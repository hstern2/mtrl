from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory

from lad import busters, gnina, roshambo
from lad.sdf import extract_scores, extract_tanimoto_combination, read_mol
from rdkit import Chem, log_handler, rdBase
from rdkit.Chem import Mol, rdMolAlign

from mtrl import DecodedAMSR, make_conformer
from mtrl.config import ScoringConfig


@dataclass(frozen=True)
class StructureScore:
    cnn_affinity: float | None = None
    roshambo_tanimoto_combo: float | None = None
    minimized_rmsd: float | None = None
    accepted: bool = False
    rejection_reason: str = ""
    minimized_mol: Mol | None = None


_WORKER_PIPELINE: StructureScoringPipeline | None = None


def _initialize_scoring_worker(config: ScoringConfig) -> None:
    global _WORKER_PIPELINE
    _WORKER_PIPELINE = StructureScoringPipeline(config, workers=1)


def _score_mol_in_worker(mol: Mol, name: str) -> StructureScore:
    if _WORKER_PIPELINE is None:
        raise RuntimeError("structure-scoring worker was not initialized")
    return _WORKER_PIPELINE._score_one(mol, name)


def _evaluate_decoded_in_worker(
    decoded: DecodedAMSR, name: str
) -> tuple[Mol | None, StructureScore]:
    if _WORKER_PIPELINE is None:
        raise RuntimeError("structure-scoring worker was not initialized")
    conformer = make_conformer(decoded)
    if conformer is None:
        return None, StructureScore(rejection_reason="AMSR conformer construction failed")
    return conformer, _WORKER_PIPELINE._score_one(conformer, name)


def minimized_pose_rmsd(aligned: Mol, minimized: Mol) -> float:
    """Symmetry-corrected heavy-atom RMSD without realigning the poses."""
    aligned_heavy = Chem.RemoveHs(aligned)
    minimized_heavy = Chem.RemoveHs(minimized)
    return float(rdMolAlign.CalcRMS(minimized_heavy, aligned_heavy))


def _write_mol(path: Path, mol: Mol, name: str) -> None:
    output = Chem.Mol(mol)
    output.SetProp("_Name", name)
    writer = Chem.SDWriter(str(path))
    try:
        writer.write(output)
    finally:
        writer.close()


def _read_mol(path: Path, *, quiet: bool) -> Mol:
    if not quiet:
        return read_mol(path)
    with rdBase.BlockLogs():
        return read_mol(path)


@contextmanager
def _quiet_tools(quiet: bool) -> Iterator[None]:
    if not quiet:
        yield
        return
    loggers = [logging.getLogger(name) for name in ("roshambo2", "posebusters")]
    disabled = [logger.disabled for logger in loggers]
    try:
        for logger in loggers:
            logger.disabled = True
        with Path(os.devnull).open("w") as discard, rdBase.BlockLogs():
            original_rdkit_stream = log_handler.stream
            saved_stdout = os.dup(1)
            saved_stderr = os.dup(2)
            try:
                os.dup2(discard.fileno(), 1)
                os.dup2(discard.fileno(), 2)
                with redirect_stdout(discard), redirect_stderr(discard):
                    yield
            finally:
                # PoseBusters temporarily redirects RDKit logging to sys.stderr.
                # Restore it while ``discard`` is still open so subsequent calls
                # never inherit a closed stream.
                log_handler.setStream(original_rdkit_stream)
                os.dup2(saved_stdout, 1)
                os.dup2(saved_stderr, 2)
                os.close(saved_stdout)
                os.close(saved_stderr)
    finally:
        for logger, was_disabled in zip(loggers, disabled, strict=True):
            logger.disabled = was_disabled


class StructureScoringPipeline:
    """Roshambo2 alignment, GNINA minimization, and PoseBusters gating."""

    def __init__(self, config: ScoringConfig, *, workers: int | None = None) -> None:
        config.validate()
        self.config = config
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.batch_number = 0
        self.workers = config.evaluation_workers if workers is None else workers
        self._pool: ProcessPoolExecutor | None = None
        roshambo.require()
        gnina.require()
        busters.require()
        reference = next(
            (
                mol
                for mol in Chem.SDMolSupplier(str(config.reference_sdf), removeHs=False)
                if mol is not None
            ),
            None,
        )
        if reference is None:
            raise ValueError(f"cannot read reference SDF: {config.reference_sdf}")
        self.roshambo_reference = Chem.AddHs(reference, addCoords=True)

    def score_batch(self, mols: list[Mol]) -> list[StructureScore]:
        batch_number = self.batch_number
        self.batch_number += 1
        names = [f"r{self.rank}_b{batch_number}_m{i}" for i in range(len(mols))]
        if self.workers == 1 or len(mols) <= 1:
            return [self._score_one(mol, name) for mol, name in zip(mols, names, strict=True)]
        return list(self._executor().map(_score_mol_in_worker, mols, names))

    def evaluate_decoded_batch(
        self, candidates: list[DecodedAMSR]
    ) -> list[tuple[Mol | None, StructureScore]]:
        """Construct and score conformers concurrently while preserving input order."""
        batch_number = self.batch_number
        self.batch_number += 1
        names = [f"r{self.rank}_b{batch_number}_m{i}" for i in range(len(candidates))]
        if self.workers == 1 or len(candidates) <= 1:
            results: list[tuple[Mol | None, StructureScore]] = []
            for candidate, name in zip(candidates, names, strict=True):
                conformer = make_conformer(candidate)
                if conformer is None:
                    results.append(
                        (
                            None,
                            StructureScore(rejection_reason="AMSR conformer construction failed"),
                        )
                    )
                else:
                    results.append((conformer, self._score_one(conformer, name)))
            return results
        return list(self._executor().map(_evaluate_decoded_in_worker, candidates, names))

    def _executor(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=get_context("spawn"),
                initializer=_initialize_scoring_worker,
                initargs=(self.config,),
            )
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def _score_one(self, mol: Mol, name: str) -> StructureScore:
        if mol.GetNumConformers() == 0 or not mol.GetConformer().Is3D():
            return StructureScore(rejection_reason="AMSR conformer is missing 3D coordinates")

        with TemporaryDirectory(prefix=f"mtrl_{name}_") as temporary:
            work = Path(temporary)
            reference_sdf = work / "reference.sdf"
            candidate_sdf = work / "candidate.sdf"
            aligned_sdf = work / "aligned.sdf"
            minimized_sdf = work / "minimized.sdf"
            try:
                # GNINA and Roshambo2 conventionally consume explicit-hydrogen SDFs.
                candidate = Chem.AddHs(Chem.Mol(mol), addCoords=True)
                _write_mol(reference_sdf, self.roshambo_reference, "reference")
                _write_mol(candidate_sdf, candidate, name)
                with _quiet_tools(not self.config.verbose_tools):
                    roshambo.run(
                        reference_sdf,
                        candidate_sdf,
                        aligned_sdf,
                        n_cpus_prepare=1,
                    )
                combo = extract_tanimoto_combination(aligned_sdf)
                if combo is None:
                    return StructureScore(rejection_reason="Roshambo2 score is missing")

                aligned = _read_mol(aligned_sdf, quiet=not self.config.verbose_tools)
                _write_mol(minimized_sdf, aligned, name)
                self._run_gnina(minimized_sdf)
                minimized_before_busters = _read_mol(
                    minimized_sdf,
                    quiet=not self.config.verbose_tools,
                )
                rmsd = minimized_pose_rmsd(aligned, minimized_before_busters)
                gnina_scores = extract_scores(minimized_sdf)
                affinity = gnina_scores.get("CNNaffinity")
                if affinity is None:
                    return StructureScore(
                        roshambo_tanimoto_combo=combo,
                        minimized_rmsd=rmsd,
                        rejection_reason="GNINA CNNaffinity is missing",
                        minimized_mol=Chem.Mol(minimized_before_busters),
                    )
                with _quiet_tools(not self.config.verbose_tools):
                    busters.run(self.config.receptor_pdb, minimized_sdf)
                gnina_scores = extract_scores(minimized_sdf)
                if not gnina_scores.get("posebusters_passed", False):
                    return StructureScore(
                        cnn_affinity=float(affinity),
                        roshambo_tanimoto_combo=combo,
                        minimized_rmsd=rmsd,
                        rejection_reason="PoseBusters failed",
                        minimized_mol=Chem.Mol(minimized_before_busters),
                    )

                return StructureScore(
                    cnn_affinity=float(affinity),
                    roshambo_tanimoto_combo=float(combo),
                    minimized_rmsd=rmsd,
                    accepted=True,
                    minimized_mol=Chem.Mol(minimized_before_busters),
                )
            except Exception as error:
                return StructureScore(
                    rejection_reason=f"structure scoring failed: {type(error).__name__}: {error}"
                )

    def _run_gnina(self, ligand_sdf: Path) -> None:
        command = gnina.cmd(
            self.config.receptor_pdb,
            ligand_sdf,
            autobox_ligand=self.config.reference_sdf,
            minimize=True,
        )
        # Each torchrun rank owns one physical GPU; GNINA sees it as device zero.
        env = os.environ.copy()
        visible = env.get("CUDA_VISIBLE_DEVICES")
        if visible:
            devices = [device.strip() for device in visible.split(",")]
            if self.local_rank >= len(devices):
                raise RuntimeError(
                    f"LOCAL_RANK={self.local_rank} is outside CUDA_VISIBLE_DEVICES={visible}"
                )
            env["CUDA_VISIBLE_DEVICES"] = devices[self.local_rank]
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(self.local_rank)
        result = subprocess.run(
            command,
            env=env,
            check=False,
            text=True,
            capture_output=not self.config.verbose_tools,
        )
        temporary_output = ligand_sdf.with_suffix(".gnina.sdf")
        if result.returncode != 0:
            temporary_output.unlink(missing_ok=True)
            diagnostic = ""
            if not self.config.verbose_tools:
                diagnostic = f": {(result.stderr or result.stdout).strip()[-500:]}"
            raise RuntimeError(f"gnina failed with exit code {result.returncode}{diagnostic}")
        if not temporary_output.is_file():
            raise RuntimeError("gnina did not write its expected SDF output")
        temporary_output.replace(ligand_sdf)
