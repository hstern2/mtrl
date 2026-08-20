from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from lad import busters, gnina, roshambo
from lad.sdf import extract_scores, extract_tanimoto_combination, read_mol
from rdkit import Chem, log_handler, rdBase
from rdkit.Chem import Mol, rdMolAlign

from mtrl.config import ScoringConfig


@dataclass(frozen=True)
class StructureScore:
    cnn_affinity: float | None = None
    roshambo_tanimoto_combo: float | None = None
    minimized_rmsd: float | None = None
    accepted: bool = False
    rejection_reason: str = ""
    minimized_mol: Mol | None = None


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
            try:
                with redirect_stdout(discard), redirect_stderr(discard):
                    yield
            finally:
                # PoseBusters temporarily redirects RDKit logging to sys.stderr.
                # Restore it while ``discard`` is still open so subsequent calls
                # never inherit a closed stream.
                log_handler.setStream(original_rdkit_stream)
    finally:
        for logger, was_disabled in zip(loggers, disabled, strict=True):
            logger.disabled = was_disabled


class StructureScoringPipeline:
    """Roshambo2 alignment, GNINA minimization, and PoseBusters gating."""

    def __init__(self, config: ScoringConfig) -> None:
        config.validate()
        self.config = config
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.batch_number = 0
        roshambo.require()
        gnina.require()
        busters.require()

    def score_batch(self, mols: list[Mol]) -> list[StructureScore]:
        batch_number = self.batch_number
        self.batch_number += 1
        return [
            self._score_one(mol, f"r{self.rank}_b{batch_number}_m{i}") for i, mol in enumerate(mols)
        ]

    def _score_one(self, mol: Mol, name: str) -> StructureScore:
        if mol.GetNumConformers() == 0 or not mol.GetConformer().Is3D():
            return StructureScore(rejection_reason="AMSR conformer is missing 3D coordinates")

        with TemporaryDirectory(prefix=f"mtrl_{name}_") as temporary:
            work = Path(temporary)
            candidate_sdf = work / "candidate.sdf"
            aligned_sdf = work / "aligned.sdf"
            minimized_sdf = work / "minimized.sdf"
            try:
                # GNINA and Roshambo2 conventionally consume explicit-hydrogen SDFs.
                candidate = Chem.AddHs(Chem.Mol(mol), addCoords=True)
                _write_mol(candidate_sdf, candidate, name)
                with _quiet_tools(not self.config.verbose_tools):
                    roshambo.run(self.config.reference_sdf, candidate_sdf, aligned_sdf)
                combo = extract_tanimoto_combination(aligned_sdf)
                if combo is None:
                    return StructureScore(rejection_reason="Roshambo2 score is missing")

                aligned = read_mol(aligned_sdf)
                _write_mol(minimized_sdf, aligned, name)
                self._run_gnina(minimized_sdf)
                minimized_before_busters = read_mol(minimized_sdf)
                rmsd = minimized_pose_rmsd(aligned, minimized_before_busters)

                with _quiet_tools(not self.config.verbose_tools):
                    busters.run(self.config.receptor_pdb, minimized_sdf)
                gnina_scores = extract_scores(minimized_sdf)
                affinity = gnina_scores.get("CNNaffinity")
                if affinity is None:
                    return StructureScore(
                        roshambo_tanimoto_combo=combo,
                        minimized_rmsd=rmsd,
                        rejection_reason="GNINA CNNaffinity is missing",
                    )
                if not gnina_scores.get("posebusters_passed", False):
                    return StructureScore(
                        cnn_affinity=float(affinity),
                        roshambo_tanimoto_combo=combo,
                        minimized_rmsd=rmsd,
                        rejection_reason="PoseBusters failed",
                    )
                if rmsd > self.config.max_minimized_rmsd:
                    return StructureScore(
                        cnn_affinity=float(affinity),
                        roshambo_tanimoto_combo=combo,
                        minimized_rmsd=rmsd,
                        rejection_reason=(
                            f"minimization RMSD={rmsd:.3f} A > "
                            f"{self.config.max_minimized_rmsd:.3f} A"
                        ),
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
