from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from lad import busters, gnina, roshambo
from lad.sdf import extract_scores, extract_tanimoto_combination, read_mol
from rdkit import Chem, log_handler, rdBase
from rdkit.Chem import Mol, rdMolAlign

from mtrl import DecodedAMSR, make_conformer
from mtrl.config import BoxScoringConfig, DockingTarget, ScoringConfig


@dataclass(frozen=True)
class StructureScore:
    cnn_affinity: float | None = None
    roshambo_tanimoto_combo: float | None = None
    minimized_rmsd: float | None = None
    accepted: bool = False
    rejection_reason: str = ""
    minimized_mol: Mol | None = None


_WORKER_PIPELINE: StructureScoringPipeline | None = None
_BOX_WORKER_PIPELINE: BoxDockingPipeline | None = None


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


def _initialize_box_scoring_worker(config: BoxScoringConfig) -> None:
    global _BOX_WORKER_PIPELINE
    _BOX_WORKER_PIPELINE = BoxDockingPipeline(config, workers=1)


def _score_mol_in_box_worker(mol: Mol, name: str) -> BoxDockingScore:
    if _BOX_WORKER_PIPELINE is None:
        raise RuntimeError("box-docking worker was not initialized")
    return _BOX_WORKER_PIPELINE._score_one(mol, name)


def _evaluate_decoded_in_box_worker(
    decoded: DecodedAMSR, name: str
) -> tuple[Mol | None, BoxDockingScore]:
    if _BOX_WORKER_PIPELINE is None:
        raise RuntimeError("box-docking worker was not initialized")
    conformer = make_conformer(decoded)
    if conformer is None:
        return None, BoxDockingScore(rejection_reason="AMSR conformer construction failed")
    return conformer, _BOX_WORKER_PIPELINE._score_one(conformer, name)


def minimized_pose_rmsd(aligned: Mol, minimized: Mol) -> float:
    """Symmetry-corrected heavy-atom RMSD without realigning the poses."""
    aligned_heavy = Chem.RemoveHs(aligned)
    minimized_heavy = Chem.RemoveHs(minimized)
    return float(rdMolAlign.CalcRMS(minimized_heavy, aligned_heavy))


def conformer_rmsd(reference: Mol, pose: Mol) -> float:
    """Symmetry-corrected heavy-atom RMSD after optimal rigid-body alignment."""
    reference_heavy = Chem.RemoveHs(Chem.Mol(reference))
    pose_heavy = Chem.RemoveHs(Chem.Mol(pose))
    return float(rdMolAlign.GetBestRMS(pose_heavy, reference_heavy))


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


def _read_mols(path: Path, *, quiet: bool) -> list[Mol]:
    def read() -> list[Mol]:
        return [mol for mol in Chem.SDMolSupplier(str(path), removeHs=False) if mol is not None]

    if quiet:
        with rdBase.BlockLogs():
            return read()
    return read()


def _gnina_environment(local_rank: int) -> dict[str, str]:
    """Select the physical GPU assigned to this torch rank."""
    env = os.environ.copy()
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [device.strip() for device in visible.split(",")]
        if local_rank >= len(devices):
            raise RuntimeError(f"LOCAL_RANK={local_rank} is outside CUDA_VISIBLE_DEVICES={visible}")
        env["CUDA_VISIBLE_DEVICES"] = devices[local_rank]
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    return env


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
        result = subprocess.run(
            command,
            env=_gnina_environment(self.local_rank),
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


@dataclass
class TargetDockingScore:
    """Best PoseBusters-passing pose from one receptor/box docking."""

    target_name: str
    cnn_affinity: float | None = None
    cnn_score: float | None = None
    vina_affinity: float | None = None
    pose_index: int | None = None
    pose_centroid_distance: float | None = None
    rigid_vina_affinity: float | None = None
    rigid_conformer_rmsd: float | None = None
    refined_conformer_rmsd: float | None = None
    n_poses: int = 0
    n_passing_poses: int = 0
    posebusters_failures: dict[str, int] = field(default_factory=dict)
    accepted: bool = False
    rejection_reason: str = ""
    pose: Mol | None = None

    def to_record(self) -> dict[str, Any]:
        """Return serializable diagnostics, excluding the RDKit pose."""
        return {
            "cnn_affinity": self.cnn_affinity,
            "cnn_score": self.cnn_score,
            "vina_affinity": self.vina_affinity,
            "rigid_vina_affinity": self.rigid_vina_affinity,
            "rigid_conformer_rmsd": self.rigid_conformer_rmsd,
            "refined_conformer_rmsd": self.refined_conformer_rmsd,
            "pose_index": self.pose_index,
            "pose_centroid_distance": self.pose_centroid_distance,
            "n_poses": self.n_poses,
            "n_passing_poses": self.n_passing_poses,
            "posebusters_failures": self.posebusters_failures,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class BoxDockingScore:
    """Complete objective vector and selected poses for one generated molecule."""

    targets: dict[str, TargetDockingScore] = field(default_factory=dict)
    accepted: bool = False
    rejection_reason: str = ""


def _float_property(properties: dict, *names: str) -> float | None:
    for name in names:
        value = properties.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


class BoxDockingPipeline:
    """GNINA box docking with optional rigid placement and PoseBusters gating."""

    def __init__(self, config: BoxScoringConfig, *, workers: int | None = None) -> None:
        config.validate()
        self.config = config
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.batch_number = 0
        self.workers = config.evaluation_workers if workers is None else workers
        self._pool: ProcessPoolExecutor | None = None
        gnina.require()
        busters.require()
        if config.docking_mode in {"rigid", "rigid-refine"} and shutil.which("obabel") is None:
            raise RuntimeError("rigid docking requires the Open Babel `obabel` executable")

    def score_batch(self, mols: list[Mol]) -> list[BoxDockingScore]:
        batch_number = self.batch_number
        self.batch_number += 1
        names = [f"r{self.rank}_b{batch_number}_m{i}" for i in range(len(mols))]
        if self.workers == 1 or len(mols) <= 1:
            return [self._score_one(mol, name) for mol, name in zip(mols, names, strict=True)]
        return list(self._executor().map(_score_mol_in_box_worker, mols, names))

    def evaluate_decoded_batch(
        self, candidates: list[DecodedAMSR]
    ) -> list[tuple[Mol | None, BoxDockingScore]]:
        batch_number = self.batch_number
        self.batch_number += 1
        names = [f"r{self.rank}_b{batch_number}_m{i}" for i in range(len(candidates))]
        if self.workers == 1 or len(candidates) <= 1:
            results: list[tuple[Mol | None, BoxDockingScore]] = []
            for candidate, name in zip(candidates, names, strict=True):
                conformer = make_conformer(candidate)
                if conformer is None:
                    results.append(
                        (
                            None,
                            BoxDockingScore(rejection_reason="AMSR conformer construction failed"),
                        )
                    )
                else:
                    results.append((conformer, self._score_one(conformer, name)))
            return results
        return list(self._executor().map(_evaluate_decoded_in_box_worker, candidates, names))

    def _executor(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=get_context("spawn"),
                initializer=_initialize_box_scoring_worker,
                initargs=(self.config,),
            )
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def _score_one(self, mol: Mol, name: str) -> BoxDockingScore:
        if mol.GetNumConformers() == 0 or not mol.GetConformer().Is3D():
            return BoxDockingScore(rejection_reason="AMSR conformer is missing 3D coordinates")

        with TemporaryDirectory(prefix=f"mtrl_box_{name}_") as temporary:
            work = Path(temporary)
            ligand_sdf = work / "candidate.sdf"
            try:
                candidate = Chem.AddHs(Chem.Mol(mol), addCoords=True)
                _write_mol(ligand_sdf, candidate, name)
                docking_ligand = ligand_sdf
                if self.config.docking_mode in {"rigid", "rigid-refine"}:
                    docking_ligand = work / "candidate_rigid.pdbqt"
                    self._prepare_rigid_ligand(ligand_sdf, docking_ligand)
            except Exception as error:
                reason = f"ligand preparation failed: {type(error).__name__}: {error}"
                return BoxDockingScore(
                    targets={
                        target.name: TargetDockingScore(
                            target_name=target.name, rejection_reason=reason
                        )
                        for target in self.config.targets
                    },
                    rejection_reason=reason,
                )
            target_results: dict[str, TargetDockingScore] = {}
            for target in self.config.targets:
                output_sdf = work / f"{target.name}.sdf"
                try:
                    if self.config.docking_mode == "flexible":
                        self._run_gnina(target, docking_ligand, output_sdf)
                    else:
                        rigid_output = work / f"{target.name}_rigid.sdf"
                        self._run_gnina(
                            target,
                            docking_ligand,
                            rigid_output,
                            cnn_scoring=(
                                "none" if self.config.docking_mode == "rigid-refine" else "rescore"
                            ),
                        )
                        if self.config.docking_mode == "rigid-refine":
                            self._refine_rigid_poses(target, candidate, rigid_output, output_sdf)
                        else:
                            self._annotate_rigid_poses(candidate, rigid_output, output_sdf)
                    self._run_posebusters(target.receptor_pdb, output_sdf)
                    target_results[target.name] = self._select_passing_pose(
                        target.name,
                        output_sdf,
                        center=target.center,
                        quiet=not self.config.verbose_tools,
                    )
                except Exception as error:
                    target_results[target.name] = TargetDockingScore(
                        target_name=target.name,
                        rejection_reason=(f"docking failed: {type(error).__name__}: {error}"),
                    )

            failures = [
                f"{name}: {result.rejection_reason}"
                for name, result in target_results.items()
                if not result.accepted
            ]
            target_acceptance = [result.accepted for result in target_results.values()]
            accepted = (
                any(target_acceptance)
                if self.config.accept_targets == "any"
                else all(target_acceptance)
            )
            return BoxDockingScore(
                targets=target_results,
                accepted=accepted,
                rejection_reason=("" if accepted else "; ".join(failures)),
            )

    def _prepare_rigid_ligand(self, ligand_sdf: Path, output_pdbqt: Path) -> None:
        raw_pdbqt = output_pdbqt.with_name(f"{output_pdbqt.stem}_raw.pdbqt")
        command = ["obabel", str(ligand_sdf), "-O", str(raw_pdbqt), "-xr", "-xh"]
        result = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=not self.config.verbose_tools,
        )
        if result.returncode != 0 or not raw_pdbqt.is_file():
            diagnostic = ""
            if not self.config.verbose_tools:
                diagnostic = f": {(result.stderr or result.stdout).strip()[-500:]}"
            raise RuntimeError(
                f"Open Babel rigid-PDBQT conversion failed with exit code "
                f"{result.returncode}{diagnostic}"
            )
        lines = raw_pdbqt.read_text().splitlines()
        header = []
        atoms = []
        for line in lines:
            if line.startswith(("ATOM  ", "HETATM")):
                atoms.append(line)
            elif not atoms and line.startswith("REMARK"):
                header.append(line)
        if not atoms:
            raise RuntimeError("Open Babel rigid-PDBQT conversion produced no atoms")
        output_pdbqt.write_text("\n".join([*header, "ROOT", *atoms, "ENDROOT", "TORSDOF 0", ""]))

    def _run_gnina(
        self,
        target: DockingTarget,
        ligand_path: Path,
        output_sdf: Path,
        *,
        cnn_scoring: str = "rescore",
    ) -> None:
        command = [
            "gnina",
            "--receptor",
            str(target.receptor_pdb),
            "--ligand",
            str(ligand_path),
            *self._box_arguments(target),
            "--exhaustiveness",
            str(target.exhaustiveness),
            "--num_modes",
            str(target.num_modes),
            "--cnn_scoring",
            cnn_scoring,
            "--pose_sort_order",
            "Energy" if cnn_scoring == "none" else "CNNscore",
            "--out",
            str(output_sdf),
        ]
        self._execute_gnina(command, output_sdf, action="gnina")

    def _run_gnina_minimize(
        self, target: DockingTarget, ligand_sdf: Path, output_sdf: Path
    ) -> None:
        command = [
            "gnina",
            "--minimize",
            "--receptor",
            str(target.receptor_pdb),
            "--ligand",
            str(ligand_sdf),
            *self._box_arguments(target),
            "--cnn_scoring",
            "rescore",
            "--out",
            str(output_sdf),
        ]
        self._execute_gnina(command, output_sdf, action="gnina --minimize")

    @staticmethod
    def _box_arguments(target: DockingTarget) -> list[str]:
        return [
            "--center_x",
            str(target.center[0]),
            "--center_y",
            str(target.center[1]),
            "--center_z",
            str(target.center[2]),
            "--size_x",
            str(target.size[0]),
            "--size_y",
            str(target.size[1]),
            "--size_z",
            str(target.size[2]),
        ]

    def _execute_gnina(self, command: list[str], output_sdf: Path, *, action: str) -> None:
        try:
            result = subprocess.run(
                command,
                env=_gnina_environment(self.local_rank),
                check=False,
                text=True,
                capture_output=not self.config.verbose_tools,
                timeout=self.config.gnina_timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            output_sdf.unlink(missing_ok=True)
            raise RuntimeError(
                f"{action} exceeded its {self.config.gnina_timeout_seconds}-second timeout"
            ) from error
        if result.returncode != 0:
            output_sdf.unlink(missing_ok=True)
            diagnostic = ""
            if not self.config.verbose_tools:
                diagnostic = f": {(result.stderr or result.stdout).strip()[-500:]}"
            raise RuntimeError(f"{action} failed with exit code {result.returncode}{diagnostic}")
        if not output_sdf.is_file():
            raise RuntimeError(f"{action} did not write its expected SDF output")

    def _run_posebusters(self, receptor_pdb: Path, poses_sdf: Path) -> None:
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mtrl.posebusters_runner",
                    self.config.posebusters_config,
                    str(receptor_pdb),
                    str(poses_sdf),
                ],
                check=False,
                text=True,
                capture_output=not self.config.verbose_tools,
                timeout=self.config.posebusters_timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"PoseBusters exceeded its {self.config.posebusters_timeout_seconds}-second timeout"
            ) from error
        if result.returncode != 0:
            diagnostic = ""
            if not self.config.verbose_tools:
                diagnostic = f": {(result.stderr or result.stdout).strip()[-500:]}"
            raise RuntimeError(f"PoseBusters failed with exit code {result.returncode}{diagnostic}")

    def _annotate_rigid_poses(
        self, decoded: Mol, rigid_output: Path, annotated_output: Path
    ) -> None:
        writer = Chem.SDWriter(str(annotated_output))
        written = 0
        try:
            for pose in _read_mols(rigid_output, quiet=not self.config.verbose_tools):
                try:
                    pose.SetDoubleProp("rigid_conformer_rmsd", conformer_rmsd(decoded, pose))
                except (RuntimeError, ValueError):
                    pass
                rigid_affinity = _float_property(
                    pose.GetPropsAsDict(), "minimizedAffinity", "Affinity"
                )
                if rigid_affinity is not None:
                    pose.SetDoubleProp("rigid_vina_affinity", rigid_affinity)
                writer.write(pose)
                written += 1
        finally:
            writer.close()
        if not written:
            annotated_output.unlink(missing_ok=True)
            raise RuntimeError("rigid GNINA docking produced no readable poses")

    def _refine_rigid_poses(
        self,
        target: DockingTarget,
        decoded: Mol,
        rigid_output: Path,
        refined_output: Path,
    ) -> None:
        quiet = not self.config.verbose_tools
        rigid_poses = _read_mols(rigid_output, quiet=quiet)
        if not rigid_poses:
            raise RuntimeError("rigid GNINA docking produced no readable poses")

        raw_refined_output = refined_output.with_name(f"{refined_output.stem}_raw.sdf")
        self._run_gnina_minimize(target, rigid_output, raw_refined_output)
        refined_poses = _read_mols(raw_refined_output, quiet=quiet)
        if len(refined_poses) != len(rigid_poses):
            raise RuntimeError(
                "GNINA minimization returned "
                f"{len(refined_poses)} poses for {len(rigid_poses)} rigid inputs"
            )

        writer = Chem.SDWriter(str(refined_output))
        try:
            for pose_index, (rigid_pose, refined_pose) in enumerate(
                zip(rigid_poses, refined_poses, strict=True), start=1
            ):
                rigid_affinity = _float_property(
                    rigid_pose.GetPropsAsDict(), "minimizedAffinity", "Affinity"
                )
                if rigid_affinity is not None:
                    refined_pose.SetDoubleProp("rigid_vina_affinity", rigid_affinity)
                refined_pose.SetIntProp("rigid_pose_index", pose_index)
                try:
                    refined_pose.SetDoubleProp(
                        "rigid_conformer_rmsd", conformer_rmsd(decoded, rigid_pose)
                    )
                    refined_pose.SetDoubleProp(
                        "refined_conformer_rmsd", conformer_rmsd(decoded, refined_pose)
                    )
                except (RuntimeError, ValueError):
                    pass
                writer.write(refined_pose)
        finally:
            writer.close()

    @staticmethod
    def _select_passing_pose(
        target_name: str,
        output_sdf: Path,
        *,
        center: tuple[float, float, float] | None = None,
        quiet: bool = True,
    ) -> TargetDockingScore:
        candidates: list[TargetDockingScore] = []
        n_poses = 0
        n_passing = 0
        failures: Counter[str] = Counter()
        for pose_index, pose in enumerate(_read_mols(output_sdf, quiet=quiet), start=1):
            n_poses += 1
            properties = pose.GetPropsAsDict()
            passed_value = properties.get("posebusters_passed", False)
            passed = passed_value is True or str(passed_value).lower() in {"true", "1"}
            if not passed:
                failed_checks = str(properties.get("posebusters_failed_checks", "unknown"))
                failures.update(check for check in failed_checks.split(",") if check)
                continue
            n_passing += 1
            affinity = _float_property(properties, "CNNaffinity")
            if affinity is None:
                continue
            candidates.append(
                TargetDockingScore(
                    target_name=target_name,
                    cnn_affinity=affinity,
                    cnn_score=_float_property(properties, "CNNscore"),
                    vina_affinity=_float_property(properties, "minimizedAffinity", "Affinity"),
                    rigid_vina_affinity=_float_property(properties, "rigid_vina_affinity"),
                    rigid_conformer_rmsd=_float_property(properties, "rigid_conformer_rmsd"),
                    refined_conformer_rmsd=_float_property(properties, "refined_conformer_rmsd"),
                    pose_index=pose_index,
                    accepted=True,
                    pose=Chem.Mol(pose),
                )
            )

        if not candidates:
            reason = (
                "PoseBusters failed for every docking pose"
                if n_passing == 0
                else "CNNaffinity is missing from every PoseBusters-passing pose"
            )
            return TargetDockingScore(
                target_name=target_name,
                n_poses=n_poses,
                n_passing_poses=n_passing,
                posebusters_failures=dict(failures),
                rejection_reason=reason,
            )

        selected = max(
            candidates,
            key=lambda result: (
                result.cnn_affinity if result.cnn_affinity is not None else -math.inf
            ),
        )
        selected.n_poses = n_poses
        selected.n_passing_poses = n_passing
        selected.posebusters_failures = dict(failures)
        if center is not None and selected.pose is not None:
            conformer = selected.pose.GetConformer()
            heavy_atom_indices = [
                atom.GetIdx() for atom in selected.pose.GetAtoms() if atom.GetAtomicNum() > 1
            ]
            if heavy_atom_indices:
                centroid = tuple(
                    sum(conformer.GetAtomPosition(index)[axis] for index in heavy_atom_indices)
                    / len(heavy_atom_indices)
                    for axis in range(3)
                )
                selected.pose_centroid_distance = math.dist(centroid, center)
        return selected
