from __future__ import annotations

import csv
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from rdkit import Chem
from rdkit.Chem import Mol

_SUPPORTED_ELEMENTS = frozenset({"B", "Br", "C", "Cl", "F", "I", "N", "O", "P", "S"})
_SUPPORTED_BONDS = frozenset(
    {
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC,
    }
)
_MAX_BATCH_SIZE = 128


@dataclass(frozen=True)
class AEVPligInput:
    """One protein-ligand pose to rescore."""

    identifier: str
    receptor_pdb: Path
    pose: Mol


class AEVPligScorer:
    """Batch adapter for the upstream AEV-PLIG prediction script."""

    def __init__(
        self,
        directory: Path,
        *,
        python: Path | None = None,
        model: str = "model_GATv2Net_ligsim90_fep_benchmark",
        workers: int = 4,
        timeout_seconds: int = 600,
        environment: dict[str, str] | None = None,
        verbose: bool = False,
    ) -> None:
        self.directory = directory
        self.python = python or Path(sys.executable)
        self.model = model
        self.workers = workers
        self.timeout_seconds = timeout_seconds
        self.environment = environment
        self.verbose = verbose

    def score(self, inputs: list[AEVPligInput]) -> dict[str, float]:
        """Return predicted pK values keyed by input identifier."""
        identifiers = [item.identifier for item in inputs]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("AEV-PLIG input identifiers must be unique")
        inputs = [item for item in inputs if _supported(item.pose)]
        if not inputs:
            return {}
        scores = {}
        for start in range(0, len(inputs), _MAX_BATCH_SIZE):
            scores.update(self._score_batch(inputs[start : start + _MAX_BATCH_SIZE]))
        return scores

    def _score_batch(self, inputs: list[AEVPligInput]) -> dict[str, float]:
        data_name = f"mtrl_{os.getpid()}_{uuid.uuid4().hex}"
        graphs = self.directory / "data" / f"{data_name}_graphs.pickle"
        processed = self.directory / "data" / "processed" / f"{data_name}.pt"
        predictions = self.directory / "output" / "predictions" / f"{data_name}_predictions.csv"
        for parent in (graphs.parent, processed.parent, predictions.parent):
            parent.mkdir(parents=True, exist_ok=True)

        try:
            with TemporaryDirectory(prefix="mtrl_aev_plig_") as temporary:
                work = Path(temporary)
                dataset = work / "inputs.csv"
                with dataset.open("w", newline="", encoding="utf-8") as output:
                    writer = csv.writer(output)
                    writer.writerow(("unique_id", "sdf_file", "pdb_file"))
                    for index, item in enumerate(inputs):
                        ligand = work / f"pose_{index:06d}.sdf"
                        pose = Chem.AddHs(Chem.Mol(item.pose), addCoords=True)
                        sdf = Chem.SDWriter(str(ligand))
                        try:
                            sdf.write(pose)
                        finally:
                            sdf.close()
                        writer.writerow((item.identifier, ligand, item.receptor_pdb))

                command = [
                    str(self.python),
                    "process_and_predict.py",
                    "--dataset_csv",
                    str(dataset),
                    "--data_name",
                    data_name,
                    "--trained_model_name",
                    self.model,
                    "--num_workers",
                    str(self.workers),
                    "--device",
                    "auto",
                ]
                environment = (
                    os.environ.copy() if self.environment is None else self.environment.copy()
                )
                environment.setdefault("PYTHONDONTWRITEBYTECODE", "1")
                try:
                    result = subprocess.run(
                        command,
                        cwd=self.directory,
                        env=environment,
                        check=False,
                        text=True,
                        capture_output=not self.verbose,
                        timeout=self.timeout_seconds,
                    )
                except subprocess.TimeoutExpired as error:
                    raise RuntimeError(
                        f"AEV-PLIG exceeded its {self.timeout_seconds}-second timeout"
                    ) from error
                if result.returncode != 0:
                    diagnostic = ""
                    if not self.verbose:
                        diagnostic = f": {(result.stderr or result.stdout).strip()[-500:]}"
                    raise RuntimeError(
                        f"AEV-PLIG failed with exit code {result.returncode}{diagnostic}"
                    )
                if not predictions.is_file():
                    raise RuntimeError("AEV-PLIG did not write its expected predictions CSV")
                with predictions.open(newline="", encoding="utf-8") as source:
                    rows = csv.DictReader(source)
                    return {
                        row["unique_id"]: float(row["preds"])
                        for row in rows
                        if row.get("unique_id") and row.get("preds")
                    }
        finally:
            for path in (graphs, processed, predictions):
                path.unlink(missing_ok=True)


def _supported(mol: Mol) -> bool:
    return all(
        atom.GetAtomicNum() == 1 or atom.GetSymbol() in _SUPPORTED_ELEMENTS
        for atom in mol.GetAtoms()
    ) and all(bond.GetBondType() in _SUPPORTED_BONDS for bond in mol.GetBonds())
