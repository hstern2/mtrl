from __future__ import annotations

from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import Mol


def read_mol(path: Path) -> Mol:
    """Read the first valid molecule from an SDF."""
    with rdBase.BlockLogs():
        molecule = next(
            (
                molecule
                for molecule in Chem.SDMolSupplier(str(path), removeHs=False)
                if molecule is not None
            ),
            None,
        )
    if molecule is None:
        raise RuntimeError(f"no valid molecule in {path}")
    return molecule


def extract_scores(path: Path) -> dict[str, float | bool]:
    """Return the best GNINA affinity and whether any pose passed PoseBusters."""
    scores: dict[str, float | bool] = {"posebusters_passed": False}
    try:
        with rdBase.BlockLogs():
            for molecule in Chem.SDMolSupplier(str(path), removeHs=False):
                if molecule is None:
                    continue
                properties = molecule.GetPropsAsDict()
                affinity = properties.get("CNNaffinity")
                if affinity is not None:
                    value = float(affinity)
                    previous = scores.get("CNNaffinity")
                    if previous is None or value > float(previous):
                        scores["CNNaffinity"] = value
                passed = properties.get("posebusters_passed", False)
                if passed is True or str(passed).lower() in {"true", "1"}:
                    scores["posebusters_passed"] = True
    except Exception:
        pass
    return scores


def extract_tanimoto_combination(path: Path) -> float | None:
    """Return the best Roshambo2 combination score in an SDF."""
    best = None
    try:
        with rdBase.BlockLogs():
            for molecule in Chem.SDMolSupplier(str(path), removeHs=False):
                if molecule is None:
                    continue
                value = molecule.GetPropsAsDict().get("tanimoto_combination")
                if value is not None:
                    score = float(value)
                    best = score if best is None else max(best, score)
    except Exception:
        pass
    return best
