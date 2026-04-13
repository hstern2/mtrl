from __future__ import annotations

from typing import Any

from rdkit.Chem import QED as QEDModule
from rdkit.Chem import Descriptors, FilterCatalog, Mol
from rdkit.Contrib.SA_Score import sascorer
from trl.objectives.base import Objective, Objectives

from mtrl import detokenize


class QEDObjective(Objective):
    """Quantitative Estimate of Drug-likeness (higher is more drug-like)."""

    def __init__(self, name: str = "qed", direction: str = "maximize") -> None:
        super().__init__(name=name, direction=direction)

    def score_batch(self, items: list[Any]) -> list[float]:
        return [QEDModule.qed(m) if isinstance(m, Mol) else 0.0 for m in items]


class SAScoreObjective(Objective):
    """Synthetic Accessibility score (lower is easier to synthesize)."""

    def __init__(
        self,
        name: str = "sa",
        direction: str = "minimize",
        reject_above: float | None = 6.0,
    ) -> None:
        super().__init__(name=name, direction=direction, reject_above=reject_above)

    def score_batch(self, items: list[Any]) -> list[float]:
        return [sascorer.calculateScore(m) if isinstance(m, Mol) else 10.0 for m in items]


_PAINS_CATALOG: FilterCatalog.FilterCatalog | None = None


def _get_pains_catalog() -> FilterCatalog.FilterCatalog:
    global _PAINS_CATALOG
    if _PAINS_CATALOG is None:
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        _PAINS_CATALOG = FilterCatalog.FilterCatalog(params)
    return _PAINS_CATALOG


def druglike_filter(mol: Mol) -> tuple[bool, str]:
    """Check drug-likeness. Returns (rejected, reason)."""
    mw = Descriptors.MolWt(mol)
    if mw < 150 or mw > 600:
        return True, f"MW={mw:.0f} outside [150, 600]"

    logp = Descriptors.MolLogP(mol)
    if logp < -1 or logp > 6:
        return True, f"logP={logp:.1f} outside [-1, 6]"

    if Descriptors.NumHDonors(mol) > 5:
        return True, "HBD > 5"

    if Descriptors.NumHAcceptors(mol) > 10:
        return True, "HBA > 10"

    if _get_pains_catalog().HasMatch(mol):
        return True, "PAINS match"

    return False, ""


def build() -> Objectives:
    """Default molecular objectives. Called by: trl rl ... --objectives mtrl.objectives:build"""
    return Objectives(
        objectives=[
            SAScoreObjective(name="sa", direction="minimize", reject_above=6.0),
            QEDObjective(name="qed", direction="maximize"),
        ],
        decode_fn=detokenize,
        pareto_lambda=0.1,
        extra_rejection_fn=druglike_filter,
    )
