from __future__ import annotations

from typing import Any

from rdkit.Chem import Mol
from rdkit.Chem import QED as QEDModule

from trl.objectives.base import Objective


class QEDObjective(Objective):
    """Quantitative Estimate of Drug-likeness (higher is more drug-like)."""

    def __init__(
        self,
        name: str = "qed",
        direction: str = "maximize",
    ) -> None:
        super().__init__(name=name, direction=direction)

    def score_batch(self, items: list[Any]) -> list[float]:
        scores: list[float] = []
        for mol in items:
            if isinstance(mol, Mol):
                scores.append(QEDModule.qed(mol))
            else:
                scores.append(0.0)
        return scores
