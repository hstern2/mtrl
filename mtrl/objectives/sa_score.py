from __future__ import annotations

from typing import Any

from rdkit.Chem import Mol
from rdkit.Contrib.SA_Score import sascorer

from trl.objectives.base import Objective


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
        scores: list[float] = []
        for mol in items:
            if isinstance(mol, Mol):
                scores.append(sascorer.calculateScore(mol))
            else:
                scores.append(10.0)  # worst score
        return scores
