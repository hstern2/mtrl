from __future__ import annotations

from typing import Any

from trl.objectives.base import Objective


class DockingObjective(Objective):
    """Molecular docking score (stub)."""

    def __init__(
        self,
        name: str = "docking",
        direction: str = "minimize",
        reject_above: float | None = -4.0,
        receptor_path: str = "",
        center: list[float] | None = None,
        box_size: list[float] | None = None,
    ) -> None:
        super().__init__(name=name, direction=direction, reject_above=reject_above)
        self.receptor_path = receptor_path
        self.center = center or [0.0, 0.0, 0.0]
        self.box_size = box_size or [20.0, 20.0, 20.0]

    def score_batch(self, items: list[Any]) -> list[float]:
        raise NotImplementedError("Docking scoring not yet implemented")
