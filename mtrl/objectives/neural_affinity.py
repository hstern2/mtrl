from __future__ import annotations

from typing import Any

from trl.objectives.base import Objective


class NeuralAffinityObjective(Objective):
    """Neural network-based binding affinity predictor (stub)."""

    def __init__(
        self,
        name: str = "affinity",
        direction: str = "minimize",
        model_path: str = "",
    ) -> None:
        super().__init__(name=name, direction=direction)
        self.model_path = model_path

    def score_batch(self, items: list[Any]) -> list[float]:
        raise NotImplementedError("Neural affinity scoring not yet implemented")
