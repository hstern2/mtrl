from __future__ import annotations

from mtrl.config import BoxScoringConfig
from mtrl.scoring import BoxDockingScore


def cnn_affinity_scores(
    result: BoxDockingScore,
    config: BoxScoringConfig,
    *,
    penalty: float = 0.0,
    adjusted: bool = False,
) -> dict[str, float]:
    """Return per-target CNNaffinities, optionally adjusted by a shared penalty."""
    prefix = "clogp_adjusted_cnn_affinity" if adjusted else "gnina_cnn_affinity"
    scores = {}
    for target in config.targets:
        target_result = result.targets[target.name]
        scores[f"{prefix}__{target.name}"] = (
            float(target_result.cnn_affinity) - penalty
            if target_result.accepted and target_result.cnn_affinity is not None
            else config.target_failure_score
        )
    return scores
