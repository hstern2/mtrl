from __future__ import annotations

from mtrl.data.amsr_wrapper import detokenize
from mtrl.objectives.filters import druglike_filter
from mtrl.objectives.qed import QEDObjective
from mtrl.objectives.sa_score import SAScoreObjective
from trl.objectives.base import Objectives


def build() -> Objectives:
    """Build the default molecular objectives suite.

    Called by: trl rl ... --objectives mtrl.suite:build
    """
    return Objectives(
        objectives=[
            SAScoreObjective(name="sa", direction="minimize", reject_above=6.0),
            QEDObjective(name="qed", direction="maximize"),
        ],
        decode_fn=detokenize,
        pareto_lambda=0.1,
        extra_rejection_fn=druglike_filter,
    )
