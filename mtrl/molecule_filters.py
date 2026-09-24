from __future__ import annotations

from rdkit.Chem import Mol

from mtrl.druglike import (
    brenk_rejection_reason,
    druglike_rejection_reason,
    muegge_rejection_reason,
)
from mtrl.synthetic_accessibility import br_sascore_rejection_reason


def molecule_filter_rejection_reason(
    mol: Mol,
    *,
    rdkit_druglike: bool = False,
    muegge: bool = False,
    brenk: bool = False,
    max_br_sascore: float | None = None,
) -> str | None:
    """Apply enabled inexpensive 2D gates in a stable order."""
    checks = (
        (rdkit_druglike, druglike_rejection_reason),
        (muegge, muegge_rejection_reason),
        (brenk, brenk_rejection_reason),
    )
    for enabled, rejection_reason in checks:
        if enabled and (reason := rejection_reason(mol)):
            return reason
    if max_br_sascore is not None:
        return br_sascore_rejection_reason(mol, max_br_sascore)
    return None
