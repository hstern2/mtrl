from __future__ import annotations

from typing import TYPE_CHECKING

from amsr import ToMol

if TYPE_CHECKING:
    from rdkit.Chem import Mol


def detokenize(tokens: list[str]) -> Mol | None:
    """Convert AMSR token strings to an RDKit Mol.

    This is the decode_fn bridge between trl's domain-agnostic
    objectives system and molecular scoring.
    """
    amsr_string = "".join(tokens)
    try:
        return ToMol(amsr_string)
    except Exception:
        return None
