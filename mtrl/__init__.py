from __future__ import annotations

from typing import TYPE_CHECKING

from amsr import ToMol

if TYPE_CHECKING:
    from rdkit.Chem import Mol

__version__ = "0.1.0"


def detokenize(tokens: list[str]) -> Mol | None:
    """Decode AMSR tokens to an RDKit Mol (bridge for trl's decode_fn)."""
    try:
        return ToMol("".join(tokens))
    except Exception:
        return None
