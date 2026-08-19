from __future__ import annotations

from typing import TYPE_CHECKING

from mtrl import detokenize

if TYPE_CHECKING:
    from rdkit.Chem import Mol


def build_conformer(tokens: list[str]) -> Mol | None:
    """Decode one AMSR token sequence and construct its encoded conformer."""
    return detokenize(tokens)
