from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from amsr import GetConformer, ToMol

if TYPE_CHECKING:
    from rdkit.Chem import Mol

__version__ = "0.1.0"


@dataclass(frozen=True)
class DecodedAMSR:
    mol: Mol
    dihedrals: dict[tuple[int, int, int, int], int]


def decode_amsr(tokens: list[str]) -> DecodedAMSR | None:
    """Decode AMSR tokens without performing the expensive 3D construction."""
    try:
        dihedrals: dict[tuple[int, int, int, int], int] = {}
        mol = ToMol("".join(tokens), stringent=True, dihedral=dihedrals)
        return DecodedAMSR(mol=mol, dihedrals=dihedrals)
    except Exception:
        return None


def make_conformer(decoded: DecodedAMSR) -> Mol | None:
    """Construct the AMSR dihedral-derived 3D conformer."""
    try:
        return GetConformer(decoded.mol, dihedral=decoded.dihedrals)
    except Exception:
        return None


def detokenize(tokens: list[str]) -> Mol | None:
    """Decode AMSR tokens to their dihedral-derived 3D conformer."""
    decoded = decode_amsr(tokens)
    return make_conformer(decoded) if decoded is not None else None
