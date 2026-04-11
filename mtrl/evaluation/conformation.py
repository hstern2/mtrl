from __future__ import annotations

from typing import TYPE_CHECKING

from amsr import GetConformerAndEnergy

from mtrl.data.amsr_wrapper import detokenize

if TYPE_CHECKING:
    from rdkit.Chem import Mol


def strain_energy(tokens: list[str]) -> float | None:
    """Compute strain energy for a molecule from its AMSR tokens.

    Returns energy in kcal/mol, or None if computation fails.
    """
    amsr_string = "".join(tokens)
    try:
        _mol, energy = GetConformerAndEnergy(amsr_string)
        return float(energy)
    except Exception:
        return None


def steric_clashes(mol: Mol, threshold: float = 0.7) -> int:
    """Count atom pairs closer than threshold * sum of vdW radii.

    Requires the molecule to have a conformer.
    """
    if mol.GetNumConformers() == 0:
        return 0

    conf = mol.GetConformer()
    positions = conf.GetPositions()
    n_atoms = mol.GetNumAtoms()
    clashes = 0

    # Simplified vdW radii (Angstroms)
    vdw = {"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8, "F": 1.47, "Cl": 1.75, "Br": 1.85}

    for i in range(n_atoms):
        ri = vdw.get(mol.GetAtomWithIdx(i).GetSymbol(), 1.7)
        for j in range(i + 1, n_atoms):
            rj = vdw.get(mol.GetAtomWithIdx(j).GetSymbol(), 1.7)
            dist = float(((positions[i] - positions[j]) ** 2).sum() ** 0.5)
            if dist < threshold * (ri + rj):
                clashes += 1

    return clashes
