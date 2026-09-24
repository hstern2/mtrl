from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, FilterCatalog, Lipinski, Mol, rdMolDescriptors


@dataclass(frozen=True)
class DruglikeProperties:
    molecular_weight: float
    clogp: float
    hbd: int
    hba: int
    rotatable_bonds: int
    tpsa: float


def druglike_properties(mol: Mol) -> DruglikeProperties:
    """Calculate the simple RDKit descriptors used by the drug-likeness gate."""
    parent = Chem.RemoveHs(mol)
    return DruglikeProperties(
        molecular_weight=float(Descriptors.MolWt(parent)),  # type: ignore[attr-defined]
        clogp=float(Crippen.MolLogP(parent)),  # type: ignore[attr-defined]
        hbd=int(Lipinski.NumHDonors(parent)),  # type: ignore[attr-defined]
        hba=int(Lipinski.NumHAcceptors(parent)),  # type: ignore[attr-defined]
        rotatable_bonds=int(
            Lipinski.NumRotatableBonds(parent)  # type: ignore[attr-defined]
        ),
        tpsa=float(rdMolDescriptors.CalcTPSA(parent)),
    )


def druglike_rejection_reason(mol: Mol) -> str | None:
    """Return Rule-of-Five/Veber violations, or ``None`` when the molecule passes."""
    properties = druglike_properties(mol)
    checks = (
        ("molecular weight", properties.molecular_weight, 500.0),
        ("cLogP", properties.clogp, 5.0),
        ("H-bond donors", properties.hbd, 5),
        ("H-bond acceptors", properties.hba, 10),
        ("rotatable bonds", properties.rotatable_bonds, 10),
        ("TPSA", properties.tpsa, 140.0),
    )
    violations = [
        f"{name} {value:g} > {limit:g}"
        for name, value, limit in checks
        if not math.isfinite(value) or value > limit
    ]
    if not violations:
        return None
    return f"RDKit drug-likeness failed: {'; '.join(violations)}"


def muegge_rejection_reason(mol: Mol) -> str | None:
    """Return Muegge drug-likeness violations, or ``None`` when it passes."""
    parent = Chem.RemoveHs(mol)
    properties = druglike_properties(parent)
    rings = int(rdMolDescriptors.CalcNumRings(parent))
    carbons = sum(atom.GetAtomicNum() == 6 for atom in parent.GetAtoms())
    heteroatoms = int(rdMolDescriptors.CalcNumHeteroatoms(parent))
    checks = (
        ("molecular weight", 200.0 <= properties.molecular_weight <= 600.0),
        ("cLogP", -2.0 <= properties.clogp <= 5.0),
        ("TPSA", properties.tpsa <= 150.0),
        ("rings", rings <= 7),
        ("carbon atoms", carbons > 4),
        ("heteroatoms", heteroatoms > 1),
        ("rotatable bonds", properties.rotatable_bonds <= 15),
        ("H-bond acceptors", properties.hba <= 10),
        ("H-bond donors", properties.hbd <= 5),
    )
    violations = [name for name, passes in checks if not passes]
    if not violations:
        return None
    return f"Muegge filter failed: {', '.join(violations)}"


@cache
def _brenk_catalog() -> FilterCatalog.FilterCatalog:
    parameters = FilterCatalog.FilterCatalogParams()
    parameters.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog.FilterCatalog(parameters)


def brenk_rejection_reason(mol: Mol) -> str | None:
    """Return matching RDKit Brenk alerts, or ``None`` when there are none."""
    matches = _brenk_catalog().GetMatches(Chem.RemoveHs(mol))
    alerts = list(dict.fromkeys(match.GetDescription() for match in matches))
    if not alerts:
        return None
    return f"Brenk filter failed: {', '.join(alerts)}"
