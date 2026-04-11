from __future__ import annotations

from rdkit.Chem import Descriptors, FilterCatalog, Mol


_PAINS_CATALOG: FilterCatalog.FilterCatalog | None = None


def _get_pains_catalog() -> FilterCatalog.FilterCatalog:
    global _PAINS_CATALOG
    if _PAINS_CATALOG is None:
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        _PAINS_CATALOG = FilterCatalog.FilterCatalog(params)
    return _PAINS_CATALOG


def druglike_filter(mol: Mol) -> tuple[bool, str]:
    """Check drug-likeness. Returns (rejected, reason)."""
    mw = Descriptors.MolWt(mol)
    if mw < 150 or mw > 600:
        return True, f"MW={mw:.0f} outside [150, 600]"

    logp = Descriptors.MolLogP(mol)
    if logp < -1 or logp > 6:
        return True, f"logP={logp:.1f} outside [-1, 6]"

    hbd = Descriptors.NumHDonors(mol)
    if hbd > 5:
        return True, f"HBD={hbd} > 5"

    hba = Descriptors.NumHAcceptors(mol)
    if hba > 10:
        return True, f"HBA={hba} > 10"

    # PAINS filter
    catalog = _get_pains_catalog()
    if catalog.HasMatch(mol):
        return True, "PAINS match"

    return False, ""
