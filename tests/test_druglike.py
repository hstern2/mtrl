from rdkit import Chem

from mtrl.druglike import (
    brenk_rejection_reason,
    clogp_rejection_reason,
    druglike_properties,
    druglike_rejection_reason,
    muegge_rejection_reason,
    pfizer_3_75_rejection_reason,
)


def test_common_druglike_molecule_passes() -> None:
    caffeine = Chem.MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")

    assert druglike_rejection_reason(caffeine) is None
    properties = druglike_properties(caffeine)
    assert properties.molecular_weight < 500
    assert properties.clogp < 5
    assert properties.tpsa < 140


def test_explicit_clogp_gate_can_be_stricter_than_muegge() -> None:
    ibuprofen = Chem.MolFromSmiles("CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1")

    assert clogp_rejection_reason(ibuprofen, 4.0) is None
    assert clogp_rejection_reason(ibuprofen, 3.0).startswith("cLogP filter failed:")


def test_pfizer_3_75_gate_rejects_lipophilic_low_polarity_molecule() -> None:
    molecule = Chem.MolFromSmiles("Cc1cccc(NC(=O)CCc2ccccc2C)c1")

    reason = pfizer_3_75_rejection_reason(molecule)

    assert reason is not None
    assert reason.startswith("Pfizer 3/75 filter failed:")
    assert "cLogP" in reason
    assert "TPSA" in reason


def test_pfizer_3_75_gate_accepts_high_clogp_molecule_with_tpsa_above_75() -> None:
    molecule = Chem.MolFromSmiles("CCCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1")

    assert pfizer_3_75_rejection_reason(molecule) is None


def test_pfizer_3_75_gate_accepts_low_clogp_molecule_with_tpsa_below_75() -> None:
    molecule = Chem.MolFromSmiles("CCO")

    assert pfizer_3_75_rejection_reason(molecule) is None


def test_greasy_alkane_is_rejected_by_clogp() -> None:
    molecule = Chem.MolFromSmiles("CCCCCCCCCCCCCCCC")

    reason = druglike_rejection_reason(molecule)

    assert reason is not None
    assert reason.startswith("RDKit drug-likeness failed:")
    assert "cLogP" in reason


def test_all_property_violations_are_reported() -> None:
    molecule = Chem.MolFromSmiles("C" * 40)

    reason = druglike_rejection_reason(molecule)

    assert reason is not None
    assert "molecular weight" in reason
    assert "cLogP" in reason


def test_muegge_rejects_underfunctionalized_molecule() -> None:
    molecule = Chem.MolFromSmiles("O=C(CCCCCCc1ccccc1)Cc1ccccc1")

    reason = muegge_rejection_reason(molecule)

    assert reason == "Muegge filter failed: heteroatoms"


def test_brenk_rejects_long_aliphatic_chain() -> None:
    molecule = Chem.MolFromSmiles("O=C(CCCCCCc1ccccc1)Cc1ccccc1")

    reason = brenk_rejection_reason(molecule)

    assert reason == "Brenk filter failed: Aliphatic_long_chain"


def test_muegge_and_brenk_accept_common_leadlike_molecule() -> None:
    ibuprofen = Chem.MolFromSmiles("CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1")

    assert muegge_rejection_reason(ibuprofen) is None
    assert brenk_rejection_reason(ibuprofen) is None
