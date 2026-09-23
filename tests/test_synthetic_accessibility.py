import pytest
from rdkit import Chem

from mtrl.synthetic_accessibility import br_sascore, br_sascore_rejection_reason


def test_br_sascore_matches_published_aspirin_example() -> None:
    aspirin = Chem.MolFromSmiles("CC(OC1=CC=CC=C1C(O)=O)=O")

    assert br_sascore(aspirin) == pytest.approx(2.0898575151)
    assert br_sascore_rejection_reason(aspirin, 5.0) is None


def test_br_sascore_rejects_difficult_molecule_with_score() -> None:
    molecule = Chem.MolFromSmiles("CN(C)C1=C2CC(Cc3ccccc3)CCC2C=C1")

    reason = br_sascore_rejection_reason(molecule, 5.0)

    assert reason is not None
    assert reason.startswith("BR-SAScore failed:")
    assert reason.endswith("> 5")
