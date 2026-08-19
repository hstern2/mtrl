from unittest.mock import patch

from rdkit import Chem

from mtrl import decode_amsr, detokenize


def test_detokenize_builds_conformer_from_decoded_dihedrals() -> None:
    topology = Chem.MolFromSmiles("CCCC")
    conformer = Chem.Mol(topology)
    dihedrals = {(0, 1, 2, 3): 60}

    def fake_to_mol(value, *, stringent, dihedral):
        assert value == "CCCC"
        assert stringent is True
        dihedral.update(dihedrals)
        return topology

    with (
        patch("mtrl.ToMol", side_effect=fake_to_mol),
        patch("mtrl.GetConformer", return_value=conformer) as get_conformer,
    ):
        decoded = decode_amsr(["C", "C", "C", "C"])
        result = detokenize(["C", "C", "C", "C"])

    assert decoded is not None
    assert decoded.dihedrals == dihedrals
    assert result is conformer
    get_conformer.assert_called_once_with(topology, dihedral=dihedrals)


def test_detokenize_returns_none_on_decode_error() -> None:
    with patch("mtrl.ToMol", side_effect=ValueError("bad AMSR")):
        assert detokenize(["bad"]) is None
