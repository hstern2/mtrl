import json

from rdkit import Chem

from mtrl.generate import write_conformers


def test_write_conformers_records_strings_sdf_and_summary(tmp_path, monkeypatch) -> None:
    def fake_detokenize(tokens: list[str]):
        if tokens == ["bad"]:
            return None
        mol = Chem.MolFromSmiles("CCO")
        conformer = Chem.Conformer(mol.GetNumAtoms())
        for atom_index in range(mol.GetNumAtoms()):
            conformer.SetAtomPosition(atom_index, (float(atom_index), 0.0, 0.0))
        mol.AddConformer(conformer)
        return mol

    monkeypatch.setattr("mtrl.generate.detokenize", fake_detokenize)
    summary = write_conformers(
        [["C", "C", "O"], ["bad"]],
        tmp_path,
        provenance={"seed": 7},
    )

    assert (tmp_path / "strings.amsr").read_text().splitlines() == ["CCO", "bad"]
    assert len([mol for mol in Chem.SDMolSupplier(str(tmp_path / "conformers.sdf")) if mol]) == 1
    assert summary["decoded_conformers"] == 1
    assert summary["decode_failures"] == 1
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
