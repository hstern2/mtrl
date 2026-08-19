import io

from rdkit import Chem

from mtrl.generate import write_conformers


def test_write_conformers_streams_sdf_and_returns_counts(monkeypatch) -> None:
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
    output = io.StringIO()
    summary = write_conformers([["C", "C", "O"], ["bad"]], output)

    sdf = output.getvalue()
    assert sdf.count("$$$$") == 1
    assert ">  <AMSR>" in sdf
    assert "CCO" in sdf
    assert summary["decoded_conformers"] == 1
    assert summary["decode_failures"] == 1
