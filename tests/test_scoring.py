from pathlib import Path
from unittest.mock import patch

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from mtrl.config import ScoringConfig
from mtrl.scoring import StructureScoringPipeline, minimized_pose_rmsd


def _mol3d() -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    return mol


def _translate(mol: Chem.Mol, distance: float) -> Chem.Mol:
    moved = Chem.Mol(mol)
    conformer = moved.GetConformer()
    for index in range(moved.GetNumAtoms()):
        position = conformer.GetAtomPosition(index)
        conformer.SetAtomPosition(index, (position.x + distance, position.y, position.z))
    return moved


def _write(path: Path, mol: Chem.Mol) -> None:
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def test_minimized_pose_rmsd_is_in_place_and_heavy_atom_only() -> None:
    aligned = _mol3d()
    moved = _translate(aligned, 0.75)
    assert minimized_pose_rmsd(aligned, moved) == pytest.approx(0.75, abs=1e-6)


@pytest.mark.parametrize(
    ("movement", "posebusters_passed", "accepted", "reason"),
    [
        (0.4, True, True, ""),
        (1.2, True, False, "minimization RMSD"),
        (0.4, False, False, "PoseBusters failed"),
    ],
)
def test_structure_pipeline_hard_gates(
    tmp_path, monkeypatch, movement, posebusters_passed, accepted, reason
) -> None:
    receptor = tmp_path / "receptor.pdb"
    reference = tmp_path / "reference.sdf"
    receptor.write_text("END\n")
    _write(reference, _mol3d())
    scratch = tmp_path / "system-temp"
    scratch.mkdir()
    monkeypatch.setattr("tempfile.tempdir", str(scratch))
    config = ScoringConfig(
        receptor_pdb=receptor,
        reference_sdf=reference,
        output_dir=tmp_path / "output",
        max_minimized_rmsd=1.0,
    )

    def fake_roshambo(query, candidate, output):
        mol = next(m for m in Chem.SDMolSupplier(str(candidate), removeHs=False) if m)
        mol.SetProp("tanimoto_combination", "0.77")
        _write(output, mol)

    def fake_gnina(path):
        mol = next(m for m in Chem.SDMolSupplier(str(path), removeHs=False) if m)
        mol = _translate(mol, movement)
        mol.SetProp("CNNaffinity", "7.25")
        _write(path, mol)

    def fake_busters(pdb, path):
        mol = next(m for m in Chem.SDMolSupplier(str(path), removeHs=False) if m)
        mol.SetProp("posebusters_passed", str(posebusters_passed))
        _write(path, mol)

    with (
        patch("mtrl.scoring.roshambo.require"),
        patch("mtrl.scoring.gnina.require"),
        patch("mtrl.scoring.busters.require"),
        patch("mtrl.scoring.roshambo.run", side_effect=fake_roshambo),
        patch("mtrl.scoring.busters.run", side_effect=fake_busters),
    ):
        pipeline = StructureScoringPipeline(config)
        with patch.object(pipeline, "_run_gnina", side_effect=fake_gnina):
            score = pipeline.score_batch([_mol3d()])[0]

    assert score.accepted is accepted
    assert score.cnn_affinity == pytest.approx(7.25)
    assert score.roshambo_tanimoto_combo == pytest.approx(0.77)
    assert score.minimized_rmsd == pytest.approx(movement, abs=1e-4)
    assert reason in score.rejection_reason
    assert (score.minimized_mol is not None) is accepted
    assert not any(scratch.iterdir())
