import json

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from mtrl.box_score import score_sdf
from mtrl.config import BoxScoringConfig, DockingTarget
from mtrl.scoring import BoxDockingScore, TargetDockingScore


class FakePipeline:
    def __init__(self, result: BoxDockingScore) -> None:
        self.result = result
        self.seen = []

    def score_batch(self, mols):
        self.seen = mols
        return [self.result for _ in mols]


def _mol3d() -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles("CCOc1ccccc1"))
    assert AllChem.EmbedMolecule(mol, randomSeed=11) == 0
    mol.SetProp("_Name", "candidate_a")
    return mol


def _mol3d_from_smiles(smiles: str, name: str) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=11) == 0
    mol.SetProp("_Name", name)
    return mol


def test_score_sdf_keeps_partial_hits_and_writes_only_real_poses(tmp_path) -> None:
    input_sdf = tmp_path / "input.sdf"
    writer = Chem.SDWriter(str(input_sdf))
    writer.write(_mol3d())
    writer.close()

    targets = []
    for name in ("site_a", "site_b"):
        receptor = tmp_path / f"{name}.pdb"
        receptor.write_text("END\n")
        targets.append(
            DockingTarget(
                name=name,
                receptor_pdb=receptor,
                center=(1.0, 2.0, 3.0),
                size=(20.0, 20.0, 20.0),
            )
        )
    pose = _mol3d()
    result = BoxDockingScore(
        targets={
            "site_a": TargetDockingScore(
                target_name="site_a",
                cnn_affinity=6.75,
                rigid_vina_affinity=-6.1,
                rigid_conformer_rmsd=0.002,
                refined_conformer_rmsd=0.84,
                accepted=True,
                pose=pose,
            ),
            "site_b": TargetDockingScore(
                target_name="site_b",
                rejection_reason="PoseBusters failed for every docking pose",
            ),
        },
        accepted=True,
    )
    config = BoxScoringConfig(
        targets=tuple(targets),
        output_dir=tmp_path / "scored",
        target_failure_score=-1.0,
        accept_targets="any",
        qed_objective=True,
    )

    summary = score_sdf(input_sdf, config, pipeline=FakePipeline(result))

    assert summary["retained_molecules"] == 1
    assert summary["all_targets_valid"] == 0
    assert summary["per_target_valid_poses"] == {"site_a": 1, "site_b": 0}
    record = json.loads((config.output_dir / "scores.jsonl").read_text())
    assert record["accepted"]
    assert record["objectives"] == {
        "gnina_cnn_affinity__site_a": 6.75,
        "gnina_cnn_affinity__site_b": -1.0,
        "qed": pytest.approx(0.5832, abs=1e-4),
    }
    assert record["targets"]["site_a"]["rigid_vina_affinity"] == -6.1
    assert record["targets"]["site_a"]["rigid_conformer_rmsd"] == 0.002
    assert record["targets"]["site_a"]["refined_conformer_rmsd"] == 0.84
    assert (config.output_dir / "poses" / "site_a.sdf").is_file()
    assert not (config.output_dir / "poses" / "site_b.sdf").exists()


def test_score_sdf_applies_property_and_br_sascore_filters_before_docking(tmp_path) -> None:
    input_sdf = tmp_path / "input.sdf"
    writer = Chem.SDWriter(str(input_sdf))
    writer.write(_mol3d_from_smiles("CCCCCCCCCCCCCCCC", "greasy"))
    writer.write(_mol3d_from_smiles("CN(C)C1=C2CC(Cc3ccccc3)CCC2C=C1", "difficult"))
    writer.write(_mol3d_from_smiles("CC(OC1=CC=CC=C1C(O)=O)=O", "aspirin"))
    writer.close()

    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("END\n")
    target = DockingTarget(
        name="site_a",
        receptor_pdb=receptor,
        center=(1.0, 2.0, 3.0),
        size=(20.0, 20.0, 20.0),
    )
    pose = _mol3d()
    result = BoxDockingScore(
        targets={
            "site_a": TargetDockingScore(
                target_name="site_a",
                cnn_affinity=6.0,
                accepted=True,
                pose=pose,
            )
        },
        accepted=True,
    )
    pipeline = FakePipeline(result)
    config = BoxScoringConfig(
        targets=(target,),
        output_dir=tmp_path / "scored",
        rdkit_druglike_filter=True,
        max_br_sascore=5.0,
    )

    summary = score_sdf(input_sdf, config, pipeline=pipeline)

    assert summary["input_records"] == 3
    assert summary["retained_molecules"] == 1
    assert [mol.GetProp("_Name") for mol in pipeline.seen] == ["aspirin"]
    records = [
        json.loads(line) for line in (config.output_dir / "scores.jsonl").read_text().splitlines()
    ]
    assert records[0]["rejection_reason"].startswith("RDKit drug-likeness failed:")
    assert records[1]["rejection_reason"].startswith("BR-SAScore failed:")
    assert records[2]["accepted"]
