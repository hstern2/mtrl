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

    def score_batch(self, mols):
        return [self.result for _ in mols]


def _mol3d() -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles("CCOc1ccccc1"))
    assert AllChem.EmbedMolecule(mol, randomSeed=11) == 0
    mol.SetProp("_Name", "candidate_a")
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
