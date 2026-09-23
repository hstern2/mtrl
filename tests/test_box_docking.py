import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from trl.objectives.base import ScoredItem

from mtrl import DecodedAMSR
from mtrl.config import CONFIG_ENV, BoxScoringConfig, DockingTarget, load_docking_targets
from mtrl.objectives import BoxDockingObjectives
from mtrl.scoring import BoxDockingPipeline, BoxDockingScore, TargetDockingScore


def _mol3d(smiles: str = "CCO") -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    return mol


def _write_many(path: Path, mols: list[Chem.Mol]) -> None:
    writer = Chem.SDWriter(str(path))
    try:
        for mol in mols:
            writer.write(mol)
    finally:
        writer.close()


def _targets(tmp_path: Path) -> tuple[DockingTarget, ...]:
    receptors = []
    for name in ("q_open", "c_open", "r_open"):
        receptor = tmp_path / f"{name}.pdb"
        receptor.write_text("END\n")
        receptors.append(
            DockingTarget(
                name=name,
                receptor_pdb=receptor,
                center=(1.0, 2.0, 3.0),
                size=(20.0, 21.0, 22.0),
            )
        )
    return tuple(receptors)


def test_target_manifest_loads_relative_receptors(tmp_path) -> None:
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("END\n")
    manifest = tmp_path / "targets.json"
    manifest.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "name": "q_open",
                        "receptor_pdb": "receptor.pdb",
                        "center": [1, 2, 3],
                        "size": [20, 21, 22],
                        "exhaustiveness": 4,
                        "num_modes": 5,
                    }
                ]
            }
        )
    )

    target = load_docking_targets(manifest)[0]

    assert target.receptor_pdb == receptor.resolve()
    assert target.center == (1.0, 2.0, 3.0)
    assert target.size == (20.0, 21.0, 22.0)
    assert target.exhaustiveness == 4
    assert target.num_modes == 5


@pytest.mark.parametrize(
    ("target_update", "message"),
    [
        ({"name": 7}, "name must be a string"),
        ({"center": "123"}, "center must be an array of three numbers"),
        ({"size": [20, True, 20]}, "size must be an array of three numbers"),
        ({"exhaustiveness": 2.5}, "exhaustiveness must be a positive integer"),
        ({"num_modes": 0}, "num_modes must be a positive integer"),
    ],
)
def test_target_manifest_rejects_malformed_fields(tmp_path, target_update, message) -> None:
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("END\n")
    target = {
        "name": "site_a",
        "receptor_pdb": "receptor.pdb",
        "center": [1, 2, 3],
        "size": [20, 20, 20],
    }
    target.update(target_update)
    manifest = tmp_path / "targets.json"
    manifest.write_text(json.dumps({"targets": [target]}))

    with pytest.raises(ValueError, match=message):
        load_docking_targets(manifest)


def test_target_manifest_requires_target_objects(tmp_path) -> None:
    manifest = tmp_path / "targets.json"
    manifest.write_text(json.dumps({"targets": ["not-an-object"]}))

    with pytest.raises(ValueError, match="each docking target must be a JSON object"):
        load_docking_targets(manifest)


def test_best_affinity_is_selected_only_from_posebusters_passing_poses(tmp_path) -> None:
    output = tmp_path / "poses.sdf"
    poses = []
    for affinity, passed in ((9.0, False), (7.5, True), (6.0, True)):
        pose = _mol3d()
        pose.SetProp("CNNaffinity", str(affinity))
        pose.SetProp("CNNscore", "0.8")
        pose.SetProp("minimizedAffinity", "-7.2")
        pose.SetProp("rigid_vina_affinity", "-6.4")
        pose.SetProp("rigid_conformer_rmsd", "0.001")
        pose.SetProp("refined_conformer_rmsd", "0.91")
        pose.SetProp("posebusters_passed", str(passed))
        if not passed:
            pose.SetProp("posebusters_failed_checks", "volume_overlap_with_protein")
        poses.append(pose)
    _write_many(output, poses)

    result = BoxDockingPipeline._select_passing_pose("q_open", output, center=(1.0, 2.0, 3.0))

    assert result.accepted
    assert result.cnn_affinity == pytest.approx(7.5)
    assert result.pose_index == 2
    assert result.rigid_vina_affinity == pytest.approx(-6.4)
    assert result.rigid_conformer_rmsd == pytest.approx(0.001)
    assert result.refined_conformer_rmsd == pytest.approx(0.91)
    assert result.pose_centroid_distance is not None
    assert result.n_poses == 3
    assert result.n_passing_poses == 2
    assert result.posebusters_failures == {"volume_overlap_with_protein": 1}


def test_full_docking_command_uses_explicit_box_without_minimize(tmp_path) -> None:
    target = _targets(tmp_path)[0]
    config = BoxScoringConfig(targets=(target,), output_dir=tmp_path / "out")
    ligand = tmp_path / "ligand.sdf"
    output = tmp_path / "output.sdf"
    _write_many(ligand, [_mol3d()])
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        output.write_text("$$$$\n")
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    with (
        patch("mtrl.scoring.gnina.require"),
        patch("mtrl.scoring.busters.require"),
        patch("mtrl.scoring.subprocess.run", side_effect=fake_run),
    ):
        pipeline = BoxDockingPipeline(config)
        pipeline._run_gnina(target, ligand, output)

    command = seen["command"]
    assert "--center_x" in command
    assert "--size_z" in command
    assert "--num_modes" in command
    assert "--minimize" not in command
    assert "--autobox_ligand" not in command


def test_rigid_pdbqt_has_single_root_and_no_torsions(tmp_path) -> None:
    target = _targets(tmp_path)[0]
    config = BoxScoringConfig(
        targets=(target,),
        output_dir=tmp_path / "out",
        docking_mode="rigid-refine",
    )
    ligand = tmp_path / "ligand.sdf"
    output = tmp_path / "ligand_rigid.pdbqt"
    _write_many(ligand, [_mol3d()])
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        raw_output = Path(command[command.index("-O") + 1])
        raw_output.write_text(
            "REMARK  Name = ligand\n"
            "ATOM      1  C   UNL     1       1.000   2.000   3.000  0.00  0.00    +0.000 C\n"
            "ATOM      2  O   UNL     1       2.000   2.000   3.000  0.00  0.00    +0.000 OA\n"
            "TORSDOF 1\n"
        )
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    with (
        patch("mtrl.scoring.gnina.require"),
        patch("mtrl.scoring.busters.require"),
        patch("mtrl.scoring.shutil.which", return_value="/usr/bin/obabel"),
        patch("mtrl.scoring.subprocess.run", side_effect=fake_run),
    ):
        BoxDockingPipeline(config)._prepare_rigid_ligand(ligand, output)

    command = seen["command"]
    assert "-xr" in command
    assert "-xh" in command
    rigid_pdbqt = output.read_text()
    rigid_lines = rigid_pdbqt.splitlines()
    assert rigid_lines.count("ROOT") == 1
    assert rigid_lines.count("ENDROOT") == 1
    assert rigid_pdbqt.endswith("TORSDOF 0\n")
    assert "BRANCH" not in rigid_pdbqt
    assert "TORSDOF 1" not in rigid_pdbqt


def test_rigid_refine_commands_disable_cnn_then_minimize_and_rescore(tmp_path) -> None:
    target = _targets(tmp_path)[0]
    config = BoxScoringConfig(
        targets=(target,),
        output_dir=tmp_path / "out",
        docking_mode="rigid-refine",
        gnina_timeout_seconds=123,
    )
    ligand = tmp_path / "ligand.pdbqt"
    rigid_output = tmp_path / "rigid.sdf"
    refined_input = tmp_path / "rigid_pose.sdf"
    refined_output = tmp_path / "refined.sdf"
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        assert kwargs["timeout"] == 123
        Path(command[command.index("--out") + 1]).write_text("$$$$\n")
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    with (
        patch("mtrl.scoring.gnina.require"),
        patch("mtrl.scoring.busters.require"),
        patch("mtrl.scoring.shutil.which", return_value="/usr/bin/obabel"),
        patch("mtrl.scoring.subprocess.run", side_effect=fake_run),
    ):
        pipeline = BoxDockingPipeline(config)
        pipeline._run_gnina(target, ligand, rigid_output, cnn_scoring="none")
        pipeline._run_gnina_minimize(target, refined_input, refined_output)

    rigid_command, refine_command = commands
    assert rigid_command[rigid_command.index("--cnn_scoring") + 1] == "none"
    assert rigid_command[rigid_command.index("--pose_sort_order") + 1] == "Energy"
    assert "--minimize" not in rigid_command
    assert "--center_x" in rigid_command
    assert "--minimize" in refine_command
    assert refine_command[refine_command.index("--cnn_scoring") + 1] == "rescore"
    assert "--autobox_ligand" not in refine_command
    assert "--center_x" in refine_command


@pytest.mark.parametrize(
    ("minimize", "message"),
    [(False, "gnina exceeded"), (True, "gnina --minimize exceeded")],
)
def test_gnina_timeout_reports_the_failed_stage(tmp_path, minimize, message) -> None:
    target = _targets(tmp_path)[0]
    config = BoxScoringConfig(
        targets=(target,),
        output_dir=tmp_path / "out",
        gnina_timeout_seconds=2,
    )
    ligand = tmp_path / "ligand.sdf"
    output = tmp_path / "output.sdf"

    with (
        patch("mtrl.scoring.gnina.require"),
        patch("mtrl.scoring.busters.require"),
        patch(
            "mtrl.scoring.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["gnina"], 2),
        ),
    ):
        pipeline = BoxDockingPipeline(config)
        with pytest.raises(RuntimeError, match=message):
            if minimize:
                pipeline._run_gnina_minimize(target, ligand, output)
            else:
                pipeline._run_gnina(target, ligand, output)


def test_rigid_poses_are_refined_in_one_batch_and_annotated(tmp_path) -> None:
    target = _targets(tmp_path)[0]
    config = BoxScoringConfig(
        targets=(target,),
        output_dir=tmp_path / "out",
        docking_mode="rigid-refine",
    )
    decoded = _mol3d("CCCO")
    rigid_output = tmp_path / "rigid.sdf"
    refined_output = tmp_path / "refined.sdf"
    rigid_poses = [Chem.Mol(decoded), Chem.Mol(decoded)]
    rigid_poses[0].SetProp("minimizedAffinity", "-5.1")
    rigid_poses[1].SetProp("minimizedAffinity", "-4.8")
    _write_many(rigid_output, rigid_poses)

    def fake_minimize(target_arg, ligand_arg, output_arg):
        assert target_arg == target
        assert ligand_arg == rigid_output
        refined_poses = [Chem.Mol(decoded), Chem.Mol(decoded)]
        refined_poses[0].SetProp("CNNaffinity", "6.2")
        refined_poses[1].SetProp("CNNaffinity", "5.9")
        _write_many(output_arg, refined_poses)

    with (
        patch("mtrl.scoring.gnina.require"),
        patch("mtrl.scoring.busters.require"),
        patch("mtrl.scoring.shutil.which", return_value="/usr/bin/obabel"),
        patch.object(
            BoxDockingPipeline,
            "_run_gnina_minimize",
            side_effect=fake_minimize,
        ) as minimize,
    ):
        BoxDockingPipeline(config)._refine_rigid_poses(
            target, decoded, rigid_output, refined_output
        )

    assert minimize.call_count == 1
    refined = [
        pose for pose in Chem.SDMolSupplier(str(refined_output), removeHs=False) if pose is not None
    ]
    assert len(refined) == 2
    assert refined[0].GetIntProp("rigid_pose_index") == 1
    assert refined[1].GetDoubleProp("rigid_vina_affinity") == pytest.approx(-4.8)
    assert refined[0].GetDoubleProp("rigid_conformer_rmsd") < 0.01
    assert refined[0].GetDoubleProp("refined_conformer_rmsd") < 0.01


def test_box_config_round_trips_rigid_refine_mode(tmp_path, monkeypatch) -> None:
    config = BoxScoringConfig(
        targets=(_targets(tmp_path)[0],),
        output_dir=tmp_path / "out",
        docking_mode="rigid-refine",
        gnina_timeout_seconds=321,
        qed_objective=True,
        rdkit_druglike_filter=True,
        max_br_sascore=5.0,
        posebusters_config="dock-fast",
        posebusters_timeout_seconds=123,
    )

    monkeypatch.setenv(CONFIG_ENV, json.dumps(config.to_dict()))
    restored = BoxScoringConfig.from_env()

    assert restored.docking_mode == "rigid-refine"
    assert restored.gnina_timeout_seconds == 321
    assert restored.qed_objective is True
    assert restored.rdkit_druglike_filter is True
    assert restored.max_br_sascore == 5.0
    assert restored.posebusters_config == "dock-fast"
    assert restored.posebusters_timeout_seconds == 123
    assert restored.targets == config.targets
    assert os.environ[CONFIG_ENV]


def test_box_config_rejects_invalid_br_sascore_cutoff(tmp_path) -> None:
    config = BoxScoringConfig(
        targets=(_targets(tmp_path)[0],),
        output_dir=tmp_path / "out",
        max_br_sascore=0.0,
    )

    with pytest.raises(ValueError, match="max_br_sascore must be between 1 and 10"):
        config.validate()


@pytest.mark.parametrize(
    ("accept_targets", "expected"),
    (("any", True), ("all", False)),
)
def test_pipeline_target_acceptance_policy(tmp_path, accept_targets, expected) -> None:
    targets = _targets(tmp_path)
    config = BoxScoringConfig(
        targets=targets,
        output_dir=tmp_path / "out",
        accept_targets=accept_targets,
    )
    passing = TargetDockingScore(
        target_name="q_open",
        cnn_affinity=6.0,
        accepted=True,
        pose=_mol3d(),
    )
    failed = TargetDockingScore(
        target_name="failed",
        rejection_reason="PoseBusters failed for every docking pose",
    )
    with (
        patch("mtrl.scoring.gnina.require"),
        patch("mtrl.scoring.busters.require"),
        patch.object(BoxDockingPipeline, "_run_gnina"),
        patch.object(BoxDockingPipeline, "_run_posebusters"),
        patch.object(
            BoxDockingPipeline,
            "_select_passing_pose",
            side_effect=[passing, failed, failed],
        ),
    ):
        result = BoxDockingPipeline(config)._score_one(_mol3d(), "candidate")

    assert result.accepted is expected


def test_posebusters_runs_in_bounded_child_process(tmp_path) -> None:
    target = _targets(tmp_path)[0]
    poses = tmp_path / "poses.sdf"
    _write_many(poses, [_mol3d()])
    config = BoxScoringConfig(
        targets=(target,),
        output_dir=tmp_path / "out",
        posebusters_timeout_seconds=47,
    )

    with (
        patch("mtrl.scoring.gnina.require"),
        patch("mtrl.scoring.busters.require"),
        patch("mtrl.scoring.subprocess.run") as run,
    ):
        run.return_value.returncode = 0
        BoxDockingPipeline(config)._run_posebusters(target.receptor_pdb, poses)

    command = run.call_args.args[0]
    assert command[0] == sys.executable
    assert command[-3:] == ["dock", str(target.receptor_pdb), str(poses)]
    assert run.call_args.kwargs["timeout"] == 47


def test_posebusters_timeout_fails_target_evaluation(tmp_path) -> None:
    target = _targets(tmp_path)[0]
    poses = tmp_path / "poses.sdf"
    _write_many(poses, [_mol3d()])
    config = BoxScoringConfig(
        targets=(target,),
        output_dir=tmp_path / "out",
        posebusters_timeout_seconds=2,
    )

    with (
        patch("mtrl.scoring.gnina.require"),
        patch("mtrl.scoring.busters.require"),
        patch(
            "mtrl.scoring.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["python"], 2),
        ),
    ):
        with pytest.raises(RuntimeError, match="PoseBusters exceeded its 2-second timeout"):
            BoxDockingPipeline(config)._run_posebusters(target.receptor_pdb, poses)


class FakeBoxPipeline:
    def __init__(self, result: BoxDockingScore) -> None:
        self.result = result
        self.seen = []

    def score_batch(self, mols):
        self.seen = mols
        return [self.result for _ in mols]


class FakeLilly:
    def __init__(self, accepted: list[bool]) -> None:
        self.accepted = accepted
        self.seen = []

    def accept_batch(self, mols):
        self.seen = mols
        return self.accepted[: len(mols)]


def test_box_objectives_apply_lilly_then_emit_one_score_per_target(tmp_path) -> None:
    targets = _targets(tmp_path)
    pose = _mol3d()
    result = BoxDockingScore(
        targets={
            target.name: TargetDockingScore(
                target_name=target.name,
                cnn_affinity=6.0 + index,
                pose_index=1,
                n_poses=4,
                n_passing_poses=3,
                accepted=True,
                pose=Chem.Mol(pose),
            )
            for index, target in enumerate(targets)
        },
        accepted=True,
    )
    pipeline = FakeBoxPipeline(result)
    lilly = FakeLilly([False, True])
    decoded = [
        DecodedAMSR(Chem.MolFromSmiles("CC"), {}),
        DecodedAMSR(Chem.MolFromSmiles("CCC"), {}),
    ]
    suite = BoxDockingObjectives(
        BoxScoringConfig(
            targets=targets,
            output_dir=tmp_path / "out",
            lilly_medchem_rules=True,
        ),
        decode_fn=lambda tokens: decoded[int(tokens[0])],
        conformer_fn=lambda candidate: pose,
        pipeline=pipeline,
        lilly_filter=lilly,
    )

    scored = suite.evaluate([["0"], ["1"]])

    assert not scored[0].valid
    assert scored[0].rejection_reason == "Lilly Medchem Rules (-relaxed) failed"
    assert scored[1].valid
    assert scored[1].scores == {
        "gnina_cnn_affinity__q_open": 6.0,
        "gnina_cnn_affinity__c_open": 7.0,
        "gnina_cnn_affinity__r_open": 8.0,
    }
    assert len(pipeline.seen) == 1
    assert (tmp_path / "out" / "best" / "overall" / "q_open.sdf").is_file()
    record = json.loads((tmp_path / "out" / "scores.jsonl").read_text().splitlines()[1])
    assert record["targets"]["q_open"]["n_passing_poses"] == 3
    summary = (tmp_path / "out" / "summary.txt").read_text()
    assert "Target acceptance policy: any" in summary
    assert "Accepted by target policy: 1" in summary


def test_box_rewards_use_all_target_affinities_as_pareto_axes(tmp_path) -> None:
    targets = _targets(tmp_path)
    suite = BoxDockingObjectives(
        BoxScoringConfig(
            targets=targets,
            output_dir=tmp_path / "out",
            target_failure_score=-2.0,
            accept_targets="any",
        ),
        pipeline=FakeBoxPipeline(BoxDockingScore()),
    )
    names = [objective.name for objective in suite.objectives]
    scored = [
        ScoredItem(token_ids=[], scores=dict(zip(names, values, strict=True)))
        for values in ((8.0, 2.0, 2.0), (2.0, 8.0, 2.0), (2.0, 2.0, 8.0), (1.0, 1.0, 1.0))
    ]

    rewards = suite.get_rewards(scored)

    assert all(rewards[index] > rewards[3] for index in range(3))


def test_qed_is_an_optional_independent_box_objective(tmp_path) -> None:
    targets = _targets(tmp_path)
    pose = _mol3d()
    decoded = DecodedAMSR(Chem.MolFromSmiles("CCO"), {})
    result = BoxDockingScore(
        targets={
            target.name: TargetDockingScore(
                target_name=target.name,
                cnn_affinity=5.0,
                accepted=True,
                pose=Chem.Mol(pose),
            )
            for target in targets
        },
        accepted=True,
    )
    suite = BoxDockingObjectives(
        BoxScoringConfig(
            targets=targets,
            output_dir=tmp_path / "out",
            qed_objective=True,
        ),
        decode_fn=lambda _: decoded,
        conformer_fn=lambda _: pose,
        pipeline=FakeBoxPipeline(result),
    )

    scored = suite.evaluate([["candidate"]])[0]

    assert [objective.name for objective in suite.objectives] == [
        "gnina_cnn_affinity__q_open",
        "gnina_cnn_affinity__c_open",
        "gnina_cnn_affinity__r_open",
        "qed",
    ]
    assert scored.scores["qed"] == pytest.approx(0.4068, abs=1e-4)
    record = json.loads((tmp_path / "out" / "scores.jsonl").read_text())
    assert record["objectives"]["qed"] == pytest.approx(0.4068, abs=1e-4)


def test_qed_changes_four_objective_pareto_reward(tmp_path) -> None:
    suite = BoxDockingObjectives(
        BoxScoringConfig(
            targets=_targets(tmp_path),
            output_dir=tmp_path / "out",
            qed_objective=True,
        ),
        pipeline=FakeBoxPipeline(BoxDockingScore()),
    )
    names = [objective.name for objective in suite.objectives]
    scored = [
        ScoredItem(token_ids=[], scores=dict(zip(names, values, strict=True)))
        for values in ((5.0, 5.0, 5.0, 0.8), (5.0, 5.0, 5.0, 0.2))
    ]

    rewards = suite.get_rewards(scored)

    assert rewards[0] > rewards[1]


def test_one_passing_target_keeps_molecule_and_floors_failed_objectives(tmp_path) -> None:
    targets = _targets(tmp_path)
    pose = _mol3d()
    result = BoxDockingScore(
        targets={
            "q_open": TargetDockingScore(
                target_name="q_open",
                cnn_affinity=7.25,
                accepted=True,
                pose=Chem.Mol(pose),
            ),
            "c_open": TargetDockingScore(
                target_name="c_open",
                rejection_reason="PoseBusters failed for every docking pose",
            ),
            "r_open": TargetDockingScore(
                target_name="r_open",
                rejection_reason="PoseBusters failed for every docking pose",
            ),
        },
        accepted=True,
    )
    suite = BoxDockingObjectives(
        BoxScoringConfig(
            targets=targets,
            output_dir=tmp_path / "out",
            target_failure_score=-2.0,
            accept_targets="any",
        ),
        decode_fn=lambda _: DecodedAMSR(Chem.MolFromSmiles("CCC"), {}),
        conformer_fn=lambda _: pose,
        pipeline=FakeBoxPipeline(result),
    )

    scored = suite.evaluate([["candidate"]])[0]

    assert scored.valid
    assert scored.scores == {
        "gnina_cnn_affinity__q_open": pytest.approx(7.25),
        "gnina_cnn_affinity__c_open": -2.0,
        "gnina_cnn_affinity__r_open": -2.0,
    }
    overall = tmp_path / "out" / "best" / "overall"
    assert (overall / "q_open.sdf").is_file()
    assert not (overall / "c_open.sdf").exists()
    assert not (overall / "r_open.sdf").exists()
    record = json.loads((tmp_path / "out" / "scores.jsonl").read_text())
    assert record["accepted"]
    assert not record["targets"]["c_open"]["accepted"]


def test_pose_names_are_stable_across_target_files_with_partial_hits(tmp_path) -> None:
    targets = _targets(tmp_path)[:2]
    pose = _mol3d()
    results = [
        BoxDockingScore(
            targets={
                "q_open": TargetDockingScore(
                    target_name="q_open", rejection_reason="no passing pose"
                ),
                "c_open": TargetDockingScore(
                    target_name="c_open", cnn_affinity=5.0, accepted=True, pose=Chem.Mol(pose)
                ),
            },
            accepted=True,
        ),
        BoxDockingScore(
            targets={
                target.name: TargetDockingScore(
                    target_name=target.name,
                    cnn_affinity=6.0,
                    accepted=True,
                    pose=Chem.Mol(pose),
                )
                for target in targets
            },
            accepted=True,
        ),
    ]

    class ResultPipeline:
        def score_batch(self, mols):
            assert len(mols) == len(results)
            return results

    decoded = [
        DecodedAMSR(Chem.MolFromSmiles("CC"), {}),
        DecodedAMSR(Chem.MolFromSmiles("CCC"), {}),
    ]
    suite = BoxDockingObjectives(
        BoxScoringConfig(targets=targets, output_dir=tmp_path / "out"),
        decode_fn=lambda tokens: decoded[int(tokens[0])],
        conformer_fn=lambda _: pose,
        pipeline=ResultPipeline(),
    )

    suite.evaluate([["0"], ["1"]])

    output = tmp_path / "out" / "generations" / "generation_000001"
    q_names = [
        mol.GetProp("_Name")
        for mol in Chem.SDMolSupplier(str(output / "q_open.sdf"), removeHs=False)
        if mol is not None
    ]
    c_names = [
        mol.GetProp("_Name")
        for mol in Chem.SDMolSupplier(str(output / "c_open.sdf"), removeHs=False)
        if mol is not None
    ]
    assert q_names == ["generation_000001_molecule_0002"]
    assert c_names == [
        "generation_000001_molecule_0001",
        "generation_000001_molecule_0002",
    ]
    score_records = [
        json.loads(line) for line in (tmp_path / "out" / "scores.jsonl").read_text().splitlines()
    ]
    assert [record["name"] for record in score_records] == c_names
