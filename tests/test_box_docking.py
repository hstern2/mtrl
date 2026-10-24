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
    assert target.num_modes == 5


@pytest.mark.parametrize(
    ("target_update", "message"),
    [
        ({"name": 7}, "name must be a string"),
        ({"center": "123"}, "center must be an array of three numbers"),
        ({"size": [20, True, 20]}, "size must be an array of three numbers"),
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

    result = BoxDockingPipeline._select_passing_pose(
        "q_open",
        output,
        center=(1.0, 2.0, 3.0),
        retain_passing_poses=True,
    )

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
    assert [pose.GetIntProp("_mtrl_pose_index") for pose in result.passing_poses] == [2, 3]


def test_aev_plig_rescores_all_passing_poses_and_selects_its_own_best(tmp_path) -> None:
    target = _targets(tmp_path)[0]
    first = _mol3d("CCO")
    first.SetIntProp("_mtrl_pose_index", 2)
    second = _mol3d("CCO")
    second.SetIntProp("_mtrl_pose_index", 5)
    target_result = TargetDockingScore(
        target_name=target.name,
        cnn_affinity=8.0,
        pose_index=2,
        accepted=True,
        pose=Chem.Mol(first),
        passing_poses=[first, second],
    )
    result = BoxDockingScore(targets={target.name: target_result}, accepted=True)
    config = BoxScoringConfig(
        targets=(target,),
        output_dir=tmp_path / "out",
        aev_plig_dir=tmp_path / "AEV-PLIG",
    )

    with (
        patch.object(BoxScoringConfig, "validate"),
        patch("mtrl.scoring.gnina.require"),
        patch("mtrl.scoring.busters.require"),
        patch(
            "mtrl.scoring.AEVPligScorer.score",
            return_value={"m0_t0_p2": 6.1, "m0_t0_p5": 7.4},
        ) as score,
    ):
        pipeline = BoxDockingPipeline(config)
        pipeline._apply_aev_plig([result])

    assert score.call_count == 1
    assert target_result.cnn_affinity == 8.0
    assert target_result.pose_index == 2
    assert target_result.aev_plig_affinity == pytest.approx(7.4)
    assert target_result.aev_plig_pose_index == 5
    assert target_result.aev_plig_pose is not None
    assert not target_result.passing_poses


def test_rigid_pdbqt_has_single_root_and_no_torsions(tmp_path) -> None:
    target = _targets(tmp_path)[0]
    config = BoxScoringConfig(targets=(target,), output_dir=tmp_path / "out")
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


def test_unidock_batches_ligands_in_explicit_box(tmp_path) -> None:
    target = _targets(tmp_path)[0]
    config = BoxScoringConfig(
        targets=(target,),
        output_dir=tmp_path / "out",
        unidock_executable="unidock-custom",
        unidock_search_mode="fast",
        unidock_seed=17,
        unidock_timeout_seconds=123,
    )
    ligands = [tmp_path / "a.pdbqt", tmp_path / "b.pdbqt"]
    output = tmp_path / "poses"
    output.mkdir()
    receptor_pdbqt = tmp_path / "receptor.pdbqt"
    receptor_pdbqt.touch()
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        assert kwargs["timeout"] == 123
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    with (
        patch("mtrl.scoring.gnina.require"),
        patch("mtrl.scoring.busters.require"),
        patch("mtrl.scoring.shutil.which", return_value="/tools/unidock"),
        patch("mtrl.scoring.subprocess.run", side_effect=fake_run),
        patch.object(BoxDockingPipeline, "_prepare_receptor", return_value=receptor_pdbqt),
    ):
        pipeline = BoxDockingPipeline(config)
        pipeline._run_unidock(target, ligands, output)

    command = seen["command"]
    assert command[0] == "/tools/unidock"
    batch_index = command.index("--gpu_batch")
    assert command[batch_index + 1 : batch_index + 3] == [str(path) for path in ligands]
    assert command[command.index("--receptor") + 1] == str(receptor_pdbqt)
    assert command[command.index("--search_mode") + 1] == "fast"
    assert command[command.index("--seed") + 1] == "17"
    assert command[command.index("--energy_range") + 1] == "100"
    assert "--center_x" in command
    assert "--size_z" in command
    assert command[command.index("--num_modes") + 1] == str(target.num_modes)


def test_score_batch_runs_unidock_once_per_target(tmp_path) -> None:
    targets = _targets(tmp_path)
    config = BoxScoringConfig(targets=targets, output_dir=tmp_path / "out")

    def prepare_ligand(_self, _input, output):
        output.write_text("TORSDOF 0\n")

    def convert_pose(_self, template, _input, output):
        _write_many(output, [template])

    def refine(task):
        index, target, _molecule, _rigid_sdf, _refined_sdf = task
        return (
            index,
            target.name,
            TargetDockingScore(target_name=target.name, cnn_affinity=6.0, accepted=True),
        )

    with (
        patch("mtrl.scoring.gnina.require"),
        patch("mtrl.scoring.busters.require"),
        patch.object(BoxDockingPipeline, "_prepare_rigid_ligand", prepare_ligand),
        patch.object(BoxDockingPipeline, "_run_unidock") as run_unidock,
        patch.object(BoxDockingPipeline, "_unidock_poses_to_sdf", convert_pose),
        patch.object(BoxDockingPipeline, "_refine_and_check", side_effect=refine),
    ):
        results = BoxDockingPipeline(config, workers=1).score_batch([_mol3d(), _mol3d("CCN")])

    assert run_unidock.call_count == len(targets)
    assert all(len(call.args[1]) == 2 for call in run_unidock.call_args_list)
    assert all(result.accepted for result in results)


def test_unidock_timeout_is_bounded(tmp_path) -> None:
    target = _targets(tmp_path)[0]
    config = BoxScoringConfig(
        targets=(target,),
        output_dir=tmp_path / "out",
        unidock_timeout_seconds=2,
    )
    ligand = tmp_path / "ligand.pdbqt"
    output = tmp_path / "output"
    output.mkdir()

    with (
        patch("mtrl.scoring.gnina.require"),
        patch("mtrl.scoring.busters.require"),
        patch("mtrl.scoring.shutil.which", return_value="/tools/unidock"),
        patch.object(
            BoxDockingPipeline, "_prepare_receptor", return_value=tmp_path / "receptor.pdbqt"
        ),
        patch(
            "mtrl.scoring.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["unidock"], 2),
        ),
    ):
        pipeline = BoxDockingPipeline(config)
        with pytest.raises(RuntimeError, match="Uni-Dock exceeded its 2-second timeout"):
            pipeline._run_unidock(target, [ligand], output)


def test_unidock_output_uses_original_topology_and_coordinates(tmp_path) -> None:
    template = _mol3d("CO")
    pdbqt = tmp_path / "poses.pdbqt"
    sdf = tmp_path / "poses.sdf"
    lines = ["MODEL 1", "REMARK VINA RESULT:    -5.750      0.000      0.000"]
    conformer = template.GetConformer()
    for atom in template.GetAtoms():
        index = atom.GetIdx()
        position = conformer.GetAtomPosition(index)
        lines.append(
            f"ATOM  {index + 1:5d}  {atom.GetSymbol():<2}  UNL     1    "
            f"{position.x + 4:8.3f}{position.y - 2:8.3f}{position.z + 1:8.3f}"
            "  0.00  0.00    +0.000 C"
        )
    lines.append("ENDMDL")
    pdbqt.write_text("\n".join(lines) + "\n")

    BoxDockingPipeline._unidock_poses_to_sdf(template, pdbqt, sdf)

    pose = next(mol for mol in Chem.SDMolSupplier(str(sdf), removeHs=False) if mol is not None)
    assert Chem.MolToSmiles(pose) == Chem.MolToSmiles(template)
    assert pose.GetDoubleProp("rigid_vina_affinity") == pytest.approx(-5.75)
    assert pose.GetDoubleProp("rigid_conformer_rmsd") < 0.002
    assert pose.GetConformer().GetAtomPosition(0).x == pytest.approx(
        template.GetConformer().GetAtomPosition(0).x + 4, abs=0.001
    )


def test_gnina_minimizes_and_rescores_unidock_poses(tmp_path) -> None:
    target = _targets(tmp_path)[0]
    config = BoxScoringConfig(
        targets=(target,), output_dir=tmp_path / "out", gnina_cpu=1, gnina_timeout_seconds=123
    )
    ligand = tmp_path / "rigid.sdf"
    output = tmp_path / "refined.sdf"
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        assert kwargs["timeout"] == 123
        output.write_text("$$$$\n")
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    with (
        patch("mtrl.scoring.gnina.require"),
        patch("mtrl.scoring.busters.require"),
        patch("mtrl.scoring.subprocess.run", side_effect=fake_run),
    ):
        BoxDockingPipeline(config)._run_gnina_minimize(target, ligand, output)

    command = seen["command"]
    assert "--minimize" in command
    assert command[command.index("--cnn_scoring") + 1] == "rescore"
    assert command[command.index("--cpu") + 1] == "1"
    assert "--center_x" in command


def test_rigid_poses_are_refined_in_one_batch_and_annotated(tmp_path) -> None:
    target = _targets(tmp_path)[0]
    config = BoxScoringConfig(targets=(target,), output_dir=tmp_path / "out")
    decoded = _mol3d("CCCO")
    rigid_output = tmp_path / "rigid.sdf"
    refined_output = tmp_path / "refined.sdf"
    rigid_poses = [Chem.Mol(decoded), Chem.Mol(decoded)]
    rigid_poses[0].SetProp("rigid_vina_affinity", "-5.1")
    rigid_poses[1].SetProp("rigid_vina_affinity", "-4.8")
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


def test_box_config_round_trips_unidock_settings(tmp_path, monkeypatch) -> None:
    config = BoxScoringConfig(
        targets=(_targets(tmp_path)[0],),
        output_dir=tmp_path / "out",
        unidock_executable="/tools/unidock",
        unidock_search_mode="balance",
        unidock_seed=91,
        unidock_timeout_seconds=222,
        gnina_cpu=2,
        gnina_timeout_seconds=321,
        rdkit_druglike_filter=True,
        muegge_filter=True,
        brenk_filter=True,
        pfizer_3_75_filter=True,
        max_br_sascore=5.0,
        max_clogp=4.0,
        posebusters_config="dock-fast",
        posebusters_timeout_seconds=123,
    )

    monkeypatch.setenv(CONFIG_ENV, json.dumps(config.to_dict()))
    restored = BoxScoringConfig.from_env()

    assert restored.unidock_executable == "/tools/unidock"
    assert restored.unidock_search_mode == "balance"
    assert restored.unidock_seed == 91
    assert restored.unidock_timeout_seconds == 222
    assert restored.gnina_cpu == 2
    assert restored.gnina_timeout_seconds == 321
    assert restored.rdkit_druglike_filter is True
    assert restored.muegge_filter is True
    assert restored.brenk_filter is True
    assert restored.pfizer_3_75_filter is True
    assert restored.max_br_sascore == 5.0
    assert restored.max_clogp == 4.0
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


def test_box_config_rejects_nonfinite_clogp_cutoff(tmp_path) -> None:
    config = BoxScoringConfig(
        targets=(_targets(tmp_path)[0],),
        output_dir=tmp_path / "out",
        max_clogp=float("inf"),
    )

    with pytest.raises(ValueError, match="max_clogp must be finite"):
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
    with (
        patch("mtrl.scoring.gnina.require"),
        patch("mtrl.scoring.busters.require"),
    ):
        pipeline = BoxDockingPipeline(config)
        result = pipeline._finish_score(
            {
                "q_open": TargetDockingScore(
                    target_name="q_open",
                    cnn_affinity=6.0,
                    accepted=True,
                    pose=_mol3d(),
                ),
                "c_open": TargetDockingScore(
                    target_name="c_open",
                    rejection_reason="PoseBusters failed for every docking pose",
                ),
                "r_open": TargetDockingScore(
                    target_name="r_open",
                    rejection_reason="PoseBusters failed for every docking pose",
                ),
            }
        )

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


def test_aev_plig_adds_one_objective_and_pose_file_per_target(tmp_path) -> None:
    targets = _targets(tmp_path)
    cnn_pose = _mol3d("CCO")
    aev_pose = _mol3d("CCO")
    result = BoxDockingScore(
        targets={
            target.name: TargetDockingScore(
                target_name=target.name,
                cnn_affinity=6.0 + index,
                aev_plig_affinity=7.0 + index,
                pose_index=1,
                aev_plig_pose_index=2,
                accepted=True,
                pose=Chem.Mol(cnn_pose),
                aev_plig_pose=Chem.Mol(aev_pose),
            )
            for index, target in enumerate(targets)
        },
        accepted=True,
    )
    suite = BoxDockingObjectives(
        BoxScoringConfig(
            targets=targets,
            output_dir=tmp_path / "out",
            aev_plig_dir=tmp_path / "AEV-PLIG",
        ),
        decode_fn=lambda _: DecodedAMSR(Chem.MolFromSmiles("CCO"), {}),
        conformer_fn=lambda _: cnn_pose,
        pipeline=FakeBoxPipeline(result),
    )

    scored = suite.evaluate([["candidate"]])[0]

    assert list(scored.scores) == [
        "gnina_cnn_affinity__q_open",
        "gnina_cnn_affinity__c_open",
        "gnina_cnn_affinity__r_open",
        "aev_plig_affinity__q_open",
        "aev_plig_affinity__c_open",
        "aev_plig_affinity__r_open",
    ]
    assert scored.scores["aev_plig_affinity__r_open"] == pytest.approx(9.0)
    overall = tmp_path / "out" / "best" / "overall"
    assert (overall / "q_open.sdf").is_file()
    aev_output = overall / "q_open__aev_plig.sdf"
    assert aev_output.is_file()
    output_pose = next(
        mol for mol in Chem.SDMolSupplier(str(aev_output), removeHs=False) if mol is not None
    )
    assert output_pose.GetProp("pose_selection_method") == "aev_plig_affinity"


def test_box_objectives_score_all_distinct_emitted_conformers(tmp_path) -> None:
    targets = _targets(tmp_path)
    pose = _mol3d()
    result = BoxDockingScore(
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
    )
    pipeline = FakeBoxPipeline(result)
    decoded = DecodedAMSR(Chem.MolFromSmiles("CCO"), {})
    suite = BoxDockingObjectives(
        BoxScoringConfig(targets=targets, output_dir=tmp_path / "out"),
        decode_fn=lambda _: decoded,
        conformer_fn=lambda _: pose,
        pipeline=pipeline,
    )

    scored = suite.evaluate([["a"], ["a"], ["b"], ["c"]])

    assert len(pipeline.seen) == 3
    assert [item.valid for item in scored] == [True, False, True, True]
    assert scored[1].rejection_reason == "duplicate AMSR string"
    assert {item.group_key for item in scored} == {"CCO"}


def test_box_rewards_are_aggregated_and_balanced_by_molecule(tmp_path) -> None:
    targets = _targets(tmp_path)
    suite = BoxDockingObjectives(
        BoxScoringConfig(targets=targets, output_dir=tmp_path / "out"),
        pipeline=FakeBoxPipeline(BoxDockingScore()),
    )
    names = [objective.name for objective in suite.objectives]
    first_scores = dict(zip(names, (8.0, 2.0, 2.0), strict=True))
    second_scores = dict(zip(names, (2.0, 8.0, 2.0), strict=True))
    dominated_scores = dict(zip(names, (1.0, 1.0, 1.0), strict=True))
    other_scores = dict(zip(names, (2.0, 2.0, 8.0), strict=True))
    scored = [
        ScoredItem(token_ids=[], scores=first_scores, group_key="molecule-a"),
        ScoredItem(token_ids=[], scores=second_scores, group_key="molecule-a"),
        ScoredItem(token_ids=[], scores=dominated_scores, group_key="molecule-a"),
        ScoredItem(token_ids=[], scores=other_scores, group_key="molecule-b"),
    ]
    expected = super(BoxDockingObjectives, suite).get_rewards(
        [
            ScoredItem(
                token_ids=[],
                scores=dict(zip(names, (8.0, 8.0, 2.0), strict=True)),
            ),
            ScoredItem(token_ids=[], scores=other_scores),
        ]
    )

    rewards = suite.get_rewards(scored)

    assert rewards[0] == pytest.approx(rewards[1])
    assert rewards[0] + rewards[1] == pytest.approx(expected[0])
    assert rewards[2] == 0.0
    assert rewards[3] == pytest.approx(expected[1])


def test_molecule_pareto_record_keeps_each_targets_best_conformer(tmp_path) -> None:
    targets = _targets(tmp_path)
    suite = BoxDockingObjectives(
        BoxScoringConfig(targets=targets, output_dir=tmp_path / "out"),
        pipeline=FakeBoxPipeline(BoxDockingScore()),
    )
    names = [objective.name for objective in suite.objectives]

    def record(amsr, values):
        return {
            "name": amsr,
            "amsr": amsr,
            "smiles": "CCO",
            "scores": dict(zip(names, values, strict=True)),
            "targets": {target.name: {"accepted": True} for target in targets},
            "pose_blocks": {target.name: f"{amsr}-{target.name}" for target in targets},
        }

    front = suite._pareto_front(
        [
            record("conformer-a", (8.0, 2.0, 4.0)),
            record("conformer-b", (7.0, 9.0, 3.0)),
        ]
    )

    assert len(front) == 1
    assert [front[0]["scores"][name] for name in names] == [8.0, 9.0, 4.0]
    assert front[0]["pose_amsrs"] == {
        "q_open": "conformer-a",
        "c_open": "conformer-b",
        "r_open": "conformer-a",
    }
    assert front[0]["conformer_count"] == 2


def test_molecule_record_tracks_separate_cnn_and_aev_pose_conformers(tmp_path) -> None:
    target = _targets(tmp_path)[0]
    suite = BoxDockingObjectives(
        BoxScoringConfig(
            targets=(target,),
            output_dir=tmp_path / "out",
            aev_plig_dir=tmp_path / "AEV-PLIG",
        ),
        pipeline=FakeBoxPipeline(BoxDockingScore()),
    )
    cnn_name, aev_name = [objective.name for objective in suite.objectives]

    def record(amsr, cnn, aev):
        return {
            "name": amsr,
            "amsr": amsr,
            "smiles": "CCO",
            "scores": {cnn_name: cnn, aev_name: aev},
            "targets": {target.name: {"accepted": True}},
            "pose_blocks": {target.name: f"cnn-{amsr}"},
            "aev_pose_blocks": {target.name: f"aev-{amsr}"},
        }

    front = suite._pareto_front([record("conformer-a", 8.0, 6.0), record("conformer-b", 7.0, 9.0)])

    assert len(front) == 1
    assert front[0]["pose_amsrs"][target.name] == "conformer-a"
    assert front[0]["aev_pose_amsrs"][target.name] == "conformer-b"


def test_additional_objectives_receive_one_batch_of_molecules_and_poses(tmp_path) -> None:
    from trl.objectives.base import Objective

    class HeavyAtomObjective(Objective):
        def __init__(self) -> None:
            super().__init__("heavy_atoms", direction="minimize")
            self.batch_sizes = []

        def score_batch(self, items):
            self.batch_sizes.append(len(items))
            assert all(item.docking.accepted for item in items)
            return [float(item.molecule.GetNumHeavyAtoms()) for item in items]

    targets = _targets(tmp_path)
    pose = _mol3d()
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
    decoded = [
        DecodedAMSR(Chem.MolFromSmiles("CC"), {}),
        DecodedAMSR(Chem.MolFromSmiles("CCO"), {}),
    ]
    objective = HeavyAtomObjective()
    suite = BoxDockingObjectives(
        BoxScoringConfig(targets=targets, output_dir=tmp_path / "out"),
        decode_fn=lambda tokens: decoded[int(tokens[0])],
        conformer_fn=lambda _: pose,
        pipeline=FakeBoxPipeline(result),
        additional_objectives=[objective],
    )

    scored = suite.evaluate([["0"], ["1"]])

    assert objective.batch_sizes == [2]
    assert [item.scores["heavy_atoms"] for item in scored] == [2.0, 3.0]


def test_cumulative_front_respects_minimizing_objectives(tmp_path) -> None:
    from trl.objectives.base import Objective

    class MinimizeObjective(Objective):
        def __init__(self) -> None:
            super().__init__("cost", direction="minimize")

        def score_batch(self, items):
            return [float(item.molecule.GetNumHeavyAtoms()) for item in items]

    suite = BoxDockingObjectives(
        BoxScoringConfig(
            targets=_targets(tmp_path),
            output_dir=tmp_path / "out",
        ),
        pipeline=FakeBoxPipeline(BoxDockingScore()),
        additional_objectives=[MinimizeObjective()],
    )
    records = [
        {
            "amsr": label,
            "smiles": smiles,
            "scores": {
                **{
                    objective.name: 5.0
                    for objective in suite.objectives
                    if objective.name != "cost"
                },
                "cost": cost,
            },
            "targets": {},
            "pose_blocks": {},
        }
        for label, smiles, cost in (("lower", "C", 1.0), ("higher", "CC", 2.0))
    ]

    front = suite._pareto_front(records)

    assert len(front) == 1
    assert front[0]["amsr"] == "lower"
    assert front[0]["scores"]["cost"] == 1.0


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
