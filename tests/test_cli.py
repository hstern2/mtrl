import json

from typer.testing import CliRunner

from mtrl.cli import (
    _CLI_BATCH_SIZE,
    _CLI_CONFORMER_WORKERS,
    _CLI_EVALUATION_WORKERS,
    app,
)


def _normalized(text: str) -> str:
    return " ".join(text.split())


def test_short_help_option() -> None:
    result = CliRunner().invoke(app, ["-h"])

    assert result.exit_code == 0
    assert "generate" in result.stdout
    assert "rl" in result.stdout
    assert "score" in result.stdout
    assert "validate-targets" in result.stdout
    assert "evaluate" not in result.stdout
    assert "default: 0" not in result.stdout

    generate_help = CliRunner().invoke(app, ["generate", "-h"])
    assert generate_help.exit_code == 0
    assert f"[default: {_CLI_BATCH_SIZE}]" in generate_help.stdout
    assert f"[default: {_CLI_CONFORMER_WORKERS}]" in generate_help.stdout
    help_text = _normalized(generate_help.stdout)
    assert "Sampling randomness at each token" in help_text
    assert "most likely choices" in help_text
    assert "Maximum number of likely choices" in help_text
    assert "probability" in help_text
    assert "fraction; 1.0 means no restriction" in help_text
    assert "usually appropriate" in help_text
    assert "random each run" in help_text
    assert "reproduce a run" in help_text


def test_rl_help_explains_training_and_output_options() -> None:
    result = CliRunner().invoke(app, ["rl", "-h"], terminal_width=120)

    assert result.exit_code == 0
    assert "max-minimized-rmsd" not in result.stdout
    help_text = _normalized(result.stdout)
    for explanation in (
        "initial policy",
        "regularization",
        "GNINA's",
        "minimization box",
        "Worker processes used concurrently",
        "Final RL iteration",
        "Restore optimizer",
        "Peak AdamW",
        "short RL runs",
        "starting checkpoint",
        "absolute joint",
        "temperature changes",
        "new seed each run",
        "V100-era CUDA GPUs",
        "rl_final.pt",
        "generation screens",
        "Empty directory",
        "Pareto SDFs",
        "scores.jsonl",
        "Weights & Biases",
    ):
        assert explanation in help_text


def test_generate_writes_only_sdf_to_stdout(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")

    received = {}
    monkeypatch.setattr("mtrl.cli.secrets.randbits", lambda bits: 8675309)

    def fake_generate(checkpoint_path, output, **kwargs):
        received.update(kwargs)
        output.write("molecule\n$$$$\n")
        return {"sampled_strings": 1, "decoded_conformers": 1}

    monkeypatch.setattr("mtrl.generate.generate", fake_generate)
    result = CliRunner().invoke(app, ["generate", str(checkpoint), "-n", "1"])

    assert result.exit_code == 0
    assert result.stdout == "molecule\n$$$$\n"
    assert result.stderr == ""
    assert received["batch_size"] == _CLI_BATCH_SIZE
    assert received["conformer_workers"] == _CLI_CONFORMER_WORKERS
    assert received["seed"] == 8675309

    received.clear()
    explicit = CliRunner().invoke(
        app,
        ["generate", str(checkpoint), "-n", "1", "--seed", "123"],
    )
    assert explicit.exit_code == 0
    assert received["seed"] == 123


def test_rl_refuses_to_mix_results_in_a_nonempty_output_directory(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    receptor = tmp_path / "receptor.pdb"
    reference = tmp_path / "reference.sdf"
    output = tmp_path / "output"
    checkpoint.touch()
    receptor.touch()
    reference.touch()
    output.mkdir()
    (output / "old-result").touch()

    result = CliRunner().invoke(
        app,
        [
            "rl",
            str(checkpoint),
            "--receptor-pdb",
            str(receptor),
            "--reference-sdf",
            str(reference),
            "--output-dir",
            str(output),
            "--iterations",
            "1",
        ],
    )

    assert result.exit_code == 2
    assert "--output-dir must be empty" in result.stderr


def test_rl_chooses_and_records_random_seed(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "model.pt"
    receptor = tmp_path / "receptor.pdb"
    reference = tmp_path / "reference.sdf"
    output = tmp_path / "output"
    checkpoint.touch()
    receptor.touch()
    reference.touch()
    received = {}

    monkeypatch.setattr("mtrl.cli.secrets.randbits", lambda bits: 8675309)

    def fake_rl_train(**kwargs):
        received.update(kwargs)

    monkeypatch.setattr("trl.training.rl_train.rl_train", fake_rl_train)
    result = CliRunner().invoke(
        app,
        [
            "rl",
            str(checkpoint),
            "--receptor-pdb",
            str(receptor),
            "--reference-sdf",
            str(reference),
            "--output-dir",
            str(output),
            "--iterations",
            "1",
            "--checkpoint-every",
            "0",
            "--no-save-final-checkpoint",
        ],
    )

    assert result.exit_code == 0
    assert received["seed"] == 8675309
    assert received["reference_checkpoint_path"] is None
    assert received["warmup_steps"] == 100
    assert received["save_final_checkpoint"] is False
    assert received["resume_training_state"] is False
    run_config = json.loads((output / "run_config.json").read_text())
    assert run_config["seed"] == 8675309
    assert run_config["warmup_steps"] == 100
    assert run_config["save_final_checkpoint"] is False
    assert run_config["evaluation_workers"] == _CLI_EVALUATION_WORKERS
    assert run_config["kl_reference_checkpoint"] == str(checkpoint.resolve())
    assert run_config["resume_training_state"] is False
    assert run_config["resumed_from_step"] == 0


def test_rl_records_a_separate_kl_reference_checkpoint(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "policy.pt"
    kl_reference = tmp_path / "reference.pt"
    receptor = tmp_path / "receptor.pdb"
    reference_sdf = tmp_path / "reference.sdf"
    output = tmp_path / "output"
    for path in (checkpoint, kl_reference, receptor, reference_sdf):
        path.touch()
    received = {}

    monkeypatch.setattr("trl.training.rl_train.rl_train", lambda **kwargs: received.update(kwargs))
    result = CliRunner().invoke(
        app,
        [
            "rl",
            str(checkpoint),
            "--kl-reference-checkpoint",
            str(kl_reference),
            "--receptor-pdb",
            str(receptor),
            "--reference-sdf",
            str(reference_sdf),
            "--output-dir",
            str(output),
            "--iterations",
            "1",
            "--checkpoint-every",
            "0",
            "--no-save-final-checkpoint",
        ],
    )

    assert result.exit_code == 0
    assert received["reference_checkpoint_path"] == str(kl_reference.resolve())
    run_config = json.loads((output / "run_config.json").read_text())
    assert run_config["kl_reference_checkpoint"] == str(kl_reference.resolve())


def test_rl_resumes_available_training_state(tmp_path, monkeypatch) -> None:
    import torch

    checkpoint = tmp_path / "policy.pt"
    receptor = tmp_path / "receptor.pdb"
    reference_sdf = tmp_path / "reference.sdf"
    output = tmp_path / "output"
    torch.save(
        {
            "step": 375,
            "optimizer": {},
            "scheduler": {},
            "rng_states": [{}],
        },
        checkpoint,
    )
    receptor.touch()
    reference_sdf.touch()
    received = {}

    monkeypatch.setattr("trl.training.rl_train.rl_train", lambda **kwargs: received.update(kwargs))
    result = CliRunner().invoke(
        app,
        [
            "rl",
            str(checkpoint),
            "--resume-training-state",
            "--receptor-pdb",
            str(receptor),
            "--reference-sdf",
            str(reference_sdf),
            "--output-dir",
            str(output),
            "--iterations",
            "1000",
            "--checkpoint-every",
            "0",
            "--no-save-final-checkpoint",
        ],
    )

    assert result.exit_code == 0
    assert received["resume_training_state"] is True
    run_config = json.loads((output / "run_config.json").read_text())
    assert run_config["resumed_from_step"] == 375
    assert run_config["value_head_resume"] == "initialized fresh (legacy checkpoint)"


def test_rl_accepts_box_docking_target_manifest(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "policy.pt"
    receptor = tmp_path / "receptor.pdb"
    targets = tmp_path / "targets.json"
    output = tmp_path / "output"
    checkpoint.touch()
    receptor.write_text("END\n")
    targets.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "name": "q_open",
                        "receptor_pdb": "receptor.pdb",
                        "center": [1, 2, 3],
                        "size": [20, 20, 20],
                    }
                ]
            }
        )
    )
    received = {}
    monkeypatch.setattr("trl.training.rl_train.rl_train", lambda **kwargs: received.update(kwargs))

    result = CliRunner().invoke(
        app,
        [
            "rl",
            str(checkpoint),
            "--docking-targets",
            str(targets),
            "--lilly-medchem-rules",
            "--rdkit-druglike-filter",
            "--muegge-filter",
            "--brenk-filter",
            "--max-br-sascore",
            "5",
            "--target-failure-score",
            "-1.5",
            "--accept-targets",
            "all",
            "--docking-mode",
            "rigid-refine",
            "--gnina-timeout-seconds",
            "300",
            "--posebusters-timeout-seconds",
            "90",
            "--posebusters-config",
            "dock-fast",
            "--qed-objective",
            "--output-dir",
            str(output),
            "--iterations",
            "1",
            "--checkpoint-every",
            "0",
            "--no-save-final-checkpoint",
        ],
    )

    assert result.exit_code == 0
    scoring_config = json.loads((output / "scoring_config.json").read_text())
    assert scoring_config["mode"] == "box_docking"
    assert scoring_config["targets"][0]["name"] == "q_open"
    assert scoring_config["lilly_medchem_rules"] is True
    assert scoring_config["rdkit_druglike_filter"] is True
    assert scoring_config["muegge_filter"] is True
    assert scoring_config["brenk_filter"] is True
    assert scoring_config["max_br_sascore"] == 5.0
    assert scoring_config["target_failure_score"] == -1.5
    assert scoring_config["accept_targets"] == "all"
    assert scoring_config["docking_mode"] == "rigid-refine"
    assert scoring_config["gnina_timeout_seconds"] == 300
    assert scoring_config["posebusters_timeout_seconds"] == 90
    assert scoring_config["posebusters_config"] == "dock-fast"
    assert scoring_config["qed_objective"] is True
    run_config = json.loads((output / "run_config.json").read_text())
    assert run_config["rdkit_druglike_filter"] is True
    assert run_config["muegge_filter"] is True
    assert run_config["brenk_filter"] is True
    assert run_config["max_br_sascore"] == 5.0
    assert "N-dimensional Pareto" in run_config["reward"]
    assert received["objectives_path"] == "mtrl.objectives:build"


def test_validate_targets_reports_named_boxes(tmp_path) -> None:
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("END\n")
    targets = tmp_path / "targets.json"
    targets.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "name": "site_a",
                        "receptor_pdb": "receptor.pdb",
                        "center": [1, 2, 3],
                        "size": [20, 21, 22],
                    }
                ]
            }
        )
    )

    result = CliRunner().invoke(app, ["validate-targets", str(targets)])

    assert result.exit_code == 0
    assert "Valid target manifest: 1 target(s)" in result.stdout
    assert "site_a" in result.stdout
    assert "center=(1.0, 2.0, 3.0)" in result.stdout


def test_score_command_builds_generic_box_configuration(tmp_path, monkeypatch) -> None:
    molecules = tmp_path / "molecules.sdf"
    receptor = tmp_path / "receptor.pdb"
    targets = tmp_path / "targets.json"
    output = tmp_path / "scored"
    molecules.write_text("$$$$\n")
    receptor.write_text("END\n")
    targets.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "name": "site_a",
                        "receptor_pdb": "receptor.pdb",
                        "center": [1, 2, 3],
                        "size": [20, 20, 20],
                    }
                ]
            }
        )
    )
    received = {}

    def fake_score_sdf(input_sdf, config):
        received["input_sdf"] = input_sdf
        received["config"] = config
        return {"input_records": 1, "retained_molecules": 1}

    monkeypatch.setattr("mtrl.box_score.score_sdf", fake_score_sdf)
    result = CliRunner().invoke(
        app,
        [
            "score",
            str(molecules),
            "--docking-targets",
            str(targets),
            "--target-failure-score",
            "-2",
            "--accept-targets",
            "any",
            "--docking-mode",
            "rigid-refine",
            "--gnina-timeout-seconds",
            "300",
            "--posebusters-timeout-seconds",
            "90",
            "--posebusters-config",
            "dock-fast",
            "--qed-objective",
            "--rdkit-druglike-filter",
            "--muegge-filter",
            "--brenk-filter",
            "--max-br-sascore",
            "5",
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert received["input_sdf"] == molecules.resolve()
    assert received["config"].target_failure_score == -2.0
    assert received["config"].accept_targets == "any"
    assert received["config"].docking_mode == "rigid-refine"
    assert received["config"].gnina_timeout_seconds == 300
    assert received["config"].posebusters_timeout_seconds == 90
    assert received["config"].posebusters_config == "dock-fast"
    assert received["config"].qed_objective is True
    assert received["config"].rdkit_druglike_filter is True
    assert received["config"].muegge_filter is True
    assert received["config"].brenk_filter is True
    assert received["config"].max_br_sascore == 5.0
    assert '"retained_molecules": 1' in result.stdout
