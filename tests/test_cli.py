import json

from typer.testing import CliRunner

from mtrl.cli import _CLI_BATCH_SIZE, _CLI_CONFORMER_WORKERS, app


def _normalized(text: str) -> str:
    return " ".join(text.split())


def test_short_help_option() -> None:
    result = CliRunner().invoke(app, ["-h"])

    assert result.exit_code == 0
    assert "generate" in result.stdout
    assert "rl" in result.stdout
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
    result = CliRunner().invoke(app, ["rl", "-h"])

    assert result.exit_code == 0
    help_text = _normalized(result.stdout)
    for explanation in (
        "initial policy",
        "GNINA's minimization box",
        "not realigned",
        "total generated molecules",
        "Peak AdamW learning rate",
        "short RL runs",
        "starting checkpoint",
        "Pareto rank alone",
        "temperature changes linearly",
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
    assert received["warmup_steps"] == 100
    assert received["save_final_checkpoint"] is False
    run_config = json.loads((output / "run_config.json").read_text())
    assert run_config["seed"] == 8675309
    assert run_config["warmup_steps"] == 100
    assert run_config["save_final_checkpoint"] is False
