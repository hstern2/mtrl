from typer.testing import CliRunner

from mtrl.cli import _CLI_BATCH_SIZE, _CLI_CONFORMER_WORKERS, app


def test_short_help_option() -> None:
    result = CliRunner().invoke(app, ["-h"])

    assert result.exit_code == 0
    assert "generate" in result.stdout
    assert "rl" in result.stdout
    assert "default: 0" not in result.stdout

    generate_help = CliRunner().invoke(app, ["generate", "-h"])
    assert generate_help.exit_code == 0
    assert f"[default: {_CLI_BATCH_SIZE}]" in generate_help.stdout
    assert f"[default: {_CLI_CONFORMER_WORKERS}]" in generate_help.stdout
    assert "most probable choices" in generate_help.stdout
    assert "cumulative" in generate_help.stdout
    assert "probability reaches P" in generate_help.stdout


def test_generate_writes_only_sdf_to_stdout(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")

    received = {}

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
