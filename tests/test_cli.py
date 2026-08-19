from typer.testing import CliRunner

from mtrl.cli import app


def test_short_help_option() -> None:
    result = CliRunner().invoke(app, ["-h"])

    assert result.exit_code == 0
    assert "generate" in result.stdout
    assert "rl" in result.stdout


def test_generate_writes_only_sdf_to_stdout(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")

    def fake_generate(checkpoint_path, output, **kwargs):
        output.write("molecule\n$$$$\n")
        return {"sampled_strings": 1, "decoded_conformers": 1}

    monkeypatch.setattr("mtrl.generate.generate", fake_generate)
    result = CliRunner().invoke(app, ["generate", str(checkpoint), "-n", "1"])

    assert result.exit_code == 0
    assert result.stdout == "molecule\n$$$$\n"
    assert result.stderr == ""
