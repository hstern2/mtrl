import json

from mtrl.report import _pareto_front, write_affinity_progress, write_pareto_progress


def test_pareto_progress_plot_uses_accepted_molecules(tmp_path) -> None:
    records = [
        {"generation": 1, "accepted": True, "cnn_affinity": 4.0, "tanimoto_combo": 0.7},
        {"generation": 2, "accepted": True, "cnn_affinity": 5.0, "tanimoto_combo": 0.5},
        {"generation": 3, "accepted": True, "cnn_affinity": 3.0, "tanimoto_combo": 0.4},
        {"generation": 3, "accepted": False, "cnn_affinity": 9.0, "tanimoto_combo": 0.9},
    ]
    (tmp_path / "scores.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    (tmp_path / "reference.json").write_text(
        json.dumps({"cnn_affinity": 7.0, "tanimoto_combo": 0.85})
    )

    output = write_pareto_progress(tmp_path)

    assert output == tmp_path / "pareto_progress.png"
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert _pareto_front(records[:3]) == records[:2]


def test_no_plot_before_first_acceptance(tmp_path) -> None:
    (tmp_path / "scores.jsonl").write_text(json.dumps({"generation": 1, "accepted": False}) + "\n")

    assert write_pareto_progress(tmp_path) is None
    assert not (tmp_path / "pareto_progress.png").exists()


def test_affinity_progress_plot(tmp_path) -> None:
    (tmp_path / "progress.csv").write_text(
        "generation,best_cnn_affinity,running_best_cnn_affinity,original_t9c_cnn_affinity\n"
        "1,4.0,4.0,7.0\n"
        "2,,4.0,7.0\n"
    )

    output = write_affinity_progress(tmp_path)

    assert output == tmp_path / "progress.png"
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
