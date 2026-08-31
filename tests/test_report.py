import json

from mtrl.report import (
    _pareto_front,
    write_affinity_progress,
    write_pareto_progress,
    write_run_summary,
)


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


def test_run_summary_reports_cumulative_filters_and_scores(tmp_path) -> None:
    (tmp_path / "progress.csv").write_text(
        "generation,generated,cumulative_generated,accepted,accepted_percent,"
        "cumulative_accepted,decode_failed,disconnected_failed,lilly_failed,"
        "conformer_failed,posebusters_failed,scoring_failed,mean_cnn_affinity,"
        "best_cnn_affinity,mean_tanimoto_combo,best_tanimoto_combo,"
        "original_t9c_cnn_affinity\n"
        "1,10,10,5,50,5,1,0,2,0,1,1,4.0,6.0,0.4,0.6,7.0\n"
        "2,10,20,8,80,13,0,1,0,0,1,0,5.0,6.5,0.5,0.7,7.0\n"
    )

    output = write_run_summary(tmp_path)

    assert output == tmp_path / "summary.txt"
    summary = output.read_text()
    assert "Generations in this run: 2" in summary
    assert "Latest generation number: 2" in summary
    assert "Strings generated: 20" in summary
    assert "Passed all gates: 13 (65.00%)" in summary
    assert "Lilly Medchem Rules (-relaxed): 2 (10.00%)" in summary
    assert "PoseBusters: 2 (10.00%)" in summary
    assert "Mean gnina CNNaffinity: 4.615" in summary
    assert "Best Tanimoto similarity: 0.700" in summary
    assert "Latest generation: 8/10 accepted (80.00%)" in summary
