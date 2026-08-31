from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def _progress_rows(progress_path: Path) -> list[dict[str, str]]:
    with progress_path.open(newline="") as source:
        return list(csv.DictReader(source))


def _weighted_mean(rows: list[dict[str, str]], field: str) -> float | None:
    weighted_sum = 0.0
    count = 0
    for row in rows:
        value = row.get(field)
        accepted = int(row.get("accepted") or 0)
        if value and accepted:
            weighted_sum += float(value) * accepted
            count += accepted
    return weighted_sum / count if count else None


def write_run_summary(output_dir: Path, destination: Path | None = None) -> Path | None:
    """Write a compact cumulative, human-readable RL status report."""
    progress_path = output_dir / "progress.csv"
    if not progress_path.is_file():
        return None
    rows = _progress_rows(progress_path)
    if not rows:
        return None

    latest = rows[-1]
    generated = int(latest["cumulative_generated"])
    accepted = int(latest["cumulative_accepted"])
    rejected = generated - accepted
    failures = (
        ("AMSR decode", "decode_failed"),
        ("Disconnected molecule", "disconnected_failed"),
        ("Lilly Medchem Rules (-relaxed)", "lilly_failed"),
        ("AMSR conformer construction", "conformer_failed"),
        ("PoseBusters", "posebusters_failed"),
        ("Alignment/minimization/scoring", "scoring_failed"),
    )

    def percentage(count: int) -> str:
        return f"{100.0 * count / max(1, generated):.2f}%"

    lines = [
        "mtrl RL summary",
        "",
        f"Generations completed: {int(latest['generation']):,}",
        f"Strings generated: {generated:,}",
        f"Passed all gates: {accepted:,} ({percentage(accepted)})",
        f"Rejected: {rejected:,} ({percentage(rejected)})",
        "",
        "Gate failures (percentage of all generated strings):",
    ]
    for label, field in failures:
        count = sum(int(row.get(field) or 0) for row in rows)
        lines.append(f"  {label}: {count:,} ({percentage(count)})")

    mean_affinity = _weighted_mean(rows, "mean_cnn_affinity")
    mean_similarity = _weighted_mean(rows, "mean_tanimoto_combo")
    best_affinity = max(
        (float(row["best_cnn_affinity"]) for row in rows if row.get("best_cnn_affinity")),
        default=None,
    )
    best_similarity = max(
        (float(row["best_tanimoto_combo"]) for row in rows if row.get("best_tanimoto_combo")),
        default=None,
    )
    lines.extend(
        [
            "",
            "Accepted-molecule scores:",
            f"  Mean gnina CNNaffinity: {mean_affinity:.3f}"
            if mean_affinity is not None
            else "  Mean gnina CNNaffinity: n/a",
            f"  Best gnina CNNaffinity: {best_affinity:.3f}"
            if best_affinity is not None
            else "  Best gnina CNNaffinity: n/a",
            f"  Mean Tanimoto similarity: {mean_similarity:.3f}"
            if mean_similarity is not None
            else "  Mean Tanimoto similarity: n/a",
            f"  Best Tanimoto similarity: {best_similarity:.3f}"
            if best_similarity is not None
            else "  Best Tanimoto similarity: n/a",
        ]
    )
    reference_affinity = latest.get("original_t9c_cnn_affinity")
    if reference_affinity:
        lines.append(f"  Reference-ligand gnina CNNaffinity: {float(reference_affinity):.3f}")
    lines.extend(
        [
            "",
            f"Latest generation: {int(latest['accepted']):,}/{int(latest['generated']):,} "
            f"accepted ({float(latest['accepted_percent']):.2f}%)",
        ]
    )

    output_path = destination or output_dir / "summary.txt"
    temporary = Path(f"{output_path}.tmp.{os.getpid()}")
    try:
        temporary.write_text("\n".join(lines) + "\n")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def _accepted_records(scores_path: Path) -> list[dict[str, Any]]:
    records = []
    with scores_path.open() as source:
        for line in source:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                break  # The live writer may be midway through its final line.
            if (
                record.get("accepted")
                and record.get("cnn_affinity") is not None
                and record.get("tanimoto_combo") is not None
            ):
                records.append(record)
    return records


def _pareto_front(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the two-objective maximizing front, ordered by CNNaffinity."""
    ordered = sorted(
        records,
        key=lambda record: (-float(record["cnn_affinity"]), -float(record["tanimoto_combo"])),
    )
    front = []
    best_similarity = float("-inf")
    for record in ordered:
        similarity = float(record["tanimoto_combo"])
        if similarity > best_similarity:
            front.append(record)
            best_similarity = similarity
    return sorted(front, key=lambda record: float(record["cnn_affinity"]))


def write_pareto_progress(output_dir: Path, destination: Path | None = None) -> Path | None:
    scores_path = output_dir / "scores.jsonl"
    reference_path = output_dir / "reference.json"
    records = _accepted_records(scores_path)
    reference = json.loads(reference_path.read_text()) if reference_path.is_file() else {}
    if not records:
        return None

    generations = [int(record["generation"]) for record in records]
    affinities = [float(record["cnn_affinity"]) for record in records]
    similarities = [float(record["tanimoto_combo"]) for record in records]
    front = _pareto_front(records)

    figure, axis = plt.subplots(figsize=(8.5, 6.5))
    points = axis.scatter(
        affinities,
        similarities,
        c=generations,
        cmap="viridis",
        s=20,
        alpha=0.58,
        linewidths=0,
        rasterized=True,
    )
    colorbar = figure.colorbar(points, ax=axis, pad=0.02)
    colorbar.set_label("RL generation")

    front_x = [float(record["cnn_affinity"]) for record in front]
    front_y = [float(record["tanimoto_combo"]) for record in front]
    axis.plot(front_x, front_y, color="black", linewidth=1.4, alpha=0.8)
    axis.scatter(
        front_x,
        front_y,
        facecolors="none",
        edgecolors="black",
        s=72,
        linewidths=1.3,
        label="Pareto front",
    )

    reference_affinity = reference.get("cnn_affinity")
    reference_similarity = reference.get("identity_tanimoto_similarity", 1.0)
    if reference_affinity is not None and reference_similarity is not None:
        axis.scatter(
            [float(reference_affinity)],
            [float(reference_similarity)],
            marker="*",
            s=240,
            color="#d62728",
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label="reference ligand",
        )

    axis.set(
        xlabel="gnina CNNaffinity",
        ylabel="Tanimoto similarity",
    )
    axis.grid(alpha=0.18)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()

    output_path = destination or output_dir / "pareto_progress.png"
    temporary = Path(f"{output_path}.tmp.{os.getpid()}")
    try:
        figure.savefig(temporary, format="png", dpi=160)
        os.replace(temporary, output_path)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)
    return output_path


def write_affinity_progress(output_dir: Path, destination: Path | None = None) -> Path | None:
    progress_path = output_dir / "progress.csv"
    rows = _progress_rows(progress_path)
    if not rows:
        return None

    generations = [int(row["generation"]) for row in rows]
    generation_best = [
        float(row["best_cnn_affinity"]) if row["best_cnn_affinity"] else float("nan")
        for row in rows
    ]
    running_best = [
        float(row["running_best_cnn_affinity"])
        if row["running_best_cnn_affinity"]
        else float("nan")
        for row in rows
    ]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        generations,
        generation_best,
        "o",
        markersize=3,
        alpha=0.65,
        label="generation best",
    )
    axis.plot(generations, running_best, linewidth=2, label="running best")
    reference_affinity = rows[-1].get("original_t9c_cnn_affinity")
    if reference_affinity:
        axis.axhline(
            float(reference_affinity),
            color="black",
            linestyle="--",
            label="reference ligand",
        )
    axis.set(xlabel="RL generation", ylabel="gnina CNNaffinity")
    axis.grid(alpha=0.2)
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    figure.tight_layout()

    output_path = destination or output_dir / "progress.png"
    temporary = Path(f"{output_path}.tmp.{os.getpid()}")
    try:
        figure.savefig(temporary, format="png", dpi=150)
        os.replace(temporary, output_path)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)
    return output_path


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot live mtrl affinity/similarity progress")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--watch", type=float, default=0, metavar="SECONDS")
    parser.add_argument("--pid", type=int, help="Stop watching when this process exits")
    args = parser.parse_args()

    previous_sizes = (-1, -1)
    while True:
        scores_path = args.output_dir / "scores.jsonl"
        progress_path = args.output_dir / "progress.csv"
        if not scores_path.is_file() or not progress_path.is_file():
            if not args.watch or (args.pid is not None and not _process_exists(args.pid)):
                break
            time.sleep(args.watch)
            continue
        sizes = (scores_path.stat().st_size, progress_path.stat().st_size)
        if sizes != previous_sizes:
            paths = (
                write_pareto_progress(args.output_dir),
                write_affinity_progress(args.output_dir),
                write_run_summary(args.output_dir),
            )
            for path in paths:
                if path is not None:
                    print(path, flush=True)
            previous_sizes = sizes
        if not args.watch or (args.pid is not None and not _process_exists(args.pid)):
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
