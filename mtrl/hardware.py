from __future__ import annotations

import os
import platform
import subprocess
from math import ceil


def sampling_batch_from_free_gib(free_gib: float) -> int:
    if free_gib >= 10:
        return 256
    if free_gib >= 5:
        return 128
    if free_gib >= 2.5:
        return 64
    return 32


def available_cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def default_conformer_workers(task_count: int) -> int:
    hardware_limit = min(16, max(1, available_cpu_count() - 1))
    workload_limit = max(1, ceil(task_count / 4))
    return min(hardware_limit, workload_limit)


def default_evaluation_workers() -> int:
    """Choose a conservative process count for mixed CPU/GPU structure scoring."""
    return min(8, max(1, available_cpu_count() // 2))


def _first_visible_gpu() -> str | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return None
    first = visible.split(",", 1)[0].strip()
    return first if first and first != "-1" else "disabled"


def fast_cli_sampling_batch_size() -> int:
    """Select a numeric CLI default with one short, optional GPU query."""
    visible = _first_visible_gpu()
    if visible == "disabled":
        return 32
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=0.5,
        )
        rows: list[tuple[str, str, float]] = []
        for line in result.stdout.splitlines():
            index, uuid, free_mib = (part.strip() for part in line.split(",", 2))
            rows.append((index, uuid, float(free_mib)))
        if rows:
            selected = rows[0]
            if visible is not None:
                matches = [row for row in rows if visible in (row[0], row[1])]
                if matches:
                    selected = matches[0]
            return sampling_batch_from_free_gib(selected[2] / 1024)
    except (FileNotFoundError, PermissionError, subprocess.SubprocessError, ValueError):
        pass
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return 64
    return 32
