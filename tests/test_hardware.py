import subprocess

from mtrl.hardware import default_evaluation_workers, fast_cli_sampling_batch_size


def test_fast_gpu_query_respects_visible_device(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    monkeypatch.setattr(
        "mtrl.hardware.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="0, GPU-zero, 12288\n1, GPU-one, 6144\n",
            stderr="",
        ),
    )

    assert fast_cli_sampling_batch_size() == 128


def test_fast_gpu_query_skips_disabled_cuda(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    assert fast_cli_sampling_batch_size() == 32


def test_fast_gpu_query_has_cpu_fallback(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        "mtrl.hardware.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )
    monkeypatch.setattr("mtrl.hardware.platform.system", lambda: "Linux")

    assert fast_cli_sampling_batch_size() == 32


def test_evaluation_workers_are_bounded(monkeypatch) -> None:
    monkeypatch.setattr("mtrl.hardware.available_cpu_count", lambda: 30)
    assert default_evaluation_workers() == 8

    monkeypatch.setattr("mtrl.hardware.available_cpu_count", lambda: 2)
    assert default_evaluation_workers() == 1
