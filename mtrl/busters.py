from __future__ import annotations

from pathlib import Path


def require() -> None:
    """Require the PoseBusters Python package."""
    try:
        import posebusters  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "posebusters is not installed: https://github.com/maabuu/posebusters"
        ) from error


def run(receptor_pdb: Path, poses_sdf: Path) -> None:
    """Run the standard receptor-aware checks and annotate each SDF pose."""
    from mtrl.posebusters_runner import run as run_posebusters

    run_posebusters("dock", receptor_pdb, poses_sdf)
