from __future__ import annotations

from pathlib import Path


def require() -> None:
    """Require Roshambo2 for the optional reference-ligand workflow."""
    try:
        import roshambo2  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "roshambo2 is not installed; install mtrl with the 'reference' extra"
        ) from error


def run(
    query_sdf: Path,
    dataset_sdf: Path,
    output_sdf: Path,
    *,
    n_cpus_prepare: int = 1,
) -> None:
    """Overlay one dataset molecule on a reference ligand with Roshambo2."""
    from roshambo2 import Roshambo2

    prefix = str(dataset_sdf.with_suffix(""))
    scorer = Roshambo2(
        str(query_sdf),
        str(dataset_sdf),
        color=True,
        n_cpus_prepare=n_cpus_prepare,
    )
    scorer.compute(
        backend="cpp",
        optim_mode="combination",
        combination_param=0.5,
        write_scores=False,
    )
    scorer.write_best_fit_structures(top_n=1, hits_sdf_prefix=prefix)
    with query_sdf.open() as handle:
        reference_name = handle.readline().strip()
    Path(f"{prefix}_{reference_name}.sdf").replace(output_sdf)
