import json
import os
import secrets
import sys
from pathlib import Path

import typer

from mtrl.hardware import (
    default_conformer_workers,
    default_evaluation_workers,
    fast_cli_sampling_batch_size,
)

_CLI_BATCH_SIZE = fast_cli_sampling_batch_size()
_CLI_CONFORMER_WORKERS = default_conformer_workers(100)
_CLI_EVALUATION_WORKERS = default_evaluation_workers()

app = typer.Typer(
    help="mtrl: molecular generation with AMSR + trl",
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command()
def generate(
    checkpoint: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="Pretrained AMSR checkpoint (.pt) to sample from",
    ),
    n: int = typer.Option(
        100,
        "-n",
        help=(
            "Number of AMSR strings to sample; fewer SDF records may be written "
            "if decoding or conformer construction fails"
        ),
        rich_help_panel="Sampling",
    ),
    batch_size: int = typer.Option(
        _CLI_BATCH_SIZE,
        help=(
            "AMSR strings sampled together on the selected device; larger batches "
            "are usually faster but require more device memory"
        ),
        rich_help_panel="Sampling",
    ),
    conformer_workers: int = typer.Option(
        _CLI_CONFORMER_WORKERS,
        help=(
            "CPU processes used to decode sampled AMSR strings into 3D conformers; "
            "larger values use more CPU and memory"
        ),
        rich_help_panel="Conformer construction",
    ),
    temperature: float = typer.Option(
        0.8,
        help=(
            "Sampling randomness at each token: lower favors the model's most likely "
            "choices; higher increases variety and usually increases invalid output"
        ),
        rich_help_panel="Sampling",
    ),
    top_k: int = typer.Option(
        0,
        help=(
            "Maximum number of likely choices kept for each next AMSR token; "
            "0 means no limit and is usually appropriate"
        ),
        rich_help_panel="Sampling",
    ),
    top_p: float = typer.Option(
        1.0,
        help=(
            "Keep enough likely choices for each next AMSR token to cover this "
            "probability fraction; 1.0 means no restriction and is usually appropriate"
        ),
        rich_help_panel="Sampling",
    ),
    seed: int | None = typer.Option(
        None,
        help=(
            "Random seed for token sampling; provide an integer to reproduce a run, "
            "or omit it to choose a new seed each run"
        ),
        show_default="random each run",
        rich_help_panel="Sampling",
    ),
    device: str = typer.Option(
        "auto",
        help=(
            "Device used for transformer sampling: auto selects CUDA when available; "
            "otherwise use cpu, cuda, or a specific device such as cuda:0"
        ),
        rich_help_panel="Sampling",
    ),
) -> None:
    """Sample AMSR strings and decode their encoded 3D conformers without scoring."""
    from mtrl.generate import generate as generate_conformers

    if n <= 0:
        raise typer.BadParameter("-n must be > 0")
    if batch_size <= 0:
        raise typer.BadParameter("--batch-size must be > 0")
    if conformer_workers <= 0:
        raise typer.BadParameter("--conformer-workers must be > 0")
    if temperature <= 0:
        raise typer.BadParameter("--temperature must be > 0")
    if top_k < 0:
        raise typer.BadParameter("--top-k must be >= 0")
    if not 0 < top_p <= 1:
        raise typer.BadParameter("--top-p must be in (0, 1]")
    if seed is None:
        seed = secrets.randbits(63)

    try:
        generate_conformers(
            checkpoint.resolve(),
            sys.stdout,
            n=n,
            batch_size=batch_size,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            device_name=device,
            conformer_workers=conformer_workers,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error


@app.command()
def rl(
    checkpoint: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help=(
            "Pretrained AMSR checkpoint (.pt) used as both the initial policy and "
            "the frozen KL-reference model"
        ),
    ),
    receptor_pdb: Path = typer.Option(
        ...,
        "--receptor-pdb",
        exists=True,
        dir_okay=False,
        help="Target receptor passed to GNINA and receptor-aware PoseBusters checks",
        rich_help_panel="Scoring inputs",
    ),
    reference_sdf: Path = typer.Option(
        ...,
        "--reference-sdf",
        exists=True,
        dir_okay=False,
        help=("3D reference ligand used for Roshambo2 alignment and as GNINA's minimization box"),
        rich_help_panel="Scoring inputs",
    ),
    evaluation_workers: int = typer.Option(
        _CLI_EVALUATION_WORKERS,
        "--evaluation-workers",
        help=(
            "Worker processes used concurrently for AMSR conformer construction, "
            "Roshambo2 alignment, GNINA minimization, and PoseBusters"
        ),
        rich_help_panel="Parallel evaluation",
    ),
    lilly_medchem_rules: bool = typer.Option(
        False,
        "--lilly-medchem-rules/--no-lilly-medchem-rules",
        help=(
            "Apply Lilly Medchem Rules in -relaxed mode before costly 3D scoring; "
            "failures receive no reward"
        ),
        rich_help_panel="Molecule gates",
    ),
    lilly_rules_executable: str = typer.Option(
        "Lilly_Medchem_Rules.rb",
        "--lilly-rules-executable",
        help=(
            "Command name or path for Lilly_Medchem_Rules.rb; used only when "
            "--lilly-medchem-rules is enabled"
        ),
        rich_help_panel="Molecule gates",
    ),
    iterations: int = typer.Option(
        1000,
        help=(
            "Number of sample-score-update cycles; total generated molecules are "
            "iterations multiplied by batch size"
        ),
        rich_help_panel="RL training",
    ),
    batch_size: int = typer.Option(
        16,
        help=(
            "Total molecules generated per RL iteration across all GPUs; must be "
            "divisible by WORLD_SIZE"
        ),
        rich_help_panel="RL training",
    ),
    lr: float = typer.Option(
        1e-5,
        help=(
            "Peak AdamW learning rate; the schedule warms up for 100 iterations, "
            "then decays to zero"
        ),
        rich_help_panel="RL training",
    ),
    warmup_steps: int = typer.Option(
        100,
        help=("Number of learning-rate warmup iterations; use a smaller value for short RL runs"),
        rich_help_panel="RL training",
    ),
    kl_beta: float = typer.Option(
        0.05,
        help=(
            "Penalty for moving away from the starting checkpoint; higher values "
            "keep the policy closer, while 0 disables the KL penalty"
        ),
        rich_help_panel="RL training",
    ),
    pareto_lambda: float = typer.Option(
        0.1,
        help=(
            "Bonus for a molecule that adds a new point to the cumulative "
            "affinity/similarity Pareto front; absolute joint quality supplies the "
            "base reward"
        ),
        rich_help_panel="RL training",
    ),
    temperature: float = typer.Option(
        1.0,
        help=(
            "Sampling temperature at the start of RL; higher values explore more "
            "and usually produce more invalid molecules"
        ),
        rich_help_panel="RL training",
    ),
    temperature_final: float = typer.Option(
        0.8,
        help=(
            "Sampling temperature at the end of RL; temperature changes linearly "
            "from --temperature to this value"
        ),
        rich_help_panel="RL training",
    ),
    seed: int | None = typer.Option(
        None,
        help=(
            "Random seed for sampling and RL initialization; provide an integer to "
            "reproduce a run, or omit it to choose a new seed each run"
        ),
        show_default="random each run",
        rich_help_panel="RL training",
    ),
    replay_fraction: float = typer.Option(0.0, hidden=True),
    precision: str = typer.Option(
        "auto",
        help=(
            "Training precision: auto uses FP16 on V100-era CUDA GPUs, BF16 on "
            "Ampere or newer, and FP32 on CPU; explicit choices are fp32/fp16/bf16"
        ),
        rich_help_panel="RL training",
    ),
    checkpoint_every: int = typer.Option(
        100,
        help=("Save rl_step_N.pt every N iterations; 0 disables intermediate saves"),
        rich_help_panel="Output and logging",
    ),
    save_final_checkpoint: bool = typer.Option(
        True,
        "--save-final-checkpoint/--no-save-final-checkpoint",
        help=(
            "Write rl_final.pt after the last iteration; disable this for independent "
            "generation screens that only need molecules and scores"
        ),
        rich_help_panel="Output and logging",
    ),
    log_every: int = typer.Option(
        10,
        help="Print reward, validity, KL, objective, and rejection summaries every N iterations",
        rich_help_panel="Output and logging",
    ),
    output_dir: Path = typer.Option(
        Path("mtrl_output/"),
        "--output-dir",
        help=(
            "Empty directory for generation SDFs, Pareto SDFs, progress reports, "
            "scores.jsonl, configuration, and RL checkpoints"
        ),
        rich_help_panel="Output and logging",
    ),
    verbose_tools: bool = typer.Option(
        False,
        help=(
            "Show Roshambo2, GNINA, and PoseBusters output; by default their routine "
            "output is suppressed"
        ),
        rich_help_panel="Output and logging",
    ),
    wandb_project: str | None = typer.Option(
        None,
        help="Weights & Biases project name; omit to disable W&B logging",
        rich_help_panel="Output and logging",
    ),
) -> None:
    """Pareto RL using GNINA affinity and Roshambo2 shape/color similarity."""
    from trl.training.rl_train import rl_train

    from mtrl.config import ScoringConfig

    for name, value in (
        ("--iterations", iterations),
        ("--batch-size", batch_size),
        ("--lr", lr),
        ("--temperature", temperature),
        ("--temperature-final", temperature_final),
        ("--log-every", log_every),
        ("--evaluation-workers", evaluation_workers),
    ):
        if value <= 0:
            raise typer.BadParameter(f"{name} must be > 0")
    for name, value in (
        ("--kl-beta", kl_beta),
        ("--pareto-lambda", pareto_lambda),
        ("--checkpoint-every", checkpoint_every),
        ("--warmup-steps", warmup_steps),
    ):
        if value < 0:
            raise typer.BadParameter(f"{name} must be >= 0")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if batch_size % world_size:
        raise typer.BadParameter(
            f"--batch-size={batch_size} must be divisible by WORLD_SIZE={world_size}"
        )
    if seed is None:
        seed = secrets.randbits(63)
    if seed < 0 or seed >= 2**63:
        raise typer.BadParameter("--seed must be in [0, 2^63)")
    output_dir = output_dir.resolve()
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0 and output_dir.exists():
        if not output_dir.is_dir():
            raise typer.BadParameter(f"--output-dir is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise typer.BadParameter(f"--output-dir must be empty: {output_dir}")
    config = ScoringConfig(
        receptor_pdb=receptor_pdb.resolve(),
        reference_sdf=reference_sdf.resolve(),
        output_dir=output_dir,
        lilly_medchem_rules=lilly_medchem_rules,
        lilly_rules_executable=lilly_rules_executable,
        verbose_tools=verbose_tools,
        evaluation_workers=evaluation_workers,
    )
    config.install()

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "scoring_config.json").write_text(
            json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n"
        )
        (output_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "batch_size": batch_size,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_every": checkpoint_every,
                    "evaluation_workers": evaluation_workers,
                    "iterations": iterations,
                    "kl_beta": kl_beta,
                    "lr": lr,
                    "pareto_lambda": pareto_lambda,
                    "reward": (
                        "reference-normalized CNNaffinity * Tanimoto similarity, "
                        "plus cumulative-Pareto-front bonus"
                    ),
                    "precision": precision,
                    "save_final_checkpoint": save_final_checkpoint,
                    "seed": seed,
                    "temperature": temperature,
                    "temperature_final": temperature_final,
                    "warmup_steps": warmup_steps,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    rl_train(
        checkpoint_path=str(checkpoint),
        vocab_path=None,
        objectives_path="mtrl.objectives:build",
        iterations=iterations,
        batch_size=batch_size,
        lr=lr,
        warmup_steps=warmup_steps,
        kl_beta=kl_beta,
        pareto_lambda=pareto_lambda,
        temperature=temperature,
        temperature_final=temperature_final,
        replay_fraction=replay_fraction,
        precision=precision,
        checkpoint_every=checkpoint_every,
        save_final_checkpoint=save_final_checkpoint,
        log_every=log_every,
        checkpoint_dir=str(output_dir),
        wandb_project=wandb_project,
        seed=seed,
    )
