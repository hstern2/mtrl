import json
import math
import os
import secrets
import sys
from pathlib import Path
from typing import cast

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


@app.command("validate-targets")
def validate_targets(
    targets_json: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="Portable JSON manifest containing named receptor/box targets",
    ),
) -> None:
    """Validate a receptor/box target manifest without running docking."""
    from mtrl.config import load_docking_targets

    try:
        targets = load_docking_targets(targets_json.resolve())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise typer.BadParameter(str(error), param_hint="targets_json") from error
    typer.echo(f"Valid target manifest: {len(targets)} target(s)")
    for target in targets:
        typer.echo(
            f"{target.name}: receptor={target.receptor_pdb} "
            f"center={target.center} size={target.size} "
            f"exhaustiveness={target.exhaustiveness} num_modes={target.num_modes}"
        )


@app.command()
def score(
    molecules_sdf: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="Existing molecules with 3D coordinates in SDF format",
    ),
    docking_targets: Path = typer.Option(
        ...,
        "--docking-targets",
        exists=True,
        dir_okay=False,
        help="JSON file containing named receptor/center/size targets",
    ),
    output_dir: Path = typer.Option(
        Path("mtrl_scored/"),
        "--output-dir",
        help="Empty directory for scores, selected poses, and configuration",
    ),
    target_failure_score: float = typer.Option(
        0.0,
        "--target-failure-score",
        help="Objective value assigned to a target with no valid pose",
    ),
    accept_targets: str = typer.Option(
        "any",
        "--accept-targets",
        help="Retain a molecule when 'any' or 'all' targets have a valid pose",
    ),
    docking_mode: str = typer.Option(
        "flexible",
        "--docking-mode",
        help=(
            "Docking workflow: 'flexible', 'rigid', or 'rigid-refine' "
            "(rigid placement without CNN scoring, then GNINA minimization/rescoring)"
        ),
    ),
    gnina_timeout_seconds: int = typer.Option(
        600,
        "--gnina-timeout-seconds",
        help="Maximum wall time for each GNINA docking or minimization call",
    ),
    posebusters_timeout_seconds: int = typer.Option(
        600,
        "--posebusters-timeout-seconds",
        help="Maximum wall time for each receptor-aware PoseBusters call",
    ),
    posebusters_config: str = typer.Option(
        "dock",
        "--posebusters-config",
        help="PoseBusters checks: 'dock' or 'dock-fast' (omits ETKDG energy ratio)",
    ),
    qed_objective: bool = typer.Option(
        False,
        "--qed-objective/--no-qed-objective",
        help="Include RDKit QED in reported objective scores",
    ),
    evaluation_workers: int = typer.Option(
        _CLI_EVALUATION_WORKERS,
        "--evaluation-workers",
        help="Worker processes used concurrently for GNINA and PoseBusters",
    ),
    lilly_medchem_rules: bool = typer.Option(
        False,
        "--lilly-medchem-rules/--no-lilly-medchem-rules",
        help="Apply Lilly Medchem Rules in -relaxed mode before docking",
    ),
    lilly_rules_executable: str = typer.Option(
        "Lilly_Medchem_Rules.rb",
        "--lilly-rules-executable",
        help="Command name or path for Lilly_Medchem_Rules.rb",
    ),
    verbose_tools: bool = typer.Option(
        False,
        help="Show GNINA and PoseBusters output",
    ),
) -> None:
    """Dock and score existing 3D molecules without running RL."""
    from mtrl.box_score import score_sdf
    from mtrl.config import (
        BoxScoringConfig,
        DockingMode,
        PoseBustersConfig,
        TargetAcceptance,
        load_docking_targets,
    )

    if evaluation_workers <= 0:
        raise typer.BadParameter("--evaluation-workers must be > 0")
    if gnina_timeout_seconds <= 0:
        raise typer.BadParameter("--gnina-timeout-seconds must be > 0")
    if posebusters_timeout_seconds <= 0:
        raise typer.BadParameter("--posebusters-timeout-seconds must be > 0")
    if posebusters_config not in {"dock", "dock-fast"}:
        raise typer.BadParameter("--posebusters-config must be 'dock' or 'dock-fast'")
    if accept_targets not in {"any", "all"}:
        raise typer.BadParameter("--accept-targets must be 'any' or 'all'")
    if docking_mode not in {"flexible", "rigid", "rigid-refine"}:
        raise typer.BadParameter("--docking-mode must be 'flexible', 'rigid', or 'rigid-refine'")
    if not math.isfinite(target_failure_score):
        raise typer.BadParameter("--target-failure-score must be finite")
    try:
        targets = load_docking_targets(docking_targets.resolve())
        config = BoxScoringConfig(
            targets=targets,
            output_dir=output_dir.resolve(),
            lilly_medchem_rules=lilly_medchem_rules,
            lilly_rules_executable=lilly_rules_executable,
            verbose_tools=verbose_tools,
            evaluation_workers=evaluation_workers,
            target_failure_score=target_failure_score,
            accept_targets=cast(TargetAcceptance, accept_targets),
            docking_mode=cast(DockingMode, docking_mode),
            gnina_timeout_seconds=gnina_timeout_seconds,
            qed_objective=qed_objective,
            posebusters_config=cast(PoseBustersConfig, posebusters_config),
            posebusters_timeout_seconds=posebusters_timeout_seconds,
        )
        summary = score_sdf(molecules_sdf.resolve(), config)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


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
            "AMSR checkpoint (.pt) used as the initial policy; also used as the "
            "KL reference unless --kl-reference-checkpoint is provided"
        ),
    ),
    kl_reference_checkpoint: Path | None = typer.Option(
        None,
        "--kl-reference-checkpoint",
        exists=True,
        dir_okay=False,
        help=(
            "Optional frozen checkpoint used only for the KL penalty; use this when "
            "warm-starting from an RL checkpoint while retaining the original model "
            "as the regularization anchor"
        ),
        rich_help_panel="RL training",
    ),
    receptor_pdb: Path | None = typer.Option(
        None,
        "--receptor-pdb",
        exists=True,
        dir_okay=False,
        help=(
            "Legacy reference-minimization receptor; use with --reference-sdf, "
            "or use --docking-targets for full box docking"
        ),
        rich_help_panel="Scoring inputs",
    ),
    reference_sdf: Path | None = typer.Option(
        None,
        "--reference-sdf",
        exists=True,
        dir_okay=False,
        help=("Legacy 3D ligand used as GNINA's minimization box and for Roshambo2 alignment"),
        rich_help_panel="Scoring inputs",
    ),
    docking_targets: Path | None = typer.Option(
        None,
        "--docking-targets",
        exists=True,
        dir_okay=False,
        help=(
            "JSON file containing named receptor/center/size targets; each target "
            "becomes an independent full-docking CNNaffinity objective"
        ),
        rich_help_panel="Scoring inputs",
    ),
    target_failure_score: float = typer.Option(
        0.0,
        "--target-failure-score",
        help="Box mode: objective value assigned to a target with no valid pose",
        rich_help_panel="Scoring inputs",
    ),
    accept_targets: str = typer.Option(
        "any",
        "--accept-targets",
        help="Box mode: retain a molecule when 'any' or 'all' targets have a valid pose",
        rich_help_panel="Scoring inputs",
    ),
    docking_mode: str = typer.Option(
        "flexible",
        "--docking-mode",
        help=(
            "Box mode: 'flexible', 'rigid', or 'rigid-refine' (rigid placement "
            "without CNN scoring, then GNINA minimization/rescoring)"
        ),
        rich_help_panel="Scoring inputs",
    ),
    gnina_timeout_seconds: int = typer.Option(
        600,
        "--gnina-timeout-seconds",
        help="Box mode: maximum wall time for each GNINA docking or minimization call",
        rich_help_panel="Scoring inputs",
    ),
    posebusters_timeout_seconds: int = typer.Option(
        600,
        "--posebusters-timeout-seconds",
        help="Box mode: maximum wall time for each receptor-aware PoseBusters call",
        rich_help_panel="Scoring inputs",
    ),
    posebusters_config: str = typer.Option(
        "dock",
        "--posebusters-config",
        help="Box mode: 'dock' or 'dock-fast' (omits ETKDG energy ratio)",
        rich_help_panel="Scoring inputs",
    ),
    qed_objective: bool = typer.Option(
        False,
        "--qed-objective/--no-qed-objective",
        help="Box mode: add RDKit QED as an independent Pareto objective",
        rich_help_panel="Scoring inputs",
    ),
    evaluation_workers: int = typer.Option(
        _CLI_EVALUATION_WORKERS,
        "--evaluation-workers",
        help=(
            "Worker processes used concurrently for AMSR conformer construction, "
            "GNINA evaluation, and PoseBusters"
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
            "Final RL iteration number; for a fresh run this is the number of "
            "sample-score-update cycles, while a resumed run continues up to this step"
        ),
        rich_help_panel="RL training",
    ),
    resume_training_state: bool = typer.Option(
        False,
        "--resume-training-state/--fresh-training-state",
        help=(
            "Restore optimizer, learning-rate schedule, scaler, RNG, and saved value "
            "head from the policy checkpoint; legacy checkpoints without a value head "
            "restore everything else and initialize only that head"
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
            "Pareto diversity weight: a cumulative-front bonus in reference mode "
            "and a crowding-distance bonus in explicit-box mode; absolute joint "
            "quality supplies the reference-mode base reward"
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
    """Molecular Pareto RL with reference minimization or explicit-box docking."""
    from trl.training.rl_train import rl_train

    from mtrl.config import (
        BoxScoringConfig,
        DockingMode,
        PoseBustersConfig,
        ScoringConfig,
        TargetAcceptance,
        load_docking_targets,
    )

    for name, value in (
        ("--iterations", iterations),
        ("--batch-size", batch_size),
        ("--lr", lr),
        ("--temperature", temperature),
        ("--temperature-final", temperature_final),
        ("--log-every", log_every),
        ("--evaluation-workers", evaluation_workers),
        ("--gnina-timeout-seconds", gnina_timeout_seconds),
        ("--posebusters-timeout-seconds", posebusters_timeout_seconds),
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
    if accept_targets not in {"any", "all"}:
        raise typer.BadParameter("--accept-targets must be 'any' or 'all'")
    if docking_mode not in {"flexible", "rigid", "rigid-refine"}:
        raise typer.BadParameter("--docking-mode must be 'flexible', 'rigid', or 'rigid-refine'")
    if posebusters_config not in {"dock", "dock-fast"}:
        raise typer.BadParameter("--posebusters-config must be 'dock' or 'dock-fast'")
    if not math.isfinite(target_failure_score):
        raise typer.BadParameter("--target-failure-score must be finite")
    if docking_targets is not None and (receptor_pdb is not None or reference_sdf is not None):
        raise typer.BadParameter(
            "--docking-targets cannot be combined with --receptor-pdb or --reference-sdf"
        )
    if docking_targets is None and (receptor_pdb is None or reference_sdf is None):
        raise typer.BadParameter(
            "provide --docking-targets, or provide both --receptor-pdb and --reference-sdf"
        )
    output_dir = output_dir.resolve()
    rank = int(os.environ.get("RANK", "0"))
    resume_step = 0
    value_head_resume = "not requested"
    if resume_training_state:
        import torch

        resume_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
        required = ("optimizer", "scheduler", "rng_states", "step")
        missing = [key for key in required if key not in resume_checkpoint]
        if missing:
            raise typer.BadParameter(f"resume checkpoint is missing fields: {missing}")
        resume_step = int(resume_checkpoint["step"])
        if resume_step >= iterations:
            raise typer.BadParameter(
                f"resume step {resume_step} must be below --iterations={iterations}"
            )
        value_head_resume = (
            "restored"
            if resume_checkpoint.get("training_state", {}).get("value_head") is not None
            else "initialized fresh (legacy checkpoint)"
        )
    if rank == 0 and output_dir.exists():
        if not output_dir.is_dir():
            raise typer.BadParameter(f"--output-dir is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise typer.BadParameter(f"--output-dir must be empty: {output_dir}")
    if docking_targets is not None:
        try:
            targets = load_docking_targets(docking_targets.resolve())
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise typer.BadParameter(str(error), param_hint="--docking-targets") from error
        config: ScoringConfig | BoxScoringConfig = BoxScoringConfig(
            targets=targets,
            output_dir=output_dir,
            lilly_medchem_rules=lilly_medchem_rules,
            lilly_rules_executable=lilly_rules_executable,
            verbose_tools=verbose_tools,
            evaluation_workers=evaluation_workers,
            target_failure_score=target_failure_score,
            accept_targets=cast(TargetAcceptance, accept_targets),
            docking_mode=cast(DockingMode, docking_mode),
            gnina_timeout_seconds=gnina_timeout_seconds,
            qed_objective=qed_objective,
            posebusters_config=cast(PoseBustersConfig, posebusters_config),
            posebusters_timeout_seconds=posebusters_timeout_seconds,
        )
        reward_description = "global-batch N-dimensional Pareto rank and crowding distance"
    else:
        assert receptor_pdb is not None and reference_sdf is not None
        config = ScoringConfig(
            receptor_pdb=receptor_pdb.resolve(),
            reference_sdf=reference_sdf.resolve(),
            output_dir=output_dir,
            lilly_medchem_rules=lilly_medchem_rules,
            lilly_rules_executable=lilly_rules_executable,
            verbose_tools=verbose_tools,
            evaluation_workers=evaluation_workers,
        )
        reward_description = (
            "reference-normalized CNNaffinity * Tanimoto similarity, "
            "plus cumulative-Pareto-front bonus"
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
                    "accept_targets": accept_targets if docking_targets is not None else None,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_every": checkpoint_every,
                    "docking_mode": docking_mode if docking_targets is not None else None,
                    "evaluation_workers": evaluation_workers,
                    "gnina_timeout_seconds": (
                        gnina_timeout_seconds if docking_targets is not None else None
                    ),
                    "posebusters_timeout_seconds": (
                        posebusters_timeout_seconds if docking_targets is not None else None
                    ),
                    "posebusters_config": (
                        posebusters_config if docking_targets is not None else None
                    ),
                    "qed_objective": qed_objective if docking_targets is not None else None,
                    "iterations": iterations,
                    "kl_beta": kl_beta,
                    "kl_reference_checkpoint": str(kl_reference_checkpoint.resolve())
                    if kl_reference_checkpoint is not None
                    else str(checkpoint.resolve()),
                    "lr": lr,
                    "pareto_lambda": pareto_lambda,
                    "reward": reward_description,
                    "precision": precision,
                    "resume_training_state": resume_training_state,
                    "resumed_from_step": resume_step,
                    "save_final_checkpoint": save_final_checkpoint,
                    "seed": seed,
                    "temperature": temperature,
                    "temperature_final": temperature_final,
                    "target_failure_score": (
                        target_failure_score if docking_targets is not None else None
                    ),
                    "warmup_steps": warmup_steps,
                    "value_head_resume": value_head_resume,
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
        reference_checkpoint_path=(
            str(kl_reference_checkpoint.resolve()) if kl_reference_checkpoint is not None else None
        ),
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
        resume_training_state=resume_training_state,
    )
