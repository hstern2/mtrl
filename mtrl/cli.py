import json
import os
import sys
from pathlib import Path

import typer

app = typer.Typer(
    help="mtrl: molecular generation with AMSR + trl",
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command()
def generate(
    checkpoint: Path = typer.Argument(..., exists=True, dir_okay=False),
    n: int = typer.Option(100, "-n", help="Number of AMSR strings to sample"),
    batch_size: int = typer.Option(256, help="AMSR strings sampled together on the GPU"),
    conformer_workers: int = typer.Option(
        0,
        help="Parallel CPU conformer workers; 0 selects up to 16",
    ),
    temperature: float = typer.Option(0.8),
    top_k: int = typer.Option(0, help="Top-k sampling; 0 disables it"),
    top_p: float = typer.Option(1.0, help="Nucleus sampling threshold"),
    seed: int = typer.Option(0),
    device: str = typer.Option("auto", help="auto, cpu, cuda, or a CUDA device such as cuda:0"),
) -> None:
    """Sample AMSR strings and decode their encoded 3D conformers without scoring."""
    from mtrl.generate import generate as generate_conformers

    if n <= 0:
        raise typer.BadParameter("-n must be > 0")
    if batch_size <= 0:
        raise typer.BadParameter("--batch-size must be > 0")
    if conformer_workers < 0:
        raise typer.BadParameter("--conformer-workers must be >= 0")
    if temperature <= 0:
        raise typer.BadParameter("--temperature must be > 0")
    if top_k < 0:
        raise typer.BadParameter("--top-k must be >= 0")
    if not 0 < top_p <= 1:
        raise typer.BadParameter("--top-p must be in (0, 1]")

    if conformer_workers == 0:
        from mtrl.generate import default_conformer_workers

        conformer_workers = default_conformer_workers()

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
    checkpoint: Path = typer.Argument(..., exists=True, dir_okay=False),
    receptor_pdb: Path = typer.Option(..., "--receptor-pdb", exists=True, dir_okay=False),
    reference_sdf: Path = typer.Option(..., "--reference-sdf", exists=True, dir_okay=False),
    max_minimized_rmsd: float = typer.Option(
        1.0,
        "--max-minimized-rmsd",
        help="Reject when the aligned-to-minimized heavy-atom RMSD exceeds this value (A)",
    ),
    lilly_medchem_rules: bool = typer.Option(
        False,
        "--lilly-medchem-rules/--no-lilly-medchem-rules",
        help="Apply Lilly Medchem Rules in -relaxed mode before structure scoring",
    ),
    lilly_rules_executable: str = typer.Option(
        "Lilly_Medchem_Rules.rb",
        "--lilly-rules-executable",
    ),
    iterations: int = typer.Option(1000),
    batch_size: int = typer.Option(16, help="Total generated molecules per RL iteration"),
    lr: float = typer.Option(1e-5),
    kl_beta: float = typer.Option(0.05),
    pareto_lambda: float = typer.Option(0.1),
    temperature: float = typer.Option(1.0),
    temperature_final: float = typer.Option(0.8),
    replay_fraction: float = typer.Option(0.0, hidden=True),
    precision: str = typer.Option("auto", help="Precision: auto, fp32, fp16, or bf16"),
    checkpoint_every: int = typer.Option(100),
    log_every: int = typer.Option(10),
    checkpoint_dir: Path = typer.Option(Path("checkpoints_rl/")),
    work_dir: Path | None = typer.Option(None),
    keep_poses: bool = typer.Option(False, help="Keep accepted aligned/minimized SDF pairs"),
    record_scores: bool = typer.Option(
        True, help="Record every generated AMSR string, score, and rejection reason"
    ),
    verbose_tools: bool = typer.Option(False, help="Show Roshambo2, GNINA, and PoseBusters output"),
    wandb_project: str | None = typer.Option(None),
) -> None:
    """Pareto RL using GNINA affinity and Roshambo2 shape/color similarity."""
    from trl.training.rl_train import rl_train

    from mtrl.config import ScoringConfig

    if iterations <= 0:
        raise typer.BadParameter("--iterations must be > 0")
    if batch_size <= 0:
        raise typer.BadParameter("--batch-size must be > 0")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if batch_size % world_size:
        raise typer.BadParameter(
            f"--batch-size={batch_size} must be divisible by WORLD_SIZE={world_size}"
        )
    output_dir = checkpoint_dir.resolve()
    scoring_work_dir = (work_dir or (output_dir / "scoring")).resolve()
    config = ScoringConfig(
        receptor_pdb=receptor_pdb.resolve(),
        reference_sdf=reference_sdf.resolve(),
        work_dir=scoring_work_dir,
        max_minimized_rmsd=max_minimized_rmsd,
        lilly_medchem_rules=lilly_medchem_rules,
        lilly_rules_executable=lilly_rules_executable,
        keep_poses=keep_poses,
        record_scores=record_scores,
        verbose_tools=verbose_tools,
    )
    config.install()

    if int(os.environ.get("RANK", "0")) == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "scoring_config.json").write_text(
            json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n"
        )

    rl_train(
        checkpoint_path=str(checkpoint),
        vocab_path=None,
        objectives_path="mtrl.objectives:build",
        iterations=iterations,
        batch_size=batch_size,
        lr=lr,
        kl_beta=kl_beta,
        pareto_lambda=pareto_lambda,
        temperature=temperature,
        temperature_final=temperature_final,
        replay_fraction=replay_fraction,
        precision=precision,
        checkpoint_every=checkpoint_every,
        log_every=log_every,
        checkpoint_dir=str(output_dir),
        wandb_project=wandb_project,
    )


@app.command()
def evaluate(
    checkpoint: str = typer.Argument(...),
    vocab: str = typer.Option("vocab.json"),
    n: int = typer.Option(1000),
    output_dir: str = typer.Option("eval_results/"),
) -> None:
    """Sample from a checkpoint, decode, and compute validity/uniqueness/novelty."""
    import torch
    from trl.data.vocab import Vocab
    from trl.generation.sampler import sample
    from trl.model.transformer import TransformerConfig, TransformerLM

    from mtrl.metrics import novelty_rate, uniqueness_rate, validity_rate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    v = Vocab.load(vocab)

    config = TransformerConfig(**ckpt["config"])
    model = TransformerLM(config).to(device)
    model.load_state_dict(ckpt["model"])

    sequences = sample(model, n, device=device)
    token_seqs = [v.decode(seq) for seq in sequences]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    val = validity_rate(token_seqs)
    uniq = uniqueness_rate(token_seqs)
    nov = novelty_rate(token_seqs, set())

    results = {"n": n, "validity": val, "uniqueness": uniq, "novelty": nov}
    (out / "metrics.json").write_text(json.dumps(results, indent=2))

    typer.echo(f"Validity: {val:.2%}  Uniqueness: {uniq:.2%}  Novelty: {nov:.2%}")
    typer.echo(f"Results saved to {output_dir}")
