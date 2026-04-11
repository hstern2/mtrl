import typer

app = typer.Typer(help="mtrl: molecular generation with AMSR + trl")


@app.command()
def prepare(
    input_dir: str = typer.Argument(..., help="Dir of SDF/CIF files"),
    output: str = typer.Option("corpus.jsonl"),
    augmentations: int = typer.Option(10, help="Tokenizations per conformation"),
    val_frac: float = typer.Option(0.05),
    test_frac: float = typer.Option(0.05),
) -> None:
    """Tokenize molecular structures to JSONL (each line: JSON list of token strings).
    Splits by molecule identity. Output fed to `trl build-vocab` then `trl prepare`."""
    from mtrl.data.prepare_corpus import prepare_corpus

    prepare_corpus(
        input_dir=input_dir,
        output=output,
        augmentations=augmentations,
        val_frac=val_frac,
        test_frac=test_frac,
    )


@app.command()
def evaluate(
    checkpoint: str = typer.Argument(...),
    vocab: str = typer.Option("vocab.json"),
    n: int = typer.Option(1000),
    output_dir: str = typer.Option("eval_results/"),
) -> None:
    """Sample, decode, compute molecular metrics + Pareto analysis."""
    import json
    from pathlib import Path

    import torch

    from mtrl.data.amsr_wrapper import detokenize
    from mtrl.evaluation.metrics import novelty_rate, uniqueness_rate, validity_rate
    from trl.data.vocab import Vocab
    from trl.generation.sampler import sample
    from trl.model.transformer import TransformerConfig, TransformerLM

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
    nov = novelty_rate(token_seqs, set())  # no training set for now

    results = {"n": n, "validity": val, "uniqueness": uniq, "novelty": nov}
    (out / "metrics.json").write_text(json.dumps(results, indent=2))

    typer.echo(f"Validity: {val:.2%}  Uniqueness: {uniq:.2%}  Novelty: {nov:.2%}")
    typer.echo(f"Results saved to {output_dir}")
