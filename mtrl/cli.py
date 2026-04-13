import json
from pathlib import Path

import typer

app = typer.Typer(help="mtrl: molecular generation with AMSR + trl")


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
