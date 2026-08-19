from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TextIO

import torch
from rdkit import Chem
from trl.data.vocab import Vocab
from trl.generation.sampler import sample
from trl.model.transformer import TransformerConfig, TransformerLM

from mtrl import detokenize

ProgressCallback = Callable[[str, int, int], None]


def _strip_wrapper_prefix(name: str) -> str:
    for prefix in ("module.", "_orig_mod."):
        if name.startswith(prefix):
            return _strip_wrapper_prefix(name.removeprefix(prefix))
    return name


def load_model(checkpoint: Path, device: torch.device) -> tuple[TransformerLM, Vocab, int]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for key in ("config", "model", "vocab"):
        if key not in state:
            raise ValueError(f"checkpoint has no {key!r}: {checkpoint}")

    vocab = Vocab(dict(state["vocab"]))
    config = TransformerConfig(**state["config"])
    model = TransformerLM(config)
    weights = {_strip_wrapper_prefix(name): value for name, value in state["model"].items()}
    model.load_state_dict(weights)
    model.to(device).eval()
    return model, vocab, config.max_seq_len


def sample_tokens(
    model: TransformerLM,
    vocab: Vocab,
    *,
    n: int,
    batch_size: int,
    max_len: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: torch.device,
    progress: ProgressCallback | None = None,
) -> list[list[str]]:
    token_sequences: list[list[str]] = []
    while len(token_sequences) < n:
        current_batch = min(batch_size, n - len(token_sequences))
        ids = sample(
            model,
            current_batch,
            max_len=max_len,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
        )
        token_sequences.extend(vocab.decode(sequence) for sequence in ids)
        if progress is not None:
            progress("sample", len(token_sequences), n)
    return token_sequences


def write_conformers(
    token_sequences: list[list[str]],
    output: TextIO,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, float | int]:
    writer = Chem.SDWriter(output)
    generated = 0
    total = len(token_sequences)
    try:
        for sample_index, tokens in enumerate(token_sequences):
            amsr = "".join(tokens)
            mol = detokenize(tokens)
            if mol is not None:
                mol.SetProp("_Name", f"sample_{sample_index:06d}")
                mol.SetIntProp("MTRL_SAMPLE_INDEX", sample_index)
                mol.SetProp("AMSR", amsr)
                mol.SetProp("CANONICAL_SMILES", Chem.MolToSmiles(mol))
                writer.write(mol)
                generated += 1
            completed = sample_index + 1
            if progress is not None and (completed % 50 == 0 or completed == total):
                writer.flush()
                progress("conformer", completed, total)
    finally:
        writer.close()

    return {
        "sampled_strings": len(token_sequences),
        "decoded_conformers": generated,
        "decode_failures": len(token_sequences) - generated,
        "valid_fraction": generated / len(token_sequences) if token_sequences else 0.0,
    }


def generate(
    checkpoint: Path,
    output: TextIO,
    *,
    n: int,
    batch_size: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    device_name: str,
    progress: ProgressCallback | None = None,
) -> dict[str, float | int]:
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model, vocab, max_len = load_model(checkpoint, device)
    token_sequences = sample_tokens(
        model,
        vocab,
        n=n,
        batch_size=batch_size,
        max_len=max_len,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
        progress=progress,
    )
    return write_conformers(
        token_sequences,
        output,
        progress=progress,
    )
