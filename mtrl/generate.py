from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import ceil
from pathlib import Path
from typing import TextIO

import torch
from rdkit import Chem
from trl.data.vocab import Vocab
from trl.generation.sampler import sample
from trl.model.transformer import TransformerConfig, TransformerLM

from mtrl import detokenize
from mtrl.conformer import build_conformer


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
    return token_sequences


def default_sampling_batch_size(device: torch.device) -> int:
    """Choose a conservative sampling batch with one inexpensive device query."""
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(device)
        free_gib = free_bytes / 1024**3
        if free_gib >= 10:
            return 256
        if free_gib >= 5:
            return 128
        if free_gib >= 2.5:
            return 64
        return 32
    if device.type == "mps":
        return 64
    return 32


def available_cpu_count() -> int:
    """Return CPUs available to this process, respecting affinity when possible."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def default_conformer_workers(task_count: int) -> int:
    """Choose enough workers for throughput without excessive process startup."""
    hardware_limit = min(16, max(1, available_cpu_count() - 1))
    workload_limit = max(1, ceil(task_count / 4))
    return min(hardware_limit, workload_limit)


def _write_mol(
    writer: Chem.SDWriter,
    mol: Chem.Mol,
    sample_index: int,
    tokens: list[str],
) -> None:
    amsr = "".join(tokens)
    mol.SetProp("_Name", f"sample_{sample_index:06d}")
    mol.SetIntProp("MTRL_SAMPLE_INDEX", sample_index)
    mol.SetProp("AMSR", amsr)
    mol.SetProp("CANONICAL_SMILES", Chem.MolToSmiles(mol))
    writer.write(mol)
    writer.flush()


def write_conformers(
    token_sequences: list[list[str]],
    output: TextIO,
    *,
    workers: int = 1,
) -> dict[str, float | int]:
    if workers <= 0:
        raise ValueError("conformer workers must be positive")
    workers = min(workers, max(1, len(token_sequences)))
    writer = Chem.SDWriter(output)
    generated = 0
    try:
        if workers == 1:
            for sample_index, tokens in enumerate(token_sequences):
                mol = detokenize(tokens)
                if mol is not None:
                    _write_mol(writer, mol, sample_index, tokens)
                    generated += 1
        else:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                futures = {
                    pool.submit(build_conformer, tokens): (sample_index, tokens)
                    for sample_index, tokens in enumerate(token_sequences)
                }
                for future in as_completed(futures):
                    sample_index, tokens = futures[future]
                    mol = future.result()
                    if mol is not None:
                        _write_mol(writer, mol, sample_index, tokens)
                        generated += 1
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
    conformer_workers: int,
) -> dict[str, float | int]:
    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model, vocab, max_len = load_model(checkpoint, device)
    if batch_size == 0:
        batch_size = default_sampling_batch_size(device)
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
    )
    if conformer_workers == 0:
        conformer_workers = default_conformer_workers(len(token_sequences))
    return write_conformers(
        token_sequences,
        output,
        workers=conformer_workers,
    )
