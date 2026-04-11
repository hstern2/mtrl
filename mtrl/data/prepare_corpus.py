from __future__ import annotations

import json
from pathlib import Path
from random import shuffle

from amsr import FromMolToTokens
from rdkit.Chem import MolFromMolFile, MolFromSmiles, RenumberAtoms, SDMolSupplier


def tokenize_mol_augmented(mol, n_augmentations: int = 10) -> list[list[str]]:  # type: ignore[no-untyped-def]
    """Generate multiple AMSR tokenizations via atom-order shuffling."""
    results: list[list[str]] = []
    n_atoms = mol.GetNumAtoms()
    for _ in range(n_augmentations):
        order = list(range(n_atoms))
        shuffle(order)
        renum = RenumberAtoms(mol, order)
        try:
            tokens = FromMolToTokens(renum)
            if tokens:
                results.append(tokens)
        except Exception:
            continue
    return results


def prepare_corpus(
    input_dir: str,
    output: str = "corpus.jsonl",
    augmentations: int = 10,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
) -> None:
    """Read SDF files from input_dir, tokenize to JSONL."""
    input_path = Path(input_dir)
    all_token_lists: list[list[str]] = []

    # Process SDF files
    for sdf_file in sorted(input_path.glob("*.sdf")):
        supplier = SDMolSupplier(str(sdf_file), removeHs=True)
        for mol in supplier:
            if mol is None:
                continue
            all_token_lists.extend(tokenize_mol_augmented(mol, augmentations))

    # Process individual mol files
    for mol_file in sorted(input_path.glob("*.mol")):
        mol = MolFromMolFile(str(mol_file), removeHs=True)
        if mol is not None:
            all_token_lists.extend(tokenize_mol_augmented(mol, augmentations))

    # Process SMILES files (one SMILES per line)
    for smi_file in sorted(input_path.glob("*.smi")):
        with open(smi_file) as f:
            for line in f:
                smi = line.strip().split()[0]
                if not smi:
                    continue
                mol = MolFromSmiles(smi)
                if mol is not None:
                    all_token_lists.extend(tokenize_mol_augmented(mol, augmentations))

    # Split into train/val/test
    n = len(all_token_lists)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)

    test_data = all_token_lists[:n_test]
    val_data = all_token_lists[n_test : n_test + n_val]
    train_data = all_token_lists[n_test + n_val :]

    # Write JSONL files
    output_path = Path(output)
    _write_jsonl(train_data, str(output_path))
    if val_data:
        _write_jsonl(val_data, str(output_path.with_suffix(".val.jsonl")))
    if test_data:
        _write_jsonl(test_data, str(output_path.with_suffix(".test.jsonl")))

    print(f"Wrote {len(train_data)} train, {len(val_data)} val, {len(test_data)} test to {output}")


def _write_jsonl(data: list[list[str]], path: str) -> None:
    with open(path, "w") as f:
        for tokens in data:
            f.write(json.dumps(tokens) + "\n")
