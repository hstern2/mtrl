from __future__ import annotations

from rdkit.Chem import MolToSmiles

from mtrl import detokenize


def validity_rate(token_sequences: list[list[str]]) -> float:
    if not token_sequences:
        return 0.0
    valid = sum(1 for seq in token_sequences if detokenize(seq) is not None)
    return valid / len(token_sequences)


def uniqueness_rate(token_sequences: list[list[str]]) -> float:
    smiles_set: set[str] = set()
    valid_count = 0
    for seq in token_sequences:
        mol = detokenize(seq)
        if mol is not None:
            valid_count += 1
            smi = MolToSmiles(mol)
            if smi:
                smiles_set.add(smi)
    if valid_count == 0:
        return 0.0
    return len(smiles_set) / valid_count


def novelty_rate(
    token_sequences: list[list[str]],
    training_smiles: set[str],
) -> float:
    novel = 0
    total_unique = 0
    for seq in token_sequences:
        mol = detokenize(seq)
        if mol is not None:
            smi = MolToSmiles(mol)
            if smi:
                total_unique += 1
                if smi not in training_smiles:
                    novel += 1
    if total_unique == 0:
        return 0.0
    return novel / total_unique
