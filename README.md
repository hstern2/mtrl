# mtrl

Molecular generation with [AMSR](https://github.com/hstern2/amsr) token sequences, fine-tuned via multi-objective reinforcement learning. Built on [trl](https://github.com/hstern2/trl).

## How it works

1. **Tokenize** molecular structures (SDF, SMILES) into AMSR token sequences with data augmentation (randomized atom ordering).
2. **Pretrain** a transformer on AMSR token sequences using trl.
3. **RL fine-tune** with molecular objectives (synthetic accessibility, QED, drug-likeness filters) via Pareto-ranked REINFORCE.
4. **Evaluate** generated molecules for validity, uniqueness, novelty, and multi-objective Pareto analysis.

## Installation

```bash
# Install trl and amsr as editable dependencies
uv pip install -e ../trl -e ../amsr -e ".[dev]"
```

## Usage

```bash
# 1. Tokenize structures to JSONL
mtrl prepare ./raw_structures/ --output corpus.jsonl --augmentations 10

# 2. Build vocab + pretrain (via trl)
trl build-vocab corpus.jsonl --output vocab.json
torchrun --nproc_per_node=4 -m trl pretrain corpus.jsonl \
    --vocab vocab.json --epochs 10

# 3. RL fine-tune with molecular objectives
torchrun --nproc_per_node=4 -m trl rl checkpoints/best.pt corpus.jsonl \
    --vocab vocab.json --objectives mtrl.suite:build

# 4. Evaluate
mtrl evaluate checkpoints_rl/rl_final.pt --vocab vocab.json --n 5000
```

## Objectives

The default suite (`mtrl.suite:build`) includes:

| Objective | Direction | Description |
|-----------|-----------|-------------|
| SA score  | minimize  | Synthetic accessibility (RDKit), reject > 6.0 |
| QED       | maximize  | Quantitative estimate of drug-likeness |
| Drug-likeness filter | reject | MW, logP, HBD/HBA, PAINS |

Stubs for docking and neural affinity scoring are included for future use.

## Structure

```
mtrl/
  data/        AMSR detokenize bridge, SDF/SMILES corpus preparation
  objectives/  SA score, QED, drug-likeness filters, docking stub
  evaluation/  validity/uniqueness/novelty, Pareto analysis, conformational checks
  suite.py     objectives factory for trl RL
```
