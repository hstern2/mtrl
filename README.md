# mtrl

Molecular generation with [AMSR](https://github.com/hstern2/amsr) token sequences, fine-tuned via multi-objective reinforcement learning. Built on [trl](https://github.com/hstern2/trl).

## How it works

1. **Pretrain** a transformer on AMSR token sequences (JSONL corpus) using trl.
2. **RL fine-tune** with molecular objectives (synthetic accessibility, QED, drug-likeness filters) via Pareto-ranked REINFORCE.
3. **Evaluate** generated molecules for validity, uniqueness, and novelty.

Corpus preparation (tokenizing SDF/SMILES into AMSR JSONL) lives in the [amsr](https://github.com/hstern2/amsr) repo.

## Installation

`trl` and `amsr` are pulled from GitHub via `tool.uv.sources` in `pyproject.toml`.

```bash
uv sync
```

## Usage

```bash
# 1. Build vocab + pretrain (assumes corpus.jsonl already exists)
trl build-vocab corpus.jsonl --output vocab.json
torchrun --nproc_per_node=4 -m trl pretrain corpus.jsonl \
    --vocab vocab.json --epochs 10

# 2. RL fine-tune with molecular objectives
torchrun --nproc_per_node=4 -m trl rl checkpoints/best.pt corpus.jsonl \
    --vocab vocab.json --objectives mtrl.objectives:build

# 3. Evaluate
mtrl evaluate checkpoints_rl/rl_final.pt --vocab vocab.json --n 5000
```

## Objectives

The default suite (`mtrl.objectives:build`) includes:

| Objective | Direction | Description |
|-----------|-----------|-------------|
| SA score  | minimize  | Synthetic accessibility (RDKit), reject > 6.0 |
| QED       | maximize  | Quantitative estimate of drug-likeness |
| Drug-likeness filter | reject | MW, logP, HBD/HBA, PAINS |

## Structure

```
mtrl/
  __init__.py    detokenize bridge (AMSR tokens -> RDKit Mol)
  cli.py         `mtrl evaluate` command
  objectives.py  QED, SA score, drug-likeness filter, build() factory for trl RL
  metrics.py     validity / uniqueness / novelty
```
