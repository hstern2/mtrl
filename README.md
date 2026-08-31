# mtrl

AMSR molecular generation and reinforcement learning built on the generic
token-sequence package [trl](https://github.com/hstern2/trl). `mtrl` owns AMSR
decoding, conformer construction, molecular filters, structure scoring, and the
molecular RL workflow.

## Generate conformers

Sample a pretrained checkpoint and decode the emitted AMSR geometry without
filters, cost functions, minimization, or RL:

```bash
uv run mtrl generate /path/to/best.pt -n 100 > conformers.sdf
```

Only SDF is written to stdout, with each record flushed as it is completed.
Each record includes the emitted AMSR string as a property. Transformer
sampling selects a conservative batch from available device memory, capped at
256; conformer construction respects CPU affinity, workload size, and a
16-worker cap. Override either with `--batch-size` or `--conformer-workers`.
Parallel construction preserves sampling order. The default seed is random;
pass `--seed INTEGER` to reproduce a run. Each record stores that seed as
`MTRL_SEED`. A stringent AMSR decode and successful conformer construction are
the only requirements for inclusion.

## Structure-scored RL

For each generated AMSR string, `mtrl`:

1. decodes the stringent topology and encoded dihedrals;
2. rejects disconnected molecules and, optionally, failures of Lilly Medchem
   Rules with `-relaxed`;
3. constructs the AMSR 3D conformer and aligns it to a reference ligand with
   Roshambo2;
4. minimizes the aligned pose with GNINA and records how far minimization moves
   it, without using that movement as a rejection gate;
5. runs receptor-aware PoseBusters on the minimized pose.

Accepted molecules maximize GNINA `CNNaffinity` and Roshambo2
`tanimoto_combination`. Their base reward is fixed between generations:
reference-normalized affinity multiplied by Tanimoto similarity. A molecule
that extends the cumulative Pareto front receives a small bonus. QED and the
former generic drug-likeness filter are not used. The model-emitted conformer
is scored; mtrl does not generate replacement conformers.

### Install

GNINA and LillyMol (when its optional filter is enabled) must be available in
`PATH`.

```bash
uv sync
```

### Run

Start from the final pretrained `best.pt` checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 uv run mtrl rl /path/to/best.pt \
  --receptor-pdb receptor.pdb \
  --reference-sdf reference_ligand.sdf \
  --lilly-medchem-rules \
  --output-dir run_rl
```

For a warm start from an RL checkpoint, add
`--kl-reference-checkpoint /path/to/original.pt` to keep the original model as
the fixed KL anchor.

The default batch is 16 molecules and the default run is 1,000 iterations.
Conformer construction and structure evaluation use a hardware-based worker
default, configurable with `--evaluation-workers`.
`run_rl/generations/generation_NNNNNN.sdf` contains every accepted molecule in
that generation; generations with no accepted molecules have no SDF.
`run_rl/best/overall.sdf` is the Pareto front across the whole run. The original
reference ligand is minimized and scored once in `reference_minimized.sdf` and
`reference.json`. `progress.csv` and `pareto_progress.png` summarize improvement
by generation. Every generated string, score, RMSD, and rejection reason is in
`scores.jsonl`. Temporary scoring files use the system temporary directory and
are removed after each molecule. External-tool chatter is hidden unless
`--verbose-tools` is set.
