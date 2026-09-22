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
`--resume-training-state --kl-reference-checkpoint /path/to/original.pt` to
restore its training state while keeping the original model as the fixed KL
anchor. Legacy checkpoints that omitted the value head restore all available
state and initialize only that head.

The default batch is 16 molecules and the default run is 1,000 iterations.
Conformer construction and structure evaluation use a hardware-based worker
default, configurable with `--evaluation-workers`.
`run_rl/generations/generation_NNNNNN.sdf` contains every accepted molecule in
that generation; generations with no accepted molecules have no SDF.
`run_rl/best/overall.sdf` is the Pareto front across the whole run. The original
reference ligand is minimized and scored once in `reference_minimized.sdf` and
`reference.json`. `summary.txt` gives concise cumulative acceptance, gate-failure,
and score statistics. `progress.csv` and `pareto_progress.png` show improvement
by generation. Every generated string, score, RMSD, and rejection reason is in
`scores.jsonl`. Temporary scoring files use the system temporary directory and
are removed after each molecule. External-tool chatter is hidden unless
`--verbose-tools` is set.

## Explicit-box, multi-receptor RL

Full GNINA docking without a reference ligand is configured with a JSON target
manifest. Each entry is one inseparable receptor/box specification and becomes
one independently maximized `CNNaffinity` objective:

```json
{
  "targets": [
    {
      "name": "site_a_open_1",
      "receptor_pdb": "receptors/site_a_open_1.pdb",
      "center": [4.108, 21.909, -18.092],
      "size": [23.8, 23.8, 23.8],
      "exhaustiveness": 8,
      "num_modes": 9
    }
  ]
}
```

Receptor paths may be absolute or relative to the manifest. Run with:

```bash
uv run mtrl validate-targets targets.json

CUDA_VISIBLE_DEVICES=0 uv run mtrl score molecules.sdf \
  --docking-targets targets.json \
  --docking-mode rigid-refine \
  --gnina-timeout-seconds 600 \
  --posebusters-timeout-seconds 600 \
  --posebusters-config dock-fast \
  --qed-objective \
  --target-failure-score 0 \
  --accept-targets any \
  --lilly-medchem-rules \
  --output-dir scored

CUDA_VISIBLE_DEVICES=0 uv run mtrl rl /path/to/best.pt \
  --docking-targets targets.json \
  --docking-mode rigid-refine \
  --gnina-timeout-seconds 600 \
  --posebusters-timeout-seconds 600 \
  --posebusters-config dock-fast \
  --qed-objective \
  --target-failure-score 0 \
  --accept-targets any \
  --lilly-medchem-rules \
  --evaluation-workers 1 \
  --output-dir run_box_rl
```

Lilly Medchem Rules remains a molecule-level gate before 3D work. GNINA performs
a full search in every configured box. Every returned pose is checked with the
receptor-aware PoseBusters docking configuration, and each objective uses the
highest `CNNaffinity` among that target's passing poses. A molecule is retained
when at least one target has a passing pose. Targets without a passing pose get
a dominated objective value of `0.0`; no pose is fabricated for them. Selected
valid poses are written as one SDF per target under each generation and
cumulative-front directory. The reference-aligned minimization workflow above
remains available unchanged.

`mtrl score` applies the same target loading, LillyMol filtering, full docking,
PoseBusters checks, target acceptance policy, and failed-target score without
sampling or training a model. Its input SDF must contain 3D coordinates. It
writes `scores.jsonl`, `summary.json`, `scoring_config.json`, and one selected-pose
SDF per target under `poses/`. `mtrl validate-targets` resolves relative receptor
paths and validates all names, files, centers, box sizes, exhaustiveness values,
and mode counts without invoking external scoring tools.

For each selected pose, `scores.jsonl` and the output SDF metadata also record
`pose_centroid_distance`, the distance in angstroms from the ligand heavy-atom
centroid to that target's configured box center. This supports calibration and
auditing of site attribution when padded search boxes overlap.

`--docking-mode rigid-refine` preserves the input conformer during global
placement. Open Babel writes a rigid PDBQT torsion tree (`TORSDOF 0`), and GNINA
searches translations and rotations with `--cnn_scoring none`. Each rigid pose
is then independently passed through GNINA `--minimize --cnn_scoring rescore`.
PoseBusters gates the resulting refined poses, and the highest refined
`CNNaffinity` among passing poses supplies that target's objective. Reports also
include the rigid-stage Vina affinity, the rigid pose's conformer RMSD from the
input, and the refined pose's conformer RMSD. `flexible` remains the default for
backwards compatibility; `rigid` performs rigid placement with CNN rescoring but
no minimization.

`--gnina-timeout-seconds` bounds each individual docking or minimization
subprocess. A timeout fails only that receptor/box evaluation, allowing long
multi-target and distributed runs to continue instead of waiting indefinitely
on a pathological ligand.

`--posebusters-timeout-seconds` similarly bounds each receptor-aware
PoseBusters evaluation. PoseBusters still gates every final refined pose; a
timeout rejects that target evaluation instead of blocking the scoring batch.

`--posebusters-config dock-fast` retains PoseBusters' chemistry, geometry,
ring, receptor-distance, clash, and volume-overlap checks while omitting its
ETKDG-based energy-ratio module. This avoids generating a second conformer
ensemble when the input conformer came from another model. The default is
`dock`, preserving the complete standard PoseBusters configuration.

`--qed-objective` adds RDKit QED as another independently maximized Pareto
axis. It is calculated from the decoded molecule without generating another
conformer. The option is off by default for backwards compatibility.
