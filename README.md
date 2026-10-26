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
2. rejects disconnected molecules and, optionally, failures of the RDKit
   drug-likeness, BR-SAScore, and Lilly Medchem Rules gates;
3. constructs the AMSR 3D conformer and aligns it to a reference ligand with
   Roshambo2;
4. minimizes the aligned pose with GNINA and records how far minimization moves
   it, without using that movement as a rejection gate;
5. runs receptor-aware PoseBusters on the minimized pose.

Accepted molecules maximize GNINA `CNNaffinity` and Roshambo2
`tanimoto_combination`. Their base reward is fixed between generations:
reference-normalized affinity multiplied by Tanimoto similarity. A molecule
that extends the cumulative Pareto front receives a small bonus. These gates do
not become reward objectives. The model-emitted conformer is scored; mtrl does
not generate replacement conformers.

### Install

GNINA and Open Babel's `obabel` must be available in `PATH`; the optional
medchem filter requires LillyMol. Explicit-box scoring also requires Uni-Dock
1.x (for example, `conda install -c conda-forge unidock`). Its executable may
be outside `PATH` when supplied with `--unidock-executable`.

```bash
uv sync
```

### Run

Start from the final pretrained `best.pt` checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 uv run mtrl rl /path/to/best.pt \
  --receptor-pdb receptor.pdb \
  --reference-sdf reference_ligand.sdf \
  --rdkit-druglike-filter \
  --max-br-sascore 5 \
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

## Explicit-box, multi-receptor docking and RL

Docking without a reference ligand is configured with a JSON target manifest.
Each receptor/box pair becomes one independently maximized `CNNaffinity`
objective. Optional AEV-PLIG rescoring adds a second objective per target:

```json
{
  "targets": [
    {
      "name": "site_a_open_1",
      "receptor_pdb": "receptors/site_a_open_1.pdb",
      "center": [4.108, 21.909, -18.092],
      "size": [23.8, 23.8, 23.8],
      "num_modes": 9,
      "region_points_pdb": "regions/site_a_open_1.pdb",
      "region_model": 1,
      "region_radius": 1.7,
      "min_region_heavy_atom_fraction": 0.9,
      "max_region_outside_distance": 1.5
    }
  ]
}
```

Centers, box sizes, radii, and distances are in angstroms. A region-points PDB
defines an optional irregular target region as the union of spheres with
`region_radius`. The fraction and maximum-outside-distance gates reject poses
that do not occupy that region after PoseBusters and before objective scoring.
`max_pose_centroid_distance` remains available for spherical regions. Receptor
and region paths may be absolute or relative to the manifest. Run with:

```bash
uv run mtrl validate-targets targets.json

CUDA_VISIBLE_DEVICES=0 uv run mtrl score molecules.sdf \
  --docking-targets targets.json \
  --unidock-executable /path/to/unidock \
  --gnina-timeout-seconds 600 \
  --posebusters-timeout-seconds 600 \
  --posebusters-config dock-fast \
  --rdkit-druglike-filter \
  --muegge-filter \
  --brenk-filter \
  --pfizer-3-75-filter \
  --max-br-sascore 5 \
  --max-clogp 4 \
  --target-failure-score 0 \
  --accept-targets any \
  --lilly-medchem-rules \
  --output-dir scored

CUDA_VISIBLE_DEVICES=0 uv run mtrl rl /path/to/best.pt \
  --docking-targets targets.json \
  --unidock-executable /path/to/unidock \
  --gnina-timeout-seconds 600 \
  --posebusters-timeout-seconds 600 \
  --posebusters-config dock-fast \
  --rdkit-druglike-filter \
  --muegge-filter \
  --brenk-filter \
  --pfizer-3-75-filter \
  --max-br-sascore 5 \
  --max-clogp 4 \
  --target-failure-score 0 \
  --accept-targets any \
  --lilly-medchem-rules \
  --evaluation-workers 1 \
  --output-dir run_box_rl
```

All enabled molecule filters run before conformer construction and docking.
The protonation filter is enabled by default; use `--no-protonation-filter` to
disable it. It checks whether the emitted protonation state is among
Dimorphite-DL predictions for pH 7.0–7.8. It rejects a molecule if the state is
absent, prediction fails, or more than 8 states are predicted. Set
`--max-protonation-variants N` to change that limit. The emitted AMSR molecule is
never altered; the filter applies to `mtrl score` and both RL modes. It accepts
any predicted state in the range rather than selecting a dominant state at pH
7.4. Repeated molecular structures are cached within each scoring process.

To enable [AEV-PLIG](https://github.com/isakvals/AEV-PLIG), provide its checkout
and Python environment to either `mtrl score` or box-mode `mtrl rl`:

```bash
--aev-plig-dir /path/to/AEV-PLIG \
--aev-plig-python /path/to/aev-plig/bin/python
```

Every PoseBusters-passing GNINA-minimized pose is rescored in bounded batches.
Explicit ligand hydrogens are restored without changing the
emitted ionization state. Each target gains a separately maximized
`aev_plig_affinity__<target>` predicted-pK objective; its best pose may differ
from the CNNaffinity-selected pose. AEV-selected poses use the
`__aev_plig.sdf` suffix. Tool-level failures stop the run; a target for which
AEV-PLIG produces no pose score receives the configured failure score.

`--rdkit-druglike-filter` enforces molecular weight ≤ 500, cLogP ≤ 5, H-bond
donors ≤ 5, H-bond acceptors ≤ 10, rotatable bonds ≤ 10, and TPSA ≤ 140.
`--muegge-filter` applies the published
[Muegge drug-likeness criteria](https://doi.org/10.1021/jm015507e), including
the minimum functionality requirement of more than one heteroatom.
`--brenk-filter` rejects structures matching RDKit's built-in
[Brenk alert catalog](https://doi.org/10.1002/cmdc.200700139), including long
aliphatic chains and other undesirable motifs.
`--pfizer-3-75-filter` applies the published
[Pfizer 3/75 guideline](https://doi.org/10.1016/j.bmcl.2008.07.071) as a hard
gate: it rejects a molecule when RDKit cLogP is greater than 3 and TPSA is less
than 75 Å².
`--max-br-sascore 5` applies the published USPTO/eMolecules BR-SAScore, on which
1 is easiest and 10 is hardest. Lilly Medchem Rules remains an independent
structural-alert gate. `--max-clogp 4` independently rejects molecules whose
RDKit cLogP exceeds 4.

For each target, mtrl sends the whole scoring batch to Uni-Dock in one GPU call.
The input conformers are rigid (`TORSDOF 0`), so Uni-Dock searches only rotation
and translation. GNINA then locally minimizes and CNN-rescores every pose.
PoseBusters checks every refined pose, and each objective uses its highest score
among the passing poses. A molecule is retained when at least one target has a
passing pose. Failed targets receive the configured failure score (`0.0` by
default). Selected poses retain the Uni-Dock Vina score and conformer RMSDs
before and after GNINA refinement.

`--unidock-search-mode` selects Uni-Dock's `fast`, `balance`, or `detail` preset;
`fast` is the default. `--evaluation-workers` controls concurrent GNINA
refinement/PoseBusters tasks, and `--gnina-cpu N` bounds each GNINA process.

For RL, every distinct emitted AMSR conformer is docked. Exact duplicate AMSR
strings within a generation are skipped, while candidates are grouped by
canonical isomeric SMILES for Pareto ranking. Each objective uses the best
passing conformer of a molecule, and each molecule receives one reward budget,
divided only among conformers that supplied at least one best objective value.
This preserves conformer search without allowing repeated conformers to
multiply a molecule's training influence. Cumulative Pareto outputs contain
one record per canonical molecule; each target SDF contains the conformer and
pose that supplied that target's score. Progress reports include unique-
molecule and Bemis–Murcko scaffold counts and the largest molecule and scaffold
shares, both before and after scoring gates.

`mtrl score` applies the same target loading, molecule filters, docking,
PoseBusters checks, target acceptance policy, and failed-target score without
sampling or training a model. Its input SDF must contain 3D coordinates. It
writes `scores.jsonl`, `summary.json`, `scoring_config.json`, and selected-pose
SDFs under `poses/`. `mtrl validate-targets` resolves relative receptor paths and
validates all names, files, centers, box sizes, and mode counts without invoking
external scoring tools.

The Python API can add molecular objectives without changing the docking
pipeline. Pass `Objective` instances through
`BoxDockingObjectives(additional_objectives=...)`; each objective receives a
batch of `BoxObjectiveInput` values containing the decoded RDKit molecule and
all selected per-target poses and diagnostics. Objectives may maximize or
minimize, and batched evaluation leaves room for more expensive methods such as
ABFE. Adding such methods to the CLI still requires an explicit configuration
and execution backend; ABFE itself is not implemented here.

For each selected pose, `scores.jsonl` and the output SDF metadata also record
`pose_centroid_distance`, the distance in angstroms from the ligand heavy-atom
centroid to that target's configured box center. This supports calibration and
auditing of site attribution when padded search boxes overlap.

`--unidock-timeout-seconds` bounds each batched placement call;
`--gnina-timeout-seconds` bounds each local refinement. A timeout fails that
target evaluation without blocking the full run.

`--posebusters-timeout-seconds` similarly bounds each receptor-aware
PoseBusters evaluation. PoseBusters still gates every final refined pose; a
timeout rejects that target evaluation instead of blocking the scoring batch.

`--posebusters-config dock-fast` retains PoseBusters' chemistry, geometry,
ring, receptor-distance, clash, and volume-overlap checks while omitting its
ETKDG-based energy-ratio module. This avoids generating a second conformer
ensemble when the input conformer came from another model. The default is
`dock`, preserving the complete standard PoseBusters configuration.
