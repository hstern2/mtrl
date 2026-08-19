# mtrl

Pareto reinforcement learning for the 3D AMSR transformer in
[trl](https://github.com/hstern2/trl).

For each generated AMSR string, `mtrl`:

1. decodes the stringent topology and encoded dihedrals;
2. rejects disconnected molecules and, optionally, failures of Lilly Medchem
   Rules with `-relaxed`;
3. constructs the AMSR 3D conformer and aligns it to a reference ligand with
   Roshambo2;
4. minimizes the aligned pose with GNINA, then runs receptor-aware PoseBusters;
5. rejects PoseBusters failures and poses whose symmetry-corrected heavy-atom
   movement during minimization exceeds 1.0 A by default.

Accepted molecules have two separately maximized Pareto objectives:
`CNNaffinity` and Roshambo2 `tanimoto_combination`. QED and the former generic
drug-likeness filter are not used. The model-emitted conformer is scored; mtrl
does not generate replacement conformers.

## Install

GNINA and LillyMol (when its optional filter is enabled) must be available in
`PATH`.

```bash
uv sync
```

## Run

Start from the final pretrained `best.pt` checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 uv run mtrl rl /path/to/best.pt \
  --receptor-pdb receptor.pdb \
  --reference-sdf reference_ligand.sdf \
  --lilly-medchem-rules \
  --max-minimized-rmsd 1.0 \
  --checkpoint-dir run_rl
```

The default batch is 16 molecules and the default run is 1,000 iterations.
`run_rl/scoring_config.json` records the exact scoring configuration. Set
`--keep-poses` to retain accepted aligned/minimized SDF pairs. Every generated
string, score, RMSD, and rejection reason is recorded under
`run_rl/scoring/rank_*/scores.jsonl`. External-tool chatter is hidden unless
`--verbose-tools` is set.
