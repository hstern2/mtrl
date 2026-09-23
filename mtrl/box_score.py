from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import QED, Mol

from mtrl.config import BoxScoringConfig
from mtrl.druglike import druglike_rejection_reason
from mtrl.lilly import LillyMedchemFilter
from mtrl.scoring import BoxDockingPipeline, BoxDockingScore
from mtrl.synthetic_accessibility import br_sascore_rejection_reason


def _objective_scores(
    result: BoxDockingScore, config: BoxScoringConfig, mol: Mol
) -> dict[str, float]:
    scores = {}
    for target in config.targets:
        target_result = result.targets[target.name]
        value = (
            float(target_result.cnn_affinity)
            if target_result.accepted and target_result.cnn_affinity is not None
            else config.target_failure_score
        )
        scores[f"gnina_cnn_affinity__{target.name}"] = value
    if config.qed_objective:
        scores["qed"] = float(QED.qed(Chem.RemoveHs(mol)))
    return scores


def _write_pose_sdfs(
    output_dir: Path,
    pose_records: dict[str, list[tuple[dict[str, Any], Mol]]],
) -> None:
    poses_dir = output_dir / "poses"
    if not any(pose_records.values()):
        return
    poses_dir.mkdir(parents=True, exist_ok=True)
    for target_name, records in pose_records.items():
        if not records:
            continue
        writer = Chem.SDWriter(str(poses_dir / f"{target_name}.sdf"))
        try:
            for record, pose in records:
                output = Chem.Mol(pose)
                output.SetProp("_Name", record["name"])
                output.SetProp("input_index", str(record["index"]))
                output.SetProp("SMILES", record["smiles"])
                output.SetProp("docking_target", target_name)
                for objective_name, value in record["objectives"].items():
                    output.SetProp(objective_name, str(value))
                for key, value in record["targets"][target_name].items():
                    if value is not None:
                        encoded = (
                            json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
                        )
                        output.SetProp(f"selected_pose_{key}", str(encoded))
                writer.write(output)
        finally:
            writer.close()


def score_sdf(
    input_sdf: Path,
    config: BoxScoringConfig,
    *,
    pipeline: BoxDockingPipeline | None = None,
    lilly_filter: LillyMedchemFilter | None = None,
) -> dict[str, Any]:
    """Dock an existing 3D SDF against named targets and write poses and scores."""
    config.validate()
    if not input_sdf.is_file():
        raise ValueError(f"input SDF does not exist: {input_sdf}")
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise ValueError(f"output directory must be empty: {config.output_dir}")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    candidate_indices: list[int] = []
    candidates: list[Mol] = []
    for index, mol in enumerate(Chem.SDMolSupplier(str(input_sdf), removeHs=False)):
        record: dict[str, Any] = {
            "index": index,
            "name": f"molecule_{index + 1:06d}",
            "smiles": None,
            "accepted": False,
            "rejection_reason": "",
            "objectives": {},
            "targets": {},
        }
        if mol is None:
            record["rejection_reason"] = "SDF record could not be parsed"
            records.append(record)
            continue
        if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
            record["name"] = mol.GetProp("_Name").strip()
        record["smiles"] = Chem.MolToSmiles(mol, isomericSmiles=True)
        if len(Chem.GetMolFrags(mol)) != 1:
            record["rejection_reason"] = "molecule is disconnected"
        elif config.rdkit_druglike_filter and (reason := druglike_rejection_reason(mol)):
            record["rejection_reason"] = reason
        elif config.max_br_sascore is not None and (
            reason := br_sascore_rejection_reason(mol, config.max_br_sascore)
        ):
            record["rejection_reason"] = reason
        elif mol.GetNumConformers() == 0 or not mol.GetConformer().Is3D():
            record["rejection_reason"] = "input molecule is missing 3D coordinates"
        else:
            candidate_indices.append(index)
            candidates.append(mol)
        records.append(record)

    if config.lilly_medchem_rules and lilly_filter is None:
        lilly_filter = LillyMedchemFilter(config.lilly_rules_executable)
    if lilly_filter is not None and candidates:
        passed = lilly_filter.accept_batch(candidates)
        retained_indices = []
        retained_candidates = []
        for record_index, mol, accepted in zip(candidate_indices, candidates, passed, strict=True):
            if accepted:
                retained_indices.append(record_index)
                retained_candidates.append(mol)
            else:
                records[record_index]["rejection_reason"] = "Lilly Medchem Rules (-relaxed) failed"
        candidate_indices = retained_indices
        candidates = retained_candidates

    owns_pipeline = pipeline is None
    pipeline = pipeline or BoxDockingPipeline(config)
    try:
        results = pipeline.score_batch(candidates) if candidates else []
    finally:
        if owns_pipeline:
            pipeline.close()

    pose_records: dict[str, list[tuple[dict[str, Any], Mol]]] = {
        target.name: [] for target in config.targets
    }
    for record_index, mol, result in zip(candidate_indices, candidates, results, strict=True):
        record = records[record_index]
        record["accepted"] = result.accepted
        record["rejection_reason"] = result.rejection_reason
        record["objectives"] = _objective_scores(result, config, mol)
        record["targets"] = {
            name: target_result.to_record() for name, target_result in result.targets.items()
        }
        if not result.accepted:
            continue
        for target_name, target_result in result.targets.items():
            if target_result.accepted and target_result.pose is not None:
                pose_records[target_name].append((record, target_result.pose))

    scores_path = config.output_dir / "scores.jsonl"
    scores_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    (config.output_dir / "scoring_config.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    _write_pose_sdfs(config.output_dir, pose_records)

    summary = {
        "input_records": len(records),
        "retained_molecules": sum(record["accepted"] for record in records),
        "all_targets_valid": sum(
            record["accepted"]
            and record["targets"]
            and all(
                record["targets"].get(target.name, {}).get("accepted", False)
                for target in config.targets
            )
            for record in records
        ),
        "per_target_valid_poses": {
            target.name: len(pose_records[target.name]) for target in config.targets
        },
        "scores_jsonl": str(scores_path),
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary
