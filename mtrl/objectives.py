from __future__ import annotations

import atexit
import json
import os
from collections.abc import Callable
from csv import DictWriter
from typing import Any

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Mol
from trl.objectives.base import Objective, Objectives, ScoredItem
from trl.objectives.pareto import nsga2_sort

from mtrl import DecodedAMSR, decode_amsr, make_conformer
from mtrl.config import ScoringConfig
from mtrl.lilly import LillyMedchemFilter
from mtrl.scoring import StructureScore, StructureScoringPipeline


class CNNaffinityObjective(Objective):
    def __init__(self) -> None:
        super().__init__(name="gnina_cnn_affinity", direction="maximize")

    def score_batch(self, items: list[Any]) -> list[float]:
        return [float(item.cnn_affinity) for item in items]


class RoshamboComboObjective(Objective):
    def __init__(self) -> None:
        super().__init__(name="roshambo_tanimoto_combo", direction="maximize")

    def score_batch(self, items: list[Any]) -> list[float]:
        return [float(item.roshambo_tanimoto_combo) for item in items]


class DockingObjectives(Objectives):
    """Two-objective molecular suite with absolute, Pareto-aware rewards."""

    DEFAULT_AFFINITY_TARGET = 10.0

    def __init__(
        self,
        config: ScoringConfig,
        *,
        decode_fn: Callable[[list[str]], DecodedAMSR | None] = decode_amsr,
        conformer_fn: Callable[[DecodedAMSR], Mol | None] = make_conformer,
        pipeline: StructureScoringPipeline | None = None,
        lilly_filter: LillyMedchemFilter | None = None,
    ) -> None:
        super().__init__(
            objectives=[CNNaffinityObjective(), RoshamboComboObjective()],
            decode_fn=decode_fn,
        )
        self.config = config
        self.generation = 1
        self.rank = int(os.environ.get("RANK", "0"))
        self.score_log = config.output_dir / "scores.jsonl"
        self.best_dir = config.output_dir / "best"
        self.generations_dir = config.output_dir / "generations"
        self.progress_file = config.output_dir / "progress.csv"
        self._progress: list[dict[str, Any]] = []
        self._overall_front: list[dict[str, Any]] = []
        self._previous_front: list[dict[str, Any]] = []
        if self.rank == 0:
            self.best_dir.mkdir(parents=True, exist_ok=True)
            self.generations_dir.mkdir(parents=True, exist_ok=True)
        owns_pipeline = pipeline is None
        self._pipeline_owns_conformer_construction = (
            owns_pipeline and conformer_fn is make_conformer
        )
        self.conformer_fn = conformer_fn
        self.pipeline = pipeline or StructureScoringPipeline(config)
        if owns_pipeline:
            atexit.register(self.close)
        self.reference_score: StructureScore | None = None
        if self.rank == 0 and owns_pipeline:
            self.reference_score = self._score_reference()
        self._share_reference_score()
        self.lilly_filter = lilly_filter
        if config.lilly_medchem_rules and self.lilly_filter is None:
            self.lilly_filter = LillyMedchemFilter(config.lilly_rules_executable)

    def evaluate(self, token_sequences: list[list[str]]) -> list[ScoredItem]:
        decoded = [self.decode_fn(sequence) for sequence in token_sequences]
        items = [ScoredItem(token_ids=[]) for _ in decoded]
        diagnostics: list[StructureScore | None] = [None] * len(decoded)

        candidate_indices = []
        candidate_decoded = []
        for index, candidate in enumerate(decoded):
            if candidate is None:
                items[index].valid = False
                items[index].rejection_reason = "AMSR decode failed"
                continue
            if len(Chem.GetMolFrags(candidate.mol)) != 1:
                items[index].valid = False
                items[index].rejection_reason = "molecule is disconnected"
                continue
            candidate_indices.append(index)
            candidate_decoded.append(candidate)

        if self.lilly_filter is not None and candidate_decoded:
            accepted = self.lilly_filter.accept_batch(
                [candidate.mol for candidate in candidate_decoded]
            )
            retained_indices = []
            retained_decoded = []
            for index, candidate, passed in zip(
                candidate_indices, candidate_decoded, accepted, strict=True
            ):
                if passed:
                    retained_indices.append(index)
                    retained_decoded.append(candidate)
                else:
                    items[index].valid = False
                    items[index].rejection_reason = "Lilly Medchem Rules (-relaxed) failed"
            candidate_indices = retained_indices
            candidate_decoded = retained_decoded

        conformer_indices = []
        candidate_mols = []
        results = []
        if self._pipeline_owns_conformer_construction and candidate_decoded:
            evaluated = self.pipeline.evaluate_decoded_batch(candidate_decoded)
            for index, (conformer, result) in zip(candidate_indices, evaluated, strict=True):
                if conformer is None:
                    items[index].valid = False
                    items[index].rejection_reason = result.rejection_reason
                    continue
                conformer_indices.append(index)
                candidate_mols.append(conformer)
                results.append(result)
        else:
            for index, candidate in zip(candidate_indices, candidate_decoded, strict=True):
                conformer = self.conformer_fn(candidate)
                if conformer is None:
                    items[index].valid = False
                    items[index].rejection_reason = "AMSR conformer construction failed"
                    continue
                conformer_indices.append(index)
                candidate_mols.append(conformer)
            results = self.pipeline.score_batch(candidate_mols) if candidate_mols else []
        output_mols: list[Mol | None] = [None] * len(decoded)
        for index, result, candidate_mol in zip(
            conformer_indices, results, candidate_mols, strict=True
        ):
            diagnostics[index] = result
            if not result.accepted:
                items[index].valid = False
                items[index].rejection_reason = result.rejection_reason
                continue
            output_mols[index] = (
                result.minimized_mol if result.minimized_mol is not None else candidate_mol
            )
            for objective in self.objectives:
                score = objective.score_batch([result])[0]
                items[index].scores[objective.name] = score

        payload = {
            "scores": self._score_records(token_sequences, decoded, items, diagnostics),
            "accepted": self._accepted_records(
                token_sequences,
                decoded,
                items,
                diagnostics,
                output_mols,
            ),
        }
        gathered = self._gather_output(payload)
        score_records = [record for part in gathered for record in part["scores"]]
        accepted_records = [record for part in gathered for record in part["accepted"]]
        self._previous_front = self._overall_front
        self._overall_front = self._pareto_front([*self._overall_front, *accepted_records])
        if self.rank == 0:
            if score_records:
                self._append_scores(score_records)
            self._write_pareto_fronts(accepted_records)
            self._write_progress(score_records, accepted_records)
        self.generation += 1
        return items

    def get_rewards(self, scored: list[ScoredItem]) -> np.ndarray:
        """Reward fixed-scale joint quality, plus genuine cumulative-front progress.

        The base reward is the dominated area from a fixed zero nadir: reference-
        normalized CNNaffinity multiplied by Tanimoto similarity. This is monotonic
        in both objectives and remains comparable between generations. A molecule
        that adds a new point to the cumulative Pareto front receives the configured
        ``pareto_lambda`` bonus. Invalid molecules receive zero.
        """
        rewards = np.zeros(len(scored), dtype=float)
        affinity_target = self._affinity_target()
        previous_scores = self._front_scores(self._previous_front)
        current_scores = self._front_scores(self._overall_front)

        for index, item in enumerate(scored):
            if not item.valid:
                continue
            affinity = float(item.scores["gnina_cnn_affinity"])
            similarity = float(item.scores["roshambo_tanimoto_combo"])
            affinity_fraction = float(np.clip(affinity / affinity_target, 0.0, 1.0))
            similarity_fraction = float(np.clip(similarity, 0.0, 1.0))
            reward = affinity_fraction * similarity_fraction
            score = (affinity, similarity)
            if self._contains_score(current_scores, score) and not self._contains_score(
                previous_scores, score
            ):
                reward += self.pareto_lambda
            rewards[index] = reward
        return rewards

    def _affinity_target(self) -> float:
        if self.reference_score is None or self.reference_score.cnn_affinity is None:
            return self.DEFAULT_AFFINITY_TARGET
        affinity = float(self.reference_score.cnn_affinity)
        return affinity if affinity > 0 else self.DEFAULT_AFFINITY_TARGET

    @staticmethod
    def _front_scores(records: list[dict[str, Any]]) -> list[tuple[float, float]]:
        return [
            (float(record["cnn_affinity"]), float(record["tanimoto_combo"]))
            for record in records
        ]

    @staticmethod
    def _contains_score(scores: list[tuple[float, float]], target: tuple[float, float]) -> bool:
        return any(np.allclose(score, target, rtol=1e-12, atol=1e-12) for score in scores)

    def _share_reference_score(self) -> None:
        if not torch.distributed.is_initialized():
            return
        payload: list[dict[str, Any] | None] = [
            (
                {
                    "cnn_affinity": self.reference_score.cnn_affinity,
                    "roshambo_tanimoto_combo": self.reference_score.roshambo_tanimoto_combo,
                    "minimized_rmsd": self.reference_score.minimized_rmsd,
                    "accepted": self.reference_score.accepted,
                    "rejection_reason": self.reference_score.rejection_reason,
                }
                if self.reference_score is not None
                else None
            )
        ]
        torch.distributed.broadcast_object_list(payload, src=0)
        if self.rank != 0:
            values = payload[0]
            if values is None:
                raise RuntimeError("rank 0 did not provide the reference-ligand score")
            self.reference_score = StructureScore(**values)

    def _score_records(
        self,
        token_sequences: list[list[str]],
        decoded: list[DecodedAMSR | None],
        items: list[ScoredItem],
        diagnostics: list[StructureScore | None],
    ) -> list[dict[str, Any]]:
        records = []
        for index, (tokens, candidate, item, diagnostic) in enumerate(
            zip(token_sequences, decoded, items, diagnostics, strict=True)
        ):
            records.append(
                {
                    "generation": self.generation,
                    "rank": self.rank,
                    "index": index,
                    "amsr": "".join(tokens),
                    "smiles": (
                        Chem.MolToSmiles(candidate.mol, isomericSmiles=True)
                        if candidate is not None
                        else None
                    ),
                    "accepted": item.valid,
                    "rejection_reason": item.rejection_reason,
                    "cnn_affinity": (diagnostic.cnn_affinity if diagnostic is not None else None),
                    "tanimoto_combo": (
                        diagnostic.roshambo_tanimoto_combo if diagnostic is not None else None
                    ),
                    "minimized_rmsd": (
                        diagnostic.minimized_rmsd if diagnostic is not None else None
                    ),
                }
            )
        return records

    def _accepted_records(
        self,
        token_sequences: list[list[str]],
        decoded: list[DecodedAMSR | None],
        items: list[ScoredItem],
        diagnostics: list[StructureScore | None],
        output_mols: list[Mol | None],
    ) -> list[dict[str, Any]]:
        records = []
        for index, (tokens, candidate, item, diagnostic, mol) in enumerate(
            zip(
                token_sequences,
                decoded,
                items,
                diagnostics,
                output_mols,
                strict=True,
            )
        ):
            if not item.valid or candidate is None or diagnostic is None or mol is None:
                continue
            affinity = diagnostic.cnn_affinity
            combo = diagnostic.roshambo_tanimoto_combo
            rmsd = diagnostic.minimized_rmsd
            if affinity is None or combo is None or rmsd is None:
                raise RuntimeError("accepted structure is missing a score or minimized RMSD")
            records.append(
                {
                    "generation": self.generation,
                    "rank": self.rank,
                    "index": index,
                    "amsr": "".join(tokens),
                    "smiles": Chem.MolToSmiles(candidate.mol, isomericSmiles=True),
                    "cnn_affinity": float(affinity),
                    "tanimoto_combo": float(combo),
                    "minimized_rmsd": float(rmsd),
                    "mol_block": Chem.MolToMolBlock(mol),
                }
            )
        return records

    @staticmethod
    def _gather_output(payload: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        if not torch.distributed.is_initialized():
            return [payload]
        gathered: list[dict[str, Any] | None] = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, payload)
        return [part for part in gathered if part is not None]

    def _append_scores(self, records: list[dict[str, Any]]) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        with self.score_log.open("a") as output:
            for record in records:
                output.write(json.dumps(record, sort_keys=True) + "\n")

    @staticmethod
    def _pareto_front(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not records:
            return []
        scores = np.asarray(
            [[record["cnn_affinity"], record["tanimoto_combo"]] for record in records],
            dtype=float,
        )
        fronts, _ = nsga2_sort(scores)
        front = [records[index] for index in fronts[0]]
        unique = {
            (
                record["amsr"],
                record["cnn_affinity"],
                record["tanimoto_combo"],
            ): record
            for record in front
        }
        return sorted(
            unique.values(),
            key=lambda record: (
                -record["cnn_affinity"],
                -record["tanimoto_combo"],
                record["amsr"],
            ),
        )

    def _write_pareto_fronts(self, accepted: list[dict[str, Any]]) -> None:
        generation = self.generation
        if accepted:
            self._write_sdf(
                self.generations_dir / f"generation_{generation:06d}.sdf",
                accepted,
            )
        generation_front = self._pareto_front(accepted)
        if generation_front:
            self._write_sdf(
                self.best_dir / f"generation_{generation:06d}.sdf",
                generation_front,
            )
        if self._overall_front:
            self._write_sdf(
                self.best_dir / "overall.sdf",
                self._overall_front,
            )

    def _score_reference(self) -> StructureScore:
        reference = next(
            (
                mol
                for mol in Chem.SDMolSupplier(str(self.config.reference_sdf), removeHs=False)
                if mol is not None
            ),
            None,
        )
        if reference is None:
            raise RuntimeError(f"cannot read original ligand: {self.config.reference_sdf}")
        score = self.pipeline.score_batch([reference])[0]
        payload = {
            "label": "original T9C",
            "input_sdf": str(self.config.reference_sdf),
            "cnn_affinity": score.cnn_affinity,
            "tanimoto_combo": score.roshambo_tanimoto_combo,
            "identity_tanimoto_similarity": 1.0,
            "minimized_rmsd": score.minimized_rmsd,
            "accepted": score.accepted,
            "rejection_reason": score.rejection_reason,
        }
        (self.config.output_dir / "reference.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        if score.minimized_mol is not None:
            mol = Chem.Mol(score.minimized_mol)
            mol.SetProp("_Name", "original_T9C_GNINA_minimized")
            for key, value in payload.items():
                if value is not None:
                    mol.SetProp(key, str(value))
            output = self.config.output_dir / "reference_minimized.sdf"
            temporary = output.with_suffix(".sdf.tmp")
            writer = Chem.SDWriter(str(temporary))
            try:
                writer.write(mol)
            finally:
                writer.close()
            os.replace(temporary, output)
        return score

    def _write_progress(
        self, score_records: list[dict[str, Any]], accepted: list[dict[str, Any]]
    ) -> None:
        reasons = [record["rejection_reason"] for record in score_records if not record["accepted"]]
        affinities = [float(record["cnn_affinity"]) for record in accepted]
        similarities = [float(record["tanimoto_combo"]) for record in accepted]
        prior_generated = int(self._progress[-1]["cumulative_generated"]) if self._progress else 0
        prior_accepted = int(self._progress[-1]["cumulative_accepted"]) if self._progress else 0
        prior_best = self._progress[-1]["running_best_cnn_affinity"] if self._progress else None
        best_affinity = max(affinities) if affinities else None
        running_best = (
            max(value for value in (prior_best, best_affinity) if value is not None)
            if prior_best is not None or best_affinity is not None
            else None
        )
        reference = self.reference_score
        row: dict[str, Any] = {
            "generation": self.generation,
            "generated": len(score_records),
            "cumulative_generated": prior_generated + len(score_records),
            "accepted": len(accepted),
            "accepted_percent": 100.0 * len(accepted) / max(1, len(score_records)),
            "cumulative_accepted": prior_accepted + len(accepted),
            "decode_failed": reasons.count("AMSR decode failed"),
            "disconnected_failed": reasons.count("molecule is disconnected"),
            "lilly_failed": sum(reason.startswith("Lilly Medchem Rules") for reason in reasons),
            "conformer_failed": reasons.count("AMSR conformer construction failed"),
            "posebusters_failed": reasons.count("PoseBusters failed"),
            "scoring_failed": sum(
                not (
                    reason == "AMSR decode failed"
                    or reason == "molecule is disconnected"
                    or reason.startswith("Lilly Medchem Rules")
                    or reason == "AMSR conformer construction failed"
                    or reason == "PoseBusters failed"
                )
                for reason in reasons
            ),
            "mean_cnn_affinity": float(np.mean(affinities)) if affinities else None,
            "best_cnn_affinity": best_affinity,
            "running_best_cnn_affinity": running_best,
            "mean_tanimoto_combo": float(np.mean(similarities)) if similarities else None,
            "best_tanimoto_combo": max(similarities) if similarities else None,
            "original_t9c_cnn_affinity": (
                reference.cnn_affinity if reference is not None else None
            ),
            "original_t9c_tanimoto_combo": (
                reference.roshambo_tanimoto_combo if reference is not None else None
            ),
        }
        self._progress.append(row)
        write_header = not self.progress_file.exists()
        with self.progress_file.open("a", newline="") as output:
            writer = DictWriter(output, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self._plot_progress()

    def _plot_progress(self) -> None:
        from mtrl.report import write_affinity_progress, write_pareto_progress

        write_pareto_progress(self.config.output_dir)
        write_affinity_progress(self.config.output_dir)

    def close(self) -> None:
        self.pipeline.close()

    @staticmethod
    def _write_sdf(path: os.PathLike[str], records: list[dict[str, Any]]) -> None:
        output_path = os.fspath(path)
        temporary = f"{output_path}.tmp"
        writer = Chem.SDWriter(temporary)
        try:
            for molecule_number, record in enumerate(records, start=1):
                mol = Chem.MolFromMolBlock(record["mol_block"], removeHs=False)
                if mol is None:
                    continue
                mol.SetProp(
                    "_Name",
                    f"generation_{record['generation']:06d}_molecule_{molecule_number:04d}",
                )
                properties = {
                    "AMSR": record["amsr"],
                    "SMILES": record["smiles"],
                    "generation": record["generation"],
                    "CNNaffinity": record["cnn_affinity"],
                    "tanimoto_combo": record["tanimoto_combo"],
                    "minimized_rmsd": record["minimized_rmsd"],
                }
                for key, value in properties.items():
                    mol.SetProp(key, str(value))
                writer.write(mol)
        finally:
            writer.close()
        os.replace(temporary, output_path)


def build() -> DockingObjectives:
    """Factory loaded by trl after `mtrl rl` installs its scoring configuration."""
    return DockingObjectives(ScoringConfig.from_env())
