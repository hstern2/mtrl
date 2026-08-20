from __future__ import annotations

import json
import os
from collections.abc import Callable
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
    """Two-objective Pareto suite with topology and structure hard gates."""

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
        self._overall_front: list[dict[str, Any]] = []
        if self.rank == 0:
            self.best_dir.mkdir(parents=True, exist_ok=True)
        self.conformer_fn = conformer_fn
        self.pipeline = pipeline or StructureScoringPipeline(config)
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
        if self.rank == 0:
            score_records = [record for part in gathered for record in part["scores"]]
            accepted_records = [record for part in gathered for record in part["accepted"]]
            if score_records:
                self._append_scores(score_records)
            self._write_pareto_fronts(accepted_records)
        self.generation += 1
        return items

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
        generation_front = self._pareto_front(accepted)
        if generation_front:
            self._write_sdf(
                self.best_dir / f"generation_{generation:06d}.sdf",
                generation_front,
            )
        self._overall_front = self._pareto_front([*self._overall_front, *accepted])
        if self._overall_front:
            self._write_sdf(
                self.best_dir / "overall.sdf",
                self._overall_front,
            )

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
