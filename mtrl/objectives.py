from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol
from trl.objectives.base import Objective, Objectives, ScoredItem

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
        self.evaluation_number = 0
        self.rank = int(os.environ.get("RANK", "0"))
        self.score_log = config.work_dir / f"rank_{self.rank}" / "scores.jsonl"
        if config.record_scores:
            self.score_log.parent.mkdir(parents=True, exist_ok=True)
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
        for index, result in zip(conformer_indices, results, strict=True):
            diagnostics[index] = result
            if not result.accepted:
                items[index].valid = False
                items[index].rejection_reason = result.rejection_reason
                continue
            for objective in self.objectives:
                score = objective.score_batch([result])[0]
                items[index].scores[objective.name] = score

        if self.config.record_scores:
            self._record(token_sequences, decoded, items, diagnostics)
        self.evaluation_number += 1
        return items

    def get_rewards(self, scored: list[ScoredItem]) -> np.ndarray:
        return super().get_rewards(scored)

    def _record(
        self,
        token_sequences: list[list[str]],
        decoded: list[DecodedAMSR | None],
        items: list[ScoredItem],
        diagnostics: list[StructureScore | None],
    ) -> None:
        with self.score_log.open("a") as output:
            for index, (tokens, candidate, item, diagnostic) in enumerate(
                zip(token_sequences, decoded, items, diagnostics, strict=True)
            ):
                record = {
                    "evaluation": self.evaluation_number,
                    "index": index,
                    "amsr": "".join(tokens),
                    "smiles": (
                        Chem.MolToSmiles(candidate.mol, isomericSmiles=True)
                        if candidate is not None
                        else None
                    ),
                    "accepted": item.valid,
                    "rejection_reason": item.rejection_reason,
                    "scores": item.scores,
                    "gnina_cnn_affinity": (
                        diagnostic.cnn_affinity if diagnostic is not None else None
                    ),
                    "roshambo_tanimoto_combo": (
                        diagnostic.roshambo_tanimoto_combo if diagnostic is not None else None
                    ),
                    "minimized_rmsd": (
                        diagnostic.minimized_rmsd if diagnostic is not None else None
                    ),
                }
                output.write(json.dumps(record, sort_keys=True) + "\n")


def build() -> DockingObjectives:
    """Factory loaded by trl after `mtrl rl` installs its scoring configuration."""
    return DockingObjectives(ScoringConfig.from_env())
