import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from mtrl import DecodedAMSR
from mtrl.config import ScoringConfig
from mtrl.objectives import DockingObjectives
from mtrl.scoring import StructureScore


class FakePipeline:
    def __init__(self, results: list[StructureScore]) -> None:
        self.results = results
        self.seen = []

    def score_batch(self, mols):
        self.seen = mols
        return self.results[: len(mols)]


class FakeLilly:
    def __init__(self, accepted: list[bool]) -> None:
        self.accepted = accepted
        self.seen = []

    def accept_batch(self, mols):
        self.seen = mols
        return self.accepted[: len(mols)]


def _decoded(smiles: str) -> DecodedAMSR:
    return DecodedAMSR(Chem.MolFromSmiles(smiles), {})


def _config(*, lilly: bool = False) -> ScoringConfig:
    return ScoringConfig(
        receptor_pdb=Path("receptor.pdb"),
        reference_sdf=Path("reference.sdf"),
        work_dir=Path("work"),
        lilly_medchem_rules=lilly,
        record_scores=False,
    )


def test_only_accepted_structures_receive_the_two_pareto_scores() -> None:
    decoded = [_decoded("CCCCCCCC"), _decoded("c1ccccc1"), None]
    pipeline = FakePipeline(
        [
            StructureScore(
                cnn_affinity=7.4,
                roshambo_tanimoto_combo=0.82,
                minimized_rmsd=0.4,
                accepted=True,
            ),
            StructureScore(rejection_reason="PoseBusters failed"),
        ]
    )
    suite = DockingObjectives(
        _config(),
        decode_fn=lambda tokens: decoded[int(tokens[0])],
        conformer_fn=lambda candidate: candidate.mol,
        pipeline=pipeline,
    )

    scored = suite.evaluate([["0"], ["1"], ["2"]])

    assert [objective.name for objective in suite.objectives] == [
        "gnina_cnn_affinity",
        "roshambo_tanimoto_combo",
    ]
    assert all(objective.direction == "maximize" for objective in suite.objectives)
    assert scored[0].valid
    assert scored[0].scores == {
        "gnina_cnn_affinity": 7.4,
        "roshambo_tanimoto_combo": 0.82,
    }
    assert not scored[1].valid
    assert scored[1].rejection_reason == "PoseBusters failed"
    assert not scored[2].valid
    assert scored[2].rejection_reason == "AMSR decode failed"
    rewards = suite.get_rewards(scored)
    assert rewards[0] > 0
    assert np.array_equal(rewards[1:], np.zeros(2))


def test_lilly_runs_before_conformer_construction_and_structure_scoring() -> None:
    decoded = [_decoded("CCCCCCCC"), _decoded("c1ccccc1C")]
    lilly = FakeLilly([False, True])
    built = []
    pipeline = FakePipeline(
        [
            StructureScore(
                cnn_affinity=6.0,
                roshambo_tanimoto_combo=0.5,
                accepted=True,
            )
        ]
    )

    def make_conformer(candidate):
        built.append(candidate)
        return candidate.mol

    suite = DockingObjectives(
        _config(lilly=True),
        decode_fn=lambda tokens: decoded[int(tokens[0])],
        conformer_fn=make_conformer,
        pipeline=pipeline,
        lilly_filter=lilly,
    )
    scored = suite.evaluate([["0"], ["1"]])

    assert len(lilly.seen) == 2
    assert built == [decoded[1]]
    assert pipeline.seen == [decoded[1].mol]
    assert not scored[0].valid
    assert scored[0].rejection_reason == "Lilly Medchem Rules (-relaxed) failed"
    assert scored[1].valid


def test_disconnected_molecules_are_rejected_before_scoring() -> None:
    pipeline = FakePipeline([])
    suite = DockingObjectives(
        _config(),
        decode_fn=lambda tokens: _decoded("CC.CC"),
        conformer_fn=lambda candidate: candidate.mol,
        pipeline=pipeline,
    )

    scored = suite.evaluate([["0"]])

    assert not scored[0].valid
    assert scored[0].rejection_reason == "molecule is disconnected"
    assert pipeline.seen == []


def test_score_audit_records_accepted_scores_and_rejections(tmp_path) -> None:
    decoded = [_decoded("CCCCCCCC"), None]
    config = ScoringConfig(
        receptor_pdb=Path("receptor.pdb"),
        reference_sdf=Path("reference.sdf"),
        work_dir=tmp_path,
        record_scores=True,
    )
    pipeline = FakePipeline(
        [
            StructureScore(
                cnn_affinity=7.1,
                roshambo_tanimoto_combo=0.7,
                minimized_rmsd=0.25,
                accepted=True,
            )
        ]
    )
    suite = DockingObjectives(
        config,
        decode_fn=lambda tokens: decoded[int(tokens[0])],
        conformer_fn=lambda candidate: candidate.mol,
        pipeline=pipeline,
    )

    suite.evaluate([["0"], ["1"]])

    records = [
        json.loads(line) for line in (tmp_path / "rank_0" / "scores.jsonl").read_text().splitlines()
    ]
    assert records[0]["accepted"] is True
    assert records[0]["gnina_cnn_affinity"] == pytest.approx(7.1)
    assert records[0]["minimized_rmsd"] == pytest.approx(0.25)
    assert records[1]["accepted"] is False
    assert records[1]["rejection_reason"] == "AMSR decode failed"
