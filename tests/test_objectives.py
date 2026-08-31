import csv
import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from trl.objectives.base import ScoredItem

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


def _config(output_dir: Path, *, lilly: bool = False) -> ScoringConfig:
    return ScoringConfig(
        receptor_pdb=Path("receptor.pdb"),
        reference_sdf=Path("reference.sdf"),
        output_dir=output_dir,
        lilly_medchem_rules=lilly,
    )


def test_only_accepted_structures_receive_the_two_pareto_scores(tmp_path) -> None:
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
        _config(tmp_path),
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


def test_rewards_are_absolute_across_batches_and_increase_with_both_objectives(
    tmp_path,
) -> None:
    suite = DockingObjectives(
        _config(tmp_path),
        pipeline=FakePipeline([]),
    )
    suite.reference_score = StructureScore(cnn_affinity=8.0)
    weak = ScoredItem(
        token_ids=[],
        scores={"gnina_cnn_affinity": 4.0, "roshambo_tanimoto_combo": 0.25},
    )
    strong = ScoredItem(
        token_ids=[],
        scores={"gnina_cnn_affinity": 8.0, "roshambo_tanimoto_combo": 0.50},
    )
    invalid = ScoredItem(token_ids=[], valid=False)

    # A one-item weak batch must not receive the same reward as a one-item strong
    # batch. That was the failure mode of per-batch Pareto ranking.
    assert suite.get_rewards([weak])[0] == pytest.approx(0.125)
    assert suite.get_rewards([strong])[0] == pytest.approx(0.50)
    assert suite.get_rewards([strong, weak, invalid]).tolist() == pytest.approx(
        [0.50, 0.125, 0.0]
    )


def test_only_new_cumulative_pareto_points_receive_the_bonus(tmp_path) -> None:
    suite = DockingObjectives(
        _config(tmp_path),
        pipeline=FakePipeline([]),
    )
    suite.reference_score = StructureScore(cnn_affinity=8.0)
    suite.pareto_lambda = 0.1
    suite._previous_front = [
        {"cnn_affinity": 6.0, "tanimoto_combo": 0.4},
    ]
    suite._overall_front = [
        {"cnn_affinity": 6.0, "tanimoto_combo": 0.4},
        {"cnn_affinity": 5.0, "tanimoto_combo": 0.6},
    ]
    old_front = ScoredItem(
        token_ids=[],
        scores={"gnina_cnn_affinity": 6.0, "roshambo_tanimoto_combo": 0.4},
    )
    new_front = ScoredItem(
        token_ids=[],
        scores={"gnina_cnn_affinity": 5.0, "roshambo_tanimoto_combo": 0.6},
    )

    rewards = suite.get_rewards([old_front, new_front])
    assert rewards[0] == pytest.approx(0.30)
    assert rewards[1] == pytest.approx(0.475)


def test_lilly_runs_before_conformer_construction_and_structure_scoring(tmp_path) -> None:
    decoded = [_decoded("CCCCCCCC"), _decoded("c1ccccc1C")]
    lilly = FakeLilly([False, True])
    built = []
    pipeline = FakePipeline(
        [
            StructureScore(
                cnn_affinity=6.0,
                roshambo_tanimoto_combo=0.5,
                minimized_rmsd=0.3,
                accepted=True,
            )
        ]
    )

    def make_conformer(candidate):
        built.append(candidate)
        return candidate.mol

    suite = DockingObjectives(
        _config(tmp_path, lilly=True),
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


def test_disconnected_molecules_are_rejected_before_scoring(tmp_path) -> None:
    pipeline = FakePipeline([])
    suite = DockingObjectives(
        _config(tmp_path),
        decode_fn=lambda tokens: _decoded("CC.CC"),
        conformer_fn=lambda candidate: candidate.mol,
        pipeline=pipeline,
    )

    scored = suite.evaluate([["0"]])

    assert not scored[0].valid
    assert scored[0].rejection_reason == "molecule is disconnected"
    assert pipeline.seen == []
    assert not list((tmp_path / "best").glob("*.sdf"))


def test_score_audit_records_accepted_scores_and_rejections(tmp_path) -> None:
    decoded = [_decoded("CCCCCCCC"), None]
    config = ScoringConfig(
        receptor_pdb=Path("receptor.pdb"),
        reference_sdf=Path("reference.sdf"),
        output_dir=tmp_path,
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

    records = [json.loads(line) for line in (tmp_path / "scores.jsonl").read_text().splitlines()]
    assert records[0]["accepted"] is True
    assert records[0]["cnn_affinity"] == pytest.approx(7.1)
    assert records[0]["tanimoto_combo"] == pytest.approx(0.7)
    assert "scores" not in records[0]
    assert records[0]["minimized_rmsd"] == pytest.approx(0.25)
    assert records[1]["accepted"] is False
    assert records[1]["rejection_reason"] == "AMSR decode failed"
    with (tmp_path / "progress.csv").open(newline="") as source:
        progress = next(csv.DictReader(source))
    assert progress["generated"] == "2"
    assert progress["accepted"] == "1"
    assert progress["decode_failed"] == "1"
    assert progress["disconnected_failed"] == "0"
    assert progress["lilly_failed"] == "0"
    assert progress["conformer_failed"] == "0"
    assert progress["posebusters_failed"] == "0"
    assert progress["scoring_failed"] == "0"


def test_generation_and_overall_sdfs_contain_complete_pareto_fronts(tmp_path) -> None:
    smiles = {"0": "CC", "1": "CCC", "2": "CCCC", "3": "CCO", "4": "CCN"}
    pipeline = FakePipeline(
        [
            StructureScore(
                cnn_affinity=8.0,
                roshambo_tanimoto_combo=0.5,
                minimized_rmsd=0.2,
                accepted=True,
            ),
            StructureScore(
                cnn_affinity=7.0,
                roshambo_tanimoto_combo=0.8,
                minimized_rmsd=0.3,
                accepted=True,
            ),
            StructureScore(
                cnn_affinity=6.0,
                roshambo_tanimoto_combo=0.4,
                minimized_rmsd=0.4,
                accepted=True,
            ),
        ]
    )
    suite = DockingObjectives(
        _config(tmp_path),
        decode_fn=lambda tokens: _decoded(smiles[tokens[0]]),
        conformer_fn=lambda candidate: candidate.mol,
        pipeline=pipeline,
    )

    suite.evaluate([["0"], ["1"], ["2"]])

    generation_one = [
        mol
        for mol in Chem.SDMolSupplier(
            str(tmp_path / "best" / "generation_000001.sdf"), removeHs=False
        )
        if mol is not None
    ]
    assert len(generation_one) == 2
    assert {mol.GetProp("AMSR") for mol in generation_one} == {"0", "1"}

    pipeline.results = [
        StructureScore(
            cnn_affinity=8.5,
            roshambo_tanimoto_combo=0.6,
            minimized_rmsd=0.2,
            accepted=True,
        ),
        StructureScore(
            cnn_affinity=6.5,
            roshambo_tanimoto_combo=0.9,
            minimized_rmsd=0.3,
            accepted=True,
        ),
    ]
    suite.evaluate([["3"], ["4"]])

    generation_two = [
        mol
        for mol in Chem.SDMolSupplier(
            str(tmp_path / "best" / "generation_000002.sdf"), removeHs=False
        )
        if mol is not None
    ]
    overall = [
        mol
        for mol in Chem.SDMolSupplier(str(tmp_path / "best" / "overall.sdf"), removeHs=False)
        if mol is not None
    ]
    all_generation_one = [
        mol
        for mol in Chem.SDMolSupplier(
            str(tmp_path / "generations" / "generation_000001.sdf"), removeHs=False
        )
        if mol is not None
    ]
    assert len(generation_two) == 2
    assert len(all_generation_one) == 3
    assert {mol.GetProp("AMSR") for mol in overall} == {"1", "3", "4"}
    for mol in overall:
        assert mol.HasProp("CNNaffinity")
        assert mol.HasProp("tanimoto_combo")
        assert mol.HasProp("minimized_rmsd")
    output_files = {
        path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file()
    }
    assert output_files == {
        "best/generation_000001.sdf",
        "best/generation_000002.sdf",
        "best/overall.sdf",
        "generations/generation_000001.sdf",
        "generations/generation_000002.sdf",
        "progress.csv",
        "pareto_progress.png",
        "progress.png",
            "scores.jsonl",
            "summary.txt",
        }
    progress = (tmp_path / "progress.csv").read_text().splitlines()
    assert len(progress) == 3
    assert "generated" in progress[0]
    assert ",3,3," in progress[1]


def test_multi_gpu_output_is_gathered_before_writing(monkeypatch) -> None:
    local = {"scores": [{"rank": 0}], "accepted": []}
    remote = {"scores": [{"rank": 1}], "accepted": []}

    monkeypatch.setattr("mtrl.objectives.torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("mtrl.objectives.torch.distributed.get_world_size", lambda: 2)

    def gather(output, payload):
        output[:] = [payload, remote]

    monkeypatch.setattr("mtrl.objectives.torch.distributed.all_gather_object", gather)

    assert DockingObjectives._gather_output(local) == [local, remote]
