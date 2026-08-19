import io

import pytest
import torch
from rdkit import Chem

from mtrl.generate import (
    default_sampling_batch_size,
    write_conformers,
)
from mtrl.hardware import default_conformer_workers


@pytest.mark.parametrize(
    ("free_gib", "expected"),
    [(12.0, 256), (8.0, 128), (4.0, 64), (2.0, 32)],
)
def test_auto_sampling_batch_uses_free_device_memory(
    free_gib, expected, monkeypatch
) -> None:
    gib = 1024**3
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda device: (int(free_gib * gib), 24 * gib),
    )

    assert default_sampling_batch_size(torch.device("cuda")) == expected


def test_auto_conformer_workers_respects_affinity_and_workload(monkeypatch) -> None:
    monkeypatch.setattr("mtrl.hardware.os.sched_getaffinity", lambda pid: set(range(32)))

    assert default_conformer_workers(1) == 1
    assert default_conformer_workers(20) == 5
    assert default_conformer_workers(1000) == 16


def test_write_conformers_streams_sdf_and_returns_counts(monkeypatch) -> None:
    def fake_detokenize(tokens: list[str]):
        if tokens == ["bad"]:
            return None
        mol = Chem.MolFromSmiles("CCO")
        conformer = Chem.Conformer(mol.GetNumAtoms())
        for atom_index in range(mol.GetNumAtoms()):
            conformer.SetAtomPosition(atom_index, (float(atom_index), 0.0, 0.0))
        mol.AddConformer(conformer)
        return mol

    monkeypatch.setattr("mtrl.generate.detokenize", fake_detokenize)
    output = io.StringIO()
    summary = write_conformers([["C", "C", "O"], ["bad"]], output)

    sdf = output.getvalue()
    assert sdf.count("$$$$") == 1
    assert ">  <AMSR>" in sdf
    assert "CCO" in sdf
    assert summary["decoded_conformers"] == 1
    assert summary["decode_failures"] == 1
