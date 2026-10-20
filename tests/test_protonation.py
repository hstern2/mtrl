from __future__ import annotations

from unittest.mock import patch

from rdkit import Chem

from mtrl.protonation import ProtonationFilter


def test_common_solution_states_at_ph_7_are_checked() -> None:
    check = ProtonationFilter()

    assert check.rejection_reason(Chem.MolFromSmiles("CC(=O)[O-]")) is None
    assert check.rejection_reason(Chem.MolFromSmiles("CC[NH3+]")) is None
    assert "emitted state not predicted" in check.rejection_reason(Chem.MolFromSmiles("CC(=O)O"))
    assert "emitted state not predicted" in check.rejection_reason(Chem.MolFromSmiles("CCN"))
    assert check.rejection_reason(Chem.MolFromSmiles("[NH3+]CC(=O)[O-]")) is None
    assert "emitted state not predicted" in check.rejection_reason(Chem.MolFromSmiles("NCC(=O)O"))


def test_many_titratable_sites_exceed_default_limit() -> None:
    reason = ProtonationFilter().rejection_reason(Chem.MolFromSmiles("NCCNCCNCCN"))
    assert reason is not None and "more than 8 predicted states" in reason


def test_variant_limit_accepts_eight_and_rejects_nine() -> None:
    mol = Chem.MolFromSmiles("CCO")
    check = ProtonationFilter()
    with patch("mtrl.protonation.Protonate") as predict:
        predict.return_value.to_list.return_value = ["CCO"] * 8
        predict.return_value.stats.fallback_used = 0
        assert check.rejection_reason(mol) is None
        assert predict.call_args.kwargs == {
            "smiles_input": "CCO",
            "ph_min": 7.0,
            "ph_max": 7.8,
            "max_variants": 9,
        }
    with patch("mtrl.protonation.Protonate") as predict:
        predict.return_value.to_list.return_value = ["CCO"] * 9
        predict.return_value.stats.fallback_used = 0
        assert "more than 8 predicted states" in ProtonationFilter().rejection_reason(mol)


def test_prediction_failure_and_empty_output_reject() -> None:
    mol = Chem.MolFromSmiles("CCO")
    with patch("mtrl.protonation.Protonate", side_effect=RuntimeError("broken")):
        assert ProtonationFilter().rejection_reason(mol) == (
            "Protonation filter failed: prediction error"
        )
    with patch("mtrl.protonation.Protonate") as predict:
        predict.return_value.to_list.return_value = []
        predict.return_value.stats.fallback_used = 0
        assert ProtonationFilter().rejection_reason(mol) == (
            "Protonation filter failed: no states predicted"
        )


def test_library_fallback_rejects_even_when_original_smiles_is_returned() -> None:
    mol = Chem.MolFromSmiles("CCO")
    with patch(
        "dimorphite_dl.protonate.run.Protonate._detect_protonation_sites",
        side_effect=RuntimeError("site detection failed"),
    ):
        assert ProtonationFilter().rejection_reason(mol) == (
            "Protonation filter failed: prediction error"
        )


def test_repeated_structure_is_predicted_once() -> None:
    check = ProtonationFilter()
    mol = Chem.MolFromSmiles("CCO")
    with patch("mtrl.protonation.Protonate") as predict:
        predict.return_value.to_list.return_value = ["CCO"]
        predict.return_value.stats.fallback_used = 0
        assert check.rejection_reason(mol) is None
        assert check.rejection_reason(Chem.AddHs(mol)) is None
        assert predict.call_count == 1
