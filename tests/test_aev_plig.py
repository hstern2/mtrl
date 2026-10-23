import csv
from pathlib import Path
from unittest.mock import patch

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from mtrl.aev_plig import AEVPligInput, AEVPligScorer


def _mol3d() -> Chem.Mol:
    mol = Chem.MolFromSmiles("C[NH2+]C")
    assert mol is not None
    mol = Chem.AddHs(mol)
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    return Chem.RemoveHs(mol)


def test_aev_plig_scores_a_batch_with_explicit_hydrogens(tmp_path) -> None:
    checkout = tmp_path / "AEV-PLIG"
    (checkout / "data" / "processed").mkdir(parents=True)
    (checkout / "output" / "predictions").mkdir(parents=True)
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("END\n")
    seen = {}

    def fake_run(command, **kwargs):
        dataset = Path(command[command.index("--dataset_csv") + 1])
        data_name = command[command.index("--data_name") + 1]
        rows = list(csv.DictReader(dataset.open()))
        ligands = [
            next(
                mol
                for mol in Chem.SDMolSupplier(row["sdf_file"], removeHs=False)
                if mol is not None
            )
            for row in rows
        ]
        seen["command"] = command
        seen["kwargs"] = kwargs
        seen["hydrogens"] = [mol.GetNumAtoms() - mol.GetNumHeavyAtoms() for mol in ligands]
        predictions = checkout / "output" / "predictions" / f"{data_name}_predictions.csv"
        with predictions.open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(("unique_id", "preds"))
            writer.writerows(((row["unique_id"], 6.5 + index) for index, row in enumerate(rows)))
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    scorer = AEVPligScorer(
        checkout,
        python=Path("/usr/bin/python3"),
        workers=3,
        environment={"CUDA_VISIBLE_DEVICES": "1"},
    )
    inputs = [
        AEVPligInput("first", receptor, _mol3d()),
        AEVPligInput("second", receptor, _mol3d()),
    ]
    with patch("mtrl.aev_plig.subprocess.run", side_effect=fake_run):
        scores = scorer.score(inputs)

    assert scores == {"first": 6.5, "second": 7.5}
    assert seen["hydrogens"] == [8, 8]
    assert seen["kwargs"]["cwd"] == checkout
    assert seen["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert seen["kwargs"]["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert seen["command"][seen["command"].index("--num_workers") + 1] == "3"
    assert seen["command"][seen["command"].index("--device") + 1] == "auto"
    assert not list((checkout / "output" / "predictions").iterdir())


def test_aev_plig_failure_is_not_silently_converted_to_scores(tmp_path) -> None:
    checkout = tmp_path / "AEV-PLIG"
    checkout.mkdir()
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("END\n")
    scorer = AEVPligScorer(checkout, python=Path("/usr/bin/python3"))
    failed = type("Result", (), {"returncode": 2, "stdout": "", "stderr": "bad model"})()

    with patch("mtrl.aev_plig.subprocess.run", return_value=failed):
        with pytest.raises(RuntimeError, match="AEV-PLIG failed.*bad model"):
            scorer.score([AEVPligInput("pose", receptor, _mol3d())])


def test_aev_plig_omits_unsupported_ligand_elements_without_starting_tool(tmp_path) -> None:
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("END\n")
    sodium = Chem.MolFromSmiles("[Na+]")
    assert sodium is not None
    scorer = AEVPligScorer(tmp_path, python=Path("/usr/bin/python3"))

    with patch("mtrl.aev_plig.subprocess.run") as run:
        scores = scorer.score([AEVPligInput("sodium", receptor, sodium)])

    assert scores == {}
    run.assert_not_called()


def test_aev_plig_bounds_upstream_prediction_batch_size(tmp_path) -> None:
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("END\n")
    mol = _mol3d()
    inputs = [AEVPligInput(f"pose_{index}", receptor, mol) for index in range(129)]
    scorer = AEVPligScorer(tmp_path, python=Path("/usr/bin/python3"))

    with patch.object(scorer, "_score_batch", side_effect=lambda batch: {}) as score_batch:
        scorer.score(inputs)

    assert [len(call.args[0]) for call in score_batch.call_args_list] == [128, 1]
