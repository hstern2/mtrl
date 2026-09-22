from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from mtrl.posebusters_runner import run


def _write_poses(path: Path) -> None:
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.write(mol)
    writer.close()


@pytest.mark.parametrize(
    ("config", "expected_posebusters_config"),
    (("dock", "dock"), ("dock-fast", "dock_fast")),
)
def test_runner_selects_config_and_annotates_poses(
    tmp_path, config, expected_posebusters_config
) -> None:
    receptor = tmp_path / "receptor.pdb"
    poses = tmp_path / "poses.sdf"
    receptor.write_text("END\n")
    _write_poses(poses)
    results = pd.DataFrame(
        {
            "sanitization": [True, False],
            "minimum_distance_to_protein": [True, True],
        }
    )

    with patch("mtrl.posebusters_runner.PoseBusters") as posebusters:
        posebusters.return_value.bust.return_value = results
        run(config, receptor, poses)

    posebusters.assert_called_once_with(config=expected_posebusters_config)
    posebusters.return_value.bust.assert_called_once_with(
        mol_pred=str(poses), mol_cond=str(receptor)
    )
    annotated = [mol for mol in Chem.SDMolSupplier(str(poses), removeHs=False) if mol]
    assert annotated[0].GetProp("posebusters_passed") == "True"
    assert annotated[0].GetProp("posebusters_failed_checks") == ""
    assert annotated[1].GetProp("posebusters_passed") == "False"
    assert annotated[1].GetProp("posebusters_failed_checks") == "sanitization"
