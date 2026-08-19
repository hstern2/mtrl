from types import SimpleNamespace
from unittest.mock import patch

import pytest
from rdkit import Chem

from mtrl.lilly import LillyMedchemFilter


def test_lilly_filter_always_uses_relaxed_rules() -> None:
    mols = [Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"), Chem.MolFromSmiles("CCCCCCCC")]
    completed = SimpleNamespace(
        returncode=0,
        stdout="CC(=O)Oc1ccccc1C(=O)O mtrl_0\n",
        stderr="",
    )

    with (
        patch("mtrl.lilly.shutil.which", return_value="/tools/Lilly_Medchem_Rules.rb"),
        patch("mtrl.lilly.subprocess.run", return_value=completed) as run,
    ):
        accepted = LillyMedchemFilter().accept_batch(mols)

    assert accepted == [True, False]
    command = run.call_args.args[0]
    assert command[1:4] == ["-relaxed", "-noapdm", "-nobadfiles"]


def test_lilly_filter_propagates_tool_failure() -> None:
    completed = SimpleNamespace(returncode=2, stdout="", stderr="failed")
    with (
        patch("mtrl.lilly.shutil.which", return_value="/tools/rules"),
        patch("mtrl.lilly.subprocess.run", return_value=completed),
    ):
        with pytest.raises(RuntimeError, match="exit code 2"):
            LillyMedchemFilter().accept_batch([Chem.MolFromSmiles("CCCCCCCC")])
