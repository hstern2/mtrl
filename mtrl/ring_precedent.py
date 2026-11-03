from __future__ import annotations

import json
import subprocess
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Mol


class RingPrecedentFilter:
    """Query LillyMol ring precedent in a separate Python environment."""

    def __init__(self, database: Path, python: Path) -> None:
        self.database = database
        self.python = python

    def accept_batch(self, mols: list[Mol]) -> list[bool]:
        if not mols:
            return []
        smiles = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in mols]
        worker = Path(__file__).with_name("_ring_precedent_worker.py")
        try:
            result = subprocess.run(
                [str(self.python), str(worker), str(self.database)],
                input=json.dumps(smiles),
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"LillyMol ring precedent lookup failed: {error}") from error
        if result.returncode != 0:
            detail = result.stderr.strip() or str(result.returncode)
            raise RuntimeError(f"LillyMol ring precedent lookup failed: {detail}")
        try:
            counts = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError("LillyMol ring precedent lookup returned invalid JSON") from error
        if (
            not isinstance(counts, list)
            or len(counts) != len(mols)
            or any(
                count is not None and (isinstance(count, bool) or not isinstance(count, int))
                for count in counts
            )
        ):
            raise RuntimeError("LillyMol ring precedent lookup returned invalid counts")
        # None means no eligible ring system; acyclic molecules pass.
        return [count != 0 for count in counts]
