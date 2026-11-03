"""Read one JSON batch of SMILES and return ChEMBL ring precedent counts."""

from __future__ import annotations

import json
import sys

import lillymol
from lillymol_bdb import RingDatabase


def main() -> None:
    database = sys.argv[1]
    smiles = json.load(sys.stdin)
    if not isinstance(smiles, list) or not all(isinstance(item, str) for item in smiles):
        raise ValueError("expected a JSON list of SMILES")
    counts = []
    with RingDatabase(
        database,
        standardize=True,
        bare_ring_fallback=True,
        max_ring_size=8,
        max_rings_per_system=10,
        atom_typing="UST:ACHRY",
    ) as rings:
        for smi in smiles:
            mol = lillymol.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"LillyMol cannot parse SMILES: {smi}")
            counts.append(rings.rarest_ring_count(mol))
    json.dump(counts, sys.stdout)


if __name__ == "__main__":
    main()
