from __future__ import annotations

import sys
from pathlib import Path

from posebusters import PoseBusters
from rdkit import Chem

from mtrl.config import PoseBustersConfig


def run(config: PoseBustersConfig, receptor_pdb: Path, poses_sdf: Path) -> None:
    """Apply receptor-aware PoseBusters checks and annotate every SDF record."""
    posebusters_config = "dock_fast" if config == "dock-fast" else "dock"
    results = PoseBusters(config=posebusters_config).bust(
        mol_pred=str(poses_sdf), mol_cond=str(receptor_pdb)
    )
    poses = list(Chem.SDMolSupplier(str(poses_sdf), removeHs=False))
    writer = Chem.SDWriter(str(poses_sdf))
    try:
        for index, pose in enumerate(poses):
            if pose is None:
                continue
            if results.empty or index >= len(results):
                passed = False
                failed_checks = "no_results"
            else:
                row = results.iloc[index]
                passed = bool(row.notna().all() and row.all())
                failed_checks = (
                    "" if passed else ",".join(row.index[~row.fillna(False).astype(bool)].tolist())
                )
            pose.SetProp("posebusters_passed", str(passed))
            pose.SetProp("posebusters_failed_checks", failed_checks)
            writer.write(pose)
    finally:
        writer.close()


def main() -> None:
    if len(sys.argv) != 4 or sys.argv[1] not in {"dock", "dock-fast"}:
        raise SystemExit(
            "usage: python -m mtrl.posebusters_runner {dock,dock-fast} RECEPTOR.pdb POSES.sdf"
        )
    run(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]))  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
