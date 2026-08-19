from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CONFIG_ENV = "MTRL_SCORING_CONFIG"


@dataclass(frozen=True)
class ScoringConfig:
    receptor_pdb: Path
    reference_sdf: Path
    work_dir: Path
    max_minimized_rmsd: float = 1.0
    lilly_medchem_rules: bool = False
    lilly_rules_executable: str = "Lilly_Medchem_Rules.rb"
    keep_poses: bool = False
    record_scores: bool = True
    verbose_tools: bool = False

    def validate(self) -> None:
        if not self.receptor_pdb.is_file():
            raise ValueError(f"receptor PDB does not exist: {self.receptor_pdb}")
        if not self.reference_sdf.is_file():
            raise ValueError(f"reference SDF does not exist: {self.reference_sdf}")
        if self.max_minimized_rmsd <= 0:
            raise ValueError("max_minimized_rmsd must be > 0")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("receptor_pdb", "reference_sdf", "work_dir"):
            result[key] = str(result[key])
        return result

    def install(self) -> None:
        self.validate()
        os.environ[CONFIG_ENV] = json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_env(cls) -> ScoringConfig:
        encoded = os.environ.get(CONFIG_ENV)
        if not encoded:
            raise RuntimeError(
                f"{CONFIG_ENV} is not set; launch with `mtrl rl` so the "
                "structure-scoring inputs are configured"
            )
        values = json.loads(encoded)
        for key in ("receptor_pdb", "reference_sdf", "work_dir"):
            values[key] = Path(values[key])
        config = cls(**values)
        config.validate()
        return config
