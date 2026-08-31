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
    output_dir: Path
    lilly_medchem_rules: bool = False
    lilly_rules_executable: str = "Lilly_Medchem_Rules.rb"
    verbose_tools: bool = False
    evaluation_workers: int = 1
    initial_generation: int = 1

    def validate(self) -> None:
        if not self.receptor_pdb.is_file():
            raise ValueError(f"receptor PDB does not exist: {self.receptor_pdb}")
        if not self.reference_sdf.is_file():
            raise ValueError(f"reference SDF does not exist: {self.reference_sdf}")
        if self.evaluation_workers <= 0:
            raise ValueError("evaluation_workers must be > 0")
        if self.initial_generation <= 0:
            raise ValueError("initial_generation must be > 0")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("receptor_pdb", "reference_sdf", "output_dir"):
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
        for key in ("receptor_pdb", "reference_sdf", "output_dir"):
            values[key] = Path(values[key])
        config = cls(**values)
        config.validate()
        return config
