from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

CONFIG_ENV = "MTRL_SCORING_CONFIG"
_TARGET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
TargetAcceptance = Literal["any", "all"]
DockingMode = Literal["flexible", "rigid", "rigid-refine"]
PoseBustersConfig = Literal["dock", "dock-fast"]


@dataclass(frozen=True)
class DockingTarget:
    """One receptor conformation and search box, yielding one RL objective."""

    name: str
    receptor_pdb: Path
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    exhaustiveness: int = 8
    num_modes: int = 9

    def validate(self) -> None:
        if not _TARGET_NAME.fullmatch(self.name):
            raise ValueError(
                f"invalid docking target name {self.name!r}; use letters, numbers, '.', '_', or '-'"
            )
        if not self.receptor_pdb.is_file():
            raise ValueError(f"docking target receptor PDB does not exist: {self.receptor_pdb}")
        if len(self.center) != 3 or not all(math.isfinite(value) for value in self.center):
            raise ValueError(f"docking target {self.name!r} center must contain three numbers")
        if len(self.size) != 3 or any(
            not math.isfinite(value) or value <= 0 for value in self.size
        ):
            raise ValueError(
                f"docking target {self.name!r} size must contain three positive numbers"
            )
        if self.exhaustiveness <= 0:
            raise ValueError(f"docking target {self.name!r} exhaustiveness must be > 0")
        if self.num_modes <= 0:
            raise ValueError(f"docking target {self.name!r} num_modes must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "receptor_pdb": str(self.receptor_pdb),
            "center": list(self.center),
            "size": list(self.size),
            "exhaustiveness": self.exhaustiveness,
            "num_modes": self.num_modes,
        }

    @classmethod
    def from_dict(cls, values: object, *, base_dir: Path | None = None) -> DockingTarget:
        if not isinstance(values, dict):
            raise ValueError("each docking target must be a JSON object")
        required = {"name", "receptor_pdb", "center", "size"}
        missing = sorted(required - values.keys())
        if missing:
            raise ValueError(f"docking target is missing fields: {', '.join(missing)}")

        name = values["name"]
        if not isinstance(name, str):
            raise ValueError("docking target name must be a string")
        receptor_value = values["receptor_pdb"]
        if not isinstance(receptor_value, str) or not receptor_value:
            raise ValueError(f"docking target {name!r} receptor_pdb must be a path string")
        receptor = Path(receptor_value)
        if base_dir is not None and not receptor.is_absolute():
            receptor = base_dir / receptor
        center = _numeric_triplet(values["center"], target_name=name, field="center")
        size = _numeric_triplet(values["size"], target_name=name, field="size")
        exhaustiveness = _positive_int(
            values.get("exhaustiveness", 8), target_name=name, field="exhaustiveness"
        )
        num_modes = _positive_int(values.get("num_modes", 9), target_name=name, field="num_modes")
        target = cls(
            name=name,
            receptor_pdb=receptor.resolve(),
            center=center,
            size=size,
            exhaustiveness=exhaustiveness,
            num_modes=num_modes,
        )
        target.validate()
        return target


def _numeric_triplet(value: object, *, target_name: str, field: str) -> tuple[float, float, float]:
    message = f"docking target {target_name!r} {field} must be an array of three numbers"
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(message)
    if any(isinstance(item, bool) or not isinstance(item, int | float) for item in value):
        raise ValueError(message)
    result = (float(value[0]), float(value[1]), float(value[2]))
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"docking target {target_name!r} {field} values must be finite")
    return result


def _positive_int(value: object, *, target_name: str, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"docking target {target_name!r} {field} must be a positive integer")
    return value


@dataclass(frozen=True)
class ScoringConfig:
    receptor_pdb: Path
    reference_sdf: Path
    output_dir: Path
    lilly_medchem_rules: bool = False
    lilly_rules_executable: str = "Lilly_Medchem_Rules.rb"
    verbose_tools: bool = False
    evaluation_workers: int = 1

    def validate(self) -> None:
        if not self.receptor_pdb.is_file():
            raise ValueError(f"receptor PDB does not exist: {self.receptor_pdb}")
        if not self.reference_sdf.is_file():
            raise ValueError(f"reference SDF does not exist: {self.reference_sdf}")
        if self.evaluation_workers <= 0:
            raise ValueError("evaluation_workers must be > 0")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("receptor_pdb", "reference_sdf", "output_dir"):
            result[key] = str(result[key])
        result["mode"] = "reference_minimize"
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
        mode = values.pop("mode", "reference_minimize")
        if mode != "reference_minimize":
            raise RuntimeError(f"expected reference-minimize scoring configuration, found {mode!r}")
        for key in ("receptor_pdb", "reference_sdf", "output_dir"):
            values[key] = Path(values[key])
        config = cls(**values)
        config.validate()
        return config


@dataclass(frozen=True)
class BoxScoringConfig:
    """Configuration for independent full-docking objectives."""

    targets: tuple[DockingTarget, ...]
    output_dir: Path
    lilly_medchem_rules: bool = False
    lilly_rules_executable: str = "Lilly_Medchem_Rules.rb"
    verbose_tools: bool = False
    evaluation_workers: int = 1
    target_failure_score: float = 0.0
    accept_targets: TargetAcceptance = "any"
    docking_mode: DockingMode = "flexible"
    gnina_timeout_seconds: int = 600
    qed_objective: bool = False
    posebusters_config: PoseBustersConfig = "dock"
    posebusters_timeout_seconds: int = 600

    def validate(self) -> None:
        if not self.targets:
            raise ValueError("at least one docking target is required")
        names = [target.name for target in self.targets]
        if len(set(names)) != len(names):
            raise ValueError("docking target names must be unique")
        for target in self.targets:
            target.validate()
        if self.evaluation_workers <= 0:
            raise ValueError("evaluation_workers must be > 0")
        if not math.isfinite(self.target_failure_score):
            raise ValueError("target_failure_score must be finite")
        if self.accept_targets not in {"any", "all"}:
            raise ValueError("accept_targets must be 'any' or 'all'")
        if self.docking_mode not in {"flexible", "rigid", "rigid-refine"}:
            raise ValueError("docking_mode must be 'flexible', 'rigid', or 'rigid-refine'")
        if self.gnina_timeout_seconds <= 0:
            raise ValueError("gnina_timeout_seconds must be > 0")
        if self.posebusters_config not in {"dock", "dock-fast"}:
            raise ValueError("posebusters_config must be 'dock' or 'dock-fast'")
        if self.posebusters_timeout_seconds <= 0:
            raise ValueError("posebusters_timeout_seconds must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": "box_docking",
            "targets": [target.to_dict() for target in self.targets],
            "output_dir": str(self.output_dir),
            "lilly_medchem_rules": self.lilly_medchem_rules,
            "lilly_rules_executable": self.lilly_rules_executable,
            "verbose_tools": self.verbose_tools,
            "evaluation_workers": self.evaluation_workers,
            "target_failure_score": self.target_failure_score,
            "accept_targets": self.accept_targets,
            "docking_mode": self.docking_mode,
            "gnina_timeout_seconds": self.gnina_timeout_seconds,
            "qed_objective": self.qed_objective,
            "posebusters_config": self.posebusters_config,
            "posebusters_timeout_seconds": self.posebusters_timeout_seconds,
        }

    def install(self) -> None:
        self.validate()
        os.environ[CONFIG_ENV] = json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_env(cls) -> BoxScoringConfig:
        encoded = os.environ.get(CONFIG_ENV)
        if not encoded:
            raise RuntimeError(
                f"{CONFIG_ENV} is not set; launch with `mtrl rl --docking-targets ...`"
            )
        values = json.loads(encoded)
        mode = values.pop("mode", None)
        if mode != "box_docking":
            raise RuntimeError(f"expected box-docking scoring configuration, found {mode!r}")
        values["output_dir"] = Path(values["output_dir"])
        values["targets"] = tuple(DockingTarget.from_dict(target) for target in values["targets"])
        config = cls(**values)
        config.validate()
        return config


def scoring_mode_from_env() -> str:
    encoded = os.environ.get(CONFIG_ENV)
    if not encoded:
        raise RuntimeError(
            f"{CONFIG_ENV} is not set; launch with `mtrl rl` so scoring is configured"
        )
    return str(json.loads(encoded).get("mode", "reference_minimize"))


def load_docking_targets(path: Path) -> tuple[DockingTarget, ...]:
    """Load a reproducible list of receptor/box objectives from JSON."""
    payload = json.loads(path.read_text())
    records = payload.get("targets") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("docking-targets JSON must be a list or contain a 'targets' list")
    targets = tuple(DockingTarget.from_dict(record, base_dir=path.parent) for record in records)
    if not targets:
        raise ValueError("docking-targets JSON contains no targets")
    names = [target.name for target in targets]
    if len(set(names)) != len(names):
        raise ValueError("docking target names must be unique")
    return targets
