from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

CONFIG_ENV = "MTRL_SCORING_CONFIG"
_TARGET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
TargetAcceptance = Literal["any", "all"]
PoseBustersConfig = Literal["dock", "dock-fast"]
UniDockSearchMode = Literal["fast", "balance", "detail"]


@dataclass(frozen=True)
class DockingTarget:
    """One receptor conformation and search box, yielding one RL objective."""

    name: str
    receptor_pdb: Path
    center: tuple[float, float, float]
    size: tuple[float, float, float]
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
        if self.num_modes <= 0:
            raise ValueError(f"docking target {self.name!r} num_modes must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "receptor_pdb": str(self.receptor_pdb),
            "center": list(self.center),
            "size": list(self.size),
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
        num_modes = _positive_int(values.get("num_modes", 9), target_name=name, field="num_modes")
        target = cls(
            name=name,
            receptor_pdb=receptor.resolve(),
            center=center,
            size=size,
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


def _validate_max_br_sascore(value: float | None) -> None:
    if value is not None and (not math.isfinite(value) or not 1.0 <= value <= 10.0):
        raise ValueError("max_br_sascore must be between 1 and 10")


def _validate_max_clogp(value: float | None) -> None:
    if value is not None and not math.isfinite(value):
        raise ValueError("max_clogp must be finite")


def _validate_max_protonation_variants(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value < 10000:
        raise ValueError("max_protonation_variants must be between 1 and 9999")


@dataclass(frozen=True)
class ScoringConfig:
    receptor_pdb: Path
    reference_sdf: Path
    output_dir: Path
    rdkit_druglike_filter: bool = False
    max_br_sascore: float | None = None
    lilly_medchem_rules: bool = False
    lilly_rules_executable: str = "Lilly_Medchem_Rules.rb"
    verbose_tools: bool = False
    evaluation_workers: int = 1
    muegge_filter: bool = False
    brenk_filter: bool = False
    pfizer_3_75_filter: bool = False
    max_clogp: float | None = None
    protonation_filter: bool = True
    max_protonation_variants: int = 8

    def validate(self) -> None:
        if not self.receptor_pdb.is_file():
            raise ValueError(f"receptor PDB does not exist: {self.receptor_pdb}")
        if not self.reference_sdf.is_file():
            raise ValueError(f"reference SDF does not exist: {self.reference_sdf}")
        if self.evaluation_workers <= 0:
            raise ValueError("evaluation_workers must be > 0")
        _validate_max_br_sascore(self.max_br_sascore)
        _validate_max_clogp(self.max_clogp)
        _validate_max_protonation_variants(self.max_protonation_variants)

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
    """Configuration for independent rigid-placement docking objectives."""

    targets: tuple[DockingTarget, ...]
    output_dir: Path
    rdkit_druglike_filter: bool = False
    max_br_sascore: float | None = None
    lilly_medchem_rules: bool = False
    lilly_rules_executable: str = "Lilly_Medchem_Rules.rb"
    verbose_tools: bool = False
    evaluation_workers: int = 1
    target_failure_score: float = 0.0
    accept_targets: TargetAcceptance = "any"
    unidock_executable: str = "unidock"
    unidock_search_mode: UniDockSearchMode = "fast"
    unidock_seed: int = 42
    unidock_timeout_seconds: int = 600
    gnina_cpu: int | None = None
    gnina_timeout_seconds: int = 600
    posebusters_config: PoseBustersConfig = "dock"
    posebusters_timeout_seconds: int = 600
    muegge_filter: bool = False
    brenk_filter: bool = False
    pfizer_3_75_filter: bool = False
    max_clogp: float | None = None
    protonation_filter: bool = True
    max_protonation_variants: int = 8
    aev_plig_dir: Path | None = None
    aev_plig_python: Path | None = None
    aev_plig_model: str = "model_GATv2Net_ligsim90_fep_benchmark"
    aev_plig_workers: int = 4
    aev_plig_timeout_seconds: int = 600

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
        _validate_max_br_sascore(self.max_br_sascore)
        _validate_max_clogp(self.max_clogp)
        _validate_max_protonation_variants(self.max_protonation_variants)
        if not math.isfinite(self.target_failure_score):
            raise ValueError("target_failure_score must be finite")
        if self.accept_targets not in {"any", "all"}:
            raise ValueError("accept_targets must be 'any' or 'all'")
        if not self.unidock_executable:
            raise ValueError("unidock_executable must not be empty")
        if self.unidock_search_mode not in {"fast", "balance", "detail"}:
            raise ValueError("unidock_search_mode must be 'fast', 'balance', or 'detail'")
        if self.unidock_seed < 0:
            raise ValueError("unidock_seed must be >= 0")
        if self.unidock_timeout_seconds <= 0:
            raise ValueError("unidock_timeout_seconds must be > 0")
        if self.gnina_cpu is not None and self.gnina_cpu <= 0:
            raise ValueError("gnina_cpu must be > 0")
        if self.gnina_timeout_seconds <= 0:
            raise ValueError("gnina_timeout_seconds must be > 0")
        if self.posebusters_config not in {"dock", "dock-fast"}:
            raise ValueError("posebusters_config must be 'dock' or 'dock-fast'")
        if self.posebusters_timeout_seconds <= 0:
            raise ValueError("posebusters_timeout_seconds must be > 0")
        if self.aev_plig_dir is None:
            if self.aev_plig_python is not None:
                raise ValueError("aev_plig_python requires aev_plig_dir")
            return
        if not self.aev_plig_dir.is_dir():
            raise ValueError(f"AEV-PLIG directory does not exist: {self.aev_plig_dir}")
        script = self.aev_plig_dir / "process_and_predict.py"
        if not script.is_file():
            raise ValueError(f"AEV-PLIG predictor does not exist: {script}")
        python = self.aev_plig_python or Path(sys.executable)
        if not python.is_file():
            raise ValueError(f"AEV-PLIG Python executable does not exist: {python}")
        if not self.aev_plig_model:
            raise ValueError("aev_plig_model must not be empty")
        model_dir = self.aev_plig_dir / "output" / "trained_models"
        model_files = [
            model_dir / f"{self.aev_plig_model}.pickle",
            *(model_dir / f"{self.aev_plig_model}_{index}.model" for index in range(10)),
        ]
        missing = [str(path) for path in model_files if not path.is_file()]
        if missing:
            raise ValueError(f"AEV-PLIG model is incomplete; missing: {', '.join(missing)}")
        if self.aev_plig_workers <= 0:
            raise ValueError("aev_plig_workers must be > 0")
        if self.aev_plig_timeout_seconds <= 0:
            raise ValueError("aev_plig_timeout_seconds must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": "box_docking",
            "targets": [target.to_dict() for target in self.targets],
            "output_dir": str(self.output_dir),
            "rdkit_druglike_filter": self.rdkit_druglike_filter,
            "muegge_filter": self.muegge_filter,
            "brenk_filter": self.brenk_filter,
            "pfizer_3_75_filter": self.pfizer_3_75_filter,
            "max_br_sascore": self.max_br_sascore,
            "max_clogp": self.max_clogp,
            "protonation_filter": self.protonation_filter,
            "max_protonation_variants": self.max_protonation_variants,
            "lilly_medchem_rules": self.lilly_medchem_rules,
            "lilly_rules_executable": self.lilly_rules_executable,
            "verbose_tools": self.verbose_tools,
            "evaluation_workers": self.evaluation_workers,
            "target_failure_score": self.target_failure_score,
            "accept_targets": self.accept_targets,
            "unidock_executable": self.unidock_executable,
            "unidock_search_mode": self.unidock_search_mode,
            "unidock_seed": self.unidock_seed,
            "unidock_timeout_seconds": self.unidock_timeout_seconds,
            "gnina_cpu": self.gnina_cpu,
            "gnina_timeout_seconds": self.gnina_timeout_seconds,
            "posebusters_config": self.posebusters_config,
            "posebusters_timeout_seconds": self.posebusters_timeout_seconds,
            "aev_plig_dir": str(self.aev_plig_dir) if self.aev_plig_dir is not None else None,
            "aev_plig_python": (
                str(self.aev_plig_python) if self.aev_plig_python is not None else None
            ),
            "aev_plig_model": self.aev_plig_model,
            "aev_plig_workers": self.aev_plig_workers,
            "aev_plig_timeout_seconds": self.aev_plig_timeout_seconds,
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
        for key in ("aev_plig_dir", "aev_plig_python"):
            if values.get(key) is not None:
                values[key] = Path(values[key])
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
