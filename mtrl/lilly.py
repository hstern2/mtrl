from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from rdkit import Chem
from rdkit.Chem import Mol


class LillyMedchemFilter:
    """Batched Lilly Medchem Rules filter using the relaxed rule set."""

    def __init__(self, executable: str = "Lilly_Medchem_Rules.rb") -> None:
        resolved = shutil.which(executable)
        if resolved is None and Path(executable).is_file():
            resolved = str(Path(executable).resolve())
        if resolved is None:
            raise RuntimeError(f"Lilly Medchem Rules executable not found: {executable}")
        self.executable = Path(resolved)

    def accept_batch(self, mols: list[Mol]) -> list[bool]:
        if not mols:
            return []

        names = [f"mtrl_{i}" for i in range(len(mols))]
        lines = []
        for mol, name in zip(mols, names, strict=True):
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            lines.append(f"{smiles} {name}")

        with TemporaryDirectory(prefix="mtrl_lilly_") as temporary:
            work = Path(temporary)
            input_path = work / "candidates.smi"
            rejected_stem = work / "rejected"
            log_stem = work / "stage"
            input_path.write_text("\n".join(lines) + "\n")
            env = os.environ.copy()
            if "LILLYMOL_HOME" not in env:
                inferred_home = self._infer_lillymol_home()
                if inferred_home is not None:
                    env["LILLYMOL_HOME"] = str(inferred_home)
            result = subprocess.run(
                [
                    str(self.executable),
                    "-relaxed",
                    "-noapdm",
                    "-B",
                    str(rejected_stem),
                    "-log",
                    str(log_stem),
                    str(input_path),
                ],
                cwd=temporary,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                message = result.stderr.strip() or "no diagnostic output"
                raise RuntimeError(
                    f"Lilly Medchem Rules failed with exit code {result.returncode}: {message}"
                )

            known_names = set(names)
            accepted = _record_names(result.stdout.splitlines(), known_names)
            rejected = set()
            for path in work.glob(f"{rejected_stem.name}*"):
                rejected.update(_record_names(path.read_text().splitlines(), known_names))
            if accepted & rejected or accepted | rejected != known_names:
                missing = sorted(known_names - accepted - rejected)
                logs = "\n".join(
                    path.read_text(errors="replace") for path in sorted(work.glob("stage*.log"))
                )
                diagnostic = (result.stderr + "\n" + logs).strip()[-1000:]
                raise RuntimeError(
                    "Lilly Medchem Rules did not classify every molecule"
                    f" (missing: {', '.join(missing) or 'none'}): "
                    f"{diagnostic or 'no diagnostic output'}"
                )
            return [name in accepted for name in names]

    def _infer_lillymol_home(self) -> Path | None:
        path = self.executable.resolve()
        if len(path.parents) >= 3 and path.parent.name == "bin":
            candidate = path.parents[2]
            if (candidate / "data" / "queries" / "LillyMedchemRules").is_dir():
                return candidate
        return None


def _record_names(lines: list[str], known_names: set[str]) -> set[str]:
    return {
        field
        for line in lines
        for field in line.split()[1:]
        if field in known_names
    }
