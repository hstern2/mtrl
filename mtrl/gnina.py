from __future__ import annotations

import shutil
from pathlib import Path


def require() -> None:
    """Require the external GNINA executable."""
    if shutil.which("gnina") is None:
        raise RuntimeError(
            "'gnina' not found in PATH. Install it from https://github.com/gnina/gnina"
        )


def cmd(
    receptor_pdb: Path,
    ligand_sdf: Path,
    autobox_ligand: Path,
    *,
    minimize: bool = False,
) -> list[str]:
    """Build the legacy reference-ligand GNINA command."""
    command = [
        "gnina",
        "-r",
        str(receptor_pdb),
        "-l",
        str(ligand_sdf),
        "--autobox_ligand",
        str(autobox_ligand),
        "--autobox_extend",
        "1",
        "--num_modes",
        "1",
        "-o",
        str(ligand_sdf.with_suffix(".gnina.sdf")),
    ]
    if minimize:
        command.insert(1, "--minimize")
    return command
