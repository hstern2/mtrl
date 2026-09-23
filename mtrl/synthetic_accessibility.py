from __future__ import annotations

import gzip
import math
import pickle
from functools import cache
from importlib import resources
from typing import cast

from rdkit import Chem
from rdkit.Chem import Mol, rdFingerprintGenerator, rdMolDescriptors

_FRAGMENT_PENALTY = -6.0
_COMPLEXITY_BUFFER = 1.0
_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, includeChirality=True)


@cache
def _fragment_scores() -> dict[int, float]:
    resource = resources.files("mtrl").joinpath("data/BRScores_uspto_emolecules.pkl.gz")
    with resource.open("rb") as source, gzip.GzipFile(fileobj=source) as compressed:
        scores = pickle.load(compressed)
    if not isinstance(scores, dict):
        raise RuntimeError("invalid packaged BR-SAScore fragment table")
    return cast(dict[int, float], scores)


def br_sascore(mol: Mol) -> float:
    """Calculate BR-SAScore using the published USPTO/eMolecules fragment table."""
    molecule = Chem.RemoveHs(mol)
    bit_output = rdFingerprintGenerator.AdditionalOutput()
    bit_output.AllocateBitInfoMap()
    fingerprint = _MORGAN_GENERATOR.GetSparseCountFingerprint(
        molecule, additionalOutput=bit_output
    )
    bit_information = bit_output.GetBitInfoMap()
    fragment_scores = _fragment_scores()
    rare_scores = []
    for bit_id, environments in bit_information.items():
        if environments[0][1] != 2:
            continue
        score = fragment_scores.get(bit_id, _FRAGMENT_PENALTY)
        if score < 0:
            rare_scores.append(score)
    fragment_score = sum(rare_scores) / len(rare_scores) if rare_scores else 0.0

    atom_count = molecule.GetNumAtoms()
    rings = molecule.GetRingInfo().AtomRings()
    ring_memberships = [0] * atom_count
    for ring in rings:
        for atom_index in ring:
            ring_memberships[atom_index] += 1
    multicycle_atoms = sum(count - 1 for count in ring_memberships if count > 1)
    complexity_penalty = (
        atom_count**1.005
        - atom_count
        + math.log10(len(Chem.FindMolChiralCenters(molecule, includeUnassigned=True)) + 1)
        + math.log10(rdMolDescriptors.CalcNumSpiroAtoms(molecule) + 1)
        + math.log10(rdMolDescriptors.CalcNumBridgeheadAtoms(molecule) + 1)
        + (math.log10(2) if any(len(ring) > 6 for ring in rings) else 0.0)
        + math.log10(multicycle_atoms + 1)
    )
    fingerprint_size = len(fingerprint.GetNonzeroElements())
    density_correction = (
        0.5 * math.log(atom_count / fingerprint_size)
        if fingerprint_size and atom_count > fingerprint_size
        else 0.0
    )

    raw_score = fragment_score - complexity_penalty + density_correction
    minimum = _FRAGMENT_PENALTY - _COMPLEXITY_BUFFER
    normalized = max(0.0, min(1.0, (raw_score - minimum) / -minimum))
    return 10.0 - 9.0 * normalized


def br_sascore_rejection_reason(mol: Mol, maximum: float) -> str | None:
    score = br_sascore(mol)
    if score <= maximum:
        return None
    return f"BR-SAScore failed: {score:.3f} > {maximum:g}"
