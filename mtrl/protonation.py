from __future__ import annotations

from functools import lru_cache

from dimorphite_dl.protonate.run import Protonate
from rdkit import Chem
from rdkit.Chem import Mol

PH_MIN = 7.0
PH_MAX = 7.8


class ProtonationFilter:
    """Check the emitted state against bounded solution-state predictions."""

    def __init__(self, max_variants: int = 8) -> None:
        if isinstance(max_variants, bool) or not 1 <= max_variants < 10000:
            raise ValueError("max_protonation_variants must be between 1 and 9999")
        self.max_variants = max_variants

    def rejection_reason(self, mol: Mol) -> str | None:
        try:
            smiles = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
            return self._rejection_for_smiles(smiles)
        except Exception:
            return "Protonation filter failed: prediction error"

    @lru_cache(maxsize=4096)
    def _rejection_for_smiles(self, smiles: str) -> str | None:
        try:
            # Request one extra state so exactly max_variants remains acceptable.
            protonator = Protonate(
                smiles_input=smiles,
                ph_min=PH_MIN,
                ph_max=PH_MAX,
                max_variants=self.max_variants + 1,
            )
            variants = protonator.to_list()
            if protonator.stats.fallback_used:
                return "Protonation filter failed: prediction error"
            if len(variants) > self.max_variants:
                return (
                    "Protonation filter failed: more than "
                    f"{self.max_variants} predicted states at pH {PH_MIN:g}-{PH_MAX:g}"
                )
            if not variants:
                return "Protonation filter failed: no states predicted"
            predicted = set()
            for variant in variants:
                predicted_mol = Chem.MolFromSmiles(variant)
                if predicted_mol is None:
                    return "Protonation filter failed: invalid predicted state"
                predicted.add(Chem.MolToSmiles(predicted_mol, isomericSmiles=True))
            if smiles not in predicted:
                return (
                    "Protonation filter failed: emitted state not predicted "
                    f"at pH {PH_MIN:g}-{PH_MAX:g}"
                )
            return None
        except Exception:
            return "Protonation filter failed: prediction error"
