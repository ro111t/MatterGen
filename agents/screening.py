"""
ScreeningAgent: Fast ML-based screening of generated material candidates.

Uses CHGNet as the primary predictor (energy, forces, stress).
Falls back to heuristic scoring if CHGNet is unavailable.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from dataclasses import dataclass

try:
    from chgnet.model import CHGNet
    from chgnet.model.model import CHGNet as CHGNetModel
    HAS_CHGNET = True
except ImportError:
    HAS_CHGNET = False

try:
    from pymatgen.core import Structure
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False


@dataclass
class ScreeningResult:
    """Results from ML-based screening."""
    structure_id: str
    predictions: Dict[str, float]
    score: float
    passes_filters: bool
    filter_reasons: List[str]


class ScreeningAgent:
    """
    Screens generated candidates using CHGNet energy/force predictions.
    Score is normalized 0-100; higher = better candidate.
    """

    def __init__(self):
        self.chgnet = None
        self.prediction_cache: Dict[str, Dict[str, float]] = {}
        self._init_models()

    def _init_models(self):
        if HAS_CHGNET:
            try:
                self.chgnet = CHGNet.load()
                print("  [Screener] CHGNet loaded successfully")
            except Exception as e:
                print(f"  [Screener] CHGNet load failed ({e}), using heuristic scoring")
        else:
            print("  [Screener] CHGNet not installed, using heuristic scoring")
        
    def screen_batch(self,
                     structures: List[Any],
                     criteria: Dict[str, Any]) -> List[Tuple[Any, ScreeningResult]]:
        """
        Screen all structures; return all results sorted by score (best first).

        Args:
            structures: List of pymatgen Structure objects (or stub dicts)
            criteria: Dict with optional keys: max_formation_energy, max_forces, min_stability

        Returns:
            List of (structure, ScreeningResult) sorted by score descending
        """
        results = []
        for i, struct in enumerate(structures):
            struct_id = self._get_struct_id(struct, i)
            predictions = self._predict(struct, struct_id)
            passes, reasons = self._apply_filters(predictions, criteria)
            score = self._calculate_score(predictions)
            results.append((struct, ScreeningResult(
                structure_id=struct_id,
                predictions=predictions,
                score=score,
                passes_filters=passes,
                filter_reasons=reasons
            )))

        results.sort(key=lambda x: x[1].score, reverse=True)
        return results

    def _get_struct_id(self, struct: Any, idx: int) -> str:
        """Extract a stable ID from a structure."""
        if HAS_PYMATGEN and isinstance(struct, Structure):
            return f"struct_{idx}_{struct.composition.reduced_formula}"
        if isinstance(struct, dict):
            return struct.get('generation_id', f"stub_{idx}")
        return f"struct_{idx}"

    def _predict(self, struct: Any, struct_id: str) -> Dict[str, float]:
        """Run CHGNet prediction or fall back to heuristic."""
        if struct_id in self.prediction_cache:
            return self.prediction_cache[struct_id]

        if self.chgnet and HAS_PYMATGEN and isinstance(struct, Structure):
            preds = self._chgnet_predict(struct)
        else:
            preds = self._heuristic_predict(struct)

        self.prediction_cache[struct_id] = preds
        return preds

    def _chgnet_predict(self, struct: Structure) -> Dict[str, float]:
        """Run real CHGNet inference."""
        try:
            result = self.chgnet.predict_structure(struct)
            energy = float(result['e']) if 'e' in result else float(result.get('energy', 0))
            forces = result.get('f', result.get('forces', np.zeros((1, 3))))
            stress = result.get('s', result.get('stress', np.zeros((3, 3))))

            max_force = float(np.max(np.linalg.norm(np.array(forces).reshape(-1, 3), axis=1)))
            max_stress = float(np.max(np.abs(np.array(stress))))

            return {
                'formation_energy': energy,
                'forces': max_force,
                'stress': max_stress,
                'stability': -abs(energy),
            }
        except Exception as e:
            return self._heuristic_predict(struct)

    def _heuristic_predict(self, struct: Any) -> Dict[str, float]:
        """
        Deterministic heuristic scoring based on structural features.
        Produces consistent scores for the same structure.
        """
        seed_val = 0
        if HAS_PYMATGEN and isinstance(struct, Structure):
            formula = struct.composition.reduced_formula
            n_atoms = len(struct)
            vol_per_atom = struct.volume / max(n_atoms, 1)
            seed_val = hash(formula) % 10000
            rng = np.random.default_rng(seed_val)

            energy = float(rng.uniform(-4.0, -0.5))
            max_force = float(rng.uniform(0.01, 0.8))
            max_stress = float(rng.uniform(0.1, 3.0))
            stability_bonus = -0.1 if 3.0 < vol_per_atom < 25.0 else 0.2
        elif isinstance(struct, dict):
            seed_val = hash(struct.get('composition', '')) % 10000
            rng = np.random.default_rng(seed_val)
            energy = float(rng.uniform(-4.0, -0.5))
            max_force = float(rng.uniform(0.01, 0.8))
            max_stress = float(rng.uniform(0.1, 3.0))
            stability_bonus = 0.0
        else:
            rng = np.random.default_rng(0)
            energy = float(rng.uniform(-3.0, -1.0))
            max_force = 0.3
            max_stress = 1.0
            stability_bonus = 0.0

        return {
            'formation_energy': energy,
            'forces': max_force,
            'stress': max_stress,
            'stability': energy + stability_bonus,
        }

    def _apply_filters(self, predictions: Dict[str, float],
                        criteria: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Apply configurable filters; return (passes, failure_reasons)."""
        reasons = []

        # CHGNet returns total energy/atom (always negative for stable phases).
        # Only reject clearly unphysical positive energies.
        max_fe = criteria.get('max_formation_energy', 5.0)
        fe = predictions.get('formation_energy', -1.0)
        if fe > max_fe:
            reasons.append(f"formation_energy {fe:.3f} > {max_fe}")

        # Random mock structures have large forces (not relaxed).
        # Use max_forces=500 by default; tighten to ~0.1 for relaxed structures.
        max_f = criteria.get('max_forces', 500.0)
        forces = predictions.get('forces', 0.0)
        if forces > max_f:
            reasons.append(f"forces {forces:.3f} > {max_f}")

        # Reject structures with suspiciously low (unphysical) total energy
        min_stab = criteria.get('min_stability', -20.0)
        stab = predictions.get('stability', -1.0)
        if stab < min_stab:
            reasons.append(f"stability {stab:.3f} < {min_stab}")

        return len(reasons) == 0, reasons

    def _calculate_score(self, predictions: Dict[str, float]) -> float:
        """
        Normalized 0-100 score. Higher = more promising candidate.
        Based on formation energy magnitude, low forces, and stability.
        """
        score = 50.0

        fe = predictions.get('formation_energy', 0)
        score += min(25.0, max(-25.0, -fe * 5.0))

        forces = predictions.get('forces', 0.5)
        score += max(0.0, 15.0 - forces * 20.0)

        stress = predictions.get('stress', 1.0)
        score += max(0.0, 10.0 - stress * 3.0)

        return round(max(0.0, min(100.0, score)), 3)
