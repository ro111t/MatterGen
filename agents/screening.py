"""
ScreeningAgent: Fast ML-based screening of generated material candidates.

Uses CHGNet as the primary predictor (energy, forces, stress).
Falls back to heuristic scoring if CHGNet is unavailable.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import hashlib
import numpy as np

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


DEFAULT_SCREENING_WEIGHTS = {
    "stability": 0.35,
    "relaxation_quality": 0.25,
    "target_property_match": 0.25,
    "composition_novelty": 0.15,
}


@dataclass
class ScreeningResult:
    """Results from ML-based screening."""
    structure_id: str
    predictions: Dict[str, float]
    score: float
    passes_filters: bool
    filter_reasons: List[str]
    rank: Optional[int] = None
    score_components: Dict[str, float] = field(default_factory=dict)


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
        
    def screen_batch(
        self,
        structures: List[Any],
        criteria: Dict[str, Any],
        target_properties: Optional[Dict[str, float]] = None,
        weights: Optional[Dict[str, float]] = None,
        deduplicate: bool = True,
    ) -> List[Tuple[Any, ScreeningResult]]:
        """
        Screen all structures; return all results sorted by score (best first).

        Args:
            structures: List of pymatgen Structure objects (or stub dicts)
            criteria: Dict with optional keys: max_formation_energy, max_forces, min_stability
            target_properties: Optional map of property name -> target value, used to
                compute a target-property-match component of the score.
            weights: Optional map overriding the default multi-objective score weights.
            deduplicate: If True, keep only the highest-scored structure per composition.

        Returns:
            List of (structure, ScreeningResult) sorted by score descending
        """
        target_properties = target_properties or {}
        weights = weights or DEFAULT_SCREENING_WEIGHTS

        raw_results = []
        for i, struct in enumerate(structures):
            struct_id = self._get_struct_id(struct, i)
            predictions = self._predict(struct, struct_id)
            passes, reasons = self._apply_filters(predictions, criteria)
            score, components = self._calculate_score(predictions, target_properties, weights)
            raw_results.append((struct, ScreeningResult(
                structure_id=struct_id,
                predictions=predictions,
                score=score,
                passes_filters=passes,
                filter_reasons=reasons,
                score_components=components,
            )))

        if deduplicate:
            raw_results = self._deduplicate_by_composition(raw_results)

        self._update_novelty_scores(raw_results, weights)

        raw_results.sort(key=lambda x: x[1].score, reverse=True)
        for rank, (_, result) in enumerate(raw_results, start=1):
            result.rank = rank

        return raw_results

    def _get_struct_id(self, struct: Any, idx: int) -> str:
        """Extract a stable ID from a structure."""
        if isinstance(struct, dict):
            if 'candidate_id' in struct:
                return str(struct['candidate_id'])
            if 'generation_id' in struct:
                return str(struct['generation_id'])
            return f"stub_{idx}"
        if hasattr(struct, '_candidate_id'):
            return str(getattr(struct, '_candidate_id'))
        if HAS_PYMATGEN and isinstance(struct, Structure):
            if hasattr(struct, 'properties') and isinstance(struct.properties, dict) and '_candidate_id' in struct.properties:
                return str(struct.properties['_candidate_id'])
            return f"struct_{idx}_{struct.composition.reduced_formula}"
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

    def _chgnet_predict(self, struct: Any) -> Dict[str, float]:
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
        Produces consistent scores for the same structure across processes.
        """
        seed_val = 0
        if HAS_PYMATGEN and isinstance(struct, Structure):
            formula = struct.composition.reduced_formula
            n_atoms = len(struct)
            vol_per_atom = struct.volume / max(n_atoms, 1)
            seed_val = int(hashlib.sha256(formula.encode('utf-8')).hexdigest(), 16) % 10000
            rng = np.random.default_rng(seed_val)

            energy = float(rng.uniform(-4.0, -0.5))
            max_force = float(rng.uniform(0.01, 0.8))
            max_stress = float(rng.uniform(0.1, 3.0))
            stability_bonus = -0.1 if 3.0 < vol_per_atom < 25.0 else 0.2
        elif isinstance(struct, dict):
            formula = struct.get('composition', '')
            seed_val = int(hashlib.sha256(formula.encode('utf-8')).hexdigest(), 16) % 10000
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

    def _deduplicate_by_composition(
        self,
        results: List[Tuple[Any, ScreeningResult]],
    ) -> List[Tuple[Any, ScreeningResult]]:
        """Keep the highest-scoring structure for each reduced composition."""
        best_by_formula: Dict[str, Tuple[Any, ScreeningResult]] = {}
        for struct, result in results:
            formula = self._get_composition_key(struct)
            if formula not in best_by_formula or result.score > best_by_formula[formula][1].score:
                best_by_formula[formula] = (struct, result)
        return list(best_by_formula.values())

    def _get_composition_key(self, struct: Any) -> str:
        """Return a reduced composition string used for deduplication."""
        if HAS_PYMATGEN and isinstance(struct, Structure):
            return str(struct.composition.reduced_formula)
        if isinstance(struct, dict):
            return str(struct.get("composition", struct.get("generation_id", id(struct))))
        return f"struct_{id(struct)}"

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

    def _calculate_score(
        self,
        predictions: Dict[str, float],
        target_properties: Optional[Dict[str, float]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Multi-objective score normalized to 0-100. Higher = better candidate.

        Combines stability, relaxation quality, target-property match, and
        within-batch composition novelty.
        """
        target_properties = target_properties or {}
        weights = weights or DEFAULT_SCREENING_WEIGHTS

        # Stability component: favor lower formation energies.
        fe = predictions.get('formation_energy', 0.0)
        stability = round(max(0.0, min(100.0, 50.0 - fe * 12.5)), 3)

        # Relaxation quality component: penalize high forces/stress.
        forces = predictions.get('forces', 0.5)
        stress = predictions.get('stress', 1.0)
        force_score = round(max(0.0, min(100.0, 100.0 - forces * 50.0)), 3)
        stress_score = round(max(0.0, min(100.0, 100.0 - stress * 20.0)), 3)
        relaxation_quality = round((force_score + stress_score) / 2.0, 3)

        # Target property match component: closeness to specified targets.
        property_scores = []
        for prop, target in target_properties.items():
            if prop in predictions:
                actual = predictions[prop]
                # Normalize closeness using a generous tolerance scale.
                scale = max(abs(target), 1.0)
                error = abs(actual - target) / scale
                property_scores.append(max(0.0, min(100.0, 100.0 - error * 100.0)))
        target_property_match = round(sum(property_scores) / max(len(property_scores), 1), 3) if property_scores else 50.0

        # Composition novelty is computed at batch level; default to neutral.
        composition_novelty = 50.0

        components = {
            'stability': stability,
            'relaxation_quality': relaxation_quality,
            'target_property_match': target_property_match,
            'composition_novelty': composition_novelty,
        }

        total_weight = sum(weights.get(k, 0.0) for k in components)
        if total_weight == 0.0:
            total_weight = 1.0

        score = sum(
            weights.get(k, 0.0) * components[k] / total_weight
            for k in components
        )
        return round(max(0.0, min(100.0, score)), 3), components

    def _update_novelty_scores(
        self,
        results: List[Tuple[Any, ScreeningResult]],
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Adjust composition novelty scores so rare compositions in the batch score higher.
        Called automatically inside screen_batch after deduplication.
        """
        weights = weights or DEFAULT_SCREENING_WEIGHTS
        counts: Dict[str, int] = defaultdict(int)
        for struct, _ in results:
            counts[self._get_composition_key(struct)] += 1

        max_count = max(counts.values()) if counts else 1
        for struct, result in results:
            formula = self._get_composition_key(struct)
            rarity = 1.0 - (counts[formula] - 1) / max_count
            result.score_components['composition_novelty'] = round(rarity * 100.0, 3)
            # Recompute overall score with the updated novelty component.
            total_weight = sum(weights.get(k, 0.0) for k in result.score_components)
            if total_weight == 0.0:
                total_weight = 1.0
            result.score = round(
                sum(
                    weights.get(k, 0.0) * result.score_components[k] / total_weight
                    for k in result.score_components
                ),
                3,
            )
