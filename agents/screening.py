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

from agents.integrity import (
    FORCE_KEY,
    SCREENING_ENERGY_KEY,
    STRESS_KEY,
    RunMode,
    normalize_run_mode,
    canonicalize_screening_predictions,
)
from agents.geometry import (
    DEFAULT_MIN_DISTANCE_ANGSTROM,
    GeometryValidationResult,
    GeometryValidator,
)
from agents.budget import DualBudgetTracker

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
    "relaxation_quality": 0.40,
    "target_property_match": 0.35,
    "composition_novelty": 0.25,
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
    backend: str = ""
    scientific_validity: str = "demo_only"
    # Geometry/provenance fields are additive to schema-v2 and intentionally
    # default to the historical screening behavior for direct callers.
    geometry_valid: Optional[bool] = None
    geometry_failure_code: Optional[str] = None
    geometry_details: Dict[str, Any] = field(default_factory=dict)
    provenance_stage: str = "screening"
    oracle_evaluated: Optional[bool] = None
    oracle_cache_hit: Optional[bool] = None

    def __post_init__(self) -> None:
        # Ensure explicitly constructed results use schema-v2 vocabulary.
        # Invalid geometry and budget-exhausted candidates must contain no
        # synthetic energy/heuristic prediction at all.
        if self.geometry_valid is False or self.geometry_failure_code in {
            "INVALID_GEOMETRY", "ORACLE_BUDGET_EXHAUSTED", "PREDICTION_FAILED"
        }:
            self.predictions = {}
        else:
            self.predictions = canonicalize_screening_predictions(
                self.predictions, backend=self.backend or "heuristic"
            )

    @property
    def failure_code(self) -> Optional[str]:
        return self.geometry_failure_code

    @property
    def details(self) -> Dict[str, Any]:
        return self.geometry_details


class ScreeningAgent:
    """
    Screens generated candidates using CHGNet energy/force predictions.
    Score is normalized 0-100; higher = better candidate.
    """

    def __init__(self, run_mode: RunMode | str = RunMode.DEVELOPMENT, mode: Optional[str] = None,
                 min_distance: float = DEFAULT_MIN_DISTANCE_ANGSTROM,
                 geometry_validator: Optional[GeometryValidator] = None,
                 minimum_distance: Optional[float] = None):
        self.run_mode = normalize_run_mode(mode if mode is not None else run_mode)
        self.chgnet = None
        self.last_backend_used: str = "uninitialized"
        self.prediction_cache: Dict[str, Dict[str, float]] = {}
        self.geometry_validator = geometry_validator or GeometryValidator(
            min_distance=(minimum_distance if minimum_distance is not None else min_distance)
        )
        self.last_geometry_results: Dict[str, GeometryValidationResult] = {}
        self.last_budget_snapshot: Dict[str, Any] = {}
        self._init_models()

    def _init_models(self):
        if HAS_CHGNET:
            try:
                self.chgnet = CHGNet.load()
                self.last_backend_used = "chgnet"
                print("  [Screener] CHGNet loaded successfully")
            except Exception as e:
                if self.run_mode == RunMode.RESEARCH:
                    raise RuntimeError(
                        f"Research mode requires CHGNet screening; model initialization failed: {e}"
                    ) from e
                self.last_backend_used = "heuristic"
                print(f"  [Screener] CHGNet load failed ({e}), using heuristic scoring")
        else:
            if self.run_mode == RunMode.RESEARCH:
                raise RuntimeError(
                    "Research mode requires CHGNet screening, but CHGNet is not installed."
                )
            self.last_backend_used = "heuristic"
            print("  [Screener] CHGNet not installed, using heuristic scoring")
        
    def screen_batch(
        self,
        structures: List[Any],
        criteria: Dict[str, Any],
        target_properties: Optional[Dict[str, float]] = None,
        weights: Optional[Dict[str, float]] = None,
        deduplicate: bool = True,
        budget_tracker: Optional[Any] = None,
        oracle_budget_tracker: Optional[Any] = None,
        oracle_budget: Optional[int] = None,
    ) -> List[Tuple[Any, ScreeningResult]]:
        """
        Screen all structures; return all results sorted by score (best first).

        Args:
            structures: List of pymatgen Structure objects (or stub dicts)
            criteria: Dict with optional force/stress thresholds.
            target_properties: Optional map of property name -> target value, used to
                compute a target-property-match component of the score.
            weights: Optional map overriding the default multi-objective score weights.
            deduplicate: If True, keep only the highest-scored structure per composition.

        Returns:
            List of (structure, ScreeningResult) sorted by score descending
        """
        target_properties = target_properties or {}
        weights = dict(weights or DEFAULT_SCREENING_WEIGHTS)

        # ``oracle_budget_tracker`` is a readable alias for integrations that
        # pass only the oracle accounting object.  A campaign passes the dual
        # tracker through ``budget_tracker``.
        budget_tracker = budget_tracker or oracle_budget_tracker
        if budget_tracker is None and oracle_budget is not None:
            budget_tracker = DualBudgetTracker(oracle_budget=oracle_budget)
        raw_results = []
        for i, struct in enumerate(structures):
            struct_id = self._get_struct_id(struct, i)
            geometry = self.geometry_validator.validate(struct)
            self.last_geometry_results[struct_id] = geometry
            if budget_tracker is not None:
                budget_tracker.record_geometry(geometry.valid)

            if not geometry.valid:
                details = dict(geometry.details)
                details.setdefault("geometry_code", geometry.code)
                if geometry.minimum_distance is not None:
                    details.setdefault("minimum_distance", geometry.minimum_distance)
                if geometry.offending_pair is not None:
                    details.setdefault("offending_pair", list(geometry.offending_pair))
                reason = "INVALID_GEOMETRY"
                if details:
                    reason = f"{reason}: {details}"
                raw_results.append((struct, ScreeningResult(
                    structure_id=struct_id,
                    predictions={},
                    score=0.0,
                    passes_filters=False,
                    filter_reasons=["INVALID_GEOMETRY", reason],
                    score_components={},
                    backend="geometry_validation",
                    scientific_validity=("research_valid" if self.run_mode == RunMode.RESEARCH else "demo_only"),
                    geometry_valid=False,
                    geometry_failure_code="INVALID_GEOMETRY",
                    geometry_details=details,
                    provenance_stage="geometry_validation",
                    oracle_evaluated=False,
                )))
                continue

            cache_hit = struct_id in self.prediction_cache
            if budget_tracker is not None and not budget_tracker.admit_oracle(cache_hit=cache_hit):
                raw_results.append((struct, ScreeningResult(
                    structure_id=struct_id,
                    predictions={},
                    score=0.0,
                    passes_filters=False,
                    filter_reasons=["ORACLE_BUDGET_EXHAUSTED"],
                    score_components={},
                    backend="oracle_budget",
                    scientific_validity=("research_valid" if self.run_mode == RunMode.RESEARCH else "demo_only"),
                    geometry_valid=True,
                    geometry_failure_code="ORACLE_BUDGET_EXHAUSTED",
                    geometry_details={"oracle_budget_remaining": budget_tracker.oracle_budget_remaining},
                    provenance_stage="oracle_budget",
                    oracle_evaluated=False,
                    oracle_cache_hit=cache_hit,
                )))
                continue

            try:
                predictions = self._predict(struct, struct_id)
            except Exception as exc:
                # Research mode remains fail-closed (the underlying _predict
                # exception is part of Sprint 1's execution boundary).  In
                # development, preserve candidate provenance without inventing
                # a score when a caller's oracle actually fails.
                if self.run_mode == RunMode.RESEARCH:
                    raise
                raw_results.append((struct, ScreeningResult(
                    structure_id=struct_id,
                    predictions={},
                    score=0.0,
                    passes_filters=False,
                    filter_reasons=["PREDICTION_FAILED", str(exc)],
                    score_components={},
                    backend="prediction_failure",
                    scientific_validity="demo_only",
                    geometry_valid=True,
                    geometry_failure_code="PREDICTION_FAILED",
                    geometry_details={"error": str(exc)},
                    provenance_stage="screening",
                    oracle_evaluated=True,
                    oracle_cache_hit=cache_hit,
                )))
                continue
            passes, reasons = self._apply_filters(predictions, criteria)
            score, components = self._calculate_score(predictions, target_properties, weights)
            raw_results.append((struct, ScreeningResult(
                structure_id=struct_id,
                predictions=predictions,
                score=score,
                passes_filters=passes,
                filter_reasons=reasons,
                score_components=components,
                backend=self.last_backend_used,
                scientific_validity=("research_valid" if self.run_mode == RunMode.RESEARCH else "demo_only"),
                geometry_valid=True,
                geometry_details=geometry.details,
                provenance_stage="screening",
                oracle_evaluated=True,
                oracle_cache_hit=cache_hit,
            )))

        if deduplicate:
            raw_results = self._deduplicate_by_composition(raw_results)

        self._update_novelty_scores(raw_results, weights)

        raw_results.sort(key=lambda x: x[1].score, reverse=True)
        for rank, (_, result) in enumerate(raw_results, start=1):
            result.rank = rank

        if budget_tracker is not None and hasattr(budget_tracker, "to_dict"):
            self.last_budget_snapshot = budget_tracker.to_dict()

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
            try:
                preds = self._chgnet_predict(struct)
                self.last_backend_used = "chgnet"
            except Exception as exc:
                if self.run_mode == RunMode.RESEARCH:
                    raise RuntimeError(
                        f"CHGNet prediction failed for {struct_id} in research mode: {exc}"
                    ) from exc
                preds = self._heuristic_predict(struct)
                self.last_backend_used = "heuristic"
        else:
            if self.run_mode == RunMode.RESEARCH:
                raise RuntimeError(
                    f"CHGNet cannot screen candidate {struct_id} in research mode; "
                    "heuristic substitution is disabled."
                )
            preds = self._heuristic_predict(struct)
            self.last_backend_used = "heuristic"

        self.prediction_cache[struct_id] = preds
        return preds

    def _chgnet_predict(self, struct: Any) -> Dict[str, float]:
        """Run real CHGNet inference."""
        result = self.chgnet.predict_structure(struct)
        energy = float(result['e']) if 'e' in result else float(result.get('energy', 0))
        forces = result.get('f', result.get('forces', np.zeros((1, 3))))
        stress = result.get('s', result.get('stress', np.zeros((3, 3))))

        max_force = float(np.max(np.linalg.norm(np.array(forces).reshape(-1, 3), axis=1)))
        max_stress = float(np.max(np.abs(np.array(stress))))

        return {
            SCREENING_ENERGY_KEY: energy,
            FORCE_KEY: max_force,
            STRESS_KEY: max_stress,
            'energy_semantics': 'raw_predicted_per_atom',
        }

    def _heuristic_predict(self, struct: Any) -> Dict[str, float]:
        """
        Deterministic heuristic scoring based on structural features.
        Produces consistent scores for the same structure across processes.
        """
        seed_val = 0
        if HAS_PYMATGEN and isinstance(struct, Structure):
            formula = struct.composition.reduced_formula
            n_atoms = len(struct)
            seed_val = int(hashlib.sha256(formula.encode('utf-8')).hexdigest(), 16) % 10000
            rng = np.random.default_rng(seed_val)

            energy = float(rng.uniform(-4.0, -0.5))
            max_force = float(rng.uniform(0.01, 0.8))
            max_stress = float(rng.uniform(0.1, 3.0))
        elif isinstance(struct, dict):
            formula = struct.get('composition', '')
            seed_val = int(hashlib.sha256(formula.encode('utf-8')).hexdigest(), 16) % 10000
            rng = np.random.default_rng(seed_val)
            energy = float(rng.uniform(-4.0, -0.5))
            max_force = float(rng.uniform(0.01, 0.8))
            max_stress = float(rng.uniform(0.1, 3.0))
        else:
            rng = np.random.default_rng(0)
            energy = float(rng.uniform(-3.0, -1.0))
            max_force = 0.3
            max_stress = 1.0

        return {
            'mock_predicted_energy_per_atom_ev': energy,
            FORCE_KEY: max_force,
            STRESS_KEY: max_stress,
            'energy_semantics': 'mock_raw_per_atom',
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

        # Raw model energy is recorded for audit only.  Until a reference-set
        # thermodynamic calculation exists (Sprint 3), it must not be used as
        # a cross-composition filter.

        # Random mock structures have large forces (not relaxed).
        # Generated structures are unrelaxed; force/stress thresholds are
        # geometry diagnostics, not thermodynamic claims.
        max_f = criteria.get('max_force_ev_per_angstrom', 500.0)
        forces = predictions.get(FORCE_KEY, 0.0)
        if forces > max_f:
            reasons.append(f"max_force_ev_per_angstrom {forces:.3f} > {max_f}")

        max_stress = criteria.get('max_stress_gpa')
        stress = predictions.get(STRESS_KEY, 0.0)
        if max_stress is not None and stress > max_stress:
            reasons.append(f"max_stress_gpa {stress:.3f} > {max_stress}")

        return len(reasons) == 0, reasons

    def _calculate_score(
        self,
        predictions: Dict[str, float],
        target_properties: Optional[Dict[str, float]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Multi-objective score normalized to 0-100. Higher = better candidate.

        Combines relaxation quality, target-property match, and within-batch
        composition novelty. Raw energy is diagnostic-only until Sprint 3.
        """
        target_properties = target_properties or {}
        weights = weights or DEFAULT_SCREENING_WEIGHTS
        weights = dict(weights)

        # Relaxation quality component: penalize high forces/stress.
        forces = predictions.get(FORCE_KEY, 0.5)
        stress = predictions.get(STRESS_KEY, 1.0)
        force_score = round(max(0.0, min(100.0, 100.0 - forces * 50.0)), 3)
        stress_score = round(max(0.0, min(100.0, 100.0 - stress * 20.0)), 3)
        relaxation_quality = round((force_score + stress_score) / 2.0, 3)

        # Target property match component: closeness to specified targets.
        property_scores = []
        for prop, target in target_properties.items():
            if prop in predictions and prop != SCREENING_ENERGY_KEY:
                actual = predictions[prop]
                # Normalize closeness using a generous tolerance scale.
                scale = max(abs(target), 1.0)
                error = abs(actual - target) / scale
                property_scores.append(max(0.0, min(100.0, 100.0 - error * 100.0)))
        target_property_match = round(sum(property_scores) / max(len(property_scores), 1), 3) if property_scores else 50.0

        # Composition novelty is computed at batch level; default to neutral.
        composition_novelty = 50.0

        components = {
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
            # Invalid geometry and candidates denied an oracle slot retain an
            # explicit score of zero; novelty must never resurrect them.
            if result.geometry_valid is False or result.provenance_stage == "oracle_budget":
                continue
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
