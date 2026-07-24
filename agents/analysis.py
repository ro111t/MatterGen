"""
AnalysisAgent: Compare ML screening predictions against high-fidelity validation
results, extract calibration metrics, identify failure modes, and surface the most
promising candidates.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np


@dataclass
class AnalysisResult:
    """Structured analysis of one iteration's results."""
    num_screened: int = 0
    num_validated: int = 0
    num_converged: int = 0
    ml_vs_dft_mae: Dict[str, float] = field(default_factory=dict)
    ml_vs_dft_rmse: Dict[str, float] = field(default_factory=dict)
    ml_vs_dft_bias: Dict[str, float] = field(default_factory=dict)
    pearson_r: Dict[str, float] = field(default_factory=dict)
    top_candidate_id: str = ""
    top_candidate_score: float = 0.0
    top_candidate_properties: Dict[str, float] = field(default_factory=dict)
    failure_modes: List[str] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)


class AnalysisAgent:
    """
    Analyzes the relationship between fast ML screening and expensive DFT validation.

    Args:
        properties_to_compare: List of property names to compare between screening
            predictions and validation results. Defaults to energy-related fields.
        top_k: Number of top candidates to retain in the analysis output.
    """

    def __init__(
        self,
        properties_to_compare: Optional[List[str]] = None,
        top_k: int = 3,
    ):
        self.properties_to_compare = properties_to_compare or [
            "formation_energy",
            "energy",
            "stability",
            "forces",
        ]
        self.top_k = top_k
        self.iteration_history: List[AnalysisResult] = []

    def analyze_batch(
        self,
        screening_results: List[Tuple[Any, Any]],
        validation_results: List[Any],
    ) -> AnalysisResult:
        """
        Compare screening predictions with validation results.

        Args:
            screening_results: List of (structure, ScreeningResult) tuples.
            validation_results: List of ValidationResult objects.

        Returns:
            AnalysisResult with calibration metrics and candidate rankings.
        """
        result = AnalysisResult(
            num_screened=len(screening_results),
            num_validated=len(validation_results),
            num_converged=sum(1 for v in validation_results if getattr(v, 'converged', False)),
        )

        if not validation_results:
            result.insights.append("No validation data available for analysis.")
            self.iteration_history.append(result)
            return result

        # Build a lookup from structure_id to screening prediction.
        screening_lookup: Dict[str, Dict[str, float]] = {}
        for struct, screen_res in screening_results:
            sid = getattr(screen_res, 'structure_id', '')
            if sid:
                screening_lookup[sid] = getattr(screen_res, 'predictions', {})

        # Pair validation results with their screening predictions.
        paired: List[Tuple[str, Dict[str, float], Dict[str, float]]] = []
        for v in validation_results:
            if not getattr(v, 'converged', False):
                continue
            sid = getattr(v, 'structure_id', '')
            dft_props = getattr(v, 'properties', {})
            ml_props = screening_lookup.get(sid, {})
            paired.append((sid, ml_props, dft_props))

        # Compute per-property metrics.
        for prop in self.properties_to_compare:
            ml_vals, dft_vals, ids = [], [], []
            for sid, ml_props, dft_props in paired:
                ml_val = ml_props.get(prop)
                dft_val = dft_props.get(prop)
                if ml_val is not None and dft_val is not None:
                    ml_vals.append(float(ml_val))
                    dft_vals.append(float(dft_val))
                    ids.append(sid)

            if not ml_vals:
                continue

            ml_vals = np.asarray(ml_vals, dtype=float)
            dft_vals = np.asarray(dft_vals, dtype=float)
            errors = ml_vals - dft_vals

            result.ml_vs_dft_mae[prop] = round(float(np.mean(np.abs(errors))), 4)
            result.ml_vs_dft_rmse[prop] = round(float(math.sqrt(np.mean(errors ** 2))), 4)
            result.ml_vs_dft_bias[prop] = round(float(np.mean(errors)), 4)

            if len(ml_vals) > 1:
                corr = self._safe_pearson(ml_vals, dft_vals)
                if corr is not None:
                    result.pearson_r[prop] = round(corr, 4)

            # Failure mode: large systematic over/under-estimation
            if result.ml_vs_dft_bias[prop] > 0.5 * result.ml_vs_dft_mae.get(prop, 1e-9):
                result.failure_modes.append(f"ML systematically overestimates {prop}.")
            elif result.ml_vs_dft_bias[prop] < -0.5 * result.ml_vs_dft_mae.get(prop, 1e-9):
                result.failure_modes.append(f"ML systematically underestimates {prop}.")

            # Failure mode: poor correlation on multiple samples
            if result.pearson_r.get(prop, 0.0) < 0.3 and len(ml_vals) > 2:
                result.failure_modes.append(f"Weak correlation for {prop} (r={result.pearson_r[prop]:.2f}).")

        # Identify top candidates by a combined DFT + ML score.
        candidates = []
        for sid, ml_props, dft_props in paired:
            stability = dft_props.get('stability', dft_props.get('formation_energy', 0.0))
            ml_score = 0.0
            for p in ['formation_energy', 'stability']:
                val = ml_props.get(p)
                if val is not None:
                    ml_score = float(val)
                    break
            # Lower (more negative) stability/energy is better; negate for ranking.
            score = -float(stability) if stability is not None else 0.0
            candidates.append({
                'structure_id': sid,
                'score': score,
                'properties': {**ml_props, **dft_props},
            })

        candidates.sort(key=lambda x: x['score'], reverse=True)
        if candidates:
            best = candidates[0]
            result.top_candidate_id = best['structure_id']
            result.top_candidate_score = round(best['score'], 4)
            result.top_candidate_properties = best['properties']

        # Natural-language insights
        result.insights = self._generate_insights(result)
        self.iteration_history.append(result)
        return result

    def _safe_pearson(self, x: np.ndarray, y: np.ndarray) -> Optional[float]:
        """Compute Pearson r, returning None if not defined."""
        if x.size < 2 or y.size < 2:
            return None
        if np.std(x) == 0 or np.std(y) == 0:
            return None
        try:
            return float(np.corrcoef(x, y)[0, 1])
        except Exception:
            return None

    def _generate_insights(self, result: AnalysisResult) -> List[str]:
        """Produce concise, human-readable insights."""
        insights: List[str] = []
        insights.append(
            f"Validated {result.num_converged}/{result.num_validated} structures."
        )

        for prop, mae in result.ml_vs_dft_mae.items():
            rmse = result.ml_vs_dft_rmse.get(prop, 0.0)
            bias = result.ml_vs_dft_bias.get(prop, 0.0)
            r = result.pearson_r.get(prop)
            line = f"{prop}: MAE={mae:.3f}, RMSE={rmse:.3f}, bias={bias:.3f}"
            if r is not None:
                line += f", r={r:.3f}"
            insights.append(line)

        if result.failure_modes:
            insights.append("Failure modes: " + "; ".join(result.failure_modes[:3]))
        else:
            insights.append("No major systematic discrepancies detected between ML and DFT.")

        if result.top_candidate_id:
            insights.append(
                f"Top candidate: {result.top_candidate_id} "
                f"(score={result.top_candidate_score:.3f})."
            )

        return insights

    def get_calibration_summary(self) -> Dict[str, Any]:
        """Aggregate calibration metrics across all analyzed iterations."""
        if not self.iteration_history:
            return {}

        summary: Dict[str, Dict[str, List[float]]] = {}
        for res in self.iteration_history:
            for prop, mae in res.ml_vs_dft_mae.items():
                summary.setdefault(prop, {'mae': [], 'rmse': [], 'bias': [], 'r': []})
                summary[prop]['mae'].append(mae)
                summary[prop]['rmse'].append(res.ml_vs_dft_rmse.get(prop, 0.0))
                summary[prop]['bias'].append(res.ml_vs_dft_bias.get(prop, 0.0))
                summary[prop]['r'].append(res.pearson_r.get(prop, 0.0))

        return {
            prop: {
                'avg_mae': round(float(np.mean(vals['mae'])), 4),
                'avg_rmse': round(float(np.mean(vals['rmse'])), 4),
                'avg_bias': round(float(np.mean(vals['bias'])), 4),
                'avg_r': round(float(np.mean([v for v in vals['r'] if v is not None])), 4) if vals['r'] else None,
            }
            for prop, vals in summary.items()
        }
