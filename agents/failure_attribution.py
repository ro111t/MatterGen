"""
failure_attribution.py — classifies and records why candidates failed screening.

Extracted from ExperienceDistiller to keep each module focused.
Called by ExperienceDistiller.distill_iteration() after each screening pass.
"""

from typing import Dict, List, Tuple, Any

from agents.career_memory import CareerMemory
from agents.integrity import FORCE_KEY, STRESS_KEY


# Map filter reason keywords to canonical failure mode strings
_FAILURE_MODE_KEYWORDS = {
    'band gap':       'wrong_band_gap',
    'max_force_ev_per_angstrom': 'high_residual_forces',
    'max force':      'high_residual_forces',
    'forces':         'high_residual_forces',
    'max_stress_gpa': 'high_stress',
    'max stress':     'high_stress',
    'stress':         'high_stress',
    'geometry':       'invalid_geometry',
}


def classify_failure_mode(filter_reasons: List[str]) -> str:
    """Map a list of filter failure reasons to a canonical mode string."""
    if not filter_reasons:
        return 'unknown'
    combined = ' '.join(filter_reasons).lower()
    for keyword, mode in _FAILURE_MODE_KEYWORDS.items():
        if keyword in combined:
            return mode
    return 'multi_criteria_failure'


def attribute_cause(
    failure_mode: str,
    predictions: Dict[str, float],
    domain: str,
) -> str:
    """Return a human-readable explanation for a canonical failure mode."""
    f    = predictions.get(FORCE_KEY, predictions.get('forces', 0.0))
    s    = predictions.get(STRESS_KEY, predictions.get('stress', 0.0))
    bg   = predictions.get('band_gap', 0.0)

    causes = {
        'wrong_band_gap': (
            f"Band gap {bg:.3f} eV outside target range for {domain}."
        ),
        'high_residual_forces': (
            f"Max forces {f:.3f} eV/Å — structure is far from a local "
            f"energy minimum; geometry is likely unrealistic."
        ),
        'high_stress': (
            f"Max stress {s:.3f} GPa — internal strain suggests "
            f"unrealistic bond lengths or angles."
        ),
        'invalid_geometry': (
            f"Geometry validation failed for {domain}; inspect lattice, atom "
            "positions, and periodic boundary conditions."
        ),
    }
    return causes.get(
        failure_mode,
        f"Failed multiple screening criteria in {domain}.",
    )


def record_failures(
    failing: List[Tuple],
    campaign_id: str,
    domain: str,
    memory: CareerMemory,
    max_record: int = 20,
) -> List[str]:
    """
    Record failure attributions for up to `max_record` failing candidates.

    Args:
        failing:     List of (structure, ScreeningResult) tuples.
        campaign_id: Current campaign ID.
        domain:      Materials domain string.
        memory:      CareerMemory instance to write into.
        max_record:  Cap on how many failures to store per iteration.

    Returns:
        List of stored failure attribution IDs.
    """
    ids = []
    for struct, result in failing[:max_record]:
        composition = getattr(struct, 'composition', None)
        if hasattr(composition, 'reduced_formula'):
            formula = composition.reduced_formula
        elif isinstance(struct, dict):
            formula = struct.get('composition', 'Unknown')
        else:
            formula = str(composition or 'Unknown')

        mode  = classify_failure_mode(result.filter_reasons)
        cause = attribute_cause(mode, result.predictions, domain)

        fa_id = memory.store_failure(
            campaign_id=campaign_id,
            domain=domain,
            formula=formula,
            failure_mode=mode,
            structural_features={'formula': formula},
            properties=result.predictions,
            attributed_cause=cause,
        )
        ids.append(fa_id)

    return ids
