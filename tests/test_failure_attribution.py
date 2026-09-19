"""Failure attribution must remain limited to observed diagnostics."""

from agents.failure_attribution import attribute_cause, classify_failure_mode
from agents.integrity import FORCE_KEY, STRESS_KEY


def test_force_and_stress_failures_are_explicit():
    assert classify_failure_mode([f"{FORCE_KEY} 2.0 > 1.0"]) == "high_residual_forces"
    assert classify_failure_mode([f"{STRESS_KEY} 3.0 > 1.0"]) == "high_stress"
    cause = attribute_cause(
        "high_residual_forces",
        {FORCE_KEY: 2.0},
        "test_domain",
    )
    assert "2.000" in cause
    assert "reference" not in cause.lower()


def test_unobserved_thermodynamic_failure_is_not_fabricated():
    mode = classify_failure_mode(["thermodynamic metric unavailable"])
    assert mode == "multi_criteria_failure"
    cause = attribute_cause(mode, {}, "test_domain")
    assert "multiple screening criteria" in cause
