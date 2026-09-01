"""Shared execution-boundary and scientific-integrity helpers.

Sprint 1 keeps the campaign usable offline for development while making the
research boundary explicit.  This module deliberately contains no scientific
models; it only describes the capabilities a campaign requested and the
capabilities that were actually initialized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional


SCHEMA_VERSION = "2.0.0"
LEGACY_SCHEMA_VERSION = "1.0.0"

# Canonical schema-v2 vocabulary.  The aliases are accepted at input
# boundaries so old callers can still be audited/replayed in development, but
# new records must use the canonical names below.
SCREENING_ENERGY_KEY = "predicted_energy_per_atom_ev"
VALIDATION_ENERGY_KEY = "energy_per_atom_ev"
FORCE_KEY = "max_force_ev_per_angstrom"
STRESS_KEY = "max_stress_gpa"


class RunMode(str, Enum):
    """Execution mode for a campaign."""

    DEVELOPMENT = "development"
    RESEARCH = "research"


class ScientificValidity(str, Enum):
    """Validity label attached to v2 provenance records."""

    DEMO_ONLY = "demo_only"
    RESEARCH_VALID = "research_valid"
    LEGACY_INVALID_ENERGY_SEMANTICS = "legacy_invalid_energy_semantics"


class LegacySchemaError(ValueError):
    """Raised when a v1 artifact is used as v2 scientific evidence."""


class ScientificPreflightError(RuntimeError):
    """Raised when research mode cannot prove all required capabilities."""

    def __init__(self, report: "ScientificPreflightReport") -> None:
        self.report = report
        super().__init__(report.message)


def normalize_run_mode(value: Any) -> RunMode:
    """Normalize a user/config value to :class:`RunMode`.

    A string is accepted for JSON/CLI/UI compatibility; invalid values fail at
    configuration construction rather than much later in a campaign.
    """

    if isinstance(value, RunMode):
        return value
    try:
        return RunMode(str(value).strip().lower())
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(mode.value for mode in RunMode)
        raise ValueError(f"Invalid run_mode {value!r}; expected one of: {allowed}") from exc


def schema_version_of(payload: Mapping[str, Any] | None) -> str:
    """Return an artifact schema version, treating missing versions as v1."""

    if not payload:
        return LEGACY_SCHEMA_VERSION
    value = payload.get("schema_version", LEGACY_SCHEMA_VERSION)
    return str(value)


def is_schema_v2(payload: Mapping[str, Any] | None) -> bool:
    """Whether an artifact is explicitly a schema-v2 artifact."""

    return schema_version_of(payload) == SCHEMA_VERSION


def is_legacy_schema(payload: Mapping[str, Any] | None) -> bool:
    """Whether an artifact must be quarantined from v2 scientific retrieval."""

    return not is_schema_v2(payload)


def assert_schema_v2_compatible(payload: Mapping[str, Any] | None, *, context: str = "artifact") -> None:
    """Reject v1/unknown artifacts at v2 scientific retrieval boundaries."""

    version = schema_version_of(payload)
    if version != SCHEMA_VERSION:
        raise LegacySchemaError(
            f"{context} has schema_version {version!r}; legacy/unknown artifacts "
            "are audit-only and are quarantined from schema-v2 scientific retrieval."
        )


def _canonical_energy_key(source: Mapping[str, Any], *, backend: str, validation: bool) -> Optional[str]:
    """Choose the *source* energy key for a result input.

    Legacy generic keys are mapped to a clearly labelled mock key when their
    source is not an explicitly real backend.  They are never copied through
    to v2 output under ``formation_energy`` or ``stability``.
    """

    target = VALIDATION_ENERGY_KEY if validation else SCREENING_ENERGY_KEY
    if target in source:
        return target
    if validation and "mock_energy_per_atom_ev" in source:
        return "mock_energy_per_atom_ev"
    if not validation and "mock_predicted_energy_per_atom_ev" in source:
        return "mock_predicted_energy_per_atom_ev"
    for key in ("energy_per_atom", "energy", "formation_energy_per_atom", "formation_energy"):
        if key in source:
            return key
    return None


def canonicalize_screening_predictions(
    predictions: Mapping[str, Any] | None,
    *,
    backend: str = "heuristic",
) -> Dict[str, Any]:
    """Convert screening results to schema-v2 vocabulary.

    This is intentionally lossy for legacy ``formation_energy`` and
    ``stability`` fields: those names carried ambiguous/incorrect semantics.
    """

    source = dict(predictions or {})
    result: Dict[str, Any] = {}
    energy_key = _canonical_energy_key(source, backend=backend, validation=False)
    if energy_key:
        output_key = (
            SCREENING_ENERGY_KEY
            if energy_key in {SCREENING_ENERGY_KEY, "energy", "energy_per_atom"}
            and backend.lower() in {"chgnet", "chgnet_thermodynamic_oracle"}
            else ("mock_predicted_energy_per_atom_ev" if energy_key not in {SCREENING_ENERGY_KEY, "mock_predicted_energy_per_atom_ev"} else energy_key)
        )
        result[output_key] = source[energy_key]
    if "max_force_ev_per_angstrom" in source:
        result[FORCE_KEY] = source["max_force_ev_per_angstrom"]
    elif "forces" in source:
        result[FORCE_KEY] = source["forces"]
    if "max_stress_gpa" in source:
        result[STRESS_KEY] = source["max_stress_gpa"]
    elif "stress" in source:
        result[STRESS_KEY] = source["stress"]
    # Preserve explicitly named non-thermodynamic properties (e.g. band gap)
    # while dropping generic energy/stability aliases.
    for key, value in source.items():
        if key in {
            "formation_energy", "formation_energy_per_atom", "stability",
            "energy", "energy_per_atom", "forces", "stress",
            SCREENING_ENERGY_KEY, "mock_predicted_energy_per_atom_ev",
            FORCE_KEY, STRESS_KEY,
        }:
            continue
        result[key] = value
    result.setdefault("energy_semantics", "raw_predicted_per_atom" if backend.lower() in {"chgnet", "chgnet_thermodynamic_oracle"} else "mock_raw_per_atom")
    return result


def canonicalize_validation_properties(
    properties: Mapping[str, Any] | None,
    *,
    calculator: str = "mock",
) -> Dict[str, Any]:
    """Convert validation results to schema-v2 vocabulary."""

    source = dict(properties or {})
    result: Dict[str, Any] = {}
    energy_key = _canonical_energy_key(source, backend=calculator, validation=True)
    if energy_key:
        output_key = (
            VALIDATION_ENERGY_KEY
            if energy_key in {VALIDATION_ENERGY_KEY, "energy", "energy_per_atom"}
            and calculator.lower() in {"ase", "vasp", "qe", "gpaw"}
            else ("mock_energy_per_atom_ev" if energy_key not in {VALIDATION_ENERGY_KEY, "mock_energy_per_atom_ev"} else energy_key)
        )
        result[output_key] = source[energy_key]
    if "total_energy_ev" in source:
        result["total_energy_ev"] = source["total_energy_ev"]
    elif "energy" in source:
        result["total_energy_ev"] = source["energy"]
    if FORCE_KEY in source:
        result[FORCE_KEY] = source[FORCE_KEY]
    elif "forces" in source:
        result[FORCE_KEY] = source["forces"]
    if STRESS_KEY in source:
        result[STRESS_KEY] = source[STRESS_KEY]
    elif "stress" in source:
        result[STRESS_KEY] = source["stress"]
    for key, value in source.items():
        if key in {
            "formation_energy", "formation_energy_per_atom", "energy",
            "energy_per_atom", "forces", "stress", "stability",
            VALIDATION_ENERGY_KEY, "mock_energy_per_atom_ev", "total_energy_ev",
            FORCE_KEY, STRESS_KEY,
        }:
            continue
        result[key] = value
    result.setdefault("energy_semantics", "raw_per_atom" if calculator.lower() != "mock" else "mock_raw_per_atom")
    return result


@dataclass
class ScientificPreflightReport:
    """Requested/actual capability report used by campaign startup."""

    run_mode: RunMode
    requested_backends: Dict[str, Any] = field(default_factory=dict)
    actual_backends: Dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def message(self) -> str:
        if self.valid:
            return "Research preflight passed: all required scientific components are configured."
        return "Research mode preflight failed; required scientific components are not configured: " + "; ".join(self.errors)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_mode": self.run_mode.value,
            "requested_backends": dict(self.requested_backends),
            "actual_backends": dict(self.actual_backends),
            "errors": list(self.errors),
            "valid": self.valid,
        }


def build_scientific_preflight(
    *,
    run_mode: Any,
    requested_backends: Mapping[str, Any],
    actual_backends: Mapping[str, Any],
    require_thermodynamics: bool = False,
    thermodynamics_available: bool = False,
) -> ScientificPreflightReport:
    """Validate the execution boundary without knowing model internals.

    The capability names are stable API.  Later sprints can set actual
    backends/thermodynamics availability without changing this gate.
    """

    mode = normalize_run_mode(run_mode)
    report = ScientificPreflightReport(
        run_mode=mode,
        requested_backends=dict(requested_backends),
        actual_backends=dict(actual_backends),
    )
    if mode != RunMode.RESEARCH:
        return report

    if report.requested_backends.get("generation") != "mattergen":
        report.errors.append("generation must request the MatterGen backend")
    if report.actual_backends.get("generation") != "mattergen":
        report.errors.append("MatterGen generation is unavailable or fell back to a development backend")
    if report.actual_backends.get("screening") not in {"chgnet", "chgnet_thermodynamic_oracle"}:
        report.errors.append("CHGNet screening is unavailable or fell back to heuristic scoring")
    requested_validation = report.requested_backends.get("validation")
    actual_validation = report.actual_backends.get("validation")
    if requested_validation == "disabled":
        if actual_validation not in {None, "disabled"}:
            report.errors.append("validation was requested disabled but an unexpected backend was initialized")
    elif actual_validation in {None, "mock", "heuristic", "disabled"}:
        report.errors.append("validation must use a configured non-mock scientific calculator")
    if report.requested_backends.get("synthesis") not in {None, "disabled"}:
        report.errors.append("synthesis is development-only and must be disabled in research mode")
    if report.actual_backends.get("synthesis") not in {None, "disabled"}:
        report.errors.append("synthesis must be disabled in research mode")
    if require_thermodynamics and not thermodynamics_available:
        report.errors.append("thermodynamics/hull capability is required but not configured (Sprint 3)")
    return report


def validity_for_mode(run_mode: Any) -> ScientificValidity:
    """Return the default validity label for a mode."""

    return (
        ScientificValidity.RESEARCH_VALID
        if normalize_run_mode(run_mode) == RunMode.RESEARCH
        else ScientificValidity.DEMO_ONLY
    )
