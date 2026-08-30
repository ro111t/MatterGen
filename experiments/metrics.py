"""Fail-closed metrics over complete campaign provenance.

``top_candidates`` is a presentation projection and is deliberately never used
by this module. Metrics are computed from the ordered ``candidates`` records in
``campaign_provenance.json`` (or an explicitly embedded equivalent), preserving
proposal order, oracle order, failures, and geometry rejections.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple


class ProvenanceIntegrityError(ValueError):
    """Raised when a metric request has no complete structured provenance."""


@dataclass
class CandidateRecord:
    """Normalized candidate lifecycle record used by analysis and QE selection."""

    candidate_id: str
    run_id: str
    task_id: str
    condition: str
    seed: int
    iteration: int
    proposal_index: int
    oracle_call_index: Optional[int]
    reduced_formula: Optional[str]
    anonymous_stoichiometry: Optional[str]
    structural_prototype: Optional[str]
    geometry_valid: bool
    geometry_failure_reason: Optional[str]
    evaluated_by_oracle: bool
    oracle_success: bool
    oracle_failure_code: Optional[str]
    predicted_energy_above_hull_ev_per_atom: Optional[float]
    predicted_thermodynamically_stable: Optional[bool]
    retained_by_hull: Optional[bool]
    decomposition_products: List[Dict[str, Any]] = field(default_factory=list)
    memory_priority_score: Optional[float] = None
    was_cached: bool = False
    structure: Optional[Dict[str, Any]] = None
    structure_path: Optional[str] = None
    structure_hash: Optional[str] = None
    provenance_missing_fields: List[str] = field(default_factory=list)
    record_status: Optional[str] = None
    rejection_stage: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunMetrics:
    """Metrics computed from a complete or explicitly incomplete run."""

    run_id: str
    task_id: str
    condition: str
    seed: int
    proposals_generated: int
    geometry_valid_count: int
    invalid_geometry_count: int
    geometry_yield: float
    oracle_evaluations: int
    oracle_budget: int
    oracle_success_count: int
    oracle_failure_count: int
    oracle_success_rate: float
    oracle_calls_to_first_candidate_at_or_below_0_10: Optional[int]
    reached_0_10_threshold: bool
    count_at_or_below_0_00: int
    count_at_or_below_0_03: int
    count_at_or_below_0_05: int
    count_at_or_below_0_10: int
    fraction_at_or_below_0_00: float
    fraction_at_or_below_0_03: float
    fraction_at_or_below_0_05: float
    fraction_at_or_below_0_10: float
    best_energy_above_hull_overall: Optional[float]
    best_energy_at_fixed_oracle_budgets: Dict[int, Optional[float]]
    area_under_best_curve: Optional[float]
    unique_reduced_compositions_count: int
    unique_anonymous_stoichiometries_count: int
    unique_prototypes_count: int
    memory_directives_applied_count: int
    memory_directives_rejected_count: int
    memory_directives_unsupported_count: int
    memory_prioritized_candidates_count: int
    total_candidates_recorded: int
    missing_hull_energy_count: int
    primary_endpoint_censored: bool = False
    provenance_complete: bool = True
    provenance_missing_fields: List[str] = field(default_factory=list)
    oracle_failure_codes: Dict[str, int] = field(default_factory=dict)
    oracle_order_source: str = "structured_record_or_list_order"
    run_status: str = "unknown"
    shuffle_validation: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _embedded_provenance(data: Mapping[str, Any]) -> Mapping[str, Any]:
    """Resolve an authoritative campaign provenance object."""

    for key in ("campaign_provenance", "provenance"):
        value = data.get(key)
        if isinstance(value, Mapping) and isinstance(value.get("candidates"), list):
            return value
    if isinstance(data.get("candidates"), list):
        return data
    if isinstance(data.get("candidate_records"), list):
        return {**data, "candidates": data["candidate_records"]}
    raise ProvenanceIntegrityError(
        "Complete structured provenance is required; metrics never consume top_candidates"
    )


def _as_bool(value: Any, *, default: Optional[bool] = None) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return default


def _finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _energy_from_record(item: Mapping[str, Any]) -> Optional[float]:
    """Read only explicit above-hull values from canonical nested fields."""

    values: List[Any] = [
        item.get("predicted_energy_above_hull_ev_per_atom"),
        item.get("energy_above_hull_ev_per_atom"),
    ]
    for key in ("screening_predictions", "predictions"):
        nested = item.get(key)
        if isinstance(nested, Mapping):
            values.extend(
                nested.get(name)
                for name in (
                    "predicted_energy_above_hull_ev_per_atom",
                    "energy_above_hull_ev_per_atom",
                )
            )
    for value in values:
        result = _finite_float(value)
        if result is not None:
            return result
    return None


def _decomposition(item: Mapping[str, Any]) -> List[Dict[str, Any]]:
    value = item.get("decomposition_products")
    if value is None and isinstance(item.get("screening_predictions"), Mapping):
        value = item["screening_predictions"].get("decomposition_products")
    if value is None and isinstance(item.get("predictions"), Mapping):
        value = item["predictions"].get("decomposition_products")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return []
    if isinstance(value, Mapping):
        if "formula" in value or "composition" in value:
            value = [value]
        else:
            # A compact mapping {formula: coefficient} is a supported source
            # schema, but atom counts/structures are still not fabricated.
            value = [{"formula": formula, "amount": coefficient} for formula, coefficient in value.items()]
    return [dict(v) for v in value if isinstance(v, Mapping)] if isinstance(value, list) else []


def extract_candidates_from_manifest(
    manifest_data: Dict[str, Any],
    run_id: str,
    task_id: str,
    condition: str,
    seed: int,
) -> List[CandidateRecord]:
    """Extract every candidate from complete structured campaign provenance."""

    source = _embedded_provenance(manifest_data)
    raw_candidates = source.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ProvenanceIntegrityError("Structured provenance candidates must be a list")

    oracle_events: List[Mapping[str, Any]] = []
    for key in ("oracle_events", "oracle_evaluations_records", "oracle_records"):
        if isinstance(source.get(key), list):
            oracle_events = [x for x in source[key] if isinstance(x, Mapping)]
            break
    oracle_index_by_id: Dict[str, int] = {}
    for i, event in enumerate(oracle_events, 1):
        cid = event.get("candidate_id") or event.get("structure_id")
        if cid is not None and event.get("oracle_call_index") is not None:
            try:
                oracle_index_by_id[str(cid)] = int(event["oracle_call_index"])
            except (TypeError, ValueError):
                # Invalid event order is not replaced with list order: memory
                # arms can reorder candidates before oracle admission.
                pass

    priority_audits = source.get("memory_priority_audit", [])
    priority_map = {
        str(p.get("candidate_id")): _finite_float(p.get("score"))
        for p in priority_audits
        if isinstance(p, Mapping) and p.get("candidate_id") is not None
    }
    records: List[CandidateRecord] = []
    used_oracle_indices: set[int] = set()

    for list_index, raw in enumerate(raw_candidates, 1):
        if not isinstance(raw, Mapping):
            raise ProvenanceIntegrityError(f"Candidate record {list_index} is not an object")
        cid_value = raw.get("candidate_id") or raw.get("structure_id")
        if cid_value is None or str(cid_value).strip() == "":
            raise ProvenanceIntegrityError(f"Candidate record {list_index} has no candidate_id")
        cid = str(cid_value)
        missing: List[str] = []

        geom = _as_bool(raw.get("geometry_valid"), default=None)
        geometry_reason = (
            raw.get("geometry_failure_code")
            or raw.get("geometry_failure_reason")
            or raw.get("geometry_rejection_reason")
        )
        if geom is None:
            missing.append("geometry_valid")
            geom = False

        evaluated = _as_bool(
            raw.get("oracle_evaluated", raw.get("evaluated_by_oracle")), default=None
        )
        if evaluated is None:
            missing.append("oracle_evaluated")
            evaluated = False
        if not geom:
            evaluated = False

        explicit_oracle_index = raw.get("oracle_call_index")
        if explicit_oracle_index is None:
            explicit_oracle_index = raw.get("oracle_sequence", raw.get("oracle_order"))
        if explicit_oracle_index is None:
            explicit_oracle_index = oracle_index_by_id.get(cid)
        oracle_idx: Optional[int] = None
        if evaluated and explicit_oracle_index is not None:
            try:
                if isinstance(explicit_oracle_index, bool):
                    raise ValueError("boolean oracle_call_index")
                oracle_idx = int(explicit_oracle_index)
                # Reject fractional values that int() would otherwise
                # truncate into a plausible-looking call index.
                if isinstance(explicit_oracle_index, float) and not explicit_oracle_index.is_integer():
                    raise ValueError("fractional oracle_call_index")
            except (TypeError, ValueError):
                missing.append("oracle_call_index")
            if oracle_idx is not None and oracle_idx <= 0:
                missing.append("invalid_oracle_call_index")
                oracle_idx = None
        if oracle_idx is not None:
            used_oracle_indices.add(oracle_idx)

        try:
            iteration_i = int(raw.get("iteration", 0))
        except (TypeError, ValueError):
            iteration_i = 0
            missing.append("iteration")
        proposal_value = raw.get(
            "proposal_index", raw.get("proposal_order", raw.get("generation_order"))
        )
        try:
            proposal_index = int(proposal_value) if proposal_value is not None else list_index
        except (TypeError, ValueError):
            proposal_index = list_index
            missing.append("proposal_index")

        energy = _energy_from_record(raw)
        failure_code = raw.get("oracle_failure_code") or raw.get("failure_code")
        evaluated_result = _as_bool(raw.get("oracle_success"), default=None)
        if evaluated_result is None:
            evaluated_result = evaluated and energy is not None and not failure_code
        success = bool(evaluated and evaluated_result and energy is not None)
        if evaluated and not success and failure_code is None and energy is None:
            failure_code = "MISSING_ORACLE_RESULT"

        formula = raw.get("composition") or raw.get("formula") or raw.get("reduced_formula")
        structure = raw.get("structure")
        if structure is not None and not isinstance(structure, Mapping):
            missing.append("structure")
            structure = None

        records.append(
            CandidateRecord(
                candidate_id=cid,
                run_id=run_id,
                task_id=task_id,
                condition=condition,
                seed=seed,
                iteration=iteration_i,
                proposal_index=proposal_index,
                oracle_call_index=oracle_idx,
                reduced_formula=str(formula) if formula is not None else None,
                anonymous_stoichiometry=raw.get("anonymous_stoichiometry"),
                structural_prototype=raw.get("structural_prototype") or raw.get("prototype"),
                geometry_valid=bool(geom),
                geometry_failure_reason=str(geometry_reason) if geometry_reason else None,
                evaluated_by_oracle=bool(evaluated),
                oracle_success=success,
                oracle_failure_code=str(failure_code) if failure_code else None,
                predicted_energy_above_hull_ev_per_atom=energy,
                predicted_thermodynamically_stable=_as_bool(
                    raw.get("predicted_thermodynamically_stable"), default=None
                ),
                retained_by_hull=_as_bool(raw.get("retained_by_hull_threshold"), default=None),
                decomposition_products=_decomposition(raw),
                memory_priority_score=priority_map.get(cid),
                was_cached=bool(_as_bool(raw.get("oracle_cache_hit"), default=False)),
                structure=dict(structure) if isinstance(structure, Mapping) else None,
                structure_path=str(raw.get("structure_path")) if raw.get("structure_path") else None,
                structure_hash=str(raw.get("structure_hash")) if raw.get("structure_hash") else None,
                provenance_missing_fields=missing,
                record_status=raw.get("status"),
                rejection_stage=raw.get("rejection_stage"),
            )
        )

    explicit_indices = [
        record.oracle_call_index
        for record in records
        if record.evaluated_by_oracle and record.oracle_call_index is not None
    ]
    duplicate_indices = {idx for idx in explicit_indices if explicit_indices.count(idx) > 1}
    if duplicate_indices:
        for record in records:
            if record.oracle_call_index in duplicate_indices:
                record.provenance_missing_fields.append("duplicate_oracle_call_index")
                record.oracle_call_index = None
    # Never synthesize oracle order from generation/list order. Memory can
    # reorder candidates before admission, so a missing index is a provenance
    # failure and the run is excluded from order-dependent scientific metrics.
    for record in records:
        if record.evaluated_by_oracle and record.oracle_call_index is None:
            if "oracle_call_index" not in record.provenance_missing_fields:
                record.provenance_missing_fields.append("oracle_call_index")
    return records


def compute_run_metrics(
    manifest_data: Dict[str, Any],
    run_id: str,
    task_id: str,
    condition: str,
    seed: int,
    oracle_budget: int = 100,
) -> Tuple[RunMetrics, List[CandidateRecord]]:
    """Compute metrics with explicit denominators and fixed-budget censoring."""

    if oracle_budget <= 0:
        raise ValueError("oracle_budget must be positive")
    source = _embedded_provenance(manifest_data)
    candidates = extract_candidates_from_manifest(manifest_data, run_id, task_id, condition, seed)
    manifest = source.get("manifest") if isinstance(source.get("manifest"), Mapping) else source
    if not isinstance(manifest, Mapping):
        manifest = {}

    proposals_generated = manifest.get("proposals_generated", source.get("proposals_generated", len(candidates)))
    try:
        proposals_generated = int(proposals_generated)
    except (TypeError, ValueError):
        proposals_generated = len(candidates)
    geometry_valid = sum(1 for c in candidates if c.geometry_valid)
    invalid_geometry = sum(1 for c in candidates if not c.geometry_valid)
    oracle_candidates = [c for c in candidates if c.evaluated_by_oracle]
    oracle_evals = len(oracle_candidates)
    counter_discrepancy = None
    if isinstance(manifest.get("oracle_evaluations"), int) and int(manifest["oracle_evaluations"]) != oracle_evals:
        counter_discrepancy = "oracle_evaluations_counter_mismatch"
    proposal_discrepancy = None
    if isinstance(manifest.get("proposals_generated"), int) and int(manifest["proposals_generated"]) != len(candidates):
        proposal_discrepancy = "proposals_generated_counter_mismatch"
    budget_discrepancy = "oracle_budget_exceeded" if oracle_evals > int(oracle_budget) else None
    successes = [
        c for c in oracle_candidates
        if c.oracle_success and c.predicted_energy_above_hull_ev_per_atom is not None
    ]
    failures = [c for c in oracle_candidates if c not in successes]
    failure_codes: Dict[str, int] = {}
    for c in failures:
        code = c.oracle_failure_code or "MISSING_ORACLE_RESULT"
        failure_codes[code] = failure_codes.get(code, 0) + 1
    missing_fields = sorted({f for c in candidates for f in c.provenance_missing_fields})

    # Oracle order is a global post-admission index, not candidate/list order.
    # Every evaluated candidate must have one unique contiguous index.  A gap
    # is also incomplete evidence: it can represent a failed/cache-hit oracle
    # attempt that was omitted from the persisted candidate list.
    explicit_indices = [c.oracle_call_index for c in oracle_candidates]
    order_complete = (
        all(index is not None for index in explicit_indices)
        and len(set(explicit_indices)) == len(explicit_indices)
        and sorted(int(index) for index in explicit_indices if index is not None)
        == list(range(1, len(oracle_candidates) + 1))
    )
    if not order_complete:
        missing_fields.append("incomplete_oracle_call_index_sequence")
    evaluated = sorted(
        [c for c in oracle_candidates if c.oracle_call_index is not None],
        key=lambda c: (int(c.oracle_call_index or 0), c.proposal_index, c.candidate_id),
    ) if order_complete else []
    first: Optional[int] = None
    if order_complete:
        for c in evaluated:
            e = c.predicted_energy_above_hull_ev_per_atom
            if e is not None and e <= 0.10:
                first = min(oracle_budget, int(c.oracle_call_index or oracle_budget))
                break
    reached = first is not None
    # An incomplete order is missing endpoint evidence, not a censored
    # endpoint observed at the budget.  This prevents downstream survival
    # analysis from treating an unsequenced success as a valid censoring event.
    endpoint_time = first if order_complete else None
    if order_complete and not reached:
        endpoint_time = oracle_budget

    def count(threshold: float) -> int:
        return sum(
            1 for c in oracle_candidates
            if c.predicted_energy_above_hull_ev_per_atom is not None
            and c.predicted_energy_above_hull_ev_per_atom <= threshold
        )

    c00, c03, c05, c10 = (count(x) for x in (0.00, 0.03, 0.05, 0.10))
    denom = oracle_evals
    fractions = tuple(x / denom if denom else 0.0 for x in (c00, c03, c05, c10))

    fixed_budgets = [10, 25, 50, 75, 100]
    best_at: Dict[int, Optional[float]] = {}
    trajectory: List[Tuple[int, float]] = []
    best = float("inf")
    for c in evaluated:
        e = c.predicted_energy_above_hull_ev_per_atom
        if e is not None and e < best:
            best = e
        if math.isfinite(best):
            trajectory.append((int(c.oracle_call_index or 0), best))
    for budget in fixed_budgets:
        values = [v for idx, v in trajectory if idx <= budget]
        best_at[budget] = min(values) if values else None
    best_overall = min((v for _, v in trajectory), default=None)
    auc: Optional[float] = None
    if trajectory:
        auc = 0.0
        prev_x = 0
        prev_y = trajectory[0][1]
        for x, y in trajectory:
            auc += max(0, x - prev_x) * prev_y
            prev_x, prev_y = x, y
        auc += max(0, oracle_budget - prev_x) * prev_y

    unique_formulas = {c.reduced_formula for c in oracle_candidates if c.reduced_formula}
    unique_anon = {c.anonymous_stoichiometry for c in oracle_candidates if c.anonymous_stoichiometry}
    unique_proto = {c.structural_prototype for c in oracle_candidates if c.structural_prototype}
    applied = manifest.get("memory_directives_applied", source.get("memory_directives_applied", []))
    rejected = manifest.get("memory_directives_rejected", source.get("memory_directives_rejected", []))
    unsupported = manifest.get("memory_directives_unsupported", source.get("memory_directives_unsupported", []))
    if counter_discrepancy:
        missing_fields.append(counter_discrepancy)
    if proposal_discrepancy:
        missing_fields.append(proposal_discrepancy)
    if budget_discrepancy:
        missing_fields.append(budget_discrepancy)

    metrics = RunMetrics(
        run_id=run_id,
        task_id=task_id,
        condition=condition,
        seed=seed,
        proposals_generated=proposals_generated,
        geometry_valid_count=geometry_valid,
        invalid_geometry_count=invalid_geometry,
        geometry_yield=geometry_valid / proposals_generated if proposals_generated else 0.0,
        oracle_evaluations=oracle_evals,
        oracle_budget=int(oracle_budget),
        oracle_success_count=len(successes),
        oracle_failure_count=len(failures),
        oracle_success_rate=len(successes) / oracle_evals if oracle_evals else 0.0,
        oracle_calls_to_first_candidate_at_or_below_0_10=endpoint_time,
        reached_0_10_threshold=reached,
        count_at_or_below_0_00=c00,
        count_at_or_below_0_03=c03,
        count_at_or_below_0_05=c05,
        count_at_or_below_0_10=c10,
        fraction_at_or_below_0_00=fractions[0],
        fraction_at_or_below_0_03=fractions[1],
        fraction_at_or_below_0_05=fractions[2],
        fraction_at_or_below_0_10=fractions[3],
        best_energy_above_hull_overall=best_overall,
        best_energy_at_fixed_oracle_budgets=best_at,
        area_under_best_curve=auc,
        unique_reduced_compositions_count=len(unique_formulas),
        unique_anonymous_stoichiometries_count=len(unique_anon),
        unique_prototypes_count=len(unique_proto),
        memory_directives_applied_count=len(applied) if isinstance(applied, list) else 0,
        memory_directives_rejected_count=len(rejected) if isinstance(rejected, list) else 0,
        memory_directives_unsupported_count=len(unsupported) if isinstance(unsupported, list) else 0,
        memory_prioritized_candidates_count=sum(
            1 for c in candidates if c.memory_priority_score is not None and c.memory_priority_score > 0
        ),
        total_candidates_recorded=len(candidates),
        missing_hull_energy_count=sum(1 for c in oracle_candidates if c.predicted_energy_above_hull_ev_per_atom is None),
        primary_endpoint_censored=not reached,
        provenance_complete=not missing_fields,
        provenance_missing_fields=missing_fields,
        oracle_failure_codes=failure_codes,
        oracle_order_source=(
            "oracle_events" if order_complete and source.get("oracle_events")
            else "structured_oracle_call_index" if order_complete
            else "incomplete_oracle_call_index"
        ),
        run_status=str(manifest.get("status", source.get("status", "unknown"))),
        shuffle_validation=(manifest.get("memory_shuffle_audit") if isinstance(manifest.get("memory_shuffle_audit"), Mapping) else None),
    )
    return metrics, candidates


__all__ = [
    "CandidateRecord",
    "RunMetrics",
    "ProvenanceIntegrityError",
    "extract_candidates_from_manifest",
    "compute_run_metrics",
]
