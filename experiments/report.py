"""Conservative, artifact-derived benchmark reporting.

Reports never assert a paper claim merely because a row exists. Every status is
derived from validated run/statistics/QE artifacts; missing or failed evidence
is rendered as ``Unavailable`` or ``Inconclusive``. JSON artifacts are retained
for machine reproducibility and PNGs are emitted when matplotlib is available.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from experiments.spec import (
    calculate_minimum_exact_test_sample_size,
    compute_sha256,
    CANONICAL_RESEARCH_PREFLIGHT_CHECKS,
    CONFIRMATORY_CONTROLS,
    CONFIRMATORY_METRICS,
    ExperimentSpec,
    ExperimentSpecError,
    MAX_EXACT_PERMUTATION_SAMPLE_SIZE,
)
from experiments.statistics import (
    CONFIRMATORY_FAMILY_NAME,
    HOLM_ADJUSTMENT_METHOD_NAME,
    PRIMARY_CENSORED_ENDPOINT_NAME,
    PRIMARY_CENSORED_METHOD_NAME,
    THRESHOLD_YIELD_ENDPOINT_NAME,
    THRESHOLD_YIELD_METHOD_NAME,
    holm_bonferroni_adjust,
)


@dataclass
class FrozenSpecValidation:
    valid: bool
    spec: Optional[ExperimentSpec] = None
    spec_hash: Optional[str] = None
    run_mode: str = "development"
    tasks: List[str] = field(default_factory=list)
    seeds: List[int] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    proposals_per_run: Optional[int] = None
    oracle_budget_per_run: Optional[int] = None
    source_task: Optional[str] = None
    expected_target_run_ids: List[str] = field(default_factory=list)
    expected_shuffled_run_ids: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)


def _is_hex_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(c in "0123456789abcdefABCDEF" for c in value)


def validate_frozen_spec(artifacts: Optional[Mapping[str, Any]]) -> FrozenSpecValidation:
    failures: List[str] = []
    if not isinstance(artifacts, Mapping):
        return FrozenSpecValidation(valid=False, failures=["missing_artifacts_mapping"])

    # 1. Check for frozen spec mapping under canonical report metadata field
    spec_dict = artifacts.get("spec")
    if spec_dict is None:
        spec_dict = artifacts.get("experiment_spec")
    if not isinstance(spec_dict, Mapping):
        failures.append("missing_frozen_spec")

    # 2. Check top-level spec_hash
    top_hash = artifacts.get("spec_hash")
    if not _is_hex_sha256(top_hash):
        failures.append("missing_or_invalid_top_level_spec_hash")

    # 3. Check dag
    dag_dict = artifacts.get("dag")
    if not isinstance(dag_dict, Mapping):
        failures.append("missing_dag")
        dag_spec_hash = None
    else:
        dag_spec_hash = dag_dict.get("spec_hash")
        if not _is_hex_sha256(dag_spec_hash):
            failures.append("missing_or_invalid_dag_spec_hash")

    # 4. Check preflight
    preflight_dict = artifacts.get("preflight")
    if not isinstance(preflight_dict, Mapping):
        failures.append("missing_preflight")
        preflight_spec_hash = None
    else:
        preflight_spec_hash = preflight_dict.get("spec_hash")
        if not _is_hex_sha256(preflight_spec_hash):
            failures.append("missing_or_invalid_preflight_spec_hash")

    if failures:
        return FrozenSpecValidation(valid=False, failures=failures)

    # 5. Recompute canonical specification digest
    try:
        recomputed_hash = compute_sha256(dict(spec_dict)).lower()
    except Exception as exc:
        failures.append(f"recompute_hash_failed:{exc}")
        return FrozenSpecValidation(valid=False, failures=failures)

    # 6. Parse specification through ExperimentSpec.from_dict
    try:
        parsed_spec = ExperimentSpec.from_dict(dict(spec_dict))
    except Exception as exc:
        failures.append(f"spec_parsing_failed:{exc}")
        return FrozenSpecValidation(valid=False, failures=failures)

    parsed_hash = parsed_spec.spec_hash.lower()

    # 7. Require exact equality among recomputed, parsed, top-level, dag, preflight hashes
    top_hash_str = str(top_hash).lower()
    dag_hash_str = str(dag_spec_hash).lower()
    preflight_hash_str = str(preflight_spec_hash).lower()

    if not (recomputed_hash == parsed_hash == top_hash_str == dag_hash_str == preflight_hash_str):
        failures.append("spec_hash_mismatch")

    # 8. Check DAG experiment_id
    if dag_dict and "experiment_id" in dag_dict:
        if str(dag_dict.get("experiment_id")) != parsed_spec.experiment_id:
            failures.append("dag_experiment_id_mismatch")

    # 9. Check run_mode agreement between parsed spec and preflight
    if preflight_dict and "run_mode" in preflight_dict:
        preflight_mode = str(preflight_dict.get("run_mode"))
        if parsed_spec.run_mode != preflight_mode:
            failures.append(f"run_mode_mismatch:{parsed_spec.run_mode}!={preflight_mode}")

    # 10. Check mathematical sample size feasibility and exact permutation boundary
    try:
        min_seeds = calculate_minimum_exact_test_sample_size(len(parsed_spec.target_tasks))
    except (ValueError, ExperimentSpecError) as exc:
        failures.append(f"statistically_infeasible_sample_size:{exc}")
        min_seeds = None

    unique_seeds_count = len(set(parsed_spec.master_seeds))
    if min_seeds is not None and unique_seeds_count < min_seeds:
        failures.append(f"statistically_infeasible_sample_size:{unique_seeds_count}<{min_seeds}")
    if parsed_spec.run_mode == "research" and unique_seeds_count > MAX_EXACT_PERMUTATION_SAMPLE_SIZE:
        failures.append(f"statistically_infeasible_sample_size:{unique_seeds_count}>{MAX_EXACT_PERMUTATION_SAMPLE_SIZE}")

    # 11. Derive expected context exclusively from parsed_spec
    tasks = sorted(str(t.task_id) for t in parsed_spec.target_tasks)
    seeds = sorted(int(s) for s in parsed_spec.master_seeds)
    conditions = list(parsed_spec.conditions)
    proposals = int(parsed_spec.proposals_per_run)
    oracle = int(parsed_spec.oracle_budget_per_run)
    source_task = str(parsed_spec.source_task.task_id)
    expected_target_run_ids = [
        f"run_{cond}_{task}_seed{seed}"
        for task in tasks
        for cond in conditions
        for seed in seeds
    ]
    expected_shuffled_run_ids = [
        f"run_shuffled_memory_control_{task}_seed{seed}"
        for task in tasks
        for seed in seeds
    ]

    # 11. Check that caller-supplied metadata does not subset or mismatch
    for task_key in ("expected_target_task_ids", "expected_target_tasks", "expected_tasks"):
        if task_key in artifacts:
            raw_tasks = artifacts[task_key]
            if raw_tasks is None:
                failures.append("caller_task_subset_or_mismatch")
            else:
                passed_tasks = sorted(str(x.get("task_id") if isinstance(x, Mapping) else x) for x in _as_list(raw_tasks) if x is not None)
                if passed_tasks != tasks:
                    failures.append("caller_task_subset_or_mismatch")

    for seed_key in ("expected_seeds", "master_seeds"):
        if seed_key in artifacts:
            raw_seeds = artifacts[seed_key]
            if raw_seeds is None:
                failures.append("caller_seed_subset_or_mismatch")
            else:
                passed_seeds = []
                for item in _as_list(raw_seeds):
                    try:
                        if not isinstance(item, bool):
                            passed_seeds.append(int(item))
                    except (TypeError, ValueError):
                        continue
                if sorted(passed_seeds) != seeds:
                    failures.append("caller_seed_subset_or_mismatch")

    for cond_key in ("expected_conditions", "conditions"):
        if cond_key in artifacts:
            raw_conditions = artifacts[cond_key]
            if raw_conditions is None:
                failures.append("caller_condition_mismatch")
            else:
                passed_conds = list(dict.fromkeys(str(item) for item in _as_list(raw_conditions) if item is not None))
                if passed_conds != conditions:
                    failures.append("caller_condition_mismatch")

    if "expected_target_run_ids" in artifacts:
        raw_run_ids = artifacts["expected_target_run_ids"]
        if raw_run_ids is None:
            failures.append("caller_run_ids_mismatch")
        elif sorted(str(r) for r in _as_list(raw_run_ids)) != sorted(expected_target_run_ids):
            failures.append("caller_run_ids_mismatch")

    if failures:
        return FrozenSpecValidation(valid=False, failures=failures)

    return FrozenSpecValidation(
        valid=True,
        spec=parsed_spec,
        spec_hash=recomputed_hash,
        run_mode=parsed_spec.run_mode,
        tasks=tasks,
        seeds=seeds,
        conditions=conditions,
        proposals_per_run=proposals,
        oracle_budget_per_run=oracle,
        source_task=source_task,
        expected_target_run_ids=expected_target_run_ids,
        expected_shuffled_run_ids=expected_shuffled_run_ids,
        failures=[],
    )


_QE_RESULT_FIELDS = (
    "candidate_id", "target_task", "condition", "seed", "formula", "num_atoms",
    "candidate_status", "candidate_energy_per_atom_ev", "competing_phases_count",
    "competing_phases_all_converged", "chgnet_predicted_hull_distance_ev_per_atom",
    "dft_local_decomposition_margin_ev_per_atom", "sign_agreement",
    "margin_difference_ev_per_atom", "reaction_equation", "reaction_balanced",
    "status", "candidate_input_hash", "candidate_result_hash", "candidate_output_hash",
    "qe_executable", "qe_executable_version", "sssp_manifest_sha256",
    "participating_phase_results",
)


def _dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, Mapping):
        return dict(obj)
    if hasattr(obj, "to_dict"):
        return dict(obj.to_dict())
    return dict(getattr(obj, "__dict__", {}))


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _is_true(value: Any) -> bool:
    """Accept only explicit boolean true values from JSON/CSV artifacts."""
    return value is True or (isinstance(value, str) and value.strip().lower() == "true")


def _nonempty_hash(value: Any) -> bool:
    """Return true for a present, non-placeholder provenance digest."""
    if not isinstance(value, str):
        return False
    text = value.strip().lower()
    return bool(text) and text not in {"none", "null", "missing", "unknown"}


def _sha256(value: Any) -> bool:
    """Require a real SHA-256 digest for publication-facing provenance."""
    if not isinstance(value, str):
        return False
    text = value.strip().lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Mapping):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return [value]


def _file_sha256(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, ValueError):
        return None


def _qe_csv_row(row: Mapping[str, Any]) -> Dict[str, str]:
    """Reproduce the exact scalar representation emitted by QEAuditRunner."""
    normalized = {
        key: "" if row.get(key) is None else str(row.get(key, ""))
        for key in _QE_RESULT_FIELDS
    }
    phases = row.get("participating_phase_results", [])
    if isinstance(phases, str):
        try:
            phases = json.loads(phases)
        except (TypeError, ValueError):
            phases = None
    try:
        normalized["participating_phase_results"] = json.dumps(
            phases, sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
    except (TypeError, ValueError):
        normalized["participating_phase_results"] = "<invalid>"
    return normalized


def _qe_csv_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    return [_qe_csv_row(row) for row in rows]


class ReportGenerator:
    """Generate plots, tables, and conservative paper-facing artifacts."""

    def __init__(self, experiment_root: Path):
        self.root = Path(experiment_root)
        self.figures_dir = self.root / "figures"
        self.tables_dir = self.root / "tables"
        self.paper_dir = self.root / "paper"
        for path in (self.figures_dir, self.tables_dir, self.paper_dir):
            path.mkdir(parents=True, exist_ok=True)

    def generate_all_reports(
        self,
        runs_metrics: Sequence[Any],
        statistical_results: Sequence[Any],
        qe_results: Sequence[Any],
        validated_artifacts: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, str]:
        runs = [_dict(r) for r in runs_metrics]
        stats = [_dict(r) for r in statistical_results]
        qe = [_dict(r) for r in qe_results]
        generated = {
            "fig1_best_so_far": str(self._generate_best_so_far_curve(runs)),
            "fig2_time_to_threshold": str(self._generate_time_to_threshold_survival(runs)),
            "fig3_paired_effects": str(self._generate_paired_effect_forest_plot(stats)),
            "fig4_failure_breakdown": str(self._generate_failure_breakdown_plot(runs)),
            "fig5_memory_applicability": str(self._generate_memory_applicability_plot(runs)),
            "fig6_shuffled_validation": str(self._generate_shuffled_control_validation_plot(runs)),
            "fig7_qe_vs_chgnet": str(self._generate_qe_vs_chgnet_parity_plot(qe)),
            "fig8_threshold_sensitivity": str(self._generate_threshold_sensitivity_summary(runs)),
        }
        matrix, claim_status = self._generate_claim_evidence_matrix(stats, qe, runs, validated_artifacts)
        checklist = self._generate_reproducibility_checklist(stats, qe, runs, validated_artifacts)
        generated["claim_matrix"] = str(matrix)
        generated["checklist"] = str(checklist)
        (self.paper_dir / "claim_status.json").write_text(json.dumps(claim_status, indent=2, sort_keys=True), encoding="utf-8")
        return generated

    def _write_json(self, name: str, data: Any, *, png_data: Optional[List[tuple]] = None) -> Path:
        path = self.figures_dir / name
        path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
        if png_data:
            self._try_png(path.with_suffix(".png"), png_data)
        return path

    @staticmethod
    def _try_png(path: Path, points: List[tuple]) -> None:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6.5, 4.0))
            for label, xs, ys in points:
                ax.plot(xs, ys, marker="o", label=label)
            if len(points) > 1:
                ax.legend(frameon=False)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(path, dpi=180)
            plt.close(fig)
        except Exception:
            # PNG is opportunistic; JSON remains the canonical artifact.
            return

    def _generate_best_so_far_curve(self, runs: Sequence[Mapping[str, Any]]) -> Path:
        data: Dict[str, List[Dict[str, Any]]] = {}
        points: Dict[str, tuple] = {}
        for rm in runs:
            condition = str(rm.get("condition", "unknown"))
            budgets = rm.get("best_energy_at_fixed_oracle_budgets", {}) or {}
            row = {"task": rm.get("task_id"), "seed": rm.get("seed"), "best_at_budgets": budgets, "best_overall": rm.get("best_energy_above_hull_overall"), "status": rm.get("run_status", "unknown")}
            data.setdefault(condition, []).append(row)
            xs, ys = [], []
            for x, y in sorted(((int(k), v) for k, v in budgets.items() if str(k).isdigit() and _finite(v)), key=lambda z: z[0]):
                xs.append(x); ys.append(float(y))
            if xs:
                points[condition] = (condition, xs, ys)
        return self._write_json("fig1_best_so_far_vs_oracle_calls.json", data, png_data=list(points.values()))

    def _generate_time_to_threshold_survival(self, runs: Sequence[Mapping[str, Any]]) -> Path:
        data: Dict[str, List[Dict[str, Any]]] = {}
        for rm in runs:
            reached = bool(rm.get("reached_0_10_threshold", False))
            budget = rm.get("oracle_budget")
            time = rm.get("oracle_calls_to_first_candidate_at_or_below_0_10")
            data.setdefault(str(rm.get("condition", "unknown")), []).append({
                "task": rm.get("task_id"), "seed": rm.get("seed"),
                "oracle_calls_to_threshold": time, "event_observed": reached,
                "censored": not reached, "budget": budget,
            })
        return self._write_json("fig2_time_to_threshold_censored.json", data)

    def _generate_paired_effect_forest_plot(self, stats: Sequence[Mapping[str, Any]]) -> Path:
        points = []
        for row in stats:
            if row.get("status") == "ANALYZED" and _finite(row.get("mean_difference")):
                points.append({"label": f"{row.get('target_task')}:{row.get('condition_b', '')}", "effect": row.get("mean_difference"), "low": row.get("ci_95_lower"), "high": row.get("ci_95_upper")})
        return self._write_json("fig3_paired_effect_sizes.json", points)

    def _generate_failure_breakdown_plot(self, runs: Sequence[Mapping[str, Any]]) -> Path:
        rows = [{"run_id": rm.get("run_id"), "task": rm.get("task_id"), "condition": rm.get("condition"), "seed": rm.get("seed"), "proposals": rm.get("proposals_generated", 0), "invalid_geometry": rm.get("invalid_geometry_count", 0), "geometry_yield": rm.get("geometry_yield", 0.0), "oracle_evaluations": rm.get("oracle_evaluations", 0), "oracle_failures": rm.get("oracle_failure_count", 0), "failure_codes": rm.get("oracle_failure_codes", {})} for rm in runs]
        return self._write_json("fig4_geometry_and_oracle_failures.json", rows)

    def _generate_memory_applicability_plot(self, runs: Sequence[Mapping[str, Any]]) -> Path:
        rows = [{"run_id": rm.get("run_id"), "task": rm.get("task_id"), "condition": rm.get("condition"), "seed": rm.get("seed"), "applied": rm.get("memory_directives_applied_count", 0), "rejected": rm.get("memory_directives_rejected_count", 0), "unsupported": rm.get("memory_directives_unsupported_count", 0), "prioritized_candidates": rm.get("memory_prioritized_candidates_count", 0)} for rm in runs]
        return self._write_json("fig5_memory_directives_breakdown.json", rows)

    def _generate_shuffled_control_validation_plot(self, runs: Sequence[Mapping[str, Any]]) -> Path:
        shuffled = [rm for rm in runs if rm.get("condition") == "shuffled_memory_control"]
        evidence = [rm.get("shuffle_validation") for rm in shuffled if isinstance(rm.get("shuffle_validation"), Mapping)]
        def valid_audit(audit: Mapping[str, Any]) -> bool:
            if not _is_true(audit.get("valid", False)):
                return False
            if "fixed_points" in audit:
                try:
                    return int(audit["fixed_points"]) == 0
                except (TypeError, ValueError):
                    return False
            return _is_true(audit.get("no_fixed_points", audit.get("is_derangement", audit.get("derangement", False))))
        valid = bool(evidence) and all(valid_audit(x) for x in evidence)
        status = "VALID_DERANGEMENT" if valid else ("INCONCLUSIVE" if shuffled else "UNAVAILABLE")
        return self._write_json("fig6_shuffled_control_validation.json", {"shuffled_runs_count": len(shuffled), "scientific_decision_support": False, "status": status, "evidence_present": bool(evidence), "runs": [rm.get("run_id") for rm in shuffled]})

    def _generate_qe_vs_chgnet_parity_plot(self, qe: Sequence[Mapping[str, Any]]) -> Path:
        validated = [r for r in qe if r.get("status") in {"VALIDATED", "CONVERGED"} and _finite(r.get("dft_local_decomposition_margin_ev_per_atom")) and _finite(r.get("chgnet_predicted_hull_distance_ev_per_atom"))]
        return self._write_json("fig7_chgnet_vs_qe_local_decomposition.json", list(qe), png_data=[("CHGNet vs QE", [float(r["chgnet_predicted_hull_distance_ev_per_atom"]) for r in validated], [float(r["dft_local_decomposition_margin_ev_per_atom"]) for r in validated])] if validated else None)

    def _generate_threshold_sensitivity_summary(self, runs: Sequence[Mapping[str, Any]]) -> Path:
        summary: Dict[str, Dict[str, Any]] = {}
        for rm in runs:
            key = f"{rm.get('task_id')}_{rm.get('condition')}"
            row = summary.setdefault(key, {"task": rm.get("task_id"), "condition": rm.get("condition"), "fractions_0_00": [], "fractions_0_03": [], "fractions_0_05": [], "fractions_0_10": []})
            for key_name in ("0_00", "0_03", "0_05", "0_10"):
                row[f"fractions_{key_name}"].append(rm.get(f"fraction_at_or_below_{key_name}"))
        path = self.tables_dir / "table_threshold_sensitivity.json"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
        # A publication table with one row per run is also emitted.
        with (self.tables_dir / "table_threshold_sensitivity.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f); writer.writerow(["task", "condition", "fraction_0_00", "fraction_0_03", "fraction_0_05", "fraction_0_10"])
            for row in summary.values():
                n = max((len(row[k]) for k in row if k.startswith("fractions_")), default=0)
                for i in range(n):
                    writer.writerow([row["task"], row["condition"], *[row[f"fractions_{x}"][i] for x in ("0_00", "0_03", "0_05", "0_10")]])
        return path

    @staticmethod
    def _stat_ok(stats: Sequence[Mapping[str, Any]], metric: str, control: str, task: Optional[str] = None) -> bool:
        rows = [r for r in stats if r.get("metric_name") == metric and r.get("condition_b") == control and (task is None or r.get("target_task") == task)]
        def usable(row: Mapping[str, Any]) -> bool:
            try:
                n = int(row.get("sample_size_n", 0) or 0)
                missing = int(row.get("missing_count", 0) or 0)
            except (TypeError, ValueError):
                return False
            return row.get("status") == "ANALYZED" and n > 0 and missing == 0 and _finite(row.get("mean_difference"))
        return any(usable(row) for row in rows)

    @staticmethod
    def _research_preflight(artifacts: Optional[Mapping[str, Any]]) -> tuple[str, bool, Dict[str, Any]]:
        """Classify the preflight artifact without trusting partial evidence."""
        preflight = _dict(artifacts.get("preflight")) if isinstance(artifacts, Mapping) and isinstance(artifacts.get("preflight"), Mapping) else {}
        if not preflight:
            return "missing", False, preflight
        required = preflight.get("required_checks")
        checks = preflight.get("checks")
        # The CLI's canonical preflight schema stores names in
        # ``required_checks`` and their results in ``checks``.  Accept the
        # older compact {name: true} form only when it is itself explicit; a
        # bare PASSED marker is never enough.
        if isinstance(required, Mapping):
            required_results = dict(required)
        elif isinstance(required, (list, tuple, set)) and required and isinstance(checks, Mapping):
            required_results = {str(name): checks.get(name) for name in required if isinstance(name, str)}
            if len(required_results) != len(required):
                required_results = {}
        else:
            # Metadata propagated by callers may expose the names/results
            # outside the nested preflight object.
            outer = artifacts if isinstance(artifacts, Mapping) else {}
            names = preflight.get("required_preflight_checks", outer.get("required_preflight_checks"))
            results = preflight.get("required_preflight_results", outer.get("required_preflight_results"))
            if names is None:
                names = outer.get("required_checks")
            if results is None:
                results = outer.get("checks")
            if isinstance(names, (list, tuple, set)) and names and isinstance(results, Mapping):
                required_results = {str(name): results.get(name) for name in names if isinstance(name, str)}
                if len(required_results) != len(names):
                    required_results = {}
            else:
                required_results = {}
        validation = validate_frozen_spec(artifacts)
        passed = (
            preflight.get("status") == "PASSED"
            and preflight.get("run_mode") == "research"
            and validation.valid
            and validation.spec is not None
            and validation.spec.run_mode == "research"
            and bool(required_results)
            and set(CANONICAL_RESEARCH_PREFLIGHT_CHECKS).issubset(set(required_results.keys()))
            and all(value is True for value in required_results.values())
        )
        if preflight.get("run_mode") != "research" or (validation.valid and validation.spec is not None and validation.spec.run_mode != "research"):
            if preflight.get("run_mode") == "research" or (validation.valid and validation.spec is not None and validation.spec.run_mode == "research"):
                return "invalid_research", False, preflight
            return "demo", False, preflight
        # A failed or malformed research preflight is an ineligible research
        # artifact, not a development/demo run.  This distinction keeps the
        # UI honest while remaining fail-closed for claims.
        return ("research" if passed else "invalid_research"), passed, preflight

    @staticmethod
    def _expected_context(artifacts: Optional[Mapping[str, Any]]) -> tuple[List[str], List[int], List[str], Dict[str, Any]]:
        # Normalize nested run_set / experiment / metadata
        source: Dict[str, Any] = {}
        if isinstance(artifacts, Mapping):
            for nested_name in ("run_set", "experiment", "metadata"):
                nested = artifacts.get(nested_name)
                if isinstance(nested, Mapping):
                    source.update(nested)
            source.update(dict(artifacts))

        validation = validate_frozen_spec(artifacts)
        if validation.valid and validation.spec is not None:
            source.update({
                "spec_valid": True,
                "spec": validation.spec.to_dict(),
                "spec_hash": validation.spec_hash,
                "expected_target_task_ids": validation.tasks,
                "expected_target_tasks": validation.tasks,
                "expected_seeds": validation.seeds,
                "expected_conditions": validation.conditions,
                "expected_target_run_ids": validation.expected_target_run_ids,
                "expected_target_run_count": len(validation.expected_target_run_ids),
                "proposal_budget_per_run": validation.proposals_per_run,
                "oracle_budget_per_run": validation.oracle_budget_per_run,
                "expected_source_task": validation.source_task,
            })
            return validation.tasks, validation.seeds, validation.conditions, source

        # If frozen spec validation failed when spec or spec_hash was supplied
        spec_present = isinstance(artifacts, Mapping) and (
            artifacts.get("spec") is not None
            or artifacts.get("experiment_spec") is not None
            or artifacts.get("spec_hash") is not None
        )
        if spec_present:
            return [], [], [], {**source, "spec_valid": False, "spec_failures": validation.failures, "spec_integrity_failed": True}

        # Legacy fallback for demo / developer helpers without frozen spec
        def values(*keys: str) -> Any:
            for key in keys:
                if key in source and source[key] is not None:
                    return source[key]
            return None

        raw_tasks = values("expected_target_task_ids", "expected_target_tasks", "expected_tasks")
        tasks = []
        for item in _as_list(raw_tasks):
            if isinstance(item, Mapping):
                item = item.get("task_id")
            if item is not None:
                tasks.append(str(item))
        raw_seeds = values("expected_seeds", "master_seeds")
        seeds = []
        for item in _as_list(raw_seeds):
            try:
                if not isinstance(item, bool):
                    seeds.append(int(item))
            except (TypeError, ValueError):
                continue
        raw_conditions = values("expected_conditions", "conditions")
        conditions = list(dict.fromkeys(str(item) for item in _as_list(raw_conditions) if item is not None))
        if not conditions:
            conditions = [
                "structured_provenance_memory", "adaptive_no_memory", "text_summary_memory",
                "shuffled_memory_control", "random_mattergen",
            ]
        return sorted(set(tasks)), sorted(set(seeds)), conditions, {**source, "spec_valid": False}

    @staticmethod
    def _target_runs_complete(
        runs: Sequence[Mapping[str, Any]], artifacts: Optional[Mapping[str, Any]],
    ) -> tuple[bool, bool, List[Mapping[str, Any]]]:
        """Return (complete, has_evidence, target_rows) for the preregistered set."""
        tasks, seeds, conditions, source = ReportGenerator._expected_context(artifacts)
        if not tasks or not seeds or not conditions:
            return False, bool(runs), []
        target_rows = [r for r in runs if str(r.get("task_id")) in tasks and str(r.get("condition")) in conditions]
        source_task = source.get("expected_source_task", source.get("source_task"))
        source_task = str(source_task) if source_task is not None else None
        # A target-set check must not make an unexpected non-source run vanish
        # by filtering it out.  Source rows are allowed here because the CLI
        # aggregates the source campaign alongside target campaigns; every
        # other row belongs to the target accounting boundary and must match
        # the preregistered set below.
        non_source_rows = [
            row for row in runs
            if source_task is None or str(row.get("task_id")) != source_task
        ]
        expected_ids_raw = source.get("expected_target_run_ids")
        expected_ids = {str(x) for x in _as_list(expected_ids_raw) if x is not None}
        if expected_ids:
            observed_ids = [str(r.get("run_id")) for r in non_source_rows]
            declared_count = source.get("expected_target_run_count")
            count_ok = True
            if declared_count is not None:
                try:
                    count_ok = int(declared_count) == len(expected_ids)
                except (TypeError, ValueError):
                    count_ok = False
            exact_set = count_ok and len(observed_ids) == len(expected_ids) and set(observed_ids) == expected_ids
        else:
            expected_keys = {(task, condition, seed) for task in tasks for condition in conditions for seed in seeds}
            observed_keys = set()
            if len(non_source_rows) != len(expected_keys):
                return False, bool(target_rows), target_rows
            for row in non_source_rows:
                try:
                    row_seed = int(row.get("seed"))
                except (TypeError, ValueError):
                    row_seed = None
                observed_keys.add((str(row.get("task_id")), str(row.get("condition")), row_seed))
            exact_set = observed_keys == expected_keys
        if not exact_set:
            return False, bool(target_rows), target_rows

        def accounting_ok(row: Mapping[str, Any]) -> bool:
            try:
                oracle = int(row.get("oracle_evaluations"))
                budget = int(row.get("oracle_budget"))
                proposals = int(row.get("proposals_generated"))
                geometry_valid = int(row.get("geometry_valid_count"))
                invalid_geometry = int(row.get("invalid_geometry_count"))
                total_candidates = int(row.get("total_candidates_recorded"))
                oracle_success = int(row.get("oracle_success_count"))
                oracle_failure = int(row.get("oracle_failure_count"))
            except (TypeError, ValueError):
                return False
            expected_proposals = source.get("proposal_budget_per_run", source.get("expected_proposal_budget"))
            expected_oracle = source.get("oracle_budget_per_run", source.get("expected_oracle_budget"))
            if expected_proposals is not None:
                try:
                    if proposals != int(expected_proposals):
                        return False
                except (TypeError, ValueError):
                    return False
            if expected_oracle is not None:
                try:
                    if budget != int(expected_oracle):
                        return False
                except (TypeError, ValueError):
                    return False
            return (
                _is_true(row.get("provenance_complete"))
                and oracle >= 0 and budget >= 0 and oracle <= budget
                and proposals >= 0 and geometry_valid >= 0 and invalid_geometry >= 0
                and total_candidates == proposals
                and geometry_valid + invalid_geometry == total_candidates
                and oracle_success >= 0 and oracle_failure >= 0
                and oracle_success + oracle_failure == oracle
                and str(row.get("run_status", "")).lower() in {"completed", "success", "succeeded"}
            )

        return bool(target_rows) and all(accounting_ok(row) for row in target_rows), True, target_rows

    @staticmethod
    def _shuffle_audit_complete(
        runs: Sequence[Mapping[str, Any]], artifacts: Optional[Mapping[str, Any]],
    ) -> bool:
        tasks, seeds, _, source = ReportGenerator._expected_context(artifacts)
        if not tasks or not seeds:
            return False
        shuffled = [r for r in runs if str(r.get("condition")) == "shuffled_memory_control"]
        expected_ids_raw = source.get("expected_shuffled_run_ids")
        expected_ids = {str(x) for x in _as_list(expected_ids_raw) if x is not None}
        if expected_ids:
            if len(shuffled) != len(expected_ids) or {str(r.get("run_id")) for r in shuffled} != expected_ids:
                return False
        else:
            expected_keys = {(task, seed) for task in tasks for seed in seeds}
            observed_keys = set()
            for row in shuffled:
                try:
                    row_seed = int(row.get("seed"))
                except (TypeError, ValueError):
                    row_seed = None
                observed_keys.add((str(row.get("task_id")), row_seed))
            if len(shuffled) != len(expected_keys) or observed_keys != expected_keys:
                return False

        def valid(row: Mapping[str, Any]) -> bool:
            audit = row.get("shuffle_validation")
            if not isinstance(audit, Mapping) or not _is_true(audit.get("valid", False)):
                return False
            if "fixed_points" in audit:
                try:
                    if int(audit["fixed_points"]) != 0:
                        return False
                except (TypeError, ValueError):
                    return False
            elif not _is_true(audit.get("no_fixed_points", audit.get("is_derangement", audit.get("derangement", False)))):
                return False
            return (
                _is_true(row.get("provenance_complete"))
                and str(row.get("run_status", "")).lower() in {"completed", "success", "succeeded"}
            )

        return all(valid(row) for row in shuffled)

    def _qe_manifest_integrity(self, qe_meta: Mapping[str, Any]) -> bool:
        """Verify the immutable QE artifact bindings before reading claims."""
        manifest = qe_meta.get("qe_audit")
        if not isinstance(manifest, Mapping):
            return False
        artifacts = manifest.get("artifacts")
        provenance = manifest.get("result_provenance")
        if not isinstance(artifacts, Mapping) or not isinstance(provenance, list):
            return False
        qe_dir = self.root / "qe_audit"
        for filename in ("selection.json", "results.csv"):
            expected = artifacts.get(filename)
            actual = _file_sha256(qe_dir / filename)
            if not _sha256(expected) or actual != str(expected).lower():
                return False
        try:
            selection = json.loads((qe_dir / "selection.json").read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return False
        if not isinstance(selection, Mapping) or not isinstance(selection.get("candidates"), list):
            return False
        manifest_selection_count = manifest.get("selection_count")
        if manifest_selection_count is None:
            return False
        try:
            if int(manifest_selection_count) != len(selection["candidates"]):
                return False
        except (TypeError, ValueError):
            return False
        if selection.get("insufficiency") != manifest.get("selection_insufficiency"):
            return False
        try:
            disk_rows = list(csv.DictReader((qe_dir / "results.csv").open("r", encoding="utf-8", newline="")))
        except (OSError, csv.Error, UnicodeError):
            return False
        if _qe_csv_rows(provenance) != _qe_csv_rows(disk_rows):
            return False
        try:
            canonical = json.dumps(
                provenance, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError):
            return False
        expected_canonical = artifacts.get("canonical_results_sha256")
        return _sha256(expected_canonical) and hashlib.sha256(canonical).hexdigest() == str(expected_canonical).lower()

    def _qe_evidence_complete(self, qe: Sequence[Mapping[str, Any]], artifacts: Optional[Mapping[str, Any]]) -> bool:
        source = artifacts if isinstance(artifacts, Mapping) else {}
        qe_meta: Dict[str, Any] = {}
        for key in ("qe_audit", "qe_selection", "qe_metadata"):
            nested = source.get(key)
            if isinstance(nested, Mapping):
                qe_meta.update(nested)
        qe_meta.update(source)
        expected = qe_meta.get("expected_qe_candidate_count", qe_meta.get("qe_expected_candidate_count"))
        try:
            expected_count = int(expected)
        except (TypeError, ValueError):
            return False
        if expected_count <= 0:
            return False
        # Selection and audit manifests are part of the evidence boundary.
        # If present, their count must agree exactly with both the requested
        # count and the emitted result rows; extra rows are not silently
        # treated as harmless surplus calculations.
        selected_count = qe_meta.get("qe_selection_count", qe_meta.get("selection_count"))
        if selected_count is not None:
            try:
                if int(selected_count) != expected_count or len(qe) != int(selected_count):
                    return False
            except (TypeError, ValueError):
                return False
        elif len(qe) != expected_count:
            return False
        insufficiency = qe_meta.get("qe_selection_insufficiency", qe_meta.get("selection_insufficiency"))
        if insufficiency not in (None, "", False):
            return False

        # A report cannot infer production provenance from populated-looking
        # CSV columns.  Require the audit manifest's explicit research/nonmock
        # declaration whenever a manifest was supplied; absent metadata is
        # deliberately insufficient for a paper claim.
        audit_run_mode = qe_meta.get("run_mode")
        audit_mock = qe_meta.get("mock_execution")
        if audit_run_mode != "research" or audit_mock is not False:
            return False
        if not self._qe_manifest_integrity(qe_meta):
            return False
        try:
            disk_rows = list(csv.DictReader((self.root / "qe_audit" / "results.csv").open("r", encoding="utf-8", newline="")))
        except (OSError, csv.Error, UnicodeError):
            return False
        if _qe_csv_rows(qe) != _qe_csv_rows(disk_rows):
            return False

        required_hashes = (
            "candidate_input_hash", "candidate_result_hash", "candidate_output_hash",
            "qe_executable", "qe_executable_version", "sssp_manifest_sha256",
        )
        phase_hashes = (
            "input_hash", "result_hash", "output_hash", "executable",
            "executable_version", "sssp_manifest_sha256",
        )
        for row in qe:
            if (
                row.get("status") != "VALIDATED"
                or row.get("candidate_status") != "CONVERGED"
                or not _is_true(row.get("reaction_balanced"))
                or not _is_true(row.get("sign_agreement"))
            ):
                return False
            if (
                not _finite(row.get("chgnet_predicted_hull_distance_ev_per_atom"))
                or not _finite(row.get("dft_local_decomposition_margin_ev_per_atom"))
                or not all(_sha256(row.get(key)) for key in required_hashes[:3])
                or not _nonempty_hash(row.get("qe_executable"))
                or not _nonempty_hash(row.get("qe_executable_version"))
                or not _sha256(row.get("sssp_manifest_sha256"))
            ):
                return False
            raw_phases = row.get("participating_phase_results")
            if isinstance(raw_phases, str):
                try:
                    raw_phases = json.loads(raw_phases)
                except (TypeError, ValueError):
                    return False
            if not isinstance(raw_phases, list) or not raw_phases:
                return False
            for phase in raw_phases:
                if not isinstance(phase, Mapping) or phase.get("status") != "CONVERGED":
                    return False
                if not all(_sha256(phase.get(key)) for key in phase_hashes[:3]):
                    return False
                if not _nonempty_hash(phase.get("executable")) or not _nonempty_hash(phase.get("executable_version")):
                    return False
                if not _sha256(phase.get("sssp_manifest_sha256")):
                    return False
            # A candidate and each participating phase must agree on the same
            # executable/SSSP provenance; otherwise the local decomposition is
            # not an auditable comparison.
            if any(
                phase.get("executable") != row.get("qe_executable")
                or phase.get("executable_version") != row.get("qe_executable_version")
                or phase.get("sssp_manifest_sha256") != row.get("sssp_manifest_sha256")
                for phase in raw_phases
            ):
                return False
        return True

    def _statistical_evidence_complete(
        self,
        stats: Sequence[Mapping[str, Any]],
        artifacts: Optional[Mapping[str, Any]],
    ) -> bool:
        """Bind statistical rows to cryptographically verified analysis artifacts on disk."""
        validation = validate_frozen_spec(artifacts)
        if not validation.valid or validation.spec is None:
            return False

        root_dir = getattr(self, "root", getattr(self, "experiment_root", Path(".")))
        stats_dir = root_dir / "statistics"
        stats_manifest_path = stats_dir / "analysis_manifest.json"
        effects_file = stats_dir / "effects.csv"
        results_file = stats_dir / "results.json"

        # 1. On-disk directory and all three artifact files MUST exist on disk
        if not (stats_dir.is_dir() and stats_manifest_path.is_file() and effects_file.is_file() and results_file.is_file()):
            return False

        # 2. Read on-disk manifest directly from disk (never bypass with in-memory data)
        try:
            on_disk_manifest = json.loads(stats_manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False

        if not isinstance(on_disk_manifest, Mapping):
            return False

        # If caller supplied an in-memory manifest, require exact equality with on-disk manifest
        if isinstance(artifacts, Mapping):
            for key in ("analysis_manifest", "statistics", "analysis"):
                val = artifacts.get(key)
                if isinstance(val, Mapping) and val != on_disk_manifest:
                    return False

        # 3. Spec hash and experiment ID matching in manifest
        if str(on_disk_manifest.get("spec_hash", "")).lower() != validation.spec_hash.lower():
            return False
        if str(on_disk_manifest.get("experiment_id", "")) != validation.spec.experiment_id:
            return False

        artifacts_map = on_disk_manifest.get("artifacts")
        if not isinstance(artifacts_map, Mapping):
            return False

        # Check for unexpected artifact keys or path traversal
        expected_keys = {"effects.csv", "results.json", "canonical_results_sha256"}
        if set(artifacts_map.keys()) != expected_keys:
            return False
        for art_key in artifacts_map:
            if not isinstance(art_key, str) or Path(art_key).name != art_key or ".." in art_key:
                return False

        effects_hash = artifacts_map.get("effects.csv")
        results_hash = artifacts_map.get("results.json")
        canonical_results_hash = artifacts_map.get("canonical_results_sha256")

        if not (_is_hex_sha256(effects_hash) and _is_hex_sha256(results_hash) and _is_hex_sha256(canonical_results_hash)):
            return False

        # 4. Require results_hash and canonical_results_hash to be identical
        if results_hash.lower() != canonical_results_hash.lower():
            return False

        # 5. Check actual file sha256 digests on disk
        actual_effects_sha256 = _file_sha256(effects_file)
        actual_results_sha256 = _file_sha256(results_file)
        if not actual_effects_sha256 or actual_effects_sha256.lower() != effects_hash.lower():
            return False
        if not actual_results_sha256 or actual_results_sha256.lower() != results_hash.lower():
            return False

        # 6. Parse results.json on disk and recompute its canonical SHA-256 digest
        try:
            on_disk_results = json.loads(results_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if not isinstance(on_disk_results, list) or not on_disk_results:
            return False

        canonical_disk_bytes = json.dumps(on_disk_results, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        canonical_disk_sha256 = hashlib.sha256(canonical_disk_bytes).hexdigest().lower()
        if canonical_disk_sha256 != canonical_results_hash.lower():
            return False

        # 7. Check manifest["results"] if present
        manifest_results = on_disk_manifest.get("results")
        if manifest_results is not None:
            if not isinstance(manifest_results, list) or manifest_results != on_disk_results:
                return False

        # 8. In-memory stats list must match on-disk results.json exactly
        stats_list = [dict(s) for s in stats]
        canonical_in_mem_bytes = json.dumps(stats_list, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        canonical_in_mem_sha256 = hashlib.sha256(canonical_in_mem_bytes).hexdigest().lower()
        if canonical_in_mem_sha256 != canonical_results_hash.lower():
            return False
        if stats_list != on_disk_results:
            return False

        # 9. Derive expected canonical effects.csv bytes from on_disk_results and compare byte-for-byte
        csv_fields = [
            "analysis_version", "target_task", "metric_name", "condition_a", "condition_b",
            "comparison_type", "sample_size_n", "missing_count", "mean_a", "mean_b",
            "mean_difference", "median_difference", "ci_95_lower", "ci_95_upper",
            "p_value_raw", "p_value_adjusted", "adjustment_method", "method", "status",
        ]
        csv_buffer = io.StringIO()
        csv_writer = csv.DictWriter(csv_buffer, fieldnames=csv_fields, lineterminator="\n")
        csv_writer.writeheader()
        for r in on_disk_results:
            csv_writer.writerow({k: "" if r.get(k) is None else str(r.get(k)) for k in csv_fields})
        expected_csv_bytes = csv_buffer.getvalue().encode("utf-8")
        actual_csv_bytes = effects_file.read_bytes()
        if actual_csv_bytes != expected_csv_bytes:
            return False

        # 10. Require complete preregistered confirmatory family (tasks x 3 controls x 2 endpoints)
        expected_confirmatory_tuples = [
            (task, control, endpoint)
            for task in validation.tasks
            for control in CONFIRMATORY_CONTROLS
            for endpoint in (PRIMARY_CENSORED_ENDPOINT_NAME, THRESHOLD_YIELD_ENDPOINT_NAME)
        ]

        confirmatory_rows_ordered = []
        for task, control, endpoint in expected_confirmatory_tuples:
            matched = [
                r for r in on_disk_results
                if r.get("target_task") == task
                and r.get("condition_a") == "structured_provenance_memory"
                and r.get("condition_b") == control
                and r.get("metric_name") == endpoint
                and r.get("comparison_type") == "confirmatory"
                and r.get("confirmatory_family") == CONFIRMATORY_FAMILY_NAME
                and r.get("adjustment_method") == HOLM_ADJUSTMENT_METHOD_NAME
                and r.get("status") == "ANALYZED"
            ]
            if len(matched) != 1:
                return False
            confirmatory_rows_ordered.append(matched[0])

        all_confirmatory_in_results = [
            r for r in on_disk_results
            if r.get("confirmatory_family") == CONFIRMATORY_FAMILY_NAME
        ]
        if len(all_confirmatory_in_results) != len(expected_confirmatory_tuples):
            return False

        # Extract raw p-values and recompute Holm adjustment from full family
        n_seeds = len(validation.seeds)
        min_possible_p = 2.0 ** (1 - n_seeds)
        raw_p_values = []
        for r in confirmatory_rows_ordered:
            p_raw = r.get("p_value_raw")
            if p_raw is None or not _finite(p_raw):
                return False
            p_raw_f = float(p_raw)
            if not (0.0 <= p_raw_f <= 1.0) or p_raw_f < min_possible_p - 1e-12:
                return False
            raw_p_values.append(p_raw_f)

        recomputed_adj_p_values = holm_bonferroni_adjust(raw_p_values)
        for r, recomputed_adj in zip(confirmatory_rows_ordered, recomputed_adj_p_values):
            p_adj = r.get("p_value_adjusted")
            if p_adj is None or not _finite(p_adj):
                return False
            if abs(float(p_adj) - recomputed_adj) > 1e-6:
                return False

        return True

    def _generate_claim_evidence_matrix(self, stats: Sequence[Mapping[str, Any]], qe: Sequence[Mapping[str, Any]], runs: Sequence[Mapping[str, Any]], artifacts: Optional[Mapping[str, Any]]) -> tuple[Path, Dict[str, str]]:
        claim_status: Dict[str, str] = {f"C{i}": "Unavailable" for i in range(1, 6)}
        preflight_state, research_eligible, preflight = self._research_preflight(artifacts)
        if preflight_state == "demo":
            claim_status = {f"C{i}": "Demo only — not scientifically eligible" for i in range(1, 6)}
        elif research_eligible:
            expected_tasks, expected_seeds, _, context = self._expected_context(artifacts)
            oracle_budget = int(context.get("oracle_budget_per_run", 0))

            def _validate_stat_row(
                row: Optional[Mapping[str, Any]],
                expected_seeds: Sequence[int],
                expected_tasks: Sequence[str],
                expected_controls: Sequence[str],
                oracle_budget: int,
                *,
                is_time_endpoint: bool,
            ) -> bool:
                if row is None or row.get("status") != "ANALYZED":
                    return False
                if str(row.get("target_task")) not in set(expected_tasks):
                    return False
                if str(row.get("condition_b")) not in set(expected_controls):
                    return False
                if str(row.get("condition_a")) != "structured_provenance_memory":
                    return False
                if row.get("comparison_type") != "confirmatory":
                    return False
                if row.get("adjustment_method") != HOLM_ADJUSTMENT_METHOD_NAME:
                    return False
                if row.get("confirmatory_family") != CONFIRMATORY_FAMILY_NAME:
                    return False
                try:
                    n = int(row.get("sample_size_n"))
                    missing = int(row.get("missing_count"))
                except (TypeError, ValueError):
                    return False
                if n != len(expected_seeds) or missing != 0 or n < 1:
                    return False

                p_raw = row.get("p_value_raw")
                p_adj = row.get("p_value_adjusted")
                if p_raw is None or p_adj is None:
                    return False
                if not (_finite(p_raw) and _finite(p_adj)):
                    return False
                p_raw_f = float(p_raw)
                p_adj_f = float(p_adj)
                if not (0.0 <= p_raw_f <= 1.0 and 0.0 <= p_adj_f <= 1.0):
                    return False
                if p_adj_f > 0.05:
                    return False

                min_possible_raw_p = 2.0 ** (1 - n)
                if p_raw_f < min_possible_raw_p - 1e-12:
                    return False

                mean_diff = row.get("mean_difference")
                if not _finite(mean_diff):
                    return False
                mean_diff_f = float(mean_diff)

                details = row.get("seed_level_differences")
                if isinstance(details, str):
                    try:
                        details = json.loads(details)
                    except (TypeError, ValueError):
                        return False
                if not isinstance(details, list) or len(details) != n:
                    return False

                seen_seeds = set()
                for d in details:
                    if not isinstance(d, Mapping) or d.get("status") != "paired":
                        return False
                    try:
                        seed = int(d.get("seed"))
                    except (TypeError, ValueError):
                        return False
                    if seed in seen_seeds or seed not in set(expected_seeds):
                        return False
                    seen_seeds.add(seed)

                if seen_seeds != set(expected_seeds):
                    return False

                if is_time_endpoint:
                    if row.get("metric_name") != PRIMARY_CENSORED_ENDPOINT_NAME:
                        return False
                    if row.get("method") != PRIMARY_CENSORED_METHOD_NAME:
                        return False
                    if mean_diff_f >= 0.0:
                        return False

                    km_a = row.get("kaplan_meier_a")
                    km_b = row.get("kaplan_meier_b")
                    if not isinstance(km_a, Mapping) or not isinstance(km_b, Mapping):
                        return False
                    for km in (km_a, km_b):
                        try:
                            km_n = int(km.get("n"))
                            events = int(km.get("events"))
                            censored = int(km.get("censorings"))
                            rmst = float(km.get("restricted_mean_survival_time"))
                        except (TypeError, ValueError):
                            return False
                        if events < 0 or censored < 0 or km_n != n:
                            return False
                        if events + censored != n:
                            return False
                        if not _finite(rmst) or rmst <= 0.0:
                            return False
                        if oracle_budget > 0 and rmst > oracle_budget + 1e-6:
                            return False

                    rmst_diff = float(km_a["restricted_mean_survival_time"]) - float(km_b["restricted_mean_survival_time"])
                    if abs(mean_diff_f - rmst_diff) > 1e-5:
                        return False

                    for d in details:
                        ta = d.get("time_a")
                        tb = d.get("time_b")
                        ea = d.get("event_a")
                        eb = d.get("event_b")
                        if not (_finite(ta) and _finite(tb)):
                            return False
                        ta_f = float(ta)
                        tb_f = float(tb)
                        if oracle_budget > 0 and not (0.0 < ta_f <= oracle_budget + 1e-6 and 0.0 < tb_f <= oracle_budget + 1e-6):
                            return False
                        if ea not in (True, False, 0, 1) or eb not in (True, False, 0, 1):
                            return False
                else:
                    if row.get("metric_name") != THRESHOLD_YIELD_ENDPOINT_NAME:
                        return False
                    if row.get("method") != THRESHOLD_YIELD_METHOD_NAME:
                        return False
                    if mean_diff_f <= 0.0:
                        return False
                    for d in details:
                        va = d.get("val_a")
                        vb = d.get("val_b")
                        if not (_finite(va) and _finite(vb)):
                            return False
                        if not (0.0 <= float(va) <= 1.0 and 0.0 <= float(vb) <= 1.0):
                            return False

                return True

            def support(control: str) -> bool:
                # Claim support is an intersection over the preregistered
                # target-task set, requiring both the primary time-to-threshold
                # endpoint and threshold yield to be significant in correct directions.
                if not expected_tasks or not expected_seeds:
                    return False
                for task in expected_tasks:
                    primary_rows = [
                        row for row in stats
                        if str(row.get("target_task")) == task
                        and row.get("metric_name") == PRIMARY_CENSORED_ENDPOINT_NAME
                        and row.get("condition_a") == "structured_provenance_memory"
                        and row.get("condition_b") == control
                    ]
                    if len(primary_rows) != 1:
                        return False
                    if not _validate_stat_row(
                        primary_rows[0], expected_seeds, expected_tasks, [control],
                        oracle_budget, is_time_endpoint=True
                    ):
                        return False

                    yield_rows = [
                        row for row in stats
                        if str(row.get("target_task")) == task
                        and row.get("metric_name") == THRESHOLD_YIELD_ENDPOINT_NAME
                        and row.get("condition_a") == "structured_provenance_memory"
                        and row.get("condition_b") == control
                    ]
                    if len(yield_rows) != 1:
                        return False
                    if not _validate_stat_row(
                        yield_rows[0], expected_seeds, expected_tasks, [control],
                        oracle_budget, is_time_endpoint=False
                    ):
                        return False
                return True

            complete_runs, has_runs, _ = self._target_runs_complete(runs, artifacts)
            stats_complete = self._statistical_evidence_complete(stats, artifacts)
            for cid, control in (("C1", "adaptive_no_memory"), ("C2", "text_summary_memory"), ("C3", "shuffled_memory_control")):
                relevant = [row for row in stats if row.get("condition_b") == control]
                is_supported = (
                    complete_runs
                    and stats_complete
                    and support(control)
                    and (cid != "C3" or self._shuffle_audit_complete(runs, artifacts))
                )
                claim_status[cid] = "Supported" if is_supported else ("Inconclusive" if relevant or has_runs else "Unavailable")

            claim_status["C4"] = "Supported" if complete_runs else ("Inconclusive" if has_runs else "Unavailable")
            claim_status["C5"] = "Supported" if self._qe_evidence_complete(qe, artifacts) else ("Inconclusive" if qe else "Unavailable")
        elif preflight_state == "invalid_research":
            # Research artifacts with a failed/malformed preflight are not
            # development evidence and must never be relabelled as demo-only.
            evidence = bool(stats or qe or runs)
            claim_status = {f"C{i}": ("Inconclusive" if evidence else "Unavailable") for i in range(1, 6)}
        rows = [("C1", "Structured memory improves discovery yield", "statistics/effects.csv"), ("C2", "Structured directives outperform text summary", "statistics/effects.csv"), ("C3", "Transfer signal survives shuffled control", "statistics/effects.csv"), ("C4", "Dual proposal/oracle budget and geometry gate are preserved", "aggregates/runs.json"), ("C5", "CHGNet retention agrees with local QE decomposition margins", "qe_audit/results.csv")]
        text = "# Paper Claim-Evidence Matrix\n\nStatuses are derived from validated artifacts; unavailable or incomplete evidence is not a positive result.\n\n| Claim | Evidence | Artifact | Status |\n| :--- | :--- | :--- | :--- |\n"
        text += "\n".join(f"| **{cid}**: {description} | preregistered run/statistics/QE artifact | `{artifact}` | **{claim_status[cid]}** |" for cid, description, artifact in rows) + "\n"
        path = self.paper_dir / "claim_evidence_matrix.md"
        path.write_text(text, encoding="utf-8")
        return path, claim_status

    def _generate_reproducibility_checklist(self, stats: Sequence[Mapping[str, Any]], qe: Sequence[Mapping[str, Any]], runs: Sequence[Mapping[str, Any]], artifacts: Optional[Mapping[str, Any]]) -> Path:
        preflight_state, research_eligible, preflight = self._research_preflight(artifacts)
        target_runs_complete, _, _ = self._target_runs_complete(runs, artifacts)
        checks = {
            "validated research preflight artifact": research_eligible,
            "complete preregistered target run set": target_runs_complete,
            "complete structured candidate provenance": target_runs_complete,
            "dual proposal/oracle accounting": target_runs_complete,
            "seed-paired censored endpoint analysis": any(r.get("metric_name") == "oracle_calls_to_first_candidate_at_or_below_0_10" and "censored" in str(r.get("method", "")).lower() for r in stats),
            "Holm correction declared for confirmatory family": any(r.get("adjustment_method") == "Holm-Bonferroni" and r.get("confirmatory_family") for r in stats),
            "verified production QE/SSSP audit": research_eligible and self._qe_evidence_complete(qe, artifacts),
            "shuffle audit evidence": research_eligible and self._shuffle_audit_complete(runs, artifacts),
        }
        text = "# Reproducibility Checklist\n\n" + "\n".join(f"- [{'x' if value else ' '}] {label}" for label, value in checks.items()) + "\n"
        path = self.paper_dir / "reproducibility_checklist.md"
        path.write_text(text, encoding="utf-8")
        return path


__all__ = ["ReportGenerator"]
