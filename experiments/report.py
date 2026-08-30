"""Conservative, artifact-derived benchmark reporting.

Reports never assert a paper claim merely because a row exists. Every status is
derived from validated run/statistics/QE artifacts; missing or failed evidence
is rendered as ``Unavailable`` or ``Inconclusive``. JSON artifacts are retained
for machine reproducibility and PNGs are emitted when matplotlib is available.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


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
        passed = (
            preflight.get("status") == "PASSED"
            and preflight.get("run_mode") == "research"
            and bool(required_results)
            and all(value is True for value in required_results.values())
        )
        if preflight.get("run_mode") != "research":
            return "demo", False, preflight
        # A failed or malformed research preflight is an ineligible research
        # artifact, not a development/demo run.  This distinction keeps the
        # UI honest while remaining fail-closed for claims.
        return ("research" if passed else "invalid_research"), passed, preflight

    @staticmethod
    def _expected_context(artifacts: Optional[Mapping[str, Any]]) -> tuple[List[str], List[int], List[str], Dict[str, Any]]:
        # Campaign execution nests expected run metadata under ``run_set``;
        # direct report callers commonly pass those fields at the top level.
        # Normalize both forms before evaluating any claim.
        source: Dict[str, Any] = {}
        if isinstance(artifacts, Mapping):
            for nested_name in ("run_set", "experiment", "metadata"):
                nested = artifacts.get(nested_name)
                if isinstance(nested, Mapping):
                    source.update(nested)
            source.update(dict(artifacts))

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
                if isinstance(item, bool):
                    continue
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
        return sorted(set(tasks)), sorted(set(seeds)), conditions, source

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

    def _generate_claim_evidence_matrix(self, stats: Sequence[Mapping[str, Any]], qe: Sequence[Mapping[str, Any]], runs: Sequence[Mapping[str, Any]], artifacts: Optional[Mapping[str, Any]]) -> tuple[Path, Dict[str, str]]:
        claim_status: Dict[str, str] = {f"C{i}": "Unavailable" for i in range(1, 6)}
        preflight_state, research_eligible, preflight = self._research_preflight(artifacts)
        if preflight_state == "demo":
            claim_status = {f"C{i}": "Demo only — not scientifically eligible" for i in range(1, 6)}
        elif research_eligible:
            expected_tasks, expected_seeds, _, _ = self._expected_context(artifacts)

            def support(control: str) -> bool:
                # Claim support is an intersection over the preregistered
                # target-task set, never an existential test over rows.
                if not expected_tasks or not expected_seeds:
                    return False
                for task in expected_tasks:
                    rows = [
                        row for row in stats
                        if str(row.get("target_task")) == task
                        and row.get("metric_name") == "fraction_at_or_below_0_10"
                        and row.get("condition_a") == "structured_provenance_memory"
                        and row.get("condition_b") == control
                    ]
                    row = rows[0] if len(rows) == 1 else None
                    if row is None or row.get("status") != "ANALYZED":
                        return False
                    try:
                        n = int(row.get("sample_size_n"))
                        missing = int(row.get("missing_count"))
                    except (TypeError, ValueError):
                        return False
                    if n != len(expected_seeds) or missing != 0:
                        return False
                    if not (_finite(row.get("mean_difference")) and float(row["mean_difference"]) > 0.0):
                        return False
                    if not (_finite(row.get("p_value_adjusted")) and float(row["p_value_adjusted"]) <= 0.05):
                        return False
                    if row.get("comparison_type") not in {None, "confirmatory"}:
                        return False
                    # Do not trust sample_size_n as a substitute for the
                    # seed-level evidence itself.  Every expected seed must
                    # have one paired observation in the canonical details.
                    details = row.get("seed_level_differences")
                    if isinstance(details, str):
                        try:
                            details = json.loads(details)
                        except (TypeError, ValueError):
                            return False
                    if not isinstance(details, list):
                        return False
                    paired = set()
                    for detail in details:
                        if not isinstance(detail, Mapping) or detail.get("status") != "paired":
                            continue
                        try:
                            seed = int(detail.get("seed"))
                        except (TypeError, ValueError):
                            continue
                        if seed in paired or seed not in set(expected_seeds):
                            return False
                        if not (_finite(detail.get("val_a", detail.get("time_a"))) and _finite(detail.get("val_b", detail.get("time_b")))):
                            return False
                        paired.add(seed)
                    if paired != set(expected_seeds):
                        return False
                return True

            for cid, control in (("C1", "adaptive_no_memory"), ("C2", "text_summary_memory"), ("C3", "shuffled_memory_control")):
                relevant = [row for row in stats if row.get("condition_b") == control]
                claim_status[cid] = "Supported" if support(control) and (cid != "C3" or self._shuffle_audit_complete(runs, artifacts)) else ("Inconclusive" if relevant or runs else "Unavailable")

            complete_runs, has_runs, _ = self._target_runs_complete(runs, artifacts)
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
