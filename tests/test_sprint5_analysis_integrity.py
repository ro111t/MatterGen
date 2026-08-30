"""Adversarial tests for the repaired Sprint 5 analysis boundary."""

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pytest

from experiments.metrics import ProvenanceIntegrityError, compute_run_metrics
from experiments.qe_audit import (
    FakeQECalculator,
    QEAuditCandidate,
    QEAuditError,
    QEAuditRunner,
    QEResultRecord,
    QECalculationStatus,
    select_audit_candidates,
)
from experiments.report import ReportGenerator
from experiments.spec import QEAuditConfig
from experiments.statistics import compute_paired_censored_comparison, kaplan_meier, run_statistical_analysis_pipeline


def _structure(species):
    return {
        "lattice": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        "positions": [[0.0, 0.0, i / max(1, len(species))] for i in range(len(species))],
        "species": list(species),
        "fractional_coordinates": True,
    }


def _candidate_provenance():
    return {
        "manifest": {"status": "completed", "proposals_generated": 5, "oracle_evaluations": 4},
        # Deliberately contradictory projection: it must never be read.
        "top_candidates": [{"candidate_id": "wrong", "predictions": {"predicted_energy_above_hull_ev_per_atom": 0.0}}],
        "candidates": [
            {"candidate_id": "c1", "composition": "LiPS", "geometry_valid": True, "oracle_evaluated": True, "oracle_call_index": 2, "screening_predictions": {"predicted_energy_above_hull_ev_per_atom": 0.2}},
            {"candidate_id": "c2", "composition": "Li2PS3", "geometry_valid": False, "geometry_failure_code": "INVALID_GEOMETRY", "oracle_evaluated": False},
            {"candidate_id": "c3", "composition": "LiPS", "geometry_valid": True, "oracle_evaluated": True, "oracle_call_index": 1, "screening_predictions": {"predicted_energy_above_hull_ev_per_atom": 0.08}},
            {"candidate_id": "c4", "composition": "LiPS", "geometry_valid": True, "oracle_evaluated": True, "oracle_call_index": 3, "oracle_failure_code": "SCF_FAILED"},
            {"candidate_id": "c5", "composition": "LiPS", "geometry_valid": True, "oracle_evaluated": True, "oracle_call_index": 4, "screening_predictions": {"predicted_energy_above_hull_ev_per_atom": 0.3}},
        ],
    }


def test_metrics_use_complete_provenance_and_preserve_order_failures():
    metrics, records = compute_run_metrics(_candidate_provenance(), "r", "Li-P-Se", "adaptive_no_memory", 42, oracle_budget=4)
    assert [record.candidate_id for record in records] == ["c1", "c2", "c3", "c4", "c5"]
    assert [record.candidate_id for record in sorted(records, key=lambda x: x.oracle_call_index or 99) if record.evaluated_by_oracle] == ["c3", "c1", "c4", "c5"]
    assert metrics.proposals_generated == 5
    assert metrics.geometry_valid_count == 4
    assert metrics.invalid_geometry_count == 1
    assert metrics.oracle_evaluations == 4
    assert metrics.oracle_failure_codes == {"SCF_FAILED": 1}
    assert metrics.oracle_calls_to_first_candidate_at_or_below_0_10 == 1
    assert metrics.reached_0_10_threshold is True


def test_top_candidates_only_is_rejected_fail_closed():
    with pytest.raises(ProvenanceIntegrityError):
        compute_run_metrics({"top_candidates": [{"candidate_id": "fake"}]}, "r", "t", "c", 1, 2)


def test_censored_endpoint_and_kaplan_meier_retain_budget():
    metrics_data = []
    for condition in ("structured_provenance_memory", "adaptive_no_memory"):
        payload = _candidate_provenance()
        payload["candidates"] = [
            {"candidate_id": "c", "composition": "LiPS", "geometry_valid": True, "oracle_evaluated": True, "oracle_call_index": 1, "oracle_failure_code": "SCF_FAILED"},
            {"candidate_id": "d", "composition": "LiPS", "geometry_valid": True, "oracle_evaluated": True, "oracle_call_index": 2, "oracle_failure_code": "SCF_FAILED"},
        ]
        metrics_data.append(compute_run_metrics(payload, condition, "Li-P-Se", condition, 42, oracle_budget=2)[0])
    assert metrics_data[0].oracle_calls_to_first_candidate_at_or_below_0_10 == 2
    assert metrics_data[0].primary_endpoint_censored is True
    km = kaplan_meier([(2, False), (1, True)], budget=2)
    assert km.n == 2 and km.events == 1 and km.restricted_mean_survival_time == pytest.approx(1.5)
    result = compute_paired_censored_comparison("Li-P-Se", "endpoint", {42: (2, False)}, {42: (1, True)}, "a", "b", 2, expected_seeds=[42, 137])
    assert result.missing_count == 1
    assert result.kaplan_meier_a["censorings"] == 1


def test_invalid_censored_time_and_mixed_fixed_budgets_are_missing():
    invalid = compute_paired_censored_comparison(
        "task", "endpoint", {1: (0, False)}, {1: (1, True)}, "a", "b", 5,
        expected_seeds=[1],
    )
    assert invalid.status == "UNAVAILABLE_MISSING_PAIRS"
    assert invalid.sample_size_n == 0 and invalid.missing_count == 1
    mixed = [
        {"task_id": "task", "condition": "structured_provenance_memory", "seed": 1,
         "oracle_calls_to_first_candidate_at_or_below_0_10": 6, "oracle_budget": 5,
         "primary_endpoint_censored": True},
        {"task_id": "task", "condition": "adaptive_no_memory", "seed": 1,
         "oracle_calls_to_first_candidate_at_or_below_0_10": 4, "oracle_budget": 10,
         "primary_endpoint_censored": True},
    ]
    results, _ = run_statistical_analysis_pipeline(mixed, expected_seeds=[1], expected_tasks=["task"])
    endpoint = next(row for row in results if row.metric_name == "oracle_calls_to_first_candidate_at_or_below_0_10" and row.condition_b == "adaptive_no_memory")
    assert endpoint.status == "UNAVAILABLE_MISSING_PAIRS"
    assert endpoint.sample_size_n == 0 and endpoint.missing_count == 1


def test_statistics_keep_all_preregistered_arms_and_holm_family(tmp_path):
    metric_a = compute_run_metrics(_candidate_provenance(), "a", "Li-P-Se", "structured_provenance_memory", 42, 4)[0]
    metric_b = compute_run_metrics(_candidate_provenance(), "b", "Li-P-Se", "adaptive_no_memory", 42, 4)[0]
    results, summary = run_statistical_analysis_pipeline([metric_a, metric_b], output_dir=tmp_path, expected_seeds=[42, 137])
    assert len(results) == 32  # every declared target/control/metric row remains present
    assert all(row.missing_count >= 1 for row in results)
    manifest = json.loads((tmp_path / "analysis_manifest.json").read_text())
    assert manifest["expected_seeds"] == [42, 137]
    assert manifest["confirmatory_family"]
    assert manifest["adjustment_method"] == "Holm-Bonferroni"


class _InjectedCalculator:
    def run_calculation(self, formula, structure, config, calc_dir, *, is_candidate=True):
        energies = {"LiPS": -2.0, "LiP": -1.0, "S": -0.2}
        calc_dir.mkdir(parents=True, exist_ok=True)
        value = energies[formula]
        return QEResultRecord(
            calculation_id=formula, formula=formula, is_candidate=is_candidate,
            status=QECalculationStatus.CONVERGED.value, total_energy_ev=value,
            num_atoms=len(structure["species"]), energy_per_atom_ev=value / len(structure["species"]),
            scf_steps=2, relaxation_steps=0, max_force_ev_per_ang=0.01,
            calculation_dir=str(calc_dir), input_hash=f"hash-{formula}",
        )


def test_qe_research_gate_balance_margin_and_no_fake_structure(tmp_path):
    with pytest.raises(QEAuditError):
        QEAuditRunner(QEAuditConfig(mock_execution=True), tmp_path, run_mode="research")
    with pytest.raises(QEAuditError):
        QEAuditRunner(QEAuditConfig(mock_execution=False), tmp_path)
    candidate = QEAuditCandidate(
        candidate_id="c", target_task="Li-P-Se", condition="structured_provenance_memory", seed=42,
        reduced_formula="LiPS", structure=_structure(["Li", "P", "S"]),
        predicted_energy_above_hull_ev_per_atom=0.08,
        predicted_decomposition_products=[
            {"formula": "LiP", "amount": 1.0, "structure": _structure(["Li", "P"])},
            {"formula": "S", "amount": 1.0, "structure": _structure(["S"])},
        ], selection_rank=1, selection_reason="test",
    )
    result = QEAuditRunner(QEAuditConfig(mock_execution=True), tmp_path, calculator=_InjectedCalculator()).audit_candidate(candidate)
    assert result.reaction_balanced is True
    assert result.status == "VALIDATED"
    assert result.dft_local_decomposition_margin_ev_per_atom == pytest.approx((-2.0 - (-1.2)) / 3.0)
    with pytest.raises(QEAuditError):
        QEAuditRunner(QEAuditConfig(mock_execution=True), tmp_path, calculator=FakeQECalculator()).audit_candidate(
            QEAuditCandidate("bad", "Li-P-Se", "structured_provenance_memory", 1, "LiPS", {"composition": "LiPS"}, 0.1, [], 1, "bad")
        )


def test_selection_reports_insufficiency_without_fabricating_structures():
    selected = select_audit_candidates([], target_count=2)
    assert len(selected) == 0
    assert selected.insufficiency == "insufficient_valid_target_candidates:0/2"


def test_report_c4_rejects_unexpected_non_source_target_rows(tmp_path):
    row = {
        "run_id": "run-1", "task_id": "task", "condition": "structured_provenance_memory", "seed": 1,
        "run_status": "completed", "proposals_generated": 2, "total_candidates_recorded": 2,
        "geometry_valid_count": 2, "invalid_geometry_count": 0, "oracle_evaluations": 2,
        "oracle_budget": 2, "oracle_success_count": 2, "oracle_failure_count": 0,
        "provenance_complete": True,
    }
    artifacts = {
        "expected_target_task_ids": ["task"], "expected_seeds": [1],
        "expected_conditions": ["structured_provenance_memory"],
        "expected_target_run_ids": ["run-1"], "expected_source_task": "source",
    }
    complete, _, _ = ReportGenerator._target_runs_complete([row], artifacts)
    assert complete is True
    extra = {**row, "run_id": "run-extra", "task_id": "unexpected"}
    complete, _, _ = ReportGenerator._target_runs_complete([row, extra], artifacts)
    assert complete is False


def test_report_c5_binds_supplied_rows_to_qe_manifest_provenance(tmp_path):
    phase = {
        "status": "CONVERGED", "input_hash": "a" * 64, "result_hash": "b" * 64,
        "output_hash": "c" * 64, "executable": "pw.x", "executable_version": "qe-1",
        "sssp_manifest_sha256": "d" * 64,
    }
    row = {
        "candidate_id": "c", "target_task": "task", "condition": "structured_provenance_memory", "seed": 1,
        "formula": "Li", "num_atoms": 1, "candidate_status": "CONVERGED",
        "candidate_energy_per_atom_ev": "-1", "competing_phases_count": 1,
        "competing_phases_all_converged": True, "chgnet_predicted_hull_distance_ev_per_atom": 0.01,
        "dft_local_decomposition_margin_ev_per_atom": 0.01, "sign_agreement": True,
        "margin_difference_ev_per_atom": 0.0, "reaction_equation": "Li -> Li", "reaction_balanced": True,
        "status": "VALIDATED", "candidate_input_hash": "a" * 64,
        "candidate_result_hash": "b" * 64, "candidate_output_hash": "c" * 64,
        "qe_executable": "pw.x", "qe_executable_version": "qe-1", "sssp_manifest_sha256": "d" * 64,
        "participating_phase_results": [phase],
    }
    qe_dir = tmp_path / "qe_audit"
    qe_dir.mkdir()
    (qe_dir / "selection.json").write_text(json.dumps({"candidates": [{"candidate_id": "c"}], "insufficiency": None}), encoding="utf-8")
    with (qe_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        csv_row = dict(row)
        csv_row["participating_phase_results"] = json.dumps(phase and [phase], sort_keys=True, separators=(",", ":"))
        writer.writerow(csv_row)
    manifest = {
        "run_mode": "research", "mock_execution": False, "selection_count": 1,
        "selection_insufficiency": None, "result_provenance": [row],
    }
    manifest["artifacts"] = {
        "selection.json": hashlib.sha256((qe_dir / "selection.json").read_bytes()).hexdigest(),
        "results.csv": hashlib.sha256((qe_dir / "results.csv").read_bytes()).hexdigest(),
        "canonical_results_sha256": hashlib.sha256(json.dumps([row], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
    }
    report = ReportGenerator(tmp_path)
    supplied = dict(csv_row)
    assert report._qe_evidence_complete([supplied], {"expected_qe_candidate_count": 1, "qe_selection_count": 1, "qe_audit": manifest}) is True
    altered = {**supplied, "candidate_id": "injected"}
    assert report._qe_evidence_complete([altered], {"expected_qe_candidate_count": 1, "qe_selection_count": 1, "qe_audit": manifest}) is False


def test_report_marks_missing_evidence_conservatively(tmp_path):
    generated = ReportGenerator(tmp_path).generate_all_reports([], [], [])
    matrix = (tmp_path / "paper" / "claim_evidence_matrix.md").read_text()
    checklist = (tmp_path / "paper" / "reproducibility_checklist.md").read_text()
    assert "Unavailable" in matrix
    assert "[x]" not in checklist
    assert Path(generated["fig1_best_so_far"]).exists()


def _make_valid_research_spec(tasks=("task1", "task2"), seeds=None, oracle_budget=10):
    from experiments.spec import ExperimentSpec, TaskDefinition, QEAuditConfig
    if seeds is None:
        seeds = [42, 137, 2024, 777, 999, 31415, 27182, 16180, 104729]
    return ExperimentSpec(
        experiment_id="exp_sprint5_test",
        run_mode="research",
        generation_backend="mattergen",
        pinned_model_identity={"name": "chgnet", "version": "0.3.0", "checkpoint_sha256": "0" * 64},
        pinned_relaxation_settings={"fmax": 0.05},
        mattergen_model_path="models/mattergen.pt",
        mattergen_checkpoint_sha256="0" * 64,
        mattergen_sampling_config_sha256="0" * 64,
        qe_audit_config=QEAuditConfig(
            mock_execution=False,
            sssp_manifest_path="pseudopotentials/sssp.json",
            sssp_manifest_sha256="0" * 64,
            qe_executable_version="7.2",
            qe_executable_sha256="0" * 64,
        ),
        validation_calculator="vasp",
        synthesis_mode="offline_reference",
        target_tasks=[TaskDefinition(t, ["Li", "P", "Se"]) for t in tasks],
        master_seeds=list(seeds),
        proposals_per_run=20,
        oracle_budget_per_run=oracle_budget,
    )


def _make_valid_run_rows(tasks=("task1", "task2"), seeds=(42, 137, 2024, 777, 999, 31415, 27182, 16180, 104729), conditions=("structured_provenance_memory", "adaptive_no_memory", "text_summary_memory", "shuffled_memory_control", "random_mattergen")):
    rows = []
    for task in tasks:
        for cond in conditions:
            for seed in seeds:
                row = {
                    "run_id": f"run_{cond}_{task}_seed{seed}",
                    "task_id": task,
                    "condition": cond,
                    "seed": seed,
                    "run_status": "completed",
                    "proposals_generated": 20,
                    "total_candidates_recorded": 20,
                    "geometry_valid_count": 20,
                    "invalid_geometry_count": 0,
                    "oracle_evaluations": 10,
                    "oracle_budget": 10,
                    "oracle_success_count": 10,
                    "oracle_failure_count": 0,
                    "provenance_complete": True,
                }
                if cond == "shuffled_memory_control":
                    row["shuffle_validation"] = {"valid": True, "fixed_points": 0}
                rows.append(row)
    return rows


def _make_valid_stat_rows(
    tasks=("task1", "task2"),
    seeds=(42, 137, 2024, 777, 999, 31415, 27182, 16180, 104729),
    controls=("adaptive_no_memory", "text_summary_memory", "shuffled_memory_control"),
    time_diff=-2.0,
    yield_diff=0.3,
    p_raw=None,
    p_adj=None,
    oracle_budget=10,
):
    from experiments.statistics import holm_bonferroni_adjust
    n = len(seeds)
    if p_raw is None:
        p_raw = 2.0 ** (1 - n)  # minimal attainable raw p-value
    rows = []
    raw_p_list = []
    for task in tasks:
        for control in controls:
            # 1. Primary time-to-threshold endpoint
            p_details = [
                {"status": "paired", "seed": s, "time_a": 2.0, "time_b": 4.0, "event_a": True, "event_b": True}
                for s in seeds
            ]
            km_a = {"n": n, "events": n, "censorings": 0, "restricted_mean_survival_time": 2.0}
            km_b = {"n": n, "events": n, "censorings": 0, "restricted_mean_survival_time": 4.0}
            rows.append({
                "target_task": task,
                "condition_a": "structured_provenance_memory",
                "condition_b": control,
                "metric_name": "oracle_calls_to_first_candidate_at_or_below_0_10",
                "status": "ANALYZED",
                "sample_size_n": n,
                "missing_count": 0,
                "mean_a": 2.0,
                "mean_b": 4.0,
                "mean_difference": time_diff,
                "median_difference": time_diff,
                "ci_95_lower": time_diff - 0.5,
                "ci_95_upper": time_diff + 0.5,
                "p_value_raw": p_raw,
                "p_value_adjusted": None,
                "adjustment_method": "Holm-Bonferroni",
                "confirmatory_family": "primary_endpoint_and_threshold_yield_family",
                "comparison_type": "confirmatory",
                "method": "paired_censored_RMST_within_seed_randomization",
                "kaplan_meier_a": km_a,
                "kaplan_meier_b": km_b,
                "seed_level_differences": json.dumps(p_details),
            })
            raw_p_list.append(p_raw)

            # 2. Secondary yield endpoint
            y_details = [
                {"status": "paired", "seed": s, "val_a": 0.6, "val_b": 0.3}
                for s in seeds
            ]
            rows.append({
                "target_task": task,
                "condition_a": "structured_provenance_memory",
                "condition_b": control,
                "metric_name": "fraction_at_or_below_0_10",
                "status": "ANALYZED",
                "sample_size_n": n,
                "missing_count": 0,
                "mean_a": 0.6,
                "mean_b": 0.3,
                "mean_difference": yield_diff,
                "median_difference": yield_diff,
                "ci_95_lower": yield_diff - 0.1,
                "ci_95_upper": yield_diff + 0.1,
                "p_value_raw": p_raw,
                "p_value_adjusted": None,
                "adjustment_method": "Holm-Bonferroni",
                "confirmatory_family": "primary_endpoint_and_threshold_yield_family",
                "comparison_type": "confirmatory",
                "method": "paired_sign_flip",
                "seed_level_differences": json.dumps(y_details),
            })
            raw_p_list.append(p_raw)

    adjusted_p_list = holm_bonferroni_adjust(raw_p_list)
    for row, adj in zip(rows, adjusted_p_list):
        row["p_value_adjusted"] = p_adj if p_adj is not None else adj

    return rows


def _make_valid_bundle(spec, tmp_path, stats_rows=None):
    from experiments.spec import CANONICAL_RESEARCH_PREFLIGHT_CHECKS
    spec_dict = spec.to_dict()
    spec_hash = spec.spec_hash
    stats_dir = tmp_path / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)

    if stats_rows is None:
        stats_rows = _make_valid_stat_rows(
            tasks=[t.task_id for t in spec.target_tasks],
            seeds=spec.master_seeds,
            oracle_budget=spec.oracle_budget_per_run,
        )

    # Write effects.csv using canonical serialization
    effects_path = stats_dir / "effects.csv"
    fields = [
        "analysis_version", "target_task", "metric_name", "condition_a", "condition_b",
        "comparison_type", "sample_size_n", "missing_count", "mean_a", "mean_b",
        "mean_difference", "median_difference", "ci_95_lower", "ci_95_upper",
        "p_value_raw", "p_value_adjusted", "adjustment_method", "method", "status",
    ]
    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in stats_rows:
        writer.writerow({k: "" if row.get(k) is None else str(row.get(k)) for k in fields})
    csv_bytes = buf.getvalue().encode("utf-8")
    effects_path.write_bytes(csv_bytes)
    effects_sha256 = hashlib.sha256(csv_bytes).hexdigest()

    # Write results.json
    results_path = stats_dir / "results.json"
    results_json_bytes = json.dumps(stats_rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    results_path.write_bytes(results_json_bytes)
    results_sha256 = hashlib.sha256(results_json_bytes).hexdigest()
    canonical_results_sha256 = results_sha256

    # Write analysis_manifest.json
    manifest = {
        "analysis_version": "1.0.0",
        "experiment_id": spec.experiment_id,
        "spec_hash": spec_hash,
        "primary_endpoint": "oracle_calls_to_first_candidate_at_or_below_0_10",
        "confirmatory_family": "primary_endpoint_and_threshold_yield_family",
        "adjustment_method": "Holm-Bonferroni",
        "artifacts": {
            "effects.csv": effects_sha256,
            "results.json": results_sha256,
            "canonical_results_sha256": canonical_results_sha256,
        },
        "results": stats_rows,
    }
    (stats_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    preflight_checks = {name: True for name in CANONICAL_RESEARCH_PREFLIGHT_CHECKS}
    artifacts = {
        "spec": spec_dict,
        "spec_hash": spec_hash,
        "dag": {"experiment_id": spec.experiment_id, "spec_hash": spec_hash},
        "preflight": {
            "status": "PASSED",
            "run_mode": spec.run_mode,
            "spec_hash": spec_hash,
            "required_checks": list(CANONICAL_RESEARCH_PREFLIGHT_CHECKS),
            "checks": preflight_checks,
        },
        "analysis_manifest": manifest,
    }
    return artifacts, stats_rows


def test_task1_frozen_spec_15_failure_modes_and_derivation(tmp_path):
    from experiments.report import validate_frozen_spec
    from experiments.spec import CANONICAL_RESEARCH_PREFLIGHT_CHECKS

    spec = _make_valid_research_spec()
    spec_dict = spec.to_dict()
    spec_hash = spec.spec_hash

    # Base valid bundle
    def _base():
        return {
            "spec": dict(spec_dict),
            "spec_hash": spec_hash,
            "dag": {"experiment_id": spec.experiment_id, "spec_hash": spec_hash},
            "preflight": {
                "status": "PASSED",
                "run_mode": "research",
                "spec_hash": spec_hash,
                "required_checks": list(CANONICAL_RESEARCH_PREFLIGHT_CHECKS),
                "checks": {name: True for name in CANONICAL_RESEARCH_PREFLIGHT_CHECKS},
            },
        }

    # 1. Missing frozen spec
    b1 = _base()
    del b1["spec"]
    assert validate_frozen_spec(b1).valid is False

    # 2. Missing top-level spec_hash
    b2 = _base()
    del b2["spec_hash"]
    assert validate_frozen_spec(b2).valid is False

    # 3. Missing DAG
    b3 = _base()
    del b3["dag"]
    assert validate_frozen_spec(b3).valid is False

    # 4. Missing dag.spec_hash
    b4 = _base()
    b4["dag"] = {"experiment_id": spec.experiment_id}
    assert validate_frozen_spec(b4).valid is False

    # 5. Missing preflight
    b5 = _base()
    del b5["preflight"]
    assert validate_frozen_spec(b5).valid is False

    # 6. Missing preflight.spec_hash
    b6 = _base()
    b6["preflight"] = {
        "status": "PASSED", "run_mode": "research",
        "required_checks": list(CANONICAL_RESEARCH_PREFLIGHT_CHECKS),
        "checks": {name: True for name in CANONICAL_RESEARCH_PREFLIGHT_CHECKS},
    }
    assert validate_frozen_spec(b6).valid is False

    # 7. Malformed non-hex hash
    b7 = _base()
    b7["spec_hash"] = "not_a_hex_digest"
    assert validate_frozen_spec(b7).valid is False

    # 8. Top-level hash mismatch
    b8 = _base()
    b8["spec_hash"] = "0" * 64
    assert validate_frozen_spec(b8).valid is False

    # 9. DAG hash mismatch
    b9 = _base()
    b9["dag"]["spec_hash"] = "1" * 64
    assert validate_frozen_spec(b9).valid is False

    # 10. Preflight hash mismatch
    b10 = _base()
    b10["preflight"]["spec_hash"] = "2" * 64
    assert validate_frozen_spec(b10).valid is False

    # 11. Spec content tampered after hashing
    b11 = _base()
    b11["spec"]["proposals_per_run"] = 9999
    assert validate_frozen_spec(b11).valid is False

    # 12. DAG experiment ID mismatch
    b12 = _base()
    b12["dag"]["experiment_id"] = "different_exp_id"
    assert validate_frozen_spec(b12).valid is False

    # 13. Caller task subset
    b13 = _base()
    b13["expected_target_task_ids"] = ["task1"]
    assert validate_frozen_spec(b13).valid is False

    # 14. Caller seed subset
    b14 = _base()
    b14["expected_seeds"] = [42, 137]
    assert validate_frozen_spec(b14).valid is False

    # 15. Valid bundle derivation
    v = validate_frozen_spec(_base())
    assert v.valid is True
    assert v.tasks == ["task1", "task2"]
    assert v.seeds == sorted(spec.master_seeds)
    assert len(v.expected_target_run_ids) == 2 * 5 * 9


def test_task1_empty_and_subset_caller_metadata_fails_closed(tmp_path):
    from experiments.report import validate_frozen_spec
    from experiments.spec import CANONICAL_RESEARCH_PREFLIGHT_CHECKS

    spec = _make_valid_research_spec()
    spec_dict = spec.to_dict()
    spec_hash = spec.spec_hash

    def _base():
        return {
            "spec": dict(spec_dict),
            "spec_hash": spec_hash,
            "dag": {"experiment_id": spec.experiment_id, "spec_hash": spec_hash},
            "preflight": {
                "status": "PASSED",
                "run_mode": "research",
                "spec_hash": spec_hash,
                "required_checks": list(CANONICAL_RESEARCH_PREFLIGHT_CHECKS),
                "checks": {name: True for name in CANONICAL_RESEARCH_PREFLIGHT_CHECKS},
            },
        }

    # Explicit empty lists must fail closed
    b_empty_tasks = _base()
    b_empty_tasks["expected_target_task_ids"] = []
    assert validate_frozen_spec(b_empty_tasks).valid is False

    b_empty_seeds = _base()
    b_empty_seeds["expected_seeds"] = []
    assert validate_frozen_spec(b_empty_seeds).valid is False

    b_empty_conds = _base()
    b_empty_conds["expected_conditions"] = []
    assert validate_frozen_spec(b_empty_conds).valid is False

    b_empty_runs = _base()
    b_empty_runs["expected_target_run_ids"] = []
    assert validate_frozen_spec(b_empty_runs).valid is False

    # Partial non-empty subsets must fail closed
    b_subset_conds = _base()
    b_subset_conds["expected_conditions"] = ["structured_provenance_memory", "adaptive_no_memory"]
    assert validate_frozen_spec(b_subset_conds).valid is False


def test_task1_research_preflight_canonical_checks_enforcement(tmp_path):
    from experiments.report import ReportGenerator
    from experiments.spec import CANONICAL_RESEARCH_PREFLIGHT_CHECKS

    generator = ReportGenerator(tmp_path)
    spec = _make_valid_research_spec()

    # 1. Arbitrary single check is rejected for research mode
    artifacts_arbitrary = {
        "spec": spec.to_dict(),
        "spec_hash": spec.spec_hash,
        "dag": {"experiment_id": spec.experiment_id, "spec_hash": spec.spec_hash},
        "preflight": {
            "status": "PASSED",
            "run_mode": "research",
            "spec_hash": spec.spec_hash,
            "required_checks": {"arbitrary_check": True},
            "checks": {"arbitrary_check": True},
        },
    }
    state, eligible, _ = generator._research_preflight(artifacts_arbitrary)
    assert state == "invalid_research"
    assert eligible is False

    # 2. Truncated check set (missing some required research checks) is rejected
    truncated_checks = list(CANONICAL_RESEARCH_PREFLIGHT_CHECKS)[:5]
    artifacts_truncated = {
        "spec": spec.to_dict(),
        "spec_hash": spec.spec_hash,
        "dag": {"experiment_id": spec.experiment_id, "spec_hash": spec.spec_hash},
        "preflight": {
            "status": "PASSED",
            "run_mode": "research",
            "spec_hash": spec.spec_hash,
            "required_checks": truncated_checks,
            "checks": {name: True for name in truncated_checks},
        },
    }
    state_t, eligible_t, _ = generator._research_preflight(artifacts_truncated)
    assert state_t == "invalid_research"
    assert eligible_t is False

    # 3. Full canonical check set with all True is accepted
    artifacts_full = {
        "spec": spec.to_dict(),
        "spec_hash": spec.spec_hash,
        "dag": {"experiment_id": spec.experiment_id, "spec_hash": spec.spec_hash},
        "preflight": {
            "status": "PASSED",
            "run_mode": "research",
            "spec_hash": spec.spec_hash,
            "required_checks": list(CANONICAL_RESEARCH_PREFLIGHT_CHECKS),
            "checks": {name: True for name in CANONICAL_RESEARCH_PREFLIGHT_CHECKS},
        },
    }
    state_f, eligible_f, _ = generator._research_preflight(artifacts_full)
    assert state_f == "research"
    assert eligible_f is True


def _make_research_mode_spec(tasks=("t1", "t2"), seeds=(1, 2, 3, 4, 5, 6, 7, 8, 9)):
    from experiments.spec import ExperimentSpec, TaskDefinition, QEAuditConfig
    return ExperimentSpec(
        "exp_research",
        run_mode="research",
        generation_backend="mattergen",
        pinned_model_identity={"name": "chgnet", "version": "0.3.0", "checkpoint_sha256": "0" * 64},
        pinned_relaxation_settings={"fmax": 0.05},
        mattergen_model_path="models/mattergen.pt",
        mattergen_checkpoint_sha256="0" * 64,
        mattergen_sampling_config_sha256="0" * 64,
        qe_audit_config=QEAuditConfig(
            mock_execution=False,
            sssp_manifest_path="pseudopotentials/sssp.json",
            sssp_manifest_sha256="0" * 64,
            qe_executable_version="7.2",
            qe_executable_sha256="0" * 64,
        ),
        validation_calculator="vasp",
        synthesis_mode="offline_reference",
        target_tasks=[TaskDefinition(t, ["Li", "P", "Se"]) for t in tasks],
        master_seeds=list(seeds),
    )


def test_task2_mathematical_sample_size_feasibility():
    from experiments.spec import calculate_minimum_exact_test_sample_size, ExperimentSpecError

    # Formula check: n_min = ceil(1 + log2(m / 0.05))
    # T = 2 -> m = 12 -> 1 + log2(240) = 8.9069 -> 9
    assert calculate_minimum_exact_test_sample_size(2) == 9
    # T = 1 -> m = 6 -> 1 + log2(120) = 7.9069 -> 8
    assert calculate_minimum_exact_test_sample_size(1) == 8
    # Custom T = 3 -> m = 18 -> 1 + log2(360) = 9.49 -> 10
    assert calculate_minimum_exact_test_sample_size(3) == 10
    # Custom T = 4 -> m = 24 -> 1 + log2(480) = 9.9069 -> 10
    assert calculate_minimum_exact_test_sample_size(4) == 10
    # Custom T = 5 -> m = 30 -> 1 + log2(600) = 10.2288 -> 11
    assert calculate_minimum_exact_test_sample_size(5) == 11

    # Exceeding exact permutation enumeration boundary (n > 16): T=274 requires n=17 -> raises ValueError
    with pytest.raises(ValueError, match="exceeds exact permutation enumeration boundary"):
        calculate_minimum_exact_test_sample_size(274)

    # 16 research seeds accepted
    spec16 = _make_research_mode_spec(tasks=["t1", "t2"], seeds=list(range(1, 17)))
    assert len(spec16.master_seeds) == 16

    # 17 research seeds rejected in research mode
    with pytest.raises(ExperimentSpecError, match="supports at most 16 unique master seeds"):
        _make_research_mode_spec(tasks=["t1", "t2"], seeds=list(range(1, 18)))

    # Development mode accepts 17 seeds without restriction
    from experiments.spec import ExperimentSpec, TaskDefinition
    dev_spec_17 = ExperimentSpec(
        "exp_dev_17",
        run_mode="development",
        target_tasks=[TaskDefinition("t1", ["Li"]), TaskDefinition("t2", ["Na"])],
        master_seeds=list(range(1, 18)),
    )
    assert len(dev_spec_17.master_seeds) == 17

    # Report validation rejects frozen 17-seed research spec
    from experiments.report import validate_frozen_spec
    from experiments.spec import CANONICAL_RESEARCH_PREFLIGHT_CHECKS, compute_sha256
    valid_spec = _make_research_mode_spec(tasks=["t1", "t2"], seeds=list(range(1, 10)))
    research_17_dict = valid_spec.to_dict()
    research_17_dict["master_seeds"] = list(range(1, 18))
    hash_17 = compute_sha256(research_17_dict)
    artifacts_17 = {
        "spec": research_17_dict,
        "spec_hash": hash_17,
        "dag": {"experiment_id": valid_spec.experiment_id, "spec_hash": hash_17},
        "preflight": {
            "status": "PASSED",
            "run_mode": "research",
            "spec_hash": hash_17,
            "required_checks": list(CANONICAL_RESEARCH_PREFLIGHT_CHECKS),
            "checks": {name: True for name in CANONICAL_RESEARCH_PREFLIGHT_CHECKS},
        },
    }
    v_17 = validate_frozen_spec(artifacts_17)
    assert v_17.valid is False
    assert any("supports at most 16 unique master seeds" in f or "17>16" in f for f in v_17.failures)


def _make_run_metric(
    run_id: str,
    task_id: str,
    condition: str,
    seed: int,
    oracle_calls_to_threshold: Optional[int] = 2,
    yield_0_10: float = 0.8,
    censored: bool = False,
    shuffle_val: Optional[dict] = None,
):
    from experiments.metrics import RunMetrics
    return RunMetrics(
        run_id=run_id,
        task_id=task_id,
        condition=condition,
        seed=seed,
        proposals_generated=20,
        geometry_valid_count=20,
        invalid_geometry_count=0,
        geometry_yield=1.0,
        oracle_evaluations=10,
        oracle_budget=10,
        oracle_success_count=10,
        oracle_failure_count=0,
        oracle_success_rate=1.0,
        oracle_calls_to_first_candidate_at_or_below_0_10=oracle_calls_to_threshold,
        reached_0_10_threshold=not censored,
        count_at_or_below_0_00=0,
        count_at_or_below_0_03=0,
        count_at_or_below_0_05=0,
        count_at_or_below_0_10=int(yield_0_10 * 10),
        fraction_at_or_below_0_00=0.0,
        fraction_at_or_below_0_03=0.0,
        fraction_at_or_below_0_05=0.0,
        fraction_at_or_below_0_10=yield_0_10,
        best_energy_above_hull_overall=0.02,
        best_energy_at_fixed_oracle_budgets={10: 0.02},
        area_under_best_curve=0.2,
        unique_reduced_compositions_count=10,
        unique_anonymous_stoichiometries_count=5,
        unique_prototypes_count=5,
        memory_directives_applied_count=1,
        memory_directives_rejected_count=0,
        memory_directives_unsupported_count=0,
        memory_prioritized_candidates_count=1,
        total_candidates_recorded=20,
        missing_hull_energy_count=0,
        primary_endpoint_censored=censored,
        provenance_complete=True,
        run_status="completed",
        shuffle_validation=shuffle_val,
    )


def test_task2_end_to_end_real_pipeline_positive_claim_supported(tmp_path):
    from experiments.spec import CANONICAL_RESEARCH_PREFLIGHT_CHECKS

    spec = _make_valid_research_spec()
    spec_dict = spec.to_dict()
    spec_hash = spec.spec_hash
    stats_dir = tmp_path / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)

    metrics_list = []
    for task in ("task1", "task2"):
        for seed in spec.master_seeds:
            # 1. structured_provenance_memory (effective: candidate found at oracle call 2, yield 0.8)
            metrics_list.append(_make_run_metric(
                run_id=f"run_structured_provenance_memory_{task}_seed{seed}",
                task_id=task,
                condition="structured_provenance_memory",
                seed=seed,
                oracle_calls_to_threshold=2,
                yield_0_10=0.8,
                censored=False,
            ))
            # 2. adaptive_no_memory (poor: censored at call 10, yield 0.0)
            metrics_list.append(_make_run_metric(
                run_id=f"run_adaptive_no_memory_{task}_seed{seed}",
                task_id=task,
                condition="adaptive_no_memory",
                seed=seed,
                oracle_calls_to_threshold=10,
                yield_0_10=0.0,
                censored=True,
            ))
            # 3. text_summary_memory (censored at call 10, yield 0.1)
            metrics_list.append(_make_run_metric(
                run_id=f"run_text_summary_memory_{task}_seed{seed}",
                task_id=task,
                condition="text_summary_memory",
                seed=seed,
                oracle_calls_to_threshold=10,
                yield_0_10=0.1,
                censored=True,
            ))
            # 4. shuffled_memory_control (censored at call 10, yield 0.1)
            metrics_list.append(_make_run_metric(
                run_id=f"run_shuffled_memory_control_{task}_seed{seed}",
                task_id=task,
                condition="shuffled_memory_control",
                seed=seed,
                oracle_calls_to_threshold=10,
                yield_0_10=0.1,
                censored=True,
                shuffle_val={"valid": True, "fixed_points": 0},
            ))
            # 5. random_mattergen
            metrics_list.append(_make_run_metric(
                run_id=f"run_random_mattergen_{task}_seed{seed}",
                task_id=task,
                condition="random_mattergen",
                seed=seed,
                oracle_calls_to_threshold=10,
                yield_0_10=0.0,
                censored=True,
            ))

    runs_list = [m.to_dict() for m in metrics_list]

    # Execute the REAL statistical analysis pipeline
    results, summary = run_statistical_analysis_pipeline(
        metrics_list,
        output_dir=stats_dir,
        expected_seeds=spec.master_seeds,
        expected_tasks=[t.task_id for t in spec.target_tasks],
        experiment_id=spec.experiment_id,
        spec_hash=spec.spec_hash,
    )

    manifest = json.loads((stats_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
    artifacts = {
        "spec": spec_dict,
        "spec_hash": spec_hash,
        "dag": {"experiment_id": spec.experiment_id, "spec_hash": spec_hash},
        "preflight": {
            "status": "PASSED",
            "run_mode": "research",
            "spec_hash": spec_hash,
            "required_checks": list(CANONICAL_RESEARCH_PREFLIGHT_CHECKS),
            "checks": {name: True for name in CANONICAL_RESEARCH_PREFLIGHT_CHECKS},
        },
        "analysis_manifest": manifest,
    }

    generator = ReportGenerator(tmp_path)
    _, status = generator._generate_claim_evidence_matrix(
        [r.to_dict() for r in results],
        [],
        runs_list,
        artifacts,
    )
    assert status["C1"] == "Supported"
    assert status["C2"] == "Supported"
    assert status["C3"] == "Supported"
    assert status["C4"] == "Supported"


def test_task3_stat_row_validation_adversarial_modes(tmp_path):
    spec = _make_valid_research_spec()
    generator = ReportGenerator(tmp_path)
    runs = _make_valid_run_rows(tasks=[t.task_id for t in spec.target_tasks], seeds=spec.master_seeds)

    # Helper to test invalid stat rows
    def _check_inconclusive(mutated_stats):
        art, _ = _make_valid_bundle(spec, tmp_path, mutated_stats)
        _, s = generator._generate_claim_evidence_matrix(mutated_stats, [], runs, art)
        return s["C1"]

    # 1. Negative raw p-value
    s_neg_raw = _make_valid_stat_rows(tasks=["task1", "task2"], seeds=spec.master_seeds, p_raw=-0.01)
    assert _check_inconclusive(s_neg_raw) == "Inconclusive"

    # 2. Negative adjusted p-value
    s_neg_adj = _make_valid_stat_rows(tasks=["task1", "task2"], seeds=spec.master_seeds, p_adj=-0.05)
    assert _check_inconclusive(s_neg_adj) == "Inconclusive"

    # 3. NaN p-value
    s_nan = _make_valid_stat_rows(tasks=["task1", "task2"], seeds=spec.master_seeds)
    s_nan[0]["p_value_raw"] = float("nan")
    assert _check_inconclusive(s_nan) == "Inconclusive"

    # 4. Infinite p-value
    s_inf = _make_valid_stat_rows(tasks=["task1", "task2"], seeds=spec.master_seeds)
    s_inf[0]["p_value_raw"] = float("inf")
    assert _check_inconclusive(s_inf) == "Inconclusive"

    # 5. p-value > 1.0
    s_gt1 = _make_valid_stat_rows(tasks=["task1", "task2"], seeds=spec.master_seeds, p_raw=1.5)
    assert _check_inconclusive(s_gt1) == "Inconclusive"

    # 6. Impossible raw p-value (< 2^(1 - n) - 1e-12)
    impossible_p = 1e-10  # 2^(1-9) = 1/256 = 0.00390625; 1e-10 is impossible for exact permutation
    s_imp = _make_valid_stat_rows(tasks=["task1", "task2"], seeds=spec.master_seeds, p_raw=impossible_p)
    assert _check_inconclusive(s_imp) == "Inconclusive"

    # 7. Wrong comparison type
    s_comp = _make_valid_stat_rows(tasks=["task1", "task2"], seeds=spec.master_seeds)
    s_comp[0]["comparison_type"] = "exploratory"
    assert _check_inconclusive(s_comp) == "Inconclusive"

    # 8. Wrong adjustment method
    s_adj = _make_valid_stat_rows(tasks=["task1", "task2"], seeds=spec.master_seeds)
    s_adj[0]["adjustment_method"] = "Bonferroni"
    assert _check_inconclusive(s_adj) == "Inconclusive"

    # 9. Wrong confirmatory family
    s_fam = _make_valid_stat_rows(tasks=["task1", "task2"], seeds=spec.master_seeds)
    s_fam[0]["confirmatory_family"] = "wrong_family"
    assert _check_inconclusive(s_fam) == "Inconclusive"

    # 10. Wrong statistical method
    s_meth = _make_valid_stat_rows(tasks=["task1", "task2"], seeds=spec.master_seeds)
    s_meth[0]["method"] = "unpaired_t_test"
    assert _check_inconclusive(s_meth) == "Inconclusive"

    # 11. Missing seed in seed_level_differences
    s_seed_miss = _make_valid_stat_rows(tasks=["task1", "task2"], seeds=spec.master_seeds)
    details = json.loads(s_seed_miss[0]["seed_level_differences"])[:-1]  # 8 seeds instead of 9
    s_seed_miss[0]["seed_level_differences"] = json.dumps(details)
    assert _check_inconclusive(s_seed_miss) == "Inconclusive"

    # 12. Duplicate seed in seed_level_differences
    s_seed_dup = _make_valid_stat_rows(tasks=["task1", "task2"], seeds=spec.master_seeds)
    details_dup = json.loads(s_seed_dup[0]["seed_level_differences"])
    details_dup[0]["seed"] = details_dup[1]["seed"]
    s_seed_dup[0]["seed_level_differences"] = json.dumps(details_dup)
    assert _check_inconclusive(s_seed_dup) == "Inconclusive"

    # 13. Extra seed in seed_level_differences
    s_seed_extra = _make_valid_stat_rows(tasks=["task1", "task2"], seeds=spec.master_seeds)
    details_extra = json.loads(s_seed_extra[0]["seed_level_differences"]) + [{"status": "paired", "seed": 99999, "time_a": 2.0, "time_b": 4.0}]
    s_seed_extra[0]["seed_level_differences"] = json.dumps(details_extra)
    assert _check_inconclusive(s_seed_extra) == "Inconclusive"

    # 14. Sample size mismatch (sample_size_n != len(expected_seeds))
    s_n_mismatch = _make_valid_stat_rows(tasks=["task1", "task2"], seeds=spec.master_seeds)
    s_n_mismatch[0]["sample_size_n"] = 5
    assert _check_inconclusive(s_n_mismatch) == "Inconclusive"

    # 15. Inconsistent events + censorings in Kaplan-Meier
    s_km_incons = _make_valid_stat_rows(tasks=["task1", "task2"], seeds=spec.master_seeds)
    s_km_incons[0]["kaplan_meier_a"]["events"] = 99
    assert _check_inconclusive(s_km_incons) == "Inconclusive"

    # 16. Missing Kaplan-Meier structure
    s_km_missing = _make_valid_stat_rows(tasks=["task1", "task2"], seeds=spec.master_seeds)
    s_km_missing[0]["kaplan_meier_a"] = None
    assert _check_inconclusive(s_km_missing) == "Inconclusive"

    # 17. Over-budget seed time
    s_over_budget = _make_valid_stat_rows(tasks=["task1", "task2"], seeds=spec.master_seeds)
    details_ob = json.loads(s_over_budget[0]["seed_level_differences"])
    details_ob[0]["time_a"] = 999.0  # Budget is 10
    s_over_budget[0]["seed_level_differences"] = json.dumps(details_ob)
    assert _check_inconclusive(s_over_budget) == "Inconclusive"

    # 18. Positive time difference (worse time-to-threshold)
    s_pos_time = _make_valid_stat_rows(tasks=["task1", "task2"], seeds=spec.master_seeds, time_diff=2.0)
    assert _check_inconclusive(s_pos_time) == "Inconclusive"

    # 19. Negative yield difference (worse yield)
    s_neg_yield = _make_valid_stat_rows(tasks=["task1", "task2"], seeds=spec.master_seeds, yield_diff=-0.2)
    assert _check_inconclusive(s_neg_yield) == "Inconclusive"


def test_task3_artifact_verification_missing_tampered_split_hash(tmp_path):
    import shutil
    spec = _make_valid_research_spec()
    generator = ReportGenerator(tmp_path)
    stats_dir = tmp_path / "statistics"

    # Helper to regenerate valid state
    def _reset():
        if stats_dir.exists():
            shutil.rmtree(stats_dir)
        return _make_valid_bundle(spec, tmp_path)

    # 1. Base valid bundle succeeds
    artifacts, stats = _reset()
    assert generator._statistical_evidence_complete(stats, artifacts) is True

    # 2. Entire statistics/ directory deleted -> fails closed
    artifacts, stats = _reset()
    shutil.rmtree(stats_dir)
    assert generator._statistical_evidence_complete(stats, artifacts) is False

    # 3. Missing analysis_manifest.json -> fails closed
    artifacts, stats = _reset()
    (stats_dir / "analysis_manifest.json").unlink()
    assert generator._statistical_evidence_complete(stats, artifacts) is False

    # 4. Missing effects.csv -> fails closed
    artifacts, stats = _reset()
    (stats_dir / "effects.csv").unlink()
    assert generator._statistical_evidence_complete(stats, artifacts) is False

    # 5. Missing results.json -> fails closed
    artifacts, stats = _reset()
    (stats_dir / "results.json").unlink()
    assert generator._statistical_evidence_complete(stats, artifacts) is False

    # 6. Tampered spec_hash in on-disk manifest -> fails closed
    artifacts, stats = _reset()
    manifest = json.loads((stats_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
    manifest["spec_hash"] = "0" * 64
    (stats_dir / "analysis_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert generator._statistical_evidence_complete(stats, artifacts) is False

    # 7. Tampered experiment_id in on-disk manifest -> fails closed
    artifacts, stats = _reset()
    manifest = json.loads((stats_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
    manifest["experiment_id"] = "different_exp"
    (stats_dir / "analysis_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert generator._statistical_evidence_complete(stats, artifacts) is False

    # 8. On-disk manifest tampered while valid in-memory manifest is supplied -> in-memory CANNOT bypass disk
    artifacts, stats = _reset()
    manifest_disk = json.loads((stats_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
    manifest_disk["spec_hash"] = "0" * 64
    (stats_dir / "analysis_manifest.json").write_text(json.dumps(manifest_disk), encoding="utf-8")
    # Pass valid manifest in artifacts
    assert generator._statistical_evidence_complete(stats, artifacts) is False

    # 9. Split-hash forgery: genuine results.json on disk, altered in-memory rows with canonical_results_sha256 matching in-memory rows -> fails closed
    artifacts, stats = _reset()
    altered_in_memory = [dict(s) for s in stats]
    altered_in_memory[0]["mean_difference"] = 999.0
    altered_canonical_hash = hashlib.sha256(
        json.dumps(altered_in_memory, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    # If caller forged the in-memory manifest canonical_results_sha256
    artifacts_forged = dict(artifacts)
    artifacts_forged["analysis_manifest"] = dict(artifacts["analysis_manifest"])
    artifacts_forged["analysis_manifest"]["artifacts"]["canonical_results_sha256"] = altered_canonical_hash
    assert generator._statistical_evidence_complete(altered_in_memory, artifacts_forged) is False

    # 10. Tampered effects.csv content on disk -> fails closed
    artifacts, stats = _reset()
    (stats_dir / "effects.csv").write_bytes(b"tampered,data\n")
    assert generator._statistical_evidence_complete(stats, artifacts) is False

    # 11. Tampered results.json content on disk -> fails closed
    artifacts, stats = _reset()
    (stats_dir / "results.json").write_bytes(b"[{\"tampered\": true}]")
    assert generator._statistical_evidence_complete(stats, artifacts) is False

    # 12. effects.csv rows mismatch with results.json -> fails closed
    artifacts, stats = _reset()
    fields = [
        "analysis_version", "target_task", "metric_name", "condition_a", "condition_b",
        "comparison_type", "sample_size_n", "missing_count", "mean_a", "mean_b",
        "mean_difference", "median_difference", "ci_95_lower", "ci_95_upper",
        "p_value_raw", "p_value_adjusted", "adjustment_method", "method", "status",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in stats:
        r = dict(row)
        r["mean_difference"] = 0.0  # Mismatch with results.json
        writer.writerow({k: "" if r.get(k) is None else str(r.get(k)) for k in fields})
    csv_bytes = buf.getvalue().encode("utf-8")
    (stats_dir / "effects.csv").write_bytes(csv_bytes)
    manifest = json.loads((stats_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"]["effects.csv"] = hashlib.sha256(csv_bytes).hexdigest()
    (stats_dir / "analysis_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    artifacts["analysis_manifest"] = manifest
    assert generator._statistical_evidence_complete(stats, artifacts) is False

    # 13. Altering median_difference to 999999 in effects.csv -> fails canonical byte-for-byte check
    artifacts, stats = _reset()
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for i, row in enumerate(stats):
        r = dict(row)
        if i == 0:
            r["median_difference"] = 999999.0
        writer.writerow({k: "" if r.get(k) is None else str(r.get(k)) for k in fields})
    csv_bytes = buf.getvalue().encode("utf-8")
    (stats_dir / "effects.csv").write_bytes(csv_bytes)
    manifest = json.loads((stats_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"]["effects.csv"] = hashlib.sha256(csv_bytes).hexdigest()
    (stats_dir / "analysis_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    artifacts["analysis_manifest"] = manifest
    assert generator._statistical_evidence_complete(stats, artifacts) is False

    # 14. Omission of C2/C3 hypothesis rows (only C1 rows present) -> fails complete Holm family validation
    artifacts, stats = _reset()
    c1_only_stats = [
        r for r in stats
        if r.get("condition_b") == "adaptive_no_memory"
    ]
    # Re-write on-disk files with C1 only
    results_bytes = json.dumps(c1_only_stats, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    (stats_dir / "results.json").write_bytes(results_bytes)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in c1_only_stats:
        writer.writerow({k: "" if row.get(k) is None else str(row.get(k)) for k in fields})
    csv_bytes = buf.getvalue().encode("utf-8")
    (stats_dir / "effects.csv").write_bytes(csv_bytes)
    manifest = json.loads((stats_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"]["effects.csv"] = hashlib.sha256(csv_bytes).hexdigest()
    manifest["artifacts"]["results.json"] = hashlib.sha256(results_bytes).hexdigest()
    manifest["artifacts"]["canonical_results_sha256"] = hashlib.sha256(results_bytes).hexdigest()
    manifest["results"] = c1_only_stats
    (stats_dir / "analysis_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    artifacts["analysis_manifest"] = manifest
    assert generator._statistical_evidence_complete(c1_only_stats, artifacts) is False

    # 15. Fabricated / un-adjusted p_value_adjusted in confirmatory family -> fails recomputed Holm adjustment check
    artifacts, stats = _reset()
    tampered_adj_stats = [dict(r) for r in stats]
    tampered_adj_stats[0]["p_value_adjusted"] = 0.001  # Too optimistic compared to Holm adjustment
    results_bytes = json.dumps(tampered_adj_stats, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    (stats_dir / "results.json").write_bytes(results_bytes)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in tampered_adj_stats:
        writer.writerow({k: "" if row.get(k) is None else str(row.get(k)) for k in fields})
    csv_bytes = buf.getvalue().encode("utf-8")
    (stats_dir / "effects.csv").write_bytes(csv_bytes)
    manifest = json.loads((stats_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"]["effects.csv"] = hashlib.sha256(csv_bytes).hexdigest()
    manifest["artifacts"]["results.json"] = hashlib.sha256(results_bytes).hexdigest()
    manifest["artifacts"]["canonical_results_sha256"] = hashlib.sha256(results_bytes).hexdigest()
    manifest["results"] = tampered_adj_stats
    (stats_dir / "analysis_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    artifacts["analysis_manifest"] = manifest
    assert generator._statistical_evidence_complete(tampered_adj_stats, artifacts) is False

    # 16. Development mode spec claiming research mode in preflight -> fails closed (invalid_research)
    from experiments.report import validate_frozen_spec
    from experiments.spec import CANONICAL_RESEARCH_PREFLIGHT_CHECKS, ExperimentSpec, TaskDefinition
    dev_spec = ExperimentSpec(
        "exp_dev_mode",
        run_mode="development",
        target_tasks=[TaskDefinition("t1", ["Li"]), TaskDefinition("t2", ["Na"])],
        master_seeds=[42, 137, 2024, 777, 999, 31415, 27182, 16180, 104729],
    )
    dev_artifacts = {
        "spec": dev_spec.to_dict(),
        "spec_hash": dev_spec.spec_hash,
        "dag": {"experiment_id": dev_spec.experiment_id, "spec_hash": dev_spec.spec_hash},
        "preflight": {
            "status": "PASSED",
            "run_mode": "research",  # Preflight claims research while spec is development
            "spec_hash": dev_spec.spec_hash,
            "required_checks": list(CANONICAL_RESEARCH_PREFLIGHT_CHECKS),
            "checks": {name: True for name in CANONICAL_RESEARCH_PREFLIGHT_CHECKS},
        },
    }
    state_d, eligible_d, _ = generator._research_preflight(dev_artifacts)
    assert state_d == "invalid_research"
    assert eligible_d is False
    v_dev = validate_frozen_spec(dev_artifacts)
    assert v_dev.valid is False
    assert any("run_mode_mismatch" in f for f in v_dev.failures)


def test_task4_transferable_memory_cartesian_and_fractional():
    from agents.transferable_memory import _coordination_summary
    import numpy as np

    lattice_matrix = [[5.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.3, 3.0]]

    # Site 1: [0.0, 0.0, 0.0]
    # Site 2: fractional [0.4, 0.5, 0.2]
    # Expected Cartesian site 2:
    # x = 0.4 * 5.0 = 2.0
    # y = 0.5 * 4.0 = 2.0
    # z = 0.5 * 0.3 + 0.2 * 3.0 = 0.15 + 0.6 = 0.75
    # Exact Cartesian distance: sqrt(2^2 + 2^2 + 0.75^2) = sqrt(8.5625) ~ 2.92617497426

    # 1. Fractional coordinates with boolean flag
    struct_frac = {
        "composition": "Li2",
        "lattice": {"matrix": lattice_matrix},
        "positions": [[0.0, 0.0, 0.0], [0.4, 0.5, 0.2]],
        "fractional_coordinates": True,
    }
    coord_frac, err_frac = _coordination_summary(struct_frac)
    assert err_frac is None
    assert coord_frac["method"] == "provided_fractional_distance_v1"

    # 2. Equivalent Cartesian coordinates with boolean flag
    struct_cart = {
        "composition": "Li2",
        "lattice": {"matrix": lattice_matrix},
        "positions": [[0.0, 0.0, 0.0], [2.0, 2.0, 0.75]],
        "fractional_coordinates": False,
    }
    coord_cart, err_cart = _coordination_summary(struct_cart)
    assert err_cart is None
    assert coord_cart["method"] == "provided_cartesian_distance_v1"

    # Both coordinate representations yield the exact same coordination numbers
    assert coord_frac["coordination_number_min"] == coord_cart["coordination_number_min"]
    assert coord_frac["coordination_number_max"] == coord_cart["coordination_number_max"]
    assert coord_frac["coordination_number_mean"] == coord_cart["coordination_number_mean"]
    assert coord_frac["coordination_number_counts"] == coord_cart["coordination_number_counts"]

    # 3. Explicit frac_coords and cart_coords
    struct_explicit_cart = {
        "composition": "Li2",
        "lattice": {"matrix": lattice_matrix},
        "cart_coords": [[0.0, 0.0, 0.0], [2.0, 2.0, 0.75]],
    }
    coord_exp_cart, _ = _coordination_summary(struct_explicit_cart)
    assert coord_exp_cart["method"] == "provided_cartesian_distance_v1"
    assert coord_exp_cart["coordination_number_mean"] == coord_frac["coordination_number_mean"]

    struct_explicit_frac = {
        "composition": "Li2",
        "lattice": {"matrix": lattice_matrix},
        "frac_coords": [[0.0, 0.0, 0.0], [0.4, 0.5, 0.2]],
    }
    coord_exp_frac, _ = _coordination_summary(struct_explicit_frac)
    assert coord_exp_frac["method"] == "provided_fractional_distance_v1"
    assert coord_exp_frac["coordination_number_mean"] == coord_frac["coordination_number_mean"]

    # 4. Numpy boolean flags
    struct_np_bool = {
        "composition": "Li2",
        "lattice": {"matrix": lattice_matrix},
        "positions": [[0.0, 0.0, 0.0], [2.0, 2.0, 0.75]],
        "fractional_coordinates": np.bool_(False),
    }
    coord_np, _ = _coordination_summary(struct_np_bool)
    assert coord_np["method"] == "provided_cartesian_distance_v1"
    assert coord_np["coordination_number_mean"] == coord_frac["coordination_number_mean"]


def test_dictionary_coordinate_schemas_in_geometry_and_thermodynamics():
    from agents.geometry import GeometryValidator
    from agents.thermodynamics import _composition_and_count
    import numpy as np

    validator = GeometryValidator(min_distance=0.8)

    # 1. Boolean fractional_coordinates=True
    struct_bool_true = {
        "composition": "Li2",
        "lattice": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        "positions": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        "species": ["Li", "Li"],
        "fractional_coordinates": True,
    }
    geom_res = validator.validate(struct_bool_true)
    assert geom_res.valid is True
    formula, count = _composition_and_count(struct_bool_true)
    assert count == 2.0

    # 2. Boolean fractional_coordinates=False (Cartesian positions)
    struct_bool_false = {
        "composition": "Li2",
        "lattice": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        "positions": [[0.0, 0.0, 0.0], [2.5, 2.5, 2.5]],
        "species": ["Li", "Li"],
        "fractional_coordinates": False,
    }
    geom_res_cart = validator.validate(struct_bool_false)
    assert geom_res_cart.valid is True

    # 3. Numpy bool flag
    struct_np = {
        "composition": "Li2",
        "lattice": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        "positions": [[0.0, 0.0, 0.0], [2.5, 2.5, 2.5]],
        "species": ["Li", "Li"],
        "fractional_coordinates": np.bool_(False),
    }
    assert validator.validate(struct_np).valid is True
    formula, count = _composition_and_count(struct_np)
    assert count == 2.0

    # 4. Invalid coordinate shapes still failing
    struct_bad_shape = {
        "composition": "Li2",
        "lattice": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        "positions": [[0.0, 0.0]],  # 2D coords
        "species": ["Li", "Li"],
    }
    assert validator.validate(struct_bad_shape).valid is False

    # 5. Species count mismatch still failing
    struct_count_mismatch = {
        "composition": "Li2",
        "lattice": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        "positions": [[0.0, 0.0, 0.0]],  # only 1 coordinate
        "species": ["Li", "Li"],  # 2 species declared
        "fractional_coordinates": True,
    }
    res_mismatch = validator.validate(struct_count_mismatch)
    assert res_mismatch.valid is False
    assert res_mismatch.code == "SITE_COUNT_MISMATCH"
