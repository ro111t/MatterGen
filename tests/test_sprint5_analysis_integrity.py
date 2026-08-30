"""Adversarial tests for the repaired Sprint 5 analysis boundary."""

import csv
import hashlib
import json
from pathlib import Path

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
