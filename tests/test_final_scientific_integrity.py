"""Final adversarial checks for scientific-order and QE provenance gates."""

import csv
import hashlib
import json

import pytest

from agents.budget import DualBudgetTracker
from agents.orchestrator import OrchestratorAgent
from agents.screening import ScreeningAgent
from experiments.metrics import compute_run_metrics
from experiments.qe_audit import (
    QEAuditCandidate,
    QEAuditError,
    QEAuditRunner,
    QEResultRecord,
    QECalculationStatus,
    select_audit_candidates,
)
from experiments.spec import QEAuditConfig


def _structure(candidate_id="c"):
    return {
        "candidate_id": candidate_id,
        "composition": "Li",
        "lattice": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        "positions": [[0.0, 0.0, 0.0]],
    }


def _qe_structure(candidate_id="c"):
    value = _structure(candidate_id)
    value.update({"species": ["Li"], "fractional_coordinates": True})
    return value


def test_screening_persists_actual_oracle_admission_order(monkeypatch):
    monkeypatch.setattr("agents.screening.HAS_CHGNET", False)
    screener = ScreeningAgent()
    screener._predict = lambda struct, sid: {
        "formation_energy": -1.0, "band_gap": 1.0, "stability_score": 0.5,
    }
    tracker = DualBudgetTracker(oracle_budget=3)
    results = screener.screen_batch(
        [_structure("priority-third"), _structure("priority-first"), _structure("priority-second")],
        {}, deduplicate=False, budget_tracker=tracker,
    )
    assert [result.oracle_call_index for _, result in results] == [1, 2, 3]


def test_metrics_never_infers_oracle_order_from_event_list_order():
    payload = {
        "manifest": {"status": "completed", "proposals_generated": 1, "oracle_evaluations": 1},
        "oracle_events": [{"candidate_id": "c"}],
        "candidates": [{
            "candidate_id": "c", "composition": "Li", "geometry_valid": True,
            "oracle_evaluated": True,
            "screening_predictions": {"predicted_energy_above_hull_ev_per_atom": 0.01},
        }],
    }
    metrics, records = compute_run_metrics(payload, "r", "target", "adaptive_no_memory", 1, 1)
    assert records[0].oracle_call_index is None
    assert "oracle_call_index" in records[0].provenance_missing_fields
    assert metrics.oracle_calls_to_first_candidate_at_or_below_0_10 is None
    assert metrics.primary_endpoint_censored is True


def test_metrics_rejects_gapped_oracle_order_for_all_order_dependent_metrics():
    payload = {
        "manifest": {"status": "completed", "proposals_generated": 2, "oracle_evaluations": 2},
        "candidates": [
            {"candidate_id": "late", "composition": "Li", "geometry_valid": True,
             "oracle_evaluated": True, "oracle_call_index": 3,
             "screening_predictions": {"predicted_energy_above_hull_ev_per_atom": 0.01}},
            {"candidate_id": "early", "composition": "Li", "geometry_valid": True,
             "oracle_evaluated": True, "oracle_call_index": 1,
             "screening_predictions": {"predicted_energy_above_hull_ev_per_atom": 0.20}},
        ],
    }
    metrics, _ = compute_run_metrics(payload, "r", "target", "adaptive_no_memory", 1, 3)
    assert metrics.oracle_calls_to_first_candidate_at_or_below_0_10 is None
    assert metrics.best_energy_above_hull_overall is None
    assert metrics.area_under_best_curve is None
    assert metrics.provenance_complete is False
    assert metrics.oracle_order_source == "incomplete_oracle_call_index"


def test_invalid_shuffled_audit_cannot_change_policy():
    strategy = {"diversity_weight": 0.4}
    transfer = {
        "control_only": True,
        "shuffle_audit": {"valid": False, "fixed_points": 1},
        "directives": [{"record_id": "r", "exploration_weight": 0.8}],
    }
    result = OrchestratorAgent._apply_transfer_policy(strategy, transfer)
    assert result["diversity_weight"] == 0.4
    assert result["memory_policy"]["applied"] == []
    assert result["memory_policy"]["control_only"] is True


class _PinnedCalculator:
    executable_version = "qe-test-1"

    def run_calculation(self, formula, structure, config, calc_dir, *, is_candidate=True):
        calc_dir.mkdir(parents=True, exist_ok=True)
        energy = -1.0
        return QEResultRecord(
            calculation_id=formula, formula=formula, is_candidate=is_candidate,
            status=QECalculationStatus.CONVERGED.value, total_energy_ev=energy,
            num_atoms=len(structure["species"]), energy_per_atom_ev=energy / len(structure["species"]),
            scf_steps=1, relaxation_steps=1, max_force_ev_per_ang=0.01,
            calculation_dir=str(calc_dir), input_hash="a" * 64,
            result_hash="b" * 64, output_hash="c" * 64,
            executable="pw.x", executable_version=self.executable_version,
            sssp_manifest_sha256="d" * 64,
        )


def test_qe_selection_rejects_nonfinite_and_outputs_bound_provenance(tmp_path):
    invalid = {
        "candidate_id": "nan", "task_id": "target", "condition": "adaptive_no_memory",
        "seed": 1, "oracle_evaluated": True, "oracle_success": True,
        "predicted_energy_above_hull_ev_per_atom": float("nan"),
        "reduced_formula": "Li", "structure": _qe_structure(),
    }
    assert not select_audit_candidates([invalid], target_count=1)

    candidate = QEAuditCandidate(
        "c", "target", "adaptive_no_memory", 1, "Li", _qe_structure(), 0.0,
        [{"formula": "Li", "amount": 1.0, "structure": _qe_structure("phase")}], 1, "test",
    )
    out = tmp_path / "qe"
    out.mkdir()
    (out / "selection.json").write_text(
        json.dumps({"candidates": [candidate.to_dict()], "insufficiency": None}),
        encoding="utf-8",
    )
    results = QEAuditRunner(
        QEAuditConfig(mock_execution=True), out, calculator=_PinnedCalculator(),
    ).run_full_audit([candidate])
    assert results[0].candidate_result_hash == "b" * 64
    row = next(csv.DictReader((out / "results.csv").open(encoding="utf-8")))
    assert row["candidate_input_hash"] == "a" * 64
    assert json.loads(row["participating_phase_results"])[0]["output_hash"] == "c" * 64
    manifest = json.loads((out / "audit_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"]["results.csv"] == hashlib.sha256((out / "results.csv").read_bytes()).hexdigest()
    assert manifest["artifacts"]["selection.json"] == hashlib.sha256((out / "selection.json").read_bytes()).hexdigest()
    canonical = json.dumps(
        manifest["result_provenance"], sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    assert manifest["artifacts"]["canonical_results_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert json.loads(row["candidate_provenance"])["result"]["result_hash"] == "b" * 64
    assert json.loads(row["participating_phase_results"])[0]["phase_index"] == 0
    assert not list(out.glob("*.tmp"))

    # The manifest is an integrity binding, not merely descriptive metadata:
    # a post-run artifact mutation is detectable from its recorded digest.
    (out / "results.csv").write_text(
        (out / "results.csv").read_text(encoding="utf-8") + "tampered\n", encoding="utf-8",
    )
    assert manifest["artifacts"]["results.csv"] != hashlib.sha256((out / "results.csv").read_bytes()).hexdigest()


def test_direct_production_qe_requires_expected_sssp_digest(tmp_path):
    pseudo = tmp_path / "Li.upf"
    pseudo.write_text("pseudo", encoding="utf-8")
    manifest_path = tmp_path / "sssp.json"
    manifest_path.write_text(json.dumps({"pseudopotentials": {"Li": {
        "path": pseudo.name,
        "sha256": hashlib.sha256(pseudo.read_bytes()).hexdigest(),
    }}}), encoding="utf-8")
    with pytest.raises(QEAuditError, match="expected SHA256"):
        QEAuditRunner(
            QEAuditConfig(mock_execution=False, sssp_manifest_path=str(manifest_path)),
            tmp_path / "out", calculator=_PinnedCalculator(),
        )


def test_direct_production_qe_rejects_mismatched_sssp_digest(tmp_path):
    pseudo = tmp_path / "Li.upf"
    pseudo.write_text("pseudo", encoding="utf-8")
    manifest_path = tmp_path / "sssp.json"
    manifest_path.write_text(json.dumps({"pseudopotentials": {"Li": {
        "path": pseudo.name,
        "sha256": hashlib.sha256(pseudo.read_bytes()).hexdigest(),
    }}}, sort_keys=True), encoding="utf-8")
    with pytest.raises(QEAuditError, match="SSSP manifest SHA256 mismatch"):
        QEAuditRunner(
            QEAuditConfig(
                mock_execution=False,
                sssp_manifest_path=str(manifest_path),
                sssp_manifest_sha256="0" * 64,
            ),
            tmp_path / "out", calculator=_PinnedCalculator(),
        )


def test_qe_audit_rejects_selection_candidate_mismatch(tmp_path):
    candidate = QEAuditCandidate(
        "expected", "target", "adaptive_no_memory", 1, "Li", _structure(), 0.0,
        [{"formula": "Li", "amount": 1.0, "structure": _structure("phase")}], 1, "test",
    )
    selected_on_disk = candidate.to_dict()
    selected_on_disk["candidate_id"] = "different"
    out = tmp_path / "qe"
    out.mkdir()
    (out / "selection.json").write_text(
        json.dumps({"candidates": [selected_on_disk], "insufficiency": None}),
        encoding="utf-8",
    )
    with pytest.raises(QEAuditError, match="candidates do not match"):
        QEAuditRunner(
            QEAuditConfig(mock_execution=True), out, calculator=_PinnedCalculator(),
        ).run_full_audit([candidate])


def test_qe_selection_excludes_positive_infinity_and_negative_infinity():
    rows = []
    for candidate_id, energy in (("pos-inf", float("inf")), ("neg-inf", float("-inf"))):
        rows.append({
            "candidate_id": candidate_id, "task_id": "target", "condition": "adaptive_no_memory",
            "seed": 1, "oracle_evaluated": True, "oracle_success": True,
            "predicted_energy_above_hull_ev_per_atom": energy,
            "reduced_formula": "Li", "structure": _structure(candidate_id),
        })
    selected = select_audit_candidates(rows, target_count=1)
    assert selected == []
    assert selected.insufficiency == "insufficient_valid_target_candidates:0/1"
