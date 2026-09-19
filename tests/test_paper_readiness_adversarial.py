"""Adversarial tests for scientific correctness, budgets, censoring, resume, and provenance."""

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
import pytest
import numpy as np

from agents.budget import DualBudgetTracker
from agents.geometry import closest_lattice_image
from agents.provenance import CandidateStatus, CandidateRecord, ProvenanceTracker
from agents.thermodynamics import (
    ModelIdentity,
    ReferencePhaseInput,
    ReferenceSetError,
    RelaxationSettings,
    build_frozen_reference_set as _build_frozen_reference_set,
    load_frozen_reference_set,
    sha256_payload,
    write_frozen_reference_set,
)
from agents.transferable_memory import extract_transferable_features
from campaign import CampaignConfig, MaterialsDiscoveryCampaign
from experiments.dag import DAGNode, DAGValidationError, ExperimentDAG, NodeType
from experiments.metrics import compute_run_metrics
from experiments.qe_audit import (
    FakeQECalculator,
    QEAuditCandidate,
    QEAuditError,
    QEAuditRunner,
    QECalculationStatus,
    QEResultRecord,
)
from experiments.report import ReportGenerator
from experiments.runner import CampaignRunner
from experiments.spec import (
    FIVE_CONDITIONS,
    ExperimentSpec,
    QEAuditConfig,
    TaskDefinition,
    TransferDeclaration,
)

MODEL = ModelIdentity("fake-chgnet", "0.3.0", "a" * 64)
SETTINGS = RelaxationSettings(fmax_ev_per_angstrom=0.05, max_steps=500, relax_cell=True)


def _coverage_manifest(inputs, chemical_system):
    from pymatgen.core import Composition

    items = list(inputs)
    endpoints = [item.source_id for item in items if len(Composition(item.structure["composition"]).elements) == 1]
    compounds = [item.source_id for item in items if len(Composition(item.structure["composition"]).elements) >= 2]
    return {
        "manifest_schema_version": "1.0.0",
        "source_dataset": "unit-test-fixture",
        "dataset_version": "fixed-v1",
        "snapshot_digest": "2" * 64,
        "chemical_system": sorted(set(chemical_system)),
        "selection_procedure": "all declared unit-test phases",
        "expected_source_phase_ids": [item.source_id for item in items],
        "elemental_endpoint_ids": endpoints,
        "required_compounds": compounds,
    }


def build_frozen_reference_set(*, inputs, chemical_system, source_selection=None, **kwargs):
    items = list(inputs)
    return _build_frozen_reference_set(
        inputs=items,
        chemical_system=chemical_system,
        source_selection=(
            _coverage_manifest(items, chemical_system)
            if source_selection is None else source_selection
        ),
        **kwargs,
    )


class MockEvaluator:
    model_identity = MODEL
    relaxation_settings = SETTINGS

    def __init__(self, energies):
        self.energies = dict(energies)

    def relax(self, value):
        key = value.get("candidate_id", value.get("composition", "Li"))
        return {
            "converged": True,
            "relaxed_structure": value,
            "energy_per_atom_ev": self.energies.get(key, -1.0),
            "max_force_ev_per_angstrom": 0.01,
            "max_stress_gpa": 0.02,
        }


def _struct(species, candidate_id=None):
    return {
        "candidate_id": candidate_id or "".join(species),
        "composition": "".join(species),
        "lattice": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        "positions": [[0.0, 0.0, i / max(1, len(species))] for i in range(len(species))],
        "species": list(species),
        "fractional_coordinates": True,
    }


def _skew_cell():
    # Monoclinic/triclinic skew cell
    return {
        "lattice": [[5.0, 0.0, 0.0], [2.0, 5.0, 0.0], [1.0, 1.0, 5.0]],
        "positions": [[0.1, 0.2, 0.3], [0.8, 0.9, 0.7]],
        "species": ["Li", "P"],
        "fractional_coordinates": True,
    }


# =========================================================================
# Phase 1 & 5: Reference-set digest contract and non-vacuous certification
# =========================================================================

def test_frozen_reference_set_canonical_byte_digest(tmp_path):
    inputs = [
        ReferencePhaseInput(source_id="Li-ref", structure=_struct(["Li"], "Li-ref")),
        ReferencePhaseInput(source_id="P-ref", structure=_struct(["P"], "P-ref")),
        ReferencePhaseInput(source_id="Li3P-ref", structure=_struct(["Li", "Li", "Li", "P"], "Li3P-ref"), source_energy_above_hull_ev_per_atom=0.0),
    ]
    evaluator = MockEvaluator({"Li-ref": -1.0, "P-ref": -2.0, "Li3P-ref": -2.5})
    json_path = tmp_path / "ref_set.json"
    ref_set = build_frozen_reference_set(
        reference_set_id="Li-P-test",
        chemical_system=["Li", "P"],
        inputs=inputs,
        evaluator=evaluator,
        output_path=json_path,
        created_at_iso="2026-08-30T00:00:00+00:00",
    )
    assert ref_set.certification.certified is True
    assert ref_set.certification.compound_coverage_valid is True

    file_bytes = json_path.read_bytes()
    expected_digest = hashlib.sha256(file_bytes).hexdigest()

    assert ref_set.reference_set_hash == expected_digest
    sidecar_path = json_path.with_suffix(json_path.suffix + ".sha256")
    assert sidecar_path.read_text(encoding="ascii").strip() == expected_digest

    loaded = load_frozen_reference_set(json_path, expected_sha256=expected_digest)
    assert loaded.reference_set_hash == expected_digest


def test_frozen_reference_set_rejects_binary_without_compounds(tmp_path):
    # Only elemental phases for a binary system: must fail certification
    inputs = [
        ReferencePhaseInput(source_id="Li-ref", structure=_struct(["Li"], "Li-ref")),
        ReferencePhaseInput(source_id="P-ref", structure=_struct(["P"], "P-ref")),
    ]
    evaluator = MockEvaluator({"Li-ref": -1.0, "P-ref": -2.0})
    json_path = tmp_path / "ref_no_compounds.json"
    ref_set = build_frozen_reference_set(
        reference_set_id="Li-P-no-compounds",
        chemical_system=["Li", "P"],
        inputs=inputs,
        evaluator=evaluator,
        output_path=json_path,
        created_at_iso="2026-08-30T00:00:00+00:00",
    )
    assert ref_set.certification.certified is False
    assert ref_set.certification.compound_coverage_valid is False
    with pytest.raises(ReferenceSetError, match="not certified"):
        load_frozen_reference_set(json_path, require_certified=True)


def test_frozen_reference_set_rejects_missing_expected_source_phases(tmp_path):
    inputs = [
        ReferencePhaseInput(source_id="mp-1", structure=_struct(["Li"], "mp-1")),
        ReferencePhaseInput(source_id="mp-2", structure=_struct(["P"], "mp-2")),
        ReferencePhaseInput(source_id="mp-3", structure=_struct(["Li", "Li", "Li", "P"], "mp-3"), source_energy_above_hull_ev_per_atom=0.0),
    ]
    evaluator = MockEvaluator({"mp-1": -1.0, "mp-2": -2.0, "mp-3": -2.5})
    source_selection = {
        "source_dataset": "MaterialsProject",
        "expected_source_phase_ids": ["mp-1", "mp-2", "mp-3", "mp-999_missing"],
    }
    json_path = tmp_path / "ref_missing.json"
    ref_set = build_frozen_reference_set(
        reference_set_id="Li-P-missing",
        chemical_system=["Li", "P"],
        inputs=inputs,
        evaluator=evaluator,
        output_path=json_path,
        created_at_iso="2026-08-30T00:00:00+00:00",
        source_selection=source_selection,
    )
    assert ref_set.certification.certified is False
    assert "mp-999_missing" in ref_set.certification.missing_expected_source_phases
    with pytest.raises(ReferenceSetError, match="not certified"):
        load_frozen_reference_set(json_path, require_certified=True)


# =========================================================================
# Phase 2: Budgets, exhaustion, and shortfall
# =========================================================================

def test_campaign_continues_generating_after_oracle_exhaustion(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.screening.HAS_CHGNET", False)
    from agents.orchestrator import CampaignObjective
    objective = CampaignObjective(
        target_properties={}, constraints={"elements": ["Li", "P"]},
        success_criteria={"min_score": 999}, domain="budget_test", max_iterations=2,
    )
    # Prepare mock reference set
    ref_inputs = [
        ReferencePhaseInput(source_id="Li-ref", structure=_struct(["Li"], "Li-ref")),
        ReferencePhaseInput(source_id="P-ref", structure=_struct(["P"], "P-ref")),
        ReferencePhaseInput(source_id="Li3P-ref", structure=_struct(["Li", "Li", "Li", "P"], "Li3P-ref"), source_energy_above_hull_ev_per_atom=0.0),
    ]
    ref_path = tmp_path / "ref.json"
    evaluator = MockEvaluator({"Li-ref": -1.0, "P-ref": -2.0, "Li3P-ref": -2.5})
    build_frozen_reference_set(
        reference_set_id="ref",
        chemical_system=["Li", "P"],
        inputs=ref_inputs,
        evaluator=evaluator,
        output_path=ref_path,
        created_at_iso="2026-08-30T00:00:00+00:00",
    )

    # 6 proposals total, oracle budget of 2
    config = CampaignConfig(
        name="test_oracle_exhaust",
        objective=objective,
        output_dir=tmp_path,
        proposal_budget=6,
        oracle_budget=2,
        num_candidates=3,
        thermodynamics_reference_set_path=str(ref_path),
        thermodynamics_evaluator=evaluator,
        geometry_min_distance=0.01,
        use_career_memory=False,
        use_validation=False,
        use_synthesis=False,
        verbose=False,
    )
    campaign = MaterialsDiscoveryCampaign(config=config)
    campaign.run_campaign()

    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["proposals_generated"] == 6
    assert manifest["oracle_evaluations"] == 2
    assert manifest["termination_reason"] in ("PROPOSAL_BUDGET_EXHAUSTED", "MAX_ITERATIONS_REACHED")

    prov_path = tmp_path / "campaign_provenance.json"
    prov_data = json.loads(prov_path.read_text(encoding="utf-8"))
    candidates = prov_data["candidates"]
    assert len(candidates) == 6
    oracle_evaluated = [c for c in candidates if c.get("oracle_evaluated") is True]
    assert len(oracle_evaluated) == 2
    rejected_budget = [
        c for c in candidates
        if c.get("rejection_stage") == "oracle_budget"
        or c.get("geometry_failure_code") == "ORACLE_BUDGET_EXHAUSTED"
        or c.get("rejection_reason") == "ORACLE_BUDGET_EXHAUSTED"
    ]
    assert len(rejected_budget) == 4


def test_generator_overproduction_fails_closed(tmp_path):
    tracker = DualBudgetTracker(proposal_budget=5, oracle_budget=5)
    tracker.record_proposals(5)
    assert tracker.proposal_budget_remaining == 0
    with pytest.raises(Exception):
        tracker.record_proposals(1)


# =========================================================================
# Phase 3: Censoring and AUC follow-up
# =========================================================================

def test_censoring_time_is_last_observed_call_not_horizon():
    # 2 oracle evaluations made out of a budget of 10, neither reaches threshold <= 0.10
    manifest_payload = {
        "manifest": {"status": "completed", "proposals_generated": 2, "oracle_evaluations": 2},
        "candidates": [
            {
                "candidate_id": "c1",
                "composition": "Li",
                "geometry_valid": True,
                "oracle_evaluated": True,
                "oracle_call_index": 1,
                "proposal_index": 0,
                "screening_predictions": {"predicted_energy_above_hull_ev_per_atom": 0.35},
            },
            {
                "candidate_id": "c2",
                "composition": "Li",
                "geometry_valid": True,
                "oracle_evaluated": True,
                "oracle_call_index": 2,
                "proposal_index": 1,
                "screening_predictions": {"predicted_energy_above_hull_ev_per_atom": 0.25},
            },
        ],
    }
    metrics, _ = compute_run_metrics(manifest_payload, "r1", "target", "adaptive_no_memory", 1, oracle_budget=10)
    assert metrics.reached_0_10_threshold is False
    assert metrics.primary_endpoint_censored is True
    # Normalized AUC across fixed horizon of 10 calls:
    # step 1: 0.35; step 2: 0.25; steps 3..10: 0.25 -> (0.35 + 0.25 + 8*0.25) / 10 = 0.26
    assert metrics.area_under_best_curve == pytest.approx(0.26)


def test_zero_oracle_calls_yields_incomplete_evidence():
    manifest_payload = {
        "manifest": {"status": "completed", "proposals_generated": 2, "oracle_evaluations": 0},
        "candidates": [
            {
                "candidate_id": "c1",
                "composition": "Li",
                "geometry_valid": False,
                "oracle_evaluated": False,
                "proposal_index": 0,
            },
        ],
    }
    metrics, _ = compute_run_metrics(manifest_payload, "r0", "target", "adaptive_no_memory", 1, oracle_budget=10)
    assert metrics.reached_0_10_threshold is False
    assert metrics.primary_endpoint_censored is True
    assert metrics.oracle_calls_to_first_candidate_at_or_below_0_10 is None
    assert metrics.area_under_best_curve is None


# =========================================================================
# Phase 4: DAG Resume & Hash-Verified Node Skipping
# =========================================================================

def test_dag_persists_and_validates_spec_hash(tmp_path):
    spec = ExperimentSpec(
        experiment_id="test_dag_resume",
        output_root=str(tmp_path / "out"),
        master_seeds=[42],
        conditions=list(FIVE_CONDITIONS),
    )
    dag = ExperimentDAG(spec)
    dag_dict = dag.to_dict()
    assert dag_dict["spec_hash"] == spec.spec_hash

    # Loading with matching spec succeeds
    loaded_dag = ExperimentDAG.from_dict(dag_dict, spec)
    assert len(loaded_dag.nodes) == len(dag.nodes)

    # Tampered spec_hash raises DAGValidationError
    tampered_dict = dict(dag_dict)
    tampered_dict["spec_hash"] = "tampered_hash_00000000000000000000000000000000000000000000000000"
    with pytest.raises(DAGValidationError, match="spec_hash mismatch"):
        ExperimentDAG.from_dict(tampered_dict, spec)


# =========================================================================
# Phase 6: Research QE Executable Provenance
# =========================================================================

def test_research_qe_requires_pinned_sha256(tmp_path):
    config = QEAuditConfig(
        candidate_count=2,
        mock_execution=False,
        qe_executable="pw.x",
        qe_executable_sha256=None,  # Missing pinned sha256
        sssp_manifest_sha256="a" * 64,
    )
    with pytest.raises(QEAuditError, match="qe_executable_sha256"):
        QEAuditRunner(config=config, output_dir=tmp_path, run_mode="research")


def test_research_qe_discrepancy_between_candidate_and_phases(tmp_path):
    config = QEAuditConfig(
        candidate_count=1,
        mock_execution=True,
        qe_executable="pw.x",
        qe_executable_version="7.2",
        qe_executable_sha256="b" * 64,
        sssp_manifest_sha256="c" * 64,
    )
    fake_calc = FakeQECalculator()
    runner = QEAuditRunner(config=config, output_dir=tmp_path, calculator=fake_calc, run_mode="development")

    cand = QEAuditCandidate(
        candidate_id="c_prov",
        target_task="Li-P-Se",
        condition="adaptive_no_memory",
        seed=42,
        reduced_formula="Li3P",
        structure=_struct(["Li", "Li", "Li", "P"], "c_prov"),
        predicted_energy_above_hull_ev_per_atom=0.01,
        predicted_decomposition_products=[
            {"formula": "Li", "amount": 3.0, "structure": _struct(["Li"], "Li-p")},
            {"formula": "P", "amount": 1.0, "structure": _struct(["P"], "P-p")},
        ],
        selection_rank=1,
        selection_reason="prov test",
    )
    res = runner.audit_candidate(cand)
    assert res.qe_executable_sha256 == "b" * 64
    assert res.sssp_manifest_sha256 == "c" * 64


# =========================================================================
# Phase 7: Skew-Cell Transferable Coordination Translation Invariance
# =========================================================================

def test_skew_cell_coordination_translation_invariance():
    struct1 = _skew_cell()
    # Shift fractional positions by integer lattice translation
    struct2 = {
        "lattice": struct1["lattice"],
        "positions": [
            [struct1["positions"][0][0] + 1.0, struct1["positions"][0][1] - 2.0, struct1["positions"][0][2] + 3.0],
            [struct1["positions"][1][0] - 1.0, struct1["positions"][1][1] + 1.0, struct1["positions"][1][2] - 1.0],
        ],
        "species": struct1["species"],
        "fractional_coordinates": True,
    }
    feat1 = extract_transferable_features(struct1)
    feat2 = extract_transferable_features(struct2)

    assert feat1.coordination_summary is not None
    assert feat2.coordination_summary is not None
    assert feat1.coordination_summary["coordination_number_mean"] == feat2.coordination_summary["coordination_number_mean"]
    assert feat1.coordination_summary["coordination_number_counts"] == feat2.coordination_summary["coordination_number_counts"]


# =========================================================================
# Phase 4 & Resume Integrity: End-to-End Resume, Skipping, and Tampering
# =========================================================================

def _build_tiny_spec(root: Path, source_task: Optional[TaskDefinition] = None) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="tiny_resume_test",
        source_task=source_task or TaskDefinition(
            task_id="source_li_p",
            elements=["Li", "P"],
            domain="Li-P",
            target_properties={"band_gap": 2.0},
            constraints={},
        ),
        target_tasks=[
            TaskDefinition(
                task_id="target_li_s",
                elements=["Li", "S"],
                domain="Li-S",
                target_properties={"band_gap": 2.5},
                constraints={},
            )
        ],
        conditions=list(FIVE_CONDITIONS),
        master_seeds=[42],
        transfer_declarations=[TransferDeclaration("source_li_p", "target_li_s", "positive_match", ["Li"])],
        proposals_per_run=2,
        oracle_budget_per_run=2,
        iterations_per_run=1,
        output_root=str(root).replace("\\", "/"),
    )


def test_e2e_pipeline_resume_zero_recomputation_and_artifact_equality(tmp_path, monkeypatch):
    from experiments.cli import execute_full_experiment_pipeline
    spec = _build_tiny_spec(tmp_path / "e2e_out")

    # First run
    res1 = execute_full_experiment_pipeline(spec, force_rerun=True)
    assert res1["status"] == "SUCCESS"

    out_dir = Path(spec.output_root)
    # Collect artifact hashes from first run
    files_to_check = [
        out_dir / "experiment_dag.json",
        out_dir / "preflight.json",
        out_dir / "aggregates" / "runs.json",
        out_dir / "aggregates" / "candidates.json",
        out_dir / "statistics" / "analysis_manifest.json",
        out_dir / "qe_audit" / "results.csv",
        out_dir / "paper" / "claim_evidence_matrix.md",
        out_dir / "paper" / "reproducibility_checklist.md",
    ]
    original_hashes = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files_to_check if f.exists()}

    # Instrument CampaignRunner and QEAuditRunner to ensure zero recomputation
    campaign_calls = []
    orig_exec = CampaignRunner.execute_run
    def instrumented_run(run_spec, force_rerun=False):
        campaign_calls.append(run_spec.run_id)
        return orig_exec(run_spec, force_rerun=force_rerun)
    monkeypatch.setattr(CampaignRunner, "execute_run", instrumented_run)

    qe_calls = []
    orig_qe = QEAuditRunner.run_full_audit
    def instrumented_qe(self, candidates):
        qe_calls.append(len(candidates))
        return orig_qe(self, candidates)
    monkeypatch.setattr(QEAuditRunner, "run_full_audit", instrumented_qe)

    # Second invocation (resumed)
    res2 = execute_full_experiment_pipeline(spec, force_rerun=False)
    assert res2["status"] == "SUCCESS"
    assert len(campaign_calls) == 0, f"Expected 0 campaign recomputations on resume, got {campaign_calls}"
    assert len(qe_calls) == 0, f"Expected 0 QE recomputations on resume, got {qe_calls}"

    # Verify all artifacts are byte-for-byte identical
    for f_str, orig_hash in original_hashes.items():
        assert hashlib.sha256(Path(f_str).read_bytes()).hexdigest() == orig_hash


def test_resume_fails_when_campaign_provenance_missing(tmp_path):
    from experiments.cli import execute_full_experiment_pipeline
    spec = _build_tiny_spec(tmp_path / "prov_missing_out")
    execute_full_experiment_pipeline(spec, force_rerun=True)

    out_dir = Path(spec.output_root)
    # Remove campaign_provenance.json in one target run directory while leaving manifest.json intact
    prov_files = list((out_dir / "runs").rglob("campaign_provenance.json"))
    assert prov_files, "Expected target run directory to exist"
    prov_file = prov_files[0]
    prov_file.unlink()

    with pytest.raises(RuntimeError, match="integrity verification|missing"):
        execute_full_experiment_pipeline(spec, force_rerun=False)


def test_resume_fails_when_campaign_provenance_modified(tmp_path):
    from experiments.cli import execute_full_experiment_pipeline
    spec = _build_tiny_spec(tmp_path / "prov_mod_out")
    execute_full_experiment_pipeline(spec, force_rerun=True)

    out_dir = Path(spec.output_root)
    prov_files = list((out_dir / "runs").rglob("campaign_provenance.json"))
    assert prov_files, "Expected target run directory to exist"
    prov_file = prov_files[0]
    prov_file.write_text(prov_file.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(RuntimeError, match="integrity verification|digest mismatch"):
        execute_full_experiment_pipeline(spec, force_rerun=False)


def test_resume_fails_when_audit_manifest_missing(tmp_path):
    from experiments.cli import execute_full_experiment_pipeline
    spec = _build_tiny_spec(tmp_path / "audit_missing_out")
    execute_full_experiment_pipeline(spec, force_rerun=True)

    out_dir = Path(spec.output_root)
    aud_manifest = out_dir / "qe_audit" / "audit_manifest.json"
    aud_manifest.unlink()

    with pytest.raises(RuntimeError, match="audit_manifest|missing"):
        execute_full_experiment_pipeline(spec, force_rerun=False)


def test_resume_fails_when_statistics_artifact_modified(tmp_path):
    from experiments.cli import execute_full_experiment_pipeline
    spec = _build_tiny_spec(tmp_path / "stats_mod_out")
    execute_full_experiment_pipeline(spec, force_rerun=True)

    out_dir = Path(spec.output_root)
    effects_file = out_dir / "statistics" / "effects.csv"
    effects_file.write_text(effects_file.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="digest mismatch"):
        execute_full_experiment_pipeline(spec, force_rerun=False)


def test_dag_persisted_identity_and_spec_hash_validation(tmp_path):
    from experiments.cli import execute_full_experiment_pipeline
    spec = _build_tiny_spec(tmp_path / "dag_ident_out")
    execute_full_experiment_pipeline(spec, force_rerun=True)

    out_dir = Path(spec.output_root)
    dag_file = out_dir / "experiment_dag.json"
    original_bytes = dag_file.read_bytes()

    # 1. Missing spec_hash
    data = json.loads(original_bytes.decode("utf-8"))
    del data["spec_hash"]
    dag_file.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(DAGValidationError, match="spec_hash"):
        execute_full_experiment_pipeline(spec, force_rerun=False)

    # 2. Mismatched spec_hash
    data["spec_hash"] = "0" * 64
    dag_file.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(DAGValidationError, match="spec_hash"):
        execute_full_experiment_pipeline(spec, force_rerun=False)

    # 3. Mismatched experiment_id
    data["spec_hash"] = spec.spec_hash
    data["experiment_id"] = "different_exp_id"
    dag_file.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(DAGValidationError, match="experiment_id"):
        execute_full_experiment_pipeline(spec, force_rerun=False)

    # 4. Malformed JSON
    dag_file.write_text("{malformed json", encoding="utf-8")
    with pytest.raises(DAGValidationError, match="malformed"):
        execute_full_experiment_pipeline(spec, force_rerun=False)

    # 5. Unexpected / missing node IDs
    data = json.loads(original_bytes.decode("utf-8"))
    data["nodes"]["unexpected_evil_node"] = {"node_type": "preflight", "dependencies": []}
    dag_file.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(DAGValidationError, match="node IDs mismatch"):
        execute_full_experiment_pipeline(spec, force_rerun=False)


def test_dag_rejects_redirected_persisted_output_paths(tmp_path):
    spec = _build_tiny_spec(tmp_path / "redirect_out")
    dag = ExperimentDAG(spec)
    dag_dict = dag.to_dict()

    # 1. Redirect fixed node output
    tampered = dict(dag_dict)
    nodes = dict(tampered["nodes"])
    preflight_id = next(nid for nid, n in nodes.items() if n["node_type"] == "preflight")
    preflight_node = dict(nodes[preflight_id])
    preflight_node["expected_output_path"] = "/evil/path/preflight.json"
    nodes[preflight_id] = preflight_node
    tampered["nodes"] = nodes
    with pytest.raises(DAGValidationError, match="persisted output path"):
        ExperimentDAG.from_dict(tampered, spec)

    # 2. Redirect snapshot node output outside expected directory
    tampered = dict(dag_dict)
    nodes = dict(tampered["nodes"])
    snap_node_id = next(nid for nid, n in nodes.items() if n["node_type"] == "source_memory_snapshot")
    snap_node = dict(nodes[snap_node_id])
    snap_node["expected_output_path"] = "/evil/other_dir/source_snap.db"
    nodes[snap_node_id] = snap_node
    tampered["nodes"] = nodes
    with pytest.raises(DAGValidationError, match="persisted snapshot path"):
        ExperimentDAG.from_dict(tampered, spec)


# =========================================================================
# Phase 6: Injected Research QE Calculator & Phase Digest Validation
# =========================================================================

class _InjectedNoDigestCalc:
    executable_version = "7.2"
    called = False

    def run_calculation(self, formula, structure, config, calc_dir, *, is_candidate=True):
        self.called = True
        raise NotImplementedError()


def test_injected_research_calculator_without_digest_fails_construction(tmp_path):
    p_file = tmp_path / "Li.upf"
    p_file.write_bytes(b"pseudo_data")
    sssp_file = tmp_path / "sssp.json"
    sssp_bytes = json.dumps({
        "pseudopotentials": {
            "Li": {"path": str(p_file), "sha256": hashlib.sha256(b"pseudo_data").hexdigest()}
        }
    }).encode("utf-8")
    sssp_file.write_bytes(sssp_bytes)
    config = QEAuditConfig(
        candidate_count=1,
        mock_execution=False,
        qe_executable="pw.x",
        qe_executable_version="7.2",
        qe_executable_sha256="a" * 64,
        sssp_manifest_path=str(sssp_file),
        sssp_manifest_sha256=hashlib.sha256(sssp_bytes).hexdigest(),
    )
    calc = _InjectedNoDigestCalc()
    with pytest.raises(QEAuditError, match="qe_executable_sha256"):
        QEAuditRunner(config=config, output_dir=tmp_path, calculator=calc, run_mode="research")
    assert calc.called is False, "Rejected calculator run_calculation was called"


def test_report_fails_when_participating_phase_missing_executable_digest(tmp_path):
    phase_without_sha = {
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
        "qe_executable": "pw.x", "qe_executable_version": "qe-1",
        "qe_executable_sha256": "e" * 64, "sssp_manifest_sha256": "d" * 64,
        "participating_phase_results": [phase_without_sha],
    }
    qe_dir = tmp_path / "qe_audit"
    qe_dir.mkdir(parents=True, exist_ok=True)
    (qe_dir / "selection.json").write_text(json.dumps({"candidates": [{"candidate_id": "c"}], "insufficiency": None}), encoding="utf-8")

    def _write_csv_and_manifest(cand_row):
        with (qe_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(cand_row))
            writer.writeheader()
            c_row = dict(cand_row)
            c_row["participating_phase_results"] = json.dumps(cand_row["participating_phase_results"], sort_keys=True, separators=(",", ":"))
            writer.writerow(c_row)
        manifest = {
            "run_mode": "research", "mock_execution": False, "selection_count": 1,
            "selection_insufficiency": None, "result_provenance": [cand_row],
            "config": {"qe_executable_sha256": "e" * 64, "sssp_manifest_sha256": "d" * 64},
        }
        manifest["artifacts"] = {
            "selection.json": hashlib.sha256((qe_dir / "selection.json").read_bytes()).hexdigest(),
            "results.csv": hashlib.sha256((qe_dir / "results.csv").read_bytes()).hexdigest(),
            "canonical_results_sha256": hashlib.sha256(json.dumps([cand_row], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        }
        return manifest

    report = ReportGenerator(tmp_path)

    # 1. Phase missing qe_executable_sha256 must fail
    m1 = _write_csv_and_manifest(row)
    assert report._qe_evidence_complete([row], {"expected_qe_candidate_count": 1, "qe_selection_count": 1, "qe_audit": m1}) is False

    # 2. Phase with invalid (short) qe_executable_sha256 must fail
    phase_invalid = dict(phase_without_sha, qe_executable_sha256="short")
    row_invalid = dict(row, participating_phase_results=[phase_invalid])
    m2 = _write_csv_and_manifest(row_invalid)
    assert report._qe_evidence_complete([row_invalid], {"expected_qe_candidate_count": 1, "qe_selection_count": 1, "qe_audit": m2}) is False

    # 3. Phase with mismatched qe_executable_sha256 must fail
    phase_mismatch = dict(phase_without_sha, qe_executable_sha256="f" * 64)
    row_mismatch = dict(row, participating_phase_results=[phase_mismatch])
    m3 = _write_csv_and_manifest(row_mismatch)
    assert report._qe_evidence_complete([row_mismatch], {"expected_qe_candidate_count": 1, "qe_selection_count": 1, "qe_audit": m3}) is False

    # 4. Fully matching phase passes
    phase_valid = dict(phase_without_sha, qe_executable_sha256="e" * 64)
    row_valid = dict(row, participating_phase_results=[phase_valid])
    m4 = _write_csv_and_manifest(row_valid)
    assert report._qe_evidence_complete([row_valid], {"expected_qe_candidate_count": 1, "qe_selection_count": 1, "qe_audit": m4}) is True


# =========================================================================
# Phase 2 & 3: Adaptive Strategy Proposal Budget Allocation & Exhaustion
# =========================================================================

def test_adaptive_strategy_attempting_to_underfill_produces_full_budget(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.screening.HAS_CHGNET", False)
    from agents.orchestrator import CampaignObjective
    objective = CampaignObjective(
        target_properties={}, constraints={"elements": ["Li", "P"]},
        success_criteria={"min_score": 999}, domain="budget_underfill_test", max_iterations=5,
    )
    ref_inputs = [
        ReferencePhaseInput(source_id="Li-ref", structure=_struct(["Li"], "Li-ref")),
        ReferencePhaseInput(source_id="P-ref", structure=_struct(["P"], "P-ref")),
        ReferencePhaseInput(source_id="Li3P-ref", structure=_struct(["Li", "Li", "Li", "P"], "Li3P-ref"), source_energy_above_hull_ev_per_atom=0.0),
    ]
    ref_path = tmp_path / "ref.json"
    evaluator = MockEvaluator({"Li-ref": -1.0, "P-ref": -2.0, "Li3P-ref": -2.5})
    build_frozen_reference_set(
        reference_set_id="ref", chemical_system=["Li", "P"], inputs=ref_inputs,
        evaluator=evaluator, output_path=ref_path, created_at_iso="2026-08-30T00:00:00+00:00",
    )

    config = CampaignConfig(
        name="test_underfill",
        objective=objective,
        output_dir=tmp_path / "underfill_out",
        proposal_budget=200,
        oracle_budget=100,
        thermodynamics_reference_set_path=str(ref_path),
        thermodynamics_evaluator=evaluator,
        geometry_min_distance=0.01,
        use_career_memory=False,
        use_validation=False,
        use_synthesis=False,
        verbose=False,
    )
    campaign = MaterialsDiscoveryCampaign(config=config)
    # StrategyAgent recommends small batch size of 10, attempting to underfill 200 over 5 iterations
    campaign.orchestrator.plan_iteration = lambda **kwargs: {
        "elements": ["Li", "P"], "num_candidates": 10,
        "screening_criteria": {}, "memory_directives": [],
        "rationale": "underfill attempt", "hypothesis": None,
    }
    campaign.run_campaign()

    manifest = json.loads((tmp_path / "underfill_out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["proposals_generated"] == 200
    assert manifest["oracle_evaluations"] == 100


def test_uneven_budget_and_shortfall_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.screening.HAS_CHGNET", False)
    from agents.orchestrator import CampaignObjective
    objective = CampaignObjective(
        target_properties={}, constraints={"elements": ["Li", "P"]},
        success_criteria={"min_score": 999}, domain="uneven_test", max_iterations=3,
    )
    ref_inputs = [
        ReferencePhaseInput(source_id="Li-ref", structure=_struct(["Li"], "Li-ref")),
        ReferencePhaseInput(source_id="P-ref", structure=_struct(["P"], "P-ref")),
        ReferencePhaseInput(source_id="Li3P-ref", structure=_struct(["Li", "Li", "Li", "P"], "Li3P-ref"), source_energy_above_hull_ev_per_atom=0.0),
    ]
    ref_path = tmp_path / "ref.json"
    evaluator = MockEvaluator({"Li-ref": -1.0, "P-ref": -2.0, "Li3P-ref": -2.5})
    build_frozen_reference_set(
        reference_set_id="ref", chemical_system=["Li", "P"], inputs=ref_inputs,
        evaluator=evaluator, output_path=ref_path, created_at_iso="2026-08-30T00:00:00+00:00",
    )

    # 7 proposals over 3 iterations (allocation: ceil(7/3)=3, ceil(4/2)=2, ceil(2/1)=2 -> total 7)
    config = CampaignConfig(
        name="test_uneven",
        objective=objective,
        output_dir=tmp_path / "uneven_out",
        proposal_budget=7,
        oracle_budget=7,
        thermodynamics_reference_set_path=str(ref_path),
        thermodynamics_evaluator=evaluator,
        geometry_min_distance=0.01,
        use_career_memory=False,
        use_validation=False,
        use_synthesis=False,
        verbose=False,
    )
    campaign = MaterialsDiscoveryCampaign(config=config)
    campaign.run_campaign()
    manifest = json.loads((tmp_path / "uneven_out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["proposals_generated"] == 7


# =========================================================================
# Phase 7: Real Shortfall Recovery & Unrecovered Shortfall
# =========================================================================

class _ShortfallRecoveryGenerator:
    def __init__(self, first_batch_yield=2):
        self.first_batch_yield = first_batch_yield
        self.call_count = 0
        self.requested_counts = []
        self.actual_counts = []
        self.last_generation_backend = "mock_shortfall_gen"
        self.backend_name = "mock_shortfall_gen"

    def generate_batch(self, elements, num_candidates=15, seed=42, **kwargs):
        self.call_count += 1
        self.requested_counts.append(num_candidates)
        if self.call_count == 1:
            yield_count = min(self.first_batch_yield, num_candidates)
        else:
            yield_count = num_candidates
        self.actual_counts.append(yield_count)
        return [_struct(["Li", "P"], candidate_id=f"sf_{self.call_count}_{i}") for i in range(yield_count)]


def test_real_shortfall_followed_by_recovery_consumes_full_budget(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.screening.HAS_CHGNET", False)
    from agents.orchestrator import CampaignObjective
    objective = CampaignObjective(
        target_properties={}, constraints={"elements": ["Li", "P"]},
        success_criteria={"min_score": 999}, domain="shortfall_rec", max_iterations=3,
    )
    ref_inputs = [
        ReferencePhaseInput(source_id="Li-ref", structure=_struct(["Li"], "Li-ref")),
        ReferencePhaseInput(source_id="P-ref", structure=_struct(["P"], "P-ref")),
        ReferencePhaseInput(source_id="Li3P-ref", structure=_struct(["Li", "Li", "Li", "P"], "Li3P-ref"), source_energy_above_hull_ev_per_atom=0.0),
    ]
    ref_path = tmp_path / "ref.json"
    evaluator = MockEvaluator({"Li-ref": -1.0, "P-ref": -2.0, "Li3P-ref": -2.5})
    build_frozen_reference_set(
        reference_set_id="ref", chemical_system=["Li", "P"], inputs=ref_inputs,
        evaluator=evaluator, output_path=ref_path, created_at_iso="2026-08-30T00:00:00+00:00",
    )

    # 7 proposals over 3 iterations:
    # iter 0: requested 3 -> returns 2 (rem = 5)
    # iter 1: rem 5 over 2 iters -> requested 3 -> returns 3 (rem = 2)
    # iter 2: rem 2 over 1 iter -> requested 2 -> returns 2 (rem = 0)
    config = CampaignConfig(
        name="shortfall_recovery_campaign",
        objective=objective,
        output_dir=tmp_path / "shortfall_rec_out",
        proposal_budget=7,
        oracle_budget=7,
        thermodynamics_reference_set_path=str(ref_path),
        thermodynamics_evaluator=evaluator,
        geometry_min_distance=0.01,
        use_career_memory=False,
        use_validation=False,
        use_synthesis=False,
        verbose=False,
    )
    campaign = MaterialsDiscoveryCampaign(config=config)
    generator = _ShortfallRecoveryGenerator(first_batch_yield=2)
    campaign.generator = generator

    res = campaign.run_campaign()
    assert generator.requested_counts == [3, 3, 2]
    assert generator.actual_counts == [2, 3, 2]
    assert res["proposals_generated"] == 7
    assert res["proposal_budget_remaining"] == 0

    manifest = json.loads((tmp_path / "shortfall_rec_out" / "manifest.json").read_text(encoding="utf-8"))
    events = manifest.get("generation_shortfall_events", []) or manifest.get("metadata", {}).get("generation_shortfall_events", [])
    assert len(events) == 2
    assert events[0]["iteration"] == 0
    assert events[0]["requested_count"] == 3
    assert events[0]["actual_count"] == 2
    assert events[0]["new_shortfall"] == 1
    assert events[1]["iteration"] == 1
    assert events[1]["recovered_count"] == 1
    assert manifest.get("backend_generation_shortfall") in (0, None) and manifest.get("metadata", {}).get("backend_generation_shortfall") in (0, None)

    prov_p = tmp_path / "shortfall_rec_out" / "campaign_provenance.json"
    rep_p = tmp_path / "shortfall_rec_out" / "report.json"
    m_data = {**json.loads(rep_p.read_text(encoding="utf-8")), **json.loads(prov_p.read_text(encoding="utf-8"))}
    rm, cands = compute_run_metrics(
        manifest_data=m_data,
        run_id="run_sf",
        task_id="shortfall_rec",
        condition="adaptive_no_memory",
        seed=42,
        oracle_budget=7,
    )
    assert rm.proposals_generated == 7
    assert rm.provenance_complete is True


def test_unrecovered_final_shortfall_marks_provenance_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.screening.HAS_CHGNET", False)
    from agents.orchestrator import CampaignObjective
    objective = CampaignObjective(
        target_properties={}, constraints={"elements": ["Li", "P"]},
        success_criteria={"min_score": 999}, domain="shortfall_unrec", max_iterations=2,
    )
    ref_inputs = [
        ReferencePhaseInput(source_id="Li-ref", structure=_struct(["Li"], "Li-ref")),
        ReferencePhaseInput(source_id="P-ref", structure=_struct(["P"], "P-ref")),
        ReferencePhaseInput(source_id="Li3P-ref", structure=_struct(["Li", "Li", "Li", "P"], "Li3P-ref"), source_energy_above_hull_ev_per_atom=0.0),
    ]
    ref_path = tmp_path / "ref.json"
    evaluator = MockEvaluator({"Li-ref": -1.0, "P-ref": -2.0, "Li3P-ref": -2.5})
    build_frozen_reference_set(
        reference_set_id="ref", chemical_system=["Li", "P"], inputs=ref_inputs,
        evaluator=evaluator, output_path=ref_path, created_at_iso="2026-08-30T00:00:00+00:00",
    )

    config = CampaignConfig(
        name="shortfall_unrecovered_campaign",
        objective=objective,
        output_dir=tmp_path / "shortfall_unrec_out",
        proposal_budget=7,
        oracle_budget=7,
        thermodynamics_reference_set_path=str(ref_path),
        thermodynamics_evaluator=evaluator,
        geometry_min_distance=0.01,
        use_career_memory=False,
        use_validation=False,
        use_synthesis=False,
        verbose=False,
    )
    campaign = MaterialsDiscoveryCampaign(config=config)
    # Generator always produces only 2 candidates
    generator = _ShortfallRecoveryGenerator(first_batch_yield=2)
    generator.generate_batch = lambda *args, **kwargs: [_struct(["Li", "P"], candidate_id=f"u_{i}") for i in range(2)]
    campaign.generator = generator

    res = campaign.run_campaign()
    assert res["proposals_generated"] == 4
    assert res["proposal_budget_remaining"] == 3

    manifest = json.loads((tmp_path / "shortfall_unrec_out" / "manifest.json").read_text(encoding="utf-8"))
    shortfall_val = manifest.get("backend_generation_shortfall") or manifest.get("metadata", {}).get("backend_generation_shortfall")
    assert shortfall_val == 3

    prov_p = tmp_path / "shortfall_unrec_out" / "campaign_provenance.json"
    rep_p = tmp_path / "shortfall_unrec_out" / "report.json"
    m_data = {**json.loads(rep_p.read_text(encoding="utf-8")), **json.loads(prov_p.read_text(encoding="utf-8"))}
    rm, cands = compute_run_metrics(
        manifest_data=m_data,
        run_id="run_unrec",
        task_id="shortfall_unrec",
        condition="adaptive_no_memory",
        seed=42,
        oracle_budget=7,
    )
    assert rm.proposals_generated == 4
    assert rm.provenance_complete is False
    assert "backend_generation_shortfall" in rm.provenance_missing_fields


# =========================================================================
# Phase 8: Partial Resume, Preflight State Hydration & DAG State Validation
# =========================================================================

def test_partial_resume_with_report_pending_hydrates_preflight_and_avoids_recomputation(tmp_path, monkeypatch):
    from experiments.cli import execute_full_experiment_pipeline
    spec = _build_tiny_spec(tmp_path / "partial_resume_out")
    res1 = execute_full_experiment_pipeline(spec, force_rerun=True)
    assert res1["status"] == "SUCCESS"

    out_dir = Path(spec.output_root)
    dag_p = out_dir / "experiment_dag.json"
    dag_dict = json.loads(dag_p.read_text(encoding="utf-8"))
    report_node_id = next(nid for nid, n in dag_dict["nodes"].items() if n["node_type"] == "report_generation")
    dag_dict["nodes"][report_node_id]["executed"] = False
    dag_dict["nodes"][report_node_id]["success"] = False
    dag_dict["nodes"][report_node_id]["result_hash"] = None
    dag_dict["nodes"][report_node_id]["result_artifacts"] = {}
    dag_p.write_text(json.dumps(dag_dict, indent=2), encoding="utf-8")

    # Remove generated paper and figure artifacts to ensure they are regenerated
    matrix_p = out_dir / "paper" / "claim_evidence_matrix.md"
    if matrix_p.exists():
        matrix_p.unlink()

    campaign_calls = []
    orig_exec = CampaignRunner.execute_run
    monkeypatch.setattr(CampaignRunner, "execute_run", lambda run_spec, force_rerun=False: campaign_calls.append(run_spec.run_id) or orig_exec(run_spec, force_rerun=force_rerun))

    qe_calls = []
    orig_qe = QEAuditRunner.run_full_audit
    monkeypatch.setattr(QEAuditRunner, "run_full_audit", lambda self, cands: qe_calls.append(len(cands)) or orig_qe(self, cands))

    # Resuming should successfully hydrate preflight, regenerate report, and not rerun campaigns or QE
    res2 = execute_full_experiment_pipeline(spec, force_rerun=False)
    assert res2["status"] == "SUCCESS"
    assert len(campaign_calls) == 0, f"Expected 0 campaign recomputations, got {campaign_calls}"
    assert len(qe_calls) == 0, f"Expected 0 QE recomputations, got {qe_calls}"
    assert matrix_p.exists(), "Expected report claim matrix to be generated on partial resume"


def test_dag_resume_rejects_successful_node_missing_result_hash(tmp_path):
    spec = _build_tiny_spec(tmp_path / "dag_missing_hash")
    dag = ExperimentDAG(spec)
    dag_dict = dag.to_dict()

    # Successful fixed node with result_hash = None
    preflight_id = next(nid for nid, n in dag_dict["nodes"].items() if n["node_type"] == "preflight")
    dag_dict["nodes"][preflight_id]["executed"] = True
    dag_dict["nodes"][preflight_id]["success"] = True
    dag_dict["nodes"][preflight_id]["result_hash"] = None

    with pytest.raises(DAGValidationError, match="result_hash"):
        ExperimentDAG.from_dict(dag_dict, spec)


def test_dag_resume_rejects_malformed_result_hash(tmp_path):
    spec = _build_tiny_spec(tmp_path / "dag_malformed_hash")
    dag = ExperimentDAG(spec)
    dag_dict = dag.to_dict()

    preflight_id = next(nid for nid, n in dag_dict["nodes"].items() if n["node_type"] == "preflight")
    dag_dict["nodes"][preflight_id]["executed"] = True
    dag_dict["nodes"][preflight_id]["success"] = True
    dag_dict["nodes"][preflight_id]["result_hash"] = "not_a_valid_64_char_hex_hash"

    with pytest.raises(DAGValidationError, match="result_hash"):
        ExperimentDAG.from_dict(dag_dict, spec)


def test_dag_resume_rejects_inconsistent_executed_success_state(tmp_path):
    spec = _build_tiny_spec(tmp_path / "dag_inconsistent_state")
    dag = ExperimentDAG(spec)
    dag_dict = dag.to_dict()

    preflight_id = next(nid for nid, n in dag_dict["nodes"].items() if n["node_type"] == "preflight")
    dag_dict["nodes"][preflight_id]["executed"] = False
    dag_dict["nodes"][preflight_id]["success"] = True

    with pytest.raises(DAGValidationError, match="state inconsistent"):
        ExperimentDAG.from_dict(dag_dict, spec)


def test_dag_resume_rejects_successful_fixed_output_node_missing_expected_output_path(tmp_path):
    spec = _build_tiny_spec(tmp_path / "dag_missing_out_path")
    dag = ExperimentDAG(spec)
    dag_dict = dag.to_dict()

    preflight_id = next(nid for nid, n in dag_dict["nodes"].items() if n["node_type"] == "preflight")
    dag_dict["nodes"][preflight_id]["executed"] = True
    dag_dict["nodes"][preflight_id]["success"] = True
    dag_dict["nodes"][preflight_id]["result_hash"] = "a" * 64
    dag_dict["nodes"][preflight_id]["expected_output_path"] = None

    with pytest.raises(DAGValidationError, match="missing expected_output_path"):
        ExperimentDAG.from_dict(dag_dict, spec)


# =========================================================================
# Phase 9: Verified Reference Set Deletion & Alteration Detection
# =========================================================================

def test_resume_fails_when_reference_set_deleted_or_altered(tmp_path):
    from experiments.cli import execute_full_experiment_pipeline
    ref_inputs = [
        ReferencePhaseInput(source_id="Li-ref", structure=_struct(["Li"], "Li-ref")),
        ReferencePhaseInput(source_id="P-ref", structure=_struct(["P"], "P-ref")),
        ReferencePhaseInput(source_id="Li3P-ref", structure=_struct(["Li", "Li", "Li", "P"], "Li3P-ref"), source_energy_above_hull_ev_per_atom=0.0),
    ]
    ref_path = tmp_path / "ref_source.json"
    try:
        from agents.thermodynamics import CHGNetRelaxationEvaluator
        chg_eval = CHGNetRelaxationEvaluator()
        evaluator = MockEvaluator({"Li-ref": -1.0, "P-ref": -2.0, "Li3P-ref": -2.5})
        evaluator.model_identity = chg_eval.model_identity
        evaluator.relaxation_settings = chg_eval.relaxation_settings
    except Exception:
        evaluator = MockEvaluator({"Li-ref": -1.0, "P-ref": -2.0, "Li3P-ref": -2.5})
    ref_meta = build_frozen_reference_set(
        reference_set_id="ref_source", chemical_system=["Li", "P"], inputs=ref_inputs,
        evaluator=evaluator, output_path=ref_path, created_at_iso="2026-08-30T00:00:00+00:00",
    )
    source_task = TaskDefinition(
        task_id="source_li_p",
        elements=["Li", "P"],
        domain="Li-P",
        target_properties={"band_gap": 2.0},
        constraints={},
        reference_set_path=str(ref_path).replace("\\", "/"),
        reference_set_sha256=hashlib.sha256(ref_path.read_bytes()).hexdigest(),
    )
    spec = _build_tiny_spec(tmp_path / "ref_alter_out", source_task=source_task)

    execute_full_experiment_pipeline(spec, force_rerun=True)

    orig_bytes = ref_path.read_bytes()

    # 1. Deleted reference set file
    ref_path.unlink()
    with pytest.raises(RuntimeError, match="reference set.*missing|missing or not a file"):
        execute_full_experiment_pipeline(spec, force_rerun=False)

    # 2. Altered reference set file bytes
    ref_path.write_bytes(orig_bytes + b"tampered")
    with pytest.raises(RuntimeError, match="reference set SHA256 mismatch"):
        execute_full_experiment_pipeline(spec, force_rerun=False)

    # 3. Tampered .verified.json artifact
    ref_path.write_bytes(orig_bytes)
    ver_p = Path(spec.output_root) / "references" / f"{spec.source_task.task_id}.verified.json"
    ver_dict = json.loads(ver_p.read_text(encoding="utf-8"))
    ver_dict["sha256"] = "0" * 64
    ver_p.write_text(json.dumps(ver_dict), encoding="utf-8")
    with pytest.raises(RuntimeError, match="reference verification artifact|digest mismatch"):
        execute_full_experiment_pipeline(spec, force_rerun=False)


# =========================================================================
# Phase 10: Complete Multi-Artifact Node Contracts
# =========================================================================

def test_resume_fails_when_secondary_aggregation_artifact_deleted_or_tampered(tmp_path):
    from experiments.cli import execute_full_experiment_pipeline
    spec = _build_tiny_spec(tmp_path / "agg_art_out")
    execute_full_experiment_pipeline(spec, force_rerun=True)

    out_dir = Path(spec.output_root)
    cands_pq = out_dir / "aggregates" / "candidates.parquet"
    cands_pq_bytes = cands_pq.read_bytes()

    # 1. Delete candidates.parquet
    cands_pq.unlink()
    with pytest.raises(RuntimeError, match="missing aggregate file|missing"):
        execute_full_experiment_pipeline(spec, force_rerun=False)

    # 2. Tamper candidates.json
    cands_pq.write_bytes(cands_pq_bytes)
    cands_json = out_dir / "aggregates" / "candidates.json"
    cands_json.write_text(cands_json.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        execute_full_experiment_pipeline(spec, force_rerun=False)


def test_resume_fails_when_secondary_report_artifact_deleted_or_tampered(tmp_path):
    from experiments.cli import execute_full_experiment_pipeline
    spec = _build_tiny_spec(tmp_path / "rep_art_out")
    execute_full_experiment_pipeline(spec, force_rerun=True)

    out_dir = Path(spec.output_root)
    chk_file = out_dir / "paper" / "reproducibility_checklist.md"
    chk_bytes = chk_file.read_bytes()

    # 1. Delete checklist
    chk_file.unlink()
    with pytest.raises(RuntimeError, match="missing report artifact|missing"):
        execute_full_experiment_pipeline(spec, force_rerun=False)

    # 2. Tamper fig1 json
    chk_file.write_bytes(chk_bytes)
    fig1 = out_dir / "figures" / "fig1_best_so_far_vs_oracle_calls.json"
    fig1.write_text(fig1.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        execute_full_experiment_pipeline(spec, force_rerun=False)


def test_resume_fails_when_qe_csv_disagrees_with_result_provenance(tmp_path):
    from experiments.cli import execute_full_experiment_pipeline
    spec = _build_tiny_spec(tmp_path / "qe_csv_disagree_out")
    execute_full_experiment_pipeline(spec, force_rerun=True)

    out_dir = Path(spec.output_root)
    csv_file = out_dir / "qe_audit" / "results.csv"
    csv_file.write_text(csv_file.read_text(encoding="utf-8") + "tampered_cand,task,cond,1,Li,1,CONVERGED,-1,1,True,0.01,0.01,True,0.0,Li->Li,True,VALIDATED\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="results.csv.*disagree|digest mismatch|QE results.csv"):
        execute_full_experiment_pipeline(spec, force_rerun=False)


def test_resume_fails_when_statistics_canonical_hash_tampered(tmp_path):
    from experiments.cli import execute_full_experiment_pipeline
    spec = _build_tiny_spec(tmp_path / "stats_canon_tamper_out")
    execute_full_experiment_pipeline(spec, force_rerun=True)

    out_dir = Path(spec.output_root)
    manifest_p = out_dir / "statistics" / "analysis_manifest.json"
    m_dict = json.loads(manifest_p.read_text(encoding="utf-8"))
    m_dict["artifacts"]["canonical_results_sha256"] = "0" * 64
    manifest_p.write_text(json.dumps(m_dict, indent=2), encoding="utf-8")

    with pytest.raises(RuntimeError, match="canonical_results_sha256 digest mismatch|digest mismatch"):
        execute_full_experiment_pipeline(spec, force_rerun=False)


# =========================================================================
# Phase 11: Mandatory Configured QE and SSSP Digests at Report Boundary
# =========================================================================

def test_report_boundary_requires_audit_manifest_config_digests(tmp_path):
    phase = {
        "status": "CONVERGED", "input_hash": "a" * 64, "result_hash": "b" * 64,
        "output_hash": "c" * 64, "executable": "pw.x", "executable_version": "qe-1",
        "qe_executable_sha256": "e" * 64, "sssp_manifest_sha256": "d" * 64,
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
        "qe_executable": "pw.x", "qe_executable_version": "qe-1",
        "qe_executable_sha256": "e" * 64, "sssp_manifest_sha256": "d" * 64,
        "participating_phase_results": [phase],
    }
    qe_dir = tmp_path / "qe_audit"
    qe_dir.mkdir(parents=True, exist_ok=True)
    (qe_dir / "selection.json").write_text(json.dumps({"candidates": [{"candidate_id": "c"}], "insufficiency": None}), encoding="utf-8")
    with (qe_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        c_row = dict(row)
        c_row["participating_phase_results"] = json.dumps([phase], sort_keys=True, separators=(",", ":"))
        writer.writerow(c_row)

    def _make_manifest(config=None):
        m = {
            "run_mode": "research", "mock_execution": False, "selection_count": 1,
            "selection_insufficiency": None, "result_provenance": [row],
        }
        if config is not None:
            m["config"] = config
        m["artifacts"] = {
            "selection.json": hashlib.sha256((qe_dir / "selection.json").read_bytes()).hexdigest(),
            "results.csv": hashlib.sha256((qe_dir / "results.csv").read_bytes()).hexdigest(),
            "canonical_results_sha256": hashlib.sha256(json.dumps([row], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        }
        return m

    report = ReportGenerator(tmp_path)
    supplied = dict(row)

    # 1. Missing config mapping returns False
    m_no_config = _make_manifest(config=None)
    assert report._qe_evidence_complete([supplied], {"expected_qe_candidate_count": 1, "qe_selection_count": 1, "qe_audit": m_no_config}) is False

    # 2. Missing qe_executable_sha256 in config returns False
    m_no_qe = _make_manifest(config={"sssp_manifest_sha256": "d" * 64})
    assert report._qe_evidence_complete([supplied], {"expected_qe_candidate_count": 1, "qe_selection_count": 1, "qe_audit": m_no_qe}) is False

    # 3. Missing sssp_manifest_sha256 in config returns False
    m_no_sssp = _make_manifest(config={"qe_executable_sha256": "e" * 64})
    assert report._qe_evidence_complete([supplied], {"expected_qe_candidate_count": 1, "qe_selection_count": 1, "qe_audit": m_no_sssp}) is False

    # 4. Malformed (short) qe_executable_sha256 in config returns False
    m_short_qe = _make_manifest(config={"qe_executable_sha256": "short", "sssp_manifest_sha256": "d" * 64})
    assert report._qe_evidence_complete([supplied], {"expected_qe_candidate_count": 1, "qe_selection_count": 1, "qe_audit": m_short_qe}) is False

    # 5. Candidate qe_executable_sha256 mismatch with config returns False
    m_mismatch = _make_manifest(config={"qe_executable_sha256": "f" * 64, "sssp_manifest_sha256": "d" * 64})
    assert report._qe_evidence_complete([supplied], {"expected_qe_candidate_count": 1, "qe_selection_count": 1, "qe_audit": m_mismatch}) is False

    # 6. Valid fully matching config returns True
    m_valid = _make_manifest(config={"qe_executable_sha256": "e" * 64, "sssp_manifest_sha256": "d" * 64})
    assert report._qe_evidence_complete([supplied], {"expected_qe_candidate_count": 1, "qe_selection_count": 1, "qe_audit": m_valid}) is True


# =========================================================================
# Phase 12: Mandatory Multi-Artifact Contracts, Shortfall Attribution & Semantic References
# =========================================================================

def test_dag_resume_fails_when_result_artifacts_deleted_or_empty(tmp_path):
    from experiments.cli import execute_full_experiment_pipeline
    spec = _build_tiny_spec(tmp_path / "art_del_empty_out")
    execute_full_experiment_pipeline(spec, force_rerun=True)

    dag_p = Path(spec.output_root) / "experiment_dag.json"
    dag_dict = json.loads(dag_p.read_text(encoding="utf-8"))
    agg_id = next(nid for nid, n in dag_dict["nodes"].items() if n["node_type"] == "aggregation")

    # 1. result_artifacts deleted
    del dag_dict["nodes"][agg_id]["result_artifacts"]
    dag_p.write_text(json.dumps(dag_dict, indent=2), encoding="utf-8")
    with pytest.raises(DAGValidationError, match="missing required result_artifacts contract"):
        execute_full_experiment_pipeline(spec, force_rerun=False)

    # 2. result_artifacts is {}
    dag_dict["nodes"][agg_id]["result_artifacts"] = {}
    dag_p.write_text(json.dumps(dag_dict, indent=2), encoding="utf-8")
    with pytest.raises(DAGValidationError, match="missing required result_artifacts contract"):
        execute_full_experiment_pipeline(spec, force_rerun=False)


def test_dag_resume_fails_when_required_artifact_key_omitted_and_file_altered(tmp_path):
    from experiments.cli import execute_full_experiment_pipeline
    spec = _build_tiny_spec(tmp_path / "art_key_omitted_out")
    execute_full_experiment_pipeline(spec, force_rerun=True)

    out_dir = Path(spec.output_root)
    dag_p = out_dir / "experiment_dag.json"
    dag_dict = json.loads(dag_p.read_text(encoding="utf-8"))
    agg_id = next(nid for nid, n in dag_dict["nodes"].items() if n["node_type"] == "aggregation")

    # Remove candidates.json from result_artifacts
    dag_dict["nodes"][agg_id]["result_artifacts"].pop("aggregates/candidates.json", None)
    dag_p.write_text(json.dumps(dag_dict, indent=2), encoding="utf-8")

    # Alter candidates.json
    c_json = out_dir / "aggregates" / "candidates.json"
    c_json.write_text(c_json.read_text(encoding="utf-8") + " ", encoding="utf-8")

    # Resuming must fail closed at DAG validation before bypassing artifact verification
    with pytest.raises(DAGValidationError, match="missing required contract keys"):
        execute_full_experiment_pipeline(spec, force_rerun=False)


def test_dag_resume_fails_when_report_contract_missing_static_artifact(tmp_path):
    from experiments.cli import execute_full_experiment_pipeline
    spec = _build_tiny_spec(tmp_path / "rep_contract_missing_out")
    execute_full_experiment_pipeline(spec, force_rerun=True)

    dag_p = Path(spec.output_root) / "experiment_dag.json"
    dag_dict = json.loads(dag_p.read_text(encoding="utf-8"))
    rep_id = next(nid for nid, n in dag_dict["nodes"].items() if n["node_type"] == "report_generation")

    # Remove required table artifact from report contract
    dag_dict["nodes"][rep_id]["result_artifacts"].pop("tables/table_threshold_sensitivity.csv", None)
    dag_p.write_text(json.dumps(dag_dict, indent=2), encoding="utf-8")

    with pytest.raises(DAGValidationError, match="missing required contract keys"):
        execute_full_experiment_pipeline(spec, force_rerun=False)


def test_dag_unexecuted_nodes_may_have_empty_contract(tmp_path):
    spec = _build_tiny_spec(tmp_path / "unexecuted_dag_out")
    dag = ExperimentDAG(spec)
    dag_dict = dag.to_dict()
    # All nodes are unexecuted with empty result_artifacts
    hydrated = ExperimentDAG.from_dict(dag_dict, spec)
    assert len(hydrated.nodes) == len(dag.nodes)


def test_full_backend_batches_with_early_termination_has_no_backend_shortfall(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.screening.HAS_CHGNET", False)
    from agents.orchestrator import CampaignObjective
    objective = CampaignObjective(
        target_properties={}, constraints={"elements": ["Li", "P"]},
        success_criteria={"min_score": -999.0}, domain="early_stop_no_shortfall", max_iterations=5,
    )
    ref_inputs = [
        ReferencePhaseInput(source_id="Li-ref", structure=_struct(["Li"], "Li-ref")),
        ReferencePhaseInput(source_id="P-ref", structure=_struct(["P"], "P-ref")),
        ReferencePhaseInput(source_id="Li3P-ref", structure=_struct(["Li", "Li", "Li", "P"], "Li3P-ref"), source_energy_above_hull_ev_per_atom=0.0),
    ]
    ref_path = tmp_path / "ref.json"
    evaluator = MockEvaluator({"Li-ref": -1.0, "P-ref": -2.0, "Li3P-ref": -2.5})
    build_frozen_reference_set(
        reference_set_id="ref", chemical_system=["Li", "P"], inputs=ref_inputs,
        evaluator=evaluator, output_path=ref_path, created_at_iso="2026-08-30T00:00:00+00:00",
    )

    config = CampaignConfig(
        name="early_stop_campaign",
        objective=objective,
        output_dir=tmp_path / "early_stop_out",
        proposal_budget=20,
        oracle_budget=20,
        thermodynamics_reference_set_path=str(ref_path),
        thermodynamics_evaluator=evaluator,
        geometry_min_distance=0.01,
        use_career_memory=False,
        use_validation=False,
        use_synthesis=False,
        verbose=False,
    )
    campaign = MaterialsDiscoveryCampaign(config=config)
    # Generator produces 100% of requested candidates
    generator = _ShortfallRecoveryGenerator(first_batch_yield=4)
    generator.generate_batch = lambda elements, num_candidates, **kwargs: [
        _struct(elements, candidate_id=f"c_{i}") for i in range(num_candidates)
    ]
    campaign.generator = generator

    res = campaign.run_campaign()
    assert res["proposal_budget_remaining"] > 0
    manifest = json.loads((tmp_path / "early_stop_out" / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "early_stop_out" / "report.json").read_text(encoding="utf-8"))
    assert manifest.get("backend_generation_shortfall") is None
    assert report.get("backend_generation_shortfall") is None
    assert manifest.get("generation_shortfall_events", []) == []


def test_fully_recovered_before_unrelated_early_stop(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.screening.HAS_CHGNET", False)
    from agents.orchestrator import CampaignObjective
    objective = CampaignObjective(
        target_properties={}, constraints={"elements": ["Li", "P"]},
        success_criteria={"min_score": 999.0}, domain="fully_rec_early_stop", max_iterations=3,
    )
    ref_inputs = [
        ReferencePhaseInput(source_id="Li-ref", structure=_struct(["Li"], "Li-ref")),
        ReferencePhaseInput(source_id="P-ref", structure=_struct(["P"], "P-ref")),
        ReferencePhaseInput(source_id="Li3P-ref", structure=_struct(["Li", "Li", "Li", "P"], "Li3P-ref"), source_energy_above_hull_ev_per_atom=0.0),
    ]
    ref_path = tmp_path / "ref.json"
    evaluator = MockEvaluator({"Li-ref": -1.0, "P-ref": -2.0, "Li3P-ref": -2.5})
    build_frozen_reference_set(
        reference_set_id="ref", chemical_system=["Li", "P"], inputs=ref_inputs,
        evaluator=evaluator, output_path=ref_path, created_at_iso="2026-08-30T00:00:00+00:00",
    )

    config = CampaignConfig(
        name="fully_rec_campaign",
        objective=objective,
        output_dir=tmp_path / "fully_rec_out",
        proposal_budget=7,
        oracle_budget=7,
        thermodynamics_reference_set_path=str(ref_path),
        thermodynamics_evaluator=evaluator,
        geometry_min_distance=0.01,
        use_career_memory=False,
        use_validation=False,
        use_synthesis=False,
        verbose=False,
    )
    campaign = MaterialsDiscoveryCampaign(config=config)
    # Iteration 0: requests 3, returns 2 (debt becomes 1)
    # Iteration 1: requests 3 (baseline 2, recovery 1), returns 3 (debt becomes 0)
    # Stop after iteration 1 for an unrelated reason
    gen_call = [0]
    def _gen_batch(elements, num_candidates, **kwargs):
        gen_call[0] += 1
        if gen_call[0] == 1:
            return [_struct(elements, candidate_id=f"p_{i}") for i in range(2)]
        return [_struct(elements, candidate_id=f"p_{i}") for i in range(num_candidates)]

    campaign.generator.generate_batch = _gen_batch
    campaign._check_termination = lambda iter_result: (True, "Early stop unrelated to budget") if campaign.iteration > 0 else (False, "Continue")

    res = campaign.run_campaign()
    assert res["proposals_generated"] == 5
    assert res["proposal_budget_remaining"] == 2

    manifest = json.loads((tmp_path / "fully_rec_out" / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "fully_rec_out" / "report.json").read_text(encoding="utf-8"))
    assert manifest.get("backend_generation_shortfall") is None
    assert report.get("backend_generation_shortfall") is None

    events = manifest.get("generation_shortfall_events", [])
    assert len(events) == 2
    ev0, ev1 = events[0], events[1]
    assert ev0["iteration"] == 0
    assert ev0["requested_count"] == 3
    assert ev0["baseline_requested_count"] == 3
    assert ev0["recovery_requested_count"] == 0
    assert ev0["actual_count"] == 2
    assert ev0["new_shortfall"] == 1
    assert ev0["recovered_count"] == 0
    assert ev0["outstanding_before"] == 0
    assert ev0["outstanding_after"] == 1

    assert ev1["iteration"] == 1
    assert ev1["requested_count"] == 3
    assert ev1["baseline_requested_count"] == 2
    assert ev1["recovery_requested_count"] == 1
    assert ev1["actual_count"] == 3
    assert ev1["new_shortfall"] == 0
    assert ev1["recovered_count"] == 1
    assert ev1["outstanding_before"] == 1
    assert ev1["outstanding_after"] == 0


def test_partially_recovered_before_early_stop(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.screening.HAS_CHGNET", False)
    from agents.orchestrator import CampaignObjective
    objective = CampaignObjective(
        target_properties={}, constraints={"elements": ["Li", "P"]},
        success_criteria={"min_score": 999.0}, domain="partially_rec_early_stop", max_iterations=4,
    )
    ref_inputs = [
        ReferencePhaseInput(source_id="Li-ref", structure=_struct(["Li"], "Li-ref")),
        ReferencePhaseInput(source_id="P-ref", structure=_struct(["P"], "P-ref")),
        ReferencePhaseInput(source_id="Li3P-ref", structure=_struct(["Li", "Li", "Li", "P"], "Li3P-ref"), source_energy_above_hull_ev_per_atom=0.0),
    ]
    ref_path = tmp_path / "ref.json"
    evaluator = MockEvaluator({"Li-ref": -1.0, "P-ref": -2.0, "Li3P-ref": -2.5})
    build_frozen_reference_set(
        reference_set_id="ref", chemical_system=["Li", "P"], inputs=ref_inputs,
        evaluator=evaluator, output_path=ref_path, created_at_iso="2026-08-30T00:00:00+00:00",
    )

    config = CampaignConfig(
        name="partially_rec_campaign",
        objective=objective,
        output_dir=tmp_path / "partially_rec_out",
        proposal_budget=10,
        oracle_budget=10,
        thermodynamics_reference_set_path=str(ref_path),
        thermodynamics_evaluator=evaluator,
        geometry_min_distance=0.01,
        use_career_memory=False,
        use_validation=False,
        use_synthesis=False,
        verbose=False,
    )
    campaign = MaterialsDiscoveryCampaign(config=config)
    # Iteration 0: requests 3, returns 1 (deficit 2, debt becomes 2). Proposals = 1, remaining = 9.
    # Iteration 1: requests 5 (baseline 3, recovery 2), returns 4 (baseline 3, recovered 1, debt becomes 1). Proposals = 5, remaining = 5.
    # Early stop fires after iteration 1.
    gen_call = [0]
    def _gen_batch(elements, num_candidates, **kwargs):
        gen_call[0] += 1
        if gen_call[0] == 1:
            return [_struct(elements, candidate_id=f"p_{i}") for i in range(1)]
        return [_struct(elements, candidate_id=f"p_{i}") for i in range(4)]

    campaign.generator.generate_batch = _gen_batch
    campaign._check_termination = lambda iter_result: (True, "Early stop unrelated to budget") if campaign.iteration > 0 else (False, "Continue")

    res = campaign.run_campaign()
    assert res["proposals_generated"] == 5
    assert res["proposal_budget_remaining"] == 5

    manifest = json.loads((tmp_path / "partially_rec_out" / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "partially_rec_out" / "report.json").read_text(encoding="utf-8"))
    # Final shortfall must be exactly 1 (not historical sum of shortfalls and not total remaining budget)
    assert manifest.get("backend_generation_shortfall") == 1
    assert report.get("backend_generation_shortfall") == 1

    events = manifest.get("generation_shortfall_events", [])
    assert len(events) == 2
    assert events[0]["outstanding_after"] == 2
    assert events[1]["outstanding_before"] == 2
    assert events[1]["recovered_count"] == 1
    assert events[1]["outstanding_after"] == 1


def test_unlimited_proposal_budget_records_underproduction_honestly(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.screening.HAS_CHGNET", False)
    from agents.orchestrator import CampaignObjective
    objective = CampaignObjective(
        target_properties={}, constraints={"elements": ["Li", "P"]},
        success_criteria={"min_score": 999.0},
        domain="unlimited_prop", max_iterations=1,
    )
    ref_inputs = [
        ReferencePhaseInput(source_id="Li-ref", structure=_struct(["Li"], "Li-ref")),
        ReferencePhaseInput(source_id="P-ref", structure=_struct(["P"], "P-ref")),
        ReferencePhaseInput(source_id="Li3P-ref", structure=_struct(["Li", "Li", "Li", "P"], "Li3P-ref"), source_energy_above_hull_ev_per_atom=0.0),
    ]
    ref_path = tmp_path / "ref.json"
    evaluator = MockEvaluator({"Li-ref": -1.0, "P-ref": -2.0, "Li3P-ref": -2.5})
    build_frozen_reference_set(
        reference_set_id="ref", chemical_system=["Li", "P"], inputs=ref_inputs,
        evaluator=evaluator, output_path=ref_path, created_at_iso="2026-08-30T00:00:00+00:00",
    )

    config = CampaignConfig(
        name="unlimited_campaign",
        objective=objective,
        output_dir=tmp_path / "unlimited_out",
        proposal_budget=None,
        oracle_budget=10,
        num_candidates=5,
        thermodynamics_reference_set_path=str(ref_path),
        thermodynamics_evaluator=evaluator,
        geometry_min_distance=0.01,
        use_career_memory=False,
        use_validation=False,
        use_synthesis=False,
        verbose=False,
    )
    campaign = MaterialsDiscoveryCampaign(config=config)
    # Returns 3 for request of 5
    campaign.generator.generate_batch = lambda elements, num_candidates, **kwargs: [
        _struct(elements, candidate_id=f"p_{i}") for i in range(3)
    ]

    res = campaign.run_campaign()
    manifest = json.loads((tmp_path / "unlimited_out" / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "unlimited_out" / "report.json").read_text(encoding="utf-8"))
    assert manifest.get("backend_generation_shortfall") == 2
    assert report.get("backend_generation_shortfall") == 2


def test_final_report_fails_closed_on_accounting_error(tmp_path):
    from agents.orchestrator import CampaignObjective
    objective = CampaignObjective(
        target_properties={}, constraints={"elements": ["Li", "P"]},
        success_criteria={"min_score": 999.0},
        domain="accounting_err", max_iterations=1,
    )
    config = CampaignConfig(
        name="acct_err_campaign",
        objective=objective,
        output_dir=tmp_path / "acct_err_out",
        proposal_budget=5,
        oracle_budget=5,
        use_career_memory=False,
        use_validation=False,
        use_synthesis=False,
        verbose=False,
    )
    campaign = MaterialsDiscoveryCampaign(config=config)
    campaign.backend_generation_shortfall_debt = 10
    with pytest.raises(RuntimeError, match="Accounting error: outstanding backend debt.*exceeds remaining proposal budget"):
        campaign._generate_final_report(1.0)


def test_resume_fails_when_reference_verification_missing_sidecar_sha256(tmp_path):
    from experiments.cli import execute_full_experiment_pipeline
    ref_inputs = [
        ReferencePhaseInput(source_id="Li-ref", structure=_struct(["Li"], "Li-ref")),
        ReferencePhaseInput(source_id="P-ref", structure=_struct(["P"], "P-ref")),
        ReferencePhaseInput(source_id="Li3P-ref", structure=_struct(["Li", "Li", "Li", "P"], "Li3P-ref"), source_energy_above_hull_ev_per_atom=0.0),
    ]
    ref_path = tmp_path / "ref_source.json"
    try:
        from agents.thermodynamics import CHGNetRelaxationEvaluator
        chg_eval = CHGNetRelaxationEvaluator()
        evaluator = MockEvaluator({"Li-ref": -1.0, "P-ref": -2.0, "Li3P-ref": -2.5})
        evaluator.model_identity = chg_eval.model_identity
        evaluator.relaxation_settings = chg_eval.relaxation_settings
    except Exception:
        evaluator = MockEvaluator({"Li-ref": -1.0, "P-ref": -2.0, "Li3P-ref": -2.5})
    ref_meta = build_frozen_reference_set(
        reference_set_id="ref_source", chemical_system=["Li", "P"], inputs=ref_inputs,
        evaluator=evaluator, output_path=ref_path, created_at_iso="2026-08-30T00:00:00+00:00",
    )
    source_task = TaskDefinition(
        task_id="source_li_p",
        elements=["Li", "P"],
        domain="Li-P",
        target_properties={"band_gap": 2.0},
        constraints={},
        reference_set_path=str(ref_path).replace("\\", "/"),
        reference_set_sha256=hashlib.sha256(ref_path.read_bytes()).hexdigest(),
    )
    spec = _build_tiny_spec(tmp_path / "ref_semantic_out", source_task=source_task)
    execute_full_experiment_pipeline(spec, force_rerun=True)

    out_dir = Path(spec.output_root)
    ver_p = out_dir / "references" / f"{source_task.task_id}.verified.json"
    ver_dict = json.loads(ver_p.read_text(encoding="utf-8"))

    # Remove sha256 from verification artifact
    ver_dict.pop("sha256", None)
    ver_p.write_text(json.dumps(ver_dict), encoding="utf-8")

    # Update generic file digest in DAG for node_refset_source_li_p to avoid masking the semantic defect
    new_ver_hash = hashlib.sha256(ver_p.read_bytes()).hexdigest()
    dag_p = out_dir / "experiment_dag.json"
    dag_dict = json.loads(dag_p.read_text(encoding="utf-8"))
    ref_nid = f"node_refset_{source_task.task_id}"
    dag_dict["nodes"][ref_nid]["result_hash"] = new_ver_hash
    rel_ver = str(ver_p.relative_to(out_dir)).replace("\\", "/")
    dag_dict["nodes"][ref_nid]["result_artifacts"][rel_ver] = new_ver_hash
    dag_p.write_text(json.dumps(dag_dict, indent=2), encoding="utf-8")

    with pytest.raises(RuntimeError, match="reference verification artifact missing valid.*sha256"):
        execute_full_experiment_pipeline(spec, force_rerun=False)


def test_resume_fails_when_reference_verification_task_id_mismatched(tmp_path):
    from experiments.cli import execute_full_experiment_pipeline
    ref_inputs = [
        ReferencePhaseInput(source_id="Li-ref", structure=_struct(["Li"], "Li-ref")),
        ReferencePhaseInput(source_id="P-ref", structure=_struct(["P"], "P-ref")),
        ReferencePhaseInput(source_id="Li3P-ref", structure=_struct(["Li", "Li", "Li", "P"], "Li3P-ref"), source_energy_above_hull_ev_per_atom=0.0),
    ]
    ref_path = tmp_path / "ref_source.json"
    try:
        from agents.thermodynamics import CHGNetRelaxationEvaluator
        chg_eval = CHGNetRelaxationEvaluator()
        evaluator = MockEvaluator({"Li-ref": -1.0, "P-ref": -2.0, "Li3P-ref": -2.5})
        evaluator.model_identity = chg_eval.model_identity
        evaluator.relaxation_settings = chg_eval.relaxation_settings
    except Exception:
        evaluator = MockEvaluator({"Li-ref": -1.0, "P-ref": -2.0, "Li3P-ref": -2.5})
    ref_meta = build_frozen_reference_set(
        reference_set_id="ref_source", chemical_system=["Li", "P"], inputs=ref_inputs,
        evaluator=evaluator, output_path=ref_path, created_at_iso="2026-08-30T00:00:00+00:00",
    )
    source_task = TaskDefinition(
        task_id="source_li_p",
        elements=["Li", "P"],
        domain="Li-P",
        target_properties={"band_gap": 2.0},
        constraints={},
        reference_set_path=str(ref_path).replace("\\", "/"),
        reference_set_sha256=hashlib.sha256(ref_path.read_bytes()).hexdigest(),
    )
    spec = _build_tiny_spec(tmp_path / "ref_task_mismatch_out", source_task=source_task)
    execute_full_experiment_pipeline(spec, force_rerun=True)

    out_dir = Path(spec.output_root)
    ver_p = out_dir / "references" / f"{source_task.task_id}.verified.json"
    ver_dict = json.loads(ver_p.read_text(encoding="utf-8"))

    # Tamper task_id in verification artifact
    ver_dict["task_id"] = "wrong_task_id"
    ver_p.write_text(json.dumps(ver_dict), encoding="utf-8")

    # Update generic file digest in DAG for node_refset_source_li_p to avoid masking
    new_ver_hash = hashlib.sha256(ver_p.read_bytes()).hexdigest()
    dag_p = out_dir / "experiment_dag.json"
    dag_dict = json.loads(dag_p.read_text(encoding="utf-8"))
    ref_nid = f"node_refset_{source_task.task_id}"
    dag_dict["nodes"][ref_nid]["result_hash"] = new_ver_hash
    rel_ver = str(ver_p.relative_to(out_dir)).replace("\\", "/")
    dag_dict["nodes"][ref_nid]["result_artifacts"][rel_ver] = new_ver_hash
    dag_p.write_text(json.dumps(dag_dict, indent=2), encoding="utf-8")

    with pytest.raises(RuntimeError, match="reference verification artifact task_id mismatch"):
        execute_full_experiment_pipeline(spec, force_rerun=False)


def test_resume_fails_when_reference_verification_path_mismatched(tmp_path):
    from experiments.cli import execute_full_experiment_pipeline
    ref_inputs = [
        ReferencePhaseInput(source_id="Li-ref", structure=_struct(["Li"], "Li-ref")),
        ReferencePhaseInput(source_id="P-ref", structure=_struct(["P"], "P-ref")),
        ReferencePhaseInput(source_id="Li3P-ref", structure=_struct(["Li", "Li", "Li", "P"], "Li3P-ref"), source_energy_above_hull_ev_per_atom=0.0),
    ]
    ref_path = tmp_path / "ref_source.json"
    try:
        from agents.thermodynamics import CHGNetRelaxationEvaluator
        chg_eval = CHGNetRelaxationEvaluator()
        evaluator = MockEvaluator({"Li-ref": -1.0, "P-ref": -2.0, "Li3P-ref": -2.5})
        evaluator.model_identity = chg_eval.model_identity
        evaluator.relaxation_settings = chg_eval.relaxation_settings
    except Exception:
        evaluator = MockEvaluator({"Li-ref": -1.0, "P-ref": -2.0, "Li3P-ref": -2.5})
    ref_meta = build_frozen_reference_set(
        reference_set_id="ref_source", chemical_system=["Li", "P"], inputs=ref_inputs,
        evaluator=evaluator, output_path=ref_path, created_at_iso="2026-08-30T00:00:00+00:00",
    )
    source_task = TaskDefinition(
        task_id="source_li_p",
        elements=["Li", "P"],
        domain="Li-P",
        target_properties={"band_gap": 2.0},
        constraints={},
        reference_set_path=str(ref_path).replace("\\", "/"),
        reference_set_sha256=hashlib.sha256(ref_path.read_bytes()).hexdigest(),
    )
    spec = _build_tiny_spec(tmp_path / "ref_path_mismatch_out", source_task=source_task)
    execute_full_experiment_pipeline(spec, force_rerun=True)

    out_dir = Path(spec.output_root)
    ver_p = out_dir / "references" / f"{source_task.task_id}.verified.json"
    ver_dict = json.loads(ver_p.read_text(encoding="utf-8"))

    # Tamper reference_set_path in verification artifact
    ver_dict["reference_set_path"] = str(tmp_path / "non_matching_path.json").replace("\\", "/")
    ver_p.write_text(json.dumps(ver_dict), encoding="utf-8")

    # Update generic file digest in DAG for node_refset_source_li_p to avoid masking
    new_ver_hash = hashlib.sha256(ver_p.read_bytes()).hexdigest()
    dag_p = out_dir / "experiment_dag.json"
    dag_dict = json.loads(dag_p.read_text(encoding="utf-8"))
    ref_nid = f"node_refset_{source_task.task_id}"
    dag_dict["nodes"][ref_nid]["result_hash"] = new_ver_hash
    rel_ver = str(ver_p.relative_to(out_dir)).replace("\\", "/")
    dag_dict["nodes"][ref_nid]["result_artifacts"][rel_ver] = new_ver_hash
    dag_p.write_text(json.dumps(dag_dict, indent=2), encoding="utf-8")

    with pytest.raises(RuntimeError, match="reference verification artifact reference_set_path mismatch"):
        execute_full_experiment_pipeline(spec, force_rerun=False)


@pytest.mark.parametrize(
    "invalid_key",
    [
        "..",
        "../artifact.json",
        "nested/..",
        "nested/../artifact.json",
        "./artifact.json",
        "nested/./artifact.json",
        "nested//artifact.json",
        "nested/",
        "nested\\..\\artifact.json",
        "/abs/artifact.json",
        "C:/drive/artifact.json",
    ],
)
def test_dag_pending_node_with_invalid_path_components_fails(tmp_path, invalid_key):
    spec = _build_tiny_spec(tmp_path / "pending_invalid_path_out")
    dag = ExperimentDAG(spec)
    dag_dict = dag.to_dict()
    pending_id = next(nid for nid, n in dag_dict["nodes"].items() if not n["executed"])
    dag_dict["nodes"][pending_id]["result_artifacts"] = {invalid_key: "0" * 64}
    with pytest.raises(DAGValidationError, match="noncanonical, absolute, empty, or traversing path"):
        ExperimentDAG.from_dict(dag_dict, spec)


def test_dag_pending_node_with_normalized_collision_fails(tmp_path):
    spec = _build_tiny_spec(tmp_path / "collision_out")
    dag = ExperimentDAG(spec)
    dag_dict = dag.to_dict()
    pending_id = next(nid for nid, n in dag_dict["nodes"].items() if not n["executed"])
    dag_dict["nodes"][pending_id]["result_artifacts"] = {
        "nested\\artifact.json": "0" * 64,
        "nested/artifact.json": "1" * 64,
    }
    with pytest.raises(DAGValidationError, match="duplicate keys after path normalization"):
        ExperimentDAG.from_dict(dag_dict, spec)


def test_dag_pending_node_with_malformed_digest_fails(tmp_path):
    spec = _build_tiny_spec(tmp_path / "pending_malformed_digest_out")
    dag = ExperimentDAG(spec)
    dag_dict = dag.to_dict()
    pending_id = next(nid for nid, n in dag_dict["nodes"].items() if not n["executed"])
    dag_dict["nodes"][pending_id]["result_artifacts"] = {"valid/relative/path.json": "invalid-digest"}
    with pytest.raises(DAGValidationError, match="malformed digest"):
        ExperimentDAG.from_dict(dag_dict, spec)


@pytest.mark.parametrize(
    "valid_key",
    [
        "nested/artifact.json",
        "figures/fig1.json",
        "paper/claim_status.json",
    ],
)
def test_dag_pending_node_with_valid_artifact_keys_succeeds(tmp_path, valid_key):
    spec = _build_tiny_spec(tmp_path / "valid_keys_out")
    dag = ExperimentDAG(spec)
    dag_dict = dag.to_dict()
    pending_id = next(nid for nid, n in dag_dict["nodes"].items() if not n["executed"])
    dag_dict["nodes"][pending_id]["result_artifacts"] = {valid_key: "a" * 64}
    hydrated = ExperimentDAG.from_dict(dag_dict, spec)
    assert hydrated.nodes[pending_id].result_artifacts[valid_key] == "a" * 64


# =========================================================================
# Phase 7: Scientific Correctness Hardening & Acceptance Gate
# =========================================================================

def test_orchestrator_locked_elements_and_ambient_key_ignored(monkeypatch, tmp_path):
    from agents.orchestrator import OrchestratorAgent, CampaignObjective
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key-for-test")
    # In canonical benchmark runs, allow_llm=False prevents reading ambient key
    orch = OrchestratorAgent(
        allow_llm=False,
        locked_elements=["Na", "Cl"],
    )
    assert orch.llm_available is False
    assert orch.llm is None

    obj = CampaignObjective(
        target_properties={"density": 2.1},
        constraints={"elements": ["Na", "Cl"]},
        success_criteria={},
    )
    strat = orch.plan_campaign(obj)
    assert strat["elements"] == ["Na", "Cl"]

    # Recommendations/expansions cannot change locked elements
    strat2 = orch.plan_iteration(obj, history=[], recommendations={"elements": ["Na", "Cl", "K"]})
    assert strat2["elements"] == ["Na", "Cl"]

    monkeypatch.setattr("agents.screening.HAS_CHGNET", False)
    campaign = MaterialsDiscoveryCampaign(CampaignConfig(
        name="locked-planner-provenance",
        objective=obj,
        output_dir=tmp_path / "locked_planner",
        use_career_memory=False,
        use_mattergen=False,
        use_validation=False,
        use_synthesis=False,
        locked_elements=["Na", "Cl"],
        allow_llm_orchestration=False,
    ))
    assert campaign.provenance.manifest.config["locked_elements"] == ["Na", "Cl"]
    assert campaign.provenance.manifest.config["allow_llm_orchestration"] is False


def test_reference_set_coverage_manifest_required_and_failed_compound_rejected(tmp_path):
    inputs = [
        ReferencePhaseInput(source_id="Li-ref", structure=_struct(["Li"], "Li-ref")),
        ReferencePhaseInput(source_id="P-ref", structure=_struct(["P"], "P-ref")),
        ReferencePhaseInput(source_id="Li3P-ref", structure=_struct(["Li", "Li", "Li", "P"], "Li3P-ref"), source_energy_above_hull_ev_per_atom=0.0),
        ReferencePhaseInput(source_id="LiP-fail", structure=_struct(["Li", "P"], "LiP-fail")),
    ]
    class PartialFailEvaluator(MockEvaluator):
        def relax(self, value):
            key = value.get("candidate_id", value.get("composition", "Li"))
            if key == "LiP-fail":
                return {"converged": False, "error": "relaxation failed"}
            return super().relax(value)

    evaluator = PartialFailEvaluator({"Li-ref": -1.0, "P-ref": -2.0, "Li3P-ref": -2.5})
    coverage = {
        "manifest_schema_version": "1.0.0",
        "source_dataset": "materials_project",
        "dataset_version": "test-snapshot-v1",
        "snapshot_digest": "3" * 64,
        "chemical_system": ["Li", "P"],
        "selection_procedure": "all unit-test phases",
        "expected_source_phase_ids": ["Li-ref", "P-ref", "Li3P-ref", "LiP-fail"],
        "elemental_endpoint_ids": ["Li-ref", "P-ref"],
        "required_compounds": ["LiP-fail"],
    }
    json_path = tmp_path / "ref_set_cov.json"
    ref_set = build_frozen_reference_set(
        reference_set_id="Li-P-cov-test",
        chemical_system=["Li", "P"],
        inputs=inputs,
        evaluator=evaluator,
        output_path=json_path,
        source_selection=coverage,
    )
    assert ref_set.certification.certified is False
    assert "LiP-fail" in ref_set.certification.missing_expected_source_phases
    assert "LiP-fail" in ref_set.certification.missing_required_compounds


def test_reference_set_without_coverage_manifest_cannot_certify(tmp_path):
    inputs = [
        ReferencePhaseInput(source_id="Li-ref", structure=_struct(["Li"], "Li-ref")),
        ReferencePhaseInput(source_id="P-ref", structure=_struct(["P"], "P-ref")),
        ReferencePhaseInput(
            source_id="Li3P-ref",
            structure=_struct(["Li", "Li", "Li", "P"], "Li3P-ref"),
            source_energy_above_hull_ev_per_atom=0.0,
        ),
    ]
    frozen = _build_frozen_reference_set(
        reference_set_id="no-coverage",
        chemical_system=["Li", "P"],
        inputs=inputs,
        evaluator=MockEvaluator({"Li-ref": -1.0, "P-ref": -2.0, "Li3P-ref": -2.5}),
        output_path=tmp_path / "no_coverage.json",
        source_selection=None,
    )
    assert frozen.certification.certified is False
    assert frozen.certification.coverage_manifest_errors == ["coverage_manifest_required"]
    with pytest.raises(ReferenceSetError, match="not certified"):
        load_frozen_reference_set(tmp_path / "no_coverage.json", require_certified=True)


def test_thermodynamic_oracle_decomposition_products_schema(tmp_path):
    from agents.thermodynamics import ThermodynamicOracle
    inputs = [
        ReferencePhaseInput(source_id="Li-ref", structure=_struct(["Li"], "Li-ref")),
        ReferencePhaseInput(source_id="P-ref", structure=_struct(["P"], "P-ref")),
        ReferencePhaseInput(source_id="Li3P-ref", structure=_struct(["Li", "Li", "Li", "P"], "Li3P-ref"), source_energy_above_hull_ev_per_atom=0.0),
    ]
    evaluator = MockEvaluator({"Li-ref": -1.0, "P-ref": -2.0, "Li3P-ref": -2.5})
    json_path = tmp_path / "ref_set_decomp.json"
    ref_set = build_frozen_reference_set(
        reference_set_id="Li-P-decomp",
        chemical_system=["Li", "P"],
        inputs=inputs,
        evaluator=evaluator,
        output_path=json_path,
    )
    oracle = ThermodynamicOracle(ref_set, evaluator)
    candidate_struct = _struct(["Li", "Li", "P"], "Li2P-cand")
    res = oracle.evaluate(candidate_struct)
    assert res.success is True
    assert len(res.decomposition_products) > 0
    prod = res.decomposition_products[0]
    assert "source_id" in prod
    assert "formula" in prod
    assert "coefficient" in prod
    assert "coefficient_basis" in prod
    assert "structure" in prod
    assert "structure_hash" in prod
    assert prod["reference_set_id"] == "Li-P-decomp"
    from experiments.qe_audit import validate_atom_balanced_reaction
    assert validate_atom_balanced_reaction(candidate_struct, res.decomposition_products)


def test_qe_supercell_multiplier_accounting_and_margin():
    from experiments.qe_audit import (
        _formula_unit_multiplier,
        validate_atom_balanced_reaction,
        compute_local_decomposition_margin,
    )
    cand_struct = _struct(["Li", "Li", "P", "P", "S", "S"], "cand_2x")
    num_atoms, m_cand = _formula_unit_multiplier("LiPS", cand_struct)
    assert num_atoms == 6
    assert m_cand == 2

    p1_struct = _struct(["Li", "Li"], "p1_2x")
    p2_struct = _struct(["P", "P", "S", "S"], "p2_2x")
    num_atoms_p1, m_p1 = _formula_unit_multiplier("Li", p1_struct)
    num_atoms_p2, m_p2 = _formula_unit_multiplier("PS", p2_struct)
    assert m_p1 == 2
    assert m_p2 == 2

    products = [
        {"formula": "Li", "coefficient": 2.0, "structure": p1_struct, "formula_unit_multiplier": 2},
        {"formula": "PS", "coefficient": 2.0, "structure": p2_struct, "formula_unit_multiplier": 2},
    ]
    assert validate_atom_balanced_reaction(cand_struct, products) is True

    phase_results = [
        {"status": "CONVERGED", "total_energy_ev": -4.0},
        {"status": "CONVERGED", "total_energy_ev": -8.0},
    ]
    margin = compute_local_decomposition_margin(-18.0, 6, phase_results, products)
    # Candidate cell: -18 eV. Product cells: 2*(-4) + 2*(-8) = -24 eV.
    # (-18 - -24) / 6 atoms = +1 eV/atom.
    assert margin == pytest.approx(1.0)


def test_confirmatory_holm_family_preserves_multiplicity_with_missing_arm():
    from experiments.statistics import (
        run_statistical_analysis_pipeline,
        PRIMARY_CENSORED_ENDPOINT_NAME,
        THRESHOLD_YIELD_ENDPOINT_NAME,
    )
    run_metrics = [
        {
            "task_id": "target_1",
            "condition": "structured_provenance_memory",
            "seed": 1,
            "provenance_complete": True,
            "oracle_budget": 100,
            PRIMARY_CENSORED_ENDPOINT_NAME: 10,
            THRESHOLD_YIELD_ENDPOINT_NAME: 0.5,
            "reached_0_10_threshold": True,
        },
        {
            "task_id": "target_1",
            "condition": "adaptive_no_memory",
            "seed": 1,
            "provenance_complete": True,
            "oracle_budget": 100,
            PRIMARY_CENSORED_ENDPOINT_NAME: 20,
            THRESHOLD_YIELD_ENDPOINT_NAME: 0.2,
            "reached_0_10_threshold": True,
        },
    ]
    results, manifest = run_statistical_analysis_pipeline(
        run_metrics,
        expected_tasks=["target_1", "target_2"],
        expected_seeds=[1],
    )
    assert manifest["planned_family_size"] == 12
    assert manifest["available_comparison_count"] == 2
    assert manifest["unavailable_comparison_count"] == 10
    assert manifest["manifest"]["auc_definition"]["worst_value_cap_ev_per_atom"] == 1.0
    available_results = [r for r in results if r.p_value_adjusted is not None]
    assert len(available_results) == 2
    for r in available_results:
        assert r.p_value_adjusted >= r.p_value_raw


def test_step_curve_auc_without_backfilling_when_first_call_fails():
    manifest = {
        "manifest": {"status": "completed", "proposals_generated": 2, "oracle_evaluations": 2},
        "candidates": [
            {
                "candidate_id": "c1",
                "composition": "Li",
                "geometry_valid": True,
                "oracle_evaluated": True,
                "oracle_call_index": 1,
                "oracle_success": False,
                "proposal_index": 0,
                "screening_predictions": {},
            },
            {
                "candidate_id": "c2",
                "composition": "Li",
                "geometry_valid": True,
                "oracle_evaluated": True,
                "oracle_call_index": 2,
                "oracle_success": True,
                "proposal_index": 1,
                "screening_predictions": {"predicted_energy_above_hull_ev_per_atom": 0.20},
            },
        ],
    }
    metrics, _ = compute_run_metrics(manifest, "r_test", "task", "adaptive_no_memory", 1, oracle_budget=10)
    assert metrics.area_under_best_curve == pytest.approx(0.28)


def test_step_curve_auc_is_unavailable_for_interrupted_run():
    manifest = {
        "manifest": {"status": "interrupted", "proposals_generated": 1, "oracle_evaluations": 1},
        "candidates": [{
            "candidate_id": "c1",
            "composition": "Li",
            "geometry_valid": True,
            "oracle_evaluated": True,
            "oracle_call_index": 1,
            "oracle_success": True,
            "proposal_index": 0,
            "screening_predictions": {"predicted_energy_above_hull_ev_per_atom": 0.2},
        }],
    }
    metrics, _ = compute_run_metrics(
        manifest, "interrupted", "task", "adaptive_no_memory", 1, oracle_budget=10,
    )
    assert metrics.area_under_best_curve is None
    assert "run_not_completed" in metrics.provenance_missing_fields


def test_qe_selection_exclusions_persist_to_artifacts(tmp_path):
    from experiments.qe_audit import select_audit_candidates

    selection = select_audit_candidates(
        [{"candidate_id": "source", "task_id": "Li-P-S"}], target_count=1,
    )
    assert selection.exclusions_by_reason == {"source_task_excluded": 1}
    runner = QEAuditRunner(QEAuditConfig(mock_execution=True), tmp_path / "qe", FakeQECalculator())
    runner.run_full_audit(selection)
    selection_payload = json.loads((tmp_path / "qe" / "selection.json").read_text(encoding="utf-8"))
    audit_payload = json.loads((tmp_path / "qe" / "audit_manifest.json").read_text(encoding="utf-8"))
    assert selection_payload["exclusions_by_reason"] == {"source_task_excluded": 1}
    assert audit_payload["selection_exclusions_by_reason"] == {"source_task_excluded": 1}


def test_screening_integrity_canonicalizer_oracle_backend():
    from agents.integrity import canonicalize_screening_predictions, SCREENING_ENERGY_KEY
    raw = {
        "energy_per_atom": -3.45,
        "max_force_ev_per_angstrom": 0.02,
        "max_stress_gpa": 0.5,
    }
    res = canonicalize_screening_predictions(raw, backend="chgnet_thermodynamic_oracle")
    assert res["energy_semantics"] == "raw_predicted_per_atom"
    assert SCREENING_ENERGY_KEY in res
    assert res[SCREENING_ENERGY_KEY] == -3.45


def test_end_to_end_full_acceptance_gate(tmp_path, monkeypatch):
    """End-to-end integration test verifying all Phase 2 scientific contracts."""
    from agents.thermodynamics import ThermodynamicOracle
    from agents.screening import ScreeningAgent
    from experiments.qe_audit import select_audit_candidates, QEAuditRunner
    from experiments.cli import _attach_reference_phase_structures, _load_candidate_structure
    from experiments.statistics import run_statistical_analysis_pipeline, PRIMARY_CENSORED_ENDPOINT_NAME, THRESHOLD_YIELD_ENDPOINT_NAME
    from agents.integrity import SCREENING_ENERGY_KEY

    # 1. Ambient API key must not activate LLM
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ambient-unauthorized-token")

    # 2. Build certified reference set for Na-Cl with full coverage manifest
    na_ref = ReferencePhaseInput(source_id="mp-Na", structure=_struct(["Na"], "mp-Na"))
    cl_ref = ReferencePhaseInput(source_id="mp-Cl", structure=_struct(["Cl"], "mp-Cl"))
    nacl_ref = ReferencePhaseInput(source_id="mp-NaCl", structure=_struct(["Na", "Cl"], "mp-NaCl"), source_energy_above_hull_ev_per_atom=0.0)
    evaluator = MockEvaluator({"mp-Na": -1.3, "mp-Cl": -2.1, "mp-NaCl": -3.2})
    ref_path = tmp_path / "nacl_ref.json"
    coverage = {
        "manifest_schema_version": "1.0.0",
        "source_dataset": "materials_project",
        "dataset_version": "test-snapshot-v1",
        "snapshot_digest": "4" * 64,
        "chemical_system": ["Cl", "Na"],
        "selection_procedure": "all unit-test phases",
        "expected_source_phase_ids": ["mp-Na", "mp-Cl", "mp-NaCl"],
        "elemental_endpoint_ids": ["mp-Na", "mp-Cl"],
        "required_compounds": ["mp-NaCl"],
    }
    ref_set = build_frozen_reference_set(
        reference_set_id="Na-Cl-certified",
        chemical_system=["Na", "Cl"],
        inputs=[na_ref, cl_ref, nacl_ref],
        evaluator=evaluator,
        output_path=ref_path,
        source_selection=coverage,
    )
    assert ref_set.certification.certified is True

    # 3. Oracle and Screening produces structured decomposition products and correct energy semantics
    oracle = ThermodynamicOracle(ref_set, evaluator)
    screener = ScreeningAgent(thermodynamic_oracle=oracle)
    cand_struct = _struct(["Na", "Na", "Cl", "Cl"], "cand_nacl_2x") # 2x supercell
    batch_results = screener.screen_batch([cand_struct], criteria={})
    assert len(batch_results) == 1
    _, screening_res = batch_results[0]
    assert screening_res.backend == "chgnet_thermodynamic_oracle"
    thermo_result = oracle.evaluate(cand_struct)
    assert len(thermo_result.decomposition_products) > 0
    decomp_prod = thermo_result.decomposition_products[0]
    assert decomp_prod["reference_set_id"] == "Na-Cl-certified"
    assert decomp_prod["structure"] is not None

    # 4. Persist through the production provenance/metrics hydration path.
    prov_dir = tmp_path / "campaign_provenance"
    tracker = ProvenanceTracker(
        campaign_id="campaign_nacl",
        campaign_name="NaCl acceptance",
        domain="test",
        output_dir=prov_dir,
        master_seed=42,
        config={"proposal_budget": 1, "oracle_budget": 1},
        objective={},
        constraints={"elements": ["Na", "Cl"]},
        run_mode="research",
        scientific_validity="research_valid",
        actual_backends={"screening": "chgnet_thermodynamic_oracle"},
    )
    tracker.register_generation(
        [cand_struct], iteration=0, backend="test", seed=42,
        target_elements=["Na", "Cl"],
    )
    budget = DualBudgetTracker(proposal_budget=1, oracle_budget=1)
    budget.record_proposals(1)
    persisted_batch = screener.screen_batch([cand_struct], criteria={}, budget_tracker=budget)
    tracker.record_screening(
        persisted_batch, criteria={}, iteration=0, backend="chgnet_thermodynamic_oracle"
    )
    tracker.sync_budget(budget, iteration=0)
    tracker.finalize("completed")
    manifest_payload = json.loads(tracker.campaign_json_path.read_text(encoding="utf-8"))
    _, extracted = compute_run_metrics(
        manifest_payload, "run_nacl", "target_nacl",
        "structured_provenance_memory", 42, oracle_budget=1,
    )
    assert len(extracted) == 1
    _load_candidate_structure(extracted[0], prov_dir)
    _attach_reference_phase_structures(extracted[0], str(ref_path))
    assert extracted[0].structure is not None, (
        extracted[0].structure_path,
        extracted[0].provenance_missing_fields,
    )
    assert extracted[0].decomposition_products
    assert extracted[0].decomposition_products[0]["coefficient"] == pytest.approx(2.0)

    selection = select_audit_candidates(extracted, target_count=1, target_tasks=["target_nacl"])
    assert len(selection) == 1, selection.exclusions_by_reason
    selected_cand = selection[0]
    assert len(selected_cand.predicted_decomposition_products) > 0

    qe_cfg = QEAuditConfig(mock_execution=True)
    runner = QEAuditRunner(qe_cfg, tmp_path / "qe_out", FakeQECalculator())
    audit_res = runner.audit_candidate(selected_cand)
    assert audit_res.reaction_balanced is True
    assert audit_res.candidate_provenance["candidate_formula_unit_multiplier"] == 2

    # 5. Holm family preservation with missing arm
    stat_records = [
        {
            "task_id": "target_nacl",
            "condition": "structured_provenance_memory",
            "seed": 42,
            "provenance_complete": True,
            "oracle_budget": 100,
            PRIMARY_CENSORED_ENDPOINT_NAME: 15,
            THRESHOLD_YIELD_ENDPOINT_NAME: 0.6,
            "reached_0_10_threshold": True,
        },
        {
            "task_id": "target_nacl",
            "condition": "adaptive_no_memory",
            "seed": 42,
            "provenance_complete": True,
            "oracle_budget": 100,
            PRIMARY_CENSORED_ENDPOINT_NAME: 35,
            THRESHOLD_YIELD_ENDPOINT_NAME: 0.1,
            "reached_0_10_threshold": True,
        },
    ]
    stat_results, stat_manifest = run_statistical_analysis_pipeline(
        stat_records,
        expected_tasks=["target_nacl", "target_other"],
        expected_seeds=[42],
        output_dir=tmp_path / "stats_out",
    )
    assert stat_manifest["planned_family_size"] == 12
    assert stat_manifest["available_comparison_count"] == 2
    assert stat_manifest["unavailable_comparison_count"] == 10

    # 6. Step curve AUC
    run_manifest_payload = {
        "manifest": {"status": "completed", "proposals_generated": 2, "oracle_evaluations": 2},
        "candidates": [
            {
                "candidate_id": "c1",
                "composition": "NaCl",
                "geometry_valid": True,
                "oracle_evaluated": True,
                "oracle_call_index": 1,
                "oracle_success": False,
                "proposal_index": 0,
                "screening_predictions": {},
            },
            {
                "candidate_id": "c2",
                "composition": "NaCl",
                "geometry_valid": True,
                "oracle_evaluated": True,
                "oracle_call_index": 2,
                "oracle_success": True,
                "proposal_index": 1,
                "screening_predictions": {"predicted_energy_above_hull_ev_per_atom": 0.15},
            },
        ],
    }
    metrics, _ = compute_run_metrics(run_manifest_payload, "r_gate", "target_nacl", "structured_provenance_memory", 42, oracle_budget=10)
    # Step 1: cap = 1.0; Step 2: 0.15; Steps 3..10: 0.15 -> sum = 1.0 + 0.15 + 8*0.15 = 2.35 -> AUC = 2.35 / 10 = 0.235
    assert metrics.area_under_best_curve == pytest.approx(0.235)
