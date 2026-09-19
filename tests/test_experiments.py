"""Sprint 5 acceptance test suite for experiment DAG, statistical protocol, and QE audit."""

import json
from pathlib import Path
import pytest

from experiments.dag import DAGNode, DAGValidationError, ExperimentDAG, NodeType
from experiments.memory_snapshots import (
    MemorySnapshotError,
    MemorySnapshotManager,
    compute_file_sha256,
)
from experiments.metrics import (
    CandidateRecord,
    compute_run_metrics,
    extract_candidates_from_manifest,
)
from experiments.qe_audit import (
    FakeQECalculator,
    QEAuditCandidate,
    QEAuditRunner,
    QECalculationStatus,
    generate_kpoints_grid,
    select_audit_candidates,
)
from experiments.report import ReportGenerator
from experiments.runner import CampaignRunner, CampaignRunnerError, RunTerminalState
from experiments.spec import (
    BASE_COMMIT_SHA,
    FIVE_CONDITIONS,
    MEMORY_ARMS,
    ExperimentSpec,
    ExperimentSpecError,
    QEAuditConfig,
    RunSpec,
    SPEC_SCHEMA_VERSION,
    TaskDefinition,
    TransferDeclaration,
    compute_sha256,
)
from experiments.statistics import (
    bootstrap_paired_ci,
    compute_paired_comparison,
    holm_bonferroni_adjust,
    paired_permutation_test_pvalue,
    run_statistical_analysis_pipeline,
)


def _qe_structure(species):
    return {
        "lattice": [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
        "positions": [[0.0, 0.0, i / max(1, len(species))] for i in range(len(species))],
        "species": list(species),
        "fractional_coordinates": True,
    }


def test_spec_canonical_hashing_and_fail_closed():
    spec1 = ExperimentSpec(experiment_id="exp_001", master_seeds=[42, 137])
    spec2 = ExperimentSpec(experiment_id="exp_001", master_seeds=[42, 137])
    assert spec1.spec_hash == spec2.spec_hash
    assert spec1.code_commit == BASE_COMMIT_SHA

    # Fails closed on unknown keys
    data = spec1.to_dict()
    data["unknown_random_key"] = "malicious_payload"
    with pytest.raises(ExperimentSpecError, match="Unknown keys"):
        ExperimentSpec.from_dict(data)

    # Fails closed on empty seeds
    with pytest.raises(ExperimentSpecError, match="master_seeds list cannot be empty"):
        ExperimentSpec(experiment_id="bad", master_seeds=[])

    # Fails closed on invalid condition
    with pytest.raises(ExperimentSpecError, match="Unknown condition"):
        ExperimentSpec(experiment_id="bad", conditions=["invalid_arm_name"])


def test_dag_construction_exact_run_counts_and_validation():
    # 5 conditions x 5 seeds x 2 targets = 50 target runs + 5 source runs = 55 total runs
    spec = ExperimentSpec(
        experiment_id="exp_5x5",
        master_seeds=[1, 2, 3, 4, 5],
        conditions=list(FIVE_CONDITIONS),
    )
    dag = ExperimentDAG(spec)
    
    assert dag.total_run_count == 55
    target_nodes = [n for n in dag.nodes.values() if n.node_type == NodeType.TARGET_CAMPAIGN_RUN]
    assert len(target_nodes) == 50
    source_nodes = [n for n in dag.nodes.values() if n.node_type == NodeType.SOURCE_MEMORY_RUN]
    assert len(source_nodes) == 5

    order = dag.topological_order()
    assert len(order) == len(dag.nodes)
    assert order[0].node_type == NodeType.PREFLIGHT

    # Cycle detection
    bad_node = DAGNode(
        node_id="cycle_node",
        node_type=NodeType.PREFLIGHT,
        description="cyclic node",
        dependencies={"node_report_generation"},
    )
    dag.nodes["node_preflight"].dependencies.add("node_report_generation")
    with pytest.raises(DAGValidationError, match="Cycle detected"):
        dag.validate()


def test_source_snapshot_immutability_and_byte_identity(tmp_path):
    # Create fake source career memory database
    import sqlite3
    db_path = tmp_path / "source.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE campaigns (id TEXT PRIMARY KEY, name TEXT, domain TEXT, start_time REAL, end_time REAL, iterations INT, total_generated INT, total_screened INT)"
    )
    conn.execute(
        "CREATE TABLE transferable_records (record_id TEXT PRIMARY KEY, schema_version TEXT, evidence_hash TEXT, source_domain TEXT, payload TEXT)"
    )
    conn.execute(
        "INSERT INTO campaigns VALUES ('c1', 'source', 'materials', 0.0, 10.0, 5, 20, 10)"
    )
    conn.execute(
        "INSERT INTO transferable_records VALUES ('r1', '2.0.0', 'hash_1', 'materials', '{\"record_id\": \"r1\"}')"
    )
    conn.commit()
    conn.close()

    snap_path = tmp_path / "snapshot.db"
    meta = MemorySnapshotManager.create_snapshot(
        source_db_path=db_path,
        destination_snapshot_path=snap_path,
        source_task="Li-P-S",
        master_seed=42,
    )
    assert meta.sqlite_file_sha256 == compute_file_sha256(snap_path)
    assert meta.record_ids == ["r1"]

    # Verify snapshot integrity
    assert MemorySnapshotManager.verify_snapshot_integrity(
        snap_path, expected_sqlite_sha256=meta.sqlite_file_sha256
    )

    # Clones for target runs are byte-identical
    target1_db = tmp_path / "target1.db"
    target2_db = tmp_path / "target2.db"
    MemorySnapshotManager.clone_for_target_run(snap_path, target1_db)
    MemorySnapshotManager.clone_for_target_run(snap_path, target2_db)
    # The source snapshot remains untouched and immutable
    assert compute_file_sha256(snap_path) == meta.sqlite_file_sha256
    # Target clones are identical to each other
    assert compute_file_sha256(target1_db) == compute_file_sha256(target2_db)


def test_hand_calculated_metrics_and_censoring():
    # Construct synthetic manifest
    manifest_data = {
        "proposals_generated": 5,
        "geometry_valid": 4,
        "invalid_geometry": 1,
        "oracle_evaluations": 3,
        "memory_directives_applied": [{"record_id": "r1"}],
        "candidates": [
            {
                "candidate_id": "c1",
                "formula": "LiPS",
                "geometry_valid": True,
                "evaluated_by_oracle": True,
                "oracle_call_index": 1,
                "predictions": {"predicted_energy_above_hull_ev_per_atom": 0.15},
            },
            {
                "candidate_id": "c2",
                "formula": "Li2PS3",
                "geometry_valid": True,
                "evaluated_by_oracle": True,
                "oracle_call_index": 2,
                "predictions": {"predicted_energy_above_hull_ev_per_atom": 0.08},
            },
            {
                "candidate_id": "c3",
                "formula": "Li3PS4",
                "geometry_valid": True,
                "evaluated_by_oracle": True,
                "oracle_call_index": 3,
                "predictions": {"predicted_energy_above_hull_ev_per_atom": 0.02},
            },
            {
                "candidate_id": "c4",
                "formula": "BadGeom",
                "geometry_valid": False,
                "screening_status": "INVALID_GEOMETRY",
                "evaluated_by_oracle": False,
            },
            {
                "candidate_id": "c5",
                "formula": "LiP2S",
                "geometry_valid": True,
                "evaluated_by_oracle": False,
            },
        ],
    }

    metrics, cands = compute_run_metrics(
        manifest_data=manifest_data,
        run_id="run_test",
        task_id="Li-P-Se",
        condition="structured_provenance_memory",
        seed=42,
        oracle_budget=10,
    )

    assert metrics.proposals_generated == 5
    assert metrics.geometry_valid_count == 4
    assert metrics.invalid_geometry_count == 1
    assert metrics.geometry_yield == pytest.approx(4.0 / 5.0)
    assert metrics.oracle_evaluations == 3
    assert metrics.oracle_calls_to_first_candidate_at_or_below_0_10 == 2  # candidate c2 reached at call 2
    assert metrics.reached_0_10_threshold is True
    assert metrics.count_at_or_below_0_10 == 2  # c2 (0.08) and c3 (0.02)
    assert metrics.count_at_or_below_0_03 == 1  # c3 (0.02)
    assert metrics.fraction_at_or_below_0_10 == pytest.approx(2.0 / 3.0)
    assert metrics.best_energy_above_hull_overall == pytest.approx(0.02)


def test_hand_calculated_statistics_and_holm_bonferroni():
    # Hand-calculated paired differences
    diffs = [0.10, 0.20, 0.30, 0.40, 0.50]
    mean, lower, upper = bootstrap_paired_ci(diffs, n_resamples=5000, random_seed=42)
    assert mean == pytest.approx(0.30)
    assert 0.15 <= lower <= 0.25
    assert 0.35 <= upper <= 0.45

    # Multiplicity adjustment test with known toy p-values
    raw_p = [0.01, 0.04, 0.03]
    # Sorted: p1=0.01 (m=3 -> 0.03), p2=0.03 (m=2 -> 0.06), p3=0.04 (m=1 -> 0.04 -> cummax 0.06)
    adj = holm_bonferroni_adjust(raw_p)
    assert adj[0] == pytest.approx(0.03)  # 0.01 * 3
    assert adj[1] == pytest.approx(0.06)  # max(0.06, 0.04 * 1) = 0.06
    assert adj[2] == pytest.approx(0.06)  # 0.03 * 2


def test_qe_audit_selection_and_local_decomposition_margin(tmp_path):
    # Create fake evaluated candidates
    cands = []
    for i in range(12):
        alkali = "Li" if i % 2 == 0 else "Na"
        chalcogen = "Se" if i % 2 == 0 else "S"
        count = i + 1
        formula = f"{alkali}{count}P{chalcogen}"
        phase_formula = f"{alkali}{count}P"
        cands.append(CandidateRecord(
            candidate_id=f"c_{i}",
            run_id=f"run_{i}",
            task_id="Li-P-Se" if i % 2 == 0 else "Na-P-S",
            condition="structured_provenance_memory" if i < 5 else "adaptive_no_memory",
            seed=42 + i,
            iteration=0,
            proposal_index=i,
            oracle_call_index=i,
            reduced_formula=formula,
            anonymous_stoichiometry="AB",
            structural_prototype="prototype_1",
            geometry_valid=True,
            geometry_failure_reason=None,
            evaluated_by_oracle=True,
            oracle_success=True,
            oracle_failure_code=None,
            predicted_energy_above_hull_ev_per_atom=0.02 * i,
            predicted_thermodynamically_stable=i < 3,
            retained_by_hull=True,
            structure=_qe_structure([alkali] * count + ["P", chalcogen]),
            decomposition_products=[
                {
                    "formula": phase_formula,
                    "amount": 1.0,
                    "structure": _qe_structure([alkali] * count + ["P"]),
                },
                {
                    "formula": chalcogen,
                    "amount": 1.0,
                    "structure": _qe_structure([chalcogen]),
                },
            ],
        ))

    selected = select_audit_candidates(cands, target_count=5)
    assert len(selected) == 5
    assert len({c.candidate_id for c in selected}) == 5

    # Test k-points generation
    k1, k2, k3 = generate_kpoints_grid([10.0, 10.0, 10.0], target_spacing_inv_ang=0.25)
    assert k1 >= 2 and k2 >= 2 and k3 >= 2

    # Test QE Runner with Fake Calculator
    config = QEAuditConfig(candidate_count=5)
    runner = QEAuditRunner(config=config, output_dir=tmp_path)
    res = runner.audit_candidate(selected[0])

    assert res.candidate_status == QECalculationStatus.CONVERGED.value
    assert res.competing_phases_all_converged is True
    assert res.dft_local_decomposition_margin_ev_per_atom is not None
    assert isinstance(res.reaction_equation, str)

    # Full audit
    all_res = runner.run_full_audit(selected)
    assert len(all_res) == 5
    assert (tmp_path / "results.csv").exists()


def test_runner_safe_resume_and_hash_verified_skip(tmp_path):
    placeholder_dir = tmp_path / "presentation_placeholder"
    placeholder_dir.mkdir(parents=True, exist_ok=True)
    manifest_data = {
        "provenance_stage": "completed",
        "total_proposals": 10,
        "proposals_generated": 10,
        "oracle_evaluations": 5,
    }
    (placeholder_dir / "manifest.json").write_text(json.dumps(manifest_data))
    
    # A presentation-style manifest without a spec/integrity sidecar must not
    # be accepted as a resumable scientific run.
    assert CampaignRunner.is_run_completed(placeholder_dir) is False
    
    spec = RunSpec(
        run_id="run_skip_test",
        experiment_id="test",
        task_id="Li-P-Se",
        elements=["Li", "P", "Se"],
        condition="adaptive_no_memory",
        seed=42,
        iteration_seeds=[42],
        proposal_budget=10,
        oracle_budget=5,
        geometry_min_distance=0.8,
        thermodynamics_retain_threshold_ev_per_atom=0.10,
        thermodynamics_stable_threshold_ev_per_atom=0.03,
        run_mode="development",
        generation_backend="mock",
        memory_mode="none",
        memory_seed=42,
        output_dir=str(placeholder_dir).replace("\\", "/"),
    )
    with pytest.raises(CampaignRunnerError, match="non-empty output directory"):
        CampaignRunner.execute_run(spec, force_rerun=True)

    # A forced run may execute only in a genuinely fresh directory.
    output_dir = tmp_path / "completed_run"
    spec = RunSpec(**{**spec.to_dict(), "output_dir": str(output_dir).replace("\\", "/")})
    first = CampaignRunner.execute_run(spec, force_rerun=True)
    assert first["skipped"] is False
    res = CampaignRunner.execute_run(spec, force_rerun=False)
    assert res["skipped"] is True
    assert res["status"] == RunTerminalState.SUCCESS.value


def test_preflight_fails_closed_on_invalid_commit_or_budgets(tmp_path):
    from experiments.cli import run_preflight_check
    
    # Valid preflight
    valid_spec = ExperimentSpec(
        experiment_id="valid_preflight",
        output_root=str(tmp_path / "valid").replace("\\", "/"),
    )
    res = run_preflight_check(valid_spec)
    assert res["status"] == "PASSED"
    
    # Invalid base commit
    bad_spec = ExperimentSpec(
        experiment_id="bad_commit",
        code_commit="unauthorized_dirty_commit_00000",
        output_root=str(tmp_path / "bad").replace("\\", "/"),
    )
    with pytest.raises(RuntimeError, match="Preflight validation failed"):
        run_preflight_check(bad_spec)


def test_qe_audit_runner_failure_handling(tmp_path):
    config = QEAuditConfig(candidate_count=2)
    fake_calc = FakeQECalculator(failure_modes={"Li2PSe3": QECalculationStatus.SCF_FAILED.value})
    runner = QEAuditRunner(config=config, output_dir=tmp_path, calculator=fake_calc)
    
    cand = QEAuditCandidate(
        candidate_id="cand_fail",
        target_task="Li-P-Se",
        condition="structured_provenance_memory",
        seed=42,
        reduced_formula="Li2PSe3",
        structure=_qe_structure(["Li", "Li", "P", "Se", "Se", "Se"]),
        predicted_energy_above_hull_ev_per_atom=0.05,
        predicted_decomposition_products=[
            {
                "formula": "Li2P",
                "amount": 1.0,
                "structure": _qe_structure(["Li", "Li", "P"]),
            },
            {
                "formula": "Se",
                "amount": 3.0,
                "structure": _qe_structure(["Se"]),
            },
        ],
        selection_rank=1,
        selection_reason="Test failure handling",
    )
    res = runner.audit_candidate(cand)
    assert res.candidate_status == QECalculationStatus.SCF_FAILED.value
    assert res.dft_local_decomposition_margin_ev_per_atom is None


def test_end_to_end_tiny_experiment_pipeline(tmp_path):
    """Executes all 5 conditions, aggregation, statistics, QE audit, and report generation."""
    spec = ExperimentSpec(
        experiment_id="tiny_integration_test",
        master_seeds=[42, 137],
        proposals_per_run=4,
        oracle_budget_per_run=2,
        iterations_per_run=1,
        output_root=str(tmp_path / "benchmark_out").replace("\\", "/"),
    )

    from experiments.cli import execute_full_experiment_pipeline
    result = execute_full_experiment_pipeline(spec, force_rerun=True)

    assert result["status"] == "SUCCESS"
    out_dir = Path(spec.output_root)
    assert (out_dir / "preflight.json").exists()
    assert (out_dir / "experiment_dag.json").exists()
    assert (out_dir / "aggregates" / "runs.json").exists()
    assert (out_dir / "statistics" / "effects.csv").exists()
    assert (out_dir / "statistics" / "analysis_manifest.json").exists()
    assert (out_dir / "qe_audit" / "results.csv").exists()
    assert (out_dir / "paper" / "claim_evidence_matrix.md").exists()
    assert (out_dir / "paper" / "reproducibility_checklist.md").exists()
    assert (out_dir / "figures" / "fig1_best_so_far_vs_oracle_calls.json").exists()
    assert (out_dir / "figures" / "fig2_time_to_threshold_censored.json").exists()
    assert (out_dir / "figures" / "fig3_paired_effect_sizes.json").exists()
    assert (out_dir / "figures" / "fig4_geometry_and_oracle_failures.json").exists()
    assert (out_dir / "figures" / "fig5_memory_directives_breakdown.json").exists()
    assert (out_dir / "figures" / "fig6_shuffled_control_validation.json").exists()
    assert (out_dir / "figures" / "fig7_chgnet_vs_qe_local_decomposition.json").exists()
    assert (out_dir / "tables" / "table_threshold_sensitivity.json").exists()
