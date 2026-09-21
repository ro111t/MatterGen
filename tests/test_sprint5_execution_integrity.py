"""Adversarial execution-boundary tests for the Sprint 5 repair.

These tests deliberately exercise the fail-closed paths; they do not require
MatterGen, QE, CHGNet, or network access.
"""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess

import pytest

import experiments.cli as experiment_cli
import experiments.runner as experiment_runner
from experiments.cli import (
    _build_run_spec,
    _find_reusable_snapshot,
    _shard_manifest_payload,
    execute_full_experiment_pipeline,
    merge_shard_outputs,
    run_preflight_check,
)
from agents.career_memory import CareerMemory
from agents.provenance import RunManifest
from experiments.dag import DAGNode, DAGValidationError, ExperimentDAG, NodeType
from experiments.memory_snapshots import MemorySnapshotError, MemorySnapshotManager
from experiments.runner import CampaignRunner, CampaignRunnerError
from experiments.spec import (
    FIVE_CONDITIONS,
    ExperimentSpec,
    ExperimentSpecError,
    RunSpec,
    TaskDefinition,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_db(path: Path, *, finalized: bool = True, with_evidence: bool = True) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE campaigns (id TEXT PRIMARY KEY, name TEXT, domain TEXT,
          start_time REAL, end_time REAL, iterations INT, total_generated INT,
          total_screened INT, best_score REAL, success_rate REAL, summary TEXT);
        CREATE TABLE transferable_records (record_id TEXT PRIMARY KEY,
          schema_version TEXT, evidence_hash TEXT, source_domain TEXT,
          campaign_ids TEXT, source_candidate_ids TEXT, source_formulas TEXT,
          outcome_label TEXT, outcome_value REAL, evidence_count INT,
          confidence REAL, finalized INT, payload TEXT, created_at REAL);
        """
    )
    conn.execute("INSERT INTO campaigns VALUES ('c1','source','materials',0,?,1,1,1,0,0,'{}')", (1 if finalized else None,))
    if with_evidence:
        conn.execute(
            "INSERT INTO transferable_records VALUES ('r1','2.0.0','e1','materials','[\"c1\"]','[]','[]','positive',1,1,1,?,'{}',0)",
            (1 if finalized else 0,),
        )
    conn.commit(); conn.close()


def _write_completed_run(spec: RunSpec, *, source_snapshot: Path | None = None) -> None:
    output = Path(spec.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_spec.json").write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    memory_db = output / "career_memory.db"
    if spec.condition == "source_neutral":
        _source_db(memory_db)
    elif spec.condition in {"text_summary_memory", "structured_provenance_memory", "shuffled_memory_control"}:
        assert source_snapshot is not None
        source_snapshot = Path(source_snapshot)
        shutil.copyfile(source_snapshot, memory_db)
        digest = _sha(memory_db)
        (output / "memory_clone_integrity.json").write_text(json.dumps({
            "run_id": spec.run_id,
            "source_snapshot": str(source_snapshot.resolve()),
            "source_sha256": _sha(source_snapshot),
            "clone_sha256": digest,
            "clone_sha256_before_execution": digest,
            "memory_transfer_declaration": spec.memory_transfer_declaration,
            "finalized": True,
        }), encoding="utf-8")
    manifest = RunManifest(
        campaign_id=f"campaign-{spec.run_id}", campaign_name=spec.run_id,
        master_seed=spec.seed, iteration_seeds=list(spec.iteration_seeds),
        proposal_budget=spec.proposal_budget, oracle_budget=spec.oracle_budget,
        status="completed",
    )
    manifest.manifest_hash = manifest.compute_manifest_hash()
    manifest_data = manifest.to_dict()
    (output / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")
    provenance = {
        "campaign_id": manifest.campaign_id,
        "campaign_name": manifest.campaign_name,
        "manifest": manifest_data,
        "candidates": [],
    }
    (output / "campaign_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    (output / "report.json").write_text("{}", encoding="utf-8")
    artifacts = {
        "manifest.json": _sha(output / "manifest.json"),
        "campaign_provenance.json": _sha(output / "campaign_provenance.json"),
        "report.json": _sha(output / "report.json"),
    }
    if memory_db.exists():
        artifacts["career_memory.db"] = _sha(memory_db)
    marker = output / "memory_clone_integrity.json"
    if marker.exists():
        artifacts["memory_clone_integrity.json"] = _sha(marker)
    (output / "run_integrity.json").write_text(json.dumps({
        "schema_version": 1,
        "run_id": spec.run_id,
        "spec_hash": spec.spec_hash,
        "manifest_sha256": artifacts["manifest.json"],
        "artifacts": artifacts,
    }), encoding="utf-8")


def _write_completed_shard(root: Path, canonical_spec: ExperimentSpec, seed: int, worker_id: str) -> None:
    shard_data = canonical_spec.to_dict()
    shard_data["output_root"] = str(root)
    shard_spec = ExperimentSpec.from_dict(shard_data)
    dag = ExperimentDAG(shard_spec, seed_subset=[seed], worker_id=worker_id)
    source_node = next(node for node in dag.nodes.values() if node.node_type == NodeType.SOURCE_MEMORY_RUN)
    source_spec = _build_run_spec(shard_spec, source_node, dag)
    _write_completed_run(source_spec)
    snapshot_node = dag.nodes[f"node_snapshot_source_seed{seed}"]
    snapshot = MemorySnapshotManager.create_snapshot(
        Path(source_spec.output_dir) / "career_memory.db",
        Path(snapshot_node.payload["snapshot_path"]),
        shard_spec.source_task.task_id,
        seed,
        allowed_transfer_declarations=[asdict(declaration) for declaration in shard_spec.transfer_declarations],
        content_addressed=True,
    )
    snapshot_node.payload["snapshot_path"] = str(snapshot.snapshot_path)
    snapshot_node.payload["snapshot_sha256"] = snapshot.sqlite_file_sha256
    snapshot_node.expected_output_path = str(snapshot.snapshot_path)
    for node in dag.nodes.values():
        if snapshot_node.node_id in node.dependencies:
            node.payload["source_memory_snapshot_path"] = str(snapshot.snapshot_path)
            node.payload["source_memory_snapshot_sha256"] = snapshot.sqlite_file_sha256
    for node in dag.nodes.values():
        if node.node_type != NodeType.TARGET_CAMPAIGN_RUN:
            continue
        run_spec = _build_run_spec(shard_spec, node, dag)
        _write_completed_run(
            run_spec,
            source_snapshot=snapshot.snapshot_path if run_spec.condition in {
                "text_summary_memory", "structured_provenance_memory", "shuffled_memory_control"
            } else None,
        )
    for node in dag.nodes.values():
        node.executed = True
        node.success = True
    (root / "shard_manifest.json").write_text(
        json.dumps(_shard_manifest_payload(shard_spec, dag, completed=True)), encoding="utf-8"
    )


def test_merged_shards_resume_relocated_runs_and_reach_statistics_once(tmp_path, monkeypatch):
    merged_root = tmp_path / "merged"
    canonical_spec = ExperimentSpec(
        experiment_id="relocation",
        master_seeds=[42, 137],
        target_tasks=[TaskDefinition(task_id="Li-P-Se", elements=["Li", "P", "Se"])],
        proposals_per_run=2,
        oracle_budget_per_run=1,
        iterations_per_run=1,
        output_root=str(merged_root),
    )
    first_root, second_root = tmp_path / "worker-a", tmp_path / "worker-b"
    _write_completed_shard(first_root, canonical_spec, 42, "worker-a")
    _write_completed_shard(second_root, canonical_spec, 137, "worker-b")

    merge_shard_outputs(canonical_spec, [first_root, second_root], merged_root)

    canonical_dag = ExperimentDAG(canonical_spec)
    adaptive_node = next(
        node for node in canonical_dag.nodes.values()
        if node.node_type == NodeType.TARGET_CAMPAIGN_RUN
        and node.payload["condition"] == "adaptive_no_memory"
        and node.payload["seed"] == 42
    )
    adaptive_spec = _build_run_spec(canonical_spec, adaptive_node, canonical_dag)
    assert CampaignRunner.is_run_completed(Path(adaptive_spec.output_dir), adaptive_spec)
    assert not CampaignRunner.is_run_completed(
        Path(adaptive_spec.output_dir), replace(adaptive_spec, proposal_budget=3)
    )

    class UnexpectedCampaign:
        def __init__(self, *args, **kwargs):
            raise AssertionError("a verified relocated run was re-executed")

    class ReachedStatistics(RuntimeError):
        pass

    statistics_calls = []
    real_statistics = experiment_cli.run_statistical_analysis_pipeline

    def run_statistics_once(**kwargs):
        statistics_calls.append(kwargs)
        real_statistics(**kwargs)
        raise ReachedStatistics("canonical statistics reached")

    monkeypatch.setattr(experiment_runner, "MaterialsDiscoveryCampaign", UnexpectedCampaign)
    monkeypatch.setattr(experiment_cli, "run_statistical_analysis_pipeline", run_statistics_once)
    with pytest.raises(ReachedStatistics, match="canonical statistics reached"):
        execute_full_experiment_pipeline(canonical_spec)

    assert len(statistics_calls) == 1
    assert (merged_root / "aggregates" / "runs.json").is_file()
    assert (merged_root / "statistics" / "analysis_manifest.json").is_file()
    memory_node = next(
        node for node in canonical_dag.nodes.values()
        if node.node_type == NodeType.TARGET_CAMPAIGN_RUN
        and node.payload["condition"] == "structured_provenance_memory"
        and node.payload["seed"] == 42
    )
    memory_snapshot = next((merged_root / "memory_snapshots").glob("source_Li-P-S_seed42_*.db"))
    memory_node.payload["source_memory_snapshot_path"] = str(memory_snapshot)
    memory_node.payload["source_memory_snapshot_sha256"] = _sha(memory_snapshot)
    memory_spec = _build_run_spec(canonical_spec, memory_node, canonical_dag)
    assert CampaignRunner.is_run_completed(Path(memory_spec.output_dir), memory_spec)


def test_duplicate_conditions_rejected():
    with pytest.raises(ExperimentSpecError, match="duplicates"):
        ExperimentSpec(experiment_id="dup", conditions=[FIVE_CONDITIONS[0]] * 5)


def test_research_spec_rejects_mock_and_missing_pins():
    with pytest.raises(ExperimentSpecError, match="generation_backend"):
        ExperimentSpec(experiment_id="research", run_mode="research")


def test_preflight_rejects_dirty_research_and_reference_hash_failure(tmp_path, monkeypatch):
    ref = tmp_path / "refs.json"; ref.write_text("{}", encoding="utf-8")
    # Construct without invoking strict research __post_init__ so this test
    # specifically covers the preflight file/hash/certification boundary.
    spec = ExperimentSpec(
        experiment_id="research-preflight", run_mode="development", output_root=str(tmp_path / "out"),
    )
    bad = spec.source_task
    object.__setattr__(bad, "reference_set_path", str(ref))
    object.__setattr__(bad, "reference_set_sha256", "0" * 64)
    object.__setattr__(bad, "reference_set_certified", True)
    object.__setattr__(spec, "run_mode", "research")
    object.__setattr__(spec, "generation_backend", "mattergen")
    real_check_output = subprocess.check_output

    def _fake_check_output(cmd, **kwargs):
        if isinstance(cmd, (list, tuple)) and cmd and cmd[0] == "git":
            return "HEAD" if any("rev-parse" in str(arg) for arg in cmd) else "experiments/dirty.py"
        return real_check_output(cmd, **kwargs)

    monkeypatch.setattr("experiments.cli.subprocess.check_output", _fake_check_output)
    object.__setattr__(spec, "code_commit", "HEAD")
    with pytest.raises(RuntimeError, match="Preflight validation failed"):
        run_preflight_check(spec)


def test_snapshot_requires_finalized_nonempty_source_and_is_no_overwrite(tmp_path):
    source = tmp_path / "source.db"; _source_db(source, finalized=False)
    with pytest.raises(MemorySnapshotError, match="finalized"):
        MemorySnapshotManager.create_snapshot(source, tmp_path / "snap.db", "Li-P-S", 1)
    source.unlink(); _source_db(source)
    snap = tmp_path / "snap.db"
    meta = MemorySnapshotManager.create_snapshot(source, snap, "Li-P-S", 1)
    assert _sha(snap) == meta.sqlite_file_sha256
    with pytest.raises(MemorySnapshotError, match="overwrite"):
        MemorySnapshotManager.create_snapshot(source, snap, "Li-P-S", 1)


def test_dag_dependency_failure_is_not_skippable():
    spec = ExperimentSpec(experiment_id="dag", master_seeds=[1])
    dag = ExperimentDAG(spec)
    dag.nodes["node_preflight"].executed = True
    dag.nodes["node_preflight"].success = False
    with pytest.raises(DAGValidationError, match="did not succeed"):
        dag.assert_dependencies_succeeded("node_refset_Li-P-S")


def test_completed_run_requires_integrity_sidecar(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "completed", "total_proposals": 1}))
    assert CampaignRunner.is_run_completed(tmp_path) is False


def test_run_spec_condition_configuration_is_distinct(tmp_path):
    base = dict(
        run_id="r", experiment_id="x", task_id="Li-P-Se", elements=["Li", "P", "Se"], seed=1,
        iteration_seeds=[1], proposal_budget=4, oracle_budget=2,
        geometry_min_distance=.8, thermodynamics_retain_threshold_ev_per_atom=.1,
        thermodynamics_stable_threshold_ev_per_atom=.03, run_mode="development",
        generation_backend="mock", memory_seed=1, output_dir=str(tmp_path),
    )
    random = RunSpec(condition="random_mattergen", memory_mode="none", strategy_mode="fixed", **base)
    adaptive = RunSpec(condition="adaptive_no_memory", memory_mode="none", **base)
    assert random.condition != adaptive.condition
    assert random.memory_mode == adaptive.memory_mode == "none"
    assert random.strategy_mode == "fixed"
    assert adaptive.strategy_mode == "adaptive"
    assert random.spec_hash != adaptive.spec_hash


def _memory_run_spec(output_dir: Path, snapshot: Path, snapshot_sha: str) -> RunSpec:
    return RunSpec(
        run_id="target-memory-seed1", experiment_id="isolation", task_id="Li-P-Se",
        elements=["Li", "P", "Se"], condition="structured_provenance_memory", seed=1,
        iteration_seeds=[1], proposal_budget=4, oracle_budget=2,
        geometry_min_distance=.8, thermodynamics_retain_threshold_ev_per_atom=.1,
        thermodynamics_stable_threshold_ev_per_atom=.03, run_mode="development",
        generation_backend="mock", memory_mode="structured_provenance", memory_seed=1,
        output_dir=str(output_dir), source_memory_snapshot_path=str(snapshot),
        source_memory_snapshot_sha256=snapshot_sha,
    )


def test_tampered_target_clone_fails_before_campaign_construction(tmp_path):
    source = tmp_path / "source.db"
    _source_db(source)
    snapshot = tmp_path / "source_snapshot.db"
    metadata = MemorySnapshotManager.create_snapshot(source, snapshot, "Li-P-S", 1)
    output = tmp_path / "target"
    output.mkdir()
    clone = output / "career_memory.db"
    MemorySnapshotManager.clone_for_target_run(snapshot, clone)
    (output / "memory_clone_integrity.json").write_text(json.dumps({
        "run_id": "target-memory-seed1",
        "source_snapshot": str(snapshot.resolve()).replace("\\", "/"),
        "source_sha256": metadata.sqlite_file_sha256,
        "clone_sha256": _sha(clone),
        "memory_transfer_declaration": None,
    }), encoding="utf-8")
    clone.open("ab").write(b"tampered")
    spec = _memory_run_spec(output, snapshot, metadata.sqlite_file_sha256)
    with pytest.raises(CampaignRunnerError, match="mutated"):
        CampaignRunner.execute_run(spec)


def test_force_rerun_rejects_populated_output_directory(tmp_path):
    output = tmp_path / "populated"
    output.mkdir()
    (output / "scientific_artifact.json").write_text("{}", encoding="utf-8")
    spec = RunSpec(
        run_id="force-populated", experiment_id="isolation", task_id="Li-P-Se",
        elements=["Li", "P", "Se"], condition="adaptive_no_memory", seed=1,
        iteration_seeds=[1], proposal_budget=2, oracle_budget=1,
        geometry_min_distance=.8, thermodynamics_retain_threshold_ev_per_atom=.1,
        thermodynamics_stable_threshold_ev_per_atom=.03, run_mode="development",
        generation_backend="mock", memory_mode="none", memory_seed=1,
        output_dir=str(output),
    )
    with pytest.raises(CampaignRunnerError, match="non-empty output directory"):
        CampaignRunner.execute_run(spec, force_rerun=True)


def test_no_memory_run_rejects_preexisting_run_local_database(tmp_path):
    output = tmp_path / "no_memory"
    output.mkdir()
    (output / "career_memory.db").write_bytes(b"state from another run")
    spec = RunSpec(
        run_id="no-memory-contamination", experiment_id="isolation", task_id="Li-P-Se",
        elements=["Li", "P", "Se"], condition="adaptive_no_memory", seed=1,
        iteration_seeds=[1], proposal_budget=2, oracle_budget=1,
        geometry_min_distance=.8, thermodynamics_retain_threshold_ev_per_atom=.1,
        thermodynamics_stable_threshold_ev_per_atom=.03, run_mode="development",
        generation_backend="mock", memory_mode="none", memory_seed=1,
        output_dir=str(output),
    )
    with pytest.raises(CampaignRunnerError, match="pre-existing CareerMemory"):
        CampaignRunner.execute_run(spec)


def test_restart_reuses_valid_content_addressed_snapshot(tmp_path):
    source = tmp_path / "source.db"
    _source_db(source)
    base = tmp_path / "source_Li-P-S_seed1.db"
    metadata = MemorySnapshotManager.create_snapshot(
        source, base, "Li-P-S", 1, content_addressed=True,
    )
    found = _find_reusable_snapshot({
        "snapshot_path": str(base), "source_task": "Li-P-S", "seed": 1,
    })
    assert found is not None
    assert found["sha256"] == metadata.sqlite_file_sha256
    assert Path(found["path"]).name.endswith(f"_{metadata.sqlite_file_sha256}.db")


def test_completed_run_with_spec_or_artifact_mismatch_is_not_skippable(tmp_path):
    output = tmp_path / "mismatch"
    output.mkdir()
    spec = RunSpec(
        run_id="mismatch", experiment_id="isolation", task_id="Li-P-Se",
        elements=["Li", "P", "Se"], condition="adaptive_no_memory", seed=1,
        iteration_seeds=[1], proposal_budget=2, oracle_budget=1,
        geometry_min_distance=.8, thermodynamics_retain_threshold_ev_per_atom=.1,
        thermodynamics_stable_threshold_ev_per_atom=.03, run_mode="development",
        generation_backend="mock", memory_mode="none", memory_seed=1,
        output_dir=str(output),
    )
    (output / "run_spec.json").write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    (output / "campaign_provenance.json").write_text("{}", encoding="utf-8")
    from agents.provenance import RunManifest
    manifest = RunManifest(campaign_name=spec.run_id, master_seed=spec.seed,
                           iteration_seeds=list(spec.iteration_seeds), status="completed")
    manifest.manifest_hash = manifest.compute_manifest_hash()
    (output / "manifest.json").write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    (output / "run_integrity.json").write_text(json.dumps({
        "schema_version": 1, "run_id": spec.run_id, "spec_hash": spec.spec_hash,
        "manifest_sha256": _sha(output / "manifest.json"),
        "artifacts": {"manifest.json": _sha(output / "manifest.json"),
                       "campaign_provenance.json": "0" * 64},
    }), encoding="utf-8")
    assert CampaignRunner.is_run_completed(output, spec) is False


def test_career_memory_rejects_implicit_home_database():
    with pytest.raises(ValueError, match="explicit run-local"):
        CareerMemory("~/.matagent_career.db")
