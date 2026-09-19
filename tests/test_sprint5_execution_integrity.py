"""Adversarial execution-boundary tests for the Sprint 5 repair.

These tests deliberately exercise the fail-closed paths; they do not require
MatterGen, QE, CHGNet, or network access.
"""

import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess

import pytest

from experiments.cli import _find_reusable_snapshot, run_preflight_check
from agents.career_memory import CareerMemory
from experiments.dag import DAGNode, DAGValidationError, ExperimentDAG, NodeType
from experiments.memory_snapshots import MemorySnapshotError, MemorySnapshotManager
from experiments.runner import CampaignRunner, CampaignRunnerError
from experiments.spec import (
    FIVE_CONDITIONS,
    ExperimentSpec,
    ExperimentSpecError,
    RunSpec,
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
    random = RunSpec(condition="random_mattergen", memory_mode="none", **base)
    adaptive = RunSpec(condition="adaptive_no_memory", memory_mode="none", **base)
    assert random.condition != adaptive.condition
    assert random.memory_mode == adaptive.memory_mode == "none"


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
