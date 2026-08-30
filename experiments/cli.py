"""Command-line interface for the Sprint 5 experiment and audit benchmark.

Supports preflight verification, full execution, safe resume, aggregation,
paired statistical analysis, Quantum ESPRESSO audit, and report generation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from experiments.dag import ExperimentDAG, NodeType
from experiments.memory_snapshots import MemorySnapshotManager
from experiments.metrics import compute_run_metrics
from experiments.report import ReportGenerator
from experiments.runner import CampaignRunner
from experiments.spec import (
    BASE_COMMIT_SHA,
    ExperimentSpec,
    FIVE_CONDITIONS,
    RunSpec,
    SPEC_SCHEMA_VERSION,
)
from experiments.statistics import run_statistical_analysis_pipeline
from agents.thermodynamics import load_frozen_reference_set

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("experiments.cli")


def _get_run_id(value: Any) -> Any:
    return value.get("run_id") if isinstance(value, dict) else getattr(value, "run_id", None)


def _get_condition(value: Any) -> Any:
    return value.get("condition") if isinstance(value, dict) else getattr(value, "condition", None)


def run_preflight_check(spec: ExperimentSpec) -> Dict[str, Any]:
    """Execute fail-closed checks on all behavior-affecting research inputs.

    Development specs intentionally retain the offline mock path, but every
    research assertion is still evaluated and must pass before the DAG can
    execute.  The current checkout is inspected rather than trusting a
    hard-coded commit embedded in an old handoff.
    """
    logger.info("Executing preflight integrity checks...")
    output_root = Path(spec.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    preflight_path = output_root / "preflight.json"

    def _git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "-c", "safe.directory=*", *args],
                cwd=Path(__file__).resolve().parents[1],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    repo_root = Path(__file__).resolve().parents[1]
    current_commit = _git("rev-parse", "HEAD")
    status_lines = [x for x in _git("status", "--porcelain=v1").splitlines() if x.strip()]
    output_abs = output_root.resolve()

    def _is_output_path(line: str) -> bool:
        # Porcelain v1 paths can be quoted; source trees used here are normal
        # ASCII paths, so the final field is sufficient and conservative.
        raw = line[3:].strip().strip('"')
        if " -> " in raw:
            raw = raw.split(" -> ", 1)[1]
        candidate = (repo_root / raw).resolve()
        try:
            candidate.relative_to(output_abs)
            return True
        except ValueError:
            return False

    research_dirty = [line for line in status_lines if not _is_output_path(line)]

    def _digest(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            return None
        h = hashlib.sha256()
        if p.is_file():
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
        elif p.is_dir():
            # Directory checkpoints are hashed over sorted relative names and
            # bytes, avoiding dependence on filesystem enumeration order.
            for child in sorted(x for x in p.rglob("*") if x.is_file()):
                h.update(str(child.relative_to(p)).replace("\\", "/").encode("utf-8"))
                h.update(b"\0")
                with child.open("rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
        return h.hexdigest()

    tasks = [spec.source_task, *spec.target_tasks]
    references_ok = True
    reference_details: Dict[str, Any] = {}
    for task in tasks:
        if not task.reference_set_path:
            task_ok = spec.run_mode != "research"
            reference_details[task.task_id] = {"present": False, "ok": task_ok}
            references_ok = references_ok and task_ok
            continue
        actual = _digest(task.reference_set_path)
        task_ok = bool(
            actual
            and task.reference_set_sha256
            and actual.lower() == task.reference_set_sha256.lower()
            and task.reference_set_certified
        )
        certification_error = None
        frozen = None
        if task_ok:
            try:
                frozen = load_frozen_reference_set(Path(task.reference_set_path))
                task_ok = bool(frozen.certification.certified)
                if spec.run_mode == "research":
                    task_ok = task_ok and {
                        "name": frozen.model.name,
                        "version": frozen.model.version,
                        "checkpoint_sha256": frozen.model.checkpoint_sha256,
                    } == (spec.pinned_model_identity or {})
                    task_ok = task_ok and frozen.relaxation_settings.__dict__ == (
                        spec.pinned_relaxation_settings or {}
                    )
            except Exception as exc:
                task_ok = False
                certification_error = str(exc)
        reference_details[task.task_id] = {
            "path": str(Path(task.reference_set_path).resolve()),
            "expected_sha256": task.reference_set_sha256,
            "actual_sha256": actual,
            "certified": bool(task.reference_set_certified),
            "artifact_certified": bool(frozen and frozen.certification.certified),
            "certification_error": certification_error,
            "ok": task_ok,
        }
        references_ok = references_ok and (task_ok if spec.run_mode == "research" else True)

    qe = spec.qe_audit_config
    mattergen_ok = bool(
        spec.mattergen_model_path
        and Path(spec.mattergen_model_path).exists()
        and spec.mattergen_checkpoint_sha256
        and _digest(spec.mattergen_model_path) == spec.mattergen_checkpoint_sha256.lower()
    )
    sssp_ok = bool(
        qe.sssp_manifest_path
        and Path(qe.sssp_manifest_path).exists()
        and qe.sssp_manifest_sha256
        and _digest(qe.sssp_manifest_path) == qe.sssp_manifest_sha256.lower()
    )
    qe_executable_path = shutil.which(qe.qe_executable)
    if qe_executable_path is None and Path(qe.qe_executable).is_file():
        qe_executable_path = str(Path(qe.qe_executable).resolve())
    qe_version_text = ""
    if qe_executable_path:
        try:
            proc = subprocess.run(
                [qe_executable_path, "-h"], capture_output=True, text=True,
                timeout=10, check=False,
            )
            qe_version_text = (proc.stdout or "") + (proc.stderr or "")
        except Exception:
            qe_version_text = ""
    qe_executable_ok = bool(
        qe_executable_path
        and qe.qe_executable_sha256
        and _digest(qe_executable_path) == qe.qe_executable_sha256.lower()
        and qe.qe_executable_version
        and qe.qe_executable_version in qe_version_text
    )
    sampling_ok = bool(
        not spec.mattergen_sampling_config_path
        or (
            Path(spec.mattergen_sampling_config_path).exists()
            and spec.mattergen_sampling_config_sha256
            and _digest(spec.mattergen_sampling_config_path) == spec.mattergen_sampling_config_sha256.lower()
        )
    )
    checks = {
        "schema_version_ok": spec.schema_version == SPEC_SCHEMA_VERSION,
        # Development fixtures historically use the Sprint-4 base constant;
        # only research mode is allowed to proceed when it names the actual
        # checkout commit (and therefore cannot accidentally use a stale pin).
        "base_commit_ok": bool(current_commit) and (
            spec.code_commit == current_commit
            or (spec.run_mode != "research" and spec.code_commit == BASE_COMMIT_SHA)
        ),
        "conditions_valid": len(spec.conditions) == 5 and len(set(spec.conditions)) == 5 and set(spec.conditions) == set(FIVE_CONDITIONS),
        "master_seeds_valid": len(spec.master_seeds) >= 1 and len(set(spec.master_seeds)) == len(spec.master_seeds),
        "budgets_valid": spec.proposals_per_run > 0 and 0 < spec.oracle_budget_per_run <= spec.proposals_per_run,
        "reference_sets_ok": references_ok,
        "research_tree_clean": not research_dirty,
        "pinned_chgnet_ok": bool(
            spec.pinned_model_identity
            and str(spec.pinned_model_identity.get("name", "")).lower() == "chgnet"
            and spec.pinned_model_identity.get("version")
            and spec.pinned_relaxation_settings
        ),
        "mattergen_checkpoint_ok": mattergen_ok,
        "sssp_manifest_ok": sssp_ok,
        "qe_executable_ok": qe_executable_ok,
        "mattergen_sampling_config_ok": sampling_ok,
        "no_research_mocks": not (
            spec.qe_audit_config.mock_execution
            or str(spec.validation_calculator).lower() in {"mock", "fake", "stub"}
            or str(spec.synthesis_mode).lower() in {"mock", "fake", "stub"}
        ),
        "research_backend_ok": spec.generation_backend == "mattergen",
        "environment_ok": bool(sys.version and platform.platform()),
    }
    # Development mode is the explicitly supported offline path.  It still
    # checks schema/budgets/current commit, while research-only capabilities
    # are reported for audit but are not required.
    required_checks = (
        ["schema_version_ok", "base_commit_ok", "conditions_valid", "master_seeds_valid", "budgets_valid"]
        if spec.run_mode != "research"
        else list(checks)
    )

    all_passed = all(checks[name] for name in required_checks)
    preflight_result = {
        "status": "PASSED" if all_passed else "FAILED",
        "timestamp": time.time(),
        "spec_hash": spec.spec_hash,
        "run_mode": spec.run_mode,
        "current_commit": current_commit,
        "dirty_entries": status_lines,
        "research_dirty_entries": research_dirty,
        "reference_details": reference_details,
        "environment": {"python": sys.version, "platform": platform.platform(), "executable": sys.executable},
        "checks": checks,
        "required_checks": required_checks,
    }

    _atomic_write_json(preflight_path, preflight_result)
    if not all_passed:
        logger.error(f"Preflight check failed: {checks}")
        raise RuntimeError(f"Preflight validation failed: {checks}")

    logger.info("Preflight checks PASSED.")
    return preflight_result


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON with replace-once semantics, never a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
    try:
        with tmp.open("rb") as handle:
            os.fsync(handle.fileno())
    except OSError:
        # Some Windows runtimes expose read-only handles without fsync; the
        # atomic replace still prevents readers from observing partial JSON.
        pass
    os.replace(tmp, path)


def _structure_payload(structure: Any) -> Dict[str, Any]:
    """Convert a persisted/reference structure into the strict QE schema."""
    value = structure
    if isinstance(value, dict) and value.get("format") in {"pymatgen", "dict"}:
        value = value.get("data")
    try:
        from pymatgen.core import Structure
        if isinstance(value, dict) and value.get("@class") == "Structure":
            value = Structure.from_dict(value)
        if isinstance(value, Structure):
            return {
                "lattice": value.lattice.matrix.tolist(),
                "positions": value.frac_coords.tolist(),
                "species": [str(site.specie) for site in value],
                "fractional_coordinates": True,
            }
    except ImportError as exc:
        raise RuntimeError("pymatgen is required to prepare structures for QE") from exc
    if isinstance(value, dict):
        lattice = value.get("lattice") or value.get("cell")
        positions = value.get("positions") or value.get("fractional_coordinates") or value.get("coordinates")
        species = value.get("species") or value.get("symbols") or value.get("elements")
        if lattice and positions and species:
            return {
                "lattice": lattice,
                "positions": positions,
                "species": species,
                "fractional_coordinates": "fractional_coordinates" in value,
            }
    raise RuntimeError("Structure cannot be converted to the QE lattice/positions/species schema")


def _load_candidate_structure(candidate: Any, run_dir: Path) -> None:
    """Resolve and checksum the campaign CIF without fabricating geometry."""
    if candidate.structure is not None or not candidate.structure_path:
        return
    path = Path(candidate.structure_path)
    if not path.is_absolute():
        path = run_dir / path
    if not path.exists() or not path.is_file():
        candidate.provenance_missing_fields.append("structure_file_missing")
        return
    if candidate.structure_hash and _file_sha256(path) != candidate.structure_hash:
        candidate.provenance_missing_fields.append("structure_hash_mismatch")
        return
    from pymatgen.core import Structure
    candidate.structure = _structure_payload(Structure.from_file(path))


def _attach_reference_phase_structures(candidate: Any, reference_set_path: Optional[str]) -> None:
    """Resolve decomposition source IDs against the frozen reference artifact."""
    if not reference_set_path or not candidate.decomposition_products:
        return
    frozen = load_frozen_reference_set(Path(reference_set_path))
    phases = {
        phase.source_id: phase
        for phase in frozen.phases
        if phase.success and (phase.relaxed_structure is not None or phase.structure is not None)
    }
    enriched = []
    for product in candidate.decomposition_products:
        item = dict(product)
        source_id = str(item.get("source_id") or item.get("formula") or item.get("composition") or "")
        phase = phases.get(source_id)
        if phase is None:
            candidate.provenance_missing_fields.append(f"decomposition_phase_missing:{source_id}")
            enriched.append(item)
            continue
        item.update({
            "source_id": source_id,
            "formula": phase.composition,
            "structure": _structure_payload(phase.relaxed_structure or phase.structure),
        })
        enriched.append(item)
    candidate.decomposition_products = enriched


def _write_parquet(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write genuine Parquet, serializing nested audit fields canonically."""
    import pandas as pd
    flattened = []
    for row in rows:
        flattened.append({
            key: json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
            if isinstance(value, (dict, list, tuple, set)) else value
            for key, value in row.items()
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    pd.DataFrame(flattened).to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _find_reusable_snapshot(
    payload: Dict[str, Any],
    expected_transfer_declarations: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Find one valid immutable content-addressed snapshot for a DAG node.

    A restarted pipeline constructs a fresh in-memory DAG, so its placeholder
    path is the unhashed base name.  Previously published hashed siblings are
    the only artifacts eligible for reuse.  Any matching but invalid sibling
    is an integrity failure, never a reason to create a second snapshot.
    """
    base = Path(payload["snapshot_path"])
    candidates: List[Path] = []
    content_addressed_name = re.compile(
        rf"^{re.escape(base.stem)}_[0-9a-fA-F]{{64}}{re.escape(base.suffix)}$"
    )
    if base.exists():
        # The unhashed placeholder is never a resumable snapshot.  Silently
        # treating it as the source for a new content-addressed copy could
        # leave two competing evidence artifacts after a restart.
        if not content_addressed_name.fullmatch(base.name):
            raise RuntimeError(
                f"Unaddressed source snapshot exists and cannot be reused: {base}"
            )
        candidates.append(base)
    if base.parent.exists():
        candidates.extend(sorted(
            path for path in base.parent.glob(f"{base.stem}_*{base.suffix}")
            if content_addressed_name.fullmatch(path.name)
        ))
    # Avoid duplicate entries when the base itself happens to match the
    # content-addressed pattern.
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        return None

    valid: List[Dict[str, Any]] = []
    for candidate in candidates:
        try:
            MemorySnapshotManager.verify_snapshot_integrity(
                candidate,
                expected_source_task=payload.get("source_task"),
                expected_master_seed=payload.get("seed"),
            )
            declarations = expected_transfer_declarations
            if declarations is None:
                declarations = payload.get("allowed_transfer_declarations")
            for declaration in declarations or []:
                MemorySnapshotManager.verify_snapshot_integrity(
                    candidate,
                    expected_source_task=payload.get("source_task"),
                    expected_master_seed=payload.get("seed"),
                    expected_transfer_declaration=declaration,
                )
            sidecar = candidate.with_suffix(".json")
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            valid.append({
                "path": str(candidate.resolve()).replace("\\", "/"),
                "sha256": _file_sha256(candidate),
                "metadata": metadata,
            })
        except Exception as exc:
            raise RuntimeError(
                f"Existing source snapshot candidate is invalid and cannot be replaced: {candidate}: {exc}"
            ) from exc
    if len(valid) > 1:
        paths = ", ".join(str(item["path"]) for item in valid)
        raise RuntimeError(f"Multiple valid snapshots match source node {payload['source_task']} seed {payload['seed']}: {paths}")
    return valid[0]


def execute_full_experiment_pipeline(
    spec: ExperimentSpec,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    """Execute the entire benchmark DAG end-to-end."""
    logger.info(f"Starting Experiment '{spec.experiment_id}'...")
    # Keep CLI preflight usable in minimal/offline checkouts where optional QE
    # modules are not installed; execution imports them only at their DAG node.
    from experiments.qe_audit import QEAuditRunner, select_audit_candidates
    output_root = Path(spec.output_root)
    if force_rerun and output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(
            "force_rerun requires a fresh, empty experiment output directory; "
            "existing run databases/artifacts will never be reused"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    # 1. Preflight
    preflight = run_preflight_check(spec)

    # 2. Build DAG
    dag = ExperimentDAG(spec)
    logger.info(f"Experiment DAG constructed with {len(dag.nodes)} nodes and {dag.total_run_count} runs.")

    # Write DAG manifest
    dag_path = output_root / "experiment_dag.json"
    _atomic_write_json(dag_path, dag.to_dict())

    # 3. Execute nodes in topological order
    nodes = dag.topological_order()
    all_runs_metrics: List[Any] = []
    all_candidates: List[Any] = []
    expected_run_ids = [
        f"run_{condition}_{task.task_id}_seed{seed}"
        for task in spec.target_tasks
        for condition in spec.conditions
        for seed in spec.master_seeds
    ]
    run_set_metadata: Dict[str, Any] = {
        "expected_target_task_ids": [task.task_id for task in spec.target_tasks],
        "expected_target_tasks": [task.task_id for task in spec.target_tasks],
        "expected_source_task": spec.source_task.task_id,
        "expected_seeds": list(spec.master_seeds),
        "expected_target_seeds": list(spec.master_seeds),
        "expected_conditions": list(spec.conditions),
        "expected_target_run_ids": expected_run_ids,
        "expected_target_run_count": len(expected_run_ids),
        "expected_source_run_count": len(spec.master_seeds),
        "proposal_budget_per_run": spec.proposals_per_run,
        "oracle_budget_per_run": spec.oracle_budget_per_run,
        "expected_proposal_budget": spec.proposals_per_run,
        "expected_oracle_budget": spec.oracle_budget_per_run,
    }
    qe_selection_metadata: Dict[str, Any] = {
        "expected_qe_candidate_count": spec.qe_audit_config.candidate_count,
        "qe_selection_count": 0,
        "qe_selection_insufficiency": None,
    }

    def _persist_dag() -> None:
        _atomic_write_json(dag_path, dag.to_dict())

    for node in nodes:
        logger.info(f"Executing DAG Node: [{node.node_type.value}] {node.node_id}...")
        # A node can run only after every parent has completed successfully.
        dag.assert_dependencies_succeeded(node.node_id)
        try:
            if node.node_type in (NodeType.SOURCE_MEMORY_RUN, NodeType.TARGET_CAMPAIGN_RUN):
                payload = node.payload
                task = next(t for t in [spec.source_task, *spec.target_tasks] if t.task_id == payload["task_id"])
                snapshot_sha = payload.get("source_memory_snapshot_sha256")
                if payload.get("source_memory_snapshot_path") and not snapshot_sha:
                    for dep_id in node.dependencies:
                        dep = dag.nodes.get(dep_id)
                        if dep and dep.node_type == NodeType.SOURCE_MEMORY_SNAPSHOT:
                            snapshot_sha = dep.payload.get("snapshot_sha256")
                            break
                run_spec = RunSpec(
                    run_id=payload["run_id"], experiment_id=spec.experiment_id,
                    task_id=payload["task_id"], elements=payload["elements"],
                    condition=payload.get("condition", "adaptive_no_memory"), seed=payload["seed"],
                    iteration_seeds=[payload["seed"] + i for i in range(spec.iterations_per_run)],
                    proposal_budget=payload["proposal_budget"], oracle_budget=payload["oracle_budget"],
                    geometry_min_distance=spec.geometry_min_distance,
                    thermodynamics_retain_threshold_ev_per_atom=spec.thermodynamics_retain_threshold_ev_per_atom,
                    thermodynamics_stable_threshold_ev_per_atom=spec.thermodynamics_stable_threshold_ev_per_atom,
                    run_mode=spec.run_mode, generation_backend=spec.generation_backend,
                    memory_mode=payload.get("memory_mode", "none"), memory_seed=payload.get("memory_seed", payload["seed"]),
                    output_dir=payload["output_dir"], memory_transfer_declaration=payload.get("memory_transfer_declaration"),
                    reference_set_path=task.reference_set_path, reference_set_sha256=task.reference_set_sha256,
                    reference_set_certified=task.reference_set_certified,
                    source_memory_snapshot_path=payload.get("source_memory_snapshot_path"),
                    source_memory_snapshot_sha256=snapshot_sha,
                    pinned_model_identity=spec.pinned_model_identity,
                    pinned_relaxation_settings=spec.pinned_relaxation_settings,
                    career_db_path=str(Path(payload["output_dir"]) / "career_memory.db"),
                    mattergen_pretrained=spec.mattergen_pretrained,
                    mattergen_model_path=spec.mattergen_model_path,
                    mattergen_checkpoint_sha256=spec.mattergen_checkpoint_sha256,
                    mattergen_sampling_config_path=spec.mattergen_sampling_config_path,
                    mattergen_sampling_config_sha256=spec.mattergen_sampling_config_sha256,
                    mattergen_batch_size=spec.mattergen_batch_size,
                    domain=task.domain,
                    target_properties=dict(task.target_properties),
                    task_constraints=dict(task.constraints),
                    validation_calculator=spec.validation_calculator,
                    synthesis_mode=spec.synthesis_mode,
                )
                run_res = CampaignRunner.execute_run(run_spec, force_rerun=force_rerun)
                # Metrics are derived from the campaign's report, never from a
                # mutable global CareerMemory query or a synthetic placeholder.
                report_p = Path(run_spec.output_dir) / "report.json"
                provenance_p = Path(run_spec.output_dir) / "campaign_provenance.json"
                if provenance_p.exists():
                    m_data = json.loads(provenance_p.read_text(encoding="utf-8"))
                    # Preserve run counters/status from the report as
                    # metadata while keeping candidate order authoritative.
                    if report_p.exists():
                        m_data = {**json.loads(report_p.read_text(encoding="utf-8")), **m_data}
                    rm, cands = compute_run_metrics(
                        manifest_data=m_data, run_id=run_spec.run_id, task_id=run_spec.task_id,
                        condition=run_spec.condition, seed=run_spec.seed, oracle_budget=run_spec.oracle_budget,
                    )
                    for candidate in cands:
                        _load_candidate_structure(candidate, Path(run_spec.output_dir))
                        _attach_reference_phase_structures(candidate, run_spec.reference_set_path)
                    all_runs_metrics.append(rm); all_candidates.extend(cands)

            elif node.node_type == NodeType.REFERENCE_SET_VERIFICATION:
                payload = node.payload
                p = payload.get("reference_set_path")
                expected = payload.get("reference_set_sha256")
                if spec.run_mode == "research":
                    if not p or not expected or not Path(p).exists():
                        raise RuntimeError(f"Reference set missing for {payload['task_id']}")
                    actual = _file_sha256(Path(p))
                    if actual != expected.lower():
                        raise RuntimeError(f"Reference set hash mismatch for {payload['task_id']}")
                out = output_root / "references" / f"{payload['task_id']}.verified.json"
                _atomic_write_json(out, {**payload, "verified": True, "sha256": _file_sha256(Path(p)) if p and Path(p).is_file() else None})
                node.expected_output_path = str(out).replace("\\", "/")

            elif node.node_type == NodeType.SOURCE_MEMORY_SNAPSHOT:
                payload = node.payload
                src_run_dir = Path(output_root / "runs" / "source" / payload["source_task"] / str(payload["seed"]))
                src_db = src_run_dir / "career_memory.db"
                dest_snap = Path(payload["snapshot_path"])
                reusable = _find_reusable_snapshot(payload)
                if reusable is None:
                    meta = MemorySnapshotManager.create_snapshot(
                        source_db_path=src_db, destination_snapshot_path=dest_snap,
                        source_task=payload["source_task"], master_seed=payload["seed"],
                        allowed_transfer_declarations=[
                            td.to_dict() if hasattr(td, "to_dict") else td.__dict__
                            for td in spec.transfer_declarations
                        ],
                        content_addressed=True,
                    )
                    snapshot_sha256 = meta.sqlite_file_sha256
                    actual_snapshot_path = str(meta.snapshot_path)
                else:
                    snapshot_sha256 = str(reusable["sha256"])
                    actual_snapshot_path = str(reusable["path"])
                node.payload["snapshot_sha256"] = snapshot_sha256
                node.payload["snapshot_path"] = actual_snapshot_path
                node.expected_output_path = actual_snapshot_path
                for dependent in dag.nodes.values():
                    if node.node_id in dependent.dependencies:
                        dependent.payload["source_memory_snapshot_path"] = actual_snapshot_path
                        dependent.payload["source_memory_snapshot_sha256"] = snapshot_sha256

            elif node.node_type == NodeType.AGGREGATION:
                agg_dir = output_root / "aggregates"; agg_dir.mkdir(parents=True, exist_ok=True)
                run_data = [rm.to_dict() for rm in all_runs_metrics]
                cand_data = [c.to_dict() for c in all_candidates]
                _atomic_write_json(agg_dir / "runs.json", run_data)
                _atomic_write_json(agg_dir / "candidates.json", cand_data)
                # The aggregate contract is content-addressed.  JSON is used
                # as a deterministic fallback when pyarrow is unavailable.
                _write_parquet(agg_dir / "runs.parquet", run_data)
                _write_parquet(agg_dir / "candidates.parquet", cand_data)

            elif node.node_type == NodeType.STATISTICAL_ANALYSIS:
                stats_dir = output_root / "statistics"
                stats_kwargs = {
                    "run_metrics_list": all_runs_metrics,
                    "analysis_version": spec.analysis_version,
                    "output_dir": stats_dir,
                    "expected_seeds": spec.master_seeds,
                }
                stats_kwargs["expected_tasks"] = [task.task_id for task in spec.target_tasks]
                run_statistical_analysis_pipeline(**stats_kwargs)

            elif node.node_type == NodeType.QE_AUDIT_SELECTION:
                qe_dir = output_root / "qe_audit"; qe_dir.mkdir(parents=True, exist_ok=True)
                selected_cands = select_audit_candidates(candidates=all_candidates, target_count=spec.qe_audit_config.candidate_count)
                qe_selection_metadata = {
                    "expected_qe_candidate_count": spec.qe_audit_config.candidate_count,
                    "qe_selection_count": len(selected_cands),
                    "qe_selection_insufficiency": getattr(selected_cands, "insufficiency", None),
                }
                _atomic_write_json(qe_dir / "selection.json", {
                    "candidates": [c.to_dict() for c in selected_cands],
                    "insufficiency": getattr(selected_cands, "insufficiency", None),
                })

            elif node.node_type == NodeType.QE_AUDIT_EXECUTION:
                qe_dir = output_root / "qe_audit"; selected_path = qe_dir / "selection.json"
                sel_data = json.loads(selected_path.read_text(encoding="utf-8"))
                from experiments.qe_audit import QEAuditCandidate
                selected_rows = sel_data.get("candidates", []) if isinstance(sel_data, dict) else sel_data
                QEAuditRunner(
                    config=spec.qe_audit_config,
                    output_dir=qe_dir,
                    run_mode=spec.run_mode,
                    qe_executable=spec.qe_audit_config.qe_executable,
                ).run_full_audit(
                    [QEAuditCandidate(**d) for d in selected_rows]
                )

            elif node.node_type == NodeType.REPORT_GENERATION:
                rep = ReportGenerator(experiment_root=output_root); stats_dir = output_root / "statistics"; qe_dir = output_root / "qe_audit"
                stats_results_list = json.loads((stats_dir / "analysis_manifest.json").read_text()).get("results", []) if (stats_dir / "analysis_manifest.json").exists() else []
                qe_results_list = list(csv.DictReader((qe_dir / "results.csv").open("r", encoding="utf-8"))) if (qe_dir / "results.csv").exists() else []
                qe_manifest: Dict[str, Any] = {}
                qe_manifest_path = qe_dir / "audit_manifest.json"
                if qe_manifest_path.exists():
                    try:
                        loaded_qe_manifest = json.loads(qe_manifest_path.read_text(encoding="utf-8"))
                        if isinstance(loaded_qe_manifest, dict):
                            qe_manifest = loaded_qe_manifest
                    except (OSError, ValueError):
                        # Report generation remains fail-closed when the QE
                        # manifest is absent or malformed.
                        qe_manifest = {}
                selection_data: Dict[str, Any] = {}
                selection_path = qe_dir / "selection.json"
                if selection_path.exists():
                    try:
                        loaded_selection = json.loads(selection_path.read_text(encoding="utf-8"))
                        if isinstance(loaded_selection, dict):
                            selection_data = loaded_selection
                    except (OSError, ValueError):
                        selection_data = {}
                qe_context = {
                    **qe_selection_metadata,
                    "qe_audit": qe_manifest,
                    "selection": selection_data,
                }
                required_names = preflight.get("required_checks", []) if isinstance(preflight, dict) else []
                required_results = {
                    str(name): preflight.get("checks", {}).get(name)
                    for name in required_names
                    if isinstance(preflight, dict) and isinstance(preflight.get("checks"), dict)
                }
                rep.generate_all_reports(
                    runs_metrics=all_runs_metrics,
                    statistical_results=stats_results_list,
                    qe_results=qe_results_list,
                    validated_artifacts={
                        "preflight": preflight,
                        "required_preflight_checks": required_names,
                        "required_preflight_results": required_results,
                        "expected_target_task_ids": [task.task_id for task in spec.target_tasks],
                        "expected_seeds": list(spec.master_seeds),
                        "expected_conditions": list(spec.conditions),
                        "expected_target_run_ids": expected_run_ids,
                        "expected_target_run_count": len(expected_run_ids),
                        "expected_proposal_budget": spec.proposals_per_run,
                        "expected_oracle_budget": spec.oracle_budget_per_run,
                        **qe_context,
                        "run_set": {
                            **run_set_metadata,
                            "actual_run_ids": sorted(str(_get_run_id(rm)) for rm in all_runs_metrics),
                            "actual_target_run_ids": sorted(
                                str(_get_run_id(rm)) for rm in all_runs_metrics
                                if str(_get_condition(rm)) in set(spec.conditions)
                            ),
                            "actual_target_run_count": sum(
                                1 for rm in all_runs_metrics if str(_get_condition(rm)) in set(spec.conditions)
                            ),
                        },
                        "dag": dag.to_dict(),
                    },
                )

            node.executed = True
            node.success = True
            node.result_hash = dag.hash_node_output(node)
            _persist_dag()
        except Exception:
            node.executed = True; node.success = False
            _persist_dag()
            raise

    logger.info("Experiment pipeline finished successfully.")
    return {
        "status": "SUCCESS",
        "experiment_id": spec.experiment_id,
        "runs_completed": len(all_runs_metrics),
        "candidates_evaluated": len(all_candidates),
    }


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Sprint 5 Experiment and Audit Benchmark CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Preflight
    p_preflight = subparsers.add_parser("preflight", help="Run preflight validation")
    p_preflight.add_argument("--spec-file", type=str, help="Path to ExperimentSpec JSON file")

    # Run
    p_run = subparsers.add_parser("run", help="Run full experiment pipeline")
    p_run.add_argument("--spec-file", type=str, help="Path to ExperimentSpec JSON file")
    p_run.add_argument("--force-rerun", action="store_true", help="Force rerun of completed runs")

    # Dry-run
    p_dry = subparsers.add_parser("dry-run", help="Run tiny injected-backend experiment offline")
    p_dry.add_argument("--output-dir", type=str, default="./dry_run_results")

    args = parser.parse_args()

    if args.command == "dry-run":
        spec = ExperimentSpec(
            experiment_id="dry_run_test",
            master_seeds=[42, 137],
            proposals_per_run=10,
            oracle_budget_per_run=5,
            iterations_per_run=2,
            output_root=args.output_dir,
        )
        res = execute_full_experiment_pipeline(spec, force_rerun=True)
        print(json.dumps(res, indent=2))
    elif args.command in {"preflight", "run"}:
        if not args.spec_file:
            parser.error(f"{args.command} requires --spec-file")
        spec_data = json.loads(Path(args.spec_file).read_text(encoding="utf-8"))
        spec = ExperimentSpec.from_dict(spec_data)
        if args.command == "preflight":
            print(json.dumps(run_preflight_check(spec), indent=2))
        else:
            print(json.dumps(execute_full_experiment_pipeline(spec, force_rerun=args.force_rerun), indent=2))


if __name__ == "__main__":
    main()
