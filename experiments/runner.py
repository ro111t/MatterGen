"""Deterministic, isolated execution of one Sprint 5 campaign run."""

from __future__ import annotations

from enum import Enum
import json
import hashlib
import logging
import math
import os
from pathlib import Path
import time
import traceback
from typing import Any, Dict, Mapping, Optional

from agents.integrity import RunMode
from agents.orchestrator import CampaignObjective
from agents.thermodynamics import load_frozen_reference_set
from campaign import CampaignConfig, MaterialsDiscoveryCampaign
from experiments.memory_snapshots import MemorySnapshotManager, compute_file_sha256
from experiments.spec import MEMORY_ARMS, RunSpec, compute_sha256

logger = logging.getLogger(__name__)


class RunTerminalState(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED_PREFLIGHT = "FAILED_PREFLIGHT"
    FAILED_GENERATION = "FAILED_GENERATION"
    FAILED_ORACLE = "FAILED_ORACLE"
    INTERRUPTED = "INTERRUPTED"
    FAILED_GENERIC = "FAILED_GENERIC"


class CampaignRunnerError(RuntimeError):
    """Raised when a run cannot be safely started, resumed, or finalized."""

    def __init__(self, message: str, state: RunTerminalState = RunTerminalState.FAILED_GENERIC):
        super().__init__(message)
        self.state = state


class CampaignRunner:
    """Execute one fully resolved :class:`RunSpec`."""

    @staticmethod
    def _atomic_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
        try:
            with tmp.open("rb") as handle:
                os.fsync(handle.fileno())
        except OSError:
            pass
        os.replace(tmp, path)

    @staticmethod
    def _canonical_manifest_hash(data: Mapping[str, Any]) -> Optional[str]:
        embedded = data.get("manifest_hash")
        if not embedded:
            return None
        try:
            from agents.provenance import RunManifest
            computed = RunManifest.from_dict(dict(data)).compute_manifest_hash()
            return computed if computed == embedded else None
        except Exception:
            return None

    @staticmethod
    def _hash_path(path: Path) -> str:
        h = hashlib.sha256()
        if path.is_file():
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
        elif path.is_dir():
            for child in sorted(x for x in path.rglob("*") if x.is_file()):
                h.update(str(child.relative_to(path)).replace("\\", "/").encode("utf-8")); h.update(b"\0")
                with child.open("rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
        else:
            raise FileNotFoundError(path)
        return h.hexdigest()

    @classmethod
    def is_run_completed(cls, output_dir: Path, spec: Optional[RunSpec] = None) -> bool:
        """Verify completion only when spec, provenance, and artifact hashes agree."""
        output_dir = Path(output_dir)
        manifest_path = output_dir / "manifest.json"
        integrity_path = output_dir / "run_integrity.json"
        spec_path = output_dir / "run_spec.json"
        if not manifest_path.exists() or not integrity_path.exists() or not spec_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
            saved_spec = json.loads(spec_path.read_text(encoding="utf-8"))
            saved_hash = compute_sha256(saved_spec)
            if spec is not None and saved_hash != spec.spec_hash:
                return False
            if integrity.get("schema_version") != 1 or integrity.get("run_id") != saved_spec.get("run_id"):
                return False
            if integrity.get("spec_hash") != saved_hash:
                return False
            if manifest.get("status") != "completed" or not cls._canonical_manifest_hash(manifest):
                return False
            if spec is not None:
                # The manifest is part of the run identity.  A valid hash on a
                # manifest from a different run must not make this run
                # resumable merely because its artifact filenames happen to
                # match.
                if manifest.get("campaign_name") != spec.run_id:
                    return False
                if manifest.get("master_seed") != spec.seed:
                    return False
                if list(manifest.get("iteration_seeds") or []) != list(spec.iteration_seeds):
                    return False
            if integrity.get("manifest_sha256") != compute_file_sha256(manifest_path):
                return False
            artifacts = integrity.get("artifacts")
            if not isinstance(artifacts, dict) or {
                "manifest.json", "campaign_provenance.json",
            } - set(artifacts):
                return False
            for relative, expected in artifacts.items():
                # Artifact names are persisted data.  Do not let a forged
                # sidecar authorize hashing a path outside this run directory.
                artifact = Path(relative)
                if artifact.is_absolute():
                    return False
                artifact = output_dir / artifact
                try:
                    artifact.resolve().relative_to(output_dir.resolve())
                except ValueError:
                    return False
                if not artifact.exists() or compute_file_sha256(artifact) != expected:
                    return False
            try:
                campaign_provenance = json.loads(
                    (output_dir / "campaign_provenance.json").read_text(encoding="utf-8")
                )
                provenance_manifest = campaign_provenance.get("manifest")
                if not isinstance(provenance_manifest, dict):
                    return False
                if provenance_manifest.get("manifest_hash") != manifest.get("manifest_hash"):
                    return False
                if campaign_provenance.get("campaign_id") != manifest.get("campaign_id"):
                    return False
                if campaign_provenance.get("campaign_name") != manifest.get("campaign_name"):
                    return False
            except Exception:
                return False
            if spec is not None and spec.condition in MEMORY_ARMS:
                # Completion is not resumable evidence if the private clone
                # was changed after finalization.  Verify both the immutable
                # source snapshot and the target clone before allowing a skip.
                memory_db_path = cls._private_memory_path(spec, output_dir)
                cls._verify_memory_clone(spec, output_dir, memory_db_path)
            if spec is not None and spec.condition == "source_neutral":
                # Source-neutral runs intentionally create the source database;
                # its final bytes are still an artifact and must be present in
                # the completion contract before a restart can skip it.
                memory_db_path = cls._private_memory_path(spec, output_dir)
                if "career_memory.db" not in artifacts or not memory_db_path.exists():
                    return False
                if compute_file_sha256(memory_db_path) != artifacts["career_memory.db"]:
                    return False
            return True
        except Exception:
            return False

    @staticmethod
    def _existing_entries(output_dir: Path) -> list[Path]:
        """Return persisted run state, excluding no files by convention."""
        try:
            return sorted(output_dir.iterdir(), key=lambda path: path.name)
        except FileNotFoundError:
            return []

    @staticmethod
    def _verify_memory_clone(spec: RunSpec, output_dir: Path, memory_db_path: Path) -> None:
        """Verify source/clone hashes and transfer authorization before use."""
        if not spec.source_memory_snapshot_path:
            raise CampaignRunnerError(
                "Memory condition requires a source snapshot",
                RunTerminalState.FAILED_PREFLIGHT,
            )
        snapshot_path = Path(spec.source_memory_snapshot_path)
        MemorySnapshotManager.verify_snapshot_integrity(
            snapshot_path,
            expected_sqlite_sha256=spec.source_memory_snapshot_sha256,
            expected_transfer_declaration=spec.memory_transfer_declaration,
        )
        clone_marker = output_dir / "memory_clone_integrity.json"
        if not memory_db_path.exists() or memory_db_path.stat().st_size == 0:
            raise CampaignRunnerError(
                "Existing memory run is missing its private CareerMemory clone",
                RunTerminalState.FAILED_PREFLIGHT,
            )
        if not clone_marker.exists():
            raise CampaignRunnerError(
                "Existing memory database lacks clone integrity metadata",
                RunTerminalState.FAILED_PREFLIGHT,
            )
        try:
            marker = json.loads(clone_marker.read_text(encoding="utf-8"))
        except Exception as exc:
            raise CampaignRunnerError(
                "Memory clone integrity metadata is invalid",
                RunTerminalState.FAILED_PREFLIGHT,
            ) from exc
        if marker.get("run_id") != spec.run_id:
            raise CampaignRunnerError(
                "Memory clone integrity metadata belongs to a different run",
                RunTerminalState.FAILED_PREFLIGHT,
            )
        recorded_snapshot = marker.get("source_snapshot")
        if not recorded_snapshot or Path(recorded_snapshot).resolve() != snapshot_path.resolve():
            raise CampaignRunnerError(
                "Memory clone source snapshot path differs from the run specification",
                RunTerminalState.FAILED_PREFLIGHT,
            )
        source_hash = str(marker.get("source_sha256") or "").lower()
        current_source_hash = compute_file_sha256(snapshot_path).lower()
        if not source_hash or source_hash != current_source_hash:
            raise CampaignRunnerError(
                "Memory source snapshot changed since the clone was recorded",
                RunTerminalState.FAILED_PREFLIGHT,
            )
        if spec.source_memory_snapshot_sha256 and source_hash != str(spec.source_memory_snapshot_sha256).lower():
            raise CampaignRunnerError(
                "Memory clone source hash differs from the run specification",
                RunTerminalState.FAILED_PREFLIGHT,
            )
        expected_clone_hash = str(marker.get("clone_sha256") or "").lower()
        current_clone_hash = compute_file_sha256(memory_db_path).lower()
        if not expected_clone_hash or current_clone_hash != expected_clone_hash:
            raise CampaignRunnerError(
                "Private CareerMemory clone was mutated after it was recorded",
                RunTerminalState.FAILED_PREFLIGHT,
            )
        if "memory_transfer_declaration" not in marker:
            raise CampaignRunnerError(
                "Memory clone integrity metadata lacks transfer authorization",
                RunTerminalState.FAILED_PREFLIGHT,
            )
        marker_transfer = marker.get("memory_transfer_declaration")
        if marker_transfer != spec.memory_transfer_declaration:
            raise CampaignRunnerError(
                "Memory clone transfer declaration differs from the run specification",
                RunTerminalState.FAILED_PREFLIGHT,
            )

    @staticmethod
    def _private_memory_path(spec: RunSpec, output_dir: Path) -> Path:
        """Resolve a CareerMemory path and require it to be run-local.

        A historical home-directory default existed in earlier campaign
        versions.  Accepting such a path here would reintroduce cross-run
        contamination even though the experimental runner appears isolated at
        the output layer.
        """
        path = Path(spec.career_db_path) if spec.career_db_path else output_dir / "career_memory.db"
        try:
            path.resolve().relative_to(output_dir.resolve())
        except ValueError as exc:
            raise CampaignRunnerError(
                "CareerMemory database must be inside the run output directory; "
                "global or shared memory paths are not allowed",
                RunTerminalState.FAILED_PREFLIGHT,
            ) from exc
        return path

    @staticmethod
    def _finalize_memory_clone_marker(output_dir: Path, memory_db_path: Path) -> None:
        """Record the post-campaign clone digest after a successful run.

        The initial digest protects the execution boundary: a partial run is
        never replayed.  Once the campaign has completed, its private clone may
        legitimately contain newly distilled target evidence, so the marker is
        advanced to that finalized byte hash before completion artifacts are
        sealed.
        """
        marker_path = output_dir / "memory_clone_integrity.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["clone_sha256_before_execution"] = marker.get("clone_sha256")
        marker["clone_sha256"] = compute_file_sha256(memory_db_path)
        marker["finalized"] = True
        CampaignRunner._atomic_json(marker_path, marker)

    @staticmethod
    def _classify_failure(exc: BaseException) -> RunTerminalState:
        if isinstance(exc, CampaignRunnerError):
            return exc.state
        message = str(exc).lower()
        if any(x in message for x in ("preflight", "research", "reference", "snapshot", "pinned")):
            return RunTerminalState.FAILED_PREFLIGHT
        if any(x in message for x in ("generation", "mattergen")):
            return RunTerminalState.FAILED_GENERATION
        if any(x in message for x in ("oracle", "thermodynamic")):
            return RunTerminalState.FAILED_ORACLE
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            return RunTerminalState.INTERRUPTED
        return RunTerminalState.FAILED_GENERIC

    @classmethod
    def execute_run(cls, spec: RunSpec, force_rerun: bool = False) -> Dict[str, Any]:
        """Execute a run, or return a verified completion on safe resume."""
        output_dir = Path(spec.output_dir)
        if output_dir.exists() and not output_dir.is_dir():
            raise CampaignRunnerError(
                f"Run output path is not a directory: {output_dir}", RunTerminalState.FAILED_PREFLIGHT
            )
        existing_entries = cls._existing_entries(output_dir) if output_dir.exists() else []
        manifest_path = output_dir / "manifest.json"
        spec_record_path = output_dir / "run_spec.json"
        try:
            if spec.run_mode == "research":
                if not spec.reference_set_path or not Path(spec.reference_set_path).exists() or not spec.reference_set_certified:
                    raise CampaignRunnerError("Research run lacks a certified reference set", RunTerminalState.FAILED_PREFLIGHT)
                if cls._hash_path(Path(spec.reference_set_path)) != str(spec.reference_set_sha256).lower():
                    raise CampaignRunnerError("Research reference set SHA256 mismatch", RunTerminalState.FAILED_PREFLIGHT)
                if not spec.mattergen_model_path or not Path(spec.mattergen_model_path).exists():
                    raise CampaignRunnerError("Research run lacks a local MatterGen checkpoint", RunTerminalState.FAILED_PREFLIGHT)
                if cls._hash_path(Path(spec.mattergen_model_path)) != str(spec.mattergen_checkpoint_sha256).lower():
                    raise CampaignRunnerError("Research MatterGen checkpoint SHA256 mismatch", RunTerminalState.FAILED_PREFLIGHT)
                if spec.mattergen_sampling_config_path:
                    sampling_path = Path(spec.mattergen_sampling_config_path)
                    if (
                        not sampling_path.exists()
                        or not spec.mattergen_sampling_config_sha256
                        or cls._hash_path(sampling_path) != spec.mattergen_sampling_config_sha256.lower()
                    ):
                        raise CampaignRunnerError(
                            "Research MatterGen sampling configuration SHA256 mismatch",
                            RunTerminalState.FAILED_PREFLIGHT,
                        )
                try:
                    frozen = load_frozen_reference_set(Path(spec.reference_set_path))
                except Exception as exc:
                    raise CampaignRunnerError(
                        f"Research reference set failed certification: {exc}",
                        RunTerminalState.FAILED_PREFLIGHT,
                    ) from exc
                if not frozen.certification.certified:
                    raise CampaignRunnerError("Research reference set is not certified", RunTerminalState.FAILED_PREFLIGHT)
                expected_model = spec.pinned_model_identity or {}
                actual_model = {
                    "name": frozen.model.name,
                    "version": frozen.model.version,
                    "checkpoint_sha256": frozen.model.checkpoint_sha256,
                }
                if actual_model != expected_model:
                    raise CampaignRunnerError(
                        "Research reference model identity differs from the pinned candidate evaluator",
                        RunTerminalState.FAILED_PREFLIGHT,
                    )
                if frozen.relaxation_settings.__dict__ != (spec.pinned_relaxation_settings or {}):
                    raise CampaignRunnerError(
                        "Research reference relaxation settings differ from the pinned candidate settings",
                        RunTerminalState.FAILED_PREFLIGHT,
                    )
            if spec_record_path.exists():
                saved = json.loads(spec_record_path.read_text(encoding="utf-8"))
                if compute_sha256(saved) != spec.spec_hash:
                    raise CampaignRunnerError(
                        f"Run '{spec.run_id}' specification mismatch; refusing resume/overwrite",
                        RunTerminalState.FAILED_PREFLIGHT,
                    )
            if not force_rerun and cls.is_run_completed(output_dir, spec):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                return {"run_id": spec.run_id, "status": RunTerminalState.SUCCESS.value,
                        "manifest_path": str(manifest_path).replace("\\", "/"),
                        "skipped": True, "manifest": manifest}

            # No restoration API exists in MaterialsDiscoveryCampaign.  A
            # silent replay from iteration zero would duplicate budgeted work.
            # A forced rerun is deliberately a new run, never an instruction
            # to reuse an existing database or artifact tree.
            if existing_entries:
                # Surface clone tampering before the generic unsupported
                # checkpoint error.  In particular, a maliciously modified
                # target database must never reach campaign construction.
                memory_db_path = cls._private_memory_path(spec, output_dir)
                if spec.condition in MEMORY_ARMS and (
                    memory_db_path.exists() or (output_dir / "memory_clone_integrity.json").exists()
                ):
                    cls._verify_memory_clone(spec, output_dir, memory_db_path)
                # A forced rerun always requires a genuinely fresh output
                # directory.  Even a presentation-only manifest can be stale
                # scientific state and must never be overwritten in place.
                if force_rerun:
                    raise CampaignRunnerError(
                        f"Run '{spec.run_id}' cannot be force-rerun in a non-empty output directory; "
                        "choose a fresh output directory",
                        RunTerminalState.FAILED_PREFLIGHT,
                    )
                if memory_db_path.exists() and spec.condition not in MEMORY_ARMS:
                    raise CampaignRunnerError(
                        "Run has a pre-existing CareerMemory database; contaminated "
                        "source state cannot be reused",
                        RunTerminalState.INTERRUPTED,
                    )
                raise CampaignRunnerError(
                    f"Run '{spec.run_id}' has persisted artifacts but is not a verified completion; "
                    "checkpoint restoration is unsupported",
                    RunTerminalState.INTERRUPTED,
                )

            output_dir.mkdir(parents=True, exist_ok=True)
            cls._atomic_json(spec_record_path, spec.to_dict())

            memory_db_path = cls._private_memory_path(spec, output_dir)
            use_career_memory = spec.condition in MEMORY_ARMS or spec.condition == "source_neutral"
            memory_mode = spec.memory_mode if spec.condition in MEMORY_ARMS else "none"
            if spec.condition == "source_neutral":
                memory_mode = "structured_provenance"
            if memory_db_path.exists():
                raise CampaignRunnerError(
                    "Run has a pre-existing CareerMemory database; contaminated "
                    "source state cannot be reused",
                    RunTerminalState.INTERRUPTED,
                )

            if spec.condition in MEMORY_ARMS:
                if not spec.source_memory_snapshot_path:
                    raise CampaignRunnerError("Memory condition requires a source snapshot", RunTerminalState.FAILED_PREFLIGHT)
                snapshot_path = Path(spec.source_memory_snapshot_path)
                MemorySnapshotManager.verify_snapshot_integrity(
                    snapshot_path,
                    expected_sqlite_sha256=spec.source_memory_snapshot_sha256,
                    expected_transfer_declaration=spec.memory_transfer_declaration,
                )
                clone_marker = output_dir / "memory_clone_integrity.json"
                if not memory_db_path.exists() or memory_db_path.stat().st_size == 0:
                    MemorySnapshotManager.clone_for_target_run(snapshot_path, memory_db_path)
                    cls._atomic_json(clone_marker, {
                        "run_id": spec.run_id,
                        "source_snapshot": str(snapshot_path.resolve()),
                        "source_sha256": compute_file_sha256(snapshot_path),
                        "clone_sha256": compute_file_sha256(memory_db_path),
                        "clone_sha256_before_execution": compute_file_sha256(memory_db_path),
                        "memory_transfer_declaration": spec.memory_transfer_declaration,
                        "finalized": False,
                    })
                else:
                    cls._verify_memory_clone(spec, output_dir, memory_db_path)

            objective = CampaignObjective(
                target_properties=dict(spec.target_properties),
                constraints={
                    **dict(spec.task_constraints),
                    "elements": list(spec.elements),
                    "memory_transfer_declaration": spec.memory_transfer_declaration,
                    "experiment_condition": spec.condition,
                },
                success_criteria={}, domain=spec.domain,
                max_iterations=len(spec.iteration_seeds))
            run_mode = RunMode.RESEARCH if spec.run_mode == "research" else RunMode.DEVELOPMENT
            num_iterations = max(1, len(spec.iteration_seeds))
            target_candidates_per_iter = int(math.ceil(spec.proposal_budget / num_iterations))
            config = CampaignConfig(
                name=spec.run_id, objective=objective, output_dir=output_dir,
                master_seed=spec.seed, career_db_path=str(memory_db_path),
                proposal_budget=spec.proposal_budget, oracle_budget=spec.oracle_budget,
                geometry_min_distance=spec.geometry_min_distance,
                thermodynamics_reference_set_path=spec.reference_set_path,
                thermodynamics_retain_threshold_ev_per_atom=spec.thermodynamics_retain_threshold_ev_per_atom,
                thermodynamics_stable_threshold_ev_per_atom=spec.thermodynamics_stable_threshold_ev_per_atom,
                use_career_memory=use_career_memory, memory_mode=memory_mode,
                memory_seed=spec.memory_seed, run_mode=run_mode, verbose=False,
                use_validation=False, use_synthesis=False,
                require_thermodynamics=spec.reference_set_path is not None,
                thermodynamics_backend="chgnet" if spec.run_mode == "research" else None,
                num_candidates=target_candidates_per_iter,
                # Research specs require MatterGen. Development/mock specs use
                # the deterministic injected generator for every arm, while
                # the random baseline remains distinct through its fixed,
                # memory-free planner below.
                use_mattergen=(spec.generation_backend == "mattergen"),
                mattergen_pretrained=spec.mattergen_pretrained or "mattergen_base",
                mattergen_model_path=spec.mattergen_model_path,
                mattergen_sampling_config_path=spec.mattergen_sampling_config_path,
                mattergen_batch_size=spec.mattergen_batch_size,
                validation_calculator=spec.validation_calculator,
                synthesis_mode=spec.synthesis_mode,
                locked_elements=list(spec.elements),
                allow_llm_orchestration=False)
            start = time.time()
            campaign = MaterialsDiscoveryCampaign(config=config)
            if spec.condition == "random_mattergen":
                campaign.orchestrator.plan_iteration = lambda **kwargs: {
                    "elements": list(spec.elements), "num_candidates": target_candidates_per_iter,
                    "screening_criteria": {}, "memory_directives": [],
                    "rationale": "fixed random MatterGen baseline", "hypothesis": None}
            campaign.run_campaign()

            if spec.condition in MEMORY_ARMS:
                # Target memory is writable and can gain new evidence while it
                # runs.  Seal its final digest before recording completion so a
                # later restart verifies the exact finalized database.
                cls._finalize_memory_clone_marker(output_dir, memory_db_path)

            # ProvenanceTracker owns canonical manifest.json.  Never replace it
            # with the campaign report (which is a separate artifact).
            if not manifest_path.exists():
                raise CampaignRunnerError("Campaign completed without canonical provenance manifest")
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_data.get("status") != "completed" or not cls._canonical_manifest_hash(manifest_data):
                raise CampaignRunnerError("Campaign emitted invalid or non-finalized provenance")
            artifacts = {"manifest.json": compute_file_sha256(manifest_path)}
            campaign_provenance_path = output_dir / "campaign_provenance.json"
            if not campaign_provenance_path.exists():
                raise CampaignRunnerError("Campaign completed without complete campaign_provenance.json")
            # The metrics layer consumes this ordered candidate/event record;
            # include it in the completion contract so it cannot disappear or
            # be replaced between resume and analysis.
            artifacts["campaign_provenance.json"] = compute_file_sha256(campaign_provenance_path)
            if memory_db_path.exists() and memory_db_path.is_file() and memory_db_path.parent.resolve() == output_dir.resolve():
                artifacts["career_memory.db"] = compute_file_sha256(memory_db_path)
            clone_marker = output_dir / "memory_clone_integrity.json"
            if clone_marker.exists():
                artifacts["memory_clone_integrity.json"] = compute_file_sha256(clone_marker)
            report_path = output_dir / "report.json"
            if report_path.exists():
                artifacts["report.json"] = compute_file_sha256(report_path)
            cls._atomic_json(output_dir / "run_integrity.json", {
                "schema_version": 1, "run_id": spec.run_id, "spec_hash": spec.spec_hash,
                "manifest_sha256": artifacts["manifest.json"], "artifacts": artifacts})
            return {"run_id": spec.run_id, "status": RunTerminalState.SUCCESS.value,
                    "manifest_path": str(manifest_path).replace("\\", "/"),
                    "duration_seconds": time.time() - start, "skipped": False}
        except BaseException as exc:
            state = cls._classify_failure(exc)
            cls._atomic_json(output_dir / "failure_record.json", {
                "run_id": spec.run_id, "terminal_state": state.value, "error": str(exc),
                "traceback": traceback.format_exc(), "timestamp": time.time()})
            if isinstance(exc, CampaignRunnerError):
                raise
            raise CampaignRunnerError(f"Campaign run '{spec.run_id}' failed [{state.value}]: {exc}", state) from exc
