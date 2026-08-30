"""Content-addressed, immutable CareerMemory snapshot management.

Ensures that text_summary_memory, structured_provenance_memory, and
shuffled_memory_control start from the exact byte-identical, frozen snapshot
of the neutral source campaign.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from agents.integrity import SCHEMA_VERSION
from experiments.spec import compute_sha256


class MemorySnapshotError(RuntimeError):
    """Raised when memory snapshot creation, verification, or isolation fails."""


@dataclass(frozen=True)
class MemorySnapshotMetadata:
    """Metadata describing a frozen CareerMemory snapshot."""
    snapshot_id: str
    source_task: str
    master_seed: int
    schema_version: str
    source_campaign_ids: List[str]
    completion_timestamps: List[float]
    record_ids: List[str]
    evidence_hashes: List[str]
    sqlite_file_sha256: str
    canonical_export_sha256: str
    allowed_transfer_declarations: List[Dict[str, Any]] = field(default_factory=list)
    snapshot_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemorySnapshotMetadata:
        return cls(**data)


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 of file on disk."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def export_canonical_snapshot_data(db_path: Path) -> Dict[str, Any]:
    """Export canonical dictionary of transferable records and campaigns for hashing."""
    if not db_path.exists():
        raise MemorySnapshotError(f"Database path '{db_path}' does not exist")

    conn = sqlite3.connect(str(db_path))
    try:
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        required = {"campaigns", "transferable_records"}
        if not required.issubset(tables):
            raise MemorySnapshotError(
                f"CareerMemory snapshot is missing required tables: {sorted(required - tables)}"
            )
        # 1. Fetch campaigns.  Keep the explicit column list stable across
        # old CareerMemory databases while retaining every field that affects
        # source identity or finalization.
        campaign_columns = {row[1] for row in conn.execute("PRAGMA table_info(campaigns)").fetchall()}
        campaign_required = {"id", "name", "domain", "start_time", "end_time", "iterations", "total_generated", "total_screened"}
        if not campaign_required.issubset(campaign_columns):
            raise MemorySnapshotError(
                f"CareerMemory snapshot campaigns table is missing required columns: "
                f"{sorted(campaign_required - campaign_columns)}"
            )
        campaign_rows = conn.execute(
            "SELECT id, name, domain, start_time, end_time, iterations, total_generated, total_screened FROM campaigns ORDER BY id"
        ).fetchall()
        campaigns = [
            {
                "id": row[0],
                "name": row[1],
                "domain": row[2],
                "start_time": row[3],
                "end_time": row[4],
                "iterations": row[5],
                "total_generated": row[6],
                "total_screened": row[7],
            }
            for row in campaign_rows
        ]

        # 2. Fetch transferable records.  Some legacy fixtures lack additive
        # columns.  Represent missing columns explicitly rather than silently
        # dropping provenance fields from the canonical export.
        record_columns = {row[1] for row in conn.execute("PRAGMA table_info(transferable_records)").fetchall()}
        record_required = {"record_id", "schema_version", "evidence_hash", "source_domain", "payload"}
        if not record_required.issubset(record_columns):
            raise MemorySnapshotError(
                f"CareerMemory snapshot transferable_records table is missing required columns: "
                f"{sorted(record_required - record_columns)}"
            )
        optional_defaults = {
            "campaign_ids": [],
            "source_candidate_ids": [],
            "source_formulas": [],
            "outcome_label": "unknown",
            "outcome_value": None,
            "evidence_count": 1,
            "confidence": 0.0,
            "finalized": None,
            "created_at": None,
        }
        canonical_record_columns = ["record_id", "schema_version", "evidence_hash", "source_domain", "payload"]
        canonical_record_columns.extend(
            key for key in optional_defaults if key in record_columns
        )
        record_rows = conn.execute(
            f"SELECT {', '.join(canonical_record_columns)} "
            "FROM transferable_records ORDER BY record_id"
        ).fetchall()
        records = []
        for row in record_rows:
            values = dict(zip(canonical_record_columns, row))
            for key, default in optional_defaults.items():
                values.setdefault(key, default)
            records.append(values)

        for record in records:
            try:
                record["payload"] = json.loads(record["payload"]) if isinstance(record["payload"], str) else record["payload"]
                for key in ("campaign_ids", "source_candidate_ids", "source_formulas"):
                    value = record[key]
                    record[key] = json.loads(value) if isinstance(value, str) else (value if value is not None else [])
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise MemorySnapshotError(
                    f"CareerMemory snapshot has invalid JSON in transferable record {record.get('record_id')}"
                ) from exc

        return {
            "campaigns": campaigns,
            "transferable_records": records,
        }
    finally:
        conn.close()


class MemorySnapshotManager:
    """Manages creation, immutability, and verification of CareerMemory snapshots."""

    @staticmethod
    def create_snapshot(
        source_db_path: Path,
        destination_snapshot_path: Path,
        source_task: str,
        master_seed: int,
        allowed_transfer_declarations: Optional[List[Dict[str, Any]]] = None,
        content_addressed: bool = False,
    ) -> MemorySnapshotMetadata:
        """Create a frozen, read-only CareerMemory snapshot from completed source run."""
        if not source_db_path.exists():
            raise MemorySnapshotError(f"Source database '{source_db_path}' does not exist")

        destination_snapshot_path = Path(destination_snapshot_path)
        if destination_snapshot_path.exists() or destination_snapshot_path.with_suffix(".json").exists():
            raise MemorySnapshotError(
                f"Refusing to overwrite existing snapshot '{destination_snapshot_path}'"
            )

        # A snapshot is evidence, not a live database.  Require at least one
        # completed source campaign and finalized transferable evidence.
        try:
            conn = sqlite3.connect(str(source_db_path))
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            if not {"campaigns", "transferable_records"}.issubset(tables):
                raise MemorySnapshotError("Source database is not a CareerMemory database")
            campaigns = conn.execute("SELECT id, end_time FROM campaigns").fetchall()
            if not campaigns or any(row[1] is None for row in campaigns):
                raise MemorySnapshotError("Source memory snapshot requires finalized source campaigns")
            record_count = conn.execute("SELECT COUNT(*) FROM transferable_records").fetchone()[0]
            if record_count <= 0:
                raise MemorySnapshotError("Source memory snapshot requires non-empty transferable evidence")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(transferable_records)").fetchall()}
            if "finalized" in columns:
                unfinished = conn.execute("SELECT COUNT(*) FROM transferable_records WHERE finalized != 1").fetchone()[0]
                if unfinished:
                    raise MemorySnapshotError("Source memory snapshot contains unfinalized transferable evidence")
        finally:
            try:
                conn.close()
            except UnboundLocalError:
                pass

        destination_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        # Copy to a same-directory temporary file and atomically publish.  The
        # destination is never replaced, so a second producer cannot silently
        # change the evidence used by already scheduled target arms.
        tmp = destination_snapshot_path.with_name(f".{destination_snapshot_path.name}.tmp-{os.getpid()}")
        if tmp.exists():
            tmp.unlink()
        shutil.copy2(source_db_path, tmp)
        tmp_hash = compute_file_sha256(tmp)
        if content_addressed:
            destination_snapshot_path = destination_snapshot_path.with_name(
                f"{destination_snapshot_path.stem}_{tmp_hash}{destination_snapshot_path.suffix}"
            )
            if destination_snapshot_path.exists() or destination_snapshot_path.with_suffix(".json").exists():
                tmp.unlink(missing_ok=True)
                raise MemorySnapshotError(
                    f"Refusing to overwrite existing snapshot '{destination_snapshot_path}'"
                )
        try:
            with tmp.open("rb") as handle:
                os.fsync(handle.fileno())
        except OSError:
            pass
        try:
            os.link(tmp, destination_snapshot_path)
            tmp.unlink()
        except FileExistsError as exc:
            tmp.unlink(missing_ok=True)
            raise MemorySnapshotError(f"Refusing to overwrite existing snapshot '{destination_snapshot_path}'") from exc
        except OSError:
            # Windows filesystems may not support hard links.  Replace is safe
            # here only after the existence check and same-process temp copy.
            if destination_snapshot_path.exists():
                tmp.unlink(missing_ok=True)
                raise MemorySnapshotError(f"Refusing to overwrite existing snapshot '{destination_snapshot_path}'")
            os.replace(tmp, destination_snapshot_path)

        # Compute file hash
        sqlite_hash = compute_file_sha256(destination_snapshot_path)
        canonical_data = export_canonical_snapshot_data(destination_snapshot_path)
        canonical_hash = compute_sha256(canonical_data)

        # Extract record IDs and evidence hashes
        record_ids = [r["record_id"] for r in canonical_data["transferable_records"]]
        evidence_hashes = [r["evidence_hash"] for r in canonical_data["transferable_records"]]
        campaign_ids = [c["id"] for c in canonical_data["campaigns"]]
        completion_timestamps = [c["end_time"] for c in canonical_data["campaigns"] if c["end_time"] is not None]

        snapshot_id = f"snapshot_{source_task}_seed{master_seed}_{sqlite_hash[:12]}"
        metadata = MemorySnapshotMetadata(
            snapshot_id=snapshot_id,
            source_task=source_task,
            master_seed=master_seed,
            schema_version=SCHEMA_VERSION,
            source_campaign_ids=campaign_ids,
            completion_timestamps=completion_timestamps,
            record_ids=record_ids,
            evidence_hashes=evidence_hashes,
            sqlite_file_sha256=sqlite_hash,
            canonical_export_sha256=canonical_hash,
            allowed_transfer_declarations=allowed_transfer_declarations or [],
            snapshot_path=str(destination_snapshot_path.resolve()).replace("\\", "/"),
        )

        # Write sidecar metadata JSON atomically and make the SQLite evidence
        # read-only.  Target clones are writable copies and are separately
        # verified against this digest.
        meta_path = destination_snapshot_path.with_suffix(".json")
        meta_tmp = meta_path.with_name(f".{meta_path.name}.tmp-{os.getpid()}")
        meta_tmp.write_text(json.dumps(metadata.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        try:
            with meta_tmp.open("rb") as handle:
                os.fsync(handle.fileno())
        except OSError:
            pass
        if meta_path.exists():
            meta_tmp.unlink(missing_ok=True)
            raise MemorySnapshotError(f"Refusing to overwrite snapshot metadata '{meta_path}'")
        os.replace(meta_tmp, meta_path)
        try:
            destination_snapshot_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        except OSError:
            # Read-only mode is an additional defense; hash verification is
            # authoritative on filesystems without POSIX permissions.
            pass
        return metadata

    @staticmethod
    def verify_snapshot_integrity(
        snapshot_path: Path,
        expected_sqlite_sha256: Optional[str] = None,
        expected_canonical_sha256: Optional[str] = None,
        expected_transfer_declaration: Optional[Mapping[str, Any]] = None,
        expected_source_task: Optional[str] = None,
        expected_master_seed: Optional[int] = None,
    ) -> bool:
        """Verify that a snapshot file has not been tampered with or mutated."""
        if not snapshot_path.exists():
            raise MemorySnapshotError(f"Snapshot path '{snapshot_path}' does not exist")

        meta_path = snapshot_path.with_suffix(".json")
        if not meta_path.exists():
            raise MemorySnapshotError(f"Snapshot sidecar '{meta_path}' does not exist")
        try:
            metadata = MemorySnapshotMetadata.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))
        except Exception as exc:
            raise MemorySnapshotError(f"Invalid snapshot sidecar '{meta_path}'") from exc

        if metadata.schema_version != SCHEMA_VERSION:
            raise MemorySnapshotError(
                f"Unsupported snapshot schema_version '{metadata.schema_version}'; "
                f"expected '{SCHEMA_VERSION}'"
            )
        if not metadata.source_task or not isinstance(metadata.record_ids, list) or not isinstance(metadata.evidence_hashes, list):
            raise MemorySnapshotError("Snapshot sidecar has invalid source or evidence identity metadata")

        actual_file_hash = compute_file_sha256(snapshot_path)
        if metadata.sqlite_file_sha256 != actual_file_hash:
            raise MemorySnapshotError(
                f"Snapshot sidecar hash mismatch: expected {metadata.sqlite_file_sha256}, got {actual_file_hash}"
            )
        if metadata.snapshot_path and Path(metadata.snapshot_path).resolve() != snapshot_path.resolve():
            raise MemorySnapshotError("Snapshot sidecar path does not match the snapshot being verified")
        if expected_source_task is not None and metadata.source_task != str(expected_source_task):
            raise MemorySnapshotError(
                f"Snapshot source task mismatch: expected {expected_source_task}, got {metadata.source_task}"
            )
        if expected_master_seed is not None and int(metadata.master_seed) != int(expected_master_seed):
            raise MemorySnapshotError(
                f"Snapshot master seed mismatch: expected {expected_master_seed}, got {metadata.master_seed}"
            )
        if expected_sqlite_sha256 and actual_file_hash != expected_sqlite_sha256:
            raise MemorySnapshotError(
                f"Snapshot file hash mismatch: expected {expected_sqlite_sha256}, got {actual_file_hash}"
            )

        actual_canonical_data = export_canonical_snapshot_data(snapshot_path)
        actual_campaigns = actual_canonical_data["campaigns"]
        if not actual_campaigns or any(c.get("end_time") is None for c in actual_campaigns):
            raise MemorySnapshotError("Snapshot database contains an unfinalized source campaign")
        records = actual_canonical_data["transferable_records"]
        if not records:
            raise MemorySnapshotError("Snapshot database contains no transferable evidence")
        campaign_ids = {str(c["id"]) for c in actual_campaigns}
        for record in records:
            referenced = record.get("campaign_ids") or []
            if not isinstance(referenced, list):
                raise MemorySnapshotError(
                    f"Snapshot record {record.get('record_id')} has non-list source campaign IDs"
                )
            if any(str(campaign_id) not in campaign_ids for campaign_id in referenced):
                raise MemorySnapshotError(
                    f"Snapshot record {record.get('record_id')} references an unknown source campaign"
                )
            if record.get("finalized") not in (None, 1, True):
                raise MemorySnapshotError(
                    f"Snapshot record {record.get('record_id')} is not finalized"
                )
        actual_canonical_hash = compute_sha256(actual_canonical_data)
        if metadata.canonical_export_sha256 != actual_canonical_hash:
            raise MemorySnapshotError(
                f"Snapshot sidecar canonical hash mismatch: expected {metadata.canonical_export_sha256}, got {actual_canonical_hash}"
            )
        if expected_canonical_sha256 and actual_canonical_hash != expected_canonical_sha256:
            raise MemorySnapshotError(
                f"Snapshot canonical hash mismatch: expected {expected_canonical_sha256}, got {actual_canonical_hash}"
            )

        # The sidecar is part of the evidence contract, rather than merely a
        # cached digest.  Reconcile every source campaign/record identity with
        # the database actually opened above so a forged or stale sidecar
        # cannot authorize transfer from different evidence.
        actual_record_ids = [r["record_id"] for r in actual_canonical_data["transferable_records"]]
        actual_evidence_hashes = [r["evidence_hash"] for r in actual_canonical_data["transferable_records"]]
        actual_campaign_ids = [c["id"] for c in actual_canonical_data["campaigns"]]
        actual_completion_timestamps = [
            c["end_time"] for c in actual_canonical_data["campaigns"] if c["end_time"] is not None
        ]
        if list(metadata.record_ids) != actual_record_ids:
            raise MemorySnapshotError("Snapshot sidecar record_ids do not match the canonical database export")
        if list(metadata.evidence_hashes) != actual_evidence_hashes:
            raise MemorySnapshotError("Snapshot sidecar evidence_hashes do not match the canonical database export")
        if list(metadata.source_campaign_ids) != actual_campaign_ids:
            raise MemorySnapshotError("Snapshot sidecar source_campaign_ids do not match the canonical database export")
        if list(metadata.completion_timestamps) != actual_completion_timestamps:
            raise MemorySnapshotError("Snapshot sidecar completion_timestamps do not match the canonical database export")
        expected_snapshot_id = f"snapshot_{metadata.source_task}_seed{metadata.master_seed}_{actual_file_hash[:12]}"
        if metadata.snapshot_id != expected_snapshot_id:
            raise MemorySnapshotError("Snapshot sidecar snapshot_id does not match the content address")

        if not isinstance(metadata.allowed_transfer_declarations, list) or any(
            not isinstance(item, Mapping) for item in metadata.allowed_transfer_declarations
        ):
            raise MemorySnapshotError("Snapshot sidecar transfer declarations are invalid")

        if expected_transfer_declaration is not None:
            requested = dict(expected_transfer_declaration)
            allowed = [dict(item) for item in metadata.allowed_transfer_declarations]
            # Target payloads may carry a deliberately smaller declaration
            # than the full TransferDeclaration stored in the snapshot.  A
            # declaration is authorized only when every requested key/value is
            # present in one of the frozen allowed declarations.
            authorized = any(
                all(
                    compute_sha256(item.get(key)) == compute_sha256(value)
                    for key, value in requested.items()
                )
                for item in allowed
            )
            if not authorized:
                raise MemorySnapshotError(
                    "Requested memory-transfer declaration is not authorized by the frozen snapshot"
                )

        return True

    @staticmethod
    def clone_for_target_run(
        snapshot_path: Path,
        target_run_db_path: Path,
    ) -> Path:
        """Clone the frozen snapshot into an isolated target run directory."""
        if not snapshot_path.exists():
            raise MemorySnapshotError(f"Snapshot path '{snapshot_path}' does not exist")
        # Validate the source and sidecar before cloning.  This also gives the
        # runner a stable source hash to record in the per-run provenance.
        MemorySnapshotManager.verify_snapshot_integrity(snapshot_path)
        target_run_db_path = Path(target_run_db_path)
        if target_run_db_path.exists():
            raise MemorySnapshotError(f"Refusing to overwrite target CareerMemory '{target_run_db_path}'")
        target_run_db_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = target_run_db_path.with_name(f".{target_run_db_path.name}.tmp-{os.getpid()}")
        shutil.copy2(snapshot_path, tmp)
        if compute_file_sha256(tmp) != compute_file_sha256(snapshot_path):
            tmp.unlink(missing_ok=True)
            raise MemorySnapshotError("Target CareerMemory clone hash mismatch")
        os.replace(tmp, target_run_db_path)
        # ``copy2`` preserves the immutable source mode on POSIX/Windows;
        # target campaigns must mutate their private clone without ever
        # changing the snapshot itself.
        try:
            target_run_db_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        except OSError:
            pass
        # Opening a CareerMemory connection here would leak a live SQLite
        # handle into the runner before the campaign opens its own connection.
        # Return the verified path and let the campaign own the sole handle.
        return target_run_db_path
