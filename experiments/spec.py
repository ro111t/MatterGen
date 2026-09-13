"""Typed, validated experiment specifications and configuration models.

Fails closed on unknown keys, invalid condition combinations, missing research
budgets, or unpinned inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from agents.integrity import SCHEMA_VERSION

SPEC_SCHEMA_VERSION = "2.0.0"
BASE_COMMIT_SHA = "b5162348fe17db62480882d197f01bf768214f3e"

FIVE_CONDITIONS = (
    "random_mattergen",
    "adaptive_no_memory",
    "text_summary_memory",
    "structured_provenance_memory",
    "shuffled_memory_control",
)

MEMORY_ARMS = (
    "text_summary_memory",
    "structured_provenance_memory",
    "shuffled_memory_control",
)

STANDARD_TASKS = ("Li-P-S", "Li-P-Se", "Na-P-S")
SOURCE_TASK = "Li-P-S"
TARGET_TASKS = ("Li-P-Se", "Na-P-S")

DEFAULT_FAMILY_WISE_ALPHA = 0.05
CONFIRMATORY_CONTROLS = (
    "adaptive_no_memory",
    "text_summary_memory",
    "shuffled_memory_control",
)
CONFIRMATORY_METRICS = (
    "oracle_calls_to_first_candidate_at_or_below_0_10",
    "fraction_at_or_below_0_10",
)
DEFAULT_MASTER_SEEDS = (42, 137, 2024, 777, 999, 31415, 27182, 16180, 104729)

CANONICAL_RESEARCH_PREFLIGHT_CHECKS = (
    "schema_version_ok",
    "base_commit_ok",
    "conditions_valid",
    "master_seeds_valid",
    "budgets_valid",
    "reference_sets_ok",
    "research_tree_clean",
    "pinned_chgnet_ok",
    "mattergen_checkpoint_ok",
    "sssp_manifest_ok",
    "qe_executable_ok",
    "mattergen_sampling_config_ok",
    "no_research_mocks",
    "research_backend_ok",
    "environment_ok",
)


class ExperimentSpecError(ValueError):
    """Raised when an experiment specification is invalid or malformed."""


MAX_EXACT_PERMUTATION_SAMPLE_SIZE = 16


def calculate_minimum_exact_test_sample_size(
    target_tasks_count: int,
    confirmatory_controls_count: int = len(CONFIRMATORY_CONTROLS),
    confirmatory_endpoints_count: int = len(CONFIRMATORY_METRICS),
    alpha: float = DEFAULT_FAMILY_WISE_ALPHA,
) -> int:
    """Calculate minimum paired sample size n for exact randomization test under Holm correction.

    For n nonzero paired differences, the minimum possible two-sided exact p-value is 2^(1 - n).
    In a family of m = target_tasks * confirmatory_controls * confirmatory_endpoints tests,
    the most favorable Holm-adjusted p-value is m * 2^(1 - n).
    The smallest n satisfying m * 2^(1 - n) <= alpha is ceil(1 + log2(m / alpha)).
    """
    if target_tasks_count <= 0 or confirmatory_controls_count <= 0 or confirmatory_endpoints_count <= 0 or alpha <= 0:
        raise ValueError("All task/control/endpoint counts and alpha must be positive")
    family_size = int(target_tasks_count * confirmatory_controls_count * confirmatory_endpoints_count)
    n = 1
    while family_size * (2.0 ** (1 - n)) > alpha + 1e-15:
        n += 1
    if n > MAX_EXACT_PERMUTATION_SAMPLE_SIZE:
        max_tasks = int((alpha * (2.0 ** (MAX_EXACT_PERMUTATION_SAMPLE_SIZE - 1))) / (confirmatory_controls_count * confirmatory_endpoints_count))
        raise ExperimentSpecError(
            f"Calculated sample size n={n} exceeds exact permutation enumeration boundary (n <= {MAX_EXACT_PERMUTATION_SAMPLE_SIZE}). "
            f"Reduce target_tasks_count (max supported: {max_tasks})."
        )
    return n


def _canonical_json(data: Any) -> str:
    """Return deterministic canonical JSON string."""
    def _normalize(val: Any) -> Any:
        if isinstance(val, Path):
            return str(val).replace("\\", "/")
        if isinstance(val, Mapping):
            return {str(k): _normalize(v) for k, v in sorted(val.items(), key=lambda x: str(x[0]))}
        if isinstance(val, (list, tuple, set)):
            return [_normalize(x) for x in val]
        return val

    return json.dumps(_normalize(data), sort_keys=True, separators=(",", ":"), default=str)


def compute_sha256(data: Any) -> str:
    """Compute SHA256 hex digest of canonical JSON or bytes."""
    if isinstance(data, (bytes, bytearray)):
        return hashlib.sha256(data).hexdigest()
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TaskDefinition:
    """Specification of a chemical system task."""
    task_id: str
    elements: List[str]
    domain: str = "materials_discovery"
    target_properties: Dict[str, float] = field(default_factory=lambda: {"screening_quality": 1.0})
    constraints: Dict[str, Any] = field(default_factory=dict)
    reference_set_path: Optional[str] = None
    reference_set_sha256: Optional[str] = None
    # A reference file is not a certified scientific input merely because it
    # exists.  Certification is an explicit part of the frozen task spec.
    reference_set_certified: bool = False

    def __post_init__(self):
        if not self.task_id or not isinstance(self.task_id, str):
            raise ExperimentSpecError("task_id must be a non-empty string")
        if not self.elements or not isinstance(self.elements, list) or len(self.elements) < 1:
            raise ExperimentSpecError(f"Task '{self.task_id}' must specify at least one element")
        if self.reference_set_path and self.reference_set_sha256:
            digest = str(self.reference_set_sha256).lower()
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ExperimentSpecError(
                    f"Task '{self.task_id}' reference_set_sha256 must be a 64-character hexadecimal digest"
                )


@dataclass(frozen=True)
class TransferDeclaration:
    """Explicit declaration of experimental transfer relationship between tasks."""
    source_task: str
    target_task: str
    allowed_relationship: str = "homologous_series"
    source_chemical_system: List[str] = field(default_factory=list)
    target_chemical_system: List[str] = field(default_factory=list)
    confidence: float = 0.8

    def __post_init__(self):
        if not self.source_task or not self.target_task:
            raise ExperimentSpecError("source_task and target_task are required in TransferDeclaration")
        if self.source_task == self.target_task and self.allowed_relationship != "same_system":
            raise ExperimentSpecError("Self-transfer must specify allowed_relationship='same_system'")


@dataclass(frozen=True)
class QEAuditConfig:
    """Configuration for Quantum ESPRESSO local decomposition audit."""
    candidate_count: int = 10
    ecutwfc_ry: float = 60.0
    ecutrho_ry: Optional[float] = 480.0
    kpoints_spacing_inv_ang: float = 0.25
    conv_thr_ev: float = 1e-5
    force_conv_thr_ev_per_ang: float = 0.05
    stress_conv_thr_gpa: float = 0.5
    smearing_degauss_ry: float = 0.01
    occupations: str = "smearing"
    smearing_type: str = "gaussian"
    sssp_manifest_path: Optional[str] = None
    sssp_manifest_sha256: Optional[str] = None
    timeout_seconds_per_job: int = 3600
    mock_execution: bool = True  # Default offline mock execution for automated testing
    qe_executable: str = "pw.x"
    qe_executable_version: Optional[str] = None
    qe_executable_sha256: Optional[str] = None

    def __post_init__(self):
        if self.ecutwfc_ry < 60.0:
            raise ExperimentSpecError(f"ecutwfc_ry ({self.ecutwfc_ry}) is too low (minimum 60.0 Ry)")
        if self.kpoints_spacing_inv_ang > 0.25:
            raise ExperimentSpecError("k-point mesh spacing target must be <= 0.25 inverse Angstrom")
        if self.qe_executable_sha256:
            digest = str(self.qe_executable_sha256).lower()
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ExperimentSpecError("qe_executable_sha256 must be a 64-character hexadecimal digest")


@dataclass(frozen=True)
class RunSpec:
    """Fully resolved specification for a single campaign run."""
    run_id: str
    experiment_id: str
    task_id: str
    elements: List[str]
    condition: str
    seed: int
    iteration_seeds: List[int]
    proposal_budget: int
    oracle_budget: int
    geometry_min_distance: float
    thermodynamics_retain_threshold_ev_per_atom: float
    thermodynamics_stable_threshold_ev_per_atom: float
    run_mode: str
    generation_backend: str
    memory_mode: str
    memory_seed: int
    output_dir: str
    memory_transfer_declaration: Optional[Dict[str, Any]] = None
    reference_set_path: Optional[str] = None
    reference_set_sha256: Optional[str] = None
    reference_set_certified: bool = False
    source_memory_snapshot_path: Optional[str] = None
    source_memory_snapshot_sha256: Optional[str] = None
    pinned_model_identity: Optional[Dict[str, str]] = None
    pinned_relaxation_settings: Optional[Dict[str, Any]] = None
    # Resolved execution inputs.  Keeping these on the per-run spec prevents
    # a runner from silently falling back to process/global defaults.
    career_db_path: Optional[str] = None
    mattergen_pretrained: Optional[str] = None
    mattergen_model_path: Optional[str] = None
    mattergen_checkpoint_sha256: Optional[str] = None
    mattergen_sampling_config_path: Optional[str] = None
    mattergen_sampling_config_sha256: Optional[str] = None
    mattergen_batch_size: int = 16
    domain: str = "materials_discovery"
    target_properties: Dict[str, float] = field(default_factory=lambda: {"screening_quality": 1.0})
    task_constraints: Dict[str, Any] = field(default_factory=dict)
    validation_calculator: str = "mock"
    synthesis_mode: str = "mock"
    allow_llm_orchestration: bool = True

    def __post_init__(self):
        valid_conditions = FIVE_CONDITIONS + ("source_neutral",)
        if self.condition not in valid_conditions:
            raise ExperimentSpecError(f"Invalid condition '{self.condition}'; must be one of {valid_conditions}")
        if self.proposal_budget <= 0:
            raise ExperimentSpecError(f"proposal_budget must be positive, got {self.proposal_budget}")
        if self.oracle_budget <= 0:
            raise ExperimentSpecError(f"oracle_budget must be positive, got {self.oracle_budget}")
        if self.oracle_budget > self.proposal_budget:
            raise ExperimentSpecError(f"oracle_budget ({self.oracle_budget}) cannot exceed proposal_budget ({self.proposal_budget})")
        if self.condition in MEMORY_ARMS and not self.source_memory_snapshot_path:
            raise ExperimentSpecError(f"Condition '{self.condition}' requires source_memory_snapshot_path")
        if self.run_mode not in {"development", "research"}:
            raise ExperimentSpecError("run_mode must be 'development' or 'research'")
        if self.generation_backend not in {"mock", "mattergen"}:
            raise ExperimentSpecError("generation_backend must be one of {'mock', 'mattergen'}")
        if self.run_mode == "research" and self.generation_backend != "mattergen":
            raise ExperimentSpecError("research runs require generation_backend='mattergen'")
        if self.run_mode == "research":
            if not self.reference_set_path or not self.reference_set_sha256 or not self.reference_set_certified:
                raise ExperimentSpecError("research runs require a certified reference set path and SHA256")
            if not self.pinned_model_identity or str(self.pinned_model_identity.get("name", "")).lower() != "chgnet":
                raise ExperimentSpecError("research runs require a pinned CHGNet identity")
            if (
                not self.pinned_model_identity.get("version")
                or not self.pinned_model_identity.get("checkpoint_sha256")
                or not self.pinned_relaxation_settings
            ):
                raise ExperimentSpecError(
                    "research runs require pinned model version, checkpoint SHA256, and relaxation settings"
                )
            if not self.mattergen_model_path or not self.mattergen_checkpoint_sha256:
                raise ExperimentSpecError("research runs require a local MatterGen checkpoint and SHA256")
        if self.mattergen_checkpoint_sha256:
            digest = str(self.mattergen_checkpoint_sha256).lower()
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ExperimentSpecError("mattergen_checkpoint_sha256 must be a 64-character hexadecimal digest")

    @property
    def spec_hash(self) -> str:
        return compute_sha256(asdict(self))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RunSpec:
        valid_keys = {f for f in cls.__dataclass_fields__}
        extra = set(data.keys()) - valid_keys
        if extra:
            raise ExperimentSpecError(f"Unknown keys in RunSpec: {sorted(extra)}")
        return cls(**data)


@dataclass(frozen=True)
class ExperimentSpec:
    """Top-level immutable specification for the complete experiment benchmark."""
    experiment_id: str
    schema_version: str = SPEC_SCHEMA_VERSION
    code_commit: str = BASE_COMMIT_SHA
    analysis_version: str = "1.0.0"
    source_task: TaskDefinition = field(
        default_factory=lambda: TaskDefinition(task_id=SOURCE_TASK, elements=["Li", "P", "S"])
    )
    target_tasks: List[TaskDefinition] = field(
        default_factory=lambda: [
            TaskDefinition(task_id="Li-P-Se", elements=["Li", "P", "Se"]),
            TaskDefinition(task_id="Na-P-S", elements=["Na", "P", "S"]),
        ]
    )
    transfer_declarations: List[TransferDeclaration] = field(
        default_factory=lambda: [
            TransferDeclaration(
                source_task="Li-P-S", target_task="Li-P-Se",
                allowed_relationship="homologous_series",
                source_chemical_system=["Li", "P", "S"],
                target_chemical_system=["Li", "P", "Se"],
            ),
            TransferDeclaration(
                source_task="Li-P-S", target_task="Na-P-S",
                allowed_relationship="homologous_series",
                source_chemical_system=["Li", "P", "S"],
                target_chemical_system=["Na", "P", "S"],
            ),
        ]
    )
    conditions: List[str] = field(default_factory=lambda: list(FIVE_CONDITIONS))
    master_seeds: List[int] = field(default_factory=lambda: list(DEFAULT_MASTER_SEEDS))
    proposals_per_run: int = 200
    oracle_budget_per_run: int = 100
    iterations_per_run: int = 5
    geometry_min_distance: float = 0.8
    thermodynamics_retain_threshold_ev_per_atom: float = 0.10
    thermodynamics_stable_threshold_ev_per_atom: float = 0.03
    run_mode: str = "development"
    generation_backend: str = "mock"
    output_root: str = "./benchmark_results"
    qe_audit_config: QEAuditConfig = field(default_factory=QEAuditConfig)
    pinned_model_identity: Optional[Dict[str, str]] = field(
        default_factory=lambda: {"name": "CHGNet", "version": "0.3.0"}
    )
    pinned_relaxation_settings: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            "fmax_ev_per_angstrom": 0.05,
            "max_steps": 500,
            "relax_cell": True,
        }
    )
    # Explicit model/backend inputs are kept at the experiment level and
    # copied into every RunSpec by the CLI/DAG builder.
    mattergen_pretrained: Optional[str] = None
    mattergen_model_path: Optional[str] = None
    mattergen_checkpoint_sha256: Optional[str] = None
    mattergen_sampling_config_path: Optional[str] = None
    mattergen_sampling_config_sha256: Optional[str] = None
    mattergen_batch_size: int = 16
    validation_calculator: str = "mock"
    synthesis_mode: str = "mock"
    allow_llm_orchestration: bool = True

    def __post_init__(self):
        if not self.experiment_id:
            raise ExperimentSpecError("experiment_id is required")
        if self.schema_version != SPEC_SCHEMA_VERSION:
            raise ExperimentSpecError(f"Unsupported schema_version '{self.schema_version}'; expected '{SPEC_SCHEMA_VERSION}'")
        if not self.conditions:
            raise ExperimentSpecError("conditions list cannot be empty")
        for cond in self.conditions:
            if cond not in FIVE_CONDITIONS:
                raise ExperimentSpecError(f"Unknown condition '{cond}' in conditions list")
        if len(self.conditions) != len(FIVE_CONDITIONS) or set(self.conditions) != set(FIVE_CONDITIONS):
            duplicates = sorted({c for c in self.conditions if self.conditions.count(c) > 1})
            if duplicates:
                raise ExperimentSpecError(f"conditions list contains duplicates: {duplicates}")
            raise ExperimentSpecError(
                f"conditions must contain exactly the five unique benchmark conditions {list(FIVE_CONDITIONS)}"
            )
        if not self.master_seeds:
            raise ExperimentSpecError("master_seeds list cannot be empty")
        if len(set(self.master_seeds)) != len(self.master_seeds):
            raise ExperimentSpecError("master_seeds contains duplicates")
        if self.proposals_per_run <= 0 or self.oracle_budget_per_run <= 0:
            raise ExperimentSpecError("proposals_per_run and oracle_budget_per_run must be positive integers")
        if self.oracle_budget_per_run > self.proposals_per_run:
            raise ExperimentSpecError("oracle_budget_per_run cannot exceed proposals_per_run")
        if self.run_mode not in {"development", "research"}:
            raise ExperimentSpecError("run_mode must be 'development' or 'research'")
        if self.generation_backend not in {"mock", "mattergen"}:
            raise ExperimentSpecError("generation_backend must be one of {'mock', 'mattergen'}")
        if not isinstance(self.mattergen_batch_size, int) or self.mattergen_batch_size <= 0:
            raise ExperimentSpecError("mattergen_batch_size must be a positive integer")
        if self.pinned_model_identity is not None and not isinstance(self.pinned_model_identity, dict):
            raise ExperimentSpecError("pinned_model_identity must be a mapping or None")
        if self.pinned_relaxation_settings is not None and not isinstance(self.pinned_relaxation_settings, dict):
            raise ExperimentSpecError("pinned_relaxation_settings must be a mapping or None")
        for task in [self.source_task, *self.target_tasks]:
            if task.reference_set_path and not task.reference_set_sha256:
                raise ExperimentSpecError(f"Task '{task.task_id}' reference_set_path requires reference_set_sha256")
        for digest_name in ("mattergen_checkpoint_sha256", "mattergen_sampling_config_sha256"):
            digest = getattr(self, digest_name)
            if digest:
                digest = str(digest).lower()
                if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ExperimentSpecError(f"{digest_name} must be a 64-character hexadecimal digest")
        if self.run_mode == "research":
            if self.generation_backend != "mattergen":
                raise ExperimentSpecError("research mode requires generation_backend='mattergen'")
            if not self.pinned_model_identity or str(self.pinned_model_identity.get("name", "")).lower() != "chgnet":
                raise ExperimentSpecError("research mode requires a pinned CHGNet model identity")
            if not self.pinned_model_identity.get("version") or not self.pinned_model_identity.get("checkpoint_sha256"):
                raise ExperimentSpecError("research mode requires a pinned CHGNet version and checkpoint SHA256")
            if not self.pinned_relaxation_settings:
                raise ExperimentSpecError("research mode requires pinned relaxation settings")
            if not self.mattergen_model_path or not self.mattergen_checkpoint_sha256:
                raise ExperimentSpecError("research mode requires a local MatterGen checkpoint and SHA256")
            if self.qe_audit_config.mock_execution:
                raise ExperimentSpecError("research mode is incompatible with mock QE execution")
            if not self.qe_audit_config.sssp_manifest_path or not self.qe_audit_config.sssp_manifest_sha256:
                raise ExperimentSpecError("research mode requires a pinned SSSP manifest and SHA256")
            if not self.qe_audit_config.qe_executable_version or not self.qe_audit_config.qe_executable_sha256:
                raise ExperimentSpecError("research mode requires a pinned QE executable version and SHA256")
            if str(self.validation_calculator).lower() in {"mock", "fake", "stub"}:
                raise ExperimentSpecError("research mode is incompatible with mock validation")
            if str(self.synthesis_mode).lower() in {"mock", "fake", "stub"}:
                raise ExperimentSpecError("research mode is incompatible with mock synthesis")
            try:
                min_seeds = calculate_minimum_exact_test_sample_size(len(self.target_tasks))
            except ValueError as exc:
                raise ExperimentSpecError(str(exc)) from exc
            unique_seeds_count = len(set(self.master_seeds))
            if unique_seeds_count < min_seeds:
                raise ExperimentSpecError(
                    f"research mode requires at least {min_seeds} unique master seeds for {len(self.target_tasks)} "
                    f"target tasks to be statistically feasible under Holm-Bonferroni exact testing; got {unique_seeds_count}"
                )
            if unique_seeds_count > MAX_EXACT_PERMUTATION_SAMPLE_SIZE:
                raise ExperimentSpecError(
                    f"research mode supports at most {MAX_EXACT_PERMUTATION_SAMPLE_SIZE} unique master seeds "
                    f"to remain within exact permutation enumeration; got {unique_seeds_count}"
                )

    @property
    def spec_hash(self) -> str:
        return compute_sha256(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "schema_version": self.schema_version,
            "code_commit": self.code_commit,
            "analysis_version": self.analysis_version,
            "source_task": asdict(self.source_task),
            "target_tasks": [asdict(t) for t in self.target_tasks],
            "transfer_declarations": [asdict(td) for td in self.transfer_declarations],
            "conditions": list(self.conditions),
            "master_seeds": list(self.master_seeds),
            "proposals_per_run": self.proposals_per_run,
            "oracle_budget_per_run": self.oracle_budget_per_run,
            "iterations_per_run": self.iterations_per_run,
            "geometry_min_distance": self.geometry_min_distance,
            "thermodynamics_retain_threshold_ev_per_atom": self.thermodynamics_retain_threshold_ev_per_atom,
            "thermodynamics_stable_threshold_ev_per_atom": self.thermodynamics_stable_threshold_ev_per_atom,
            "run_mode": self.run_mode,
            "generation_backend": self.generation_backend,
            "output_root": str(self.output_root).replace("\\", "/"),
            "qe_audit_config": asdict(self.qe_audit_config),
            "pinned_model_identity": self.pinned_model_identity,
            "pinned_relaxation_settings": self.pinned_relaxation_settings,
            "mattergen_pretrained": self.mattergen_pretrained,
            "mattergen_model_path": self.mattergen_model_path,
            "mattergen_checkpoint_sha256": self.mattergen_checkpoint_sha256,
            "mattergen_sampling_config_path": self.mattergen_sampling_config_path,
            "mattergen_sampling_config_sha256": self.mattergen_sampling_config_sha256,
            "mattergen_batch_size": self.mattergen_batch_size,
            "validation_calculator": self.validation_calculator,
            "synthesis_mode": self.synthesis_mode,
            "allow_llm_orchestration": self.allow_llm_orchestration,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExperimentSpec:
        expected_keys = {
            "experiment_id", "schema_version", "code_commit", "analysis_version",
            "source_task", "target_tasks", "transfer_declarations", "conditions",
            "master_seeds", "proposals_per_run", "oracle_budget_per_run",
            "iterations_per_run", "geometry_min_distance",
            "thermodynamics_retain_threshold_ev_per_atom",
            "thermodynamics_stable_threshold_ev_per_atom", "run_mode",
            "generation_backend", "output_root", "qe_audit_config",
            "pinned_model_identity", "pinned_relaxation_settings",
            "mattergen_pretrained", "mattergen_model_path", "mattergen_checkpoint_sha256",
            "mattergen_sampling_config_path", "mattergen_sampling_config_sha256", "mattergen_batch_size",
            "validation_calculator", "synthesis_mode", "allow_llm_orchestration",
        }
        unknown = set(data.keys()) - expected_keys
        if unknown:
            raise ExperimentSpecError(f"Unknown keys in ExperimentSpec: {sorted(unknown)}")
        
        parsed = dict(data)
        if "source_task" in parsed and isinstance(parsed["source_task"], dict):
            parsed["source_task"] = TaskDefinition(**parsed["source_task"])
        if "target_tasks" in parsed and isinstance(parsed["target_tasks"], list):
            parsed["target_tasks"] = [
                TaskDefinition(**t) if isinstance(t, dict) else t for t in parsed["target_tasks"]
            ]
        if "transfer_declarations" in parsed and isinstance(parsed["transfer_declarations"], list):
            parsed["transfer_declarations"] = [
                TransferDeclaration(**td) if isinstance(td, dict) else td for td in parsed["transfer_declarations"]
            ]
        if "qe_audit_config" in parsed and isinstance(parsed["qe_audit_config"], dict):
            parsed["qe_audit_config"] = QEAuditConfig(**parsed["qe_audit_config"])
        return cls(**parsed)
