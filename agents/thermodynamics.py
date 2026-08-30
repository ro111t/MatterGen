"""Frozen-reference thermodynamics for research-grade screening.

Reference phases are prepared offline with the same evaluator and relaxation
settings used for campaign candidates.  Runtime code only loads a certified,
checksummed artifact and never retrieves reference data from the network.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

from agents.geometry import GeometryValidator
from agents.integrity import SCHEMA_VERSION

try:
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
    from pymatgen.core import Composition, Structure
except ImportError:  # pragma: no cover - research preflight rejects this case
    PDEntry = PhaseDiagram = Composition = Structure = None


THERMODYNAMICS_SCHEMA_VERSION = "1.0.0"
DEFAULT_RETAIN_THRESHOLD_EV_PER_ATOM = 0.10
DEFAULT_STABLE_THRESHOLD_EV_PER_ATOM = 0.03
DEFAULT_SENSITIVITY_THRESHOLDS = (0.03, 0.05, 0.10)


class ThermodynamicFailureCode(str, Enum):
    RELAXATION_FAILED = "RELAXATION_FAILED"
    INVALID_RELAXED_GEOMETRY = "INVALID_RELAXED_GEOMETRY"
    NONFINITE_ENERGY = "NONFINITE_ENERGY"
    INCOMPATIBLE_CHEMICAL_SYSTEM = "INCOMPATIBLE_CHEMICAL_SYSTEM"
    PHASE_DIAGRAM_FAILED = "PHASE_DIAGRAM_FAILED"


class ReferenceSetError(ValueError):
    """A frozen reference artifact is invalid, uncertified, or incompatible."""


class ThermodynamicOracleError(RuntimeError):
    """A candidate could not receive a scientifically valid oracle result."""


@dataclass(frozen=True)
class ModelIdentity:
    name: str
    version: str
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        if not self.name or not self.version or not self.checkpoint_sha256:
            raise ValueError("model name, version, and checkpoint_sha256 are required")


@dataclass(frozen=True)
class RelaxationSettings:
    fmax_ev_per_angstrom: float = 0.05
    max_steps: int = 500
    relax_cell: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.fmax_ev_per_angstrom) or not (0 < self.fmax_ev_per_angstrom <= 0.05):
            raise ValueError("fmax_ev_per_angstrom must be finite, positive, and <= 0.05")
        if (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or not (0 < self.max_steps <= 500)
        ):
            raise ValueError("max_steps must be a positive integer <= 500")
        if not self.relax_cell:
            raise ValueError("research thermodynamics requires cell relaxation")


@dataclass
class ReferencePhaseInput:
    source_id: str
    structure: Any
    source: str = "local"
    source_energy_above_hull_ev_per_atom: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReferencePhaseRecord:
    source_id: str
    source: str
    composition: str
    structure: Any
    source_energy_above_hull_ev_per_atom: Optional[float]
    metadata: Dict[str, Any]
    dedup_key: str
    success: bool
    failure_code: Optional[str] = None
    failure_message: Optional[str] = None
    relaxed_structure: Any = None
    energy_per_atom_ev: Optional[float] = None
    total_energy_ev: Optional[float] = None
    atom_count: Optional[float] = None
    max_force_ev_per_angstrom: Optional[float] = None
    max_stress_gpa: Optional[float] = None


@dataclass
class CertificationReport:
    certified: bool
    chemical_system: List[str]
    required_elemental_endpoints: List[str]
    missing_or_failed_elemental_endpoints: List[str]
    near_hull_total: int
    near_hull_succeeded: int
    other_total: int
    other_succeeded: int
    other_success_fraction: float
    failures: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class FrozenReferenceSet:
    reference_set_id: str
    chemical_system: List[str]
    model: ModelIdentity
    relaxation_settings: RelaxationSettings
    phases: List[ReferencePhaseRecord]
    certification: CertificationReport
    created_at_iso: str
    schema_version: str = THERMODYNAMICS_SCHEMA_VERSION
    reference_set_hash: Optional[str] = None

    def payload(self) -> Dict[str, Any]:
        value = asdict(self)
        value.pop("reference_set_hash", None)
        for phase in value["phases"]:
            phase["structure"] = serialize_structure(phase["structure"])
            phase["relaxed_structure"] = serialize_structure(phase["relaxed_structure"])
        return value


@dataclass
class ThermodynamicResult:
    success: bool
    predicted_energy_per_atom_ev: Optional[float] = None
    predicted_total_energy_ev: Optional[float] = None
    predicted_formation_energy_ev_per_atom: Optional[float] = None
    predicted_energy_above_hull_ev_per_atom: Optional[float] = None
    predicted_signed_hull_delta_ev_per_atom: Optional[float] = None
    predicted_thermodynamically_stable: Optional[bool] = None
    retained_by_hull_threshold: Optional[bool] = None
    decomposition: Dict[str, float] = field(default_factory=dict)
    relaxed_structure: Any = None
    max_force_ev_per_angstrom: Optional[float] = None
    max_stress_gpa: Optional[float] = None
    reference_set_id: Optional[str] = None
    reference_set_hash: Optional[str] = None
    model: Optional[Dict[str, Any]] = None
    relaxation_settings: Optional[Dict[str, Any]] = None
    failure_code: Optional[str] = None
    failure_message: Optional[str] = None

    def scientific_values(self) -> Dict[str, Any]:
        if not self.success:
            return {}
        values = {
            key: value for key, value in asdict(self).items()
            if value is not None and key not in {"success", "failure_code", "failure_message", "relaxed_structure"}
        }
        # Explicit provenance contract consumed by schema-v2 CareerMemory.
        # A hull label without this identity is not scientific evidence.
        values["provenance_schema_version"] = SCHEMA_VERSION
        values["thermodynamic_schema_version"] = THERMODYNAMICS_SCHEMA_VERSION
        values["thermodynamics_certified"] = bool(
            self.reference_set_id and self.reference_set_hash and self.model and self.relaxation_settings
        )
        return values


class StructureEvaluator(Protocol):
    model_identity: ModelIdentity
    relaxation_settings: RelaxationSettings

    def relax(self, structure: Any) -> Mapping[str, Any]: ...


class CHGNetRelaxationEvaluator:
    """Pinned CHGNet/StructOptimizer adapter used by builder and candidates."""

    def __init__(self, *, checkpoint_sha256: Optional[str] = None, model: Any = None,
                 settings: Optional[RelaxationSettings] = None,
                 model_name: str = "CHGNet"):
        try:
            from chgnet.model import CHGNet
            from chgnet.model.dynamics import StructOptimizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ReferenceSetError("CHGNet is required for the thermodynamic evaluator") from exc
        self.relaxation_settings = settings or RelaxationSettings()
        self.model = model or CHGNet.load()
        version = importlib.metadata.version("chgnet")
        digest = hashlib.sha256()
        for key, tensor in sorted(self.model.state_dict().items()):
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(key.encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(tuple(array.shape)).encode("ascii"))
            digest.update(array.tobytes())
        actual_hash = digest.hexdigest()
        if checkpoint_sha256 is not None and checkpoint_sha256 != actual_hash:
            raise ReferenceSetError("Loaded CHGNet weights do not match checkpoint_sha256")
        self.model_identity = ModelIdentity(model_name, version, actual_hash)
        self.optimizer = StructOptimizer(model=self.model)

    def relax(self, structure: Any) -> Mapping[str, Any]:
        result = self.optimizer.relax(
            structure, fmax=self.relaxation_settings.fmax_ev_per_angstrom,
            steps=self.relaxation_settings.max_steps,
            relax_cell=self.relaxation_settings.relax_cell, verbose=False,
        )
        relaxed = result["final_structure"]
        trajectory = result["trajectory"]
        forces = trajectory.forces[-1]
        stresses = trajectory.stresses[-1]
        max_force = float(max((sum(float(v) ** 2 for v in row) ** 0.5 for row in forces), default=math.inf))
        max_stress = float(max((abs(float(value)) for value in stresses.ravel()), default=math.inf)) * 160.21766208
        return {
            "converged": max_force <= self.relaxation_settings.fmax_ev_per_angstrom + 1e-12,
            "relaxed_structure": relaxed,
            "total_energy_ev": float(trajectory.energies[-1]),
            "max_force_ev_per_angstrom": max_force,
            "max_stress_gpa": max_stress,
        }


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def serialize_structure(structure: Any) -> Any:
    if structure is None:
        return None
    if hasattr(structure, "as_dict"):
        return {"format": "pymatgen", "data": structure.as_dict()}
    if isinstance(structure, Mapping):
        return {"format": "dict", "data": dict(structure)}
    raise TypeError(f"Unsupported structure type: {type(structure).__name__}")


def deserialize_structure(payload: Any) -> Any:
    if payload is None:
        return None
    if not isinstance(payload, Mapping) or "format" not in payload:
        raise ReferenceSetError("Invalid serialized structure")
    if payload["format"] == "pymatgen":
        if Structure is None:
            raise ReferenceSetError("pymatgen is required to load reference structures")
        return Structure.from_dict(payload["data"])
    if payload["format"] == "dict":
        return dict(payload["data"])
    raise ReferenceSetError(f"Unsupported structure format: {payload['format']}")


def _composition_and_count(structure: Any) -> Tuple[str, float]:
    if Structure is not None and isinstance(structure, Structure):
        return structure.composition.formula, float(structure.composition.num_atoms)
    if isinstance(structure, Mapping):
        formula = structure.get("composition", structure.get("formula"))
        if not formula or Composition is None:
            raise ValueError("Structure composition is unavailable")
        composition = Composition(str(formula))
        coords = structure.get("fractional_coordinates", structure.get("positions", structure.get("coordinates")))
        count = float(len(coords)) if coords is not None else float(composition.num_atoms)
        # ``reduced_formula`` applies special molecular conventions (for
        # example O -> O2) that would corrupt total-energy normalization.
        # The explicit formula preserves the actual composition represented by
        # this cell; ``count`` remains the number of sites used for E_total.
        return composition.formula, count
    raise ValueError("Unsupported structure type")


def _elements(formula: str) -> List[str]:
    if Composition is None:
        raise ReferenceSetError("pymatgen is required for thermodynamics")
    return sorted(str(element) for element in Composition(formula).elements)


def _dedup_key(item: ReferencePhaseInput) -> str:
    structure_payload = serialize_structure(item.structure)
    if structure_payload.get("format") == "dict":
        structure_payload = dict(structure_payload)
        structure_payload["data"] = {
            key: value for key, value in structure_payload["data"].items()
            if key not in {"candidate_id", "generation_id", "source_id"}
        }
    payload = {
        "composition": _composition_and_count(item.structure)[0],
        "structure": structure_payload,
    }
    return sha256_payload(payload)


def _finite_optional(value: Any) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _threshold_value(value: Any, name: str) -> float:
    """Normalize a nonnegative finite threshold used in scientific decisions."""
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return normalized


def _same_composition(first: str, second: str) -> bool:
    """Compare full cell stoichiometry, rather than reduced formula only."""
    if Composition is None:
        return first == second
    try:
        left = Composition(first).get_el_amt_dict()
        right = Composition(second).get_el_amt_dict()
    except Exception:
        return False
    if set(left) != set(right):
        return False
    return all(math.isclose(float(left[element]), float(right[element]), rel_tol=0.0, abs_tol=1e-8)
               for element in left)


def _relax_record(item: ReferencePhaseInput, evaluator: StructureEvaluator,
                  geometry_validator: GeometryValidator, dedup_key: str,
                  chemical_system: Sequence[str]) -> ReferencePhaseRecord:
    composition, input_atom_count = _composition_and_count(item.structure)
    base = dict(
        source_id=str(item.source_id), source=str(item.source), composition=composition,
        structure=item.structure,
        source_energy_above_hull_ev_per_atom=item.source_energy_above_hull_ev_per_atom,
        metadata=dict(item.metadata), dedup_key=dedup_key, success=False,
    )
    phase_elements = _elements(composition)
    if not set(phase_elements).issubset(set(chemical_system)):
        return ReferencePhaseRecord(
            **base,
            failure_code=ThermodynamicFailureCode.INCOMPATIBLE_CHEMICAL_SYSTEM.value,
            failure_message=(
                f"phase elements {phase_elements} are outside declared chemical "
                f"system {sorted(set(chemical_system))}"
            ),
        )
    try:
        outcome = dict(evaluator.relax(item.structure))
    except Exception as exc:
        return ReferencePhaseRecord(**base, failure_code=ThermodynamicFailureCode.RELAXATION_FAILED.value,
                                    failure_message=str(exc))
    # Scientific certification requires an explicit convergence signal and an
    # observed final force at or below the pinned tolerance.  Missing fields
    # are failures, never implicit success.
    if outcome.get("converged") is not True:
        return ReferencePhaseRecord(**base, failure_code=ThermodynamicFailureCode.RELAXATION_FAILED.value,
                                    failure_message=str(outcome.get("error", "relaxation did not converge")))
    relaxed = outcome.get("relaxed_structure")
    geometry = geometry_validator.validate(relaxed)
    if not geometry.valid:
        return ReferencePhaseRecord(**base, failure_code=ThermodynamicFailureCode.INVALID_RELAXED_GEOMETRY.value,
                                    failure_message=geometry.code, relaxed_structure=relaxed)
    relaxed_formula, atom_count = _composition_and_count(relaxed)
    if not _same_composition(relaxed_formula, composition) or not math.isclose(
        atom_count, input_atom_count, rel_tol=0.0, abs_tol=1e-8
    ):
        return ReferencePhaseRecord(**base, failure_code=ThermodynamicFailureCode.INVALID_RELAXED_GEOMETRY.value,
                                    failure_message="composition changed during relaxation", relaxed_structure=relaxed)
    energy_pa = _finite_optional(outcome.get("energy_per_atom_ev"))
    total = _finite_optional(outcome.get("total_energy_ev"))
    supplied_energy_pa = energy_pa
    supplied_total = total
    if energy_pa is None and total is not None:
        energy_pa = total / atom_count
    if total is None and energy_pa is not None:
        total = energy_pa * atom_count
    if energy_pa is None or total is None:
        return ReferencePhaseRecord(**base, failure_code=ThermodynamicFailureCode.NONFINITE_ENERGY.value,
                                    failure_message="finite energy_per_atom_ev or total_energy_ev is required",
                                    relaxed_structure=relaxed, atom_count=atom_count)
    if supplied_energy_pa is not None and supplied_total is not None and not math.isclose(
        supplied_total, supplied_energy_pa * atom_count, rel_tol=1e-8, abs_tol=1e-8
    ):
        return ReferencePhaseRecord(**base, failure_code=ThermodynamicFailureCode.NONFINITE_ENERGY.value,
                                    failure_message="energy_per_atom_ev and total_energy_ev are inconsistent",
                                    relaxed_structure=relaxed, atom_count=atom_count)
    observed_force = _finite_optional(
        outcome.get("max_force_ev_per_angstrom", outcome.get("max_force"))
    )
    if observed_force is None or observed_force > evaluator.relaxation_settings.fmax_ev_per_angstrom + 1e-12:
        return ReferencePhaseRecord(**base, failure_code=ThermodynamicFailureCode.RELAXATION_FAILED.value,
                                    failure_message="relaxation did not meet the configured force tolerance",
                                    relaxed_structure=relaxed, atom_count=atom_count,
                                    max_force_ev_per_angstrom=observed_force)
    success_values = dict(base)
    success_values["success"] = True
    return ReferencePhaseRecord(
        **success_values, relaxed_structure=relaxed, energy_per_atom_ev=energy_pa,
        total_energy_ev=total, atom_count=atom_count,
        max_force_ev_per_angstrom=observed_force,
        max_stress_gpa=_finite_optional(outcome.get("max_stress_gpa")),
    )


def _certify(phases: Sequence[ReferencePhaseRecord], chemical_system: Sequence[str]) -> CertificationReport:
    endpoints = sorted(set(chemical_system))
    endpoint_status = {element: False for element in endpoints}
    near: List[ReferencePhaseRecord] = []
    other: List[ReferencePhaseRecord] = []
    for phase in phases:
        phase_elements = _elements(phase.composition)
        if len(phase_elements) == 1 and phase_elements[0] in endpoint_status:
            endpoint_status[phase_elements[0]] = endpoint_status[phase_elements[0]] or phase.success
            continue
        source_hull = phase.source_energy_above_hull_ev_per_atom
        if source_hull is not None and float(source_hull) <= 0.05:
            near.append(phase)
        else:
            other.append(phase)
    missing = sorted(element for element, success in endpoint_status.items() if not success)
    near_ok = sum(phase.success for phase in near)
    other_ok = sum(phase.success for phase in other)
    fraction = 1.0 if not other else other_ok / len(other)
    failures = [
        {"source_id": phase.source_id, "failure_code": phase.failure_code or "UNKNOWN"}
        for phase in phases if not phase.success
    ]
    certified = not missing and near_ok == len(near) and fraction >= 0.95
    return CertificationReport(
        certified=certified, chemical_system=endpoints,
        required_elemental_endpoints=endpoints,
        missing_or_failed_elemental_endpoints=missing,
        near_hull_total=len(near), near_hull_succeeded=near_ok,
        other_total=len(other), other_succeeded=other_ok,
        other_success_fraction=fraction, failures=failures,
    )


def build_frozen_reference_set(
    *, reference_set_id: str, chemical_system: Sequence[str], inputs: Iterable[ReferencePhaseInput],
    evaluator: StructureEvaluator, output_path: Path | str, created_at_iso: Optional[str] = None,
    geometry_min_distance: float = 0.8,
) -> FrozenReferenceSet:
    """Relax, certify, and freeze a local reference set without network access."""
    system = sorted(set(str(element) for element in chemical_system))
    if not reference_set_id or not system:
        raise ValueError("reference_set_id and chemical_system are required")
    grouped: Dict[str, List[ReferencePhaseInput]] = {}
    for item in inputs:
        composition = _composition_and_count(item.structure)[0]
        phase_elements = _elements(composition)
        if not set(phase_elements).issubset(set(system)):
            raise ValueError(f"Reference phase {item.source_id!r} is outside chemical system {system}")
        grouped.setdefault(_dedup_key(item), []).append(item)

    # Merge exact structure duplicates deterministically.  The lowest known
    # source hull distance wins, so a near-hull record cannot be hidden by a
    # lexicographically earlier far-hull duplicate.  All source provenance is
    # retained in the selected record's metadata.
    unique: Dict[str, ReferencePhaseInput] = {}
    for key, members in grouped.items():
        def priority(value: ReferencePhaseInput) -> Tuple[float, str, str]:
            hull = _finite_optional(value.source_energy_above_hull_ev_per_atom)
            return (hull if hull is not None else math.inf, str(value.source), str(value.source_id))

        chosen = min(members, key=priority)
        if len(members) > 1:
            metadata = dict(chosen.metadata)
            metadata["duplicate_source_ids"] = sorted(str(value.source_id) for value in members)
            metadata["duplicate_sources"] = [
                {
                    "source_id": str(value.source_id),
                    "source": str(value.source),
                    "source_energy_above_hull_ev_per_atom": _finite_optional(value.source_energy_above_hull_ev_per_atom),
                }
                for value in sorted(members, key=priority)
            ]
            known_hulls = [
                float(value.source_energy_above_hull_ev_per_atom)
                for value in members
                if _finite_optional(value.source_energy_above_hull_ev_per_atom) is not None
            ]
            chosen = replace(
                chosen,
                metadata=metadata,
                source_energy_above_hull_ev_per_atom=min(known_hulls) if known_hulls else None,
            )
        unique[key] = chosen
    validator = GeometryValidator(min_distance=geometry_min_distance)
    phases = [
        _relax_record(item, evaluator, validator, key, system)
        for key, item in sorted(unique.items())
    ]
    certification = _certify(phases, system)
    frozen = FrozenReferenceSet(
        reference_set_id=reference_set_id, chemical_system=system,
        model=evaluator.model_identity, relaxation_settings=evaluator.relaxation_settings,
        phases=phases, certification=certification,
        created_at_iso=created_at_iso or datetime.now(timezone.utc).isoformat(),
    )
    write_frozen_reference_set(frozen, output_path)
    return frozen


def write_frozen_reference_set(reference_set: FrozenReferenceSet, output_path: Path | str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = reference_set.payload()
    digest = sha256_payload(payload)
    canonical = canonical_json(payload) + "\n"
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    if path.exists():
        # Frozen artifacts are immutable and must remain byte-canonical;
        # changing scientific content or formatting requires a new path/version.
        try:
            existing_raw = path.read_text(encoding="utf-8")
            existing_payload = json.loads(existing_raw)
            existing_digest = sha256_payload(existing_payload)
        except Exception as exc:
            raise ReferenceSetError(f"Existing reference-set artifact is unreadable: {path}") from exc
        if existing_digest != digest:
            raise ReferenceSetError(
                f"Reference-set artifact already exists with different content: {path}"
            )
        if existing_raw != canonical:
            raise ReferenceSetError(
                f"Existing reference-set artifact is not canonical JSON: {path}"
            )
        if not checksum_path.exists() or checksum_path.read_text(encoding="ascii").strip() != digest:
            raise ReferenceSetError(f"Existing reference-set checksum is missing or invalid: {checksum_path}")
    else:
        path.write_text(canonical, encoding="utf-8")
        checksum_path.write_text(digest + "\n", encoding="ascii")
    reference_set.reference_set_hash = digest
    return digest


def _frozen_from_payload(payload: Mapping[str, Any], digest: str) -> FrozenReferenceSet:
    if payload.get("schema_version") != THERMODYNAMICS_SCHEMA_VERSION:
        raise ReferenceSetError(f"Unsupported thermodynamics schema: {payload.get('schema_version')!r}")
    phases = []
    for raw in payload.get("phases", []):
        value = dict(raw)
        value["structure"] = deserialize_structure(value.get("structure"))
        value["relaxed_structure"] = deserialize_structure(value.get("relaxed_structure"))
        phases.append(ReferencePhaseRecord(**value))
    return FrozenReferenceSet(
        reference_set_id=payload["reference_set_id"], chemical_system=list(payload["chemical_system"]),
        model=ModelIdentity(**payload["model"]),
        relaxation_settings=RelaxationSettings(**payload["relaxation_settings"]),
        phases=phases, certification=CertificationReport(**payload["certification"]),
        created_at_iso=payload["created_at_iso"], schema_version=payload["schema_version"],
        reference_set_hash=digest,
    )


def load_frozen_reference_set(
    path: Path | str, *, expected_model: Optional[ModelIdentity] = None,
    expected_settings: Optional[RelaxationSettings] = None,
    required_chemical_system: Optional[Sequence[str]] = None,
    require_certified: bool = True,
) -> FrozenReferenceSet:
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except Exception as exc:
        raise ReferenceSetError(f"Reference-set artifact is unreadable: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ReferenceSetError("Reference-set artifact root must be a JSON object")
    digest = sha256_payload(payload)
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    try:
        checksum = checksum_path.read_text(encoding="ascii").strip()
    except Exception:
        checksum = None
    if checksum != digest:
        raise ReferenceSetError("Reference-set SHA256 verification failed")
    if raw != canonical_json(payload) + "\n":
        raise ReferenceSetError("Reference-set artifact is not canonical JSON")
    try:
        frozen = _frozen_from_payload(payload, digest)
    except Exception as exc:
        if isinstance(exc, ReferenceSetError):
            raise
        raise ReferenceSetError("Reference-set payload is invalid") from exc
    try:
        recomputed_certification = _certify(frozen.phases, frozen.chemical_system)
    except Exception as exc:
        raise ReferenceSetError("Reference-set certification cannot be recomputed") from exc
    if recomputed_certification != frozen.certification:
        raise ReferenceSetError("Reference-set certification does not match its phase records")
    if require_certified and not frozen.certification.certified:
        raise ReferenceSetError("Reference set is not certified for research use")
    if expected_model is not None and frozen.model != expected_model:
        raise ReferenceSetError("Reference-set model/checkpoint identity mismatch")
    if expected_settings is not None and frozen.relaxation_settings != expected_settings:
        raise ReferenceSetError("Reference-set relaxation settings mismatch")
    if required_chemical_system is not None and set(required_chemical_system) != set(frozen.chemical_system):
        raise ReferenceSetError("Reference-set chemical system mismatch")
    return frozen


class ThermodynamicOracle:
    """Evaluate relaxed candidates against an immutable reference hull."""

    def __init__(self, reference_set: FrozenReferenceSet, evaluator: StructureEvaluator, *,
                 retain_threshold_ev_per_atom: float = DEFAULT_RETAIN_THRESHOLD_EV_PER_ATOM,
                 stable_threshold_ev_per_atom: float = DEFAULT_STABLE_THRESHOLD_EV_PER_ATOM,
                 research: bool = False, geometry_min_distance: float = 0.8):
        if not reference_set.certification.certified:
            raise ReferenceSetError("Certified reference set required")
        if evaluator.model_identity != reference_set.model:
            raise ReferenceSetError("Candidate evaluator model/checkpoint differs from frozen references")
        if evaluator.relaxation_settings != reference_set.relaxation_settings:
            raise ReferenceSetError("Candidate evaluator settings differ from frozen references")
        if PhaseDiagram is None:
            raise ReferenceSetError("pymatgen phase-diagram support is required")
        self.reference_set = reference_set
        self.evaluator = evaluator
        self.research = research
        self.geometry_validator = GeometryValidator(min_distance=geometry_min_distance)
        self.retain_threshold_ev_per_atom = _threshold_value(
            retain_threshold_ev_per_atom, "retain_threshold_ev_per_atom"
        )
        self.stable_threshold_ev_per_atom = _threshold_value(
            stable_threshold_ev_per_atom, "stable_threshold_ev_per_atom"
        )
        entries = [PDEntry(phase.composition, phase.total_energy_ev, name=phase.source_id)
                   for phase in reference_set.phases if phase.success]
        self.phase_diagram = PhaseDiagram(entries)
        self._cache: Dict[str, ThermodynamicResult] = {}

    @property
    def capability(self) -> bool:
        return True

    def cache_key_for(self, structure: Any, cache_key: Optional[str] = None) -> str:
        """Return the stable key used by the oracle's result cache."""
        if cache_key is not None:
            return str(cache_key)
        try:
            return sha256_payload({"structure": serialize_structure(structure)})
        except Exception:
            return sha256_payload({"structure_repr": repr(structure)})

    def has_cached_result(self, cache_key: str) -> bool:
        return str(cache_key) in self._cache

    @property
    def cache(self) -> Mapping[str, ThermodynamicResult]:
        """Read-only view of cached oracle results for budget integrations."""
        return self._cache

    def _failure(self, code: ThermodynamicFailureCode, message: str,
                 *, cache_key: Optional[str] = None) -> ThermodynamicResult:
        if self.research:
            raise ThermodynamicOracleError(f"{code.value}: {message}")
        result = ThermodynamicResult(
            success=False, failure_code=code.value, failure_message=message,
            reference_set_id=self.reference_set.reference_set_id,
            reference_set_hash=self.reference_set.reference_set_hash,
        )
        if cache_key is not None:
            self._cache[str(cache_key)] = result
        return result

    def evaluate(self, structure: Any, *, cache_key: Optional[str] = None) -> ThermodynamicResult:
        key = self.cache_key_for(structure, cache_key)
        if key in self._cache:
            return self._cache[key]
        try:
            input_formula, input_atom_count = _composition_and_count(structure)
        except Exception as exc:
            return self._failure(
                ThermodynamicFailureCode.INVALID_RELAXED_GEOMETRY,
                f"candidate composition is unavailable: {exc}", cache_key=key,
            )
        try:
            outcome = dict(self.evaluator.relax(structure))
        except Exception as exc:
            return self._failure(ThermodynamicFailureCode.RELAXATION_FAILED, str(exc), cache_key=key)
        if outcome.get("converged") is not True:
            return self._failure(ThermodynamicFailureCode.RELAXATION_FAILED,
                                 str(outcome.get("error", "relaxation did not converge")), cache_key=key)
        observed_force = _finite_optional(
            outcome.get("max_force_ev_per_angstrom", outcome.get("max_force"))
        )
        if observed_force is None or observed_force > self.reference_set.relaxation_settings.fmax_ev_per_angstrom + 1e-12:
            return self._failure(ThermodynamicFailureCode.RELAXATION_FAILED,
                                 "relaxation did not meet the configured force tolerance", cache_key=key)
        relaxed = outcome.get("relaxed_structure")
        geometry = self.geometry_validator.validate(relaxed)
        if not geometry.valid:
            return self._failure(ThermodynamicFailureCode.INVALID_RELAXED_GEOMETRY, geometry.code, cache_key=key)
        try:
            formula, atom_count = _composition_and_count(relaxed)
        except Exception as exc:
            return self._failure(
                ThermodynamicFailureCode.INVALID_RELAXED_GEOMETRY,
                f"relaxed composition is unavailable: {exc}", cache_key=key,
            )
        if not _same_composition(formula, input_formula) or not math.isclose(
            atom_count, input_atom_count, rel_tol=0.0, abs_tol=1e-8
        ):
            return self._failure(
                ThermodynamicFailureCode.INVALID_RELAXED_GEOMETRY,
                "composition changed during relaxation", cache_key=key,
            )
        if not set(_elements(formula)).issubset(set(self.reference_set.chemical_system)):
            return self._failure(ThermodynamicFailureCode.INCOMPATIBLE_CHEMICAL_SYSTEM, formula, cache_key=key)
        energy_pa = _finite_optional(outcome.get("energy_per_atom_ev"))
        total = _finite_optional(outcome.get("total_energy_ev"))
        supplied_energy_pa = energy_pa
        supplied_total = total
        if energy_pa is None and total is not None:
            energy_pa = total / atom_count
        if total is None and energy_pa is not None:
            total = energy_pa * atom_count
        if energy_pa is None or total is None:
            return self._failure(ThermodynamicFailureCode.NONFINITE_ENERGY, "finite energy is required", cache_key=key)
        if supplied_energy_pa is not None and supplied_total is not None and not math.isclose(
            supplied_total, supplied_energy_pa * atom_count, rel_tol=1e-8, abs_tol=1e-8
        ):
            return self._failure(
                ThermodynamicFailureCode.NONFINITE_ENERGY,
                "energy_per_atom_ev and total_energy_ev are inconsistent", cache_key=key,
            )
        try:
            entry = PDEntry(formula, total, name=cache_key or formula)
            decomposition, signed_delta = self.phase_diagram.get_decomp_and_e_above_hull(
                entry, allow_negative=True, check_stable=False
            )
            formation = float(self.phase_diagram.get_form_energy_per_atom(entry))
            signed_delta = float(signed_delta)
            above_hull = max(0.0, signed_delta)
            # Reference source IDs are unambiguous even when pymatgen applies
            # molecular formula conventions (for example LiO -> Li2O2).
            products = {str(product.name): float(amount)
                        for product, amount in decomposition.items()}
        except Exception as exc:
            return self._failure(ThermodynamicFailureCode.PHASE_DIAGRAM_FAILED, str(exc), cache_key=key)
        result = ThermodynamicResult(
            success=True, predicted_energy_per_atom_ev=energy_pa,
            predicted_total_energy_ev=total,
            predicted_formation_energy_ev_per_atom=formation,
            predicted_energy_above_hull_ev_per_atom=above_hull,
            predicted_signed_hull_delta_ev_per_atom=signed_delta,
            predicted_thermodynamically_stable=above_hull <= self.stable_threshold_ev_per_atom + 1e-12,
            retained_by_hull_threshold=above_hull <= self.retain_threshold_ev_per_atom + 1e-12,
            decomposition=products, relaxed_structure=relaxed,
            max_force_ev_per_angstrom=observed_force,
            max_stress_gpa=_finite_optional(outcome.get("max_stress_gpa")),
            reference_set_id=self.reference_set.reference_set_id,
            reference_set_hash=self.reference_set.reference_set_hash,
            model=asdict(self.reference_set.model),
            relaxation_settings=asdict(self.reference_set.relaxation_settings),
        )
        self._cache[key] = result
        return result


def threshold_sensitivity(values: Iterable[Optional[float]],
                          thresholds: Sequence[float] = DEFAULT_SENSITIVITY_THRESHOLDS) -> Dict[str, int]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return {f"{float(threshold):.2f}": sum(value <= float(threshold) for value in finite)
            for threshold in thresholds}


__all__ = [
    "THERMODYNAMICS_SCHEMA_VERSION", "DEFAULT_RETAIN_THRESHOLD_EV_PER_ATOM",
    "DEFAULT_STABLE_THRESHOLD_EV_PER_ATOM", "DEFAULT_SENSITIVITY_THRESHOLDS",
    "ThermodynamicFailureCode", "ReferenceSetError", "ThermodynamicOracleError",
    "ModelIdentity", "RelaxationSettings", "ReferencePhaseInput", "ReferencePhaseRecord",
    "CertificationReport", "FrozenReferenceSet", "ThermodynamicResult", "StructureEvaluator",
    "CHGNetRelaxationEvaluator",
    "canonical_json", "sha256_payload", "serialize_structure", "deserialize_structure",
    "build_frozen_reference_set", "write_frozen_reference_set", "load_frozen_reference_set",
    "ThermodynamicOracle", "threshold_sensitivity",
]
