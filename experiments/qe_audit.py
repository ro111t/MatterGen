"""Local Quantum ESPRESSO decomposition audit with strict provenance.

Production execution is provided by :class:`ASEQuantumEspressoCalculator`
(direct ``pw.x`` invocation with an ASE-compatible structure schema). A fake
calculator is available solely when ``QEAuditConfig.mock_execution`` is
explicitly true and the runner is not in research mode.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from experiments.spec import QEAuditConfig, compute_sha256


RY_TO_EV = 13.605693122994


class QECalculationStatus(str, Enum):
    PREPARED = "PREPARED"
    RUNNING = "RUNNING"
    CONVERGED = "CONVERGED"
    SCF_FAILED = "SCF_FAILED"
    RELAXATION_NOT_CONVERGED = "RELAXATION_NOT_CONVERGED"
    PARSE_FAILED = "PARSE_FAILED"
    INVALID_INPUT = "INVALID_INPUT"


class QEAuditError(ValueError):
    """Invalid audit configuration, structure, pseudopotential, or reaction."""


class QESelectionInsufficiency(QEAuditError):
    """Raised only by callers opting into strict selection."""


class AuditSelection(list):
    """List-compatible selection carrying an explicit insufficiency reason."""

    def __init__(self, values: Sequence[Any] = (), *, insufficiency: Optional[str] = None):
        super().__init__(values)
        self.insufficiency = insufficiency


@dataclass
class QEAuditCandidate:
    candidate_id: str
    target_task: str
    condition: str
    seed: int
    reduced_formula: str
    structure: Dict[str, Any]
    predicted_energy_above_hull_ev_per_atom: float
    predicted_decomposition_products: List[Dict[str, Any]]
    selection_rank: int
    selection_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QEResultRecord:
    calculation_id: str
    formula: str
    is_candidate: bool
    status: str
    total_energy_ev: Optional[float]
    num_atoms: int
    energy_per_atom_ev: Optional[float]
    scf_steps: int
    relaxation_steps: int
    max_force_ev_per_ang: Optional[float]
    calculation_dir: str
    input_hash: str
    error_message: Optional[str] = None
    result_hash: Optional[str] = None
    output_hash: Optional[str] = None
    executable: Optional[str] = None
    executable_version: Optional[str] = None
    sssp_manifest_sha256: Optional[str] = None
    qe_executable_sha256: Optional[str] = None
    kpoints: Optional[Tuple[int, int, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QELocalDecompositionAuditResult:
    candidate_id: str
    target_task: str
    condition: str
    seed: int
    formula: str
    num_atoms: int
    candidate_status: str
    candidate_energy_per_atom_ev: Optional[float]
    competing_phases_count: int
    competing_phases_all_converged: bool
    chgnet_predicted_hull_distance_ev_per_atom: float
    dft_local_decomposition_margin_ev_per_atom: Optional[float]
    sign_agreement: Optional[bool]
    margin_difference_ev_per_atom: Optional[float]
    reaction_equation: str
    participating_phase_results: List[Dict[str, Any]] = field(default_factory=list)
    reaction_balanced: bool = False
    status: str = "INCONCLUSIVE"
    sensitivity: Optional[Dict[str, Any]] = None
    candidate_input_hash: Optional[str] = None
    candidate_result_hash: Optional[str] = None
    candidate_output_hash: Optional[str] = None
    qe_executable: Optional[str] = None
    qe_executable_version: Optional[str] = None
    qe_executable_sha256: Optional[str] = None
    sssp_manifest_sha256: Optional[str] = None
    # Keep the complete candidate calculation record alongside the flattened
    # C5-facing fields.  This avoids losing calculation ids/directories when a
    # CSV consumer does not also have access to the per-calculation directory.
    candidate_provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def _structure_parts(structure: Mapping[str, Any]) -> Tuple[List[List[float]], List[List[float]], List[str], bool]:
    if not isinstance(structure, Mapping):
        raise QEAuditError("A crystal structure object is required; composition-only records are invalid")
    lattice = structure.get("lattice") or structure.get("cell")
    positions = structure.get("positions") or structure.get("coords") or structure.get("coordinates")
    species = structure.get("species") or structure.get("symbols") or structure.get("elements")
    if isinstance(positions, list) and positions and isinstance(positions[0], Mapping):
        parsed_positions: List[List[float]] = []
        parsed_species: List[str] = []
        for site in positions:
            pos = site.get("frac_coords") or site.get("cart_coords") or site.get("position") or site.get("coords")
            el = site.get("element") or site.get("symbol")
            if not isinstance(pos, (list, tuple)) or len(pos) < 3 or not el:
                raise QEAuditError("Each structure site requires element and 3 coordinates")
            parsed_positions.append([float(pos[0]), float(pos[1]), float(pos[2])])
            parsed_species.append(str(el))
        positions, species = parsed_positions, parsed_species
    if not isinstance(lattice, list) or len(lattice) != 3 or any(not isinstance(v, (list, tuple)) or len(v) != 3 for v in lattice):
        raise QEAuditError("Structure requires a 3x3 lattice")
    if not isinstance(positions, list) or not positions:
        raise QEAuditError("Structure requires non-empty lattice coordinates")
    if not isinstance(species, list) or len(species) != len(positions):
        raise QEAuditError("Structure requires one element symbol per coordinate")
    try:
        lat = [[float(x) for x in row] for row in lattice]
        pos = [[float(x) for x in row[:3]] for row in positions]
    except (TypeError, ValueError, IndexError) as exc:
        raise QEAuditError(f"Non-numeric lattice/coordinates: {exc}") from exc
    if any(not all(math.isfinite(x) for x in row) for row in lat + pos):
        raise QEAuditError("Structure lattice/coordinates must be finite")
    fractional = bool(structure.get("fractional_coordinates", structure.get("fractional", True)))
    return lat, pos, [str(s) for s in species], fractional


def _formula_counts(formula: str) -> Dict[str, int]:
    if not isinstance(formula, str) or not formula.strip():
        raise QEAuditError("Formula is required for atom balancing")
    tokens = re.findall(r"([A-Z][a-z]*)([0-9]*(?:\.[0-9]+)?)", formula)
    if not tokens or "".join(a + b for a, b in tokens) != formula:
        raise QEAuditError(f"Cannot parse chemical formula {formula!r}")
    counts: Dict[str, int] = {}
    for element, number in tokens:
        if "." in number:
            raise QEAuditError("Fractional formula atom counts are not supported")
        n = int(number) if number else 1
        if n <= 0:
            raise QEAuditError("Formula atom counts must be positive")
        counts[element] = counts.get(element, 0) + n
    return counts


def _structure_counts(structure: Mapping[str, Any]) -> Dict[str, int]:
    _, _, species, _ = _structure_parts(structure)
    counts: Dict[str, int] = {}
    for element in species:
        counts[element] = counts.get(element, 0) + 1
    return counts


def _validate_formula_structure(formula: str, structure: Mapping[str, Any]) -> int:
    counts = _structure_counts(structure)
    formula_counts = _formula_counts(formula)
    if counts != formula_counts:
        raise QEAuditError(f"Structure species {counts} do not match formula {formula} ({formula_counts})")
    return sum(counts.values())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_dict(config: QEAuditConfig) -> Dict[str, Any]:
    return asdict(config)


def _canonical_json_bytes(value: Any) -> bytes:
    """Serialize canonical JSON while rejecting non-finite numbers."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise QEAuditError(
            "QE audit artifacts must contain only JSON values and finite numbers"
        ) from exc


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically replace *path* with *data* in the same directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                # The replace is still atomic on filesystems where fsync is
                # unavailable (notably some Windows temporary volumes).
                pass
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(Path(path), text.encode("utf-8"))


def _atomic_write_json(path: Path, value: Any, *, indent: Optional[int] = None) -> None:
    if indent is None:
        payload = _canonical_json_bytes(value)
    else:
        try:
            payload = json.dumps(
                value,
                sort_keys=True,
                indent=indent,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise QEAuditError(
                "QE audit artifacts must contain only JSON values and finite numbers"
            ) from exc
    _atomic_write_bytes(Path(path), payload)


def _validate_qe_result_record(record: QEResultRecord) -> None:
    """Reject calculator output that could poison canonical audit artifacts."""
    if not isinstance(record, QEResultRecord):
        raise QEAuditError("QE calculators must return QEResultRecord instances")
    for field_name in (
        "total_energy_ev", "energy_per_atom_ev", "max_force_ev_per_ang",
    ):
        value = getattr(record, field_name)
        if value is not None and not _finite_number(value):
            raise QEAuditError(f"QE result field {field_name} must be finite")
    # Hashes and provenance are included in canonical result content.  This
    # check catches non-JSON values before any CSV or manifest is replaced.
    _canonical_json_bytes(record.to_dict())


def _manifest_info(config: QEAuditConfig) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    path_value = getattr(config, "sssp_manifest_path", None)
    expected = getattr(config, "sssp_manifest_sha256", None)
    if not path_value:
        if expected:
            raise QEAuditError("An SSSP manifest path is required when an expected SHA256 digest is supplied")
        return None, None
    if not expected:
        raise QEAuditError("Production SSSP manifest requires an expected SHA256 digest")
    expected_text = str(expected).strip().lower()
    if len(expected_text) != 64 or any(char not in "0123456789abcdef" for char in expected_text):
        raise QEAuditError("Expected SSSP manifest SHA256 digest must be 64 hexadecimal characters")
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        raise QEAuditError(f"SSSP manifest does not exist: {path}")
    digest = _file_sha256(path)
    if digest != expected_text:
        raise QEAuditError(f"SSSP manifest SHA256 mismatch: expected {expected}, got {digest}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise QEAuditError(f"SSSP manifest is not valid JSON: {exc}") from exc
    pseudo = data.get("pseudopotentials", data.get("pseudos", {})) if isinstance(data, Mapping) else {}
    if not isinstance(pseudo, Mapping):
        raise QEAuditError("SSSP manifest requires a pseudopotentials mapping")
    result: Dict[str, str] = {}
    for element, value in pseudo.items():
        if isinstance(value, Mapping):
            p = value.get("path") or value.get("file") or value.get("filename")
            declared = value.get("sha256")
        else:
            p, declared = value, None
        if not p:
            raise QEAuditError(f"SSSP manifest entry for {element} has no file")
        pseudo_path = Path(str(p))
        if not pseudo_path.is_absolute():
            pseudo_path = path.parent / pseudo_path
        if not pseudo_path.exists():
            raise QEAuditError(f"Pseudopotential missing for {element}: {pseudo_path}")
        actual = _file_sha256(pseudo_path)
        if not declared:
            raise QEAuditError(f"SSSP manifest entry for {element} requires a SHA256 digest")
        if actual.lower() != str(declared).lower():
            raise QEAuditError(f"Pseudopotential SHA256 mismatch for {element}")
        result[str(element)] = str(pseudo_path)
    return result, digest


def select_audit_candidates(
    candidates: Sequence[Any], target_count: int = 10,
    target_tasks: Optional[Sequence[str]] = None,
    relevant_conditions: Optional[Sequence[str]] = None,
    strict: bool = False,
) -> AuditSelection:
    """Select deterministic, target-only, structure-bearing audit candidates.

    When fewer than ``target_count`` candidates are available, no fabricated
    structures are created. The returned list carries ``.insufficiency``; set
    ``strict=True`` to raise ``QESelectionInsufficiency`` instead.
    """
    if target_count <= 0:
        raise QEAuditError("target_count must be positive")
    target_set = set(target_tasks) if target_tasks else None
    condition_set = set(relevant_conditions) if relevant_conditions else None
    eligible: List[Any] = []
    for c in candidates:
        task = str(_get(c, "task_id", _get(c, "target_task", "")))
        if target_set is not None:
            if task not in target_set:
                continue
        elif task in {"Li-P-S", "source", "source_task"}:
            continue
        cond = str(_get(c, "condition", ""))
        if condition_set is not None and cond not in condition_set:
            continue
        if not bool(_get(c, "evaluated_by_oracle", _get(c, "oracle_evaluated", False))):
            continue
        if not bool(_get(c, "oracle_success", True)):
            continue
        energy = _get(c, "predicted_energy_above_hull_ev_per_atom")
        try:
            energy_f = float(energy)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(energy_f):
            continue
        structure = _get(c, "structure")
        formula = _get(c, "reduced_formula", _get(c, "formula", _get(c, "composition")))
        if not isinstance(structure, Mapping) or not formula:
            continue
        try:
            _validate_formula_structure(str(formula), structure)
        except QEAuditError:
            continue
        eligible.append(c)
    eligible.sort(key=lambda c: (
        str(_get(c, "task_id", _get(c, "target_task", ""))),
        str(_get(c, "condition", "")),
        int(_get(c, "seed", 0)), int(_get(c, "proposal_index", 0)),
        str(_get(c, "candidate_id", "")),
    ))
    strata: Dict[Tuple[str, str], List[Any]] = {}
    for c in eligible:
        strata.setdefault((str(_get(c, "task_id", "")), str(_get(c, "condition", ""))), []).append(c)
    selected: List[Any] = []
    while len(selected) < target_count and strata:
        progressed = False
        for key in sorted(list(strata)):
            values = strata.get(key, [])
            if not values:
                strata.pop(key, None)
                continue
            selected.append(values.pop(0))
            progressed = True
            if len(selected) >= target_count:
                break
        if not progressed:
            break
    reason = None if len(selected) >= target_count else f"insufficient_valid_target_candidates:{len(selected)}/{target_count}"
    if reason and strict:
        raise QESelectionInsufficiency(reason)
    result = AuditSelection(insufficiency=reason)
    for rank, c in enumerate(selected, 1):
        result.append(QEAuditCandidate(
            candidate_id=str(_get(c, "candidate_id")),
            target_task=str(_get(c, "task_id", _get(c, "target_task"))),
            condition=str(_get(c, "condition")),
            seed=int(_get(c, "seed")),
            reduced_formula=str(_get(c, "reduced_formula", _get(c, "formula", _get(c, "composition")))),
            structure=dict(_get(c, "structure")),
            predicted_energy_above_hull_ev_per_atom=float(_get(c, "predicted_energy_above_hull_ev_per_atom")),
            predicted_decomposition_products=[dict(x) for x in (_get(c, "decomposition_products", []) or []) if isinstance(x, Mapping)],
            selection_rank=rank,
            selection_reason=f"balanced_target_stratum:{_get(c, 'task_id', '')}:{_get(c, 'condition', '')}",
        ))
    return result


def generate_kpoints_grid(lattice_lengths: Sequence[float], target_spacing_inv_ang: float = 0.25) -> Tuple[int, int, int]:
    if target_spacing_inv_ang <= 0 or target_spacing_inv_ang > 0.25:
        raise QEAuditError("k-point target spacing must be in (0, 0.25] inverse Angstrom")
    if len(lattice_lengths) != 3:
        raise QEAuditError("Three lattice lengths are required")
    grid = []
    for length in lattice_lengths:
        if float(length) <= 0:
            raise QEAuditError("Lattice lengths must be positive")
        grid.append(max(1, math.ceil((2.0 * math.pi) / (float(length) * target_spacing_inv_ang))))
    return tuple(grid)  # type: ignore[return-value]


class FakeQECalculator:
    """Deterministic test calculator; forbidden unless mock execution is explicit."""

    def __init__(self, failure_modes: Optional[Mapping[str, str]] = None):
        self.failure_modes = dict(failure_modes or {})

    def run_calculation(self, formula: str, structure: Dict[str, Any], config: QEAuditConfig, calc_dir: Path, *, is_candidate: bool = True) -> QEResultRecord:
        num_atoms = _validate_formula_structure(formula, structure)
        calc_dir.mkdir(parents=True, exist_ok=True)
        input_data = {"formula": formula, "structure": structure, "config": _config_dict(config), "is_candidate": is_candidate}
        input_hash = compute_sha256(input_data)
        input_path = calc_dir / "input.json"
        failure = self.failure_modes.get(formula)
        executable = getattr(config, "qe_executable", "pw.x")
        executable_version = getattr(config, "qe_executable_version", None)
        sssp_sha = getattr(config, "sssp_manifest_sha256", None)
        qe_sha = getattr(config, "qe_executable_sha256", None)
        if failure:
            result = QEResultRecord(f"calc_{input_hash[:16]}", formula, is_candidate, failure, None, num_atoms, None, 100, 0, None, str(calc_dir).replace("\\", "/"), input_hash, f"Injected failure: {failure}", None, None, executable, executable_version, sssp_sha, qe_sha)
        else:
            f_hash = int(hashlib.sha256(formula.encode("utf-8")).hexdigest()[:12], 16)
            energy_pa = -5.0 - (f_hash % 200) / 100.0
            result = QEResultRecord(f"calc_{input_hash[:16]}", formula, is_candidate, QECalculationStatus.CONVERGED.value, energy_pa * num_atoms, num_atoms, energy_pa, 18, 12, 0.015, str(calc_dir).replace("\\", "/"), input_hash, None, None, None, executable, executable_version, sssp_sha, qe_sha)
        result_path = calc_dir / "result.json"
        result_hash = compute_sha256(result.to_dict())
        result.result_hash = result_hash
        _atomic_write_json(result_path, result.to_dict(), indent=2)
        return result


class ASEQuantumEspressoCalculator:
    """Production calculator invoking ASE-compatible QE ``pw.x`` directly."""

    def __init__(self, executable: str = "pw.x", pseudopotentials: Optional[Mapping[str, str]] = None, executable_version: Optional[str] = None):
        self.executable = executable
        self.pseudopotentials = dict(pseudopotentials or {})
        self.executable_version = executable_version or self._detect_version()
        resolved = shutil.which(self.executable) or self.executable
        self.executable_sha256 = _file_sha256(Path(resolved)) if Path(resolved).is_file() else None
        self.qe_executable_sha256 = self.executable_sha256

    def _detect_version(self) -> Optional[str]:
        exe = shutil.which(self.executable) or self.executable
        try:
            proc = subprocess.run([exe, "-h"], capture_output=True, text=True, timeout=10, check=False)
            text = (proc.stdout or "") + (proc.stderr or "")
            match = re.search(r"(?:Program|version)\s+([0-9]+(?:\.[0-9]+)+)", text, re.I)
            return match.group(1) if match else None
        except Exception:
            return None

    def _input_text(self, formula: str, structure: Mapping[str, Any], config: QEAuditConfig) -> Tuple[str, Tuple[int, int, int], int]:
        lattice, positions, species, fractional = _structure_parts(structure)
        num_atoms = _validate_formula_structure(formula, structure)
        lengths = [math.sqrt(sum(x * x for x in row)) for row in lattice]
        kpoints = generate_kpoints_grid(lengths, float(config.kpoints_spacing_inv_ang))
        unique = list(dict.fromkeys(species))
        missing = [s for s in unique if s not in self.pseudopotentials]
        if missing:
            raise QEAuditError(f"No SSSP pseudopotential for: {missing}")
        conv_thr_ry = float(config.conv_thr_ev) / RY_TO_EV
        force_thr_ry_bohr = float(config.force_conv_thr_ev_per_ang) * 0.529177210903 / RY_TO_EV
        pressure_thr_kbar = float(config.stress_conv_thr_gpa) * 10.0
        lines = [
            "&CONTROL", " calculation='vc-relax',", " prefix='mattergen_audit',",
            " pseudo_dir='./pseudos',", f" forc_conv_thr={force_thr_ry_bohr:.8e},", "/",
            "&SYSTEM", " ibrav=0,", f" nat={num_atoms},", f" ntyp={len(unique)},",
            f" ecutwfc={float(config.ecutwfc_ry):.8f},",
            f" ecutrho={float(config.ecutrho_ry or config.ecutwfc_ry * 8):.8f},",
            f" occupations='{config.occupations}',", f" smearing='{config.smearing_type}',",
            f" degauss={float(config.smearing_degauss_ry):.8f},", "/",
            "&ELECTRONS", f" conv_thr={conv_thr_ry:.8e},", "/",
            "&IONS", " ion_dynamics='bfgs',", "/",
            "&CELL", " cell_dynamics='bfgs',", f" press_conv_thr={pressure_thr_kbar:.8f},", "/",
            "ATOMIC_SPECIES",
        ]
        for element in unique:
            lines.append(f"{element} 1.0 {Path(self.pseudopotentials[element]).name}")
        lines.append("CELL_PARAMETERS angstrom")
        lines.extend(" ".join(f"{v:.12f}" for v in row) for row in lattice)
        lines.append("ATOMIC_POSITIONS crystal" if fractional else "ATOMIC_POSITIONS angstrom")
        lines.extend(f"{el} {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}" for el, p in zip(species, positions))
        lines.extend(["K_POINTS automatic", f"{kpoints[0]} {kpoints[1]} {kpoints[2]} 0 0 0", ""])
        return "\n".join(lines), kpoints, num_atoms

    def run_calculation(self, formula: str, structure: Dict[str, Any], config: QEAuditConfig, calc_dir: Path, *, is_candidate: bool = True) -> QEResultRecord:
        text, kpoints, num_atoms = self._input_text(formula, structure, config)
        _, _, structure_species, _ = _structure_parts(structure)
        pseudo_hashes = {
            element: _file_sha256(Path(path))
            for element, path in sorted(self.pseudopotentials.items())
            if element in set(structure_species)
        }
        input_hash = compute_sha256({
            "input": text,
            "pseudopotentials": pseudo_hashes,
            "executable": self.executable,
            "executable_version": self.executable_version,
        })
        calc_dir.mkdir(parents=True, exist_ok=True)
        input_path = calc_dir / "qe.in"
        result_path = calc_dir / "result.json"
        if result_path.exists():
            try:
                cached = QEResultRecord(**json.loads(result_path.read_text(encoding="utf-8")))
                output_path = calc_dir / "qe.out"
                if (
                    cached.input_hash == input_hash
                    and cached.result_hash == compute_sha256({k: v for k, v in cached.to_dict().items() if k != "result_hash"})
                    and output_path.exists()
                    and cached.output_hash == _file_sha256(output_path)
                    and (
                        (cached.sssp_manifest_sha256 or "").lower()
                        == str(getattr(config, "sssp_manifest_sha256", None) or "").lower()
                    )
                ):
                    return cached
            except Exception:
                pass
        pseudo_dir = calc_dir / "pseudos"
        pseudo_dir.mkdir(parents=True, exist_ok=True)
        for element, source in self.pseudopotentials.items():
            if element not in pseudo_hashes:
                continue
            source_path = Path(source)
            destination = pseudo_dir / source_path.name
            if destination.exists() and _file_sha256(destination) != pseudo_hashes[element]:
                raise QEAuditError(f"Cached pseudopotential changed for {element}")
            if not destination.exists():
                tmp_pseudo = destination.with_suffix(destination.suffix + ".tmp")
                shutil.copy2(source_path, tmp_pseudo)
                if _file_sha256(tmp_pseudo) != pseudo_hashes[element]:
                    tmp_pseudo.unlink(missing_ok=True)
                    raise QEAuditError(f"Pseudopotential copy hash mismatch for {element}")
                os.replace(tmp_pseudo, destination)
        _atomic_write_text(input_path, text)
        output_path = calc_dir / "qe.out"
        exe = shutil.which(self.executable) or self.executable
        try:
            proc = subprocess.run([exe, "-in", str(input_path)], cwd=str(calc_dir), capture_output=True, text=True, timeout=int(config.timeout_seconds_per_job), check=False)
            output = (proc.stdout or "") + (proc.stderr or "")
            _atomic_write_text(output_path, output)
        except Exception as exc:
            output = ""
            _atomic_write_text(output_path, str(exc))
            proc = None
        output_hash = _file_sha256(output_path)
        energy_matches = re.findall(r"!\s+total energy\s*=\s*([-+0-9.eE]+)\s+Ry", output)
        converged = bool(
            proc is not None
            and proc.returncode == 0
            and re.search(r"JOB DONE", output, re.I)
            and energy_matches
        )
        energy = float(energy_matches[-1]) * RY_TO_EV if energy_matches else None
        status = QECalculationStatus.CONVERGED.value if converged else (QECalculationStatus.SCF_FAILED.value if proc is not None else QECalculationStatus.PARSE_FAILED.value)
        result = QEResultRecord(f"calc_{input_hash[:16]}", formula, is_candidate, status, energy, num_atoms, energy / num_atoms if energy is not None else None, len(re.findall(r"iteration #", output, re.I)), 0, None, str(calc_dir).replace("\\", "/"), input_hash, None if converged else "QE did not provide a converged total energy", None, output_hash, self.executable, self.executable_version, getattr(config, "sssp_manifest_sha256", None), self.executable_sha256 or getattr(config, "qe_executable_sha256", None), kpoints)
        canonical = result.to_dict()
        canonical.pop("result_hash", None)
        result.result_hash = compute_sha256(canonical)
        _atomic_write_json(result_path, result.to_dict(), indent=2)
        return result


def _run_calc(calculator: Any, formula: str, structure: Dict[str, Any], config: QEAuditConfig, directory: Path, *, is_candidate: bool) -> QEResultRecord:
    try:
        return calculator.run_calculation(formula, structure, config, directory, is_candidate=is_candidate)
    except TypeError:
        return calculator.run_calculation(formula, structure, config, directory)


def _product_schema(candidate: QEAuditCandidate) -> List[Tuple[str, float, Dict[str, Any]]]:
    products = candidate.predicted_decomposition_products
    if isinstance(products, Mapping):
        products = [
            {"formula": formula, "amount": coefficient}
            for formula, coefficient in products.items()
        ]
    if not isinstance(products, list) or not products:
        raise QEAuditError("An atom-balanced decomposition product list is required")
    parsed: List[Tuple[str, float, Dict[str, Any]]] = []
    for product in products:
        if not isinstance(product, Mapping):
            raise QEAuditError("Each decomposition product must be a mapping")
        formula = product.get("formula") or product.get("composition")
        coefficient = product.get("coefficient", product.get("amount"))
        structure = product.get("structure") or product.get("phase_structure")
        if not formula or coefficient is None or not isinstance(structure, Mapping):
            raise QEAuditError("Each decomposition product requires formula, coefficient/amount, and structure")
        try:
            coeff = float(coefficient)
        except (TypeError, ValueError):
            raise QEAuditError("Decomposition coefficients must be finite positive numbers")
        if not math.isfinite(coeff) or coeff <= 0:
            raise QEAuditError("Decomposition coefficients must be finite positive numbers")
        _validate_formula_structure(str(formula), structure)
        parsed.append((str(formula), coeff, dict(structure)))
    return parsed


def validate_atom_balanced_reaction(
    candidate_formula: str,
    products: Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> bool:
    """Validate elemental balance without using guessed atom counts."""
    if isinstance(products, Mapping):
        products = [
            {"formula": formula, "coefficient": coefficient}
            for formula, coefficient in products.items()
        ]
    candidate_counts = _formula_counts(candidate_formula)
    product_counts: Dict[str, float] = {}
    for product in products:
        formula = product.get("formula") or product.get("composition")
        coefficient = product.get("coefficient", product.get("amount"))
        if not formula or coefficient is None:
            return False
        try:
            coefficient_f = float(coefficient)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(coefficient_f) or coefficient_f <= 0:
            return False
        for element, count in _formula_counts(str(formula)).items():
            product_counts[element] = product_counts.get(element, 0.0) + coefficient_f * count
    return set(candidate_counts) == set(product_counts) and all(
        math.isclose(float(candidate_counts[element]), product_counts[element], rel_tol=0, abs_tol=1e-8)
        for element in candidate_counts
    )


def compute_local_decomposition_margin(
    candidate_total_energy_ev: float,
    candidate_num_atoms: int,
    phase_results: Sequence[Mapping[str, Any] | QEResultRecord],
    products: Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> float:
    """Compute ``(E_candidate - sum(c_i E_phase_i))/N_candidate``.

    Total energies are used for every phase; weighted phase energies per atom
    are deliberately not averaged, which would be wrong for unlike formulas.
    """
    if candidate_num_atoms <= 0 or not math.isfinite(float(candidate_total_energy_ev)):
        raise QEAuditError("Candidate total energy and atom count are required")
    if isinstance(products, Mapping):
        products = [{"formula": formula, "coefficient": coefficient} for formula, coefficient in products.items()]
    if len(phase_results) != len(products):
        raise QEAuditError("One converged phase result is required per decomposition product")
    reference_total = 0.0
    for phase, product in zip(phase_results, products):
        status = _get(phase, "status")
        total_energy = _get(phase, "total_energy_ev")
        coefficient = product.get("coefficient", product.get("amount"))
        if status != QECalculationStatus.CONVERGED.value or not _finite_number(total_energy) or not _finite_number(coefficient) or float(coefficient) <= 0:
            raise QEAuditError("All decomposition phases must have finite converged total energies")
        reference_total += float(coefficient) * float(total_energy)
    return (float(candidate_total_energy_ev) - reference_total) / int(candidate_num_atoms)


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


class QEAuditRunner:
    """Execute candidate and competing-phase calculations and local margins."""

    def __init__(self, config: QEAuditConfig, output_dir: Path, calculator: Optional[Any] = None, *, run_mode: str = "development", qe_executable: str = "pw.x"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.calc_root = self.output_dir / "calculations"
        self.calc_root.mkdir(parents=True, exist_ok=True)
        self.run_mode = run_mode
        if float(config.ecutwfc_ry) < 60.0:
            raise QEAuditError("QE audit requires ecutwfc >= 60 Ry")
        if float(config.kpoints_spacing_inv_ang) > 0.25:
            raise QEAuditError("QE audit requires k-point spacing <= 0.25 inverse Angstrom")
        if run_mode == "research" and bool(config.mock_execution):
            raise QEAuditError("Fake QE execution is forbidden in research mode; set mock_execution=False")
        if run_mode == "research":
            if not config.qe_executable_sha256 or len(config.qe_executable_sha256) != 64:
                raise QEAuditError("Research mode requires pinned 64-character qe_executable_sha256")
            if not config.sssp_manifest_sha256 or len(config.sssp_manifest_sha256) != 64:
                raise QEAuditError("Research mode requires pinned 64-character sssp_manifest_sha256")
        # Resolve and verify the manifest once, before selecting a calculator.
        # In production this makes the expected digest an explicit input to
        # the runner and prevents a calculator from silently choosing a
        # different SSSP database.
        if bool(config.mock_execution):
            self.sssp_pseudopotentials, self.sssp_manifest_digest = None, None
        else:
            self.sssp_pseudopotentials, self.sssp_manifest_digest = _manifest_info(config)
            if not self.sssp_pseudopotentials or not self.sssp_manifest_digest:
                raise QEAuditError("Production QE requires a verified SSSP manifest and expected SHA256 digest")
        if calculator is None:
            if bool(config.mock_execution):
                calculator = FakeQECalculator()
            else:
                resolved_executable = shutil.which(qe_executable)
                if resolved_executable is None and Path(qe_executable).is_file():
                    resolved_executable = str(Path(qe_executable).resolve())
                if not resolved_executable:
                    raise QEAuditError(f"Production QE executable was not found: {qe_executable}")
                if config.qe_executable_sha256 and _file_sha256(Path(resolved_executable)) != config.qe_executable_sha256.lower():
                    raise QEAuditError("Production QE executable SHA256 mismatch")
                calculator = ASEQuantumEspressoCalculator(resolved_executable, self.sssp_pseudopotentials)
                if not calculator.executable_version:
                    raise QEAuditError("Production QE executable/version could not be verified")
                if config.qe_executable_version and calculator.executable_version != config.qe_executable_version:
                    raise QEAuditError("Production QE executable version mismatch")
        if isinstance(calculator, FakeQECalculator) and (not bool(config.mock_execution) or run_mode == "research"):
            raise QEAuditError("FakeQECalculator requires explicit mock_execution=True outside research mode")
        if run_mode == "research" and not isinstance(calculator, ASEQuantumEspressoCalculator):
            if not getattr(calculator, "executable_version", None) or not isinstance(getattr(calculator, "executable_version"), str):
                raise QEAuditError("Injected research QE calculators must expose a non-empty executable_version")
            if config.qe_executable_version and getattr(calculator, "executable_version") != config.qe_executable_version:
                raise QEAuditError(f"Injected research QE calculator executable version mismatch: expected {config.qe_executable_version}, got {getattr(calculator, 'executable_version')}")
            calc_sha = getattr(calculator, "qe_executable_sha256", getattr(calculator, "executable_sha256", None))
            if not calc_sha or not isinstance(calc_sha, str) or len(calc_sha) != 64 or not all(c in "0123456789abcdefABCDEF" for c in calc_sha):
                raise QEAuditError("Injected research QE calculators must expose a valid 64-character hexadecimal qe_executable_sha256")
            calc_sha = calc_sha.lower()
            if config.qe_executable_sha256 and calc_sha != config.qe_executable_sha256.lower():
                raise QEAuditError(f"Injected research QE calculator executable SHA256 mismatch: expected {config.qe_executable_sha256.lower()}, got {calc_sha}")
        self.calculator = calculator

    def _validate_result_provenance(self, result: QEResultRecord) -> None:
        _validate_qe_result_record(result)
        if not bool(self.config.mock_execution):
            if (
                not result.sssp_manifest_sha256
                or str(result.sssp_manifest_sha256).lower() != str(self.sssp_manifest_digest).lower()
            ):
                raise QEAuditError(
                    "QE result SSSP manifest SHA256 does not match the verified manifest"
                )
        if self.config.qe_executable_sha256:
            if (
                not result.qe_executable_sha256
                or str(result.qe_executable_sha256).lower() != str(self.config.qe_executable_sha256).lower()
            ):
                raise QEAuditError(
                    "QE result executable SHA256 does not match the configured digest"
                )

    def audit_candidate(self, candidate: QEAuditCandidate) -> QELocalDecompositionAuditResult:
        if not _finite_number(candidate.predicted_energy_above_hull_ev_per_atom):
            raise QEAuditError("Candidate predicted energy above hull must be finite")
        num_atoms = _validate_formula_structure(candidate.reduced_formula, candidate.structure)
        products = _product_schema(candidate)
        candidate_counts = _formula_counts(candidate.reduced_formula)
        product_counts: Dict[str, float] = {}
        for formula, coeff, _ in products:
            for element, count in _formula_counts(formula).items():
                product_counts[element] = product_counts.get(element, 0.0) + coeff * count
        balanced = validate_atom_balanced_reaction(candidate.reduced_formula, [{"formula": f, "coefficient": c} for f, c, _ in products])
        equation = f"{candidate.reduced_formula} -> " + " + ".join(f"{coeff:g} {formula}" for formula, coeff, _ in products)
        if not balanced:
            return QELocalDecompositionAuditResult(candidate.candidate_id, candidate.target_task, candidate.condition, candidate.seed, candidate.reduced_formula, num_atoms, QECalculationStatus.INVALID_INPUT.value, None, len(products), False, candidate.predicted_energy_above_hull_ev_per_atom, None, None, None, equation, reaction_balanced=False, status="INVALID_UNBALANCED_REACTION")

        candidate_dir = self.calc_root / f"candidate_{candidate.candidate_id}"
        cand_res = _run_calc(self.calculator, candidate.reduced_formula, candidate.structure, self.config, candidate_dir, is_candidate=True)
        self._validate_result_provenance(cand_res)
        phase_results: List[QEResultRecord] = []
        phase_provenance: List[Dict[str, Any]] = []
        all_converged = cand_res.status == QECalculationStatus.CONVERGED.value
        reference_total = 0.0
        for index, (formula, coeff, structure) in enumerate(products):
            directory = self.calc_root / f"phase_{index}_{hashlib.sha256(formula.encode()).hexdigest()[:12]}"
            phase_res = _run_calc(self.calculator, formula, structure, self.config, directory, is_candidate=False)
            self._validate_result_provenance(phase_res)
            phase_results.append(phase_res)
            phase_record = phase_res.to_dict()
            phase_record.update({
                "phase_index": index,
                "decomposition_coefficient": coeff,
            })
            phase_provenance.append(phase_record)
            if phase_res.status != QECalculationStatus.CONVERGED.value or phase_res.total_energy_ev is None:
                all_converged = False
            else:
                reference_total += coeff * phase_res.total_energy_ev
        margin = None
        sign = None
        difference = None
        if cand_res.status == QECalculationStatus.CONVERGED.value and cand_res.total_energy_ev is not None and all_converged:
            margin = compute_local_decomposition_margin(cand_res.total_energy_ev, num_atoms, phase_results, [{"formula": f, "coefficient": c} for f, c, _ in products])
            sign = (margin <= 0.03) == (candidate.predicted_energy_above_hull_ev_per_atom <= 0.03)
            difference = margin - candidate.predicted_energy_above_hull_ev_per_atom
        status = "VALIDATED" if margin is not None else "INCONCLUSIVE_CALCULATION_FAILURE"
        return QELocalDecompositionAuditResult(
            candidate.candidate_id, candidate.target_task, candidate.condition,
            candidate.seed, candidate.reduced_formula, num_atoms,
            cand_res.status, cand_res.energy_per_atom_ev, len(products),
            all_converged, candidate.predicted_energy_above_hull_ev_per_atom,
            margin, sign, difference, equation,
            phase_provenance, reaction_balanced=True,
            status=status, candidate_input_hash=cand_res.input_hash,
            candidate_result_hash=cand_res.result_hash,
            candidate_output_hash=cand_res.output_hash,
            qe_executable=cand_res.executable,
            qe_executable_version=cand_res.executable_version,
            qe_executable_sha256=cand_res.qe_executable_sha256,
            sssp_manifest_sha256=cand_res.sssp_manifest_sha256,
            candidate_provenance={
                "candidate_id": candidate.candidate_id,
                "target_task": candidate.target_task,
                "condition": candidate.condition,
                "seed": candidate.seed,
                "selection_rank": candidate.selection_rank,
                "selection_reason": candidate.selection_reason,
                "result": cand_res.to_dict(),
            },
        )

    def run_full_audit(self, candidates: Sequence[QEAuditCandidate]) -> List[QELocalDecompositionAuditResult]:
        # Materialize once: callers may pass an AuditSelection carrying its
        # insufficiency metadata, and the exact same selection must be bound
        # to the emitted audit manifest.
        selection_has_insufficiency = hasattr(candidates, "insufficiency")
        selection_insufficiency = getattr(candidates, "insufficiency", None)
        candidates = list(candidates)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        selection_path = self.output_dir / "selection.json"
        selection_payload = {
            "candidates": [candidate.to_dict() for candidate in candidates],
            "insufficiency": selection_insufficiency,
        }
        bound_selection_insufficiency = selection_insufficiency
        # A selection node normally writes this artifact before execution. A
        # direct runner invocation still gets a complete, atomically-written
        # selection artifact; an existing artifact is never overwritten.
        if selection_path.exists():
            if not selection_path.is_file():
                raise QEAuditError("QE selection artifact is not a regular file")
            try:
                existing_selection = json.loads(selection_path.read_text(encoding="utf-8"))
                _canonical_json_bytes(existing_selection)
            except (OSError, json.JSONDecodeError, QEAuditError) as exc:
                raise QEAuditError("QE selection artifact is invalid or contains non-finite values") from exc
            if not isinstance(existing_selection, Mapping) or not isinstance(existing_selection.get("candidates"), list):
                raise QEAuditError("QE selection artifact must contain a candidates list")
            expected_selection_rows = [candidate.to_dict() for candidate in candidates]
            try:
                selection_matches = _canonical_json_bytes(existing_selection["candidates"]) == _canonical_json_bytes(expected_selection_rows)
            except QEAuditError as exc:
                raise QEAuditError("QE selection artifact contains non-canonical candidate values") from exc
            if not selection_matches:
                raise QEAuditError(
                    "QE selection artifact candidates do not match the candidates supplied for audit"
                )
            existing_has_insufficiency = "insufficiency" in existing_selection
            existing_insufficiency = existing_selection.get("insufficiency")
            if selection_has_insufficiency and existing_insufficiency != selection_insufficiency:
                raise QEAuditError(
                    "QE selection artifact insufficiency does not match the supplied selection"
                )
            # A plain list has no insufficiency metadata of its own (as in the
            # CLI execution node), so an explicit artifact value is the
            # authoritative value in that case. AuditSelection callers must
            # agree explicitly with the artifact.
            if not selection_has_insufficiency and existing_has_insufficiency:
                bound_selection_insufficiency = existing_insufficiency
        else:
            _atomic_write_json(selection_path, selection_payload, indent=2)

        results = [self.audit_candidate(candidate) for candidate in candidates]
        results_path = self.output_dir / "results.csv"
        fields = ["candidate_id", "target_task", "condition", "seed", "formula", "num_atoms", "candidate_status", "candidate_energy_per_atom_ev", "competing_phases_count", "competing_phases_all_converged", "chgnet_predicted_hull_distance_ev_per_atom", "dft_local_decomposition_margin_ev_per_atom", "sign_agreement", "margin_difference_ev_per_atom", "reaction_equation", "reaction_balanced", "status", "candidate_input_hash", "candidate_result_hash", "candidate_output_hash", "qe_executable", "qe_executable_version", "qe_executable_sha256", "sssp_manifest_sha256", "candidate_provenance", "participating_phase_results"]
        # csv.writer is used only to build a complete in-memory document;
        # replacing the destination happens exactly once after all rows have
        # passed strict JSON/finite-value validation.
        import io
        csv_buffer = io.StringIO(newline="")
        writer = csv.DictWriter(csv_buffer, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = result.to_dict()
            _canonical_json_bytes(row)
            csv_row = {key: row.get(key, "") for key in fields}
            csv_row["candidate_provenance"] = json.dumps(
                row.get("candidate_provenance", {}), sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            )
            csv_row["participating_phase_results"] = json.dumps(
                row.get("participating_phase_results", []), sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            )
            writer.writerow(csv_row)
        _atomic_write_text(results_path, csv_buffer.getvalue())
        canonical_results = _canonical_json_bytes([r.to_dict() for r in results])
        selection_digest = _file_sha256(selection_path)
        results_digest = _file_sha256(results_path)
        metadata = {
            "run_mode": self.run_mode,
            "mock_execution": bool(self.config.mock_execution),
            "selection_count": len(candidates),
            "selection_insufficiency": bound_selection_insufficiency,
            "config": _config_dict(self.config),
            "artifacts": {
                "selection.json": selection_digest,
                "results.csv": results_digest,
                "canonical_results_sha256": hashlib.sha256(canonical_results).hexdigest(),
            },
            "result_provenance": [r.to_dict() for r in results],
        }
        manifest_path = self.output_dir / "audit_manifest.json"
        _atomic_write_json(manifest_path, metadata, indent=2)
        return results


__all__ = [
    "QECalculationStatus", "QEAuditError", "QESelectionInsufficiency", "AuditSelection",
    "QEAuditCandidate", "QEResultRecord", "QELocalDecompositionAuditResult",
    "select_audit_candidates", "generate_kpoints_grid", "FakeQECalculator",
    "ASEQuantumEspressoCalculator", "ASEQECalculator", "QECalculator", "QEAuditRunner",
    "validate_atom_balanced_reaction", "compute_local_decomposition_margin",
]

# Compatibility aliases used by downstream audit scripts.
ASEQECalculator = ASEQuantumEspressoCalculator
QECalculator = ASEQuantumEspressoCalculator
