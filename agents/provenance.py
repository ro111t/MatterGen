"""
provenance.py — Comprehensive candidate lifecycle and experiment reproducibility tracking for AMDA.

Provides:
  - CandidateStatus (Enum): 7-stage candidate lifecycle states
  - SoftwareEnvironment: Captures host environment, git commit SHA, package versions, and hardware
  - CandidateRecord: Rich versioned record schema ("1.0.0") tracking every candidate from birth to disposition
  - RunManifest: Campaign run manifest ("1.0.0") capturing configuration, seeds, strategies, and environment
  - ProvenanceTracker: Orchestrator for registering lifecycle events, persisting standardized .cif structures,
    streaming real-time event logs (provenance.jsonl), and exporting consolidated JSON and CSV records.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Optional chemical and computational libraries
try:
    from pymatgen.core import Composition, Structure
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False

try:
    from ase import Atoms
    from ase.io import write as ase_write
    HAS_ASE = True
except ImportError:
    HAS_ASE = False


class CandidateStatus(str, Enum):
    """Explicit candidate lifecycle states."""
    GENERATED = "generated"
    SCREENED = "screened"
    VALIDATED = "validated"
    SYNTHESIS_ASSESSED = "synthesis_assessed"
    RANKED = "ranked"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


def _get_utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def extract_candidate_id(struct: Any, fallback_idx: Optional[int] = None) -> str:
    """Extract canonical candidate ID (MAT-xxxxxx) from pymatgen Structure, dict stub, or Atoms."""
    if isinstance(struct, dict):
        cid = struct.get("candidate_id") or struct.get("generation_id") or struct.get("structure_id")
        if cid:
            return str(cid)
    if hasattr(struct, "_candidate_id") and getattr(struct, "_candidate_id"):
        return str(getattr(struct, "_candidate_id"))
    if hasattr(struct, "properties") and isinstance(struct.properties, dict):
        cid = struct.properties.get("_candidate_id") or struct.properties.get("candidate_id") or struct.properties.get("structure_id")
        if cid:
            return str(cid)
    if hasattr(struct, "info") and isinstance(struct.info, dict):
        cid = struct.info.get("_candidate_id") or struct.info.get("candidate_id") or struct.info.get("structure_id")
        if cid:
            return str(cid)
    if fallback_idx is not None:
        return f"MAT-{fallback_idx + 1:06d}"
    return "MAT-UNKNOWN"


def _parse_formula_elements(formula: str) -> List[str]:
    """Extract constituent chemical element symbols from formula string (e.g. 'Li3PS4' -> ['Li', 'P', 'S'])."""
    if not formula or formula == "Unknown":
        return []
    # Match standard IUPAC element symbols (e.g., Li, Na, P, S, Cl, La, Zr)
    matches = re.findall(r'([A-Z][a-z]*)', formula)
    return sorted(list(set(matches)))


def extract_formula_and_elements(struct: Any) -> Tuple[str, List[str], str]:
    """
    Extract reduced formula, element list, and chemical system string.
    Returns: (formula, elements_list, chemical_system)
    """
    if HAS_PYMATGEN and isinstance(struct, Structure):
        formula = struct.composition.reduced_formula
        elements = sorted(list({el.symbol for el in struct.composition.elements}))
        chem_sys = "-".join(elements)
        return formula, elements, chem_sys

    if isinstance(struct, dict):
        formula = struct.get("composition") or struct.get("formula") or "Unknown"
        elements = struct.get("elements", [])
        if not elements and formula != "Unknown":
            if HAS_PYMATGEN:
                try:
                    comp = Composition(formula)
                    elements = sorted(list({el.symbol for el in comp.elements}))
                    formula = comp.reduced_formula
                except Exception:
                    elements = _parse_formula_elements(str(formula))
            else:
                elements = _parse_formula_elements(str(formula))
        if not elements and isinstance(struct.get("positions"), list):
            site_elements = [p.get("element") for p in struct.get("positions", []) if isinstance(p, dict) and "element" in p]
            if site_elements:
                elements = sorted(list(set(site_elements)))
        elements = sorted(list(set(elements)))
        chem_sys = "-".join(elements) if elements else "Unknown"
        return str(formula), elements, chem_sys

    if HAS_ASE and isinstance(struct, Atoms):
        formula = struct.get_chemical_formula(mode="reduced")
        elements = sorted(list(set(struct.get_chemical_symbols())))
        chem_sys = "-".join(elements)
        return formula, elements, chem_sys

    if isinstance(struct, str):
        formula = struct
        elements = _parse_formula_elements(struct)
        chem_sys = "-".join(elements) if elements else struct
        return formula, elements, chem_sys

    comp_attr = getattr(struct, "composition", None)
    if comp_attr is not None:
        if hasattr(comp_attr, "reduced_formula"):
            formula = comp_attr.reduced_formula
            elements = sorted(list({el.symbol for el in comp_attr.elements}))
            chem_sys = "-".join(elements)
            return formula, elements, chem_sys
        formula = str(comp_attr)
        elements = _parse_formula_elements(formula)
        chem_sys = "-".join(elements) if elements else formula
        return formula, elements, chem_sys

    return "Unknown", [], "Unknown"


@dataclass
class SoftwareEnvironment:
    """Captures runtime environment and dependency versions for full reproducibility."""
    python_version: str
    os_name: str
    os_release: str
    os_version: str
    platform_system: str
    git_commit_sha: Optional[str]
    packages: Dict[str, Optional[str]]
    cuda_available: bool
    cuda_device_count: int
    gpu_device_name: Optional[str]

    @classmethod
    def capture(cls) -> SoftwareEnvironment:
        """Safely inspect environment metadata without throwing on missing packages."""
        # Git commit SHA
        git_sha = None
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False
            )
            if res.returncode == 0:
                git_sha = res.stdout.strip()
        except Exception:
            git_sha = None

        # Package versions
        tracked_pkgs = [
            "pymatgen", "chgnet", "ase", "torch", "mattergen",
            "numpy", "scipy", "pandas", "pytest"
        ]
        pkg_versions: Dict[str, Optional[str]] = {}
        for pkg in tracked_pkgs:
            try:
                import importlib.metadata
                pkg_versions[pkg] = importlib.metadata.version(pkg)
            except Exception:
                try:
                    mod = sys.modules.get(pkg) or __import__(pkg)
                    pkg_versions[pkg] = getattr(mod, "__version__", "unknown")
                except Exception:
                    pkg_versions[pkg] = None

        # CUDA / GPU metadata
        cuda_avail = False
        cuda_count = 0
        gpu_name = None
        try:
            import torch
            cuda_avail = torch.cuda.is_available()
            if cuda_avail:
                cuda_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if cuda_count > 0 else None
        except Exception:
            pass

        return cls(
            python_version=platform.python_version(),
            os_name=platform.system(),
            os_release=platform.release(),
            os_version=platform.version(),
            platform_system=platform.platform(),
            git_commit_sha=git_sha,
            packages=pkg_versions,
            cuda_available=cuda_avail,
            cuda_device_count=cuda_count,
            gpu_device_name=gpu_name,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateRecord:
    """
    Versioned record schema representing the complete lifecycle of a single material candidate.
    Schema Version: 1.0.0
    """
    schema_version: str = "1.0.0"

    # Identity
    candidate_id: str = ""
    campaign_id: str = ""
    iteration: int = 0
    created_at_iso: str = field(default_factory=_get_utc_now_iso)

    # Chemistry & Structure
    composition: str = "Unknown"
    chemical_system: str = "Unknown"
    elements: List[str] = field(default_factory=list)
    num_elements: int = 0
    structure_path: Optional[str] = None
    structure_hash: Optional[str] = None

    # Generation
    generation_backend: str = "stub"
    model_name_or_path: Optional[str] = None
    checkpoint: Optional[str] = None
    generation_seed: Optional[int] = None
    generation_parameters: Dict[str, Any] = field(default_factory=dict)
    target_elements: List[str] = field(default_factory=list)
    generation_timestamp_iso: Optional[str] = None

    # Screening
    screening_backend: Optional[str] = None
    screening_predictions: Dict[str, float] = field(default_factory=dict)
    screening_score: Optional[float] = None
    screening_score_components: Dict[str, float] = field(default_factory=dict)
    passes_screening_filters: Optional[bool] = None
    screening_filter_reasons: List[str] = field(default_factory=list)
    screening_rank: Optional[int] = None
    screening_timestamp_iso: Optional[str] = None

    # Validation
    validation_calculator: Optional[str] = None
    validation_converged: Optional[bool] = None
    validation_properties: Dict[str, float] = field(default_factory=dict)
    validation_cost_hours: Optional[float] = None
    validation_error_message: Optional[str] = None
    validation_timestamp_iso: Optional[str] = None

    # Synthesis Feasibility
    synthesis_mode: Optional[str] = None
    synthesis_feasible: Optional[bool] = None
    synthesis_feasibility_score: Optional[float] = None
    synthesis_difficulty_score: Optional[float] = None
    synthesis_estimated_cost: Optional[float] = None
    synthesis_route: Optional[str] = None
    synthesis_route_reason: Optional[str] = None
    synthesis_similar_known_phases: List[str] = field(default_factory=list)
    synthesis_warnings: List[str] = field(default_factory=list)
    synthesis_timestamp_iso: Optional[str] = None

    # Decision & Scientific Memory
    status: str = CandidateStatus.GENERATED.value
    rejection_stage: Optional[str] = None
    rejection_reason: Optional[str] = None
    ranking_score: Optional[float] = None
    iteration_rank: Optional[int] = None
    stored_in_memory: bool = False
    strategy_influence: Optional[str] = None
    decision_timestamp_iso: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Export as structured dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CandidateRecord:
        """Construct CandidateRecord from dictionary."""
        known_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Flatten nested structures into tabular format suitable for Pandas DataFrame or CSV export.
        """
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "campaign_id": self.campaign_id,
            "iteration": self.iteration,
            "created_at_iso": self.created_at_iso,
            "status": self.status,
            "composition": self.composition,
            "chemical_system": self.chemical_system,
            "elements": ";".join(self.elements),
            "num_elements": self.num_elements,
            "structure_path": self.structure_path or "",
            "structure_hash": self.structure_hash or "",
            # Generation
            "generation_backend": self.generation_backend,
            "model_name_or_path": self.model_name_or_path or "",
            "checkpoint": self.checkpoint or "",
            "generation_seed": self.generation_seed if self.generation_seed is not None else "",
            "target_elements": ";".join(self.target_elements),
            "generation_timestamp_iso": self.generation_timestamp_iso or "",
            # Screening
            "screening_backend": self.screening_backend or "",
            "screening_score": self.screening_score if self.screening_score is not None else "",
            "passes_screening_filters": self.passes_screening_filters if self.passes_screening_filters is not None else "",
            "screening_filter_reasons": "; ".join(self.screening_filter_reasons),
            "screening_rank": self.screening_rank if self.screening_rank is not None else "",
            "screening_formation_energy": self.screening_predictions.get("formation_energy", ""),
            "screening_forces": self.screening_predictions.get("forces", ""),
            "screening_stress": self.screening_predictions.get("stress", ""),
            "screening_stability": self.screening_predictions.get("stability", ""),
            "screening_timestamp_iso": self.screening_timestamp_iso or "",
            # Validation
            "validation_calculator": self.validation_calculator or "",
            "validation_converged": self.validation_converged if self.validation_converged is not None else "",
            "validation_cost_hours": self.validation_cost_hours if self.validation_cost_hours is not None else "",
            "validation_energy_per_atom": self.validation_properties.get("energy_per_atom", ""),
            "validation_stability": self.validation_properties.get("stability", ""),
            "validation_band_gap": self.validation_properties.get("band_gap", ""),
            "validation_bulk_modulus": self.validation_properties.get("bulk_modulus", ""),
            "validation_error_message": self.validation_error_message or "",
            "validation_timestamp_iso": self.validation_timestamp_iso or "",
            # Synthesis
            "synthesis_mode": self.synthesis_mode or "",
            "synthesis_feasible": self.synthesis_feasible if self.synthesis_feasible is not None else "",
            "synthesis_feasibility_score": self.synthesis_feasibility_score if self.synthesis_feasibility_score is not None else "",
            "synthesis_difficulty_score": self.synthesis_difficulty_score if self.synthesis_difficulty_score is not None else "",
            "synthesis_estimated_cost": self.synthesis_estimated_cost if self.synthesis_estimated_cost is not None else "",
            "synthesis_route": self.synthesis_route or "",
            "synthesis_route_reason": self.synthesis_route_reason or "",
            "synthesis_warnings": "; ".join(self.synthesis_warnings),
            "synthesis_timestamp_iso": self.synthesis_timestamp_iso or "",
            # Decision & Memory
            "rejection_stage": self.rejection_stage or "",
            "rejection_reason": self.rejection_reason or "",
            "ranking_score": self.ranking_score if self.ranking_score is not None else "",
            "iteration_rank": self.iteration_rank if self.iteration_rank is not None else "",
            "stored_in_memory": self.stored_in_memory,
            "strategy_influence": self.strategy_influence or "",
            "decision_timestamp_iso": self.decision_timestamp_iso or "",
        }


@dataclass
class RunManifest:
    """
    Campaign execution manifest ("1.0.0").
    Records startup configuration, environment, seeds, and planning strategies.
    """
    schema_version: str = "1.0.0"
    campaign_id: str = ""
    campaign_name: str = ""
    domain: str = ""
    git_commit_sha: Optional[str] = None
    environment: Dict[str, Any] = field(default_factory=dict)
    master_seed: int = 42
    iteration_seeds: List[int] = field(default_factory=list)
    objective: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    strategies: List[Dict[str, Any]] = field(default_factory=list)
    start_time_iso: str = field(default_factory=_get_utc_now_iso)
    end_time_iso: Optional[str] = None
    elapsed_time_seconds: Optional[float] = None
    status: str = "running"
    total_candidates_generated: int = 0
    total_candidates_accepted: int = 0
    total_candidates_rejected: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RunManifest:
        known_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path

    def is_consistent_with(self, other: RunManifest) -> Tuple[bool, List[str]]:
        """Verify configuration consistency between an original and a reproduced run."""
        discrepancies = []
        if self.master_seed != other.master_seed:
            discrepancies.append(f"Master seed mismatch: {self.master_seed} != {other.master_seed}")
        if self.domain != other.domain:
            discrepancies.append(f"Domain mismatch: {self.domain} != {other.domain}")
        if self.iteration_seeds != other.iteration_seeds:
            discrepancies.append(f"Iteration seeds mismatch: {self.iteration_seeds} != {other.iteration_seeds}")
        return (len(discrepancies) == 0, discrepancies)


class ProvenanceTracker:
    """
    Central provenance engine managing candidate lifecycle transitions, structure serialization,
    real-time event streaming (JSONL), manifest generation, and CSV exports.
    """

    def __init__(
        self,
        campaign_id: str,
        campaign_name: str,
        domain: str,
        output_dir: Union[str, Path],
        master_seed: int = 42,
        config: Optional[Dict[str, Any]] = None,
        objective: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ):
        self.campaign_id = campaign_id
        self.campaign_name = campaign_name
        self.domain = domain
        self.output_dir = Path(output_dir).resolve()
        self.master_seed = master_seed
        self.config_dict = config or {}
        self.objective_dict = objective or {}
        self.constraints_dict = constraints or {}

        self.structures_dir = self.output_dir / "structures"
        self.structures_dir.mkdir(parents=True, exist_ok=True)

        self.jsonl_path = self.output_dir / "provenance.jsonl"
        self.manifest_path = self.output_dir / "manifest.json"
        self.campaign_json_path = self.output_dir / "campaign_provenance.json"
        self.csv_path = self.output_dir / "candidates_provenance.csv"

        self.records: Dict[str, CandidateRecord] = {}
        self.environment = SoftwareEnvironment.capture()
        self.manifest = RunManifest(
            campaign_id=self.campaign_id,
            campaign_name=self.campaign_name,
            domain=self.domain,
            git_commit_sha=self.environment.git_commit_sha,
            environment=self.environment.to_dict(),
            master_seed=self.master_seed,
            iteration_seeds=[],
            objective=self.objective_dict,
            constraints=self.constraints_dict,
            config=self.config_dict,
            strategies=[],
            start_time_iso=_get_utc_now_iso(),
            status="running",
        )

    def write_manifest(self) -> Path:
        """Write current manifest state to manifest.json."""
        return self.manifest.save(self.manifest_path)

    def record_strategy(self, iteration: int, strategy: Dict[str, Any]) -> None:
        """Record planned strategy for the iteration to ensure deterministic replay."""
        clean_strat = {
            "iteration": iteration,
            "elements": strategy.get("elements", []),
            "num_candidates": strategy.get("num_candidates", 15),
            "screening_criteria": strategy.get("screening_criteria", {}),
            "diversity_weight": strategy.get("diversity_weight", 0.3),
            "rationale": strategy.get("rationale", ""),
            "hypothesis": strategy.get("hypothesis", ""),
        }
        self.manifest.strategies.append(clean_strat)
        self.write_manifest()

    # -------------------------------------------------------------------------
    # Structure Serialization & Hashing
    # -------------------------------------------------------------------------

    def save_structure_file(self, candidate_id: str, struct: Any) -> Tuple[str, str]:
        """
        Persist a crystal structure into a standardized .cif file under structures/
        and return its relative path and SHA256 checksum.
        """
        cif_filename = f"{candidate_id}.cif"
        cif_path = self.structures_dir / cif_filename
        relative_path = f"structures/{cif_filename}"

        cif_content = self._serialize_to_cif(candidate_id, struct)
        with open(cif_path, "w", encoding="utf-8") as f:
            f.write(cif_content)

        sha256_hash = hashlib.sha256(cif_content.encode("utf-8")).hexdigest()
        return relative_path, sha256_hash

    def _serialize_to_cif(self, candidate_id: str, struct: Any) -> str:
        """Polymorphic CIF serializer handling Structure, Atoms, and stub dicts."""
        if HAS_PYMATGEN and isinstance(struct, Structure):
            try:
                from pymatgen.io.cif import CifWriter
                return str(CifWriter(struct))
            except Exception:
                try:
                    return struct.to(fmt="cif")
                except Exception:
                    pass

        if HAS_ASE and isinstance(struct, Atoms):
            try:
                import io
                buf = io.StringIO()
                ase_write(buf, struct, format="cif")
                return buf.getvalue()
            except Exception:
                pass

        # Fallback standard CIF format for stub dictionary or offline structures
        formula, elements, _ = extract_formula_and_elements(struct)
        lattice_vectors = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
        positions = [[0.0, 0.0, 0.0]]
        site_elements = elements if elements else ["Li"]

        if isinstance(struct, dict):
            if "lattice" in struct and isinstance(struct["lattice"], list):
                lattice_vectors = struct["lattice"]
            if "positions" in struct and isinstance(struct["positions"], list):
                positions = struct["positions"]

        # Compute cell dimensions: a, b, c lengths
        a_vec = lattice_vectors[0] if len(lattice_vectors) > 0 else [5.0, 0.0, 0.0]
        b_vec = lattice_vectors[1] if len(lattice_vectors) > 1 else [0.0, 5.0, 0.0]
        c_vec = lattice_vectors[2] if len(lattice_vectors) > 2 else [0.0, 0.0, 5.0]

        a_len = math.sqrt(sum(v**2 for v in a_vec)) or 5.0
        b_len = math.sqrt(sum(v**2 for v in b_vec)) or 5.0
        c_len = math.sqrt(sum(v**2 for v in c_vec)) or 5.0

        cif_lines = [
            f"data_{candidate_id}",
            f"_chemical_formula_sum '{formula}'",
            f"_cell_length_a {a_len:.5f}",
            f"_cell_length_b {b_len:.5f}",
            f"_cell_length_c {c_len:.5f}",
            "_cell_angle_alpha 90.00000",
            "_cell_angle_beta 90.00000",
            "_cell_angle_gamma 90.00000",
            "_symmetry_space_group_name_H-M 'P 1'",
            "_symmetry_Int_Tables_number 1",
            "loop_",
            "_atom_site_label",
            "_atom_site_type_symbol",
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
        ]

        for i, pos in enumerate(positions):
            el_sym = site_elements[i % len(site_elements)]
            x, y, z = (pos[0], pos[1], pos[2]) if isinstance(pos, (list, tuple)) and len(pos) >= 3 else (0.0, 0.0, 0.0)
            # Ensure coordinates are in fractional [0, 1) range
            if abs(x) > 1.0 or abs(y) > 1.0 or abs(z) > 1.0:
                fx = (x / a_len) % 1.0
                fy = (y / b_len) % 1.0
                fz = (z / c_len) % 1.0
            else:
                fx = x % 1.0
                fy = y % 1.0
                fz = z % 1.0
            cif_lines.append(f"  {el_sym}{i+1} {el_sym} {fx:.5f} {fy:.5f} {fz:.5f}")

        cif_lines.append("")
        return "\n".join(cif_lines)

    # -------------------------------------------------------------------------
    # Lifecycle Event Recording
    # -------------------------------------------------------------------------

    def register_generation(
        self,
        candidates: List[Any],
        iteration: int,
        backend: str,
        seed: int,
        target_elements: List[str],
        parameters: Optional[Dict[str, Any]] = None,
        model_name_or_path: Optional[str] = None,
        checkpoint: Optional[str] = None,
    ) -> List[str]:
        """Stage 1: Register newly generated candidates at birth and save CIF structures."""
        if seed not in self.manifest.iteration_seeds:
            self.manifest.iteration_seeds.append(seed)

        registered_ids = []
        now_iso = _get_utc_now_iso()

        for idx, struct in enumerate(candidates):
            cand_id = extract_candidate_id(struct, fallback_idx=len(self.records))
            formula, elements, chem_sys = extract_formula_and_elements(struct)
            struct_path, struct_hash = self.save_structure_file(cand_id, struct)

            record = CandidateRecord(
                candidate_id=cand_id,
                campaign_id=self.campaign_id,
                iteration=iteration,
                created_at_iso=now_iso,
                composition=formula,
                chemical_system=chem_sys,
                elements=elements,
                num_elements=len(elements),
                structure_path=struct_path,
                structure_hash=struct_hash,
                generation_backend=backend,
                model_name_or_path=model_name_or_path,
                checkpoint=checkpoint,
                generation_seed=seed,
                generation_parameters=parameters or {},
                target_elements=target_elements,
                generation_timestamp_iso=now_iso,
                status=CandidateStatus.GENERATED.value,
            )

            self.records[cand_id] = record
            registered_ids.append(cand_id)
            self.stream_candidate_event(cand_id)

        self.manifest.total_candidates_generated = len(self.records)
        return registered_ids

    def record_screening(
        self,
        screening_results: List[Tuple[Any, Any]],
        criteria: Dict[str, Any],
        iteration: int,
        backend: str = "chgnet",
    ) -> None:
        """Stage 2: Record ML screening outputs, filters, pass/fail status, and numerical rejection reasons."""
        now_iso = _get_utc_now_iso()

        for rank_idx, (struct, res) in enumerate(screening_results, 1):
            cand_id = extract_candidate_id(struct) if struct is not None else (getattr(res, "structure_id", None) or f"MAT-{rank_idx:06d}")
            record = self.records.get(cand_id)
            if not record:
                res_id = getattr(res, "structure_id", None)
                if res_id and res_id in self.records:
                    cand_id = res_id
                    record = self.records[cand_id]
            if not record:
                formula, elements, chem_sys = extract_formula_and_elements(struct)
                struct_path, struct_hash = self.save_structure_file(cand_id, struct)
                record = CandidateRecord(
                    candidate_id=cand_id,
                    campaign_id=self.campaign_id,
                    iteration=iteration,
                    composition=formula,
                    chemical_system=chem_sys,
                    elements=elements,
                    num_elements=len(elements),
                    structure_path=struct_path,
                    structure_hash=struct_hash,
                )
                self.records[cand_id] = record

            record.screening_backend = backend
            record.screening_predictions = getattr(res, "predictions", {}) or {}
            record.screening_score = getattr(res, "score", None)
            record.screening_score_components = getattr(res, "score_components", {}) or {}
            record.passes_screening_filters = getattr(res, "passes_filters", True)
            record.screening_filter_reasons = getattr(res, "filter_reasons", []) or []
            record.screening_rank = getattr(res, "rank", rank_idx)
            record.screening_timestamp_iso = now_iso

            if not record.passes_screening_filters:
                record.status = CandidateStatus.REJECTED.value
                record.rejection_stage = "screening"
                reasons_str = "; ".join(record.screening_filter_reasons) if record.screening_filter_reasons else "Failed screening criteria"
                record.rejection_reason = f"Screening filter failed: {reasons_str}"
                record.decision_timestamp_iso = now_iso
            elif record.status != CandidateStatus.REJECTED.value:
                record.status = CandidateStatus.SCREENED.value

            self.stream_candidate_event(cand_id)

    def record_validation(
        self,
        validation_results: List[Any],
        iteration: int,
    ) -> None:
        """Stage 3: Record high-fidelity (DFT or mock DFT) validation results and convergence."""
        now_iso = _get_utc_now_iso()

        for v in validation_results:
            struct = getattr(v, "structure", None)
            cand_id = extract_candidate_id(struct) if struct is not None else getattr(v, "structure_id", None)
            record = self.records.get(cand_id)
            if not record:
                res_id = getattr(v, "structure_id", None)
                if res_id and res_id in self.records:
                    cand_id = res_id
                    record = self.records[cand_id]
            if not record:
                continue

            record.validation_calculator = getattr(v, "calculator", "mock")
            record.validation_converged = getattr(v, "converged", False)
            record.validation_properties = getattr(v, "properties", {}) or {}
            record.validation_cost_hours = getattr(v, "cost_hours", 0.0)
            record.validation_error_message = getattr(v, "error_message", "") or None
            record.validation_timestamp_iso = now_iso

            if not record.validation_converged:
                record.status = CandidateStatus.REJECTED.value
                record.rejection_stage = "validation"
                err = record.validation_error_message or "Calculation failed to converge"
                record.rejection_reason = f"Validation non-convergence: {err}"
                record.decision_timestamp_iso = now_iso
            elif record.status != CandidateStatus.REJECTED.value:
                record.status = CandidateStatus.VALIDATED.value

            self.stream_candidate_event(cand_id)

    def record_synthesis(
        self,
        synthesis_results: List[Any],
        iteration: int,
        mode: str = "mock",
    ) -> None:
        """Stage 4: Record synthesis feasibility assessment results."""
        now_iso = _get_utc_now_iso()

        for s in synthesis_results:
            cand_id = getattr(s, "structure_id", None)
            record = self.records.get(cand_id)
            if not record:
                continue

            record.synthesis_mode = mode
            record.synthesis_feasible = getattr(s, "feasible", False)
            record.synthesis_feasibility_score = getattr(s, "feasibility_score", 0.0)
            record.synthesis_difficulty_score = getattr(s, "difficulty_score", 0.0)
            record.synthesis_estimated_cost = getattr(s, "estimated_cost", 0.0)
            record.synthesis_route = getattr(s, "synthesis_route", "unknown")
            record.synthesis_route_reason = getattr(s, "route_reason", "")
            record.synthesis_similar_known_phases = getattr(s, "similar_known_phases", []) or []
            record.synthesis_warnings = getattr(s, "warnings", []) or []
            record.synthesis_timestamp_iso = now_iso

            if not record.synthesis_feasible:
                record.status = CandidateStatus.REJECTED.value
                record.rejection_stage = "synthesis"
                record.rejection_reason = (
                    f"Synthesis infeasible: score {record.synthesis_feasibility_score:.2f} "
                    f"(difficulty {record.synthesis_difficulty_score:.2f})"
                )
                record.decision_timestamp_iso = now_iso
            elif record.status != CandidateStatus.REJECTED.value:
                record.status = CandidateStatus.SYNTHESIS_ASSESSED.value

            self.stream_candidate_event(cand_id)

    def record_ranking(
        self,
        ranking_entries: List[Tuple[str, float, int]],
        iteration: int,
    ) -> None:
        """Stage 5: Record global multi-objective ranking within iteration batch."""
        now_iso = _get_utc_now_iso()
        for cand_id, score, rank in ranking_entries:
            record = self.records.get(cand_id)
            if not record:
                continue
            record.ranking_score = score
            record.iteration_rank = rank
            if record.status != CandidateStatus.REJECTED.value:
                record.status = CandidateStatus.RANKED.value
            self.stream_candidate_event(cand_id)

    def record_decision(
        self,
        candidate_id: str,
        status: Union[CandidateStatus, str],
        rejection_stage: Optional[str] = None,
        rejection_reason: Optional[str] = None,
        stored_in_memory: bool = False,
        strategy_influence: Optional[str] = None,
        ranking_score: Optional[float] = None,
        iteration_rank: Optional[int] = None,
    ) -> None:
        """Stage 6: Record final acceptance/rejection decision and memory storage influence."""
        record = self.records.get(candidate_id)
        if not record:
            return

        status_val = status.value if isinstance(status, CandidateStatus) else str(status)
        record.status = status_val
        if rejection_stage:
            record.rejection_stage = rejection_stage
        if rejection_reason:
            record.rejection_reason = rejection_reason
        if stored_in_memory:
            record.stored_in_memory = stored_in_memory
        if strategy_influence:
            record.strategy_influence = strategy_influence
        if ranking_score is not None:
            record.ranking_score = ranking_score
        if iteration_rank is not None:
            record.iteration_rank = iteration_rank

        record.decision_timestamp_iso = _get_utc_now_iso()
        self.stream_candidate_event(candidate_id)

    # -------------------------------------------------------------------------
    # Real-Time Event Streaming & File Exports
    # -------------------------------------------------------------------------

    def stream_candidate_event(self, candidate_id: str) -> None:
        """Append candidate event line to provenance.jsonl in real time."""
        record = self.records.get(candidate_id)
        if not record:
            return
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), default=str) + "\n")

    def save_campaign_provenance(self) -> Path:
        """Write consolidated campaign JSON provenance record."""
        payload = {
            "schema_version": "1.0.0",
            "campaign_id": self.campaign_id,
            "campaign_name": self.campaign_name,
            "domain": self.domain,
            "manifest": self.manifest.to_dict(),
            "summary_stats": self.get_summary_stats(),
            "candidates": [r.to_dict() for r in self.records.values()],
        }
        with open(self.campaign_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        return self.campaign_json_path

    def export_csv(self) -> Path:
        """Export flat tabular CSV representation of all candidates."""
        flat_records = [r.to_flat_dict() for r in self.records.values()]
        if not flat_records:
            fieldnames = list(CandidateRecord().to_flat_dict().keys())
        else:
            fieldnames = list(flat_records[0].keys())

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in flat_records:
                writer.writerow(row)
        return self.csv_path

    def finalize(self, status: str = "completed") -> Dict[str, Any]:
        """Finalize manifest and export all consolidated records."""
        now_iso = _get_utc_now_iso()
        self.manifest.end_time_iso = now_iso
        self.manifest.status = status

        start_dt = datetime.fromisoformat(self.manifest.start_time_iso)
        end_dt = datetime.fromisoformat(now_iso)
        self.manifest.elapsed_time_seconds = (end_dt - start_dt).total_seconds()

        accepted_count = sum(1 for r in self.records.values() if r.status == CandidateStatus.ACCEPTED.value)
        rejected_count = sum(1 for r in self.records.values() if r.status == CandidateStatus.REJECTED.value)
        self.manifest.total_candidates_accepted = accepted_count
        self.manifest.total_candidates_rejected = rejected_count

        self.write_manifest()
        self.save_campaign_provenance()
        self.export_csv()
        return self.get_summary_stats()

    # -------------------------------------------------------------------------
    # Source of Truth Statistics
    # -------------------------------------------------------------------------

    def get_summary_stats(self) -> Dict[str, Any]:
        """Compute accurate campaign-level summary metrics directly from CandidateRecords."""
        total_generated = len(self.records)
        all_records = list(self.records.values())

        screened = [r for r in all_records if r.screening_score is not None]
        passed_screening = [r for r in screened if r.passes_screening_filters is True]
        validated = [r for r in all_records if r.validation_converged is not None]
        converged = [r for r in validated if r.validation_converged is True]
        synthesis_assessed = [r for r in all_records if r.synthesis_feasible is not None]
        synthesis_feasible = [r for r in synthesis_assessed if r.synthesis_feasible is True]

        scores = [r.screening_score for r in screened if r.screening_score is not None]
        best_score = max(scores) if scores else 0.0

        val_stabilities = [
            r.validation_properties.get("stability")
            for r in validated
            if "stability" in r.validation_properties
        ]
        best_validated_stability = max(val_stabilities) if val_stabilities else 0.0

        synth_scores = [
            r.synthesis_feasibility_score
            for r in synthesis_assessed
            if r.synthesis_feasibility_score is not None
        ]
        best_synthesis_feasibility = max(synth_scores) if synth_scores else 0.0

        total_validation_cost = sum(
            r.validation_cost_hours for r in validated if r.validation_cost_hours is not None
        )

        backend_counts: Dict[str, int] = {}
        for r in all_records:
            b = r.generation_backend
            backend_counts[b] = backend_counts.get(b, 0) + 1

        return {
            "total_generated": total_generated,
            "total_screened": len(screened),
            "total_passed_screening": len(passed_screening),
            "overall_pass_rate": len(passed_screening) / total_generated if total_generated > 0 else 0.0,
            "total_validated": len(validated),
            "total_converged": len(converged),
            "convergence_rate": len(converged) / len(validated) if len(validated) > 0 else 0.0,
            "total_validation_cost_hours": total_validation_cost,
            "total_synthesis_assessed": len(synthesis_assessed),
            "total_synthesis_feasible": len(synthesis_feasible),
            "synthesis_feasibility_rate": len(synthesis_feasible) / len(synthesis_assessed) if len(synthesis_assessed) > 0 else 0.0,
            "best_score_ever": best_score,
            "best_validated_stability_ever": best_validated_stability,
            "best_synthesis_feasibility_ever": best_synthesis_feasibility,
            "generation_backend_counts": backend_counts,
        }
