"""
ValidationAgent: High-fidelity computational validation of screened candidates.

Supports multiple backends via ASE:
  - mock    : deterministic fake DFT for testing and offline development
  - ase     : any ASE-compatible calculator passed in at init
  - vasp    : VASP (requires license + VASP binary)
  - qe      : Quantum ESPRESSO (requires pw.x)
  - gpaw    : GPAW (requires gpaw install)

The agent runs structure relaxation and property calculations in parallel
where possible, and reports per-structure computational cost estimates.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import time
import warnings

import numpy as np

try:
    from ase import Atoms
    from ase.calculators.calculator import Calculator as ASECalculator
    from ase.optimize import BFGS
    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    warnings.warn("ASE not installed; ValidationAgent will use mock mode only.")

try:
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False


@dataclass
class ValidationResult:
    """Results from a validation calculation."""
    structure_id: str
    structure: Any
    calculator: str
    converged: bool
    properties: Dict[str, float] = field(default_factory=dict)
    cost_hours: float = 0.0
    error_message: str = ""


class ValidationAgent:
    """
    Validates material candidates using DFT or mock DFT backends.

    Args:
        calculator: Backend name ('mock', 'ase', 'vasp', 'qe', 'gpaw') or an
            already-initialized ASE Calculator instance.
        n_workers: Number of parallel validation jobs. Mock calculations are
            fast enough that parallelism is mainly useful for real DFT.
        properties_to_compute: List of property names to compute. Currently
            supported: 'energy', 'forces', 'stress', 'band_gap'.
        max_relax_steps: Maximum relaxation steps for structural relaxation.
        fmax: Force convergence threshold for relaxation (eV/Angstrom).
    """

    def __init__(
        self,
        calculator: Any = "mock",
        n_workers: int = 1,
        properties_to_compute: Optional[List[str]] = None,
        max_relax_steps: int = 100,
        fmax: float = 0.05,
    ):
        self.calculator_name = calculator if isinstance(calculator, str) else "ase"
        self.calculator = calculator if not isinstance(calculator, str) else None
        self.n_workers = max(1, n_workers)
        self.properties_to_compute = properties_to_compute or [
            "energy",
            "forces",
            "stress",
        ]
        self.max_relax_steps = max_relax_steps
        self.fmax = fmax
        self._backend = self._init_backend()

    def _init_backend(self):
        """Resolve calculator backend."""
        if isinstance(self.calculator, str):
            name = self.calculator.lower()
            if name == "mock":
                return "mock"
            if name == "vasp":
                return self._init_vasp()
            if name == "qe":
                return self._init_qe()
            if name == "gpaw":
                return self._init_gpaw()
            raise ValueError(f"Unknown calculator backend: {self.calculator}")
        if self.calculator is None:
            return "mock"
        if HAS_ASE and isinstance(self.calculator, ASECalculator):
            return self.calculator
        return "mock"

    def _init_vasp(self):
        try:
            from ase.calculators.vasp import Vasp
            return Vasp()
        except Exception as e:
            warnings.warn(f"VASP calculator init failed: {e}. Using mock mode.")
            return "mock"

    def _init_qe(self):
        try:
            from ase.calculators.espresso import Espresso
            return Espresso(command="pw.x < PREFIX.pwi > PREFIX.pwo")
        except Exception as e:
            warnings.warn(f"Quantum ESPRESSO calculator init failed: {e}. Using mock mode.")
            return "mock"

    def _init_gpaw(self):
        try:
            from gpaw import GPAW
            return GPAW()
        except Exception as e:
            warnings.warn(f"GPAW calculator init failed: {e}. Using mock mode.")
            return "mock"

    def validate_structure(
        self,
        structure: Any,
        structure_id: str = "",
        properties_to_compute: Optional[List[str]] = None,
    ) -> ValidationResult:
        """
        Validate a single structure.

        Returns a ValidationResult with computed properties and estimated cost.
        """
        props = properties_to_compute or self.properties_to_compute
        sid = structure_id or self._structure_id(structure)

        if self._backend == "mock":
            return self._mock_validate(structure, sid, props)

        if not HAS_ASE:
            return ValidationResult(
                structure_id=sid,
                structure=structure,
                calculator=self.calculator_name,
                converged=False,
                error_message="ASE not installed; cannot run real DFT validation.",
            )

        return self._ase_validate(structure, sid, props)

    def batch_validate(
        self,
        structures: List[Any],
        priority_scores: Optional[List[float]] = None,
        properties_to_compute: Optional[List[str]] = None,
    ) -> List[ValidationResult]:
        """
        Validate a batch of structures. Sorts by priority_scores if provided.

        Returns a list of ValidationResult objects in the same order as input.
        """
        if not structures:
            return []

        indexed = list(enumerate(structures))
        if priority_scores:
            indexed.sort(key=lambda x: priority_scores[x[0]], reverse=True)

        results_map: Dict[int, ValidationResult] = {}

        if self.n_workers == 1 or self._backend == "mock":
            # Mock mode is CPU-trivial; run sequentially to avoid process overhead.
            for idx, struct in indexed:
                sid = self._structure_id(struct, idx)
                res = self.validate_structure(struct, sid, properties_to_compute)
                results_map[idx] = res
        else:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {
                    idx: executor.submit(
                        self.validate_structure,
                        struct,
                        self._structure_id(struct, idx),
                        properties_to_compute,
                    )
                    for idx, struct in indexed
                }
                for idx, future in futures.items():
                    try:
                        results_map[idx] = future.result()
                    except Exception as e:
                        sid = self._structure_id(structures[idx], idx)
                        results_map[idx] = ValidationResult(
                            structure_id=sid,
                            structure=structures[idx],
                            calculator=self.calculator_name,
                            converged=False,
                            error_message=str(e),
                        )

        # Restore original order
        return [results_map[i] for i in range(len(structures))]

    def _structure_id(self, structure: Any, idx: int = 0) -> str:
        """Generate a stable identifier for a structure."""
        if isinstance(structure, dict):
            if "candidate_id" in structure:
                return str(structure["candidate_id"])
            if "generation_id" in structure:
                return str(structure["generation_id"])
            formula = structure.get("composition", f"unknown_{idx}")
            return f"{formula}_{idx}"
        if hasattr(structure, "_candidate_id"):
            return str(getattr(structure, "_candidate_id"))
        if HAS_PYMATGEN and isinstance(structure, Structure):
            if hasattr(structure, "properties") and isinstance(structure.properties, dict) and "_candidate_id" in structure.properties:
                return str(structure.properties["_candidate_id"])
            return f"{structure.composition.reduced_formula}_{idx}"
        if HAS_ASE and isinstance(structure, Atoms):
            return f"{structure.get_chemical_formula()}_{idx}"
        return f"struct_{idx}"

    def _structure_to_atoms(self, structure: Any) -> Optional[Any]:
        """Convert pymatgen Structure or dict to ASE Atoms."""
        if HAS_ASE and isinstance(structure, Atoms):
            return structure
        if HAS_PYMATGEN and isinstance(structure, Structure):
            return AseAtomsAdaptor().get_atoms(structure)
        if isinstance(structure, dict):
            symbols = []
            for el, count in zip(structure.get("elements", []), structure.get("stoich", [])):
                symbols.extend([el] * int(count))
            if not symbols and "composition" in structure:
                # Fallback: parse simple formula like "Li2PS3"
                symbols = self._parse_formula(structure["composition"])
            positions = structure.get("positions")
            cell = structure.get("lattice")
            if positions is not None and cell is not None:
                return Atoms(
                    symbols=symbols,
                    positions=np.asarray(positions),
                    cell=np.asarray(cell),
                    pbc=True,
                )
        return None

    @staticmethod
    def _parse_formula(formula: str) -> List[str]:
        """Very basic formula parser for fallback dict structures."""
        import re
        pattern = re.compile(r"([A-Z][a-z]?)(\d*)")
        symbols = []
        for element, count in pattern.findall(formula):
            symbols.extend([element] * max(1, int(count) if count else 1))
        return symbols

    def _mock_validate(
        self,
        structure: Any,
        structure_id: str,
        properties_to_compute: List[str],
    ) -> ValidationResult:
        """
        Deterministic mock DFT validator.

        Produces plausible property values based on a structure hash so the
        same composition/geometry yields reproducible numbers across runs.
        """
        seed = int(hashlib.md5(structure_id.encode()).hexdigest(), 16) % (2**31)
        rng = np.random.default_rng(seed)

        n_atoms = 1
        if HAS_PYMATGEN and isinstance(structure, Structure):
            n_atoms = len(structure)
        elif isinstance(structure, dict):
            positions = structure.get("positions", [])
            n_atoms = len(positions) if positions else 1
        elif HAS_ASE and isinstance(structure, Atoms):
            n_atoms = len(structure)

        # Mock relaxation: shift energy slightly lower than screening estimate
        energy = float(rng.uniform(-5.5, -1.0)) * n_atoms
        max_force = float(rng.uniform(0.001, 0.08))
        max_stress = float(rng.uniform(0.01, 0.5))
        band_gap = float(rng.uniform(0.0, 4.5))

        properties: Dict[str, float] = {}
        if "energy" in properties_to_compute:
            properties["energy"] = energy
        if "forces" in properties_to_compute:
            properties["forces"] = max_force
        if "stress" in properties_to_compute:
            properties["stress"] = max_stress
        if "band_gap" in properties_to_compute:
            properties["band_gap"] = band_gap

        # Formation energy per atom (fake but consistent)
        properties["formation_energy_per_atom"] = energy / max(n_atoms, 1)
        properties["stability"] = -abs(properties["formation_energy_per_atom"])

        # Mock cost: ~1-6 CPU hours per structure
        cost_hours = float(rng.uniform(1.0, 6.0))

        return ValidationResult(
            structure_id=structure_id,
            structure=structure,
            calculator="mock",
            converged=True,
            properties=properties,
            cost_hours=cost_hours,
        )

    def _ase_validate(
        self,
        structure: Any,
        structure_id: str,
        properties_to_compute: List[str],
    ) -> ValidationResult:
        """Run a real ASE-based relaxation + single-point calculation."""
        atoms = self._structure_to_atoms(structure)
        if atoms is None:
            return ValidationResult(
                structure_id=structure_id,
                structure=structure,
                calculator=self.calculator_name,
                converged=False,
                error_message="Could not convert structure to ASE Atoms.",
            )

        start = time.time()
        try:
            atoms.calc = self._backend
            relax = BFGS(atoms, logfile=None)
            relax.run(fmax=self.fmax, steps=self.max_relax_steps)
            converged = relax.get_number_of_steps() < self.max_relax_steps

            energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces()
            max_force = float(np.max(np.linalg.norm(forces, axis=1)))
            stress = atoms.get_stress(voigt=True)
            max_stress = float(np.max(np.abs(stress))) if stress is not None else 0.0

            properties: Dict[str, float] = {}
            if "energy" in properties_to_compute:
                properties["energy"] = energy
            if "forces" in properties_to_compute:
                properties["forces"] = max_force
            if "stress" in properties_to_compute:
                properties["stress"] = max_stress

            n_atoms = len(atoms)
            properties["formation_energy_per_atom"] = energy / max(n_atoms, 1)
            properties["stability"] = -abs(properties["formation_energy_per_atom"])

            # Very rough wall-time cost estimate
            elapsed_hours = (time.time() - start) / 3600.0
            cost_hours = max(elapsed_hours, 0.5)

            return ValidationResult(
                structure_id=structure_id,
                structure=structure,
                calculator=self.calculator_name,
                converged=converged,
                properties=properties,
                cost_hours=cost_hours,
            )
        except Exception as e:
            return ValidationResult(
                structure_id=structure_id,
                structure=structure,
                calculator=self.calculator_name,
                converged=False,
                error_message=str(e),
            )

    def get_backend_info(self) -> Dict[str, Any]:
        """Return a summary of the configured validation backend."""
        return {
            "calculator": self.calculator_name,
            "backend_type": "mock" if self._backend == "mock" else "ase",
            "n_workers": self.n_workers,
            "properties_to_compute": self.properties_to_compute,
        }
