"""Fail-closed validation of generated periodic crystal geometries.

The generation backends used by this repository return either pymatgen
``Structure`` objects, ASE ``Atoms`` objects, or small dictionary records.  A
geometry gate is deliberately kept independent from the screening model so a
malformed candidate can never accidentally reach CHGNet (or be replaced by a
heuristic score).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import itertools
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # Optional dependency in development installations.
    from pymatgen.core import Composition, Structure
except Exception:  # pragma: no cover - exercised when pymatgen is unavailable
    Composition = None
    Structure = None

try:  # Optional dependency in development installations.
    from ase import Atoms
except Exception:  # pragma: no cover - exercised when ASE is unavailable
    Atoms = None


DEFAULT_MIN_DISTANCE_ANGSTROM = 0.8


class GeometryFailureCode(str, Enum):
    """Canonical, machine-readable geometry validation outcomes."""

    VALID = "VALID"
    INVALID_GEOMETRY = "INVALID_GEOMETRY"
    INVALID_STRUCTURE_TYPE = "INVALID_STRUCTURE_TYPE"
    INVALID_COMPOSITION = "INVALID_COMPOSITION"
    EMPTY_COMPOSITION = "EMPTY_COMPOSITION"
    EMPTY_STRUCTURE = "EMPTY_STRUCTURE"
    INVALID_PERIODIC_REPRESENTATION = "INVALID_PERIODIC_REPRESENTATION"
    NONFINITE_LATTICE = "NONFINITE_LATTICE"
    NONFINITE_COORDINATES = "NONFINITE_COORDINATES"
    INVALID_LATTICE_SHAPE = "INVALID_LATTICE_SHAPE"
    INVALID_COORDINATE_SHAPE = "INVALID_COORDINATE_SHAPE"
    NONPOSITIVE_CELL_VOLUME = "NONPOSITIVE_CELL_VOLUME"
    SITE_COUNT_MISMATCH = "SITE_COUNT_MISMATCH"
    PERIODIC_MIN_DISTANCE = "PERIODIC_MIN_DISTANCE"

    # Friendly aliases used by callers that describe the physical failure.
    PERIODIC_COLLISION = "PERIODIC_MIN_DISTANCE"
    DEGENERATE_CELL = "NONPOSITIVE_CELL_VOLUME"


GeometryValidationCode = GeometryFailureCode


@dataclass
class GeometryValidationResult:
    """Typed result of validating one periodic structure."""

    valid: bool
    code: str = GeometryFailureCode.VALID.value
    details: Dict[str, Any] = field(default_factory=dict)
    minimum_distance: Optional[float] = None
    offending_pair: Optional[Tuple[int, int]] = None
    representation: Optional[str] = None

    @property
    def failure_code(self) -> str:
        """Alias retained for provenance and API readability."""

        return self.code

    @property
    def failure_reason(self) -> str:
        return self.code

    @property
    def minimum_distance_angstrom(self) -> Optional[float]:
        return self.minimum_distance

    @property
    def measured_minimum_distance(self) -> Optional[float]:
        return self.minimum_distance

    @property
    def min_distance(self) -> Optional[float]:
        return self.minimum_distance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "code": self.code,
            "failure_code": self.code,
            "details": dict(self.details),
            "minimum_distance": self.minimum_distance,
            "offending_pair": list(self.offending_pair) if self.offending_pair is not None else None,
            "representation": self.representation,
        }


# Element symbols are only used by the dependency-free composition parser.
_ELEMENT_SYMBOLS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
    "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs",
    "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
}


def _failure(code: GeometryFailureCode | str, *, details: Optional[Dict[str, Any]] = None,
             representation: Optional[str] = None, minimum_distance: Optional[float] = None,
             offending_pair: Optional[Tuple[int, int]] = None) -> GeometryValidationResult:
    value = code.value if isinstance(code, GeometryFailureCode) else str(code)
    return GeometryValidationResult(
        valid=False,
        code=value,
        details=dict(details or {}),
        minimum_distance=minimum_distance,
        offending_pair=offending_pair,
        representation=representation,
    )


def _composition_is_parseable(composition: Any) -> bool:
    if composition is None or not str(composition).strip():
        return False
    text = str(composition).strip().replace(" ", "")
    if Composition is not None:
        try:
            parsed = Composition(text)
            return bool(parsed and parsed.num_atoms > 0 and all(float(v) > 0 for v in parsed.values()))
        except Exception:
            return False
    # Dependency-free parser: support ordinary chemical formulae and reject
    # arbitrary prose, malformed counts, and unknown element symbols.
    matches = list(re.finditer(r"([A-Z][a-z]?)(\d*(?:\.\d+)?)", text))
    if not matches or "".join(m.group(0) for m in matches) != text:
        return False
    for match in matches:
        if match.group(1) not in _ELEMENT_SYMBOLS:
            return False
        count = match.group(2)
        if count and float(count) <= 0:
            return False
    return True


def _expanded_formula_species(composition: str) -> List[str]:
    """Expand an ordinary formula enough to associate dict sites with species."""
    text = str(composition).strip().replace(" ", "")
    if Composition is not None:
        try:
            parsed = Composition(text)
            # Keep a deterministic order.  Site chemistry is not used for the
            # distance calculation, but a count mismatch must not be hidden.
            species: List[str] = []
            for element, amount in parsed.items():
                n = int(round(float(amount)))
                if n <= 0 or abs(float(amount) - n) > 1e-8:
                    return []
                species.extend([str(element)] * n)
            return species
        except Exception:
            return []
    species = []
    for match in re.finditer(r"([A-Z][a-z]?)(\d*(?:\.\d+)?)", text):
        amount = float(match.group(2) or 1)
        n = int(round(amount))
        if n <= 0 or abs(amount - n) > 1e-8:
            return []
        species.extend([match.group(1)] * n)
    return species


def _as_numeric_array(value: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return None
    return arr


def _closest_lattice_image(frac_delta: np.ndarray, lattice: np.ndarray,
                           *, exclude_zero: bool = False) -> float:
    """Solve the 3-D closest-lattice-image problem for one fractional delta.

    A component-wise ``[-1, 0, 1]`` stencil is not reliable for skew cells:
    the shortest lattice vector can require coefficients larger than one.  We
    instead use a finite, mathematically bounded integer search.  A rounded
    image supplies an initial upper bound; the smallest eigenvalue of the
    lattice Gram matrix bounds every integer coefficient that could improve
    it.  In ordinary cells this examines only a few dozen images and remains
    exact for arbitrarily skew (but nondegenerate) cells.
    """
    frac_delta = np.asarray(frac_delta, dtype=float)
    gram = np.dot(lattice, lattice.T)
    min_eigenvalue = float(np.min(np.linalg.eigvalsh(gram)))
    if not math.isfinite(min_eigenvalue) or min_eigenvalue <= 0:
        return math.inf

    center = -frac_delta
    candidates = [np.rint(center).astype(int)]
    if exclude_zero and np.all(candidates[0] == 0):
        candidates.extend(np.eye(3, dtype=int))
        candidates.extend(-np.eye(3, dtype=int))
    best = math.inf
    for integer_shift in candidates:
        if exclude_zero and np.all(integer_shift == 0):
            continue
        vector = frac_delta + integer_shift
        best = min(best, float(np.sqrt(np.dot(vector, np.dot(gram, vector)))))

    # For any candidate with norm <= best, lambda_min * ||delta+n||^2 <= best^2.
    radius = best / math.sqrt(min_eigenvalue) + 1e-12
    lower = np.ceil(center - radius).astype(int)
    upper = np.floor(center + radius).astype(int)
    for integer_shift in itertools.product(
        *(range(int(lower[k]), int(upper[k]) + 1) for k in range(3))
    ):
        integer_shift = np.asarray(integer_shift, dtype=int)
        if exclude_zero and np.all(integer_shift == 0):
            continue
        vector = frac_delta + integer_shift
        distance = float(np.sqrt(np.dot(vector, np.dot(gram, vector))))
        if distance < best:
            best = distance
    return best


def _minimum_periodic_distance(frac_coords: np.ndarray, lattice: np.ndarray) -> Tuple[float, Tuple[int, int]]:
    """Return the exact minimum inter-site distance including periodic images."""
    n_sites = len(frac_coords)
    best = math.inf
    pair: Tuple[int, int] = (-1, -1)
    for i in range(n_sites):
        # Self images are part of the periodic minimum-distance definition.
        self_distance = _closest_lattice_image(frac_coords[i] - frac_coords[i], lattice, exclude_zero=True)
        if self_distance < best:
            best, pair = self_distance, (i, i)
        for j in range(i + 1, n_sites):
            distance = _closest_lattice_image(frac_coords[j] - frac_coords[i], lattice)
            if distance < best:
                best, pair = distance, (i, j)
    return best, pair


class GeometryValidator:
    """Validate periodic cell, finite coordinates, chemistry, and collisions."""

    def __init__(self, min_distance: float = DEFAULT_MIN_DISTANCE_ANGSTROM,
                 *, minimum_distance: Optional[float] = None):
        if minimum_distance is not None:
            min_distance = minimum_distance
        try:
            self.min_distance = float(min_distance)
        except (TypeError, ValueError) as exc:
            raise ValueError("min_distance must be a finite positive number") from exc
        if not math.isfinite(self.min_distance) or self.min_distance <= 0:
            raise ValueError("min_distance must be a finite positive number")

    def validate(self, structure: Any) -> GeometryValidationResult:
        if Structure is not None and isinstance(structure, Structure):
            return self._validate_pymatgen(structure)
        if Atoms is not None and isinstance(structure, Atoms):
            return self._validate_ase(structure)
        if isinstance(structure, dict):
            return self._validate_dict(structure)
        return _failure(GeometryFailureCode.INVALID_STRUCTURE_TYPE,
                        details={"type": type(structure).__name__})

    validate_structure = validate

    def _validate_arrays(self, *, composition: Any, lattice: Any, coords: Any,
                         representation: str, periodic: bool = True,
                         species_count: Optional[int] = None,
                         coords_are_cartesian: bool = False) -> GeometryValidationResult:
        if not _composition_is_parseable(composition):
            code = GeometryFailureCode.EMPTY_COMPOSITION if composition is None or not str(composition).strip() else GeometryFailureCode.INVALID_COMPOSITION
            return _failure(code, details={"composition": composition}, representation=representation)
        if not periodic:
            return _failure(GeometryFailureCode.INVALID_PERIODIC_REPRESENTATION,
                            details={"periodic": periodic}, representation=representation)
        lattice_arr = _as_numeric_array(lattice)
        if lattice_arr is None or lattice_arr.shape != (3, 3):
            return _failure(GeometryFailureCode.INVALID_LATTICE_SHAPE,
                            details={"shape": None if lattice_arr is None else list(lattice_arr.shape)}, representation=representation)
        if not np.isfinite(lattice_arr).all():
            return _failure(GeometryFailureCode.NONFINITE_LATTICE, representation=representation)
        volume = float(abs(np.linalg.det(lattice_arr)))
        if not math.isfinite(volume) or volume <= 1e-12:
            return _failure(GeometryFailureCode.NONPOSITIVE_CELL_VOLUME,
                            details={"cell_volume": volume}, representation=representation)
        coords_arr = _as_numeric_array(coords)
        if coords_arr is None or coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
            return _failure(GeometryFailureCode.INVALID_COORDINATE_SHAPE,
                            details={"shape": None if coords_arr is None else list(coords_arr.shape)}, representation=representation)
        if len(coords_arr) == 0:
            return _failure(GeometryFailureCode.EMPTY_STRUCTURE, representation=representation)
        if not np.isfinite(coords_arr).all():
            return _failure(GeometryFailureCode.NONFINITE_COORDINATES, representation=representation)
        if species_count is not None and species_count != len(coords_arr):
            return _failure(GeometryFailureCode.SITE_COUNT_MISMATCH,
                            details={"species_count": species_count, "coordinate_count": len(coords_arr)}, representation=representation)

        if coords_are_cartesian:
            try:
                frac_coords = np.dot(coords_arr, np.linalg.inv(lattice_arr))
            except np.linalg.LinAlgError:
                return _failure(GeometryFailureCode.NONPOSITIVE_CELL_VOLUME,
                                details={"cell_volume": volume}, representation=representation)
        else:
            frac_coords = coords_arr
        minimum_distance, pair = _minimum_periodic_distance(frac_coords, lattice_arr)
        if not math.isfinite(minimum_distance):
            return _failure(GeometryFailureCode.EMPTY_STRUCTURE, representation=representation)
        if minimum_distance < self.min_distance:
            return _failure(
                GeometryFailureCode.PERIODIC_MIN_DISTANCE,
                details={
                    "minimum_distance": minimum_distance,
                    "threshold": self.min_distance,
                    "offending_pair": list(pair),
                    "cell_volume": volume,
                },
                representation=representation,
                minimum_distance=minimum_distance,
                offending_pair=pair,
            )
        return GeometryValidationResult(
            valid=True,
            code=GeometryFailureCode.VALID.value,
            details={"cell_volume": volume, "minimum_distance": minimum_distance, "threshold": self.min_distance},
            minimum_distance=minimum_distance,
            offending_pair=pair,
            representation=representation,
        )

    def _validate_pymatgen(self, structure: Any) -> GeometryValidationResult:
        try:
            composition = structure.composition.reduced_formula
            lattice = structure.lattice.matrix
            coords = structure.frac_coords
            count = len(structure)
        except Exception as exc:
            return _failure(GeometryFailureCode.INVALID_STRUCTURE_TYPE,
                            details={"error": str(exc)}, representation="pymatgen")
        return self._validate_arrays(composition=composition, lattice=lattice, coords=coords,
                                     representation="pymatgen", species_count=count)

    def _validate_ase(self, structure: Any) -> GeometryValidationResult:
        try:
            symbols = structure.get_chemical_symbols()
            composition = structure.get_chemical_formula(mode="reduced")
            lattice = np.asarray(structure.cell.array, dtype=float)
            coords = np.asarray(structure.get_positions(), dtype=float)
            pbc = np.asarray(structure.get_pbc(), dtype=bool)
        except Exception as exc:
            return _failure(GeometryFailureCode.INVALID_STRUCTURE_TYPE,
                            details={"error": str(exc)}, representation="ase")
        periodic = bool(pbc.shape == (3,) and pbc.all())
        return self._validate_arrays(composition=composition, lattice=lattice, coords=coords,
                                     representation="ase", periodic=periodic,
                                     species_count=len(symbols), coords_are_cartesian=True)

    def _validate_dict(self, structure: Dict[str, Any]) -> GeometryValidationResult:
        composition = structure.get("composition", structure.get("formula"))
        lattice = structure.get("lattice", structure.get("cell"))
        coords = structure.get("positions", structure.get("coordinates"))
        if "fractional_coordinates" in structure:
            coords = structure.get("fractional_coordinates")
            coords_are_cartesian = False
        else:
            coords_are_cartesian = bool(structure.get("coords_are_cartesian", structure.get("cartesian", False)))
        species = structure.get("species")
        if species is None:
            species = structure.get("site_species")
        if isinstance(species, list):
            normalized = []
            for item in species:
                if isinstance(item, dict):
                    normalized.append(item.get("element", item.get("symbol")))
                else:
                    normalized.append(item)
            species_count = len(normalized)
        else:
            expanded = _expanded_formula_species(str(composition)) if composition is not None else []
            species_count = len(expanded) if expanded else None
        # A dict with a lattice is the repository's periodic representation;
        # explicit pbc=False still fails closed.  Missing pbc retains backward
        # compatibility with generated dicts, which historically omit it.
        pbc = structure.get("pbc", structure.get("periodic", True))
        if isinstance(pbc, (list, tuple, np.ndarray)):
            periodic = len(pbc) == 3 and bool(np.asarray(pbc, dtype=bool).all())
        else:
            periodic = bool(pbc)
        return self._validate_arrays(composition=composition, lattice=lattice, coords=coords,
                                     representation="dict", periodic=periodic,
                                     species_count=species_count,
                                     coords_are_cartesian=coords_are_cartesian)


def validate_geometry(structure: Any, min_distance: float = DEFAULT_MIN_DISTANCE_ANGSTROM,
                      *, minimum_distance: Optional[float] = None) -> GeometryValidationResult:
    """Convenience function for one-shot geometry validation."""

    return GeometryValidator(min_distance, minimum_distance=minimum_distance).validate(structure)


__all__ = [
    "DEFAULT_MIN_DISTANCE_ANGSTROM",
    "GeometryFailureCode",
    "GeometryValidationCode",
    "GeometryValidationResult",
    "GeometryValidator",
    "validate_geometry",
]
