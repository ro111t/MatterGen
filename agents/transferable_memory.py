"""Schema-v2, structure-aware career memory primitives.

The original career-memory tables predate the scientific execution boundary and
mostly contain formula strings and scores.  This module is deliberately kept
independent from SQLite so that the representation can be tested, serialized,
and replayed without a model or a database.  ``CareerMemory`` owns persistence
and delegates extraction/transfer decisions here.

Transfer is conservative by construction: missing descriptors are not guessed,
chemical-system relationships are explicit, and a directive is only a bounded
hint to a planner.  It can never constitute an acceptance decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import random
import re
import copy
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from agents.integrity import SCHEMA_VERSION

try:  # Optional in the repository's offline development environment.
    from pymatgen.core import Composition, Element, Structure
    HAS_PYMATGEN = True
except Exception:  # pragma: no cover - exercised on minimal installations
    Composition = Element = Structure = None
    HAS_PYMATGEN = False


try:
    import numpy as np
    _BOOL_TYPES = (bool, np.bool_)
except Exception:
    _BOOL_TYPES = (bool,)


TRANSFERABLE_SCHEMA_VERSION = SCHEMA_VERSION
MEMORY_MODES = ("none", "text_summary", "structured_provenance", "shuffled_control")
RELATIONSHIPS = (
    "same_system",
    "isoelectronic_substitution",
    "chalcogen_substitution",
    "alkali_substitution",
    "homologous_series",
    "element_class_mapping",
)

_MISSING = "missing"


def _finite(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), default=str)


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _parse_formula(formula: Any) -> Tuple[Dict[str, float], Optional[str]]:
    """Return deterministic element counts and a reduced anonymous pattern."""
    if formula is None:
        return {}, None
    if HAS_PYMATGEN:
        try:
            comp = formula if isinstance(formula, Composition) else Composition(str(formula))
            amounts = {str(k): float(v) for k, v in comp.get_el_amt_dict().items()}
            # ``anonymized_formula`` uses A/B/C labels in abundance order and
            # is invariant to Li/Na or S/Se substitution.
            anon = getattr(comp, "anonymized_formula", None)
            if anon:
                return amounts, str(anon)
            reduced = comp.reduced_composition
            elems = sorted(reduced.get_el_amt_dict())
            vals = [reduced.get_el_amt_dict()[e] for e in elems]
        except Exception:
            amounts = {}
    else:
        amounts = {}
    if not amounts:
        text = str(formula)
        pairs = re.findall(r"([A-Z][a-z]?)(?:([0-9]+(?:\.[0-9]+)?))?", text)
        for element, amount in pairs:
            amounts[element] = amounts.get(element, 0.0) + (float(amount) if amount else 1.0)
    if not amounts:
        return {}, None
    vals = list(amounts.values())
    # Reduced ratio, then canonical abundance pattern.  Element ordering is
    # irrelevant; ties are sorted lexically only for deterministic output.
    scale = min(v for v in vals if v > 0)
    ratios = [round(v / scale, 8) for v in vals]
    ordered = sorted(ratios, reverse=True)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    anon = "".join(f"{letters[i]}{int(v) if float(v).is_integer() else v:g}" for i, v in enumerate(ordered))
    return amounts, anon


def _get_formula(structure: Any, result: Any = None) -> Optional[str]:
    if isinstance(structure, Mapping):
        for key in ("composition", "formula", "reduced_formula"):
            if structure.get(key):
                return str(structure[key])
    comp = getattr(structure, "composition", None)
    if comp is not None:
        try:
            return str(getattr(comp, "reduced_formula", comp))
        except Exception:
            return str(comp)
    if result is not None:
        return str(getattr(result, "formula", "")) or None
    return None


def _elements_from_structure(structure: Any, amounts: Mapping[str, float]) -> List[str]:
    if isinstance(structure, Mapping) and structure.get("elements"):
        return sorted({str(x) for x in structure["elements"]})
    return sorted(str(x) for x in amounts)


def _normalize_chemical_system(value: Any) -> List[str]:
    """Normalize an explicit chemical-system declaration.

    Configuration files commonly use either ``["Li", "P", "Se"]`` or the
    compact ``"Li-P-Se"`` spelling.  Treating a string as an iterable of
    characters would silently weaken the fail-closed target check, so parse
    both forms here and reject malformed values by returning an empty list.
    """
    if isinstance(value, str):
        tokens = re.findall(r"[A-Z][a-z]?", value)
    elif isinstance(value, (list, tuple, set, frozenset)):
        tokens = [str(item) for item in value]
    else:
        return []
    return sorted({token for token in tokens if re.fullmatch(r"[A-Z][a-z]?", token)})


def _feature_missing(reason: str) -> Dict[str, str]:
    return {"value": _MISSING, "reason": reason}


def _periodic_class(symbol: str) -> str:
    # Keep a small standards-independent class map for offline stubs.  The
    # values are broad periodic classes, never fabricated oxidation states.
    fallback = {
        "H": "nonmetal", "Li": "alkali", "Na": "alkali", "K": "alkali", "Rb": "alkali", "Cs": "alkali",
        "Be": "alkaline_earth", "Mg": "alkaline_earth", "Ca": "alkaline_earth", "Sr": "alkaline_earth", "Ba": "alkaline_earth",
        "B": "p_block", "C": "p_block", "Si": "p_block", "Ge": "p_block", "Sn": "p_block", "Pb": "p_block",
        "N": "p_block", "P": "p_block", "As": "p_block", "Sb": "p_block", "Bi": "p_block",
        "O": "chalcogen", "S": "chalcogen", "Se": "chalcogen", "Te": "chalcogen", "Po": "chalcogen",
        "F": "halogen", "Cl": "halogen", "Br": "halogen", "I": "halogen",
    }
    if not HAS_PYMATGEN:
        return fallback.get(symbol, "transition_metal" if symbol else "unknown")
    try:
        e = Element(symbol)
        group = getattr(e, "group", None)
        row = getattr(e, "row", None)
        if group in (1, 2):
            return "alkali" if group == 1 else "alkaline_earth"
        if group in (16, 17):
            return "chalcogen" if group == 16 else "halogen"
        if group is not None and group >= 3 and group <= 12:
            return "transition_metal"
        if group in (13, 14, 15):
            return "p_block"
        if row is not None and row >= 6:
            return "heavy_element"
    except Exception:
        pass
    return "unknown"


def _element_stats(elements: Sequence[str]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    classes = {el: _periodic_class(el) for el in sorted(set(elements))}
    missing: Dict[str, str] = {}
    out: Dict[str, Any] = {"element_classes": classes}
    if not HAS_PYMATGEN:
        # Pauling electronegativities and empirical atomic radii for the common
        # campaign elements keep test/offline extraction deterministic.  An
        # unknown element remains missing rather than receiving a mean value.
        en_table = {"Li": 0.98, "Na": 0.93, "K": 0.82, "P": 2.19, "S": 2.58, "Se": 2.55, "Te": 2.10, "O": 3.44, "Cl": 3.16, "Br": 2.96, "I": 2.66, "Si": 1.90, "Ge": 2.01, "As": 2.18, "Sb": 2.05, "Bi": 2.02}
        radius_table = {"Li": 1.52, "Na": 1.86, "K": 2.27, "P": 1.06, "S": 1.05, "Se": 1.20, "Te": 1.38, "O": 0.66, "Cl": 1.02, "Br": 1.20, "I": 1.39, "Si": 1.11, "Ge": 1.25, "As": 1.19, "Sb": 1.45, "Bi": 1.60}
        en = [en_table[e] for e in sorted(set(elements)) if e in en_table]
        radii = [radius_table[e] for e in sorted(set(elements)) if e in radius_table]
        if en:
            out["electronegativity_stats"] = {"min": min(en), "max": max(en), "mean": sum(en) / len(en), "n": len(en)}
        else:
            out["electronegativity_stats"] = None
            missing["electronegativity_stats"] = "ELEMENT_PROPERTY_UNAVAILABLE"
        if radii:
            out["radius_ratio_stats"] = {"min": min(radii) / max(radii), "max": max(radii) / min(radii), "mean": sum(radii) / len(radii), "n": len(radii)}
        else:
            out["radius_ratio_stats"] = None
            missing["radius_ratio_stats"] = "ELEMENT_PROPERTY_UNAVAILABLE"
        return out, missing
    en: List[float] = []
    radii: List[float] = []
    for symbol in sorted(set(elements)):
        try:
            el = Element(symbol)
            x = _finite(getattr(el, "X", None))
            r = _finite(getattr(el, "atomic_radius", None))
            if x is not None:
                en.append(x)
            if r is not None and r > 0:
                radii.append(r)
        except Exception:
            continue
    if en:
        out["electronegativity_stats"] = {"min": min(en), "max": max(en), "mean": sum(en) / len(en), "n": len(en)}
    else:
        out["electronegativity_stats"] = None
        missing["electronegativity_stats"] = "ELEMENT_PROPERTY_UNAVAILABLE"
    if radii:
        out["radius_ratio_stats"] = {
            "min": min(radii) / max(radii),
            "max": max(radii) / min(radii),
            "mean": sum(radii) / len(radii),
            "n": len(radii),
        }
    else:
        out["radius_ratio_stats"] = None
        missing["radius_ratio_stats"] = "ELEMENT_PROPERTY_UNAVAILABLE"
    return out, missing


def _structure_value(structure: Any, key: str) -> Any:
    if isinstance(structure, Mapping):
        return structure.get(key)
    return getattr(structure, key, None)


def _first_present(*values: Any) -> Any:
    """Return the first non-None value without truth-testing array objects."""
    for value in values:
        if value is not None:
            return value
    return None


def _metadata_value(structure: Any, *keys: str) -> Any:
    for key in keys:
        value = _structure_value(structure, key)
        if value is not None:
            return value
    props = _structure_value(structure, "properties")
    if isinstance(props, Mapping):
        for key in keys:
            if props.get(key) is not None:
                return props[key]
    return None


def _lattice_volume(lattice: Any) -> Optional[float]:
    """Compute a 3x3 lattice determinant without requiring numpy."""
    if isinstance(lattice, Mapping):
        lattice = _first_present(lattice.get("matrix"), lattice.get("vectors"))
    if not isinstance(lattice, (list, tuple)) or len(lattice) != 3:
        return None
    try:
        a, b, c = ([float(x) for x in row] for row in lattice)
        return abs(a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]))
    except (TypeError, ValueError, IndexError):
        return None


def _coordination_summary(structure: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Compute a simple deterministic local-neighbour summary.

    We intentionally avoid CrystalNN's optional oxidation-state heuristics.
    For pymatgen structures, every periodic-image neighbour within a fixed
    covalent-radius envelope is counted; dict stubs may provide an explicit
    coordination mapping.  No coordination is fabricated when positions or
    lattice are absent.
    """
    explicit = _structure_value(structure, "coordination_summary")
    if explicit is not None:
        return dict(explicit) if isinstance(explicit, Mapping) else {"value": explicit}, None
    if isinstance(structure, Mapping):
        values = _first_present(structure.get("coordination_numbers"), structure.get("coordination"))
        if isinstance(values, Mapping):
            vals = [int(v) for v in values.values()]
        elif isinstance(values, (list, tuple)) and values:
            vals = [int(v) for v in values]
        else:
            vals = []
        if vals:
            return {
                "coordination_number_min": min(vals), "coordination_number_max": max(vals),
                "coordination_number_mean": sum(vals) / len(vals),
                "coordination_number_counts": {str(n): vals.count(n) for n in sorted(set(vals))},
                "method": "provided_local_coordination_v1",
            }, None
    # Lightweight object/dict fallback for local structure adapters that do
    # not depend on pymatgen.  Coordinates are treated as fractional when a
    # 3x3 lattice is supplied; this is intentionally a bounded distance count,
    # not a chemical-neighbour inference.
    raw_frac = _structure_value(structure, "fractional_coordinates")
    coords_cart_flag = _structure_value(structure, "coords_are_cartesian")
    cartesian_flag = _structure_value(structure, "cartesian")
    explicit_frac = _structure_value(structure, "frac_coords")
    explicit_cart = _structure_value(structure, "cart_coords")

    if isinstance(raw_frac, _BOOL_TYPES):
        if bool(raw_frac):
            positions = _first_present(
                explicit_frac,
                _structure_value(structure, "positions"),
                _structure_value(structure, "coordinates"),
            )
            is_fractional = True
        else:
            positions = _first_present(
                explicit_cart,
                _structure_value(structure, "positions"),
                _structure_value(structure, "coordinates"),
            )
            is_fractional = False
    elif raw_frac is not None:
        positions = raw_frac
        is_fractional = True
    elif explicit_frac is not None:
        positions = explicit_frac
        is_fractional = True
    elif explicit_cart is not None:
        positions = explicit_cart
        is_fractional = False
    else:
        positions = _first_present(
            _structure_value(structure, "positions"),
            _structure_value(structure, "coordinates"),
        )
        if isinstance(coords_cart_flag, _BOOL_TYPES) and bool(coords_cart_flag):
            is_fractional = False
        elif isinstance(cartesian_flag, _BOOL_TYPES) and bool(cartesian_flag):
            is_fractional = False
        else:
            is_fractional = True

    lattice = _structure_value(structure, "lattice")
    matrix = lattice.get("matrix") if isinstance(lattice, Mapping) else lattice
    if isinstance(positions, (list, tuple)) and positions and isinstance(matrix, (list, tuple)) and len(matrix) == 3:
        try:
            vectors = [[float(x) for x in row] for row in matrix]
            # Use the shortest lattice-vector norm as a conservative local
            # envelope, capped to avoid counting distant periodic shells.
            lengths = [sum(x * x for x in row) ** 0.5 for row in vectors]
            cutoff = min(3.0, max(min(lengths) * 0.55, 1e-8))
            if is_fractional:
                cart = []
                for pos in positions:
                    frac = [float(x) for x in pos[:3]]
                    cart.append([sum(frac[k] * vectors[k][j] for k in range(3)) for j in range(3)])
                method_name = "provided_fractional_distance_v1"
            else:
                cart = [[float(x) for x in pos[:3]] for pos in positions]
                method_name = "provided_cartesian_distance_v1"
            values = []
            for i, left in enumerate(cart):
                n = 0
                for j, right in enumerate(cart):
                    if i == j:
                        continue
                    delta = [right[k] - left[k] for k in range(3)]
                    # Orthorhombic minimum-image handling is deterministic;
                    # skew cells use the direct local image conservatively.
                    if all(abs(vectors[k][j]) < 1e-10 for k in range(3) for j in range(3) if j != k):
                        for k in range(3):
                            length = abs(vectors[k][k])
                            if length:
                                delta[k] -= round(delta[k] / length) * length
                    distance = sum(x * x for x in delta) ** 0.5
                    if distance <= cutoff:
                        n += 1
                values.append(n)
            return {
                "coordination_number_min": min(values), "coordination_number_max": max(values),
                "coordination_number_mean": sum(values) / len(values),
                "coordination_number_counts": {str(n): values.count(n) for n in sorted(set(values))},
                "method": method_name,
            }, None
        except (TypeError, ValueError, IndexError, ZeroDivisionError):
            return None, "COORDINATION_COMPUTATION_FAILED"
    if not HAS_PYMATGEN or not isinstance(structure, Structure):
        return None, "STRUCTURE_COORDINATION_UNAVAILABLE"
    try:
        values: List[int] = []
        for site in structure:
            centre = site.specie.symbol
            radius = _finite(getattr(Element(centre), "covalent_radius", None)) or 1.5
            # A broad, chemistry-neutral envelope avoids optional neighbour
            # packages while remaining deterministic on local structures.
            neighbours = structure.get_neighbors(site, 1.35 * radius + 2.0)
            values.append(len(neighbours))
        if not values:
            return None, "EMPTY_STRUCTURE"
        return {
            "coordination_number_min": min(values),
            "coordination_number_max": max(values),
            "coordination_number_mean": sum(values) / len(values),
            "coordination_number_counts": {str(n): values.count(n) for n in sorted(set(values))},
            "method": "periodic_distance_covalent_envelope_v1",
        }, None
    except Exception:
        return None, "COORDINATION_COMPUTATION_FAILED"


def _space_group_summary(structure: Any) -> Tuple[Dict[str, Any], Dict[str, str]]:
    out: Dict[str, Any] = {}
    missing: Dict[str, str] = {}
    explicit = _structure_value(structure, "space_group")
    explicit_system = _structure_value(structure, "crystal_system")
    if explicit is not None:
        out["space_group"] = explicit
    if explicit_system is not None:
        out["crystal_system"] = explicit_system
    if ("space_group" not in out or "crystal_system" not in out) and HAS_PYMATGEN and isinstance(structure, Structure):
        try:
            symbol, number = structure.get_space_group_info(symprec=0.01)
            out.setdefault("space_group", {"symbol": str(symbol), "number": int(number)})
            # SpaceGroup has crystal_system in newer pymatgen; keep a small
            # deterministic fallback for versions where it is absent.
            try:
                from pymatgen.symmetry.groups import SpaceGroup
                out.setdefault("crystal_system", str(SpaceGroup(symbol).crystal_system))
            except Exception:
                pass
        except Exception:
            pass
    for key in ("space_group", "crystal_system"):
        if key not in out:
            out[key] = None
            missing[key] = "SPACE_GROUP_UNAVAILABLE"
    return out, missing


def _oxidation_summary(structure: Any, amounts: Mapping[str, float]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    out: Dict[str, Any] = {"oxidation_state_feasible": None, "oxidation_state_balance": None}
    missing: Dict[str, str] = {}
    explicit = _structure_value(structure, "oxidation_state_feasible")
    if explicit is not None:
        out["oxidation_state_feasible"] = bool(explicit)
    if isinstance(structure, Mapping) and structure.get("oxidation_states"):
        states = structure["oxidation_states"]
        try:
            total = sum(float(amounts.get(str(k), 0)) * float(v) for k, v in states.items())
            out["oxidation_state_balance"] = total
            out["oxidation_state_feasible"] = abs(total) < 1e-8
        except Exception:
            pass
    if HAS_PYMATGEN and isinstance(structure, Structure):
        try:
            comp = structure.composition
            guesses = comp.oxi_state_guesses()
            if guesses:
                best = sorted(guesses, key=lambda g: _canonical_json(g))[0]
                total = sum(float(amounts.get(str(k), 0.0)) * float(v) for k, v in best.items())
                out["oxidation_state_balance"] = total
                out["oxidation_state_feasible"] = abs(total) < 1e-8
        except Exception:
            pass
    for key, reason in (("oxidation_state_feasible", "OXIDATION_STATES_UNAVAILABLE"), ("oxidation_state_balance", "OXIDATION_STATES_UNAVAILABLE")):
        if out[key] is None:
            missing[key] = reason
    return out, missing


@dataclass
class TransferableFeatures:
    """Formula-independent descriptors used as a transfer query/record."""

    anonymous_stoichiometric_pattern: Optional[str] = None
    structural_prototype: Optional[str] = None
    coordination_summary: Optional[Dict[str, Any]] = None
    space_group: Any = None
    crystal_system: Optional[str] = None
    volume_per_atom: Optional[float] = None
    density: Optional[float] = None
    packing_fraction: Optional[float] = None
    oxidation_state_feasible: Optional[bool] = None
    oxidation_state_balance: Optional[float] = None
    element_classes: Dict[str, str] = field(default_factory=dict)
    electronegativity_stats: Optional[Dict[str, Any]] = None
    radius_ratio_stats: Optional[Dict[str, Any]] = None
    thermodynamic_label: Optional[str] = None
    thermodynamic_threshold_ev_per_atom: Optional[float] = None
    missing_features: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # Read-only vocabulary aliases used by early experiment notebooks.
    @property
    def anonymous_stoichiometry(self) -> Optional[str]:
        return self.anonymous_stoichiometric_pattern

    @property
    def prototype_id(self) -> Optional[str]:
        return self.structural_prototype

    @property
    def coordination_number_summary(self) -> Optional[Dict[str, Any]]:
        return self.coordination_summary

    @property
    def missing(self) -> Dict[str, str]:
        return self.missing_features

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "TransferableFeatures":
        data = dict(payload or {})
        known = cls.__dataclass_fields__
        return cls(**{k: data[k] for k in known if k in data})


def extract_transferable_features(
    structure: Any,
    result: Any = None,
    validation: Any = None,
    thermodynamics: Any = None,
) -> TransferableFeatures:
    """Extract deterministic structure/result descriptors.

    ``None`` plus a reason code is used for every unavailable value.  This is
    important: a missing field must not accidentally act as a transferable
    zero or a successful outcome.
    """
    formula = _get_formula(structure, result)
    amounts, anonymous = _parse_formula(formula)
    missing: Dict[str, str] = {}
    if anonymous is None:
        missing["anonymous_stoichiometric_pattern"] = "COMPOSITION_UNAVAILABLE"

    explicit_proto = _metadata_value(structure, "structural_prototype", "prototype", "prototype_id", "motif_id")
    prototype = str(explicit_proto) if explicit_proto is not None else None
    if prototype is None:
        # Prototype identifiers are deliberately conservative.  A reduced
        # formula is not itself a prototype; only explicit metadata qualifies.
        missing["structural_prototype"] = "PROTOTYPE_UNAVAILABLE"

    coord, coord_reason = _coordination_summary(structure)
    if coord is None:
        missing["coordination_summary"] = coord_reason or "COORDINATION_UNAVAILABLE"
    sg, sg_missing = _space_group_summary(structure)
    missing.update(sg_missing)

    volume = _finite(_structure_value(structure, "volume_per_atom"))
    density = _finite(_structure_value(structure, "density"))
    packing = _finite(_structure_value(structure, "packing_fraction"))
    if volume is None and HAS_PYMATGEN and isinstance(structure, Structure):
        try:
            volume = float(structure.volume) / max(len(structure), 1)
        except Exception:
            pass
    if volume is None and isinstance(structure, Mapping):
        cell_volume = _finite(structure.get("volume")) or _lattice_volume(structure.get("lattice"))
        positions = _first_present(structure.get("positions"), structure.get("sites"))
        n_atoms = len(positions) if isinstance(positions, (list, tuple)) else 0
        if cell_volume is not None and n_atoms > 0:
            volume = cell_volume / n_atoms
    if density is None and HAS_PYMATGEN and isinstance(structure, Structure):
        try:
            density = float(structure.density)
        except Exception:
            pass
    if volume is None:
        missing["volume_per_atom"] = "VOLUME_UNAVAILABLE"
    if density is None:
        missing["density"] = "DENSITY_UNAVAILABLE"
    if packing is None:
        missing["packing_fraction"] = "PACKING_FRACTION_UNAVAILABLE"

    elem = _elements_from_structure(structure, amounts)
    stats, stat_missing = _element_stats(elem)
    missing.update(stat_missing)
    ox, ox_missing = _oxidation_summary(structure, amounts)
    missing.update(ox_missing)

    if prototype is None and coord:
        # A stable motif identifier based on coordination only is useful but
        # is explicitly labelled as a motif, never as a crystallographic
        # prototype.
        prototype = None

    thermo_source: Any = thermodynamics if thermodynamics is not None else result
    label = None
    threshold = None
    if thermo_source is not None:
        source_mapping = thermo_source if isinstance(thermo_source, Mapping) else getattr(thermo_source, "predictions", None)
        certified = _certified_thermodynamic_source(thermo_source, source_mapping)
        if isinstance(source_mapping, Mapping) and certified:
            label = source_mapping.get("thermodynamic_label") or source_mapping.get("stability_label")
            stable_flag = source_mapping.get("predicted_thermodynamically_stable")
            retained_flag = source_mapping.get("retained_by_hull_threshold")
            if label is None and stable_flag is not None:
                label = "stable" if bool(stable_flag) else "unstable"
            if label is None and retained_flag is not None:
                label = "retained" if bool(retained_flag) else "rejected"
            threshold = _finite(source_mapping.get("predicted_energy_above_hull_ev_per_atom"))
            threshold = threshold if threshold is not None else _finite(source_mapping.get("energy_above_hull_ev_per_atom"))
        elif certified:
            label = getattr(thermo_source, "thermodynamic_label", None)
            if label is None and getattr(thermo_source, "predicted_thermodynamically_stable", None) is not None:
                label = "stable" if bool(thermo_source.predicted_thermodynamically_stable) else "unstable"
            if label is None and getattr(thermo_source, "retained_by_hull_threshold", None) is not None:
                label = "retained" if bool(thermo_source.retained_by_hull_threshold) else "rejected"
            threshold = _finite(getattr(thermo_source, "predicted_energy_above_hull_ev_per_atom", None))
    if label is None:
        missing["thermodynamic_label"] = (
            "UNCERTIFIED_THERMODYNAMIC_OUTCOME" if thermo_source is not None else "THERMODYNAMIC_OUTCOME_UNAVAILABLE"
        )
    if threshold is None:
        missing["thermodynamic_threshold_ev_per_atom"] = (
            "UNCERTIFIED_THERMODYNAMIC_OUTCOME" if thermo_source is not None else "THERMODYNAMIC_THRESHOLD_UNAVAILABLE"
        )

    return TransferableFeatures(
        anonymous_stoichiometric_pattern=anonymous,
        structural_prototype=prototype,
        coordination_summary=coord,
        space_group=sg.get("space_group"),
        crystal_system=sg.get("crystal_system"),
        volume_per_atom=volume,
        density=density,
        packing_fraction=packing,
        oxidation_state_feasible=ox["oxidation_state_feasible"],
        oxidation_state_balance=ox["oxidation_state_balance"],
        element_classes=stats.get("element_classes", {}),
        electronegativity_stats=stats.get("electronegativity_stats"),
        radius_ratio_stats=stats.get("radius_ratio_stats"),
        thermodynamic_label=str(label) if label is not None else None,
        thermodynamic_threshold_ev_per_atom=threshold,
        missing_features=missing,
    )


def _certified_thermodynamic_source(source: Any, mapping: Any = None) -> bool:
    """Require the Sprint-3 oracle identity before accepting hull labels."""
    payload = mapping if isinstance(mapping, Mapping) else source if isinstance(source, Mapping) else {}
    if isinstance(source, Mapping):
        # A free dict is deliberately insufficient, even when it self-asserts
        # ``thermodynamics_certified``.  Only an actual typed oracle result or
        # a ScreeningResult produced by the certified oracle is trusted.
        return False
    try:
        from agents.thermodynamics import ThermodynamicResult
    except Exception:  # pragma: no cover - import failure is fail-closed
        ThermodynamicResult = ()
    if isinstance(source, ThermodynamicResult):
        return bool(
            getattr(source, "success", False) is True
            and getattr(source, "reference_set_id", None)
            and getattr(source, "reference_set_hash", None)
            and isinstance(getattr(source, "model", None), Mapping)
            and isinstance(getattr(source, "relaxation_settings", None), Mapping)
        )
    try:
        from agents.screening import ScreeningResult
    except Exception:  # pragma: no cover
        ScreeningResult = ()
    if isinstance(source, ScreeningResult):
        return bool(
            source.backend == "chgnet_thermodynamic_oracle"
            and source.provenance_stage == "thermodynamics"
            and isinstance(source.predictions, Mapping)
            and source.predictions.get("provenance_schema_version") == SCHEMA_VERSION
            and source.predictions.get("thermodynamics_certified") is True
            and source.predictions.get("reference_set_id")
            and source.predictions.get("reference_set_hash")
            and isinstance(source.predictions.get("model"), Mapping)
            and isinstance(source.predictions.get("relaxation_settings"), Mapping)
            and ("predicted_energy_above_hull_ev_per_atom" in source.predictions
                 or "predicted_thermodynamically_stable" in source.predictions)
        )
    return False


@dataclass
class ApplicabilityConstraint:
    """Fail-closed relationship and descriptor constraints for a record."""

    allowed_relationship: str = "same_system"
    feature_ranges: Dict[str, Tuple[Optional[float], Optional[float]]] = field(default_factory=dict)
    allowed_element_classes: Dict[str, List[str]] = field(default_factory=dict)
    source_chemical_system: List[str] = field(default_factory=list)
    target_chemical_system: List[str] = field(default_factory=list)
    min_evidence: int = 1
    evidence_count: int = 1
    uncertainty: Optional[float] = None
    confidence: float = 0.0
    observed_outcome_direction: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def allowed_chemical_system_relationship(self) -> str:
        return self.allowed_relationship

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ApplicabilityConstraint":
        data = dict(payload or {})
        if "feature_ranges" in data:
            data["feature_ranges"] = {
                str(k): tuple(v) if isinstance(v, (list, tuple)) else v
                for k, v in data["feature_ranges"].items()
            }
        known = cls.__dataclass_fields__
        return cls(**{k: data[k] for k in known if k in data})


def applicability_from_declaration(
    declaration: Mapping[str, Any] | None,
    *,
    source_features: Optional[TransferableFeatures] = None,
) -> Optional[ApplicabilityConstraint]:
    """Build constraints only from an explicit campaign declaration.

    The default ``None`` is significant: callers then receive the strict
    ``same_system`` default.  No substitution relationship is inferred from
    element names, formulas, or broad periodic classes.
    """
    if not declaration:
        return None
    raw = dict(declaration)
    nested = raw.get("applicability")
    if isinstance(nested, Mapping):
        raw = dict(nested)
    relationship = raw.get("allowed_relationship", raw.get("relationship", raw.get("allowed_chemical_system_relationship")))
    if relationship is None:
        return None
    source_system = raw.get("source_chemical_system", raw.get("source_elements"))
    if source_system is None and source_features is not None:
        source_system = sorted(source_features.element_classes)
    target_system = raw.get("target_chemical_system", raw.get("target_elements", []))
    ranges = raw.get("feature_ranges", {})
    return ApplicabilityConstraint(
        allowed_relationship=str(relationship),
        feature_ranges={str(k): tuple(v) for k, v in ranges.items()},
        allowed_element_classes={str(k): list(v) for k, v in (raw.get("allowed_element_classes", raw.get("element_class_mapping", {})) or {}).items()},
        source_chemical_system=_normalize_chemical_system(source_system),
        target_chemical_system=_normalize_chemical_system(target_system),
        min_evidence=max(1, int(raw.get("min_evidence", 1))),
        evidence_count=max(1, int(raw.get("evidence_count", 1))),
        uncertainty=_finite(raw.get("uncertainty")),
        confidence=min(1.0, max(0.0, float(raw.get("confidence", 0.5)))),
        observed_outcome_direction=str(raw.get("observed_outcome_direction", "unknown")),
    )


@dataclass
class TransferDirective:
    """Bounded planning hints; no field is an acceptance override."""

    preferred_anonymous_stoichiometries: List[str] = field(default_factory=list)
    preferred_prototypes: List[str] = field(default_factory=list)
    preferred_coordination_motifs: List[str] = field(default_factory=list)
    volume_per_atom_range: Optional[Tuple[float, float]] = None
    permitted_element_class_substitutions: Dict[str, List[str]] = field(default_factory=dict)
    exploration_weight: Optional[float] = None
    exploitation_weight: Optional[float] = None
    source_principle_ids: List[str] = field(default_factory=list)
    source_evidence_ids: List[str] = field(default_factory=list)
    source_campaign_ids: List[str] = field(default_factory=list)
    bounded: bool = True
    bypasses_geometry_gate: bool = False
    bypasses_thermodynamic_gate: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TransferableMemoryRecord:
    """One schema-v2 transferable evidence record."""

    record_id: str
    schema_version: str = TRANSFERABLE_SCHEMA_VERSION
    principle_id: Optional[str] = None
    evidence_ids: List[str] = field(default_factory=list)
    campaign_ids: List[str] = field(default_factory=list)
    source_candidate_ids: List[str] = field(default_factory=list)
    source_formulas: List[str] = field(default_factory=list)
    source_domain: str = ""
    outcome_label: str = "unknown"
    outcome_value: Optional[float] = None
    features: TransferableFeatures = field(default_factory=TransferableFeatures)
    applicability: ApplicabilityConstraint = field(default_factory=ApplicabilityConstraint)
    directive: TransferDirective = field(default_factory=TransferDirective)
    evidence_hash: str = ""
    finalized: bool = False
    created_at: Optional[float] = None

    @property
    def provenance(self) -> Dict[str, Any]:
        return {
            "campaign_ids": list(self.campaign_ids),
            "source_candidate_ids": list(self.source_candidate_ids),
            "source_formulas": list(self.source_formulas),
            "source_domain": self.source_domain,
            "principle_id": self.principle_id,
            "evidence_ids": list(self.evidence_ids),
        }

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["features"] = self.features.to_dict()
        payload["applicability"] = self.applicability.to_dict()
        payload["directive"] = self.directive.to_dict()
        payload["provenance"] = self.provenance
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransferableMemoryRecord":
        data = dict(payload)
        data["features"] = TransferableFeatures.from_dict(data.get("features"))
        applicability = dict(data.get("applicability") or {})
        if "allowed_relationship" not in applicability and "allowed_chemical_system_relationship" in applicability:
            applicability["allowed_relationship"] = applicability["allowed_chemical_system_relationship"]
        data["applicability"] = ApplicabilityConstraint.from_dict(applicability)
        data["directive"] = TransferDirective(**{
            k: data.get("directive", {}).get(k, v)
            for k, v in TransferDirective().__dict__.items()
        })
        known = cls.__dataclass_fields__
        return cls(**{k: data[k] for k in known if k in data})


def evidence_fingerprint(
    features: TransferableFeatures,
    outcome_label: str,
    outcome_value: Optional[float],
    source_domain: str,
    principle_id: Optional[str] = None,
) -> str:
    """Hash evidence content, intentionally excluding campaign/formula IDs."""
    return _stable_hash({
        "features": features.to_dict(),
        "outcome_label": outcome_label,
        "outcome_value": outcome_value,
        "source_domain": source_domain,
        "principle_id": principle_id,
    })


def _feature_value(features: TransferableFeatures, key: str) -> Any:
    value = getattr(features, key, None)
    return value


def _chemical_system(features: TransferableFeatures) -> set[str]:
    return set(features.element_classes)


def applicability_check(
    record: TransferableMemoryRecord,
    target_features: TransferableFeatures,
    *,
    target_domain: Optional[str] = None,
    target_elements: Optional[Iterable[str]] = None,
) -> Tuple[bool, List[str]]:
    """Return ``(applicable, reasons)`` using fail-closed semantics."""
    app = record.applicability
    if isinstance(target_features, Mapping):
        if "element_classes" in target_features or "anonymous_stoichiometric_pattern" in target_features:
            target_features = TransferableFeatures.from_dict(target_features)
        else:
            target_features = extract_transferable_features(target_features)
    reasons: List[str] = []
    if record.schema_version != TRANSFERABLE_SCHEMA_VERSION:
        return False, ["LEGACY_SCHEMA_QUARANTINED"]
    if not record.finalized:
        return False, ["SOURCE_CAMPAIGN_NOT_FINALIZED"]
    if record.source_domain and target_domain and record.source_domain != target_domain:
        # Cross-domain transfer needs an explicit element-class relationship;
        # plain domain names never imply transfer.
        if app.allowed_relationship not in {"element_class_mapping", "homologous_series"}:
            reasons.append("UNRELATED_DOMAIN")
    source = set(_normalize_chemical_system(app.source_chemical_system)) or _chemical_system(record.features)
    declared_target = set(_normalize_chemical_system(app.target_chemical_system))
    # Always derive the actual query system from the query features when the
    # caller does not provide a separate element list.  Otherwise a declared
    # target could accidentally replace (rather than constrain) the query.
    actual_target = (
        {str(x) for x in target_elements}
        if target_elements is not None
        else _chemical_system(target_features)
    )
    target = actual_target
    record_system = _chemical_system(record.features)
    if app.source_chemical_system and source != record_system:
        reasons.append("DECLARED_SOURCE_SYSTEM_MISMATCH")
    if declared_target and actual_target != declared_target:
        reasons.append("DECLARED_TARGET_SYSTEM_MISMATCH")
    relation = app.allowed_relationship
    if relation not in RELATIONSHIPS:
        reasons.append("UNKNOWN_RELATIONSHIP")
    elif relation == "same_system":
        if source and target and source != target:
            reasons.append("CHEMICAL_SYSTEM_MISMATCH")
        elif not source or not target:
            reasons.append("CHEMICAL_SYSTEM_MISSING")
    elif relation in {"chalcogen_substitution", "alkali_substitution", "isoelectronic_substitution", "homologous_series", "element_class_mapping"}:
        if not declared_target and not app.allowed_element_classes:
            reasons.append("EXPLICIT_TRANSFER_DECLARATION_REQUIRED")
        source_classes = set(record.features.element_classes.values())
        target_classes = set(target_features.element_classes.values())
        if not source_classes or not target_classes:
            reasons.append("ELEMENT_CLASS_MISSING")
        elif not (source_classes & target_classes) and relation != "element_class_mapping":
            reasons.append("NO_SHARED_FRAMEWORK_CLASS")
        if relation == "chalcogen_substitution":
            if "chalcogen" not in source_classes or "chalcogen" not in target_classes:
                reasons.append("CHALCOGEN_RELATION_UNSUPPORTED")
        if relation == "alkali_substitution":
            if "alkali" not in source_classes or "alkali" not in target_classes:
                reasons.append("ALKALI_RELATION_UNSUPPORTED")
        if app.allowed_element_classes:
            mapping_ok = False
            source_symbols = set(record.features.element_classes)
            target_symbols = set(target)
            for source_class, allowed_targets in app.allowed_element_classes.items():
                if source_class in source_classes or source_class in source_symbols:
                    if set(allowed_targets) & target_classes or set(allowed_targets) & target_symbols:
                        mapping_ok = True
            if not mapping_ok:
                reasons.append("ELEMENT_CLASS_MAPPING_MISMATCH")
    for key, bounds in app.feature_ranges.items():
        value = _feature_value(target_features, key)
        if value is None:
            reasons.append(f"FEATURE_MISSING:{key}")
            continue
        try:
            low, high = bounds
            if low is not None and float(value) < float(low):
                reasons.append(f"FEATURE_BELOW_RANGE:{key}")
            if high is not None and float(value) > float(high):
                reasons.append(f"FEATURE_ABOVE_RANGE:{key}")
        except (TypeError, ValueError):
            reasons.append(f"FEATURE_RANGE_INVALID:{key}")
    if app.evidence_count < app.min_evidence:
        reasons.append("INSUFFICIENT_EVIDENCE")
    if app.confidence <= 0:
        reasons.append("ZERO_CONFIDENCE")
    return not reasons, reasons or ["APPLICABLE"]


def make_directive(record: TransferableMemoryRecord, *, max_exploration: float = 0.8) -> Dict[str, Any]:
    """Serialize a safe directive with explicit citation and gate guarantees."""
    directive = record.directive.to_dict()
    features = record.features
    if not directive.get("preferred_anonymous_stoichiometries") and features.anonymous_stoichiometric_pattern:
        directive["preferred_anonymous_stoichiometries"] = [features.anonymous_stoichiometric_pattern]
    if not directive.get("preferred_prototypes") and features.structural_prototype:
        directive["preferred_prototypes"] = [features.structural_prototype]
    if not directive.get("preferred_coordination_motifs") and features.coordination_summary:
        summary = features.coordination_summary
        mean_cn = summary.get("coordination_number_mean") if isinstance(summary, Mapping) else None
        directive["preferred_coordination_motifs"] = [
            f"coordination_number_mean={mean_cn}" if mean_cn is not None else "coordination_summary_available"
        ]
    if directive.get("volume_per_atom_range") is None and features.volume_per_atom is not None:
        value = float(features.volume_per_atom)
        directive["volume_per_atom_range"] = (max(0.0, value * 0.9), value * 1.1)
    if not directive.get("permitted_element_class_substitutions") and record.applicability.allowed_element_classes:
        directive["permitted_element_class_substitutions"] = dict(record.applicability.allowed_element_classes)
    # Every positive evidence record receives a bounded policy signal derived
    # from its recorded confidence.  This is the only currently supported
    # executable effect; structural preferences remain advisory metadata until
    # a generator explicitly supports those condition fields.
    if directive.get("exploration_weight") is None and directive.get("exploitation_weight") is None:
        confidence = min(1.0, max(0.0, float(record.applicability.confidence)))
        directive["exploration_weight"] = min(0.8, max(0.1, 0.5 - 0.3 * confidence))
        directive["exploitation_weight"] = 1.0 - directive["exploration_weight"]
    unsupported = []
    for field_name, effect_name in (
        ("preferred_anonymous_stoichiometries", "anonymous_stoichiometry_conditioning"),
        ("preferred_prototypes", "prototype_conditioning"),
        ("preferred_coordination_motifs", "coordination_conditioning"),
        ("volume_per_atom_range", "volume_conditioning"),
        ("permitted_element_class_substitutions", "element_substitution_conditioning"),
    ):
        if directive.get(field_name):
            unsupported.append(effect_name)
    directive["supported_policy_effects"] = ["exploration_weight", "exploitation_weight"]
    directive["unsupported_policy_effects"] = unsupported
    directive["exploration_weight"] = min(max(float(directive.get("exploration_weight") or 0.0), 0.0), max_exploration)
    directive["exploitation_weight"] = min(max(float(directive.get("exploitation_weight") or 0.0), 0.0), 1.0)
    directive["source_principle_ids"] = sorted(set(directive.get("source_principle_ids", [])) | ({record.principle_id} if record.principle_id else set()))
    directive["source_evidence_ids"] = sorted(set(directive.get("source_evidence_ids", [])) | set(record.evidence_ids or [record.record_id]))
    directive["source_campaign_ids"] = sorted(set(directive.get("source_campaign_ids", [])) | set(record.campaign_ids))
    directive["bounded"] = True
    directive["bypasses_geometry_gate"] = False
    directive["bypasses_thermodynamic_gate"] = False
    directive["record_id"] = record.record_id
    directive["applicability"] = record.applicability.to_dict()
    return directive


def view_records(
    records: Sequence[TransferableMemoryRecord | Mapping[str, Any]],
    mode: str = "structured_provenance",
    *,
    seed: int = 0,
    query: Any = None,
) -> Dict[str, Any]:
    """Return one of four reproducible experimental memory views.

    ``shuffled_control`` permutes complete records, preserving count and every
    record marginal, while severing positional query-to-record pairing.  It is
    labelled invalid for scientific decision support and should only be used as
    a control experiment.
    """
    if mode not in MEMORY_MODES:
        raise ValueError(f"unknown memory mode {mode!r}; expected one of {MEMORY_MODES}")
    normalized = [copy.deepcopy(r.to_dict() if isinstance(r, TransferableMemoryRecord) else dict(r)) for r in records]
    base = {
        "mode": mode,
        "seed": int(seed),
        "record_count": len(normalized),
        "scientific_decision_support": mode != "shuffled_control" and mode != "none",
        "label": mode,
    }
    if mode == "none":
        base.update({"records": [], "text": "Career memory disabled."})
    elif mode == "text_summary":
        lines = []
        for rec in normalized:
            features = rec.get("features", {})
            lines.append(
                f"{rec.get('record_id', '?')}: outcome={rec.get('outcome_label', 'unknown')}; "
                f"anonymous_stoichiometry={features.get('anonymous_stoichiometric_pattern')}; "
                f"prototype={features.get('structural_prototype')}"
            )
        base.update({"records": [], "text": "\n".join(lines)})
    else:
        shown = list(normalized)
        if mode == "shuffled_control":
            n = len(shown)
            base["scientific_decision_support"] = False
            base["label"] = "shuffled_control_invalid_for_scientific_decision_support"
            if n < 2:
                # A singleton cannot be deranged, so exposing it would leave
                # the query/evidence pairing intact and make a misleading
                # control.  Preserve the corpus count in the audit only.
                base.update({
                    "records": [],
                    "shuffle_audit": {
                        "seed": int(seed), "permutation": [], "mapping": [],
                        "input_hash": _stable_hash(normalized), "output_hash": _stable_hash([]),
                        "fixed_points": 0, "valid": False,
                        "reason": "SHUFFLE_CONTROL_UNAVAILABLE_SINGLETON" if n == 1 else "SHUFFLE_CONTROL_EMPTY",
                    },
                })
                return base
            # Build a deterministic derangement.  The random stream is seeded
            # and retries are bounded; a cyclic rotation fallback guarantees no
            # fixed points for every n > 1.
            rng = random.Random(int(seed))
            permutation = list(range(n))
            for _ in range(max(16, n * 4)):
                rng.shuffle(permutation)
                if all(i != permutation[i] for i in range(n)):
                    break
            else:
                shift = 1 + (int(seed) % (n - 1))
                permutation = [(i + shift) % n for i in range(n)]
            if any(i == permutation[i] for i in range(n)):
                shift = 1
                permutation = [(i + shift) % n for i in range(n)]

            # Never expose a partially shuffled control.  In addition to the
            # no-fixed-point requirement, the mapping must be a true
            # permutation; this protects the control if the random backend is
            # replaced or returns malformed state.
            permutation_valid = (
                len(permutation) == n
                and sorted(permutation) == list(range(n))
                and all(i != permutation[i] for i in range(n))
            )
            if not permutation_valid:
                base.update({
                    "records": [],
                    "shuffle_audit": {
                        "seed": int(seed), "permutation": list(permutation),
                        "mapping": [],
                        "input_hash": _stable_hash(normalized),
                        "output_hash": _stable_hash([]),
                        "fixed_points": sum(
                            i == permutation[i]
                            for i in range(min(n, len(permutation)))
                        ),
                        "valid": False,
                        "reason": "SHUFFLE_CONTROL_INVALID_DERANGEMENT",
                    },
                })
                return base

            # Keep descriptor/query-side fields from row i and response-side
            # evidence/directive/provenance from row permutation[i].  This
            # preserves both marginals while severing the pairing.
            response_fields = {
                "outcome_label", "outcome_value", "directive", "principle_id",
                "evidence_ids", "campaign_ids", "source_candidate_ids",
                "source_formulas", "source_domain", "evidence_hash",
                "applicability", "finalized", "created_at", "provenance",
            }
            shown = []
            for i, descriptor in enumerate(normalized):
                response = normalized[permutation[i]]
                mixed = copy.deepcopy(descriptor)
                for key in response_fields:
                    if key in response:
                        mixed[key] = copy.deepcopy(response[key])
                mixed["record_id"] = f"shuffle_descriptor_{i:04d}"
                mixed["paired_descriptor_record_id"] = descriptor.get("record_id")
                mixed["paired_response_record_id"] = response.get("record_id")
                mixed["shuffle_control"] = True
                shown.append(mixed)
            base["shuffle_audit"] = {
                "seed": int(seed), "permutation": list(permutation),
                "mapping": [
                    {"descriptor_index": i, "response_index": permutation[i],
                     "descriptor_record_id": normalized[i].get("record_id"),
                     "response_record_id": normalized[permutation[i]].get("record_id")}
                    for i in range(n)
                ],
                "input_hash": _stable_hash(normalized), "output_hash": _stable_hash(shown),
                "fixed_points": sum(i == permutation[i] for i in range(n)),
                "valid": True,
                "reason": "DERANGED_DESCRIPTOR_RESPONSE_PAIRING",
            }
        base["records"] = shown
        if query is not None:
            base["query"] = query
    return base


def summarize_records(records: Sequence[TransferableMemoryRecord]) -> str:
    """Compact deterministic text view used in prompts and audit logs."""
    view = view_records(records, "text_summary")
    return str(view.get("text", ""))


def directive_match_score(structure: Any, directives: Sequence[Mapping[str, Any]]) -> Tuple[float, List[str]]:
    """Score a candidate against bounded memory hints without evaluating it.

    This is deliberately a pre-oracle prioritization signal.  It only uses
    descriptors available from the candidate structure and cannot alter any
    geometry, thermodynamic, or resource gate.
    """
    if not directives:
        return 0.0, []
    features = extract_transferable_features(structure)
    best_score = 0.0
    matched: List[str] = []
    for directive in directives:
        score = 0.0
        hits: List[str] = []
        patterns = directive.get("preferred_anonymous_stoichiometries", []) or []
        if features.anonymous_stoichiometric_pattern and features.anonymous_stoichiometric_pattern in patterns:
            score += 1.0
            hits.append("anonymous_stoichiometry")
        prototypes = directive.get("preferred_prototypes", []) or []
        if features.structural_prototype and features.structural_prototype in prototypes:
            score += 1.0
            hits.append("prototype")
        bounds = directive.get("volume_per_atom_range")
        if features.volume_per_atom is not None and isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            try:
                if float(bounds[0]) <= float(features.volume_per_atom) <= float(bounds[1]):
                    score += 1.0
                    hits.append("volume_per_atom")
            except (TypeError, ValueError):
                pass
        source_id = str(directive.get("record_id", ""))
        exploitation = min(1.0, max(0.0, float(directive.get("exploitation_weight") or 0.0)))
        score *= 0.5 + exploitation
        if score > best_score:
            best_score = score
            matched = [source_id + ":" + hit for hit in hits] if source_id else hits
    return best_score, matched


def prioritize_candidates(
    candidates: Sequence[Any], directives: Sequence[Mapping[str, Any]]
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Stable deterministic pre-oracle prioritization and per-candidate audit."""
    scored = []
    audit = []
    for index, candidate in enumerate(candidates):
        score, matched = directive_match_score(candidate, directives)
        candidate_id = None
        if isinstance(candidate, Mapping):
            candidate_id = candidate.get("candidate_id") or candidate.get("generation_id")
        else:
            candidate_id = getattr(candidate, "_candidate_id", None)
            if candidate_id is None and isinstance(getattr(candidate, "properties", None), Mapping):
                candidate_id = candidate.properties.get("_candidate_id")
        candidate_id = str(candidate_id) if candidate_id is not None else f"index_{index:06d}"
        scored.append((score, index, candidate_id, candidate))
        audit.append({"candidate_id": candidate_id, "score": score, "matched": matched, "original_index": index})
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [item[3] for item in scored], [next(entry for entry in audit if entry["candidate_id"] == item[2] and entry["original_index"] == item[1]) for item in scored]


# Public compatibility aliases make the typed boundary easy to use from
# notebooks and downstream campaign adapters without duplicating schemas.
TransferableRecord = TransferableMemoryRecord
extract_features = extract_transferable_features
is_applicable = applicability_check
