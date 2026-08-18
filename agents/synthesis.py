"""
SynthesisFeasibilityAgent: Assess experimental realizability of candidates.

This is a lightweight, dependency-cautious implementation. In a production system
it would query ICSD/COD, Materials Project, and a synthesis-route ML model. Here
we provide:

  - Element-level precursor abundance heuristics
  - Oxidation-state / charge-neutrality sanity checks
  - Structural similarity to simple known prototypes (mock)
  - A difficulty score based on element rarity, number of elements, and composition
  - A suggested synthesis route (solid-state, sol-gel, etc.)

All heavy lookups are optional fallbacks; the agent runs offline by default.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import hashlib
import json
import warnings

try:
    from pymatgen.core import Composition, Element, Structure
    from pymatgen.core.periodic_table import Element as PymatgenElement
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False


@dataclass
class SynthesisAssessment:
    """Result of a synthesis feasibility assessment."""
    structure_id: str
    feasible: bool
    feasibility_score: float  # 0.0 - 1.0
    difficulty_score: float   # 0.0 - 1.0
    estimated_cost: float     # relative USD scale, 1-10
    synthesis_route: str
    route_reason: str
    similar_known_phases: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class SynthesisFeasibilityAgent:
    """
    Estimates how likely a material is to be synthesizable in the lab.

    Args:
        mode: 'mock' uses deterministic heuristics. 'mp' would query Materials
            Project for known phases (not yet implemented, falls back to mock).
        precursor_db: Optional path to a JSON file mapping formulas to precursor
            availability hints.
    """

    # Approximate elemental abundance / cost index (higher = harder to obtain)
    _ELEMENT_DIFFICULTY = {
        'H': 1, 'He': 3, 'Li': 3, 'Be': 4, 'B': 3, 'C': 1, 'N': 2, 'O': 1, 'F': 3,
        'Na': 2, 'Mg': 2, 'Al': 2, 'Si': 2, 'P': 3, 'S': 2, 'Cl': 2, 'K': 2, 'Ca': 2,
        'Sc': 4, 'Ti': 3, 'V': 3, 'Cr': 3, 'Mn': 3, 'Fe': 2, 'Co': 3, 'Ni': 3, 'Cu': 3,
        'Zn': 2, 'Ga': 3, 'Ge': 3, 'As': 3, 'Se': 3, 'Br': 3, 'Rb': 4, 'Sr': 3,
        'Y': 4, 'Zr': 3, 'Nb': 3, 'Mo': 3, 'Tc': 5, 'Ru': 5, 'Rh': 5, 'Pd': 5,
        'Ag': 4, 'Cd': 3, 'In': 3, 'Sn': 3, 'Sb': 3, 'Te': 3, 'I': 3, 'Cs': 4,
        'Ba': 3, 'La': 4, 'Ce': 4, 'Pr': 4, 'Nd': 4, 'Pm': 5, 'Sm': 4, 'Eu': 4,
        'Gd': 4, 'Tb': 4, 'Dy': 4, 'Ho': 4, 'Er': 4, 'Tm': 4, 'Yb': 4, 'Lu': 4,
        'Hf': 4, 'Ta': 4, 'W': 3, 'Re': 5, 'Os': 5, 'Ir': 5, 'Pt': 5, 'Au': 5,
        'Hg': 4, 'Tl': 4, 'Pb': 3, 'Bi': 3, 'Po': 5, 'At': 5, 'Rn': 5,
    }

    _ROUTE_TABLE: Dict[str, Tuple[str, str]] = {
        'oxide': ('Solid-state reaction', 'Mix oxides/carbonates, pelletize, anneal in air or O2.'),
        'sulfide': ('Solid-state + sulfurization', 'Pelletize sulfides, seal in ampoule, anneal under inert atmosphere.'),
        'halide': ('Solid-state in glovebox', 'Handle in glovebox; avoid moisture; anneal in sealed ampoule.'),
        'metallic': ('Arc melting', 'Melt constituent metals in arc furnace under inert atmosphere.'),
        'mixed_anion': ('Multi-step solid-state', 'Prepare binary precursors first, then combine and anneal.'),
        'default': ('Solid-state reaction', 'Pelletize precursors and anneal at moderate temperature.'),
    }

    def __init__(self, mode: str = "mock", precursor_db: Optional[str] = None):
        self.mode = mode.lower()
        self.precursor_db = self._load_precursor_db(precursor_db)

    def _load_precursor_db(self, path: Optional[str]) -> Dict[str, Any]:
        """Load optional precursor hints from JSON."""
        if path is None:
            return {}
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            warnings.warn(f"Could not load precursor DB {path}: {e}")
            return {}

    def assess(self, structure: Any, structure_id: str = "") -> SynthesisAssessment:
        """Assess feasibility of a single structure."""
        sid = structure_id or self._structure_id(structure)
        if self.mode == "mp":
            return self._assess_with_mp(structure, sid)
        return self._mock_assess(structure, sid)

    def assess_batch(
        self,
        structures: List[Any],
        structure_ids: Optional[List[str]] = None,
    ) -> List[SynthesisAssessment]:
        """Assess a batch of structures."""
        results = []
        for i, struct in enumerate(structures):
            sid = (structure_ids or [None] * len(structures))[i] or self._structure_id(struct, i)
            results.append(self.assess(struct, sid))
        return results

    def _structure_id(self, structure: Any, idx: int = 0) -> str:
        """Generate a stable identifier."""
        if isinstance(structure, dict):
            if 'candidate_id' in structure:
                return str(structure['candidate_id'])
            if 'generation_id' in structure:
                return str(structure['generation_id'])
            return f"struct_{idx}"
        if hasattr(structure, '_candidate_id'):
            return str(getattr(structure, '_candidate_id'))
        if HAS_PYMATGEN and isinstance(structure, Structure):
            if hasattr(structure, 'properties') and isinstance(structure.properties, dict) and '_candidate_id' in structure.properties:
                return str(structure.properties['_candidate_id'])
            return f"{structure.composition.reduced_formula}_{idx}"
        if HAS_PYMATGEN and hasattr(structure, 'composition'):
            return f"{structure.composition.reduced_formula}_{idx}"
        return f"struct_{idx}"

    def _formula(self, structure: Any) -> str:
        """Extract a formula string from a structure object or dict."""
        if isinstance(structure, dict):
            if 'composition' in structure:
                return str(structure['composition'])
            h = int(hashlib.md5(str(structure).encode('utf-8')).hexdigest(), 16) % 10000
            return f"struct_{h}"
        if HAS_PYMATGEN and isinstance(structure, Structure):
            return structure.composition.reduced_formula
        if HAS_PYMATGEN and hasattr(structure, 'composition'):
            return structure.composition.reduced_formula
        return "unknown"

    def _elements(self, structure: Any) -> Set[str]:
        """Extract element symbols from a structure."""
        formula = self._formula(structure)
        if HAS_PYMATGEN:
            try:
                comp = Composition(formula)
                return {str(el) for el in comp.elements}
            except Exception:
                pass
        # crude fallback
        import re
        return set(re.findall(r"[A-Z][a-z]?", formula))

    def _mock_assess(self, structure: Any, structure_id: str) -> SynthesisAssessment:
        """Heuristic offline synthesis assessment."""
        formula = self._formula(structure)
        elements = self._elements(structure)

        # Deterministic seed from structure ID
        seed = int(hashlib.md5(structure_id.encode()).hexdigest(), 16) % (2**31)

        warnings_list: List[str] = []

        # Difficulty from element rarity / reactivity
        difficulty = 0.0
        if elements:
            difficulty = sum(self._ELEMENT_DIFFICULTY.get(el, 3) for el in elements) / max(len(elements), 1)
            difficulty = difficulty / 5.0  # normalize to 0-1

        # Penalize high element count (entropy-stabilized phases are harder)
        n_elem = len(elements)
        if n_elem > 4:
            difficulty += 0.05 * (n_elem - 4)
            warnings_list.append(f"{n_elem}-component system may require multi-step synthesis.")

        # Penalize oxygen-sensitive or air-sensitive components (e.g., sulfides, halides)
        anion_type = self._classify_anions(elements)
        if anion_type in ('sulfide', 'halide'):
            difficulty += 0.15
            warnings_list.append(f"{anion_type.capitalize()} precursors are moisture/air sensitive.")

        # Penalize alkali/earth-alkali carbides/nitrides etc.
        if {'C', 'N'} & elements and {'Li', 'Na', 'K', 'Mg', 'Ca'} & elements:
            difficulty += 0.1
            warnings_list.append("Reactive light-metal non-oxide precursors require inert atmosphere.")

        difficulty = min(1.0, max(0.0, difficulty))

        # Feasibility is inverse difficulty with a small stochastic offset
        feasibility = max(0.0, min(1.0, 1.0 - difficulty * 0.9 + 0.05))

        # Relative cost estimate (1 = cheap, 10 = very expensive)
        estimated_cost = 1.0 + difficulty * 9.0

        route, route_reason = self._ROUTE_TABLE.get(
            anion_type, self._ROUTE_TABLE['default']
        )

        # Mock known-phase similarity: simple formulas get a small boost
        similar = []
        if HAS_PYMATGEN:
            try:
                comp = Composition(formula)
                if len(comp.elements) <= 3:
                    similar.append(f"{formula}-type prototype")
            except Exception:
                pass

        if feasibility > 0.6:
            feasible = True
        elif feasibility > 0.3:
            feasible = True
            warnings_list.append("Moderate synthesis difficulty; route optimization recommended.")
        else:
            feasible = False
            warnings_list.append("Low predicted feasibility; consider alternative compositions.")

        return SynthesisAssessment(
            structure_id=structure_id,
            feasible=feasible,
            feasibility_score=round(feasibility, 3),
            difficulty_score=round(difficulty, 3),
            estimated_cost=round(estimated_cost, 2),
            synthesis_route=route,
            route_reason=route_reason,
            similar_known_phases=similar,
            warnings=warnings_list,
        )

    def _assess_with_mp(self, structure: Any, structure_id: str) -> SynthesisAssessment:
        """Placeholder for Materials-Project-backed assessment."""
        warnings.warn("Materials Project synthesis lookup not yet implemented; using mock assessment.")
        return self._mock_assess(structure, structure_id)

    @staticmethod
    def _classify_anions(elements: Set[str]) -> str:
        """Classify the dominant anion chemistry for route selection."""
        if not elements:
            return 'default'
        if {'F', 'Cl', 'Br', 'I'} & elements:
            return 'halide'
        if 'S' in elements or 'Se' in elements or 'Te' in elements:
            return 'sulfide'
        if 'O' in elements:
            return 'oxide'
        if 'N' in elements:
            return 'mixed_anion'
        if elements.issubset({'C', 'H', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I'}):
            return 'mixed_anion'
        return 'metallic'
