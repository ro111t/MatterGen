"""
GenerationAgent: Generates material structure candidates.

Primary production path: Microsoft MatterGen diffusion model, loaded from a
HuggingFace pretrained checkpoint or a local checkpoint directory.
Fallback path: pymatgen-based randomized crystal structures.

MatterGen requires a compatible PyTorch / numpy environment and the model
sampling configs packaged with the library. If either is missing, the agent
gracefully falls back to the mock generator and logs the reason.
"""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from math import ceil
import random
import tempfile
import numpy as np
from dataclasses import dataclass

try:
    from pymatgen.core import Structure, Lattice, Element
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False
    print("[Generator] pymatgen not found — minimal stub structures will be used")


# Optional MatterGen check — defer the heavy import until it is actually used.
import importlib.util

HAS_MATTERGEN = importlib.util.find_spec("mattergen") is not None and HAS_PYMATGEN


# Common solid electrolyte prototype compositions for mock generation
_PROTOTYPE_COMPOSITIONS = {
    "li_solid_electrolyte": [
        ("Li", "P", "S"),
        ("Li", "P", "S", "Cl"),
        ("Li", "P", "S", "Br"),
        ("Li", "La", "Zr", "O"),
        ("Li", "Al", "Ti", "P", "O"),
        ("Li", "Ge", "P", "S"),
        ("Li", "Si", "P", "S"),
        ("Li", "N", "H"),
    ],
    "na_solid_electrolyte": [
        ("Na", "Zr", "Si", "P", "O"),
        ("Na", "Al", "Si", "O"),
        ("Na", "P", "S"),
        ("Na", "La", "Zr", "O"),
    ],
    "thermoelectric": [
        ("Bi", "Te"),
        ("Pb", "Te"),
        ("Ge", "Te"),
        ("Si", "Ge"),
        ("Co", "Sb"),
    ],
    "battery_cathode": [
        ("Li", "Ni", "Mn", "Co", "O"),
        ("Li", "Fe", "P", "O"),
        ("Li", "Mn", "O"),
        ("Li", "Co", "O"),
    ],
}





class MattergenGenerator:
    """Thin wrapper around Microsoft's MatterGen CrystalGenerator."""

    def __init__(
        self,
        pretrained_name: Optional[str] = "mattergen_base",
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 16,
        properties_to_condition_on: Optional[Dict[str, Any]] = None,
        target_compositions: Optional[List[Dict[str, int]]] = None,
        config_overrides: Optional[List[str]] = None,
        sampling_config_path: Optional[str] = None,
        sampling_config_name: str = "default",
    ):
        if not HAS_MATTERGEN:
            raise ImportError(
                "MatterGen is not installed or failed to import. "
                "Install the official mattergen package to use this generator."
            )

        self.pretrained_name = pretrained_name
        self.model_path = model_path
        self.device = device
        if batch_size < 1:
            raise ValueError("MatterGen batch_size must be at least 1")
        self.batch_size = batch_size
        self.properties_to_condition_on = properties_to_condition_on or {}
        self.target_compositions = target_compositions or []
        self.config_overrides = config_overrides or []
        self.sampling_config_name = sampling_config_name
        self.sampling_config_path = self._resolve_sampling_config_path(sampling_config_path)
        self._generator: Any = None
        self._load_model()

    @staticmethod
    def _resolve_sampling_config_path(path: Optional[str]) -> Path:
        if path:
            return Path(path).resolve()
        return Path(__file__).parent / "mattergen_sampling_conf"

    def _load_model(self) -> None:
        from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
        from mattergen.generator import CrystalGenerator

        # Restrict element vocabulary to the user's desired chemical system.
        # Also ensure hardcoded training paths that are absent from the pip wheel
        # are populated from the bundled data files shipped with this project.
        try:
            import shutil
            import mattergen.common.utils.globals as g
            pkg_scale_file = Path(g.MODELS_PROJECT_ROOT) / "common" / "gemnet" / "gemnet-dT.json"
            bundled_scale_file = self.sampling_config_path / "gemnet-dT.json"
            if bundled_scale_file.exists() and not pkg_scale_file.exists():
                pkg_scale_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(bundled_scale_file, pkg_scale_file)
        except Exception as e:
            print(f"  [Generator] Note: could not bootstrap GemNet scale file to package directory ({e}); relying on Hydra config override")


        scale_file_path = (self.sampling_config_path / "gemnet-dT.json").as_posix()
        overrides = self.config_overrides + [
            "++lightning_module.diffusion_module.model.element_mask_func={_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}",
            f"++lightning_module.diffusion_module.model.gemnet.scale_file={scale_file_path}",
        ]

        if self.model_path:
            checkpoint_info = MatterGenCheckpointInfo(
                model_path=Path(self.model_path).resolve(),
                load_epoch="last",
                config_overrides=overrides,
            )
        else:
            checkpoint_info = MatterGenCheckpointInfo.from_hf_hub(
                self.pretrained_name,
                config_overrides=overrides,
            )

        self._generator = CrystalGenerator(
            checkpoint_info=checkpoint_info,
            batch_size=self.batch_size,
            properties_to_condition_on=self.properties_to_condition_on,
            target_compositions_dict=self.target_compositions,
            record_trajectories=False,
            sampling_config_path=self.sampling_config_path,
            sampling_config_name=self.sampling_config_name,
        )
        # Verify the sampling config is actually present so we can fall back
        # early instead of failing halfway through the campaign.
        self._generator.load_sampling_config(
            batch_size=self.batch_size, num_batches=1
        )

    def generate(
        self,
        num_candidates: int,
        elements: Optional[List[str]] = None,
        target_properties: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Generate up to num_candidates structures."""
        if num_candidates < 1:
            return []

        num_batches = ceil(num_candidates / self.batch_size)

        properties = dict(self.properties_to_condition_on)
        if target_properties:
            properties.update(target_properties)

        target_comps = list(self.target_compositions)
        if elements and not target_comps and not properties:
            # Best-effort chemical-system conditioning when no other conditioning
            # is supplied. The "chemical_system" model handles this natively.
            system = "-".join(sorted(elements))
            properties["chemical_system"] = system

        with tempfile.TemporaryDirectory() as tmpdir:
            self._generator.properties_to_condition_on = properties
            structures = self._generator.generate(
                batch_size=self.batch_size,
                num_batches=num_batches,
                target_compositions_dict=target_comps,
                output_dir=Path(tmpdir),
            )
        # MatterGen produces whole batches. Keep the adapter contract exact when
        # the requested count is not a multiple of the MatterGen batch size.
        return list(structures)[:num_candidates]


class GenerationAgent:
    """
    Generates candidate material structures.
    Primary path: MatterGen diffusion model (when enabled and available).
    Fallback path: pymatgen-based randomized crystal structures.
    """

    def __init__(
        self,
        diversity_threshold: float = 0.85,
        use_mattergen: bool = False,
        mattergen_pretrained: str = "mattergen_base",
        mattergen_model_path: Optional[str] = None,
        mattergen_batch_size: int = 16,
        mattergen_sampling_config_path: Optional[str] = None,
        mattergen_sampling_config_name: str = "default",
    ):
        self.generation_history = []
        self.diversity_threshold = diversity_threshold
        self._total_generated = 0
        self.use_mattergen = use_mattergen
        self._mattergen: Optional[MattergenGenerator] = None
        self.last_generation_backend: Optional[str] = None
        self._generation_batch_backends: List[str] = []

        if self.use_mattergen:
            try:
                self._mattergen = MattergenGenerator(
                    pretrained_name=mattergen_pretrained,
                    model_path=mattergen_model_path,
                    batch_size=mattergen_batch_size,
                    sampling_config_path=mattergen_sampling_config_path,
                    sampling_config_name=mattergen_sampling_config_name,
                )
                print(f"  [Generator] MatterGen backend loaded ({mattergen_pretrained})")
            except Exception as e:
                print(f"  [Generator] MatterGen unavailable ({e}); falling back to pymatgen mock")
                self.use_mattergen = False

    def generate_batch(
        self,
        elements: List[str],
        num_candidates: int = 15,
        seed: int = 42,
        domain: str = "",
    ) -> List[Any]:
        """
        Generate num_candidates structures using the given elements.

        Args:
            elements: Element symbols to build compositions from (e.g. ['Li','P','S','Cl'])
            num_candidates: How many structures to produce
            seed: Random seed for reproducibility
            domain: Optional domain hint for prototype selection

        Returns:
            List of pymatgen Structure objects (or stub dicts if pymatgen unavailable)
        """
        backend = "pymatgen_mock"
        if self.use_mattergen and self._mattergen is not None:
            try:
                structures = self._mattergen.generate(num_candidates, elements=elements)
                backend = "mattergen"
            except Exception as e:
                print(
                    f"  [Generator] MatterGen generation failed ({e}); "
                    "falling back to pymatgen mock for this batch"
                )
                structures = self._generate_pymatgen_fallback(elements, num_candidates, seed)
        else:
            structures = self._generate_pymatgen_fallback(elements, num_candidates, seed)

        self.last_generation_backend = backend
        self._generation_batch_backends.append(backend)
        self._total_generated += len(structures)
        self.generation_history.extend(structures)
        return structures

    def _generate_pymatgen_fallback(self, elements: List[str],
                                     num_candidates: int,
                                     seed: int) -> List[Any]:
        """Use pymatgen mock (or stub) with a deterministic RNG."""
        rng = random.Random(seed)
        if HAS_PYMATGEN:
            return self._generate_pymatgen_structures(elements, num_candidates, rng)
        return self._generate_stub_structures(elements, num_candidates, rng)

    def _generate_pymatgen_structures(self, elements: List[str],
                                       num_candidates: int,
                                       rng: random.Random) -> List[Any]:
        """Generate realistic mock structures using pymatgen."""
        structures = []
        valid_elements = self._filter_valid_elements(elements)
        if not valid_elements:
            valid_elements = ['Li', 'P', 'S']

        for i in range(num_candidates):
            try:
                struct = self._build_random_structure(valid_elements, rng, i)
                structures.append(struct)
            except Exception:
                structures.append(self._build_minimal_structure(valid_elements, rng, i))

        return structures

    def _filter_valid_elements(self, elements: List[str]) -> List[str]:
        """Keep only elements pymatgen recognises."""
        valid = []
        for el in elements:
            try:
                Element(el)
                valid.append(el)
            except Exception:
                pass
        return valid

    def _build_random_structure(self, elements: List[str],
                                 rng: random.Random, idx: int) -> Any:
        """Build a random crystal structure with given elements."""
        n_formula_units = rng.choice([1, 2, 4])
        n_elem_types = rng.randint(2, min(4, len(elements)))
        chosen = rng.sample(elements, n_elem_types)

        # Random stoichiometry (small integers)
        stoich = [rng.choice([1, 2, 3, 4]) for _ in chosen]
        species = []
        coords = []
        for el, n in zip(chosen, stoich):
            for _ in range(n * n_formula_units):
                species.append(el)
                coords.append([rng.random(), rng.random(), rng.random()])

        # Random cubic-ish lattice (3-12 Å)
        a = rng.uniform(3.5, 12.0)
        b = rng.uniform(3.5, 12.0)
        c = rng.uniform(3.5, 12.0)
        alpha = rng.uniform(60, 120)
        beta  = rng.uniform(60, 120)
        gamma = rng.uniform(60, 120)

        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        struct = Structure(lattice, species, coords)
        return struct

    def _build_minimal_structure(self, elements: List[str],
                                  rng: random.Random, idx: int) -> Any:
        """Fallback: simple cubic with 2 species."""
        el1 = elements[0]
        el2 = elements[1] if len(elements) > 1 else elements[0]
        a = rng.uniform(4.0, 8.0)
        lattice = Lattice.cubic(a)
        return Structure(lattice, [el1, el2],
                         [[0, 0, 0], [0.5, 0.5, 0.5]])

    def _generate_stub_structures(self, elements: List[str],
                                   num_candidates: int,
                                   rng: random.Random) -> List[Dict[str, Any]]:
        """Fallback when pymatgen is unavailable — returns dicts."""
        structs = []
        for i in range(num_candidates):
            chosen = rng.sample(elements, min(3, len(elements)))
            stoich = [rng.randint(1, 4) for _ in chosen]
            formula = ''.join(f"{e}{s}" for e, s in zip(chosen, stoich))
            structs.append({
                'composition': formula,
                'lattice': np.eye(3) * rng.uniform(4.0, 8.0),
                'positions': np.random.rand(sum(stoich), 3),
                'elements': chosen,
                'generation_id': f"stub_{self._total_generated}_{i}"
            })
        return structs

    def get_statistics(self) -> Dict[str, Any]:
        """Generation statistics."""
        return {
            'total_generated': self._total_generated,
            'session_generated': len(self.generation_history),
            'last_generation_backend': self.last_generation_backend,
            'generation_batch_backends': list(self._generation_batch_backends),
        }
