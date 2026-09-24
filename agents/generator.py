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
import re
import tempfile
import numpy as np
from dataclasses import dataclass

from agents.integrity import RunMode, normalize_run_mode

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
        run_mode: RunMode | str = RunMode.DEVELOPMENT,
    ):
        if not HAS_MATTERGEN:
            raise ImportError(
                "MatterGen is not installed or failed to import. "
                "Install the official mattergen package to use this generator."
            )

        self.pretrained_name = pretrained_name
        self.run_mode = normalize_run_mode(run_mode)
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

    @staticmethod
    def _local_checkpoint_target(path: str) -> Tuple[Path, Any]:
        # Inspect the supplied layout before resolving symlinks: HF snapshots
        # may link last.ckpt to a blob outside the model directory.
        supplied = Path(path).absolute()
        if supplied.is_dir():
            return supplied.resolve(), "last"
        if supplied.parent.name != "checkpoints" or supplied.suffix != ".ckpt":
            raise ValueError("Explicit MatterGen checkpoint must be <model_dir>/checkpoints/<epoch>.ckpt")
        if not supplied.is_file():
            raise ValueError(f"MatterGen checkpoint does not exist: {supplied}")
        model_dir = supplied.parent.parent.resolve()
        if not (model_dir / "config.yaml").is_file():
            raise ValueError(f"MatterGen model directory is missing config.yaml: {model_dir}")
        if supplied.name == "last.ckpt":
            epoch = "last"
        else:
            match = re.fullmatch(r"epoch=(\d+)(?:-[^/]+)?\.ckpt", supplied.name)
            if not match:
                raise ValueError(f"Unsupported explicit MatterGen checkpoint filename: {supplied.name}")
            epoch = int(match.group(1))

        # Mirror MatterGen's recursive selector, but reject multiple matches
        # instead of allowing filesystem iteration order to choose the artifact.
        matches = []
        for candidate in model_dir.rglob("*.ckpt"):
            if not candidate.is_file():
                continue
            if epoch == "last":
                selected = candidate.name.endswith("last.ckpt")
            elif candidate.name.endswith("last.ckpt"):
                continue
            else:
                try:
                    selected = int(candidate.name.split(".ckpt")[0].split("-")[0].split("=")[1]) == epoch
                except (ValueError, IndexError) as exc:
                    raise ValueError(f"Malformed MatterGen checkpoint filename: {candidate}") from exc
            if selected:
                matches.append(candidate)
        if len(matches) != 1 or matches[0].resolve() != supplied.resolve():
            raise ValueError(f"Ambiguous MatterGen checkpoint selection for {supplied}")
        return model_dir, epoch

    def _load_model(self) -> None:
        from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
        from mattergen.generator import CrystalGenerator

        # Bootstrap hardcoded training paths that are absent from the pip wheel
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
            model_dir, load_epoch = self._local_checkpoint_target(self.model_path)
            checkpoint_info = MatterGenCheckpointInfo(
                model_path=model_dir,
                load_epoch=load_epoch,
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
        seed: Optional[int] = None,
        target_compositions_dict: Optional[List[Dict[str, float]]] = None,
    ) -> List[Any]:
        """Generate up to num_candidates structures."""
        if num_candidates < 1:
            return []

        from mattergen.common.utils.globals import SELECTED_ATOMIC_NUMBERS

        if not elements:
            raise ValueError("MatterGen generation requires a nonempty allowed element set")
        if any(not isinstance(symbol, str) or not Element.is_valid_symbol(symbol)
               for symbol in elements):
            raise ValueError(f"Invalid MatterGen element symbols: {elements}")
        allowed_symbols = frozenset(elements)
        allowed_numbers = frozenset(Element(symbol).Z for symbol in allowed_symbols)
        if not allowed_numbers.issubset(SELECTED_ATOMIC_NUMBERS):
            raise ValueError(f"Requested elements violate MatterGen's global element restrictions: {elements}")

        target_comps = list(target_compositions_dict if target_compositions_dict is not None else self.target_compositions)
        if target_comps and self.run_mode == RunMode.RESEARCH:
            raise RuntimeError(
                "Research mode cannot use target_compositions_dict: runtime compositions "
                "are not verified fixed-composition conditioning for the MatterGen base "
                "atom-type-denoising sampler; the memory intervention requires validation."
            )

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2**32))
            try:
                import torch
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            except ImportError:
                pass

        num_batches = ceil(num_candidates / self.batch_size)

        # CrystalGenerator.model prepares the checkpoint and exposes the actual
        # denoiser. A missing/incompatible extension point must fail, not sample
        # without the declared chemical-system invariant.
        denoiser = self._generator.model.diffusion_module.model
        original_mask = denoiser.element_mask_func
        if not callable(original_mask):
            raise RuntimeError("Cannot install MatterGen element restriction: missing callable element_mask_func")
        allowed_cond = set(denoiser.cond_fields_model_was_trained_on)

        properties = dict(self.properties_to_condition_on)
        if target_properties:
            properties.update(target_properties)

        # Filter properties strictly to those the model checkpoint was trained on
        if allowed_cond:
            properties = {k: v for k, v in properties.items() if k in allowed_cond}
        else:
            properties = {}

        def restrict_elements(logits, x=None, batch_idx=None, predictions_are_zero_based=True):
            import torch
            from mattergen.denoiser import mask_disallowed_elements

            kwargs = dict(x=x, batch_idx=batch_idx,
                          predictions_are_zero_based=predictions_are_zero_based)
            masked = original_mask(logits=logits, **kwargs)
            # Retain global restrictions even if the existing hook is customized.
            masked = mask_disallowed_elements(
                logits=masked, predictions_are_zero_based=predictions_are_zero_based
            )
            numbers = torch.arange(masked.shape[-1], device=masked.device)
            if predictions_are_zero_based:
                numbers = numbers + 1
            keep = torch.zeros_like(numbers, dtype=torch.bool)
            for number in allowed_numbers:
                keep |= numbers == number
            # Match MatterGen's finite logit floor: classifier-free guidance
            # interpolates logits, for which -inf - -inf would produce NaNs.
            return masked.masked_fill(~keep, -1e10)

        try:
            denoiser.element_mask_func = restrict_elements
            if denoiser.element_mask_func is not restrict_elements:
                raise RuntimeError("Cannot install MatterGen element restriction")
            with tempfile.TemporaryDirectory() as tmpdir:
                self._generator.properties_to_condition_on = properties
                structures = list(self._generator.generate(
                    batch_size=self.batch_size,
                    num_batches=num_batches,
                    target_compositions_dict=target_comps,
                    output_dir=Path(tmpdir),
                ))
            # Check the complete backend output, including the final partial
            # batch, before trimming. Never discard or resample a violation.
            for index, structure in enumerate(structures):
                species = {element.symbol for element in structure.composition.elements}
                if not species or not species.issubset(allowed_symbols):
                    raise RuntimeError(
                        f"MatterGen chemical-system violation at output {index}: "
                        f"{sorted(species)} outside allowed {sorted(allowed_symbols)}"
                    )
        finally:
            denoiser.element_mask_func = original_mask
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
        run_mode: RunMode | str = RunMode.DEVELOPMENT,
        mode: Optional[str] = None,
    ):
        self.generation_history = []
        self.diversity_threshold = diversity_threshold
        self._total_generated = 0
        self.run_mode = normalize_run_mode(mode if mode is not None else run_mode)
        self.use_mattergen = use_mattergen
        self._mattergen: Optional[MattergenGenerator] = None
        self.last_generation_backend: Optional[str] = None
        self._generation_batch_backends: List[str] = []
        # Last applied CareerMemory directives are retained for provenance and
        # audit only; they do not bypass geometry or thermodynamic gates.
        self.last_memory_directives: List[Dict[str, Any]] = []

        if self.run_mode == RunMode.RESEARCH and not self.use_mattergen:
            raise RuntimeError(
                "Research mode requires MatterGen generation; refusing the development generator."
            )

        if self.use_mattergen:
            try:
                self._mattergen = MattergenGenerator(
                    pretrained_name=mattergen_pretrained,
                    model_path=mattergen_model_path,
                    batch_size=mattergen_batch_size,
                    sampling_config_path=mattergen_sampling_config_path,
                    sampling_config_name=mattergen_sampling_config_name,
                    run_mode=self.run_mode,
                )
                print(f"  [Generator] MatterGen backend loaded ({mattergen_pretrained})")
            except Exception as e:
                if self.run_mode == RunMode.RESEARCH:
                    raise RuntimeError(
                        f"Research mode requires an available MatterGen backend; initialization failed: {e}"
                    ) from e
                print(f"  [Generator] MatterGen unavailable ({e}); falling back to pymatgen mock")
                self.use_mattergen = False

    @property
    def backend_name(self) -> str:
        """Accurately report the active generation backend."""
        if self.run_mode == RunMode.RESEARCH and (not self.use_mattergen or self._mattergen is None):
            raise RuntimeError("Research mode cannot generate candidates without MatterGen.")

        if self.use_mattergen and self._mattergen is not None:
            return "mattergen"
        if HAS_PYMATGEN:
            return "pymatgen_mock"
        return "stub"

    def generate_batch(
        self,
        elements: List[str],
        num_candidates: int = 15,
        seed: int = 42,
        domain: str = "",
        memory_directives: Optional[List[Dict[str, Any]]] = None,
        directives: Optional[List[Dict[str, Any]]] = None,
        target_compositions_dict: Optional[List[Dict[str, float]]] = None,
        diversity_weight: float = 0.4,
    ) -> List[Any]:
        """
        Generate num_candidates structures using the given elements.

        Args:
            elements: Element symbols to build compositions from (e.g. ['Li','P','S','Cl'])
            num_candidates: How many structures to produce
            seed: Random seed for reproducibility
            domain: Optional domain hint for prototype selection
            target_compositions_dict: Optional list of target compositions for memory conditioning
            diversity_weight: Policy diversity weight used for provenance/audit

        Returns:
            List of pymatgen Structure objects (or stub dicts if pymatgen unavailable)
        """
        start_idx = self._total_generated
        self.last_memory_directives = list(memory_directives or directives or [])
        self.last_target_compositions_dict = list(target_compositions_dict or [])
        self.last_diversity_weight = float(diversity_weight)
        backend = "pymatgen_mock" if HAS_PYMATGEN else "stub"
        if self.use_mattergen and self._mattergen is not None:
            try:
                try:
                    structures = self._mattergen.generate(num_candidates, elements=elements, seed=seed, target_compositions_dict=target_compositions_dict)
                except TypeError:
                    if self.run_mode == RunMode.RESEARCH:
                        raise
                    structures = self._mattergen.generate(num_candidates, elements=elements, target_compositions_dict=target_compositions_dict)
                backend = "mattergen"
            except Exception as e:
                if self.run_mode == RunMode.RESEARCH:
                    raise RuntimeError(
                        f"MatterGen generation failed in research mode; no fallback is permitted: {e}"
                    ) from e
                print(
                    f"  [Generator] MatterGen generation failed ({e}); "
                    "falling back to mock for this batch"
                )
                structures = self._generate_pymatgen_fallback(elements, num_candidates, seed, start_idx=start_idx, target_compositions_dict=target_compositions_dict)
                backend = "pymatgen_mock" if HAS_PYMATGEN else "stub"
        else:
            structures = self._generate_pymatgen_fallback(elements, num_candidates, seed, start_idx=start_idx, target_compositions_dict=target_compositions_dict)

        # Tag each structure with an immutable candidate ID at birth
        for i, struct in enumerate(structures):
            cand_id = f"MAT-{start_idx + i + 1:06d}"
            if isinstance(struct, dict):
                struct['candidate_id'] = cand_id
                struct['generation_id'] = cand_id
            else:
                try:
                    setattr(struct, '_candidate_id', cand_id)
                except (AttributeError, TypeError):
                    pass
                if hasattr(struct, 'properties') and isinstance(struct.properties, dict):
                    struct.properties['_candidate_id'] = cand_id

        self.last_generation_backend = backend
        self._generation_batch_backends.append(backend)
        self._total_generated += len(structures)
        self.generation_history.extend(structures)
        return structures

    def _generate_pymatgen_fallback(self, elements: List[str],
                                     num_candidates: int,
                                     seed: int,
                                     start_idx: int = 0,
                                     target_compositions_dict: Optional[List[Dict[str, float]]] = None) -> List[Any]:
        """Use pymatgen mock (or stub) with a deterministic RNG."""
        rng = random.Random(seed)
        if HAS_PYMATGEN:
            return self._generate_pymatgen_structures(elements, num_candidates, rng, start_idx=start_idx, target_compositions_dict=target_compositions_dict)
        return self._generate_stub_structures(elements, num_candidates, rng, start_idx=start_idx, target_compositions_dict=target_compositions_dict)

    def _generate_pymatgen_structures(self, elements: List[str],
                                       num_candidates: int,
                                       rng: random.Random,
                                       start_idx: int = 0,
                                       target_compositions_dict: Optional[List[Dict[str, float]]] = None) -> List[Any]:
        """Generate realistic mock structures using pymatgen."""
        structures = []
        valid_elements = self._filter_valid_elements(elements)
        if not valid_elements:
            valid_elements = ['Li', 'P', 'S']

        target_compositions_dict = target_compositions_dict or []
        for i in range(num_candidates):
            target_comp = target_compositions_dict[i] if i < len(target_compositions_dict) else None
            try:
                struct = self._build_random_structure(valid_elements, rng, i, target_composition=target_comp)
                structures.append(struct)
            except Exception:
                structures.append(self._build_minimal_structure(valid_elements, rng, i, target_composition=target_comp))

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
                                 rng: random.Random, idx: int,
                                 target_composition: Optional[Dict[str, float]] = None) -> Any:
        """Build a random crystal structure with given elements.

        If target_composition is provided and all of its elements are available,
        it is used instead of a fully random stoichiometry.  This implements
        memory-guided mock generation while keeping the same lattice randomness.
        """
        n_formula_units = rng.choice([1, 2, 4])

        if (
            target_composition is not None
            and all(el in elements for el in target_composition)
            and any(v > 0 for v in target_composition.values())
        ):
            chosen = sorted([el for el in target_composition if target_composition[el] > 0])
            base_stoich = [max(1, int(round(float(target_composition[el])))) for el in chosen]
        else:
            n_elem_types = rng.randint(2, min(4, len(elements)))
            chosen = sorted(rng.sample(elements, n_elem_types))
            base_stoich = [rng.choice([1, 2, 3, 4]) for _ in chosen]

        stoich = [n * n_formula_units for n in base_stoich]
        species = []
        coords = []
        for el, n in zip(chosen, stoich):
            for _ in range(n):
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
                                  rng: random.Random, idx: int,
                                  target_composition: Optional[Dict[str, float]] = None) -> Any:
        """Fallback: simple cubic with 2 species."""
        if (
            target_composition is not None
            and all(el in elements for el in target_composition)
            and any(v > 0 for v in target_composition.values())
        ):
            chosen = sorted([el for el in target_composition if target_composition[el] > 0])[:2]
            if len(chosen) == 1:
                chosen = chosen * 2
        else:
            el1 = elements[0]
            el2 = elements[1] if len(elements) > 1 else elements[0]
            chosen = [el1, el2]
        a = rng.uniform(4.0, 8.0)
        lattice = Lattice.cubic(a)
        return Structure(lattice, chosen,
                         [[0, 0, 0], [0.5, 0.5, 0.5]])

    def _generate_stub_structures(self, elements: List[str],
                                   num_candidates: int,
                                   rng: random.Random,
                                   start_idx: int = 0,
                                   target_compositions_dict: Optional[List[Dict[str, float]]] = None) -> List[Dict[str, Any]]:
        """Fallback when pymatgen is unavailable — returns dicts with deterministic rng positions."""
        structs = []
        target_compositions_dict = target_compositions_dict or []
        for i in range(num_candidates):
            target_comp = target_compositions_dict[i] if i < len(target_compositions_dict) else None
            if (
                target_comp is not None
                and all(el in elements for el in target_comp)
                and any(v > 0 for v in target_comp.values())
            ):
                chosen = sorted([el for el in target_comp if target_comp[el] > 0])
                stoich = [max(1, int(round(float(target_comp[el])))) for el in chosen]
            else:
                chosen = sorted(rng.sample(elements, min(3, len(elements))))
                stoich = [rng.randint(1, 4) for _ in chosen]
            formula = ''.join(f"{e}{s}" for e, s in zip(chosen, stoich))
            total_atoms = sum(stoich)
            positions = [[rng.random(), rng.random(), rng.random()] for _ in range(total_atoms)]
            lattice_scaling = rng.uniform(4.0, 8.0)
            lattice = [[lattice_scaling, 0.0, 0.0], [0.0, lattice_scaling, 0.0], [0.0, 0.0, lattice_scaling]]
            cand_id = f"MAT-{start_idx + i + 1:06d}"
            structs.append({
                'composition': formula,
                'lattice': lattice,
                'positions': positions,
                'elements': chosen,
                'generation_id': cand_id,
                'candidate_id': cand_id,
            })
        return structs

    def get_statistics(self) -> Dict[str, Any]:
        """Generation statistics."""
        return {
            'total_generated': self._total_generated,
            'session_generated': len(self.generation_history),
            'last_generation_backend': self.last_generation_backend,
            'generation_batch_backends': list(self._generation_batch_backends),
            'backend_name': self.backend_name,
        }
