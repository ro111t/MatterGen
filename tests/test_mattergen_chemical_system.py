"""CPU contract tests using MatterGen's real mask, loaders and diffusion API."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.generator import GenerationAgent, MattergenGenerator
from experiments.spec import FIVE_CONDITIONS


@pytest.fixture
def backend(monkeypatch):
    torch = pytest.importorskip("torch")
    denoiser_api = pytest.importorskip("mattergen.denoiser")
    from pymatgen.core import Lattice, Structure

    denoiser = SimpleNamespace(
        element_mask_func=denoiser_api.mask_disallowed_elements,
        cond_fields_model_was_trained_on=[],
    )
    calls = []

    def generate(**kwargs):
        # No chemical_system field and no learned conditioning: unconditional.
        logits = denoiser.element_mask_func(torch.zeros(1, 101))
        calls.append((kwargs, logits.clone()))
        species = int(logits.argmax(-1).item()) + 1
        return [Structure(Lattice.cubic(5), [species], [[0, 0, 0]])
                for _ in range(kwargs["batch_size"] * kwargs["num_batches"])]

    crystal = SimpleNamespace(
        model=SimpleNamespace(diffusion_module=SimpleNamespace(model=denoiser)),
        generate=generate,
    )
    monkeypatch.setattr("agents.generator.HAS_MATTERGEN", True)
    monkeypatch.setattr(MattergenGenerator, "_load_model", lambda self: setattr(self, "_generator", crystal))
    adapter = MattergenGenerator()
    adapter.calls = calls
    return adapter


def test_unconditional_species_restriction_and_partial_batch(backend):
    import torch

    original = backend._generator.model.diffusion_module.model.element_mask_func
    result = backend.generate(5, elements=["Li", "P", "Se"])
    logits = backend.calls[0][1]
    probs = logits.softmax(-1)
    assert len(result) == 5
    assert set(torch.nonzero(probs[0]).flatten().tolist()) == {2, 14, 33}
    assert (probs[0, [20, 47, 65]] == 0).all()  # Sc, Cd, Dy
    assert all({e.symbol for e in s.composition.elements} <= {"Li", "P", "Se"} for s in result)
    assert backend._generator.model.diffusion_module.model.element_mask_func is original
    assert backend._generator.properties_to_condition_on == {}


@pytest.mark.parametrize("elements", [None, [], ["li"], ["Unobtainium"], ["Li", "X"], ["He"], [3]])
def test_invalid_elements_fail_before_sampling(backend, elements):
    with pytest.raises(ValueError):
        backend.generate(1, elements=elements)
    assert not backend.calls


def test_sequential_calls_and_failure_restore_original_mask(backend):
    denoiser = backend._generator.model.diffusion_module.model
    original = denoiser.element_mask_func
    backend.generate(1, elements=["Li", "P", "Se"])
    backend.generate(1, elements=["Na", "S"])
    assert set(backend.calls[1][1].softmax(-1)[0].nonzero().flatten().tolist()) == {10, 15}
    assert denoiser.element_mask_func is original
    backend._generator.generate = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("sampling failed"))
    with pytest.raises(RuntimeError, match="sampling failed"):
        backend.generate(1, elements=["Li"])
    assert denoiser.element_mask_func is original


@pytest.mark.parametrize("readonly", [False, True])
def test_mask_installation_failure_does_not_sample(backend, readonly):
    if readonly:
        class ReadOnlyDenoiser:
            cond_fields_model_was_trained_on = []

            @property
            def element_mask_func(self):
                from mattergen.denoiser import mask_disallowed_elements
                return mask_disallowed_elements

        backend._generator.model.diffusion_module.model = ReadOnlyDenoiser()
    else:
        backend._generator.model.diffusion_module.model.element_mask_func = None
    with pytest.raises((RuntimeError, AttributeError)):
        backend.generate(1, elements=["Li"])
    assert not backend.calls


def test_guidance_interpolation_and_one_based_logits(backend):
    import torch
    from mattergen.common.data.condition_factory import get_composition_data_loader
    from mattergen.property_embeddings import SetConditionalEmbeddingType, SetUnconditionalEmbeddingType

    def generate(**kwargs):
        graph, _ = next(iter(get_composition_data_loader([{"Li": 1}], 1, 1)))
        graph = graph.replace(chemical_system=["Li-P-Se"])
        hook = backend._generator.model.diffusion_module.model.element_mask_func
        logits = torch.zeros(1, 101)
        cond = hook(logits, SetConditionalEmbeddingType()(graph), torch.tensor([0]))
        uncond = hook(logits, SetUnconditionalEmbeddingType()(graph), torch.tensor([0]))
        for scale in [0.0, 1.0, 2.0]:
            guided = torch.lerp(uncond, cond, scale)
            assert torch.isfinite(guided).all()
            assert set(guided.softmax(-1)[0].nonzero().flatten().tolist()) == {2, 14, 33}
        one_based = hook(logits, predictions_are_zero_based=False)
        assert set(one_based.softmax(-1)[0].nonzero().flatten().tolist()) == {3, 15, 34}
        return []

    backend._generator.generate = generate
    backend.generate(1, elements=["Li", "P", "Se"])


def test_post_validation_checks_even_trimmed_outputs_and_research_never_falls_back(backend):
    from pymatgen.core import Lattice, Structure

    agent = GenerationAgent(use_mattergen=True, run_mode="research")
    original = backend._generator.model.diffusion_module.model.element_mask_func
    backend._generator.generate = lambda **kwargs: [
        Structure(Lattice.cubic(5), ["Li"], [[0, 0, 0]]),
        Structure(Lattice.cubic(5), ["Dy"], [[0, 0, 0]]),
    ]
    with pytest.raises(RuntimeError, match="no fallback is permitted.*chemical-system violation"):
        agent.generate_batch(["Li", "P", "Se"], num_candidates=1)
    assert not agent.generation_history
    assert agent.last_generation_backend is None
    assert backend._generator.model.diffusion_module.model.element_mask_func is original


@pytest.mark.parametrize("compositions", [[], [{"Li": 3, "P": 1, "Se": 4}]])
def test_development_compositions_do_not_change_common_mask(backend, compositions):
    backend.generate(1, elements=["Li", "P", "Se"], target_compositions_dict=compositions)
    kwargs, logits = backend.calls[0]
    assert kwargs["target_compositions_dict"] == compositions
    assert set(logits.softmax(-1)[0].nonzero().flatten().tolist()) == {2, 14, 33}


@pytest.mark.parametrize("constructor_default", [False, True])
def test_research_rejects_unverified_composition_intervention(backend, constructor_default):
    compositions = [{"Li": 3, "P": 1, "Se": 4}]
    agent = GenerationAgent(use_mattergen=True, run_mode="research")
    if constructor_default:
        agent._mattergen.target_compositions = compositions
    with pytest.raises(RuntimeError, match="runtime compositions.*not verified"):
        agent.generate_batch(["Li", "P", "Se"], num_candidates=1,
                             target_compositions_dict=None if constructor_default else compositions)
    assert not backend.calls
    assert not agent.generation_history


@pytest.mark.parametrize("condition", FIVE_CONDITIONS)
def test_campaign_propagates_same_restriction_for_all_conditions(backend, condition, tmp_path):
    from campaign import CampaignConfig, MaterialsDiscoveryCampaign
    from agents.orchestrator import CampaignObjective

    # Stop after the real campaign -> GenerationAgent -> adapter path, avoiding
    # unrelated screening. Memory retrieval is empty at this initial iteration.
    class GenerationObserved(Exception):
        pass

    class Campaign(MaterialsDiscoveryCampaign):
        def _init_screener(self):
            return SimpleNamespace()

    objective = CampaignObjective(target_properties={}, success_criteria={},
        constraints={"elements": ["Li", "P", "Se"], "experiment_condition": condition}, max_iterations=1)
    memory_modes = {"text_summary_memory": "text_summary",
                    "structured_provenance_memory": "structured_provenance",
                    "shuffled_memory_control": "shuffled_control"}
    campaign = Campaign(CampaignConfig(
        name=condition, objective=objective, output_dir=tmp_path / condition,
        locked_elements=["Li", "P", "Se"], use_mattergen=True,
        use_career_memory=condition in memory_modes,
        memory_mode=memory_modes.get(condition, "none"),
        use_validation=False, use_synthesis=False,
        require_thermodynamics=False, allow_llm_orchestration=False,
        strategy_mode="fixed" if condition == "random_mattergen" else "adaptive",
        proposal_budget=2, oracle_budget=2, num_candidates=2, verbose=False,
    ))
    original_generate = campaign.generator.generate_batch

    def observe(**kwargs):
        assert kwargs["elements"] == ["Li", "P", "Se"]
        result = original_generate(**kwargs)
        assert len(result) == 2
        assert set(backend.calls[-1][1].softmax(-1)[0].nonzero().flatten().tolist()) == {2, 14, 33}
        raise GenerationObserved

    campaign.generator.generate_batch = observe
    with pytest.raises(GenerationObserved):
        campaign._run_iteration()


def test_installed_runtime_compositions_are_replaced_by_base_atomic_diffusion():
    import importlib.metadata
    torch = pytest.importorskip("torch")
    api = pytest.importorskip("mattergen.generator")
    from omegaconf import OmegaConf
    from mattergen.diffusion.corruption.d3pm_corruption import D3PMCorruption
    from mattergen.diffusion.corruption.multi_corruption import MultiCorruption
    from mattergen.diffusion.d3pm.d3pm import MaskDiffusion, create_discrete_diffusion_schedule
    from mattergen.diffusion.d3pm.d3pm_predictors_correctors import D3PMAncestralSamplingPredictor
    from mattergen.diffusion.sampling.pc_sampler import _sample_prior

    assert importlib.metadata.version("mattergen") == "1.0.3", "Reverify the composition contract after upgrading MatterGen"
    # These are the pinned base config's atom-type diffusion settings. Exercise
    # installed runtime APIs without loading checkpoint weights or using a GPU.
    corruption = D3PMCorruption(MaskDiffusion(
        dim=101, schedule=create_discrete_diffusion_schedule(kind="standard", num_steps=1000)), offset=1)
    multi = MultiCorruption(discrete_corruptions={"atomic_numbers": corruption})
    config = OmegaConf.create({"lightning_module": {"diffusion_module": {
        "loss_fn": {"weights": {"atomic_numbers": 1.0}},
        "corruption": {"discrete_corruptions": {"atomic_numbers": {}}},
    }}})
    checkpoint = SimpleNamespace(config=config)
    compositions = [{"Li": 3, "P": 1, "Se": 4}]
    sampling_path = Path(__file__).resolve().parents[1] / "agents" / "mattergen_sampling_conf"
    # Constructor-time input is explicitly incompatible with atom denoising.
    with pytest.raises(AssertionError, match="not crystal structure prediction"):
        api.CrystalGenerator(checkpoint_info=checkpoint, target_compositions_dict=compositions,
                             sampling_config_path=sampling_path)
    crystal = api.CrystalGenerator(checkpoint_info=checkpoint, sampling_config_path=sampling_path)
    sampling = crystal.load_sampling_config(batch_size=1, num_batches=1, target_compositions_dict=compositions)
    assert "atomic_numbers" in sampling.sampler_partial.predictor_partials
    graph, mask = next(iter(crystal.get_condition_loader(sampling, compositions)))
    assert mask is None
    assert sorted(graph.atomic_numbers.tolist()) == [3, 3, 3, 15, 34, 34, 34, 34]
    prior = _sample_prior(multi, graph, mask)
    assert prior.num_atoms.tolist() == graph.num_atoms.tolist() == [8]
    assert prior.atomic_numbers.tolist() == [101] * 8  # All identities replaced by mask tokens.

    # The actual predictor can finish with a different stoichiometry. Force an
    # all-Li model score at the final step, which is allowed in the same system.
    logits = torch.full((8, 101), -1e10)
    logits[:, 2] = 0
    predictor = D3PMAncestralSamplingPredictor(corruption=corruption, score_fn=lambda *a: None)
    sample, mean = predictor.update_given_score(
        x=prior.atomic_numbers, t=torch.tensor([0.0]), dt=torch.tensor(-0.001),
        batch_idx=prior.get_batch_idx("atomic_numbers"), score=logits, batch=prior,
    )
    assert mean.tolist() == [3] * 8
    assert sample.tolist() == [3] * 8


def test_existing_hook_and_global_restrictions_are_preserved(backend):
    from mattergen.common.utils.globals import SELECTED_ATOMIC_NUMBERS

    before = tuple(SELECTED_ATOMIC_NUMBERS)
    denoiser = backend._generator.model.diffusion_module.model

    def existing_hook(logits, **kwargs):
        result = logits.clone()
        result[:, 14] = -1e10  # Additional existing restriction on P.
        return result

    denoiser.element_mask_func = existing_hook
    backend.generate(1, elements=["Li", "P", "Se"], target_properties={"chemical_system": "Dy-Sc-Cd"})
    assert set(backend.calls[0][1].softmax(-1)[0].nonzero().flatten().tolist()) == {2, 33}
    assert tuple(SELECTED_ATOMIC_NUMBERS) == before
    assert denoiser.element_mask_func is existing_hook
    assert backend._generator.properties_to_condition_on == {}


def test_research_type_error_does_not_retry_without_seed(backend):
    agent = GenerationAgent(use_mattergen=True, run_mode="research")
    calls = []

    def broken(**kwargs):
        calls.append(kwargs)
        raise TypeError("sampler failed")

    backend._generator.generate = broken
    with pytest.raises(RuntimeError, match="no fallback is permitted.*sampler failed"):
        agent.generate_batch(["Li"], num_candidates=1, seed=42)
    assert len(calls) == 1
