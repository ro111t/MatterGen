"""Tests for GenerationAgent backend selection and mock fallback."""

import pytest

from agents import generator
from agents.generator import GenerationAgent, MattergenGenerator, HAS_MATTERGEN


def test_generation_agent_returns_mock_candidates():
    """The default path stays offline and produces the requested candidate count."""
    agent = GenerationAgent(use_mattergen=False)
    candidates = agent.generate_batch(
        elements=["Li", "P", "S"],
        num_candidates=5,
        seed=42,
    )
    assert len(candidates) == 5
    assert agent.last_generation_backend in ("pymatgen_mock", "stub")


def test_generation_agent_pymatgen_path():
    """Explicitly disabling MatterGen should use pymatgen mock path (or stub)."""
    agent = GenerationAgent(use_mattergen=False)
    candidates = agent.generate_batch(
        elements=["Li", "P", "S"],
        num_candidates=3,
        seed=1,
    )
    assert len(candidates) == 3
    assert not agent.use_mattergen
    assert agent._mattergen is None
    assert agent.last_generation_backend in ("pymatgen_mock", "stub")


@pytest.mark.skipif(HAS_MATTERGEN, reason="only run when MatterGen is unavailable")
def test_mattergen_generator_raises_when_unavailable():
    """MattergenGenerator should raise a clear error when MatterGen is missing."""
    with pytest.raises(ImportError):
        MattergenGenerator()


def test_generation_agent_records_mattergen_when_it_succeeds(monkeypatch):
    """A successful adapter call is recorded as a real MatterGen batch."""
    class FakeMattergen:
        def __init__(self, **kwargs):
            pass

        def generate(self, num_candidates, elements=None):
            return [f"mattergen-{index}" for index in range(num_candidates)]

    monkeypatch.setattr(generator, "MattergenGenerator", FakeMattergen)
    agent = GenerationAgent(use_mattergen=True, mattergen_batch_size=2)

    candidates = agent.generate_batch(elements=["Li", "P", "S"], num_candidates=3)

    assert candidates == ["mattergen-0", "mattergen-1", "mattergen-2"]
    assert agent.last_generation_backend == "mattergen"


def test_mattergen_generator_rounds_up_and_trims_partial_batches():
    """The adapter should return exactly the count requested from a batch model."""
    class FakeCrystalGenerator:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return list(range(kwargs["batch_size"] * kwargs["num_batches"]))

    fake_generator = FakeCrystalGenerator()
    backend = object.__new__(MattergenGenerator)
    backend.batch_size = 2
    backend.properties_to_condition_on = {}
    backend.target_compositions = []
    backend._generator = fake_generator

    candidates = backend.generate(num_candidates=5, elements=["Li", "P", "S"])

    assert candidates == [0, 1, 2, 3, 4]
    assert fake_generator.calls[0]["num_batches"] == 3


def test_generation_agent_records_mock_after_mattergen_runtime_failure(monkeypatch):
    """A failed MatterGen call must not be reported as a MatterGen batch."""
    class FailingMattergen:
        def __init__(self, **kwargs):
            pass

        def generate(self, num_candidates, elements=None):
            raise RuntimeError("simulated MatterGen runtime failure")

    monkeypatch.setattr(generator, "MattergenGenerator", FailingMattergen)
    agent = GenerationAgent(use_mattergen=True, mattergen_batch_size=2)

    candidates = agent.generate_batch(elements=["Li", "P", "S"], num_candidates=3)

    assert len(candidates) == 3
    assert agent.last_generation_backend in ("pymatgen_mock", "stub")
    assert agent.use_mattergen is True


def test_generation_agent_assigns_stable_candidate_ids():
    """GenerationAgent should attach MAT-xxxxxx IDs at birth."""
    agent = GenerationAgent(use_mattergen=False)
    candidates = agent.generate_batch(elements=["Li", "P", "S"], num_candidates=3, seed=10)
    for i, c in enumerate(candidates):
        expected_id = f"MAT-{i + 1:06d}"
        if isinstance(c, dict):
            assert c.get("candidate_id") == expected_id
            assert c.get("generation_id") == expected_id
        else:
            cand_id = getattr(c, "_candidate_id", None) or (c.properties.get("_candidate_id") if hasattr(c, "properties") else None)
            assert cand_id == expected_id


def test_generation_agent_deterministic_replay():
    """Same seed should generate identical compositions and positions."""
    agent1 = GenerationAgent(use_mattergen=False)
    cands1 = agent1.generate_batch(elements=["Li", "P", "S"], num_candidates=4, seed=42)

    agent2 = GenerationAgent(use_mattergen=False)
    cands2 = agent2.generate_batch(elements=["Li", "P", "S"], num_candidates=4, seed=42)

    assert len(cands1) == len(cands2)
    for c1, c2 in zip(cands1, cands2):
        if isinstance(c1, dict):
            assert c1["composition"] == c2["composition"]
            assert c1["positions"] == c2["positions"]
            assert c1["candidate_id"] == c2["candidate_id"]
        else:
            assert c1.composition.reduced_formula == c2.composition.reduced_formula
            assert getattr(c1, "_candidate_id", None) == getattr(c2, "_candidate_id", None)


def test_backend_name_accuracy():
    """Backend name should accurately reflect whether pymatgen/mattergen is used."""
    from agents.generator import HAS_PYMATGEN
    agent = GenerationAgent(use_mattergen=False)
    if HAS_PYMATGEN:
        assert agent.backend_name == "pymatgen_mock"
    else:
        assert agent.backend_name == "stub"

