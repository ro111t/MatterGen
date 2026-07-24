"""Tests for GenerationAgent and MatterGen integration."""

import pytest

from agents.generator import GenerationAgent, MattergenGenerator, HAS_MATTERGEN, HAS_PYMATGEN


@pytest.mark.parametrize("use_mattergen", [False, True])
def test_generation_agent_returns_candidates(use_mattergen, monkeypatch):
    """GenerationAgent should always return candidate structures, falling back if MatterGen fails."""
    if use_mattergen and not HAS_MATTERGEN:
        pytest.skip("MatterGen not installed")

    agent = GenerationAgent(
        use_mattergen=use_mattergen,
        mattergen_pretrained="mattergen_base",
        mattergen_batch_size=2,
    )
    candidates = agent.generate_batch(
        elements=["Li", "P", "S"],
        num_candidates=5,
        seed=42,
    )
    assert len(candidates) == 5
    # If MatterGen is enabled but failed to initialize, the agent should have
    # transparently fallen back to the pymatgen mock path.
    if use_mattergen:
        assert agent._mattergen is not None or not agent.use_mattergen


def test_generation_agent_pymatgen_path():
    """Explicitly disabling MatterGen should use pymatgen mock path."""
    agent = GenerationAgent(use_mattergen=False)
    candidates = agent.generate_batch(
        elements=["Li", "P", "S"],
        num_candidates=3,
        seed=1,
    )
    assert len(candidates) == 3
    assert not agent.use_mattergen
    assert agent._mattergen is None


@pytest.mark.skipif(HAS_MATTERGEN, reason="only run when MatterGen is unavailable")
def test_mattergen_generator_raises_when_unavailable():
    """MattergenGenerator should raise a clear error when MatterGen is missing."""
    with pytest.raises(ImportError):
        MattergenGenerator()


@pytest.mark.skipif(not HAS_MATTERGEN, reason="MatterGen not installed")
def test_mattergen_generator_falls_back_on_bad_config(monkeypatch):
    """If MatterGen imports succeed but the runtime config is broken, init should fail gracefully."""
    from mattergen.generator import CrystalGenerator

    def broken_load_sampling_config(self, *args, **kwargs):
        raise RuntimeError("simulated broken MatterGen environment")

    monkeypatch.setattr(CrystalGenerator, "load_sampling_config", broken_load_sampling_config)

    agent = GenerationAgent(
        use_mattergen=True,
        mattergen_pretrained="mattergen_base",
        mattergen_batch_size=2,
    )
    # The init-time sampling-config check should catch the broken env and fall back.
    assert agent.use_mattergen is False

    candidates = agent.generate_batch(elements=["Li", "P", "S"], num_candidates=3)
    assert len(candidates) == 3
