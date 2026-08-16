"""Opt-in integration coverage for real MatterGen inference."""

from __future__ import annotations

import importlib.metadata
import os
import sys

import pytest

from agents.generator import GenerationAgent, HAS_MATTERGEN


@pytest.mark.mattergen_smoke
def test_mattergen_generates_a_real_pymatgen_structure():
    """Generate one structure with the actual MatterGen backend, never a mock."""
    if os.environ.get("RUN_MATTERGEN_SMOKE") != "1":
        pytest.skip("Set RUN_MATTERGEN_SMOKE=1 to download a checkpoint and run this test")

    if sys.version_info[:2] != (3, 10):
        pytest.fail(f"Expected Python 3.10, got {sys.version.split()[0]}")

    import numpy as np
    import torch
    from pymatgen.core import Structure

    assert np.__version__.startswith("1."), "MatterGen requires numpy<2.0"
    assert torch.__version__.startswith("2.4.1"), "Expected torch==2.4.1"
    assert HAS_MATTERGEN, "MatterGen must be installed and importable in the real runtime"
    try:
        assert importlib.metadata.version("mattergen") == "1.0.3", "Expected mattergen==1.0.3"
    except importlib.metadata.PackageNotFoundError:
        pytest.fail("mattergen package distribution is not installed in this environment")

    agent = GenerationAgent(
        use_mattergen=True,
        mattergen_pretrained="mattergen_base",
        mattergen_batch_size=1,
    )
    assert agent.use_mattergen, "MatterGen initialization fell back to the mock backend"

    structures = agent.generate_batch(
        elements=["Li", "P", "S"],
        num_candidates=1,
        seed=42,
    )

    assert agent.last_generation_backend == "mattergen"
    assert len(structures) == 1
    assert isinstance(structures[0], Structure)


if __name__ == "__main__":
    os.environ["RUN_MATTERGEN_SMOKE"] = "1"
    test_mattergen_generates_a_real_pymatgen_structure()
    print("MatterGen smoke test passed: successfully generated a real pymatgen.Structure.")

