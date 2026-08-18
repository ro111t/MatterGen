"""Tests verifying determinism in screening and ID preservation across all pipeline agents."""

import pytest
from agents.generator import GenerationAgent
from agents.screening import ScreeningAgent
from agents.validation import ValidationAgent
from agents.synthesis import SynthesisFeasibilityAgent


def test_screening_is_deterministic():
    """Screening scores must be invariant across different agent instances."""
    screener1 = ScreeningAgent()
    screener2 = ScreeningAgent()

    stub_struct = {
        "composition": "Li3PS4",
        "candidate_id": "MAT-000001",
        "generation_id": "MAT-000001",
        "positions": [[0.0, 0.0, 0.0]],
        "lattice": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
    }

    res1 = screener1.screen_batch([stub_struct], criteria={})
    res2 = screener2.screen_batch([stub_struct], criteria={})

    assert res1[0][1].score == res2[0][1].score
    assert res1[0][1].predictions == res2[0][1].predictions
    assert res1[0][1].structure_id == "MAT-000001"
    assert res2[0][1].structure_id == "MAT-000001"


def test_candidate_id_preserved_across_all_stages():
    """MAT-xxxxxx ID assigned at generation should be preserved across screening, validation, and synthesis."""
    gen = GenerationAgent(use_mattergen=False)
    screener = ScreeningAgent()
    validator = ValidationAgent(calculator="mock")
    synthesis = SynthesisFeasibilityAgent(mode="mock")

    candidates = gen.generate_batch(elements=["Li", "P", "S"], num_candidates=2, seed=42)
    assert len(candidates) == 2
    cand0_id = candidates[0].get("candidate_id") if isinstance(candidates[0], dict) else getattr(candidates[0], "_candidate_id", None)
    cand1_id = candidates[1].get("candidate_id") if isinstance(candidates[1], dict) else getattr(candidates[1], "_candidate_id", None)
    assert cand0_id == "MAT-000001"
    assert cand1_id == "MAT-000002"

    # Screen
    screened = screener.screen_batch(candidates, criteria={})
    screened_ids = [res.structure_id for _, res in screened]
    assert set(screened_ids) == {"MAT-000001", "MAT-000002"}

    # Validate
    validated = validator.batch_validate(candidates)
    validated_ids = [v.structure_id for v in validated]
    assert validated_ids == ["MAT-000001", "MAT-000002"]

    # Synthesis
    assessed = synthesis.assess_batch(candidates, validated_ids)
    assessed_ids = [a.structure_id for a in assessed]
    assert assessed_ids == ["MAT-000001", "MAT-000002"]
