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
    def _extract_id(c):
        if isinstance(c, dict):
            return c.get("candidate_id")
        return getattr(c, "_candidate_id", None) or (c.properties.get("_candidate_id") if hasattr(c, "properties") else None)

    assert _extract_id(candidates[0]) == "MAT-000001"
    assert _extract_id(candidates[1]) == "MAT-000002"

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


def test_multi_iteration_candidate_id_incrementing():
    """Candidate IDs should strictly increment across consecutive generate_batch calls without resetting or collisions."""
    gen = GenerationAgent(use_mattergen=False)
    def _extract_id(c):
        if isinstance(c, dict):
            return c.get("candidate_id")
        return getattr(c, "_candidate_id", None) or (c.properties.get("_candidate_id") if hasattr(c, "properties") else None)

    batch1 = gen.generate_batch(elements=["Li", "P", "S"], num_candidates=3, seed=1)
    batch2 = gen.generate_batch(elements=["Li", "P", "S", "Cl"], num_candidates=3, seed=2)

    ids1 = [_extract_id(c) for c in batch1]
    ids2 = [_extract_id(c) for c in batch2]

    assert ids1 == ["MAT-000001", "MAT-000002", "MAT-000003"]
    assert ids2 == ["MAT-000004", "MAT-000005", "MAT-000006"]
    assert len(set(ids1 + ids2)) == 6


def test_synthesis_assessment_is_deterministic():
    """Synthesis assessment must be deterministic across different agent instances."""
    synth1 = SynthesisFeasibilityAgent(mode="mock")
    synth2 = SynthesisFeasibilityAgent(mode="mock")

    stub_struct = {
        "composition": "Li3PS4",
        "candidate_id": "MAT-000001",
        "generation_id": "MAT-000001",
        "positions": [[0.0, 0.0, 0.0]],
        "lattice": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
    }

    res1 = synth1.assess(stub_struct, "MAT-000001")
    res2 = synth2.assess(stub_struct, "MAT-000001")

    assert res1.feasible == res2.feasible
    assert res1.feasibility_score == res2.feasibility_score
    assert res1.difficulty_score == res2.difficulty_score
    assert res1.synthesis_route == res2.synthesis_route
    assert res1.structure_id == "MAT-000001"
    assert res2.structure_id == "MAT-000001"

