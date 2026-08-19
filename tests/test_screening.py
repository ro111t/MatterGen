"""Tests for multi-objective screening and candidate deduplication."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from agents.screening import ScreeningAgent, ScreeningResult, DEFAULT_SCREENING_WEIGHTS


@pytest.fixture(autouse=True)
def disable_chgnet(monkeypatch):
    """Keep screening tests fast and deterministic by avoiding real CHGNet inference."""
    def _predict(self, struct, struct_id):
        return {
            "formation_energy": -2.0,
            "forces": 0.5,
            "stress": 0.05,
            "stability": -2.0,
        }

    monkeypatch.setattr(ScreeningAgent, "_predict", _predict)


class _FakeStructure:
    """Minimal stand-in for a pymatgen Structure."""

    def __init__(self, formula: str):
        self.composition = _FakeComposition(formula)


class _FakeComposition:
    def __init__(self, reduced_formula: str):
        self.reduced_formula = reduced_formula


def _make_stub(formula: str, generation_id: str = "") -> dict:
    return {
        "composition": formula,
        "generation_id": generation_id or formula,
    }


def test_screen_batch_returns_sorted_results_with_ranks():
    agent = ScreeningAgent()
    stubs = [
        _make_stub("LiP", "stub_0"),
        _make_stub("Li2S", "stub_1"),
        _make_stub("Li3PS4", "stub_2"),
    ]

    results = agent.screen_batch(stubs, criteria={})

    assert len(results) == len(stubs)
    scores = [r.score for _, r in results]
    assert scores == sorted(scores, reverse=True)
    for rank, (_, result) in enumerate(results, start=1):
        assert result.rank == rank
        assert result.passes_filters
        assert result.score_components


def test_deduplication_keeps_highest_scored_per_composition():
    agent = ScreeningAgent()
    stubs = [
        _make_stub("LiP", "a"),
        _make_stub("LiP", "b"),
        _make_stub("Li2S", "c"),
    ]

    results = agent.screen_batch(stubs, criteria={}, deduplicate=True)
    formulas = [r.structure_id.split("_")[0] if isinstance(r.structure_id, str) else r.structure_id for _, r in results]
    # The deduplication key is the composition string; two LiP stubs collapse to one.
    compositions = {agent._get_composition_key(s) for s, _ in results}
    assert len(compositions) == 2
    assert len(results) == 2


def test_no_deduplication_returns_all_structures():
    agent = ScreeningAgent()
    stubs = [
        _make_stub("LiP", "a"),
        _make_stub("LiP", "b"),
    ]

    results = agent.screen_batch(stubs, criteria={}, deduplicate=False)
    assert len(results) == 2


def test_target_property_match_affects_score():
    agent = ScreeningAgent()
    stub = _make_stub("LiP", "stub_0")

    # Patch the predictor so the stub carries a band_gap property we can target.
    agent._predict = lambda struct, sid: {
        "formation_energy": -2.0,
        "forces": 0.1,
        "stress": 0.1,
        "stability": -2.0,
        "band_gap": 2.5,
    }

    # Without a target, the property-match component is neutral.
    results_no_target = agent.screen_batch([stub], criteria={}, target_properties={})
    score_no_target = results_no_target[0][1].score

    results_with_target = agent.screen_batch(
        [stub], criteria={}, target_properties={"band_gap": 2.5}
    )
    score_with_target = results_with_target[0][1].score

    assert results_with_target[0][1].score_components["target_property_match"] == 100.0
    assert score_with_target > score_no_target


def test_custom_weights_change_score():
    agent = ScreeningAgent()
    stub = _make_stub("LiP", "stub_0")

    # Force a prediction where the components differ enough that changing weights changes the total.
    agent._predict = lambda struct, sid: {
        "formation_energy": -4.0,
        "forces": 1.0,
        "stress": 2.0,
        "stability": -4.0,
    }

    stability_focused = {"stability": 1.0, "relaxation_quality": 0.0, "target_property_match": 0.0, "composition_novelty": 0.0}
    relaxation_focused = {"stability": 0.0, "relaxation_quality": 1.0, "target_property_match": 0.0, "composition_novelty": 0.0}

    results_stable = agent.screen_batch([stub], criteria={}, weights=stability_focused)
    results_relaxation = agent.screen_batch([stub], criteria={}, weights=relaxation_focused)

    assert results_stable[0][1].score != results_relaxation[0][1].score
    assert results_stable[0][1].score_components["stability"] == 100.0
    assert results_stable[0][1].score > results_relaxation[0][1].score


def test_filter_reasons_rejected_candidates():
    agent = ScreeningAgent()
    stub = _make_stub("LiP", "stub_0")

    # The heuristic predictor produces formation energies in [-4, -0.5]; 0.0 is out of range
    # but we can force a rejection with a very low max_formation_energy threshold.
    # Instead, monkeypatch _predict to return an unphysical value.
    agent._predict = lambda struct, sid: {
        "formation_energy": 10.0,
        "forces": 0.1,
        "stress": 0.1,
        "stability": -10.0,
    }

    results = agent.screen_batch([stub], criteria={"max_formation_energy": 5.0})
    _, result = results[0]
    assert not result.passes_filters
    assert any("formation_energy" in reason for reason in result.filter_reasons)
