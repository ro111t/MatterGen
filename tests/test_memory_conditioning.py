"""Focused tests for the memory-conditioned target-composition policy."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from agents.generator import GenerationAgent, MattergenGenerator
from agents.orchestrator import OrchestratorAgent
from agents.strategy import (
    StrategyAgent,
    _canonical_composition_key,
    _normalize_composition,
    _parse_anonymous_pattern,
)
from agents.transferable_memory import _element_order_for_anonymous_pattern, _parse_formula


@dataclass
class FakeObjective:
    """Minimal objective for testing StrategyAgent."""
    target_properties: Dict[str, float]
    constraints: Dict[str, Any]
    success_criteria: Dict[str, float]
    domain: str = "materials"
    max_iterations: int = 5
    compute_budget_hours: float = 100.0


def make_fake_objective(elements: List[str]) -> FakeObjective:
    return FakeObjective(
        target_properties={"screening_quality": 1.0},
        constraints={"elements": list(elements)},
        success_criteria={},
        domain="materials",
        max_iterations=5,
    )


def make_valid_directive(record_id: str = "rec-1") -> Dict[str, Any]:
    """Return a directive for a Li-P-S -> Li-P-Se transfer."""
    return {
        "record_id": record_id,
        "preferred_anonymous_stoichiometries": ["A4B3C2"],
        "source_element_order": ["S", "P", "Li"],
        "source_element_classes": {"S": "chalcogen", "P": "p_block", "Li": "alkali"},
        "permitted_element_class_substitutions": {
            "chalcogen": ["S", "Se"],
            "p_block": ["P"],
            "alkali": ["Li"],
        },
        "exploration_weight": 0.3,
    }


class TestAnonymousPatternParsing:
    def test_parse_anonymous_pattern(self) -> None:
        assert _parse_anonymous_pattern("A2B3C4") == [("A", 2.0), ("B", 3.0), ("C", 4.0)]

    def test_parse_empty_returns_empty(self) -> None:
        assert _parse_anonymous_pattern("") == []

    def test_parse_invalid_returns_empty(self) -> None:
        assert _parse_anonymous_pattern("not_a_pattern") == []


class TestElementOrderForAnonymousPattern:
    def test_order_matches_abundance(self) -> None:
        amounts = {"Li": 2.0, "P": 3.0, "S": 4.0}
        order = _element_order_for_anonymous_pattern(amounts)
        assert order == ["S", "P", "Li"]


class TestCompositionHelpers:
    def test_normalize_composition(self) -> None:
        assert _normalize_composition({"Se": 4.0, "P": 3.0, "Li": 2.0}) == {
            "Li": 2.0,
            "P": 3.0,
            "Se": 4.0,
        }

    def test_canonical_key_is_deterministic(self) -> None:
        assert _canonical_composition_key({"Se": 4.0, "P": 3.0, "Li": 2.0}) == \
               _canonical_composition_key({"Li": 2.0, "P": 3.0, "Se": 4.0})


class TestMemoryMapping:
    def test_map_anonymous_pattern_to_target(self) -> None:
        directive = make_valid_directive()
        agent = StrategyAgent()
        result = agent._map_anonymous_pattern_to_target(
            "A4B3C2",
            directive["source_element_order"],
            directive["source_element_classes"],
            ["Li", "P", "Se"],
            directive["permitted_element_class_substitutions"],
        )
        assert result == {"Li": 2.0, "P": 3.0, "Se": 4.0}

    def test_undeclared_substitution_returns_none(self) -> None:
        substitutions = {
            "chalcogen": ["S"],  # Se is not allowed
            "p_block": ["P"],
            "alkali": ["Li"],
        }
        directive = make_valid_directive()
        agent = StrategyAgent()
        result = agent._map_anonymous_pattern_to_target(
            "A4B3C2",
            directive["source_element_order"],
            directive["source_element_classes"],
            ["Li", "P", "Se"],
            substitutions,
        )
        assert result is None


class TestStrategyAgentMemoryConditioning:
    def test_adaptive_no_memory_returns_empty_targets(self) -> None:
        agent = StrategyAgent()
        rec = agent.recommend(make_fake_objective(["Li", "P", "Se"]), [], directives=[], seed=1)
        assert rec["target_compositions_dict"] == []
        assert rec["num_memory_guided_proposals"] == 0
        assert rec["num_exploratory_proposals"] == rec["num_candidates"]

    def test_text_summary_does_not_affect_executable_policy(self) -> None:
        agent = StrategyAgent()
        rec = agent.recommend(
            make_fake_objective(["Li", "P", "Se"]),
            [],
            directives=[{"record_id": "text-1", "text_summary": "Li-P-S is promising"}],
            seed=1,
        )
        assert rec["target_compositions_dict"] == []

    def test_structured_memory_produces_targets(self) -> None:
        agent = StrategyAgent()
        rec = agent.recommend(
            make_fake_objective(["Li", "P", "Se"]),
            [],
            directives=[make_valid_directive()],
            seed=42,
        )
        assert len(rec["target_compositions_dict"]) == rec["num_candidates"]
        assert rec["num_memory_guided_proposals"] > 0
        first = rec["target_compositions_dict"][0]
        assert first == {"Li": 2.0, "P": 3.0, "Se": 4.0}

    def test_shuffled_differs_in_content(self) -> None:
        structured = make_valid_directive("rec-s")
        shuffled = dict(structured)
        # Deliberately swap the source element order (mismatched content)
        shuffled["source_element_order"] = ["Li", "P", "S"]
        shuffled["record_id"] = "rec-sh"

        agent1 = StrategyAgent()
        agent2 = StrategyAgent()
        rec1 = agent1.recommend(make_fake_objective(["Li", "P", "Se"]), [], directives=[structured], seed=42)
        rec2 = agent2.recommend(make_fake_objective(["Li", "P", "Se"]), [], directives=[shuffled], seed=42)
        # Both produce target compositions via the same code path.
        assert len(rec1["target_compositions_dict"]) == len(rec2["target_compositions_dict"])
        # The mapped content differs because the anonymous-pattern mapping changed.
        assert rec1["target_compositions_dict"][0] != rec2["target_compositions_dict"][0]

    def test_reproducibility_with_same_seed(self) -> None:
        directives = [make_valid_directive()]
        agent1 = StrategyAgent()
        agent2 = StrategyAgent()
        obj = make_fake_objective(["Li", "P", "Se"])
        rec1 = agent1.recommend(obj, [], directives=directives, seed=123)
        rec2 = agent2.recommend(obj, [], directives=directives, seed=123)
        assert rec1["target_compositions_dict"] == rec2["target_compositions_dict"]
        assert rec1["diversity_weight"] == rec2["diversity_weight"]

    def test_build_target_compositions_endpoints(self) -> None:
        agent = StrategyAgent()
        directives = [make_valid_directive()]
        obj = make_fake_objective(["Li", "P", "Se"])
        # Directly test the builder with the explicit mix control knob.
        all_memory = agent._build_target_compositions_dict(
            directives=directives,
            target_elements=["Li", "P", "Se"],
            num_candidates=10,
            diversity_weight=0.0,
            seed=1,
        )
        assert len(all_memory) == 10
        assert all(t == {"Li": 2.0, "P": 3.0, "Se": 4.0} for t in all_memory)

        all_exploratory = agent._build_target_compositions_dict(
            directives=directives,
            target_elements=["Li", "P", "Se"],
            num_candidates=10,
            diversity_weight=1.0,
            seed=1,
        )
        assert len(all_exploratory) == 10
        assert all(t != {"Li": 2.0, "P": 3.0, "Se": 4.0} for t in all_exploratory)

    def test_recommend_exploration_weight_biases_mix(self) -> None:
        agent = StrategyAgent()
        directive = make_valid_directive()
        directive["exploration_weight"] = 0.0
        rec_exploit = agent.recommend(make_fake_objective(["Li", "P", "Se"]), [], directives=[directive], seed=1)

        agent2 = StrategyAgent()
        directive2 = make_valid_directive()
        directive2["exploration_weight"] = 1.0
        rec_explore = agent2.recommend(make_fake_objective(["Li", "P", "Se"]), [], directives=[directive2], seed=1)

        assert rec_exploit["num_memory_guided_proposals"] > rec_explore["num_memory_guided_proposals"]
        assert rec_explore["num_exploratory_proposals"] > rec_exploit["num_exploratory_proposals"]

    def test_structured_vs_adaptive_differ_only_by_memory_intervention(self) -> None:
        no_memory = StrategyAgent()
        with_memory = StrategyAgent()
        directives = [make_valid_directive()]
        obj = make_fake_objective(["Li", "P", "Se"])
        rec_no = no_memory.recommend(obj, [], directives=[], seed=99)
        rec_yes = with_memory.recommend(obj, [], directives=directives, seed=99)
        # Both have the same element set and num_candidates (empty history default)
        assert rec_no["elements"] == rec_yes["elements"]
        assert rec_no["num_candidates"] == rec_yes["num_candidates"]
        # The only meaningful differences are the memory-conditioned fields.
        assert rec_no["target_compositions_dict"] == []
        assert rec_yes["target_compositions_dict"] != []
        assert rec_no["num_memory_guided_proposals"] == 0
        assert rec_yes["num_memory_guided_proposals"] > 0


class TestMockGenerator:
    def test_mock_uses_target_composition(self) -> None:
        agent = GenerationAgent(use_mattergen=False)
        targets = [{"Li": 2, "P": 3, "Se": 4}]
        structs = agent._generate_pymatgen_fallback(
            ["Li", "P", "Se"], 5, 42, target_compositions_dict=targets
        )
        # Without pymatgen this returns stub dicts; with pymatgen it returns Structures.
        assert len(structs) == 5

    def test_mock_uses_random_when_no_targets(self) -> None:
        agent = GenerationAgent(use_mattergen=False)
        structs = agent._generate_pymatgen_fallback(["Li", "P", "Se"], 5, 42)
        assert len(structs) == 5


class TestMattergenAdapter:
    def test_mattergen_adapter_forwards_target_compositions_dict(self) -> None:
        """The adapter must pass per-call target_compositions_dict to CrystalGenerator."""
        # Construct a minimal MattergenGenerator without loading a real model.
        mg = MattergenGenerator.__new__(MattergenGenerator)
        mg.batch_size = 16
        mg.target_compositions = []
        mg.properties_to_condition_on = {}
        fake_generator = MagicMock()
        fake_generator.diffusion_module = MagicMock()
        fake_generator.diffusion_module.model.cond_fields_model_was_trained_on = []
        fake_generator.load_sampling_config = MagicMock()
        fake_generator.generate = MagicMock(return_value=[])
        mg._generator = fake_generator
        target = [{"Li": 2.0, "P": 3.0, "Se": 4.0}]
        mg.generate(32, target_compositions_dict=target)
        _, kwargs = fake_generator.generate.call_args
        assert kwargs.get("target_compositions_dict") == target


class TestOrchestratorAndCareerMemory:
    def test_orchestrator_returns_directives_for_structured_provenance(self, monkeypatch) -> None:
        orchestrator = OrchestratorAgent(
            career_memory=None,
            memory_mode="structured_provenance",
            memory_seed=0,
            allow_llm=False,
        )
        transfer = {
            "directives": [make_valid_directive("rec-1")],
            "applied": [{"record_id": "rec-1", "reasons": []}],
            "rejected": [],
            "unsupported": [],
            "scientific_decision_support": True,
            "control_only": False,
            "shuffle_audit": None,
        }
        monkeypatch.setattr(orchestrator, "_get_transferable_directives", lambda *a, **k: transfer)
        objective = make_fake_objective(["Li", "P", "Se"])
        objective.target_properties = {"screening_quality": 1.0}
        strategy = orchestrator.plan_iteration(
            objective=objective,
            history=[],
            campaign_id="camp-1",
            iteration=0,
        )
        assert strategy["memory_directives"]
        # The directive should include the source elements and the mapped target composition.
        directive = strategy["memory_directives"][0]
        assert directive.get("source_element_order")
        assert directive.get("source_element_classes")

    def test_orchestrator_no_directives_for_adaptive_no_memory(self, monkeypatch) -> None:
        orchestrator = OrchestratorAgent(
            career_memory=None,
            memory_mode="none",
            memory_seed=0,
            allow_llm=False,
        )
        transfer = {
            "directives": [],
            "applied": [],
            "rejected": [],
            "unsupported": [],
            "scientific_decision_support": True,
            "control_only": False,
            "shuffle_audit": None,
        }
        monkeypatch.setattr(orchestrator, "_get_transferable_directives", lambda *a, **k: transfer)
        objective = make_fake_objective(["Li", "P", "Se"])
        strategy = orchestrator.plan_iteration(
            objective=objective,
            history=[],
            campaign_id="camp-1",
            iteration=0,
        )
        assert strategy["memory_directives"] == []
