"""Dedicated Sprint 2 tests for geometry gating and dual budgets."""

from pathlib import Path
import json

import pytest

from agents.budget import DualBudgetTracker
from agents.geometry import GeometryFailureCode, GeometryValidator, validate_geometry
from agents.integrity import FORCE_KEY, STRESS_KEY
from agents.screening import ScreeningAgent
from campaign import CampaignConfig, MaterialsDiscoveryCampaign
from agents.orchestrator import CampaignObjective


def _dict_structure(formula="Li", positions=None, *, lattice=None, **extra):
    data = {
        "candidate_id": extra.pop("candidate_id", formula),
        "composition": formula,
        "lattice": lattice or [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
        "positions": positions if positions is not None else [[0.0, 0.0, 0.0]],
    }
    data.update(extra)
    return data


def test_valid_and_periodic_colliding_structures_have_typed_results():
    validator = GeometryValidator(min_distance=0.8)
    valid = validator.validate(_dict_structure("Li2", [[0, 0, 0], [0.5, 0.5, 0.5]]))
    assert valid.valid
    assert valid.code == GeometryFailureCode.VALID.value
    assert valid.minimum_distance is not None

    # The second atom is close through the periodic boundary (0.99 == -0.01).
    colliding = validator.validate(_dict_structure("Li2", [[0, 0, 0], [0.99, 0, 0]]))
    assert not colliding.valid
    assert colliding.code == GeometryFailureCode.PERIODIC_MIN_DISTANCE.value
    assert colliding.minimum_distance == pytest.approx(0.05, abs=1e-8)
    assert colliding.offending_pair == (0, 1)


def test_skew_cell_search_finds_short_image_beyond_componentwise_unit_stencil():
    # The shortest self image is a3 - 2*a1 - 2*a2, whose integer coefficients
    # exceed a conventional [-1, 0, 1] image stencil.
    lattice = [[10.0, 0, 0], [0.0, 10.0, 0], [20.1, 20.1, 0.5]]
    result = GeometryValidator(min_distance=1.2).validate(
        _dict_structure("Li", [[0, 0, 0]], lattice=lattice)
    )
    assert not result.valid
    assert result.code == GeometryFailureCode.PERIODIC_MIN_DISTANCE.value
    assert result.minimum_distance == pytest.approx((0.1 ** 2 + 0.1 ** 2 + 0.5 ** 2) ** 0.5, abs=1e-6)


def test_one_site_self_image_and_configured_threshold_are_recorded():
    structure = _dict_structure("Li", [[0, 0, 0]], lattice=[[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    assert GeometryValidator(min_distance=0.8).validate(structure).valid
    result = GeometryValidator(min_distance=1.1).validate(structure)
    assert not result.valid
    assert result.minimum_distance == pytest.approx(1.0)
    assert result.details["threshold"] == pytest.approx(1.1)


@pytest.mark.parametrize(
    "structure, code",
    [
        (_dict_structure("Li", [[float("nan"), 0, 0]]), GeometryFailureCode.NONFINITE_COORDINATES.value),
        (_dict_structure("Li", [[0, 0, 0]], lattice=[[0, 0, 0], [0, 5, 0], [0, 0, 5]]), GeometryFailureCode.NONPOSITIVE_CELL_VOLUME.value),
        (_dict_structure("", [[0, 0, 0]]), GeometryFailureCode.EMPTY_COMPOSITION.value),
    ],
)
def test_nonfinite_degenerate_and_empty_composition_fail_closed(structure, code):
    result = validate_geometry(structure)
    assert not result.valid
    assert result.code == code


def test_dict_representation_rejects_malformed_coordinates():
    result = validate_geometry(_dict_structure("Li", [[0, 0]]))
    assert not result.valid
    assert result.code == GeometryFailureCode.INVALID_COORDINATE_SHAPE.value


def test_composition_only_dict_is_invalid_in_canonical_validator():
    result = validate_geometry({"composition": "LiPS"})
    assert not result.valid
    assert result.code == GeometryFailureCode.INVALID_LATTICE_SHAPE.value


def test_research_composition_only_dict_never_reaches_predict(monkeypatch):
    # Bypass model initialization only to exercise the geometry gate itself.
    monkeypatch.setattr("agents.screening.ScreeningAgent._init_models", lambda self: None)
    agent = ScreeningAgent(run_mode="research")
    calls = []
    agent._predict = lambda *args: calls.append(args)
    result = agent.screen_batch([{"candidate_id": "composition-only", "composition": "LiPS"}], {})[0][1]
    assert calls == []
    assert result.failure_code == "INVALID_GEOMETRY"
    assert result.predictions == {}


def test_invalid_candidates_bypass_predict_and_have_no_energy(monkeypatch):
    monkeypatch.setattr("agents.screening.HAS_CHGNET", False)
    agent = ScreeningAgent()
    calls = []

    def should_not_run(*args):
        calls.append(args)
        return {FORCE_KEY: 0.0, STRESS_KEY: 0.0}

    agent._predict = should_not_run
    invalid = _dict_structure("Li2", [[0, 0, 0], [0, 0, 0]])
    result = agent.screen_batch([invalid], criteria={}, deduplicate=False)[0][1]
    assert calls == []
    assert not result.passes_filters
    assert result.failure_code == "INVALID_GEOMETRY"
    assert result.score == 0.0
    assert result.predictions == {}
    assert result.provenance_stage == "geometry_validation"
    assert result.details["minimum_distance"] == 0.0
    assert result.details["offending_pair"] == [0, 1]


def test_invalid_geometry_is_proposal_only_and_oracle_events_are_exact(monkeypatch):
    monkeypatch.setattr("agents.screening.HAS_CHGNET", False)
    agent = ScreeningAgent()
    calls = []

    def predictor(struct, sid):
        calls.append(sid)
        if sid == "failure":
            raise RuntimeError("oracle unavailable")
        predictions = {FORCE_KEY: 0.1, STRESS_KEY: 0.1}
        agent.prediction_cache[sid] = predictions
        return predictions

    agent._predict = predictor
    tracker = DualBudgetTracker(proposal_budget=5, oracle_budget=3)
    invalid = _dict_structure("Li2", [[0, 0, 0], [0, 0, 0]], candidate_id="invalid")
    valid = _dict_structure("Li", candidate_id="valid")
    cached = _dict_structure("Li", candidate_id="valid")
    failure = _dict_structure("Li", candidate_id="failure")
    for _ in range(4):
        tracker.record_proposals(1)
    results = agent.screen_batch([invalid, valid, cached, failure], {}, deduplicate=False, budget_tracker=tracker)
    by_id = {res.structure_id: res for _, res in results}
    assert by_id["invalid"].failure_code == "INVALID_GEOMETRY"
    assert by_id["failure"].failure_code == "PREDICTION_FAILED"
    assert tracker.geometry_valid == 3
    assert tracker.invalid_geometry == 1
    assert tracker.oracle_evaluations == 3
    assert tracker.oracle_cache_hits == 1
    assert calls == ["valid", "valid", "failure"]


def test_oracle_budget_exhaustion_is_explicit_and_does_not_predict(monkeypatch):
    monkeypatch.setattr("agents.screening.HAS_CHGNET", False)
    agent = ScreeningAgent()
    calls = []
    agent._predict = lambda struct, sid: calls.append(sid) or {FORCE_KEY: 0.1, STRESS_KEY: 0.1}
    tracker = DualBudgetTracker(oracle_budget=1)
    structures = [_dict_structure("Li", candidate_id="a"), _dict_structure("Li", candidate_id="b")]
    results = agent.screen_batch(structures, {}, deduplicate=False, budget_tracker=tracker)
    by_id = {res.structure_id: res for _, res in results}
    assert calls == ["a"]
    assert tracker.oracle_evaluations == 1
    assert by_id["b"].failure_code == "ORACLE_BUDGET_EXHAUSTED"
    assert by_id["b"].predictions == {}
    assert by_id["b"].score == 0.0


class _FixedGenerator:
    backend_name = "stub"

    def __init__(self):
        self.calls = []
        self.last_generation_backend = "stub"

    def generate_batch(self, elements, num_candidates=15, seed=42, **kwargs):
        self.calls.append(num_candidates)
        return [_dict_structure("Li", candidate_id=f"g{i}") for i in range(num_candidates)]


class _BudgetCampaign(MaterialsDiscoveryCampaign):
    def _init_generator(self):
        self.fixed_generator = _FixedGenerator()
        return self.fixed_generator


class _OverproducingGenerator(_FixedGenerator):
    def generate_batch(self, elements, num_candidates=15, seed=42, **kwargs):
        self.calls.append(num_candidates)
        return [_dict_structure("Li", candidate_id=f"over{i}") for i in range(num_candidates + 1)]


class _OverproducingCampaign(MaterialsDiscoveryCampaign):
    def _init_generator(self):
        self.fixed_generator = _OverproducingGenerator()
        return self.fixed_generator


def test_campaign_truncates_generation_and_preserves_budget_totals(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.screening.HAS_CHGNET", False)
    objective = CampaignObjective(
        target_properties={}, constraints={"elements": ["Li"]},
        success_criteria={"min_score": 999}, domain="budget", max_iterations=3,
    )
    campaign = _BudgetCampaign(CampaignConfig(
        name="budget", objective=objective, output_dir=tmp_path,
        use_career_memory=False, use_validation=False, use_synthesis=False,
        num_candidates=10, proposal_budget=3, oracle_budget=2, verbose=False,
    ))
    result = campaign.run_campaign()
    assert campaign.fixed_generator.calls == [3]
    assert result["proposals_generated"] == 3
    assert result["proposal_budget_remaining"] == 0
    assert result["oracle_evaluations"] == 2
    assert result["oracle_budget_remaining"] == 0
    assert result["termination_reason"] in {"PROPOSAL_BUDGET_EXHAUSTED", "ORACLE_BUDGET_EXHAUSTED"}
    assert len(campaign.provenance.records) == 3
    assert all(r.status for r in campaign.provenance.records.values())


def test_campaign_rejects_generator_overproduction_instead_of_slicing(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.screening.HAS_CHGNET", False)
    objective = CampaignObjective(
        target_properties={}, constraints={"elements": ["Li"]},
        success_criteria={"min_score": 999}, domain="overproduction", max_iterations=1,
    )
    campaign = _OverproducingCampaign(CampaignConfig(
        name="overproduction", objective=objective, output_dir=tmp_path,
        use_career_memory=False, use_validation=False, use_synthesis=False,
        num_candidates=2, proposal_budget=2, oracle_budget=2, verbose=False,
    ))
    with pytest.raises(RuntimeError, match="overproduction"):
        campaign.run_campaign()
    assert campaign.fixed_generator.calls == [2]
    assert campaign.budget_tracker.proposals_generated == 0


def test_manifest_and_reproduction_preserve_budget_limits(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.screening.HAS_CHGNET", False)
    objective = CampaignObjective(
        target_properties={}, constraints={"elements": ["Li"]},
        success_criteria={"min_score": 999}, domain="repro", max_iterations=1,
    )
    campaign = MaterialsDiscoveryCampaign(CampaignConfig(
        name="repro", objective=objective, output_dir=tmp_path,
        use_career_memory=False, use_validation=False, use_synthesis=False,
        num_candidates=1, proposal_budget=4, oracle_budget=2, verbose=False,
    ))
    campaign.run_campaign()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["proposal_budget"] == 4
    assert manifest["oracle_budget"] == 2
    assert manifest["config"]["proposal_budget"] == 4
    assert manifest["config"]["oracle_budget"] == 2
    assert manifest["proposals_generated"] == 1
    assert manifest["iteration_budget_counters"]

    reproduced = MaterialsDiscoveryCampaign.reproduce_from_manifest(
        tmp_path / "manifest.json", output_dir=tmp_path / "reproduced"
    )
    assert reproduced.config.proposal_budget == 4
    assert reproduced.config.oracle_budget == 2
    assert reproduced.budget_tracker.proposal_budget == 4
    assert reproduced.budget_tracker.oracle_budget == 2


def test_research_requires_explicit_positive_budgets(tmp_path):
    objective = CampaignObjective(target_properties={}, constraints={}, success_criteria={}, domain="research", max_iterations=1)
    with pytest.raises(ValueError, match="explicit positive"):
        CampaignConfig(name="missing", objective=objective, output_dir=tmp_path, run_mode="research")
    with pytest.raises(ValueError, match="positive integer"):
        CampaignConfig(name="zero", objective=objective, output_dir=tmp_path, run_mode="research", proposal_budget=0, oracle_budget=1)
    with pytest.raises(ValueError, match="positive integer"):
        CampaignConfig(name="negative", objective=objective, output_dir=tmp_path, run_mode="research", proposal_budget=1, oracle_budget=-1)
