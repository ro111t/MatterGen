import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tempfile

from campaign import MaterialsDiscoveryCampaign, CampaignConfig
from agents.orchestrator import CampaignObjective
from agents.screening import ScreeningAgent, ScreeningResult
from agents.integrity import FORCE_KEY, SCREENING_ENERGY_KEY, STRESS_KEY


class FakeScreener:
    """Deterministic, CHGNet-free screener for fast integration tests."""

    def screen_batch(self, structures, criteria=None, **kwargs):
        results = []
        for i, struct in enumerate(structures):
            if isinstance(struct, dict):
                sid = struct.get("candidate_id") or struct.get("generation_id") or f"struct_{i}"
            elif hasattr(struct, "_candidate_id") and getattr(struct, "_candidate_id"):
                sid = getattr(struct, "_candidate_id")
            elif hasattr(struct, "properties") and isinstance(struct.properties, dict) and "_candidate_id" in struct.properties:
                sid = struct.properties["_candidate_id"]
            else:
                sid = f"struct_{i}"

            results.append((struct, ScreeningResult(
                structure_id=sid,
                predictions={
                    SCREENING_ENERGY_KEY: -2.0 - i * 0.1,
                    FORCE_KEY: 0.5,
                    STRESS_KEY: 0.05,
                },
                score=70.0 + i,
                passes_filters=True,
                filter_reasons=[],
            )))
        return results


class CampaignWithFakeScreener(MaterialsDiscoveryCampaign):
    """Not collected by pytest because it does not start with 'Test'."""

    def _init_screener(self):
        return FakeScreener()


def _make_campaign(tmp_path, **overrides):
    objective = CampaignObjective(
        target_properties={},
        constraints={"elements": ["Li", "P", "S"], "max_atoms": 10},
        success_criteria={"min_score": 999.0},
        domain="test_campaign",
        max_iterations=overrides.pop("max_iterations", 1),
    )

    config_kwargs = {
        "name": "test_campaign",
        "objective": objective,
        "output_dir": tmp_path,
        "use_career_memory": False,
        "verbose": False,
        "use_validation": True,
        "validation_top_k": 2,
        "use_synthesis": True,
        "num_candidates": 4,
    }
    config_kwargs.update(overrides)

    return CampaignWithFakeScreener(CampaignConfig(**config_kwargs))


def _pin_batch_size(campaign, n):
    """Force the orchestrator to use a small batch size so tests stay fast."""
    original_plan = campaign.orchestrator.plan_iteration

    def fixed_plan(*args, **kwargs):
        strategy = original_plan(*args, **kwargs)
        strategy["num_candidates"] = n
        return strategy

    campaign.orchestrator.plan_iteration = fixed_plan


def test_full_pipeline_runs_and_reports_expected_keys():
    with tempfile.TemporaryDirectory() as tmp:
        campaign = _make_campaign(Path(tmp))
        _pin_batch_size(campaign, 4)
        results = campaign.run_campaign()

        assert results["campaign_name"] == "test_campaign"
        assert results["iterations"] == 1
        assert results["total_generated"] == 4
        assert results["total_passed_screening"] == 4
        assert results["total_validated"] == 2
        assert results["total_converged"] == 2
        assert results["total_synthesis_feasible"] == 2
        expected_backend = campaign.generator.last_generation_backend
        assert results["generation_backend"] == expected_backend
        assert results["generation_backend_counts"] == {expected_backend: 1}
        assert results["thermodynamics_metrics_available"] is False
        assert "best_synthesis_feasibility_ever" in results


def test_pipeline_with_validation_disabled():
    with tempfile.TemporaryDirectory() as tmp:
        campaign = _make_campaign(
            Path(tmp),
            use_validation=False,
            use_synthesis=False,
        )
        _pin_batch_size(campaign, 4)
        results = campaign.run_campaign()

        assert results["total_validated"] == 0
        assert results["total_converged"] == 0
        assert results["total_synthesis_feasible"] == 0
        assert results["iterations"] == 1


def test_strategy_recommendation_updates():
    with tempfile.TemporaryDirectory() as tmp:
        campaign = _make_campaign(Path(tmp), max_iterations=2)
        _pin_batch_size(campaign, 4)
        campaign.run_campaign()

        # After at least one iteration, recommendations should be populated.
        assert campaign.current_recommendations is not None
        assert "elements" in campaign.current_recommendations
        assert "num_candidates" in campaign.current_recommendations
        assert campaign.strategy.outcomes


def test_report_uses_the_backend_that_generated_the_batch():
    with tempfile.TemporaryDirectory() as tmp:
        campaign = _make_campaign(Path(tmp), use_mattergen=True)
        _pin_batch_size(campaign, 2)

        # Simulate a configured MatterGen backend that fails at generation time.
        # Development mode permits the deterministic fallback.
        campaign.generator.use_mattergen = True
        campaign.generator._mattergen = object()

        results = campaign.run_campaign()

        expected_backend = campaign.generator.last_generation_backend
        assert results["generation_backend"] == expected_backend
        assert results["generation_backend_counts"] == {expected_backend: 1}
        assert results["mattergen_pretrained"] is None
