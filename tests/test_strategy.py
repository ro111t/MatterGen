import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.strategy import StrategyAgent


class FakeObjective:
    def __init__(self, elements):
        self.constraints = {"elements": elements}


def test_recommendation_uses_allowed_elements():
    s = StrategyAgent()
    obj = FakeObjective(["Li", "P", "S"])
    rec = s.recommend(obj, [])

    assert isinstance(rec["elements"], list)
    assert all(el in ["Li", "P", "S"] for el in rec["elements"])
    assert 5 <= rec["num_candidates"] <= 100
    assert 0.0 <= rec["diversity_weight"] <= 1.0


def test_update_and_ucb_selection():
    s = StrategyAgent(exploration_weight=0.0)  # pure exploitation
    strategy = {"elements": ["Li", "P", "S"], "num_candidates": 10, "diversity_weight": 0.3}
    insights = {
        "num_passed": 2, "num_screened": 10,
        "num_validated": 2, "num_converged": 2, "num_synthesis_feasible": 2,
        "best_score": 70.0, "best_validated_stability": -1.0,
        "best_synthesis_feasibility": 0.6, "validation_cost_hours": 5.0,
    }

    s.update(0, strategy, insights)
    rec = s.recommend(FakeObjective(["Li", "P", "S"]), [])
    # With no competing arms and pure exploitation, should return the observed set
    assert tuple(sorted(rec["elements"])) == ("Li", "P", "S")


def test_batch_size_adaptation():
    s = StrategyAgent()
    # Low pass rate should increase batch size
    for i in range(3):
        s.update(
            i,
            {"elements": ["Li", "P", "S"], "num_candidates": 10, "diversity_weight": 0.4},
            {
                "num_passed": 0, "num_screened": 20,
                "num_validated": 0, "num_converged": 0, "num_synthesis_feasible": 0,
                "best_score": 0.0, "best_validated_stability": 0.0,
                "best_synthesis_feasibility": 0.0, "validation_cost_hours": 0.0,
            },
        )
    rec = s.recommend(FakeObjective(["Li", "P", "S"]), [])
    assert rec["num_candidates"] > 10
