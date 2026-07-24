import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.synthesis import SynthesisFeasibilityAgent, SynthesisAssessment


def test_mock_assessment_returns_dataclass():
    s = SynthesisFeasibilityAgent(mode="mock")
    stub = {"composition": "Li2PS3"}
    result = s.assess(stub, structure_id="test_1")

    assert isinstance(result, SynthesisAssessment)
    assert result.structure_id == "test_1"
    assert 0.0 <= result.feasibility_score <= 1.0
    assert 0.0 <= result.difficulty_score <= 1.0
    assert result.synthesis_route
    assert result.estimated_cost >= 0.0


def test_batch_assessment():
    s = SynthesisFeasibilityAgent(mode="mock")
    structs = [{"composition": "Li2PS3"}, {"composition": "Li7P3S11"}]
    results = s.assess_batch(structs, structure_ids=["a", "b"])

    assert len(results) == 2
    for r in results:
        assert r.structure_id in ("a", "b")


def test_high_difficulty_for_many_elements():
    s = SynthesisFeasibilityAgent(mode="mock")
    # Six component system should be flagged as harder
    result = s.assess({"composition": "LiMnCoNiAlO2"}, structure_id="hea")
    assert result.difficulty_score > 0.2
    assert any("multi-step" in w.lower() or "component" in w.lower() for w in result.warnings)
