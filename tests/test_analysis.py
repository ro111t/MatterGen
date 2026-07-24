import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataclasses import dataclass, field
from typing import Any, Dict

from agents.analysis import AnalysisAgent


@dataclass
class FakeScreeningResult:
    structure_id: str
    predictions: Dict[str, float] = field(default_factory=dict)
    passes_filters: bool = True
    score: float = 50.0


@dataclass
class FakeValidationResult:
    structure_id: str
    converged: bool = True
    properties: Dict[str, float] = field(default_factory=dict)
    cost_hours: float = 1.0


def test_analysis_computes_mae():
    a = AnalysisAgent(properties_to_compare=["formation_energy"])

    screening_results = [
        (None, FakeScreeningResult("s1", predictions={"formation_energy": -2.0})),
        (None, FakeScreeningResult("s2", predictions={"formation_energy": -3.0})),
    ]
    validation_results = [
        FakeValidationResult("s1", properties={"formation_energy": -2.5}),
        FakeValidationResult("s2", properties={"formation_energy": -3.5}),
    ]

    result = a.analyze_batch(screening_results, validation_results)

    assert result.num_validated == 2
    assert result.num_converged == 2
    assert "formation_energy" in result.ml_vs_dft_mae
    # MAE should be 0.5 for both
    assert abs(result.ml_vs_dft_mae["formation_energy"] - 0.5) < 1e-6
    assert result.top_candidate_id in ("s1", "s2")


def test_analysis_no_validation_data():
    a = AnalysisAgent()
    result = a.analyze_batch([], [])
    assert result.num_validated == 0
    assert not result.ml_vs_dft_mae
    assert result.insights


def test_analysis_detects_bias():
    a = AnalysisAgent(properties_to_compare=["formation_energy"])
    screening_results = [
        (None, FakeScreeningResult("s1", predictions={"formation_energy": -1.0})),
        (None, FakeScreeningResult("s2", predictions={"formation_energy": -1.0})),
    ]
    validation_results = [
        FakeValidationResult("s1", properties={"formation_energy": -3.0}),
        FakeValidationResult("s2", properties={"formation_energy": -3.0}),
    ]

    result = a.analyze_batch(screening_results, validation_results)
    assert any("overestimates" in fm.lower() for fm in result.failure_modes)
