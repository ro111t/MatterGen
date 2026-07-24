import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from agents.validation import ValidationAgent, ValidationResult


def test_mock_validation_returns_result():
    v = ValidationAgent(calculator="mock")
    stub = {
        "composition": "Li2PS3",
        "elements": ["Li", "P", "S"],
        "lattice": [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        "positions": [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [0.1, 0.1, 0.1]],
    }
    result = v.validate_structure(stub)

    assert isinstance(result, ValidationResult)
    assert result.converged
    assert "energy" in result.properties
    assert "stability" in result.properties
    assert result.cost_hours > 0
    assert result.structure_id


def test_batch_validation():
    v = ValidationAgent(calculator="mock", n_workers=1)
    stubs = [
        {
            "composition": f"Li{x}PS",
            "elements": ["Li", "P", "S"],
            "lattice": [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
            "positions": [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
        }
        for x in range(2, 4)
    ]
    results = v.batch_validate(stubs)

    assert len(results) == len(stubs)
    for r in results:
        assert r.converged


def test_backend_info():
    v = ValidationAgent(calculator="mock")
    info = v.get_backend_info()
    assert info["calculator"] == "mock"
    assert info["backend_type"] == "mock"
