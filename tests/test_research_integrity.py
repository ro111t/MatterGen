"""Sprint 1 scientific execution-boundary and schema-v2 regression tests."""

import json

import pytest

from campaign import CampaignConfig, MaterialsDiscoveryCampaign
from agents.generator import GenerationAgent
from agents.integrity import (
    LEGACY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    LegacySchemaError,
    RunMode,
    ScientificPreflightError,
    ScientificValidity,
    assert_schema_v2_compatible,
    build_scientific_preflight,
    is_legacy_schema,
    normalize_run_mode,
)
from agents.orchestrator import CampaignObjective
from agents.provenance import CandidateRecord, RunManifest
from agents.screening import ScreeningAgent
from agents.synthesis import SynthesisFeasibilityAgent
from agents.validation import ValidationAgent


def _objective() -> CampaignObjective:
    return CampaignObjective(
        target_properties={},
        constraints={"elements": ["Li", "P", "S"]},
        success_criteria={"min_score": 999.0},
        domain="integrity_test",
        max_iterations=1,
    )


def test_run_mode_is_typed_and_development_is_default(tmp_path):
    config = CampaignConfig(name="dev", objective=_objective(), output_dir=tmp_path)
    assert config.run_mode is RunMode.DEVELOPMENT
    assert normalize_run_mode("RESEARCH") is RunMode.RESEARCH


def test_development_paths_keep_deterministic_fallbacks():
    generation = GenerationAgent(use_mattergen=False)
    candidates = generation.generate_batch(["Li", "P", "S"], num_candidates=1, seed=3)
    assert candidates
    assert generation.last_generation_backend in {"pymatgen_mock", "stub"}

    screening = ScreeningAgent(run_mode="development")
    screened = screening.screen_batch(candidates, criteria={}, deduplicate=False)
    predictions = screened[0][1].predictions
    assert "predicted_energy_per_atom_ev" in predictions or "mock_predicted_energy_per_atom_ev" in predictions
    assert "stability" not in predictions
    assert "formation_energy" not in predictions

    validation = ValidationAgent(calculator="mock")
    result = validation.validate_structure(candidates[0])
    assert result.converged
    assert "energy_per_atom_ev" in result.properties
    assert "stability" not in result.properties
    assert "formation_energy_per_atom" not in result.properties


def test_research_components_fail_closed():
    with pytest.raises(RuntimeError, match="MatterGen"):
        GenerationAgent(use_mattergen=False, run_mode="research")
    with pytest.raises(RuntimeError, match="validation|mock"):
        ValidationAgent(calculator="mock", run_mode="research")
    with pytest.raises(RuntimeError, match="Synthesis|development-only"):
        SynthesisFeasibilityAgent(mode="mock", run_mode="research")

    # CHGNet may be installed in some test environments.  A composition-only
    # dictionary is not a periodic structure and must fail geometry validation
    # before prediction, rather than being heuristically substituted.
    if ScreeningAgent.__module__:
        try:
            screener = ScreeningAgent(run_mode="research")
        except RuntimeError:
            pass
        else:
            calls = []
            screener._predict = lambda *args: calls.append(args)
            results = screener.screen_batch([{"composition": "LiPS"}], criteria={})
            assert calls == []
            assert results[0][1].geometry_failure_code == "INVALID_GEOMETRY"


def test_campaign_research_preflight_reports_missing_components_without_artifacts(tmp_path):
    with pytest.raises(ScientificPreflightError) as exc:
        MaterialsDiscoveryCampaign(
            CampaignConfig(
                name="research",
                objective=_objective(),
                output_dir=tmp_path,
                run_mode="research",
                proposal_budget=1,
                oracle_budget=1,
                use_mattergen=False,
                use_validation=False,
                use_synthesis=False,
            )
        )
    message = str(exc.value)
    assert "generation" in message.lower()
    assert "thermodynamics" in message.lower()
    assert not (tmp_path / "manifest.json").exists()


def test_research_preflight_allows_disabled_stages_but_requires_thermo_marker():
    report = build_scientific_preflight(
        run_mode="research",
        requested_backends={
            "generation": "mattergen",
            "screening": "chgnet",
            "validation": "disabled",
            "synthesis": "disabled",
        },
        actual_backends={
            "generation": "mattergen",
            "screening": "chgnet",
            "validation": "disabled",
            "synthesis": "disabled",
        },
        require_thermodynamics=True,
        thermodynamics_available=False,
    )
    assert not report.valid
    assert any("thermodynamics" in error for error in report.errors)
    assert not any("validation" in error for error in report.errors)


def test_schema_v2_validity_and_no_legacy_energy_labels():
    record = CandidateRecord(
        candidate_id="MAT-000001",
        screening_predictions={
            "formation_energy": -1.2,
            "stability": -1.2,
            "forces": 0.2,
            "stress": 0.1,
        },
        validation_properties={
            "formation_energy_per_atom": -1.1,
            "stability": -1.1,
            "forces": 0.1,
            "stress": 0.2,
        },
    )
    payload = record.to_dict()
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["scientific_validity"] == ScientificValidity.DEMO_ONLY.value
    assert "formation_energy" not in payload["screening_predictions"]
    assert "stability" not in payload["screening_predictions"]
    assert "formation_energy_per_atom" not in payload["validation_properties"]
    assert "stability" not in payload["validation_properties"]
    assert "-abs" not in json.dumps(payload)

    manifest = RunManifest()
    assert manifest.schema_version == SCHEMA_VERSION
    assert manifest.scientific_validity == ScientificValidity.DEMO_ONLY.value


def test_v1_is_readable_for_audit_but_quarantined():
    legacy = {"schema_version": LEGACY_SCHEMA_VERSION, "campaign_name": "old"}
    manifest = RunManifest.from_dict(legacy)
    assert manifest.schema_version == LEGACY_SCHEMA_VERSION
    assert manifest.scientific_validity == ScientificValidity.LEGACY_INVALID_ENERGY_SEMANTICS.value
    assert is_legacy_schema(legacy)
    with pytest.raises(LegacySchemaError):
        assert_schema_v2_compatible(legacy, context="test artifact")
