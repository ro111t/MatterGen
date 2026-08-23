"""
test_provenance.py — Unit and integration tests for candidate provenance tracking,
lifecycle state transitions, CIF persistence, and data export integrity.
"""

import csv
import json
from pathlib import Path
import tempfile
import pytest

from agents.provenance import (
    CandidateStatus,
    CandidateRecord,
    RunManifest,
    SoftwareEnvironment,
    ProvenanceTracker,
    extract_candidate_id,
    extract_formula_and_elements,
)
from campaign import MaterialsDiscoveryCampaign, CampaignConfig
from agents.orchestrator import CampaignObjective
from agents.screening import ScreeningResult
from agents.validation import ValidationResult
from agents.synthesis import SynthesisAssessment


def test_candidate_record_serialization_and_flat_dict():
    """Verify CandidateRecord round-trip serialization and flat dict export."""
    record = CandidateRecord(
        candidate_id="MAT-000137",
        campaign_id="camp_test_001",
        iteration=1,
        composition="Li3PS4",
        chemical_system="Li-P-S",
        elements=["Li", "P", "S"],
        num_elements=3,
        structure_path="structures/MAT-000137.cif",
        structure_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        generation_backend="pymatgen_mock",
        generation_seed=43,
        target_elements=["Li", "P", "S"],
        screening_backend="chgnet",
        screening_predictions={"formation_energy": -2.15, "stability": -0.25, "forces": 0.04},
        screening_score=88.5,
        passes_screening_filters=True,
        screening_filter_reasons=[],
        screening_rank=1,
        validation_calculator="mock",
        validation_converged=True,
        validation_properties={"energy_per_atom": -5.12, "stability": -0.22},
        validation_cost_hours=1.2,
        synthesis_mode="mock",
        synthesis_feasible=True,
        synthesis_feasibility_score=0.85,
        synthesis_difficulty_score=0.25,
        synthesis_estimated_cost=3.5,
        synthesis_route="solid_state",
        status=CandidateStatus.ACCEPTED.value,
        ranking_score=88.5,
        iteration_rank=1,
        stored_in_memory=True,
        strategy_influence="Incorporate P-S polyhedra for high conductivity",
    )

    # 1. Test to_dict / from_dict
    data = record.to_dict()
    assert data["candidate_id"] == "MAT-000137"
    assert data["status"] == "accepted"
    assert data["screening_predictions"]["formation_energy"] == -2.15

    restored = CandidateRecord.from_dict(data)
    assert restored.candidate_id == record.candidate_id
    assert restored.screening_score == 88.5
    assert restored.stored_in_memory is True

    # 2. Test to_flat_dict for CSV export
    flat = record.to_flat_dict()
    assert flat["candidate_id"] == "MAT-000137"
    assert flat["elements"] == "Li;P;S"
    assert flat["screening_formation_energy"] == -2.15
    assert flat["validation_energy_per_atom"] == -5.12
    assert flat["synthesis_route"] == "solid_state"
    assert flat["stored_in_memory"] is True


def test_software_environment_capture():
    """Verify SoftwareEnvironment safely captures host platform and package metadata."""
    env = SoftwareEnvironment.capture()
    assert env.python_version != ""
    assert env.os_name != ""
    assert isinstance(env.packages, dict)
    assert "pytest" in env.packages
    assert env.packages["pytest"] is not None

    env_dict = env.to_dict()
    assert "python_version" in env_dict
    assert "os_name" in env_dict
    assert "packages" in env_dict


def test_extract_formula_and_elements_without_pymatgen():
    """Verify element parsing works correctly from strings and dicts even without pymatgen."""
    # Test string parsing
    form, els, sys_str = extract_formula_and_elements("Li3PS4")
    assert form == "Li3PS4"
    assert els == ["Li", "P", "S"]
    assert sys_str == "Li-P-S"

    # Test complex formula
    form, els, sys_str = extract_formula_and_elements("La2Zr2O7")
    assert form == "La2Zr2O7"
    assert els == ["La", "O", "Zr"]

    # Test dictionary stub without elements list
    form, els, sys_str = extract_formula_and_elements({"composition": "Na3SbS4"})
    assert form == "Na3SbS4"
    assert els == ["Na", "S", "Sb"]


def test_manifest_creation_and_persistence():
    """Verify manifest.json is created at campaign start with complete configuration and seeds."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tracker = ProvenanceTracker(
            campaign_id="camp_manifest_test",
            campaign_name="test_campaign",
            domain="li_solid_electrolyte",
            output_dir=tmp_path,
            master_seed=1234,
            config={"num_candidates": 5},
            objective={"stability": -0.1},
            constraints={"elements": ["Li", "P", "S"]},
        )
        manifest_path = tracker.write_manifest()
        assert manifest_path.exists()

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        assert manifest_data["campaign_id"] == "camp_manifest_test"
        assert manifest_data["domain"] == "li_solid_electrolyte"
        assert manifest_data["master_seed"] == 1234
        assert manifest_data["status"] == "running"
        assert "environment" in manifest_data


def test_full_candidate_lifecycle_tracking():
    """Verify candidate record transitions through all 7 lifecycle stages."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tracker = ProvenanceTracker(
            campaign_id="camp_lifecycle_test",
            campaign_name="test_campaign",
            domain="li_solid_electrolyte",
            output_dir=tmp_path,
            master_seed=42,
        )

        dummy_candidates = [
            {
                "candidate_id": "MAT-000001",
                "composition": "Li3PS4",
                "lattice": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
                "positions": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            }
        ]

        # Stage 1: Generation
        ids = tracker.register_generation(
            candidates=dummy_candidates,
            iteration=0,
            backend="stub",
            seed=42,
            target_elements=["Li", "P", "S"],
        )
        assert ids == ["MAT-000001"]
        rec = tracker.records["MAT-000001"]
        assert rec.status == CandidateStatus.GENERATED.value
        cif_file = tmp_path / "structures" / "MAT-000001.cif"
        assert cif_file.exists()
        assert rec.structure_hash is not None

        # Verify CIF fractional coordinate validity
        with open(cif_file, "r", encoding="utf-8") as f:
            cif_text = f.read()
        assert "_atom_site_fract_x" in cif_text
        assert "0.50000 0.50000 0.50000" in cif_text

        # Stage 2: Screening
        screening_res = [
            (dummy_candidates[0], ScreeningResult(
                structure_id="MAT-000001",
                predictions={"formation_energy": -2.1, "stability": -0.2},
                score=85.0,
                passes_filters=True,
                filter_reasons=[],
                rank=1,
            ))
        ]
        tracker.record_screening(screening_res, criteria={}, iteration=0, backend="chgnet")
        assert rec.status == CandidateStatus.SCREENED.value
        assert rec.screening_score == 85.0

        # Stage 3: Validation
        val_res = [
            ValidationResult(
                structure_id="MAT-000001",
                structure=dummy_candidates[0],
                calculator="mock",
                converged=True,
                properties={"stability": -0.19, "energy_per_atom": -4.8},
                cost_hours=0.5,
            )
        ]
        tracker.record_validation(val_res, iteration=0)
        assert rec.status == CandidateStatus.VALIDATED.value
        assert rec.validation_converged is True

        # Stage 4: Synthesis
        synth_res = [
            SynthesisAssessment(
                structure_id="MAT-000001",
                feasible=True,
                feasibility_score=0.9,
                difficulty_score=0.2,
                estimated_cost=2.0,
                synthesis_route="solid_state",
                route_reason="Standard precursor availability",
            )
        ]
        tracker.record_synthesis(synth_res, iteration=0, mode="mock")
        assert rec.status == CandidateStatus.SYNTHESIS_ASSESSED.value
        assert rec.synthesis_feasible is True

        # Stage 5: Ranking
        tracker.record_ranking([("MAT-000001", 85.0, 1)], iteration=0)
        assert rec.status == CandidateStatus.RANKED.value
        assert rec.ranking_score == 85.0
        assert rec.iteration_rank == 1

        # Stage 6/7: Final Decision & Memory
        tracker.record_decision(
            candidate_id="MAT-000001",
            status=CandidateStatus.ACCEPTED,
            ranking_score=85.0,
            iteration_rank=1,
            stored_in_memory=True,
            strategy_influence="High priority target composition",
        )
        assert rec.status == CandidateStatus.ACCEPTED.value
        assert rec.stored_in_memory is True


def test_failed_candidate_rejection_reasons():
    """Verify that candidates failing screening, validation, or synthesis preserve numerical rejection reasons."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tracker = ProvenanceTracker(
            campaign_id="camp_failures_test",
            campaign_name="failures_campaign",
            domain="li_solid_electrolyte",
            output_dir=tmp_path,
            master_seed=42,
        )

        candidates = [
            {"candidate_id": "MAT-000010", "composition": "Li10S1"},
            {"candidate_id": "MAT-000011", "composition": "Li2P1"},
            {"candidate_id": "MAT-000012", "composition": "Li1P1S10"},
        ]

        tracker.register_generation(
            candidates=candidates,
            iteration=0,
            backend="stub",
            seed=42,
            target_elements=["Li", "P", "S"],
        )

        # 1. Candidate 10 fails screening with numerical reason
        screening_res = [
            (candidates[0], ScreeningResult(
                structure_id="MAT-000010",
                predictions={"formation_energy": -0.31, "stability": 0.45},
                score=35.0,
                passes_filters=False,
                filter_reasons=["formation_energy -0.31 > -0.50 eV/atom threshold"],
                rank=3,
            )),
            (candidates[1], ScreeningResult(
                structure_id="MAT-000011",
                predictions={"formation_energy": -1.8, "stability": -0.1},
                score=75.0,
                passes_filters=True,
                filter_reasons=[],
                rank=1,
            )),
            (candidates[2], ScreeningResult(
                structure_id="MAT-000012",
                predictions={"formation_energy": -1.9, "stability": -0.15},
                score=78.0,
                passes_filters=True,
                filter_reasons=[],
                rank=2,
            )),
        ]
        tracker.record_screening(screening_res, criteria={}, iteration=0)
        rec10 = tracker.records["MAT-000010"]
        assert rec10.status == CandidateStatus.REJECTED.value
        assert rec10.rejection_stage == "screening"
        assert "formation_energy -0.31 > -0.50 eV/atom" in rec10.rejection_reason

        # 2. Candidate 11 fails validation (non-converged)
        val_res = [
            ValidationResult(
                structure_id="MAT-000011",
                structure=candidates[1],
                calculator="mock",
                converged=False,
                properties={},
                cost_hours=2.0,
                error_message="unconverged relaxation after 200 steps",
            )
        ]
        tracker.record_validation(val_res, iteration=0)
        rec11 = tracker.records["MAT-000011"]
        assert rec11.status == CandidateStatus.REJECTED.value
        assert rec11.rejection_stage == "validation"
        assert "unconverged relaxation after 200 steps" in rec11.rejection_reason

        # 3. Candidate 12 fails synthesis feasibility
        synth_res = [
            SynthesisAssessment(
                structure_id="MAT-000012",
                feasible=False,
                feasibility_score=0.25,
                difficulty_score=0.85,
                estimated_cost=9.0,
                synthesis_route="unfeasible",
                route_reason="Excessive element rarity and high difficulty",
            )
        ]
        tracker.record_synthesis(synth_res, iteration=0)
        rec12 = tracker.records["MAT-000012"]
        assert rec12.status == CandidateStatus.REJECTED.value
        assert rec12.rejection_stage == "synthesis"
        assert "Synthesis infeasible" in rec12.rejection_reason


def test_provenance_jsonl_and_csv_export():
    """Verify real-time provenance.jsonl streaming and candidates_provenance.csv structure."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tracker = ProvenanceTracker(
            campaign_id="camp_export_test",
            campaign_name="export_campaign",
            domain="li_solid_electrolyte",
            output_dir=tmp_path,
            master_seed=42,
        )

        candidates = [
            {"candidate_id": "MAT-000001", "composition": "Li3PS4"},
            {"candidate_id": "MAT-000002", "composition": "Li7P3S11"},
        ]
        tracker.register_generation(candidates, iteration=0, backend="stub", seed=42, target_elements=["Li", "P", "S"])

        # Check JSONL
        jsonl_path = tmp_path / "provenance.jsonl"
        assert jsonl_path.exists()
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 2
        assert lines[0]["candidate_id"] == "MAT-000001"

        # Export CSV and check
        csv_path = tracker.export_csv()
        assert csv_path.exists()
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["candidate_id"] == "MAT-000001"
        assert rows[0]["composition"] == "Li3PS4"


def test_report_matches_provenance_source_of_truth():
    """Verify campaign report and summary stats are derived directly from ProvenanceTracker."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        objective = CampaignObjective(
            target_properties={"stability": -0.1, "formation_energy": -2.0},
            constraints={"elements": ["Li", "P", "S"], "max_atoms": 10},
            success_criteria={"min_score": 999.0},
            domain="test_campaign",
            max_iterations=1,
        )
        config = CampaignConfig(
            name="test_truth_campaign",
            objective=objective,
            output_dir=tmp_path,
            use_career_memory=False,
            verbose=False,
            use_validation=True,
            validation_top_k=2,
            use_synthesis=True,
            num_candidates=3,
        )

        campaign = MaterialsDiscoveryCampaign(config)
        results = campaign.run_campaign()

        # Check report.json and campaign_provenance.json
        report_file = tmp_path / "report.json"
        provenance_file = tmp_path / "campaign_provenance.json"
        manifest_file = tmp_path / "manifest.json"

        assert report_file.exists()
        assert provenance_file.exists()
        assert manifest_file.exists()

        with open(provenance_file, "r", encoding="utf-8") as f:
            prov_data = json.load(f)

        stats = prov_data["summary_stats"]
        assert results["total_generated"] == stats["total_generated"]
        assert results["total_passed_screening"] == stats["total_passed_screening"]
        assert results["total_validated"] == stats["total_validated"]
        assert results["total_converged"] == stats["total_converged"]
