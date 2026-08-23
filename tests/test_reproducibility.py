"""
test_reproducibility.py — End-to-end tests for campaign reproducibility from manifest.json,
CLI reproduction workflow, and configuration integrity checking.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import pytest

from campaign import MaterialsDiscoveryCampaign, CampaignConfig
from agents.orchestrator import CampaignObjective
from agents.provenance import RunManifest


def test_campaign_reproduction_from_manifest():
    """Verify that reproducing a campaign from its manifest produces identical candidates, formulas, and scores."""
    with tempfile.TemporaryDirectory() as tmp:
        base_dir = Path(tmp)
        run1_dir = base_dir / "run1"
        run2_dir = base_dir / "run2"

        objective = CampaignObjective(
            target_properties={"stability": -0.1, "formation_energy": -2.0},
            constraints={"elements": ["Li", "P", "S"], "max_atoms": 10},
            success_criteria={"min_score": 999.0},
            domain="repro_domain",
            max_iterations=2,
        )

        config1 = CampaignConfig(
            name="repro_test_campaign",
            objective=objective,
            output_dir=run1_dir,
            master_seed=999,
            use_career_memory=False,
            verbose=False,
            use_validation=True,
            validation_top_k=2,
            use_synthesis=True,
            num_candidates=3,
        )

        # 1. Execute initial run
        campaign1 = MaterialsDiscoveryCampaign(config1)
        res1 = campaign1.run_campaign()

        manifest1_path = run1_dir / "manifest.json"
        assert manifest1_path.exists()

        # 2. Reproduce run directly from saved manifest
        campaign2 = MaterialsDiscoveryCampaign.reproduce_from_manifest(
            manifest_path=manifest1_path,
            output_dir=run2_dir,
        )

        # 3. Compare outputs and provenance
        with open(run1_dir / "campaign_provenance.json", "r", encoding="utf-8") as f:
            prov1 = json.load(f)

        with open(run2_dir / "campaign_provenance.json", "r", encoding="utf-8") as f:
            prov2 = json.load(f)

        cands1 = prov1["candidates"]
        cands2 = prov2["candidates"]

        assert len(cands1) == len(cands2)
        assert len(cands1) == 6  # 2 iterations * 3 candidates

        for c1, c2 in zip(cands1, cands2):
            assert c1["candidate_id"] == c2["candidate_id"]
            assert c1["composition"] == c2["composition"]
            assert c1["elements"] == c2["elements"]
            assert c1["generation_seed"] == c2["generation_seed"]
            assert c1["status"] == c2["status"]
            if c1["screening_score"] is not None and c2["screening_score"] is not None:
                assert np.isclose(c1["screening_score"], c2["screening_score"], atol=1e-4)


def test_manifest_iteration_seeds_preserved():
    """Verify manifest records master seed and distinct iteration seeds."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "seed_test"
        objective = CampaignObjective(
            target_properties={"stability": -0.1},
            constraints={"elements": ["Li", "P", "S"]},
            success_criteria={"min_score": 999.0},
            domain="seed_domain",
            max_iterations=3,
        )
        config = CampaignConfig(
            name="seed_campaign",
            objective=objective,
            output_dir=run_dir,
            master_seed=500,
            use_career_memory=False,
            verbose=False,
            num_candidates=2,
        )
        campaign = MaterialsDiscoveryCampaign(config)
        campaign.run_campaign()

        with open(run_dir / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["master_seed"] == 500
        assert manifest["iteration_seeds"] == [500, 501, 502]


def test_reproduction_cli_replay():
    """Verify that the CLI --reproduce argument reproduces a previous run with matching candidate outputs."""
    with tempfile.TemporaryDirectory() as tmp:
        base_dir = Path(tmp)
        run1_dir = base_dir / "cli_run1"
        run2_dir = base_dir / "cli_run2"

        # 1. Run campaign CLI
        cmd1 = [
            sys.executable, "campaign.py",
            "--domain", "cli_test",
            "--iterations", "1",
            "--candidates", "2",
            "--master-seed", "123",
            "--no-career-memory",
            "--output-dir", str(run1_dir),
        ]
        res1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=60, check=False)
        assert res1.returncode == 0
        manifest_file = run1_dir / "manifest.json"
        assert manifest_file.exists()

        # 2. Reproduce via CLI using long alias --reproduce-from-manifest
        cmd2 = [
            sys.executable, "campaign.py",
            "--reproduce-from-manifest", str(manifest_file),
            "--output-dir", str(run2_dir),
        ]
        res2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=60, check=False)
        assert res2.returncode == 0
        assert (run2_dir / "manifest.json").exists()
        assert (run2_dir / "campaign_provenance.json").exists()
        assert (run2_dir / "candidates_provenance.csv").exists()

        # 3. Check matching candidate outputs
        with open(run1_dir / "campaign_provenance.json", "r", encoding="utf-8") as f:
            d1 = json.load(f)
        with open(run2_dir / "campaign_provenance.json", "r", encoding="utf-8") as f:
            d2 = json.load(f)
        assert len(d1["candidates"]) == len(d2["candidates"])
        assert len(d1["candidates"]) == 2


def test_reproduction_detects_configuration_tampering():
    """Verify that configuration tampering in manifest.json is detected by integrity checking."""
    m1 = RunManifest(
        campaign_id="c1",
        domain="li_battery",
        master_seed=42,
        iteration_seeds=[42, 43],
        objective={"stability": -0.1},
        constraints={"elements": ["Li", "P", "S"]},
        config={"num_candidates": 10},
    )
    m2 = RunManifest(
        campaign_id="c1_tampered",
        domain="li_battery",
        master_seed=999,  # tampered seed
        iteration_seeds=[42, 43],
        objective={"stability": -0.5},  # tampered objective
        constraints={"elements": ["Li", "P", "S"]},
        config={"num_candidates": 10},
    )

    consistent, discrepancies = m1.is_consistent_with(m2)
    assert not consistent
    assert any("Master seed mismatch" in d for d in discrepancies)
    assert any("Objective mismatch" in d for d in discrepancies)


def test_reproduce_from_manifest_raises_on_tampered_hash(tmp_path):
    """Verify reproduce_from_manifest raises ValueError when manifest content is tampered with."""
    manifest = RunManifest(
        campaign_id="camp_hash_test",
        campaign_name="hash_test",
        domain="li_solid_electrolyte",
        master_seed=42,
        iteration_seeds=[42],
        objective={"stability": -0.1},
        constraints={"elements": ["Li", "P", "S"]},
    )
    manifest.manifest_hash = manifest.compute_manifest_hash()
    
    # Tamper with master_seed without updating hash
    manifest_dict = manifest.to_dict()
    manifest_dict["master_seed"] = 9999

    manifest_file = tmp_path / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, indent=2)

    with pytest.raises(ValueError, match="Manifest tampering detected"):
        MaterialsDiscoveryCampaign.reproduce_from_manifest(manifest_file, output_dir=tmp_path / "out")
