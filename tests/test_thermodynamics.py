"""Sprint 3 acceptance tests for frozen-reference thermodynamics."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from agents.thermodynamics import (
    ModelIdentity,
    ReferencePhaseInput,
    ReferenceSetError,
    RelaxationSettings,
    ThermodynamicFailureCode,
    ThermodynamicOracle,
    ThermodynamicOracleError,
    build_frozen_reference_set,
    canonical_json,
    load_frozen_reference_set,
    sha256_payload,
    threshold_sensitivity,
)
from agents.budget import DualBudgetTracker
from agents.screening import ScreeningAgent
from agents.integrity import ScientificPreflightError, build_scientific_preflight
from agents.orchestrator import CampaignObjective
from campaign import CampaignConfig, MaterialsDiscoveryCampaign


MODEL = ModelIdentity("fake-chgnet", "0.3.0", "a" * 64)
SETTINGS = RelaxationSettings(fmax_ev_per_angstrom=0.05, max_steps=500, relax_cell=True)
REPO_ROOT = Path(__file__).resolve().parents[1]


def structure(formula, candidate_id=None, lattice=None, positions=None):
    from pymatgen.core import Composition

    count = int(Composition(formula).num_atoms)
    positions = positions or [[(i * 0.37) % 1, (i * 0.23) % 1, (i * 0.41) % 1] for i in range(count)]
    return {
        "candidate_id": candidate_id or formula,
        "composition": formula,
        "lattice": lattice or [[8.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]],
        "positions": positions,
    }


class FakeEvaluator:
    model_identity = MODEL
    relaxation_settings = SETTINGS

    def __init__(self, energies, failures=(), nonfinite=(), replacements=None):
        self.energies = dict(energies)
        self.failures = set(failures)
        self.nonfinite = set(nonfinite)
        self.replacements = dict(replacements or {})
        self.calls = []

    def relax(self, value):
        key = value.get("candidate_id", value["composition"])
        self.calls.append(key)
        if key in self.failures:
            raise RuntimeError(f"failed {key}")
        relaxed = self.replacements.get(key, value)
        energy = float("nan") if key in self.nonfinite else self.energies[key]
        return {
            "converged": True,
            "relaxed_structure": relaxed,
            "energy_per_atom_ev": energy,
            "max_force_ev_per_angstrom": 0.01,
            "max_stress_gpa": 0.02,
        }


class FixedThermodynamicGenerator:
    """Small deterministic generator for campaign-level thermodynamics tests."""

    backend_name = "stub"

    def __init__(self):
        self.last_generation_backend = "stub"

    def generate_batch(self, elements, num_candidates=15, seed=42, **kwargs):
        return [structure("LiO", "campaign-candidate") for _ in range(num_candidates)]


def inp(source_id, formula, energy_hull=None, **kwargs):
    return ReferencePhaseInput(
        source_id=source_id, structure=structure(formula, source_id),
        source_energy_above_hull_ev_per_atom=energy_hull, **kwargs,
    )


def build_binary(tmp_path, *, evaluator=None, created="2026-08-29T00:00:00+00:00"):
    evaluator = evaluator or FakeEvaluator({"Li-ref": -1.0, "O-ref": -2.0, "LiO-ref": -2.0})
    path = tmp_path / "frozen_Li_O.json"
    frozen = build_frozen_reference_set(
        reference_set_id="Li-O-chgnet-v1", chemical_system=["O", "Li"],
        inputs=[inp("Li-ref", "Li"), inp("O-ref", "O"), inp("LiO-ref", "LiO", 0.0)],
        evaluator=evaluator, output_path=path, created_at_iso=created,
    )
    return path, frozen, evaluator


def test_binary_hull_uses_total_energies_and_reports_true_quantities(tmp_path):
    path, _, _ = build_binary(tmp_path)
    frozen = load_frozen_reference_set(path, expected_model=MODEL, expected_settings=SETTINGS,
                                       required_chemical_system=["Li", "O"])
    evaluator = FakeEvaluator({"candidate": -1.9})
    result = ThermodynamicOracle(frozen, evaluator).evaluate(structure("LiO", "candidate"))
    assert result.success
    assert result.predicted_total_energy_ev == pytest.approx(-3.8)
    assert result.predicted_formation_energy_ev_per_atom == pytest.approx(-0.4)
    assert result.predicted_energy_above_hull_ev_per_atom == pytest.approx(0.1)
    assert result.decomposition == pytest.approx({"LiO-ref": 1.0})
    assert not result.predicted_thermodynamically_stable
    assert result.retained_by_hull_threshold


def test_unary_and_ternary_known_phase_diagrams(tmp_path):
    unary_path = tmp_path / "unary.json"
    unary_eval = FakeEvaluator({"Li-ref": -1.0})
    unary = build_frozen_reference_set(
        reference_set_id="Li", chemical_system=["Li"], inputs=[inp("Li-ref", "Li")],
        evaluator=unary_eval, output_path=unary_path, created_at_iso="fixed",
    )
    unary_result = ThermodynamicOracle(unary, FakeEvaluator({"u": -0.9})).evaluate(structure("Li", "u"))
    assert unary_result.predicted_formation_energy_ev_per_atom == pytest.approx(0.1)
    assert unary_result.predicted_energy_above_hull_ev_per_atom == pytest.approx(0.1)

    ternary_path = tmp_path / "ternary.json"
    energies = {"Li": -1.0, "P": -3.0, "S": -2.0, "LiPS": -2.5}
    ternary = build_frozen_reference_set(
        reference_set_id="Li-P-S", chemical_system=["Li", "P", "S"],
        inputs=[inp(name, name, 0.0 if name == "LiPS" else None) for name in energies],
        evaluator=FakeEvaluator(energies), output_path=ternary_path, created_at_iso="fixed",
    )
    result = ThermodynamicOracle(ternary, FakeEvaluator({"candidate": -2.4})).evaluate(
        structure("LiPS", "candidate")
    )
    assert result.predicted_formation_energy_ev_per_atom == pytest.approx(-0.4)
    assert result.predicted_energy_above_hull_ev_per_atom == pytest.approx(0.1)
    assert result.decomposition == pytest.approx({"LiPS": 1.0})


def test_below_reference_hull_does_not_redefine_hull_or_decompose_to_self(tmp_path):
    path, frozen, _ = build_binary(tmp_path)
    result = ThermodynamicOracle(frozen, FakeEvaluator({"new": -2.1})).evaluate(structure("LiO", "new"))
    assert result.predicted_signed_hull_delta_ev_per_atom == pytest.approx(-0.1)
    assert result.predicted_energy_above_hull_ev_per_atom == 0.0
    assert result.decomposition == pytest.approx({"LiO-ref": 1.0})


def test_dedup_is_deterministic_and_artifact_hash_detects_tampering(tmp_path):
    duplicate = inp("zzz-duplicate", "LiO", 0.0)
    duplicate.structure["candidate_id"] = "different-id"
    evaluator = FakeEvaluator({"Li-ref": -1.0, "O-ref": -2.0, "LiO-ref": -2.0, "different-id": -2.0})
    path = tmp_path / "set.json"
    frozen = build_frozen_reference_set(
        reference_set_id="deterministic", chemical_system=["Li", "O"],
        inputs=[duplicate, inp("O-ref", "O"), inp("LiO-ref", "LiO", 0.0), inp("Li-ref", "Li")],
        evaluator=evaluator, output_path=path, created_at_iso="fixed",
    )
    assert len(frozen.phases) == 3
    original = path.read_text(encoding="utf-8")
    digest = frozen.reference_set_hash
    assert load_frozen_reference_set(path).reference_set_hash == digest
    payload = json.loads(original)
    payload["reference_set_id"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReferenceSetError, match="SHA256"):
        load_frozen_reference_set(path)


def test_loader_recomputes_certification_even_when_sidecar_is_rewritten(tmp_path):
    path, _, _ = build_binary(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    # An attacker who can rewrite both files must not be able to mark a
    # semantically invalid certification as valid by recomputing the digest.
    payload["certification"]["near_hull_succeeded"] = 999
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(
        sha256_payload(payload) + "\n", encoding="ascii"
    )
    with pytest.raises(ReferenceSetError, match="certification"):
        load_frozen_reference_set(path, require_certified=False)


def test_reference_builder_help_runs_when_invoked_as_direct_script():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "build_reference_set.py"), "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "offline" in result.stdout.lower()


def test_certification_requires_endpoints_all_near_hull_and_95_percent_other(tmp_path):
    missing = build_frozen_reference_set(
        reference_set_id="missing", chemical_system=["Li", "O"], inputs=[inp("Li", "Li")],
        evaluator=FakeEvaluator({"Li": -1}), output_path=tmp_path / "missing.json", created_at_iso="fixed",
    )
    assert not missing.certification.certified
    assert missing.certification.missing_or_failed_elemental_endpoints == ["O"]

    near = build_frozen_reference_set(
        reference_set_id="near", chemical_system=["Li", "O"],
        inputs=[inp("Li", "Li"), inp("O", "O"), inp("near", "LiO", 0.05)],
        evaluator=FakeEvaluator({"Li": -1, "O": -2}, failures={"near"}),
        output_path=tmp_path / "near.json", created_at_iso="fixed",
    )
    assert not near.certification.certified
    assert near.certification.near_hull_total == 1
    assert near.certification.near_hull_succeeded == 0

    other_inputs = [inp("Li", "Li"), inp("O", "O")] + [
        ReferencePhaseInput(
            source_id=f"p{i}",
            structure=structure("LiO", f"p{i}", lattice=[[8.0 + i / 10, 0, 0], [0, 8, 0], [0, 0, 8]]),
            source_energy_above_hull_ev_per_atom=0.2 + i / 100,
        )
        for i in range(20)
    ]
    energies = {"Li": -1, "O": -2, **{f"p{i}": -1.5 for i in range(20)}}
    ninety = build_frozen_reference_set(
        reference_set_id="ninety", chemical_system=["Li", "O"], inputs=other_inputs,
        evaluator=FakeEvaluator(energies, failures={"p0", "p1"}),
        output_path=tmp_path / "ninety.json", created_at_iso="fixed",
    )
    assert not ninety.certification.certified
    assert ninety.certification.other_success_fraction == pytest.approx(0.9)


def test_nonfinite_and_collapsed_relaxations_are_explicit(tmp_path):
    collapsed = structure("Li", "collapsed", lattice=[[0.5, 0, 0], [0, 5, 0], [0, 0, 5]])
    evaluator = FakeEvaluator(
        {"Li": -1, "O": -2, "bad": -1, "collapsed": -1}, nonfinite={"bad"},
        replacements={"collapsed": collapsed},
    )
    collapsed_input = ReferencePhaseInput(
        source_id="collapsed", structure=structure("Li", "collapsed", lattice=[[7, 0, 0], [0, 7, 0], [0, 0, 7]]),
        source_energy_above_hull_ev_per_atom=0.2,
    )
    frozen = build_frozen_reference_set(
        reference_set_id="failures", chemical_system=["Li", "O"],
        inputs=[inp("Li", "Li"), inp("O", "O"), inp("bad", "LiO", 0.2), collapsed_input],
        evaluator=evaluator, output_path=tmp_path / "failures.json", created_at_iso="fixed",
    )
    codes = {phase.source_id: phase.failure_code for phase in frozen.phases}
    assert codes["bad"] == ThermodynamicFailureCode.NONFINITE_ENERGY.value
    assert codes["collapsed"] == ThermodynamicFailureCode.INVALID_RELAXED_GEOMETRY.value


def test_loader_rejects_model_settings_system_and_uncertified(tmp_path):
    path, _, _ = build_binary(tmp_path)
    with pytest.raises(ReferenceSetError, match="model"):
        load_frozen_reference_set(path, expected_model=replace(MODEL, version="other"))
    with pytest.raises(ReferenceSetError, match="settings"):
        load_frozen_reference_set(path, expected_settings=RelaxationSettings(max_steps=400))
    with pytest.raises(ReferenceSetError, match="chemical system"):
        load_frozen_reference_set(path, required_chemical_system=["Li", "S"])

    bad_path = tmp_path / "bad.json"
    build_frozen_reference_set(
        reference_set_id="bad", chemical_system=["Li", "O"], inputs=[inp("Li", "Li")],
        evaluator=FakeEvaluator({"Li": -1}), output_path=bad_path, created_at_iso="fixed",
    )
    with pytest.raises(ReferenceSetError, match="not certified"):
        load_frozen_reference_set(bad_path)


def test_oracle_failure_has_no_scientific_values_and_research_raises(tmp_path):
    _, frozen, _ = build_binary(tmp_path)
    candidate = structure("LiO", "failure")
    result = ThermodynamicOracle(frozen, FakeEvaluator({}, failures={"failure"})).evaluate(candidate)
    assert not result.success
    assert result.failure_code == ThermodynamicFailureCode.RELAXATION_FAILED.value
    assert result.scientific_values() == {}
    with pytest.raises(ThermodynamicOracleError, match="RELAXATION_FAILED"):
        ThermodynamicOracle(frozen, FakeEvaluator({}, failures={"failure"}), research=True).evaluate(candidate)


def test_thresholds_and_sensitivity(tmp_path):
    _, frozen, _ = build_binary(tmp_path)
    stable = ThermodynamicOracle(frozen, FakeEvaluator({"c": -1.98})).evaluate(structure("LiO", "c"))
    assert stable.predicted_thermodynamically_stable
    assert stable.retained_by_hull_threshold
    assert threshold_sensitivity([0.01, 0.03, 0.04, 0.08, None]) == {"0.03": 2, "0.05": 3, "0.10": 4}


def test_builder_is_offline_and_deterministic_for_fixed_inputs(tmp_path):
    first_path, first, _ = build_binary(tmp_path / "a")
    second_path, second, _ = build_binary(tmp_path / "b")
    assert first.reference_set_hash == second.reference_set_hash
    assert first_path.read_bytes() == second_path.read_bytes()


def test_screening_integrates_hull_filter_cache_and_exact_budget(tmp_path, monkeypatch):
    _, frozen, _ = build_binary(tmp_path)
    evaluator = FakeEvaluator({"same": -1.89})
    oracle = ThermodynamicOracle(frozen, evaluator)
    monkeypatch.setattr("agents.screening.ScreeningAgent._init_models", lambda self: None)
    screener = ScreeningAgent(thermodynamic_oracle=oracle)
    tracker = DualBudgetTracker(proposal_budget=2, oracle_budget=2)
    tracker.record_proposals(2)
    candidate = structure("LiO", "same")
    results = screener.screen_batch([candidate, candidate], {}, deduplicate=False, budget_tracker=tracker)
    assert evaluator.calls == ["same"]
    assert tracker.oracle_evaluations == 2
    assert tracker.oracle_cache_hits == 1
    assert all(not result.passes_filters for _, result in results)
    assert all("predicted_energy_above_hull_ev_per_atom" in result.predictions for _, result in results)


def test_relaxation_failure_consumes_oracle_and_has_no_thermo_values(tmp_path, monkeypatch):
    _, frozen, _ = build_binary(tmp_path)
    oracle = ThermodynamicOracle(frozen, FakeEvaluator({}, failures={"failure"}))
    monkeypatch.setattr("agents.screening.ScreeningAgent._init_models", lambda self: None)
    tracker = DualBudgetTracker(oracle_budget=1)
    result = ScreeningAgent(thermodynamic_oracle=oracle).screen_batch(
        [structure("LiO", "failure")], {}, deduplicate=False, budget_tracker=tracker
    )[0][1]
    assert tracker.oracle_evaluations == 1
    assert result.failure_code == "THERMODYNAMIC_ORACLE_FAILED"
    assert result.predictions == {}
    assert result.oracle_evaluated is True


def test_campaign_boolean_cannot_spoof_research_thermodynamics(tmp_path):
    objective = CampaignObjective({}, {"elements": ["Li", "O"]}, {}, "test", 1)
    with pytest.raises(ScientificPreflightError) as exc:
        MaterialsDiscoveryCampaign(CampaignConfig(
            name="spoof", objective=objective, output_dir=tmp_path, run_mode="research",
            proposal_budget=1, oracle_budget=1, use_mattergen=False,
            use_validation=False, use_synthesis=False, thermodynamics_available=True,
        ))
    assert "thermodynamics_reference_set_path" in str(exc.value)


def test_loaded_certified_oracle_supplies_preflight_capability(tmp_path):
    path, _, evaluator = build_binary(tmp_path)
    objective = CampaignObjective({}, {"elements": ["Li", "O"]}, {}, "test", 1)
    campaign = MaterialsDiscoveryCampaign(CampaignConfig(
        name="loaded", objective=objective, output_dir=tmp_path / "campaign",
        use_career_memory=False, use_validation=False, use_synthesis=False,
        thermodynamics_reference_set_path=str(path), thermodynamics_evaluator=evaluator,
    ))
    assert campaign.thermodynamic_oracle.capability
    report = build_scientific_preflight(
        run_mode="research",
        requested_backends={"generation": "mattergen", "validation": "disabled", "synthesis": "disabled"},
        actual_backends={"generation": "mattergen", "screening": "chgnet_thermodynamic_oracle",
                         "validation": "disabled", "synthesis": "disabled"},
        require_thermodynamics=True,
        thermodynamics_available=campaign.thermodynamic_oracle.capability,
    )
    assert report.valid
    assert campaign.provenance.manifest.config["thermodynamics_reference_set_path"] == str(path)


def test_threshold_configuration_roundtrips_through_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.screening.HAS_CHGNET", False)
    objective = CampaignObjective({}, {"elements": ["Li"]}, {"min_score": 999.0}, "roundtrip", 1)
    campaign = MaterialsDiscoveryCampaign(CampaignConfig(
        name="roundtrip", objective=objective, output_dir=tmp_path,
        use_career_memory=False, use_validation=False, use_synthesis=False,
        num_candidates=1, thermodynamics_retain_threshold_ev_per_atom=0.07,
        thermodynamics_stable_threshold_ev_per_atom=0.02,
    ))
    campaign.run_campaign()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["thermodynamics_retain_threshold_ev_per_atom"] == pytest.approx(0.07)
    assert manifest["config"]["thermodynamics_stable_threshold_ev_per_atom"] == pytest.approx(0.02)
    reproduced = MaterialsDiscoveryCampaign.reproduce_from_manifest(
        tmp_path / "manifest.json", output_dir=tmp_path / "reproduced"
    )
    assert reproduced.config.thermodynamics_retain_threshold_ev_per_atom == pytest.approx(0.07)
    assert reproduced.config.thermodynamics_stable_threshold_ev_per_atom == pytest.approx(0.02)


def test_campaign_reports_thermodynamics_metrics_when_oracle_is_loaded(tmp_path):
    path, _, evaluator = build_binary(tmp_path)
    evaluator.energies["campaign-candidate"] = -1.9
    objective = CampaignObjective(
        {}, {"elements": ["Li", "O"]}, {"min_score": 999.0}, "metrics", 1,
    )
    campaign = MaterialsDiscoveryCampaign(CampaignConfig(
        name="metrics", objective=objective, output_dir=tmp_path / "campaign",
        use_career_memory=False, use_validation=False, use_synthesis=False,
        num_candidates=1, thermodynamics_reference_set_path=str(path),
        thermodynamics_evaluator=evaluator,
    ))
    campaign._init_generator = lambda: FixedThermodynamicGenerator()
    # The generator is initialized during construction, so replace it before run.
    campaign.generator = FixedThermodynamicGenerator()
    result = campaign.run_campaign()
    assert result["thermodynamics_metrics_available"] is True
    assert campaign.results_history[0]["insights"]["thermodynamics_metrics_available"] is True
    assert result["actual_backends"]["thermodynamics"] == "certified_frozen_chgnet_hull"
