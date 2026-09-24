"""Regression tests for the VerifiedResearchExecution campaign boundary.

The generic path (run_mode="research" + interface-compatible objects + backend
names) must never be able to grant itself research validity.  These tests do
not require MatterGen/QE/network access; CHGNet is substituted only through
the verified-evaluator seam, which is still checked against the pinned model
identity.
"""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import pytest

from agents.integrity import RunMode, ScientificValidity
from agents.orchestrator import CampaignObjective
from agents.provenance import ProvenanceTracker, RunManifest
from agents.research_execution import (
    ResearchVerificationError,
    ResearchVerificationReceipt,
    VerifiedResearchExecution,
    hash_path,
)
from agents.screening import ScreeningAgent
from agents.thermodynamics import (
    ModelIdentity,
    ReferencePhaseInput,
    RelaxationSettings,
    ThermodynamicOracle,
    build_frozen_reference_set,
)
from agents.validation import ValidationAgent
from campaign import CampaignConfig, MaterialsDiscoveryCampaign
from experiments.runner import CampaignRunner, CampaignRunnerError
from experiments.spec import RunSpec


MODEL = ModelIdentity("CHGNet", "0.3.0-test", "a" * 64)
SETTINGS = RelaxationSettings(fmax_ev_per_angstrom=0.05, max_steps=500, relax_cell=True)


class FakeEvaluator:
    """Protocol-compatible evaluator; identity still verified against the pin."""

    model_identity = MODEL
    relaxation_settings = SETTINGS

    def __init__(self, energies=None, failures=()):
        self.energies = dict(energies or {})
        self.failures = set(failures)
        self.calls = []

    def relax(self, value):
        key = value.get("candidate_id", value["composition"])
        self.calls.append(key)
        if key in self.failures:
            raise RuntimeError(f"failed {key}")
        return {
            "converged": True,
            "relaxed_structure": value,
            "energy_per_atom_ev": self.energies[key],
            "max_force_ev_per_angstrom": 0.01,
            "max_stress_gpa": 0.02,
        }


def _structure(formula, candidate_id=None):
    from pymatgen.core import Composition

    count = int(Composition(formula).num_atoms)
    return {
        "candidate_id": candidate_id or formula,
        "composition": formula,
        "lattice": [[8.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]],
        "positions": [[(i * 0.37) % 1, (i * 0.23) % 1, (i * 0.41) % 1] for i in range(count)],
    }


def _inp(source_id, formula, hull=None):
    return ReferencePhaseInput(
        source_id=source_id, structure=_structure(formula, source_id),
        source_energy_above_hull_ev_per_atom=hull,
    )


def _certified_reference(tmp_path, evaluator=None):
    evaluator = evaluator or FakeEvaluator(
        {"li": -1.0, "p": -0.5, "se": -0.7, "lipse": -2.5}
    )
    inputs = [
        _inp("li", "Li"), _inp("p", "P"), _inp("se", "Se"),
        _inp("lipse", "LiPSe", 0.0),
    ]
    coverage = {
        "manifest_schema_version": "1.0.0",
        "source_dataset": "unit-test-fixture",
        "dataset_version": "fixed-v1",
        "snapshot_digest": "1" * 64,
        "chemical_system": ["Li", "P", "Se"],
        "selection_procedure": "all declared unit-test phases",
        "expected_source_phase_ids": [item.source_id for item in inputs],
        "elemental_endpoint_ids": ["li", "p", "se"],
        "required_compounds": ["lipse"],
    }
    path = tmp_path / "frozen_Li_P_Se.json"
    frozen = build_frozen_reference_set(
        reference_set_id="Li-P-Se-test", chemical_system=["Li", "P", "Se"],
        inputs=inputs, evaluator=evaluator, output_path=path,
        created_at_iso="2026-01-01T00:00:00+00:00", source_selection=coverage,
    )
    assert frozen.certification.certified
    return path, frozen, evaluator


def _research_spec(tmp_path, *, sampling_config=True, evaluator=None):
    ref_path, frozen, evaluator = _certified_reference(tmp_path, evaluator)
    checkpoint = tmp_path / "mattergen.ckpt"
    checkpoint.write_bytes(b"test-checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    sampling_path = None
    sampling_sha = None
    if sampling_config:
        sampling_path = tmp_path / "sampling.yaml"
        sampling_path.write_text("num_steps: 4\n", encoding="utf-8")
        sampling_sha = hashlib.sha256(sampling_path.read_bytes()).hexdigest()
    spec = RunSpec(
        run_id="verified-run", experiment_id="exp", task_id="Li-P-Se",
        elements=["Li", "P", "Se"], condition="adaptive_no_memory", seed=42,
        iteration_seeds=[42], proposal_budget=4, oracle_budget=2,
        geometry_min_distance=0.8, thermodynamics_retain_threshold_ev_per_atom=0.10,
        thermodynamics_stable_threshold_ev_per_atom=0.03, run_mode="research",
        generation_backend="mattergen", memory_mode="none", memory_seed=0,
        output_dir=str(tmp_path / "run"),
        reference_set_path=str(ref_path),
        reference_set_sha256=hashlib.sha256(ref_path.read_bytes()).hexdigest(),
        reference_set_certified=True,
        pinned_model_identity=asdict(MODEL),
        pinned_relaxation_settings=dict(SETTINGS.__dict__),
        mattergen_model_path=str(checkpoint),
        mattergen_checkpoint_sha256=checkpoint_sha,
        mattergen_sampling_config_path=str(sampling_path) if sampling_path else None,
        mattergen_sampling_config_sha256=sampling_sha,
        validation_calculator="disabled",
        synthesis_mode="disabled",
    )
    return spec, frozen, evaluator


def _objective():
    return CampaignObjective(
        target_properties={},
        constraints={"elements": ["Li", "P", "Se"]},
        success_criteria={"min_score": 999.0},
        domain="boundary_test",
        max_iterations=1,
    )


class _FakeMattergenGenerator:
    """Stand-in for GenerationAgent; MatterGen loading is runner-verified."""

    backend_name = "mattergen"
    last_generation_backend = "mattergen"

    def __init__(self, *args, **kwargs):
        pass


# --- ValidationAgent fail-closed -------------------------------------------------

def test_arbitrary_ase_calculators_fail_in_research_mode():
    from ase.calculators.calculator import Calculator
    from ase.calculators.lj import LennardJones

    class CustomCalculator(Calculator):
        implemented_properties = ["energy"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=None):
            super().calculate(atoms, properties, system_changes)
            self.results["energy"] = 0.0

    with pytest.raises(RuntimeError, match="verified validation"):
        ValidationAgent(calculator=LennardJones(), run_mode=RunMode.RESEARCH)
    with pytest.raises(RuntimeError, match="verified validation"):
        ValidationAgent(calculator=CustomCalculator(), run_mode="research")


def test_unverified_research_validation_strings_fail():
    for name in ("mock", "vasp", "qe", "gpaw", "ase", "VASP"):
        with pytest.raises(RuntimeError, match="verified validation"):
            ValidationAgent(calculator=name, run_mode="research")


def test_development_ase_calculator_still_allowed():
    from ase.calculators.lj import LennardJones

    agent = ValidationAgent(calculator=LennardJones(), run_mode="development")
    assert agent.get_backend_info()["backend_type"] == "ase"
    assert ValidationAgent(calculator="mock").get_backend_info()["backend_type"] == "mock"


# --- CampaignConfig / MaterialsDiscoveryCampaign boundary ------------------------

def test_direct_research_campaign_fails_before_artifacts(tmp_path):
    with pytest.raises(Exception) as exc:
        MaterialsDiscoveryCampaign(CampaignConfig(
            name="research", objective=_objective(), output_dir=tmp_path / "out",
            run_mode="research", proposal_budget=4, oracle_budget=2,
            use_mattergen=True, use_validation=False, use_synthesis=False,
        ))
    assert "VerifiedResearchExecution" in str(exc.value)
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_caller_supplied_research_thermodynamic_evaluator_fails(tmp_path):
    spec, frozen, evaluator = _research_spec(tmp_path)
    ctx = VerifiedResearchExecution.verify(spec, evaluator=evaluator)
    with pytest.raises(ValueError, match="caller-injected thermodynamic"):
        CampaignConfig(
            name="research", objective=_objective(), output_dir=tmp_path / "out",
            run_mode="research", proposal_budget=4, oracle_budget=2,
            use_mattergen=True, use_validation=False, use_synthesis=False,
            research_execution=ctx, research_spec_hash=spec.spec_hash,
            thermodynamics_evaluator=evaluator,
        )


def test_verified_context_constructs_research_campaign(tmp_path, monkeypatch):
    spec, frozen, evaluator = _research_spec(tmp_path)
    ctx = VerifiedResearchExecution.verify(spec, evaluator=evaluator)
    monkeypatch.setattr("campaign.GenerationAgent", _FakeMattergenGenerator)
    campaign = MaterialsDiscoveryCampaign(CampaignConfig(
        name="research", objective=_objective(), output_dir=tmp_path / "out",
        run_mode="research", proposal_budget=4, oracle_budget=2,
        use_career_memory=False, use_mattergen=True,
        use_validation=False, use_synthesis=False,
        mattergen_model_path=spec.mattergen_model_path,
        mattergen_sampling_config_path=spec.mattergen_sampling_config_path,
        thermodynamics_reference_set_path=spec.reference_set_path,
        research_execution=ctx, research_spec_hash=spec.spec_hash,
    ))
    assert campaign.thermodynamic_oracle.evaluator is evaluator
    assert campaign.thermodynamic_oracle.research is True
    assert campaign.validator is None and campaign.synthesis is None
    assert campaign.provenance.manifest.scientific_validity == ScientificValidity.RESEARCH_VALID.value
    assert campaign.provenance.manifest.config["research_execution"]["spec_hash"] == spec.spec_hash


def test_wrong_spec_hash_context_rejected_by_campaign(tmp_path, monkeypatch):
    spec, frozen, evaluator = _research_spec(tmp_path)
    ctx = VerifiedResearchExecution.verify(spec, evaluator=evaluator)
    monkeypatch.setattr("campaign.GenerationAgent", _FakeMattergenGenerator)
    with pytest.raises(Exception, match="spec_hash"):
        MaterialsDiscoveryCampaign(CampaignConfig(
            name="research", objective=_objective(), output_dir=tmp_path / "out",
            run_mode="research", proposal_budget=4, oracle_budget=2,
            use_career_memory=False, use_mattergen=True,
            use_validation=False, use_synthesis=False,
            mattergen_model_path=spec.mattergen_model_path,
            thermodynamics_reference_set_path=spec.reference_set_path,
            research_execution=ctx, research_spec_hash="0" * 64,
        ))


# --- VerifiedResearchExecution verification --------------------------------------

def test_verified_context_rejects_spec_hash_mismatch(tmp_path):
    spec, frozen, evaluator = _research_spec(tmp_path)
    ctx = VerifiedResearchExecution.verify(spec, evaluator=evaluator)
    ctx.assert_matches_spec(spec)
    with pytest.raises(ResearchVerificationError, match="spec_hash"):
        ctx.assert_matches_spec(replace(spec, proposal_budget=8))


def test_verified_context_rejects_mattergen_checkpoint_mismatch(tmp_path):
    spec, frozen, evaluator = _research_spec(tmp_path)
    with pytest.raises(ResearchVerificationError, match="MatterGen checkpoint"):
        VerifiedResearchExecution.verify(
            replace(spec, mattergen_checkpoint_sha256="0" * 64), evaluator=evaluator
        )


def test_verified_context_rejects_sampling_config_mismatch(tmp_path):
    spec, frozen, evaluator = _research_spec(tmp_path)
    with pytest.raises(ResearchVerificationError, match="sampling"):
        VerifiedResearchExecution.verify(
            replace(spec, mattergen_sampling_config_sha256="0" * 64), evaluator=evaluator
        )


def test_verified_context_rejects_reference_sha_mismatch(tmp_path):
    spec, frozen, evaluator = _research_spec(tmp_path)
    with pytest.raises(ResearchVerificationError, match="reference-set artifact SHA256"):
        VerifiedResearchExecution.verify(
            replace(spec, reference_set_sha256="0" * 64), evaluator=evaluator
        )


def test_verified_context_rejects_model_and_settings_mismatch(tmp_path):
    spec, frozen, evaluator = _research_spec(tmp_path)
    bad_model = dict(asdict(MODEL)); bad_model["version"] = "0.0.0-other"
    with pytest.raises(ResearchVerificationError, match="model identity"):
        VerifiedResearchExecution.verify(
            replace(spec, pinned_model_identity=bad_model), evaluator=evaluator
        )
    bad_settings = dict(SETTINGS.__dict__); bad_settings["max_steps"] = 100
    with pytest.raises(ResearchVerificationError, match="relaxation settings"):
        VerifiedResearchExecution.verify(
            replace(spec, pinned_relaxation_settings=bad_settings), evaluator=evaluator
        )
    # An evaluator whose self-reported identity disagrees with the verified
    # reference identity is rejected even when the spec pin is untouched.
    wrong_eval = FakeEvaluator()
    wrong_eval.model_identity = replace(MODEL, checkpoint_sha256="b" * 64)
    with pytest.raises(ResearchVerificationError, match="evaluator model"):
        VerifiedResearchExecution.verify(spec, evaluator=wrong_eval)
    wrong_settings_eval = FakeEvaluator()
    wrong_settings_eval.relaxation_settings = replace(SETTINGS, max_steps=100)
    with pytest.raises(ResearchVerificationError, match="relaxation-settings"):
        VerifiedResearchExecution.verify(spec, evaluator=wrong_settings_eval)


def test_verify_requires_research_mode(tmp_path):
    spec, frozen, evaluator = _research_spec(tmp_path)
    with pytest.raises(ResearchVerificationError, match="run_mode"):
        VerifiedResearchExecution.verify(
            replace(spec, run_mode="development"), evaluator=evaluator
        )


# --- ScreeningAgent / provenance --------------------------------------------------

def test_research_screener_requires_verified_oracle(tmp_path):
    _, frozen, evaluator = _certified_reference(tmp_path)
    with pytest.raises(RuntimeError, match="verified research"):
        ScreeningAgent(run_mode="research")
    # A non-research oracle is not the verified boundary either.
    oracle_dev = ThermodynamicOracle(frozen, evaluator)
    with pytest.raises(RuntimeError, match="verified research"):
        ScreeningAgent(run_mode="research", thermodynamic_oracle=oracle_dev)
    # The verified oracle path is accepted.
    oracle = ThermodynamicOracle(frozen, evaluator, research=True)
    agent = ScreeningAgent(run_mode="research", thermodynamic_oracle=oracle)
    assert agent.last_backend_used == "chgnet_thermodynamic_oracle"


def test_run_mode_research_alone_cannot_produce_research_valid(tmp_path):
    with pytest.raises(ValueError, match="ResearchVerificationReceipt"):
        ProvenanceTracker(
            campaign_id="c", campaign_name="c", domain="d",
            output_dir=tmp_path / "p", run_mode="research",
        )
    with pytest.raises(ValueError, match="ResearchVerificationReceipt"):
        ProvenanceTracker(
            campaign_id="c", campaign_name="c", domain="d",
            output_dir=tmp_path / "p2", run_mode="development",
            scientific_validity="research_valid",
        )
    tracker = ProvenanceTracker(
        campaign_id="c", campaign_name="c", domain="d", output_dir=tmp_path / "p3",
    )
    assert tracker.scientific_validity == "demo_only"


def test_receipt_derived_from_verified_execution_grants_validity(tmp_path):
    spec, frozen, evaluator = _research_spec(tmp_path)
    ctx = VerifiedResearchExecution.verify(spec, evaluator=evaluator)
    tracker = ProvenanceTracker(
        campaign_id="c", campaign_name="c", domain="d", output_dir=tmp_path / "p",
        run_mode="research", research_verification=ctx.receipt(),
    )
    assert tracker.scientific_validity == ScientificValidity.RESEARCH_VALID.value


# --- Runner wiring / manifest replay ----------------------------------------------

def test_canonical_runner_constructs_and_passes_verified_execution(tmp_path, monkeypatch):
    spec, frozen, evaluator = _research_spec(tmp_path)
    from experiments.selection_protocol import PROTOCOL, CONTRACT, generate_stream, proposal_identity
    from pymatgen.core import Structure, Lattice
    spec = replace(spec, condition="source_neutral", protocol_version=PROTOCOL, protocol=dict(CONTRACT),
                   oracle_budget=100, proposal_budget=100)
    class Proposals:
        last_generation_backend = "mattergen"
        def generate_batch(self, **kwargs):
            return [Structure(Lattice.cubic(5), ["Li"], [[0, 0, 0]]) for _ in range(kwargs["num_candidates"])]
    from tests.revised_helpers import authorize
    from experiments.release_integrity import initialize_pair
    spec = initialize_pair(authorize(spec, tmp_path, monkeypatch), Proposals)
    captured = {}

    class CaptureCampaign:
        def __init__(self, config, *args):
            captured["config"] = config
            raise RuntimeError("capture-sentinel")

    # Substitute the real CHGNet load; the evaluator identity is still checked
    # against the verified frozen/pinned identity inside verify().
    monkeypatch.setattr(
        "agents.research_execution.CHGNetRelaxationEvaluator",
        lambda **kwargs: evaluator,
    )
    monkeypatch.setattr("experiments.revised_runner.RevisedCampaign", CaptureCampaign)
    with pytest.raises(CampaignRunnerError, match="capture-sentinel"):
        CampaignRunner.execute_run(spec)

    config = captured["config"]
    assert config.run_mode == RunMode.RESEARCH
    assert config.use_validation is False
    assert config.use_synthesis is False
    assert config.research_spec_hash == spec.spec_hash
    ctx = config.research_execution
    assert isinstance(ctx, VerifiedResearchExecution)
    assert ctx.spec_hash == spec.spec_hash
    assert ctx.mattergen_checkpoint_sha256 == spec.mattergen_checkpoint_sha256
    assert ctx.reference_set_sha256 == spec.reference_set_sha256
    assert ctx.model_identity == MODEL
    assert ctx.relaxation_settings == SETTINGS
    assert ctx.validation_disabled and ctx.synthesis_disabled
    ctx.assert_matches_spec(spec)


def test_runner_rejects_research_spec_tampering(tmp_path, monkeypatch):
    spec, frozen, evaluator = _research_spec(tmp_path)
    from experiments.selection_protocol import PROTOCOL, CONTRACT
    bad = replace(spec, reference_set_sha256="0" * 64, condition="source_neutral",
                  protocol_version=PROTOCOL, protocol=dict(CONTRACT), oracle_budget=100, proposal_budget=100)
    from tests.revised_helpers import authorize
    from experiments.release_integrity import initialize_pair
    from pymatgen.core import Structure, Lattice
    class Proposals:
        last_generation_backend = "mattergen"
        def generate_batch(self, **kwargs):
            return [Structure(Lattice.cubic(5), ["Li"], [[0, 0, 0]]) for _ in range(kwargs["num_candidates"])]
    bad = initialize_pair(authorize(bad, tmp_path, monkeypatch), Proposals)
    with pytest.raises(CampaignRunnerError, match="SHA256 mismatch"):
        CampaignRunner.execute_run(bad)


def test_research_manifest_replay_requires_fresh_verification(tmp_path):
    manifest = RunManifest(run_mode="research")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    with pytest.raises(RuntimeError, match="reproduced through the direct campaign path"):
        MaterialsDiscoveryCampaign.reproduce_from_manifest(path, output_dir=tmp_path / "repro")


def test_development_manifest_replay_still_works(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.screening.HAS_CHGNET", False)
    campaign = MaterialsDiscoveryCampaign(CampaignConfig(
        name="dev", objective=_objective(), output_dir=tmp_path,
        use_career_memory=False, use_validation=False, use_synthesis=False,
        num_candidates=1, proposal_budget=2, oracle_budget=1, verbose=False,
    ))
    campaign.run_campaign()
    reproduced = MaterialsDiscoveryCampaign.reproduce_from_manifest(
        tmp_path / "manifest.json", output_dir=tmp_path / "repro"
    )
    assert reproduced.config.proposal_budget == 2
