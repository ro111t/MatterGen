"""Sprint 4 structure-aware CareerMemory contracts."""

import json
import sqlite3
from types import SimpleNamespace

import pytest

from agents.career_memory import CareerMemory
from agents.experience_distiller import ExperienceDistiller
from agents.integrity import SCHEMA_VERSION
from agents.transferable_memory import (
    ApplicabilityConstraint,
    TransferDirective,
    TransferableFeatures,
    TransferableMemoryRecord,
    applicability_check,
    extract_transferable_features,
    make_directive,
    view_records,
)
from agents.screening import ScreeningResult
from agents.thermodynamics import ThermodynamicResult
from agents.orchestrator import CampaignObjective, OrchestratorAgent
from agents.transferable_memory import prioritize_candidates
from agents.provenance import ProvenanceTracker
from agents.budget import DualBudgetTracker
from agents.screening import ScreeningAgent
from agents.integrity import FORCE_KEY, STRESS_KEY


def _structure(formula="Li2PS3", elements=None, candidate_id="c1"):
    return {
        "composition": formula,
        "elements": elements or ["Li", "P", "S"],
        "candidate_id": candidate_id,
        "lattice": [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
        "positions": [[0, 0, 0]] * 6,
        "coordination_numbers": [4, 4, 3, 3, 3, 3],
        "oxidation_states": {"Li": 1, "P": 5, "S": -2},
        "prototype": "argyrodite",
    }


def test_homologous_descriptors_transfer_without_formula_identity():
    li = extract_transferable_features(_structure("Li2PS3"))
    na = extract_transferable_features(_structure("Na2PSe3", ["Na", "P", "Se"], "c2"))
    assert li.anonymous_stoichiometric_pattern == na.anonymous_stoichiometric_pattern
    assert sorted(li.element_classes.values()) == sorted(na.element_classes.values())
    assert li.structural_prototype == na.structural_prototype
    assert li.anonymous_stoichiometric_pattern != "Li2PS3"


def test_features_are_deterministic_and_missing_is_explicit():
    features = extract_transferable_features(_structure())
    again = extract_transferable_features(_structure())
    assert features.to_dict() == again.to_dict()
    assert features.volume_per_atom == pytest.approx(125 / 6)
    assert features.coordination_summary["coordination_number_mean"] == pytest.approx(10 / 3)
    assert features.oxidation_state_balance == pytest.approx(1.0)
    assert features.missing_features["density"] == "DENSITY_UNAVAILABLE"
    assert features.missing_features["thermodynamic_label"] == "THERMODYNAMIC_OUTCOME_UNAVAILABLE"


def test_applicability_fails_closed_for_unrelated_system():
    record = TransferableMemoryRecord(
        record_id="r1", campaign_ids=["old"], source_domain="x", finalized=True,
        features=extract_transferable_features(_structure()),
        applicability=ApplicabilityConstraint(
            allowed_relationship="same_system", confidence=0.8,
            source_chemical_system=["Li", "P", "S"],
        ),
    )
    applicable, reasons = applicability_check(
        record, extract_transferable_features(_structure("Bi2Te3", ["Bi", "Te"])), target_domain="x"
    )
    assert not applicable
    assert "CHEMICAL_SYSTEM_MISMATCH" in reasons


def test_db_dedup_quarantine_and_finalized_warm_start(tmp_path):
    memory = CareerMemory(str(tmp_path / "memory.db"))
    campaign = memory.start_campaign("old", "x", {})
    structure = _structure()
    rid = memory.store_transferable_evidence(
        campaign_id=campaign, domain="x", structure=structure,
        result={"passes_filters": True}, outcome_label="stable",
    )
    # Active campaign data cannot leak into its own or a future warm start.
    assert memory.get_transferable_records() == []
    duplicate = memory.store_transferable_evidence(
        campaign_id=campaign, domain="x", structure=structure,
        result={"passes_filters": True}, outcome_label="stable",
    )
    assert duplicate == rid
    memory.end_campaign(campaign, {})
    records = memory.get_transferable_records()
    assert len(records) == 1
    assert records[0].applicability.evidence_count == 1
    memory.conn.execute(
        "INSERT INTO transferable_records (record_id, schema_version, evidence_hash, source_domain, payload) VALUES (?, ?, ?, ?, ?)",
        ("legacy", "1.0.0", "legacy-hash", "x", json.dumps({"record_id": "legacy", "schema_version": "1.0.0"})),
    )
    memory.conn.commit()
    assert all(item.record_id != "legacy" for item in memory.get_transferable_records())


def test_memory_modes_and_seeded_control():
    records = [TransferableMemoryRecord(
        record_id=str(i), campaign_ids=["c"], finalized=True,
        features=TransferableFeatures(anonymous_stoichiometric_pattern=f"A{i}"),
        outcome_label=f"outcome-{i}", evidence_ids=[f"e{i}"],
        directive=TransferDirective(exploration_weight=0.1 + i * 0.2),
    ) for i in range(3)]
    record = records[0]
    none = view_records([record], "none")
    text = view_records([record], "text_summary")
    structured = view_records([record], "structured_provenance")
    control_a = view_records(records, "shuffled_control", seed=11)
    control_b = view_records(records, "shuffled_control", seed=11)
    control_c = view_records(records, "shuffled_control", seed=1)
    assert none["records"] == []
    assert text["records"] == [] and "outcome=" in text["text"]
    assert structured["records"][0]["record_id"] == "0"
    assert control_a == control_b
    assert control_a["scientific_decision_support"] is False
    assert "invalid_for_scientific_decision_support" in control_a["label"]
    assert control_a["record_count"] == 3
    assert control_a["shuffle_audit"]["fixed_points"] == 0
    assert control_a["shuffle_audit"]["permutation"] != [0, 1, 2]
    assert control_a["shuffle_audit"]["permutation"] != control_c["shuffle_audit"]["permutation"]
    assert sorted(x["features"]["anonymous_stoichiometric_pattern"] for x in control_a["records"]) == ["A0", "A1", "A2"]
    assert sorted(x["outcome_label"] for x in control_a["records"]) == ["outcome-0", "outcome-1", "outcome-2"]
    assert all(x["paired_descriptor_record_id"] != x["paired_response_record_id"] for x in control_a["records"])
    singleton = view_records([record], "shuffled_control", seed=11)
    assert singleton["records"] == []
    assert singleton["shuffle_audit"]["reason"] == "SHUFFLE_CONTROL_UNAVAILABLE_SINGLETON"


def test_directive_is_cited_and_cannot_bypass_gates():
    record = TransferableMemoryRecord(
        record_id="r", principle_id="p", evidence_ids=["e"], campaign_ids=["c"], finalized=True,
        directive=TransferDirective(preferred_prototypes=["argyrodite"], exploration_weight=9),
    )
    payload = make_directive(record)
    assert payload["preferred_prototypes"] == ["argyrodite"]
    assert payload["source_principle_ids"] == ["p"]
    assert payload["source_evidence_ids"] == ["e"]
    assert payload["bypasses_geometry_gate"] is False
    assert payload["bypasses_thermodynamic_gate"] is False
    assert payload["exploration_weight"] <= 0.8


def test_certified_thermo_result_required_and_typed_result_accepted():
    spoof = {
        "thermodynamic_label": "stable", "predicted_energy_above_hull_ev_per_atom": 0.0,
        "reference_set_id": "ref", "reference_set_hash": "hash",
        "model": {}, "relaxation_settings": {}, "thermodynamics_certified": True,
        "provenance_schema_version": SCHEMA_VERSION,
    }
    rejected = extract_transferable_features(_structure(), result=spoof)
    assert rejected.thermodynamic_label is None
    assert rejected.missing_features["thermodynamic_label"] == "UNCERTIFIED_THERMODYNAMIC_OUTCOME"

    genuine = ThermodynamicResult(
        success=True, predicted_energy_above_hull_ev_per_atom=0.01,
        predicted_thermodynamically_stable=True, retained_by_hull_threshold=True,
        reference_set_id="ref", reference_set_hash="hash", model={"name": "CHGNet"},
        relaxation_settings={"fmax_ev_per_angstrom": 0.05},
    )
    accepted = extract_transferable_features(_structure(), result=genuine)
    assert accepted.thermodynamic_label == "stable"
    assert accepted.thermodynamic_threshold_ev_per_atom == pytest.approx(0.01)

    wrapper = ScreeningResult(
        structure_id="c", predictions=genuine.scientific_values(), score=1.0,
        passes_filters=True, filter_reasons=[], backend="chgnet_thermodynamic_oracle",
        provenance_stage="thermodynamics",
    )
    wrapped = extract_transferable_features(_structure(), result=wrapper)
    assert wrapped.thermodynamic_label == "stable"


def test_distiller_requires_explicit_homologous_transfer_declaration(tmp_path):
    def run(declaration):
        memory = CareerMemory(str(tmp_path / ("declared.db" if declaration else "default.db")))
        campaign = memory.start_campaign("source", "li_sse", {})
        result = SimpleNamespace(
            structure_id="candidate", score=0.8, passes_filters=True,
            predictions={"max_force_ev_per_angstrom": 0.1},
        )
        strategy = {"elements": ["Li", "P", "S"], "constraints": {}}
        if declaration:
            strategy["memory_transfer_declaration"] = declaration
        ExperienceDistiller(memory).distill_iteration(
            campaign, "li_sse", 0, [_structure()], [(_structure(), result)], strategy
        )
        memory.end_campaign(campaign, {})
        target = extract_transferable_features(_structure("Na2PSe3", ["Na", "P", "Se"], "target"))
        return memory.get_transferable_directives(
            target_features=target, target_domain="na_sse", target_elements=["Na", "P", "Se"]
        )

    assert run(None)["directives"] == []
    declared = run({
        "allowed_relationship": "homologous_series",
        "source_chemical_system": ["Li", "P", "S"],
        "target_chemical_system": ["Na", "P", "Se"],
        "confidence": 0.7,
    })
    assert declared["directives"]
    assert declared["directives"][0]["applicability"]["allowed_relationship"] == "homologous_series"


def test_declared_target_and_mapping_are_enforced():
    source = extract_transferable_features(_structure("Li2PS3"))
    target = extract_transferable_features(_structure("Na2PS3", ["Na", "P", "S"], "target"))
    base = TransferableMemoryRecord(
        record_id="r", finalized=True, source_domain="x", features=source,
        applicability=ApplicabilityConstraint(
            allowed_relationship="homologous_series", confidence=0.8,
            source_chemical_system=["Li", "P", "S"], target_chemical_system=["Li", "P", "Se"],
        ),
    )
    ok, reasons = __import__("agents.transferable_memory", fromlist=["applicability_check"]).applicability_check(
        base, target, target_elements=["Na", "P", "S"], target_domain="x"
    )
    assert not ok and "DECLARED_TARGET_SYSTEM_MISMATCH" in reasons
    no_target = TransferableMemoryRecord(
        record_id="r2", finalized=True, source_domain="x", features=source,
        applicability=ApplicabilityConstraint(allowed_relationship="homologous_series", confidence=0.8),
    )
    ok, reasons = __import__("agents.transferable_memory", fromlist=["applicability_check"]).applicability_check(
        no_target, target, target_domain="x"
    )
    assert not ok and "EXPLICIT_TRANSFER_DECLARATION_REQUIRED" in reasons


def test_retrieval_uses_strict_campaign_chronology_and_audits_leakage(tmp_path):
    memory = CareerMemory(str(tmp_path / "chronology.db"))
    old = memory.start_campaign("old", "x", {})
    target = memory.start_campaign("target", "x", {})
    current = memory.start_campaign("current", "x", {})
    future = memory.start_campaign("future", "x", {})
    # Finalize source rows before replacing timestamps to keep lifecycle state
    # explicit and deterministic.
    for cid in (old, future):
        memory.end_campaign(cid, {})
    base_features = extract_transferable_features(_structure())
    target_features = extract_transferable_features(_structure())
    for cid, proto, finalized in ((old, "old", True), (current, "current", False), (future, "future", True)):
        features = TransferableFeatures(**{**base_features.to_dict(), "structural_prototype": proto})
        memory.store_transferable_record(TransferableMemoryRecord(
            record_id=cid, campaign_ids=[cid], source_domain="x", finalized=finalized,
            features=features,
            applicability=ApplicabilityConstraint(
                allowed_relationship="same_system", confidence=0.8,
                source_chemical_system=["Li", "P", "S"],
            ),
        ))
    memory.conn.execute("UPDATE campaigns SET start_time=?, end_time=? WHERE id=?", (10.0, 20.0, old))
    memory.conn.execute("UPDATE campaigns SET start_time=?, end_time=NULL WHERE id=?", (30.0, target))
    memory.conn.execute("UPDATE campaigns SET start_time=?, end_time=NULL WHERE id=?", (30.0, current))
    memory.conn.execute("UPDATE campaigns SET start_time=?, end_time=? WHERE id=?", (40.0, 50.0, future))
    memory.conn.commit()
    selection = memory.get_applicable_transferable_memories(
        target_features=target_features, target_domain="x", target_campaign_id=target
    )
    assert [item["record_id"] for item in selection["applied"]] == [old]
    rejected = {item["record_id"]: item["reasons"] for item in selection["rejected"]}
    assert "SOURCE_CAMPAIGN_NOT_FINALIZED" in rejected[current]
    assert "SOURCE_CAMPAIGN_NOT_PRIOR" in rejected[future]


def test_negative_evidence_is_retained_but_never_promoted(tmp_path):
    memory = CareerMemory(str(tmp_path / "negative.db"))
    campaign = memory.start_campaign("negative", "x", {})
    memory.store_transferable_evidence(
        campaign_id=campaign, domain="x", structure=_structure(),
        result={"passes_filters": False}, outcome_label="unstable",
    )
    memory.end_campaign(campaign, {})
    target = extract_transferable_features(_structure())
    selection = memory.get_applicable_transferable_memories(
        target_features=target, target_domain="x", target_elements=["Li", "P", "S"]
    )
    assert selection["applied"] == []
    assert selection["rejected"][0]["reasons"] == ["NEGATIVE_OUTCOME_NOT_A_DIRECTIVE"]
    directives = memory.get_transferable_directives(
        target_features=target, target_domain="x", target_elements=["Li", "P", "S"]
    )
    assert directives["directives"] == []
    assert directives["rejected"][0]["record"]["outcome_label"] == "screening_reject"


def test_duplicate_provenance_ids_merge_without_confidence_inflation(tmp_path):
    memory = CareerMemory(str(tmp_path / "dedup-provenance.db"))
    campaign_a = memory.start_campaign("a", "x", {})
    campaign_b = memory.start_campaign("b", "x", {})
    features = extract_transferable_features(_structure())
    base = TransferableMemoryRecord(
        record_id="evidence-a", evidence_ids=["evidence-a"], campaign_ids=[campaign_a],
        source_candidate_ids=["candidate-a"], source_formulas=["Li2PS3"], source_domain="x",
        outcome_label="stable", features=features,
        applicability=ApplicabilityConstraint(
            allowed_relationship="same_system", source_chemical_system=["Li", "P", "S"],
            confidence=0.4, evidence_count=1,
        ),
        directive=TransferDirective(source_evidence_ids=["evidence-a"], source_campaign_ids=[campaign_a]),
    )
    duplicate = TransferableMemoryRecord(
        record_id="evidence-b", evidence_ids=["evidence-b"], campaign_ids=[campaign_b],
        source_candidate_ids=["candidate-b"], source_formulas=["Li2PS3"], source_domain="x",
        outcome_label="stable", features=features,
        applicability=ApplicabilityConstraint(
            allowed_relationship="same_system", source_chemical_system=["Li", "P", "S"],
            confidence=0.9, evidence_count=9,
        ),
        directive=TransferDirective(source_evidence_ids=["evidence-b"], source_campaign_ids=[campaign_b]),
    )
    memory.store_transferable_record(base)
    memory.store_transferable_record(duplicate)
    merged = memory.conn.execute(
        "SELECT payload FROM transferable_records"
    ).fetchone()[0]
    record = TransferableMemoryRecord.from_dict(json.loads(merged))
    assert record.applicability.evidence_count == 1
    assert record.applicability.confidence == pytest.approx(0.4)
    assert record.evidence_ids == ["evidence-a", "evidence-b"]
    assert record.directive.source_evidence_ids == ["evidence-a", "evidence-b"]
    assert record.directive.source_campaign_ids == sorted([campaign_a, campaign_b])


def test_memory_modes_reach_distinct_planning_context_and_policy(tmp_path):
    memory = CareerMemory(str(tmp_path / "modes.db"))
    source = memory.start_campaign("source", "x", {})
    features = extract_transferable_features(_structure())
    for index, weight in enumerate((0.1, 0.4, 0.7)):
        memory.store_transferable_record(TransferableMemoryRecord(
            record_id=f"r{index}", campaign_ids=[source], finalized=False, source_domain="x",
            outcome_label="stable", features=TransferableFeatures(**{
                **features.to_dict(), "structural_prototype": f"prototype-{index}"
            }),
            applicability=ApplicabilityConstraint(
                allowed_relationship="same_system", source_chemical_system=["Li", "P", "S"],
                confidence=0.8,
            ),
            directive=TransferDirective(exploration_weight=weight),
        ))
    memory.end_campaign(source, {})
    objective = CampaignObjective(
        target_properties={"screening_quality": 1.0},
        constraints={"elements": ["Li", "P", "S"]},
        success_criteria={},
        domain="x",
    )
    prompts = {}
    strategies = {}
    for mode in ("none", "text_summary", "structured_provenance", "shuffled_control"):
        agent = OrchestratorAgent(career_memory=memory, memory_mode=mode, memory_seed=11)
        def capture(prompt, *, _mode=mode):
            prompts[_mode] = prompt
            return agent._default_strategy()
        agent._call_llm_for_strategy = capture
        strategies[mode] = agent.plan_iteration(objective, [], campaign_id="", iteration=0)
    assert "TRANSFERABLE MEMORY" not in prompts["none"]
    assert "TEXT VIEW" in prompts["text_summary"]
    assert "STRUCTURED TRANSFERABLE MEMORY" in prompts["structured_provenance"]
    assert "invalid for scientific decisions" in prompts["shuffled_control"]
    assert strategies["none"]["memory_policy"]["applied"] == []
    assert strategies["text_summary"]["memory_policy"]["applied"] == []
    assert strategies["structured_provenance"]["memory_policy"]["applied"]
    assert strategies["shuffled_control"]["memory_policy"]["applied"]
    assert strategies["shuffled_control"]["memory_directive_audit"]["scientific_decision_support"] is False


def test_pre_oracle_priority_is_bounded_and_changes_order_without_gate_changes():
    candidates = [
        _structure("Li2PS3", candidate_id="match"),
        _structure("LiPS", candidate_id="other"),
    ]
    directives = [{
        "record_id": "memory-r", "preferred_anonymous_stoichiometries": [
            extract_transferable_features(candidates[0]).anonymous_stoichiometric_pattern
        ],
        "exploitation_weight": 0.8, "bypasses_geometry_gate": False,
        "bypasses_thermodynamic_gate": False,
    }]
    ordered, audit = prioritize_candidates(candidates, directives)
    assert ordered[0]["candidate_id"] == "match"
    assert audit[0]["candidate_id"] == "match"
    assert audit[0]["matched"]
    assert all(item["score"] >= 0 for item in audit)


def test_priority_changes_oracle_admission_but_not_geometry_or_budget_accounting(monkeypatch):
    monkeypatch.setattr("agents.screening.HAS_CHGNET", False)
    match = _structure("Li2O", ["Li", "O"], "match")
    match["positions"] = [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]]
    # This candidate has the same proposal representation but occupies the
    # scarce oracle slot first in the baseline ordering.
    other = _structure("Na", ["Na"], "other")
    other["positions"] = [[0, 0, 0]]
    invalid = _structure("Bi2", ["Bi"], "invalid")
    invalid["positions"] = [[0, 0, 0], [0, 0, 0]]
    pattern = extract_transferable_features(match).anonymous_stoichiometric_pattern
    directives = [{
        "record_id": "memory-r", "preferred_anonymous_stoichiometries": [pattern],
        "exploitation_weight": 0.8, "bypasses_geometry_gate": False,
        "bypasses_thermodynamic_gate": False,
    }]
    candidates = [invalid, other, match]
    prioritized, priority_audit = prioritize_candidates(candidates, directives)
    assert prioritized[0]["candidate_id"] == "match"

    def run(batch):
        agent = ScreeningAgent()
        calls = []
        agent._predict = lambda struct, sid: calls.append(sid) or {FORCE_KEY: 0.1, STRESS_KEY: 0.1}
        tracker = DualBudgetTracker(proposal_budget=3, oracle_budget=1)
        tracker.record_proposals(len(batch))
        result = agent.screen_batch(batch, criteria={"max_force_ev_per_angstrom": 1.0},
                                    deduplicate=False, budget_tracker=tracker)
        return result, calls, tracker.to_dict()

    baseline, baseline_calls, baseline_budget = run(candidates)
    controlled, controlled_calls, controlled_budget = run(prioritized)
    assert baseline_calls == ["other"]
    assert controlled_calls == ["match"]
    assert baseline_budget["proposals_generated"] == controlled_budget["proposals_generated"] == 3
    assert baseline_budget["geometry_valid"] == controlled_budget["geometry_valid"] == 2
    assert baseline_budget["invalid_geometry"] == controlled_budget["invalid_geometry"] == 1
    assert baseline_budget["oracle_evaluations"] == controlled_budget["oracle_evaluations"] == 1
    assert baseline_budget["oracle_evaluations"] <= baseline_budget["oracle_budget"]
    assert controlled_budget["oracle_evaluations"] <= controlled_budget["oracle_budget"]
    # The geometry threshold and screening threshold remain caller-owned and
    # identical; memory only changed which valid proposal got the one slot.
    baseline_invalid = next(result for struct, result in baseline if result.structure_id == "invalid")
    controlled_invalid = next(result for struct, result in controlled if result.structure_id == "invalid")
    assert baseline_invalid.geometry_details["threshold"] == controlled_invalid.geometry_details["threshold"]
    assert all(item["score"] >= 0 for item in priority_audit)


def test_extraction_failure_is_structured_in_development_and_fail_closed_in_research(tmp_path, monkeypatch):
    memory = CareerMemory(str(tmp_path / "failure.db"))
    campaign = memory.start_campaign("source", "x", {})
    result = SimpleNamespace(structure_id="candidate", score=0.2, passes_filters=False, predictions={})
    distiller = ExperienceDistiller(memory)

    def fail(*args, **kwargs):
        raise ValueError("malformed structure adapter")

    monkeypatch.setattr("agents.experience_distiller.extract_transferable_features", fail)
    failures = distiller._store_candidates_with_provenance(
        campaign, "x", 0, [(_structure(), result)], [], [], strategy={}, fail_closed=False
    )
    assert failures and failures[0]["code"] == "TRANSFERABLE_FEATURE_EXTRACTION_FAILED"
    assert failures[0]["scientific_decision_support"] is False
    with pytest.raises(RuntimeError, match="Transferable memory extraction failed"):
        distiller._store_candidates_with_provenance(
            campaign, "x", 0, [(_structure(), result)], [], [], strategy={}, fail_closed=True
        )


def test_memory_replay_fields_roundtrip_through_manifest(tmp_path):
    tracker = ProvenanceTracker(
        campaign_id="target", campaign_name="target", domain="x",
        output_dir=tmp_path, master_seed=7,
        config={"memory_mode": "shuffled_control", "memory_seed": 19,
                "memory_transfer_declaration": {"allowed_relationship": "homologous_series"}},
        constraints={"memory_transfer_declaration": {"allowed_relationship": "homologous_series"}},
    )
    audit = {
        "mode": "shuffled_control", "seed": 19,
        "applied": ["r"], "rejected": [],
        "scientific_decision_support": False,
        "unsupported": [{"record_id": "r", "reasons": ["SHUFFLED_CONTROL"]}],
        "shuffle_audit": {"seed": 19, "permutation": [1, 2, 0], "valid": True},
    }
    tracker.record_strategy(0, {
        "elements": ["Li", "P", "S"], "memory_directives": [{"record_id": "r"}],
        "memory_directive_audit": audit,
        "memory_transfer_declaration": {"allowed_relationship": "homologous_series"},
    })
    tracker.record_memory_prioritization(
        [{"candidate_id": "c", "score": 1.0, "matched": ["r:prototype"]}], iteration=0
    )
    tracker.record_memory_extraction_audit(
        [{"candidate_id": "bad", "code": "TRANSFERABLE_FEATURE_EXTRACTION_FAILED"}], iteration=0
    )
    tracker.write_manifest()
    payload = json.loads((tmp_path / "manifest.json").read_text())
    assert payload["memory_mode"] == "shuffled_control"
    assert payload["memory_seed"] == 19
    assert payload["memory_transfer_declaration"]["allowed_relationship"] == "homologous_series"
    assert payload["memory_shuffle_audit"][0]["permutation"] == [1, 2, 0]
    assert payload["memory_priority_audit"][0]["candidate_id"] == "c"
    assert payload["memory_extraction_failures"][0]["code"] == "TRANSFERABLE_FEATURE_EXTRACTION_FAILED"
    assert payload["strategies"][0]["memory_directive_audit"]["unsupported"]
    roundtrip = tracker.manifest.from_dict(payload)
    assert roundtrip.memory_mode == "shuffled_control"
    assert roundtrip.memory_transfer_declaration["allowed_relationship"] == "homologous_series"


def test_checkpoint_and_reproduction_restore_memory_controls(tmp_path):
    declaration = {"allowed_relationship": "homologous_series", "target_chemical_system": ["Na", "P", "Se"]}
    tracker = ProvenanceTracker(
        campaign_id="target", campaign_name="target", domain="x",
        output_dir=tmp_path / "original", master_seed=7,
        config={"memory_mode": "shuffled_control", "memory_seed": 19,
                "memory_transfer_declaration": declaration,
                "use_career_memory": False, "use_validation": False,
                "use_synthesis": False, "require_thermodynamics": False,
                "num_candidates": 1},
        constraints={"elements": ["Li", "P", "S"], "memory_transfer_declaration": declaration},
        objective={},
    )
    tracker.manifest.iteration_seeds = [42]
    tracker.record_strategy(0, {
        "elements": ["Li", "P", "S"], "num_candidates": 1,
        "memory_directives": [{"record_id": "r"}],
        "memory_directive_audit": {
            "mode": "shuffled_control", "seed": 19, "applied": ["r"],
            "rejected": [], "unsupported": [{"record_id": "r", "reasons": ["CONTROL"]}],
            "scientific_decision_support": False,
            "shuffle_audit": {"seed": 19, "permutation": [1, 2, 0], "valid": True},
        },
        "memory_transfer_declaration": declaration,
    })
    tracker.record_memory_prioritization([{"candidate_id": "c", "score": 1.0}], iteration=0)
    tracker.write_manifest()

    # Exercise the actual campaign checkpoint serializer with a light-weight
    # object; the full reproduction below exercises constructor/config replay.
    from campaign import MaterialsDiscoveryCampaign
    checkpoint_campaign = object.__new__(MaterialsDiscoveryCampaign)
    checkpoint_campaign.iteration = 0
    checkpoint_campaign.campaign_id = "target"
    checkpoint_campaign.results_history = []
    checkpoint_campaign.termination_reason = None
    checkpoint_campaign.provenance = tracker
    checkpoint_campaign.budget_tracker = DualBudgetTracker(proposal_budget=1, oracle_budget=1)
    checkpoint_campaign._log = lambda message: None
    checkpoint_campaign.config = SimpleNamespace(
        output_dir=tmp_path / "original", name="target", geometry_min_distance=0.8,
        proposal_budget=1, oracle_budget=1,
        thermodynamics_reference_set_path=None,
        thermodynamics_retain_threshold_ev_per_atom=0.1,
        thermodynamics_stable_threshold_ev_per_atom=0.03,
        memory_mode="shuffled_control", memory_seed=19,
        objective=SimpleNamespace(domain="x", constraints={"elements": ["Li", "P", "S"],
                                               "memory_transfer_declaration": declaration}),
    )
    checkpoint_campaign._save_checkpoint()
    checkpoint = json.loads((tmp_path / "original" / "checkpoint_0.json").read_text())
    assert checkpoint["memory"]["mode"] == "shuffled_control"
    assert checkpoint["memory"]["seed"] == 19
    assert checkpoint["memory"]["transfer_declaration"] == declaration
    assert checkpoint["memory"]["shuffle_audit"][0]["permutation"] == [1, 2, 0]
    assert checkpoint["memory"]["priority_audit"][0]["candidate_id"] == "c"

    reproduced = MaterialsDiscoveryCampaign.reproduce_from_manifest(
        tmp_path / "original" / "manifest.json", output_dir=tmp_path / "reproduced"
    )
    assert reproduced.config.memory_mode == "shuffled_control"
    assert reproduced.config.memory_seed == 19
    assert reproduced.config.objective.constraints["memory_transfer_declaration"] == declaration
    assert reproduced.provenance.manifest.strategies[0]["memory_directive_audit"]["shuffle_audit"]["permutation"] == [1, 2, 0]


def test_legacy_database_migration_is_additive_and_quarantined(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE campaigns (id TEXT PRIMARY KEY, name TEXT NOT NULL, domain TEXT NOT NULL, objective TEXT NOT NULL, start_time REAL, end_time REAL, iterations INTEGER DEFAULT 0, total_generated INTEGER DEFAULT 0, total_screened INTEGER DEFAULT 0, best_score REAL DEFAULT 0.0, success_rate REAL DEFAULT 0.0, summary TEXT)")
    conn.execute("CREATE TABLE candidates (id TEXT PRIMARY KEY, campaign_id TEXT, domain TEXT, formula TEXT, score REAL, passed_screening INTEGER, hypothesis_ids TEXT, principle_ids TEXT, properties TEXT, iteration INTEGER, created_at REAL)")
    conn.execute("INSERT INTO campaigns (id,name,domain,objective) VALUES ('legacy','legacy','x','{}')")
    conn.execute("INSERT INTO candidates (id,campaign_id,domain,formula,score,passed_screening,properties) VALUES ('legacy-c','legacy','x','LiPS',0.9,1,'{}')")
    conn.commit()
    conn.close()
    memory = CareerMemory(str(path))
    tables = {row[0] for row in memory.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "transferable_records" in tables
    assert memory.get_transferable_records() == []
    assert memory.get_top_candidates_ever(domain="x") == []
    props = memory.conn.execute("SELECT properties FROM candidates WHERE id='legacy-c'").fetchone()[0]
    assert json.loads(props) == {}
