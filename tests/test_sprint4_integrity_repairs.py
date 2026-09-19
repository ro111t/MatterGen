"""Adversarial regression tests for Sprint 4 scientific-integrity repairs."""

from __future__ import annotations

from agents.career_memory import CareerMemory
from agents.provenance import RunManifest
from agents.transferable_memory import (
    ApplicabilityConstraint,
    TransferDirective,
    TransferableFeatures,
    TransferableMemoryRecord,
    view_records,
)


def _features(index: int = 0) -> TransferableFeatures:
    return TransferableFeatures(
        anonymous_stoichiometric_pattern=f"A{index}B",
        structural_prototype=f"prototype-{index}",
        element_classes={"Li": "alkali", "P": "pnictogen", "S": "chalcogen"},
    )


def _record(index: int, outcome_label: str | None) -> TransferableMemoryRecord:
    return TransferableMemoryRecord(
        record_id=f"record-{index}",
        campaign_ids=["source"],
        source_domain="li_sse",
        outcome_label=outcome_label,
        features=_features(index),
        applicability=ApplicabilityConstraint(
            allowed_relationship="same_system",
            source_chemical_system=["Li", "P", "S"],
            confidence=0.8,
        ),
        directive=TransferDirective(
            exploration_weight=0.1 + index / 10,
            exploitation_weight=0.9 - index / 10,
        ),
    )


def test_only_explicit_positive_outcomes_can_become_directives(tmp_path):
    memory = CareerMemory(str(tmp_path / "outcome-gate.db"))
    source = memory.start_campaign("source", "li_sse", {})
    # Include spelling/format aliases that have historically slipped through
    # a negative-only filter.  None and unrecognized labels must also fail.
    labels = [
        "unknown",
        None,
        "missing",
        "failure",
        "failed",
        "rejected",
        "rejection",
        "unstable",
        "thermodynamically-unstable",
        "instability",
        "screening failed",
        "arbitrary_success_claim",
        "stable",
        "retained",
        "screening_pass",
        "thermodynamically_stable",
    ]
    for index, label in enumerate(labels):
        record = _record(index, label)
        record.campaign_ids = [source]
        memory.store_transferable_record(record)
    memory.end_campaign(source, {})

    selected = memory.get_transferable_directives(
        target_features=_features(0),
        target_domain="li_sse",
        target_elements=["Li", "P", "S"],
    )
    assert {item["record"]["outcome_label"] for item in selected["rejected"]} >= {
        "unknown", "missing", "failure", "failed", "rejected", "rejection",
        "unstable", "thermodynamically-unstable", "instability", "screening failed",
        "arbitrary_success_claim",
    }
    assert {item["record"]["outcome_label"] for item in selected["applied"]} == {
        "stable", "retained", "screening_pass", "thermodynamically_stable",
    }
    assert {
        directive["record_id"] for directive in selected["directives"]
    } == {f"record-{labels.index(label)}" for label in labels[-4:]}


def test_manifest_hash_canonicalizes_and_covers_memory_extraction_failures():
    first = RunManifest(
        domain="li_sse",
        config={"z": {3, 1}, "a": {"nested": [2, 1]}},
        memory_transfer_declaration={"target": ["Se", "P"], "source": ["S", "P"]},
        memory_extraction_failures=[
            {"candidate_id": "c1", "code": "FEATURE_EXTRACTION_FAILED", "details": {"b": 2, "a": 1}}
        ],
    )
    second = RunManifest(
        domain="li_sse",
        config={"a": {"nested": [2, 1]}, "z": {1, 3}},
        memory_transfer_declaration={"source": ["S", "P"], "target": ["Se", "P"]},
        memory_extraction_failures=[
            {"candidate_id": "c1", "code": "FEATURE_EXTRACTION_FAILED", "details": {"a": 1, "b": 2}}
        ],
    )
    assert first.compute_manifest_hash() == second.compute_manifest_hash()

    original_hash = first.compute_manifest_hash()
    first.memory_extraction_failures[0]["code"] = "DIFFERENT_FAILURE"
    assert first.compute_manifest_hash() != original_hash


def test_shuffled_control_moves_nested_provenance_with_response_bundle():
    records = []
    for index in range(4):
        record = _record(index, f"outcome-{index}").to_dict()
        # Deliberately use a nested marker that cannot be reconstructed from
        # the descriptor, so stale provenance is detected directly.
        record["provenance"] = {
            "evidence_marker": index,
            "source_candidate_ids": [f"candidate-{index}"],
        }
        records.append(record)

    first = view_records(records, mode="shuffled_control", seed=37)
    second = view_records(records, mode="shuffled_control", seed=37)
    assert first == second
    audit = first["shuffle_audit"]
    assert audit["valid"] is True
    assert audit["fixed_points"] == 0
    assert sorted(audit["permutation"]) == list(range(len(records)))

    # Every descriptor remains at its original position, while every response
    # field—including nested provenance—comes from the paired response row.
    for descriptor_index, shown in enumerate(first["records"]):
        response_index = audit["permutation"][descriptor_index]
        assert shown["features"] == records[descriptor_index]["features"]
        assert shown["paired_descriptor_record_id"] == records[descriptor_index]["record_id"]
        assert shown["paired_response_record_id"] == records[response_index]["record_id"]
        assert shown["outcome_label"] == records[response_index]["outcome_label"]
        assert shown["directive"] == records[response_index]["directive"]
        assert shown["evidence_ids"] == records[response_index]["evidence_ids"]
        assert shown["evidence_hash"] == records[response_index]["evidence_hash"]
        assert shown["provenance"] == records[response_index]["provenance"]

    # The control preserves response marginals exactly, even though pairing is
    # intentionally broken.
    for field in ("outcome_label", "evidence_ids", "evidence_hash", "provenance"):
        assert sorted(
            repr(item[field]) for item in first["records"]
        ) == sorted(repr(item[field]) for item in records)


def test_shuffled_control_is_unavailable_for_empty_or_singleton_inputs():
    empty = view_records([], mode="shuffled_control", seed=9)
    singleton = view_records([_record(0, "stable")], mode="shuffled_control", seed=9)

    assert empty["records"] == []
    assert empty["shuffle_audit"]["valid"] is False
    assert empty["shuffle_audit"]["reason"] == "SHUFFLE_CONTROL_EMPTY"
    assert singleton["records"] == []
    assert singleton["shuffle_audit"]["valid"] is False
    assert singleton["shuffle_audit"]["reason"] == "SHUFFLE_CONTROL_UNAVAILABLE_SINGLETON"
