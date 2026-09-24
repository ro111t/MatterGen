"""Acceptance tests for the frozen common-proposal selection experiment."""
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace
import json

import pytest
from pymatgen.core import Lattice, Structure

from experiments.selection_protocol import (
    CONTRACT, PROTOCOL, BatchPolicy, ProposalReplay, batch_sizes, canonical, counts,
    create_only, decode_structure, descriptor, digest, encode_structure, file_hash,
    generate_stream, parse_text, priority, proposal_identity, read_json, render_text, shuffle,
)
from experiments.revised_runner import Journal


def structure(species):
    return Structure(Lattice.cubic(10), species,
                     [[i / len(species), 0, 0] for i in range(len(species))])


@pytest.mark.parametrize("species,expected", [
    (["Li"] * 3 + ["P"] + ["Se"] * 4, (1, 3, 4)),
    (["Li"] * 6 + ["P"] * 2 + ["Se"] * 8, (1, 3, 4)),
    (["Se", "Li", "P"], (1, 1, 1)), (["Li", "Li", "Se"], (1, 2)), (["Li"] * 4, (1,)),
])
def test_descriptor_and_lossless_structure(species, expected):
    s = structure(species)
    assert descriptor(s) == expected
    assert descriptor(structure(list(reversed(species)))) == expected
    encoded = encode_structure(s)
    assert encode_structure(decode_structure(encoded)) == encoded


def test_fractional_dummy_and_empty_descriptors_fail():
    for s in [Structure(Lattice.cubic(5), [{"Li": 0.5}], [[0, 0, 0]]),
              Structure(Lattice.cubic(5), ["X"], [[0, 0, 0]]), Structure(Lattice.cubic(5), [], [])]:
        with pytest.raises(ValueError):
            descriptor(s)


def test_exact_beta_equation_and_source_cap():
    d = (1, 3, 4)
    assert priority(d, {}, {}) == Fraction(1, 2)
    assert priority(d, {}, {d: (10, 10)}) == Fraction(2, 3)
    assert priority(d, {}, {d: (1000, 1000)}) == Fraction(2, 3)
    assert priority(d, {d: (2, 0)}, {d: (3, 2)}) == Fraction(1, 3)
    assert priority(d, {d: (100, 0)}, {d: (1, 1)}) == Fraction(2, 103)
    assert priority(d, {}, {(1,): (4, 4)}) == Fraction(1, 2)


def evidence():
    return [{"id": "a", "descriptor": [1], "y": 1},
            {"id": "b", "descriptor": [1, 1], "y": 0}]


def test_shuffle_exact_cyclic_contract_and_marginals():
    source = evidence()
    mixed, audit = shuffle(source, 42)
    ordered = sorted(source, key=lambda r: (digest(["source-shuffle-v1", 42, r["id"]]), r["id"]))
    assert audit["mapping"] == [{"descriptor_id": ordered[i]["id"],
                                 "response_id": ordered[(i + 1) % 2]["id"]} for i in range(2)]
    assert all(x["descriptor_id"] != x["response_id"] for x in audit["mapping"])
    assert sorted(x["y"] for x in mixed) == sorted(x["y"] for x in source)
    assert sorted(x["descriptor"] for x in mixed) == sorted(x["descriptor"] for x in source)
    assert counts(mixed) != counts(source)
    assert shuffle(source, 42) == (mixed, audit)


@pytest.mark.parametrize("source", [[], evidence()[:1],
    [{"id": "a", "descriptor": [1], "y": 0}, {"id": "b", "descriptor": [1], "y": 1}],
    [{"id": "a", "descriptor": [1], "y": 1}, {"id": "b", "descriptor": [1, 1], "y": 1}]])
def test_shuffle_degeneracy_fails_without_retry(source):
    with pytest.raises(ValueError):
        shuffle(source, 42)


def test_text_is_exact_sufficient_statistic_not_a_hidden_lookup():
    stats = counts(evidence())
    raw = render_text(stats).encode()
    assert b"Descriptor [1]: evaluated=1; successful=1.\n" in raw
    assert parse_text(raw) == stats
    for bad in [raw.replace(b"\n", b"\r\n"), raw[:-1], raw + b"explanation\n",
                raw.replace(b"successful=1", b"successful=2"), raw + raw.splitlines(keepends=True)[2],
                raw.replace(b"evaluated=1", b"evaluated=01")]:
        with pytest.raises(ValueError):
            parse_text(bad)


def identity():
    return {"elements": ["Li", "P"], "task_id": "Li-P", "seed": 42,
            "batch_sizes": [2, 2], "iteration_seeds": [42, 43], "protocol": CONTRACT}


class Factory:
    last_generation_backend = "mattergen"
    def generate_batch(self, **kwargs):
        assert kwargs["target_compositions_dict"] == []
        return [structure(["Li", "P"]), structure(["Li"])]


def test_common_stream_replay_identity_hash_and_no_regeneration(tmp_path):
    path = tmp_path / "stream.json"
    sha = generate_stream(path, identity(), Factory)
    assert generate_stream(path, identity(), lambda: pytest.fail("regenerated")) == sha
    streams = []
    for _ in range(5):
        replay = ProposalReplay(path, sha, identity())
        batches = [replay.generate_batch(["Li", "P"], 2, seed) for seed in [42, 43]]
        streams.append([[encode_structure(s) for s in batch] for batch in batches])
    assert all(x == streams[0] for x in streams)
    with pytest.raises(ValueError):
        ProposalReplay(path, "0" * 64, identity())
    wrong = dict(identity(), seed=43)
    with pytest.raises(ValueError):
        ProposalReplay(path, sha, wrong)
    with pytest.raises(FileExistsError):
        create_only(path, b"overwrite")


def test_stream_generation_shortfall_is_not_retried(tmp_path):
    path = tmp_path / "stream.json"
    with pytest.raises(ValueError):
        generate_stream(path, identity(), lambda: SimpleNamespace(
            generate_batch=lambda **k: [], last_generation_backend="mattergen"))
    with pytest.raises(ValueError, match="interrupted"):
        generate_stream(path, identity(), Factory)


def test_batch_evidence_isolation_exact_ties_and_condition_behavior(tmp_path):
    path = tmp_path / "stream.json"
    sha = generate_stream(path, identity(), Factory)
    replay = ProposalReplay(path, sha, identity())
    candidates = replay.generate_batch(["Li", "P"], 2, 42)
    rows = replay.stream["batches"][0]["rows"]
    stats = counts(evidence())
    a = BatchPolicy("structured_provenance_memory", stats)
    t = BatchPolicy("text_summary_memory", parse_text(render_text(stats).encode()))
    assert a.rank(candidates, rows, 0)[1] == t.rank(candidates, rows, 0)[1]
    assert a.pending["order"] == [rows[1]["candidate_id"], rows[0]["candidate_id"]]
    random = BatchPolicy("random_mattergen", stats)
    assert random.rank(candidates, rows, 0)[1]["order"] == [r["candidate_id"] for r in rows]
    local = BatchPolicy("adaptive_no_memory")
    assert local.rank(candidates, rows, 0)[1]["order"] == [r["candidate_id"] for r in rows]
    with pytest.raises(ValueError):
        local.rank(candidates, rows, 1)
    sci = {"reference_set_hash": "ref", "model": {}, "relaxation_settings": {}}
    from agents.screening import ScreeningResult
    results = [(s, ScreeningResult(structure_id=r["candidate_id"], score=0, passes_filters=True,
        filter_reasons=[], geometry_valid=True, oracle_evaluated=True, oracle_call_index=i + 1,
        scientific_validity="research_valid", backend="chgnet_thermodynamic_oracle", provenance_stage="thermodynamics",
        predictions={**sci, "thermodynamics_certified": True,
                     "predicted_energy_above_hull_ev_per_atom": 0.0 if len(s) == 1 else 0.2}))
        for i, (s, r) in enumerate(zip(candidates, rows))]
    assert local.local == []
    local.complete(results, rows, 0, sci)
    next_candidates = replay.generate_batch(["Li", "P"], 2, 43)
    next_rows = replay.stream["batches"][1]["rows"]
    assert local.rank(next_candidates, next_rows, 1)[1]["order"] == [next_rows[1]["candidate_id"], next_rows[0]["candidate_id"]]


def test_journal_replay_hashes_and_inflight_cost(tmp_path):
    j = Journal(tmp_path / "journal")
    j.append({"type": "identity"})
    j.append({"type": "ranking"})
    assert Journal(tmp_path / "journal").events == j.events
    j.append({"type": "oracle_begin"})
    with pytest.raises(ValueError, match="cost uncertain"):
        Journal(tmp_path / "journal")
    j.append({"type": "oracle_result"})
    assert Journal(tmp_path / "journal").tail == j.tail


def test_schema_rejects_old_experiments_and_modified_equation():
    from experiments.spec import ExperimentSpec, ExperimentSpecError
    with pytest.raises(ExperimentSpecError):
        ExperimentSpec(experiment_id="old", schema_version="2.0.0")
    spec = ExperimentSpec(experiment_id="new")
    assert spec.to_dict()["selection_protocol"] == PROTOCOL
    assert ExperimentSpec.from_dict(spec.to_dict()).spec_hash == spec.spec_hash


@pytest.fixture
def revised_fixture(tmp_path, monkeypatch):
    from tests.test_research_execution_boundary import _research_spec, VerifiedResearchExecution
    from experiments.revised_runner import RevisedCampaign
    from agents.screening import ScreeningResult
    from dataclasses import asdict
    original, _, evaluator = _research_spec(tmp_path)
    original_verify = VerifiedResearchExecution.verify
    monkeypatch.setattr(VerifiedResearchExecution, "verify", classmethod(
        lambda cls, spec: original_verify(spec, evaluator=evaluator)))
    spec = replace(original, run_id="source", condition="source_neutral", proposal_budget=200,
                   oracle_budget=100, iteration_seeds=[42, 43, 44, 45, 46],
                   protocol_version=PROTOCOL, protocol=dict(CONTRACT), output_dir=str(tmp_path / "source"))
    calls = []

    class Screener:
        last_backend_used = "chgnet_thermodynamic_oracle"
        def screen_batch(self, structures, budget_tracker, **kwargs):
            s = structures[0]
            cid = s.properties["_candidate_id"]
            calls.append(cid)
            budget_tracker.record_geometry(True)
            if not budget_tracker.admit_oracle():
                return [(s, ScreeningResult(cid, {}, 0, False, ["ORACLE_BUDGET_EXHAUSTED"],
                    geometry_valid=True, oracle_evaluated=False, provenance_stage="oracle_budget",
                    geometry_failure_code="ORACLE_BUDGET_EXHAUSTED"))]
            predictions = {"reference_set_hash": original_verify(spec, evaluator=evaluator).frozen_reference_set.reference_set_hash,
                "model": asdict(evaluator.model_identity), "relaxation_settings": asdict(evaluator.relaxation_settings),
                "thermodynamics_certified": True, "predicted_energy_above_hull_ev_per_atom": 0.0 if len(s) == 1 else 0.2}
            return [(s, ScreeningResult(cid, predictions, 50, len(s) == 1, [],
                geometry_valid=True, oracle_evaluated=True, oracle_call_index=budget_tracker.oracle_evaluations,
                backend=self.last_backend_used, provenance_stage="thermodynamics", scientific_validity="research_valid"))]

    monkeypatch.setattr(RevisedCampaign, "_init_screener", lambda self: Screener())
    # Real verified context/preflight remain in force; only expensive candidate evaluation is substituted.
    class Proposals:
        last_generation_backend = "mattergen"
        def generate_batch(self, **kwargs):
            return [structure(["Li"] if i % 2 else ["Li", "P"]) for i in range(kwargs["num_candidates"])]
    from tests.revised_helpers import authorize
    from experiments.release_integrity import initialize_pair
    spec = initialize_pair(authorize(spec, tmp_path, monkeypatch), Proposals)
    return spec, calls, Proposals


def test_full_five_condition_execution_and_equivalent_text(revised_fixture, tmp_path):
    from experiments.revised_runner import execute, completed, freeze_source
    spec, calls, factory = revised_fixture
    execute(spec)
    assert completed(spec)
    source = json.loads((tmp_path / "source" / "selection_observations.json").read_text())
    assert source["budget"]["oracle_evaluations"] == 100
    assert source["budget"]["proposals_generated"] == 200
    assert sum(r["eligible"] for r in source["observations"]) == 100
    assert all(r["y"] is None for r in source["observations"] if not r["eligible"])
    corpus = tmp_path / "corpus.json"
    sha, text_path, text_sha = freeze_source(tmp_path / "source", corpus, 42)
    stream = spec.proposal_stream_manifest
    stream_sha = spec.proposal_stream_sha256
    from experiments.spec import FIVE_CONDITIONS
    outputs = {}
    for condition in FIVE_CONDITIONS:
        target = replace(spec, run_id=condition, condition=condition,
            strategy_mode="fixed" if condition == "random_mattergen" else "adaptive",
            source_memory_snapshot_path=str(corpus), source_corpus_manifest=str(corpus.relative_to(tmp_path)),
            source_corpus_sha256=sha, source_text_path=str(__import__("pathlib").Path(text_path).relative_to(tmp_path)), source_text_sha256=text_sha,
            source_receipt_manifest="corpus.receipt.json", source_receipt_sha256=file_hash(tmp_path / "corpus.receipt.json"),
            proposal_stream_manifest=str(stream), proposal_stream_sha256=stream_sha,
            memory_transfer_declaration={"allowed_relationship": "same_system",
                "source_chemical_system": spec.elements, "target_chemical_system": spec.elements},
            output_dir=str(tmp_path / condition))
        execute(target)
        assert completed(target)
        outputs[condition] = json.loads((tmp_path / condition / "selection_observations.json").read_text())
        assert outputs[condition]["budget"]["oracle_evaluations"] == 100
        assert outputs[condition]["budget"]["proposals_generated"] == 200
        prior_calls = len(calls)
        assert execute(target)["skipped"] is True
        assert len(calls) == prior_calls
    structured = outputs["structured_provenance_memory"]
    text = outputs["text_summary_memory"]
    assert [b["order"] for b in structured["batches"]] == [b["order"] for b in text["batches"]]
    assert structured["observations"] == text["observations"]
    assert outputs["adaptive_no_memory"]["batches"][0]["order"] == outputs["random_mattergen"]["batches"][0]["order"]
    assert outputs["adaptive_no_memory"]["batches"][1]["order"] != outputs["random_mattergen"]["batches"][1]["order"]
    assert structured["batches"][0]["historical_counts_hash"] != outputs["shuffled_memory_control"]["batches"][0]["historical_counts_hash"]


def test_interruption_after_durable_result_resumes_without_repaying(revised_fixture, monkeypatch, tmp_path):
    from experiments.revised_runner import execute, completed
    spec, calls, _ = revised_fixture
    original_append = Journal.append
    triggered = []
    def interrupt(self, event):
        original_append(self, event)
        if event["type"] == "oracle_result" and not triggered:
            triggered.append(True)
            raise KeyboardInterrupt("simulated post-result interruption")
    monkeypatch.setattr(Journal, "append", interrupt)
    with pytest.raises(KeyboardInterrupt):
        execute(spec)
    first = calls[0]
    monkeypatch.setattr(Journal, "append", original_append)
    execute(spec)
    assert completed(spec)
    assert calls.count(first) == 1
    data = json.loads((tmp_path / "source" / "selection_observations.json").read_text())
    assert data["budget"]["oracle_evaluations"] == 100


def test_research_dag_roundtrip_uses_corpus_for_every_arm(tmp_path):
    from experiments.spec import ExperimentSpec
    from experiments.dag import ExperimentDAG, NodeType
    # Schema's scientific pin validation is exercised separately; this test
    # isolates the research DAG topology/path rules without real dependencies.
    spec = ExperimentSpec(experiment_id="dag", master_seeds=[42], output_root=str(tmp_path))
    object.__setattr__(spec, "run_mode", "research")
    dag = ExperimentDAG(spec)
    snapshot = next(n for n in dag.nodes.values() if n.node_type == NodeType.SOURCE_MEMORY_SNAPSHOT)
    assert snapshot.expected_output_path == str(tmp_path / "source_evidence" / "seed42.json")
    for node in dag.nodes.values():
        if node.node_type == NodeType.TARGET_CAMPAIGN_RUN:
            assert snapshot.node_id in node.dependencies
    assert ExperimentDAG.from_dict(dag.to_dict(), spec).to_dict() == dag.to_dict()


def test_proposal_seal_rejects_rehashed_modified_stream(tmp_path):
    path = tmp_path / "stream.json"
    sha = generate_stream(path, identity(), Factory)
    data = read_json(path, sha)
    data["identity"]["seed"] = 99
    path.chmod(0o644)
    path.write_bytes(canonical(data))
    with pytest.raises(ValueError, match="seal"):
        ProposalReplay(path, file_hash(path), dict(identity(), seed=99))


def test_finalized_artifact_tampering_cannot_trigger_repair(revised_fixture, tmp_path):
    from experiments.revised_runner import execute, completed, freeze_source
    spec, calls, _ = revised_fixture
    execute(spec)
    before = len(calls)
    (tmp_path / "source" / "report.json").write_text("{}")
    assert not completed(spec)
    with pytest.raises(ValueError, match="Finalized"):
        execute(spec)
    with pytest.raises(ValueError, match="verified completed"):
        freeze_source(tmp_path / "source", tmp_path / "bad.json", 42)
    assert len(calls) == before


def test_invalid_or_missing_source_results_are_not_negatives(tmp_path):
    from agents.screening import ScreeningResult
    path = tmp_path / "stream.json"
    sha = generate_stream(path, identity(), Factory)
    replay = ProposalReplay(path, sha, identity())
    candidates = replay.generate_batch(["Li", "P"], 2, 42)
    rows = replay.stream["batches"][0]["rows"]
    scientific = {"reference_set_hash": "ref", "model": {}, "relaxation_settings": {}}
    for changes in [{"oracle_evaluated": False}, {"geometry_valid": False},
                    {"scientific_validity": "demo_only"}, {"backend": "heuristic"},
                    {"provenance_stage": "oracle_budget"},
                    {"predictions": {}},
                    {"predictions": {**scientific, "thermodynamics_certified": False,
                                     "predicted_energy_above_hull_ev_per_atom": 0.2}}]:
        policy = BatchPolicy("source_neutral")
        policy.rank(candidates, rows, 0)
        results = []
        for s, row in zip(candidates, rows):
            kwargs = dict(structure_id=row["candidate_id"], score=0, passes_filters=False,
                filter_reasons=[], geometry_valid=True, oracle_evaluated=True, oracle_call_index=1,
                scientific_validity="research_valid", backend="chgnet_thermodynamic_oracle",
                provenance_stage="thermodynamics", predictions={**scientific,
                "thermodynamics_certified": True, "predicted_energy_above_hull_ev_per_atom": 0.2})
            kwargs.update(changes)
            results.append((s, ScreeningResult(**kwargs)))
        policy.complete(results, rows, 0, scientific)
        assert policy.local == []
        assert all(r["y"] is None and not r["eligible"] for r in policy.observations)
