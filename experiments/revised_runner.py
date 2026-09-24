"""Execution boundary for the frozen common-proposal ICLR protocol.

Durable per-request journal permits replay of completed work. A crash with an
in-flight oracle request is intentionally not retryable: its external cost is
uncertain and cannot be silently charged twice or made free.
"""
from dataclasses import asdict, replace
from pathlib import Path
import json
import time

from agents.generator import GenerationAgent
from agents.integrity import RunMode
from agents.orchestrator import CampaignObjective
from agents.provenance import CandidateStatus
from agents.research_execution import VerifiedResearchExecution, hash_path
from agents.screening import ScreeningResult
from campaign import CampaignConfig, MaterialsDiscoveryCampaign
from experiments.release_integrity import locate
from experiments.selection_protocol import (
    PROTOCOL, CONTRACT, BatchPolicy, ProposalReplay, canonical, counts, create_only,
    digest, file_hash, generate_stream, parse_text, proposal_identity, read_json,
    render_text, shuffle,
)


def proposal_factory(spec):
    if hash_path(Path(spec.mattergen_model_path)) != spec.mattergen_checkpoint_sha256:
        raise ValueError("MatterGen checkpoint SHA256 mismatch")
    if hash_path(Path(spec.mattergen_sampling_config_path)) != spec.mattergen_sampling_config_sha256:
        raise ValueError("Sampling configuration SHA256 mismatch")
    return lambda: GenerationAgent(use_mattergen=True, run_mode="research",
        mattergen_model_path=spec.mattergen_model_path, mattergen_pretrained=spec.mattergen_pretrained,
        mattergen_batch_size=spec.mattergen_batch_size,
        mattergen_sampling_config_path=spec.mattergen_sampling_config_path)


def prepare(spec, stream_path=None):
    # Compatibility name is replay-only. Initialization belongs to its DAG node.
    from experiments.release_integrity import bind_pair
    return bind_pair(spec)


def load_run_spec(directory, root):
    from experiments.spec import RunSpec
    spec = RunSpec.from_dict(json.loads((Path(directory) / "run_spec.json").read_text()))
    changes = {"artifact_root": str(root), "output_dir": str(directory)}
    for field in ("reference_set_path", "mattergen_model_path", "mattergen_sampling_config_path"):
        value = getattr(spec, field)
        if value and spec.artifact_root:
            try:
                changes[field] = str(Path(root) / Path(value).relative_to(spec.artifact_root))
            except ValueError:
                pass
    return replace(spec, **changes)


def freeze_source(source_dir, path, seed, root=None):
    """Export only a verified completed source run; never trust caller labels."""
    source_dir = Path(source_dir)
    from experiments.spec import RunSpec
    source_spec = load_run_spec(source_dir, root) if root is not None else RunSpec.from_dict(json.loads((source_dir / "run_spec.json").read_text()))
    if source_spec.condition != "source_neutral" or source_spec.seed != seed or not completed(source_spec):
        raise ValueError("Source acquisition is not a verified completed paired source campaign")
    body = json.loads((source_dir / "selection_observations.json").read_text())
    observations = [{"id": row["id"], "descriptor": row["descriptor"], "y": row["y"]}
                    for row in body["observations"] if row["eligible"]]
    counts(observations)
    # Freeze one control before any target arm. Degeneracy blocks the complete
    # paired comparison rather than selecting a favorable alternative shuffle.
    mixed, audit = shuffle(observations, seed)
    corpus = {"protocol": CONTRACT, "seed": seed, "source_spec_hash": source_spec.spec_hash,
              "source_run_dir": str(source_dir.resolve().relative_to(Path(source_spec.artifact_root).resolve())), "source_task": source_spec.task_id,
              "source_elements": sorted(source_spec.elements),
              "source_integrity_sha256": file_hash(source_dir / "run_integrity.json"),
              "observations_sha256": file_hash(source_dir / "selection_observations.json"),
              "observations": observations, "eligibility_audit": body["observations"],
              "acquisition_cost": body["budget"], "shuffled": mixed, "shuffle_audit": audit}
    raw = canonical(corpus)
    text_path = Path(path).with_suffix(".txt")
    text = render_text(counts(observations)).encode()
    if Path(path).exists():
        if Path(path).read_bytes() != raw or text_path.read_bytes() != text:
            raise ValueError("Frozen source artifact cannot be replaced")
    else:
        create_only(text_path, text)
        create_only(path, raw)
    receipt_path = Path(path).with_suffix(".receipt.json")
    receipt = {"protocol": CONTRACT, "parent": source_spec.parent_experiment_hash,
               "seed": seed, "source_elements": sorted(source_spec.elements),
               "corpus_sha256": file_hash(path), "text_sha256": file_hash(text_path),
               "source_integrity_sha256": corpus["source_integrity_sha256"],
               "acquisition_cost": corpus["acquisition_cost"], "shuffle_valid": True}
    if receipt_path.exists():
        if receipt_path.read_bytes() != canonical(receipt):
            raise ValueError("Frozen verification receipt changed")
    else:
        create_only(receipt_path, canonical(receipt))
    return file_hash(path), str(text_path), file_hash(text_path)


def historical(spec):
    metadata = {"protocol": CONTRACT, "proposal_stream_sha256": spec.proposal_stream_sha256}
    if spec.condition == "source_neutral":
        return {}, metadata
    from experiments.release_integrity import locate
    receipt = read_json(locate(spec, spec.source_receipt_manifest), spec.source_receipt_sha256)
    if (receipt["protocol"] != CONTRACT or receipt["parent"] != spec.parent_experiment_hash
            or receipt["seed"] != spec.seed or receipt["shuffle_valid"] is not True
            or receipt["text_sha256"] != spec.source_text_sha256
            or receipt["corpus_sha256"] != spec.source_corpus_sha256):
        raise ValueError("Frozen source verification receipt mismatch")
    declaration = spec.memory_transfer_declaration or {}
    if (sorted(declaration.get("source_chemical_system", [])) != receipt["source_elements"]
            or sorted(declaration.get("target_chemical_system", [])) != sorted(spec.elements)
            or declaration.get("allowed_relationship") not in {"same_system", "homologous_series"}):
        raise ValueError("Explicit source-target transfer declaration required")
    metadata.update(source_receipt_sha256=spec.source_receipt_sha256,
                    parent_experiment_hash=spec.parent_experiment_hash,
                    proposal_binding_sha256=spec.proposal_binding_sha256,
                    source_acquisition_cost=receipt["acquisition_cost"])
    stats = {}
    if spec.condition == "text_summary_memory":
        text_path = locate(spec, spec.source_text_path)
        text_bytes = text_path.read_bytes()
        import hashlib
        if hashlib.sha256(text_bytes).hexdigest() != receipt["text_sha256"]:
            raise ValueError("Frozen text SHA256 mismatch")
        stats = parse_text(text_bytes)
    elif spec.condition in {"structured_provenance_memory", "shuffled_memory_control"}:
        corpus = read_json(locate(spec, spec.source_corpus_manifest), receipt["corpus_sha256"])
        if corpus["protocol"] != CONTRACT or corpus["seed"] != spec.seed:
            raise ValueError("Source protocol/seed mismatch")
        if spec.condition == "structured_provenance_memory":
            stats = counts(corpus["observations"])
        else:
            mixed, audit = shuffle(corpus["observations"], spec.seed)
            if mixed != corpus["shuffled"] or audit != corpus["shuffle_audit"]:
                raise ValueError("Frozen shuffle mismatch")
            stats = counts(mixed)
            metadata["shuffle_audit"] = audit
    metadata["translated_counts_hash"] = digest(render_text(stats))
    return stats, metadata


class Journal:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        from experiments.selection_protocol import fsync_directory
        fsync_directory(self.directory.parent)
        self.events = []
        self.tail = None
        for i, path in enumerate(sorted(self.directory.glob("*.json"))):
            if path.name != f"{i:06d}.json":
                raise ValueError("Journal gap")
            event = json.loads(path.read_bytes())
            if event["previous"] != self.tail or event["sha256"] != digest(event["payload"]):
                raise ValueError("Journal hash chain mismatch")
            if path.read_bytes() != canonical(event):
                raise ValueError("Noncanonical journal")
            self.events.append(event["payload"])
            self.tail = digest(event)
        if self.events and self.events[-1]["type"] == "oracle_begin":
            raise ValueError("Interrupted in-flight oracle request: cost uncertain; automatic retry forbidden")

    def append(self, event):
        body = {"previous": self.tail, "payload": event, "sha256": digest(event)}
        create_only(self.directory / f"{len(self.events):06d}.json", canonical(body))
        self.events.append(event)
        self.tail = digest(body)


class RevisedCampaign(MaterialsDiscoveryCampaign):
    def __init__(self, config, spec, replay, policy, journal):
        self.revised_spec, self.replay, self.policy, self.journal = spec, replay, policy, journal
        super().__init__(config)

    def _init_generator(self):
        return self.replay

    def run_campaign(self):
        spec = self.revised_spec
        self.campaign_id = "selection-" + spec.spec_hash[:24]
        self.provenance.campaign_id = self.campaign_id
        self.provenance.manifest.campaign_id = self.campaign_id
        self.provenance.manifest.config["selection_protocol"] = CONTRACT
        self.provenance.manifest.config["selection_artifacts"] = self.policy.metadata
        if spec.condition == "shuffled_memory_control":
            self.provenance.manifest.memory_shuffle_audit = self.policy.metadata["shuffle_audit"]
        if not self.journal.events:
            self.journal.append({"type": "identity", "spec_hash": spec.spec_hash,
                                 "metadata": self.policy.metadata})
        expected = {"type": "identity", "spec_hash": spec.spec_hash, "metadata": self.policy.metadata}
        if self.journal.events[0] != expected:
            raise ValueError("Journal belongs to another run/protocol")
        event_cursor = 1
        scientific_identity = {"reference_set_hash": self._research_execution.frozen_reference_set.reference_set_hash,
                               "model": asdict(self._research_execution.model_identity),
                               "relaxation_settings": asdict(self._research_execution.relaxation_settings)}
        started = time.monotonic()
        for b, size in enumerate(self.replay.identity["batch_sizes"]):
            self.iteration = b
            candidates = self.replay.generate_batch(spec.elements, size, spec.iteration_seeds[b])
            rows = self.replay.stream["batches"][b]["rows"]
            self.budget_tracker.record_proposals(len(candidates))
            self.provenance.register_generation(candidates=candidates, iteration=b, backend="mattergen",
                seed=spec.iteration_seeds[b], target_elements=spec.elements,
                parameters={"proposal_stream_sha256": spec.proposal_stream_sha256,
                            "batch_sha256": self.replay.stream["batches"][b]["sha256"],
                            "target_compositions_dict": [], "selection_protocol": PROTOCOL},
                model_name_or_path=spec.mattergen_model_path, checkpoint=spec.mattergen_pretrained)
            ordered, audit = self.policy.rank(candidates, rows, b)
            rank_event = {"type": "ranking", "audit": audit}
            if event_cursor < len(self.journal.events):
                if canonical(self.journal.events[event_cursor]) != canonical(rank_event):
                    raise ValueError("Resume ranking/evidence cutoff mismatch")
            else:
                self.journal.append(rank_event)
            event_cursor += 1
            self.provenance.manifest.config.setdefault("selection_ranking_audit", []).append(audit)
            if spec.condition in {"structured_provenance_memory", "text_summary_memory", "shuffled_memory_control"}:
                self.provenance.record_memory_prioritization(
                    [row for row in audit["scores"] if row["historical_n"] > 0], iteration=b)
            screened = []
            for s in ordered:
                cid = s.properties["_candidate_id"]
                before = self.budget_tracker.to_dict()
                if event_cursor < len(self.journal.events):
                    begin = self.journal.events[event_cursor]
                    end = self.journal.events[event_cursor + 1]
                    if (begin != {"type": "oracle_begin", "candidate_id": cid, "budget_before": before}
                            or end["type"] != "oracle_result" or end["candidate_id"] != cid):
                        raise ValueError("Resume oracle order/budget mismatch")
                    result = ScreeningResult(**end["result"])
                    self.budget_tracker.record_geometry(result.geometry_valid is True)
                    if result.oracle_evaluated:
                        if not self.budget_tracker.admit_oracle(cache_hit=bool(result.oracle_cache_hit)):
                            raise ValueError("Journal exceeds oracle budget")
                        if result.oracle_call_index != self.budget_tracker.oracle_evaluations:
                            raise ValueError("Journal oracle index mismatch")
                    if self.budget_tracker.to_dict() != end["budget_after"]:
                        raise ValueError("Journal budget accounting mismatch")
                else:
                    self.journal.append({"type": "oracle_begin", "candidate_id": cid, "budget_before": before})
                    result = self.screener.screen_batch([s], criteria={}, target_properties=spec.target_properties,
                                deduplicate=False, budget_tracker=self.budget_tracker)[0][1]
                    self.journal.append({"type": "oracle_result", "candidate_id": cid,
                                         "result": asdict(result), "budget_after": self.budget_tracker.to_dict()})
                event_cursor += 2
                screened.append((s, result))
            self.policy.complete(screened, rows, b, scientific_identity)
            batch_event = {"type": "batch_complete", "snapshot": self.policy.snapshot(),
                           "budget": self.budget_tracker.to_dict(), "proposal_cursor": self.replay.cursor}
            if event_cursor < len(self.journal.events):
                if canonical(self.journal.events[event_cursor]) != canonical(batch_event):
                    raise ValueError("Resume completed batch state mismatch")
            else:
                self.journal.append(batch_event)
            event_cursor += 1
            if hasattr(self.screener, "_update_novelty_scores"):
                self.screener._update_novelty_scores(screened)
            screened.sort(key=lambda item: item[1].score, reverse=True)
            for rank, (_, result) in enumerate(screened, 1):
                result.rank = rank
            self.provenance.record_screening(screened, criteria={}, iteration=b, backend="chgnet_thermodynamic_oracle")
            for s, result in screened:
                if result.passes_filters:
                    self.provenance.record_decision(candidate_id=result.structure_id,
                        status=CandidateStatus.ACCEPTED, ranking_score=result.score, iteration_rank=result.rank)
            self.results_history.append({"generation_backend": "mattergen",
                                         "insights": {"thermodynamics_metrics_available": True}})
        if event_cursor != len(self.journal.events):
            raise ValueError("Unexpected trailing journal events")
        self.termination_reason = "PROPOSAL_BUDGET_EXHAUSTED"
        self.budget_tracker.set_termination(self.termination_reason)
        data = {"protocol": CONTRACT, "spec_hash": spec.spec_hash, "observations": self.policy.observations,
                "budget": self.budget_tracker.to_dict(), "batches": self.policy.audit,
                "journal_tail": self.journal.tail}
        output = self.config.output_dir / "selection_observations.json"
        if output.exists():
            if output.read_bytes() != canonical(data):
                raise ValueError("Completed observations cannot be overwritten")
        else:
            create_only(output, canonical(data))
        self.provenance.manifest.config["selection_trajectory_sha256"] = digest({
            "orders": [b["order"] for b in self.policy.audit],
            "outcomes": [{k: row[k] for k in ("id", "eligible", "y", "oracle_call_index")}
                         for row in self.policy.observations]})
        return self._generate_final_report(time.monotonic() - started)


def completed(spec):
    directory = Path(spec.output_dir)
    try:
        saved = json.loads((directory / "run_spec.json").read_text())
        from experiments.spec import RunSpec
        from experiments.release_integrity import bind_pair
        spec = bind_pair(spec)
        if RunSpec.from_dict(saved).identity_dict() != spec.identity_dict():
            return False
        integrity = json.loads((directory / "run_integrity.json").read_text())
        if integrity["protocol_version"] != PROTOCOL or integrity["spec_hash"] != spec.spec_hash:
            return False
        if integrity["proposal_stream_sha256"] != spec.proposal_stream_sha256:
            return False
        ProposalReplay(locate(spec, spec.proposal_stream_manifest), spec.proposal_stream_sha256, proposal_identity(spec))
        if spec.condition != "source_neutral":
            historical(spec)
        for name, sha in integrity["artifacts"].items():
            if Path(name).is_absolute() or ".." in Path(name).parts or file_hash(directory / name) != sha:
                return False
        required = {"manifest.json", "campaign_provenance.json", "selection_observations.json", "report.json"}
        if not required <= set(integrity["artifacts"]):
            return False
        journal = Journal(directory / "selection_journal")
        if journal.tail != integrity["journal_tail"]:
            return False
        from experiments.runner import CampaignRunner
        manifest = json.loads((directory / "manifest.json").read_text())
        return manifest["status"] == "completed" and bool(CampaignRunner._canonical_manifest_hash(manifest))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def execute(spec, force_rerun=False):
    if spec.protocol_version != PROTOCOL or spec.protocol != CONTRACT or spec.run_mode != "research":
        raise ValueError("Unsupported revised protocol")
    if spec.oracle_budget != 100 or spec.thermodynamics_retain_threshold_ev_per_atom != 0.10:
        raise ValueError("Frozen revised research requires the 100-call/0.10 endpoint contract")
    from experiments.release_integrity import bind_pair
    spec = bind_pair(spec)
    directory = Path(spec.output_dir)
    ctx = VerifiedResearchExecution.verify(spec)
    replay = ProposalReplay(locate(spec, spec.proposal_stream_manifest), spec.proposal_stream_sha256, proposal_identity(spec))
    stats, metadata = historical(spec)
    policy = BatchPolicy(spec.condition, stats, metadata)
    if completed(spec):
        if force_rerun:
            raise ValueError("Cannot force-rerun immutable completed research artifacts")
        return {"run_id": spec.run_id, "status": "SUCCESS", "skipped": True,
                "manifest_path": str(directory / "manifest.json")}
    if (directory / "run_integrity.json").exists():
        raise ValueError("Finalized research artifact failed integrity verification; refusing repair or rerun")
    spec_path = directory / "run_spec.json"
    if directory.exists() and any(directory.iterdir()):
        if force_rerun or not spec_path.exists() or load_run_spec(directory, spec.artifact_root).identity_dict() != spec.identity_dict():
            raise ValueError("Old/unrelated experiment artifacts are not resume compatible")
        if not (directory / "selection_journal" / "000000.json").exists():
            raise ValueError("Missing revised execution journal")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        create_only(spec_path, canonical(spec.to_dict()))
    journal = Journal(directory / "selection_journal")
    objective = CampaignObjective(target_properties=dict(spec.target_properties),
        constraints={"elements": list(spec.elements), "experiment_condition": spec.condition},
        success_criteria={}, domain=spec.domain, max_iterations=len(spec.iteration_seeds))
    config = CampaignConfig(name=spec.run_id, objective=objective, output_dir=directory,
        strategy_mode=spec.strategy_mode, master_seed=spec.seed, proposal_budget=spec.proposal_budget, oracle_budget=spec.oracle_budget,
        geometry_min_distance=spec.geometry_min_distance, use_career_memory=False,
        use_validation=False, use_synthesis=False, use_mattergen=True, run_mode=RunMode.RESEARCH,
        mattergen_pretrained=spec.mattergen_pretrained, mattergen_model_path=spec.mattergen_model_path,
        mattergen_batch_size=spec.mattergen_batch_size, mattergen_sampling_config_path=spec.mattergen_sampling_config_path,
        thermodynamics_reference_set_path=spec.reference_set_path, require_thermodynamics=True,
        thermodynamics_backend="chgnet", research_execution=ctx, research_spec_hash=spec.spec_hash,
        thermodynamics_retain_threshold_ev_per_atom=spec.thermodynamics_retain_threshold_ev_per_atom,
        thermodynamics_stable_threshold_ev_per_atom=spec.thermodynamics_stable_threshold_ev_per_atom,
        locked_elements=spec.elements, allow_llm_orchestration=False, validation_calculator="disabled",
        synthesis_mode="disabled", memory_mode=spec.memory_mode, memory_seed=spec.memory_seed)
    campaign = RevisedCampaign(config, spec, replay, policy, journal)
    campaign.run_campaign()
    artifacts = {name: file_hash(directory / name) for name in
                 ("manifest.json", "campaign_provenance.json", "report.json", "selection_observations.json")}
    integrity = {"protocol_version": PROTOCOL, "spec_hash": spec.spec_hash,
                 "proposal_stream_sha256": spec.proposal_stream_sha256,
                 "journal_tail": journal.tail, "artifacts": artifacts}
    create_only(directory / "run_integrity.json", canonical(integrity))
    return {"run_id": spec.run_id, "status": "SUCCESS", "skipped": False,
            "manifest_path": str(directory / "manifest.json")}
