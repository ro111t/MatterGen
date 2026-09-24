"""Frozen ICLR selection protocol. No oracle or model is available to the ranker.

Artifacts are create-only canonical bytes. Hashes, rather than file permissions,
are authoritative. Generation and scoring are separated by ProposalReplay.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
from functools import reduce
from math import gcd, isfinite
from pathlib import Path
import hashlib
import json
import os
import re
import tempfile

PROTOCOL = "iclr_common_proposals_v1"
CONTRACT = {
    "protocol_version": PROTOCOL,
    "generation_role": "common_proposal_stream",
    "research_composition_conditioning": "disabled",
    "descriptor_version": "anonymous_integer_ratio_v1",
    "ranker_version": "beta_exact_match_v1",
    "prior_alpha": 1, "prior_beta": 1,
    "historical_effective_weight": 1, "local_observation_weight": 1,
    "update_timing": "between_batches", "tie_break": "proposal_index",
    "response_threshold_ev_per_atom": 0.10, "response_comparison": "<=",
    "shuffle_version": "sha256_order_cyclic_v1",
    "text_format_version": "anonymous_source_counts_v1",
}
HEADER = ("Anonymous-stoichiometry source evidence v1\n"
          "Outcome: predicted E_hull <= 0.10 eV/atom in the source campaign.\n")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def create_only(path, data):
    """Atomically publish complete bytes without replacing any existing artifact."""
    path = Path(path)
    missing = []
    parent = path.parent
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        fsync_directory(directory.parent)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".publish-")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)  # Atomic, and fails if destination exists.
        fsync_directory(path.parent)  # Marker publication is durable before returning.
    finally:
        os.unlink(temporary)


def read_json(path, expected):
    raw = Path(path).read_bytes()
    if not expected or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Protocol artifact SHA256 mismatch: {path}")
    value = json.loads(raw)
    if canonical(value) != raw:
        raise ValueError(f"Noncanonical protocol artifact: {path}")
    return value


def descriptor(structure):
    if not structure.is_ordered or len(structure) == 0:
        raise ValueError("Descriptor requires nonempty, fully occupied ordered structure")
    from pymatgen.core import Element
    counts = Counter()
    for site in structure:
        if len(site.species) != 1 or next(iter(site.species.values())) != 1:
            raise ValueError("Fractional occupancy is not supported")
        symbol = site.specie.symbol
        if not Element.is_valid_symbol(symbol):
            raise ValueError("Dummy/invalid species")
        counts[symbol] += 1
    divisor = reduce(gcd, counts.values())
    return tuple(sorted(n // divisor for n in counts.values()))


def valid_descriptor(value):
    if (not isinstance(value, (list, tuple)) or not value
            or any(type(n) is not int or n <= 0 for n in value)
            or list(value) != sorted(value) or reduce(gcd, value) != 1):
        raise ValueError("Invalid anonymous_integer_ratio_v1 descriptor")
    return tuple(value)


def encode_structure(structure):
    descriptor(structure)
    def number(value):
        value = float(value)
        if not isfinite(value):
            raise ValueError("Nonfinite structure coordinate")
        return value.hex()
    return {"lattice": [[number(x) for x in row] for row in structure.lattice.matrix],
            "species": [site.specie.symbol for site in structure],
            "fractional_coordinates": [[number(x) for x in row] for row in structure.frac_coords]}


def decode_structure(value):
    from pymatgen.core import Structure
    s = Structure([[float.fromhex(x) for x in row] for row in value["lattice"]],
                  value["species"], [[float.fromhex(x) for x in row]
                                     for row in value["fractional_coordinates"]])
    if encode_structure(s) != value:
        raise ValueError("Invalid/noncanonical structure representation")
    return s


def batch_sizes(total, count):
    if type(total) is not int or type(count) is not int or total < count or count < 1:
        raise ValueError("Every scheduled batch must contain proposals")
    sizes = []
    for remaining_batches in range(count, 0, -1):
        n = (total + remaining_batches - 1) // remaining_batches
        sizes.append(n)
        total -= n
    return sizes


def proposal_identity(spec):
    # No condition, memory, output directory or target oracle result may enter.
    return {"protocol": CONTRACT, "task_id": spec.task_id, "seed": spec.seed,
            "elements": sorted(spec.elements), "iteration_seeds": list(spec.iteration_seeds),
            "batch_sizes": batch_sizes(spec.proposal_budget, len(spec.iteration_seeds)),
            "checkpoint_sha256": spec.mattergen_checkpoint_sha256,
            "sampling_sha256": spec.mattergen_sampling_config_sha256,
            "mattergen_batch_size": spec.mattergen_batch_size,
            "generator_code_sha256": file_hash(Path(__file__).resolve().parents[1] / "agents/generator.py"),
            "artifact_code_sha256": file_hash(__file__),
            "authorized_code": spec.authorized_code_identity, "parent": spec.parent_experiment_hash}


def generate_stream(path, identity, factory):
    """factory is called only for a new artifact, under exclusive generation lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".generation-lock")
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise ValueError("Proposal generation already active or interrupted; refusing regeneration") from exc
    success = False
    try:
        if path.exists():
            stream = read_json(path, file_hash(path))
        else:
            backend = factory()
            batches = []
            index = 0
            for b, (seed, size) in enumerate(zip(identity["iteration_seeds"], identity["batch_sizes"])):
                candidates = backend.generate_batch(elements=identity["elements"], num_candidates=size,
                                                    seed=seed, target_compositions_dict=[])
                if len(candidates) != size or backend.last_generation_backend != "mattergen":
                    raise ValueError("Common proposal generation failed exact count/backend contract")
                rows = []
                for position, s in enumerate(candidates):
                    if not {e.symbol for e in s.composition.elements} <= set(identity["elements"]):
                        raise ValueError("Proposal violates common chemical system")
                    data = encode_structure(s)
                    rows.append({"candidate_id": f"P-{digest(identity)[:16]}-{index:06d}",
                                 "proposal_index": index, "batch": b, "position": position,
                                 "descriptor": descriptor(s), "structure": data,
                                 "structure_sha256": digest(data)})
                    index += 1
                batches.append({"rows": rows, "sha256": digest(rows)})
            from agents.provenance import SoftwareEnvironment
            stream = {"identity": identity, "batches": batches,
                      "generation_environment": SoftwareEnvironment.capture().to_dict()}
            create_only(path, canonical(stream))
            create_only(path.with_suffix(".sha256"), file_hash(path).encode("ascii"))
        validate_stream(stream, identity)
        if path.with_suffix(".sha256").read_text() != file_hash(path):
            raise ValueError("Proposal stream seal mismatch")
        success = True
        return file_hash(path)
    finally:
        if success:
            lock.rmdir()
        # An interrupted/failed construction is not automatically retried.


def validate_stream(stream, identity):
    if stream["identity"] != identity or len(stream["batches"]) != len(identity["batch_sizes"]):
        raise ValueError("Proposal stream identity/boundaries mismatch")
    index = 0
    for b, batch in enumerate(stream["batches"]):
        rows = batch["rows"]
        if len(rows) != identity["batch_sizes"][b] or digest(rows) != batch["sha256"]:
            raise ValueError("Proposal batch hash/count mismatch")
        for position, row in enumerate(rows):
            expected_id = f"P-{digest(identity)[:16]}-{index:06d}"
            if (row["candidate_id"] != expected_id or row["proposal_index"] != index
                    or row["batch"] != b or row["position"] != position
                    or digest(row["structure"]) != row["structure_sha256"]):
                raise ValueError("Proposal identity/order/hash mismatch")
            s = decode_structure(row["structure"])
            if descriptor(s) != valid_descriptor(row["descriptor"]):
                raise ValueError("Proposal descriptor mismatch")
            if not {e.symbol for e in s.composition.elements} <= set(identity["elements"]):
                raise ValueError("Proposal chemical-system mismatch")
            index += 1


class ProposalReplay:
    backend_name = "mattergen"
    last_generation_backend = None

    def __init__(self, path, sha256, identity):
        self.path, self.sha256, self.identity = path, sha256, identity
        if Path(path).with_suffix(".sha256").read_text() != sha256:
            raise ValueError("Proposal stream seal mismatch")
        self.stream = read_json(path, sha256)
        validate_stream(self.stream, identity)
        self.cursor = 0

    def generate_batch(self, elements, num_candidates, seed, target_compositions_dict=None, **kwargs):
        read_json(self.path, self.sha256)
        b = self.cursor
        if (target_compositions_dict or sorted(elements) != self.identity["elements"]
                or b >= len(self.stream["batches"]) or seed != self.identity["iteration_seeds"][b]
                or num_candidates != self.identity["batch_sizes"][b]):
            raise ValueError("Replay request differs from immutable proposal schedule")
        result = []
        for row in self.stream["batches"][b]["rows"]:
            s = decode_structure(row["structure"])
            s.properties["_candidate_id"] = row["candidate_id"]
            result.append(s)
        self.cursor += 1
        self.last_generation_backend = "mattergen"
        return result


def counts(observations):
    result = {}
    seen = set()
    for row in observations:
        d = valid_descriptor(row["descriptor"])
        if type(row["y"]) is not int or row["y"] not in (0, 1) or row["id"] in seen:
            raise ValueError("Invalid/duplicate evidence observation")
        seen.add(row["id"])
        n, s = result.get(d, (0, 0))
        result[d] = (n + 1, s + row["y"])
    return result


def render_text(stats):
    return HEADER + "".join(f"Descriptor {canonical(d).decode()}: evaluated={n}; successful={s}.\n"
                            for d, (n, s) in sorted(stats.items()))


def parse_text(raw):
    text = raw.decode("utf-8")
    if not text.startswith(HEADER):
        raise ValueError("Invalid source text header")
    stats = {}
    for line in text[len(HEADER):].splitlines(keepends=True):
        match = re.fullmatch(r"Descriptor (\[[0-9,]+\]): evaluated=([1-9][0-9]*); successful=(0|[1-9][0-9]*)\.\n", line)
        if not match:
            raise ValueError("Invalid source text grammar")
        d = valid_descriptor(json.loads(match[1]))
        n, s = int(match[2]), int(match[3])
        if d in stats or s > n:
            raise ValueError("Invalid/duplicate source text counts")
        stats[d] = n, s
    if render_text(stats).encode() != raw:
        raise ValueError("Source text is not canonical")
    return stats


def shuffle(observations, seed):
    counts(observations)
    ordered = sorted(observations, key=lambda r: (digest(["source-shuffle-v1", seed, r["id"]]), r["id"]))
    if len(ordered) < 2:
        raise ValueError("Shuffled contrast unavailable: fewer than two observations")
    mixed, mapping = [], []
    for i, row in enumerate(ordered):
        response = ordered[(i + 1) % len(ordered)]
        mixed.append({"id": row["id"], "descriptor": row["descriptor"], "y": response["y"]})
        mapping.append({"descriptor_id": row["id"], "response_id": response["id"]})
    if counts(mixed) == counts(observations):
        raise ValueError("Shuffled contrast degenerate: descriptor-response sufficient statistics unchanged")
    assert sorted(r["y"] for r in mixed) == sorted(r["y"] for r in observations)
    return mixed, {"version": CONTRACT["shuffle_version"], "seed": seed, "mapping": mapping,
                   "fixed_points": 0, "valid": True, "input_hash": digest(observations),
                   "output_hash": digest(mixed), "associations_changed": True}


def priority(d, local, historical):
    n, s = local.get(d, (0, 0))
    hn, hs = historical.get(d, (0, 0))
    return (Fraction(1 + s) + (Fraction(hs, hn) if hn else 0)) / (2 + n + bool(hn))


class BatchPolicy:
    """Only primitive past evidence enters scoring; no evaluator/model handles."""
    def __init__(self, condition, historical=None, metadata=None):
        self.condition = condition
        self.historical = dict(historical or {})
        self.metadata = dict(metadata or {})
        self.local = []
        self.completed_batches = 0
        self.pending = None
        self.audit = []
        self.observations = []

    def rank(self, candidates, rows, batch):
        if self.pending is not None or batch != self.completed_batches:
            raise ValueError("Evidence cutoff/batch ordering violation")
        local = counts(self.local)
        audits = []
        for s, row in zip(candidates, rows, strict=True):
            d = descriptor(s)
            if d != valid_descriptor(row["descriptor"]):
                raise ValueError("Rank input descriptor mismatch")
            score = priority(d, local, self.historical)
            n, successes = local.get(d, (0, 0))
            hn, hs = self.historical.get(d, (0, 0))
            audits.append({"candidate_id": row["candidate_id"], "descriptor": d,
                           "proposal_index": row["proposal_index"], "local_n": n, "local_s": successes,
                           "historical_n": hn, "historical_s": hs,
                           "numerator": score.numerator, "denominator": score.denominator, "score": float(score)})
        random = self.condition in {"random_mattergen", "source_neutral"}
        order = sorted(range(len(rows)), key=lambda i: (Fraction(0) if random else
                       -Fraction(audits[i]["numerator"], audits[i]["denominator"]), rows[i]["proposal_index"]))
        batch_audit = {"batch": batch, "evidence_cutoff": batch - 1, "local_evidence_hash": digest(self.local),
                       "historical_counts_hash": digest(render_text(self.historical)),
                       "order": [rows[i]["candidate_id"] for i in order],
                       "scores": [dict(audits[i], rank=rank) for rank, i in enumerate(order)],
                       "policy": CONTRACT, **self.metadata}
        batch_audit["state_hash"] = digest(batch_audit)
        self.pending = batch_audit
        return [candidates[i] for i in order], batch_audit

    def complete(self, screened, rows, batch, scientific_identity):
        if self.pending is None or batch != self.completed_batches:
            raise ValueError("No matching frozen batch")
        if [result.structure_id for _, result in screened] != self.pending["order"]:
            raise ValueError("Cannot commit incomplete or reordered batch observations")
        row_by_id = {r["candidate_id"]: r for r in rows}
        observations = []
        for structure, result in screened:
            cid = result.structure_id
            if cid not in row_by_id or any(r["id"] == cid for r in observations):
                raise ValueError("Unexpected/duplicate evaluated candidate")
            p = result.predictions
            e = p.get("predicted_energy_above_hull_ev_per_atom")
            eligible = (result.geometry_valid is True and result.oracle_evaluated is True
                        and type(result.oracle_call_index) is int and result.oracle_call_index > 0
                        and result.scientific_validity == "research_valid"
                        and result.backend == "chgnet_thermodynamic_oracle"
                        and result.provenance_stage == "thermodynamics"
                        and p.get("thermodynamics_certified") is True
                        and isinstance(e, (int, float)) and not isinstance(e, bool) and isfinite(e)
                        and p.get("reference_set_hash") == scientific_identity["reference_set_hash"]
                        and p.get("model") == scientific_identity["model"]
                        and p.get("relaxation_settings") == scientific_identity["relaxation_settings"])
            row = {"id": cid, "batch": batch, "descriptor": row_by_id[cid]["descriptor"],
                   "proposal_index": row_by_id[cid]["proposal_index"],
                   "structure_sha256": row_by_id[cid]["structure_sha256"],
                   "oracle_call_index": result.oracle_call_index,
                   "eligible": eligible, "reason": "CERTIFIED_OUTCOME" if eligible else "NO_ELIGIBLE_CERTIFIED_OUTCOME",
                   "y": int(e <= 0.10) if eligible else None,
                   "oracle_cache_hit": result.oracle_cache_hit,
                   "predicted_energy_above_hull_ev_per_atom": e,
                   "scientific_identity": scientific_identity}
            observations.append(row)
        # Commit only after the whole batch; no results can reach rank mid-batch.
        self.local.extend({"id": r["id"], "descriptor": r["descriptor"], "y": r["y"]}
                          for r in observations if r["eligible"])
        self.observations.extend(observations)
        self.audit.append(self.pending)
        self.pending = None
        self.completed_batches += 1

    def snapshot(self):
        return {"protocol": CONTRACT, "condition": self.condition, "historical": render_text(self.historical),
                "metadata": self.metadata, "local": self.local, "completed_batches": self.completed_batches,
                "pending": self.pending, "audit": self.audit, "observations": self.observations}
