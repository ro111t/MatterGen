"""
CareerMemory — persistent cross-campaign, cross-domain knowledge store.

The core novel contribution of MatAgent. Unlike session memory (resets per
campaign) or plain logging (no transfer), CareerMemory accumulates structured
scientific knowledge across every campaign and domain the agent has run:

  1. Cross-campaign learning  — iteration 1 of campaign N benefits from all
                                 prior campaigns in the same domain.
  2. Cross-domain transfer    — Li-SSE insights warm-start Na-SSE campaigns.
  3. Failure attribution      — agent learns *why* candidates failed.
  4. Hypothesis lineage       — every candidate traces back to the principle
                                 that motivated its generation.

Persists to an explicitly supplied SQLite path.  A home-directory default is
intentionally not provided: implicit global state can contaminate independent
campaigns and invalidate scientific comparisons.
"""

import sqlite3
import json
import time
import uuid
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from agents.db_schema import ALL_SCHEMAS
from agents.integrity import SCHEMA_VERSION, ScientificValidity
from agents.transferable_memory import (
    ApplicabilityConstraint,
    TransferableFeatures,
    TransferableMemoryRecord,
    TransferDirective,
    applicability_check,
    evidence_fingerprint,
    extract_transferable_features,
    make_directive,
    view_records,
)


# A transferable record may contain useful negative/unknown evidence, but only
# labels emitted by a validated positive observation are allowed to reach the
# planner.  Keep this allow-list deliberately small: treating an arbitrary
# label such as ``success`` or ``accepted`` as positive would turn malformed
# or stale records into executable scientific advice.
DIRECTIVE_ELIGIBLE_OUTCOME_LABELS = frozenset({
    "stable",
    "retained",
    "screening_pass",
    "thermodynamically_stable",
})
_DIRECTIVE_NEGATIVE_OUTCOME_LABELS = frozenset({
    "unstable",
    "thermodynamically_unstable",
    "instable",
    "instability",
    "rejected",
    "rejection",
    "screening_reject",
    "screening_failed",
    "failed",
    "failure",
    "error",
})
_DIRECTIVE_UNKNOWN_OUTCOME_LABELS = frozenset({
    "",
    "unknown",
    "missing",
    "none",
    "null",
    "na",
    "n_a",
    "not_available",
    "unavailable",
    "no_result",
})


def _normalize_outcome_label(label: Any) -> str:
    """Canonicalize outcome labels before applying the directive gate."""
    if label is None:
        return ""
    # Labels are persisted as user/backend data, so tolerate common spelling
    # variants without ever expanding the positive allow-list implicitly.
    return re.sub(r"_+", "_", re.sub(r"[\s-]+", "_", str(label).strip().casefold()))


def _directive_outcome_rejection_reason(label: Any) -> Optional[str]:
    """Return a stable audit reason, or ``None`` for an eligible outcome."""
    normalized = _normalize_outcome_label(label)
    if normalized in DIRECTIVE_ELIGIBLE_OUTCOME_LABELS:
        return None
    if normalized in _DIRECTIVE_NEGATIVE_OUTCOME_LABELS:
        return "NEGATIVE_OUTCOME_NOT_A_DIRECTIVE"
    if normalized in _DIRECTIVE_UNKNOWN_OUTCOME_LABELS:
        return "OUTCOME_MISSING_OR_UNKNOWN"
    return "OUTCOME_NOT_ELIGIBLE_FOR_DIRECTIVE"


class CareerMemory:
    """
    Persistent knowledge store that accumulates scientific experience across
    all campaigns and domains. Survives process restarts via SQLite.
    """

    def __init__(self, db_path: str):
        legacy_home_db = Path.home() / ".matagent_career.db"
        if (
            not db_path
            or str(db_path).strip() == "~"
            or Path(db_path).expanduser().resolve() == legacy_home_db.resolve()
        ):
            raise ValueError("CareerMemory requires an explicit run-local db_path")
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        c = self.conn.cursor()
        for schema in ALL_SCHEMAS:
            c.execute(schema)
        self.conn.commit()

    # -------------------------------------------------------------------------
    # Campaign lifecycle
    # -------------------------------------------------------------------------

    def start_campaign(self, name: str, domain: str, objective: Dict[str, Any]) -> str:
        """Register a new campaign and return its ID."""
        campaign_id = str(uuid.uuid4())[:8]
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO campaigns (id, name, domain, objective, start_time)
            VALUES (?, ?, ?, ?, ?)
        """, (campaign_id, name, domain, json.dumps(objective), time.time()))
        self.conn.commit()
        return campaign_id

    def end_campaign(self, campaign_id: str, summary: Dict[str, Any]):
        """Mark campaign as complete and store summary."""
        c = self.conn.cursor()
        c.execute("""
            UPDATE campaigns SET
                end_time=?, iterations=?, total_generated=?,
                total_screened=?, best_score=?, success_rate=?, summary=?
            WHERE id=?
        """, (
            time.time(),
            summary.get('iterations', 0),
            summary.get('total_generated', 0),
            summary.get('total_screened', 0),
            summary.get('best_score', 0.0),
            summary.get('success_rate', 0.0),
            json.dumps(summary),
            campaign_id
        ))
        # Evidence written during an iteration is not eligible to warm-start
        # another campaign until its source campaign has a terminal summary.
        c.execute(
            "UPDATE transferable_records SET finalized=1 "
            "WHERE schema_version=? AND campaign_ids LIKE ?",
            (SCHEMA_VERSION, f'%"{campaign_id}"%'),
        )
        self.conn.commit()

    # -------------------------------------------------------------------------
    # Candidate storage with hypothesis linkage
    # -------------------------------------------------------------------------

    def store_candidate(self,
                        campaign_id: str,
                        domain: str,
                        formula: str,
                        score: float,
                        passed: bool,
                        properties: Dict[str, float],
                        hypothesis_ids: List[str],
                        principle_ids: List[str],
                        iteration: int,
                        candidate_id: Optional[str] = None,
                        schema_version: str = SCHEMA_VERSION,
                        scientific_validity: str = ScientificValidity.DEMO_ONLY.value):
        """Store a generated candidate with full provenance."""
        if not candidate_id:
            record_id = str(uuid.uuid4())[:8]
        elif candidate_id.startswith(f"{campaign_id}_"):
            record_id = candidate_id
        else:
            record_id = f"{campaign_id}_{candidate_id}"

        c = self.conn.cursor()
        # The existing SQLite table intentionally remains unchanged for legacy
        # DB compatibility.  Provenance metadata is embedded in the JSON
        # payload; legacy rows without this marker are quarantined at read
        # time and are never used as v2 scientific evidence.
        properties_payload = dict(properties or {})
        properties_payload.setdefault("_schema_version", schema_version)
        properties_payload.setdefault("_scientific_validity", scientific_validity)
        c.execute("""
            INSERT OR REPLACE INTO candidates
            (id, campaign_id, domain, formula, score, passed_screening,
             hypothesis_ids, principle_ids, properties, iteration, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record_id, campaign_id, domain, formula, score,
            int(passed), json.dumps(hypothesis_ids), json.dumps(principle_ids),
            json.dumps(properties_payload), iteration, time.time()
        ))
        self.conn.commit()
        return record_id

    def store_failure(self,
                      campaign_id: str,
                      domain: str,
                      formula: str,
                      failure_mode: str,
                      structural_features: Dict[str, Any],
                      properties: Dict[str, float],
                      attributed_cause: str):
        """Record a failure attribution for future learning."""
        fa_id = str(uuid.uuid4())[:8]
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO failure_attributions
            (id, campaign_id, domain, formula, failure_mode,
             structural_features, property_predictions, attributed_cause, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fa_id, campaign_id, domain, formula, failure_mode,
            json.dumps(structural_features), json.dumps(properties),
            attributed_cause, time.time()
        ))
        self.conn.commit()
        return fa_id

    # -------------------------------------------------------------------------
    # Principle management
    # -------------------------------------------------------------------------

    def store_principle(self,
                        domain: str,
                        statement: str,
                        property_target: str,
                        structural_motif: str,
                        campaign_id: str,
                        confidence: float = 0.5,
                        source_type: str = "inferred") -> str:
        """Store or update a distilled scientific principle."""
        # Check if a very similar principle exists for this domain+target
        existing = self._find_similar_principle(domain, property_target, structural_motif)
        now = time.time()

        if existing:
            # Update confidence and add supporting campaign
            p_id, supporting = existing
            supporting_list = json.loads(supporting)
            if campaign_id not in supporting_list:
                supporting_list.append(campaign_id)
            new_conf = min(0.95, confidence + 0.05 * len(supporting_list))
            c = self.conn.cursor()
            c.execute("""
                UPDATE principles SET confidence=?, supporting_campaigns=?, updated_at=?
                WHERE id=?
            """, (new_conf, json.dumps(supporting_list), now, p_id))
            self.conn.commit()
            return p_id
        else:
            p_id = str(uuid.uuid4())[:8]
            c = self.conn.cursor()
            c.execute("""
                INSERT INTO principles
                (id, domain, statement, confidence, supporting_campaigns, refuting_campaigns,
                 property_target, structural_motif, source_type, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                p_id, domain, statement, confidence,
                json.dumps([campaign_id]), json.dumps([]),
                property_target, structural_motif, source_type, now, now
            ))
            self.conn.commit()
            return p_id

    def _find_similar_principle(self, domain: str, property_target: str,
                                 structural_motif: str) -> Optional[Tuple[str, str]]:
        """Find an existing principle with same domain+property+motif."""
        c = self.conn.cursor()
        c.execute("""
            SELECT id, supporting_campaigns FROM principles
            WHERE domain=? AND property_target=? AND structural_motif=?
            LIMIT 1
        """, (domain, property_target, structural_motif))
        row = c.fetchone()
        return (row[0], row[1]) if row else None

    def refute_principle(self, principle_id: str, campaign_id: str):
        """Reduce confidence in a principle based on contradicting evidence."""
        c = self.conn.cursor()
        c.execute("SELECT confidence, refuting_campaigns FROM principles WHERE id=?",
                  (principle_id,))
        row = c.fetchone()
        if not row:
            return
        conf, refuting = row
        refuting_list = json.loads(refuting)
        if campaign_id not in refuting_list:
            refuting_list.append(campaign_id)
        new_conf = max(0.05, conf - 0.1 * len(refuting_list))
        c.execute("""
            UPDATE principles SET confidence=?, refuting_campaigns=?, updated_at=?
            WHERE id=?
        """, (new_conf, json.dumps(refuting_list), time.time(), principle_id))
        self.conn.commit()

    # -------------------------------------------------------------------------
    # Hypothesis tracking
    # -------------------------------------------------------------------------

    def store_hypothesis(self,
                         campaign_id: str,
                         iteration: int,
                         statement: str,
                         basis: str,
                         source_principle_ids: List[str],
                         source_domains: List[str],
                         confidence: float = 0.5) -> str:
        """Record a hypothesis before testing it."""
        h_id = str(uuid.uuid4())[:8]
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO hypotheses
            (id, campaign_id, iteration, statement, basis, source_principle_ids,
             source_domains, confidence_before, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            h_id, campaign_id, iteration, statement, basis,
            json.dumps(source_principle_ids), json.dumps(source_domains),
            confidence, time.time()
        ))
        self.conn.commit()
        return h_id

    def resolve_hypothesis(self, hypothesis_id: str, outcome: str, confidence_after: float):
        """Update a hypothesis with the outcome of testing it."""
        c = self.conn.cursor()
        c.execute("""
            UPDATE hypotheses SET outcome=?, confidence_after=?
            WHERE id=?
        """, (outcome, confidence_after, hypothesis_id))
        self.conn.commit()

    # -------------------------------------------------------------------------
    # Cross-domain transfer
    # -------------------------------------------------------------------------

    def store_cross_domain_link(self,
                                 source_domain: str,
                                 target_domain: str,
                                 source_principle_id: str,
                                 analogy: str,
                                 confidence: float = 0.4) -> str:
        """Record an analogical link between domains."""
        link_id = str(uuid.uuid4())[:8]
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO cross_domain_links
            (id, source_domain, target_domain, source_principle_id,
             analogy_description, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (link_id, source_domain, target_domain, source_principle_id,
              analogy, confidence, time.time()))
        self.conn.commit()
        return link_id

    # -------------------------------------------------------------------------
    # Retrieval for warm-start and planning
    # -------------------------------------------------------------------------

    def get_relevant_principles(self,
                                 domain: str,
                                 property_target: str,
                                 min_confidence: float = 0.4) -> List[Dict[str, Any]]:
        """
        Retrieve high-confidence principles relevant to the current domain+property.
        Used for warm-starting a new campaign iteration.
        """
        c = self.conn.cursor()
        c.execute("""
            SELECT id, statement, confidence, structural_motif, source_type, supporting_campaigns
            FROM principles
            WHERE domain=? AND property_target=? AND confidence >= ?
            ORDER BY confidence DESC
            LIMIT 10
        """, (domain, property_target, min_confidence))

        rows = c.fetchall()
        return [
            {
                'id': r[0], 'statement': r[1], 'confidence': r[2],
                'structural_motif': r[3], 'source_type': r[4],
                'n_campaigns': len(json.loads(r[5]))
            }
            for r in rows
        ]

    def get_cross_domain_insights(self,
                                   target_domain: str,
                                   min_confidence: float = 0.3) -> List[Dict[str, Any]]:
        """
        Retrieve insights from other domains that may transfer to this one.
        The key novel capability: warm-start from analogous domains.
        """
        c = self.conn.cursor()
        c.execute("""
            SELECT cdl.id, cdl.source_domain, cdl.analogy_description, cdl.confidence,
                   p.statement, p.structural_motif, p.property_target
            FROM cross_domain_links cdl
            JOIN principles p ON cdl.source_principle_id = p.id
            WHERE cdl.target_domain=? AND cdl.confidence >= ?
            ORDER BY cdl.confidence DESC
            LIMIT 10
        """, (target_domain, min_confidence))

        rows = c.fetchall()
        return [
            {
                'link_id': r[0], 'source_domain': r[1], 'analogy': r[2],
                'confidence': r[3], 'source_principle': r[4],
                'structural_motif': r[5], 'property_target': r[6]
            }
            for r in rows
        ]

    def get_common_failure_modes(self,
                                  domain: str,
                                  top_n: int = 5) -> List[Dict[str, Any]]:
        """Return the most frequent failure modes in this domain to avoid."""
        c = self.conn.cursor()
        c.execute("""
            SELECT failure_mode, attributed_cause, COUNT(*) as cnt
            FROM failure_attributions
            WHERE domain=?
            GROUP BY failure_mode
            ORDER BY cnt DESC
            LIMIT ?
        """, (domain, top_n))

        rows = c.fetchall()
        return [
            {'failure_mode': r[0], 'cause': r[1], 'occurrences': r[2]}
            for r in rows
        ]

    def get_career_summary(self) -> Dict[str, Any]:
        """High-level summary of the agent's career knowledge."""
        c = self.conn.cursor()

        c.execute("SELECT COUNT(*), COUNT(DISTINCT domain) FROM campaigns")
        n_campaigns, n_domains = c.fetchone()

        c.execute("SELECT COUNT(*) FROM principles WHERE confidence >= 0.6")
        n_strong_principles = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM failure_attributions")
        n_failures = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM cross_domain_links WHERE confidence >= 0.4")
        n_cross_links = c.fetchone()[0]

        c.execute("SELECT DISTINCT domain FROM campaigns")
        domains = [r[0] for r in c.fetchall()]

        c.execute("""
            SELECT domain, AVG(success_rate), COUNT(*)
            FROM campaigns WHERE success_rate IS NOT NULL
            GROUP BY domain
        """)
        domain_stats = {r[0]: {'avg_success_rate': r[1], 'n_campaigns': r[2]}
                        for r in c.fetchall()}

        return {
            'total_campaigns': n_campaigns,
            'domains_explored': domains,
            'n_domains': n_domains,
            'high_confidence_principles': n_strong_principles,
            'failure_attributions': n_failures,
            'cross_domain_links': n_cross_links,
            'domain_stats': domain_stats
        }

    def get_top_candidates_ever(self, domain: Optional[str] = None,
                                 top_n: int = 10) -> List[Dict[str, Any]]:
        """Best candidates across all campaigns (optionally filtered by domain)."""
        c = self.conn.cursor()
        if domain:
            c.execute("""
                SELECT formula, score, domain, campaign_id, iteration, properties
                FROM candidates WHERE domain=?
                ORDER BY score DESC
            """, (domain,))
        else:
            c.execute("""
                SELECT formula, score, domain, campaign_id, iteration, properties
                FROM candidates ORDER BY score DESC
            """)

        rows = c.fetchall()
        results = []
        for r in rows:
            try:
                props = json.loads(r[5])
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(props, dict) or props.get("_schema_version") != SCHEMA_VERSION:
                # v1/unknown rows are retained for audit in the DB but are
                # quarantined from v2 scientific retrieval/statistics.
                continue
            results.append({
                'formula': r[0], 'score': r[1], 'domain': r[2],
                'campaign_id': r[3], 'iteration': r[4],
                'properties': props,
            })
            if len(results) >= top_n:
                break
        return results

    # -------------------------------------------------------------------------
    # Schema-v2 transferable structure memory
    # -------------------------------------------------------------------------

    def store_transferable_record(self, record: TransferableMemoryRecord | Dict[str, Any]) -> str:
        """Persist one structure-aware evidence record with deterministic dedup.

        The evidence fingerprint intentionally excludes campaign/formula IDs.
        Re-recording the same observation therefore preserves the original
        confidence and evidence count, while merging provenance IDs for audit.
        """
        if isinstance(record, dict):
            record = TransferableMemoryRecord.from_dict(record)
        if not isinstance(record, TransferableMemoryRecord):
            raise TypeError("record must be TransferableMemoryRecord or dict")
        if record.schema_version != SCHEMA_VERSION:
            # Legacy records are accepted only for audit storage by callers;
            # this method is a v2 scientific boundary and must fail closed.
            raise ValueError("legacy/unknown transferable records are quarantined")
        # The legacy SQLite column is NOT NULL.  Normalize a missing label to
        # an explicit unknown value so malformed evidence can still be kept
        # for audit without ever becoming directive-eligible.
        if record.outcome_label is None:
            record.outcome_label = "unknown"
        if isinstance(record.features, dict):
            record.features = TransferableFeatures.from_dict(record.features)
        if isinstance(record.applicability, dict):
            record.applicability = ApplicabilityConstraint.from_dict(record.applicability)
        if isinstance(record.directive, dict):
            record.directive = TransferDirective(**{
                k: v for k, v in record.directive.items()
                if k in TransferDirective.__dataclass_fields__
            })
        if not record.evidence_hash:
            record.evidence_hash = evidence_fingerprint(
                record.features, record.outcome_label, record.outcome_value,
                record.source_domain, record.principle_id,
            )
        if not record.evidence_ids:
            record.evidence_ids = [record.record_id]
        record.campaign_ids = sorted(set(record.campaign_ids))
        record.source_candidate_ids = sorted(set(record.source_candidate_ids))
        record.source_formulas = sorted(set(record.source_formulas))
        # A record only becomes warm-start eligible after an end_campaign call.
        finalized = bool(record.finalized and self._campaigns_finalized(record.campaign_ids))
        record.finalized = finalized
        payload = record.to_dict()
        c = self.conn.cursor()
        c.execute(
            "SELECT record_id, campaign_ids, source_candidate_ids, source_formulas, "
            "evidence_count, finalized, payload FROM transferable_records WHERE evidence_hash=?",
            (record.evidence_hash,),
        )
        existing = c.fetchone()
        if existing:
            old_payload = self._safe_json(existing[6], {})
            old = TransferableMemoryRecord.from_dict(old_payload) if old_payload else record
            merged_campaigns = sorted(set(old.campaign_ids) | set(record.campaign_ids))
            merged_candidates = sorted(set(old.source_candidate_ids) | set(record.source_candidate_ids))
            merged_formulas = sorted(set(old.source_formulas) | set(record.source_formulas))
            merged_evidence = sorted(set(old.evidence_ids) | set(record.evidence_ids))
            old.campaign_ids = merged_campaigns
            old.source_candidate_ids = merged_candidates
            old.source_formulas = merged_formulas
            old.evidence_ids = merged_evidence
            old.directive.source_evidence_ids = sorted(
                set(old.directive.source_evidence_ids)
                | set(record.directive.source_evidence_ids)
                | set(merged_evidence)
            )
            old.directive.source_campaign_ids = sorted(
                set(old.directive.source_campaign_ids)
                | set(record.directive.source_campaign_ids)
                | set(merged_campaigns)
            )
            old.finalized = bool(old.finalized or (record.finalized and self._campaigns_finalized(merged_campaigns)))
            old.created_at = old.created_at or record.created_at or time.time()
            # Never increase evidence_count/confidence for duplicate evidence.
            old.applicability.evidence_count = min(old.applicability.evidence_count, record.applicability.evidence_count)
            old.evidence_hash = record.evidence_hash
            c.execute(
                "UPDATE transferable_records SET campaign_ids=?, source_candidate_ids=?, "
                "source_formulas=?, finalized=?, payload=? WHERE evidence_hash=?",
                (json.dumps(old.campaign_ids), json.dumps(old.source_candidate_ids),
                 json.dumps(old.source_formulas), int(old.finalized), json.dumps(old.to_dict(), sort_keys=True),
                 record.evidence_hash),
            )
            self.conn.commit()
            return old.record_id
        record.created_at = record.created_at or time.time()
        c.execute(
            "INSERT INTO transferable_records "
            "(record_id, schema_version, evidence_hash, source_domain, campaign_ids, "
            "source_candidate_ids, source_formulas, outcome_label, outcome_value, "
            "evidence_count, confidence, finalized, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (record.record_id, record.schema_version, record.evidence_hash,
             record.source_domain, json.dumps(record.campaign_ids),
             json.dumps(record.source_candidate_ids), json.dumps(record.source_formulas),
             record.outcome_label, record.outcome_value,
             max(1, int(record.applicability.evidence_count)),
             float(record.applicability.confidence), int(record.finalized),
             json.dumps(record.to_dict(), sort_keys=True), record.created_at),
        )
        self.conn.commit()
        return record.record_id

    # Friendly aliases used by integrations and experiment notebooks.
    store_transferable_memory = store_transferable_record

    def store_transferable_evidence(
        self,
        *,
        campaign_id: str,
        domain: str,
        structure: Any,
        result: Any = None,
        validation: Any = None,
        thermodynamics: Any = None,
        outcome_label: Optional[str] = None,
        outcome_value: Optional[float] = None,
        principle_id: Optional[str] = None,
        source_candidate_id: Optional[str] = None,
        applicability: Optional[ApplicabilityConstraint | Dict[str, Any]] = None,
        directive: Optional[TransferDirective | Dict[str, Any]] = None,
        finalized: bool = False,
    ) -> str:
        """Extract and store one candidate observation as schema-v2 evidence."""
        features = extract_transferable_features(
            structure, result=result, validation=validation, thermodynamics=thermodynamics
        )
        result_map = result if isinstance(result, dict) else getattr(result, "predictions", {}) or {}
        thermodynamic_labels = {"stable", "unstable", "retained", "rejected", "thermodynamically_stable", "thermodynamically_unstable"}
        if outcome_label is None:
            label = features.thermodynamic_label
            if label is None and result is not None:
                label = getattr(result, "thermodynamic_label", None)
            if label is None:
                passed = result_map.get("passes_filters")
                passed = getattr(result, "passes_filters", passed)
                label = "screening_pass" if passed is True else ("screening_reject" if passed is False else "unknown")
            outcome_label = str(label)
        elif str(outcome_label) in thermodynamic_labels and features.thermodynamic_label is None:
            # A caller cannot promote an arbitrary/stale stability label into
            # evidence merely by passing it as an argument.  Preserve a
            # separately observed screening rejection, however, so negative
            # transfer remains queryable rather than being silently erased.
            passed = result_map.get("passes_filters")
            passed = getattr(result, "passes_filters", passed)
            outcome_label = "screening_reject" if passed is False else "unknown"
        if outcome_value is None:
            outcome_value = features.thermodynamic_threshold_ev_per_atom
        if applicability is None:
            applicability = ApplicabilityConstraint(
                allowed_relationship="same_system",
                source_chemical_system=sorted(features.element_classes),
                confidence=0.5 if outcome_label not in {"unknown", "screening_reject"} else 0.25,
                observed_outcome_direction=("positive" if outcome_label in {"stable", "retained", "screening_pass"} else "negative" if outcome_label in {"unstable", "rejected", "screening_reject"} else "unknown"),
            )
        elif isinstance(applicability, dict):
            applicability = ApplicabilityConstraint.from_dict(applicability)
        if directive is None:
            directive = TransferDirective()
        elif isinstance(directive, dict):
            directive = TransferDirective(**{k: v for k, v in directive.items() if k in TransferDirective.__dataclass_fields__})
        formula = self._formula_for_transfer(structure)
        candidate_id = source_candidate_id or self._candidate_id_for_transfer(structure)
        record_id = "tm_" + evidence_fingerprint(features, outcome_label, outcome_value, domain, principle_id)[:16]
        record = TransferableMemoryRecord(
            record_id=record_id,
            evidence_ids=[record_id],
            principle_id=principle_id,
            campaign_ids=[campaign_id],
            source_candidate_ids=[candidate_id] if candidate_id else [],
            source_formulas=[formula] if formula else [],
            source_domain=domain,
            outcome_label=outcome_label,
            outcome_value=outcome_value,
            features=features,
            applicability=applicability,
            directive=directive,
            finalized=finalized,
        )
        return self.store_transferable_record(record)

    def get_transferable_records(
        self,
        *,
        source_domain: Optional[str] = None,
        finalized_only: bool = True,
        include_legacy: bool = False,
    ) -> List[TransferableMemoryRecord]:
        """Load typed v2 records; legacy/active records are quarantined by default."""
        sql = "SELECT payload, finalized FROM transferable_records WHERE 1=1"
        args: List[Any] = []
        if source_domain:
            sql += " AND source_domain=?"
            args.append(source_domain)
        if finalized_only:
            sql += " AND finalized=1"
        if not include_legacy:
            sql += " AND schema_version=?"
            args.append(SCHEMA_VERSION)
        sql += " ORDER BY evidence_hash ASC"
        rows = self.conn.cursor().execute(sql, args).fetchall()
        result: List[TransferableMemoryRecord] = []
        for payload, finalized in rows:
            try:
                record = TransferableMemoryRecord.from_dict(json.loads(payload))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not include_legacy and record.schema_version != SCHEMA_VERSION:
                continue
            # The normalized column is the lifecycle source of truth.  The
            # embedded payload predates finalization updates and may still
            # carry ``false`` after end_campaign has marked the row eligible.
            record.finalized = bool(finalized)
            result.append(record)
        return result

    def get_applicable_transferable_memories(
        self,
        *,
        target_features: TransferableFeatures | Dict[str, Any],
        target_domain: Optional[str] = None,
        target_elements: Optional[List[str]] = None,
        target_campaign_id: Optional[str] = None,
        source_domain: Optional[str] = None,
        include_rejected: bool = False,
        target_start_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Return applied/rejected records and reasons for a reproducible audit."""
        if isinstance(target_features, dict):
            if "element_classes" in target_features or "anonymous_stoichiometric_pattern" in target_features:
                target_features = TransferableFeatures.from_dict(target_features)
            else:
                target_features = extract_transferable_features(target_features)
        applied, rejected = [], []
        if target_start_time is None and target_campaign_id:
            row = self.conn.cursor().execute(
                "SELECT start_time FROM campaigns WHERE id=?", (target_campaign_id,)
            ).fetchone()
            target_start_time = row[0] if row else None
        # Include active rows for the rejection audit; chronology/lifecycle
        # checks below ensure they never become actionable directives.
        for record in self.get_transferable_records(source_domain=source_domain, finalized_only=False):
            if target_campaign_id and target_campaign_id in record.campaign_ids:
                rejected.append({
                    "record_id": record.record_id,
                    "reasons": ["CURRENT_CAMPAIGN_EXCLUDED"],
                    "record": record.to_dict(),
                })
                continue
            chronology_reason = self._prior_campaign_reason(
                record.campaign_ids, target_campaign_id, target_start_time
            )
            if chronology_reason is not None:
                rejected.append({
                    "record_id": record.record_id,
                    "reasons": [chronology_reason],
                    "record": record.to_dict(),
                })
                continue
            outcome_reason = _directive_outcome_rejection_reason(record.outcome_label)
            if outcome_reason == "NEGATIVE_OUTCOME_NOT_A_DIRECTIVE":
                # Preserve the existing negative-evidence audit contract.
                # Unknown/malformed records are handled by the directive
                # wrapper below, where the executable-policy boundary lives.
                ok, reasons = False, [outcome_reason]
            else:
                ok, reasons = applicability_check(
                    record, target_features, target_domain=target_domain, target_elements=target_elements
                )
            item = {"record_id": record.record_id, "reasons": reasons, "record": record.to_dict()}
            (applied if ok else rejected).append(item)
        return {"applied": applied, "rejected": rejected, "record_count": len(applied) + len(rejected)}

    def get_transferable_directives(self, **kwargs: Any) -> Dict[str, Any]:
        """Convenience wrapper yielding planner directives plus audit reasons."""
        selection = self.get_applicable_transferable_memories(**kwargs)
        directives = []
        unsupported = []
        eligible_items = []
        for item in selection["applied"]:
            record = TransferableMemoryRecord.from_dict(item["record"])
            outcome_reason = _directive_outcome_rejection_reason(record.outcome_label)
            if outcome_reason is not None:
                # ``get_applicable_transferable_memories`` remains useful for
                # querying/auditing unknown observations, but this wrapper is
                # the executable planner boundary and is strictly positive-
                # allow-listed.  Reclassify the item here so callers cannot
                # mistake it for a directive-ready record.
                rejected_item = dict(item)
                rejected_item["reasons"] = [outcome_reason]
                selection["rejected"].append(rejected_item)
                continue
            eligible_items.append(item)
            directive = make_directive(record)
            directives.append(directive)
            if directive.get("unsupported_policy_effects"):
                unsupported.append({
                    "record_id": record.record_id,
                    "reasons": [f"UNSUPPORTED_EFFECT:{name}" for name in directive["unsupported_policy_effects"]],
                })
        selection["applied"] = eligible_items
        selection["directives"] = directives
        selection["unsupported"] = unsupported
        return selection

    get_applicable_memories = get_applicable_transferable_memories
    get_directives = get_transferable_directives

    def memory_view(
        self,
        mode: str = "structured_provenance",
        *,
        seed: int = 0,
        target_features: Optional[TransferableFeatures] = None,
        target_domain: Optional[str] = None,
        target_elements: Optional[List[str]] = None,
        target_campaign_id: Optional[str] = None,
        target_start_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Read an experimental memory view, including deterministic controls."""
        records = self.get_transferable_records()
        chronology_rejected: List[Dict[str, Any]] = []
        if target_campaign_id or target_start_time is not None:
            prior = []
            for record in records:
                reason = self._prior_campaign_reason(record.campaign_ids, target_campaign_id, target_start_time)
                if reason is None:
                    prior.append(record)
                else:
                    chronology_rejected.append({"record_id": record.record_id, "reasons": [reason]})
            records = prior
        if target_features is not None:
            selected = self.get_applicable_transferable_memories(
                target_features=target_features, target_domain=target_domain, target_elements=target_elements,
                target_campaign_id=target_campaign_id, target_start_time=target_start_time,
            )
            records = [TransferableMemoryRecord.from_dict(item["record"]) for item in selected["applied"]]
        view = view_records(records, mode, seed=seed)
        view["chronology_rejected"] = chronology_rejected
        return view

    # Additional names make the boundary easy to discover without exposing SQL.
    get_memory_view = memory_view

    def _campaigns_finalized(self, campaign_ids: List[str]) -> bool:
        if not campaign_ids:
            return False
        placeholders = ",".join("?" for _ in campaign_ids)
        rows = self.conn.cursor().execute(
            f"SELECT id, end_time FROM campaigns WHERE id IN ({placeholders})", campaign_ids
        ).fetchall()
        return len(rows) == len(set(campaign_ids)) and all(row[1] is not None for row in rows)

    def _prior_campaign_reason(
        self,
        source_campaign_ids: List[str],
        target_campaign_id: Optional[str],
        target_start_time: Optional[float],
    ) -> Optional[str]:
        """Explain why source evidence is not strictly prior to target start."""
        if target_campaign_id and target_campaign_id in source_campaign_ids:
            return "CURRENT_CAMPAIGN_EXCLUDED"
        if not target_campaign_id and target_start_time is None:
            # Backwards-compatible query outside a live campaign: finalized
            # records are already lifecycle-gated, and no chronology anchor
            # was supplied by the caller to compare against.
            return None
        if target_start_time is None:
            return "TARGET_CAMPAIGN_TIME_UNAVAILABLE"
        if not source_campaign_ids:
            return "SOURCE_CAMPAIGN_TIME_UNAVAILABLE"
        placeholders = ",".join("?" for _ in source_campaign_ids)
        rows = self.conn.cursor().execute(
            f"SELECT id, start_time, end_time FROM campaigns WHERE id IN ({placeholders})",
            source_campaign_ids,
        ).fetchall()
        if len(rows) != len(set(source_campaign_ids)):
            return "SOURCE_CAMPAIGN_TIME_UNAVAILABLE"
        for _cid, start_time, end_time in rows:
            if start_time is None or end_time is None:
                return "SOURCE_CAMPAIGN_NOT_FINALIZED"
            if float(start_time) >= float(target_start_time):
                return "SOURCE_CAMPAIGN_NOT_PRIOR"
            if float(end_time) >= float(target_start_time):
                return "SOURCE_FINALIZED_AFTER_TARGET_START"
        return None

    @staticmethod
    def _safe_json(payload: Any, default: Any) -> Any:
        try:
            return json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            return default

    @staticmethod
    def _formula_for_transfer(structure: Any) -> Optional[str]:
        if isinstance(structure, dict):
            return str(structure.get("composition") or structure.get("formula")) if (structure.get("composition") or structure.get("formula")) else None
        comp = getattr(structure, "composition", None)
        if comp is None:
            return None
        return str(getattr(comp, "reduced_formula", comp))

    @staticmethod
    def _candidate_id_for_transfer(structure: Any) -> Optional[str]:
        if isinstance(structure, dict):
            value = structure.get("candidate_id") or structure.get("generation_id")
        else:
            value = getattr(structure, "_candidate_id", None)
            if value is None and isinstance(getattr(structure, "properties", None), dict):
                value = structure.properties.get("_candidate_id")
        return str(value) if value is not None else None
