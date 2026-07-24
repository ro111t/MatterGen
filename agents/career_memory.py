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

Persists to ~/.matagent_career.db (SQLite). Schema defined in db_schema.py.
"""

import sqlite3
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from agents.db_schema import ALL_SCHEMAS


class CareerMemory:
    """
    Persistent knowledge store that accumulates scientific experience across
    all campaigns and domains. Survives process restarts via SQLite.
    """

    def __init__(self, db_path: str = "~/.matagent_career.db"):
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
                        iteration: int):
        """Store a generated candidate with full provenance."""
        candidate_id = str(uuid.uuid4())[:8]
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO candidates
            (id, campaign_id, domain, formula, score, passed_screening,
             hypothesis_ids, principle_ids, properties, iteration, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            candidate_id, campaign_id, domain, formula, score,
            int(passed), json.dumps(hypothesis_ids), json.dumps(principle_ids),
            json.dumps(properties), iteration, time.time()
        ))
        self.conn.commit()
        return candidate_id

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
                ORDER BY score DESC LIMIT ?
            """, (domain, top_n))
        else:
            c.execute("""
                SELECT formula, score, domain, campaign_id, iteration, properties
                FROM candidates ORDER BY score DESC LIMIT ?
            """, (top_n,))

        rows = c.fetchall()
        return [
            {
                'formula': r[0], 'score': r[1], 'domain': r[2],
                'campaign_id': r[3], 'iteration': r[4],
                'properties': json.loads(r[5])
            }
            for r in rows
        ]
