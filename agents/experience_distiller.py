"""
ExperienceDistiller — turns raw screening results into reusable knowledge.

After each campaign iteration this module:
  1. Extracts principles from successful candidates (LLM or heuristic).
  2. Records failure attributions via failure_attribution.py.
  3. Writes cross-domain analogy links for known domain pairs.
  4. Stores every candidate with full hypothesis/principle provenance.

This is what makes the agent smarter over time — not just logging results,
but distilling them into actionable priors stored in CareerMemory.
"""

import json
from typing import Dict, List, Any, Optional, Tuple

from agents.career_memory import CareerMemory
from agents.failure_attribution import record_failures


KNOWN_DOMAIN_ANALOGIES = {
    ("li_solid_electrolyte", "na_solid_electrolyte"): (
        "Structural motifs enabling fast Li+ transport (e.g., argyrodite, NASICON) "
        "have direct Na+ analogs due to similar ionic radius trends"
    ),
    ("li_solid_electrolyte", "mg_solid_electrolyte"): (
        "High ionic conductivity frameworks for monovalent Li+ may adapt to divalent "
        "Mg2+ with modified vacancy concentrations and lattice softness"
    ),
    ("thermoelectric", "li_solid_electrolyte"): (
        "Low phonon mean free path (high anharmonicity) in thermoelectrics correlates "
        "with lattice softness, which also lowers migration barriers in solid electrolytes"
    ),
    ("thermoelectric", "na_solid_electrolyte"): (
        "Same anharmonicity-conductivity analogy as thermoelectric-Li SSE applies to Na SSE"
    ),
    ("battery_cathode", "li_solid_electrolyte"): (
        "Layered oxide frameworks studied as cathodes share structural features with "
        "lithium-conducting oxides; stability criteria overlap"
    ),
}


class ExperienceDistiller:
    """
    Distills campaign iteration results into reusable scientific principles
    stored in CareerMemory.
    """

    def __init__(self, career_memory: CareerMemory, llm_client=None):
        self.memory = career_memory
        self.llm = llm_client

    def distill_iteration(self,
                          campaign_id: str,
                          domain: str,
                          iteration: int,
                          candidates: List[Any],
                          screening_results: List[Any],
                          strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point. Call after each campaign iteration.

        Args:
            campaign_id: Current campaign ID
            domain: Domain string (e.g., "li_solid_electrolyte")
            iteration: Current iteration number
            candidates: Generated structures
            screening_results: List of (structure, ScreeningResult) tuples
            strategy: Planning strategy that was used this iteration

        Returns:
            Distillation summary with principles written and hypotheses resolved
        """
        if not screening_results:
            return {'principles_written': 0, 'failures_recorded': 0}

        # Separate passing and failing candidates
        passing = [(s, r) for s, r in screening_results if r.passes_filters]
        failing = [(s, r) for s, r in screening_results if not r.passes_filters]

        principles_written = []
        failures_recorded = []
        cross_links = []

        if self.memory is None:
            return {'principles_written': 0, 'failures_recorded': 0,
                    'cross_domain_links': 0, 'top_principles': []}

        # 1. Extract principles from passing candidates
        if passing:
            principles_written = self._extract_success_principles(
                domain, campaign_id, iteration, passing, strategy
            )

        # 2. Record failure attributions
        if failing:
            failures_recorded = record_failures(
                failing, campaign_id, domain, self.memory
            )

        # 3. Write cross-domain analogy links
        cross_links = self._generate_cross_domain_links(
            domain, campaign_id, principles_written
        )

        # 4. Store all candidates with full provenance
        current_hypotheses = self._get_current_hypotheses(campaign_id, iteration)
        self._store_candidates_with_provenance(
            campaign_id, domain, iteration, screening_results,
            principles_written, current_hypotheses
        )

        return {
            'principles_written': len(principles_written),
            'failures_recorded': len(failures_recorded),
            'cross_domain_links': len(cross_links),
            'top_principles': [p['statement'] for p in principles_written[:3]]
        }

    def distill_campaign_end(self,
                              campaign_id: str,
                              domain: str,
                              all_results: List[Dict[str, Any]],
                              objective: Dict[str, Any]) -> str:
        """
        End-of-campaign distillation. Extracts higher-level principles
        that span multiple iterations.

        Returns a human-readable career summary paragraph.
        """
        if not all_results:
            return "No results to distill."

        # Aggregate stats
        total_generated = sum(r.get('num_generated', 0) for r in all_results)
        total_screened = sum(r.get('num_screened', 0) for r in all_results)
        best_scores = [r.get('insights', {}).get('avg_stability', 0) for r in all_results]
        improving = len(best_scores) > 2 and best_scores[-1] > best_scores[0]

        # Build summary for LLM
        if self.llm:
            summary = self._llm_campaign_summary(
                domain, objective, all_results, improving
            )
        else:
            summary = self._heuristic_campaign_summary(
                domain, total_generated, total_screened, improving, all_results
            )

        return summary

    # -------------------------------------------------------------------------
    # Internal: principle extraction
    # -------------------------------------------------------------------------

    def _extract_success_principles(self,
                                     domain: str,
                                     campaign_id: str,
                                     iteration: int,
                                     passing: List[Tuple],
                                     strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract principles from passing candidates."""
        principles = []
        elements = strategy.get('constraints', {}).get('elements', [])
        target_props = strategy.get('target_properties', {})

        if self.llm:
            extracted = self._llm_extract_principles(
                domain, passing, elements, target_props
            )
        else:
            extracted = self._heuristic_extract_principles(
                domain, passing, elements, target_props
            )

        for p in extracted:
            p_id = self.memory.store_principle(
                domain=domain,
                statement=p['statement'],
                property_target=p.get('property_target', 'stability'),
                structural_motif=p.get('structural_motif', 'unknown'),
                campaign_id=campaign_id,
                confidence=p.get('confidence', 0.5),
                source_type='inferred'
            )
            p['id'] = p_id
            principles.append(p)

        return principles

    def _heuristic_extract_principles(self,
                                       domain: str,
                                       passing: List[Tuple],
                                       elements: List[str],
                                       target_props: Dict) -> List[Dict]:
        """Rule-based principle extraction when no LLM is available."""
        principles = []

        # Analyze element patterns in passing candidates
        elem_counts: Dict[str, int] = {}
        for struct, result in passing:
            formula = getattr(struct, 'composition', None)
            if formula:
                for el in elements:
                    if el in str(formula):
                        elem_counts[el] = elem_counts.get(el, 0) + 1

        # Elements that appear in >60% of passing candidates
        threshold = len(passing) * 0.6
        key_elements = [el for el, cnt in elem_counts.items() if cnt >= threshold]

        if key_elements:
            elem_str = '+'.join(key_elements)
            principles.append({
                'statement': (
                    f"In {domain}, combinations including {elem_str} consistently "
                    f"pass stability screening — prioritize these in generation"
                ),
                'property_target': 'stability',
                'structural_motif': elem_str,
                'confidence': min(0.7, 0.4 + 0.05 * len(passing))
            })

        # Score-based principle
        scores = [r.score for _, r in passing]
        if scores:
            avg_score = sum(scores) / len(scores)
            principles.append({
                'statement': (
                    f"Iteration {domain} screening achieved avg score {avg_score:.2f} "
                    f"with elements {elements} — continue prioritizing this space"
                ),
                'property_target': list(target_props.keys())[0] if target_props else 'stability',
                'structural_motif': '+'.join(elements[:3]) if elements else 'mixed',
                'confidence': 0.45
            })

        return principles

    def _llm_extract_principles(self,
                                 domain: str,
                                 passing: List[Tuple],
                                 elements: List[str],
                                 target_props: Dict) -> List[Dict]:
        """Use LLM to extract rich structural principles."""
        formulas = []
        scores = []
        for struct, result in passing[:10]:
            formulas.append(self._formula_from_struct(struct))
            scores.append(round(result.score, 3))

        prompt = f"""You are a materials scientist analyzing successful candidates from a discovery campaign.

Domain: {domain}
Target properties: {json.dumps(target_props)}
Candidate elements used: {elements}

Top passing candidates (formula: score):
{chr(10).join(f'  {f}: {s}' for f, s in zip(formulas, scores))}

Extract 1-3 generalizable scientific principles from these successes.
For each principle, identify:
- The structural motif or chemical pattern responsible
- Which target property it relates to
- Your confidence (0-1) that this will generalize

Respond as JSON array:
[
  {{
    "statement": "...",
    "structural_motif": "...",
    "property_target": "...",
    "confidence": 0.6
  }}
]"""

        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a materials science expert. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=512
            )
            text = response.choices[0].message.content.strip()
            start = text.find('[')
            end = text.rfind(']') + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
        except Exception as e:
            print(f"  [Distiller] LLM extraction failed: {e}, using heuristic")

        return self._heuristic_extract_principles(domain, passing, elements, target_props)

    # -------------------------------------------------------------------------
    # Cross-domain links
    # -------------------------------------------------------------------------

    def _generate_cross_domain_links(self,
                                      domain: str,
                                      campaign_id: str,
                                      principles: List[Dict]) -> List[str]:
        """Generate cross-domain analogy links for new principles."""
        link_ids = []

        for p in principles:
            p_id = p.get('id')
            if not p_id:
                continue
            for (src, tgt), analogy in KNOWN_DOMAIN_ANALOGIES.items():
                if src == domain:
                    link_id = self.memory.store_cross_domain_link(
                        source_domain=src,
                        target_domain=tgt,
                        source_principle_id=p_id,
                        analogy=analogy,
                        confidence=0.4
                    )
                    link_ids.append(link_id)

        return link_ids

    # -------------------------------------------------------------------------
    # Internal: candidate provenance storage
    # -------------------------------------------------------------------------

    def _get_current_hypotheses(self, campaign_id: str,
                                 iteration: int) -> List[str]:
        """Retrieve hypothesis IDs active this iteration."""
        c = self.memory.conn.cursor()
        c.execute(
            "SELECT id FROM hypotheses WHERE campaign_id=? AND iteration=?",
            (campaign_id, iteration),
        )
        return [r[0] for r in c.fetchall()]

    @staticmethod
    def _formula_from_struct(struct: Any) -> str:
        """Extract a reduced formula string from a pymatgen Structure or stub dict."""
        comp = getattr(struct, 'composition', None)
        if hasattr(comp, 'reduced_formula'):
            return comp.reduced_formula
        if isinstance(struct, dict):
            return struct.get('composition', 'Unknown')
        return str(comp or 'Unknown')

    def _store_candidates_with_provenance(
        self,
        campaign_id: str,
        domain: str,
        iteration: int,
        screening_results: List[Tuple],
        principles: List[Dict],
        hypothesis_ids: List[str],
    ):
        """Store all screened candidates with full provenance."""
        principle_ids = [p['id'] for p in principles if 'id' in p]
        for struct, result in screening_results:
            self.memory.store_candidate(
                campaign_id=campaign_id,
                domain=domain,
                formula=self._formula_from_struct(struct),
                score=result.score,
                passed=result.passes_filters,
                properties=result.predictions,
                hypothesis_ids=hypothesis_ids,
                principle_ids=principle_ids,
                iteration=iteration,
            )

    # -------------------------------------------------------------------------
    # Campaign-level summaries
    # -------------------------------------------------------------------------

    def _heuristic_campaign_summary(self,
                                     domain: str,
                                     total_generated: int,
                                     total_screened: int,
                                     improving: bool,
                                     all_results: List[Dict]) -> str:
        n_iter = len(all_results)
        trend = "showed improvement over iterations" if improving else "was stable across iterations"
        return (
            f"Campaign in domain '{domain}': {n_iter} iterations, "
            f"{total_generated} structures generated, {total_screened} passed screening. "
            f"Performance {trend}."
        )

    def _llm_campaign_summary(self,
                               domain: str,
                               objective: Dict,
                               all_results: List[Dict],
                               improving: bool) -> str:
        """Generate a rich LLM campaign summary for career memory."""
        stats = {
            'iterations': len(all_results),
            'total_generated': sum(r.get('num_generated', 0) for r in all_results),
            'improving': improving,
            'final_success_rate': all_results[-1].get('insights', {}).get('success_rate', 0)
        }

        prompt = f"""Summarize this materials discovery campaign for long-term memory storage.
Domain: {domain}
Objective: {json.dumps(objective)}
Stats: {json.dumps(stats)}

Write 2-3 sentences capturing: what worked, what failed, and what a future agent
should know before starting a similar campaign. Be specific about chemistry and
structural features. Do not use markdown.
"""
        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a materials science expert writing concise research notes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=256
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return self._heuristic_campaign_summary(
                domain, stats['total_generated'], 0, improving, all_results
            )
