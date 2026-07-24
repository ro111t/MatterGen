from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json
import os

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from agents.career_memory import CareerMemory


@dataclass
class CampaignObjective:
    """Defines the goal of a materials discovery campaign"""
    target_properties: Dict[str, float]
    constraints: Dict[str, Any]
    success_criteria: Dict[str, float]
    domain: str = "materials"
    max_iterations: int = 20
    compute_budget_hours: float = 100.0


class OrchestratorAgent:
    """
    High-level strategic agent that coordinates the materials discovery campaign.
    Uses LLM for reasoning, and CareerMemory for warm-starting from past experience.
    """

    def __init__(self, career_memory: Optional[CareerMemory] = None,
                 api_key: Optional[str] = None):
        self.career_memory = career_memory
        self.current_strategy = None
        self.current_hypothesis_ids: List[str] = []

        # Initialize OpenAI client
        key = api_key or os.environ.get('OPENAI_API_KEY')
        if HAS_OPENAI and key:
            self.llm = OpenAI(api_key=key)
            self.llm_available = True
        else:
            self.llm = None
            self.llm_available = False
            print("  [Orchestrator] No OpenAI key — using heuristic planning")
        
    def plan_campaign(self, objective: CampaignObjective) -> Dict[str, Any]:
        """
        Create initial search strategy, warm-started from CareerMemory if available.
        """
        warm_start = self._get_career_warm_start(objective)
        prompt = self._build_planning_prompt(objective, warm_start)
        strategy = self._call_llm_for_strategy(prompt)
        self.current_strategy = strategy
        return strategy
        
    def plan_iteration(self, objective: CampaignObjective, history: List[Dict],
                        campaign_id: str = "", iteration: int = 0,
                        recommendations: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Plan next iteration using campaign history + CareerMemory principles.
        Optional `recommendations` from the StrategyAgent override base choices.
        """
        recent_results = history[-5:] if len(history) > 5 else history

        # Pull relevant career principles for this domain
        career_context = ""
        hypothesis_basis = ""
        source_principle_ids = []
        if self.career_memory:
            principles = self.career_memory.get_relevant_principles(
                domain=objective.domain,
                property_target=list(objective.target_properties.keys())[0]
                    if objective.target_properties else 'stability'
            )
            cross_domain = self.career_memory.get_cross_domain_insights(
                target_domain=objective.domain
            )
            failures = self.career_memory.get_common_failure_modes(
                domain=objective.domain
            )

            if principles:
                career_context += "\nKNOWN PRINCIPLES FROM CAREER MEMORY:\n"
                for p in principles[:4]:
                    career_context += f"  [{p['confidence']:.2f}] {p['statement']}\n"
                    source_principle_ids.append(p['id'])

            if cross_domain:
                career_context += "\nCROSS-DOMAIN INSIGHTS:\n"
                for cd in cross_domain[:3]:
                    career_context += f"  (from {cd['source_domain']}) {cd['analogy']}\n"

            if failures:
                career_context += "\nKNOWN FAILURE MODES TO AVOID:\n"
                for f in failures[:3]:
                    career_context += f"  {f['failure_mode']}: {f['cause']}\n"

            hypothesis_basis = career_context

        prompt = f"""You are a materials science AI planning iteration {iteration} of a discovery campaign.

OBJECTIVE:
{json.dumps({'domain': objective.domain, 'target_properties': objective.target_properties,
             'constraints': objective.constraints}, indent=2)}

RECENT CAMPAIGN RESULTS:
{self._summarize_results(recent_results)}
{career_context}

Based on this, design the generation strategy for the next batch.
Respond ONLY with JSON with these keys:
- elements: list of element symbols to focus on (3-5 elements)
- num_candidates: integer (suggest 10-30 for local testing)
- screening_criteria: dict with max_formation_energy, max_forces, min_stability
- diversity_weight: float 0-1
- rationale: one sentence explanation
- hypothesis: one testable scientific hypothesis for this iteration

Example:
{{"elements": ["Li", "P", "S", "Cl"], "num_candidates": 20,
  "screening_criteria": {{"max_formation_energy": 2.0, "max_forces": 1.0}},
  "diversity_weight": 0.4,
  "rationale": "Argyrodite Li-P-S-Cl space has strong literature support.",
  "hypothesis": "Cl substitution in Li6PS5X improves stability scores above 15."}}"""

        strategy = self._call_llm_for_strategy(prompt)
        strategy = self._validate_strategy(strategy, objective, recommendations=recommendations)

        # Record hypothesis in career memory
        if self.career_memory and campaign_id and strategy.get('hypothesis'):
            h_id = self.career_memory.store_hypothesis(
                campaign_id=campaign_id,
                iteration=iteration,
                statement=strategy['hypothesis'],
                basis=hypothesis_basis or "campaign history",
                source_principle_ids=source_principle_ids,
                source_domains=[objective.domain],
                confidence=0.5
            )
            self.current_hypothesis_ids = [h_id]
        else:
            self.current_hypothesis_ids = []

        self.current_strategy = strategy
        return strategy
        
    def interpret_results(self, batch_results: Dict[str, Any]) -> str:
        """
        Generate natural language interpretation of batch results.
        """
        if not self.llm_available:
            n_gen = batch_results.get('num_generated', 0)
            n_scr = batch_results.get('num_screened', 0)
            rate = batch_results.get('screening_rate', 0)
            return (
                f"Generated {n_gen} candidates, {n_scr} passed screening "
                f"({rate:.1%} pass rate). "
                f"Avg stability: {batch_results.get('avg_stability', 0):.3f} eV/atom."
            )

        prompt = f"""Analyze these materials discovery results briefly (2-3 sentences):
- Generated: {batch_results.get('num_generated', 0)} candidates
- Passed screening: {batch_results.get('num_screened', 0)} ({batch_results.get('screening_rate', 0):.1%})
- Avg stability: {batch_results.get('avg_stability', 0):.3f} eV/atom
- Successful: {batch_results.get('num_successful', 0)}

Focus on: what worked, what failed, one concrete recommendation."""

        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a materials science expert. Be concise."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"LLM interpretation failed ({e}). {batch_results.get('num_screened', 0)} candidates passed screening."
        
    def adjust_strategy(self, performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Modify search strategy based on performance metrics."""
        if not self.current_strategy:
            return self._default_strategy()

        if performance_metrics.get('success_rate', 0) < 0.05:
            adjustment = {
                'diversity_weight': min(self.current_strategy.get('diversity_weight', 0.3) + 0.1, 0.8),
                'num_candidates': int(self.current_strategy.get('num_candidates', 20) * 1.3),
            }
        elif performance_metrics.get('success_rate', 0) > 0.3:
            adjustment = {
                'diversity_weight': max(self.current_strategy.get('diversity_weight', 0.3) - 0.1, 0.1),
                'num_candidates': int(self.current_strategy.get('num_candidates', 20) * 0.8),
            }
        else:
            adjustment = {}

        self.current_strategy.update(adjustment)
        return self.current_strategy
        
    def _get_career_warm_start(self, objective: CampaignObjective) -> str:
        """Query CareerMemory for relevant past experience."""
        if not self.career_memory:
            return ""

        summary = self.career_memory.get_career_summary()
        if summary['total_campaigns'] == 0:
            return ""

        context = f"\nCAREER MEMORY ({summary['total_campaigns']} prior campaigns):\n"

        principles = self.career_memory.get_relevant_principles(
            domain=objective.domain,
            property_target=list(objective.target_properties.keys())[0]
                if objective.target_properties else 'stability'
        )
        if principles:
            context += "Known principles:\n"
            for p in principles[:3]:
                context += f"  - {p['statement']} (confidence={p['confidence']:.2f})\n"

        cross = self.career_memory.get_cross_domain_insights(objective.domain)
        if cross:
            context += "Cross-domain insights:\n"
            for cd in cross[:2]:
                context += f"  - From {cd['source_domain']}: {cd['analogy']}\n"

        top = self.career_memory.get_top_candidates_ever(domain=objective.domain, top_n=3)
        if top:
            context += "Best prior candidates in this domain:\n"
            for c in top:
                context += f"  - {c['formula']} (score={c['score']:.2f})\n"

        return context

    def _build_planning_prompt(self, objective: CampaignObjective,
                                warm_start: str = "") -> str:
        """Build LLM prompt for initial planning with career warm-start."""
        return f"""Design a materials discovery strategy for this objective.

DOMAIN: {objective.domain}
TARGET PROPERTIES: {json.dumps(objective.target_properties)}
CONSTRAINTS: {json.dumps(objective.constraints)}
{warm_start}
Respond ONLY with JSON with keys:
- elements (list), num_candidates (int), screening_criteria (dict),
  diversity_weight (float), rationale (str), hypothesis (str)"""
        
    def _call_llm_for_strategy(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI and parse the JSON strategy response."""
        if not self.llm_available:
            return self._default_strategy()

        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a materials science expert. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=512
            )
            text = response.choices[0].message.content.strip()
            return self._parse_json(text)
        except Exception as e:
            print(f"  [Orchestrator] LLM call failed: {e}")
            return self._default_strategy()

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response."""
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        return self._default_strategy()
            
    def _default_strategy(self) -> Dict[str, Any]:
        """Default strategy if LLM is unavailable.

        Screening criteria are kept lenient because generated structures are
        unrelaxed; CHGNet will report large forces for random coordinates.
        Tighten these after a relaxation/validation step is added.
        """
        return {
            'elements': ['Li', 'P', 'S', 'O'],
            'num_candidates': 15,
            'screening_criteria': {
                'max_formation_energy': 5.0,
                'max_forces': 500.0,
                'min_stability': -20.0,
            },
            'diversity_weight': 0.4,
            'rationale': 'Default Li-P-S-O space for solid electrolytes.',
            'hypothesis': 'Li-P-S compositions with low formation energy will pass screening.'
        }
        
    def _validate_strategy(self, strategy: Dict[str, Any],
                             objective: CampaignObjective,
                             recommendations: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Ensure strategy is within safe bounds; merge StrategyAgent recommendations first."""
        if recommendations:
            if recommendations.get('elements'):
                strategy['elements'] = recommendations['elements']
            if recommendations.get('num_candidates') is not None:
                strategy['num_candidates'] = int(recommendations['num_candidates'])
            if recommendations.get('diversity_weight') is not None:
                strategy['diversity_weight'] = float(recommendations['diversity_weight'])
            if recommendations.get('screening_criteria'):
                strategy['screening_criteria'] = recommendations['screening_criteria']

        strategy['num_candidates'] = min(max(int(strategy.get('num_candidates', 15)), 5), 100)
        strategy['diversity_weight'] = min(max(float(strategy.get('diversity_weight', 0.3)), 0.0), 1.0)
        if 'elements' not in strategy or not strategy['elements']:
            strategy['elements'] = objective.constraints.get('elements', ['Li', 'P', 'S', 'O'])

        # Generated structures are unrelaxed, so CHGNet will report large forces.
        # Enforce lenient screening defaults so candidates reach validation.
        criteria = strategy.get('screening_criteria', {})
        criteria['max_formation_energy'] = max(float(criteria.get('max_formation_energy', 5.0)), 5.0)
        criteria['max_forces'] = max(float(criteria.get('max_forces', 500.0)), 100.0)
        criteria['min_stability'] = min(float(criteria.get('min_stability', -20.0)), -10.0)
        strategy['screening_criteria'] = criteria

        return strategy
        
    def _summarize_results(self, results: List[Dict]) -> str:
        """Concise summary of recent results for LLM context."""
        if not results:
            return "No previous results."

        lines = []
        for r in results:
            insights = r.get('insights', {})
            lines.append(
                f"  Iter {r.get('iteration', '?')}: "
                f"{r.get('num_generated', 0)} generated, "
                f"{r.get('num_screened', 0)} passed screening, "
                f"best_score={insights.get('best_score', 0):.2f}, "
                f"success_rate={insights.get('success_rate', 0):.1%}"
            )
        return '\n'.join(lines)
