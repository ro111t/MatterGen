"""
StrategyAgent: Meta-learning / adaptive search optimization for the discovery loop.

This lightweight implementation maintains a history of which element sets and
hyperparameters produced good outcomes, then uses a simple UCB-style bandit to
recommend the next element set, batch size, and diversity weight.

Future upgrades can replace the bandit with Bayesian optimization over a
structured search space.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math
import warnings

import numpy as np


@dataclass
class StrategyOutcome:
    """One data point: which strategy was tried and how it performed."""
    iteration: int
    elements: Tuple[str, ...]
    num_candidates: int
    diversity_weight: float
    num_passed: int
    num_screened: int
    num_validated: int
    num_converged: int
    num_synthesis_feasible: int
    best_score: float
    best_validated_stability: float
    best_synthesis_feasibility: float
    validation_cost_hours: float


class StrategyAgent:
    """
    Recommends the next search strategy based on historical campaign outcomes.

    Args:
        exploration_weight: UCB exploration coefficient. Higher values favor
            trying less-tested element sets. 0.0 means pure exploitation.
        target_metric: Which metric to optimize. Options:
            'best_score', 'best_validated_stability', 'best_synthesis_feasibility',
            'combined'.
    """

    def __init__(
        self,
        exploration_weight: float = 0.2,
        target_metric: str = "combined",
    ):
        self.exploration_weight = exploration_weight
        self.target_metric = target_metric
        self.outcomes: List[StrategyOutcome] = []
        self.element_rewards: Dict[Tuple[str, ...], List[float]] = defaultdict(list)
        self.total_trials = 0

    def update(
        self,
        iteration: int,
        strategy: Dict[str, Any],
        insights: Dict[str, Any],
    ) -> None:
        """Record the outcome of an iteration for future recommendations."""
        elements = tuple(sorted(strategy.get('elements', [])))
        outcome = StrategyOutcome(
            iteration=iteration,
            elements=elements,
            num_candidates=int(strategy.get('num_candidates', 15)),
            diversity_weight=float(strategy.get('diversity_weight', 0.3)),
            num_passed=int(insights.get('num_passed', 0)),
            num_screened=int(insights.get('num_screened', 0)),
            num_validated=int(insights.get('num_validated', 0)),
            num_converged=int(insights.get('num_converged', 0)),
            num_synthesis_feasible=int(insights.get('num_synthesis_feasible', 0)),
            best_score=float(insights.get('best_score', 0.0)),
            best_validated_stability=float(insights.get('best_validated_stability', 0.0)),
            best_synthesis_feasibility=float(insights.get('best_synthesis_feasibility', 0.0)),
            validation_cost_hours=float(insights.get('validation_cost_hours', 0.0)),
        )
        self.outcomes.append(outcome)
        reward = self._compute_reward(outcome)
        self.element_rewards[elements].append(reward)
        self.total_trials += 1

    def recommend(
        self,
        objective: Any,
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Recommend the next strategy parameters.

        Returns a dict with keys:
            - elements: list of element symbols
            - num_candidates: suggested batch size
            - diversity_weight: suggested diversity weight
            - screening_criteria: optional criteria adjustments
            - rationale: explanation of recommendation
        """
        allowed_elements = objective.constraints.get('elements', [])

        # 1. Choose element set via UCB over observed element sets.
        recommended_elements = self._recommend_elements(allowed_elements)

        # 2. Adapt batch size based on recent yield.
        num_candidates = self._recommend_batch_size()

        # 3. Adapt diversity weight based on whether recent iterations improved.
        diversity_weight = self._recommend_diversity()

        # 4. Optionally tighten screening once validation starts producing data.
        screening_criteria = self._recommend_screening_criteria()

        return {
            'elements': list(recommended_elements),
            'num_candidates': num_candidates,
            'diversity_weight': diversity_weight,
            'screening_criteria': screening_criteria,
            'rationale': (
                f"UCB selected {'-'.join(recommended_elements)} "
                f"(trials={len(self.element_rewards.get(recommended_elements, []))}, "
                f"exploration={self.exploration_weight}). Batch={num_candidates}, "
                f"diversity={diversity_weight:.2f}."
            ),
        }

    def _compute_reward(self, outcome: StrategyOutcome) -> float:
        """Combine multiple objectives into a single reward signal."""
        if self.target_metric == 'best_score':
            return outcome.best_score / 100.0  # score is 0-100
        if self.target_metric == 'best_validated_stability':
            return max(0.0, -outcome.best_validated_stability)
        if self.target_metric == 'best_synthesis_feasibility':
            return outcome.best_synthesis_feasibility

        # combined: normalize and weight each signal
        score_component = outcome.best_score / 100.0
        stability_component = max(0.0, -outcome.best_validated_stability) / 5.0
        synthesis_component = outcome.best_synthesis_feasibility
        return 0.3 * score_component + 0.4 * stability_component + 0.3 * synthesis_component

    def _recommend_elements(self, allowed_elements: List[str]) -> Tuple[str, ...]:
        """Use UCB to pick the most promising element set."""
        # If we have no data yet, use the allowed elements directly.
        if not self.element_rewards:
            return tuple(sorted(set(allowed_elements)))

        # Build candidate element sets: observed ones plus one fallback of all allowed elements.
        candidates = set(self.element_rewards.keys())
        if allowed_elements:
            candidates.add(tuple(sorted(set(allowed_elements))))

        best_ucb = float('-inf')
        best_elements: Optional[Tuple[str, ...]] = None

        for elements in candidates:
            rewards = self.element_rewards[elements]
            n = len(rewards)
            mean_reward = sum(rewards) / n if n > 0 else 0.0
            # UCB bonus; large bonus for unseen or rarely seen arms
            if self.total_trials > 0 and n > 0:
                bonus = self.exploration_weight * math.sqrt(2.0 * math.log(self.total_trials) / n)
            else:
                bonus = 1.0  # strong prior for untried arms

            ucb = mean_reward + bonus
            if ucb > best_ucb:
                best_ucb = ucb
                best_elements = elements

        # Ensure recommendation only contains allowed elements if a constraint is given.
        if allowed_elements and best_elements:
            filtered = [e for e in best_elements if e in allowed_elements]
            if filtered:
                return tuple(filtered)

        return best_elements or tuple(sorted(set(allowed_elements)))

    def _recommend_batch_size(self) -> int:
        """Increase batch size when yield is low; shrink when yield is high."""
        if len(self.outcomes) < 2:
            return 15

        recent = self.outcomes[-3:]
        pass_rates = [
            o.num_passed / max(o.num_screened, 1)
            for o in recent
        ]
        avg_rate = sum(pass_rates) / len(pass_rates)
        last_batch = self.outcomes[-1].num_candidates

        if avg_rate < 0.05:
            return min(100, int(last_batch * 1.3))
        if avg_rate > 0.3:
            return max(5, int(last_batch * 0.85))
        return last_batch

    def _recommend_diversity(self) -> float:
        """Explore more when improvement stalls; exploit when improving."""
        if len(self.outcomes) < 2:
            return 0.4

        recent_rewards = [self._compute_reward(o) for o in self.outcomes[-3:]]
        if len(recent_rewards) >= 2:
            slope = recent_rewards[-1] - recent_rewards[0]
        else:
            slope = 0.0

        current_diversity = self.outcomes[-1].diversity_weight
        if slope > 0.05:
            # Improving: reduce diversity to exploit the current region.
            return max(0.1, current_diversity - 0.05)
        if slope < -0.05 or slope == 0.0:
            # Stagnant or worsening: increase diversity to explore.
            return min(0.8, current_diversity + 0.05)
        return current_diversity

    def _recommend_screening_criteria(self) -> Optional[Dict[str, float]]:
        """
        Optional tightening of screening once we have validation data.
        For now, return None; this hook is available for future Bayesian opt.
        """
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Return a human-readable summary of strategy history."""
        return {
            'total_trials': self.total_trials,
            'element_sets_tested': len(self.element_rewards),
            'best_element_set': self._recommend_elements([]) if self.element_rewards else None,
            'recent_rewards': [
                {
                    'iteration': o.iteration,
                    'elements': list(o.elements),
                    'reward': round(self._compute_reward(o), 3),
                }
                for o in self.outcomes[-5:]
            ],
        }
