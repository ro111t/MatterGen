"""
StrategyAgent: Meta-learning / adaptive search optimization for the discovery loop.

This lightweight implementation maintains a history of which element sets and
hyperparameters produced good outcomes, then uses a simple UCB-style bandit to
recommend the next element set, batch size, and diversity weight.

The agent may also receive structured transferable directives from an upstream
memory layer. Those directives are used to derive concrete target compositions
for the generator, while the diversity weight controls the exploration-
exploitation mix between memory-derived and randomly sampled compositions.
"""

import random
import re
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
    best_synthesis_feasibility: float
    validation_cost_hours: float


class StrategyAgent:
    """
    Recommends the next search strategy based on historical campaign outcomes.

    Args:
        exploration_weight: UCB exploration coefficient. Higher values favor
            trying less-tested element sets. 0.0 means pure exploitation.
        target_metric: Retained for configuration compatibility; Sprint 1
            optimizes only the non-thermodynamic ``best_score``.
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
        history: List[Dict],
        directives: Optional[List[Dict[str, Any]]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Recommend the next strategy parameters.

        Returns a dict with keys:
            - elements: list of element symbols
            - num_candidates: suggested batch size
            - diversity_weight: suggested diversity weight
            - target_compositions_dict: list of target composition dicts
            - screening_criteria: optional criteria adjustments
            - rationale: explanation of recommendation
        """
        allowed_elements = objective.constraints.get('elements', [])

        # 1. Choose element set via UCB over observed element sets.
        recommended_elements = self._recommend_elements(allowed_elements)

        # 2. Adapt batch size based on recent yield.
        num_candidates = self._recommend_batch_size()

        # 3. Adapt diversity weight based on in-campaign history and memory directives.
        diversity_weight = self._recommend_diversity(directives=directives)

        # 4. Derive target compositions from memory directives if available.
        target_compositions_dict = self._build_target_compositions_dict(
            directives=directives,
            target_elements=list(recommended_elements),
            num_candidates=num_candidates,
            diversity_weight=diversity_weight,
            seed=seed,
        )

        # 5. Optionally tighten screening once validation starts producing data.
        screening_criteria = self._recommend_screening_criteria()

        if target_compositions_dict:
            num_memory_guided = min(num_candidates, max(0, round((1.0 - diversity_weight) * num_candidates)))
            num_exploratory = num_candidates - num_memory_guided
        else:
            num_memory_guided = 0
            num_exploratory = num_candidates

        return {
            'elements': list(recommended_elements),
            'num_candidates': num_candidates,
            'diversity_weight': diversity_weight,
            'target_compositions_dict': target_compositions_dict,
            'num_memory_guided_proposals': num_memory_guided,
            'num_exploratory_proposals': num_exploratory,
            'screening_criteria': screening_criteria,
            'rationale': (
                f"UCB selected {'-'.join(recommended_elements)} "
                f"(trials={len(self.element_rewards.get(recommended_elements, []))}, "
                f"exploration={self.exploration_weight}). Batch={num_candidates}, "
                f"diversity={diversity_weight:.2f}, "
                f"memory_targets={len(target_compositions_dict)}."
            ),
        }

    def _compute_reward(self, outcome: StrategyOutcome) -> float:
        """Combine multiple objectives into a single reward signal."""
        if self.target_metric == 'best_score':
            return outcome.best_score / 100.0  # score is 0-100
        # Sprint 1 intentionally rewards only the non-thermodynamic screening
        # score.  Raw energies and thermodynamic labels cannot be compared
        # across compositions until a reference-set/hull result exists.
        return outcome.best_score / 100.0

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
            if self.outcomes:
                return self.outcomes[-1].num_candidates
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
            return max(1, int(last_batch * 0.85))
        return last_batch

    def _recommend_diversity(
        self,
        directives: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """Explore more when improvement stalls; exploit when improving."""
        if len(self.outcomes) < 2:
            base = 0.4
        else:
            recent_rewards = [self._compute_reward(o) for o in self.outcomes[-3:]]
            if len(recent_rewards) >= 2:
                slope = recent_rewards[-1] - recent_rewards[0]
            else:
                slope = 0.0

            current_diversity = self.outcomes[-1].diversity_weight
            if slope > 0.05:
                # Improving: reduce diversity to exploit the current region.
                base = max(0.1, current_diversity - 0.05)
            elif slope < -0.05 or slope == 0.0:
                # Stagnant or worsening: increase diversity to explore.
                base = min(0.8, current_diversity + 0.05)
            else:
                base = current_diversity

        # Blend with the cross-campaign memory exploration signal when available.
        if directives:
            usable = [
                d for d in directives
                if d.get("exploration_weight") is not None
            ]
            if usable:
                mean_exploration = sum(float(d.get("exploration_weight", 0.0)) for d in usable) / len(usable)
                base = 0.5 * base + 0.5 * mean_exploration
                base = min(0.8, max(0.1, base))

        return base

    def _build_target_compositions_dict(
        self,
        directives: Optional[List[Dict[str, Any]]],
        target_elements: List[str],
        num_candidates: int,
        diversity_weight: float,
        seed: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """Build num_candidates target compositions mixing memory and exploration."""
        if not directives or not target_elements or num_candidates <= 0:
            return []

        memory_targets = self._extract_memory_target_compositions(directives, target_elements)
        if not memory_targets:
            return []

        # diversity_weight is the probability of random exploration.
        # 1 - diversity_weight is exploitation of memory-derived targets.
        n_memory = min(num_candidates, max(0, round((1.0 - diversity_weight) * num_candidates)))
        n_random = num_candidates - n_memory

        rng = random.Random(seed if seed is not None else 0)
        memory_part = [memory_targets[i % len(memory_targets)] for i in range(n_memory)]
        random_part = self._sample_random_target_compositions(target_elements, n_random, rng)
        return memory_part + random_part

    def _extract_memory_target_compositions(
        self,
        directives: List[Dict[str, Any]],
        target_elements: List[str],
    ) -> List[Dict[str, float]]:
        """Map anonymous stoichiometric patterns from directives to target elements."""
        targets: List[Dict[str, float]] = []
        for d in directives:
            patterns = d.get("preferred_anonymous_stoichiometries") or []
            source_order = d.get("source_element_order") or []
            source_classes = d.get("source_element_classes") or {}
            substitutions = d.get("permitted_element_class_substitutions") or {}
            if not patterns or not source_order or not source_classes or not substitutions:
                continue
            for pattern in patterns:
                composition = self._map_anonymous_pattern_to_target(
                    pattern, source_order, source_classes, target_elements, substitutions
                )
                if composition and any(v > 0 for v in composition.values()):
                    targets.append(composition)

        # Deduplicate while preserving order.
        seen: set = set()
        unique: List[Dict[str, float]] = []
        for t in targets:
            key = _canonical_composition_key(t)
            if key not in seen:
                seen.add(key)
                unique.append(t)
        return unique

    def _map_anonymous_pattern_to_target(
        self,
        pattern: str,
        source_order: List[str],
        source_classes: Dict[str, str],
        target_elements: List[str],
        substitutions: Dict[str, List[str]],
    ) -> Optional[Dict[str, float]]:
        """Map an anonymous source pattern to a concrete target composition."""
        pairs = _parse_anonymous_pattern(pattern)
        if not pairs or len(pairs) > len(source_order):
            return None

        target_set = set(target_elements)
        target_amounts: Dict[str, float] = {}
        for i, (label, count) in enumerate(pairs):
            source_el = source_order[i]
            source_class = source_classes.get(source_el)
            if not source_class:
                return None

            allowed = list(substitutions.get(source_class, []))
            # The original source element is also an implicit allowed target if present.
            candidates = [e for e in allowed if e in target_set]
            if source_el in target_set:
                candidates.append(source_el)
            if not candidates:
                return None

            # Deterministic tie-break: sorted order.
            chosen = sorted(candidates)[0]
            target_amounts[chosen] = target_amounts.get(chosen, 0.0) + count

        return _normalize_composition(target_amounts)

    def _sample_random_target_compositions(
        self,
        target_elements: List[str],
        n: int,
        rng: random.Random,
    ) -> List[Dict[str, float]]:
        """Sample n random exploratory target compositions from the target element set."""
        if n <= 0:
            return []
        compositions: List[Dict[str, float]] = []
        for _ in range(n):
            n_types = rng.randint(2, min(4, len(target_elements)))
            chosen = sorted(rng.sample(target_elements, n_types))
            amounts = {el: float(rng.choice([1, 2, 3, 4])) for el in chosen}
            compositions.append(_normalize_composition(amounts))
        return compositions

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


def _parse_anonymous_pattern(pattern: str) -> List[Tuple[str, float]]:
    """Parse an anonymous stoichiometric pattern such as A2B3C4 into labels and counts."""
    if not pattern:
        return []
    matches = re.findall(r"([A-Z])([0-9]+(?:\.[0-9]+)?)", pattern)
    if not matches:
        return []
    result = []
    for label, amount in matches:
        try:
            result.append((label, float(amount)))
        except ValueError:
            return []
    return result


def _canonical_composition_key(composition: Dict[str, float]) -> str:
    """Deterministic string key for deduplicating composition dicts."""
    items = sorted((str(k), float(v)) for k, v in composition.items())
    return "-".join(f"{k}:{v:g}" for k, v in items)


def _normalize_composition(amounts: Dict[str, float]) -> Dict[str, float]:
    """Return a cleaned composition dict with only positive, sorted amounts."""
    return {k: float(v) for k, v in sorted(amounts.items()) if v > 0}
