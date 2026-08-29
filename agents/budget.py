"""Exact proposal/oracle budget accounting for a campaign.

The tracker intentionally has no notion of scientific quality.  It records
resource-consuming events only: every generated candidate is a proposal, and
every geometrically valid candidate admitted to prediction consumes one oracle
slot before cache lookup.  This makes duplicate, cached, and failed oracle
requests auditable instead of silently free.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


def _validate_limit(name: str, value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer when supplied")
    return value


@dataclass
class BudgetSnapshot:
    proposals_generated: int = 0
    geometry_valid: int = 0
    invalid_geometry: int = 0
    oracle_evaluations: int = 0
    oracle_cache_hits: int = 0
    proposal_budget: Optional[int] = None
    oracle_budget: Optional[int] = None
    proposal_budget_remaining: Optional[int] = None
    oracle_budget_remaining: Optional[int] = None
    termination_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DualBudgetTracker:
    """Stateful, fail-closed tracker for proposal and oracle budgets."""

    def __init__(self, proposal_budget: Optional[int] = None,
                 oracle_budget: Optional[int] = None):
        self.proposal_budget = _validate_limit("proposal_budget", proposal_budget)
        self.oracle_budget = _validate_limit("oracle_budget", oracle_budget)
        self.proposals_generated = 0
        self.geometry_valid = 0
        self.invalid_geometry = 0
        self.oracle_evaluations = 0
        self.oracle_cache_hits = 0
        self.termination_reason: Optional[str] = None

    @property
    def proposal_budget_remaining(self) -> Optional[int]:
        if self.proposal_budget is None:
            return None
        return max(0, self.proposal_budget - self.proposals_generated)

    @property
    def oracle_budget_remaining(self) -> Optional[int]:
        if self.oracle_budget is None:
            return None
        return max(0, self.oracle_budget - self.oracle_evaluations)

    @property
    def proposals_remaining(self) -> Optional[int]:
        return self.proposal_budget_remaining

    @property
    def oracle_remaining(self) -> Optional[int]:
        return self.oracle_budget_remaining

    def generation_capacity(self, requested: int) -> int:
        """Return a request truncated to the exact remaining proposal capacity."""
        requested = max(0, int(requested))
        remaining = self.proposal_budget_remaining
        return requested if remaining is None else min(requested, remaining)

    def record_proposals(self, count: int) -> None:
        """Debit proposals for every candidate returned by generation."""
        count = int(count)
        if count < 0:
            raise ValueError("proposal count cannot be negative")
        if self.proposal_budget is not None and self.proposals_generated + count > self.proposal_budget:
            raise ValueError("proposal budget exceeded")
        self.proposals_generated += count

    # Aliases make the event semantics easy to discover from callers/tests.
    consume_proposals = record_proposals
    record_generated = record_proposals

    def record_geometry(self, valid: bool) -> None:
        if valid:
            self.geometry_valid += 1
        else:
            self.invalid_geometry += 1

    def admit_oracle(self, *, cache_hit: bool = False) -> bool:
        """Debit one oracle request before cache lookup; return admission status."""
        if self.oracle_budget is not None and self.oracle_evaluations >= self.oracle_budget:
            return False
        self.oracle_evaluations += 1
        if cache_hit:
            self.oracle_cache_hits += 1
        return True

    consume_oracle = admit_oracle
    record_oracle = admit_oracle

    def set_termination(self, reason: Optional[str]) -> None:
        self.termination_reason = reason

    def snapshot(self, *, termination_reason: Optional[str] = None) -> BudgetSnapshot:
        return BudgetSnapshot(
            proposals_generated=self.proposals_generated,
            geometry_valid=self.geometry_valid,
            invalid_geometry=self.invalid_geometry,
            oracle_evaluations=self.oracle_evaluations,
            oracle_cache_hits=self.oracle_cache_hits,
            proposal_budget=self.proposal_budget,
            oracle_budget=self.oracle_budget,
            proposal_budget_remaining=self.proposal_budget_remaining,
            oracle_budget_remaining=self.oracle_budget_remaining,
            termination_reason=(termination_reason if termination_reason is not None else self.termination_reason),
        )

    def to_dict(self, *, termination_reason: Optional[str] = None) -> Dict[str, Any]:
        return self.snapshot(termination_reason=termination_reason).to_dict()


# Explicit aliases for integrations that use the longer name.
BudgetTracker = DualBudgetTracker
CampaignBudgetTracker = DualBudgetTracker


__all__ = [
    "BudgetSnapshot",
    "DualBudgetTracker",
    "BudgetTracker",
    "CampaignBudgetTracker",
]
