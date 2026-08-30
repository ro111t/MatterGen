"""Seed-paired, fail-closed statistical analysis for the benchmark.

The primary endpoint is time (oracle calls) to the first candidate at or below
0.10 eV/atom. Runs that do not reach the endpoint are right-censored at their
fixed oracle budget and are retained in Kaplan--Meier/RMST summaries. Missing
arms and failed seeds remain explicit rows in the analysis manifest.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from experiments.spec import (
    calculate_minimum_exact_test_sample_size,
    CONFIRMATORY_CONTROLS,
    CONFIRMATORY_METRICS,
    DEFAULT_FAMILY_WISE_ALPHA,
)

CONFIRMATORY_FAMILY_NAME = "primary_endpoint_and_threshold_yield_family"
PRIMARY_CENSORED_ENDPOINT_NAME = "oracle_calls_to_first_candidate_at_or_below_0_10"
PRIMARY_CENSORED_METHOD_NAME = "paired_censored_RMST_within_seed_randomization"
THRESHOLD_YIELD_ENDPOINT_NAME = "fraction_at_or_below_0_10"
THRESHOLD_YIELD_METHOD_NAME = "paired_sign_flip"
HOLM_ADJUSTMENT_METHOD_NAME = "Holm-Bonferroni"


def _atomic_write_file(path: Path, content: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{random.randint(1000, 9999)}")
    if isinstance(content, str):
        temp.write_text(content, encoding="utf-8")
    else:
        temp.write_bytes(content)
    os.replace(temp, path)


@dataclass
class KaplanMeierEstimate:
    """Nonparametric right-censored survival estimate."""

    n: int
    events: int
    censorings: int
    restricted_mean_survival_time: float
    curve: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PairedComparisonResult:
    analysis_version: str
    target_task: str
    metric_name: str
    condition_a: str
    condition_b: str
    comparison_type: str
    sample_size_n: int
    missing_count: int
    mean_a: float
    mean_b: float
    mean_difference: float
    median_difference: float
    ci_95_lower: float
    ci_95_upper: float
    p_value_raw: Optional[float]
    p_value_adjusted: Optional[float]
    adjustment_method: Optional[str]
    seed_level_differences: List[Dict[str, Any]] = field(default_factory=list)
    method: str = "paired_sign_flip"
    confirmatory_family: Optional[str] = None
    status: str = "ANALYZED"
    kaplan_meier_a: Optional[Dict[str, Any]] = None
    kaplan_meier_b: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def bootstrap_paired_ci(
    differences: Sequence[float], n_resamples: int = 10000,
    confidence_level: float = 0.95, random_seed: int = 42,
) -> Tuple[float, float, float]:
    if not differences:
        return 0.0, 0.0, 0.0
    rng = random.Random(random_seed)
    n = len(differences)
    mean_diff = float(sum(differences) / n)
    if n == 1 or all(d == differences[0] for d in differences):
        return mean_diff, mean_diff, mean_diff
    samples = sorted(
        sum(rng.choice(differences) for _ in range(n)) / n
        for _ in range(max(1, n_resamples))
    )
    alpha = (1.0 - confidence_level) / 2.0
    lo = max(0, min(int(alpha * len(samples)), len(samples) - 1))
    hi = max(0, min(int((1.0 - alpha) * len(samples)) - 1, len(samples) - 1))
    return mean_diff, samples[lo], samples[hi]


def paired_permutation_test_pvalue(
    differences: Sequence[float], n_permutations: int = 10000, random_seed: int = 42,
) -> float:
    if not differences:
        return 1.0
    n = len(differences)
    observed = abs(sum(differences) / n)
    if observed == 0.0:
        return 1.0
    if n <= 16:
        total = 1 << n
        extreme = 0
        for mask in range(total):
            value = sum(d * (1 if mask & (1 << i) else -1) for i, d in enumerate(differences)) / n
            extreme += abs(value) >= observed - 1e-12
        return extreme / total
    rng = random.Random(random_seed)
    extreme = 0
    for _ in range(max(1, n_permutations)):
        value = sum(d * (1 if rng.random() < 0.5 else -1) for d in differences) / n
        extreme += abs(value) >= observed - 1e-12
    return max(1.0 / max(1, n_permutations), extreme / max(1, n_permutations))


def holm_bonferroni_adjust(p_values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * len(indexed)
    running = 0.0
    for rank, (idx, value) in enumerate(indexed):
        running = max(running, min(1.0, float(value) * (len(indexed) - rank)))
        adjusted[idx] = running
    return adjusted


def kaplan_meier(
    observations: Sequence[Tuple[float, bool]], budget: Optional[float] = None,
) -> KaplanMeierEstimate:
    """Estimate survival to threshold, where ``event=True`` means reached."""
    clean: List[Tuple[float, bool]] = []
    for time, event in observations:
        try:
            t = float(time)
        except (TypeError, ValueError):
            continue
        # Follow-up times are physical oracle-call counts.  Zero/negative and
        # non-finite values are invalid evidence, never values to clip into a
        # valid censoring time.
        if not math.isfinite(t) or t <= 0.0:
            continue
        clean.append((t, bool(event)))
    if budget is None:
        budget = max((t for t, _ in clean), default=0.0)
    try:
        budget = float(budget)
    except (TypeError, ValueError):
        budget = 0.0
    if not math.isfinite(budget) or budget <= 0.0:
        clean = []
        budget = 0.0
    else:
        # Do not turn over-budget values into budget censorings: that would
        # manufacture a valid observation from malformed evidence.
        clean = [(t, event) for t, event in clean if t <= budget]
    n = len(clean)
    if not n:
        return KaplanMeierEstimate(0, 0, 0, budget, [{"time": 0.0, "survival": 1.0}])

    # Events and censorings at the same time are handled as a tied risk set.
    by_time: Dict[float, Dict[str, int]] = {}
    for t, event in clean:
        item = by_time.setdefault(t, {"events": 0, "censored": 0})
        item["events" if event else "censored"] += 1
    at_risk = n
    survival = 1.0
    previous = 0.0
    rmst = 0.0
    curve: List[Dict[str, float]] = [{"time": 0.0, "survival": 1.0, "at_risk": float(n), "events": 0.0}]
    total_events = 0
    total_censorings = 0
    for t in sorted(by_time):
        rmst += max(0.0, t - previous) * survival
        counts = by_time[t]
        events = counts["events"]
        censored = counts["censored"]
        if at_risk > 0 and events:
            survival *= max(0.0, 1.0 - events / at_risk)
        total_events += events
        total_censorings += censored
        curve.append({"time": float(t), "survival": float(survival), "at_risk": float(at_risk), "events": float(events)})
        at_risk -= events + censored
        previous = t
    rmst += max(0.0, budget - previous) * survival
    curve.append({"time": budget, "survival": float(survival), "at_risk": float(max(0, at_risk)), "events": 0.0})
    return KaplanMeierEstimate(n, total_events, total_censorings, rmst, curve)


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    return float(ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0)


def compute_paired_comparison(
    task_id: str, metric_name: str, data_a: Mapping[int, float], data_b: Mapping[int, float],
    condition_a: str, condition_b: str, comparison_type: str = "confirmatory",
    analysis_version: str = "1.0.0", random_seed: int = 42,
    expected_seeds: Optional[Sequence[int]] = None,
) -> PairedComparisonResult:
    seeds = sorted(set(expected_seeds if expected_seeds is not None else set(data_a) | set(data_b)))
    complete = [s for s in seeds if s in data_a and s in data_b and _finite(data_a[s]) and _finite(data_b[s])]
    missing = len(seeds) - len(complete)
    details: List[Dict[str, Any]] = []
    for s in seeds:
        if s not in data_a and s not in data_b:
            details.append({"seed": s, "status": "missing_a_and_b"})
        elif s not in data_a:
            details.append({"seed": s, "status": "missing_a", "val_b": data_b.get(s)})
        elif s not in data_b:
            details.append({"seed": s, "status": "missing_b", "val_a": data_a.get(s)})
        elif not (_finite(data_a[s]) and _finite(data_b[s])):
            details.append({"seed": s, "status": "invalid_value", "val_a": data_a[s], "val_b": data_b[s]})
        else:
            details.append({"seed": s, "status": "paired", "val_a": data_a[s], "val_b": data_b[s], "diff": data_a[s] - data_b[s]})
    if not complete:
        return PairedComparisonResult(
            analysis_version, task_id, metric_name, condition_a, condition_b, comparison_type,
            0, missing, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None, None, details,
            status="UNAVAILABLE_MISSING_PAIRS",
        )
    vals_a = [float(data_a[s]) for s in complete]
    vals_b = [float(data_b[s]) for s in complete]
    diffs = [a - b for a, b in zip(vals_a, vals_b)]
    mean_diff, low, high = bootstrap_paired_ci(diffs, random_seed=random_seed)
    return PairedComparisonResult(
        analysis_version, task_id, metric_name, condition_a, condition_b, comparison_type,
        len(complete), missing, sum(vals_a) / len(vals_a), sum(vals_b) / len(vals_b), mean_diff,
        _median(diffs), low, high, paired_permutation_test_pvalue(diffs, random_seed=random_seed),
        None, None, details,
    )


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _valid_censored_time(value: Any, budget: Any) -> bool:
    """Whether a follow-up time is admissible before converting it to float.

    Censored times are observations, not placeholders that can be clipped to a
    budget.  In particular, zero, negative, non-finite, and over-budget values
    must become missing evidence rather than silently changing the risk set.
    """
    if not _finite(value) or not _finite(budget):
        return False
    try:
        budget_value = float(budget)
        time_value = float(value)
    except (TypeError, ValueError):
        return False
    return budget_value > 0.0 and 0.0 < time_value <= budget_value


def compute_paired_censored_comparison(
    task_id: str,
    metric_name: str,
    data_a: Mapping[int, Tuple[float, bool]],
    data_b: Mapping[int, Tuple[float, bool]],
    condition_a: str,
    condition_b: str,
    budget: float,
    comparison_type: str = "confirmatory",
    analysis_version: str = "1.0.0",
    expected_seeds: Optional[Sequence[int]] = None,
    random_seed: int = 42,
) -> PairedComparisonResult:
    """Paired RMST/KM comparison retaining censoring and missingness."""
    # A censored follow-up is still a measured time, so silently clipping an
    # invalid value to the budget would manufacture an observation.  Validate
    # both arms before constructing pairs and retain invalid seeds as explicit
    # missingness in the result manifest.
    try:
        budget_value = float(budget)
    except (TypeError, ValueError):
        budget_value = float("nan")
    seeds = sorted(set(expected_seeds if expected_seeds is not None else set(data_a) | set(data_b)))

    def _valid_observation(value: Any) -> bool:
        return _valid_censored_time(value, budget_value)

    valid_a = {s: (float(data_a[s][0]), bool(data_a[s][1])) for s in seeds if s in data_a and isinstance(data_a[s], (tuple, list)) and len(data_a[s]) >= 2 and _valid_observation(data_a[s][0])}
    valid_b = {s: (float(data_b[s][0]), bool(data_b[s][1])) for s in seeds if s in data_b and isinstance(data_b[s], (tuple, list)) and len(data_b[s]) >= 2 and _valid_observation(data_b[s][0])}
    complete = [s for s in seeds if s in valid_a and s in valid_b]
    missing = len(seeds) - len(complete)
    km_budget = budget_value if _finite(budget_value) and budget_value >= 0.0 else 0.0
    obs_a = [valid_a[s] for s in complete]
    obs_b = [valid_b[s] for s in complete]
    km_a = kaplan_meier(obs_a, km_budget)
    km_b = kaplan_meier(obs_b, km_budget)
    details: List[Dict[str, Any]] = []
    pairs: List[Tuple[Tuple[float, bool], Tuple[float, bool]]] = []
    for s in seeds:
        if s not in data_a or s not in data_b:
            details.append({"seed": s, "status": "missing_a" if s not in data_a else "missing_b"})
            continue
        if s not in valid_a or s not in valid_b:
            details.append({
                "seed": s,
                "status": "invalid_censored_time",
                "time_a": data_a[s][0] if isinstance(data_a[s], (tuple, list)) and data_a[s] else None,
                "time_b": data_b[s][0] if isinstance(data_b[s], (tuple, list)) and data_b[s] else None,
                "budget": budget_value,
            })
            continue
        ta, ea = valid_a[s]
        tb, eb = valid_b[s]
        pairs.append(((ta, ea), (tb, eb)))
        details.append({"seed": s, "status": "paired", "time_a": ta, "event_a": ea, "time_b": tb, "event_b": eb})
    if not complete or not pairs:
        status = "UNAVAILABLE_MISSING_PAIRS"
        n = 0
        mean_diff = low = high = median = 0.0
        pvalue = None
    else:
        status = "ANALYZED"
        n = len(pairs)
        mean_diff = km_a.restricted_mean_survival_time - km_b.restricted_mean_survival_time
        # Resample complete seed-pairs and recompute both KM curves. This keeps
        # event indicators attached to their observed/censored follow-up and
        # yields a paired RMST interval rather than treating censor times as
        # event times.
        rng = random.Random(random_seed)
        bootstrap = []
        for _ in range(10000):
            sample = [rng.choice(pairs) for _ in range(n)]
            bootstrap.append(
                kaplan_meier([pair[0] for pair in sample], km_budget).restricted_mean_survival_time
                - kaplan_meier([pair[1] for pair in sample], km_budget).restricted_mean_survival_time
            )
        bootstrap.sort()
        low = bootstrap[max(0, int(0.025 * len(bootstrap)))]
        high = bootstrap[min(len(bootstrap) - 1, int(0.975 * len(bootstrap)))]
        median = bootstrap[len(bootstrap) // 2]

        # Exact (small n) or Monte Carlo within-pair randomization test. Under
        # the paired randomized null, swapping complete (time,event) outcomes
        # within any seed is exchangeable and preserves censoring.
        observed = abs(mean_diff)
        permutations = 1 << n
        if n <= 16:
            masks = range(permutations)
        else:
            masks = [rng.getrandbits(n) for _ in range(10000)]
        extreme = total = 0
        for mask in masks:
            arm_a = []
            arm_b = []
            for index, pair in enumerate(pairs):
                swapped = bool(mask & (1 << index))
                arm_a.append(pair[1] if swapped else pair[0])
                arm_b.append(pair[0] if swapped else pair[1])
            statistic = abs(
                kaplan_meier(arm_a, km_budget).restricted_mean_survival_time
                - kaplan_meier(arm_b, km_budget).restricted_mean_survival_time
            )
            extreme += statistic >= observed - 1e-12
            total += 1
        pvalue = extreme / total if total else None
    return PairedComparisonResult(
        analysis_version, task_id, metric_name, condition_a, condition_b, comparison_type,
        n, missing, km_a.restricted_mean_survival_time, km_b.restricted_mean_survival_time,
        km_a.restricted_mean_survival_time - km_b.restricted_mean_survival_time,
        median, low, high, pvalue, None, None, details,
        method="paired_censored_RMST_within_seed_randomization",
        status=status, kaplan_meier_a=km_a.to_dict(), kaplan_meier_b=km_b.to_dict(),
    )


def _get(obj: Any, name: str, default: Any = None) -> Any:
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def run_statistical_analysis_pipeline(
    run_metrics_list: Sequence[Any],
    analysis_version: str = "1.0.0",
    output_dir: Optional[Path] = None,
    expected_seeds: Optional[Sequence[int]] = None,
    expected_tasks: Optional[Sequence[str]] = None,
    experiment_id: Optional[str] = None,
    spec_hash: Optional[str] = None,
) -> Tuple[List[PairedComparisonResult], Dict[str, Any]]:
    """Run the preregistered paired family without silently dropping arms.

    ``expected_tasks`` is part of the preregistered design, rather than being
    inferred from whatever runs happened to finish.  This makes an entirely
    absent task (or arm) an explicit unavailable comparison in the manifest.
    """
    conditions = (
        "structured_provenance_memory", "adaptive_no_memory", "text_summary_memory",
        "shuffled_memory_control", "random_mattergen",
    )
    # Keep the preregistered task universe even when an arm or an entire task
    # is absent.  Conversely, retain every observed task (including an
    # unexpected condition) so that an operator cannot make a task disappear
    # merely by changing its arm label.
    observed_tasks = {
        str(_get(r, "task_id")) for r in run_metrics_list
        if _get(r, "task_id") is not None
    }
    tasks = sorted({str(task) for task in (expected_tasks or []) if task is not None} | observed_tasks)

    def _seed(value: Any) -> Optional[int]:
        try:
            # bool is an integer subclass but is not a valid experimental seed.
            if isinstance(value, bool):
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    all_seeds = sorted(
        {seed for seed in (_seed(value) for value in (expected_seeds or [])) if seed is not None}
        | {seed for seed in (_seed(_get(r, "seed")) for r in run_metrics_list) if seed is not None}
    )
    by: Dict[str, Dict[str, Dict[int, Any]]] = {}
    for rm in run_metrics_list:
        task, cond, seed = _get(rm, "task_id"), _get(rm, "condition"), _get(rm, "seed")
        seed_i = _seed(seed)
        if task is None or cond is None or seed_i is None:
            continue
        by.setdefault(str(task), {}).setdefault(str(cond), {})[seed_i] = rm

    confirmatory_metrics = {
        PRIMARY_CENSORED_ENDPOINT_NAME,
        THRESHOLD_YIELD_ENDPOINT_NAME,
    }
    analyzed_metrics = [
        PRIMARY_CENSORED_ENDPOINT_NAME, THRESHOLD_YIELD_ENDPOINT_NAME,
        "fraction_at_or_below_0_05", "fraction_at_or_below_0_03", "fraction_at_or_below_0_00",
        "geometry_yield", "oracle_success_rate", "unique_reduced_compositions_count",
    ]
    results: List[PairedComparisonResult] = []
    confirmatory_indices: List[int] = []
    confirmatory_p: List[float] = []
    for task in tasks:
        arms = by.get(task, {})
        proposed = arms.get("structured_provenance_memory", {})
        for control in conditions[1:]:
            comp_type = "confirmatory" if control != "random_mattergen" else "exploratory"
            for metric in analyzed_metrics:
                a_map: Dict[int, Any] = {}
                b_map: Dict[int, Any] = {}
                if metric == PRIMARY_CENSORED_ENDPOINT_NAME:
                    # Do not cast an endpoint until its raw value has passed
                    # the finite/positive/fixed-budget validation.  The
                    # helper repeats this check at the public API boundary so
                    # direct callers receive the same fail-closed behavior.
                    def _budget_value(rm: Any) -> Optional[float]:
                        raw_budget = _get(rm, "oracle_budget", 0)
                        if not _finite(raw_budget):
                            return None
                        try:
                            value = float(raw_budget)
                        except (TypeError, ValueError):
                            return None
                        return value if value > 0.0 else None

                    budget_values = [
                        value
                        for rm in list(proposed.values()) + list(arms.get(control, {}).values())
                        for value in [_budget_value(rm)]
                        if value is not None
                    ]
                    # Kaplan--Meier/RMST requires one fixed oracle budget for
                    # the paired comparison.  Taking max() would incorrectly
                    # admit a time that exceeded its own run budget whenever a
                    # different arm happened to have a larger budget.
                    common_budget = (
                        budget_values[0]
                        if budget_values and all(value == budget_values[0] for value in budget_values)
                        else 0.0
                    )
                    budget = common_budget
                    for s, rm in proposed.items():
                        if _get(rm, "provenance_complete") is True:
                            value = _get(rm, metric)
                            row_budget = _budget_value(rm)
                            if common_budget > 0.0 and row_budget == common_budget and _valid_censored_time(value, row_budget):
                                a_map[s] = (float(value), not bool(_get(rm, "primary_endpoint_censored", not bool(_get(rm, "reached_0_10_threshold", False)))))
                    for s, rm in arms.get(control, {}).items():
                        if _get(rm, "provenance_complete") is True:
                            value = _get(rm, metric)
                            row_budget = _budget_value(rm)
                            if common_budget > 0.0 and row_budget == common_budget and _valid_censored_time(value, row_budget):
                                b_map[s] = (float(value), not bool(_get(rm, "primary_endpoint_censored", not bool(_get(rm, "reached_0_10_threshold", False)))))
                    res = compute_paired_censored_comparison(task, metric, a_map, b_map, "structured_provenance_memory", control, budget, comp_type, analysis_version, all_seeds)
                else:
                    for s, rm in proposed.items():
                        if _get(rm, "provenance_complete") is True:
                            value = _get(rm, metric)
                            if _finite(value):
                                a_map[s] = float(value)
                    for s, rm in arms.get(control, {}).items():
                        if _get(rm, "provenance_complete") is True:
                            value = _get(rm, metric)
                            if _finite(value):
                                b_map[s] = float(value)
                    res = compute_paired_comparison(task, metric, a_map, b_map, "structured_provenance_memory", control, comp_type, analysis_version, expected_seeds=all_seeds)
                if metric in confirmatory_metrics and comp_type == "confirmatory":
                    res.confirmatory_family = CONFIRMATORY_FAMILY_NAME
                results.append(res)

                # The comparison helpers report missing seeds, but an absent
                # arm/task with no seed observations would otherwise look like
                # an ordinary empty comparison.  Keep this distinction in the
                # machine-readable result so reporting cannot mistake it for
                # negative scientific evidence.
                if not proposed or not arms.get(control):
                    res.status = "UNAVAILABLE_MISSING_TASK_OR_ARM"
                    res.p_value_raw = None
                    res.p_value_adjusted = None
                    res.adjustment_method = None
                if comp_type == "confirmatory" and metric in confirmatory_metrics and res.p_value_raw is not None:
                    confirmatory_indices.append(len(results) - 1)
                    confirmatory_p.append(res.p_value_raw)

    for idx, adjusted in zip(confirmatory_indices, holm_bonferroni_adjust(confirmatory_p)):
        results[idx].p_value_adjusted = adjusted
        results[idx].adjustment_method = HOLM_ADJUSTMENT_METHOD_NAME

    results_payload = [r.to_dict() for r in results]
    manifest = {
        "analysis_version": analysis_version,
        "experiment_id": experiment_id,
        "spec_hash": spec_hash,
        "primary_endpoint": PRIMARY_CENSORED_ENDPOINT_NAME,
        "primary_endpoint_semantics": "right_censored_at_fixed_oracle_budget",
        "confirmatory_family": CONFIRMATORY_FAMILY_NAME,
        "confirmatory_metrics": sorted(confirmatory_metrics),
        "confirmatory_controls": list(conditions[1:4]),
        "all_conditions": list(conditions),
        "expected_seeds": all_seeds,
        "expected_tasks": tasks,
        "total_comparisons": len(results),
        "confirmatory_comparisons_count": len(confirmatory_indices),
        "adjustment_method": HOLM_ADJUSTMENT_METHOD_NAME,
        "results": results_payload,
    }
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. effects.csv
        fields = [
            "analysis_version", "target_task", "metric_name", "condition_a", "condition_b",
            "comparison_type", "sample_size_n", "missing_count", "mean_a", "mean_b",
            "mean_difference", "median_difference", "ci_95_lower", "ci_95_upper",
            "p_value_raw", "p_value_adjusted", "adjustment_method", "method", "status",
        ]
        import io
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.to_dict().get(k, "") for k in fields})
        csv_bytes = csv_buffer.getvalue().encode("utf-8")
        _atomic_write_file(output_dir / "effects.csv", csv_bytes)
        effects_sha256 = hashlib.sha256(csv_bytes).hexdigest()

        # 2. results.json (canonical structured results)
        results_canonical_json = json.dumps(results_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        results_bytes = results_canonical_json.encode("utf-8")
        _atomic_write_file(output_dir / "results.json", results_bytes)
        results_sha256 = hashlib.sha256(results_bytes).hexdigest()
        canonical_results_sha256 = hashlib.sha256(results_bytes).hexdigest()

        # 3. analysis_manifest.json
        manifest["artifacts"] = {
            "effects.csv": effects_sha256,
            "results.json": results_sha256,
            "canonical_results_sha256": canonical_results_sha256,
        }
        manifest_json = json.dumps(manifest, indent=2, sort_keys=True)
        _atomic_write_file(output_dir / "analysis_manifest.json", manifest_json.encode("utf-8"))

    return results, {
        "analysis_version": analysis_version,
        "experiment_id": experiment_id,
        "spec_hash": spec_hash,
        "total_comparisons": len(results),
        "confirmatory_comparisons": len(confirmatory_indices),
        "expected_seeds": all_seeds,
        "manifest": manifest,
    }


__all__ = [
    "CONFIRMATORY_FAMILY_NAME",
    "PRIMARY_CENSORED_ENDPOINT_NAME",
    "PRIMARY_CENSORED_METHOD_NAME",
    "THRESHOLD_YIELD_ENDPOINT_NAME",
    "THRESHOLD_YIELD_METHOD_NAME",
    "HOLM_ADJUSTMENT_METHOD_NAME",
    "KaplanMeierEstimate",
    "PairedComparisonResult",
    "bootstrap_paired_ci",
    "calculate_minimum_exact_test_sample_size",
    "paired_permutation_test_pvalue",
    "holm_bonferroni_adjust",
    "kaplan_meier",
    "compute_paired_comparison",
    "compute_paired_censored_comparison",
    "run_statistical_analysis_pipeline",
]
