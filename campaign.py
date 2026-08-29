"""
campaign.py — top-level campaign orchestration for MatAgent.

Entrypoint:  python campaign.py [--domain ...] [--iterations N] [--no-career-memory]

Each campaign runs: plan → generate → screen → distill, looping until the
max iteration count is reached or a success criterion is met.
"""

import argparse
from collections import Counter
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from agents.orchestrator import OrchestratorAgent, CampaignObjective
from agents.generator import GenerationAgent
from agents.screening import ScreeningAgent
from agents.career_memory import CareerMemory
from agents.experience_distiller import ExperienceDistiller
from agents.validation import ValidationAgent, ValidationResult
from agents.synthesis import SynthesisFeasibilityAgent, SynthesisAssessment
from agents.strategy import StrategyAgent
from agents.analysis import AnalysisAgent, AnalysisResult
from agents.provenance import (
    ProvenanceTracker,
    CandidateStatus,
    CandidateRecord,
    RunManifest,
    extract_candidate_id,
)
from agents.integrity import (
    SCREENING_ENERGY_KEY,
    VALIDATION_ENERGY_KEY,
    RunMode,
    ScientificValidity,
    ScientificPreflightError,
    build_scientific_preflight,
    normalize_run_mode,
    assert_schema_v2_compatible,
)
from agents.geometry import DEFAULT_MIN_DISTANCE_ANGSTROM, GeometryValidator
from agents.budget import DualBudgetTracker
from agents.thermodynamics import (
    CHGNetRelaxationEvaluator,
    ThermodynamicOracle,
    load_frozen_reference_set,
    threshold_sensitivity,
)


@dataclass
class CampaignConfig:
    """Configuration for a materials discovery campaign"""
    name: str
    objective: CampaignObjective
    output_dir: Path
    master_seed: int = 42
    career_db_path: str = "~/.matagent_career.db"
    checkpoint_interval: int = 5
    verbose: bool = True
    use_career_memory: bool = True
    use_validation: bool = True
    validation_top_k: int = 5
    use_synthesis: bool = True
    synthesis_min_feasibility: float = 0.3
    num_candidates: int = 15
    use_mattergen: bool = False
    mattergen_pretrained: str = "mattergen_base"
    mattergen_model_path: Optional[str] = None
    mattergen_batch_size: int = 16
    mattergen_sampling_config_path: Optional[str] = None
    mattergen_sampling_config_name: str = "default"
    # Scientific execution boundary.  Development remains the backwards-
    # compatible default; research is fail-closed until all capabilities are
    # explicitly configured.
    run_mode: RunMode = RunMode.DEVELOPMENT
    validation_calculator: Any = "mock"
    synthesis_mode: str = "mock"
    require_thermodynamics: bool = True
    thermodynamics_backend: Optional[str] = None
    thermodynamics_reference_set_path: Optional[str] = None
    thermodynamics_evaluator: Any = None
    thermodynamics_available: bool = False
    thermodynamics_retain_threshold_ev_per_atom: float = 0.10
    thermodynamics_stable_threshold_ev_per_atom: float = 0.03
    # Resource accounting.  ``None`` keeps the existing unlimited development
    # behavior; research runs must provide explicit positive limits.
    proposal_budget: Optional[int] = None
    oracle_budget: Optional[int] = None
    geometry_min_distance: float = DEFAULT_MIN_DISTANCE_ANGSTROM
    geometry_minimum_distance: Optional[float] = None
    # Common aliases accepted for JSON/CLI-era callers.
    min_distance: Optional[float] = None
    minimum_distance: Optional[float] = None
    min_distance_angstrom: Optional[float] = None

    def __post_init__(self) -> None:
        self.run_mode = normalize_run_mode(self.run_mode)
        for name in ("proposal_budget", "oracle_budget"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer when supplied")
        if self.run_mode == RunMode.RESEARCH:
            missing = [name for name in ("proposal_budget", "oracle_budget") if getattr(self, name) is None]
            if missing:
                raise ValueError("Research mode requires explicit positive proposal_budget and oracle_budget")
        selected_distance = self.minimum_distance if self.minimum_distance is not None else self.min_distance
        if self.min_distance_angstrom is not None:
            selected_distance = self.min_distance_angstrom
        if self.geometry_minimum_distance is not None:
            selected_distance = self.geometry_minimum_distance
        if selected_distance is not None:
            self.geometry_min_distance = selected_distance
        try:
            self.geometry_min_distance = float(self.geometry_min_distance)
        except (TypeError, ValueError) as exc:
            raise ValueError("geometry_min_distance must be finite and positive") from exc
        if not __import__("math").isfinite(self.geometry_min_distance) or self.geometry_min_distance <= 0:
            raise ValueError("geometry_min_distance must be finite and positive")
        for name in ("thermodynamics_retain_threshold_ev_per_atom", "thermodynamics_stable_threshold_ev_per_atom"):
            value = float(getattr(self, name))
            if not __import__("math").isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            setattr(self, name, value)


class MaterialsDiscoveryCampaign:
    """
    Autonomous materials discovery campaign with persistent CareerMemory and full Candidate Provenance.
    """

    def __init__(self, config: CampaignConfig):
        self.config = config
        self.config.run_mode = normalize_run_mode(self.config.run_mode)
        self.budget_tracker = DualBudgetTracker(
            proposal_budget=self.config.proposal_budget,
            oracle_budget=self.config.oracle_budget,
        )
        self.geometry_validator = GeometryValidator(
            min_distance=self.config.geometry_min_distance,
        )
        self.iteration = 0
        self.results_history = []
        self.campaign_id = ""
        self.termination_reason: Optional[str] = None

        # Build the execution boundary before opening persistent memory or
        # creating output artifacts.  In research mode each component is
        # initialized strictly and all failures are reported together.
        if self.config.run_mode == RunMode.RESEARCH:
            self._initialize_components_research()
            self._run_research_preflight()
        else:
            self.thermodynamic_oracle = self._init_thermodynamic_oracle()
            self.generator = self._init_generator()
            self.screener = self._init_screener()
            self.validator = self._init_validator()
            self.synthesis = self._init_synthesis_agent()
            self.analyzer = self._init_analysis_agent()
            self.strategy = self._init_strategy_agent()
            self.requested_backends = self._requested_backend_metadata()
            self.actual_backends = self._actual_backend_metadata()

        # Career memory — persists across ALL campaigns
        if config.use_career_memory:
            self.career_memory = CareerMemory(db_path=config.career_db_path)
        else:
            self.career_memory = None

        self.orchestrator = OrchestratorAgent(
            career_memory=self.career_memory,
            api_key=os.environ.get('OPENAI_API_KEY')
        )
        self.current_recommendations: Optional[Dict[str, Any]] = None
        self.distiller = ExperienceDistiller(
            career_memory=self.career_memory,
            llm_client=self.orchestrator.llm
        )

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.provenance = ProvenanceTracker(
            campaign_id=self.campaign_id or f"camp_{int(time.time())}_{uuid.uuid4().hex[:6]}",
            campaign_name=self.config.name,
            domain=self.config.objective.domain,
            output_dir=self.config.output_dir,
            master_seed=getattr(self.config, "master_seed", 42),
            config={
                'name': self.config.name,
                'domain': self.config.objective.domain,
                'use_career_memory': self.config.use_career_memory,
                'use_mattergen': self.config.use_mattergen,
                'mattergen_pretrained': self.config.mattergen_pretrained,
                'mattergen_batch_size': self.config.mattergen_batch_size,
                'mattergen_model_path': self.config.mattergen_model_path,
                'mattergen_sampling_config_path': self.config.mattergen_sampling_config_path,
                'mattergen_sampling_config_name': self.config.mattergen_sampling_config_name,
                'run_mode': self.config.run_mode.value,
                'validation_calculator': self._requested_validation_backend(),
                'synthesis_mode': self.config.synthesis_mode,
                'require_thermodynamics': self.config.require_thermodynamics,
                'thermodynamics_backend': self.config.thermodynamics_backend,
                'thermodynamics_reference_set_path': self.config.thermodynamics_reference_set_path,
                'thermodynamics_retain_threshold_ev_per_atom': self.config.thermodynamics_retain_threshold_ev_per_atom,
                'thermodynamics_stable_threshold_ev_per_atom': self.config.thermodynamics_stable_threshold_ev_per_atom,
                'use_validation': self.config.use_validation,
                'validation_top_k': self.config.validation_top_k,
                'use_synthesis': self.config.use_synthesis,
                'num_candidates': self.config.num_candidates,
                'master_seed': getattr(self.config, "master_seed", 42),
                'proposal_budget': self.config.proposal_budget,
                'oracle_budget': self.config.oracle_budget,
                'geometry_min_distance': self.config.geometry_min_distance,
            },
            objective=self.config.objective.target_properties,
            constraints=self.config.objective.constraints,
            run_mode=self.config.run_mode.value,
            scientific_validity=(
                ScientificValidity.RESEARCH_VALID.value
                if self.config.run_mode == RunMode.RESEARCH
                else ScientificValidity.DEMO_ONLY.value
            ),
            requested_backends=self.requested_backends,
            actual_backends=self.actual_backends,
            proposal_budget=self.config.proposal_budget,
            oracle_budget=self.config.oracle_budget,
        )

    def _requested_validation_backend(self) -> str:
        """Stable metadata label for a configured validation backend."""
        calculator = self.config.validation_calculator
        return str(calculator) if isinstance(calculator, str) else "ase_calculator_object"

    def _requested_backend_metadata(self) -> Dict[str, Any]:
        """Describe requested scientific components before initialization."""
        return {
            "generation": "mattergen" if self.config.use_mattergen else "pymatgen_mock",
            "screening": "chgnet",
            "validation": self._requested_validation_backend() if self.config.use_validation else "disabled",
            "synthesis": self.config.synthesis_mode if self.config.use_synthesis else "disabled",
            "thermodynamics": (
                self.config.thermodynamics_backend
                if self.config.thermodynamics_backend
                else ("required" if self.config.require_thermodynamics else "not_required")
            ),
        }

    def _actual_backend_metadata(self) -> Dict[str, Any]:
        """Read actual component identities without inferring scientific validity."""
        generator_backend = getattr(self.generator, "backend_name", None)
        screener_backend = getattr(self.screener, "last_backend_used", None)
        if screener_backend in {None, "uninitialized"}:
            screener_backend = "chgnet" if getattr(self.screener, "chgnet", None) is not None else "heuristic"
        validator_info = self.validator.get_backend_info() if hasattr(self.validator, "get_backend_info") else {}
        validation_backend = (
            "disabled" if not self.config.use_validation
            else validator_info.get("backend_type") or getattr(self.validator, "calculator_name", None)
        )
        synthesis_backend = "disabled" if not self.config.use_synthesis else getattr(self.synthesis, "mode", self.config.synthesis_mode)
        return {
            "generation": generator_backend or "unavailable",
            "screening": screener_backend or "unavailable",
            "validation": validation_backend or "unavailable",
            "synthesis": synthesis_backend,
            "thermodynamics": (
                "certified_frozen_chgnet_hull"
                if getattr(self, "thermodynamic_oracle", None) is not None
                and self.thermodynamic_oracle.capability
                else "not_configured"
            ),
        }

    def _initialize_components_research(self) -> None:
        """Initialize all research components while collecting failures."""
        self.requested_backends = self._requested_backend_metadata()
        self._component_init_errors: List[str] = []

        def init(name: str, factory: Any, fallback: Any = None) -> Any:
            try:
                return factory()
            except Exception as exc:
                self._component_init_errors.append(f"{name}: {exc}")
                return fallback

        self.generator = init("generation", self._init_generator)
        self.thermodynamic_oracle = init("thermodynamics", self._init_thermodynamic_oracle)
        self.screener = init("screening", self._init_screener)
        self.validator = (
            init("validation", self._init_validator)
            if self.config.use_validation else None
        )
        self.synthesis = (
            init("synthesis", self._init_synthesis_agent)
            if self.config.use_synthesis else None
        )
        # Analysis and strategy do not provide scientific evidence, but keep
        # construction consistent for future successful research runs.
        self.analyzer = self._init_analysis_agent()
        self.strategy = self._init_strategy_agent()
        self.actual_backends = self._actual_backend_metadata()

    def _run_research_preflight(self) -> None:
        """Apply the reusable fail-closed research gate."""
        report = build_scientific_preflight(
            run_mode=self.config.run_mode,
            requested_backends=self.requested_backends,
            actual_backends=self.actual_backends,
            require_thermodynamics=self.config.require_thermodynamics,
            thermodynamics_available=(
                getattr(self, "thermodynamic_oracle", None) is not None
                and self.thermodynamic_oracle.capability
            ),
        )
        if self._component_init_errors:
            report.errors = self._component_init_errors + report.errors
        if not report.valid:
            raise ScientificPreflightError(report)
        
    def run_campaign(self) -> Dict[str, Any]:
        """Execute the full discovery campaign with career memory and provenance tracking."""
        objective = self.config.objective

        # Register campaign in career memory
        if self.career_memory:
            self.campaign_id = self.career_memory.start_campaign(
                name=self.config.name,
                domain=objective.domain,
                objective={
                    'target_properties': objective.target_properties,
                    'constraints': objective.constraints
                }
            )
            career_summary = self.career_memory.get_career_summary()
            self._log(f"\nCareer memory: {career_summary['total_campaigns']} prior campaigns, "
                      f"{career_summary['high_confidence_principles']} known principles")
        else:
            self.campaign_id = f"camp_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        self.provenance.campaign_id = self.campaign_id
        self.provenance.manifest.campaign_id = self.campaign_id
        self.provenance.sync_budget(self.budget_tracker)
        self.provenance.write_manifest()

        self._log(f"\nStarting campaign: {self.config.name}  [id={self.campaign_id}]")
        self._log(f"Domain: {objective.domain}")
        self._log(f"Target: {objective.target_properties}")

        start_time = time.time()

        while self.iteration < objective.max_iterations:
            if self.budget_tracker.proposal_budget_remaining == 0:
                self.termination_reason = "PROPOSAL_BUDGET_EXHAUSTED"
                self.budget_tracker.set_termination(self.termination_reason)
                break
            if self.budget_tracker.oracle_budget_remaining == 0:
                self.termination_reason = "ORACLE_BUDGET_EXHAUSTED"
                self.budget_tracker.set_termination(self.termination_reason)
                break
            self._log(f"\n{'='*60}")
            self._log(f"ITERATION {self.iteration}")
            self._log(f"{'='*60}")

            iteration_result = self._run_iteration()
            self.results_history.append(iteration_result)

            should_stop, reason = self._check_termination(iteration_result)
            if should_stop:
                self.termination_reason = reason
                self.budget_tracker.set_termination(reason)
                self.provenance.sync_budget(self.budget_tracker, iteration=self.iteration, termination_reason=reason)
                self._log(f"\nCampaign terminated: {reason}")
                break

            if self.iteration % self.config.checkpoint_interval == 0:
                self._save_checkpoint()

            self.iteration += 1

        if self.termination_reason is None:
            # Reaching max_iterations is a clean, explicit termination state.
            self.termination_reason = "MAX_ITERATIONS_REACHED"
            self.budget_tracker.set_termination(self.termination_reason)
        self.provenance.sync_budget(self.budget_tracker, termination_reason=self.termination_reason)

        elapsed_time = time.time() - start_time
        final_results = self._generate_final_report(elapsed_time)

        # Close campaign in career memory
        if self.career_memory:
            self.career_memory.end_campaign(self.campaign_id, final_results)

        return final_results
        
    def _run_iteration(self) -> Dict[str, Any]:
        """Execute one iteration: plan → generate → screen → validate → synthesize → distill → report."""
        budget_before = self.budget_tracker.to_dict()

        # 1. Plan with career memory warm-start and strategy-agent recommendations
        self._log("\n[1/6] Planning...")
        strategy = self.orchestrator.plan_iteration(
            objective=self.config.objective,
            history=self.results_history,
            campaign_id=self.campaign_id,
            iteration=self.iteration,
            recommendations=self.current_recommendations,
        )
        if not strategy.get("_replayed"):
            # Only set user-configured batch size on the first iteration if no recommendation exists
            if self.iteration == 0 and not self.current_recommendations and self.config.num_candidates:
                strategy['num_candidates'] = self.config.num_candidates
        self.provenance.record_strategy(self.iteration, strategy)
        self._log(f"  Elements: {strategy.get('elements', [])}")
        self._log(f"  Candidates: {strategy.get('num_candidates', self.config.num_candidates)}")
        self._log(f"  Rationale: {strategy.get('rationale', 'N/A')}")
        if strategy.get('hypothesis'):
            self._log(f"  Hypothesis: {strategy['hypothesis']}")

        # 2. Generate
        self._log("\n[2/6] Generating Candidates...")
        if (
            self.provenance
            and self.provenance.manifest.iteration_seeds
            and self.iteration < len(self.provenance.manifest.iteration_seeds)
        ):
            iter_seed = self.provenance.manifest.iteration_seeds[self.iteration]
        else:
            iter_seed = getattr(self.config, "master_seed", 42) + self.iteration
        requested_num_to_gen = strategy.get('num_candidates', self.config.num_candidates)
        try:
            requested_num_to_gen = max(0, int(requested_num_to_gen))
        except (TypeError, ValueError):
            requested_num_to_gen = max(0, int(self.config.num_candidates))
        num_to_gen = self.budget_tracker.generation_capacity(requested_num_to_gen)
        candidates = self.generator.generate_batch(
            elements=strategy.get('elements', ['Li', 'P', 'S', 'O']),
            num_candidates=num_to_gen,
            seed=iter_seed,
        )
        # A backend that overproduces has already consumed proposal resources,
        # so silently slicing would make the accounting non-auditable.  Abort
        # explicitly; callers can inspect the backend error and retry with a
        # corrected adapter.
        candidates = list(candidates or [])
        if len(candidates) > num_to_gen:
            raise RuntimeError(
                f"Generation backend returned {len(candidates)} candidates for a request of {num_to_gen}; "
                "overproduction cannot be silently discarded under the proposal budget."
            )
        self.budget_tracker.record_proposals(len(candidates))
        generation_backend = self.generator.last_generation_backend or getattr(self.generator, "backend_name", "stub")
        self.provenance.register_generation(
            candidates=candidates,
            iteration=self.iteration,
            backend=generation_backend,
            seed=iter_seed,
            target_elements=strategy.get('elements', ['Li', 'P', 'S', 'O']),
            parameters={
                'elements': strategy.get('elements', []),
                'num_candidates': len(candidates),
                'requested_num_candidates': requested_num_to_gen,
                'proposal_budget_remaining': self.budget_tracker.proposal_budget_remaining,
            },
            model_name_or_path=self.config.mattergen_model_path if generation_backend == "mattergen" else None,
            checkpoint=self.config.mattergen_pretrained if generation_backend == "mattergen" else None,
        )
        self._log(f"  Generated: {len(candidates)} structures")
        self._log(f"  Generation backend: {generation_backend}")

        if not candidates:
            # Preserve a valid iteration record for an empty backend response;
            # no screening/oracle call is made and the campaign can terminate
            # deterministically.
            self.provenance.sync_budget(self.budget_tracker, iteration=self.iteration)
            return self._empty_iteration_result(generation_backend, strategy)

        # 3. Screen with CHGNet/M3GNet
        self._log("\n[3/6] Screening with ML Models...")
        screening_criteria = strategy.get('screening_criteria', {})
        screened = self.screener.screen_batch(
            structures=candidates,
            criteria=screening_criteria,
            target_properties=self.config.objective.target_properties,
            deduplicate=False,
            budget_tracker=self.budget_tracker,
        )
        self.provenance.sync_budget(self.budget_tracker, iteration=self.iteration)
        screener_backend = getattr(self.screener, "last_backend_used", None)
        if not screener_backend or screener_backend == "uninitialized":
            screener_backend = "chgnet" if getattr(self.screener, "chgnet", None) is not None else "heuristic"
        self.provenance.record_screening(
            screening_results=screened,
            criteria=screening_criteria,
            iteration=self.iteration,
            backend=screener_backend,
        )
        n_pass = sum(1 for _, r in screened if r.passes_filters)
        scores = [r.score for _, r in screened]
        best_score = max(scores) if scores else 0.0
        # Raw per-atom model energy is retained in provenance for audit only.
        self._log(f"  Screened: {len(screened)} total, {n_pass} passed filters")
        self._log(f"  Best score: {best_score:.3f}")

        # 4. Validate top candidates with DFT or mock DFT
        validation_results: List[ValidationResult] = []
        if self.config.use_validation and screened:
            self._log("\n[4/6] Validating top candidates...")
            top_to_validate = [
                struct for struct, result in screened
                if result.passes_filters
            ][:self.config.validation_top_k]
            if top_to_validate:
                validation_results = self.validator.batch_validate(top_to_validate)
                self.provenance.record_validation(validation_results, iteration=self.iteration)
                n_converged = sum(1 for v in validation_results if v.converged)
                total_cost = sum(v.cost_hours for v in validation_results)
                self._log(f"  Validated: {len(validation_results)} structures, {n_converged} converged")
                self._log(f"  Validation cost: {total_cost:.1f} compute-hours")
            else:
                self._log("  No candidates passed screening filters; skipping validation.")
        else:
            self._log("\n[4/6] Validation disabled.")

        # 5. Analyze ML vs DFT calibration
        analysis_result: Optional[AnalysisResult] = None
        if validation_results:
            self._log("\n[5/7] Analyzing ML vs DFT Calibration...")
            analysis_result = self.analyzer.analyze_batch(screened, validation_results)
            for insight in analysis_result.insights[:4]:
                self._log(f"  - {insight}")
        else:
            self._log("\n[5/7] Skipping analysis; no validation results.")

        # 6. Synthesis feasibility assessment (evaluate only converged candidates)
        synthesis_results: List[SynthesisAssessment] = []
        converged_validations = [v for v in validation_results if v.converged]
        if self.config.use_synthesis:
            if self.config.use_validation and converged_validations:
                self._log("\n[6/7] Assessing Synthesis Feasibility...")
                validated_structures = [v.structure for v in converged_validations]
                validated_ids = [v.structure_id for v in converged_validations]
                synthesis_results = self.synthesis.assess_batch(validated_structures, validated_ids)
                self.provenance.record_synthesis(
                    synthesis_results,
                    iteration=self.iteration,
                    mode=getattr(self.synthesis, "mode", "mock"),
                )
                n_feasible = sum(1 for s in synthesis_results if s.feasible)
                avg_feasibility = sum(s.feasibility_score for s in synthesis_results) / max(len(synthesis_results), 1)
                self._log(f"  Feasible: {n_feasible}/{len(synthesis_results)} (avg score {avg_feasibility:.3f})")
            elif (not self.config.use_validation) and screened:
                top_to_synth = [
                    struct for struct, result in screened
                    if result.passes_filters
                ][:self.config.validation_top_k]
                if top_to_synth:
                    self._log("\n[6/7] Assessing Synthesis Feasibility (validation disabled)...")
                    synth_ids = [extract_candidate_id(s) for s in top_to_synth]
                    synthesis_results = self.synthesis.assess_batch(top_to_synth, synth_ids)
                    self.provenance.record_synthesis(
                        synthesis_results,
                        iteration=self.iteration,
                        mode=getattr(self.synthesis, "mode", "mock"),
                    )
                    n_feasible = sum(1 for s in synthesis_results if s.feasible)
                    avg_feasibility = sum(s.feasibility_score for s in synthesis_results) / max(len(synthesis_results), 1)
                    self._log(f"  Feasible: {n_feasible}/{len(synthesis_results)} (avg score {avg_feasibility:.3f})")
        else:
            self._log("\n[6/7] Synthesis assessment disabled or no validated structures.")

        # 7. Distill experience into CareerMemory
        self._log("\n[7/7] Distilling Experience...")
        distill_result = self.distiller.distill_iteration(
            campaign_id=self.campaign_id,
            domain=self.config.objective.domain,
            iteration=self.iteration,
            candidates=candidates,
            screening_results=screened,
            strategy=strategy
        )
        self._log(f"  Principles written: {distill_result['principles_written']}")
        self._log(f"  Failures recorded: {distill_result['failures_recorded']}")
        if distill_result.get('top_principles'):
            for p in distill_result['top_principles']:
                self._log(f"    -> {p[:80]}...")

        # Record ranking
        ranking_entries = [
            (res.structure_id or extract_candidate_id(struct), res.score, res.rank)
            for struct, res in screened
        ]
        self.provenance.record_ranking(ranking_entries, iteration=self.iteration)

        # Record candidate final acceptance / rejection decisions in provenance
        converged_ids = {v.structure_id for v in validation_results if v.converged}
        failed_val_ids = {v.structure_id for v in validation_results if not v.converged}
        feasible_synth_ids = {s.structure_id for s in synthesis_results if s.feasible}
        failed_synth_ids = {s.structure_id for s in synthesis_results if not s.feasible}

        for struct, res in screened:
            cid = res.structure_id or extract_candidate_id(struct)
            if not res.passes_filters:
                continue  # already marked REJECTED in record_screening

            if self.config.use_validation:
                if cid in failed_val_ids:
                    continue  # already marked REJECTED in record_validation
                if cid not in converged_ids:
                    # Filtered out by validation_top_k cutoff
                    self.provenance.record_decision(
                        candidate_id=cid,
                        status=CandidateStatus.REJECTED,
                        rejection_stage="ranking",
                        rejection_reason=f"Rank {res.rank} exceeds validation_top_k cutoff ({self.config.validation_top_k})",
                        ranking_score=res.score,
                        iteration_rank=res.rank,
                    )
                    continue

            if self.config.use_synthesis:
                if cid in failed_synth_ids:
                    continue  # already marked REJECTED in record_synthesis
                if cid not in feasible_synth_ids:
                    self.provenance.record_decision(
                        candidate_id=cid,
                        status=CandidateStatus.REJECTED,
                        rejection_stage="synthesis",
                        rejection_reason="Synthesis feasibility assessment not passed or missing",
                        ranking_score=res.score,
                        iteration_rank=res.rank,
                    )
                    continue

            # Accepted candidate
            self.provenance.record_decision(
                candidate_id=cid,
                status=CandidateStatus.ACCEPTED,
                ranking_score=res.score,
                iteration_rank=res.rank,
                stored_in_memory=self.config.use_career_memory,
                strategy_influence=strategy.get("hypothesis"),
            )

        n_converged = sum(1 for v in validation_results if v.converged)
        total_cost = sum(v.cost_hours for v in validation_results)
        n_synthesis_feasible = sum(1 for s in synthesis_results if s.feasible)
        avg_synthesis_feasibility = sum(s.feasibility_score for s in synthesis_results) / max(len(synthesis_results), 1)
        best_synthesis = max(
            (s.feasibility_score for s in synthesis_results),
            default=0.0
        )
        hull_values = [
            result.predictions.get("predicted_energy_above_hull_ev_per_atom")
            for _, result in screened
            if result.predictions.get("predicted_energy_above_hull_ev_per_atom") is not None
        ]

        insights = {
            'generation_backend': generation_backend,
            'num_generated': len(candidates),
            'num_screened': len(screened),
            'num_passed': n_pass,
            'success_rate': n_pass / max(len(screened), 1),
            'screening_rate': n_pass / max(len(screened), 1),
            'best_score': best_score,
            'thermodynamics_metrics_available': bool(hull_values),
            'predicted_hull_threshold_sensitivity': threshold_sensitivity(hull_values),
            'num_validated': len(validation_results),
            'num_converged': n_converged,
            'validation_cost_hours': total_cost,
            'num_synthesis_assessed': len(synthesis_results),
            'num_synthesis_feasible': n_synthesis_feasible,
            'avg_synthesis_feasibility': avg_synthesis_feasibility,
            'best_synthesis_feasibility': best_synthesis,
            'analysis_top_candidate': analysis_result.top_candidate_id if analysis_result else None,
            'analysis_top_candidate_score': analysis_result.top_candidate_score if analysis_result else 0.0,
            'ml_dft_mae': analysis_result.ml_vs_dft_mae if analysis_result else {},
            'principles_written': distill_result['principles_written'],
        }
        budget_after = self.budget_tracker.to_dict()
        budget_iteration = {
            key: (
                budget_after[key] - budget_before[key]
                if key in {"proposals_generated", "geometry_valid", "invalid_geometry", "oracle_evaluations", "oracle_cache_hits"}
                else budget_after[key]
            )
            for key in budget_after
        }
        # Iteration counters are deltas; remaining capacities are post-iteration state.
        insights.update(budget_iteration)
        self.provenance.sync_budget(self.budget_tracker, iteration=self.iteration)

        # LLM interpretation
        report = self.orchestrator.interpret_results(insights)
        self._log(f"\n  {report}")

        # Update adaptive strategy agent and get recommendation for next iteration
        self.strategy.update(
            iteration=self.iteration,
            strategy=strategy,
            insights=insights,
        )
        self.current_recommendations = self.strategy.recommend(
            objective=self.config.objective,
            history=self.results_history,
        )
        self._log(f"\n[Strategy] Next iteration recommendation: {self.current_recommendations['rationale']}")

        return {
            'iteration': self.iteration,
            'generation_backend': generation_backend,
            'num_generated': len(candidates),
            'num_screened': len(screened),
            'num_validated': len(validation_results),
            'validation_results': [
                {
                    'structure_id': v.structure_id,
                    'calculator': v.calculator,
                    'converged': v.converged,
                    'properties': v.properties,
                    'cost_hours': v.cost_hours,
                    'error_message': v.error_message,
                }
                for v in validation_results
            ],
            'synthesis_results': [
                {
                    'structure_id': s.structure_id,
                    'feasible': s.feasible,
                    'feasibility_score': s.feasibility_score,
                    'difficulty_score': s.difficulty_score,
                    'estimated_cost': s.estimated_cost,
                    'synthesis_route': s.synthesis_route,
                    'similar_known_phases': s.similar_known_phases,
                    'warnings': s.warnings,
                }
                for s in synthesis_results
            ],
            'analysis': {
                'top_candidate_id': analysis_result.top_candidate_id if analysis_result else None,
                'top_candidate_score': analysis_result.top_candidate_score if analysis_result else 0.0,
                'ml_vs_dft_mae': analysis_result.ml_vs_dft_mae if analysis_result else {},
                'ml_vs_dft_rmse': analysis_result.ml_vs_dft_rmse if analysis_result else {},
                'ml_vs_dft_bias': analysis_result.ml_vs_dft_bias if analysis_result else {},
                'pearson_r': analysis_result.pearson_r if analysis_result else {},
                'failure_modes': analysis_result.failure_modes if analysis_result else [],
                'insights': analysis_result.insights if analysis_result else [],
            },
            'insights': insights,
            'strategy': strategy,
            **budget_iteration,
        }

    def _empty_iteration_result(self, generation_backend: str, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Return a schema-compatible iteration result for an empty generation batch."""
        snapshot = self.budget_tracker.to_dict()
        insights = {
            "generation_backend": generation_backend,
            "num_generated": 0,
            "num_screened": 0,
            "num_passed": 0,
            "best_score": 0.0,
            "thermodynamics_metrics_available": False,
            "num_validated": 0,
            "num_converged": 0,
            "num_synthesis_feasible": 0,
            "principles_written": 0,
            **snapshot,
        }
        self.provenance.sync_budget(self.budget_tracker, iteration=self.iteration)
        return {
            "iteration": self.iteration,
            "generation_backend": generation_backend,
            "num_generated": 0,
            "num_screened": 0,
            "num_validated": 0,
            "validation_results": [],
            "synthesis_results": [],
            "analysis": {},
            "insights": insights,
            "strategy": strategy,
            **snapshot,
        }

    def _check_termination(self, iteration_result: Dict[str, Any]) -> tuple:
        """Check if campaign should stop."""
        insights = iteration_result.get('insights', {})
        best_score = insights.get('best_score', 0)

        if self.budget_tracker.proposal_budget_remaining == 0:
            return True, "PROPOSAL_BUDGET_EXHAUSTED"
        if self.budget_tracker.oracle_budget_remaining == 0:
            return True, "ORACLE_BUDGET_EXHAUSTED"
        if iteration_result.get("num_generated", 0) == 0:
            return True, "NO_CANDIDATES_GENERATED"

        min_score = self.config.objective.success_criteria.get('min_score', float('inf'))
        if best_score >= min_score:
            return True, f"Target score {min_score} achieved (best={best_score:.2f})"

        # Stagnation check: no improvement in last 5 iterations
        if len(self.results_history) >= 5:
            recent_scores = [r['insights'].get('best_score', 0) for r in self.results_history[-5:]]
            if max(recent_scores) == 0:
                return True, "No scoring candidates in last 5 iterations"

        return False, "Continue"

    def _save_checkpoint(self):
        """Save campaign state to disk."""
        checkpoint = {
            'iteration': self.iteration,
            'campaign_id': self.campaign_id,
            'results_history': self.results_history,
            'config': {
                'name': self.config.name,
                'domain': self.config.objective.domain,
                'proposal_budget': self.config.proposal_budget,
                'oracle_budget': self.config.oracle_budget,
                'geometry_min_distance': self.config.geometry_min_distance,
                'thermodynamics_reference_set_path': self.config.thermodynamics_reference_set_path,
                'thermodynamics_retain_threshold_ev_per_atom': self.config.thermodynamics_retain_threshold_ev_per_atom,
                'thermodynamics_stable_threshold_ev_per_atom': self.config.thermodynamics_stable_threshold_ev_per_atom,
            },
            'budget': self.budget_tracker.to_dict(termination_reason=self.termination_reason),
        }
        checkpoint_path = self.config.output_dir / f"checkpoint_{self.iteration}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
        self._log(f"Checkpoint saved: {checkpoint_path}")

    def _generate_final_report(self, elapsed_time: float) -> Dict[str, Any]:
        """Generate final report and persist to disk using ProvenanceTracker as single source of truth."""
        self.provenance.sync_budget(self.budget_tracker, termination_reason=self.termination_reason)
        stats = self.provenance.finalize(status="completed")
        total_principles = sum(r['insights'].get('principles_written', 0) for r in self.results_history)

        # Career memory top candidates
        top_candidates = []
        if self.career_memory:
            top_candidates = self.career_memory.get_top_candidates_ever(
                domain=self.config.objective.domain, top_n=5
            )

        n_iterations = len(self.results_history)
        default_backend = getattr(self.generator, "backend_name", "stub")
        batch_backends = [
            result.get('generation_backend', default_backend)
            for result in self.results_history
        ]
        backend_counts = dict(Counter(batch_backends))
        if len(set(batch_backends)) == 1:
            backend_name = batch_backends[0]
        elif len(set(batch_backends)) == 0:
            backend_name = default_backend
        else:
            backend_name = 'mixed'

        report = {
            'campaign_name': self.config.name,
            'campaign_id': self.campaign_id,
            'domain': self.config.objective.domain,
            'iterations': n_iterations,
            'elapsed_time_seconds': elapsed_time,
            'generation_backend': backend_name,
            'generation_backend_counts': backend_counts,
            'run_mode': self.config.run_mode.value,
            'scientific_validity': (
                ScientificValidity.RESEARCH_VALID.value
                if self.config.run_mode == RunMode.RESEARCH
                else ScientificValidity.DEMO_ONLY.value
            ),
            'thermodynamics_metrics_available': any(
                result.get("insights", {}).get("thermodynamics_metrics_available", False)
                for result in self.results_history
            ),
            'requested_backends': self.requested_backends,
            'actual_backends': self.actual_backends,
            'mattergen_pretrained': (
                self.config.mattergen_pretrained if 'mattergen' in backend_counts else None
            ),
            'total_generated': stats['total_generated'],
            'total_passed_screening': stats['total_passed_screening'],
            'overall_pass_rate': stats['overall_pass_rate'],
            'total_validated': stats['total_validated'],
            'total_converged': stats['total_converged'],
            'total_validation_cost_hours': stats['total_validation_cost_hours'],
            'total_synthesis_assessed': stats['total_synthesis_assessed'],
            'total_synthesis_feasible': stats['total_synthesis_feasible'],
            'best_score_ever': stats['best_score_ever'],
            'best_synthesis_feasibility_ever': stats['best_synthesis_feasibility_ever'],
            'principles_written_to_career': total_principles,
            'top_candidates': top_candidates,
            'proposals_generated': self.budget_tracker.proposals_generated,
            'geometry_valid': self.budget_tracker.geometry_valid,
            'invalid_geometry': self.budget_tracker.invalid_geometry,
            'oracle_evaluations': self.budget_tracker.oracle_evaluations,
            'oracle_cache_hits': self.budget_tracker.oracle_cache_hits,
            'proposal_budget': self.budget_tracker.proposal_budget,
            'oracle_budget': self.budget_tracker.oracle_budget,
            'proposal_budget_remaining': self.budget_tracker.proposal_budget_remaining,
            'oracle_budget_remaining': self.budget_tracker.oracle_budget_remaining,
            'termination_reason': self.termination_reason,
        }

        report_path = self.config.output_dir / f"report_{self.campaign_id}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)

        with open(self.config.output_dir / "report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)

        self._log(f"\n{'='*60}")
        self._log("CAMPAIGN COMPLETE")
        self._log(f"{'='*60}")
        self._log(f"Generation backend: {backend_name}")
        self._log(f"Iterations: {n_iterations}")
        self._log(f"Generated: {stats['total_generated']} structures")
        self._log(f"Passed screening: {stats['total_passed_screening']} ({report['overall_pass_rate']:.1%})")
        self._log(f"Best score: {stats['best_score_ever']:.3f}")
        self._log(f"Principles written to career memory: {total_principles}")
        if top_candidates:
            self._log("\nTop candidates:")
            for i, c in enumerate(top_candidates[:5], 1):
                self._log(f"  {i}. {c['formula']}  score={c['score']:.3f}")
        self._log(f"\nReport saved: {report_path}")

        return report

    @classmethod
    def reproduce_from_manifest(
        cls,
        manifest_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> "MaterialsDiscoveryCampaign":
        """
        Reconstruct and execute a campaign exactly from a saved manifest.json record.
        """
        manifest_file = Path(manifest_path).resolve()
        with open(manifest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        manifest = RunManifest.from_dict(data)
        # A v1 manifest may be parsed for audit, but cannot be executed as a
        # v2 scientific campaign.  This preserves the original artifact and
        # prevents legacy energy semantics entering new retrieval/statistics.
        assert_schema_v2_compatible(data, context="reproduction manifest")
        out_dir = Path(output_dir).resolve() if output_dir else manifest_file.parent / "reproduced"

        # Verify manifest integrity hash if present
        if manifest.manifest_hash:
            computed_hash = manifest.compute_manifest_hash()
            if manifest.manifest_hash != computed_hash:
                raise ValueError(
                    f"Manifest tampering detected! Embedded hash: {manifest.manifest_hash} != Computed hash: {computed_hash}"
                )

        obj_data = manifest.objective or {}
        constr_data = manifest.constraints or {}
        cfg_data = manifest.config or {}

        objective = CampaignObjective(
            target_properties=obj_data.get('target_properties', obj_data),
            constraints=constr_data.get('constraints', constr_data),
            success_criteria=cfg_data.get('success_criteria') or obj_data.get('success_criteria') or {'min_score': 999.0},
            domain=manifest.domain,
            max_iterations=len(manifest.iteration_seeds) if manifest.iteration_seeds else 1,
        )

        config = CampaignConfig(
            name=f"{manifest.campaign_name}_reproduced",
            objective=objective,
            output_dir=out_dir,
            master_seed=manifest.master_seed,
            use_career_memory=cfg_data.get('use_career_memory', False),
            use_validation=cfg_data.get('use_validation', True),
            validation_top_k=cfg_data.get('validation_top_k', 5),
            use_synthesis=cfg_data.get('use_synthesis', True),
            num_candidates=cfg_data.get('num_candidates', 15),
            use_mattergen=cfg_data.get('use_mattergen', False),
            mattergen_pretrained=cfg_data.get('mattergen_pretrained', 'mattergen_base'),
            mattergen_model_path=cfg_data.get('mattergen_model_path', None),
            mattergen_batch_size=cfg_data.get('mattergen_batch_size', 16),
            mattergen_sampling_config_path=cfg_data.get('mattergen_sampling_config_path', None),
            mattergen_sampling_config_name=cfg_data.get('mattergen_sampling_config_name', 'default'),
            run_mode=cfg_data.get('run_mode', manifest.run_mode),
            validation_calculator=cfg_data.get('validation_calculator', 'mock'),
            synthesis_mode=cfg_data.get('synthesis_mode', 'mock'),
            require_thermodynamics=cfg_data.get('require_thermodynamics', True),
            thermodynamics_backend=cfg_data.get('thermodynamics_backend'),
            thermodynamics_reference_set_path=cfg_data.get('thermodynamics_reference_set_path'),
            thermodynamics_retain_threshold_ev_per_atom=cfg_data.get('thermodynamics_retain_threshold_ev_per_atom', 0.10),
            thermodynamics_stable_threshold_ev_per_atom=cfg_data.get('thermodynamics_stable_threshold_ev_per_atom', 0.03),
            # Legacy Boolean is intentionally ignored: capability must be
            # established by loading the certified artifact and evaluator.
            thermodynamics_available=False,
            proposal_budget=cfg_data.get('proposal_budget', manifest.proposal_budget),
            oracle_budget=cfg_data.get('oracle_budget', manifest.oracle_budget),
            geometry_min_distance=cfg_data.get(
                'geometry_min_distance',
                cfg_data.get('min_distance', DEFAULT_MIN_DISTANCE_ANGSTROM),
            ),
            verbose=True,
        )

        campaign = cls(config)
        if manifest.iteration_seeds:
            campaign.provenance.manifest.iteration_seeds = list(manifest.iteration_seeds)

        # If strategies were recorded in the manifest, replay with exact strategy per iteration
        if manifest.strategies:
            saved_strats = {s.get("iteration", i): s for i, s in enumerate(manifest.strategies)}
            orig_plan = campaign.orchestrator.plan_iteration

            def replay_plan(*args, **kwargs):
                iter_num = kwargs.get("iteration", campaign.iteration)
                if iter_num in saved_strats:
                    strat = dict(saved_strats[iter_num])
                    strat["_replayed"] = True
                    return strat
                return orig_plan(*args, **kwargs)

            campaign.orchestrator.plan_iteration = replay_plan

        campaign.run_campaign()
        return campaign

    def _log(self, message: str):
        """Log message if verbose mode enabled."""
        if self.config.verbose:
            print(message)

    def _init_generator(self):
        """Initialize generation agent with MatterGen or pymatgen mock backend."""
        return GenerationAgent(
            use_mattergen=self.config.use_mattergen,
            mattergen_pretrained=self.config.mattergen_pretrained,
            mattergen_model_path=self.config.mattergen_model_path,
            mattergen_batch_size=self.config.mattergen_batch_size,
            mattergen_sampling_config_path=self.config.mattergen_sampling_config_path,
            mattergen_sampling_config_name=self.config.mattergen_sampling_config_name,
            run_mode=self.config.run_mode,
        )

    def _init_screener(self):
        """Initialize screening agent with real CHGNet."""
        return ScreeningAgent(
            run_mode=self.config.run_mode,
            geometry_validator=self.geometry_validator,
            thermodynamic_oracle=getattr(self, "thermodynamic_oracle", None),
        )

    def _init_thermodynamic_oracle(self):
        """Load an immutable certified set; never retrieve references at runtime."""
        path = self.config.thermodynamics_reference_set_path
        evaluator = self.config.thermodynamics_evaluator
        if not path:
            if self.config.run_mode == RunMode.RESEARCH and self.config.require_thermodynamics:
                raise RuntimeError("research thermodynamics requires thermodynamics_reference_set_path")
            return None
        configured_elements = self.config.objective.constraints.get("elements")
        required_chemical_system = (
            list(configured_elements)
            if isinstance(configured_elements, (list, tuple, set)) and configured_elements
            else None
        )
        frozen = load_frozen_reference_set(
            path, required_chemical_system=required_chemical_system,
        )
        if evaluator is None:
            evaluator = CHGNetRelaxationEvaluator(settings=frozen.relaxation_settings)
        frozen = load_frozen_reference_set(
            path, expected_model=evaluator.model_identity,
            expected_settings=evaluator.relaxation_settings,
            required_chemical_system=required_chemical_system,
        )
        return ThermodynamicOracle(
            frozen, evaluator, research=self.config.run_mode == RunMode.RESEARCH,
            geometry_min_distance=self.config.geometry_min_distance,
            retain_threshold_ev_per_atom=self.config.thermodynamics_retain_threshold_ev_per_atom,
            stable_threshold_ev_per_atom=self.config.thermodynamics_stable_threshold_ev_per_atom,
        )

    def _init_validator(self):
        """Initialize validation agent (mock DFT by default)."""
        return ValidationAgent(
            calculator=self.config.validation_calculator,
            n_workers=1,
            run_mode=self.config.run_mode,
        )

    def _init_synthesis_agent(self):
        """Initialize synthesis feasibility agent."""
        return SynthesisFeasibilityAgent(
            mode=self.config.synthesis_mode,
            run_mode=self.config.run_mode,
        )

    def _init_strategy_agent(self):
        """Initialize adaptive strategy agent."""
        return StrategyAgent(exploration_weight=0.2, target_metric="combined")

    def _init_analysis_agent(self):
        """Initialize ML-vs-DFT analysis agent."""
        return AnalysisAgent(properties_to_compare=[
            "predicted_energy_per_atom_ev",
            "energy_per_atom_ev",
            "max_force_ev_per_angstrom",
        ])


def main():
    """CLI entry-point for running a test campaign or reproducing from manifest."""
    parser = argparse.ArgumentParser(description="MatAgent Discovery Campaign")
    parser.add_argument('--reproduce', '--reproduce-from-manifest', dest='reproduce', type=str, default=None,
                        help='Path to manifest.json to reproduce a previously executed campaign')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory override (especially when reproducing)')
    parser.add_argument('--domain', default='li_solid_electrolyte')
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--candidates', type=int, default=15)
    parser.add_argument('--proposal-budget', type=int, default=None,
                        help='Maximum generated proposals (paper default: 400; omitted means unlimited in development)')
    parser.add_argument('--oracle-budget', type=int, default=None,
                        help='Maximum geometrically valid oracle evaluations (paper default: 200; omitted means unlimited in development)')
    parser.add_argument('--geometry-min-distance', '--min-distance', dest='geometry_min_distance', type=float, default=DEFAULT_MIN_DISTANCE_ANGSTROM,
                        help='Absolute periodic minimum-distance threshold in Angstrom (default: 0.8)')
    parser.add_argument('--thermodynamics-reference-set', type=str, default=None,
                        help='Path to an offline certified frozen reference-set JSON (runtime never downloads references)')
    parser.add_argument('--thermodynamics-retain-threshold', type=float, default=0.10,
                        help='Maximum predicted energy above hull retained, eV/atom (default: 0.10)')
    parser.add_argument('--thermodynamics-stable-threshold', type=float, default=0.03,
                        help='Predicted-stable energy-above-hull threshold, eV/atom (default: 0.03)')
    parser.add_argument('--master-seed', type=int, default=42)
    parser.add_argument('--no-career-memory', action='store_true')
    parser.add_argument('--no-validation', action='store_true')
    parser.add_argument('--validation-top-k', type=int, default=5)
    parser.add_argument('--no-synthesis', action='store_true')
    parser.add_argument('--run-mode', choices=[mode.value for mode in RunMode], default=RunMode.DEVELOPMENT.value,
                        help='Execution boundary: development permits deterministic mocks; research fails closed on missing scientific backends')
    parser.add_argument('--use-mattergen', action='store_true',
                        help='Use the Microsoft MatterGen diffusion model for generation')
    parser.add_argument('--mattergen-pretrained', type=str, default='mattergen_base',
                        help='MatterGen pretrained checkpoint name or "chemical_system" for element-conditioned generation')
    parser.add_argument('--mattergen-model-path', type=str, default=None,
                        help='Path to a local MatterGen checkpoint directory')
    parser.add_argument('--mattergen-batch-size', type=int, default=16,
                        help='Batch size for MatterGen generation')
    parser.add_argument('--mattergen-sampling-config-path', type=str, default=None,
                        help='Path to MatterGen sampling config directory (defaults to bundled configs)')
    parser.add_argument('--mattergen-sampling-config-name', type=str, default='default',
                        help='Name of the sampling config YAML file to use (default or csp)')
    args = parser.parse_args()

    if args.reproduce:
        print(f"Reproducing campaign from manifest: {args.reproduce}")
        campaign = MaterialsDiscoveryCampaign.reproduce_from_manifest(
            manifest_path=args.reproduce,
            output_dir=args.output_dir,
        )
        print("Reproduction complete.")
        return

    objective = CampaignObjective(
        target_properties={},
        constraints={'elements': ['Li', 'P', 'S', 'O', 'Cl'], 'max_atoms': 20},
        success_criteria={'min_score': 999.0},
        domain=args.domain,
        max_iterations=args.iterations,
    )

    out_dir = Path(args.output_dir) if args.output_dir else Path(f"./campaigns/{args.domain}")

    config = CampaignConfig(
        name=f"{args.domain}_campaign",
        objective=objective,
        output_dir=out_dir,
        master_seed=args.master_seed,
        use_career_memory=not args.no_career_memory,
        verbose=True,
        use_validation=not args.no_validation,
        validation_top_k=args.validation_top_k,
        use_synthesis=not args.no_synthesis,
        num_candidates=args.candidates,
        use_mattergen=args.use_mattergen,
        mattergen_pretrained=args.mattergen_pretrained,
        mattergen_model_path=args.mattergen_model_path,
        mattergen_batch_size=args.mattergen_batch_size,
        mattergen_sampling_config_path=args.mattergen_sampling_config_path,
        mattergen_sampling_config_name=args.mattergen_sampling_config_name,
        run_mode=args.run_mode,
        proposal_budget=args.proposal_budget,
        oracle_budget=args.oracle_budget,
        geometry_min_distance=args.geometry_min_distance,
        thermodynamics_reference_set_path=args.thermodynamics_reference_set,
        thermodynamics_retain_threshold_ev_per_atom=args.thermodynamics_retain_threshold,
        thermodynamics_stable_threshold_ev_per_atom=args.thermodynamics_stable_threshold,
    )

    campaign = MaterialsDiscoveryCampaign(config)
    results = campaign.run_campaign()

    print(f"\nDone. Best score: {results['best_score_ever']:.3f}")
    print(f"Principles written to career memory: {results['principles_written_to_career']}")


if __name__ == "__main__":
    main()
