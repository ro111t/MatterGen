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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.orchestrator import OrchestratorAgent, CampaignObjective
from agents.generator import GenerationAgent
from agents.screening import ScreeningAgent
from agents.career_memory import CareerMemory
from agents.experience_distiller import ExperienceDistiller
from agents.validation import ValidationAgent, ValidationResult
from agents.synthesis import SynthesisFeasibilityAgent, SynthesisAssessment
from agents.strategy import StrategyAgent
from agents.analysis import AnalysisAgent, AnalysisResult


@dataclass
class CampaignConfig:
    """Configuration for a materials discovery campaign"""
    name: str
    objective: CampaignObjective
    output_dir: Path
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


class MaterialsDiscoveryCampaign:
    """
    Autonomous materials discovery campaign with persistent CareerMemory.
    """

    def __init__(self, config: CampaignConfig):
        self.config = config
        self.iteration = 0
        self.results_history = []
        self.campaign_id = ""

        # Career memory — persists across ALL campaigns
        if config.use_career_memory:
            self.career_memory = CareerMemory(db_path=config.career_db_path)
        else:
            self.career_memory = None

        self.orchestrator = OrchestratorAgent(
            career_memory=self.career_memory,
            api_key=os.environ.get('OPENAI_API_KEY')
        )
        self.generator = self._init_generator()
        self.screener = self._init_screener()
        self.validator = self._init_validator()
        self.synthesis = self._init_synthesis_agent()
        self.analyzer = self._init_analysis_agent()
        self.strategy = self._init_strategy_agent()
        self.current_recommendations: Optional[Dict[str, Any]] = None
        self.distiller = ExperienceDistiller(
            career_memory=self.career_memory,
            llm_client=self.orchestrator.llm
        )

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_campaign(self) -> Dict[str, Any]:
        """Execute the full discovery campaign with career memory."""
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

        self._log(f"\nStarting campaign: {self.config.name}  [id={self.campaign_id}]")
        self._log(f"Domain: {objective.domain}")
        self._log(f"Target: {objective.target_properties}")

        start_time = time.time()

        while self.iteration < objective.max_iterations:
            self._log(f"\n{'='*60}")
            self._log(f"ITERATION {self.iteration}")
            self._log(f"{'='*60}")

            iteration_result = self._run_iteration()
            self.results_history.append(iteration_result)

            should_stop, reason = self._check_termination(iteration_result)
            if should_stop:
                self._log(f"\nCampaign terminated: {reason}")
                break

            if self.iteration % self.config.checkpoint_interval == 0:
                self._save_checkpoint()

            self.iteration += 1

        elapsed_time = time.time() - start_time
        final_results = self._generate_final_report(elapsed_time)

        # Close campaign in career memory
        if self.career_memory:
            self.career_memory.end_campaign(self.campaign_id, final_results)

        return final_results
        
    def _run_iteration(self) -> Dict[str, Any]:
        """Execute one iteration: plan → generate → screen → validate → synthesize → distill → report."""

        # 1. Plan with career memory warm-start and strategy-agent recommendations
        self._log("\n[1/6] Planning...")
        strategy = self.orchestrator.plan_iteration(
            objective=self.config.objective,
            history=self.results_history,
            campaign_id=self.campaign_id,
            iteration=self.iteration,
            recommendations=self.current_recommendations,
        )
        self._log(f"  Elements: {strategy.get('elements', [])}")
        self._log(f"  Candidates: {strategy.get('num_candidates', 15)}")
        self._log(f"  Rationale: {strategy.get('rationale', 'N/A')}")
        if strategy.get('hypothesis'):
            self._log(f"  Hypothesis: {strategy['hypothesis']}")

        # 2. Generate
        self._log("\n[2/6] Generating Candidates...")
        candidates = self.generator.generate_batch(
            elements=strategy.get('elements', ['Li', 'P', 'S', 'O']),
            num_candidates=strategy.get('num_candidates', self.config.num_candidates),
            seed=42 + self.iteration
        )
        generation_backend = self.generator.last_generation_backend or getattr(self.generator, "backend_name", "stub")
        self._log(f"  Generated: {len(candidates)} structures")
        self._log(f"  Generation backend: {generation_backend}")

        # 3. Screen with CHGNet/M3GNet
        self._log("\n[3/6] Screening with ML Models...")
        screened = self.screener.screen_batch(
            structures=candidates,
            criteria=strategy.get('screening_criteria', {}),
            target_properties=self.config.objective.target_properties,
        )
        n_pass = sum(1 for _, r in screened if r.passes_filters)
        scores = [r.score for _, r in screened]
        best_score = max(scores) if scores else 0.0
        avg_stability = sum(r.predictions.get('stability', 0) for _, r in screened) / max(len(screened), 1)
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
                n_converged = sum(1 for v in validation_results if v.converged)
                total_cost = sum(v.cost_hours for v in validation_results)
                best_validated = max(
                    (v.properties.get('stability', float('-inf')) for v in validation_results),
                    default=0.0
                )
                self._log(f"  Validated: {len(validation_results)} structures, {n_converged} converged")
                self._log(f"  Validation cost: {total_cost:.1f} compute-hours")
                self._log(f"  Best validated stability: {best_validated:.3f} eV/atom")
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

        # 6. Synthesis feasibility assessment
        synthesis_results: List[SynthesisAssessment] = []
        if self.config.use_synthesis and validation_results:
            self._log("\n[6/7] Assessing Synthesis Feasibility...")
            validated_structures = [v.structure for v in validation_results]
            validated_ids = [v.structure_id for v in validation_results]
            synthesis_results = self.synthesis.assess_batch(validated_structures, validated_ids)
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
                self._log(f"    → {p[:80]}...")

        n_converged = sum(1 for v in validation_results if v.converged)
        total_cost = sum(v.cost_hours for v in validation_results)
        best_validated = max(
            (v.properties.get('stability', float('-inf')) for v in validation_results),
            default=0.0
        )

        n_synthesis_feasible = sum(1 for s in synthesis_results if s.feasible)
        avg_synthesis_feasibility = sum(s.feasibility_score for s in synthesis_results) / max(len(synthesis_results), 1)
        best_synthesis = max(
            (s.feasibility_score for s in synthesis_results),
            default=0.0
        )

        insights = {
            'generation_backend': generation_backend,
            'num_generated': len(candidates),
            'num_screened': len(screened),
            'num_passed': n_pass,
            'success_rate': n_pass / max(len(screened), 1),
            'best_score': best_score,
            'avg_stability': avg_stability,
            'num_validated': len(validation_results),
            'num_converged': n_converged,
            'validation_cost_hours': total_cost,
            'best_validated_stability': best_validated,
            'num_synthesis_assessed': len(synthesis_results),
            'num_synthesis_feasible': n_synthesis_feasible,
            'avg_synthesis_feasibility': avg_synthesis_feasibility,
            'best_synthesis_feasibility': best_synthesis,
            'analysis_top_candidate': analysis_result.top_candidate_id if analysis_result else None,
            'analysis_top_candidate_score': analysis_result.top_candidate_score if analysis_result else 0.0,
            'ml_dft_mae': analysis_result.ml_vs_dft_mae if analysis_result else {},
            'principles_written': distill_result['principles_written'],
        }

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
            'strategy': strategy
        }

    def _check_termination(self, iteration_result: Dict[str, Any]) -> tuple:
        """Check if campaign should stop."""
        insights = iteration_result.get('insights', {})
        best_score = insights.get('best_score', 0)

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
            }
        }
        checkpoint_path = self.config.output_dir / f"checkpoint_{self.iteration}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
        self._log(f"Checkpoint saved: {checkpoint_path}")

    def _generate_final_report(self, elapsed_time: float) -> Dict[str, Any]:
        """Generate final report and persist to disk."""
        total_generated = sum(r['num_generated'] for r in self.results_history)
        total_passed = sum(r['insights'].get('num_passed', 0) for r in self.results_history)
        total_validated = sum(r['insights'].get('num_validated', 0) for r in self.results_history)
        total_converged = sum(r['insights'].get('num_converged', 0) for r in self.results_history)
        total_validation_cost = sum(r['insights'].get('validation_cost_hours', 0.0) for r in self.results_history)
        total_synthesis_assessed = sum(r['insights'].get('num_synthesis_assessed', 0) for r in self.results_history)
        total_synthesis_feasible = sum(r['insights'].get('num_synthesis_feasible', 0) for r in self.results_history)
        best_synthesis_scores = [r['insights'].get('best_synthesis_feasibility', 0.0) for r in self.results_history]
        best_synthesis_feasibility = max(best_synthesis_scores) if best_synthesis_scores else 0.0
        best_scores = [r['insights'].get('best_score', 0) for r in self.results_history]
        best_score = max(best_scores) if best_scores else 0.0
        best_validated_stabilities = [r['insights'].get('best_validated_stability', 0) for r in self.results_history]
        best_validated_stability = max(best_validated_stabilities) if best_validated_stabilities else 0.0
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
            'mattergen_pretrained': (
                self.config.mattergen_pretrained if 'mattergen' in backend_counts else None
            ),
            'total_generated': total_generated,
            'total_passed_screening': total_passed,
            'overall_pass_rate': total_passed / total_generated if total_generated > 0 else 0,
            'total_validated': total_validated,
            'total_converged': total_converged,
            'total_validation_cost_hours': total_validation_cost,
            'total_synthesis_assessed': total_synthesis_assessed,
            'total_synthesis_feasible': total_synthesis_feasible,
            'best_score_ever': best_score,
            'best_validated_stability_ever': best_validated_stability,
            'best_synthesis_feasibility_ever': best_synthesis_feasibility,
            'principles_written_to_career': total_principles,
            'top_candidates': top_candidates,
        }

        report_path = self.config.output_dir / f"report_{self.campaign_id}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        self._log(f"\n{'='*60}")
        self._log("CAMPAIGN COMPLETE")
        self._log(f"{'='*60}")
        self._log(f"Generation backend: {backend_name}")
        self._log(f"Iterations: {n_iterations}")
        self._log(f"Generated: {total_generated} structures")
        self._log(f"Passed screening: {total_passed} ({report['overall_pass_rate']:.1%})")
        self._log(f"Best score: {best_score:.3f}")
        self._log(f"Principles written to career memory: {total_principles}")
        if top_candidates:
            self._log("\nTop candidates:")
            for i, c in enumerate(top_candidates[:5], 1):
                self._log(f"  {i}. {c['formula']}  score={c['score']:.3f}")
        self._log(f"\nReport saved: {report_path}")

        return report

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
        )

    def _init_screener(self):
        """Initialize screening agent with real CHGNet."""
        return ScreeningAgent()

    def _init_validator(self):
        """Initialize validation agent (mock DFT by default)."""
        return ValidationAgent(calculator="mock", n_workers=1)

    def _init_synthesis_agent(self):
        """Initialize synthesis feasibility agent."""
        return SynthesisFeasibilityAgent(mode="mock")

    def _init_strategy_agent(self):
        """Initialize adaptive strategy agent."""
        return StrategyAgent(exploration_weight=0.2, target_metric="combined")

    def _init_analysis_agent(self):
        """Initialize ML-vs-DFT analysis agent."""
        return AnalysisAgent(properties_to_compare=["formation_energy", "energy", "stability", "forces"])


def main():
    """CLI entry-point for running a test campaign."""
    parser = argparse.ArgumentParser(description="MatAgent Discovery Campaign")
    parser.add_argument('--domain', default='li_solid_electrolyte')
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--candidates', type=int, default=15)
    parser.add_argument('--no-career-memory', action='store_true')
    parser.add_argument('--no-validation', action='store_true')
    parser.add_argument('--validation-top-k', type=int, default=5)
    parser.add_argument('--no-synthesis', action='store_true')
    parser.add_argument('--use-mattergen', action='store_true',
                        help='Use the Microsoft MatterGen diffusion model for generation (falls back to mock if unavailable)')
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

    objective = CampaignObjective(
        target_properties={'stability': -0.1, 'formation_energy': -2.0},
        constraints={'elements': ['Li', 'P', 'S', 'O', 'Cl'], 'max_atoms': 20},
        success_criteria={'min_score': 999.0},
        domain=args.domain,
        max_iterations=args.iterations,
    )

    config = CampaignConfig(
        name=f"{args.domain}_campaign",
        objective=objective,
        output_dir=Path(f"./campaigns/{args.domain}"),
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
    )

    campaign = MaterialsDiscoveryCampaign(config)
    results = campaign.run_campaign()

    print(f"\nDone. Best score: {results['best_score_ever']:.3f}")
    print(f"Principles written to career memory: {results['principles_written_to_career']}")


if __name__ == "__main__":
    main()
