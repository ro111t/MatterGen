# Mattergen Autonomous Materials Discovery Agent

## System Overview

An agentic system that autonomously generates, evaluates, and optimizes novel materials using Mattergen and computational validation tools.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Orchestrator Agent (LLM)                     │
│  - Strategic planning & hypothesis generation                   │
│  - Result interpretation & learning                              │
│  - Exploration vs exploitation decisions                         │
└───────────────┬─────────────────────────────────────────────────┘
                │
    ┌───────────┴───────────┐
    │                       │
┌───▼────────────┐    ┌────▼──────────────┐
│  Memory System │    │  Knowledge Base   │
│  - Campaign    │    │  - Materials DB   │
│    history     │    │  - Property data  │
│  - Learned     │    │  - Synthesis      │
│    patterns    │    │    feasibility    │
└───┬────────────┘    └────┬──────────────┘
    │                      │
┌───▼──────────────────────▼────────────────────────────────────┐
│                    Agent Workflow Loop                         │
└───┬────────────────────────────────────────────────────────────┘
    │
    ├──► 1. GENERATION AGENT
    │    ├─ Mattergen API interface (Microsoft MatterGen diffusion model)
    │    ├─ Bundled sampling configs & GemNet scaling factors for pip installs
    │    ├─ Graceful fallback to pymatgen mock if MatterGen is unavailable
    │    ├─ Condition on target properties / chemical system
    │    ├─ Diversity sampling strategies
    │    └─ Batch generation (10-100 candidates)
    │
    ├──► 2. SCREENING AGENT (Fast Filter)
    │    ├─ ML property predictors
    │    │  ├─ M3GNet (stability, formation energy)
    │    │  ├─ CHGNet (forces, stresses)
    │    │  └─ MEGNet (band gap, bulk modulus)
    │    ├─ Rule-based filters
    │    │  ├─ Charge neutrality
    │    │  ├─ Reasonable bond lengths
    │    │  └─ Chemical feasibility
    │    └─ Rank top 10-20% for detailed validation
    │
    ├──► 3. VALIDATION AGENT (Computational)
    │    ├─ DFT calculations (VASP/QE via ASE)
    │    ├─ Phonon calculations (stability)
    │    ├─ Electronic structure analysis
    │    ├─ Defect formation energies
    │    └─ Parallel job scheduling
    │
    ├──► 4. ANALYSIS AGENT
    │    ├─ Compare predictions vs DFT
    │    ├─ Extract structure-property relationships
    │    ├─ Identify promising candidates
    │    ├─ Failure mode analysis
    │    └─ Update success metrics
    │
    ├──► 5. SYNTHESIS FEASIBILITY AGENT
    │    ├─ Check against ICSD/COD databases
    │    ├─ Precursor availability analysis
    │    ├─ Synthesis route prediction (ML)
    │    ├─ Cost estimation
    │    └─ Experimental difficulty scoring
    │
    └──► 6. STRATEGY AGENT (Meta-learning)
         ├─ Bayesian optimization of search space
         ├─ Active learning sample selection
         ├─ Adjust generation parameters
         ├─ Exploration vs exploitation balance
         └─ Campaign termination criteria

┌─────────────────────────────────────────────────────────────────┐
│                    External Integrations                         │
├─────────────────────────────────────────────────────────────────┤
│ • Mattergen API (Microsoft)                                      │
│ • Materials Project API (property lookup)                        │
│ • ASE (Atomic Simulation Environment)                            │
│ • Pymatgen (materials analysis)                                  │
│ • VASP/Quantum ESPRESSO (DFT)                                    │
│ • MatBench (benchmarking)                                        │
│ • CIF/POSCAR parsers                                             │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Orchestrator Agent (LLM-Based)

**Role**: High-level strategic decision making

**Capabilities**:
- Formulate hypotheses about material properties
- Interpret scientific results in context
- Decide when to pivot search strategies
- Generate natural language reports
- Interface with human researchers

**Implementation**:
```python
class OrchestratorAgent:
    def __init__(self, llm_model, memory, knowledge_base):
        self.llm = llm_model  # GPT-4, Claude, etc.
        self.memory = memory
        self.kb = knowledge_base
        
    def plan_campaign(self, objective):
        """Create search strategy for material objective"""
        # e.g., "Find solid electrolyte with >1 mS/cm Li+ conductivity"
        
    def interpret_results(self, batch_results):
        """Analyze batch outcomes and extract insights"""
        
    def adjust_strategy(self, performance_metrics):
        """Modify search parameters based on success rate"""
```

### 2. Generation Agent

**Role**: Interface with Mattergen to create novel materials

**Key Functions**:
- Conditional generation based on target properties
- Diversity-aware sampling (avoid duplicates)
- Compositional constraints (e.g., "must contain Li")
- Crystal system preferences

**Implementation**:
```python
class GenerationAgent:
    def __init__(self, mattergen_client):
        self.mattergen = mattergen_client
        self.generation_history = []
        
    def generate_batch(self, 
                      target_properties: dict,
                      constraints: dict,
                      num_samples: int = 50,
                      diversity_weight: float = 0.3):
        """
        Generate novel materials conditioned on targets
        
        Args:
            target_properties: {"band_gap": 2.5, "formation_energy": -2.0}
            constraints: {"elements": ["Li", "P", "S"], "max_atoms": 20}
            num_samples: Number of candidates to generate
            diversity_weight: Balance novelty vs target matching
        """
        candidates = self.mattergen.generate(
            properties=target_properties,
            constraints=constraints,
            n=num_samples,
            temperature=diversity_weight
        )
        return candidates
        
    def adaptive_sampling(self, feedback):
        """Adjust generation parameters based on validation results"""
```

### 3. Screening Agent (Fast ML Models)

**Role**: Rapid filtering of generated candidates

**Models**:
- **M3GNet**: Universal potential for stability/energy
- **CHGNet**: Forces, stresses, magnetic properties
- **MEGNet**: Graph network for various properties
- **ALIGNN**: Lattice-based predictions

**Workflow**:
```python
class ScreeningAgent:
    def __init__(self):
        self.m3gnet = load_model("M3GNet-MP-2021.2.8")
        self.chgnet = load_model("CHGNet")
        self.alignn = load_model("ALIGNN")
        
    def screen_batch(self, structures, criteria):
        """
        Fast screening of generated structures
        
        Returns: Top candidates ranked by multi-objective score
        """
        results = []
        for struct in structures:
            # Predict properties
            energy = self.m3gnet.predict_energy(struct)
            band_gap = self.alignn.predict_bandgap(struct)
            stability = self.chgnet.predict_stability(struct)
            
            # Apply filters
            if self.passes_filters(energy, band_gap, stability, criteria):
                score = self.calculate_score(energy, band_gap, stability)
                results.append((struct, score, predictions))
                
        # Return top 20%
        return sorted(results, key=lambda x: x[1], reverse=True)[:len(structures)//5]
        
    def passes_filters(self, energy, band_gap, stability, criteria):
        """Rule-based filtering"""
        if energy > criteria.get("max_energy", 0):
            return False
        if stability < criteria.get("min_stability", -0.1):
            return False
        # Additional checks...
        return True
```

### 4. Validation Agent (DFT Calculations)

**Role**: High-fidelity computational validation

**Tasks**:
- Structure relaxation
- Total energy calculations
- Electronic band structure
- Phonon dispersion (dynamical stability)
- Elastic constants

**Implementation**:
```python
class ValidationAgent:
    def __init__(self, calculator="vasp", n_workers=8):
        self.calculator = calculator
        self.job_queue = JobQueue(n_workers)
        
    def validate_structure(self, structure, properties_to_compute):
        """
        Run DFT calculations on a structure
        
        Returns: Validated properties + computational cost
        """
        job = DFTJob(
            structure=structure,
            calculator=self.calculator,
            tasks=properties_to_compute,
            convergence_criteria={"energy": 1e-5, "forces": 0.01}
        )
        
        result = self.job_queue.submit(job)
        return result
        
    def batch_validate(self, structures, priority_queue=True):
        """Parallel validation with priority scheduling"""
        jobs = [self.create_job(s) for s in structures]
        
        if priority_queue:
            # Prioritize by screening score
            jobs = self.prioritize_jobs(jobs)
            
        results = self.job_queue.submit_batch(jobs)
        return results
```

### 5. Analysis Agent

**Role**: Extract insights and update knowledge

**Functions**:
- Compare ML predictions vs DFT ground truth
- Identify structure-property patterns
- Detect failure modes
- Update model calibration

**Implementation**:
```python
class AnalysisAgent:
    def __init__(self, llm_model):
        self.llm = llm_model
        self.pattern_db = PatternDatabase()
        
    def analyze_batch(self, screening_results, dft_results):
        """
        Compare predictions with validation
        
        Returns: Insights, model errors, promising candidates
        """
        insights = {
            "prediction_accuracy": self.compute_errors(screening_results, dft_results),
            "successful_candidates": self.filter_successful(dft_results),
            "failure_modes": self.identify_failures(dft_results),
            "structure_patterns": self.extract_patterns(dft_results)
        }
        
        # LLM-based interpretation
        narrative = self.llm.generate_report(insights)
        
        # Update knowledge base
        self.pattern_db.update(insights["structure_patterns"])
        
        return insights, narrative
        
    def extract_patterns(self, results):
        """Find correlations between structure and properties"""
        # e.g., "Materials with corner-sharing octahedra show higher stability"
```

### 6. Synthesis Feasibility Agent

**Role**: Assess experimental realizability

**Checks**:
- Similar structures in experimental databases (ICSD, COD)
- Precursor availability
- Synthesis temperature/pressure requirements
- Predicted synthesis routes

**Implementation**:
```python
class SynthesisFeasibilityAgent:
    def __init__(self, icsd_api, synthesis_model):
        self.icsd = icsd_api
        self.synthesis_predictor = synthesis_model
        
    def assess_feasibility(self, structure):
        """
        Estimate likelihood of successful synthesis
        
        Returns: Feasibility score (0-1), suggested route
        """
        # Check for similar known materials
        similar = self.icsd.find_similar_structures(structure, threshold=0.8)
        
        # Predict synthesis route
        route = self.synthesis_predictor.predict_route(structure)
        
        # Estimate difficulty
        difficulty_score = self.estimate_difficulty(structure, route)
        
        return {
            "feasibility": 1.0 - difficulty_score,
            "similar_materials": similar,
            "suggested_route": route,
            "estimated_cost": self.estimate_cost(route)
        }
```

### 7. Strategy Agent (Meta-Learning)

**Role**: Optimize the search process itself

**Techniques**:
- Bayesian optimization over generation parameters
- Active learning for sample selection
- Multi-armed bandit for exploration/exploitation
- Campaign performance tracking

**Implementation**:
```python
class StrategyAgent:
    def __init__(self):
        self.optimizer = BayesianOptimizer()
        self.performance_history = []
        
    def update_strategy(self, campaign_results):
        """
        Adjust search parameters based on outcomes
        
        Returns: Updated generation parameters
        """
        # Extract features from successful materials
        success_features = self.extract_success_features(campaign_results)
        
        # Update Bayesian model
        self.optimizer.update(success_features)
        
        # Suggest next batch parameters
        next_params = self.optimizer.suggest()
        
        return next_params
        
    def should_terminate(self, campaign_metrics):
        """Decide if search objective is met or stuck"""
        if campaign_metrics["best_score"] > campaign_metrics["target_score"]:
            return True, "Target achieved"
            
        if campaign_metrics["improvement_rate"] < 0.01:
            return True, "Diminishing returns"
            
        return False, "Continue search"
```

## Memory System

**Short-term Memory**:
- Current campaign state
- Recent batch results
- Active hypotheses

**Long-term Memory**:
- All generated structures (vector DB)
- Validated properties database
- Learned structure-property rules
- Failed attempts (to avoid repetition)

**Implementation**:
```python
class MemorySystem:
    def __init__(self):
        self.vector_db = ChromaDB()  # For structure similarity search
        self.sql_db = PostgreSQL()   # For structured property data
        self.graph_db = Neo4j()      # For relationship mapping
        
    def store_candidate(self, structure, properties, metadata):
        """Store with multiple indexing strategies"""
        
    def retrieve_similar(self, structure, k=10):
        """Find similar previously explored materials"""
        
    def query_patterns(self, pattern_description):
        """Natural language query over learned patterns"""
```

## Workflow Loop

```python
class MaterialsDiscoveryCampaign:
    def __init__(self, objective, config):
        self.objective = objective
        self.orchestrator = OrchestratorAgent(...)
        self.generator = GenerationAgent(...)
        self.screener = ScreeningAgent(...)
        self.validator = ValidationAgent(...)
        self.analyzer = AnalysisAgent(...)
        self.synthesis = SynthesisFeasibilityAgent(...)
        self.strategy = StrategyAgent(...)
        self.memory = MemorySystem(...)
        
    def run_campaign(self, max_iterations=100, budget_hours=1000):
        """
        Main discovery loop
        """
        iteration = 0
        compute_budget_used = 0
        
        while iteration < max_iterations:
            print(f"\n=== Iteration {iteration} ===")
            
            # 1. Strategic planning
            strategy = self.orchestrator.plan_iteration(
                objective=self.objective,
                history=self.memory.get_campaign_history()
            )
            
            # 2. Generate candidates
            candidates = self.generator.generate_batch(
                target_properties=strategy["target_properties"],
                constraints=strategy["constraints"],
                num_samples=strategy["batch_size"]
            )
            print(f"Generated {len(candidates)} candidates")
            
            # 3. Fast screening
            screened = self.screener.screen_batch(
                structures=candidates,
                criteria=strategy["screening_criteria"]
            )
            print(f"Screened to {len(screened)} promising candidates")
            
            # 4. DFT validation (expensive)
            validated = self.validator.batch_validate(
                structures=[s[0] for s in screened],
                priority_queue=True
            )
            compute_budget_used += sum(v["cost_hours"] for v in validated)
            print(f"Validated {len(validated)} structures (budget: {compute_budget_used}/{budget_hours}h)")
            
            # 5. Synthesis feasibility
            feasible = []
            for v in validated:
                if v["meets_criteria"]:
                    feas = self.synthesis.assess_feasibility(v["structure"])
                    if feas["feasibility"] > 0.5:
                        feasible.append((v, feas))
            print(f"Found {len(feasible)} feasible candidates")
            
            # 6. Analysis and learning
            insights, report = self.analyzer.analyze_batch(screened, validated)
            print(f"\n{report}\n")
            
            # 7. Update memory
            self.memory.store_batch(candidates, screened, validated, insights)
            
            # 8. Meta-learning strategy update
            new_strategy = self.strategy.update_strategy({
                "screened": screened,
                "validated": validated,
                "feasible": feasible,
                "insights": insights
            })
            
            # 9. Check termination
            should_stop, reason = self.strategy.should_terminate({
                "best_score": max(v["score"] for v in validated),
                "target_score": self.objective["target_score"],
                "improvement_rate": insights["improvement_rate"],
                "budget_remaining": budget_hours - compute_budget_used
            })
            
            if should_stop:
                print(f"Campaign terminated: {reason}")
                break
                
            iteration += 1
            
        # Final report
        return self.generate_final_report()
```

## Key Design Principles

### 1. **Hierarchical Decision Making**
- LLM orchestrator for high-level strategy
- Specialized agents for domain-specific tasks
- Clear separation of concerns

### 2. **Multi-Fidelity Optimization**
- Fast ML screening (seconds per material)
- Expensive DFT validation (hours per material)
- Experimental feasibility (final gate)

### 3. **Active Learning**
- Prioritize informative samples
- Balance exploration vs exploitation
- Learn from failures

### 4. **Computational Efficiency**
- Parallel job execution
- Early termination of unpromising candidates
- Budget-aware scheduling

### 5. **Human-in-the-Loop**
- Natural language reporting
- Intervention points for domain expertise
- Explainable decisions

## Technology Stack

**Core Framework**:
- Python 3.10+
- LangGraph / CrewAI for agent orchestration
- Ray for distributed computing

**Materials Science**:
- Pymatgen (structure manipulation)
- ASE (atomistic simulations)
- MatMiner (feature extraction)
- M3GNet, CHGNet (ML potentials)

**ML/AI**:
- OpenAI API / Anthropic Claude (LLM)
- scikit-learn (classical ML)
- PyTorch (custom models)
- Weights & Biases (experiment tracking)

**Data**:
- ChromaDB (vector storage)
- PostgreSQL (structured data)
- Neo4j (knowledge graph)

**Compute**:
- SLURM integration (HPC clusters)
- Docker containers
- Kubernetes (optional, for cloud)

## Success Metrics

**Discovery Metrics**:
- Number of materials meeting target criteria
- Computational cost per successful material
- Prediction accuracy (ML vs DFT)

**Efficiency Metrics**:
- Screening funnel conversion rates
- Compute hours saved by ML screening
- Iteration time

**Learning Metrics**:
- Improvement in success rate over time
- Knowledge base growth
- Strategy adaptation effectiveness

## Deployment Scenarios

### Scenario 1: Local Prototype
- Single workstation
- Small batches (10-20 materials)
- CPU-based DFT (Quantum ESPRESSO)
- SQLite database

### Scenario 2: HPC Cluster
- University/national lab cluster
- Large batches (100-500 materials)
- GPU-accelerated ML + CPU DFT
- Full database stack

### Scenario 3: Cloud Hybrid
- Cloud for orchestration + ML
- HPC for DFT calculations
- Distributed across multiple sites
- Enterprise-grade monitoring

## Next Steps for Implementation

1. **Phase 1**: Build core agents (2-3 weeks)
   - Generation agent with Mattergen API
   - Screening agent with M3GNet
   - Basic orchestrator

2. **Phase 2**: Add validation (2-3 weeks)
   - ASE/VASP integration
   - Job queue system
   - Results database

3. **Phase 3**: Close the loop (2-3 weeks)
   - Analysis agent
   - Strategy agent
   - Memory system

4. **Phase 4**: Optimization (ongoing)
   - Performance tuning
   - Model calibration
   - Human feedback integration
