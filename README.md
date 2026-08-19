# Mattergen Autonomous Materials Discovery Agent

[![Tests](https://github.com/ro111t/MatterGen/actions/workflows/pytest.yml/badge.svg)](https://github.com/ro111t/MatterGen/actions/workflows/pytest.yml)

An agentic system for autonomous discovery of novel materials using Mattergen and computational validation.

## Overview

This system implements a closed-loop materials discovery pipeline that:
1. Generates novel materials using Mattergen
2. Screens candidates with fast ML models
3. Validates promising materials with DFT
4. Learns from results to improve search strategy
5. Assesses synthesis feasibility

## Teammate Handoff

### Start Here

1. Create the Python 3.10 environment shown in [Installation](#installation).
2. Install development dependencies and run `python3 -m pytest tests/ -v`.
3. Run a small offline-safe campaign with `python campaign.py --iterations 1 --candidates 5 --no-career-memory`.
4. Launch the UI with `streamlit run ui/app.py` when you want to configure a campaign interactively.

### Repository Map

- `campaign.py`: orchestration entry point, configuration dataclass, CLI, checkpoints, and final reports.
- `agents/generator.py`: MatterGen adapter and deterministic pymatgen mock fallback.
- `agents/screening.py`: CHGNet-backed fast screening with multi-objective scoring, target-property matching, and composition deduplication.
- `agents/validation.py`: ASE validation interface; uses a deterministic mock calculator by default.
- `agents/analysis.py`, `agents/synthesis.py`, and `agents/strategy.py`: calibration analysis, synthesis heuristics, and adaptive search strategy.
- `agents/career_memory.py` and `agents/experience_distiller.py`: persistent learning between campaigns.
- `agents/mattergen_sampling_conf/`: bundled MatterGen `default`/`csp` sampling configurations and GemNet scale data required by incomplete pip wheels.
- `ui/app.py`: Streamlit interface.
- `tests/`: unit and end-to-end campaign coverage.

### Current MatterGen Status

The integration is complete and selectable through the CLI or UI. The project includes MatterGen sampling/configuration files that some pip installations omit. A generation error is caught per batch and uses the pymatgen mock fallback, so campaigns complete even if MatterGen is unavailable.

Real diffusion sampling has not been verified in the current macOS environment because it runs Python 3.13 with a NumPy version that removed `numpy.math`, which MatterGen currently expects. Use Python 3.10 with `numpy<2.0` and `torch==2.4.1` to enable the real backend. The mock path is fully tested and requires no HuggingFace access.

### Collaboration Notes

- Do not commit `.env` files, HuggingFace tokens, OpenAI keys, local virtual environments, campaign outputs, or CareerMemory databases.
- Campaign output is intentionally ignored by Git. Share a selected report or structure separately when needed.
- Run the full test suite before opening a pull request. It currently covers mock generation, MatterGen initialization/fallback, the campaign flow, validation, synthesis, analysis, and strategy.

## Architecture

```
Orchestrator (LLM) → Strategy → Generation → Screening → Validation → Analysis → Synthesis → Distill → (loop)
                        ↑                                                                               ↓
                        └────────────────────────── Learning Loop ──────────────────────────────────────┘
```

### Key Components

All agents below are implemented. Heavy external backends (Mattergen API, VASP/QE, Materials Project) are optional; the system ships with deterministic mocks/heuristics so it runs offline out of the box.

- **Orchestrator Agent**: High-level strategic planning using LLM; falls back to deterministic strategy when no API key is available.
- **Generation Agent**: Generates candidate structures using Microsoft MatterGen when available, with a pymatgen-based mock fallback.
- **Screening Agent**: Fast ML-based filtering with CHGNet/M3GNet/ALIGNN (currently CHGNet enabled by default).
- **Validation Agent**: DFT relaxation and property calculation via ASE; supports VASP, Quantum ESPRESSO, GPAW, and a deterministic `mock` backend.
- **Analysis Agent**: Compares ML screening predictions against DFT validation and reports MAE, RMSE, bias, Pearson r, failure modes, and top candidates.
- **Synthesis Feasibility Agent**: Estimates precursor difficulty, synthesis route, and experimental cost from composition heuristics.
- **Strategy Agent**: UCB-style bandit over element sets plus adaptive batch size / diversity weight.
- **CareerMemory / ExperienceDistiller**: Persistent SQLite storage of principles, hypotheses, and top candidates across campaigns.

## Installation

```bash
# Create environment
conda create -n mattergen-agent python=3.10
conda activate mattergen-agent

# Install dependencies
pip install -r requirements.txt

# Install materials science packages
pip install pymatgen ase matgl chgnet alignn

# Optional: enable the real MatterGen diffusion backend
# Follow the exact, pinned setup in the "Real MatterGen Setup" section below.
# The default environment continues to use the pymatgen mock.

# The project ships with bundled sampling configs and a GemNet scaling
# factor file so that the pip wheel can find them even when its own data
# files are missing. Use `--mattergen-sampling-config-path` to override.

# Optional: set a HuggingFace token to avoid rate limits when downloading checkpoints
export HF_TOKEN="your-hf-token"
```

## Real MatterGen Setup

The default `requirements.txt` workflow remains mock-only and does not require
MatterGen, CUDA, or a Hugging Face download. Use the separate environment below
when you need real diffusion sampling. It targets **Linux with an NVIDIA CUDA
11.8-capable GPU**; on Windows, create it inside WSL2 or use a remote Linux GPU
host.

```bash
# From a clean clone on the Linux GPU host
conda env create -f environment.yml
conda activate mattergen-agent

# Preserve the assignment's torch==2.4.1 pin while installing MatterGen 1.0.3.
python scripts/install_mattergen_runtime.py

# Normal tests remain offline-safe and continue to exercise the mock path.
python -m pytest tests/ -v

# Opt in to a checkpoint download and one real generated pymatgen.Structure.
# Option A (via pytest):
RUN_MATTERGEN_SMOKE=1 python -m pytest tests/test_mattergen_smoke.py -m mattergen_smoke -v

# Option B (standalone script):
python tests/test_mattergen_smoke.py
```

The smoke test fails if initialization or generation falls back to the mock
backend. It also checks Python 3.10, `numpy<2.0`, `torch==2.4.1`, and
`mattergen==1.0.3`. MatterGen 1.0.3 declares a Linux-specific Torch 2.2.1
dependency upstream, so the bootstrap script intentionally installs the pinned
MatterGen package without re-resolving dependencies; `environment.yml` and
`requirements-mattergen.txt` supply the configured Torch 2.4.1 runtime instead.

## Quick Start

```python
from pathlib import Path
from campaign import MaterialsDiscoveryCampaign, CampaignConfig
from agents.orchestrator import CampaignObjective

# Define what you're looking for
objective = CampaignObjective(
    target_properties={
        'band_gap': 2.5,  # eV
        'formation_energy': -2.0,  # eV/atom
    },
    constraints={
        'elements': ['Li', 'P', 'S', 'O'],
        'max_atoms': 20
    },
    success_criteria={'min_score': 0.8},
    max_iterations=50,
    compute_budget_hours=500.0
)

# Configure campaign
config = CampaignConfig(
    name="solid_electrolyte_discovery",
    objective=objective,
    output_dir=Path("./campaigns/electrolyte"),
    verbose=True,
    # use_mattergen=True,  # enable real MatterGen generation when available
    # mattergen_pretrained="chemical_system",  # element-conditioned generation
)

# Run
campaign = MaterialsDiscoveryCampaign(config)
results = campaign.run_campaign()
```

## Web UI

A lightweight Streamlit interface is available in `ui/app.py` for configuring and running campaigns interactively.

```bash
streamlit run ui/app.py
```

The UI lets you set target properties, constraints, run settings, and pipeline-stage toggles (validation, synthesis), choose the generation backend (pymatgen mock vs MatterGen), and displays summary metrics, the active backend, ML-vs-DFT calibration, validation/synthesis tables, and iteration history.

## Command-Line Interface

```bash
python campaign.py \
  --domain li_solid_electrolyte \
  --iterations 3 \
  --candidates 15 \
  --validation-top-k 3 \
  --no-synthesis \
  --use-mattergen \
  --mattergen-pretrained chemical_system \
  --mattergen-batch-size 8
```

Flags:
- `--domain`: campaign domain name
- `--iterations`: number of iterations to run
- `--candidates`: fallback/default batch size (orchestrator may adjust it)
- `--validation-top-k`: number of screened candidates to validate
- `--no-career-memory`: disable persistent learning across campaigns
- `--no-validation`: skip DFT validation
- `--no-synthesis`: skip synthesis feasibility assessment
- `--use-mattergen`: use the MatterGen diffusion model instead of the pymatgen mock
- `--mattergen-pretrained`: MatterGen checkpoint name (e.g. `mattergen_base`, `chemical_system`)
- `--mattergen-model-path`: path to a local MatterGen checkpoint directory
- `--mattergen-batch-size`: batch size for MatterGen generation
- `--mattergen-sampling-config-name`: name of the sampling config YAML file to use (`default` or `csp`)
- `--mattergen-sampling-config-path`: path to a MatterGen sampling config directory (defaults to bundled configs)

## Testing

Install dev dependencies and run the test suite:

```bash
pip install -r requirements-dev.txt
python3 -m pytest tests/ -v
```

## Configuration

### Target Properties

Specify desired material properties:
- `band_gap`: Electronic band gap (eV)
- `formation_energy`: Formation energy (eV/atom)
- `stability`: Thermodynamic stability (eV/atom)
- Custom properties supported by Mattergen

### Constraints

Control generation space:
- `elements`: Allowed chemical elements
- `max_atoms`: Maximum atoms per unit cell
- `crystal_system`: Preferred crystal symmetry
- `composition`: Stoichiometric constraints

### Screening Criteria

Filter candidates:
- `min_stability`: Minimum stability threshold
- `max_formation_energy`: Maximum formation energy
- `min_band_gap`, `max_band_gap`: Band gap range
- `max_forces`: Maximum atomic forces

### Multi-Objective Screening

`agents/screening.py` ranks candidates with a composite score that combines:
- **Stability** (favoring lower formation energy)
- **Relaxation quality** (low forces/stress)
- **Target property match** (closeness to campaign objective targets such as `band_gap`)
- **Composition novelty** (rarity within the generated batch)

The default weights are configurable via the `weights` argument to `screen_batch`. Duplicate compositions within a batch are deduplicated, keeping the highest-scoring representative.

```python
screened = screener.screen_batch(
    structures=candidates,
    criteria={"max_forces": 1.0},
    target_properties={"band_gap": 2.5},
    weights={"stability": 0.4, "target_property_match": 0.3, "relaxation_quality": 0.2, "composition_novelty": 0.1},
)
```

## Workflow

### Iteration Loop

Each iteration:
1. **Plan**: LLM or heuristic strategy decides element focus and batch size.
2. **Generate**: candidate structures via MatterGen diffusion model (when the environment satisfies its `numpy<2.0`/torch constraints) or pymatgen mock fallback.
3. **Screen**: CHGNet/M3GNet filters candidates against lenient thresholds.
4. **Validate**: ASE-backed DFT relaxation on top candidates (mock by default).
5. **Analyze**: compare ML predictions against DFT to quantify model error.
6. **Assess**: estimate synthesis feasibility and route.
7. **Learn**: CareerMemory records principles; StrategyAgent recommends next iteration.

### Multi-Fidelity Approach

- **Fast screening**: Seconds per material (ML models)
- **DFT validation**: Hours per material (accurate)
- **Synthesis check**: Minutes per material (databases)

### Active Learning

- Bayesian optimization of search space
- Exploration vs exploitation balance
- Failure mode analysis
- Pattern recognition

## Output

### Campaign Results

```
campaigns/
└── electrolyte/
    ├── checkpoint_10.json
    ├── checkpoint_20.json
    ├── final_report.json
    └── structures/
        ├── successful/
        └── all_validated/
```

### Final Report

```json
{
  "campaign_name": "solid_electrolyte_discovery",
  "domain": "li_solid_electrolyte",
  "iterations": 10,
  "generation_backend": "mattergen",
  "mattergen_pretrained": "chemical_system",
  "total_generated": 320,
  "total_passed_screening": 145,
  "overall_pass_rate": 0.45,
  "total_validated": 30,
  "total_converged": 30,
  "total_validation_cost_hours": 145.0,
  "total_synthesis_assessed": 30,
  "total_synthesis_feasible": 24,
  "best_score_ever": 78.5,
  "best_validated_stability_ever": -1.45,
  "best_synthesis_feasibility_ever": 0.82,
  "top_candidates": [...]
}
```

## Advanced Usage

### Custom Screening Models

```python
from agents.screening import ScreeningAgent

screener = ScreeningAgent(
    use_m3gnet=True,
    use_chgnet=True,
    use_alignn=True
)

# Add custom model
screener.models['custom'] = load_custom_model()
```

### Custom Strategy

```python
from agents.orchestrator import OrchestratorAgent

orchestrator = OrchestratorAgent(
    llm_client=your_llm,
    memory_system=your_memory,
    knowledge_base=your_kb
)

# Override planning
strategy = orchestrator.plan_iteration(objective, history)
```

### Parallel DFT

```python
from agents.validation import ValidationAgent

validator = ValidationAgent(
    calculator="vasp",
    n_workers=16  # Parallel jobs
)

results = validator.batch_validate(structures)
```

## Performance Tips

1. **Start small**: Begin with 20-30 candidates per batch
2. **Tune screening**: Adjust `top_k_percent` based on success rate
3. **Use checkpoints**: Resume from failures
4. **Monitor costs**: Track compute budget usage
5. **Calibrate models**: Update ML predictions with DFT results

## Integration

### Materials Project

```python
from mp_api.client import MPRester

mpr = MPRester("your-api-key")
# Query for similar materials
# Use as knowledge base
```

### Custom DFT

```python
from ase.calculators.vasp import Vasp

calc = Vasp(
    xc='PBE',
    encut=520,
    kpts=(4,4,4)
)
```

### Experiment Tracking

```python
import wandb

wandb.init(project="mattergen-discovery")
wandb.log({"success_rate": rate})
```

## Limitations

- **Computational cost**: DFT is expensive (hours per material)
- **Synthesis gap**: Computational stability ≠ synthesizability
- **Model accuracy**: ML predictions have ~10-20% error
- **Search space**: Vast - needs good priors

## Future Enhancements

- [ ] Multi-objective optimization (Pareto fronts)
- [ ] Transfer learning across campaigns
- [ ] Automated synthesis route planning
- [ ] Integration with robotic labs
- [ ] Real-time experimental feedback
- [ ] Uncertainty quantification

## References

- Mattergen: [Microsoft Research](https://www.microsoft.com/en-us/research/project/mattergen/)
- M3GNet: [Nature Computational Science](https://www.nature.com/articles/s43588-022-00349-3)
- Materials Project: [materialsproject.org](https://materialsproject.org)

## License

MIT

## Citation

If you use this system, please cite:
```bibtex
@software{mattergen_agent,
  title={Mattergen Autonomous Discovery Agent},
  authors={Rohit Vennelakanti, Joseph Press, Bradley Hawkins, Andrei Ivanou},
  year={2026}
}
```

## Support

For issues and questions:
- GitHub Issues: [[here]](https://github.com/ro111t/MatterGen/issues)
- Email: mattergenamda@gmail.com
