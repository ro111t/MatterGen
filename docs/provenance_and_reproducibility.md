# AMDA Candidate Provenance & Experiment Reproducibility Guide

This document defines the architecture, data schemas, storage conventions, reproduction workflows, and analysis tools for the **Autonomous Materials Discovery Agent (AMDA)** in the [`ro111t/MatterGen`](https://github.com/ro111t/MatterGen) repository.

---

## 1. Overview & Research Motivation

In autonomous materials discovery, discovering a candidate with promising predicted properties is only half the battle. Six months or five years from now, a researcher or experimental chemist must be able to answer:
* **Lineage**: Where did Candidate `MAT-000137` come from? What campaign, iteration, model, and seed generated it?
* **Physics & Calibration**: What were its CHGNet screening predictions? Did it converge under DFT relaxation? What were the forces, stress, and total energy?
* **Experimental Realizability**: What was its synthesis feasibility assessment, proposed route, precursor availability, and difficulty score?
* **Decision History**: Why was it accepted or rejected? Did its discovery influence subsequent exploration strategies or write principles into long-term CareerMemory?
* **Reproducibility**: Can another researcher replay the exact same campaign from a single saved manifest file?

AMDA treats **failures as critical scientific information, not disposable logs**. Every generated candidate is tracked through an explicit 7-stage state machine from birth to final disposition.

---

## 2. Candidate Lifecycle State Machine

Every candidate progresses through an explicit 7-stage lifecycle (`CandidateStatus` enum):

```
[1. GENERATED]
      │  • Canonical ID assigned (MAT-xxxxxx)
      │  • Crystal structure persisted to structures/MAT-xxxxxx.cif
      │  • Fractional coordinates computed and SHA256 checksum calculated
      ▼
[2. SCREENED]
      │  • CHGNet raw energy, force, and stress diagnostics recorded with units
      │  • Multi-objective composite score evaluated
      │  • Geometry/property thresholds applied (raw energy is diagnostic-only)
      ├──► [REJECTED (Screening)] (e.g. max_force_ev_per_angstrom exceeds threshold)
      ▼
[3. VALIDATED]
      │  • High-fidelity DFT / mock relaxation executed on top candidates
      │  • Convergence status & compute hours tracked
      ├──► [REJECTED (Validation)] (e.g. unconverged relaxation after 200 steps)
      ▼
[4. SYNTHESIS_ASSESSED]
      │  • Evaluated on converged candidates (or top screened candidates if validation is disabled)
      │  • Precursor availability & oxidation sanity checked
      │  • Synthesis route & difficulty score calculated
      ├──► [REJECTED (Synthesis)] (e.g. feasibility score 0.25 < 0.40 threshold)
      ▼
[5. RANKED]
      │  • Global multi-objective ranking score and rank assigned
      ▼
[6. ACCEPTED / REJECTED]
      │  • Candidates outside top-k cutoff marked REJECTED (Ranking cutoff)
      │  • Successful candidates marked ACCEPTED (gated by convergence and feasibility)
      ▼
[7. Career Memory Influence (Metadata)]
         • Stored in long-term CareerMemory (`stored_in_memory: bool = True`)
         • Linked with hypothesis guidance (`strategy_influence: str`)
```

---

## 3. Schemas & Specifications

### CandidateRecord Schema (`schema_version: "2.0.0"`)

| Category | Field | Type | Description |
| :--- | :--- | :--- | :--- |
| **Metadata** | `schema_version` | `string` | Version of the schema (`"2.0.0"`). |
| | `scientific_validity` | `string` | `demo_only`, `research_valid`, or `legacy_invalid_energy_semantics`. |
| | `run_mode` | `string` | `development` or fail-closed `research`. |
| **Identity** | `candidate_id` | `string` | Canonical candidate ID (e.g. `"MAT-000137"`). |
| | `campaign_id` | `string` | ID of the discovery campaign. |
| | `iteration` | `integer` | Iteration number (0-indexed). |
| | `created_at_iso` | `string` | UTC timestamp in ISO 8601 format. |
| **Chemistry** | `composition` | `string` | Reduced chemical formula (e.g. `"Li3PS4"`). |
| | `chemical_system` | `string` | Alphabetical element string (e.g. `"Li-P-S"`). |
| | `elements` | `list[str]` | List of constituent element symbols. |
| | `num_elements` | `integer` | Number of distinct elements. |
| | `structure_path` | `string` | Relative path to CIF file (`"structures/MAT-000137.cif"`). |
| | `structure_hash` | `string` | SHA256 cryptographic checksum of the CIF file. |
| **Generation**| `generation_backend` | `string` | Backend used: `"mattergen"`, `"pymatgen_mock"`, `"stub"`. |
| | `model_name_or_path`| `string?` | MatterGen pretrained name or checkpoint directory. |
| | `checkpoint` | `string?` | Model checkpoint identifier. |
| | `generation_seed` | `integer` | Random seed used for generation. |
| | `target_elements` | `list[str]` | Target element subspace requested. |
| **Screening** | `screening_backend` | `string` | Predictor used (`"chgnet"` or `"heuristic"`). |
| | `screening_predictions` | `dict` | Raw diagnostics: `predicted_energy_per_atom_ev`, `max_force_ev_per_angstrom`, `max_stress_gpa`. |
| | `screening_score` | `float` | Multi-objective composite score (0-100). |
| | `passes_screening_filters` | `bool` | Whether candidate satisfied all screening criteria. |
| | `screening_filter_reasons` | `list[str]` | Detailed filter outcomes / rejection thresholds. |
| | `screening_rank` | `integer` | Rank within screening batch. |
| **Validation**| `validation_calculator` | `string` | Calculator backend (`"vasp"`, `"gpaw"`, `"ase"`, `"mock"`). |
| | `validation_converged` | `bool` | Whether electronic/ionic relaxation converged. |
| | `validation_properties` | `dict` | Validated properties (`energy_per_atom_ev`, unit-bearing force/stress keys, `band_gap`, `bulk_modulus`). |
| | `validation_cost_hours` | `float` | Computational cost in CPU/GPU core hours. |
| | `validation_error_message` | `string?` | Error description if relaxation failed. |
| **Synthesis** | `synthesis_mode` | `string` | Synthesis agent mode (`"mock"` or `"mp"`). |
| | `synthesis_feasible` | `bool` | High-level feasibility determination. |
| | `synthesis_feasibility_score` | `float` | Realizability score (0.0 to 1.0). |
| | `synthesis_difficulty_score` | `float` | Thermodynamic / precursor difficulty (0.0 to 1.0). |
| | `synthesis_estimated_cost` | `float` | Estimated experimental cost index (1 to 10). |
| | `synthesis_route` | `string` | Recommended route (`"solid_state"`, `"sol_gel"`, `"hydrothermal"`). |
| | `synthesis_route_reason` | `string` | Chemical justification for selected route. |
| **Decision** | `status` | `string` | Final status: `"accepted"` or `"rejected"`. |
| | `rejection_stage` | `string?` | Stage where candidate was rejected (`"screening"`, `"validation"`, `"synthesis"`, `"ranking"`). |
| | `rejection_reason` | `string?` | Explicit quantitative rationale for rejection. |
| | `ranking_score` | `float?` | Final multi-objective ranking score. |
| | `stored_in_memory` | `bool` | Whether recorded in long-term CareerMemory. |
| | `strategy_influence` | `string?` | Hypothesis / strategy guidance impacted by candidate. |

---

### RunManifest Schema (`manifest.json`)

Frozen at campaign initialization, dynamically tracks per-iteration strategies, and finalizes upon completion:
```json
{
  "schema_version": "2.0.0",
  "scientific_validity": "demo_only",
  "run_mode": "development",
  "campaign_id": "camp_1787446952",
  "campaign_name": "li_solid_electrolyte_campaign",
  "domain": "li_solid_electrolyte",
  "git_commit_sha": "51f209556d40e88551ba63f1cfd578d309909bb2",
  "environment": {
    "python_version": "3.13.1",
    "os_name": "Windows",
    "packages": {
      "pymatgen": "2024.1.1",
      "chgnet": null,
      "torch": null,
      "numpy": "2.3.4"
    }
  },
  "master_seed": 42,
  "iteration_seeds": [42, 43],
  "objective": {
    "band_gap": 2.0
  },
  "constraints": {
    "elements": ["Li", "P", "S", "O", "Cl"],
    "max_atoms": 20
  },
  "strategies": [
    {
      "iteration": 0,
      "elements": ["Li", "P", "S", "O"],
      "num_candidates": 15,
      "screening_criteria": {"max_force_ev_per_angstrom": 100.0},
      "hypothesis": "Li-P-S structures with low residual force will pass screening."
    }
  ],
  "start_time_iso": "2026-08-23T01:02:32.336955+00:00",
  "end_time_iso": "2026-08-23T01:02:32.468832+00:00",
  "elapsed_time_seconds": 0.13,
  "status": "completed",
  "total_candidates_generated": 30,
  "total_candidates_accepted": 6,
  "total_candidates_rejected": 24
}
```

---

## 4. Directory Structure of Campaign Artifacts

```text
campaigns/li_solid_electrolyte/
├── manifest.json              # Run configuration, environment, and replay strategies
├── provenance.jsonl           # Real-time streamed candidate event log
├── campaign_provenance.json   # Consolidated JSON provenance record
├── candidates_provenance.csv  # Flat tabular export for pandas analysis
├── report.json                # Summary metrics derived from ProvenanceTracker
├── report_<campaign_id>.json  # Versioned summary report
└── structures/                # Standardized crystal structures (fractional coordinates)
    ├── MAT-000001.cif
    ├── MAT-000002.cif
    └── MAT-000003.cif
```

---

## 5. Campaign Reproduction Guide

### CLI Reproduction
To replay any past experiment using its saved `manifest.json`:

```bash
# Replay campaign to default reproduced/ directory
python campaign.py --reproduce campaigns/li_solid_electrolyte/manifest.json

# Or specify a custom output directory
python campaign.py --reproduce campaigns/li_solid_electrolyte/manifest.json --output-dir ./replays/run_01
```

### Programmatic Python Reproduction
```python
from pathlib import Path
from campaign import MaterialsDiscoveryCampaign

manifest_file = Path("campaigns/li_solid_electrolyte/manifest.json")
campaign = MaterialsDiscoveryCampaign.reproduce_from_manifest(
    manifest_path=manifest_file,
    output_dir="./replays/exp01"
)
```

---

## 6. Analyzing Provenance with Pandas

The flat export `candidates_provenance.csv` allows quick querying:

```python
import pandas as pd

# Load candidate provenance table
df = pd.read_csv("campaigns/li_solid_electrolyte/candidates_provenance.csv")

# 1. Inspect acceptance / rejection breakdown
print(df["status"].value_counts())

# 2. Query accepted candidates with low residual force
promising = df[
    (df["status"] == "accepted") &
    (df["screening_max_force_ev_per_angstrom"] < 1.5)
][["candidate_id", "composition", "screening_score", "synthesis_route", "structure_path"]]
print(promising.head())

# 3. Analyze failure modes across stages
failures = df[df["status"] == "rejected"][["candidate_id", "rejection_stage", "rejection_reason"]]
print(failures.head(10))
```

---

## 7. Determinism Boundaries

* **Mock / CPU Replay Pipelines**: With seeds recorded in `manifest.iteration_seeds` and strategies replayed from `manifest.strategies`, CPU/mock execution guarantees **strict bitwise equality** of candidate IDs, chemical compositions, and scores across separate runs.
* **GPU Diffusion Sampling (MatterGen / CHGNet)**: CUDA kernels and GPU non-determinism can cause slight floating-point differences. Verification across GPU environments relies on numerical tolerances (`np.isclose(atol=1e-3)`).
