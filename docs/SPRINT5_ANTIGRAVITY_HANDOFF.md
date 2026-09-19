# Sprint 5 Antigravity implementation contract

## 0. Read this first

Sprint 5 is the experiment, validation, and paper-readiness layer. It must not
repair or reinterpret scientific behavior from Sprints 1-4. It consumes those
layers through their public contracts and fails closed when a prerequisite is
absent.

At the time this handoff was written, Sprints 1-3 were accepted and committed,
while Sprint 4 was paused during its final correction/review loop. Therefore:

1. Do not start Sprint 5 implementation until the Lead Architect marks Sprint
   4 accepted and supplies its commit SHA.
2. Work on the existing isolated branch, or on a new branch forked from the
   accepted Sprint 4 commit. Never work from `main` and never merge or push
   unless explicitly authorized.
3. Record the exact base commit and refuse to run research experiments if the
   working tree contains unrecorded code changes.
4. Do not modify Sprint 1-4 scientific semantics merely to make a benchmark
   pass. Report a prerequisite failure instead.

## 1. Sprint objective

Build a reproducible benchmark and audit package capable of testing the paper's
central claim:

> Does structured, provenance-grounded, cross-campaign materials memory improve
> sample-efficient discovery in explicitly related chemical systems compared
> with matched no-memory, text-memory, and shuffled-memory controls?

The benchmark must support the homologous sequence:

1. source campaign: Li-P-S;
2. target campaign: Li-P-Se;
3. target campaign: Na-P-S.

These systems are an explicitly declared experimental relationship, not an
automatically inferred claim of chemical transferability.

Sprint 5 delivers infrastructure, frozen experiment specifications, analysis,
tests, and a Quantum ESPRESSO audit workflow. It does not fabricate benchmark
results, imply synthesizability, or claim DFT validation when QE calculations
have not actually completed.

## 2. Dependencies that must be verified before implementation

Create a machine-readable preflight report. Every research run must abort if
any required item fails.

### Sprint 1: scientific integrity

- Research mode is fail-closed.
- No mock, heuristic, or silent backend fallback is permitted.
- Only schema-v2 records are eligible for scientific conclusions.
- Legacy schema-v1 records remain available for audit but are quarantined.
- Every run records requested and actual backends.

### Sprint 2: geometry and resource accounting

- Every generated object consumes one proposal slot, including invalid
  geometries and backend failures.
- The periodic minimum-distance gate runs before the scientific oracle.
- Invalid geometry is recorded as `INVALID_GEOMETRY` and consumes no oracle
  compute.
- Every geometrically valid oracle admission consumes exactly one oracle slot,
  including cached requests and failed oracle calls.
- Proposal and oracle counters must never be reset between iterations of a run.

### Sprint 3: thermodynamics

- Each chemical system has a local, immutable, certified frozen reference set.
- Reference and candidate energies use the identical pinned CHGNet checkpoint
  and relaxation settings.
- Cell relaxation is enabled, `fmax <= 0.05 eV/Angstrom`, and
  `max_steps <= 500`.
- Research metrics use `predicted_energy_above_hull_ev_per_atom`, true predicted
  formation energy, signed hull delta, and decomposition products from the
  reference-only phase diagram.
- Raw CHGNet energy per atom is diagnostic only and must never be called
  formation energy or stability.
- Reference-set identity, SHA256, certification, model identity, actual model
  state hash, and relaxation settings are present in every relevant record.

### Sprint 4: transferable memory

- Only finalized campaigns that ended before the target campaign began are
  eligible.
- Transfer requires an explicit relationship/target declaration; unrelated
  chemistry fails closed.
- Negative evidence remains queryable but cannot become a positive directive.
- Exact duplicate evidence does not inflate confidence or evidence count.
- `none`, `text_summary`, `structured_provenance`, and `shuffled_control` are
  behaviorally distinct.
- The shuffled control performs a seeded descriptor-response derangement, not
  a harmless list reorder, and is marked invalid for scientific decisions.
- Structured and shuffled directives enter the same bounded policy mechanism.
- The policy may prioritize candidates before scarce oracle admission, but it
  cannot alter proposal counts, geometry results, hull results, thresholds, or
  one-slot oracle accounting.
- Applied, rejected, unsupported, shuffle, extraction-failure, and candidate
  priority audits survive checkpoint and manifest replay.

## 3. Required module boundaries

Use isolated modules with no circular dependency on campaign internals. Names
may vary, but responsibilities must remain separate.

### `experiments/spec.py`

Typed, validated experiment specification:

- experiment ID and schema version;
- immutable code/base commit;
- task sequence and explicit transfer declarations;
- five experimental conditions;
- seed list;
- proposal and oracle budgets;
- generation, geometry, CHGNet, reference-set, and memory configuration;
- frozen memory snapshot identity where applicable;
- output root and artifact hashes;
- QE audit configuration;
- analysis version.

Unknown keys, invalid condition combinations, missing research budgets, and
unresolved paths must fail closed.

### `experiments/dag.py`

Build and validate the global experiment dependency graph. A node may execute
only when all parent artifacts are successful and hash-verified. The graph must
include:

1. environment and code preflight;
2. frozen reference-set verification for all three systems;
3. neutral source-memory construction per seed;
4. immutable source-memory snapshot and hash;
5. target campaign runs for every condition/seed/task;
6. aggregation and statistical analysis;
7. blinded QE audit candidate selection;
8. QE calculations for candidates and required decomposition phases;
9. final tables, figures, claim matrix, and reproducibility bundle.

Reject cycles, missing parents, duplicated run IDs, and artifact reuse across
incompatible configurations.

### `experiments/runner.py`

Execute one fully resolved run specification. It must:

- use the public campaign API;
- never query Materials Project or any network service at runtime;
- create a deterministic run ID from the canonical spec;
- write atomically to an isolated run directory;
- support safe resume from verified checkpoints;
- never double-count proposals/oracle calls after resume;
- skip an already completed run only after verifying its manifest and artifact
  hashes;
- record terminal states such as `SUCCESS`, `FAILED_PREFLIGHT`,
  `FAILED_GENERATION`, `FAILED_ORACLE`, and `INTERRUPTED`;
- preserve complete failure records rather than silently dropping failed runs.

### `experiments/memory_snapshots.py`

Create read-only, content-addressed CareerMemory snapshots. Text, structured,
and shuffled arms for a paired seed must begin from the identical source-memory
snapshot. Snapshot metadata must include:

- source campaign IDs and completion times;
- record IDs and evidence hashes;
- schema version;
- SQLite file hash;
- canonical export hash;
- source task and seed;
- explicit allowed transfer declarations.

No target-run record may be written back into the frozen source snapshot.
Target outputs go to a separate database or overlay.

### `experiments/metrics.py`

Compute metrics only from manifests/provenance, never from log text. Every
metric must include an explicit denominator and missingness count.

### `experiments/statistics.py`

Perform paired, seed-aware comparisons and emit machine-readable results.
Statistical code must be deterministic and tested against hand-calculated toy
examples.

### `experiments/qe_audit.py`

Prepare, run, resume, and summarize the DFT audit. Calculator construction,
pseudopotential resolution, input generation, execution, parsing, and local
decomposition-margin analysis must be independently testable.

### `experiments/report.py`

Generate publication-ready tables and figures from frozen result artifacts.
It must never rerun campaigns or mutate raw results.

## 4. Experimental design

### 4.1 Five required conditions

Use these stable machine names:

1. `random_mattergen`
   - Fixed chemical-system-conditioned MatterGen sampling.
   - No LLM adaptation, campaign-history adaptation, CareerMemory, or memory
     prioritization.
   - This is not equivalent to `memory_mode=none`; implement it as a genuinely
     fixed proposal policy.

2. `adaptive_no_memory`
   - The same adaptive planner/campaign-history mechanism used by the proposed
     method.
   - No CareerMemory principles, summaries, structured records, or directives.

3. `text_summary_memory`
   - Uses the identical frozen source-memory snapshot as conditions 4 and 5.
   - Exposes deterministic text context only.
   - Text must not be parsed back into executable structured directives.

4. `structured_provenance_memory`
   - Proposed method.
   - Uses applicable, cited, schema-v2 structured records and the bounded
     candidate-priority policy.

5. `shuffled_memory_control`
   - Uses the same snapshot and the same bounded policy channel as condition 4.
   - Descriptor-response associations are deterministically deranged with the
     configured seed while corpus marginals and count are preserved.
   - Must be visibly labelled invalid for scientific decision support.
   - If fewer than two eligible records exist, the run fails preflight as an
     invalid control; it must not silently behave like another arm.

### 4.2 Source-memory construction

To isolate memory representation from different source trajectories:

- For each master seed, run one neutral Li-P-S source campaign with a fixed,
  preregistered policy.
- Finalize and freeze its memory once.
- Clone that exact snapshot for the text, structured, and shuffled target arms.
- The no-memory arms receive no access to the snapshot.
- Do not independently regenerate source memory for each target condition;
  doing so would confound representation with source-data quality.

An optional secondary multi-hop experiment may study Li-P-S -> Li-P-Se ->
Na-P-S with evolving memory. Keep it separate from the primary causal ablation
and label it exploratory.

### 4.3 Pairing and seeds

- Use common random numbers: a given master seed maps to identical generation
  and iteration seed schedules across conditions wherever the condition has not
  deliberately changed ordering.
- Minimum pilot: 5 paired seeds, matching the current compute estimate.
- Before final claims, perform a documented power/precision assessment from
  pilot variance. Increase the final seed count if five seeds cannot provide
  useful confidence intervals. Never choose the final seed count based on which
  condition wins.
- Store the complete seed schedule in the top-level experiment manifest.

### 4.4 Budgets

The initial target is 200 proposals per target campaign per condition and seed.
Define an explicit oracle budget separately. Do not describe 200 proposals as
200 oracle evaluations.

For the initial 5-condition x 5-seed design:

- target runs per target system: 25;
- proposals per target system at 200 each: 5,000;
- two target systems: 10,000 proposals, plus neutral source campaigns;
- actual CHGNet oracle calls depend on geometry yield and the explicit oracle
  budget.

Do not repeat the earlier unsupported estimate of 1.5 seconds per full CHGNet
relaxation as a guarantee. Add a pre-registered calibration run on the actual
GPU and report median, interquartile range, and tail latency before scheduling
the full experiment.

## 5. Frozen experiment inputs

Create a version-controlled experiment specification directory containing:

- Li-P-S, Li-P-Se, and Na-P-S task definitions;
- explicit source/target transfer declarations;
- paths and SHA256 hashes for all three certified reference sets;
- pinned MatterGen checkpoint identity/hash;
- pinned CHGNet checkpoint and actual state hash;
- relaxation settings;
- proposal/oracle budgets;
- geometry threshold;
- hull retain/stable thresholds;
- seeds;
- condition definitions;
- software environment lock/hash;
- QE settings and SSSP manifest.

The benchmark must consume local paths only. Reference-set construction and
external data retrieval are offline prerequisites, not hidden runtime steps.

## 6. Required metrics

### 6.1 Primary metrics

Pre-register one primary metric before running the full benchmark. Recommended:

- `oracle_calls_to_first_candidate_at_or_below_0.10_ev_per_atom`, right-censored
  when no candidate reaches the threshold.

Also report:

- fraction of proposals reaching the oracle;
- fraction of oracle-evaluated candidates at or below 0.10 eV/atom;
- best predicted energy above hull at fixed oracle budgets;
- area under the best-so-far-versus-oracle-calls curve.

### 6.2 Secondary metrics

- invalid-geometry fraction and reason distribution;
- oracle failure fraction and failure codes;
- thresholds at 0.00, 0.03, 0.05, and 0.10 eV/atom;
- unique reduced compositions among evaluated candidates;
- unique anonymous stoichiometries and explicitly available prototypes/motifs;
- duplicate/cached request fraction;
- decomposition-product distribution;
- wall time and GPU time, reported separately from oracle count;
- memory records eligible/applied/rejected/unsupported;
- candidate-priority score distribution and how often priority changed oracle
  admission under a finite budget;
- negative-transfer indicators: structured memory worse than adaptive no-memory
  on the paired primary metric, reported without hiding unfavorable seeds.

Never use synthesis-agent heuristic scores as evidence of synthesizability.

## 7. Statistical protocol

- Treat seed as the paired experimental unit.
- Report every seed-level point, not only means.
- Report paired effect sizes with 95% bootstrap confidence intervals using a
  fixed bootstrap seed and a documented resample count.
- For time-to-threshold with censoring, report success probability at the fixed
  budget and an appropriate paired/censored analysis; do not replace failures
  with an arbitrary large number without a sensitivity analysis.
- Correct the predeclared family of confirmatory comparisons for multiplicity.
  At minimum, the proposed method must be compared with adaptive no-memory,
  text summary, and shuffled control.
- Label all other comparisons exploratory.
- Include threshold-sensitivity results at the four hull thresholds above.
- Do not claim improvement from overlapping confidence intervals alone and do
  not equate statistical significance with chemical importance.
- Emit a tidy CSV/Parquet table containing effect, interval, test, adjusted
  p-value when applicable, sample size, missing count, and analysis version.

## 8. Quantum ESPRESSO local decomposition audit

### 8.1 Scope

Audit 10 candidates total. Selection must be deterministic and blinded to QE
outcomes. Pre-register the selection rule, for example:

- candidates spanning both target systems;
- a mix near and below the CHGNet retention threshold;
- composition/structure diversity;
- balanced representation of the proposed method and matched controls;
- no manual replacement after seeing QE results, except a documented technical
  failure replacement rule chosen in advance.

### 8.2 What must be calculated

For each candidate, use the Sprint 3 decomposition products to define a local
reaction. Run QE for:

- the candidate; and
- every required competing phase in its predicted decomposition, unless an
  identical converged calculation with the exact same settings/hash already
  exists.

Compute the DFT local decomposition margin per atom from consistent QE total
energies. A candidate-only relaxation is not a thermodynamic audit.

### 8.3 Fixed settings

- ASE Quantum ESPRESSO interface;
- SSSP pseudopotentials resolved from a local immutable manifest;
- wavefunction cutoff: 60 Ry unless the pinned SSSP recommendation is higher;
- charge-density cutoff: explicit, pseudopotential-compatible, and recorded;
- reciprocal-space spacing target: 0.25 inverse Angstrom or denser;
- cell and ionic relaxation policy stated explicitly;
- force/stress/SCF convergence thresholds stated explicitly;
- smearing, occupations, spin initialization, and metallicity policy stated
  explicitly;
- QE version, executable hash/path, MPI invocation, pseudopotential filenames
  and SHA256 hashes recorded;
- no silent retry with changed scientific settings.

If the study keeps 60 Ry and 0.25 inverse Angstrom fixed for feasibility, run a
small convergence sensitivity check on representative phases and report it as
a limitation. Never call an unconverged calculation successful.

### 8.4 QE statuses and resume

Use explicit states such as `PREPARED`, `RUNNING`, `SCF_FAILED`,
`RELAXATION_NOT_CONVERGED`, `CONVERGED`, and `PARSE_FAILED`. Inputs and outputs
must be content-addressed. Resume must not overwrite a completed calculation or
reuse an output generated with incompatible settings.

### 8.5 QE outputs

- candidate and phase energies;
- convergence evidence;
- local reaction/decomposition coefficients;
- local DFT decomposition margin in eV/atom;
- CHGNet-versus-DFT sign agreement and magnitude difference;
- failure table including every selected candidate;
- no claim of a complete DFT convex hull unless a complete consistent DFT
  reference set was actually constructed.

Unit tests must use injected fake calculators/parsers. Real QE is an opt-in
integration test and must never run in the normal unit suite.

## 9. Reproducibility and artifact layout

Every experiment root should contain, at minimum:

```text
experiment_manifest.json
preflight.json
specs/
reference_sets/
memory_snapshots/
runs/<condition>/<task>/<seed>/
aggregates/candidates.parquet
aggregates/runs.parquet
statistics/effects.csv
statistics/analysis_manifest.json
qe_audit/selection.json
qe_audit/calculations/
qe_audit/results.csv
figures/
tables/
paper/claim_evidence_matrix.md
paper/reproducibility_checklist.md
artifact_hashes.sha256
```

The top-level manifest must include code commit, dirty-tree status, environment,
all input hashes, run IDs, dependency edges, expected run count, terminal run
counts, and analysis/QE versions.

## 10. Paper-facing artifacts

Generate scripts, not hand-edited plots, for:

1. best-so-far hull energy versus oracle calls;
2. time/oracle-calls to threshold with censoring shown;
3. paired seed-level effect plots for the four confirmatory comparisons;
4. geometry and oracle failure breakdown;
5. memory applicability/applied/rejected/unsupported breakdown;
6. shuffled-control validation panel showing preserved marginals and broken
   pairings;
7. CHGNet versus QE local decomposition margins;
8. threshold-sensitivity table/figure.

Create a claim-evidence matrix with columns:

- proposed paper claim;
- required metric/figure/table;
- supporting artifact path;
- validity assumptions;
- known limitation;
- status: supported, unsupported, or not yet tested.

Claims must remain conservative. Preferred language is "predicted energy above
hull under a pinned CHGNet reference framework" and "local QE decomposition
audit," not "stable," "synthesizable," or "experimentally validated" without
the corresponding evidence.

## 11. Required tests

Use a dedicated Sprint 5 test suite plus the complete repository suite.

### Specification and DAG

- canonical spec hashing is deterministic;
- unknown/invalid configurations fail closed;
- all expected nodes/edges exist;
- cycles, duplicate run IDs, incompatible artifact reuse, and missing parents
  are rejected;
- expected run counts are exact.

### Condition isolation

- random baseline is fixed and nonadaptive;
- adaptive no-memory receives no CareerMemory signal;
- text receives text only;
- structured receives applicable structured directives;
- shuffled uses the same snapshot/policy channel with deranged pairings;
- source snapshot hash is identical across the three memory arms;
- no target-to-source or future-to-past leakage is possible.

### Budgets and resume

- invalid proposals count against proposals, not oracle calls;
- every valid oracle admission consumes one slot;
- all arms preserve the same configured budgets and gates;
- prioritization can change admission order but not counts or thresholds;
- interrupted/resumed runs exactly match uninterrupted counters and results;
- completed runs are reused only after hash verification;
- corrupted checkpoints/artifacts fail closed.

### Metrics/statistics

- hand-calculated toy trajectories produce exact metrics;
- denominators and missingness are correct;
- censoring is preserved;
- paired resampling is deterministic;
- multiplicity adjustment is correct;
- threshold sensitivity is correct;
- all failed seeds remain in outputs.

### QE

- SSSP lookup and hashes are deterministic;
- k-point mesh generation meets the spacing requirement;
- cutoffs and convergence settings appear in every input/manifest;
- candidate and all decomposition phases are required;
- local decomposition margins match hand calculations;
- incompatible cached calculations are rejected;
- failure and resume states are explicit;
- normal tests never invoke a real QE executable.

### Reporting

- figures/tables derive only from frozen aggregate artifacts;
- rerunning report generation is deterministic;
- claim-evidence matrix never marks a claim supported when required artifacts
  are missing or contain failed preflight/analysis states.

## 12. Acceptance gates

Sprint 5 is accepted only if all gates pass:

1. Sprint 4 base commit is recorded and accepted.
2. `git diff --check` is clean.
3. Dedicated Sprint 5 tests pass in the pinned MatterGen environment.
4. The complete repository test suite passes with only documented optional
   skips.
5. A tiny injected-backend end-to-end experiment executes all five conditions,
   aggregation, statistics, report generation, and resume without network or
   GPU access.
6. Research-mode preflight demonstrably rejects mock/fallback backends,
   uncertified/mismatched references, dirty code, missing budgets, invalid
   shuffled controls, and unpinned inputs.
7. Source-memory snapshots are byte/hash identical across the three memory
   conditions and remain immutable.
8. Experiment manifests prove exact run counts, seed pairing, budgets, backend
   identity, and artifact hashes.
9. The QE workflow passes injected-calculator tests and an opt-in environment
   preflight. Do not require actual QE completion for code acceptance, but do
   not produce a scientific audit result until real calculations converge.
10. All paper figures/tables are reproducible from frozen aggregate files.
11. Documentation includes setup, dry-run, full-run, resume, aggregation, QE,
    and report commands.
12. No benchmark result or paper claim is invented, manually patched, or
    silently excluded.

## 13. Required delivery report from Antigravity

Return a concise but complete report containing:

- base branch and base commit;
- files added/modified;
- final experiment DAG and run-count calculation;
- exact five-condition semantics;
- source-memory isolation method;
- metrics and preregistered primary endpoint;
- statistical methods;
- QE audit design;
- dedicated and full test commands with exact counts;
- any optional tests skipped and why;
- remaining external prerequisites, especially frozen reference artifacts,
  MatterGen/CHGNet checkpoints, SSSP files, QE executable, and compute;
- known limitations;
- `git status --short` and `git diff --check` output;
- no commit, merge, push, or destructive cleanup unless separately authorized.

## 14. Explicit non-goals

- Do not write the paper manuscript in Sprint 5.
- Do not dynamically query Materials Project during experiments.
- Do not build a global DFT convex hull from only the 10-candidate audit.
- Do not treat CHGNet raw energy as formation energy.
- Do not treat `-abs(energy)` or any heuristic score as stability.
- Do not claim synthesizability from a heuristic synthesis agent.
- Do not let an LLM modify scientific gates, budgets, seeds, or frozen inputs at
  runtime.
- Do not tune the method after examining held-out target outcomes without
  declaring a new experiment version.
- Do not silently remove failed candidates, seeds, campaigns, or QE jobs.

