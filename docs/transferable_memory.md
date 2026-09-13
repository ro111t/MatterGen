# Structure-aware CareerMemory (schema 2)

CareerMemory keeps legacy formula/score rows for audit, but only explicit
schema-v2 `transferable_records` are eligible as research evidence.  Each
record stores an anonymous stoichiometric pattern, optional prototype and
local coordination descriptors, symmetry, volume/density/packing values,
oxidation-state balance, broad periodic classes, normalized electronegativity
and radius statistics, and a thermodynamic outcome when a certified oracle
provided one.  An unavailable value is represented by `null` together with a
reason code; it is never replaced with a guessed value.

Transfer is fail-closed.  A record names its permitted relationship
(`same_system`, explicit chalcogen/alkali or isoelectronic substitution,
`homologous_series`, or `element_class_mapping`), descriptor ranges,
minimum evidence, confidence/uncertainty, and observed outcome direction.
Formula and source IDs remain provenance only.  For example, a campaign may
explicitly test Li-P-S -> Li-P-Se -> Na-P-S as a homologous sequence, while an
unrelated system is rejected unless an explicit element-class mapping is
provided.  No relationship implies a success claim.

Applicable records become bounded directives with citations to evidence,
principle, and campaign IDs.  Directives may suggest anonymous stoichiometries,
prototypes, coordination motifs, volume ranges, substitutions, or
exploration/exploitation weights.  They are planner/generator/screener hints;
they cannot accept a candidate or bypass the Sprint 2 geometry or Sprint 3
thermodynamic gates.  Manifest and provenance logs retain both applied and
rejected directives with applicability reasons.

The memory view is controlled by `memory_mode` and `memory_seed` and is saved
in the manifest/config for replay:

* `none`: no memory is exposed.
* `text_summary`: deterministic text context only.
* `structured_provenance`: typed records and executable bounded directives.
* `shuffled_control`: seeded derangement preserving record count and record
  marginals while breaking query pairing. It is labelled invalid for
  scientific decision support and exists only as an experimental control.

Experiment specifications expose `allow_llm_orchestration` (default `true`).
Set it to `false` for offline runs with heuristic planning. When enabled, the
planner uses `OPENAI_API_KEY` if available; missing credentials or API failures
fall back to heuristic planning. Text summaries enter the LLM prompt only and
do not produce candidate-prioritization directives. Their effect on discovery
has not been demonstrated under the benchmark's fixed budgets and locked
chemical systems; treat this mode as experimental, not a validated baseline.

Only finalized prior campaigns can warm-start a new run. Evidence is
deduplicated by a canonical descriptor/outcome fingerprint, so duplicate rows
do not inflate counts or confidence. Negative outcomes and failed transfer
checks are retained for audit, but negative evidence cannot become a positive
generation directive. This limits transfer leakage and makes the scientific
scope of raw formulas clear: formulas identify an observation, while the
descriptors are what can be compared across chemistry.

### Candidate Prioritization & Pre-Oracle Ordering

Applicable directives provide bounded pre-oracle prioritization hints. Candidates
are scored and reordered before scarce oracle evaluation without altering proposal
budgets, geometry validation thresholds, thermodynamic criteria, or single-slot
oracle accounting. Extraction handles arbitrary nested structures and converts
NumPy arrays/coordinates deterministically. All prioritization scores, directive
audits, derangement permutations, and extraction failures are serialized in
campaign checkpoints and manifests for exact end-to-end replay.
