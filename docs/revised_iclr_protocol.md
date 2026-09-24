# Frozen common-proposal ICLR execution

Research uses ExperimentSpec schema `3.0.0`, analysis version `2.0.0`, and
`iclr_common_proposals_v1`. Existing research artifacts are not migrated or
resume-compatible. Development retains the legacy campaign path.

## Generation and replay

An explicit `proposal_stream` DAG node initializes one proposal artifact per task and paired seed under
`proposal_streams/<task>/<seed>.json`, with a separate SHA-256 seal. Generation
uses the pinned checkpoint and unchanged sampling configuration, iteration
seeds, proposal count and batch boundaries. The common MatterGen element mask
allows subsets of the requested species. Research composition targets are
empty. Memory and target oracle results never enter generation.

The create-only artifact records ordered candidate IDs, batch/global indices,
lossless hexadecimal lattice/coordinate values, species, descriptors, structure
and batch hashes, generator/protocol code hashes, and generation environment.
All five target conditions replay these bytes. Existing artifacts are verified,
never regenerated to implement a condition. Interrupted generation leaves an
exclusive lock and fails closed rather than silently retrying. A durable intent,
create-only `proposal_bindings/<task>/<seed>.json`, and independent hash seal
bind the stream before conditions run. Condition execution and hydration are
replay-only: a missing initialized artifact is an error, never a generation
request. Each run, aggregation, statistics and report validates the revised
identity and equality of all five sibling stream/binding hashes.

## Descriptor, response and ranking

`anonymous_integer_ratio_v1` counts each species in a fully occupied ordered
structure, divides counts by their greatest common divisor, and sorts the
resulting positive integers. Species labels, geometry and absolute cell size
are not ranking features. The descriptor is computed before relaxation.

For an exact descriptor match, let `(nL,sL)` be eligible previous target counts
and `(nH,sH)` be frozen historical counts. `beta_exact_match_v1` ranks descending:

```
(1 + sL + (sH/nH if nH > 0 else 0)) / (2 + nL + (1 if nH > 0 else 0))
```

The implementation compares exact rational numbers. Proposal index breaks ties.
Historical evidence has total effective weight one per matched descriptor;
missing evidence contributes zero. All ranking is frozen before a batch starts;
local observations are committed only after every proposal in that batch is
processed. No oracle result is an input to the current batch's rank operation.

An observation is eligible only after a charged, geometrically valid, certified
research thermodynamic evaluation with matching frozen model, relaxation and
reference identities and a finite result. Its response is exactly
`int(predicted_E_hull <= 0.10)` in its own campaign. Failed and unevaluated results
remain in the eligibility audit with `y=null`; they are not negative examples.
Source energies are not interpreted as target energies.

## Conditions

| Condition | Ordering/evidence |
| --- | --- |
| `random_mattergen` | Proposal order; no evidence-based ranking |
| `adaptive_no_memory` | Ranker with completed target batches only |
| `structured_provenance_memory` | Same ranker, plus correctly paired frozen source observations |
| `shuffled_memory_control` | Same ranker, with frozen source response permutation |
| `text_summary_memory` | Same ranker, with historical counts parsed only from frozen text |

Source acquisition uses proposal order and is recorded separately. Every target
arm depends on the same validated source artifact/control preflight, including
arms that do not receive historical ranking evidence.

Source observations are ordered by SHA-256 of canonical JSON
`["source-shuffle-v1", seed, observation_id]`, breaking hash ties by ID. Each
ordered descriptor receives the next observation's response, cyclically. The
mapping has zero fixed points and preserves descriptor and response marginals.
Fewer than two observations or unchanged descriptor-level `(n,s)` statistics
blocks the paired experiment. There is no search for a more effective shuffle.
Changed sufficient statistics need not change ordering or improve performance.

The text artifact has this exact header and numerically sorted descriptor rows:

```
Anonymous-stoichiometry source evidence v1
Outcome: predicted E_hull <= 0.10 eV/atom in the source campaign.
Descriptor [1,3,4]: evaluated=7; successful=5.
```

The strict parser rejects unknown text, duplicate/noncanonical descriptors,
invalid counts and noncanonical whitespace. No LLM or structured fallback is
used by the text ranker. Structured and faithful text evidence are mathematically
equivalent; this is a representation-equivalence control, not a superiority
comparison. Text is absent from efficacy contrasts and the Holm family.
The confirmatory family is target tasks × two controls (`adaptive_no_memory`,
`shuffled_memory_control`) × two endpoints (paid calls to first threshold success,
threshold yield). Random remains exploratory. The existing eight-seed campaign
is not reduced; configured seed lists and the repository's default list remain
unchanged. Text trajectory inequality is an integrity failure, not an efficacy
result.

Common source freezing verifies the complete acquisition and publishes a hashed
receipt binding source corpus, text, source cost and shuffle validity. Text
execution reads that receipt and its text bytes only; it does not reopen source
observations or shuffled records. Controls with no historical evidence read no
source observations. Full-closure verification at freeze/merge checks all files.

## Budget and recovery

The target oracle cap remains 100. Existing geometry gates and charging of
admitted requests (including cache hits) remain authoritative. Invalid geometry
is free; exhausted-budget proposals are unevaluated. Execution does not stop at
first success. The existing charged-call primary endpoint is unchanged.

A create-only hash-chained journal records frozen rankings, pre-request budget,
results, post-request budget and completed-batch evidence snapshots. Durable
results are replayed on resume without repeating calls. An interrupted in-flight
request has uncertain cost and blocks automatic resume. Finalized artifacts
that fail integrity checks cannot be repaired by rerunning. Completion binds
run spec, stream, source artifacts, journal tail, provenance and report hashes.
Publication fsyncs file contents and the containing directory before returning
and before an oracle may start. Tests verify syscall ordering and process-crash
handling; they do not simulate arbitrary hardware failure.

Every initialization, execution and hydration checks the current clean Git
commit/tree against its parent experiment authorization. No hand-maintained
scientific-source allowlist is used. Baseline strategy provenance is fixed;
non-memory selection audits are separate from historical-memory prioritization.

Shard closure includes runs/journals, references, proposal streams, bindings,
seals/intents, source corpus (including shuffle mapping), text and verification
receipts. Scientific identities exclude deployment locations. Artifact locators
are root-relative; merge verifies every source closure before copying and the
complete destination afterward. Conflicts fail closed; missing artifacts are
never regenerated during merge or hydration.

CPU acceptance tests substitute expensive generation/evaluation while retaining
the verified research boundary. They do not establish GPU numerical determinism.
A new pinned schema-v3 spec, clean matching research code identity, complete
source acquisition with a nondegenerate frozen shuffle, and a real-GPU canary
are required before production. Existing RunPod outputs must remain untouched.
