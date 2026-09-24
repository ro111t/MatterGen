"""Schema-v3 release boundaries shared by execution, merge and analysis."""
from dataclasses import replace
from pathlib import Path
import json
import subprocess

from experiments.selection_protocol import PROTOCOL, CONTRACT, canonical, create_only, digest, file_hash, read_json


def code_identity(expected_commit, output_root=None):
    """Use the complete authorized Git tree, not a curated source-file list."""
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    commit = git('rev-parse', 'HEAD')
    if not expected_commit or commit != expected_commit:
        raise ValueError('Current code commit differs from authorized experiment')
    args = ['status', '--porcelain', '--untracked-files=all', '--', '.']
    if output_root:
        try:
            rel = Path(output_root).resolve().relative_to(root)
            if not rel.parts:
                raise ValueError('Experiment output cannot be repository root')
            if git("ls-files", "--", rel.as_posix()):
                raise RuntimeError("Experiment output overlaps tracked code")
            args.append(':(exclude)' + rel.as_posix())
        except ValueError:
            pass
    if git(*args):
        raise ValueError('Authorized research code tree is dirty')
    return {'commit': commit, 'tree': git('rev-parse', 'HEAD^{tree}')}


def parent_hash(spec):
    return parent_data_hash(spec.to_dict())


def parent_data_hash(value):
    # Deployment paths are locators. Their independently pinned hashes are identity.
    def normalize(item):
        if isinstance(item, dict):
            return {k: normalize(v) for k, v in item.items()
                    if k not in {'output_root', 'reference_set_path', 'mattergen_model_path',
                                 'mattergen_sampling_config_path', 'sssp_manifest_path', 'qe_executable'}}
        if isinstance(item, (list, tuple)):
            return [normalize(v) for v in item]
        return item
    return digest(normalize(value))


def verify_code(spec):
    actual = code_identity(spec.authorized_code_commit, spec.artifact_root)
    if actual != spec.authorized_code_identity or not spec.parent_experiment_hash:
        raise ValueError('Authorized parent/code identity mismatch')
    return actual


def locate(spec, value):
    if not value:
        raise ValueError('Missing revised artifact locator')
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Revised artifact locator must be root-relative')
    return Path(spec.artifact_root) / path


def initialize_pair(spec, factory=None):
    """Only explicit DAG initialization can create a stream. Intent is durable first."""
    from experiments.selection_protocol import generate_stream, proposal_identity
    from experiments.revised_runner import proposal_factory
    verify_code(spec)
    stream = Path('proposal_streams') / spec.task_id / f'{spec.seed}.json'
    binding = Path('proposal_bindings') / spec.task_id / f'{spec.seed}.json'
    identity = proposal_identity(spec)
    target = locate(spec, str(binding))
    intent = target.with_suffix('.intent.json')
    if target.exists():
        return bind_pair(spec)
    if intent.exists() or locate(spec, str(stream)).exists():
        raise ValueError('Initialized/interrupted paired stream cannot be regenerated')
    create_only(intent, canonical({'identity': identity, 'parent': spec.parent_experiment_hash}))
    sha = generate_stream(locate(spec, str(stream)), identity, factory or proposal_factory(spec))
    body = {'protocol': CONTRACT, 'parent': spec.parent_experiment_hash,
            'code': spec.authorized_code_identity, 'identity': identity,
            'stream': str(stream), 'sha256': sha}
    create_only(target, canonical(body))
    create_only(target.with_suffix('.sha256'), file_hash(target).encode())
    return bind_pair(spec)


def bind_pair(spec):
    from experiments.selection_protocol import ProposalReplay, proposal_identity
    verify_code(spec)
    rel = str(Path('proposal_bindings') / spec.task_id / f'{spec.seed}.json')
    path = locate(spec, rel)
    sha = path.with_suffix('.sha256').read_text()
    body = read_json(path, sha)
    if (body['protocol'] != CONTRACT or body['parent'] != spec.parent_experiment_hash
            or body['code'] != spec.authorized_code_identity or body['identity'] != proposal_identity(spec)):
        raise ValueError('Paired proposal binding identity mismatch')
    for supplied, actual in ((spec.proposal_binding_manifest, rel),
                             (spec.proposal_binding_sha256, sha),
                             (spec.proposal_stream_sha256, body['sha256']),
                             (spec.proposal_stream_manifest, body['stream'])):
        if supplied is not None and supplied != actual:
            raise ValueError('Run differs from authoritative paired proposal binding')
    ProposalReplay(locate(spec, body['stream']), body['sha256'], body['identity'])
    return replace(spec, proposal_binding_manifest=rel, proposal_binding_sha256=sha,
                   proposal_stream_manifest=body['stream'], proposal_stream_sha256=body['sha256'])


def validate_rows(rows, *, expected_tasks=None, expected_seeds=None, expected_parent=None, complete=True):
    """Do not let independently valid rows imply a valid paired experiment."""
    from experiments.spec import FIVE_CONDITIONS
    rows = [r.to_dict() if hasattr(r, 'to_dict') else r for r in rows]
    groups = {}
    parents = set()
    for row in rows:
        if row.get('condition') == 'source_neutral':
            continue
        if row.get('selection_protocol') != PROTOCOL:
            raise ValueError('Mixed/legacy research analysis rows')
        for field in ('proposal_stream_sha256', 'proposal_binding_sha256', 'parent_experiment_hash',
                      'selection_trajectory_sha256', 'source_receipt_sha256'):
            value = row.get(field)
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError(f'Missing/invalid revised binding: {field}')
        parents.add(row['parent_experiment_hash'])
        key = row['task_id'], row['seed']
        siblings = groups.setdefault(key, {})
        cond = row['condition']
        if cond not in FIVE_CONDITIONS or cond in siblings:
            raise ValueError('Invalid/duplicate paired condition')
        siblings[cond] = row
    if expected_parent is not None and parents != {expected_parent}:
        raise ValueError('Analysis rows belong to another parent experiment')
    if not groups or len(parents) != 1:
        raise ValueError('Missing or incompatible parent experiment rows')
    if complete and expected_tasks is not None and expected_seeds is not None:
        if set(groups) != {(t, s) for t in expected_tasks for s in expected_seeds}:
            raise ValueError('Incomplete paired task/seed set')
    for siblings in groups.values():
        if complete and set(siblings) != set(FIVE_CONDITIONS):
            raise ValueError('Incomplete five-condition pair')
        for field in ('proposal_stream_sha256', 'proposal_binding_sha256', 'source_receipt_sha256'):
            if len({r[field] for r in siblings.values()}) != 1:
                raise ValueError(f'Unequal sibling {field}')
        if 'structured_provenance_memory' in siblings and 'text_summary_memory' in siblings:
            if siblings['structured_provenance_memory']['selection_trajectory_sha256'] != siblings['text_summary_memory']['selection_trajectory_sha256']:
                raise ValueError('Structured/text trajectory inequality')
    return {'protocol': PROTOCOL, 'paired_groups': len(groups), 'text_equivalence': True}


def verify_research_root(spec, root, seeds):
    """Verify a complete portable research closure without regenerating anything."""
    from experiments.revised_runner import load_run_spec, completed, freeze_source
    from experiments.metrics import compute_run_metrics
    from experiments.spec import FIVE_CONDITIONS
    root = Path(root)
    code_identity(spec.code_commit, root)
    rows = []
    for seed in seeds:
        source = root / 'runs' / 'source' / spec.source_task.task_id / str(seed)
        corpus = root / 'source_evidence' / f'seed{seed}.json'
        for path in (corpus, corpus.with_suffix('.txt'), corpus.with_suffix('.receipt.json')):
            if not path.is_file():
                raise ValueError('Incomplete frozen source closure')
        freeze_source(source, corpus, seed, root=root)
        for task, condition in [(spec.source_task, 'source_neutral')] + [(t, c) for t in spec.target_tasks for c in FIVE_CONDITIONS]:
            directory = root / 'runs' / ('source' if condition == 'source_neutral' else condition) / task.task_id / str(seed)
            run = load_run_spec(directory, root)
            if (run.parent_experiment_hash != parent_hash(spec) or run.authorized_code_commit != spec.code_commit
                    or run.task_id != task.task_id or run.condition != condition or run.seed != seed or not completed(run)):
                raise ValueError('Research closure run identity/completion mismatch')
            reference = Path(run.reference_set_path)
            if (file_hash(reference) != run.reference_set_sha256
                    or reference.with_suffix(reference.suffix + ".sha256").read_text().strip() != run.reference_set_sha256):
                raise ValueError("Frozen reference closure hash mismatch")
            body = json.loads((directory / 'campaign_provenance.json').read_text())
            metric, _ = compute_run_metrics(body, run.run_id, task.task_id, condition, seed, run.oracle_budget)
            rows.append(metric)
    validate_rows(rows, expected_tasks=[t.task_id for t in spec.target_tasks], expected_seeds=seeds, expected_parent=parent_hash(spec))
    return rows
