"""Adversarial release tests; no model inference or production assets."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import os
import stat

import pytest

from experiments.selection_protocol import PROTOCOL, CONTRACT, canonical, file_hash
from experiments.release_integrity import bind_pair, initialize_pair, validate_rows
from experiments.revised_runner import execute, freeze_source, Journal
from tests.test_selection_protocol import revised_fixture


def target_spec(spec, root, condition):
    corpus = root / 'corpus.json'
    sha, text, text_sha = freeze_source(Path(spec.output_dir), corpus, spec.seed)
    return replace(spec, condition=condition, run_id=condition, output_dir=str(root / condition),
        strategy_mode='fixed' if condition == 'random_mattergen' else 'adaptive',
        source_corpus_manifest='corpus.json', source_corpus_sha256=sha,
        source_text_path='corpus.txt', source_text_sha256=text_sha,
        source_receipt_manifest='corpus.receipt.json', source_receipt_sha256=file_hash(root / 'corpus.receipt.json'),
        memory_transfer_declaration={'allowed_relationship':'same_system',
            'source_chemical_system':spec.elements, 'target_chemical_system':spec.elements})


def rows():
    from experiments.spec import FIVE_CONDITIONS
    return [dict(task_id='Li-P-Se', seed=42, condition=c, selection_protocol=PROTOCOL,
        proposal_stream_sha256='a'*64, proposal_binding_sha256='b'*64,
        parent_experiment_hash='c'*64, source_receipt_sha256='d'*64,
        selection_trajectory_sha256='e'*64, provenance_complete=True,
        oracle_budget=100, oracle_evaluations=100, proposals_generated=200,
        geometry_valid_count=200, invalid_geometry_count=0, total_candidates_recorded=200,
        oracle_success_count=100, oracle_failure_count=0, run_status='completed') for c in FIVE_CONDITIONS]


def test_bound_stream_deletion_never_regenerates(revised_fixture):
    spec, _, _ = revised_fixture
    (Path(spec.artifact_root) / spec.proposal_stream_manifest).unlink()
    with pytest.raises((OSError, ValueError)):
        bind_pair(spec)
    with pytest.raises((OSError, ValueError)):
        initialize_pair(spec, lambda: pytest.fail('regenerated bound stream'))
    with pytest.raises((OSError, ValueError)):
        execute(spec)


def test_unequal_run_pin_rejected_before_evaluation(revised_fixture):
    spec, calls, _ = revised_fixture
    with pytest.raises(ValueError, match='binding'):
        execute(replace(spec, proposal_stream_sha256='0'*64))
    assert not calls


@pytest.mark.parametrize('mutation', ['missing_protocol','missing_hash','legacy','unequal_hash','text_inequality'])
def test_analysis_rejects_incompatible_rows(mutation, tmp_path, monkeypatch):
    data = rows()
    if mutation in ('missing_protocol','legacy'):
        data[0]['selection_protocol'] = None
    elif mutation == 'missing_hash':
        del data[0]['proposal_binding_sha256']
    elif mutation == 'unequal_hash':
        data[0]['proposal_stream_sha256'] = 'f'*64
    else:
        next(r for r in data if r['condition']=='text_summary_memory')['selection_trajectory_sha256']='f'*64
    with pytest.raises(ValueError):
        validate_rows(data)
    from experiments.statistics import run_statistical_analysis_pipeline
    with pytest.raises(ValueError):
        run_statistical_analysis_pipeline(data, analysis_version='2.0.0', parent_experiment_hash='c'*64, output_dir=tmp_path / 'statistics')
    assert not (tmp_path / 'statistics').exists()
    from experiments.report import ReportGenerator
    from experiments.spec import FIVE_CONDITIONS
    monkeypatch.setattr(ReportGenerator, '_expected_context', staticmethod(lambda artifacts:
        (['Li-P-Se'],[42],list(FIVE_CONDITIONS),{'spec':{'run_mode':'research'}})))
    assert ReportGenerator._target_runs_complete(data,{})[0] is False


def test_text_execution_cannot_read_structured_evidence(revised_fixture, tmp_path, monkeypatch):
    spec, _, _ = revised_fixture
    execute(spec)
    target = target_spec(spec, tmp_path, 'text_summary_memory')
    original_open = Path.open
    forbidden = {tmp_path / 'corpus.json', Path(spec.output_dir) / 'selection_observations.json'}
    def guarded(path, *args, **kwargs):
        if path in forbidden:
            raise PermissionError('structured evidence deliberately inaccessible')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    execute(target)
    assert execute(target)['skipped']
    text = tmp_path / 'corpus.txt'
    text.chmod(0o644)
    text.write_text('tampered')
    with pytest.raises(ValueError, match='text SHA256'):
        execute(target)


def test_resume_rechecks_code_identity(revised_fixture, monkeypatch):
    spec, calls, _ = revised_fixture
    execute(spec)
    before = len(calls)
    monkeypatch.setattr('experiments.release_integrity.code_identity', lambda *args:
        {'commit':'different', 'tree':'different'})
    with pytest.raises(ValueError, match='identity'):
        execute(spec)
    assert len(calls) == before


def test_stale_preflight_does_not_authorize_changed_code(tmp_path, monkeypatch):
    from experiments.cli import _verify_and_hydrate_completed_node
    from experiments.dag import DAGNode, NodeType
    path = tmp_path / 'preflight.json'
    path.write_text('{"status":"PASSED"}')
    node = DAGNode('preflight', NodeType.PREFLIGHT, 'test', expected_output_path=str(path),result_hash=file_hash(path))
    monkeypatch.setattr('experiments.release_integrity.code_identity',lambda *args: (_ for _ in ()).throw(ValueError('changed code')))
    with pytest.raises(ValueError,match='changed code'):
        _verify_and_hydrate_completed_node(node, SimpleNamespace(run_mode='research',code_commit='old',output_root=str(tmp_path)),
            None, [], [], {}, tmp_path, {})


def test_journal_publication_is_durable_before_oracle(revised_fixture, monkeypatch):
    from experiments.revised_runner import RevisedCampaign
    spec, _, _ = revised_fixture
    events = []
    original_fsync, original_link = os.fsync, os.link
    def fsync(fd):
        events.append('directory' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file')
        return original_fsync(fd)
    def link(*args, **kwargs):
        events.append('publish')
        return original_link(*args, **kwargs)
    monkeypatch.setattr(os,'fsync',fsync)
    monkeypatch.setattr(os,'link',link)
    original_init = RevisedCampaign._init_screener
    def init(campaign):
        screener = original_init(campaign)
        original_screen = screener.screen_batch
        def screen(*args, **kwargs):
            assert campaign.journal.events[-1]['type'] == 'oracle_begin'
            assert events[-3:] == ['file','publish','directory']
            return original_screen(*args,**kwargs)
        screener.screen_batch = screen
        return screener
    monkeypatch.setattr(RevisedCampaign,'_init_screener',init)
    execute(spec)


def test_baseline_and_adaptive_provenance_are_not_memory(revised_fixture,tmp_path):
    from experiments.metrics import compute_run_metrics
    spec, _, _ = revised_fixture
    execute(spec)
    for condition in ['random_mattergen','adaptive_no_memory','structured_provenance_memory']:
        target = target_spec(spec,tmp_path,condition)
        execute(target)
        body = json.loads((Path(target.output_dir)/'campaign_provenance.json').read_text())
        metric,_ = compute_run_metrics(body,target.run_id,target.task_id,condition,target.seed,100)
        assert metric.memory_prioritized_candidates_count > 0 if condition=='structured_provenance_memory' else metric.memory_prioritized_candidates_count == 0
        assert body['manifest']['config']['strategy_mode'] == target.strategy_mode


def test_cli_shard_relocation_closure_statistics_and_report(revised_fixture,tmp_path,monkeypatch):
    """Real CLI orchestration/runner/merge, with only expensive backends substituted."""
    import shutil
    from experiments.spec import ExperimentSpec, TaskDefinition, FIVE_CONDITIONS
    from experiments.cli import execute_full_experiment_pipeline, merge_shard_outputs
    from experiments.release_integrity import verify_research_root, parent_hash
    from experiments.statistics import run_statistical_analysis_pipeline
    from experiments.report import ReportGenerator
    original, _, factory = revised_fixture
    root = tmp_path / 'shard-a'
    second_root = tmp_path / 'shard-b'
    task_args = dict(elements=original.elements,reference_set_path=original.reference_set_path,
        reference_set_sha256=original.reference_set_sha256,reference_set_certified=True)
    exp = ExperimentSpec(experiment_id='release-cli',code_commit=original.authorized_code_commit,
        master_seeds=[42,137],source_task=TaskDefinition(task_id='source',**task_args),
        target_tasks=[TaskDefinition(task_id='Li-P-Se',**task_args)],output_root=str(root),
        proposals_per_run=200,oracle_budget_per_run=100,iterations_per_run=5,
        generation_backend='mattergen',pinned_model_identity=original.pinned_model_identity,
        pinned_relaxation_settings=original.pinned_relaxation_settings,
        mattergen_model_path=original.mattergen_model_path,mattergen_checkpoint_sha256=original.mattergen_checkpoint_sha256,
        mattergen_sampling_config_path=original.mattergen_sampling_config_path,
        mattergen_sampling_config_sha256=original.mattergen_sampling_config_sha256,
        transfer_declarations=[])
    # One-seed CPU wiring fixture; production ExperimentSpec enforces preregistered n.
    object.__setattr__(exp,'run_mode','research')
    from experiments.spec import TransferDeclaration
    object.__setattr__(exp,'transfer_declarations',[TransferDeclaration(
        source_task='source',target_task='Li-P-Se',source_chemical_system=original.elements,
        target_chemical_system=original.elements,allowed_relationship='same_system')])
    monkeypatch.setattr('experiments.revised_runner.proposal_factory',lambda spec: factory)
    def preflight(spec):
        result={'status':'PASSED','spec_hash':spec.spec_hash,'run_mode':'research'}
        (Path(spec.output_root)/'preflight.json').write_bytes(canonical(result))
        return result
    monkeypatch.setattr('experiments.cli.run_preflight_check',preflight)
    monkeypatch.setattr('experiments.cli._runtime_metadata',lambda:{'git_commit':original.authorized_code_commit})
    execute_full_experiment_pipeline(exp,seed_subset=[42],worker_id='cpu')
    second_exp = replace(exp,run_mode='development',output_root=str(second_root))
    object.__setattr__(second_exp,'run_mode','research')
    execute_full_experiment_pipeline(second_exp,seed_subset=[137],worker_id='cpu-b')
    moved=tmp_path/'moved-a'
    moved_second=tmp_path/'moved-b'
    shutil.copytree(root,moved)
    shutil.copytree(second_root,moved_second)
    shutil.rmtree(root)
    shutil.rmtree(second_root)  # receipts retain their original, now stale shard-local paths
    receipt_relative=Path('references/Li-P-Se.verified.json')
    first_receipt=json.loads((moved/receipt_relative).read_text())
    second_receipt=json.loads((moved_second/receipt_relative).read_text())
    assert first_receipt['reference_set_path'] != second_receipt['reference_set_path']
    assert {k:v for k,v in first_receipt.items() if k!='reference_set_path'} == {
        k:v for k,v in second_receipt.items() if k!='reference_set_path'}
    # Original reference input is unavailable: the shard must carry its own pinned bytes.
    Path(original.reference_set_path).unlink()
    merged=tmp_path/'merged'
    merge_shard_outputs(exp,[moved,moved_second],merged)
    merged_receipt=json.loads((merged/receipt_relative).read_text())
    merged_reference=merged/'references/Li-P-Se.frozen.json'
    assert Path(merged_receipt['reference_set_path']).resolve() == merged_reference.resolve()
    assert merged_receipt['sha256'] == file_hash(merged_reference)
    metrics=verify_research_root(exp,merged,[42,137])
    endpoints = {m.condition: m.oracle_calls_to_first_candidate_at_or_below_0_10 for m in metrics}
    assert endpoints['random_mattergen'] == endpoints['adaptive_no_memory'] == 2
    assert endpoints['structured_provenance_memory'] == endpoints['text_summary_memory'] == 1
    assert all((merged/name).is_dir() for name in ['proposal_streams','proposal_bindings','source_evidence','runs','references'])
    result,summary=run_statistical_analysis_pipeline(metrics,analysis_version='2.0.0',parent_experiment_hash=parent_hash(exp),
        output_dir=tmp_path/'statistics',expected_tasks=['Li-P-Se'],expected_seeds=[42,137])
    manifest=json.loads((tmp_path/'statistics'/'analysis_manifest.json').read_text())
    assert manifest['confirmatory_controls']==['adaptive_no_memory','shuffled_memory_control']
    assert manifest['planned_family_size']==4
    assert manifest['representation_equivalence']['text_equivalence']
    assert all(r.condition_b!='text_summary_memory' for r in result)
    # Actual report completeness with a research context, rather than a synthetic success flag.
    monkeypatch.setattr(ReportGenerator,'_expected_context',staticmethod(lambda artifacts:
        (['Li-P-Se'],[42,137],list(FIVE_CONDITIONS),{'spec':{'run_mode':'research'},
            'expected_source_task':'source','proposal_budget_per_run':200,'oracle_budget_per_run':100})))
    assert ReportGenerator._target_runs_complete([m.to_dict() for m in metrics],{})[0]
    # Copy conflicts are rejected; completed shard is never silently preferred.
    conflict=merged/'references'/'Li-P-Se.verified.json'
    canonical_receipt=conflict.read_bytes()
    conflict.chmod(0o644)
    conflict.write_text('conflict')
    with pytest.raises(RuntimeError,match='collision'):
        merge_shard_outputs(exp,[moved,moved_second],merged)
    conflict.write_bytes(canonical_receipt)

    # An ordinary artifact still requires byte-identical content.
    ordinary=merged/'references/Li-P-Se.frozen.json.sha256'
    ordinary_bytes=ordinary.read_bytes()
    ordinary.chmod(0o644)
    ordinary.write_text('conflicting checksum')
    with pytest.raises(RuntimeError,match='collision'):
        merge_shard_outputs(exp,[moved,moved_second],merged)
    ordinary.write_bytes(ordinary_bytes)

    source_receipt=moved_second/receipt_relative
    original_bytes=source_receipt.read_bytes()
    changes=[
        {'reference_set_sha256':'0'*64}, {'sha256':'0'*64},
        {'task_id':'other'}, {'elements':['Li','P','S']},
        {'reference_set_certified':False}, {'verified':False},
        {'reference_set_path':'/unrelated/references/Li-P-Se.frozen.json'},
        {'unexpected_field':True},
    ]
    for change in changes:
        mutated={**second_receipt,**change}
        source_receipt.chmod(0o644)
        source_receipt.write_text(json.dumps(mutated,indent=2,sort_keys=True))
        with pytest.raises(RuntimeError,match='receipt|collision|identity'):
            merge_shard_outputs(exp,[moved,moved_second],merged)
        source_receipt.write_bytes(original_bytes)
    source_receipt.write_text('{malformed')
    with pytest.raises(RuntimeError,match='receipt'):
        merge_shard_outputs(exp,[moved,moved_second],merged)
    source_receipt.write_bytes(original_bytes)


def test_git_authorization_covers_entire_tree(monkeypatch):
    from experiments.release_integrity import code_identity
    state = {'dirty': ''}
    def git(args, **kwargs):
        if 'status' in args:
            return state['dirty']
        return ('t'*40 if args[-1]=='HEAD^{tree}' else 'c'*40) + '\n'
    monkeypatch.setattr('experiments.release_integrity.subprocess.check_output',git)
    assert code_identity('c'*40) == {'commit':'c'*40,'tree':'t'*40}
    for changed in ('experiments/revised_runner.py','agents/screening.py','new_scientific_module.py'):
        state['dirty']=' M '+changed
        with pytest.raises(ValueError,match='dirty'):
            code_identity('c'*40)
    with pytest.raises(ValueError,match='commit'):
        code_identity('other')


def test_parent_experiment_mismatch_blocks_analysis():
    data=rows()
    with pytest.raises(ValueError,match='parent experiment'):
        validate_rows(data,expected_parent='f'*64)
