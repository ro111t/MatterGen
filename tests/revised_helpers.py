"""Explicit CPU-only authorization seam; production always checks real Git."""
from dataclasses import replace


def authorize(spec, root, monkeypatch, parent='a' * 64):
    identity = {'commit': 'test-authorized', 'tree': 'b' * 40}
    monkeypatch.setattr('experiments.release_integrity.code_identity', lambda expected_commit, output_root=None: identity)
    return replace(spec, artifact_root=str(root), authorized_code_commit=identity['commit'],
                   authorized_code_identity=identity, parent_experiment_hash=parent)
