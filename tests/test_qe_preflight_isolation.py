"""The QE version probe must not contaminate the authorized research tree."""

import hashlib
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace

import pytest

import experiments.cli as experiment_cli
import experiments.release_integrity as release_integrity
from experiments.spec import ExperimentSpec, QEAuditConfig, TaskDefinition


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("probe_succeeds", [True, False])
def test_research_qe_probe_keeps_clean_tree_clean(tmp_path, monkeypatch, probe_succeeds):
    repo = tmp_path / "research-code"
    (repo / "experiments").mkdir(parents=True)
    fixture_module = repo / "experiments" / "cli.py"
    fixture_module.write_text("# clean research tree\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test",
                    "-c", "user.email=test@example.org", "commit", "-qm", "fixture"], check=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setattr(experiment_cli, "__file__", str(fixture_module))
    monkeypatch.setattr(release_integrity, "__file__", str(fixture_module))

    assets = tmp_path / "assets"
    assets.mkdir()
    checkpoint = assets / "last.ckpt"
    checkpoint.write_bytes(b"checkpoint fixture")
    sampling = assets / "sampling.yaml"
    sampling.write_bytes(b"sampling fixture")
    sssp = assets / "sssp.json"
    sssp.write_bytes(b"sssp fixture")
    source_ref = assets / "source.json"
    source_ref.write_bytes(b"source reference fixture")
    target_ref = assets / "target.json"
    target_ref.write_bytes(b"target reference fixture")

    probe_cwd_file = tmp_path / "probe-cwd.txt"
    qe = assets / "pw.x"
    qe.write_text(
        "#!/bin/sh\n"
        "printf crash > CRASH\n"
        "printf input > input_tmp.in\n"
        f"pwd > {shlex.quote(str(probe_cwd_file))}\n"
        + ("printf 'Program PWSCF v.7.5 starts\\n'\n" if probe_succeeds else "exit 2\n")
    )
    qe.chmod(0o755)

    model = {"name": "CHGNet", "version": "0.4.2", "checkpoint_sha256": "a" * 64}
    relaxation = {"fmax_ev_per_angstrom": 0.05, "max_steps": 500, "relax_cell": True}
    monkeypatch.setattr(experiment_cli, "load_frozen_reference_set", lambda path: SimpleNamespace(
        certification=SimpleNamespace(certified=True), model=SimpleNamespace(**model),
        relaxation_settings=SimpleNamespace(**relaxation)))
    spec = ExperimentSpec(
        experiment_id="qe-preflight-isolation", code_commit=commit,
        source_task=TaskDefinition("Li-P-S", ["Li", "P", "S"],
            reference_set_path=str(source_ref), reference_set_sha256=_sha(source_ref),
            reference_set_certified=True),
        target_tasks=[TaskDefinition("Li-P-Se", ["Li", "P", "Se"],
            reference_set_path=str(target_ref), reference_set_sha256=_sha(target_ref),
            reference_set_certified=True)],
        master_seeds=[42, 137, 2024, 777, 999, 31415, 27182, 16180],
        run_mode="research", generation_backend="mattergen", output_root=str(tmp_path / "results"),
        mattergen_model_path=str(checkpoint), mattergen_checkpoint_sha256=_sha(checkpoint),
        mattergen_sampling_config_path=str(sampling), mattergen_sampling_config_sha256=_sha(sampling),
        pinned_model_identity=model, pinned_relaxation_settings=relaxation,
        qe_audit_config=QEAuditConfig(qe_executable=str(qe), qe_executable_version="7.5",
            qe_executable_sha256=_sha(qe), sssp_manifest_path=str(sssp),
            sssp_manifest_sha256=_sha(sssp), mock_execution=False),
    )

    if probe_succeeds:
        result = experiment_cli.run_preflight_check(spec)
        assert result["status"] == "PASSED"
        assert result["checks"]["qe_executable_ok"] is True
    else:
        with pytest.raises(RuntimeError, match="Preflight validation failed"):
            experiment_cli.run_preflight_check(spec)

    probe_cwd = Path(probe_cwd_file.read_text().strip())
    assert not probe_cwd.is_relative_to(repo)
    assert not probe_cwd.exists()  # TemporaryDirectory removed QE's byproducts.
    assert not (repo / "CRASH").exists()
    assert not (repo / "input_tmp.in").exists()
    assert subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True) == ""
    assert release_integrity.code_identity(commit)["commit"] == commit
