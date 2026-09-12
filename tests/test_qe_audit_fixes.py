import subprocess
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from experiments.qe_audit import (
    ASEQuantumEspressoCalculator,
    QECalculationStatus,
    QEAuditConfig,
)
from experiments.spec import ExperimentSpec, RunSpec, compute_sha256
from experiments.runner import CampaignRunner
from campaign import CampaignConfig


def test_qe_version_detection_standard_banners():
    """Verify QE version detection supports Program PWSCF v.7.5 and standard banner variants."""
    banners = [
        ("     Program PWSCF v.7.5 starts on 10Sep2026 at 18:30:00", "7.5"),
        ("     Program PWSCF v.7.3.1 starts on 15Mar2024 at 12:00:00", "7.3.1"),
        ("Program PWSCF v. 7.5", "7.5"),
        ("Program PWSCF v7.5", "7.5"),
        ("Program PWSCF version 7.5", "7.5"),
        ("Program PWSCF 7.5", "7.5"),
        ("Program 7.5", "7.5"),
        ("version 7.5", "7.5"),
        ("version: 7.5", "7.5"),
        ("PWSCF v.7.5", "7.5"),
        ("v.7.5", "7.5"),
    ]
    calc = ASEQuantumEspressoCalculator(pseudopotentials={}, executable="pw.x", executable_version="7.5")
    for banner, expected_version in banners:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=banner, stderr="", returncode=0)
            detected = calc._detect_version()
            assert detected == expected_version, f"Failed for banner: {banner!r} (got {detected})"


def test_qe_relaxation_maximum_steps_reached_labeled_not_converged(tmp_path):
    """Verify that an unfinished relaxation reaching maximum steps returns RELAXATION_NOT_CONVERGED."""
    calc_dir = tmp_path / "calc_max_steps"
    pseudo_file = tmp_path / "Li.upf"
    pseudo_file.write_text("dummy pseudo", encoding="utf-8")

    calc = ASEQuantumEspressoCalculator(
        pseudopotentials={"Li": str(pseudo_file)},
        executable="pw.x",
        executable_version="7.5",
    )
    config = QEAuditConfig(ecutwfc_ry=60.0)
    structure = {
        "lattice": [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        "species": ["Li"],
        "positions": [[0.0, 0.0, 0.0]],
        "fractional": True,
    }

    # Output that indicates maximum steps were reached despite having JOB DONE and energy
    qe_output_max_steps = """
     Program PWSCF v.7.5 starts on 10Sep2026 at 18:30:00
     ...
     iteration # 10
     !    total energy              =   -15.12345678 Ry
     The maximum number of steps has been reached.
     End of BFGS geometry optimization
     Preliminary final coordinates
     ...
     =------------------------------------------------------------------------------=
     JOB DONE.
     =------------------------------------------------------------------------------=
"""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=qe_output_max_steps, stderr="", returncode=0)
        result = calc.run_calculation("Li", structure, config, calc_dir)

    assert result.status == QECalculationStatus.RELAXATION_NOT_CONVERGED.value
    assert "did not converge" in result.error_message
    assert result.error_message is not None


def test_qe_relaxation_converged_labeled_converged(tmp_path):
    """Verify that a properly converged BFGS relaxation returns CONVERGED."""
    calc_dir = tmp_path / "calc_converged"
    pseudo_file = tmp_path / "Li.upf"
    pseudo_file.write_text("dummy pseudo", encoding="utf-8")

    calc = ASEQuantumEspressoCalculator(
        pseudopotentials={"Li": str(pseudo_file)},
        executable="pw.x",
        executable_version="7.5",
    )
    config = QEAuditConfig(ecutwfc_ry=60.0)
    structure = {
        "lattice": [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        "species": ["Li"],
        "positions": [[0.0, 0.0, 0.0]],
        "fractional": True,
    }

    qe_output_converged = """
     Program PWSCF v.7.5 starts on 10Sep2026 at 18:30:00
     ...
     iteration # 5
     !    total energy              =   -15.12345678 Ry
     bfgs converged in   5 scf cycles and   3 bfgs steps
     End of BFGS geometry optimization
     Begin final coordinates
     ATOMIC_POSITIONS crystal
     Li 0.0 0.0 0.0
     End final coordinates
     =------------------------------------------------------------------------------=
     JOB DONE.
     =------------------------------------------------------------------------------=
"""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=qe_output_converged, stderr="", returncode=0)
        result = calc.run_calculation("Li", structure, config, calc_dir)

    assert result.status == QECalculationStatus.CONVERGED.value
    assert result.error_message is None
    assert result.total_energy_ev is not None


def test_qe_subprocess_uses_relative_qe_in(tmp_path):
    """Verify subprocess receives 'qe.in' relative to calc_dir rather than calc_dir/qe.in."""
    calc_dir = tmp_path / "rel_calc_dir"
    pseudo_file = tmp_path / "Li.upf"
    pseudo_file.write_text("dummy pseudo", encoding="utf-8")

    calc = ASEQuantumEspressoCalculator(
        pseudopotentials={"Li": str(pseudo_file)},
        executable="pw.x",
        executable_version="7.5",
    )
    config = QEAuditConfig(ecutwfc_ry=60.0)
    structure = {
        "lattice": [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        "species": ["Li"],
        "positions": [[0.0, 0.0, 0.0]],
        "fractional": True,
    }

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="JOB DONE", stderr="", returncode=0)
        calc.run_calculation("Li", structure, config, calc_dir)

        # Inspect the command arguments passed to subprocess.run
        args, kwargs = mock_run.call_args
        cmd = args[0]
        assert cmd[0] == "pw.x"
        assert cmd[1] == "-in"
        assert cmd[2] == "qe.in"  # Must be 'qe.in', not str(calc_dir / 'qe.in')
        assert kwargs["cwd"] == str(calc_dir)


def test_legacy_converged_cache_is_invalidated(tmp_path):
    """A valid pre-fix cache must not preserve a false convergence decision."""
    pseudo = tmp_path / "Li.upf"
    pseudo.write_text("dummy pseudo", encoding="utf-8")
    calc = ASEQuantumEspressoCalculator(
        executable="pw.x", executable_version="7.5",
        pseudopotentials={"Li": str(pseudo)},
    )
    config = QEAuditConfig()
    structure = {
        "lattice": [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
        "positions": [[0, 0, 0]], "species": ["Li"],
    }
    output = (
        "! total energy = -15 Ry\n"
        "The maximum number of steps has been reached.\nJOB DONE.\n"
    )
    calc_dir = tmp_path / "calc"
    with patch("subprocess.run", return_value=MagicMock(
        stdout=output, stderr="", returncode=0,
    )):
        calc.run_calculation("Li", structure, config, calc_dir)

    # Recreate the old parser's correctly hashed but incorrect result.
    result_path = calc_dir / "result.json"
    legacy = json.loads(result_path.read_text(encoding="utf-8"))
    input_text, _, _ = calc._input_text("Li", structure, config)
    legacy["input_hash"] = compute_sha256({
        "input": input_text,
        "pseudopotentials": {"Li": hashlib.sha256(pseudo.read_bytes()).hexdigest()},
        "executable": calc.executable,
        "executable_version": calc.executable_version,
    })
    legacy["status"] = QECalculationStatus.CONVERGED.value
    legacy["error_message"] = None
    legacy["result_hash"] = compute_sha256({
        k: v for k, v in legacy.items() if k != "result_hash"
    })
    result_path.write_text(json.dumps(legacy), encoding="utf-8")

    with patch("subprocess.run", return_value=MagicMock(
        stdout=output, stderr="", returncode=0,
    )) as run:
        result = calc.run_calculation("Li", structure, config, calc_dir)
    run.assert_called_once()
    assert result.status == QECalculationStatus.RELAXATION_NOT_CONVERGED.value

    with patch("subprocess.run") as run:
        cached = calc.run_calculation("Li", structure, config, calc_dir)
    run.assert_not_called()
    assert cached.status == QECalculationStatus.RELAXATION_NOT_CONVERGED.value


def test_allow_llm_orchestration_propagation():
    """Verify allow_llm_orchestration is preserved across ExperimentSpec, RunSpec, and CampaignConfig."""
    exp_spec = ExperimentSpec(
        experiment_id="test_exp_spec",
        master_seeds=[42],
        allow_llm_orchestration=True,
    )
    assert exp_spec.allow_llm_orchestration is True

    # Test serialization round-trip
    d = exp_spec.to_dict()
    assert d["allow_llm_orchestration"] is True
    restored = ExperimentSpec.from_dict(d)
    assert restored.allow_llm_orchestration is True

    # Test RunSpec default and assignment
    run_spec_default = RunSpec(
        run_id="run_default",
        experiment_id="test_exp",
        task_id="Li-P-S",
        elements=["Li", "P", "S"],
        condition="text_summary_memory",
        seed=42,
        iteration_seeds=[42],
        proposal_budget=10,
        oracle_budget=5,
        geometry_min_distance=0.8,
        thermodynamics_retain_threshold_ev_per_atom=0.10,
        thermodynamics_stable_threshold_ev_per_atom=0.03,
        run_mode="development",
        generation_backend="mock",
        memory_mode="text_summary",
        memory_seed=42,
        output_dir="./out",
        source_memory_snapshot_path="dummy",
    )
    assert run_spec_default.allow_llm_orchestration is True

    # Test when explicitly set to False
    run_spec_disabled = RunSpec(
        run_id="run_disabled",
        experiment_id="test_exp",
        task_id="Li-P-S",
        elements=["Li", "P", "S"],
        condition="text_summary_memory",
        seed=42,
        iteration_seeds=[42],
        proposal_budget=10,
        oracle_budget=5,
        geometry_min_distance=0.8,
        thermodynamics_retain_threshold_ev_per_atom=0.10,
        thermodynamics_stable_threshold_ev_per_atom=0.03,
        run_mode="development",
        generation_backend="mock",
        memory_mode="text_summary",
        memory_seed=42,
        output_dir="./out",
        source_memory_snapshot_path="dummy",
        allow_llm_orchestration=False,
    )
    assert run_spec_disabled.allow_llm_orchestration is False


def test_runner_passes_allow_llm_orchestration_to_campaign_config(tmp_path):
    """Verify CampaignRunner propagates spec.allow_llm_orchestration to CampaignConfig."""
    import json
    out_dir = tmp_path / "run_out"
    run_spec = RunSpec(
        run_id="run_test_llm_flag",
        experiment_id="test_exp",
        task_id="Li-P-S",
        elements=["Li", "P", "S"],
        condition="adaptive_no_memory",
        seed=42,
        iteration_seeds=[42],
        proposal_budget=10,
        oracle_budget=5,
        geometry_min_distance=0.8,
        thermodynamics_retain_threshold_ev_per_atom=0.10,
        thermodynamics_stable_threshold_ev_per_atom=0.03,
        run_mode="development",
        generation_backend="mock",
        memory_mode="none",
        memory_seed=42,
        output_dir=str(out_dir),
        allow_llm_orchestration=True,
    )

    with patch("experiments.runner.MaterialsDiscoveryCampaign") as mock_campaign_cls:
        mock_campaign = MagicMock()
        def fake_run():
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "manifest.json").write_text(json.dumps({"status": "completed", "finalized": True}))
            (out_dir / "campaign_provenance.json").write_text("{}")
        mock_campaign.run_campaign.side_effect = fake_run
        mock_campaign_cls.return_value = mock_campaign

        with patch.object(CampaignRunner, "_canonical_manifest_hash", return_value="dummy_hash"), \
             patch("experiments.runner.compute_file_sha256", return_value="dummy_sha"):
            CampaignRunner.execute_run(run_spec)

        call_args, call_kwargs = mock_campaign_cls.call_args
        config = call_kwargs["config"]
        assert config.allow_llm_orchestration is True
