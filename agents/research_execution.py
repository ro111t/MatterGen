"""Verified campaign-level research execution boundary.

The canonical experiment pipeline is the only authoritative research path:

    ExperimentSpec -> experiment preflight -> RunSpec -> CampaignRunner
        -> VerifiedResearchExecution.verify(spec) -> MaterialsDiscoveryCampaign

A ``VerifiedResearchExecution`` is the campaign-relevant proof that the
scientific dependencies of a run were actually checked: file/digest identity
of the MatterGen checkpoint and sampling configuration, the frozen
reference-set artifact hash and certification, and the pinned CHGNet
evaluator/relaxation identity.  It is bound to the exact ``RunSpec.spec_hash``
that was verified.  Campaign-level validation and synthesis are not verified
scientific dependencies of this experiment and are recorded as explicitly
disabled; QE/SSSP remain experiment/DAG-level concerns and are deliberately
out of scope here.

The object is produced only by :meth:`VerifiedResearchExecution.verify`, which
recomputes every digest/identity itself instead of trusting caller labels.
Python object construction is not cryptographic security; the guarantee comes
from controlled factory construction, real hash/identity validation, and
campaign-side invariant checking.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

from agents.thermodynamics import (
    CHGNetRelaxationEvaluator,
    FrozenReferenceSet,
    ModelIdentity,
    RelaxationSettings,
    StructureEvaluator,
    load_frozen_reference_set,
)


class ResearchVerificationError(RuntimeError):
    """A campaign-level research dependency failed identity verification."""


def hash_path(path: Path | str) -> str:
    """SHA256 over file bytes or a deterministic directory traversal."""
    path = Path(path)
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    elif path.is_dir():
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            digest.update(str(child.relative_to(path)).replace("\\", "/").encode("utf-8"))
            digest.update(b"\0")
            with child.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    else:
        raise ResearchVerificationError(f"Required research artifact is missing: {path}")
    return digest.hexdigest()


@dataclass(frozen=True)
class ResearchVerificationReceipt:
    """Identity binding emitted by a verified execution for provenance.

    This is a receipt of a completed verification, not a grant of validity on
    its own; it records which scientific identity was verified so downstream
    artifacts can be audited against it.
    """

    spec_hash: str
    run_id: str
    mattergen_model_path: str
    mattergen_checkpoint_sha256: str
    mattergen_sampling_config_path: Optional[str]
    mattergen_sampling_config_sha256: Optional[str]
    reference_set_path: str
    reference_set_sha256: str
    model_name: str
    model_version: str
    model_checkpoint_sha256: str
    relaxation_settings: Dict[str, Any]
    validation_disabled: bool
    synthesis_disabled: bool


@dataclass(frozen=True)
class VerifiedResearchExecution:
    """Verified scientific identity for one campaign run.

    Instances are produced only by :meth:`verify`.  ``frozen_reference_set``
    and ``evaluator`` are the already-verified in-memory dependencies so the
    campaign consumes them instead of opening an independent unpinned path.
    """

    spec_hash: str
    run_id: str
    mattergen_model_path: str
    mattergen_checkpoint_sha256: str
    mattergen_sampling_config_path: Optional[str]
    mattergen_sampling_config_sha256: Optional[str]
    reference_set_path: str
    reference_set_sha256: str
    model_identity: ModelIdentity
    relaxation_settings: RelaxationSettings
    frozen_reference_set: FrozenReferenceSet
    evaluator: StructureEvaluator
    validation_disabled: bool
    synthesis_disabled: bool

    @classmethod
    def verify(
        cls,
        spec: Any,
        *,
        evaluator: Optional[StructureEvaluator] = None,
    ) -> "VerifiedResearchExecution":
        """Verify the campaign-relevant scientific identity of a RunSpec.

        ``spec`` is duck-typed over :class:`experiments.spec.RunSpec` fields so
        this module stays inside the agents layer.  ``evaluator`` is an
        optional seam for tests: when omitted, a real pinned
        :class:`CHGNetRelaxationEvaluator` is constructed and its computed
        weight digest is checked against the pinned checkpoint identity.
        """

        def _fail(message: str) -> ResearchVerificationError:
            return ResearchVerificationError(message)

        if getattr(spec, "run_mode", None) != "research":
            raise _fail("VerifiedResearchExecution requires run_mode='research'")
        required = (
            "spec_hash", "run_id", "elements",
            "reference_set_path", "reference_set_sha256",
            "pinned_model_identity", "pinned_relaxation_settings",
            "mattergen_model_path", "mattergen_checkpoint_sha256",
            "mattergen_sampling_config_path", "mattergen_sampling_config_sha256",
        )
        missing = [name for name in required if getattr(spec, name, None) is None]
        if missing:
            raise _fail(f"Run spec lacks verified research fields: {missing}")
        if not getattr(spec, "reference_set_certified", False):
            raise _fail("Run spec does not declare a certified reference set")
        # Campaign-level validation and synthesis are not verified scientific
        # dependencies of the controlled experiment; the run must declare them
        # disabled rather than naming unverified backends.
        for field_name in ("validation_calculator", "synthesis_mode"):
            value = getattr(spec, field_name, "disabled")
            if str(value).lower() != "disabled":
                raise _fail(
                    f"Research run spec must declare {field_name}='disabled'; "
                    "campaign-level validation/synthesis are not verified dependencies"
                )

        mattergen_path = Path(spec.mattergen_model_path)
        if hash_path(mattergen_path) != str(spec.mattergen_checkpoint_sha256).lower():
            raise _fail("MatterGen checkpoint SHA256 mismatch")
        sampling_path = spec.mattergen_sampling_config_path
        sampling_sha = None
        if sampling_path:
            sampling_sha = str(spec.mattergen_sampling_config_sha256 or "").lower()
            if not sampling_sha or hash_path(Path(sampling_path)) != sampling_sha:
                raise _fail("MatterGen sampling configuration SHA256 mismatch")

        reference_path = Path(spec.reference_set_path)
        expected_ref_sha = str(spec.reference_set_sha256).lower()
        if hash_path(reference_path) != expected_ref_sha:
            raise _fail("Frozen reference-set artifact SHA256 mismatch")
        try:
            frozen = load_frozen_reference_set(
                reference_path,
                expected_sha256=expected_ref_sha,
                required_chemical_system=list(spec.elements),
                require_certified=True,
            )
        except Exception as exc:
            raise _fail(f"Frozen reference set failed verification: {exc}") from exc
        expected_model = ModelIdentity(**dict(spec.pinned_model_identity))
        if frozen.model != expected_model:
            raise _fail("Frozen reference-set model identity differs from the pinned CHGNet identity")
        if frozen.relaxation_settings.__dict__ != dict(spec.pinned_relaxation_settings):
            raise _fail("Frozen reference-set relaxation settings differ from the pinned settings")

        if evaluator is None:
            # Loads CHGNet once and checks the loaded weight digest against the
            # pinned checkpoint identity; the verified instance is reused by
            # the campaign so the model is never loaded twice.
            evaluator = CHGNetRelaxationEvaluator(
                checkpoint_sha256=frozen.model.checkpoint_sha256,
                settings=frozen.relaxation_settings,
            )
        if getattr(evaluator, "model_identity", None) != frozen.model:
            raise _fail("Thermodynamic evaluator model/checkpoint identity mismatch")
        if getattr(evaluator, "relaxation_settings", None) != frozen.relaxation_settings:
            raise _fail("Thermodynamic evaluator relaxation-settings mismatch")

        return cls(
            spec_hash=str(spec.spec_hash),
            run_id=str(spec.run_id),
            mattergen_model_path=str(mattergen_path),
            mattergen_checkpoint_sha256=str(spec.mattergen_checkpoint_sha256).lower(),
            mattergen_sampling_config_path=str(sampling_path) if sampling_path else None,
            mattergen_sampling_config_sha256=sampling_sha,
            reference_set_path=str(reference_path),
            reference_set_sha256=expected_ref_sha,
            model_identity=frozen.model,
            relaxation_settings=frozen.relaxation_settings,
            frozen_reference_set=frozen,
            evaluator=evaluator,
            validation_disabled=True,
            synthesis_disabled=True,
        )

    def assert_matches_spec(self, spec: Any) -> None:
        """Raise unless ``spec`` is exactly the identity this context verified."""
        if str(getattr(spec, "spec_hash", "")) != self.spec_hash:
            raise ResearchVerificationError("RunSpec spec_hash does not match the verified execution")
        if str(getattr(spec, "run_id", "")) != self.run_id:
            raise ResearchVerificationError("RunSpec run_id does not match the verified execution")
        comparisons = (
            ("mattergen_model_path", self.mattergen_model_path),
            ("mattergen_checkpoint_sha256", self.mattergen_checkpoint_sha256),
            ("mattergen_sampling_config_path", self.mattergen_sampling_config_path),
            ("mattergen_sampling_config_sha256", self.mattergen_sampling_config_sha256),
            ("reference_set_path", self.reference_set_path),
            ("reference_set_sha256", self.reference_set_sha256),
        )
        for field_name, expected in comparisons:
            actual = getattr(spec, field_name, None)
            actual = str(actual) if actual is not None else None
            if actual != expected:
                raise ResearchVerificationError(
                    f"RunSpec {field_name} does not match the verified execution"
                )
        if dict(getattr(spec, "pinned_model_identity", {}) or {}) != {
            "name": self.model_identity.name,
            "version": self.model_identity.version,
            "checkpoint_sha256": self.model_identity.checkpoint_sha256,
        }:
            raise ResearchVerificationError("RunSpec pinned model identity does not match the verified execution")
        if dict(getattr(spec, "pinned_relaxation_settings", {}) or {}) != dict(self.relaxation_settings.__dict__):
            raise ResearchVerificationError("RunSpec pinned relaxation settings do not match the verified execution")

    def receipt(self) -> ResearchVerificationReceipt:
        """Emit the provenance receipt bound to this verified execution."""
        return ResearchVerificationReceipt(
            spec_hash=self.spec_hash,
            run_id=self.run_id,
            mattergen_model_path=self.mattergen_model_path,
            mattergen_checkpoint_sha256=self.mattergen_checkpoint_sha256,
            mattergen_sampling_config_path=self.mattergen_sampling_config_path,
            mattergen_sampling_config_sha256=self.mattergen_sampling_config_sha256,
            reference_set_path=self.reference_set_path,
            reference_set_sha256=self.reference_set_sha256,
            model_name=self.model_identity.name,
            model_version=self.model_identity.version,
            model_checkpoint_sha256=self.model_identity.checkpoint_sha256,
            relaxation_settings=dict(self.relaxation_settings.__dict__),
            validation_disabled=self.validation_disabled,
            synthesis_disabled=self.synthesis_disabled,
        )


__all__ = [
    "ResearchVerificationError",
    "ResearchVerificationReceipt",
    "VerifiedResearchExecution",
    "hash_path",
]
