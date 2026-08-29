"""Build a frozen thermodynamic reference set from a local JSON input.

This command deliberately has no Materials Project or other network client.
The input must already contain downloaded structures and source metadata.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

# Support direct execution from a checkout (``python scripts/build_reference_set.py``)
# where Python otherwise puts ``scripts/`` rather than the repository root on
# ``sys.path``.  This is deliberately derived from this file, never cwd.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.thermodynamics import ReferencePhaseInput, build_frozen_reference_set, deserialize_structure


def _load_evaluator(spec: str):
    module_name, object_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), object_name)
    return factory() if callable(factory) else factory


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an offline, checksummed reference set")
    parser.add_argument("input", type=Path, help="Local JSON containing reference_set_id, chemical_system, and phases")
    parser.add_argument("output", type=Path, help="Frozen JSON output path")
    parser.add_argument("--evaluator", required=True, help="Import path module:factory for the pinned relaxer")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    inputs = [
        ReferencePhaseInput(
            source_id=phase["source_id"], source=phase.get("source", "local"),
            structure=deserialize_structure(phase["structure"]),
            source_energy_above_hull_ev_per_atom=phase.get("source_energy_above_hull_ev_per_atom"),
            metadata=phase.get("metadata", {}),
        )
        for phase in payload["phases"]
    ]
    frozen = build_frozen_reference_set(
        reference_set_id=payload["reference_set_id"], chemical_system=payload["chemical_system"],
        inputs=inputs, evaluator=_load_evaluator(args.evaluator), output_path=args.output,
        created_at_iso=payload.get("created_at_iso"),
    )
    print(json.dumps({
        "reference_set_id": frozen.reference_set_id,
        "certified": frozen.certification.certified,
        "sha256": frozen.reference_set_hash,
        "phases": len(frozen.phases),
    }, sort_keys=True))
    return 0 if frozen.certification.certified else 2


if __name__ == "__main__":
    raise SystemExit(main())
