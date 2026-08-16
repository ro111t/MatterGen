"""Install the MatterGen package while preserving this project's Torch pin.

MatterGen v1.0.3 declares a Linux-specific Torch 2.2.1 wheel dependency. This
project configures Torch 2.4.1 instead, so dependencies are installed from
environment.yml first and MatterGen itself is installed without re-resolving
them. Run the real smoke test immediately after this bootstrap.
"""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "mattergen==1.0.3",
        ],
        check=True,
    )

    # MatterGen's pip wheel omits its required GemNet scaling factors file.
    # Copy the bundled gemnet-dT.json into the package directory.
    try:
        from pathlib import Path
        import shutil
        import mattergen.common.utils.globals as g

        src = Path(__file__).resolve().parent.parent / "agents" / "mattergen_sampling_conf" / "gemnet-dT.json"
        dst = Path(g.MODELS_PROJECT_ROOT) / "common" / "gemnet" / "gemnet-dT.json"
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
            print(f"Installed bundled GemNet scaling factors to {dst}")
    except Exception as e:
        print(f"Warning: could not install GemNet scaling factors: {e}")


if __name__ == "__main__":
    main()
