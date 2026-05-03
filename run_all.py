from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(script: str) -> None:
    print(f"Running {script}")
    subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)


def main() -> None:
    run("src/benchmark_validity_pipeline.py")
    run("src/deep_research_candidate_audit.py")
    run("src/protocol_sensitivity_audit.py")


if __name__ == "__main__":
    main()

