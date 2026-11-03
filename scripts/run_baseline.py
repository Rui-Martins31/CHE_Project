"""Profile both HAR-kNN scenarios.

Writes, under results/:
  gprof-<scenario>.txt  `make runprofiling` from "kNN done." onward,
                        including the gprof report

`gmon.out` stays in knn/run/.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KNN_DIR = ROOT / "knn"
RESULTS_DIR = ROOT / "results"
SCENARIOS = ("wisdm", "pamap2")
MARKER = "kNN done."


def summary_from(text):
    start = text.find(MARKER)
    if start == -1:
        return text
    return text[start:]


def run_profiling(scenario, log_path):
    print(f"make runprofiling SCENARIO={scenario} -> {log_path.name}", flush=True)
    completed = subprocess.run(
        ["make", "runprofiling", f"SCENARIO={scenario}"],
        cwd=KNN_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    log_path.write_text(summary_from(completed.stdout), encoding="utf-8")
    if completed.returncode != 0:
        raise SystemExit(
            f"make runprofiling SCENARIO={scenario} failed with exit code "
            f"{completed.returncode}; see {log_path}"
        )
    if MARKER not in completed.stdout:
        raise SystemExit(f"{log_path} has no '{MARKER}' summary")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        run_profiling(scenario, RESULTS_DIR / f"gprof-{scenario}.txt")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
