"""Run every code variant and GCC combination from the project note.

Each configuration is built once. The timing binary is then executed 10
times by calling the same measurement used in run_config.py. The median
execution time and the accuracy are stored. Profiling is a separate
instrumented build, run once, and the top 5 functions are stored with
the flat profile under results/profiles/.

specialized=1 is only combined with K=3. sqrt() is only combined with
double, and sqrtf() only with float.

  python scripts/run_experiments.py
  python scripts/run_experiments.py --scenario wisdm --k 3 --dtype float
  python scripts/run_experiments.py --charts-only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from charts import write_charts
from experiment import REPEATS, append_row, configurations, format_row, measure


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Sweep HAR-kNN variants and plot the results.")
    parser.add_argument("--scenario", choices=("wisdm", "pamap2"))
    parser.add_argument("--k", type=int, choices=(3, 5, 20))
    parser.add_argument("--dtype", choices=("double", "float", "0", "1"))
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--list", action="store_true", help="print the configurations and exit")
    parser.add_argument("--charts-only", action="store_true")
    return parser.parse_args(argv)


def selected_dtype(value):
    if value in (None,):
        return None
    if value in ("0", "double"):
        return "double"
    return "float"


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.charts_only:
        written = write_charts()
        for path in written:
            print(path)
        return 0

    configs = list(
        configurations(
            scenario=args.scenario,
            k=args.k,
            dtype=selected_dtype(args.dtype),
        )
    )
    if args.list:
        for config in configs:
            print(config.slug)
        print(f"{len(configs)} configurations")
        return 0

    print(f"{len(configs)} configurations", flush=True)
    for index, config in enumerate(configs, start=1):
        print(f"[{index}/{len(configs)}] {config.slug}", flush=True)
        row = measure(config, repeats=args.repeats, with_profile=True)
        append_row(row)
        print(format_row(row), flush=True)
        print(flush=True)

    for path in write_charts():
        print(path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
