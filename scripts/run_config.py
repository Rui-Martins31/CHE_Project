"""Run one HAR-kNN configuration.

Example:
  python scripts/run_config.py --scenario wisdm --k 3 --dtype 0 --specialized 1 --sqrt sqrt --opt -O2 --repeats 10 --profile
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment import REPEATS, Config, format_row, measure, rejection_reason


def dtype_argument(value):
    if value in ("0", "double"):
        return "double", 0, 1
    if value in ("1", "float"):
        return "float", 1, 2
    raise argparse.ArgumentTypeError("dtype must be 0 (double), 1 (float), double, or float")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Build and run one kNN configuration.")
    parser.add_argument("--scenario", choices=("wisdm", "pamap2"), default="wisdm")
    parser.add_argument("--k", type=int, choices=(3, 5, 20), default=3)
    parser.add_argument("--dtype", type=dtype_argument, default=("double", 0, 1))
    parser.add_argument("--specialized", type=int, choices=(0, 1), default=1)
    parser.add_argument("--sqrt", choices=("sqrt", "sqrtf", "none"), default="sqrt")
    parser.add_argument("--opt", default="-O2", help="optimization flags, without -march or -flto")
    parser.add_argument("--march", action="store_true", help="add -march=native")
    parser.add_argument("--lto", action="store_true", help="add -flto")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args):
    dtype, dtype_code, dt = args.dtype
    reason = rejection_reason(args.k, args.specialized, dtype, args.sqrt)
    if reason:
        raise SystemExit(reason)
    return Config(
        scenario=args.scenario,
        k=args.k,
        dtype=dtype,
        dtype_code=dtype_code,
        dt=dt,
        specialized=args.specialized,
        sqrt_mode=args.sqrt,
        opt_level=args.opt,
        march=args.march,
        lto=args.lto,
    )


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    config = config_from_args(args)
    repeats = args.repeats if args.repeats > 0 else REPEATS
    row = measure(config, repeats=repeats, with_profile=args.profile)
    print(format_row(row))
    return 0 if row["status"] != "fail" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
