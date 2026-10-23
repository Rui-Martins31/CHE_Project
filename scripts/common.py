"""Shared configuration for the profiling scripts (checkpoint 1.1).

Holds what profile_gnu.py and profile_intel.py must agree on: the source
files, the compile-time defines (same as knn/Makefile with READ=4) and the
output folders, so both scripts always profile the same program.
"""
import argparse
import itertools
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "knn" / "src"
SRCS = ["knn.c", "utils.c", "timer.c", "io.c", "features.c", "main.c"]
IS_WINDOWS = os.name == "nt"
EXE = "harknn.exe" if IS_WINDOWS else "harknn"

# scenario -> (NUM_FEATURES, SCENARIO_FEATURES)
SCENARIOS = {"wisdm": (43, 1), "pamap2": (90, 2)}
ALL_K = [3, 5, 20]
TYPES = {1: "double", 2: "float"}


@dataclass
class Config:
    scenario: str
    k: int
    dt: int
    opt: str = "-O2"
    specialized: int = 1
    use_sqrt: int = 1

    @property
    def type(self):
        return TYPES[self.dt]

    @property
    def name(self):
        return f"{self.scenario}_k{self.k}_{self.type}"

    @property
    def out(self):
        """Results kept for the report."""
        return ROOT / "results" / "profiling" / self.name

    @property
    def work(self):
        """Build products and raw tool data."""
        return ROOT / "knn" / "run" / self.name

    def defs(self):
        num_features, scenario_features = SCENARIOS[self.scenario]
        data = ROOT / "data" / self.scenario
        # as_posix(): forward slashes work on Windows and need no escaping in a C string
        return [
            f"-I{data}", "-Wall", "-Wextra", "-std=gnu99",
            f"-DK={self.k}", f"-DDT={self.dt}", f"-DSPECIALIZED={self.specialized}",
            "-DMATH_TYPE=1", "-DDIST_METHOD=1", f"-DUSE_SQRT={self.use_sqrt}",
            f"-DNUM_FEATURES={num_features}", f"-DSCENARIO_FEATURES={scenario_features}",
            "-DREAD=4", "-DDIMEM=0", "-DVERIFY=0", "-DSTREAMING=1", "-DNORMALIZE=1",
            f'-DTRAIN_PATH="{(data / "train2_norm.dat").as_posix()}"',
            f'-DTEST_PATH="{(data / "test2_notnorm.dat").as_posix()}"',
        ]


def parse_args(description, add_arguments=None):
    """Parse the common command line; returns (list of Config, args)."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("scenario", nargs="?", default="wisdm", choices=SCENARIOS)
    p.add_argument("k", nargs="?", type=int, default=3)
    p.add_argument("dt", nargs="?", type=int, default=1, choices=TYPES,
                   help="1 = double, 2 = float (default: 1)")
    p.add_argument("--all", action="store_true",
                   help=f"run every configuration: {list(SCENARIOS)} x k={ALL_K} x {list(TYPES.values())}")
    p.add_argument("--opt", default="O2", help="optimization level, e.g. O2 or O3 (default: O2)")
    p.add_argument("--specialized", type=int, default=1)
    p.add_argument("--use-sqrt", type=int, default=1)
    if add_arguments:
        add_arguments(p)
    args = p.parse_args()

    opt = "-" + args.opt.lstrip("-")
    combos = (itertools.product(SCENARIOS, ALL_K, TYPES) if args.all
              else [(args.scenario, args.k, args.dt)])
    configs = [Config(s, k, dt, opt, args.specialized, args.use_sqrt) for s, k, dt in combos]
    return configs, args


def run(cmd, cwd=None, stdout=None):
    """Run a command (list of args, no shell); raise if it fails."""
    subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=stdout, check=True)


def run_to_file(cmd, path, cwd=None):
    with open(path, "w") as f:
        run(cmd, cwd=cwd, stdout=f)


def build(cfg, exe, flags):
    """Compile all sources into exe with the given extra flags."""
    exe.parent.mkdir(parents=True, exist_ok=True)
    run(["gcc", *flags, *cfg.defs(), *SRCS, "-o", exe, "-lm"], cwd=SRC)


def clean(folder, patterns):
    """Delete only the files/folders matching patterns (leave the other tools' results)."""
    for pattern in patterns:
        for path in folder.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def find_tool(name, override=None, fallbacks=()):
    """Return the path of a tool: explicit override, then PATH, then known install folders."""
    if override:
        return Path(override) if Path(override).exists() else None
    found = shutil.which(name)
    if found:
        return Path(found)
    for f in fallbacks:
        if Path(f).exists():
            return Path(f)
    return None


def write_program_output(raw, dest):
    """Copy the program's stdout without the per-instance 'class id' lines."""
    lines = Path(raw).read_text().splitlines(keepends=True)
    Path(dest).write_text("".join(l for l in lines if not l.startswith("class id")))


def print_highlights(program_output):
    for line in Path(program_output).read_text().splitlines():
        if "accuracy" in line or "time" in line:
            print("  " + line)
