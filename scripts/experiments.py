#!/usr/bin/env python3
"""Checkpoint 1.2 experiments: build, run, measure and profile every code variant x GCC flag set.

Code variants (per scenario):
  K            3, 5, 20
  data type    double (DT=1), float (DT=2)
  specialized  0, 1 (1 only when K=3)
  sqrt         sqrt   (USE_SQRT=1, double sqrt() also for float: MATH_TYPE=0)
               sqrtf  (USE_SQRT=1, MATH_TYPE=1; float only, for double it is the same code as sqrt)
               nosqrt (USE_SQRT=0)
GCC flags: {-Os, -O1, -O2, -O3, -Ofast, -O3 -fno-tree-vectorize, -O3 -fno-inline}
           x {nothing, -march=native, -flto, -march=native -flto}

Steps (each one skips work that is already done, so an interrupted campaign can be resumed):
  build    compile every selected variant with `make comp` (in parallel)
  run      run each binary REPS times through scripts/measure.ps1 (time, energy, power, frequency)
  profile  build with -g -pg (`make comprofiling`), run, gprof flat profile + call graph (gprof2dot)
  all      build + run + profile, then scripts/analyze.py

Examples (from the repository root, in Git Bash):
  python scripts/experiments.py all
  python scripts/experiments.py run --scenario wisdm --k 3 --opt O2 O3
  python scripts/experiments.py list --scenario pamap2
"""
import argparse
import csv
import importlib.util
import itertools
import os
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KNN = ROOT / "knn"
EXP_BIN = KNN / "bin" / "exp"
EXP_RUN = KNN / "run" / "exp"
OUT = ROOT / "results" / "experiments"

SCENARIOS = ["wisdm", "pamap2"]
KS = [3, 5, 20]
DTYPES = {"double": 1, "float": 2}
SQRTS = ["sqrt", "sqrtf", "nosqrt"]
OPTS = {  # slug -> flags
    "Os": "-Os",
    "O1": "-O1",
    "O2": "-O2",
    "O3": "-O3",
    "Ofast": "-Ofast",
    "O3-novec": "-O3 -fno-tree-vectorize",
    "O3-noinline": "-O3 -fno-inline",
}
EXTRAS = {  # slug -> flags
    "": "",
    "native": "-march=native",
    "lto": "-flto",
    "native-lto": "-march=native -flto",
}


@dataclass(frozen=True)
class Variant:
    scenario: str
    k: int
    dtype: str
    specialized: int
    sqrt: str
    opt: str
    extra: str

    @property
    def code(self):
        return f"k{self.k}_{self.dtype}_{'spec' if self.specialized else 'gen'}_{self.sqrt}"

    @property
    def flags_slug(self):
        return self.opt + (f"+{self.extra}" if self.extra else "")

    @property
    def id(self):
        return f"{self.code}__{self.flags_slug}"

    @property
    def flags(self):
        return f"{OPTS[self.opt]} {EXTRAS[self.extra]}".strip()

    def make_vars(self):
        return {
            "SCENARIO": self.scenario,
            "K": self.k,
            "DT": DTYPES[self.dtype],
            "SPECIALIZED": self.specialized,
            "USE_SQRT": 0 if self.sqrt == "nosqrt" else 1,
            # the original code uses MATH_TYPE=1; for double sqrt both values give sqrt()
            "MATH_TYPE": 0 if (self.sqrt == "sqrt" and self.dtype == "float") else 1,
            "OPT": self.flags,
        }

    def exe(self):
        return EXP_BIN / self.scenario / f"{self.id}.exe"

    def raw_csv(self):
        return OUT / "raw" / self.scenario / f"{self.id}.csv"

    def prof_dir(self):
        return OUT / "profiling" / self.scenario / self.id


def all_variants():
    for s, k, dt, spec, sq, opt, ex in itertools.product(
            SCENARIOS, KS, DTYPES, [0, 1], SQRTS, OPTS, EXTRAS):
        if spec and k != 3:
            continue  # specialization only exists for k=3
        if sq == "sqrtf" and dt != "float":
            continue  # sqrtf only differs from sqrt for float
        yield Variant(s, k, dt, spec, sq, opt, ex)


def select(args):
    vs = [v for v in all_variants()
          if v.scenario in args.scenario and v.k in args.k and v.dtype in args.dtype
          and v.sqrt in args.sqrt and v.opt in args.opt and v.extra in args.extra
          and (args.specialized is None or v.specialized in args.specialized)]
    if args.limit:
        vs = vs[:args.limit]
    return vs


def make_cmd():
    for name in ("make", "mingw32-make"):
        if shutil.which(name):
            return name
    sys.exit("make not found (on Windows install MinGW make: mingw32-make)")


def make(target, v, exe, log):
    vars_ = v.make_vars() | {"TARGET": exe.as_posix()}
    cmd = [make_cmd(), "-s", "-C", str(KNN), target] + [f"{k}={val}" for k, val in vars_.items()]
    exe.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    log.write_text(" ".join(cmd) + "\n\n" + r.stdout + r.stderr)
    return r.returncode == 0 and exe.exists()


# ---------------------------------------------------------------- build
def cmd_build(args, vs):
    todo = [v for v in vs if args.force or not v.exe().exists()]
    print(f"build: {len(todo)} of {len(vs)} variants need building ({args.jobs} jobs)")
    done = failed = 0
    t0 = time.time()

    def one(v):
        return v, make("comp", v, v.exe(), v.exe().with_suffix(".log"))

    with ThreadPoolExecutor(args.jobs) as pool:
        for v, ok in pool.map(one, todo):
            done += 1
            if not ok:
                failed += 1
                print(f"  FAILED {v.scenario}/{v.id} (see {v.exe().with_suffix('.log')})")
            if done % 20 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} built, {failed} failed, {time.time() - t0:.0f} s")


# ---------------------------------------------------------------- run
def cmd_run(args, vs):
    todo = [v for v in vs if v.exe().exists() and (args.force or not v.raw_csv().exists())]
    missing = sum(not v.exe().exists() for v in vs)
    if missing:
        print(f"run: {missing} variants are not built (run `build` first), skipping them")
    print(f"run: {len(todo)} variants x {args.reps} reps")
    if args.shuffle:  # spread slow drifts (temperature, background load) over all variants
        random.Random(42).shuffle(todo)
    EXP_RUN.mkdir(parents=True, exist_ok=True)
    measure = ROOT / "scripts" / "measure.ps1"
    t0 = time.time()
    for i, v in enumerate(todo, 1):
        out = v.raw_csv()
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(measure),
               "-Exe", str(v.exe()), "-Reps", str(args.reps), "-Warmup", str(args.warmup),
               "-Cpu", str(args.cpu), "-WorkDir", str(EXP_RUN), "-OutCsv", str(tmp)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not tmp.exists():
            print(f"  [{i}/{len(todo)}] {v.scenario}/{v.id}: measure.ps1 failed\n{r.stdout}{r.stderr}")
            continue
        tmp.replace(out)
        rows = list(csv.DictReader(out.open()))
        ok = all(row["exit_code"] == "0" for row in rows)
        med = sorted(float(row["knn_s"]) for row in rows)[len(rows) // 2] if ok else float("nan")
        eta = (time.time() - t0) / i * (len(todo) - i)
        print(f"  [{i}/{len(todo)}] {v.scenario}/{v.id}: "
              + (f"knn {med:.3f} s, acc {rows[0]['accuracy']} %" if ok else "CRASHED")
              + f"  (eta {eta / 60:.0f} min)")


# ---------------------------------------------------------------- profile
FLAT_RE = re.compile(r"^\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(?:(\d+)\s+([\d.]+)\s+([\d.]+)\s+)?(\S.*)$")


def top_functions(gprof_txt, n):
    """Parse the gprof flat profile: [(name, %time, self_s, calls)]."""
    lines = gprof_txt.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.strip().startswith("time   seconds")) + 1
    except StopIteration:
        return []
    top = []
    for l in lines[start:]:
        m = FLAT_RE.match(l)
        if not m:
            break
        top.append((m.group(7).strip(), float(m.group(1)), float(m.group(3)), m.group(4) or ""))
        if len(top) == n:
            break
    return top


def cmd_profile(args, vs):
    todo = [v for v in vs if args.force or not (v.prof_dir() / "top.csv").exists()]
    print(f"profile: {len(todo)} variants, {args.prof_runs} runs each (gmon.out summed)")
    gprof2dot = importlib.util.find_spec("gprof2dot") is not None
    dot = shutil.which("dot")
    if not gprof2dot:
        print("  gprof2dot not found (pip install gprof2dot): call graphs are skipped")
    elif not dot:
        print("  Graphviz `dot` not found: call graphs are written as .dot only")

    def build(v):
        d = v.prof_dir()
        d.mkdir(parents=True, exist_ok=True)
        return v, make("comprofiling", v, d / "harknn.exe", d / "build.log")

    with ThreadPoolExecutor(args.jobs) as pool:
        built = list(pool.map(build, todo))

    for i, (v, ok) in enumerate(built, 1):
        d = v.prof_dir()
        if not ok:
            print(f"  [{i}/{len(built)}] {v.scenario}/{v.id}: build FAILED")
            continue
        exe = d / "harknn.exe"
        for f in ("gmon.out", "gmon.sum"):
            (d / f).unlink(missing_ok=True)
        crashed = False
        for _ in range(args.prof_runs):
            r = subprocess.run([str(exe)], cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if r.returncode != 0 or not (d / "gmon.out").exists():
                crashed = True
                break
            if (d / "gmon.sum").exists():
                subprocess.run(["gprof", "-s", str(exe), "gmon.out", "gmon.sum"], cwd=d, check=True)
                (d / "gmon.out").unlink()
            else:
                (d / "gmon.out").rename(d / "gmon.sum")
        if crashed:
            print(f"  [{i}/{len(built)}] {v.scenario}/{v.id}: run CRASHED")
            continue
        txt = subprocess.run(["gprof", "-b", str(exe), "gmon.sum"], cwd=d,
                             capture_output=True, text=True).stdout
        (d / "gprof.txt").write_text(txt)
        top = top_functions(txt, 5)
        with (d / "top.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank", "function", "pct_time", "self_s", "calls"])
            for rank, row in enumerate(top, 1):
                w.writerow([rank, *row])
        if gprof2dot:
            subprocess.run([sys.executable, "-m", "gprof2dot", "-f", "prof", "-o", str(d / "callgraph.dot"), str(d / "gprof.txt")])
            if dot and (d / "callgraph.dot").exists():
                subprocess.run([dot, "-Tsvg", "-o", str(d / "callgraph.svg"), str(d / "callgraph.dot")])
        exe.unlink()  # keep only the reports
        print(f"  [{i}/{len(built)}] {v.scenario}/{v.id}: "
              + ", ".join(f"{name} {pct:.0f}%" for name, pct, _, _ in top[:3]))


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["list", "build", "run", "profile", "all"])
    p.add_argument("--scenario", nargs="+", default=SCENARIOS, choices=SCENARIOS)
    p.add_argument("--k", nargs="+", type=int, default=KS, choices=KS)
    p.add_argument("--dtype", nargs="+", default=list(DTYPES), choices=list(DTYPES))
    p.add_argument("--specialized", nargs="+", type=int, choices=[0, 1])
    p.add_argument("--sqrt", nargs="+", default=SQRTS, choices=SQRTS)
    p.add_argument("--opt", nargs="+", default=list(OPTS), choices=list(OPTS))
    p.add_argument("--extra", nargs="+", default=list(EXTRAS), choices=list(EXTRAS),
                   help="'' = no extra flags; native; lto; native-lto")
    p.add_argument("--limit", type=int, help="only the first N selected variants (for testing)")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--cpu", type=int, default=12, help="logical CPU to pin the program to (a P-core)")
    p.add_argument("--prof-runs", type=int, default=3)
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    p.add_argument("--force", action="store_true", help="redo work that is already done")
    args = p.parse_args()

    vs = select(args)
    if args.step == "list":
        for v in vs:
            print(f"{v.scenario:7} {v.id:45} make {' '.join(f'{k}={x!r}' for k, x in v.make_vars().items())}")
        print(f"{len(vs)} variants")
        return
    if args.step in ("build", "all"):
        cmd_build(args, vs)
    if args.step in ("run", "all"):
        cmd_run(args, vs)
    if args.step in ("profile", "all"):
        cmd_profile(args, vs)
    if args.step == "all":
        subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze.py")], check=True)


if __name__ == "__main__":
    main()
