"""Profile the HAR-kNN program with gprof and gcov (checkpoint 1.1).

Usage (PowerShell, Git Bash or Linux):
    python scripts/profile_gnu.py [SCENARIO] [K] [DT] [--opt O2] [--all]
      SCENARIO: wisdm | pamap2   (default: wisdm)
      K:        3 | 5 | 7 | 20   (default: 3)
      DT:       1 = double, 2 = float (default: 1)
      --all:    the 12 configurations (2 scenarios x k=3,5,20 x double/float)

Outputs go to results/profiling/<scenario>_k<K>_<type>/:
    program_output.txt   - program stdout (accuracy, timings)
    gprof_<OPT>.txt      - gprof flat profile + call graph (build with --opt)
    gprof_O0.txt         - gprof at -O0, no inlining -> complete call graph
    gcov/*.gcov          - line-by-line execution counts (-O0 --coverage)
    gcov_summary.txt     - per-file / per-function coverage and branch summary
Only these files are replaced; results from profile_intel.py are kept.
"""
from common import (EXE, IS_WINDOWS, SRC, SRCS, build, clean, parse_args, print_highlights,
                    run, run_to_file, write_program_output)

# -no-pie is required on MinGW-w64: with PIE/ASLR gprof reports an empty profile.
PG_FLAGS = ["-g", "-pg"] + (["-no-pie"] if IS_WINDOWS else [])


def run_gprof(cfg, work, label, flags):
    d = work / f"gprof_{label}"
    exe = d / EXE
    build(cfg, exe, [*flags, *PG_FLAGS])
    run_to_file([exe], d / "program_output.txt", cwd=d)  # writes gmon.out in d
    run_to_file(["gprof", "-b", exe, "gmon.out"], cfg.out / f"gprof_{label}.txt", cwd=d)
    return d


def run_gcov(cfg, work):
    gc = work / "gcov"
    gc.mkdir(parents=True)
    # Compile file by file so each .gcda lands next to its .o in gc
    for f in SRCS:
        run(["gcc", "-O0", "--coverage", *cfg.defs(), "-c", SRC / f, "-o", gc / f.replace(".c", ".o")])
    run(["gcc", "--coverage", *sorted(gc.glob("*.o")), "-o", gc / EXE, "-lm"])
    run_to_file([gc / EXE], gc / "program_output.txt", cwd=gc)
    gcov_dir = cfg.out / "gcov"
    gcov_dir.mkdir()
    run_to_file(["gcov", "-b", "-c", "-f", "-o", gc, *(SRC / f for f in SRCS)],
                cfg.out / "gcov_summary.txt", cwd=gcov_dir)


def profile(cfg):
    work = cfg.work / "gnu"
    cfg.out.mkdir(parents=True, exist_ok=True)
    clean(cfg.out, ["program_output.txt", "gprof_*.txt", "gcov", "gcov_summary.txt"])
    clean(cfg.work, ["gnu"])
    label = cfg.opt.lstrip("-")

    print(f"== {cfg.name}: gprof ({cfg.opt})")
    d = run_gprof(cfg, work, label, [cfg.opt])
    write_program_output(d / "program_output.txt", cfg.out / "program_output.txt")

    if label != "O0":
        print(f"== {cfg.name}: gprof (-O0, no inlining)")
        run_gprof(cfg, work, "O0", ["-O0", "-fno-inline"])

    print(f"== {cfg.name}: gcov (-O0 --coverage)")
    run_gcov(cfg, work)

    print(f"== done -> {cfg.out}")
    print_highlights(cfg.out / "program_output.txt")
    flat = (cfg.out / f"gprof_{label}.txt").read_text().splitlines()
    print("\n".join(flat[2:12]))


def main():
    configs, _ = parse_args("Profile HAR-kNN with gprof and gcov.")
    for cfg in configs:
        profile(cfg)


if __name__ == "__main__":
    main()
