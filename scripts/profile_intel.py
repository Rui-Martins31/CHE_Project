"""Profile the HAR-kNN program with Intel VTune and Intel Advisor (checkpoint 1.1).

Usage (PowerShell, Git Bash or Linux):
    python scripts/profile_intel.py [SCENARIO] [K] [DT] [--opt O2] [--all]
                                    [--no-vtune] [--no-advisor] [--interval MS]
      SCENARIO, K, DT, --opt, --all: same as profile_gnu.py

The program is built with <opt> -g (no -pg/--coverage instrumentation).
Outputs go to results/profiling/<scenario>_k<K>_<type>/:
    vtune_summary.txt     - VTune hotspots summary
    vtune_hotspots.csv    - VTune CPU time per function
    vtune_topdown.txt     - VTune call tree (compare with the gprof call graph)
    advisor_roofline.html - Advisor roofline chart (open in a browser)
    advisor_survey.csv    - Advisor survey: time, GFLOPS, AI per loop/function
Raw results stay in knn/run/<config>/intel/ and can be opened in the GUIs:
    vtune-gui knn/run/<config>/intel/vtune/vtune.vtune
    advisor-gui knn/run/<config>/intel/advisor
Only these files are replaced; results from profile_gnu.py are kept.
"""
import subprocess

from common import EXE, SRC, build, clean, find_tool, parse_args, run, run_to_file

ONEAPI = ["C:/Program Files (x86)/Intel/oneAPI", "/opt/intel/oneapi"]


def find_intel(name, override):
    return find_tool(name, override, [f"{root}/{name}/latest/bin64/{n}"
                                      for root in ONEAPI for n in (f"{name}.exe", name)])


def run_vtune(vtune, cfg, exe, work, interval):
    r = work / "vtune"
    knobs = ["-knob", f"sampling-interval={interval}"] if interval else []
    run_to_file([vtune, "-collect", "hotspots", *knobs, "-r", r, "--", exe],
                work / "vtune_collect.log", cwd=work)
    run([vtune, "-report", "summary", "-r", r, "-report-output", cfg.out / "vtune_summary.txt"])
    run([vtune, "-report", "hotspots", "-r", r, "-format", "csv", "-csv-delimiter", "comma",
         "-report-output", cfg.out / "vtune_hotspots.csv"])
    run([vtune, "-report", "top-down", "-r", r, "-report-output", cfg.out / "vtune_topdown.txt"])


def run_advisor(advisor, cfg, exe, work):
    proj = work / "advisor"
    # roofline = survey (time per loop) + tripcounts with FLOP/memory counting (much slower)
    run_to_file([advisor, "--collect=roofline", f"--project-dir={proj}",
                 f"--search-dir=src:r={SRC}", "--", exe],
                work / "advisor_collect.log", cwd=work)
    run([advisor, "--report=roofline", f"--project-dir={proj}",
         f"--report-output={cfg.out / 'advisor_roofline.html'}"])
    run([advisor, "--report=survey", f"--project-dir={proj}", "--format=csv",
         f"--report-output={cfg.out / 'advisor_survey.csv'}"])


def profile(cfg, tools, interval):
    work = cfg.work / "intel"
    cfg.out.mkdir(parents=True, exist_ok=True)
    clean(cfg.out, [f"{name}_*" for name in tools])
    clean(cfg.work, ["intel"])

    exe = work / EXE
    build(cfg, exe, [cfg.opt, "-g"])

    for name, path in tools.items():
        print(f"== {cfg.name}: {name} ({cfg.opt} -g)")
        try:
            if name == "vtune":
                run_vtune(path, cfg, exe, work, interval)
            else:
                run_advisor(path, cfg, exe, work)
        except subprocess.CalledProcessError as e:
            # keep going: one tool failing should not lose the other's results
            print(f"   {name} failed (exit {e.returncode}); see {work / f'{name}_collect.log'}")
    print(f"== done -> {cfg.out}")


def main():
    def add_arguments(p):
        p.add_argument("--no-vtune", action="store_true")
        p.add_argument("--no-advisor", action="store_true")
        p.add_argument("--vtune", help="path to vtune executable")
        p.add_argument("--advisor", help="path to advisor executable")
        p.add_argument("--interval", type=float,
                       help="VTune sampling interval in ms (default: VTune's, ~10 ms)")

    configs, args = parse_args("Profile HAR-kNN with Intel VTune and Advisor.", add_arguments)

    tools = {}
    for name, skip, override in [("vtune", args.no_vtune, args.vtune),
                                 ("advisor", args.no_advisor, args.advisor)]:
        if skip:
            continue
        path = find_intel(name, override)
        if path:
            tools[name] = path
        else:
            print(f"== {name} not found, skipping (pass --{name} PATH)")
    if not tools:
        return

    for cfg in configs:
        profile(cfg, tools, args.interval)


if __name__ == "__main__":
    main()
