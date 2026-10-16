#!/usr/bin/env python3
"""Summarise the checkpoint 1.2 measurements and draw the charts.

Reads   results/experiments/raw/<scenario>/<variant>.csv       (scripts/experiments.py run)
        results/experiments/profiling/<scenario>/<variant>/top.csv  (scripts/experiments.py profile)
Writes  results/experiments/summary.csv    one row per variant: medians of the 10 runs, gains vs baseline
        results/experiments/hotspots.csv   top-5 gprof functions per variant
        results/experiments/summary.md     best variants and per-option tables
        results/experiments/charts/*.png

Baseline (the original code): double, sqrt, SPECIALIZED=1 (only has an effect for k=3), -O2,
one baseline per scenario and k. Speedup = baseline time / time; energy gain = baseline energy / energy.
All times, energies, power and frequency are for the kNN phase (the timed loop in main.c);
the *_total columns are for the whole process (including reading the .dat files).
"""
import csv
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments import EXTRAS, KS, OPTS, OUT, SCENARIOS, all_variants  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm  # noqa: E402

CHARTS = OUT / "charts"

# palette (validated reference palette: categorical slots in fixed order, blue<->gray<->red diverging)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list(
    "gain", ["#b8322f", "#e98b85", "#f0efec", "#86b6ef", "#1c5cab"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
})

METRICS = ["wall_s", "knn_s", "accuracy", "pkg_J", "cores_J", "pkg_W", "freq_MHz",
           "knn_pkg_J", "knn_cores_J", "knn_pkg_W", "knn_freq_MHz"]


def baseline_of(v):
    return (v.scenario, v.k, "double", 1 if v.k == 3 else 0, "sqrt", "O2", "")


def key(v):
    return (v.scenario, v.k, v.dtype, v.specialized, v.sqrt, v.opt, v.extra)


def load():
    rows = {}
    for v in all_variants():
        f = v.raw_csv()
        if not f.exists():
            continue
        reps = list(csv.DictReader(f.open()))
        r = {"variant": v, "reps": len(reps), "status": "ok"}
        if not reps or any(x["exit_code"] != "0" for x in reps):
            r["status"] = "crashed"
        else:
            for m in METRICS:
                vals = [float(x[m]) for x in reps if x[m] not in ("NaN", "")]
                r[m] = statistics.median(vals) if vals else math.nan
            r["knn_s_min"] = min(float(x["knn_s"]) for x in reps)
            r["knn_s_max"] = max(float(x["knn_s"]) for x in reps)
        rows[key(v)] = r
    for r in rows.values():
        b = rows.get(baseline_of(r["variant"]))
        ok = r["status"] == "ok" and b and b["status"] == "ok"
        r["speedup"] = b["knn_s"] / r["knn_s"] if ok else math.nan
        r["energy_gain"] = b["knn_pkg_J"] / r["knn_pkg_J"] if ok else math.nan
        r["acc_delta"] = r["accuracy"] - b["accuracy"] if ok else math.nan
    return rows


def write_summary(rows):
    cols = ["scenario", "k", "dtype", "specialized", "sqrt", "opt", "extra", "flags", "status", "reps",
            "knn_s", "knn_s_min", "knn_s_max", "speedup", "knn_pkg_J", "knn_cores_J", "energy_gain",
            "knn_pkg_W", "knn_freq_MHz", "accuracy", "acc_delta",
            "wall_s", "pkg_J", "cores_J", "pkg_W", "freq_MHz"]
    with (OUT / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows.values():
            v = r["variant"]
            base = [v.scenario, v.k, v.dtype, v.specialized, v.sqrt, v.opt, v.extra, v.flags,
                    r["status"], r["reps"]]
            w.writerow(base + [fmt(r.get(c)) for c in cols[len(base):]])


def fmt(x):
    if isinstance(x, float):
        return "" if math.isnan(x) else f"{x:.6g}"
    return "" if x is None else x


def write_hotspots():
    out = []
    for v in all_variants():
        f = v.prof_dir() / "top.csv"
        if f.exists():
            for row in csv.DictReader(f.open()):
                out.append([v.scenario, v.id, row["rank"], row["function"], row["pct_time"],
                            row["self_s"], row["calls"]])
    with (OUT / "hotspots.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "variant", "rank", "function", "pct_time", "self_s", "calls"])
        w.writerows(out)
    return out


# ---------------------------------------------------------------- charts
def code_variants(scenario):
    seen = []
    for v in all_variants():
        if v.scenario == scenario and v.code not in seen:
            seen.append(v.code)
    return seen


def flag_sets():
    return [o + (f"+{e}" if e else "") for e in EXTRAS for o in OPTS]


def heatmap(rows, scenario, metric, title, fname):
    codes, flags = code_variants(scenario), flag_sets()
    grid = [[math.nan] * len(flags) for _ in codes]
    for r in rows.values():
        v = r["variant"]
        if v.scenario == scenario:
            grid[codes.index(v.code)][flags.index(v.flags_slug)] = r[metric]
    vals = [x for row in grid for x in row if not math.isnan(x)]
    if not vals:
        return
    span = max(max(vals), 1 / min(vals), 1.05)
    fig, ax = plt.subplots(figsize=(13, 0.32 * len(codes) + 2.2))
    im = ax.imshow(grid, cmap=DIVERGING, norm=LogNorm(1 / span, span), aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(flags)), flags, rotation=60, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(codes)), codes, fontsize=7.5)
    for i, row in enumerate(grid):
        for j, x in enumerate(row):
            label = "×" if math.isnan(x) else f"{x:.2f}"
            # dark text on light cells, light text on the dark ends of the scale
            dark = not math.isnan(x) and abs(math.log(x)) > 0.6 * math.log(span)
            ax.text(j, i, label, ha="center", va="center", fontsize=5.5,
                    color=SURFACE if dark else TEXT)
    for j in range(len(OPTS), len(flags), len(OPTS)):  # separate the -march/-flto groups
        ax.axvline(j - 0.5, color=SURFACE, linewidth=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("× vs baseline (double, sqrt, -O2); >1 is better", color=TEXT2)
    cb.outline.set_visible(False)
    ax.set_title(f"{title} — {scenario.upper()}   (× = not measured or crashed)", loc="left")
    fig.tight_layout()
    fig.savefig(CHARTS / fname, dpi=160)
    plt.close(fig)


def by_option(rows, scenario, options, fixed, label_of, title, fname):
    """Small multiples (time, energy, power, frequency) vs a GCC option, one series per k."""
    panels = [("knn_s", "kNN time (s)"), ("knn_pkg_J", "kNN energy, package (J)"),
              ("knn_pkg_W", "Average power, package (W)"), ("knn_freq_MHz", "Average frequency (MHz)")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (metric, ylabel) in zip(axes, panels):
        for color, k in zip(SERIES, KS):
            ys = []
            for o in options:
                r = rows.get(fixed(scenario, k, o))
                ys.append(r[metric] if r and r["status"] == "ok" else math.nan)
            ax.plot(range(len(options)), ys, color=color, linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"k = {k}")
        ax.set_xticks(range(len(options)), [label_of(o) for o in options], rotation=40, ha="right")
        ax.set_title(ylabel, loc="left")
        ax.set_ylim(bottom=0)
    axes[0].legend(loc="lower left")
    fig.suptitle(f"{title} — {scenario.upper()}", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHARTS / fname, dpi=160)
    plt.close(fig)


def scatter(rows, scenario, fname):
    fig, ax = plt.subplots(figsize=(7, 5))
    for color, dt in zip(SERIES, ["double", "float"]):
        pts = [(r["knn_s"], r["knn_pkg_J"]) for r in rows.values()
               if r["variant"].scenario == scenario and r["variant"].dtype == dt and r["status"] == "ok"]
        if pts:
            ax.scatter(*zip(*pts), s=22, color=color, edgecolor=SURFACE, linewidth=0.8, label=dt, alpha=0.9)
    for k in KS:
        b = rows.get(baseline_of(type("V", (), {"scenario": scenario, "k": k})))
        if b and b["status"] == "ok":
            ax.scatter([b["knn_s"]], [b["knn_pkg_J"]], s=90, facecolor="none", edgecolor=TEXT, linewidth=1.5)
            ax.annotate(f"baseline k={k}", (b["knn_s"], b["knn_pkg_J"]), xytext=(6, 4),
                        textcoords="offset points", fontsize=8, color=TEXT2)
    ax.set_xlabel("kNN time (s)")
    ax.set_ylabel("kNN energy, package (J)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(title="data type", loc="upper left")
    ax.set_title(f"Energy vs time, all variants — {scenario.upper()}", loc="left")
    fig.tight_layout()
    fig.savefig(CHARTS / fname, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- markdown
def write_markdown(rows, hotspots):
    L = ["# Checkpoint 1.2 results", "",
         "Generated by `scripts/analyze.py`. Medians of the repetitions; kNN phase only.",
         "Baseline: double, sqrt, specialized (k=3), `-O2`. Speedup and energy gain are vs that baseline.", ""]
    crashed = [r["variant"] for r in rows.values() if r["status"] == "crashed"]
    for s in SCENARIOS:
        rs = [r for r in rows.values() if r["variant"].scenario == s and r["status"] == "ok"]
        if not rs:
            continue
        L += [f"## {s.upper()}", "", "### Baseline and best variants per k", "",
              "| k | variant | flags | time (s) | speedup | energy (J) | energy gain | power (W) | freq (MHz) | accuracy (%) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for k in KS:
            rk = [r for r in rs if r["variant"].k == k]
            if not rk:
                continue
            picks = [("baseline", rows.get(baseline_of(rk[0]["variant"]))),
                     ("fastest", min(rk, key=lambda r: r["knn_s"])),
                     ("least energy", min(rk, key=lambda r: r["knn_pkg_J"]))]
            for name, r in picks:
                if r and r["status"] == "ok":
                    v = r["variant"]
                    L.append(f"| {k} | {name}: `{v.code}` | `{v.flags}` | {r['knn_s']:.3f} | {r['speedup']:.2f} | "
                             f"{r['knn_pkg_J']:.2f} | {r['energy_gain']:.2f} | {r['knn_pkg_W']:.1f} | "
                             f"{r['knn_freq_MHz']:.0f} | {r['accuracy']:.2f} |")
        L += ["", "### Median speedup per flag set (over all code variants)", "",
              "| flags | " + " | ".join(f"k={k}" for k in KS) + " |", "|---|" + "---|" * len(KS)]
        for fl in flag_sets():
            cells = []
            for k in KS:
                xs = [r["speedup"] for r in rs if r["variant"].k == k and r["variant"].flags_slug == fl
                      and not math.isnan(r["speedup"])]
                cells.append(f"{statistics.median(xs):.2f}" if xs else "–")
            L.append(f"| `{fl}` | " + " | ".join(cells) + " |")
        acc = sorted({(r["variant"].code, round(r["accuracy"], 2)) for r in rs if abs(r["acc_delta"]) > 1e-9})
        if acc:
            L += ["", "Variants whose accuracy differs from the baseline: "
                  + ", ".join(f"`{c}` ({a} %)" for c, a in acc)]
        L.append("")
    if crashed:
        L += ["## Crashed variants", ""] + [f"- {v.scenario}/`{v.id}`" for v in crashed] + [""]
    if hotspots:
        L += ["## Top-5 hotspots (gprof), baseline flags `-O2`", "",
              "Full list for every variant in `hotspots.csv`; reports in `profiling/<scenario>/<variant>/`.", ""]
        for v in all_variants():
            if v.opt == "O2" and v.extra == "" and (v.prof_dir() / "top.csv").exists():
                top = [h for h in hotspots if h[0] == v.scenario and h[1] == v.id]
                L.append(f"- {v.scenario}/`{v.code}`: " + ", ".join(f"{h[3]} {float(h[4]):.0f}%" for h in top))
        L.append("")
    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")


def main():
    rows = load()
    if not rows:
        sys.exit(f"no measurements in {OUT / 'raw'} (run scripts/experiments.py run first)")
    CHARTS.mkdir(parents=True, exist_ok=True)
    write_summary(rows)
    hotspots = write_hotspots()
    for s in SCENARIOS:
        if not any(r["variant"].scenario == s for r in rows.values()):
            continue
        heatmap(rows, s, "speedup", "kNN speedup", f"{s}_speedup_heatmap.png")
        heatmap(rows, s, "energy_gain", "kNN energy gain", f"{s}_energy_heatmap.png")
        base_code = lambda k: ("double", 1 if k == 3 else 0, "sqrt")  # noqa: E731
        by_option(rows, s, list(OPTS), lambda s_, k, o: (s_, k, *base_code(k), o, ""),
                  lambda o: OPTS[o], "Optimisation level (original code)", f"{s}_opt_levels.png")
        by_option(rows, s, list(EXTRAS), lambda s_, k, e: (s_, k, *base_code(k), "O3", e),
                  lambda e: ("-O3 " + EXTRAS[e]).strip(), "-march=native / -flto at -O3 (original code)",
                  f"{s}_extras.png")
        scatter(rows, s, f"{s}_time_vs_energy.png")
    write_markdown(rows, hotspots)
    n_ok = sum(r["status"] == "ok" for r in rows.values())
    print(f"analyze: {len(rows)} variants ({n_ok} ok) -> {OUT.relative_to(OUT.parent.parent)}/"
          "{summary.csv, summary.md, hotspots.csv, charts/}")


if __name__ == "__main__":
    main()
