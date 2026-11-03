"""Record optimization attempts and tune the OpenMP batch size and thread count.

Writes results/optimizations.csv. The compiler sweep in results/experiments.csv
is left untouched. Historical rows are the one-at-a-time attempts already run
on AC power. The grid at the bottom is measured by this script.
"""

import csv
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment import KNN_DIR, RESULTS_DIR, TIME_RE, WRONG_RE, RIGHT_RE, ACCURACY_RE

OUT = RESULTS_DIR / "optimizations.csv"
REPEATS = 7
OPT = "-Ofast -march=native -flto"
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
THREADS = (1, 2, 4, 6, 8, 16)
EXPECTED_WRONG = {("pamap2", 3): 236, ("pamap2", 5): 216, ("wisdm", 3): 791}

FIELDS = [
    "attempt",
    "outcome",
    "scenario",
    "k",
    "dtype",
    "specialized",
    "sqrt_mode",
    "opt_flags",
    "threads",
    "batch",
    "repeats",
    "time_median_s",
    "times_s",
    "accuracy_pct",
    "wrong",
    "right",
    "charge_pct",
    "note",
]

# One-at-a-time attempts, in the order they were measured. Charge is the
# Win32 battery estimate at that run. All of them were on AC (BatteryStatus 2).
HISTORY = [
    ("baseline", "reference", "pamap2", 3, 1, "", "", 0.0292, "0.0318;0.027;0.0322;0.0273;0.0285;0.0292;0.0299", "76.59", 236, 772, 39, "Original classifier. No OpenMP."),
    ("baseline", "reference", "pamap2", 5, 0, "", "", 0.0535, "0.0547;0.0515;0.0582;0.0524;0.0548;0.0535;0.0529", "78.57", 216, 792, 39, "Original classifier. No OpenMP."),
    ("baseline", "reference", "wisdm", 3, 1, "", "", 0.0679, "0.0696;0.0667;0.0679;0.0662;0.0674;0.0681;0.07", "74.99", 791, 2372, 39, "Original classifier. No OpenMP."),
    ("local_query", "reverted", "pamap2", 3, 1, "", "", 0.0312, "0.0311;0.0348;0.0312;0.0296;0.0339;0.0275;0.0351", "76.59", 236, 772, 39, "Local query[NUM_FEATURES] and a compile-time feature loop. Slower, reverted."),
    ("local_query", "reverted", "pamap2", 5, 0, "", "", 0.0586, "0.0594;0.0562;0.058;0.0607;0.0586;0.056;0.0651", "78.57", 216, 792, 39, "Local query[NUM_FEATURES] and a compile-time feature loop. Slower, reverted."),
    ("local_query", "reverted", "wisdm", 3, 1, "", "", 0.0745, "0.0748;0.074;0.073;0.0759;0.0745;0.0771;0.0711", "74.99", 791, 2372, 39, "Local query[NUM_FEATURES] and a compile-time feature loop. Slower, reverted."),
    ("fused_topk", "kept", "pamap2", 3, 1, "", "", 0.0223, "0.0218;0.0223;0.0241;0.0222;0.022;0.023;0.0242", "76.59", 236, 772, 41, "Running top-k inside the distance loop. sqrtf still applied before the comparison."),
    ("fused_topk", "kept", "pamap2", 5, 0, "", "", 0.0249, "0.024;0.0244;0.0265;0.0234;0.025;0.0254;0.0249", "78.57", 216, 792, 41, "Running top-k inside the distance loop. sqrtf still applied before the comparison."),
    ("fused_topk", "kept", "wisdm", 3, 1, "", "", 0.0496, "0.0494;0.0519;0.0517;0.0476;0.0504;0.0496;0.0494", "74.99", 791, 2372, 41, "Running top-k inside the distance loop. sqrtf still applied before the comparison."),
    ("squared_distance", "reverted", "pamap2", 3, 1, "", "", 0.0230, "0.0214;0.0239;0.023;0.025;0.0216;0.0221;0.0244", "76.59", 236, 772, 42, "On top of fused top-k. sqrtf only on the k kept distances. No gain, reverted."),
    ("squared_distance", "reverted", "pamap2", 5, 0, "", "", 0.0290, "0.0311;0.0269;0.0311;0.029;0.0274;0.029;0.0327", "78.57", 216, 792, 42, "On top of fused top-k. sqrtf only on the k kept distances. Slower, reverted."),
    ("squared_distance", "reverted", "wisdm", 3, 1, "", "", 0.0490, "0.0607;0.0497;0.0483;0.0483;0.049;0.0474;0.0505", "74.99", 791, 2372, 42, "On top of fused top-k. sqrtf only on the k kept distances. No gain, reverted."),
    ("openmp_per_instance", "reverted", "pamap2", 3, 1, 2, 1, 0.0376, "0.0339;0.043;0.0406;0.0376;0.0454;0.0353;0.0372", "76.59", 236, 772, 43, "Fused top-k plus one OpenMP region per instance. Fork/join cost dominated, reverted as a standalone step."),
    ("openmp_per_instance", "reverted", "pamap2", 5, 0, 2, 1, 0.0400, "0.0384;0.0449;0.0334;0.05;0.0351;0.04;0.0502", "78.57", 216, 792, 43, "Fused top-k plus one OpenMP region per instance. Fork/join cost dominated, reverted as a standalone step."),
    ("openmp_per_instance", "reverted", "wisdm", 3, 1, 2, 1, 0.0915, "0.0988;0.0925;0.0915;0.0856;0.0958;0.0883;0.0864", "74.99", 791, 2372, 43, "Fused top-k plus one OpenMP region per instance. Fork/join cost dominated, reverted as a standalone step."),
    ("batch16_threads2", "superseded", "pamap2", 3, 1, 2, 16, 0.0199, "0.0202;0.0199;0.0207;0.0189;0.021;0.0198;0.0192", "76.59", 236, 772, 45, "16 queries per training-set pass, 2 threads. Batching stayed; thread count was raised afterwards."),
    ("batch16_threads2", "superseded", "pamap2", 5, 0, 2, 16, 0.0208, "0.0208;0.0214;0.0194;0.0195;0.0185;0.0219;0.0225", "78.57", 216, 792, 45, "16 queries per training-set pass, 2 threads. Batching stayed; thread count was raised afterwards."),
    ("batch16_threads2", "superseded", "wisdm", 3, 1, 2, 16, 0.0454, "0.046;0.0446;0.0454;0.0451;0.0493;0.0433;0.0481", "74.99", 791, 2372, 45, "16 queries per training-set pass, 2 threads. Batching stayed; thread count was raised afterwards."),
    ("thread_sweep_batch16", "not_chosen", "pamap2", 3, 1, 2, 16, 0.0195, "0.0187;0.0192;0.0194;0.0195;0.0196;0.0204;0.0204", "76.59", 236, 772, 91, "Same batch-16 code, thread count swept. 2 threads."),
    ("thread_sweep_batch16", "not_chosen", "pamap2", 5, 0, 2, 16, 0.0198, "0.0196;0.0196;0.0197;0.0198;0.0204;0.0205;0.0213", "78.57", 216, 792, 91, "Same batch-16 code, thread count swept. 2 threads."),
    ("thread_sweep_batch16", "not_chosen", "wisdm", 3, 1, 2, 16, 0.0430, "0.0412;0.0424;0.0429;0.0430;0.0438;0.0449;0.0450", "74.99", 791, 2372, 91, "Same batch-16 code, thread count swept. 2 threads."),
    ("thread_sweep_batch16", "not_chosen", "pamap2", 3, 1, 4, 16, 0.0146, "0.0124;0.0144;0.0145;0.0146;0.0149;0.0167;0.0199", "76.59", 236, 772, 91, "Same batch-16 code, thread count swept. 4 threads."),
    ("thread_sweep_batch16", "not_chosen", "pamap2", 5, 0, 4, 16, 0.0143, "0.0129;0.0132;0.0142;0.0143;0.0149;0.0152;0.0178", "78.57", 216, 792, 91, "Same batch-16 code, thread count swept. 4 threads."),
    ("thread_sweep_batch16", "superseded", "wisdm", 3, 1, 4, 16, 0.0315, "0.0306;0.0307;0.0312;0.0315;0.0316;0.0332;0.0348", "74.99", 791, 2372, 91, "Best WISDM point in the batch-16 thread sweep. Superseded by batch 128 and 6 threads."),
    ("thread_sweep_batch16", "superseded", "pamap2", 3, 1, 6, 16, 0.0138, "0.0113;0.0131;0.0132;0.0138;0.0148;0.0153;0.0161", "76.59", 236, 772, 91, "Best PAMAP2 K=3 point in the batch-16 thread sweep. Superseded by batch 128 and 6 threads."),
    ("thread_sweep_batch16", "superseded", "pamap2", 5, 0, 6, 16, 0.0138, "0.0110;0.0118;0.0124;0.0138;0.0158;0.0162;0.0175", "78.57", 216, 792, 91, "Best PAMAP2 K=5 point in the batch-16 thread sweep. Superseded by batch 128 and 6 threads."),
    ("thread_sweep_batch16", "not_chosen", "wisdm", 3, 1, 6, 16, 0.0323, "0.0250;0.0286;0.0304;0.0323;0.0328;0.0336;0.0361", "74.99", 791, 2372, 91, "Same batch-16 code, thread count swept. 6 threads."),
    ("thread_sweep_batch16", "not_chosen", "pamap2", 3, 1, 8, 16, 0.0161, "0.0135;0.0140;0.0157;0.0161;0.0163;0.0167;0.0196", "76.59", 236, 772, 91, "Same batch-16 code, thread count swept. 8 threads."),
    ("thread_sweep_batch16", "not_chosen", "pamap2", 5, 0, 8, 16, 0.0170, "0.0153;0.0161;0.0167;0.0170;0.0171;0.0183;0.0184", "78.57", 216, 792, 91, "Same batch-16 code, thread count swept. 8 threads."),
    ("thread_sweep_batch16", "not_chosen", "wisdm", 3, 1, 8, 16, 0.0371, "0.0345;0.0359;0.0368;0.0371;0.0375;0.0383;0.0425", "74.99", 791, 2372, 91, "Same batch-16 code, thread count swept. 8 threads."),
    ("thread_sweep_batch16", "not_chosen", "pamap2", 3, 1, 16, 16, 0.0172, "0.0158;0.0168;0.0169;0.0172;0.0196;0.0201;0.0206", "76.59", 236, 772, 91, "Same batch-16 code, thread count swept. 16 threads."),
    ("thread_sweep_batch16", "not_chosen", "pamap2", 5, 0, 16, 16, 0.0181, "0.0146;0.0154;0.0176;0.0181;0.0187;0.0188;0.0188", "78.57", 216, 792, 91, "Same batch-16 code, thread count swept. 16 threads."),
    ("thread_sweep_batch16", "not_chosen", "wisdm", 3, 1, 16, 16, 0.0358, "0.0346;0.0351;0.0354;0.0358;0.0380;0.0392;0.0412", "74.99", 791, 2372, 91, "Same batch-16 code, thread count swept. 16 threads."),
]


def history_row(item):
    attempt, outcome, scenario, k, specialized, threads, batch, median, runs, accuracy, wrong, right, charge, note = item
    return {
        "attempt": attempt,
        "outcome": outcome,
        "scenario": scenario,
        "k": k,
        "dtype": "float",
        "specialized": specialized,
        "sqrt_mode": "sqrtf",
        "opt_flags": OPT,
        "threads": threads,
        "batch": batch,
        "repeats": REPEATS,
        "time_median_s": median,
        "times_s": runs,
        "accuracy_pct": accuracy,
        "wrong": wrong,
        "right": right,
        "charge_pct": charge,
        "note": note,
    }


def charge_percent():
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Battery).EstimatedChargeRemaining"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    text = completed.stdout.strip()
    return text if text.isdigit() else ""


def build(scenario, k, specialized, batch):
    command = [
        "make",
        "comp",
        f"SCENARIO={scenario}",
        f"K={k}",
        "DT=2",
        f"SPECIALIZED={specialized}",
        "USE_SQRT=1",
        "MATH_TYPE=1",
        f"OPT={OPT}",
        f"KNN_BATCH={batch}",
    ]
    completed = subprocess.run(
        command,
        cwd=KNN_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout[-4000:])
    binary = KNN_DIR / "bin" / "harknn.exe"
    if not binary.is_file():
        binary = KNN_DIR / "bin" / "harknn"
    return binary


def run_once(binary, threads):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    completed = subprocess.run(
        [str(binary)],
        cwd=KNN_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout[-4000:])
    text = completed.stdout
    time_match = TIME_RE.search(text)
    wrong = WRONG_RE.search(text)
    right = RIGHT_RE.search(text)
    accuracy = ACCURACY_RE.search(text)
    if time_match is None or wrong is None or accuracy is None:
        raise RuntimeError(text[-2000:])
    return {
        "time_s": float(time_match.group(1)),
        "wrong": int(wrong.group(1)),
        "right": int(right.group(1)) if right else "",
        "accuracy_pct": accuracy.group(1),
    }


def choose(rows):
    """One (batch, threads) pair for every scenario: smallest sum of medians."""
    grid = {}
    for row in rows:
        if row["attempt"] != "batch_thread_sweep" or row["outcome"] == "reverted":
            continue
        key = (int(row["batch"]), int(row["threads"]))
        grid.setdefault(key, {})[(row["scenario"], int(row["k"]))] = float(row["time_median_s"])
    needed = {("pamap2", 3), ("pamap2", 5), ("wisdm", 3)}
    best_key = None
    best_sum = None
    for key, times in grid.items():
        if set(times) != needed:
            continue
        total = sum(times.values())
        if best_sum is None or total < best_sum:
            best_sum = total
            best_key = key
    if best_key is None:
        return None
    batch, threads = best_key
    for row in rows:
        if row["attempt"] != "batch_thread_sweep":
            continue
        if int(row["batch"]) == batch and int(row["threads"]) == threads and row["outcome"] != "reverted":
            row["outcome"] = "kept"
            row["note"] += " Selected: lowest sum of the three medians."
    return best_key, best_sum


def main():
    charge = charge_percent()
    print(f"charge={charge}", flush=True)
    rows = [history_row(item) for item in HISTORY]
    jobs = (
        ("pamap2", 3, 1),
        ("pamap2", 5, 0),
        ("wisdm", 3, 1),
    )
    # Build once per (scenario, batch); threads do not change the binary.
    for scenario, k, specialized in jobs:
        for batch in BATCHES:
            print(f"build {scenario} k={k} batch={batch}", flush=True)
            binary = build(scenario, k, specialized, batch)
            for threads in THREADS:
                runs = [run_once(binary, threads) for _ in range(REPEATS)]
                times = [run["time_s"] for run in runs]
                wrong = runs[0]["wrong"]
                expected = EXPECTED_WRONG[(scenario, k)]
                note = f"Batch and thread sweep on the fused top-k plus OpenMP code. batch={batch}, threads={threads}."
                outcome = "not_chosen"
                if wrong != expected:
                    outcome = "reverted"
                    note += f" Accuracy mismatch: wrong={wrong}, expected {expected}."
                row = {
                    "attempt": "batch_thread_sweep",
                    "outcome": outcome,
                    "scenario": scenario,
                    "k": k,
                    "dtype": "float",
                    "specialized": specialized,
                    "sqrt_mode": "sqrtf",
                    "opt_flags": OPT,
                    "threads": threads,
                    "batch": batch,
                    "repeats": REPEATS,
                    "time_median_s": f"{statistics.median(times):.6g}",
                    "times_s": ";".join(f"{value:.6g}" for value in times),
                    "accuracy_pct": runs[0]["accuracy_pct"],
                    "wrong": wrong,
                    "right": runs[0]["right"],
                    "charge_pct": charge,
                    "note": note,
                }
                rows.append(row)
                print(
                    f"  threads={threads:<2} median={row['time_median_s']} wrong={wrong}",
                    flush=True,
                )
    chosen = choose(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if chosen is None:
        print("no selection", flush=True)
        return 1
    (batch, threads), total = chosen
    print(f"selected batch={batch} threads={threads} sum={total:.6g}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
