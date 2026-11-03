"""One HAR-kNN build and one execution.

`run_config.py` calls this for a single configuration.
`run_experiments.py` calls `measure` once per variant, which runs the
binary 10 times and profiles it once.
"""

import csv
import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KNN_DIR = ROOT / "knn"
RESULTS_DIR = ROOT / "results"
PROFILE_DIR = RESULTS_DIR / "profiles"

SCENARIOS = ("wisdm", "pamap2")
K_VALUES = (3, 5, 20)
# Professor note: D={0,1} => {double, float}. The Makefile uses DT=1/2.
DTYPES = (("double", 0, 1), ("float", 1, 2))
SQRT_MODES = ("sqrt", "sqrtf", "none")
OPT_LEVELS = (
    "-Os",
    "-O1",
    "-O2",
    "-O3",
    "-Ofast",
    "-O3 -fno-tree-vectorize",
    "-O3 -fno-inline",
)

REPEATS = 10
MARKER = "kNN done."
TIME_RE = re.compile(r"Total execution time:\s+([0-9.]+) s")
ACCURACY_RE = re.compile(r"kNN: accuracy = ([0-9.]+) %")
WRONG_RE = re.compile(r"kNN: number of classifications wrong = (\d+)")
RIGHT_RE = re.compile(r"kNN: number of classifications right = (\d+)")

CSV_FIELDS = (
    "scenario",
    "k",
    "dtype",
    "dtype_code",
    "specialized",
    "sqrt_mode",
    "opt_level",
    "march",
    "lto",
    "opt_flags",
    "status",
    "accuracy_pct",
    "wrong",
    "right",
    "time_median_s",
    "times_s",
    "hotspot_1",
    "hotspot_1_pct",
    "hotspot_2",
    "hotspot_2_pct",
    "hotspot_3",
    "hotspot_3_pct",
    "hotspot_4",
    "hotspot_4_pct",
    "hotspot_5",
    "hotspot_5_pct",
    "error",
)


@dataclass(frozen=True)
class Config:
    scenario: str
    k: int
    dtype: str
    dtype_code: int
    dt: int
    specialized: int
    sqrt_mode: str
    opt_level: str
    march: bool = False
    lto: bool = False

    @property
    def use_sqrt(self):
        return 0 if self.sqrt_mode == "none" else 1

    @property
    def math_type(self):
        # MATH_TYPE=1 picks sqrt for double and sqrtf for float.
        return 1

    @property
    def opt_flags(self):
        flags = [self.opt_level]
        if self.march:
            flags.append("-march=native")
        if self.lto:
            flags.append("-flto")
        return " ".join(flags)

    @property
    def slug(self):
        opt = self.opt_level.lstrip("-").replace(" ", "_")
        parts = [
            self.scenario,
            f"k{self.k}",
            self.dtype,
            f"spec{self.specialized}",
            self.sqrt_mode,
            opt,
        ]
        if self.march:
            parts.append("native")
        if self.lto:
            parts.append("lto")
        return "_".join(parts)


def rejection_reason(k, specialized, dtype, sqrt_mode):
    """Combinations the provided sources do not implement.

    specialized=1 is compiled only for K=3. With MATH_TYPE=1, sqrt() is
    used for double and sqrtf() for float, so the other pairing never
    changes which call is emitted.
    """
    if specialized == 1 and k != 3:
        return "specialized=1 exists only for K=3"
    if sqrt_mode == "sqrt" and dtype != "double":
        return "sqrt() is used only for double; float uses sqrtf()"
    if sqrt_mode == "sqrtf" and dtype != "float":
        return "sqrtf() is used only for float; double uses sqrt()"
    return None


def sqrt_modes_for(dtype):
    if dtype == "double":
        return ("sqrt", "none")
    return ("sqrtf", "none")


def configurations(scenario=None, k=None, dtype=None):
    scenarios = (scenario,) if scenario else SCENARIOS
    ks = (k,) if k is not None else K_VALUES
    dtypes = [item for item in DTYPES if dtype is None or item[0] == dtype]
    for scenario_name in scenarios:
        for k_value in ks:
            specialized_values = (0, 1) if k_value == 3 else (0,)
            for specialized in specialized_values:
                for dtype_name, dtype_code, dt in dtypes:
                    for sqrt_mode in sqrt_modes_for(dtype_name):
                        for opt_level in OPT_LEVELS:
                            for march in (False, True):
                                for lto in (False, True):
                                    yield Config(
                                        scenario_name,
                                        k_value,
                                        dtype_name,
                                        dtype_code,
                                        dt,
                                        specialized,
                                        sqrt_mode,
                                        opt_level,
                                        march,
                                        lto,
                                    )


def binary_path():
    for name in ("harknn.exe", "harknn"):
        path = KNN_DIR / "bin" / name
        if path.is_file():
            return path
    return KNN_DIR / "bin" / "harknn"


def _make(target, config):
    command = [
        "make",
        target,
        f"SCENARIO={config.scenario}",
        f"K={config.k}",
        f"DT={config.dt}",
        f"SPECIALIZED={config.specialized}",
        f"USE_SQRT={config.use_sqrt}",
        f"MATH_TYPE={config.math_type}",
        f"OPT={config.opt_flags}",
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
    return binary_path()


def build(config):
    return _make("comp", config)


def build_profile(config):
    return _make("comprofiling", config)


def _parse_program(text):
    time_match = TIME_RE.search(text)
    accuracy_match = ACCURACY_RE.search(text)
    if time_match is None or accuracy_match is None or MARKER not in text:
        raise RuntimeError("program output has no kNN summary")
    wrong = WRONG_RE.search(text)
    right = RIGHT_RE.search(text)
    return {
        "time_s": float(time_match.group(1)),
        "accuracy_pct": float(accuracy_match.group(1)),
        "wrong": int(wrong.group(1)) if wrong else None,
        "right": int(right.group(1)) if right else None,
    }


def execute(binary):
    """Run one binary once and return its reported time and accuracy."""
    completed = subprocess.run(
        [str(binary)],
        cwd=KNN_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout[-4000:])
    return _parse_program(completed.stdout)


def top_hotspots(gprof_text, count=5):
    lines = gprof_text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if "name" in line and "time" in line:
            start = index + 1
            break
    if start is None:
        return []
    found = []
    for line in lines[start:]:
        if not line.strip():
            break
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            percent = float(parts[0])
        except ValueError:
            continue
        found.append((parts[-1], percent))
        if len(found) == count:
            break
    return found


def profile(config):
    binary = build_profile(config)
    run_dir = KNN_DIR / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    gmon = run_dir / "gmon.out"
    if gmon.exists():
        gmon.unlink()
    completed = subprocess.run(
        [str(binary.resolve())],
        cwd=run_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout[-4000:])
    gprof = subprocess.run(
        ["gprof", "-b", "-p", str(binary.resolve())],
        cwd=run_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    if gprof.returncode != 0:
        raise RuntimeError(gprof.stdout[-4000:])
    hotspots = top_hotspots(gprof.stdout)
    profile_path = PROFILE_DIR / f"{config.slug}.txt"
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(gprof.stdout, encoding="utf-8")
    return hotspots


def _median(values):
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return statistics.median(usable)


def _join(values):
    return ";".join("" if value is None else f"{value:.6g}" for value in values)


def measure(config, repeats=REPEATS, with_profile=True):
    """Build `config` once, run it `repeats` times, and profile it once."""
    row = {
        "scenario": config.scenario,
        "k": config.k,
        "dtype": config.dtype,
        "dtype_code": config.dtype_code,
        "specialized": config.specialized,
        "sqrt_mode": config.sqrt_mode,
        "opt_level": config.opt_level,
        "march": int(config.march),
        "lto": int(config.lto),
        "opt_flags": config.opt_flags,
        "status": "ok",
        "error": "",
    }
    for name, _percent in (
        (f"hotspot_{index}", None) for index in range(1, 6)
    ):
        row[name] = ""
        row[f"{name}_pct"] = ""

    try:
        binary = build(config)
        runs = [execute(binary) for _ in range(repeats)]
    except (RuntimeError, OSError) as error:
        row["status"] = "fail"
        row["error"] = str(error).replace("\n", " ")[-500:]
        return row

    row["accuracy_pct"] = runs[0]["accuracy_pct"]
    row["wrong"] = runs[0]["wrong"]
    row["right"] = runs[0]["right"]
    row["time_median_s"] = _median([run["time_s"] for run in runs])
    row["times_s"] = _join([run["time_s"] for run in runs])

    if with_profile:
        try:
            for index, (name, percent) in enumerate(profile(config), start=1):
                row[f"hotspot_{index}"] = name
                row[f"hotspot_{index}_pct"] = percent
        except (RuntimeError, OSError) as error:
            row["status"] = "profile_fail"
            row["error"] = str(error).replace("\n", " ")[-500:]
    return row


def format_row(row):
    lines = [
        f"{row['scenario']}  K={row['k']}  {row['dtype']}  "
        f"specialized={row['specialized']}  sqrt={row['sqrt_mode']}",
        f"gcc: {row['opt_flags']}",
        f"status: {row['status']}",
    ]
    if row.get("accuracy_pct") not in (None, ""):
        lines.append(
            f"accuracy: {row['accuracy_pct']}%  "
            f"wrong={row['wrong']}  right={row['right']}"
        )
        lines.append(f"median execution time: {row['time_median_s']} s")
        lines.append(f"runs: {row['times_s']}")
        hotspots = []
        for index in range(1, 6):
            name = row.get(f"hotspot_{index}")
            if name:
                hotspots.append(f"{name} ({row.get(f'hotspot_{index}_pct')}%)")
        if hotspots:
            lines.append("top functions: " + ", ".join(hotspots))
    if row.get("error"):
        lines.append(row["error"])
    return "\n".join(lines)


def csv_path():
    return RESULTS_DIR / "experiments.csv"


def load_rows():
    path = csv_path()
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_row(row):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = csv_path()
    new_file = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(
            {
                field: "" if row.get(field) is None else row.get(field, "")
                for field in CSV_FIELDS
            }
        )

