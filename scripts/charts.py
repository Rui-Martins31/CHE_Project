"""Charts from results/experiments.csv."""

from experiment import DTYPES, K_VALUES, OPT_LEVELS, RESULTS_DIR, load_rows

CHART_DIR = RESULTS_DIR / "charts"


def _number(row, key):
    value = row.get(key, "")
    if value in ("", None):
        return None
    return float(value)


def _flag(row, key):
    return row.get(key, "0") in ("1", "True", "true")


def _reference_sqrt(dtype):
    return "sqrtf" if dtype == "float" else "sqrt"


def _specialized_for(k):
    return "1" if int(k) == 3 else "0"


def _matching(rows, **expected):
    found = []
    for row in rows:
        if row.get("status") == "fail":
            continue
        if all(str(row.get(key)) == str(value) for key, value in expected.items()):
            found.append(row)
    return found


def _lookup(rows, opt_level, k, dtype):
    matches = [
        row
        for row in rows
        if row["opt_level"] == opt_level and row["k"] == str(k) and row["dtype"] == dtype
    ]
    if not matches:
        return None
    return _number(matches[0], "time_median_s")


def _write_grouped(path, title, xlabel, ylabel, group_labels, series):
    import matplotlib.pyplot as plt

    count = len(series)
    width = 0.8 / count
    figure, axis = plt.subplots(figsize=(12, 5))
    positions = list(range(len(group_labels)))
    for index, (name, values) in enumerate(series):
        offset = (index - (count - 1) / 2) * width
        heights = [float("nan") if value is None else value for value in values]
        axis.bar([position + offset for position in positions], heights, width, label=name)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_xticks(positions, group_labels, rotation=25, ha="right")
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _base_rows(rows, scenario, dtype):
    return _matching(
        rows,
        scenario=scenario,
        dtype=dtype,
        sqrt_mode=_reference_sqrt(dtype),
        march="0",
        lto="0",
    )


def _with_default_spec(rows):
    return [row for row in rows if row["specialized"] == _specialized_for(row["k"])]


def write_charts(rows=None):
    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        print("matplotlib is not installed; CSV is written, charts were skipped")
        return []

    if rows is None:
        rows = load_rows()
    written = []
    written.extend(_time_by_opt(rows))
    written.extend(_accuracy_by_opt(rows))
    written.extend(_sqrt_chart(rows))
    written.extend(_specialized_chart(rows))
    written.extend(_march_lto_chart(rows))
    return written


def _time_by_opt(rows):
    paths = []
    for scenario in sorted({row["scenario"] for row in rows}):
        for dtype, _code, _dt in DTYPES:
            selected = _with_default_spec(_base_rows(rows, scenario, dtype))
            if not selected:
                continue
            series = []
            for k in K_VALUES:
                series.append(
                    (
                        f"K={k}",
                        [_lookup(selected, opt, k, dtype) for opt in OPT_LEVELS],
                    )
                )
            if not any(value is not None for _name, values in series for value in values):
                continue
            path = CHART_DIR / f"time_{scenario}_{dtype}.png"
            _write_grouped(
                path,
                f"{scenario}, {dtype}: median execution time",
                "GCC optimization",
                "seconds",
                OPT_LEVELS,
                series,
            )
            paths.append(path)
    return paths


def _accuracy_by_opt(rows):
    paths = []
    for scenario in sorted({row["scenario"] for row in rows}):
        for dtype, _code, _dt in DTYPES:
            selected = _with_default_spec(_base_rows(rows, scenario, dtype))
            if not selected:
                continue
            series = []
            for k in K_VALUES:
                values = []
                for opt in OPT_LEVELS:
                    matches = [
                        row
                        for row in selected
                        if row["opt_level"] == opt and row["k"] == str(k)
                    ]
                    values.append(_number(matches[0], "accuracy_pct") if matches else None)
                series.append((f"K={k}", values))
            if not any(value is not None for _name, values in series for value in values):
                continue
            path = CHART_DIR / f"accuracy_{scenario}_{dtype}.png"
            _write_grouped(
                path,
                f"{scenario}, {dtype}: accuracy",
                "GCC optimization",
                "accuracy (%)",
                OPT_LEVELS,
                series,
            )
            paths.append(path)
    return paths


def _sqrt_chart(rows):
    paths = []
    for scenario in sorted({row["scenario"] for row in rows}):
        for dtype, _code, _dt in DTYPES:
            modes = ["sqrt", "none"] if dtype == "double" else ["sqrtf", "none"]
            selected = _with_default_spec(
                _matching(
                    rows,
                    scenario=scenario,
                    dtype=dtype,
                    opt_level="-O2",
                    march="0",
                    lto="0",
                )
            )
            series = []
            for k in K_VALUES:
                values = []
                for mode in modes:
                    matches = [
                        row
                        for row in selected
                        if row["k"] == str(k) and row["sqrt_mode"] == mode
                    ]
                    values.append(_number(matches[0], "time_median_s") if matches else None)
                series.append((f"K={k}", values))
            if not any(value is not None for _name, values in series for value in values):
                continue
            path = CHART_DIR / f"sqrt_{scenario}_{dtype}.png"
            _write_grouped(
                path,
                f"{scenario}, {dtype}, -O2: sqrt mode",
                "Euclidean distance",
                "seconds",
                modes,
                series,
            )
            paths.append(path)
    return paths


def _specialized_chart(rows):
    selected = _matching(rows, k="3", opt_level="-O2", march="0", lto="0")
    if not selected:
        return []
    scenarios = sorted({row["scenario"] for row in selected})
    dtypes = [dtype for dtype, _code, _dt in DTYPES]
    labels = []
    spec0 = []
    spec1 = []
    for scenario in scenarios:
        for dtype in dtypes:
            labels.append(f"{scenario}\n{dtype}")
            for bucket, specialized in ((spec0, "0"), (spec1, "1")):
                matches = [
                    row
                    for row in selected
                    if row["scenario"] == scenario
                    and row["dtype"] == dtype
                    and row["specialized"] == specialized
                    and row["sqrt_mode"] == _reference_sqrt(dtype)
                ]
                bucket.append(_number(matches[0], "time_median_s") if matches else None)
    if not any(value is not None for value in spec0 + spec1):
        return []
    path = CHART_DIR / "specialized_k3.png"
    _write_grouped(
        path,
        "K=3, -O2: specialized vs generic classify",
        "Scenario and data type",
        "seconds",
        labels,
        [("specialized=0", spec0), ("specialized=1", spec1)],
    )
    return [path]


def _march_lto_chart(rows):
    paths = []
    labels = ["-O3", "-O3 native", "-O3 lto", "-O3 native lto"]
    flag_sets = ((False, False), (True, False), (False, True), (True, True))
    for scenario in sorted({row["scenario"] for row in rows}):
        for dtype, _code, _dt in DTYPES:
            selected = _with_default_spec(
                _matching(rows, scenario=scenario, dtype=dtype, opt_level="-O3")
            )
            selected = [
                row
                for row in selected
                if row["sqrt_mode"] == _reference_sqrt(dtype)
            ]
            series = []
            for k in K_VALUES:
                values = []
                for march, lto in flag_sets:
                    matches = [
                        row
                        for row in selected
                        if row["k"] == str(k)
                        and _flag(row, "march") == march
                        and _flag(row, "lto") == lto
                    ]
                    values.append(_number(matches[0], "time_median_s") if matches else None)
                series.append((f"K={k}", values))
            if not any(value is not None for _name, values in series for value in values):
                continue
            path = CHART_DIR / f"march_lto_{scenario}_{dtype}.png"
            _write_grouped(
                path,
                f"{scenario}, {dtype}: -march=native and -flto at -O3",
                "GCC flags",
                "seconds",
                labels,
                series,
            )
            paths.append(path)
    return paths

