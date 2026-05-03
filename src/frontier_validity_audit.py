from __future__ import annotations

import csv
import itertools
import math
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / ".python_packages"
if PACKAGE_DIR.exists():
    sys.path.insert(0, str(PACKAGE_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


INPUT = ROOT / "outputs" / "llm_benchmark_audit_extract" / "table_0.csv"
OUT_DIR = ROOT / "outputs" / "frontier_audit"


def parse_score(value: str) -> float:
    return float(value.strip().replace("%", ""))


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    out = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    xvar = sum((x - xbar) ** 2 for x in xs)
    yvar = sum((y - ybar) ** 2 for y in ys)
    if xvar == 0 or yvar == 0:
        return math.nan
    return sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / math.sqrt(xvar * yvar)


def spearman(rows: list[dict[str, object]]) -> float:
    return pearson(
        ranks([float(row["humaneval"]) for row in rows]),
        ranks([float(row["lcb_v6"]) for row in rows]),
    )


def bootstrap_ci(rows: list[dict[str, object]], draws: int = 5000) -> tuple[float, float]:
    if len(rows) < 5:
        return math.nan, math.nan
    rng = random.Random(20260502)
    values: list[float] = []
    for _ in range(draws):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        rho = spearman(sample)
        if not math.isnan(rho):
            values.append(rho)
    values.sort()
    return values[int(0.025 * len(values))], values[int(0.975 * len(values))]


def inversion_rate(rows: list[dict[str, object]]) -> float:
    inversions = 0
    comparable = 0
    for left, right in itertools.combinations(rows, 2):
        source_delta = float(left["humaneval"]) - float(right["humaneval"])
        transfer_delta = float(left["lcb_v6"]) - float(right["lcb_v6"])
        if source_delta == 0 or transfer_delta == 0:
            continue
        comparable += 1
        if source_delta * transfer_delta < 0:
            inversions += 1
    return inversions / comparable if comparable else math.nan


def topk_overlap(rows: list[dict[str, object]], k: int = 5) -> float:
    if len(rows) < k:
        return math.nan
    source_top = {
        str(row["model"])
        for row in sorted(rows, key=lambda row: float(row["humaneval"]), reverse=True)[:k]
    }
    transfer_top = {
        str(row["model"])
        for row in sorted(rows, key=lambda row: float(row["lcb_v6"]), reverse=True)[:k]
    }
    return len(source_top & transfer_top) / len(source_top | transfer_top)


def row_status(row: dict[str, object]) -> str:
    he = float(row["humaneval"])
    model = str(row["model"]).lower()
    notes = " ".join(str(row[key]).lower() for key in ("he_notes", "lcb_scaffolding", "variant"))
    if "base" in notes or "1.7b" in model or "2b" in model or "3b" in model:
        return "exclude_low_or_base_anchor"
    if he < 90:
        return "exclude_below_frontier_band"
    if row["he_type"] == "Aggregator" or row["lcb_type"] == "Aggregator":
        return "frontier_candidate_aggregator_caveat"
    return "frontier_candidate"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with INPUT.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = {
                "model": raw["Canonical Model Name"],
                "organization": raw["Organization"],
                "family": raw["Family"],
                "variant": raw["Exact Variant"],
                "release": raw["Release"],
                "humaneval": parse_score(raw["HE pass@1"]),
                "he_type": raw["HE Type"],
                "he_notes": raw["HE Notes"],
                "lcb_v6": parse_score(raw["LCB-v6 pass@1"]),
                "lcb_type": raw["LCB Type"],
                "lcb_scaffolding": raw["LCB Scaffolding"],
                "status": raw["Status"],
            }
            row["audit_status"] = row_status(row)
            rows.append(row)

    fieldnames = list(rows[0].keys())
    with (OUT_DIR / "frontier_candidate_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    thresholds = [0, 70, 80, 85, 88, 90, 92, 94, 95, 96, 97]
    summary: list[dict[str, object]] = []
    for threshold in thresholds:
        selected = [row for row in rows if float(row["humaneval"]) >= threshold]
        if len(selected) < 3:
            continue
        low, high = bootstrap_ci(selected)
        summary.append(
            {
                "analysis_set": "all_lcb_v6_candidates" if threshold == 0 else f"humaneval_ge_{threshold}",
                "threshold": threshold,
                "n": len(selected),
                "rho": spearman(selected),
                "ci_low": low,
                "ci_high": high,
                "inversion_rate": inversion_rate(selected),
                "top5_overlap": topk_overlap(selected),
            }
        )

    frontier = [row for row in rows if row["audit_status"] in {"frontier_candidate", "frontier_candidate_aggregator_caveat"}]
    if len(frontier) >= 3:
        low, high = bootstrap_ci(frontier)
        summary.append(
            {
                "analysis_set": "frontier_rule_he_ge_90_no_base_tiny",
                "threshold": 90,
                "n": len(frontier),
                "rho": spearman(frontier),
                "ci_low": low,
                "ci_high": high,
                "inversion_rate": inversion_rate(frontier),
                "top5_overlap": topk_overlap(frontier),
            }
        )

    with (OUT_DIR / "threshold_sensitivity.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    plot_rows = [
        row
        for row in summary
        if (str(row["analysis_set"]).startswith("humaneval") or row["threshold"] == 0)
        and int(row["n"]) >= 6
    ]
    x = [float(row["threshold"]) for row in plot_rows]
    y = [float(row["rho"]) for row in plot_rows]
    n = [int(row["n"]) for row in plot_rows]

    plt.figure(figsize=(6.2, 3.6))
    plt.axhline(0, color="#555555", linewidth=1)
    plt.axhline(-0.3, color="#9b2c2c", linewidth=1, linestyle="--", label="inversion flag")
    plt.plot(x, y, marker="o", color="#1f5f8b", linewidth=2)
    for idx, (xi, yi, ni) in enumerate(zip(x, y, n)):
        y_offset = 0.045 if idx % 2 == 0 else -0.085
        plt.text(xi, yi + y_offset, f"n={ni}", ha="center", fontsize=8)
    plt.xlabel("Minimum HumanEval pass@1 included")
    plt.ylabel("Spearman rank validity")
    plt.title("HumanEval -> LiveCodeBench-v6 threshold sensitivity")
    plt.ylim(-0.5, 0.9)
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=False, loc="lower left")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "threshold_sensitivity.png", dpi=220)


if __name__ == "__main__":
    main()
