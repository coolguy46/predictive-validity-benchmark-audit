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


OUT_DIR = ROOT / "outputs" / "frontier_audit"


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


def spearman(rows: list[dict[str, object]], source: str, transfer: str) -> float:
    return pearson(ranks([float(row[source]) for row in rows]), ranks([float(row[transfer]) for row in rows]))


def ci(rows: list[dict[str, object]], source: str, transfer: str, draws: int = 3000) -> tuple[float, float]:
    if len(rows) < 5:
        return math.nan, math.nan
    rng = random.Random(20260502)
    vals: list[float] = []
    for _ in range(draws):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        rho = spearman(sample, source, transfer)
        if not math.isnan(rho):
            vals.append(rho)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def inversion_rate(rows: list[dict[str, object]], source: str, transfer: str) -> float:
    inversions = 0
    comparable = 0
    for left, right in itertools.combinations(rows, 2):
        source_delta = float(left[source]) - float(right[source])
        transfer_delta = float(left[transfer]) - float(right[transfer])
        if source_delta == 0 or transfer_delta == 0:
            continue
        comparable += 1
        if source_delta * transfer_delta < 0:
            inversions += 1
    return inversions / comparable if comparable else math.nan


def top5(rows: list[dict[str, object]], source: str, transfer: str) -> float:
    if len(rows) < 5:
        return math.nan
    source_top = {str(row["model"]) for row in sorted(rows, key=lambda row: float(row[source]), reverse=True)[:5]}
    transfer_top = {str(row["model"]) for row in sorted(rows, key=lambda row: float(row[transfer]), reverse=True)[:5]}
    return len(source_top & transfer_top) / len(source_top | transfer_top)


def metric_row(
    dataset: str,
    pair: str,
    band: str,
    rows: list[dict[str, object]],
    source: str = "source",
    transfer: str = "transfer",
    ci_override: tuple[float, float] | None = None,
) -> dict[str, object]:
    low, high = ci_override if ci_override is not None else ci(rows, source, transfer)
    return {
        "dataset": dataset,
        "pair": pair,
        "band": band,
        "n": len(rows),
        "rho": spearman(rows, source, transfer),
        "ci_low": low,
        "ci_high": high,
        "inversion_rate": inversion_rate(rows, source, transfer),
        "top5_overlap": top5(rows, source, transfer),
    }


def load_canonical() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with (ROOT / "outputs" / "data" / "canonical_scores.csv").open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            try:
                source = float(raw["source_score"])
                transfer = float(raw["transfer_score_z"] if raw["pair_id"] == "math" else raw["transfer_score"])
            except ValueError:
                continue
            rows.append(
                {
                    "model": raw["model_name"],
                    "pair_id": raw["pair_id"],
                    "source": source,
                    "transfer": transfer,
                }
            )
    return rows


def load_lcbv6_candidate() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    candidate_path = ROOT / "outputs" / "deep_research_candidate_audit" / "frontier_v6_candidate_rows.csv"
    if not candidate_path.exists():
        candidate_path = OUT_DIR / "frontier_candidate_rows.csv"
    with candidate_path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "model": raw["model"],
                    "source": float(raw["humaneval"]),
                    "transfer": float(raw["lcb_v6"]),
                }
            )
    return rows


def load_latest_ci_overrides() -> dict[str, tuple[float, float]]:
    overrides: dict[str, tuple[float, float]] = {}
    latest_by_pair: dict[str, dict[str, str]] = {}
    with (ROOT / "outputs" / "analysis" / "rank_correlations.csv").open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw["confirmed_only"] != "False":
                continue
            pair = raw["analysis_pair"]
            if pair not in {"code", "knowledge", "math_zscore"}:
                continue
            if pair not in latest_by_pair or raw["cohort_date"] > latest_by_pair[pair]["cohort_date"]:
                latest_by_pair[pair] = raw
    mapping = {
        "code": "Canonical|HumanEval -> LiveCodeBench|all high-band rows",
        "knowledge": "Canonical|MMLU -> GPQA Diamond|all rows",
        "math_zscore": "Canonical|GSM8K -> MATH/AIME (z)|all rows",
    }
    for pair, raw in latest_by_pair.items():
        overrides[mapping[pair]] = (float(raw["rho_ci95_low"]), float(raw["rho_ci95_high"]))
    return overrides


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    canonical = load_canonical()
    lcbv6 = load_lcbv6_candidate()
    ci_overrides = load_latest_ci_overrides()

    rows: list[dict[str, object]] = []
    code = [row for row in canonical if row["pair_id"] == "code"]
    knowledge = [row for row in canonical if row["pair_id"] == "knowledge"]
    math_rows = [row for row in canonical if row["pair_id"] == "math"]

    configs = [
        ("Canonical", "HumanEval -> LiveCodeBench", "all high-band rows", code),
        ("Canonical", "HumanEval -> LiveCodeBench", "HumanEval >= 94", [r for r in code if float(r["source"]) >= 94]),
        ("LCB-v6 candidate", "HumanEval -> LiveCodeBench-v6", "HumanEval >= 90", [r for r in lcbv6 if float(r["source"]) >= 90]),
        ("Canonical", "MMLU -> GPQA Diamond", "all rows", knowledge),
        ("Canonical", "MMLU -> GPQA Diamond", "MMLU >= 90", [r for r in knowledge if float(r["source"]) >= 90]),
        ("Canonical", "GSM8K -> MATH/AIME (z)", "all rows", math_rows),
        ("Canonical", "GSM8K -> MATH/AIME (z)", "GSM8K >= 96", [r for r in math_rows if float(r["source"]) >= 96]),
    ]
    for dataset, pair, band, selected in configs:
        key = f"{dataset}|{pair}|{band}"
        rows.append(metric_row(dataset, pair, band, selected, ci_override=ci_overrides.get(key)))

    with (OUT_DIR / "protocol_sensitivity_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_rows = [
        row
        for row in rows
        if row["pair"]
        in {
            "HumanEval -> LiveCodeBench",
            "HumanEval -> LiveCodeBench-v6",
            "MMLU -> GPQA Diamond",
            "GSM8K -> MATH/AIME (z)",
        }
    ]
    labels = [f"{row['pair']}\n{row['band']} (n={row['n']})" for row in plot_rows]
    y = [float(row["rho"]) for row in plot_rows]
    low = [float(row["ci_low"]) if not math.isnan(float(row["ci_low"])) else float(row["rho"]) for row in plot_rows]
    high = [float(row["ci_high"]) if not math.isnan(float(row["ci_high"])) else float(row["rho"]) for row in plot_rows]
    lower_err = [max(0.0, yi - lo) for yi, lo in zip(y, low)]
    upper_err = [max(0.0, hi - yi) for yi, hi in zip(y, high)]

    palette = {
        "Canonical|HumanEval -> LiveCodeBench": "#b5532f",
        "LCB-v6 candidate|HumanEval -> LiveCodeBench-v6": "#2d6f9f",
        "Canonical|MMLU -> GPQA Diamond": "#2f855a",
        "Canonical|GSM8K -> MATH/AIME (z)": "#6b5b95",
    }
    colors = [palette.get(f"{row['dataset']}|{row['pair']}", "#555555") for row in plot_rows]
    plt.figure(figsize=(9.2, 4.6))
    plt.axhline(0, color="#444444", linewidth=1)
    plt.axhline(-0.3, color="#9b2c2c", linewidth=1, linestyle="--")
    plt.errorbar(range(len(y)), y, yerr=[lower_err, upper_err], fmt="none", ecolor="#777777", capsize=3, linewidth=1)
    plt.scatter(range(len(y)), y, s=52, color=colors, zorder=3)
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=8)
    plt.ylabel("Spearman rank validity")
    plt.title("Protocol sensitivity across benchmark pairs")
    plt.ylim(-1.05, 1.05)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "protocol_sensitivity_summary.png", dpi=220)


if __name__ == "__main__":
    main()
