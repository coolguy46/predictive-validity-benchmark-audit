"""Supplemental robustness checks reported in the manuscript.

The main pipeline writes the core rank-validity outputs. This small script
derives the release-month residualization, leave-organization-out checks, and
single-scale math correlations from the same released CSV files.
"""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "analysis"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    out = [0.0] * len(values)
    idx = 0
    while idx < len(values):
        end = idx
        while end + 1 < len(values) and values[order[end + 1]] == values[order[idx]]:
            end += 1
        avg = ((idx + 1) + (end + 1)) / 2
        for rank_idx in range(idx, end + 1):
            out[order[rank_idx]] = avg
        idx = end + 1
    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_denom = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_denom = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    return numerator / (x_denom * y_denom) if x_denom and y_denom else float("nan")


def spearman(rows: list[dict[str, str]], transfer_col: str = "transfer_score") -> float:
    return pearson(
        ranks([float(row["source_score"]) for row in rows]),
        ranks([float(row[transfer_col]) for row in rows]),
    )


def spearman_v6(rows: list[dict[str, str]]) -> float:
    return pearson(
        ranks([float(row["humaneval"]) for row in rows]),
        ranks([float(row["lcb_v6"]) for row in rows]),
    )


def inversion_rate_v6(rows: list[dict[str, str]]) -> float:
    inversions = 0
    comparable = 0
    for left_idx, left in enumerate(rows):
        for right in rows[left_idx + 1 :]:
            source_delta = float(left["humaneval"]) - float(right["humaneval"])
            transfer_delta = float(left["lcb_v6"]) - float(right["lcb_v6"])
            if source_delta == 0 or transfer_delta == 0:
                continue
            comparable += 1
            if source_delta * transfer_delta < 0:
                inversions += 1
    return inversions / comparable if comparable else float("nan")


def month_index(date_text: str) -> int:
    date = datetime.strptime(date_text, "%Y-%m-%d")
    return date.year * 12 + date.month


def residualize(values: list[float], controls: list[float]) -> list[float]:
    control_mean = sum(controls) / len(controls)
    value_mean = sum(values) / len(values)
    var = sum((control - control_mean) ** 2 for control in controls)
    beta = (
        sum((control - control_mean) * (value - value_mean) for control, value in zip(controls, values)) / var
        if var
        else 0.0
    )
    alpha = value_mean - beta * control_mean
    return [value - (alpha + beta * control) for value, control in zip(values, controls)]


def partial_spearman_by_release(rows: list[dict[str, str]]) -> float:
    source_ranks = ranks([float(row["source_score"]) for row in rows])
    transfer_ranks = ranks([float(row["transfer_score"]) for row in rows])
    month_ranks = ranks([float(month_index(row["date"])) for row in rows])
    return pearson(residualize(source_ranks, month_ranks), residualize(transfer_ranks, month_ranks))


def add_rank_disagreement_fields(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    enriched = [dict(row) for row in rows]
    source_rank = ranks([-float(row["source_score"]) for row in enriched])
    transfer_rank = ranks([-float(row["transfer_score"]) for row in enriched])
    for row, source, transfer in zip(enriched, source_rank, transfer_rank):
        row["source_rank"] = source
        row["transfer_rank"] = transfer
        row["rank_delta"] = transfer - source
    return enriched


def main() -> None:
    code = load_csv(ROOT / "outputs" / "data" / "pair_code_cleaned.csv")
    math_rows = load_csv(ROOT / "outputs" / "data" / "pair_math_cleaned.csv")
    v6_rows = load_csv(ROOT / "outputs" / "deep_research_candidate_audit" / "frontier_v6_candidate_rows.csv")

    rows: list[dict[str, object]] = [
        {
            "check": "canonical_code_partial_spearman_release_month",
            "subset": "all",
            "n": len(code),
            "rho": partial_spearman_by_release(code),
        }
    ]

    for org in sorted({row["org"] for row in code}):
        subset = [row for row in code if row["org"] != org]
        rows.append(
            {
                "check": "canonical_code_leave_organization_out",
                "subset": f"without {org}",
                "n": len(subset),
                "rho": spearman(subset),
            }
        )

    ranked_code = add_rank_disagreement_fields(code)
    for label, sorted_rows in [
        ("remove 5 largest transfer-over-source rank improvements", sorted(ranked_code, key=lambda row: row["rank_delta"])),
        ("remove 5 largest absolute rank disagreements", sorted(ranked_code, key=lambda row: abs(row["rank_delta"]), reverse=True)),
    ]:
        removed = {row["model_name"] for row in sorted_rows[:5]}
        subset = [row for row in ranked_code if row["model_name"] not in removed]
        rows.append(
            {
                "check": "canonical_code_reporting_selection_stress",
                "subset": label,
                "n": len(subset),
                "rho": spearman(subset),
            }
        )

    for transfer in ["MATH-500", "AIME2024"]:
        subset = [row for row in math_rows if row["transfer_benchmark"] == transfer]
        rows.append(
            {
                "check": "math_single_scale",
                "subset": transfer,
                "n": len(subset),
                "rho": spearman(subset),
            }
        )

    v6_release_subset = [row for row in v6_rows if row["release_month"] >= "2025-01"]
    rows.append(
        {
            "check": "v6_release_date_sensitivity",
            "subset": "release >= 2025-01",
            "n": len(v6_release_subset),
            "rho": spearman_v6(v6_release_subset),
        }
    )
    rows.append(
        {
            "check": "v6_release_date_sensitivity_inversion_rate",
            "subset": "release >= 2025-01",
            "n": len(v6_release_subset),
            "rho": inversion_rate_v6(v6_release_subset),
        }
    )

    write_csv(OUT_DIR / "supplemental_robustness_checks.csv", rows)


if __name__ == "__main__":
    main()
