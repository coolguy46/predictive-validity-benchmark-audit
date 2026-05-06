from __future__ import annotations

import csv
import itertools
import math
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
INPUT_DIR = ROOT / "outputs" / "deep_research_audit_extract"
OUT_DIR = ROOT / "outputs" / "deep_research_candidate_audit"


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
    cov = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    return cov / math.sqrt(xvar * yvar)


def spearman(rows: list[dict[str, object]]) -> float:
    return pearson(
        ranks([float(row["humaneval"]) for row in rows]),
        ranks([float(row["lcb_v6"]) for row in rows]),
    )


def bootstrap_ci(rows: list[dict[str, object]], draws: int = 5000) -> tuple[float, float]:
    if len(rows) < 5:
        return math.nan, math.nan
    rng = random.Random(20260502)
    vals: list[float] = []
    for _ in range(draws):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        rho = spearman(sample)
        if not math.isnan(rho):
            vals.append(rho)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def permutation_p(rows: list[dict[str, object]], draws: int = 10000) -> float:
    observed = abs(spearman(rows))
    if math.isnan(observed):
        return math.nan
    rng = random.Random(20260502)
    source = [float(row["humaneval"]) for row in rows]
    transfer = [float(row["lcb_v6"]) for row in rows]
    count = 0
    for _ in range(draws):
        shuffled = transfer[:]
        rng.shuffle(shuffled)
        rho = abs(pearson(ranks(source), ranks(shuffled)))
        if rho >= observed:
            count += 1
    return (count + 1) / (draws + 1)


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


def top5_overlap(rows: list[dict[str, object]]) -> float:
    if len(rows) < 5:
        return math.nan
    source_top = {
        str(row["model"])
        for row in sorted(rows, key=lambda row: float(row["humaneval"]), reverse=True)[:5]
    }
    transfer_top = {
        str(row["model"])
        for row in sorted(rows, key=lambda row: float(row["lcb_v6"]), reverse=True)[:5]
    }
    return len(source_top & transfer_top) / len(source_top | transfer_top)


def parse_float(value: str) -> float:
    parsed = float(value.strip().replace("%", ""))
    return parsed * 100 if parsed <= 1 else parsed


def parse_cutoff_and_refs(value: str) -> tuple[str, str]:
    value = value.strip()
    refs = re.findall(r"(?:^|\s)(\d+)$", value)
    if not refs:
        return value, ""
    ref = refs[-1]
    cutoff = value[: value.rfind(ref)].strip()
    return cutoff, ref


def load_source_map() -> dict[str, dict[str, str]]:
    reviewer_map = DATA_DIR / "v6_source_map.csv"
    if reviewer_map.exists():
        refs: dict[str, dict[str, str]] = {}
        with reviewer_map.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                refs[str(row["ref"])] = {
                    "title": row.get("title", ""),
                    "url": row.get("url", ""),
                }
        return refs

    lines = (INPUT_DIR / "full_text.txt").read_text(encoding="utf-8").splitlines()
    try:
        start = next(idx for idx, line in enumerate(lines) if line.strip() == "Works cited")
    except StopIteration:
        return {}
    refs: dict[str, dict[str, str]] = {}
    number = 1
    for line in lines[start + 1 :]:
        line = line.strip()
        if not line:
            continue
        url_match = re.search(r"https?://\S+", line)
        refs[str(number)] = {
            "title": re.sub(r", accessed .*", "", line),
            "url": url_match.group(0) if url_match else "",
        }
        number += 1
    return refs


def load_rows(source_map: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    reviewer_rows = DATA_DIR / "livecodebench_v6_candidate_n34.csv"
    if not reviewer_rows.exists():
        reviewer_rows = DATA_DIR / "livecodebench_v6_candidate_n30.csv"
    if reviewer_rows.exists():
        rows: list[dict[str, object]] = []
        with reviewer_rows.open(encoding="utf-8-sig", newline="") as handle:
            for raw in csv.DictReader(handle):
                rows.append(
                    {
                        "model": raw["model"],
                        "organization": raw["organization"],
                        "humaneval": float(raw["humaneval"]),
                        "lcb_v6": float(raw["lcb_v6"]),
                        "release_month": raw["release_month"],
                        "training_cutoff": raw.get("training_cutoff", ""),
                        "report_source_ref": raw.get("report_source_ref", ""),
                        "report_source_title": raw.get("report_source_title", ""),
                        "report_source_url": raw.get("report_source_url", ""),
                        "provenance_note": raw.get("provenance_note", ""),
                    }
                )
        return rows

    rows: list[dict[str, object]] = []
    with (INPUT_DIR / "table_0.csv").open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            cutoff, ref = parse_cutoff_and_refs(raw["Cutoff"])
            ref_data = source_map.get(ref, {"title": "", "url": ""})
            rows.append(
                {
                    "model": raw["Model Variant"],
                    "organization": raw["Organization"],
                    "humaneval": parse_float(raw["HE pass@1"]),
                    "lcb_v6": parse_float(raw["LCB-v6 pass@1"]),
                    "release_month": raw["Release Month"],
                    "training_cutoff": cutoff,
                    "report_source_ref": ref,
                    "report_source_title": ref_data["title"],
                    "report_source_url": ref_data["url"],
                    "provenance_note": "Extracted from Gemini report table; citation maps to report source, not guaranteed per-metric provenance.",
                }
            )
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for threshold in [90, 91, 92, 93, 94, 95, 96]:
        selected = [row for row in rows if float(row["humaneval"]) >= threshold]
        if len(selected) < 3:
            continue
        low, high = bootstrap_ci(selected)
        out.append(
            {
                "analysis_set": f"deep_research_humaneval_ge_{threshold}",
                "threshold": threshold,
                "n": len(selected),
                "rho": spearman(selected),
                "ci_low": low,
                "ci_high": high,
                "permutation_p": permutation_p(selected),
                "inversion_rate": inversion_rate(selected),
                "top5_overlap": top5_overlap(selected),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_map = load_source_map()
    source_rows = [{"ref": ref, **data} for ref, data in source_map.items()]
    rows = load_rows(source_map)
    summary = summarize(rows)
    write_csv(OUT_DIR / "source_map.csv", source_rows)
    write_csv(OUT_DIR / "frontier_v6_candidate_rows.csv", rows)
    write_csv(OUT_DIR / "threshold_summary.csv", summary)


if __name__ == "__main__":
    main()
