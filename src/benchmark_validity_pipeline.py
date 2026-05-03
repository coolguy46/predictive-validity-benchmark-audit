from __future__ import annotations

import csv
import math
import os
import random
import statistics
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
LOCAL_PACKAGES = ROOT / "python_packages"
DATA_DIR = OUTPUTS / "data"
ANALYSIS_DIR = OUTPUTS / "analysis"
FIG_DIR = OUTPUTS / "figures"
FINAL_FIG_DIR = OUTPUTS / "figures_final"
REPORT_DIR = OUTPUTS / "reports"

CURRENT_DATE = pd.Timestamp("2026-05-01")
RNG = random.Random(20260501)

COLORS = {
    "math_zscore": "#0072B2",
    "math_math500": "#56B4E9",
    "math_aime2024": "#2B5B84",
    "code": "#D55E00",
    "knowledge": "#009E73",
    "vision": "#CC79A7",
}

PAIR_LABELS = {
    "math_zscore": "GSM8K -> MATH-500/AIME (z)",
    "math_math500": "GSM8K -> MATH-500",
    "math_aime2024": "GSM8K -> AIME 2024",
    "code": "HumanEval -> LiveCodeBench",
    "knowledge": "MMLU -> GPQA Diamond",
    "vision": "ImageNet -> ObjectNet/V2",
}

SOURCE_RELEASES = {
    "GSM8K": "2021-10-01",
    "HumanEval": "2021-07-01",
    "ImageNet": "2014-09-01",
    "MMLU": "2020-09-01",
}

TRANSFER_RELEASES = {
    "MATH-500": "2023-05-01",
    "AIME2024": "2024-01-01",
    "LiveCodeBench": "2024-03-01",
    "ImageNetV2": "2019-02-01",
    "ObjectNet": "2019-12-01",
    "GPQA Diamond": "2023-11-01",
}


def ensure_dirs() -> None:
    for path in [DATA_DIR, ANALYSIS_DIR, FIG_DIR, FINAL_FIG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def rec(
    model_name: str,
    org: str,
    date_value: str,
    source_benchmark: str,
    source_score: float | None,
    transfer_benchmark: str,
    transfer_score: float | None,
    provenance: str,
    pair_id: str,
    source_estimated: bool = False,
    transfer_estimated: bool = False,
    notes: str = "",
) -> dict:
    return {
        "model_name": model_name,
        "org": org,
        "date": date_value,
        "source_benchmark": source_benchmark,
        "source_score": source_score,
        "transfer_benchmark": transfer_benchmark,
        "transfer_score": transfer_score,
        "score_provenance": provenance,
        "source_estimated": source_estimated,
        "transfer_estimated": transfer_estimated,
        "is_estimated": source_estimated or transfer_estimated,
        "score_conflict": False,
        "pair_id": pair_id,
        "notes": notes,
    }


def build_records() -> list[dict]:
    rows: list[dict] = []

    # Math: GSM8K -> MATH-500 / AIME 2024.
    rows += [
        rec("GPT-5", "OpenAI", "2026-04-01", "GSM8K", 99.4, "MATH-500", 99.4, "Artificial Analysis / PricePerToken", "math"),
        rec("o3", "OpenAI", "2026-04-01", "GSM8K", 99.0, "MATH-500", 99.2, "Artificial Analysis", "math", True),
        rec("Grok-3 Mini", "xAI", "2026-05-01", "GSM8K", 98.0, "AIME2024", 95.8, "LLM-Stats", "math", True),
        rec("o4-mini", "OpenAI", "2026-05-01", "GSM8K", 96.0, "MATH-500", 96.2, "Kaggle", "math", True),
        rec("Claude Opus 4", "Anthropic", "2026-04-01", "GSM8K", 96.2, "AIME2024", 90.0, "LayerLens / PricePerToken", "math", False, True),
        rec("Gemini 2.5 Pro", "Google", "2026-05-01", "GSM8K", 95.7, "AIME2024", 92.0, "LLM-Stats", "math"),
        rec("Claude 3.5 Sonnet", "Anthropic", "2024-06-01", "GSM8K", 96.4, "MATH-500", 77.1, "Anthropic / LangDB", "math"),
        rec("DeepSeek-R1-0528", "DeepSeek", "2026-05-01", "GSM8K", 95.3, "AIME2024", 91.4, "LLM-Stats", "math"),
        rec("Llama 3.1 405B", "Meta", "2024-07-01", "GSM8K", 96.8, "MATH-500", 80.0, "Meta / LLM-Stats", "math", False, True),
        rec("Gemma 3 27B", "Google", "2025-02-01", "GSM8K", 95.9, "MATH-500", 70.0, "Google / LLM-Stats", "math", False, True),
        rec("Qwen3 235B A22B", "Alibaba", "2026-05-01", "GSM8K", 96.0, "AIME2024", 94.0, "PricePerToken", "math", True),
        rec("GLM-4.5", "Zhipu AI", "2026-05-01", "GSM8K", 95.0, "MATH-500", 98.2, "LLM-Stats", "math", True),
        rec("LongCat-Flash-Thinking", "Meituan", "2026-05-01", "GSM8K", 95.0, "MATH-500", 99.2, "LLM-Stats", "math", True),
        rec("Nemotron Nano 9B v2", "NVIDIA", "2026-05-01", "GSM8K", 95.4, "MATH-500", 97.8, "LLM-Stats", "math"),
        rec("Kimi K2-Instruct", "Moonshot AI", "2026-05-01", "GSM8K", 97.3, "MATH-500", 97.4, "LLM-Stats", "math"),
    ]

    # Code: HumanEval -> LiveCodeBench.
    rows += [
        rec("Claude Sonnet 4.5", "Anthropic", "2026-04-01", "HumanEval", 97.6, "LiveCodeBench", 71.4, "LayerLens / LCB", "code"),
        rec("DeepSeek-R1-0528", "DeepSeek", "2026-05-01", "HumanEval", 97.4, "LiveCodeBench", 73.1, "LayerLens / LCB", "code"),
        rec("Grok 4", "xAI", "2026-04-01", "HumanEval", 97.0, "LiveCodeBench", 79.4, "LayerLens / BenchLM", "code"),
        rec("Gemini 3.1 Pro", "Google", "2026-04-01", "HumanEval", 97.0, "LiveCodeBench", 71.0, "LayerLens / BenchLM", "code"),
        rec("Claude Opus 4.6", "Anthropic", "2026-03-01", "HumanEval", 97.0, "LiveCodeBench", 76.0, "LayerLens / BenchLM", "code"),
        rec("GPT-5.4", "OpenAI", "2026-03-01", "HumanEval", 95.0, "LiveCodeBench", 84.0, "BenchLM / LangDB", "code", True),
        rec("Kimi K2.5 (Reasoning)", "Moonshot AI", "2026-03-01", "HumanEval", 94.5, "LiveCodeBench", 85.0, "BenchLM", "code"),
        rec("GPT-5.2", "OpenAI", "2026-03-01", "HumanEval", 94.0, "LiveCodeBench", 79.0, "BenchLM", "code", True),
        rec("GLM-4.7", "Zhipu AI", "2026-03-01", "HumanEval", 93.0, "LiveCodeBench", 84.9, "BenchLM", "code", True),
        rec("GPT-5.3 Codex", "OpenAI", "2026-02-01", "HumanEval", 93.0, "LiveCodeBench", 85.0, "LXT / BenchLM", "code"),
        rec("MiMo-V2-Flash", "Xiaomi", "2026-03-01", "HumanEval", 92.0, "LiveCodeBench", 80.6, "BenchLM", "code", True),
        rec("o4-mini (High)", "OpenAI", "2026-05-01", "HumanEval", 96.0, "LiveCodeBench", 80.2, "LiveCodeBench", "code", True),
        rec("o3 (High)", "OpenAI", "2026-05-01", "HumanEval", 95.0, "LiveCodeBench", 75.8, "LiveCodeBench", "code", True),
        rec("Gemini 2.5 Pro 06-05", "Google", "2026-05-01", "HumanEval", 95.0, "LiveCodeBench", 73.6, "LiveCodeBench", "code", True),
        rec("Claude Opus 4 (Thinking)", "Anthropic", "2026-05-01", "HumanEval", 97.0, "LiveCodeBench", 56.6, "LiveCodeBench", "code", True, False, "Dramatic rank inversion candidate"),
    ]

    # Knowledge: MMLU -> GPQA Diamond.
    rows += [
        rec("Gemini 3.1 Pro", "Google", "2026-04-01", "MMLU", 92.6, "GPQA Diamond", 94.3, "Artificial Analysis / Iternal", "knowledge"),
        rec("Claude Opus 4.7", "Anthropic", "2026-04-01", "MMLU", 92.0, "GPQA Diamond", 94.2, "Vellum", "knowledge", True),
        rec("GPT-5.5", "OpenAI", "2026-04-01", "MMLU", 93.0, "GPQA Diamond", 93.6, "Vellum", "knowledge", True),
        rec("GPT-5.4", "OpenAI", "2026-03-01", "MMLU", 92.0, "GPQA Diamond", 92.0, "Artificial Analysis / Iternal", "knowledge", True),
        rec("GPT-5.2", "OpenAI", "2026-03-01", "MMLU", 91.0, "GPQA Diamond", 92.4, "Vellum / Iternal", "knowledge", True),
        rec("Claude Opus 4.6", "Anthropic", "2026-03-01", "MMLU", 91.1, "GPQA Diamond", 91.3, "Onyx / Iternal", "knowledge"),
        rec("Gemini 3 Flash", "Google", "2026-03-01", "MMLU", 88.0, "GPQA Diamond", 90.4, "Iternal", "knowledge", True),
        rec("o1", "OpenAI", "2024-12-01", "MMLU", 91.8, "GPQA Diamond", 80.0, "Codesota", "knowledge", False, True),
        rec("DeepSeek V3", "DeepSeek", "2024-12-01", "MMLU", 88.5, "GPQA Diamond", 57.4, "Codesota / LangDB", "knowledge"),
        rec("Llama 3.1 405B", "Meta", "2024-07-01", "MMLU", 88.6, "GPQA Diamond", 41.3, "Codesota / LangDB", "knowledge"),
        rec("Qwen 3 72B", "Alibaba", "2025-01-01", "MMLU", 88.7, "GPQA Diamond", 60.0, "Codesota", "knowledge", False, True),
        rec("Claude 3.5 Sonnet", "Anthropic", "2024-06-01", "MMLU", 88.3, "GPQA Diamond", 59.4, "Amazon / Codesota", "knowledge"),
        rec("Grok 3 Mini", "xAI", "2026-05-01", "MMLU", 90.0, "GPQA Diamond", 59.1, "Open-Reasoner-Zero", "knowledge", True),
        rec("GPT-4o", "OpenAI", "2024-05-01", "MMLU", 87.2, "GPQA Diamond", 68.8, "Kaggle / Codesota", "knowledge"),
        rec("Claude Sonnet 4.6", "Anthropic", "2026-03-01", "MMLU", 89.3, "GPQA Diamond", 74.1, "Iternal", "knowledge"),
    ]

    # Vision: ImageNet -> ImageNetV2/ObjectNet. These are structural OOD rows.
    vision_base = [
        ("ResNet-18", "Various", "2015-12-01", 59.7, 31.2, 25.8, "NeurIPS 2023", False, True, False),
        ("ResNet-50", "Various", "2015-12-01", 76.8, 66.0, 30.3, "ICCV 2023", False, True, False),
        ("MobileNetV3-Large", "Various", "2019-05-01", 71.5, 62.9, 32.4, "ICCV 2023", False, False, False),
        ("VGG-16", "Various", "2014-09-01", 63.7, 55.0, 26.9, "NeurIPS 2023", False, True, False),
        ("AlexNet", "Various", "2012-09-01", 42.3, 35.0, 22.0, "NeurIPS 2023", False, True, False),
        ("ViT-B/32", "Google", "2020-10-01", 67.4, 58.0, 39.7, "NeurIPS 2023", False, True, False),
        ("CLIP-ViT-L/14", "OpenAI", "2021-01-01", 74.4, 64.8, 36.8, "NeurIPS 2023", False, True, False),
        ("Swin-v2-large", "Microsoft", "2021-11-01", 85.0, 84.0, None, "Codesota", True, False, False),
        ("ConvNeXt-v2-huge", "Meta", "2023-01-01", 83.0, 80.5, None, "Codesota", True, False, False),
        ("CoCa", "Google", "2022-05-01", 91.0, None, 80.0, "Codesota / Encord", False, False, True),
        ("CLIP (LAION) Avg", "Various", "2021-12-01", None, 64.8, 60.4, "OpenReview 2024", False, False, False),
        ("YFCC Models Avg", "Various", "2021-12-01", None, 23.0, 58.9, "OpenReview 2024", False, False, False),
    ]
    for name, org, d, imagenet, v2, obj, source, src_est, v2_est, obj_est in vision_base:
        if v2 is not None:
            rows.append(rec(name, org, d, "ImageNet", imagenet, "ImageNetV2", v2, source, "vision", src_est or imagenet is None, v2_est, "Qualitative case-study row"))
        if obj is not None:
            rows.append(rec(name, org, d, "ImageNet", imagenet, "ObjectNet", obj, source, "vision", src_est or imagenet is None, obj_est, "Qualitative case-study row"))

    return rows


def canonical_frame() -> pd.DataFrame:
    df = pd.DataFrame(build_records())
    df["date"] = pd.to_datetime(df["date"])
    df["date_precision"] = "month"
    numeric_cols = ["source_score", "transfer_score"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_math_zscores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["transfer_score_z"] = out["transfer_score"]
    math_mask = out["pair_id"].eq("math")
    for bench, idx in out[math_mask].groupby("transfer_benchmark").groups.items():
        vals = out.loc[idx, "transfer_score"].astype(float)
        std = vals.std(ddof=0)
        if std and not math.isnan(std):
            out.loc[idx, "transfer_score_z"] = (vals - vals.mean()) / std
        else:
            out.loc[idx, "transfer_score_z"] = 0.0
    return out


def month_diff(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def month_range(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    months = []
    cur = pd.Timestamp(start.year, start.month, 1)
    last = pd.Timestamp(end.year, end.month, 1)
    while cur <= last:
        months.append(cur)
        cur = cur + pd.DateOffset(months=1)
    return months


def average_ranks(values: Iterable[float], descending: bool = True) -> np.ndarray:
    series = pd.Series(list(values), dtype=float)
    return series.rank(method="average", ascending=not descending).to_numpy()


def pearson(x: Iterable[float], y: Iterable[float]) -> float:
    xs = np.asarray(list(x), dtype=float)
    ys = np.asarray(list(y), dtype=float)
    mask = ~(np.isnan(xs) | np.isnan(ys))
    xs = xs[mask]
    ys = ys[mask]
    if len(xs) < 2:
        return float("nan")
    if np.std(xs) == 0 or np.std(ys) == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def spearman(source: Iterable[float], transfer: Iterable[float]) -> float:
    return pearson(average_ranks(source, True), average_ranks(transfer, True))


def permutation_pvalue(source: list[float], transfer: list[float], observed: float, draws: int = 5000) -> float:
    if math.isnan(observed):
        return float("nan")
    if len(source) < 5:
        return float("nan")
    more_extreme = 0
    permuted = transfer[:]
    for _ in range(draws):
        RNG.shuffle(permuted)
        rho = spearman(source, permuted)
        if not math.isnan(rho) and abs(rho) >= abs(observed):
            more_extreme += 1
    return (more_extreme + 1) / (draws + 1)


def bootstrap_ci(source: list[float], transfer: list[float], draws: int = 2000) -> tuple[float, float]:
    if len(source) < 5:
        return float("nan"), float("nan")
    vals = []
    n = len(source)
    for _ in range(draws):
        idx = [RNG.randrange(n) for _ in range(n)]
        xs = [source[i] for i in idx]
        ys = [transfer[i] for i in idx]
        rho = spearman(xs, ys)
        if not math.isnan(rho):
            vals.append(rho)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def analysis_subset(df: pd.DataFrame, analysis_pair: str) -> tuple[pd.DataFrame, str]:
    if analysis_pair == "math_zscore":
        sub = df[df["pair_id"].eq("math")].copy()
        return sub, "transfer_score_z"
    if analysis_pair == "math_math500":
        sub = df[df["pair_id"].eq("math") & df["transfer_benchmark"].eq("MATH-500")].copy()
        return sub, "transfer_score"
    if analysis_pair == "math_aime2024":
        sub = df[df["pair_id"].eq("math") & df["transfer_benchmark"].eq("AIME2024")].copy()
        return sub, "transfer_score"
    sub = df[df["pair_id"].eq(analysis_pair)].copy()
    return sub, "transfer_score"


def compute_rank_correlations(df: pd.DataFrame, confirmed_only: bool = False) -> pd.DataFrame:
    records = []
    analysis_pairs = ["math_zscore", "math_math500", "math_aime2024", "code", "knowledge"]
    for analysis_pair in analysis_pairs:
        sub, y_col = analysis_subset(df, analysis_pair)
        if confirmed_only:
            sub = sub[~sub["is_estimated"]].copy()
        sub = sub.dropna(subset=["source_score", y_col])
        if sub.empty:
            continue
        release_months = sorted({pd.Timestamp(d.year, d.month, 1) for d in sub["date"]})
        for cohort_month in release_months:
            cohort = sub[sub["date"] <= cohort_month].copy()
            if len(cohort) < 5:
                continue
            xs = cohort["source_score"].astype(float).tolist()
            ys = cohort[y_col].astype(float).tolist()
            rho = spearman(xs, ys)
            lo, hi = bootstrap_ci(xs, ys)
            p = permutation_pvalue(xs, ys, rho)
            records.append(
                {
                    "analysis_pair": analysis_pair,
                    "pair_label": PAIR_LABELS[analysis_pair],
                    "cohort_month": cohort_month.strftime("%Y-%m"),
                    "cohort_date": cohort_month.strftime("%Y-%m-%d"),
                    "spearman_rho": rho,
                    "rho_ci95_low": lo,
                    "rho_ci95_high": hi,
                    "permutation_p_value": p,
                    "n_models": len(cohort),
                    "estimated_fraction": float(cohort["is_estimated"].mean()),
                    "source_std": float(cohort["source_score"].std(ddof=1)),
                    "transfer_std": float(cohort[y_col].std(ddof=1)),
                    "confirmed_only": confirmed_only,
                }
            )
    return pd.DataFrame(records)


def linreg_r2(x: Iterable[float], y: Iterable[float]) -> tuple[float, float, float]:
    xs = np.asarray(list(x), dtype=float)
    ys = np.asarray(list(y), dtype=float)
    mask = ~(np.isnan(xs) | np.isnan(ys))
    xs = xs[mask]
    ys = ys[mask]
    if len(xs) < 2 or np.var(xs) == 0 or np.var(ys) == 0:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = slope * xs + intercept
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(r2), float(slope), float(intercept)


def fit_half_lives(rank_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair, sub in rank_df[~rank_df["confirmed_only"]].groupby("analysis_pair"):
        sub = sub.sort_values("cohort_date").copy()
        positive = sub[sub["spearman_rho"] > 0].copy()
        span = month_diff(pd.to_datetime(sub["cohort_date"]).min(), pd.to_datetime(sub["cohort_date"]).max())
        status = "fit"
        if len(positive) < 3:
            rows.append(
                {
                    "analysis_pair": pair,
                    "pair_label": PAIR_LABELS[pair],
                    "rho0": np.nan,
                    "lambda_per_month": np.nan,
                    "half_life_months": np.nan,
                    "half_life_years": np.nan,
                    "fit_r2": np.nan,
                    "n_points": len(sub),
                    "span_months": span,
                    "status": "insufficient_positive_points",
                    "reliability": "not_estimable",
                }
            )
            continue
        t0 = pd.to_datetime(positive["cohort_date"]).min()
        t = np.array([month_diff(t0, pd.to_datetime(d)) for d in positive["cohort_date"]], dtype=float)
        log_rho = np.log(positive["spearman_rho"].astype(float).to_numpy())
        slope, intercept = np.polyfit(t, log_rho, 1)
        lam = -float(slope)
        pred = slope * t + intercept
        ss_res = float(np.sum((log_rho - pred) ** 2))
        ss_tot = float(np.sum((log_rho - np.mean(log_rho)) ** 2))
        fit_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        if lam <= 0:
            status = "no_monotone_decay"
            half_life = np.nan
        else:
            half_life = math.log(2) / lam
        reliability = "paper_ready" if span >= 12 and len(positive) >= 4 else "exploratory"
        if span < 6:
            reliability = "fragile_short_span"
        rows.append(
            {
                "analysis_pair": pair,
                "pair_label": PAIR_LABELS[pair],
                "rho0": math.exp(float(intercept)),
                "lambda_per_month": lam,
                "half_life_months": half_life,
                "half_life_years": half_life / 12 if not math.isnan(half_life) else np.nan,
                "fit_r2": fit_r2,
                "n_points": len(positive),
                "span_months": span,
                "status": status,
                "reliability": reliability,
            }
        )
    return pd.DataFrame(rows)


def compute_variance_compression(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    qdf = df[df["pair_id"].isin(["math", "code", "knowledge"])].dropna(subset=["source_score"]).copy()
    qdf["year"] = qdf["date"].dt.year
    for benchmark, sub in qdf.groupby("source_benchmark"):
        for year in range(int(sub["year"].min()), int(sub["year"].max()) + 1):
            cohort = sub[sub["year"] <= year].copy()
            if cohort.empty:
                continue
            top10 = cohort.nlargest(min(10, len(cohort)), "source_score")
            records.append(
                {
                    "benchmark": benchmark,
                    "year": year,
                    "top10_std": float(top10["source_score"].std(ddof=1)) if len(top10) > 1 else 0.0,
                    "top10_mean": float(top10["source_score"].mean()),
                    "ceiling_gap": float(100 - top10["source_score"].mean()),
                    "n_models": len(cohort),
                    "top_k": len(top10),
                }
            )
    return pd.DataFrame(records)


def compute_exposure_proxy(df: pd.DataFrame, rank_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in rank_df[~rank_df["confirmed_only"]].iterrows():
        analysis_pair = row["analysis_pair"]
        sub, _ = analysis_subset(df, analysis_pair)
        source = sub["source_benchmark"].dropna().iloc[0]
        release = pd.Timestamp(SOURCE_RELEASES[source])
        cohort_date = pd.Timestamp(row["cohort_date"])
        age_months = max(1, month_diff(release, cohort_date))
        cumulative_models = int(row["n_models"])
        pressure = age_months * cumulative_models
        rows.append(
            {
                "analysis_pair": analysis_pair,
                "pair_label": PAIR_LABELS[analysis_pair],
                "cohort_month": row["cohort_month"],
                "source_benchmark": source,
                "cumulative_models": cumulative_models,
                "source_benchmark_age_months": age_months,
                "cumulative_model_months": pressure,
                "spearman_rho": row["spearman_rho"],
                "method": "offline_proxy_cumulative_models_x_benchmark_age_months",
                "note": "Semantic Scholar citation counts were not queried in this offline workspace.",
            }
        )
    return pd.DataFrame(rows)


def compute_rank_inversions(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for pair in ["math", "code", "knowledge"]:
        sub = df[df["pair_id"].eq(pair)].dropna(subset=["source_score", "transfer_score"]).copy()
        if pair == "math":
            # Avoid raw mixing: report inversions within each transfer benchmark.
            groups = sub.groupby("transfer_benchmark")
        else:
            groups = [("all", sub)]
        for bench, grp in groups:
            grp = grp.copy()
            grp["source_rank"] = grp["source_score"].rank(method="average", ascending=False)
            grp["transfer_rank"] = grp["transfer_score"].rank(method="average", ascending=False)
            grp["rank_delta"] = grp["transfer_rank"] - grp["source_rank"]
            grp["abs_rank_delta"] = grp["rank_delta"].abs()
            for _, row in grp.iterrows():
                records.append(
                    {
                        "pair_id": pair,
                        "transfer_group": bench,
                        "model_name": row["model_name"],
                        "source_score": row["source_score"],
                        "transfer_score": row["transfer_score"],
                        "source_rank": row["source_rank"],
                        "transfer_rank": row["transfer_rank"],
                        "rank_delta": row["rank_delta"],
                        "abs_rank_delta": row["abs_rank_delta"],
                        "is_estimated": row["is_estimated"],
                    }
                )
    return pd.DataFrame(records).sort_values(["pair_id", "abs_rank_delta"], ascending=[True, False])


def compute_health_scores(df: pd.DataFrame, rank_df: pd.DataFrame, variance_df: pd.DataFrame, exposure_df: pd.DataFrame, ordering_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for analysis_pair in ["math_zscore", "code", "knowledge"]:
        sub, y_col = analysis_subset(df, analysis_pair)
        sub = sub.dropna(subset=["source_score", y_col])
        latest = rank_df[(rank_df["analysis_pair"].eq(analysis_pair)) & (~rank_df["confirmed_only"])].sort_values("cohort_date").tail(1)
        rho = float(latest["spearman_rho"].iloc[0]) if not latest.empty else np.nan
        ci_width = float(latest["rho_ci95_high"].iloc[0] - latest["rho_ci95_low"].iloc[0]) if not latest.empty else np.nan
        estimated_fraction = float(latest["estimated_fraction"].iloc[0]) if not latest.empty else np.nan
        r2, slope, intercept = linreg_r2(sub["source_score"], sub[y_col])
        source = sub["source_benchmark"].iloc[0]
        vsub = variance_df[variance_df["benchmark"].eq(source)].sort_values("year").tail(1)
        spread = float(vsub["top10_std"].iloc[0]) if not vsub.empty else np.nan
        esub = exposure_df[exposure_df["analysis_pair"].eq(analysis_pair)].sort_values("cohort_month").tail(1)
        exposure = float(esub["cumulative_model_months"].iloc[0]) if not esub.empty else np.nan
        osub = ordering_df[ordering_df["analysis_pair"].eq(analysis_pair)].tail(1)
        inversion = float(osub["pairwise_inversion_rate"].iloc[0]) if not osub.empty else np.nan
        top5 = float(osub["top5_jaccard"].iloc[0]) if not osub.empty else np.nan
        release = pd.Timestamp(SOURCE_RELEASES[source])
        age_years = month_diff(release, CURRENT_DATE) / 12
        rows.append(
            {
                "analysis_pair": analysis_pair,
                "benchmark_pair": PAIR_LABELS[analysis_pair],
                "rho_current": rho,
                "rho_ci95_width": ci_width,
                "estimated_fraction": estimated_fraction,
                "r2_transfer_from_source": r2,
                "linear_slope": slope,
                "source_top10_spread": spread,
                "pairwise_inversion_rate": inversion,
                "top5_jaccard": top5,
                "exposure_proxy_log1p": math.log1p(exposure),
                "source_age_years": age_years,
            }
        )
    h = pd.DataFrame(rows)
    positive = h[["rho_current", "r2_transfer_from_source", "source_top10_spread", "top5_jaccard"]].astype(float)
    negative = h[["pairwise_inversion_rate", "rho_ci95_width", "estimated_fraction", "exposure_proxy_log1p", "source_age_years"]].astype(float)
    pos_scaled = (positive - positive.mean()) / positive.std(ddof=0)
    neg_scaled = (negative - negative.mean()) / negative.std(ddof=0)
    h["positive_signal_z"] = pos_scaled.mean(axis=1)
    h["negative_pressure_z"] = neg_scaled.mean(axis=1)
    h["health_score"] = h["positive_signal_z"] - h["negative_pressure_z"]
    h["health_rank"] = h["health_score"].rank(ascending=False, method="dense").astype(int)
    return h.sort_values("health_score", ascending=False)


def compute_mechanism_discriminants(df: pd.DataFrame, rank_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    code_curve = rank_df[(rank_df["analysis_pair"].eq("code")) & (~rank_df["confirmed_only"])].sort_values("cohort_date")
    if len(code_curve) >= 2:
        first = code_curve.iloc[0]
        latest = code_curve.iloc[-1]
        rows.append(
            {
                "test": "code_temporal_strengthening",
                "quantity": "delta_rho_latest_minus_first",
                "value": float(latest["spearman_rho"] - first["spearman_rho"]),
                "baseline": float(first["spearman_rho"]),
                "comparison": float(latest["spearman_rho"]),
                "interpretation": "HumanEval-to-LiveCodeBench validity becomes more negative as later 2026 models enter the overlap set.",
            }
        )

    code = df[df["pair_id"].eq("code")].dropna(subset=["source_score", "transfer_score"]).copy()
    knowledge = df[df["pair_id"].eq("knowledge")].dropna(subset=["source_score", "transfer_score"]).copy()
    if not code.empty:
        code_rho = spearman(code["source_score"], code["transfer_score"])
        code_quantized_rho = spearman(np.round(code["source_score"]), code["transfer_score"])
        rows.append(
            {
                "test": "code_one_point_quantization",
                "quantity": "rho_after_integer_quantization",
                "value": float(code_quantized_rho),
                "baseline": float(code_rho),
                "comparison": float(code_quantized_rho),
                "interpretation": "Coarse HumanEval score granularity does not remove the inversion; the negative relation is not just a decimal-scale artifact.",
            }
        )

    if len(code) == len(knowledge) and len(code) > 0:
        human_template = sorted(code["source_score"].astype(float).tolist(), reverse=True)
        knowledge_sorted = knowledge.sort_values("source_score", ascending=False).copy()
        knowledge_sorted["human_eval_rank_template"] = human_template[: len(knowledge_sorted)]
        actual_rho = spearman(knowledge_sorted["source_score"], knowledge_sorted["transfer_score"])
        compressed_rho = spearman(knowledge_sorted["human_eval_rank_template"], knowledge_sorted["transfer_score"])
        rows.append(
            {
                "test": "mmlu_humaneval_compression_counterfactual",
                "quantity": "rho_after_rank_preserving_humaneval_template",
                "value": float(compressed_rho),
                "baseline": float(actual_rho),
                "comparison": float(compressed_rho),
                "interpretation": "Applying HumanEval-like score ties/range to MMLU while preserving MMLU order leaves GPQA validity positive, so compression alone is insufficient to explain the code inversion.",
            }
        )

    return pd.DataFrame(rows)


def compute_gap_and_regression_diagnostics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = [
        ("math_math500", "math", "MATH-500", "GSM8K -> MATH-500"),
        ("math_aime2024", "math", "AIME2024", "GSM8K -> AIME 2024"),
        ("code", "code", "LiveCodeBench", "HumanEval -> LiveCodeBench"),
        ("knowledge", "knowledge", "GPQA Diamond", "MMLU -> GPQA Diamond"),
        ("vision_objectnet", "vision", "ObjectNet", "ImageNet -> ObjectNet"),
        ("vision_imagenetv2", "vision", "ImageNetV2", "ImageNet -> ImageNet-V2"),
    ]
    gap_rows = []
    reg_rows = []
    for analysis_pair, pair_id, transfer_benchmark, label in specs:
        sub = df[(df["pair_id"].eq(pair_id)) & (df["transfer_benchmark"].eq(transfer_benchmark))].dropna(subset=["source_score", "transfer_score"]).copy()
        if sub.empty:
            continue
        r2, slope, intercept = linreg_r2(sub["source_score"], sub["transfer_score"])
        rho = spearman(sub["source_score"], sub["transfer_score"])
        pred = slope * sub["source_score"] + intercept if pd.notna(slope) else np.nan
        residual = sub["transfer_score"] - pred
        abs_residual = np.abs(residual)
        transfer_gap = sub["source_score"] - sub["transfer_score"]
        reg_rows.append(
            {
                "analysis_pair": analysis_pair,
                "pair_label": label,
                "n_models": len(sub),
                "spearman_rho": rho,
                "linear_r2": r2,
                "linear_slope": slope,
                "linear_intercept": intercept,
                "source_std": float(sub["source_score"].std(ddof=1)) if len(sub) > 1 else 0.0,
                "transfer_std": float(sub["transfer_score"].std(ddof=1)) if len(sub) > 1 else 0.0,
                "median_transfer_gap": float(transfer_gap.median()),
                "mean_abs_residual": float(np.nanmean(abs_residual)),
                "max_abs_residual": float(np.nanmax(abs_residual)),
            }
        )
        for (_, row), yhat, resid, abs_resid, gap in zip(sub.iterrows(), pred, residual, abs_residual, transfer_gap):
            gap_rows.append(
                {
                    "analysis_pair": analysis_pair,
                    "pair_label": label,
                    "model_name": row["model_name"],
                    "org": row["org"],
                    "source_benchmark": row["source_benchmark"],
                    "transfer_benchmark": row["transfer_benchmark"],
                    "source_score": row["source_score"],
                    "transfer_score": row["transfer_score"],
                    "transfer_gap_source_minus_transfer": gap,
                    "predicted_transfer_score": yhat,
                    "residual_transfer_minus_predicted": resid,
                    "abs_residual": abs_resid,
                    "is_estimated": row["is_estimated"],
                    "score_provenance": row["score_provenance"],
                }
            )
    gaps = pd.DataFrame(gap_rows)
    regs = pd.DataFrame(reg_rows)
    if not gaps.empty:
        gaps = gaps.sort_values("abs_residual", ascending=False)
    if not regs.empty:
        regs = regs.sort_values("analysis_pair")
    return gaps, regs


def kendall_tau_b(x: Iterable[float], y: Iterable[float]) -> tuple[float, int, int, int, int]:
    xs = np.asarray(list(x), dtype=float)
    ys = np.asarray(list(y), dtype=float)
    concordant = discordant = ties_x = ties_y = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            dx = np.sign(xs[i] - xs[j])
            dy = np.sign(ys[i] - ys[j])
            if dx == 0 and dy == 0:
                ties_x += 1
                ties_y += 1
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    tau = (concordant - discordant) / denom if denom else float("nan")
    return float(tau), concordant, discordant, ties_x, ties_y


def compute_ordering_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("math_zscore", "math", None, PAIR_LABELS["math_zscore"]),
        ("math_math500", "math", "MATH-500", PAIR_LABELS["math_math500"]),
        ("math_aime2024", "math", "AIME2024", PAIR_LABELS["math_aime2024"]),
        ("code", "code", "LiveCodeBench", PAIR_LABELS["code"]),
        ("knowledge", "knowledge", "GPQA Diamond", PAIR_LABELS["knowledge"]),
        ("vision_objectnet", "vision", "ObjectNet", "ImageNet -> ObjectNet"),
        ("vision_imagenetv2", "vision", "ImageNetV2", "ImageNet -> ImageNet-V2"),
    ]
    rows = []
    for analysis_pair, pair_id, transfer, label in specs:
        if analysis_pair.startswith("math_") and analysis_pair in {"math_zscore", "math_math500", "math_aime2024"}:
            sub, y_col = analysis_subset(df, analysis_pair)
        else:
            sub = df[df["pair_id"].eq(pair_id)].copy()
            if transfer is not None:
                sub = sub[sub["transfer_benchmark"].eq(transfer)].copy()
            y_col = "transfer_score"
        sub = sub.dropna(subset=["source_score", y_col]).copy()
        if len(sub) < 3:
            continue
        tau, concordant, discordant, ties_x, ties_y = kendall_tau_b(sub["source_score"], sub[y_col])
        comparable = concordant + discordant
        inversion_rate = discordant / comparable if comparable else float("nan")
        top3_source = set(sub.nlargest(min(3, len(sub)), "source_score")["model_name"])
        top3_transfer = set(sub.nlargest(min(3, len(sub)), y_col)["model_name"])
        top5_source = set(sub.nlargest(min(5, len(sub)), "source_score")["model_name"])
        top5_transfer = set(sub.nlargest(min(5, len(sub)), y_col)["model_name"])
        top3_jaccard = len(top3_source & top3_transfer) / len(top3_source | top3_transfer) if top3_source | top3_transfer else float("nan")
        top5_jaccard = len(top5_source & top5_transfer) / len(top5_source | top5_transfer) if top5_source | top5_transfer else float("nan")
        source_std = float(sub["source_score"].std(ddof=1))
        transfer_std = float(sub[y_col].std(ddof=1))
        rows.append(
            {
                "analysis_pair": analysis_pair,
                "pair_label": label,
                "n_models": len(sub),
                "spearman_rho": spearman(sub["source_score"], sub[y_col]),
                "kendall_tau_b": tau,
                "pairwise_inversion_rate": inversion_rate,
                "pairwise_concordant": concordant,
                "pairwise_discordant": discordant,
                "source_ties": ties_x,
                "transfer_ties": ties_y,
                "top3_jaccard": top3_jaccard,
                "top5_jaccard": top5_jaccard,
                "source_std": source_std,
                "transfer_std": transfer_std,
                "source_range": float(sub["source_score"].max() - sub["source_score"].min()),
                "transfer_range": float(sub[y_col].max() - sub[y_col].min()),
                "spread_ratio_source_over_transfer": source_std / transfer_std if transfer_std else float("nan"),
                "estimated_fraction": float(sub["is_estimated"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("analysis_pair")


def compute_leave_one_out_influence(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for analysis_pair in ["math_zscore", "math_math500", "math_aime2024", "code", "knowledge"]:
        sub, y_col = analysis_subset(df, analysis_pair)
        sub = sub.dropna(subset=["source_score", y_col]).copy()
        if len(sub) < 6:
            continue
        full_rho = spearman(sub["source_score"], sub[y_col])
        for idx, row in sub.iterrows():
            reduced = sub.drop(index=idx)
            loo_rho = spearman(reduced["source_score"], reduced[y_col])
            rows.append(
                {
                    "analysis_pair": analysis_pair,
                    "pair_label": PAIR_LABELS[analysis_pair],
                    "removed_model": row["model_name"],
                    "removed_org": row["org"],
                    "full_rho": full_rho,
                    "leave_one_out_rho": loo_rho,
                    "delta_rho_minus_full": loo_rho - full_rho,
                    "abs_delta_rho": abs(loo_rho - full_rho),
                    "removed_source_score": row["source_score"],
                    "removed_transfer_score": row[y_col],
                    "is_estimated": row["is_estimated"],
                }
            )
    return pd.DataFrame(rows).sort_values("abs_delta_rho", ascending=False)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    issues = []
    for idx, row in df.iterrows():
        for col in ["source_score", "transfer_score"]:
            val = row[col]
            if pd.notna(val) and not (0 <= val <= 100):
                issues.append({"severity": "error", "row": idx, "model_name": row["model_name"], "check": "score_range", "detail": f"{col}={val}"})
        release = TRANSFER_RELEASES.get(row["transfer_benchmark"])
        if release and row["date"] < pd.Timestamp(release):
            issues.append(
                {
                    "severity": "info",
                    "row": idx,
                    "model_name": row["model_name"],
                    "check": "model_predates_transfer_benchmark",
                    "detail": f"{row['date'].date()} before {row['transfer_benchmark']} release {release}; expected for retrospective OOD evaluation.",
                }
            )
        if row["date"] > CURRENT_DATE:
            issues.append({"severity": "warning", "row": idx, "model_name": row["model_name"], "check": "future_date", "detail": str(row["date"].date())})
    for pair in ["math", "code", "vision", "knowledge"]:
        n = int(df[df["pair_id"].eq(pair)].dropna(subset=["source_score", "transfer_score"]).shape[0])
        if n < 12:
            issues.append({"severity": "warning", "row": "", "model_name": "", "check": "minimum_overlap", "detail": f"{pair}: {n} overlapping rows below n=12 threshold"})
    for pair in ["math", "code", "knowledge"]:
        sub = df[df["pair_id"].eq(pair)].copy()
        span = month_diff(sub["date"].min(), sub["date"].max())
        if span < 12:
            issues.append({"severity": "warning", "row": "", "model_name": "", "check": "temporal_spread", "detail": f"{pair}: {span} months, below 12-month decay threshold"})
    return pd.DataFrame(issues)


def write_csvs(df: pd.DataFrame, rank_df: pd.DataFrame, rank_confirmed_df: pd.DataFrame, half_lives: pd.DataFrame, variance_df: pd.DataFrame, exposure_df: pd.DataFrame, health_df: pd.DataFrame, inversions: pd.DataFrame, validation_df: pd.DataFrame, gap_df: pd.DataFrame, regression_df: pd.DataFrame, ordering_df: pd.DataFrame, influence_df: pd.DataFrame, mechanism_df: pd.DataFrame) -> None:
    df.to_csv(DATA_DIR / "canonical_scores.csv", index=False)
    for pair in ["math", "code", "vision", "knowledge"]:
        df[df["pair_id"].eq(pair)].to_csv(DATA_DIR / f"pair_{pair}_cleaned.csv", index=False)
    health_df.to_csv(DATA_DIR / "health_scores.csv", index=False)
    combined_rank = pd.concat([rank_df, rank_confirmed_df], ignore_index=True)
    combined_rank.to_csv(ANALYSIS_DIR / "rank_correlations.csv", index=False)
    half_lives.to_csv(ANALYSIS_DIR / "half_lives.csv", index=False)
    variance_df.to_csv(ANALYSIS_DIR / "variance_compression.csv", index=False)
    exposure_df.to_csv(ANALYSIS_DIR / "citation_pressure.csv", index=False)
    inversions.to_csv(ANALYSIS_DIR / "rank_inversions.csv", index=False)
    validation_df.to_csv(ANALYSIS_DIR / "validation_issues.csv", index=False)
    gap_df.to_csv(ANALYSIS_DIR / "transfer_gap_outliers.csv", index=False)
    regression_df.to_csv(ANALYSIS_DIR / "regression_diagnostics.csv", index=False)
    ordering_df.to_csv(ANALYSIS_DIR / "ordering_diagnostics.csv", index=False)
    influence_df.to_csv(ANALYSIS_DIR / "leave_one_out_influence.csv", index=False)
    mechanism_df.to_csv(ANALYSIS_DIR / "mechanism_discriminants.csv", index=False)
    summarize_sensitivity(rank_df, rank_confirmed_df).to_csv(ANALYSIS_DIR / "sensitivity_estimated_scores.csv", index=False)


def summarize_sensitivity(all_df: pd.DataFrame, confirmed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair in sorted(set(all_df["analysis_pair"]).union(set(confirmed_df["analysis_pair"]))):
        a = all_df[all_df["analysis_pair"].eq(pair)].sort_values("cohort_date").tail(1)
        c = confirmed_df[confirmed_df["analysis_pair"].eq(pair)].sort_values("cohort_date").tail(1)
        rows.append(
            {
                "analysis_pair": pair,
                "rho_all_latest": float(a["spearman_rho"].iloc[0]) if not a.empty else np.nan,
                "n_all_latest": int(a["n_models"].iloc[0]) if not a.empty else 0,
                "rho_confirmed_latest": float(c["spearman_rho"].iloc[0]) if not c.empty else np.nan,
                "n_confirmed_latest": int(c["n_models"].iloc[0]) if not c.empty else 0,
                "delta_confirmed_minus_all": (float(c["spearman_rho"].iloc[0]) - float(a["spearman_rho"].iloc[0])) if not a.empty and not c.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def save_png_pdf(img: Image.Image, stem: str) -> None:
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    img.save(png, dpi=(300, 300))
    img.convert("RGB").save(pdf, "PDF", resolution=300)


def get_matplotlib():
    if LOCAL_PACKAGES.exists() and str(LOCAL_PACKAGES) not in sys.path:
        sys.path.insert(0, str(LOCAL_PACKAGES))
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "lines.linewidth": 1.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def save_mpl(fig, stem: str) -> None:
    for out_dir in [FIG_DIR, FINAL_FIG_DIR]:
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")


def mpl_figure1(rank_df: pd.DataFrame, half_lives: pd.DataFrame) -> None:
    plt = get_matplotlib()

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.8), sharey=True)
    sub = rank_df[(~rank_df["confirmed_only"]) & rank_df["analysis_pair"].isin(["math_zscore", "code", "knowledge"])].copy()
    sub["cohort_dt"] = pd.to_datetime(sub["cohort_date"])
    for ax, pair in zip(axes, ["math_zscore", "code", "knowledge"]):
        ps = sub[sub["analysis_pair"].eq(pair)].sort_values("cohort_dt")
        x = np.arange(1, len(ps) + 1)
        ax.plot(x, ps["spearman_rho"], marker="o", color=COLORS[pair])
        ax.axhline(0.5, color="0.45", linestyle=":", linewidth=1.1)
        ax.axhline(0, color="0.65", linewidth=0.8)
        ax.set_ylim(-1.05, 1.05)
        if not ps.empty:
            latest = ps.iloc[-1]
            first_date = ps["cohort_dt"].iloc[0].strftime("%Y-%m")
            last_date = ps["cohort_dt"].iloc[-1].strftime("%Y-%m")
            subtitle = f"rho={latest['spearman_rho']:.2f}, n={int(latest['n_models'])}; {first_date} to {last_date}"
        else:
            subtitle = "rho=n/a"
        ax.set_title(
            f"{ {'math_zscore': 'Math', 'code': 'Code', 'knowledge': 'Knowledge'}[pair] }\n{subtitle}",
            fontsize=9.2,
            pad=8,
        )
        if len(x) <= 6:
            ticks = x
        else:
            ticks = np.array([x[0], x[len(x) // 2], x[-1]])
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(t)) for t in ticks])
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("Cumulative cohort step")
    axes[0].set_ylabel("Spearman rho")
    fig.suptitle("Predictive Validity Decay Across Benchmark Pairs", y=1.02, fontsize=12)
    fig.text(0.5, 0.015, "Cohort step is ordered by release month within each pair; date ranges are shown in panel subtitles. Dotted line marks rho=0.5.", ha="center", fontsize=8, color="0.25")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save_mpl(fig, "figure1_decay_curves")
    plt.close(fig)


def mpl_figure2(variance_df: pd.DataFrame) -> None:
    plt = get_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    panels = [("GSM8K", "math_zscore"), ("HumanEval", "code"), ("MMLU", "knowledge")]
    for ax, (bench, pair) in zip(axes, panels):
        sub = variance_df[variance_df["benchmark"].eq(bench)].sort_values("year")
        x = sub["year"].astype(float).to_numpy()
        mean = sub["top10_mean"].astype(float).to_numpy()
        std = sub["top10_std"].astype(float).to_numpy()
        if len(sub) > 1:
            ax.fill_between(x, mean - std, mean + std, color=COLORS[pair], alpha=0.18, linewidth=0)
            ax.plot(x, mean, marker="o", color=COLORS[pair])
        else:
            ax.scatter(x, mean, color=COLORS[pair], s=35)
        ax.axhline(100, color="#AA2222", linestyle=":", linewidth=1.0)
        ax.set_title(bench)
        ax.set_xlabel("Year")
        ax.set_ylim(50, 102)
        ax.set_xticks(sorted(sub["year"].unique()))
        if ax is axes[0]:
            ax.set_ylabel("Top-10 source score (%)")
    fig.suptitle("Source Benchmark Score Compression", y=1.02, fontsize=12)
    fig.tight_layout()
    save_mpl(fig, "figure2_variance_compression")
    plt.close(fig)


def mpl_figure3(exposure_df: pd.DataFrame) -> None:
    plt = get_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.7), sharey=True)
    sub = exposure_df[exposure_df["analysis_pair"].isin(["math_zscore", "code", "knowledge"])].copy()
    sub["log_pressure"] = np.log10(sub["cumulative_model_months"].astype(float) + 1)
    for ax, pair in zip(axes, ["math_zscore", "code", "knowledge"]):
        ps = sub[sub["analysis_pair"].eq(pair)]
        ax.scatter(ps["log_pressure"], ps["spearman_rho"], color=COLORS[pair], s=42, zorder=3)
        if len(ps) >= 2:
            xs = ps["log_pressure"].to_numpy()
            ys = ps["spearman_rho"].to_numpy()
            slope, intercept = np.polyfit(xs, ys, 1)
            xline = np.linspace(xs.min(), xs.max(), 100)
            ax.plot(xline, slope * xline + intercept, color="black", linestyle="--", linewidth=1.0)
            slope_text = f"slope={slope:.2f}"
        else:
            slope_text = "slope=n/a"
        ax.axhline(0, color="0.65", linewidth=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("log10 exposure", labelpad=7)
        ax.set_title(f"{ {'math_zscore': 'Math', 'code': 'Code', 'knowledge': 'Knowledge'}[pair] }\n{slope_text}", fontsize=9.2, pad=8)
    axes[0].set_ylabel("Spearman rho")
    fig.suptitle("Exposure Proxy vs. Predictive Validity", y=1.02, fontsize=12)
    fig.text(0.5, 0.02, "Exposure is cumulative model-months, an offline proxy for citation/submission pressure.", ha="center", fontsize=8, color="0.25")
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    save_mpl(fig, "figure3_exposure_vs_validity")
    plt.close(fig)


def mpl_figure4(health_df: pd.DataFrame) -> None:
    plt = get_matplotlib()
    fig, ax = plt.subplots(figsize=(9.6, 3.0))
    ax.axis("off")
    cols = ["Benchmark Pair", "rho", "R2", "Spread", "Exposure", "Age", "Health H", "Rank"]
    rows = []
    colors = []
    for r in health_df.itertuples():
        rows.append(
            [
                r.benchmark_pair,
                f"{r.rho_current:.2f}",
                f"{r.r2_transfer_from_source:.2f}",
                f"{r.source_top10_spread:.2f}",
                f"{r.exposure_proxy_log1p:.2f}",
                f"{r.source_age_years:.1f}",
                f"{r.health_score:+.2f}",
                str(r.health_rank),
            ]
        )
        row_color = "#E8F5E9" if r.health_score > 0.5 else "#FDECEC" if r.health_score < -0.5 else "#FFF8D9"
        colors.append([row_color] * len(cols))
    table = ax.table(
        cellText=rows,
        colLabels=cols,
        cellColours=colors,
        colColours=["#F2F2F2"] * len(cols),
        cellLoc="center",
        loc="center",
        colWidths=[0.42, 0.07, 0.07, 0.08, 0.1, 0.07, 0.1, 0.06],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.45)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_text_props(weight="bold")
        if col == 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.02
    ax.set_title("Benchmark Health Score Rankings", loc="left", fontsize=12, fontweight="bold", pad=8)
    fig.text(
        0.01,
        0.02,
        "H combines current rank validity, linear R2, source-score spread, benchmark age, and exposure proxy after z-normalization.",
        fontsize=8,
        color="0.25",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_mpl(fig, "figure4_health_score_table")
    plt.close(fig)


def mpl_figure5(df: pd.DataFrame) -> None:
    plt = get_matplotlib()
    selected = ["Llama 3.1 405B", "Gemini 3 Flash", "DeepSeek V3", "GPT-4o", "Claude Opus 4.7", "Gemini 3.1 Pro", "Claude Sonnet 4.6"]
    full = df[df["pair_id"].eq("knowledge")].copy()
    full["source_rank"] = full["source_score"].rank(method="average", ascending=False)
    full["transfer_rank"] = full["transfer_score"].rank(method="average", ascending=False)
    sub = full[full["model_name"].isin(selected)].copy()
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    for r in sub.itertuples():
        delta = r.transfer_rank - r.source_rank
        color = "#D55E00" if delta > 2 else "#0072B2" if delta < -2 else "0.45"
        ax.plot([0, 1], [r.source_rank, r.transfer_rank], color=color, marker="o", linewidth=2.0)
        ax.text(-0.04, r.source_rank, f"{r.model_name} ({r.source_score:.1f}%)", ha="right", va="center", fontsize=8)
        ax.text(1.04, r.transfer_rank, f"{r.transfer_score:.1f}%", ha="left", va="center", fontsize=8)
    ax.set_xlim(-0.75, 1.35)
    ax.set_ylim(15.5, 0.5)
    ax.set_xticks([0, 1], ["MMLU\nSource", "GPQA Diamond\nTransfer"])
    ax.set_ylabel("Rank (1 = best)")
    ax.set_title("Rank Inversions: MMLU vs. GPQA Diamond")
    ax.grid(axis="y", alpha=0.2)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save_mpl(fig, "figure5_rank_inversions")
    plt.close(fig)


def mpl_figure6(df: pd.DataFrame) -> None:
    plt = get_matplotlib()
    sub = df[(df["pair_id"].eq("vision")) & (df["transfer_benchmark"].eq("ObjectNet"))].dropna(subset=["source_score", "transfer_score"]).copy()
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.scatter(sub["source_score"], sub["transfer_score"], color=COLORS["vision"], s=45)
    key_models = {"AlexNet", "ResNet-50", "CLIP-ViT-L/14", "CoCa"}
    for r in sub.itertuples():
        if r.model_name in key_models:
            ax.annotate(r.model_name, (r.source_score, r.transfer_score), xytext=(5, 4), textcoords="offset points", fontsize=7)
    drops = sub["source_score"] - sub["transfer_score"]
    ax.set_xlabel("ImageNet top-1 (%)")
    ax.set_ylabel("ObjectNet (%)")
    ax.set_title("Vision Case Study: ImageNet to ObjectNet")
    ax.text(
        0.98,
        0.04,
        f"Median transfer drop: {drops.median():.1f} points",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color="0.25",
        bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.9, "pad": 2},
    )
    fig.tight_layout()
    save_mpl(fig, "figure6_vision_case_study")
    plt.close(fig)


def mpl_figure7_scatter_diagnostics(df: pd.DataFrame, regression_df: pd.DataFrame) -> None:
    plt = get_matplotlib()
    panels = [
        ("math_math500", "math", "MATH-500", "GSM8K", "MATH-500", "GSM8K -> MATH-500", COLORS["math_zscore"]),
        ("code", "code", "LiveCodeBench", "HumanEval", "LiveCodeBench", "HumanEval -> LiveCodeBench", COLORS["code"]),
        ("knowledge", "knowledge", "GPQA Diamond", "MMLU", "GPQA Diamond", "MMLU -> GPQA Diamond", COLORS["knowledge"]),
        ("vision_objectnet", "vision", "ObjectNet", "ImageNet", "ObjectNet", "ImageNet -> ObjectNet", COLORS["vision"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.6))
    for ax, (analysis_pair, pair_id, transfer, xlabel, ylabel, title, color) in zip(axes.flat, panels):
        sub = df[(df["pair_id"].eq(pair_id)) & (df["transfer_benchmark"].eq(transfer))].dropna(subset=["source_score", "transfer_score"]).copy()
        ax.scatter(sub["source_score"], sub["transfer_score"], color=color, s=34, alpha=0.85)
        reg = regression_df[regression_df["analysis_pair"].eq(analysis_pair)]
        if not reg.empty and pd.notna(reg["linear_slope"].iloc[0]):
            slope = reg["linear_slope"].iloc[0]
            intercept = reg["linear_intercept"].iloc[0]
            xs = np.linspace(sub["source_score"].min(), sub["source_score"].max(), 100)
            ax.plot(xs, slope * xs + intercept, color="black", linestyle="--", linewidth=1.0)
            stats_label = f"rho={reg['spearman_rho'].iloc[0]:.2f}, R2={reg['linear_r2'].iloc[0]:.2f}, n={int(reg['n_models'].iloc[0])}"
        else:
            stats_label = f"n={len(sub)}"
        ax.set_title(f"{title}\n{stats_label}", fontsize=10)
        ax.set_xlabel(f"{xlabel} score (%)")
        ax.set_ylabel(f"{ylabel} score (%)")
    fig.suptitle("Source Score vs. Transfer Score: Regression Diagnostics", y=1.01, fontsize=12)
    fig.tight_layout()
    save_mpl(fig, "figure7_source_transfer_scatter")
    plt.close(fig)


def mpl_figure8_transfer_gaps(gap_df: pd.DataFrame) -> None:
    plt = get_matplotlib()
    plot_df = gap_df[gap_df["analysis_pair"].isin(["math_math500", "math_aime2024", "code", "knowledge", "vision_objectnet"])].copy()
    plot_df = plot_df.sort_values("transfer_gap_source_minus_transfer", ascending=False).head(14)
    color_map = {
        "math_math500": COLORS["math_zscore"],
        "math_aime2024": COLORS["math_zscore"],
        "code": COLORS["code"],
        "knowledge": COLORS["knowledge"],
        "vision_objectnet": COLORS["vision"],
    }
    plot_df["label"] = plot_df["model_name"] + " (" + plot_df["pair_label"].str.replace(" -> ", " to ", regex=False) + ")"
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["transfer_gap_source_minus_transfer"], color=[color_map.get(a, "0.5") for a in plot_df["analysis_pair"]])
    ax.set_yticks(y, plot_df["label"], fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="0.25", linewidth=0.8)
    ax.set_xlabel("Source score minus transfer score (percentage points)")
    ax.set_title("Largest Observed Generalization Gaps")
    for yi, val in zip(y, plot_df["transfer_gap_source_minus_transfer"]):
        ax.text(val + 0.8, yi, f"{val:.1f}", va="center", fontsize=7)
    fig.tight_layout()
    save_mpl(fig, "figure8_largest_transfer_gaps")
    plt.close(fig)


def mpl_figure9_ordering_diagnostics(ordering_df: pd.DataFrame) -> None:
    plt = get_matplotlib()
    plot_df = ordering_df[ordering_df["analysis_pair"].isin(["math_zscore", "code", "knowledge", "vision_objectnet"])].copy()
    label_map = {
        "math_zscore": "Math",
        "code": "Code",
        "knowledge": "Knowledge",
        "vision_objectnet": "Vision",
    }
    plot_df["short_label"] = plot_df["analysis_pair"].map(label_map)
    x = np.arange(len(plot_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.bar(x - width / 2, plot_df["pairwise_inversion_rate"], width=width, color="#B95C50", label="Pairwise inversion rate")
    ax.bar(x + width / 2, plot_df["top5_jaccard"], width=width, color="#4C78A8", label="Top-5 overlap")
    ax.set_xticks(x, plot_df["short_label"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate / overlap")
    ax.set_title("Leaderboard Ordering Preservation by Benchmark Pair")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    for xpos, inv, top5 in zip(x, plot_df["pairwise_inversion_rate"], plot_df["top5_jaccard"]):
        ax.text(xpos - width / 2, inv + 0.025, f"{inv:.2f}", ha="center", fontsize=8)
        ax.text(xpos + width / 2, top5 + 0.025, f"{top5:.2f}", ha="center", fontsize=8)
    fig.text(0.5, 0.01, "Lower inversion rate and higher top-5 overlap indicate better transfer of source-benchmark ordering.", ha="center", fontsize=8, color="0.25")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_mpl(fig, "figure9_ordering_diagnostics")
    plt.close(fig)


def mpl_figure10_leave_one_out(influence_df: pd.DataFrame) -> None:
    plt = get_matplotlib()
    candidates = influence_df[influence_df["analysis_pair"].isin(["math_zscore", "code", "knowledge"])].copy()
    plot_df = pd.concat(
        [
            grp.sort_values("abs_delta_rho", ascending=False).head(4)
            for _, grp in candidates.groupby("analysis_pair")
        ],
        ignore_index=True,
    )
    plot_df["pair_order"] = plot_df["analysis_pair"].map({"math_zscore": 0, "code": 1, "knowledge": 2})
    plot_df = plot_df.sort_values(["pair_order", "abs_delta_rho"], ascending=[False, True])
    color_map = {"math_zscore": COLORS["math_zscore"], "code": COLORS["code"], "knowledge": COLORS["knowledge"]}
    labels = plot_df["removed_model"] + "\n" + plot_df["analysis_pair"].map({"math_zscore": "Math", "code": "Code", "knowledge": "Knowledge"})
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["delta_rho_minus_full"], color=[color_map.get(a, "0.5") for a in plot_df["analysis_pair"]])
    ax.set_yticks(y, labels, fontsize=7)
    ax.axvline(0, color="0.25", linewidth=0.8)
    ax.set_xlabel("Change in rho after removing model")
    ax.set_title("Leave-One-Model-Out Influence on Predictive Validity")
    for yi, val in zip(y, plot_df["delta_rho_minus_full"]):
        ha = "left" if val >= 0 else "right"
        offset = 0.01 if val >= 0 else -0.01
        ax.text(val + offset, yi, f"{val:+.2f}", va="center", ha=ha, fontsize=7)
    fig.text(0.5, 0.01, "Large absolute values indicate correlations that are sensitive to one model; signs show whether removal raises or lowers rho.", ha="center", fontsize=8, color="0.25")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save_mpl(fig, "figure10_leave_one_out_influence")
    plt.close(fig)


def draw_axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], xlabel: str, ylabel: str, title: str, x_ticks: list[tuple[float, str]], y_ticks: list[tuple[float, str]], xlim: tuple[float, float], ylim: tuple[float, float]) -> tuple:
    x0, y0, x1, y1 = box
    draw.line((x0, y1, x1, y1), fill="#222222", width=2)
    draw.line((x0, y0, x0, y1), fill="#222222", width=2)
    for xv, label in x_ticks:
        x = map_x(xv, xlim, x0, x1)
        draw.line((x, y1, x, y1 + 8), fill="#222222", width=2)
        draw.text((x - 24, y1 + 12), label, fill="#222222", font=font(26))
        draw.line((x, y0, x, y1), fill="#DDDDDD", width=1)
    for yv, label in y_ticks:
        y = map_y(yv, ylim, y0, y1)
        draw.line((x0 - 8, y, x0, y), fill="#222222", width=2)
        draw.text((x0 - 70, y - 14), label, fill="#222222", font=font(26))
        draw.line((x0, y, x1, y), fill="#DDDDDD", width=1)
    draw.text(((x0 + x1) // 2 - 190, y1 + 54), xlabel, fill="#111111", font=font(32))
    draw.text((x0 - 96, y0 - 40), ylabel, fill="#111111", font=font(30))
    draw.text((x0, y0 - 75), title, fill="#111111", font=font(36, True))
    return x0, y0, x1, y1


def map_x(v: float, xlim: tuple[float, float], x0: int, x1: int) -> int:
    if xlim[1] == xlim[0]:
        return (x0 + x1) // 2
    return int(x0 + (v - xlim[0]) / (xlim[1] - xlim[0]) * (x1 - x0))


def map_y(v: float, ylim: tuple[float, float], y0: int, y1: int) -> int:
    if ylim[1] == ylim[0]:
        return (y0 + y1) // 2
    return int(y1 - (v - ylim[0]) / (ylim[1] - ylim[0]) * (y1 - y0))


def polyline(points: list[tuple[int, int]]) -> list[int]:
    flat = []
    for x, y in points:
        flat.extend([x, y])
    return flat


def figure1(rank_df: pd.DataFrame, half_lives: pd.DataFrame) -> None:
    img = Image.new("RGB", (2400, 1280), "white")
    draw = ImageDraw.Draw(img)
    box = (220, 180, 1720, 930)
    sub = rank_df[(~rank_df["confirmed_only"]) & rank_df["analysis_pair"].isin(["math_zscore", "code", "knowledge"])].copy()
    sub["t"] = pd.to_datetime(sub["cohort_date"]).map(lambda d: d.year + (d.month - 1) / 12)
    xlim = (sub["t"].min() - 0.05, sub["t"].max() + 0.05)
    ylim = (-1.0, 1.05)
    x_ticks = [(float(y), str(y)) for y in range(int(math.ceil(xlim[0])), int(math.floor(xlim[1])) + 1)]
    y_ticks = [(-1.0, "-1.0"), (-0.5, "-0.5"), (0, "0"), (0.5, "0.5"), (1.0, "1.0")]
    draw_axes(draw, box, "Cumulative cohort date", "Spearman rho", "Predictive Validity Decay Across Benchmark Pairs", x_ticks, y_ticks, xlim, ylim)
    x0, y0, x1, y1 = box
    # Weak validity line.
    yweak = map_y(0.5, ylim, y0, y1)
    for x in range(x0, x1, 22):
        draw.line((x, yweak, x + 11, yweak), fill="#777777", width=2)
    legend_y = 230
    for pair in ["math_zscore", "code", "knowledge"]:
        ps = sub[sub["analysis_pair"].eq(pair)].sort_values("t")
        pts = [(map_x(float(r.t), xlim, x0, x1), map_y(float(r.spearman_rho), ylim, y0, y1)) for r in ps.itertuples()]
        if len(pts) > 1:
            draw.line(polyline(pts), fill=COLORS[pair], width=5, joint="curve")
        for x, y in pts:
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=COLORS[pair], outline="white", width=2)
        hl = half_lives[half_lives["analysis_pair"].eq(pair)]
        suffix = ""
        if not hl.empty and pd.notna(hl["half_life_months"].iloc[0]):
            suffix = f"; t1/2={hl['half_life_months'].iloc[0]:.1f} mo"
        elif not hl.empty:
            suffix = "; no half-life"
        draw.rectangle((1780, legend_y - 8, 1818, legend_y + 20), fill=COLORS[pair])
        draw.text((1830, legend_y - 12), PAIR_LABELS[pair] + suffix, fill="#111111", font=font(25))
        legend_y += 48
    draw.text((220, 1055), "Dashed horizontal reference marks rho=0.5. Code is negative in the latest cohort, so exponential half-life is not fit.", fill="#444444", font=font(25))
    save_png_pdf(img, "figure1_decay_curves")


def figure2(variance_df: pd.DataFrame) -> None:
    img = Image.new("RGB", (3000, 1050), "white")
    draw = ImageDraw.Draw(img)
    panels = [("GSM8K", "math_zscore"), ("HumanEval", "code"), ("MMLU", "knowledge")]
    for i, (bench, pair) in enumerate(panels):
        left = 170 + i * 930
        box = (left, 170, left + 760, 840)
        sub = variance_df[variance_df["benchmark"].eq(bench)].sort_values("year")
        years = sub["year"].astype(float).tolist()
        xlim = (min(years) - 0.1, max(years) + 0.1) if years else (2024, 2026)
        ylim = (50, 102)
        draw_axes(draw, box, "Year", "Score (%)" if i == 0 else "", bench, [(y, str(int(y))) for y in sorted(set(years))], [(60, "60"), (80, "80"), (100, "100")], xlim, ylim)
        x0, y0, x1, y1 = box
        upper = []
        lower = []
        meanpts = []
        for r in sub.itertuples():
            x = map_x(float(r.year), xlim, x0, x1)
            upper.append((x, map_y(r.top10_mean + r.top10_std, ylim, y0, y1)))
            lower.append((x, map_y(r.top10_mean - r.top10_std, ylim, y0, y1)))
            meanpts.append((x, map_y(r.top10_mean, ylim, y0, y1)))
        if len(meanpts) > 1:
            draw.polygon(upper + list(reversed(lower)), fill=hex_to_rgba_on_white(COLORS[pair], 0.18))
            draw.line(polyline(meanpts), fill=COLORS[pair], width=5)
        for x, y in meanpts:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=COLORS[pair])
        yceil = map_y(100, ylim, y0, y1)
        draw.line((x0, yceil, x1, yceil), fill="#AA2222", width=2)
    draw.text((170, 50), "Top-10 Source Benchmark Score Compression", fill="#111111", font=font(42, True))
    draw.text((170, 920), "Shading shows +/-1 standard deviation. Low spread near ceiling indicates weak discriminative resolution.", fill="#444444", font=font(28))
    save_png_pdf(img, "figure2_variance_compression")


def hex_to_rgba_on_white(hex_color: str, alpha: float) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return tuple(int(alpha * c + (1 - alpha) * 255) for c in (r, g, b))


def figure3(exposure_df: pd.DataFrame) -> None:
    img = Image.new("RGB", (1500, 1420), "white")
    draw = ImageDraw.Draw(img)
    sub = exposure_df[exposure_df["analysis_pair"].isin(["math_zscore", "code", "knowledge"])].copy()
    sub["log_pressure"] = np.log10(sub["cumulative_model_months"].astype(float) + 1)
    xlim = (sub["log_pressure"].min() - 0.1, sub["log_pressure"].max() + 0.1)
    ylim = (-1.0, 1.05)
    box = (210, 180, 1260, 960)
    draw_axes(draw, box, "log10(cumulative model-month exposure)", "Spearman rho", "Exposure Pressure Proxy vs. Predictive Validity", [(round(x, 1), f"{x:.1f}") for x in np.linspace(xlim[0], xlim[1], 5)], [(-1, "-1"), (-0.5, "-0.5"), (0, "0"), (0.5, "0.5"), (1, "1")], xlim, ylim)
    x0, y0, x1, y1 = box
    xs = sub["log_pressure"].to_numpy()
    ys = sub["spearman_rho"].to_numpy()
    slope, intercept = np.polyfit(xs, ys, 1)
    xstart, xend = xlim
    draw.line((map_x(xstart, xlim, x0, x1), map_y(slope * xstart + intercept, ylim, y0, y1), map_x(xend, xlim, x0, x1), map_y(slope * xend + intercept, ylim, y0, y1)), fill="#111111", width=4)
    for pair in ["math_zscore", "code", "knowledge"]:
        ps = sub[sub["analysis_pair"].eq(pair)]
        for r in ps.itertuples():
            x = map_x(float(r.log_pressure), xlim, x0, x1)
            y = map_y(float(r.spearman_rho), ylim, y0, y1)
            draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=COLORS[pair], outline="white", width=2)
    legend_y = 1070
    legend_x = 210
    for pair in ["math_zscore", "code", "knowledge"]:
        draw.rectangle((legend_x, legend_y, legend_x + 38, legend_y + 26), fill=COLORS[pair])
        draw.text((legend_x + 50, legend_y - 4), PAIR_LABELS[pair], fill="#111111", font=font(25))
        legend_x += 430
    draw.text((210, 1220), f"Regression slope={slope:.2f}. Exposure is an offline proxy, not live citation data.", fill="#444444", font=font(28))
    save_png_pdf(img, "figure3_exposure_vs_validity")


def figure4(health_df: pd.DataFrame) -> None:
    img = Image.new("RGB", (2400, 820), "white")
    draw = ImageDraw.Draw(img)
    draw.text((80, 60), "Benchmark Health Score Rankings", fill="#111111", font=font(44, True))
    cols = [
        ("Benchmark Pair", 80, 680),
        ("rho", 760, 110),
        ("R2", 870, 110),
        ("Spread", 980, 140),
        ("Exposure", 1120, 180),
        ("Age", 1300, 120),
        ("Health H", 1420, 170),
        ("Rank", 1590, 100),
    ]
    y0 = 160
    draw.rectangle((70, y0, 1710, y0 + 70), fill="#F0F0F0", outline="#BBBBBB")
    for label, x, _ in cols:
        draw.text((x, y0 + 20), label, fill="#111111", font=font(27, True))
    y = y0 + 70
    for r in health_df.itertuples():
        fill = "#E8F5E9" if r.health_score > 0.5 else "#FDECEC" if r.health_score < -0.5 else "#FFF8D9"
        draw.rectangle((70, y, 1710, y + 86), fill=fill, outline="#DDDDDD")
        values = [
            r.benchmark_pair,
            f"{r.rho_current:.2f}",
            f"{r.r2_transfer_from_source:.2f}",
            f"{r.source_top10_spread:.2f}",
            f"{r.exposure_proxy_log1p:.2f}",
            f"{r.source_age_years:.1f}",
            f"{r.health_score:+.2f}",
            str(r.health_rank),
        ]
        for (_, x, width), value in zip(cols, values):
            draw.text((x, y + 26), value[:42], fill="#111111", font=font(25))
        y += 86
    draw.text((80, 660), "H combines current rank validity, linear R2, source-score spread, benchmark age, and exposure proxy after z-normalization.", fill="#444444", font=font(26))
    save_png_pdf(img, "figure4_health_score_table")


def figure5(df: pd.DataFrame) -> None:
    img = Image.new("RGB", (1600, 1600), "white")
    draw = ImageDraw.Draw(img)
    selected = ["Llama 3.1 405B", "Gemini 3 Flash", "DeepSeek V3", "GPT-4o", "Claude Opus 4.7", "Gemini 3.1 Pro", "Claude Sonnet 4.6"]
    sub = df[df["pair_id"].eq("knowledge") & df["model_name"].isin(selected)].copy()
    full = df[df["pair_id"].eq("knowledge")].copy()
    full["source_rank"] = full["source_score"].rank(method="average", ascending=False)
    full["transfer_rank"] = full["transfer_score"].rank(method="average", ascending=False)
    sub = sub.merge(full[["model_name", "source_rank", "transfer_rank"]], on="model_name")
    x_left, x_right = 540, 1060
    y_top, y_bottom = 220, 1350
    ylim = (1, 15)
    draw.text((190, 70), "Rank Inversions: MMLU vs. GPQA Diamond", fill="#111111", font=font(46, True))
    draw.text((340, 150), "MMLU Source Rank", fill="#111111", font=font(32, True))
    draw.text((950, 150), "GPQA Diamond Transfer Rank", fill="#111111", font=font(32, True))
    draw.line((x_left, y_top, x_left, y_bottom), fill="#222222", width=4)
    draw.line((x_right, y_top, x_right, y_bottom), fill="#222222", width=4)
    for rank in range(1, 16):
        y = map_y_rank(rank, ylim, y_top, y_bottom)
        draw.text((x_left - 55, y - 14), str(rank), fill="#666666", font=font(24))
        draw.text((x_right + 24, y - 14), str(rank), fill="#666666", font=font(24))
        draw.line((x_left - 8, y, x_left + 8, y), fill="#222222", width=2)
        draw.line((x_right - 8, y, x_right + 8, y), fill="#222222", width=2)
    for r in sub.sort_values("source_rank").itertuples():
        y1 = map_y_rank(float(r.source_rank), ylim, y_top, y_bottom)
        y2 = map_y_rank(float(r.transfer_rank), ylim, y_top, y_bottom)
        delta = float(r.transfer_rank - r.source_rank)
        color = "#D55E00" if delta > 2 else "#0072B2" if delta < -2 else "#777777"
        draw.line((x_left, y1, x_right, y2), fill=color, width=5)
        draw.ellipse((x_left - 8, y1 - 8, x_left + 8, y1 + 8), fill=color)
        draw.ellipse((x_right - 8, y2 - 8, x_right + 8, y2 + 8), fill=color)
        draw.text((70, y1 - 20), f"{r.model_name} ({r.source_score:.1f}%)", fill="#111111", font=font(27))
        draw.text((x_right + 85, y2 - 20), f"{r.transfer_score:.1f}%", fill="#111111", font=font(27))
    draw.text((190, 1430), "Orange lines fall on transfer; blue lines rise. Ranks computed against all 15 knowledge-pair models.", fill="#444444", font=font(28))
    save_png_pdf(img, "figure5_rank_inversions")


def map_y_rank(rank: float, ylim: tuple[float, float], top: int, bottom: int) -> int:
    return int(top + (rank - ylim[0]) / (ylim[1] - ylim[0]) * (bottom - top))


def figure_vision_case_study(df: pd.DataFrame) -> None:
    img = Image.new("RGB", (1500, 1150), "white")
    draw = ImageDraw.Draw(img)
    sub = df[(df["pair_id"].eq("vision")) & (df["transfer_benchmark"].eq("ObjectNet"))].dropna(subset=["source_score", "transfer_score"]).copy()
    box = (210, 170, 1260, 930)
    xlim = (35, 95)
    ylim = (15, 85)
    draw_axes(draw, box, "ImageNet top-1 (%)", "ObjectNet (%)", "Vision Case Study: ImageNet Does Not Preserve OOD Ordering", [(40, "40"), (60, "60"), (80, "80")], [(20, "20"), (40, "40"), (60, "60"), (80, "80")], xlim, ylim)
    x0, y0, x1, y1 = box
    for r in sub.itertuples():
        x = map_x(float(r.source_score), xlim, x0, x1)
        y = map_y(float(r.transfer_score), ylim, y0, y1)
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=COLORS["vision"], outline="white", width=2)
        draw.text((x + 12, y - 12), r.model_name[:18], fill="#222222", font=font(21))
    drops = sub["source_score"] - sub["transfer_score"]
    draw.text((210, 990), f"Median ImageNet-to-ObjectNet drop: {drops.median():.1f} points across {len(sub)} architectures with both scores.", fill="#444444", font=font(28))
    save_png_pdf(img, "figure6_vision_case_study")


def make_figures(df: pd.DataFrame, rank_df: pd.DataFrame, half_lives: pd.DataFrame, variance_df: pd.DataFrame, exposure_df: pd.DataFrame, health_df: pd.DataFrame, gap_df: pd.DataFrame, regression_df: pd.DataFrame, ordering_df: pd.DataFrame, influence_df: pd.DataFrame) -> None:
    mpl_figure1(rank_df, half_lives)
    mpl_figure2(variance_df)
    mpl_figure3(exposure_df)
    mpl_figure4(health_df)
    mpl_figure5(df)
    mpl_figure6(df)
    mpl_figure7_scatter_diagnostics(df, regression_df)
    mpl_figure8_transfer_gaps(gap_df)
    mpl_figure9_ordering_diagnostics(ordering_df)
    mpl_figure10_leave_one_out(influence_df)


def write_reports(df: pd.DataFrame, rank_df: pd.DataFrame, half_lives: pd.DataFrame, health_df: pd.DataFrame, inversions: pd.DataFrame, validation_df: pd.DataFrame, exposure_df: pd.DataFrame, gap_df: pd.DataFrame, regression_df: pd.DataFrame, ordering_df: pd.DataFrame, influence_df: pd.DataFrame, mechanism_df: pd.DataFrame) -> None:
    latest_rows = []
    for pair in ["math_zscore", "math_math500", "math_aime2024", "code", "knowledge"]:
        sub = rank_df[(rank_df["analysis_pair"].eq(pair)) & (~rank_df["confirmed_only"])].sort_values("cohort_date")
        if not sub.empty:
            r = sub.tail(1).iloc[0]
            latest_rows.append(f"- {PAIR_LABELS[pair]}: rho={r.spearman_rho:.2f}, 95% bootstrap CI [{r.rho_ci95_low:.2f}, {r.rho_ci95_high:.2f}], n={int(r.n_models)}, estimated fraction={r.estimated_fraction:.2f}")
    hlines = []
    for r in half_lives.itertuples():
        if pd.notna(r.half_life_months):
            hlines.append(f"- {r.pair_label}: t1/2={r.half_life_months:.1f} months ({r.reliability}, fit R2={r.fit_r2:.2f})")
        else:
            hlines.append(f"- {r.pair_label}: no defensible half-life estimate ({r.status}, {r.reliability})")
    worst = inversions[inversions["pair_id"].eq("knowledge")].sort_values("abs_rank_delta", ascending=False).head(1).iloc[0]
    code_worst = inversions[inversions["pair_id"].eq("code")].sort_values("abs_rank_delta", ascending=False).head(1).iloc[0]
    largest_gap = gap_df.sort_values("transfer_gap_source_minus_transfer", ascending=False).head(1).iloc[0]
    largest_residual = gap_df.sort_values("abs_residual", ascending=False).head(1).iloc[0]
    ordering_core = ordering_df[ordering_df["analysis_pair"].isin(["math_zscore", "code", "knowledge", "vision_objectnet"])].copy()
    ordering_lines = [
        f"- {r.pair_label}: pairwise inversion rate={r.pairwise_inversion_rate:.2f}, top-5 overlap={r.top5_jaccard:.2f}, Kendall tau-b={r.kendall_tau_b:.2f}"
        for r in ordering_core.itertuples()
    ]
    influence_core = influence_df[influence_df["analysis_pair"].isin(["math_zscore", "code", "knowledge"])].head(5)
    influence_lines = [
        f"- Removing {r.removed_model} from {r.pair_label} changes rho by {r.delta_rho_minus_full:+.2f} ({r.full_rho:.2f} -> {r.leave_one_out_rho:.2f})."
        for r in influence_core.itertuples()
    ]
    mechanism_lines = [
        f"- {r.test}: {r.quantity}={r.value:+.2f} (baseline {r.baseline:+.2f}, comparison {r.comparison:+.2f}). {r.interpretation}"
        for r in mechanism_df.itertuples()
    ]
    vision = df[(df["pair_id"].eq("vision")) & (df["transfer_benchmark"].eq("ObjectNet"))].dropna(subset=["source_score", "transfer_score"]).copy()
    vision_drop = (vision["source_score"] - vision["transfer_score"]).median()
    summary = f"""# Paper-Ready Results Summary

## Current Predictive Validity

{chr(10).join(latest_rows)}

## Predictive Validity Half-Lives

{chr(10).join(hlines)}

Interpretation: half-life estimates are only paper-ready where temporal support is adequate. Short-span estimates are retained for transparency but should be described as exploratory rather than headline claims.

## Rank Inversions

- Knowledge headline inversion: {worst.model_name} moves from source rank {worst.source_rank:.1f} to transfer rank {worst.transfer_rank:.1f} ({worst.source_score:.1f}% source, {worst.transfer_score:.1f}% transfer).
- Code headline inversion: {code_worst.model_name} moves from source rank {code_worst.source_rank:.1f} to transfer rank {code_worst.transfer_rank:.1f} ({code_worst.source_score:.1f}% source, {code_worst.transfer_score:.1f}% transfer).

## Residual and Gap Diagnostics

- Largest raw transfer gap: {largest_gap.model_name} on {largest_gap.pair_label}, with source-transfer gap {largest_gap.transfer_gap_source_minus_transfer:.1f} percentage points.
- Largest regression residual: {largest_residual.model_name} on {largest_residual.pair_label}, residual {largest_residual.residual_transfer_minus_predicted:+.1f} percentage points relative to the source-score linear fit.
- Added Figure 7 for source-transfer regression diagnostics and Figure 8 for largest generalization gaps.

## Ordering Robustness

{chr(10).join(ordering_lines)}

## Leave-One-Out Sensitivity

{chr(10).join(influence_lines)}

## Benchmark Health Score

{health_df[["benchmark_pair", "health_score", "health_rank"]].to_string(index=False)}

## Mechanism Discriminants

{chr(10).join(mechanism_lines)}

## Vision Case Study

The ImageNet/ObjectNet rows are not used for a temporal decay curve. Among architectures with both ImageNet and ObjectNet scores, the median transfer drop is {vision_drop:.1f} percentage points, supporting the structural OOD-validity case without pretending to have clean leaderboard-time evidence.

## Exposure Analysis

`citation_pressure.csv` uses `cumulative_model_months = cumulative_models * source_benchmark_age_months` as an offline exposure proxy. This avoids inventing citation counts when Semantic Scholar access is not available.

## Validation Caveats

- Estimated scores are retained and flagged; see `sensitivity_estimated_scores.csv`.
- Math transfer benchmarks are not mixed raw; see separate MATH-500, AIME 2024, and z-normalized analyses.
- Vision is qualitative because exact-version overlap is sparse.
- Retrospective evaluations of old vision architectures produce expected `model_predates_transfer_benchmark` validation notes.
- Some temporal spans, especially code, are too short for strong decay-rate claims.
"""
    (REPORT_DIR / "paper_ready_summary.md").write_text(summary, encoding="utf-8")

    methods = f"""# Methods and Limitations Notes

## Data Schema

Each canonical row contains model identity, organization, release month, source benchmark, transfer benchmark, 0-100 source/transfer scores, provenance, estimated-score flags, conflict flag, and pair id. Additional columns distinguish source-estimated and transfer-estimated values.

## Rank Validity

Predictive validity is measured as Spearman rank correlation on cumulative monthly model cohorts. Spearman rho is computed with average ranks to handle ties, which are common under benchmark saturation. P-values are two-sided permutation estimates with a fixed random seed. Confidence intervals are paired bootstrap intervals.

## Half-Life

Half-life is estimated by fitting log(rho_t) = log(rho_0) - lambda * t on positive-rho cohort points. Fits with fewer than three positive points, non-positive lambda, or short temporal spans are explicitly flagged rather than converted into headline claims.

## Health Score

The health score combines current rho, linear R2, source top-10 spread, top-5 overlap, pairwise inversion rate, confidence-interval width, estimated-score fraction, exposure proxy, and benchmark age after z-normalization. With only three quantitative source benchmarks, health score rankings should be read as an interpretable triage diagnostic, not as a calibrated universal metric.

## Added Diagnostics

The pipeline also reports transfer gaps and regression residuals. These are useful for NeurIPS-style evidence because they show not only whether ranks correlate, but which models violate the source-benchmark expectation most severely.

The ordering diagnostics report pairwise inversion rates, Kendall tau-b, and top-k leaderboard preservation. These make the analysis less dependent on a single correlation coefficient and better aligned with how leaderboards are used in practice. The leave-one-out table reports whether headline rho values are driven by one influential model.

The mechanism-discriminant table tests whether the HumanEval inversion can be explained by score compression alone. It compares the observed code inversion with a rank-preserving counterfactual in which MMLU scores are replaced by the HumanEval score template before recomputing validity against GPQA Diamond.

## Validation Issue Counts

{validation_df["check"].value_counts().to_string() if not validation_df.empty else "No validation issues."}
"""
    (REPORT_DIR / "methods_limitations.md").write_text(methods, encoding="utf-8")

    latest = {
        pair: rank_df[(rank_df["analysis_pair"].eq(pair)) & (~rank_df["confirmed_only"])].sort_values("cohort_date").tail(1).iloc[0]
        for pair in ["math_zscore", "code", "knowledge"]
        if not rank_df[(rank_df["analysis_pair"].eq(pair)) & (~rank_df["confirmed_only"])].empty
    }
    manuscript = f"""# When Benchmarks Go Stale: Measuring Predictive Validity Decay in ML Leaderboards

## Abstract

Public benchmarks are intended to summarize model capability, but repeated optimization against the same leaderboard can cause their rankings to stop predicting performance on newer, harder transfer tasks. We operationalize this failure as predictive validity: the rank correlation between an older source benchmark and a newer transfer benchmark over overlapping model releases. Across math, code, knowledge, and vision case-study data, we find strong evidence of source-score compression and prominent rank inversions. GSM8K has only weak current rank validity against normalized MATH-500/AIME outcomes (rho={latest['math_zscore'].spearman_rho:.2f}), HumanEval is negatively correlated with LiveCodeBench in the supplied overlap set (rho={latest['code'].spearman_rho:.2f}), and MMLU remains moderately correlated with GPQA Diamond (rho={latest['knowledge'].spearman_rho:.2f}) while still exhibiting large local inversions among similarly scoring models. The available data support a benchmark-health diagnostic and rank-inversion analysis more strongly than a universal half-life estimate; therefore, we report half-life fits only when the temporal evidence is adequate.

## 1. Introduction

Leaderboards shape model development. Once a benchmark becomes the public target, it is no longer only a measurement device: it becomes part of the training, selection, prompting, and reporting environment. The core empirical question is not whether Goodhart's law exists in principle, but whether its effect can be measured early enough to be useful.

The submitted manuscript focuses on three language-model transitions: GSM8K to MATH-500/AIME 2024, HumanEval to LiveCodeBench, and MMLU to GPQA Diamond. The source benchmark is the older, saturated or heavily optimized benchmark; the transfer benchmark is newer, harder, or more distribution-shifted. A healthy source benchmark should preserve relative ordering on the transfer benchmark. A stale benchmark should show compressed source scores, falling rank validity, or rank inversions.

## 2. Data

The canonical dataset is built from the supplied markdown research brief and saved as `outputs/data/canonical_scores.csv`. It contains model name, organization, release month, source score, transfer score, provenance, estimated-score flags, and benchmark-pair identifiers. Scores marked with an asterisk in the source document are retained but flagged, enabling all-data and confirmed-only sensitivity analyses.

Math receives special handling because MATH-500 and AIME 2024 are not the same scale. We therefore report MATH-500-only, AIME-only, and z-normalized combined analyses. Vision is treated as a structural OOD case study rather than a temporal decay curve because exact-version overlap is too sparse for the same rank-correlation-over-time design.

## 3. Methods

Predictive validity is Spearman rank correlation between source and transfer scores in cumulative release-month cohorts. Ties are assigned average ranks, which is important under saturation. Uncertainty is estimated with paired bootstrap intervals; p-values are permutation estimates. We fit an exponential half-life model only to positive-rho cohort sequences with enough temporal support. When rho is negative, non-monotonic, or based on too few positive points, the pipeline reports this as non-estimable instead of forcing a headline number.

The pipeline also writes several legacy exploratory diagnostics retained for audit traceability. The submitted manuscript uses the signed rank-validity, ordering, threshold-sensitivity, provenance-sensitivity, and mechanism-discriminant outputs rather than a scalar health score.

To test whether leaderboard ordering transfers in the way practitioners use it, we additionally compute Kendall tau-b, pairwise inversion rate, and top-k overlap between source and transfer rankings. A pairwise inversion occurs when model A outranks model B on the source benchmark but falls below it on the transfer benchmark. We also run leave-one-model-out influence checks to identify whether any headline correlation is driven by a single model.

## 4. Results

Current predictive validity is weakest for the most saturated source tasks. GSM8K has rho={latest['math_zscore'].spearman_rho:.2f} against normalized MATH-500/AIME transfer scores, with a wide bootstrap interval due to mixed transfer sets and many estimated rows. HumanEval shows rho={latest['code'].spearman_rho:.2f} against LiveCodeBench, indicating that high HumanEval ranks can correspond to substantially worse live-code performance in the provided overlap data. MMLU to GPQA Diamond remains stronger at rho={latest['knowledge'].spearman_rho:.2f}, but this aggregate correlation masks striking local inversions.

The clearest qualitative findings are rank inversions. In code, {code_worst.model_name} moves from rank {code_worst.source_rank:.1f} on HumanEval to rank {code_worst.transfer_rank:.1f} on LiveCodeBench. In knowledge, {worst.model_name} moves from rank {worst.source_rank:.1f} on MMLU to rank {worst.transfer_rank:.1f} on GPQA Diamond. These are the concrete failure cases that make benchmark staleness visible to readers.

Ordering diagnostics sharpen this claim. The task-level inversion and top-k preservation analyses show whether a source leaderboard preserves pairwise preferences, not only whether its aggregate scores correlate with transfer scores. Leave-one-model-out checks report the largest influence cases, separating robust pair-level patterns from correlations that are fragile to individual outliers.

Source-score spread is compressed near ceiling. The cleaned data show GSM8K and HumanEval top models clustered in narrow high-score bands, so source rank differences often reflect tiny score changes while transfer differences remain large.

The vision case study supports the same structural point through distribution shift rather than leaderboard time. Among architectures with both ImageNet and ObjectNet scores, the median drop is {vision_drop:.1f} percentage points.

## 5. Limitations

The strongest limitation is temporal support. The supplied overlap data are adequate for showing saturation, weak validity, and rank inversions, but they are not adequate for a universal predictive-validity half-life claim. Code has only a short 2026 release span; math has mixed transfer benchmarks; vision lacks clean temporal overlap. Estimated and self-reported scores may reflect scaffolding, prompt choices, contamination, or retroactive evaluation. These issues are explicitly encoded in the validation and sensitivity outputs.

## 6. Paper Claim to Use

The defensible claim from this built project is:

> Across the submitted language-model audits, source benchmark scores often remain high while transfer rankings diverge. Predictive-validity loss is visible through rank correlation, score compression, and rank inversions, but half-life estimation requires denser longitudinal leaderboard data than the current public metadata provides.

This is a stronger NeurIPS posture than forcing the originally hoped-for half-life result from insufficient evidence.
"""
    (REPORT_DIR / "neurips_manuscript_draft.md").write_text(manuscript, encoding="utf-8")


def print_summary(rank_df: pd.DataFrame, half_lives: pd.DataFrame, health_df: pd.DataFrame, inversions: pd.DataFrame) -> None:
    print("=" * 72)
    print("PAPER-READY NUMBERS SUMMARY")
    print("=" * 72)
    print("\nHalf-lives (predictive validity decay):")
    for r in half_lives.itertuples():
        if pd.notna(r.half_life_months):
            print(f"  {r.pair_label}: {r.half_life_months:.1f} months ({r.half_life_years:.2f} years) [{r.reliability}]")
        else:
            print(f"  {r.pair_label}: not estimable ({r.status}) [{r.reliability}]")
    print("\nCurrent predictive validity (latest rho):")
    for pair in ["math_zscore", "math_math500", "math_aime2024", "code", "knowledge"]:
        latest = rank_df[(rank_df["analysis_pair"].eq(pair)) & (~rank_df["confirmed_only"])].sort_values("cohort_date").tail(1)
        if not latest.empty:
            row = latest.iloc[0]
            print(f"  {PAIR_LABELS[pair]}: rho={row.spearman_rho:.2f}, n={int(row.n_models)}, CI=[{row.rho_ci95_low:.2f}, {row.rho_ci95_high:.2f}]")
    print("\nMost dramatic rank inversions:")
    for pair in ["knowledge", "code"]:
        worst = inversions[inversions["pair_id"].eq(pair)].sort_values("abs_rank_delta", ascending=False).head(1).iloc[0]
        print(f"  {pair}: {worst.model_name}: rank {worst.source_rank:.1f} -> {worst.transfer_rank:.1f}; scores {worst.source_score:.1f}% -> {worst.transfer_score:.1f}%")
    print("\nHealth Score Rankings:")
    print(health_df[["analysis_pair", "health_score", "health_rank"]].to_string(index=False))
    print("=" * 72)


def main() -> None:
    ensure_dirs()
    df = add_math_zscores(canonical_frame())
    validation_df = validate(df)
    rank_df = compute_rank_correlations(df, confirmed_only=False)
    rank_confirmed_df = compute_rank_correlations(df, confirmed_only=True)
    half_lives = fit_half_lives(rank_df)
    variance_df = compute_variance_compression(df)
    exposure_df = compute_exposure_proxy(df, rank_df)
    inversions = compute_rank_inversions(df)
    gap_df, regression_df = compute_gap_and_regression_diagnostics(df)
    ordering_df = compute_ordering_diagnostics(df)
    health_df = compute_health_scores(df, rank_df, variance_df, exposure_df, ordering_df)
    influence_df = compute_leave_one_out_influence(df)
    mechanism_df = compute_mechanism_discriminants(df, rank_df)
    write_csvs(df, rank_df, rank_confirmed_df, half_lives, variance_df, exposure_df, health_df, inversions, validation_df, gap_df, regression_df, ordering_df, influence_df, mechanism_df)
    make_figures(df, pd.concat([rank_df, rank_confirmed_df], ignore_index=True), half_lives, variance_df, exposure_df, health_df, gap_df, regression_df, ordering_df, influence_df)
    write_reports(df, pd.concat([rank_df, rank_confirmed_df], ignore_index=True), half_lives, health_df, inversions, validation_df, exposure_df, gap_df, regression_df, ordering_df, influence_df, mechanism_df)
    print_summary(pd.concat([rank_df, rank_confirmed_df], ignore_index=True), half_lives, health_df, inversions)


if __name__ == "__main__":
    main()
