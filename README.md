# Anonymous Reproduction Package

This folder contains the anonymized source code and derived score metadata for the paper
`When Benchmark Rankings Stop Transferring: Auditing Predictive-Validity Collapse in Frontier Benchmarks`.

The package is intended to let reviewers reproduce the reported benchmark-validity diagnostics from public score metadata. It does not include benchmark test items, model weights, copied leaderboard pages, paper drafts, or author-identifying working notes.

## Contents

- `data/canonical_n15_scores.csv`: reviewer-facing canonical 15-model HumanEval-to-LiveCodeBench score table.
- `data/livecodebench_v6_candidate_n34.csv`: reviewer-facing 34-row LiveCodeBench-v6 candidate audit table.
- `data/v6_source_map.csv`: source-reference map for the 34-row candidate audit.
- `src/benchmark_validity_pipeline.py`: builds the canonical score tables, rank-validity diagnostics, sensitivity tables, and main figures.
- `src/deep_research_candidate_audit.py`: recomputes threshold statistics for the public-metadata LiveCodeBench-v6 candidate audit.
- `src/protocol_sensitivity_audit.py`: combines canonical, candidate, positive-control, and stress-test rows into the protocol-sensitivity summary used in the paper.
- `src/supplemental_robustness_checks.py`: writes release-month residualization, leave-organization-out, and single-scale math checks.
- `outputs/data/`: generated canonical score tables.
- `outputs/analysis/`: generated rank correlations, ordering diagnostics, influence checks, and mechanism diagnostics.
- `outputs/deep_research_candidate_audit/`: generated LiveCodeBench-v6 candidate-audit rows and threshold summaries.
- `outputs/figures/`: generated figure images used by the manuscript.

## Setup

Use Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
```

## Quick inspection

Reviewers who want to inspect the score metadata before running code should start with:

```text
data/canonical_n15_scores.csv
data/livecodebench_v6_candidate_n34.csv
data/v6_source_map.csv
```

The `outputs/` directory contains generated diagnostics and figures. It is included for transparency, but the two reviewer-facing score tables are copied into `data/` so they do not have to be found among generated CSV files.

## Reproduce

Run all analysis scripts:

```bash
python run_all.py
```

Or run the scripts manually:

```bash
python src/benchmark_validity_pipeline.py
python src/deep_research_candidate_audit.py
python src/protocol_sensitivity_audit.py
python src/supplemental_robustness_checks.py
```

The scripts overwrite generated CSV/PNG files under `outputs/`.

## Notes on Evidence

The package uses derived public benchmark-score metadata and provenance fields. Some candidate rows are from secondary leaderboards or extracted public reports, so the paper treats those rows as public-metadata stress tests rather than controlled re-evaluations. Reported statistics are recomputed by the scripts in this folder; source webpages and benchmark creators remain the authority for benchmark terms and original score reporting.
