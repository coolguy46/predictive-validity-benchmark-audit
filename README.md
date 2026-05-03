# Anonymous Reproduction Package

This folder contains the anonymized source code and derived score metadata for the paper
`When Benchmark Rankings Stop Transferring: Auditing Predictive-Validity Collapse in Frontier Benchmarks`.

The package is intended to let reviewers reproduce the reported benchmark-validity diagnostics from public score metadata. It does not include benchmark test items, model weights, copied leaderboard pages, paper drafts, or author-identifying working notes.

## Contents

- `src/benchmark_validity_pipeline.py`: builds the canonical score tables, rank-validity diagnostics, sensitivity tables, and main figures.
- `src/deep_research_candidate_audit.py`: parses the public-metadata LiveCodeBench-v6 candidate audit table and recomputes threshold statistics.
- `src/protocol_sensitivity_audit.py`: combines canonical, candidate, positive-control, and stress-test rows into the protocol-sensitivity summary used in the paper.
- `src/frontier_validity_audit.py`: auxiliary threshold audit for an extracted frontier candidate table.
- `outputs/data/`: derived canonical score tables.
- `outputs/analysis/`: generated rank correlations, ordering diagnostics, influence checks, and mechanism diagnostics.
- `outputs/deep_research_candidate_audit/`: generated LiveCodeBench-v6 candidate-audit rows and threshold summaries.
- `outputs/frontier_audit/`: generated threshold and protocol-sensitivity summaries.
- `outputs/figures/`: generated figure images used by the manuscript.

## Setup

Use Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
```

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
```

The scripts overwrite generated CSV/PNG files under `outputs/`.

## Notes on Evidence

The package uses derived public benchmark-score metadata and provenance fields. Some candidate rows are from secondary leaderboards or extracted public reports, so the paper treats those rows as public-metadata stress tests rather than controlled re-evaluations. Reported statistics are recomputed by the scripts in this folder; source webpages and benchmark creators remain the authority for benchmark terms and original score reporting.

