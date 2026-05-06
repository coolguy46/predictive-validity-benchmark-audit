# Reviewer-facing data

This directory contains the two score tables most relevant to the paper.

- `canonical_n15_scores.csv`: the coherent 15-model HumanEval-to-LiveCodeBench frontier-overlap audit used for the inversion claim.
- `livecodebench_v6_candidate_n34.csv`: the broader 34-row HumanEval-to-LiveCodeBench-v6 public-metadata candidate audit used for the weak-transfer check.
- `v6_source_map.csv`: source-reference map for the 34-row candidate audit. The candidate rows also include per-row source URL fields.

The larger `outputs/` directory contains generated diagnostics and figures. Reviewers who only want to inspect the input score metadata should start here.
