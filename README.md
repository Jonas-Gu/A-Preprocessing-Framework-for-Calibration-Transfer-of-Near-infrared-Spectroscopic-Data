# Code for NIR Calibration Transfer Manuscript

This folder contains the code used for the revised manuscript.

## Folder structure

- `code_matlab/`
  - Main MATLAB implementation of the proposed preprocessing-selection framework.
  - `demo.m`: main workflow and Table 3 Top-5 candidate output.
  - `run_none_tpt_benchmarks.m`: NONE and TPT benchmark results.
  - `run_wasserstein_n_sensitivity.m`: quantile-grid sensitivity analysis.
  - `run_metric_stability_comparison.m`: WD/MMD metric-substitution comparison for Table 6.
  - `run_grid_search_benchmark.m`: GA versus grid-search comparison for Table 7.

- `dipls_python/`
  - `dipls1_original_covariance.ipynb`: di-PLS-1 benchmark.
  - `dipls2_tikhonov_gram.ipynb`: di-PLS-2 benchmark.
  - `dipals.py` and `functions.py`: local dependencies used by `dipls1_original_covariance.ipynb`.

- `umprecht_mmd_baseline/`
  - `umprecht_mmd_baseline.py`: Umprecht-style MMD-based preprocessing-selection baseline.
  - `*_funcs.py` and `utils_config.py`: preprocessing modules used by the baseline.

## Data files

The `.mat` files included here are the datasets used by the scripts. The split-index CSV files are expected to be released together with this code repository.

## Notes

The MATLAB scripts use fixed random seeds where stochastic optimization is involved. Runtime values may vary across machines, but selected preprocessing pipelines and prediction metrics should match the manuscript tables under the same software environment.
