# A Preprocessing Framework for Calibration Transfer of Near-infrared Spectroscopic Data

This folder contains the code used for the revised manuscript.

## Folder structure

- `code_matlab/`
  - Main MATLAB implementation of the proposed preprocessing-selection framework.
  - `demo.m`: main GA-based workflow and Table 3 Top-5 candidate output.
  - `preprocessCombos_domain.m`: GA-based candidate-pool generation followed by WD-based dual-metric selection.
  - `run_none_tpt_benchmarks.m`: NONE and TPT benchmark results.
  - `run_wasserstein_n_sensitivity.m`: quantile-grid sensitivity analysis.
  - `run_metric_stability_comparison.m`: WD/MMD metric-substitution comparison for Table 6.
  - `run_grid_search_benchmark.m`: GA versus exhaustive grid-search comparison for Table 7. By default, this script runs both the wheat and pharmaceutical tablet datasets and writes the selected-model summaries to `code_matlab/results/`.

- `code_matlab_grid_search/`
  - Grid-search version of the main MATLAB workflow.
  - `demo_grid_search.m`: full workflow entry point. It keeps the same WD calculation and normalized RMSECV-WD selection rule as `code_matlab/demo.m`, but replaces GA candidate-pool generation with exhaustive enumeration.
  - `preprocessCombos_domain_grid_search.m`: exhaustive grid-search candidate-pool generation followed by the same WD-based dual-metric selection.

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

## Grid-search implementation

The repository provides both a direct comparison script and a full grid-search version of the main workflow. `code_matlab/run_grid_search_benchmark.m` reproduces the GA-versus-grid-search comparison reported in Table 7. The separate `code_matlab_grid_search/` folder provides the full proposed workflow with exhaustive enumeration replacing GA candidate-pool generation. This grid-search version evaluates all 384 preprocessing pipelines combined with the predefined LV range from 6 to 10 and applies the same WD calculation and normalized RMSECV-WD final selection rule used by the GA-based workflow.
