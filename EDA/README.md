# Pre-modelling EDA and independent cross-check

This folder holds the data audit and exploratory analysis done before the
regression study in `Code/main.py`. It was produced by a separate pipeline in
[aayush-0131/fi2010-regression-deeplob](https://github.com/aayush-0131/fi2010-regression-deeplob),
which has the scripts and tests that regenerate every file here.

No predictive model was fitted at this stage, and no random split was used.

## Data used

- Source: [FreemanOne FI2010 on Kaggle](https://www.kaggle.com/datasets/freemanone/fi2010),
  NoAuction Z-score, CF9 anchored split. This is a different copy of the data
  from the `alpha000x/fi-2010` files used by `Code/main.py`.
- Train: the first 9 trading days, 362,400 rows. Test: day 10, 31,937 rows.
- 144 predictors (40 raw order book and 104 engineered) and 5 label horizons.
- The source gate checks SHA-256 hashes, dimensions, the index sequence, labels,
  missing values and infinities before computing anything
  (`tables/dataset_overview.csv`). No NaN or infinite values were found.

## Cross-check against the main results

The two pipelines are independent and use different data copies. They agree on
the structural facts the report relies on:

| Quantity | This EDA | Main README / report |
|---|---:|---:|
| Total observations (NoAuction) | 394,337 | 394,337 |
| Constant predictors | 9 | 9 |
| Predictor pairs with \|r\| >= 0.95 | 544 | 544 |
| Numerical rank of the 135 nonconstant predictors | 89 | 89 |

Sources: `tables/constant_predictors.csv`,
`tables/very_high_correlation_pairs.csv` and
`tables/multicollinearity_summary.csv`. These statistics are computed on the
training days.

## Contents

| Path | What it is |
|---|---|
| `docs/FI2010_FEATURE_DICTIONARY.csv` | Name, family and description of every one of the 144 predictors |
| `docs/DECISION_LOG.md` | Methodological decisions made before modelling (chronological split, dataset choice, and so on) |
| `tables/dataset_overview.csv`, `tables/freeman_data_audit*.{csv,json}` | File integrity and source audit |
| `tables/reference_decpre/` | Audit of the DecPre representation, kept for reference |
| `tables/class_distribution*.csv` | Up / stationary / down balance for each horizon, train vs test |
| `tables/per_feature_summary.csv`, `tables/feature_family_summary.csv` | Summary statistics for each feature and each family |
| `tables/*correlation*.csv`, `tables/multicollinearity_summary.csv`, `tables/correlation_eigen_spectrum.csv` | Collinearity and rank analysis |
| `tables/train_test_feature_drift.csv` | Kolmogorov-Smirnov drift between the training days and day 10 |
| `tables/representative_features.csv` | One representative feature for each highly correlated cluster |
| `tables/eda_run_manifest.json` | Run parameters, seeds and timestamp |
| `figures/*.png` | Plots of the tables above |
