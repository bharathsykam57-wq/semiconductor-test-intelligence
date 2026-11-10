# STIP engineering report

Experiment date: 2026-11-10. All results below were measured locally using the
existing Python 3.11 virtual environment. This is a portfolio baseline, not a
validated production diagnostic system.

## Problem and scope

Electrical anomaly detection and wafer-map pattern recognition answer different
engineering questions. The existing synthetic workflow generates electrical
measurements, aggregates device statistics, detects unusual devices with
Isolation Forest, and reports wafer/lot yield proxies. That workflow is preserved.
The new real-data workflow predicts the provided `failureType` pattern label
from compact spatial descriptors of WM-811K die pass/fail maps.

The local legacy pickle contained 811,457 wafer records: 172,950 labeled and
638,507 unlabeled. Public dataset background and references are available in
[the MathWorks WM-811K example](https://www.mathworks.com/help/images/classify-anomalies-on-wafer-defect-maps-using-deep-learning.html).
No dataset or binary model is added to Git. Source access/reuse terms should be
checked separately before redistribution.

## Method

A trusted-pickle loader handles legacy pandas index module names and latin1
encoding. Labels are normalized from nested arrays. Empty/unknown labels are
excluded, while `none` is retained as a supervised normal class. Source indices
are preserved as wafer identifiers.

Extraction validates map dimensionality and values, and computes 22 count,
yield, geometry and spatial features per wafer. Center/edge and quadrant rates,
failure centroid, radial fractions/spread and orthogonal adjacency summarize
spatial patterns. See [exact definitions](wafer_features.md). Map dimensions vary;
normalized coordinates allow comparisons without resizing maps. Zero-tested maps
have safe finite outputs and are excluded from dashboard yield histograms.

The pickle must be loaded in full; feature output is written in 2,000-wafer
Parquet batches without materializing die-level tables. Extraction completed for
all labeled wafers. Peak process memory was not instrumented, so no measured
memory ceiling is claimed. Smaller sampling caps do not avoid loading the pickle.

A smoke experiment first sampled 20 wafers from each category (180 total) using
seed 42. Extraction, Parquet serialization and a 20-tree model ran successfully.
Its small-sample performance is not used as an estimate of population performance.

Full experiments used all labeled wafers, 200 Random Forest trees,
`min_samples_leaf=2`, balanced subsample class weighting, two workers and seed 42.
Median imputation was fitted only on training data; tree models do not require
feature scaling. Only the 22 defined numerical features enter the model. No
hyperparameter tuning was performed. A most-frequent DummyClassifier uses the
same train/test partitions.

Two evaluations were run:

1. **Grouped holdout:** GroupShuffleSplit selects 20% of unique lots, seed 42.
   Training: 138,503 wafers / 8,609 lots. Test: 34,447 wafers / 2,153 lots.
2. **Official metadata:** Training: 54,355 wafers / 5,809 lots. Test: 118,595
   wafers / 4,953 lots. The dataset column is spelled `trianTestLabel`.

Both splits had **zero lot overlap**, all nine classes in both partitions and
no purged rows. The metadata implementation preserves the official test set and
purges overlapping training lots if another dataset version requires it. The
grouped split ignores official membership and is therefore a distinct experiment,
not a reproduction of the official benchmark split. Partition Parquet files
record wafer identifiers and lot membership for auditing.

## Measured results

| Evaluation | Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| Grouped holdout | Random Forest | 0.953203 | 0.742962 | 0.947203 |
| Grouped holdout | Most frequent | 0.855401 | 0.102452 | 0.788736 |
| Official metadata | Random Forest | 0.897930 | 0.539357 | 0.911862 |
| Official metadata | Most frequent | 0.933437 | 0.107286 | 0.901302 |

The official baseline exceeds Random Forest accuracy because 93.34% of its test
set is `none`. Macro F1 exposes the majority baseline's inability to detect defect
classes. Random Forest improves macro and weighted F1, but its official-split
accuracy is worse. These tradeoffs must be reported together.

| Class | Total labeled | Grouped precision | Grouped recall | Official precision | Official recall |
|---|---:|---:|---:|---:|---:|
| Center | 4,294 | 0.9163 | 0.8800 | 0.3215 | 0.5685 |
| Donut | 555 | 0.7863 | 0.7603 | 0.6122 | 0.4110 |
| Edge-Loc | 5,189 | 0.7018 | 0.5580 | 0.2254 | 0.5826 |
| Edge-Ring | 9,680 | 0.9608 | 0.9399 | 0.7326 | 0.5622 |
| Loc | 3,593 | 0.7112 | 0.3860 | 0.2230 | 0.3634 |
| Near-full | 149 | 1.0000 | 0.9000 | 0.9500 | 1.0000 |
| Random | 866 | 0.8712 | 0.9103 | 0.8333 | 0.6615 |
| Scratch | 1,193 | 0.5517 | 0.0717 | 0.1694 | 0.0303 |
| none | 147,431 | 0.9654 | 0.9926 | 0.9749 | 0.9278 |

Full confusion matrices, support counts, per-class metrics, class distributions
and feature importances are preserved in the small reviewable reports:

- [Grouped evaluation](../reports/wm811k_grouped_evaluation.json)
- [Official evaluation](../reports/wm811k_official_evaluation.json)

The highest grouped impurity importances were radius standard deviation
(0.08367), coordinate spread (0.08023), adjacent failure fraction (0.07082),
yield (0.06905), and outer failure fraction (0.06411). These are descriptive
model indicators; correlated features and continuous-variable bias limit causal
interpretation. The difference between evaluation splits suggests distribution
sensitivity but does not establish its physical cause.

## Integration and validation

The Streamlit sidebar switches between independent synthetic and real views.
Real-data analysis reads compact Parquet/NPZ/JSON artifacts, never the 2 GB pickle.
It shows yield/category distributions, example maps, baseline and model scores,
confusion matrix, per-class results, feature importance and engineering notes.
The real view can operate without synthetic device features.

Automated fixture tests cover invalid/empty maps, die counts and count integrity,
zero-tested and rectangular maps, center/edge/quadrant/centroid/adjacency behavior,
nested labels and exclusions, deterministic sampling across batch sizes, Parquet
schema, serialization, grouped leakage prevention, metadata overlap purging and
both missing-artifact and complete-artifact dashboard rendering. Ordinary tests
do not need the source dataset. Final validation: `.venv/bin/python -m pytest -q` returned **148 passed**;
`.venv/bin/python -m pip check` reported **No broken requirements found**.

## Limitations and next steps

- Only labeled maps are modeled; annotation selection can bias conclusions.
- Severe class imbalance makes aggregate accuracy insufficient. Rare-class
  metrics have limited support (only 30 Near-full test wafers in grouped holdout).
- Scratch recall is poor: 0.0717 grouped and 0.0303 official. Aggregate descriptors
  lose directional thin-line structure. Directional, connected-component and
  image-based representations are worthwhile next experiments.
- One seeded holdout and one official split do not provide confidence intervals.
  Use grouped cross-validation and nested tuning on training data before further
  model selection; keep an untouched final holdout for claims.
- Lot separation prevents one known leakage path, but geometry/process correlations
  and dataset shift remain. There are no temporal or external-fab validations.
- A label indicates a visual pattern, not a demonstrated physical root cause.
  Zero tested dies have undefined physical yield despite a safe numerical zero.
- Pickle loading uses pandas' internal compatibility API and requires trusted
  input. A separately prepared map store would avoid monolithic loading.
- Binary artifacts depend on compatible library versions and must be regenerated
  using the documented commands. Dependencies pass the current environment check;
  a fresh installation on another machine was not tested.
