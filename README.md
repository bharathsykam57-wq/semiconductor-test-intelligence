# Semiconductor Test Intelligence Platform (STIP)

**A Python-based semiconductor test analytics and machine learning portfolio project.**

STIP is an independent engineering project that demonstrates how data science techniques can be applied to synthetic semiconductor manufacturing and electrical test measurements.

The platform combines data generation, feature engineering, anomaly detection, engineering analytics, and interactive visualization to support investigation of unusual device behavior.

## Key Features

- **Synthetic Data Generation:** Configurable manufacturing lots, wafers, devices, operating conditions, and simulated failure categories.
- **Data Processing:** Parquet-based data ingestion and device-level feature engineering using Polars.
- **Anomaly Detection:** Isolation Forest identifies devices with unusual electrical measurements.
- **Model Evaluation:** Anomaly detection evaluation, including wafer-held-out testing.
- **Explainability:** Standardized feature deviations help engineers understand unusual measurements. These are descriptive indicators, not direct Isolation Forest feature attributions.
- **Wafer Analytics:** Compare electrical characteristics and investigate wafer-level differences.
- **Yield Analytics:** Calculate device passing rates at wafer and manufacturing-lot levels, including sample-size warnings.
- **Interactive Dashboard:** Explore device measurements, anomaly scores, wafer comparisons, and yield summaries using Streamlit.
- **Automated Testing:** pytest tests covering data generation, validation, feature engineering, anomaly detection, reporting, and analytics.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data processing | Polars, Pandas, NumPy |
| Machine learning | scikit-learn |
| Visualization | Streamlit, Matplotlib, Seaborn |
| Data storage | Parquet, CSV |
| Testing | pytest |
| Version control | Git |

## Dataset

The electrical-test workflow uses **synthetic semiconductor measurements**. The separate WM-811K workflow described below uses real public wafer maps.

The configurable generator supports:

- 20 manufacturing lots
- 5 wafers per lot
- 100 devices per wafer
- 100 measurements per device
- 1,000,000 expected measurement records
- Random seed: 42
- Simulated electrical variations and failure categories

These settings describe the configured full dataset, not the size of every experiment.

A smaller dataset of 501 devices was used during dashboard development and anomaly evaluation.


## Installation and Setup

### Prerequisites

- Python 3.11
- Git
- A terminal (macOS, Linux, or Windows)

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd semiconductor-test-intelligence
```

Replace `<YOUR_REPOSITORY_URL>` with the actual GitHub URL after publishing.

### 2. Create a virtual environment

```bash
python3.11 -m venv .venv
```

### 3. Activate the environment

**macOS / Linux:**

```bash
source .venv/bin/activate
```

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Verify installation

```bash
python -m pip check
python -m pytest tests/ -q
```

All automated tests should pass.

### 6. Generate synthetic data

Follow the commands in the Reproducible Data Pipeline section below.

### 7. Launch the dashboard

```bash
python -m streamlit run dashboard/app.py
```

The dashboard requires the processed device-feature dataset described below.

**Note:** Dependencies were captured from the development environment. A fresh installation has not yet been independently verified.


## Reproducible Data Pipeline

The project supports the following workflow:

1. Generate synthetic semiconductor measurements.
2. Store raw measurements in Parquet or CSV.
3. Aggregate measurements into device-level electrical features.
4. Analyze features using anomaly detection and engineering analytics.
5. Investigate results through the Streamlit dashboard.

### Generate a small test dataset

```bash
python -m src.data_generation.generator \
  --config configs/data_generation.yaml \
  --output-dir data/smoke_test \
  --max-devices 10
```

### Build device-level features

```bash
python -m scripts.build_device_features \
  --input-dir data/smoke_test \
  --output-file data/smoke_test_features/device_features.parquet
```

## Anomaly Detection Evaluation

Isolation Forest was evaluated using a wafer-held-out procedure on 500 synthetic devices from five wafers.

| Metric | Result |
|---|---:|
| True positives | 51 |
| True negatives | 438 |
| False positives | 0 |
| False negatives | 11 |
| Precision | 1.000 |
| Recall | 0.823 |
| F1 score | 0.903 |

These results describe a specific synthetic-data experiment and should not be interpreted as production semiconductor defect-detection performance.

## Yield Analytics

STIP estimates wafer-level and lot-level device passing rates using the following proxy definition:

**A device passes when its aggregated `failure_rate` equals zero.**

Lot yield is calculated using the total number of passing devices divided by the total number of tested devices, rather than averaging wafer percentages.

This is an analytical proxy, not a validated manufacturing yield specification.


## Prepare Dashboard Data

The Streamlit dashboard requires a processed device-level
Parquet dataset.

### Generate synthetic measurements

From the project root:

```bash
python -m src.data_generation.generator \
  --config configs/data_generation.yaml \
  --output-dir data/raw \
  --max-devices 500
```

This generates a limited synthetic dataset for demonstration.

### Build dashboard features

```bash
python -m scripts.build_device_features \
  --input-dir data/raw \
  --output-file data/processed/device_features/device_features.parquet
```

The feature engineering pipeline aggregates raw electrical
measurements into one row per device.

### Launch the dashboard

```bash
python -m streamlit run dashboard/app.py
```

The dashboard loads the generated features and provides
interactive anomaly detection, device investigation,
wafer analytics, and yield analysis.

**Important:** These commands write to the standard project
data directories and may replace existing generated datasets.
Use separate output paths when preserving previous experiments.


## Run the Dashboard

From the project root:

```bash
python -m streamlit run dashboard/app.py
```

The dashboard expects processed device features at:

`data/processed/device_features/device_features.parquet`

## Run Automated Tests

```bash
python -m pytest tests/ -q
```

## Limitations

- Electrical measurements are synthetic; real WM-811K data consists of public die pass/fail maps.
- Simulated failures may not reflect the complexity of real silicon defects.
- Evaluation results are specific to the synthetic dataset and experimental setup.
- Feature deviation scores are descriptive, not causal explanations.
- Yield calculations use a simplified passing criterion.
- Very small wafer or lot samples can produce unstable estimates.
- The feature-building script currently loads all input Parquet chunks into memory.

## Project Status

**Implemented:** Synthetic data generation, data ingestion, device feature engineering, anomaly detection, evaluation, feature deviation analysis, wafer analytics, yield analytics, interactive dashboard, and automated tests.

**Real-data implementation:** WM-811K spatial features, labeled Parquet pipeline, lot-disjoint Random Forest classification and a separate dashboard view (details below).

**Future enhancements:** Image-based classification, grouped cross-validation, clustering, streaming source formats and deployment improvements.

## Disclaimer

This is an independent engineering and machine learning portfolio project. It is not affiliated with, endorsed by, or developed using proprietary data from Cadence Design Systems.
## Real wafer maps: WM-811K

STIP now has two independent workflows: the original **synthetic electrical-test
analysis** and **real public wafer-map pattern classification**. The real-data
workflow uses die pass/fail maps, not electrical measurements.

Dataset reference: [WM-811K example and dataset references](https://www.mathworks.com/help/images/classify-anomalies-on-wafer-defect-maps-using-deep-learning.html).
Obtain the dataset separately and place the trusted legacy pickle at
`data/external/wm811k/LSWMD.pkl`. Dataset files are excluded from Git. Check the
source's access and reuse terms before redistributing data. Pickle loading can
execute code: use only a trusted copy. The local file contains 811,457 maps,
172,950 labeled maps and 638,507 unlabeled maps. The `none` category means a
labeled normal pattern; unlabeled maps are excluded from supervised learning.

### Architecture

```text
Synthetic measurements -> electrical features -> Isolation Forest / yield analytics
Real WM-811K pickle -> label selection -> spatial + count features -> Parquet
                                          -> lot-disjoint Random Forest evaluation
Both workflows -> separate Streamlit views
```

- `src/wafer_map_ingestion/loader.py`: legacy pandas pickle compatibility.
- `src/wafer_map_features/`: label normalization, map validation, spatial features,
  bounded output batches and example maps.
- `src/classification/wafer_classifier.py`: preprocessing, split auditing, training,
  serialized model, predictions, partitions and evaluation JSON.
- `dashboard/real_wafers.py`: compact-artifact real-data analysis view.
- [Feature definitions](docs/wafer_features.md) and [engineering report](docs/ENGINEERING_REPORT.md).

### Reproduce the real-data workflow

Use the existing `.venv`, or follow the installation instructions above. No new
dependencies are required for the real-data pipeline.

```bash
# Smoke test: sample at most 20 wafers per category, reproducibly
.venv/bin/python -m src.wafer_map_features.process_labeled_batch \
  --per-class 20 --batch-size 50 \
  --output data/processed/wm811k_smoke/features.parquet
.venv/bin/python -m src.classification.wafer_classifier \
  --input data/processed/wm811k_smoke/features.parquet \
  --output-dir data/processed/wm811k_smoke/model --trees 20

# Full labeled extraction (default: all 172,950 labeled wafers)
.venv/bin/python -m src.wafer_map_features.process_labeled_batch --batch-size 2000
# Seeded 80/20 split by lot (the fraction applies to lots, not rows)
.venv/bin/python -m src.classification.wafer_classifier
# Respect official metadata; purge training rows from shared lots if present
.venv/bin/python -m src.classification.wafer_classifier \
  --split metadata --output-dir data/processed/wm811k/model_official

.venv/bin/python -m pytest -q
.venv/bin/python -m streamlit run dashboard/app.py
```

Select **Real WM-811K wafer maps** in the sidebar. This view works even when
synthetic device features are absent. It displays yield/category distributions,
example maps, model metrics, baseline comparison, confusion matrix, per-class
metrics, train/test class counts, feature importance and engineering observations.
Its artifact directory is editable; expected filenames are `features.parquet`,
`features.examples.npz`, `features.manifest.json`, and `model/evaluation.json`.
The default dashboard shows the grouped evaluation. Official-split artifacts are
stored separately so they do not replace the grouped results.

The pickle is monolithic and must be loaded into memory once per extraction;
output batches do **not** make pickle loading streaming. The pipeline never
expands the roughly billion die observations into a table. It processes one map
at a time and writes bounded feature batches. Reduce `--batch-size` for feature
output memory; `--per-class` reduces extraction/training work but does not reduce
pickle loading memory. Do not run concurrent pickle extraction processes on a
16 GB machine. Train from Parquet after extraction exits.

### Evaluation methodology and limitations

Random Forest uses 200 trees, minimum leaf size 2, balanced subsample class
weights, seed 42 and two workers by default. Median imputation is fitted on the
training partition only. No scaling is required by this tree baseline. Neither
wafer identifiers, lot names nor training/test labels enter the model. A most
frequent class classifier is fitted on exactly the same training partition.
`failureType` is the target. JSON reports include accuracy, macro/weighted F1,
per-class precision/recall, confusion matrix, class counts and impurity feature
importance. Partition artifacts make lot separation auditable.

The grouped evaluation holds out whole lots without class stratification; rare
classes may be absent and are explicitly reported. The official metadata option
preserves test wafers and removes training wafers whose lots occur in test. Both
modes reject missing lot identifiers. A single split does not measure uncertainty
or guarantee generalization across fabrication processes. No hyperparameter
search or test-driven tuning was performed. Capped per-class smoke experiments
alter prevalence and must not be reported as full-dataset performance. See the
engineering report for measured results.

Historical sections above describing all data as synthetic apply to the original
electrical-test workflow. Real wafer maps do not provide physical root-cause
labels or electrical readings; pattern recognition is not validated process
failure diagnosis. Aggregate spatial descriptors lose fine image structure and
can depend on map resolution. Ordinary unit tests use fixtures and do not load
the external dataset. Generated data/models remain ignored, and no push is part
of this workflow.

### Verified results (2026-11-10)

All 172,950 labeled wafers were processed successfully. With the default 200-tree
configuration and seed 42:

| Split | Train / test wafers | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Lot-grouped holdout | 138,503 / 34,447 | 0.953203 | 0.742962 | 0.947203 |
| Official metadata | 54,355 / 118,595 | 0.897930 | 0.539357 | 0.911862 |

Both evaluations had zero shared lots. The majority baseline accuracy was
0.855401 on grouped holdout and 0.933437 on the official test set; the official
baseline exceeded Random Forest accuracy while having macro F1 of only 0.107286.
Scratch recall remains poor. Full results and limitations are in the engineering
report and versioned evaluation JSON files.

Validation: `.venv/bin/python -m pytest -q` completed with **148 passed**;
`.venv/bin/python -m pip check` reported **No broken requirements found**.
Real-data extraction and both model evaluations ran against the local dataset.
Fixture-based dashboard tests cover missing and populated artifacts.
