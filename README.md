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

STIP uses **synthetic semiconductor measurements**, not proprietary or production manufacturing data.

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

- All semiconductor measurements are synthetic.
- Simulated failures may not reflect the complexity of real silicon defects.
- Evaluation results are specific to the synthetic dataset and experimental setup.
- Feature deviation scores are descriptive, not causal explanations.
- Yield calculations use a simplified passing criterion.
- Very small wafer or lot samples can produce unstable estimates.
- The feature-building script currently loads all input Parquet chunks into memory.

## Project Status

**Implemented:** Synthetic data generation, data ingestion, device feature engineering, anomaly detection, evaluation, feature deviation analysis, wafer analytics, yield analytics, interactive dashboard, and automated tests.

**Future enhancements:** Failure classification, clustering, additional model evaluation, scalable data processing, and deployment improvements.

## Disclaimer

This is an independent engineering and machine learning portfolio project. It is not affiliated with, endorsed by, or developed using proprietary data from Cadence Design Systems.