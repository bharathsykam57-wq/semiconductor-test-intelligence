
# Semiconductor Test Intelligence Platform (STIP)

**Semiconductor test analytics, anomaly detection, and real wafer-map defect classification using Python and machine learning.**

STIP is an independent engineering and data science portfolio project that explores how machine learning, statistical analysis, and data engineering can support semiconductor test-data investigation.

The platform implements two complementary workflows:

1. **Synthetic electrical-test analytics:** Generate semiconductor electrical measurements, engineer device-level features, detect anomalies using Isolation Forest, and analyze wafer-level and lot-level yield.
2. **Real wafer-map classification:** Process the public WM-811K dataset, extract spatial and statistical features from wafer maps, classify defect patterns using Random Forest, and evaluate generalization with lot-disjoint train/test splits.

Both workflows are accessible through an interactive Streamlit dashboard.

The project emphasizes reproducibility, engineering interpretation, leakage-aware evaluation, and transparent reporting of model limitations.

> **Disclaimer:** This is an independent educational and portfolio project. It is not affiliated with or endorsed by Cadence Design Systems, and it does not use proprietary semiconductor manufacturing data.

---

## 1. Project Highlights

- End-to-end Python data engineering and machine learning pipelines
- Synthetic semiconductor electrical-test data generation
- Parquet-based processing using Polars, Pandas, and NumPy
- Isolation Forest anomaly detection on electrical-test features
- Wafer-level and manufacturing-lot-level yield analytics
- Processing of 172,950 labeled real WM-811K wafer maps
- Spatial and statistical wafer-map feature engineering
- Random Forest multiclass defect-pattern classification
- Lot-disjoint model evaluation to reduce data leakage
- Comparison against a majority-class baseline
- Interactive Streamlit dashboards for engineering investigation
- Automated testing with pytest
- Engineering documentation covering assumptions, limitations, and experimental results

---

## 2. Machine Learning Results

### Real WM-811K wafer-map classification

The real-data workflow processes all **172,950 labeled wafer maps** from the locally obtained WM-811K dataset.

A Random Forest classifier predicts wafer defect-pattern categories using engineered spatial and statistical features.

Two evaluation protocols were implemented.

| Evaluation protocol | Training wafers | Test wafers | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| Lot-grouped holdout | 138,503 | 34,447 | 95.32% | 0.743 | 0.947 |
| Official metadata split with lot-disjoint filtering | 54,355 | 118,595 | 89.79% | 0.539 | 0.912 |

**Evaluation safeguards:**

- Training and test partitions contain zero overlapping manufacturing lots.
- Lot identifiers and wafer identifiers are excluded from predictive features.
- Missing-value imputation is fitted on the training partition only.
- A majority-class baseline is evaluated on the same partitions.
- Class-specific precision, recall, and confusion matrices are recorded.
- Both evaluation protocols use the same model configuration.

### Baseline comparison

| Evaluation | Random Forest accuracy | Majority baseline accuracy |
|---|---:|---:|
| Lot-grouped holdout | 95.32% | 85.54% |
| Official metadata split | 89.79% | 93.34% |

For the official metadata evaluation, the majority baseline achieves a higher accuracy but only **0.107 macro F1**, compared with **0.539 macro F1** for Random Forest.

This illustrates the importance of metrics beyond accuracy when evaluating imbalanced multiclass classification problems.

**Important limitations:** Minority defect categories remain challenging, particularly Scratch. Performance varies substantially between evaluation protocols. These results do not establish production readiness or generalization to other semiconductor fabrication processes.

Detailed results are available in:

- [Engineering Report](docs/ENGINEERING_REPORT.md)
- [Grouped Evaluation Results](reports/wm811k_grouped_evaluation.json)
- [Official Evaluation Results](reports/wm811k_official_evaluation.json)

### Synthetic electrical-test anomaly detection

Isolation Forest was evaluated on 500 synthetic devices from five wafers using a wafer-held-out procedure.

| Metric | Result |
|---|---:|
| True positives | 51 |
| True negatives | 438 |
| False positives | 0 |
| False negatives | 11 |
| Precision | 1.000 |
| Recall | 0.823 |
| F1 score | 0.903 |

These results apply only to the specific synthetic-data experiment and should not be interpreted as validated semiconductor manufacturing performance.

---

## 3. Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data processing | Polars, Pandas, NumPy |
| Machine learning | scikit-learn |
| Data visualization | Streamlit, Matplotlib, Seaborn |
| Data storage | Parquet, CSV, NumPy artifacts |
| Model serialization | Joblib |
| Automated testing | pytest |
| Version control | Git, GitHub |

---

## 4. System Architecture

STIP contains two independent analytical workflows.

```text
                  Semiconductor Test Intelligence Platform
                                    |
                  +-----------------+-----------------+
                  |                                   |
          Synthetic Workflow                  Real WM-811K Workflow
                  |                                   |
       Synthetic Data Generator                 Wafer-Map Ingestion
                  |                                   |
        Electrical Measurements               Label Selection
                  |                                   |
         Device Feature Engineering          Spatial Feature Extraction
                  |                                   |
           +------+-------+                      Parquet Dataset
           |              |                            |
     Isolation Forest   Yield Analytics          Lot-Disjoint Splits
           |              |                            |
     Anomaly Scores   Wafer/Lot Metrics           Random Forest
           |              |                            |
           +------+-------+                      Model Evaluation
                  |                                   |
                  +-----------------+-----------------+
                                    |
                            Streamlit Dashboard
                                    |
                     Engineering Analysis and Reporting
```

The workflows are intentionally separate because synthetic electrical measurements and real wafer-map defect patterns represent different data modalities.

---

## 5. Repository Structure

```text
semiconductor-test-intelligence/
|
|-- configs/
|   |-- data_generation.yaml
|
|-- dashboard/
|   |-- app.py
|   |-- real_wafers.py
|
|-- data/
|   |-- external/
|   |-- processed/
|
|-- docs/
|   |-- ENGINEERING_REPORT.md
|   |-- wafer_features.md
|
|-- reports/
|   |-- wm811k_grouped_evaluation.json
|   |-- wm811k_official_evaluation.json
|
|-- scripts/
|   |-- build_device_features.py
|
|-- src/
|   |-- classification/
|   |   |-- wafer_classifier.py
|   |
|   |-- data_generation/
|   |
|   |-- wafer_map_ingestion/
|   |   |-- loader.py
|   |   |-- inspect_dataset.py
|   |
|   |-- wafer_map_features/
|       |-- batch_features.py
|       |-- process_labeled_batch.py
|       |-- process_real_batch.py
|       |-- select_labeled_wafers.py
|       |-- spatial.py
|       |-- wafer_statistics.py
|
|-- tests/
|   |-- test_real_wafer_pipeline.py
|   |-- test_real_wafer_dashboard.py
|
|-- .gitignore
|-- README.md
|-- requirements.txt
```

The structure above highlights the principal components rather than every file in the repository.

Generated datasets, trained model artifacts, and the original WM-811K dataset are excluded from Git.

---

## 6. Installation and Setup

### Prerequisites

- Python 3.11
- Git
- A terminal
- Sufficient memory and disk space for the selected workflow

The full WM-811K extraction is memory-intensive because the source dataset is stored in a monolithic legacy pickle.

### Step 1: Clone the repository

```bash
git clone https://github.com/bharathsykam57-wq/semiconductor-test-intelligence.git

cd semiconductor-test-intelligence
```

### Step 2: Create a virtual environment

```bash
python3.11 -m venv .venv
```

### Step 3: Activate the environment

**macOS / Linux:**

```bash
source .venv/bin/activate
```

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### Step 4: Install dependencies

```bash
python -m pip install --upgrade pip

python -m pip install -r requirements.txt
```

### Step 5: Verify installation

```bash
python -m pip check

python -m pytest tests/ -q
```

The automated tests use local fixtures and do not require downloading the full WM-811K dataset.

**Reproducibility note:** The dependencies were captured from the development environment. A fresh-environment installation has not yet been independently verified.

---

## 7. Workflow A: Synthetic Electrical-Test Analytics

### Overview

This workflow simulates semiconductor electrical measurements and investigates device behavior using data engineering, anomaly detection, and statistical analysis.

The generator supports configurable:

- Manufacturing lots
- Wafers per lot
- Devices per wafer
- Measurements per device
- Operating conditions
- Simulated failure categories
- Random seeds

### Configured full dataset

| Parameter | Value |
|---|---:|
| Manufacturing lots | 20 |
| Wafers per lot | 5 |
| Devices per wafer | 100 |
| Measurements per device | 100 |
| Expected measurement records | 1,000,000 |
| Random seed | 42 |

These values describe the configured full dataset, not every individual experiment.

### Step 1: Generate a small test dataset

From the repository root:

```bash
python -m src.data_generation.generator \
  --config configs/data_generation.yaml \
  --output-dir data/smoke_test \
  --max-devices 10
```

### Step 2: Build device-level features

```bash
python -m scripts.build_device_features \
  --input-dir data/smoke_test \
  --output-file data/smoke_test_features/device_features.parquet
```

The pipeline aggregates raw electrical measurements into device-level analytical features.

### Step 3: Prepare the standard dashboard dataset

```bash
python -m src.data_generation.generator \
  --config configs/data_generation.yaml \
  --output-dir data/raw \
  --max-devices 500
```

Then:

```bash
python -m scripts.build_device_features \
  --input-dir data/raw \
  --output-file data/processed/device_features/device_features.parquet
```

The synthetic dashboard expects:

```text
data/processed/device_features/device_features.parquet
```

**Warning:** Commands writing to standard output directories may replace previously generated datasets. Use alternative paths when preserving earlier experiments.

### Anomaly detection

Isolation Forest identifies devices exhibiting unusual combinations of electrical measurements.

Standardized feature deviations provide descriptive context for interpreting anomalous devices.

These deviations are not causal explanations or direct Isolation Forest feature attributions.

### Yield analytics

The project estimates wafer-level and lot-level passing rates using a simplified analytical definition:

**A device passes when its aggregated `failure_rate` equals zero.**

Lot yield is calculated as:

```text
Lot yield = Total passing devices / Total tested devices
```

This approach avoids incorrectly averaging wafer percentages when wafer sample sizes differ.

The calculated yield is an analytical proxy, not a validated manufacturing yield specification.

---

## 8. Workflow B: Real WM-811K Wafer-Map Classification

### Overview

WM-811K is a public wafer-map dataset used for semiconductor defect-pattern research.

The local dataset contains:

| Dataset component | Count |
|---|---:|
| Total wafer maps | 811,457 |
| Labeled wafer maps | 172,950 |
| Unlabeled wafer maps | 638,507 |

The supervised classification workflow uses labeled maps only.

The `none` category represents a labeled normal wafer pattern and is distinct from an unlabeled record.

### Wafer-map categories

The classification task includes the following labels:

- none
- Center
- Donut
- Edge-Loc
- Edge-Ring
- Loc
- Near-full
- Random
- Scratch

These are wafer-map pattern labels, not verified physical manufacturing root causes.

### Dataset acquisition

The WM-811K dataset must be obtained separately.

Dataset background and references:

[MATLAB — Wafer Defect Map Classification](https://www.mathworks.com/help/images/classify-anomalies-on-wafer-defect-maps-using-deep-learning.html)

Place the trusted dataset file at:

```text
data/external/wm811k/LSWMD.pkl
```

The dataset is not included in this GitHub repository.

**Security warning:** Python pickle files can execute arbitrary code during deserialization. Load only a trusted dataset copy. Review the dataset's access and reuse terms before redistribution.

### Feature engineering

The pipeline extracts numerical features describing the spatial distribution of passing and failing die positions.

Feature groups include:

- Die counts and failure proportions
- Wafer-map dimensions
- Spatial distribution of failing dies
- Position-based statistics
- Center and edge characteristics
- Other engineered spatial descriptors

The full feature definitions are documented in:

[Wafer Feature Engineering Documentation](docs/wafer_features.md)

The feature representation contains 22 numerical features.

### Step 1: Run a small reproducible experiment

```bash
python -m src.wafer_map_features.process_labeled_batch \
  --per-class 20 \
  --batch-size 50 \
  --output data/processed/wm811k_smoke/features.parquet
```

This selects at most 20 labeled wafer maps per category.

The sampling limit changes the category distribution, so smoke-test results must not be presented as full-dataset performance.

### Step 2: Train a small model

```bash
python -m src.classification.wafer_classifier \
  --input data/processed/wm811k_smoke/features.parquet \
  --output-dir data/processed/wm811k_smoke/model \
  --trees 20
```

### Step 3: Process the full labeled dataset

```bash
python -m src.wafer_map_features.process_labeled_batch \
  --batch-size 2000
```

The full extraction processes all 172,950 labeled wafer maps.

### Step 4: Run lot-grouped evaluation

```bash
python -m src.classification.wafer_classifier
```

The default evaluation uses a seeded 80/20 split by manufacturing lot.

The split fraction applies to lots rather than individual wafer rows.

### Step 5: Run official metadata evaluation

```bash
python -m src.classification.wafer_classifier \
  --split metadata \
  --output-dir data/processed/wm811k/model_official
```

This protocol respects the dataset's official metadata while removing training wafers belonging to lots represented in the test partition.

The resulting training and test sets have zero overlapping lots.

### Memory considerations

The WM-811K pickle is monolithic and must be loaded into memory during extraction.

Although extracted features are written in bounded batches, loading the source pickle is not streaming.

The pipeline processes individual wafer maps without expanding all die positions into a single massive tabular dataset.

For memory-constrained machines:

- Avoid running multiple full extraction processes simultaneously.
- Reduce `--batch-size` to control feature-output batching.
- Use `--per-class` for smaller experiments.
- Complete extraction before starting model training.

Reducing `--per-class` does not reduce the memory needed to load the original pickle.

---

## 9. Classification Model and Evaluation

### Random Forest configuration

The default full-data classifier uses:

| Parameter | Value |
|---|---|
| Algorithm | Random Forest |
| Number of trees | 200 |
| Minimum leaf size | 2 |
| Class weighting | Balanced subsample |
| Random seed | 42 |
| Parallel workers | 2 |
| Missing-value handling | Median imputation |
| Target | `failureType` |

Median imputation is fitted using training data only.

Feature scaling is not required for this tree-based baseline.

### Data leakage prevention

The evaluation pipeline includes several safeguards:

1. Training and test partitions do not share manufacturing lots.
2. Wafer and lot identifiers are not predictive inputs.
3. Training-only preprocessing is applied before evaluation.
4. A majority-class baseline uses the same data partition.
5. Split information and class counts are recorded for auditing.

### Evaluation metrics

The evaluation reports include:

- Accuracy
- Macro F1
- Weighted F1
- Per-class precision
- Per-class recall
- Confusion matrix
- Training and test class distributions
- Feature importance
- Majority-class baseline comparison

Macro F1 is particularly useful because WM-811K has substantial class imbalance.

### Interpretation

The lot-grouped and official metadata evaluations produce different results.

This difference demonstrates that model performance depends on how training and test data are partitioned.

A single holdout experiment cannot establish uncertainty bounds or guarantee generalization to new fabrication processes.

The project does not claim that Random Forest is the optimal classifier.

No hyperparameter search or test-driven tuning was performed.

---

## 10. Interactive Streamlit Dashboard

The dashboard provides two separate analytical views.

### Synthetic electrical-test analytics

Features include:

- Device-level electrical measurements
- Isolation Forest anomaly scores
- Feature deviation analysis
- Wafer comparisons
- Wafer-level yield
- Manufacturing-lot yield
- Engineering investigation summaries

### Real WM-811K wafer maps

Features include:

- Dataset and wafer-map summaries
- Wafer yield distributions
- Defect-category distributions
- Example wafer-map visualizations
- Model evaluation metrics
- Majority-baseline comparison
- Confusion matrix
- Per-class precision and recall
- Feature importance
- Train/test category counts
- Engineering observations and limitations

### Launch the dashboard

From the repository root:

```bash
python -m streamlit run dashboard/app.py
```

Select the desired workflow in the sidebar.

The real wafer-map view can operate without the synthetic device-feature dataset, provided its required real-data artifacts are available.

### Real-data artifacts

The real-data dashboard uses generated artifacts such as:

```text
features.parquet
features.examples.npz
features.manifest.json
model/evaluation.json
```

The artifact directory is configurable in the dashboard.

By default, the real-data view displays the grouped evaluation. Official-split evaluation artifacts are stored separately.

**Note:** The GitHub repository does not include the large generated datasets or serialized trained models. These must be created locally using the documented pipeline.

---

## 11. Automated Testing

The project includes automated tests covering:

- Synthetic data generation
- Data validation
- Device-level feature engineering
- Anomaly detection
- Engineering analytics
- Wafer-map ingestion
- Spatial feature extraction
- Real-data classification pipeline
- Dashboard artifact handling
- Missing-data and edge-case behavior

Run the full test suite:

```bash
python -m pytest tests/ -q
```

Verify dependency consistency:

```bash
python -m pip check
```

**Latest reported validation:** 152 tests passed after the dashboard improvements.

This count is based on the latest development validation and should be reconfirmed when reproducing the project in a fresh environment.

The full WM-811K extraction and both real-data evaluations were executed locally. Ordinary automated tests use fixtures rather than loading the external dataset.

---

## 12. Engineering Limitations

### Synthetic workflow

- Electrical measurements are simulated rather than collected from production semiconductor test equipment.
- Simulated failure mechanisms may not capture real silicon behavior.
- Anomaly detection results are specific to the experimental dataset.
- Feature deviations are descriptive rather than causal.
- Yield uses a simplified passing criterion.
- Small wafer or lot samples may produce unstable estimates.
- The synthetic feature-building script currently loads input Parquet chunks into memory.

### Real WM-811K workflow

- Wafer maps contain die pass/fail patterns rather than underlying electrical measurements.
- Defect-pattern labels are not verified physical root-cause diagnoses.
- The dataset is strongly imbalanced.
- Rare classes have limited evaluation support.
- Scratch recall remains poor.
- Hand-engineered spatial features lose some detailed image structure.
- Features can depend on wafer-map resolution.
- A single grouped split does not quantify model uncertainty.
- Performance may not generalize across fabrication processes.
- The source pickle requires substantial memory to load.
- The classifier is an experimental baseline, not a production-qualified semiconductor inspection system.

These limitations are intentionally documented to distinguish experimental machine learning results from validated manufacturing performance.

---

## 13. Project Documentation

Additional technical information is available in:

| Document | Description |
|---|---|
| [Engineering Report](docs/ENGINEERING_REPORT.md) | Experimental design, evaluation, findings, and limitations |
| [Feature Definitions](docs/wafer_features.md) | Spatial and statistical wafer-map features |
| [Grouped Evaluation JSON](reports/wm811k_grouped_evaluation.json) | Detailed lot-grouped evaluation metrics |
| [Official Evaluation JSON](reports/wm811k_official_evaluation.json) | Detailed official metadata evaluation metrics |

---

## 14. Project Status

### Implemented

- [x] Synthetic semiconductor electrical-test generation
- [x] Device-level data processing and feature engineering
- [x] Isolation Forest anomaly detection
- [x] Wafer and lot yield analytics
- [x] Synthetic anomaly detection evaluation
- [x] WM-811K dataset ingestion
- [x] Real wafer-map spatial feature engineering
- [x] Full labeled wafer-map processing
- [x] Random Forest defect-pattern classification
- [x] Lot-disjoint evaluation
- [x] Official metadata evaluation
- [x] Majority-class baseline comparison
- [x] Interactive Streamlit dashboard
- [x] Automated tests
- [x] Engineering report and evaluation artifacts

### Potential future enhancements

- Image-based wafer-map classification using convolutional neural networks
- Grouped cross-validation and uncertainty estimation
- Improved minority-class classification
- Additional defect-pattern clustering approaches
- More memory-efficient ingestion formats
- Deployment and continuous integration improvements
- Validation using additional semiconductor datasets

These are possible extensions, not requirements for the current portfolio implementation.

---

## 15. Relevance to Semiconductor Data Science

The project demonstrates several transferable engineering and data science skills:

**Data engineering:** Processing structured measurements and large wafer-map datasets, building reproducible features, and managing analytical artifacts.

**Machine learning:** Applying unsupervised anomaly detection and supervised classification to semiconductor-related data.

**Statistical evaluation:** Investigating class imbalance, selecting appropriate metrics, comparing baselines, and evaluating leakage risks.

**Engineering analytics:** Interpreting wafer patterns, yield distributions, unusual measurements, and model limitations.

**Software engineering:** Organizing modular Python pipelines, writing automated tests, documenting experiments, and using Git for version control.

STIP is intended to demonstrate practical data science methodology in a semiconductor engineering context rather than replicate proprietary industrial test infrastructure.

---

## 16. Responsible Use and Disclaimer

This repository is an independent educational and portfolio project.

It is not affiliated with, endorsed by, or developed using proprietary data from Cadence Design Systems or any semiconductor manufacturer.

The synthetic electrical-test data is generated for experimentation.

The real wafer-map workflow uses a separately obtained public research dataset, subject to its applicable access and reuse terms.

The models and analytics are intended for research, learning, and portfolio demonstration, not production manufacturing decisions.

---

## 17. Author

**Bharath Sykam**

MSc Data Science and Artificial Intelligence

GitHub: [bharathsykam57-wq](https://github.com/bharathsykam57-wq)

Project repository: [Semiconductor Test Intelligence Platform](https://github.com/bharathsykam57-wq/semiconductor-test-intelligence)
