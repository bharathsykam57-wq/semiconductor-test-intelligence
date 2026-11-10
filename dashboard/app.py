
from pathlib import Path
import sys

import polars as pl
import streamlit as st
import pandas as pd

# ==================================================
# Project configuration
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anomaly_detection.isolation_forest import (
    detect_device_anomalies,
)

from src.explainability.feature_deviation import (
    calculate_feature_deviations,
    interpret_feature_deviation,
)

from src.wafer_analytics.wafer_summary import (
    calculate_wafer_summary,
    assess_wafer_sample_sizes,
)

st.set_page_config(
    page_title="Semiconductor Test Intelligence",
    page_icon="🔬",
    layout="wide",
)


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "device_features"
    / "device_features.parquet"
)


# ==================================================
# Data loading
# ==================================================

@st.cache_data
def load_device_features():
    return pl.read_parquet(DATA_PATH)


@st.cache_data
def load_anomaly_results(features: pl.DataFrame):
    return detect_device_anomalies(
        features,
        contamination=0.10,
        random_state=42,
    )


# ==================================================
# Dashboard header
# ==================================================

st.title("🔬 Semiconductor Test Intelligence Platform")

st.caption(
    "AI-powered semiconductor test analytics | "
    "Synthetic demonstration data"
)

st.divider()


# ==================================================
# Load and validate data
# ==================================================

if not DATA_PATH.is_file():
    st.error(
        "Device feature dataset not found. "
        "Generate the processed device features first."
    )
    st.stop()

try:
    df = load_device_features()
except Exception as exc:
    st.error(f"Unable to load device features: {exc}")
    st.stop()

if df.is_empty():
    st.warning("The device feature dataset is empty.")
    st.stop()


# ==================================================
# Sidebar manufacturing filters
# ==================================================

st.sidebar.title("Manufacturing Filters")

lot_options = ["All Lots"] + sorted(
    df["lot_id"].unique().to_list()
)

selected_lot = st.sidebar.selectbox(
    "Select Manufacturing Lot",
    options=lot_options,
)

lot_filtered_df = df

if selected_lot != "All Lots":
    lot_filtered_df = lot_filtered_df.filter(
        pl.col("lot_id") == selected_lot
    )


wafer_options = ["All Wafers"] + sorted(
    lot_filtered_df["wafer_id"].unique().to_list()
)

selected_wafer = st.sidebar.selectbox(
    "Select Wafer",
    options=wafer_options,
)


filtered_df = lot_filtered_df

if selected_wafer != "All Wafers":
    filtered_df = filtered_df.filter(
        pl.col("wafer_id") == selected_wafer
    )


st.sidebar.divider()

st.sidebar.metric(
    "Visible Devices",
    filtered_df.height,
)

st.sidebar.caption(
    f"Showing {filtered_df.height:,} "
    f"of {df.height:,} devices"
)

st.sidebar.info(
    "All measurements are synthetic and intended "
    "for demonstration and portfolio purposes."
)


# ==================================================
# Manufacturing overview
# ==================================================

st.header("Manufacturing Test Overview")

total_devices = filtered_df.height
total_wafers = filtered_df["wafer_id"].n_unique()
total_lots = filtered_df["lot_id"].n_unique()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Devices",
    f"{total_devices:,}",
)

col2.metric(
    "Total Wafers",
    total_wafers,
)

col3.metric(
    "Total Lots",
    total_lots,
)


# ==================================================
# Device feature table
# ==================================================

st.subheader("Device-Level Test Features")

st.dataframe(
    filtered_df.to_pandas(),
    use_container_width=True,
    hide_index=True,
)


# ==================================================
# Electrical test analysis
# ==================================================

st.divider()

st.header("Electrical Test Analysis")

st.caption(
    "Explore electrical characteristics of the "
    "selected synthetic semiconductor devices."
)

measurement_options = {
    "Mean Leakage Current (µA)":
        "mean_leakage_current_ua",

    "Maximum Propagation Delay (ns)":
        "max_propagation_delay_ns",

    "Mean Power Consumption (mW)":
        "mean_power_mw",
}


selected_measurement = st.selectbox(
    "Select Electrical Measurement",
    options=list(measurement_options.keys()),
)

measurement_column = measurement_options[
    selected_measurement
]


chart_df = filtered_df.select([
    "device_id",
    measurement_column,
]).to_pandas()

chart_df = chart_df.sort_values(
    measurement_column,
    ascending=False,
)


st.subheader(selected_measurement)

st.bar_chart(
    chart_df.set_index("device_id")[
        [measurement_column]
    ],
    x_label="Device",
    y_label=selected_measurement,
)


# ==================================================
# Measurement statistics
# ==================================================

st.subheader("Measurement Statistics")

stats = filtered_df.select([
    pl.col(measurement_column)
    .mean()
    .alias("Mean"),

    pl.col(measurement_column)
    .median()
    .alias("Median"),

    pl.col(measurement_column)
    .min()
    .alias("Minimum"),

    pl.col(measurement_column)
    .max()
    .alias("Maximum"),
])

st.dataframe(
    stats.to_pandas(),
    hide_index=True,
    use_container_width=True,
)


# ==================================================
# AI-powered anomaly detection
# ==================================================

st.divider()

st.header("AI-Powered Device Anomaly Detection")

st.caption(
    "Isolation Forest identifies devices with unusual "
    "electrical characteristics. Synthetic data only."
)


# Train on all devices so scores remain consistent
# when the user changes manufacturing filters.

try:
    all_anomaly_results = load_anomaly_results(df)
except Exception as exc:
    st.error(f"Anomaly detection failed: {exc}")
    st.stop()


# Apply manufacturing filters to model results

anomaly_results = all_anomaly_results

if selected_lot != "All Lots":
    anomaly_results = anomaly_results.filter(
        pl.col("lot_id") == selected_lot
    )

if selected_wafer != "All Wafers":
    anomaly_results = anomaly_results.filter(
        pl.col("wafer_id") == selected_wafer
    )


# ==================================================
# Anomaly detection metrics
# ==================================================

total_analyzed = anomaly_results.height

anomaly_count = anomaly_results.filter(
    pl.col("is_anomaly")
).height

anomaly_percentage = (
    100 * anomaly_count / total_analyzed
    if total_analyzed > 0
    else 0.0
)


col1, col2, col3 = st.columns(3)

col1.metric(
    "Devices Analyzed",
    f"{total_analyzed:,}",
)

col2.metric(
    "Flagged Anomalies",
    anomaly_count,
)

col3.metric(
    "Anomaly Rate",
    f"{anomaly_percentage:.2f}%",
)


# ==================================================
# Ranked anomalous devices
# ==================================================

st.subheader("Most Unusual Devices")

ranked_devices = (
    anomaly_results
    .sort("anomaly_score", descending=True)
    .select([
        "device_id",
        "wafer_id",
        "anomaly_score",
        "is_anomaly",
        "failure_rate",
    ])
)

st.dataframe(
    ranked_devices.to_pandas(),
    use_container_width=True,
    hide_index=True,
)


st.caption(
    "Higher anomaly scores indicate more unusual "
    "devices. A flagged anomaly does not necessarily "
    "represent a confirmed physical defect."
)

st.info(
    "The model is fitted on the complete synthetic "
    "dataset. Display filters do not retrain it. "
    "These are exploratory in-sample anomaly scores, "
    "not held-out evaluation predictions."
)


# ==================================================
# Interactive anomaly score visualization
# ==================================================

st.divider()

st.header("Device Anomaly Score Visualization")

st.caption(
    "Explore Isolation Forest anomaly scores for "
    "the selected synthetic semiconductor devices."
)

# Allow users to show only flagged devices
show_only_anomalies = st.checkbox(
    "Show only flagged anomalies",
    value=False,
)

plot_df = anomaly_results

if show_only_anomalies:
    plot_df = plot_df.filter(
        pl.col("is_anomaly")
    )

# Sort devices by anomaly score
plot_df = plot_df.sort(
    "anomaly_score",
    descending=False,
)

if plot_df.is_empty():
    st.warning(
        "No devices match the selected filters."
    )
else:
    chart_data = plot_df.select([
        "device_id",
        "anomaly_score",
        "is_anomaly",
    ]).to_pandas()

    # Rank devices by their anomaly score
    chart_data["device_rank"] = range(
        1,
        len(chart_data) + 1,
    )

    # Display separate series for flagged and
    # unflagged devices
    chart_data["Flagged anomaly"] = (
        chart_data["anomaly_score"].where(
            chart_data["is_anomaly"]
        )
    )

    chart_data["Not flagged"] = (
        chart_data["anomaly_score"].where(
            ~chart_data["is_anomaly"]
        )
    )

    st.scatter_chart(
        chart_data.set_index("device_rank")[
            ["Not flagged", "Flagged anomaly"]
        ],
        x_label="Devices ranked by anomaly score",
        y_label="Isolation Forest anomaly score",
    )

    st.caption(
        "Scores above zero are flagged by the model. "
        "Higher scores indicate more unusual devices."
    )

    # Display highest-scoring devices
    st.subheader("Top 10 Most Unusual Devices")

    top_devices = (
        plot_df
        .sort("anomaly_score", descending=True)
        .head(10)
        .select([
            "device_id",
            "wafer_id",
            "anomaly_score",
            "is_anomaly",
            "failure_rate",
        ])
    )

    st.dataframe(
        top_devices.to_pandas(),
        use_container_width=True,
        hide_index=True,
    )



# ==================================================
# Individual device investigation
# ==================================================

st.divider()

st.header("Individual Device Investigation")

st.caption(
    "Inspect electrical characteristics and anomaly "
    "detection results for a selected synthetic device."
)

# Use the manufacturing filters already selected
device_options = sorted(
    anomaly_results["device_id"].to_list()
)

if not device_options:
    st.warning("No devices available for investigation.")

else:
    selected_device = st.selectbox(
        "Select Device",
        options=device_options,
    )

    device_data = anomaly_results.filter(
        pl.col("device_id") == selected_device
    )

    device = device_data.row(0, named=True)

    # Device identification
    st.subheader("Device Information")

    st.write("**Device ID:**", device["device_id"])
    st.write("**Wafer ID:**", device["wafer_id"])
    st.write("**Lot ID:**", device["lot_id"])

    # Anomaly detection results
    st.subheader("AI Anomaly Assessment")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Anomaly Score",
        f"{device['anomaly_score']:.4f}",
    )

    col2.metric(
        "Recorded Failure Rate",
        f"{device['failure_rate']:.2%}",
    )

    col3.metric(
        "Model Flag",
        "Anomaly" if device["is_anomaly"] else "Not Flagged",
    )

    # Electrical measurements
    st.subheader("Electrical Characteristics")

    electrical_features = {
        "Mean Leakage Current (µA)":
            "mean_leakage_current_ua",
        "Maximum Leakage Current (µA)":
            "max_leakage_current_ua",
        "Maximum Propagation Delay (ns)":
            "max_propagation_delay_ns",
        "Mean Power Consumption (mW)":
            "mean_power_mw",
    }

    feature_rows = [
        {
            "Measurement": label,
            "Value": device[column],
        }
        for label, column in electrical_features.items()
    ]

    st.dataframe(
        feature_rows,
        use_container_width=True,
        hide_index=True,
    )
    
    
    
    # ==============================================
    # Feature deviation analysis
    # ==============================================

    st.subheader("Electrical Feature Deviation Analysis")

    st.caption(
        "Compare this device's electrical measurements "
        "with the full synthetic device dataset."
    )

    deviation_results = calculate_feature_deviations(
        df,
        selected_device,
    )
    
    
    # Convert the Polars results into a Pandas DataFrame
    deviation_display = deviation_results.to_pandas()

    # Add a human-readable engineering interpretation
    deviation_display["interpretation"] = (
        deviation_display["z_score"].apply(
            interpret_feature_deviation
        )
    )


    # Show the most unusual measurements first
    st.dataframe(
        deviation_display,
        use_container_width=True,
        hide_index=True,
    )

    # Visualize absolute standardized deviations
    deviation_chart = (
        deviation_results
        .select(["feature", "absolute_z_score"])
        .to_pandas()
        .set_index("feature")
    )

    st.bar_chart(
        deviation_chart,
        x_label="Electrical Feature",
        y_label="Absolute Z-Score",
    )

    # Identify the feature with the largest deviation
    most_deviant = deviation_results.row(
        0,
        named=True,
    )

    st.info(
        f"Largest standardized deviation: "
        f"{most_deviant['feature']} "
        f"(Z-score: {most_deviant['z_score']:.2f}). "
        "This describes how unusual the measurement is "
        "relative to the dataset; it does not establish "
        "why Isolation Forest flagged the device."
    )


    # Engineering interpretation
    st.subheader("Engineering Interpretation")

    if device["is_anomaly"]:
        st.warning(
            "This device has unusual electrical characteristics "
            "relative to the model's synthetic-data baseline. "
            "Further investigation may be appropriate."
        )
    else:
        st.success(
            "This device was not flagged as unusual "
            "by the current Isolation Forest model."
        )

    st.caption(
        "Anomaly detection is not a definitive physical "
        "defect diagnosis. Recorded failure labels and "
        "model anomaly flags are separate indicators."
    )




# ==================================================
# Anomalous vs typical device comparison
# ==================================================

st.divider()

st.header("Anomalous vs Typical Device Comparison")

st.caption(
    "Compare average electrical characteristics of "
    "flagged and unflagged devices within the selected "
    "manufacturing filters. Synthetic data only."
)

comparison_features = {
    "Mean Leakage Current (µA)": "mean_leakage_current_ua",
    "Maximum Propagation Delay (ns)": "max_propagation_delay_ns",
    "Mean Power Consumption (mW)": "mean_power_mw",
}

comparison_measurement = st.selectbox(
    "Select Measurement for Comparison",
    options=list(comparison_features.keys()),
    key="comparison_measurement",
)

comparison_column = comparison_features[
    comparison_measurement
]

# Calculate group averages for the selected devices
comparison_df = (
    anomaly_results
    .with_columns(
        pl.when(pl.col("is_anomaly"))
        .then(pl.lit("Flagged Anomaly"))
        .otherwise(pl.lit("Not Flagged"))
        .alias("Device Group")
    )
    .group_by("Device Group")
    .agg(
        pl.col(comparison_column)
        .mean()
        .alias("Average Value"),
        pl.len().alias("Device Count"),
    )
    .sort("Device Group")
)

if comparison_df.is_empty():
    st.warning("No devices available for comparison.")

else:
    st.subheader("Average Electrical Measurement by Group")

    st.bar_chart(
        comparison_df.to_pandas().set_index(
            "Device Group"
        )[["Average Value"]],
        y_label=comparison_measurement,
    )

    st.subheader("Comparison Statistics")

    st.dataframe(
        comparison_df.to_pandas(),
        use_container_width=True,
        hide_index=True,
    )

    group_names = comparison_df[
        "Device Group"
    ].to_list()

    if len(group_names) < 2:
        st.info(
            "Only one device group is present in the "
            "current selection. Select more devices "
            "to compare flagged and unflagged groups."
        )

    st.caption(
        "These are descriptive group averages, not "
        "causal explanations of why the model flagged "
        "a device. The same electrical features were "
        "used to fit the anomaly detection model."
    )




# ==================================================
# Wafer-level anomaly analysis
# ==================================================

st.divider()

st.header("Wafer-Level Anomaly Analysis")

st.caption(
    "Compare anomaly counts and anomaly rates across "
    "manufacturing wafers. Synthetic data only."
)

# Aggregate anomaly results by manufacturing wafer
wafer_summary = (
    anomaly_results
    .group_by(["lot_id", "wafer_id"])
    .agg([
        pl.len().alias("Total Devices"),
        pl.col("is_anomaly").sum().alias("Flagged Anomalies"),
        pl.col("failure_rate")
        .mean()
        .alias("Average Recorded Failure Rate"),
    ])
    .with_columns(
        (
            pl.col("Flagged Anomalies")
            / pl.col("Total Devices")
            * 100
        ).alias("Anomaly Rate (%)")
    )
    .sort("Anomaly Rate (%)", descending=True)
)

if wafer_summary.is_empty():
    st.warning("No wafers available for analysis.")

else:
    st.subheader("Anomaly Rate by Wafer")

    st.bar_chart(
        wafer_summary.to_pandas().set_index(
            "wafer_id"
        )[["Anomaly Rate (%)"]],
        y_label="Flagged Devices (%)",
    )

    st.subheader("Wafer Anomaly Summary")

    st.dataframe(
        wafer_summary.to_pandas(),
        use_container_width=True,
        hide_index=True,
    )

    # Highlight the wafer with the highest observed rate
    highest_wafer = wafer_summary.row(0, named=True)

    st.metric(
        "Highest Observed Wafer Anomaly Rate",
        f"{highest_wafer['Anomaly Rate (%)']:.2f}%",
    )

    st.caption(
        f"Highest observed rate: {highest_wafer['wafer_id']}. "
        "Rates are descriptive and depend on the number "
        "of devices in each wafer. Small wafers may have "
        "unstable rates."
    )

    st.info(
        "These anomaly rates use predictions from the "
        "Isolation Forest fitted on all 501 synthetic "
        "devices. They are not the held-out wafer "
        "evaluation results."
    )




# ==================================================
# Held-out wafer model evaluation
# ==================================================

import json

st.divider()

st.header("Held-Out Wafer Model Evaluation")

st.caption(
    "Leave-one-wafer-out Isolation Forest evaluation "
    "using synthetic semiconductor test data."
)

EVALUATION_PATH = (
    PROJECT_ROOT
    / "reports"
    / "anomaly_evaluation.json"
)

if not EVALUATION_PATH.is_file():
    st.warning(
        "Evaluation report not found. "
        "Generate reports/anomaly_evaluation.json first."
    )

else:
    @st.cache_data
    def load_evaluation_report(path):
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    evaluation = load_evaluation_report(
        str(EVALUATION_PATH)
    )

    st.subheader("Evaluation Report")

    st.json(evaluation)
    
    
    # ==============================================
    # Model evaluation KPI cards
    # ==============================================

    st.subheader("Model Performance Metrics")

    # Read the overall metrics from the saved report.
    # Support either a direct metrics dictionary or
    # an overall_metrics dictionary.
    metrics = evaluation["evaluation"]["overall_metrics"]

    required_metrics = [
        "precision",
        "recall",
        "f1_score",
    ]

    if all(key in metrics for key in required_metrics):
        precision = float(metrics["precision"])
        recall = float(metrics["recall"])
        f1 = float(metrics["f1_score"])

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Precision",
            f"{precision:.2%}",
        )

        col2.metric(
            "Recall",
            f"{recall:.2%}",
        )

        col3.metric(
            "F1-Score",
            f"{f1:.2%}",
        )

        st.subheader("Confusion Matrix Counts")

        count_keys = [
            "true_positives",
            "true_negatives",
            "false_positives",
            "false_negatives",
        ]

        if all(key in metrics for key in count_keys):
            tp = int(metrics["true_positives"])
            tn = int(metrics["true_negatives"])
            fp = int(metrics["false_positives"])
            fn = int(metrics["false_negatives"])

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("True Positives", tp)
            c2.metric("True Negatives", tn)
            c3.metric("False Positives", fp)
            c4.metric("False Negatives", fn)

        else:
            st.info(
                "Confusion matrix counts were not found "
                "under the expected keys in the JSON report."
            )

    else:
        st.warning(
            "Evaluation metrics were not found under "
            "the expected keys. Inspect the JSON report "
            "to confirm its structure."
        )


    st.info(
        "These results come from leave-one-wafer-out "
        "evaluation on synthetic data. Each held-out "
        "wafer was evaluated using a model trained "
        "on other wafers. The dashboard's device anomaly "
        "scores use a separate model fitted on all "
        "501 synthetic devices."
    )




# ==================================================
# Held-out evaluation visualizations
# ==================================================

st.divider()

st.header("Model Evaluation Visualizations")

st.caption(
    "Held-out wafer evaluation results from synthetic "
    "semiconductor test data."
)

if EVALUATION_PATH.is_file():

    overall = evaluation["evaluation"]["overall_metrics"]
    wafer_results = evaluation["evaluation"]["per_wafer_results"]

    # ----------------------------------------------
    # Confusion matrix
    # ----------------------------------------------

    st.subheader("Confusion Matrix")

    tp = overall["true_positives"]
    tn = overall["true_negatives"]
    fp = overall["false_positives"]
    fn = overall["false_negatives"]

    confusion_data = {
        "Actual: No Recorded Failure": {
            "Predicted: No Anomaly": tn,
            "Predicted: Anomaly": fp,
        },
        "Actual: Recorded Failure": {
            "Predicted: No Anomaly": fn,
            "Predicted: Anomaly": tp,
        },
    }

    confusion_df = pd.DataFrame.from_dict(
        confusion_data,
        orient="index",
    )

    st.dataframe(
        confusion_df,
        use_container_width=True,
    )

    st.caption(
        "Rows represent recorded failure labels; "
        "columns represent model anomaly predictions."
    )

    # ----------------------------------------------
    # Wafer-level model performance
    # ----------------------------------------------

    st.subheader("Performance by Held-Out Wafer")

    wafer_df = pd.DataFrame(wafer_results)

    wafer_chart = wafer_df.set_index("wafer_id")[
        ["precision", "recall", "f1_score"]
    ]

    st.bar_chart(
        wafer_chart,
        x_label="Held-Out Wafer",
        y_label="Metric Value (0–1)",
        stack=False,
    )

    st.subheader("Per-Wafer Evaluation Results")

    st.dataframe(
        wafer_df,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "These metrics are from five held-out wafers "
        "within one synthetic manufacturing lot. "
        "They should not be interpreted as validated "
        "performance on unseen manufacturing lots."
    )



# ==============================================
# Wafer-Level Engineering Summary
# ==============================================

st.divider()
st.header("Wafer-Level Engineering Summary")

st.caption(
    "Compare electrical characteristics across wafers "
    "using synthetic semiconductor test data."
)

wafer_summary = calculate_wafer_summary(df)

st.subheader("Wafer Engineering Metrics")

st.dataframe(
    wafer_summary.to_pandas(),
    use_container_width=True,
    hide_index=True,
)

# Compare electrical characteristics across wafers
st.subheader("Electrical Characteristics by Wafer")

metric_options = {
    "Average Leakage Current (µA)": "avg_leakage_current_ua",
    "Average Maximum Propagation Delay (ns)": (
        "avg_max_propagation_delay_ns"
    ),
    "Average Power Consumption (mW)": "avg_power_mw",
    "Average Device Failure Rate": "avg_device_failure_rate",
}

selected_metric = st.selectbox(
    "Select an engineering metric",
    options=list(metric_options.keys()),
    key="wafer_engineering_metric",
)

metric_column = metric_options[selected_metric]

wafer_chart_data = (
    wafer_summary
    .select(["wafer_id", metric_column])
    .to_pandas()
    .set_index("wafer_id")
)

st.bar_chart(
    wafer_chart_data,
    x_label="Wafer",
    y_label=selected_metric,
)


# ==============================================
# Wafer Engineering Ranking
# ==============================================

# ==============================================
# Wafer Sample Size Validation
# ==============================================

st.subheader("Wafer Sample Size Assessment")


MIN_DEVICES_PER_WAFER = 30

wafer_sample_assessment = assess_wafer_sample_sizes(
    wafer_summary,
    min_devices=MIN_DEVICES_PER_WAFER,
)


st.dataframe(
    wafer_sample_assessment.select([
        "lot_id",
        "wafer_id",
        "device_count",
        "sample_size_status",
    ]).to_pandas(),
    use_container_width=True,
    hide_index=True,
)

limited_wafers = wafer_sample_assessment.filter(
    pl.col("device_count") < MIN_DEVICES_PER_WAFER
)

if limited_wafers.height > 0:
    st.warning(
        f"{limited_wafers.height} wafer(s) contain fewer "
        f"than {MIN_DEVICES_PER_WAFER} devices. "
        "Interpret their aggregate statistics cautiously."
    )
else:
    st.success(
        "All wafers meet the configured sample-size guideline."
    )

st.caption(
    "The 30-device threshold is an illustrative dashboard "
    "guideline, not a statistically justified minimum or "
    "a validated semiconductor manufacturing standard."
)


st.subheader("Wafer Engineering Ranking")

ranked_wafers = (
    wafer_summary
    .sort(metric_column, descending=True)
    .select([
        "lot_id",
        "wafer_id",
        "device_count",
        metric_column,
    ])
)

st.dataframe(
    ranked_wafers.to_pandas(),
    use_container_width=True,
    hide_index=True,
)


# ==============================================
# Wafer vs Overall Dataset Average
# ==============================================

st.subheader("Wafer vs Overall Dataset Average")

# Calculate the device-weighted overall average.
# This avoids giving a one-device wafer the same
# influence as a wafer containing 100 devices.

overall_average = (
    df.select(
        pl.col({
            "avg_leakage_current_ua": "mean_leakage_current_ua",
            "avg_max_propagation_delay_ns": "max_propagation_delay_ns",
            "avg_power_mw": "mean_power_mw",
            "avg_device_failure_rate": "failure_rate",
        }[metric_column]).mean()
    ).item()
)

comparison = wafer_summary.with_columns(
    (pl.col(metric_column) - overall_average)
    .alias("difference_from_overall")
)

if overall_average != 0:
    comparison = comparison.with_columns(
        (
            pl.col("difference_from_overall")
            / abs(overall_average)
            * 100
        ).alias("percent_difference")
    )
else:
    comparison = comparison.with_columns(
        pl.lit(None, dtype=pl.Float64)
        .alias("percent_difference")
    )

st.metric(
    "Overall Device Average",
    f"{overall_average:.4f}",
)

st.dataframe(
    comparison.select([
        "wafer_id",
        "device_count",
        metric_column,
        "difference_from_overall",
        "percent_difference",
    ]).to_pandas(),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Positive differences indicate values above the "
    "device-weighted dataset average. These are "
    "descriptive comparisons, not manufacturing "
    "pass/fail thresholds."
)


# ==============================================
# Wafer Deviation Visualization
# ==============================================

st.subheader("Wafer Deviation from Overall Average")

deviation_chart_data = (
    comparison
    .select([
        "wafer_id",
        "percent_difference",
    ])
    .to_pandas()
    .set_index("wafer_id")
)

st.bar_chart(
    deviation_chart_data,
    x_label="Wafer",
    y_label="Difference from Overall Average (%)",
)

st.caption(
    "Positive values indicate above-average measurements; "
    "negative values indicate below-average measurements. "
    "The baseline includes all devices in the dataset."
)


highest_wafer = ranked_wafers.row(0, named=True)

st.info(
    f"Highest {selected_metric}: "
    f"{highest_wafer['wafer_id']} "
    f"({highest_wafer[metric_column]:.4f}). "
    f"This wafer contains "
    f"{highest_wafer['device_count']} devices."
)


st.info(
    "These comparisons are descriptive and use synthetic data. "
    "Wafers with different device counts may have different "
    "levels of statistical reliability."
)


# ==================================================
# Dashboard footer
# ==================================================

st.divider()

st.caption(
    "STIP | Semiconductor Test Intelligence Platform "
    "| Portfolio Demonstration | Synthetic Data"
)
