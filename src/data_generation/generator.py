"""
Semiconductor Test Intelligence Platform.

Main synthetic semiconductor test data generation engine.

All generated measurements are synthetic and are not
calibrated against actual semiconductor foundry data.
"""

from pathlib import Path
import argparse
import math

import numpy as np
import polars as pl
import yaml

from .device_characteristics import get_default_device_profile

from .manufacturing import (
    ManufacturingVariationConfig,
    sample_lot_variation,
    sample_wafer_variation,
    sample_device_variation,
)

from .operating_conditions import (
    OperatingConditionConfig,
    generate_operating_conditions,
)

from .failure_simulation import (
    FailureCategory,
    FailureSimulationConfig,
)


def load_config(config_path: str = "configs/data_generation.yaml"):
    """Load and validate the YAML configuration."""

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    return config


def validate_config(config: dict):
    """Validate dataset dimensions and failure probabilities."""

    dataset = config["dataset"]

    dimensions = [
        dataset["total_lots"],
        dataset["wafers_per_lot"],
        dataset["devices_per_wafer"],
        dataset["measurements_per_device"],
    ]

    if any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
        for value in dimensions
    ):
        raise ValueError("Dataset dimensions must be positive integers.")

    expected = math.prod(dimensions)

    if expected != dataset["expected_rows"]:
        raise ValueError(
            f"Expected {dataset['expected_rows']} rows, "
            f"but dimensions produce {expected}."
        )

    chunk_size = config["output"]["chunk_size"]

    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size <= 0
    ):
        raise ValueError("Chunk size must be a positive integer.")

    if chunk_size < dataset["measurements_per_device"]:
        raise ValueError("Chunk size must fit at least one device.")

    if chunk_size % dataset["measurements_per_device"] != 0:
        raise ValueError(
            "Chunk size must be divisible by measurements_per_device."
        )

    probabilities = config["failure_simulation"]["category_probabilities"]

    if set(probabilities) != {category.value for category in FailureCategory}:
        raise ValueError("Failure probabilities must cover all eight categories.")

    probability_values = list(probabilities.values())

    if any(
        not math.isfinite(value) or value < 0 or value > 1
        for value in probability_values
    ):
        raise ValueError("Failure probabilities must be between 0 and 1.")

    if not math.isclose(sum(probability_values), 1.0, abs_tol=1e-9):
        raise ValueError("Failure probabilities must sum to 1.")

    supported_formats = {"parquet", "csv"}

    formats = set(config["output"]["formats"])

    if not formats or not formats.issubset(supported_formats):
        raise ValueError("Supported formats are parquet and csv.")


def _generate_device_records(
    rng,
    profile,
    manufacturing_effects,
    operating_config,
    failure_config,
    failure_category,
    measurements_per_device,
    device_id,
    wafer_id,
    lot_id,
    first_record_id,
):
    """Generate all measurements for one device."""

    failure_category = FailureCategory(str(failure_category))

    conditions = generate_operating_conditions(
        rng,
        operating_config,
        measurements_per_device,
    )

    voltage = np.array([x.voltage_v for x in conditions])
    frequency = np.array([x.frequency_mhz for x in conditions])
    temperature = np.array([x.temperature_c for x in conditions])

    nominal_voltage = profile.nominal_voltage_v
    nominal_frequency = profile.nominal_frequency_mhz
    nominal_temperature = profile.nominal_temperature_c

    # Hierarchical manufacturing effects.

    process_shift = manufacturing_effects.total_shift

    process_factor = np.exp(process_shift)

    # Baseline electrical behaviour.

    temperature_current_factor = np.maximum(
        0.1,
        1.0 + 0.001 * (temperature - nominal_temperature),
    )

    current = (
        profile.nominal_current_ma
        * (voltage / nominal_voltage)
        * (frequency / nominal_frequency)
        * temperature_current_factor
        * process_factor
    )

    leakage = (
        profile.nominal_leakage_current_ua
        * np.exp(0.025 * (temperature - nominal_temperature))
        * process_factor
    )

    delay = (
        profile.nominal_propagation_delay_ns
        * (nominal_voltage / voltage) ** 1.3
        * (1.0 + 0.0015 * (temperature - nominal_temperature))
        * np.exp(-process_shift)
    )

    current_multiplier = np.ones(measurements_per_device)
    leakage_multiplier = np.ones(measurements_per_device)
    delay_multiplier = np.ones(measurements_per_device)

    performance_delta = np.zeros(measurements_per_device)

    failed = np.zeros(measurements_per_device, dtype=bool)

    # Apply persistent failure mechanisms.

    if failure_category == FailureCategory.HIGH_LEAKAGE:

        leakage_multiplier *= failure_config.high_leakage_multiplier
        performance_delta -= 0.05

        failed[:] = True

    elif failure_category == FailureCategory.TIMING_FAILURE:

        delay_multiplier *= failure_config.timing_delay_multiplier
        performance_delta -= 0.20

        failed[:] = True

    elif failure_category == FailureCategory.VOLTAGE_SENSITIVITY:

        deviation = np.abs(voltage - nominal_voltage) / nominal_voltage

        multiplier = (
            1.0
            + failure_config.voltage_sensitivity_factor * deviation
        )

        current_multiplier *= multiplier
        delay_multiplier *= multiplier

        performance_delta -= np.minimum(0.30, deviation)

        failed[:] = True

    elif failure_category == FailureCategory.THERMAL_INSTABILITY:

        deviation = np.abs(
            temperature - nominal_temperature
        ) / 100.0

        multiplier = (
            1.0
            + failure_config.thermal_instability_factor * deviation
        )

        leakage_multiplier *= multiplier
        delay_multiplier *= multiplier

        performance_delta -= np.minimum(0.30, deviation)

        failed[:] = True

    elif failure_category == FailureCategory.POWER_ANOMALY:

        current_multiplier *= failure_config.power_anomaly_multiplier
        performance_delta -= 0.10

        failed[:] = True

    elif failure_category == FailureCategory.FREQUENCY_INSTABILITY:

        deviation = (
            np.abs(frequency - nominal_frequency)
            / nominal_frequency
        )

        multiplier = (
            1.0
            + failure_config.frequency_instability_multiplier
            * deviation
        )

        delay_multiplier *= multiplier

        performance_delta -= np.minimum(0.30, deviation)

        failed[:] = True

    elif failure_category == FailureCategory.INTERMITTENT_FAILURE:

        failed = (
            rng.random(measurements_per_device)
            < failure_config.intermittent_failure_probability
        )

        current_multiplier[failed] *= 1.5
        leakage_multiplier[failed] *= 2.0
        delay_multiplier[failed] *= 1.5

        performance_delta[failed] -= 0.20

    # Apply failure effects.

    current *= current_multiplier
    leakage *= leakage_multiplier
    delay *= delay_multiplier

    # Measurement noise.

    noise = {
        "current": 0.02,
        "leakage": 0.05,
        "delay": 0.02,
        "performance": 0.01,
    }

    current += rng.normal(0, noise["current"], measurements_per_device)

    leakage += rng.normal(0, noise["leakage"], measurements_per_device)

    delay += rng.normal(0, noise["delay"], measurements_per_device)

    current = np.maximum(current, 0.001)
    leakage = np.maximum(leakage, 0.0)
    delay = np.maximum(delay, 0.001)

    # Power in mW = voltage in V * current in mA.

    power = voltage * current

    # Synthetic normalized performance indicator.

    delay_penalty = np.maximum(
        delay / profile.nominal_propagation_delay_ns - 1.0,
        0.0,
    )

    performance = (
        profile.nominal_performance_score
        - 0.15 * delay_penalty
        + performance_delta
        + rng.normal(0, noise["performance"], measurements_per_device)
    )

    performance = np.clip(performance, 0.0, 1.0)

    # Assign measurement-level labels.

    failure_labels = np.full(
        measurements_per_device,
        failure_category.value,
        dtype=object,
    )

    if failure_category == FailureCategory.INTERMITTENT_FAILURE:

        failure_labels[:] = FailureCategory.NORMAL.value

        failure_labels[failed] = (
            FailureCategory.INTERMITTENT_FAILURE.value
        )

    test_status = np.where(failed, "FAIL", "PASS")

    record_ids = np.arange(
        first_record_id,
        first_record_id + measurements_per_device,
    )

    return {
        "lot_id": np.full(measurements_per_device, lot_id),
        "wafer_id": np.full(measurements_per_device, wafer_id),
        "device_id": np.full(measurements_per_device, device_id),

        "test_id": [
            f"TEST_{i:010d}" for i in record_ids
        ],

        "test_condition_id": [
            f"TC_{i:010d}" for i in record_ids
        ],

        "voltage_v": voltage,
        "frequency_mhz": frequency,
        "temperature_c": temperature,

        "current_ma": current,
        "power_mw": power,
        "leakage_current_ua": leakage,
        "propagation_delay_ns": delay,
        "performance_score": performance,

        "test_status": test_status,
        "failure_category": failure_labels,

        "lot_process_shift": np.full(
            measurements_per_device,
            manufacturing_effects.lot_shift,
        ),

        "wafer_process_shift": np.full(
            measurements_per_device,
            manufacturing_effects.wafer_shift,
        ),

        "device_process_shift": np.full(
            measurements_per_device,
            manufacturing_effects.device_shift,
        ),

        "is_synthetic": np.ones(
            measurements_per_device,
            dtype=bool,
        ),
    }


def generate_dataset(
    config,
    output_dir=None,
    max_devices=None,
):
    """Generate and export the synthetic semiconductor dataset."""

    validate_config(config)

    rng = np.random.default_rng(
        config["dataset"]["random_seed"]
    )

    profile = get_default_device_profile()

    manufacturing_config = ManufacturingVariationConfig(
        **config["manufacturing"]
    )

    operating_config = OperatingConditionConfig(
        voltage_min_v=config["operating_conditions"]["voltage_v"]["min"],
        voltage_max_v=config["operating_conditions"]["voltage_v"]["max"],

        frequency_min_mhz=config["operating_conditions"]["frequency_mhz"]["min"],
        frequency_max_mhz=config["operating_conditions"]["frequency_mhz"]["max"],

        temperature_min_c=config["operating_conditions"]["temperature_c"]["min"],
        temperature_max_c=config["operating_conditions"]["temperature_c"]["max"],
    )

    failure_parameters = config["failure_simulation"]

    failure_config = FailureSimulationConfig(
        intermittent_failure_probability=failure_parameters.get(
            "intermittent_failure_probability",
            0.10,
        )
    )

    dataset_config = config["dataset"]

    measurements = dataset_config["measurements_per_device"]

    total_devices = (
        dataset_config["total_lots"]
        * dataset_config["wafers_per_lot"]
        * dataset_config["devices_per_wafer"]
    )

    if max_devices is not None:

        if (
            not isinstance(max_devices, int)
            or isinstance(max_devices, bool)
            or max_devices <= 0
        ):
            raise ValueError("max_devices must be a positive integer.")

        total_devices = min(total_devices, max_devices)

    expected_rows = total_devices * measurements

    chunk_devices = (
        config["output"]["chunk_size"] // measurements
    )

    probabilities = failure_parameters["category_probabilities"]

    categories = [
        FailureCategory(category)
        for category in probabilities
    ]

    probabilities_array = np.array(
        list(probabilities.values()),
        dtype=float,
    )

    if not failure_parameters.get("enabled", True):

        categories = [FailureCategory.NORMAL]
        probabilities_array = np.array([1.0])

    output_dir = Path(
        output_dir or config["output"]["directory"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    formats = config["output"]["formats"]

    record_counter = 0
    device_counter = 0
    chunk_counter = 0

    batch_parts = {}

    def flush_batch():

        nonlocal record_counter, chunk_counter, batch_parts

        if not batch_parts:
            return

        combined = {
            column: np.concatenate(parts)
            for column, parts in batch_parts.items()
        }

        dataframe = pl.DataFrame(combined)

        if dataframe.height > config["output"]["chunk_size"]:
            raise RuntimeError("Generated chunk exceeds configured size.")

        for file_format in formats:

            filename = (
                output_dir
                / f"part-{chunk_counter:05d}.{file_format}"
            )

            if file_format == "parquet":
                dataframe.write_parquet(filename)

            elif file_format == "csv":
                dataframe.write_csv(filename)

        record_counter += dataframe.height
        chunk_counter += 1

        print(
            f"Chunk {chunk_counter}: "
            f"{dataframe.height:,} records | "
            f"Total: {record_counter:,}"
        )

        batch_parts = {}

    # Traverse manufacturing hierarchy.

    for lot_number in range(dataset_config["total_lots"]):

        if device_counter >= total_devices:
            break

        lot_id = f"LOT_{lot_number + 1:03d}"

        lot_shift = sample_lot_variation(
            rng,
            manufacturing_config,
        )

        for wafer_number in range(dataset_config["wafers_per_lot"]):

            if device_counter >= total_devices:
                break

            wafer_id = (
                f"{lot_id}_WAFER_{wafer_number + 1:03d}"
            )

            wafer_shift = sample_wafer_variation(
                rng,
                manufacturing_config,
            )

            for device_number in range(dataset_config["devices_per_wafer"]):

                if device_counter >= total_devices:
                    break

                device_id = (
                    f"{wafer_id}_DEVICE_{device_number + 1:04d}"
                )

                device_shift = sample_device_variation(
                    rng,
                    manufacturing_config,
                )

                from .manufacturing import combine_manufacturing_effects

                effects = combine_manufacturing_effects(
                    lot_shift,
                    wafer_shift,
                    device_shift,
                )

                category = rng.choice(
                    [item.value for item in categories],
                    p=probabilities_array,
                )

                records = _generate_device_records(
                    rng=rng,
                    profile=profile,
                    manufacturing_effects=effects,
                    operating_config=operating_config,
                    failure_config=failure_config,
                    failure_category=category,
                    measurements_per_device=measurements,
                    device_id=device_id,
                    wafer_id=wafer_id,
                    lot_id=lot_id,
                    first_record_id=device_counter * measurements,
                )

                for column, values in records.items():
                    batch_parts.setdefault(column, []).append(
                        np.asarray(values)
                    )

                device_counter += 1

                if device_counter % chunk_devices == 0:
                    flush_batch()

    flush_batch()

    if record_counter != expected_rows:
        raise RuntimeError(
            f"Generated {record_counter} rows; "
            f"expected {expected_rows}."
        )

    print("\nDATA GENERATION COMPLETED")
    print(f"Devices: {device_counter:,}")
    print(f"Records: {record_counter:,}")
    print(f"Chunks: {chunk_counter}")
    print(f"Output: {output_dir.resolve()}")

    return {
        "devices": device_counter,
        "records": record_counter,
        "chunks": chunk_counter,
        "output_directory": str(output_dir),
    }


def main():

    parser = argparse.ArgumentParser(
        description="Generate synthetic semiconductor test data."
    )

    parser.add_argument(
        "--config",
        default="configs/data_generation.yaml",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    parser.add_argument(
        "--max-devices",
        type=int,
        default=None,
        help="Generate a limited number of devices for testing.",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    generate_dataset(
        config,
        output_dir=args.output_dir,
        max_devices=args.max_devices,
    )


if __name__ == "__main__":
    main()