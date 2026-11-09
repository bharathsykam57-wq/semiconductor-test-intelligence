"""
Semiconductor Failure Simulation Engine.

Simulates synthetic electrical failure signatures.

Failure effects are configurable modelling assumptions,
not calibrated representations of real silicon defects.
"""

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np


class FailureCategory(str, Enum):
    NORMAL = "normal"
    HIGH_LEAKAGE = "high_leakage"
    TIMING_FAILURE = "timing_failure"
    VOLTAGE_SENSITIVITY = "voltage_sensitivity"
    THERMAL_INSTABILITY = "thermal_instability"
    POWER_ANOMALY = "power_anomaly"
    FREQUENCY_INSTABILITY = "frequency_instability"
    INTERMITTENT_FAILURE = "intermittent_failure"


@dataclass(frozen=True)
class FailureEffects:
    """Multiplicative and additive effects on electrical measurements."""

    current_multiplier: float = 1.0
    leakage_multiplier: float = 1.0
    delay_multiplier: float = 1.0
    performance_delta: float = 0.0

    def __post_init__(self):

        for field_name in (
            "current_multiplier",
            "leakage_multiplier",
            "delay_multiplier",
        ):

            value = getattr(self, field_name)

            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{field_name} must be finite and non-negative."
                )

        if not math.isfinite(self.performance_delta):
            raise ValueError("performance_delta must be finite.")


@dataclass(frozen=True)
class FailureSimulationConfig:
    """Synthetic failure simulation parameters."""

    high_leakage_multiplier: float = 4.0
    timing_delay_multiplier: float = 1.5

    voltage_sensitivity_factor: float = 2.0
    thermal_instability_factor: float = 3.0

    power_anomaly_multiplier: float = 1.5
    frequency_instability_multiplier: float = 1.4

    intermittent_failure_probability: float = 0.10

    def __post_init__(self):

        positive_fields = (
            "high_leakage_multiplier",
            "timing_delay_multiplier",
            "voltage_sensitivity_factor",
            "thermal_instability_factor",
            "power_anomaly_multiplier",
            "frequency_instability_multiplier",
        )

        for field_name in positive_fields:

            value = getattr(self, field_name)

            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{field_name} must be finite and positive."
                )

        probability = self.intermittent_failure_probability

        if (
            not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError(
                "intermittent_failure_probability must be between 0 and 1."
            )


def simulate_failure_effects(
    category: FailureCategory,
    config: FailureSimulationConfig,
    voltage_v: float,
    frequency_mhz: float,
    temperature_c: float,
    nominal_voltage_v: float = 0.90,
    nominal_frequency_mhz: float = 1000.0,
    nominal_temperature_c: float = 25.0,
) -> FailureEffects:
    """
    Calculate synthetic failure effects for one measurement.

    Operating-condition dependencies are explicitly modelled.
    """

    if not isinstance(category, FailureCategory):
        raise TypeError("category must be a FailureCategory.")

    numeric_values = (
        voltage_v,
        frequency_mhz,
        temperature_c,
        nominal_voltage_v,
        nominal_frequency_mhz,
        nominal_temperature_c,
    )

    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("Operating conditions must be finite.")

    if (
        voltage_v <= 0
        or frequency_mhz <= 0
        or nominal_voltage_v <= 0
        or nominal_frequency_mhz <= 0
    ):
        raise ValueError("Voltage and frequency must be positive.")

    effects = FailureEffects()

    if category == FailureCategory.NORMAL:

        return effects

    elif category == FailureCategory.HIGH_LEAKAGE:

        effects = FailureEffects(
            leakage_multiplier=config.high_leakage_multiplier,
            performance_delta=-0.05,
        )

    elif category == FailureCategory.TIMING_FAILURE:

        effects = FailureEffects(
            delay_multiplier=config.timing_delay_multiplier,
            performance_delta=-0.20,
        )

    elif category == FailureCategory.VOLTAGE_SENSITIVITY:

        voltage_deviation = abs(
            voltage_v - nominal_voltage_v
        ) / nominal_voltage_v

        multiplier = (
            1.0
            + config.voltage_sensitivity_factor * voltage_deviation
        )

        effects = FailureEffects(
            current_multiplier=multiplier,
            delay_multiplier=multiplier,
            performance_delta=-min(0.30, voltage_deviation),
        )

    elif category == FailureCategory.THERMAL_INSTABILITY:

        temperature_deviation = abs(
            temperature_c - nominal_temperature_c
        ) / 100.0

        multiplier = (
            1.0
            + config.thermal_instability_factor * temperature_deviation
        )

        effects = FailureEffects(
            leakage_multiplier=multiplier,
            delay_multiplier=multiplier,
            performance_delta=-min(0.30, temperature_deviation),
        )

    elif category == FailureCategory.POWER_ANOMALY:

        effects = FailureEffects(
            current_multiplier=config.power_anomaly_multiplier,
            performance_delta=-0.10,
        )

    elif category == FailureCategory.FREQUENCY_INSTABILITY:

        frequency_deviation = abs(
            frequency_mhz - nominal_frequency_mhz
        ) / nominal_frequency_mhz

        multiplier = (
            1.0
            + config.frequency_instability_multiplier
            * frequency_deviation
        )

        effects = FailureEffects(
            delay_multiplier=multiplier,
            performance_delta=-min(0.30, frequency_deviation),
        )

    elif category == FailureCategory.INTERMITTENT_FAILURE:

        effects = FailureEffects()

    return effects


def sample_intermittent_failure(
    rng: np.random.Generator,
    config: FailureSimulationConfig,
) -> bool:
    """
    Decide whether an intermittent failure occurs
    during one individual measurement.
    """

    return bool(
        rng.random() < config.intermittent_failure_probability
    )


def apply_intermittent_failure(
    effects: FailureEffects,
) -> FailureEffects:
    """
    Apply a synthetic transient failure signature
    to an individual measurement.
    """

    return FailureEffects(
        current_multiplier=effects.current_multiplier * 1.5,
        leakage_multiplier=effects.leakage_multiplier * 2.0,
        delay_multiplier=effects.delay_multiplier * 1.5,
        performance_delta=effects.performance_delta - 0.20,
    )