"""
Operating Conditions Engine.

Generates reproducible semiconductor test conditions
across configured voltage, frequency and temperature ranges.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class OperatingConditionConfig:

    voltage_min_v: float = 0.7
    voltage_max_v: float = 1.1

    frequency_min_mhz: float = 400.0
    frequency_max_mhz: float = 1600.0

    temperature_min_c: float = -20.0
    temperature_max_c: float = 125.0

    def __post_init__(self):

        ranges = [
            ("voltage", self.voltage_min_v, self.voltage_max_v),
            ("frequency", self.frequency_min_mhz, self.frequency_max_mhz),
            ("temperature", self.temperature_min_c, self.temperature_max_c),
        ]

        for name, minimum, maximum in ranges:

            if not math.isfinite(minimum) or not math.isfinite(maximum):
                raise ValueError(f"{name} bounds must be finite.")

            if minimum >= maximum:
                raise ValueError(
                    f"{name} minimum must be smaller than maximum."
                )


@dataclass(frozen=True)
class OperatingCondition:

    test_condition_id: str

    voltage_v: float
    frequency_mhz: float
    temperature_c: float


def generate_operating_conditions(
    rng: np.random.Generator,
    config: OperatingConditionConfig,
    n_conditions: int,
) -> list[OperatingCondition]:
    """
    Generate reproducible operating conditions.

    Each variable is sampled independently using a
    uniform distribution over its configured range.
    """

    if not isinstance(n_conditions, int) or isinstance(n_conditions, bool):
        raise TypeError("n_conditions must be an integer.")

    if n_conditions <= 0:
        raise ValueError("n_conditions must be positive.")

    voltages = rng.uniform(
        config.voltage_min_v,
        config.voltage_max_v,
        n_conditions,
    )

    frequencies = rng.uniform(
        config.frequency_min_mhz,
        config.frequency_max_mhz,
        n_conditions,
    )

    temperatures = rng.uniform(
        config.temperature_min_c,
        config.temperature_max_c,
        n_conditions,
    )

    conditions = []

    for index in range(n_conditions):

        conditions.append(
            OperatingCondition(
                test_condition_id=f"TC_{index + 1:08d}",
                voltage_v=float(voltages[index]),
                frequency_mhz=float(frequencies[index]),
                temperature_c=float(temperatures[index]),
            )
        )

    return conditions