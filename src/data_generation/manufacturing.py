"""
Manufacturing Variation Engine.

Simulates hierarchical manufacturing process variations
at lot, wafer, and device levels.

Variations are synthetic statistical assumptions and
are not calibrated against actual foundry measurements.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ManufacturingVariationConfig:
    """Configuration for hierarchical process variations."""

    lot_variation: float = 0.02
    wafer_variation: float = 0.015
    device_variation: float = 0.01

    def __post_init__(self):

        for field_name in (
            "lot_variation",
            "wafer_variation",
            "device_variation",
        ):

            value = getattr(self, field_name)

            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{field_name} must be finite and non-negative."
                )


@dataclass(frozen=True)
class ManufacturingEffects:
    """Combined manufacturing process deviations."""

    lot_shift: float
    wafer_shift: float
    device_shift: float

    @property
    def total_shift(self) -> float:
        """Return the combined additive process deviation."""

        return (
            self.lot_shift
            + self.wafer_shift
            + self.device_shift
        )


def sample_lot_variation(
    rng: np.random.Generator,
    config: ManufacturingVariationConfig,
) -> float:
    """Sample one process deviation for a manufacturing lot."""

    return float(rng.normal(0.0, config.lot_variation))


def sample_wafer_variation(
    rng: np.random.Generator,
    config: ManufacturingVariationConfig,
) -> float:
    """Sample one process deviation for a wafer."""

    return float(rng.normal(0.0, config.wafer_variation))


def sample_device_variation(
    rng: np.random.Generator,
    config: ManufacturingVariationConfig,
) -> float:
    """Sample one process deviation for a device."""

    return float(rng.normal(0.0, config.device_variation))


def combine_manufacturing_effects(
    lot_shift: float,
    wafer_shift: float,
    device_shift: float,
) -> ManufacturingEffects:
    """Combine hierarchical manufacturing deviations."""

    values = (lot_shift, wafer_shift, device_shift)

    if not all(math.isfinite(value) for value in values):
        raise ValueError("Manufacturing effects must be finite.")

    return ManufacturingEffects(
        lot_shift=lot_shift,
        wafer_shift=wafer_shift,
        device_shift=device_shift,
    )


def calculate_total_process_variation(
    config: ManufacturingVariationConfig,
) -> float:
    """
    Calculate the theoretical standard deviation of the
    combined independent manufacturing effects.
    """

    return math.sqrt(
        config.lot_variation**2
        + config.wafer_variation**2
        + config.device_variation**2
    )