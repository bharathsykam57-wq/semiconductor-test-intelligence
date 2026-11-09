"""
Device Characteristics Module.

Defines nominal electrical characteristics for synthetic
semiconductor devices.

These values represent a generic CMOS-inspired simulation
profile, not calibrated foundry measurements.
"""

from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class DeviceProfile:
    """
    Defines the nominal electrical characteristics
    of a synthetic semiconductor device.
    """

    profile_name: str = "generic_cmos_synthetic"

    # Reference operating conditions
    nominal_voltage_v: float = 0.90
    nominal_frequency_mhz: float = 1000.0
    nominal_temperature_c: float = 25.0

    # Electrical characteristics
    threshold_voltage_v: float = 0.32
    nominal_current_ma: float = 12.0
    nominal_leakage_current_ua: float = 2.0
    nominal_propagation_delay_ns: float = 0.55

    # Normalized performance indicator
    nominal_performance_score: float = 0.95

    def __post_init__(self):

        positive_fields = [
            "nominal_voltage_v",
            "nominal_frequency_mhz",
            "threshold_voltage_v",
            "nominal_current_ma",
            "nominal_leakage_current_ua",
            "nominal_propagation_delay_ns",
        ]

        for field_name in positive_fields:

            value = getattr(self, field_name)

            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{field_name} must be a finite positive number."
                )

        if not math.isfinite(self.nominal_temperature_c):
            raise ValueError(
                "nominal_temperature_c must be finite."
            )

        if not 0 <= self.nominal_performance_score <= 1:
            raise ValueError(
                "nominal_performance_score must be between 0 and 1."
            )

        if not self.profile_name.strip():
            raise ValueError("profile_name cannot be empty.")

    def to_dict(self) -> dict:
        """
        Convert the device profile into a dictionary.
        """

        return asdict(self)


def get_default_device_profile() -> DeviceProfile:
    """
    Return the default synthetic semiconductor profile.
    """

    return DeviceProfile()