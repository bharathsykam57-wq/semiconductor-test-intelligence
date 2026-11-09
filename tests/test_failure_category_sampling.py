"""Regression tests for STIP failure-category sampling."""

import numpy as np
import pytest

from src.data_generation.failure_simulation import FailureCategory


@pytest.mark.parametrize("category", list(FailureCategory))
def test_numpy_sampling_preserves_failure_category(category):
    """NumPy sampling must preserve valid enum values without truncation."""

    rng = np.random.default_rng(42)

    sampled = rng.choice([category.value])

    recovered = FailureCategory(str(sampled))

    assert recovered == category
    assert str(sampled) == category.value
