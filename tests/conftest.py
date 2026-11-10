"""Shared fixtures for oscfit test suite."""

import numpy as np
import pytest

from oscfit.synthetic import make_signal, make_xt_map, make_image_cube


SEED = 42


@pytest.fixture
def decayless_signal():
    """Clean decayless signal with moderate SNR."""
    return make_signal(
        model="decayless",
        params={"A": 80.0, "P": 150.0, "phi": 0.3, "c0": 200.0, "c1": 0.05},
        duration=900.0,
        cadence=3.0,
        snr=15.0,
        seed=SEED,
    )


@pytest.fixture
def exponential_signal():
    """Exponentially damped signal."""
    return make_signal(
        model="exponential",
        params={"A": 120.0, "P": 100.0, "phi": 0.0, "tau": 400.0, "c0": 300.0, "c1": 0.0},
        duration=600.0,
        cadence=5.0,
        snr=20.0,
        seed=SEED,
    )


@pytest.fixture
def gaussian_signal():
    """Gaussian-damped signal."""
    return make_signal(
        model="gaussian",
        params={"A": 100.0, "P": 80.0, "phi": 1.0, "tau_g": 200.0, "c0": 250.0, "c1": -0.1},
        duration=500.0,
        cadence=3.0,
        snr=12.0,
        seed=SEED,
    )


@pytest.fixture
def noisy_signal():
    """Low-SNR signal for robustness testing."""
    return make_signal(
        model="decayless",
        params={"A": 50.0, "P": 200.0, "phi": 0.0, "c0": 100.0, "c1": 0.0},
        duration=800.0,
        cadence=5.0,
        snr=3.0,
        seed=SEED,
    )


@pytest.fixture
def gapped_signal():
    """Signal with data gaps."""
    return make_signal(
        model="decayless",
        params={"A": 80.0, "P": 120.0, "phi": 0.0, "c0": 200.0, "c1": 0.0},
        duration=600.0,
        cadence=3.0,
        snr=10.0,
        seed=SEED,
        gaps=[(100.0, 115.0), (300.0, 330.0)],
    )


@pytest.fixture
def synthetic_xt():
    """Synthetic time-distance map with known oscillation."""
    return make_xt_map(
        nx=80,
        nt=200,
        loop_center=40.0,
        loop_width=3.5,
        loop_intensity=500.0,
        background=100.0,
        oscillation_amplitude=2.5,
        period=100.0,
        cadence=3.0,
        instrument="eui_hrieuv",
        seed=SEED,
    )


@pytest.fixture
def synthetic_cube():
    """Synthetic image cube with oscillating loop."""
    return make_image_cube(
        ny=128,
        nx=128,
        nt=60,
        loop_width=2.5,
        loop_intensity=300.0,
        oscillation_amplitude=1.5,
        period=80.0,
        cadence=5.0,
        seed=SEED,
    )
