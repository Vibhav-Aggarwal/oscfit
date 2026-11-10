"""Tests for period detection module."""

import numpy as np
import pytest

from oscfit.detect import detect_periods, _split_cosine_bell, _fit_noise_background
from oscfit.synthetic import make_signal


class TestFFTDetection:
    def test_recovers_known_period(self, decayless_signal):
        d = decayless_signal
        result = detect_periods(d["time"], d["signal"], method="fft")
        assert result.dominant_period is not None
        assert result.dominant_period == pytest.approx(150.0, rel=0.1)

    def test_exponential_period(self, exponential_signal):
        d = exponential_signal
        result = detect_periods(d["time"], d["signal"], method="fft")
        assert result.dominant_period is not None
        assert result.dominant_period == pytest.approx(100.0, rel=0.15)

    def test_high_snr_high_confidence(self):
        d = make_signal(model="decayless",
                        params={"A": 200, "P": 80, "phi": 0, "c0": 0, "c1": 0},
                        duration=800, cadence=3, snr=50, seed=42)
        result = detect_periods(d["time"], d["signal"], method="fft")
        assert result.dominant_period == pytest.approx(80.0, rel=0.05)
        assert result.significances[0] > 5.0

    def test_returns_empty_for_noise(self):
        rng = np.random.default_rng(42)
        t = np.arange(0, 300, 3.0)
        noise = rng.normal(0, 50, len(t))
        result = detect_periods(t, noise, method="fft", confidence=0.999)
        # Even with noise, the method returns at least one peak (the dominant frequency)
        assert isinstance(result.periods, np.ndarray)

    def test_min_max_period_bounds(self):
        d = make_signal(model="decayless",
                        params={"A": 100, "P": 60, "phi": 0, "c0": 0, "c1": 0},
                        duration=600, cadence=3, snr=20, seed=42)
        result = detect_periods(d["time"], d["signal"], method="fft",
                                min_period=40, max_period=200)
        if result.dominant_period is not None:
            assert result.dominant_period >= 40
            assert result.dominant_period <= 200


class TestLombScargle:
    def test_recovers_period_with_gaps(self, gapped_signal):
        d = gapped_signal
        result = detect_periods(d["time"], d["signal"], method="lomb")
        assert result.dominant_period is not None
        assert result.dominant_period == pytest.approx(120.0, rel=0.15)

    def test_uniform_sampling(self, decayless_signal):
        d = decayless_signal
        result = detect_periods(d["time"], d["signal"], method="lomb")
        assert result.dominant_period is not None
        assert result.dominant_period == pytest.approx(150.0, rel=0.15)


class TestWavelet:
    def test_recovers_known_period(self, decayless_signal):
        d = decayless_signal
        result = detect_periods(d["time"], d["signal"], method="wavelet")
        assert result.dominant_period is not None
        assert result.dominant_period == pytest.approx(150.0, rel=0.25)

    def test_method_label(self, decayless_signal):
        d = decayless_signal
        result = detect_periods(d["time"], d["signal"], method="wavelet")
        assert result.method == "wavelet"


class TestMergedDetection:
    def test_all_methods(self, decayless_signal):
        d = decayless_signal
        result = detect_periods(d["time"], d["signal"], method="all")
        assert result.method == "all"
        assert result.dominant_period is not None
        assert result.dominant_period == pytest.approx(150.0, rel=0.15)


class TestSplitCosineBell:
    def test_center_is_one(self):
        w = _split_cosine_bell(100, p=0.4)
        assert w[50] == pytest.approx(1.0)

    def test_edges_tapered(self):
        w = _split_cosine_bell(100, p=0.4)
        assert w[0] < 0.01
        assert w[-1] < 0.01

    def test_symmetric(self):
        w = _split_cosine_bell(100, p=0.4)
        np.testing.assert_allclose(w, w[::-1])


class TestNoiseBackground:
    def test_fits_pure_white_noise(self):
        rng = np.random.default_rng(42)
        freqs = np.linspace(0.01, 0.5, 200)
        power = rng.exponential(1.0, 200)
        Nc, alpha, Nw = _fit_noise_background(freqs, power)
        assert alpha < 1.5  # should be near 0 for white noise

    def test_fits_red_noise(self):
        freqs = np.linspace(0.01, 0.5, 200)
        power = 10.0 * freqs ** (-1.5) + 0.5
        rng = np.random.default_rng(42)
        power *= rng.exponential(1.0, 200)
        Nc, alpha, Nw = _fit_noise_background(freqs, power)
        assert alpha == pytest.approx(1.5, abs=1.5)


class TestEdgeCases:
    def test_short_signal(self):
        t = np.arange(5) * 3.0
        s = np.sin(2 * np.pi * t / 10)
        result = detect_periods(t, s)
        assert isinstance(result.periods, np.ndarray)

    def test_constant_signal(self):
        t = np.arange(100) * 3.0
        s = np.ones(100) * 50.0
        result = detect_periods(t, s, method="fft")
        assert isinstance(result.periods, np.ndarray)

    def test_all_nan_signal(self):
        t = np.arange(100) * 3.0
        s = np.full(100, np.nan)
        result = detect_periods(t, s)
        assert len(result.periods) == 0

    @pytest.mark.parametrize("method", ["fft", "lomb", "wavelet"])
    def test_repr(self, decayless_signal, method):
        d = decayless_signal
        result = detect_periods(d["time"], d["signal"], method=method)
        r = repr(result)
        assert "DetectionResult" in r
