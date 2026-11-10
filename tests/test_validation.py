"""Error-path tests: every public function must reject bad inputs cleanly."""

import numpy as np
import pytest

from oscfit.fit import fit_oscillation
from oscfit.detect import detect_periods
from oscfit.seismology import derive_parameters
from oscfit.extract import extract_slit_timeseries
from oscfit.synthetic import make_signal, make_image_cube
from oscfit.models import two_regime


class TestFitValidation:
    def test_unknown_model_raises(self):
        t = np.arange(100) * 3.0
        s = np.sin(2 * np.pi * t / 80)
        with pytest.raises(ValueError, match="Unknown model"):
            fit_oscillation(t, s, model="bogus_model")

    def test_shape_mismatch_raises(self):
        t = np.arange(100) * 3.0
        s = np.sin(2 * np.pi * t / 80)[:50]
        with pytest.raises(ValueError, match="same shape"):
            fit_oscillation(t, s)

    def test_constant_signal_raises(self):
        t = np.arange(100) * 3.0
        s = np.full(100, 50.0)
        with pytest.raises(ValueError, match="constant"):
            fit_oscillation(t, s, model="decayless", mcmc=False)

    def test_insufficient_data_raises(self):
        t = np.arange(3) * 3.0
        s = np.sin(t)
        with pytest.raises(ValueError, match="at least"):
            fit_oscillation(t, s, model="decayless", mcmc=False)

    def test_non_monotonic_time_raises(self):
        t = np.array([0, 3, 2, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        s = np.sin(t)
        with pytest.raises(ValueError, match="monotonic"):
            fit_oscillation(t, s, model="decayless", mcmc=False)

    def test_all_nan_signal_raises(self):
        t = np.arange(100) * 3.0
        s = np.full(100, np.nan)
        with pytest.raises(ValueError, match="at least"):
            fit_oscillation(t, s, model="decayless", mcmc=False)

    def test_seed_reproducibility(self):
        d = make_signal(model="decayless", duration=400, cadence=3, snr=15, seed=1)
        r1 = fit_oscillation(d["time"], d["signal"], model="decayless",
                             mcmc=True, n_walkers=8, n_steps=200, n_burn=50, seed=123)
        r2 = fit_oscillation(d["time"], d["signal"], model="decayless",
                             mcmc=True, n_walkers=8, n_steps=200, n_burn=50, seed=123)
        np.testing.assert_array_equal(r1.samples, r2.samples)


class TestDetectValidation:
    def test_bad_method_raises(self):
        t = np.arange(100) * 3.0
        s = np.sin(2 * np.pi * t / 80)
        with pytest.raises(ValueError, match="method"):
            detect_periods(t, s, method="bogus")

    def test_invalid_confidence_raises(self):
        t = np.arange(100) * 3.0
        s = np.sin(2 * np.pi * t / 80)
        with pytest.raises(ValueError, match="confidence"):
            detect_periods(t, s, confidence=1.0)
        with pytest.raises(ValueError, match="confidence"):
            detect_periods(t, s, confidence=0.0)
        with pytest.raises(ValueError, match="confidence"):
            detect_periods(t, s, confidence=-0.5)

    def test_shape_mismatch_raises(self):
        t = np.arange(100) * 3.0
        s = np.sin(t)[:50]
        with pytest.raises(ValueError, match="same shape"):
            detect_periods(t, s)

    def test_invalid_period_range_raises(self):
        t = np.arange(100) * 3.0
        s = np.sin(2 * np.pi * t / 80)
        with pytest.raises(ValueError, match="period range"):
            detect_periods(t, s, min_period=200, max_period=100)
        with pytest.raises(ValueError, match="period range"):
            detect_periods(t, s, min_period=0)

    def test_non_monotonic_time_raises(self):
        t = np.array([0, 3, 6, 4, 5, 8, 9, 10, 11, 12], dtype=float)
        s = np.sin(t)
        with pytest.raises(ValueError, match="monotonic"):
            detect_periods(t, s)


class TestSeismologyValidation:
    def test_zero_period_raises(self):
        with pytest.raises(ValueError, match="Period"):
            derive_parameters(P=0, A=100, L=15)

    def test_negative_period_raises(self):
        with pytest.raises(ValueError, match="Period"):
            derive_parameters(P=-100, A=100, L=15)

    def test_negative_amplitude_raises(self):
        with pytest.raises(ValueError, match="Amplitude"):
            derive_parameters(P=100, A=-50, L=15)

    def test_zero_loop_length_raises(self):
        with pytest.raises(ValueError, match="Loop length"):
            derive_parameters(P=100, A=50, L=0)

    def test_zeta_le_one_raises(self):
        with pytest.raises(ValueError, match="zeta"):
            derive_parameters(P=100, A=50, L=15, zeta=1.0)
        with pytest.raises(ValueError, match="zeta"):
            derive_parameters(P=100, A=50, L=15, zeta=0.5)

    def test_zero_density_raises(self):
        with pytest.raises(ValueError, match="density"):
            derive_parameters(P=100, A=50, L=15, n_e=0)

    def test_nan_inputs_rejected(self):
        with pytest.raises(ValueError):
            derive_parameters(P=np.nan, A=50, L=15)
        with pytest.raises(ValueError):
            derive_parameters(P=100, A=np.nan, L=15)
        with pytest.raises(ValueError):
            derive_parameters(P=100, A=50, L=np.nan)


class TestExtractValidation:
    def test_non_3d_cube_raises(self):
        cube_2d = np.zeros((50, 100))
        path = np.array([[10, 10], [20, 20]])
        with pytest.raises(ValueError, match="3D"):
            extract_slit_timeseries(cube_2d, path)

    def test_bad_loop_path_shape_raises(self):
        cube = np.zeros((10, 50, 50))
        bad_path = np.array([1, 2, 3])  # 1D
        with pytest.raises(ValueError, match="loop_path"):
            extract_slit_timeseries(cube, bad_path)

    def test_single_point_loop_path_raises(self):
        cube = np.zeros((10, 50, 50))
        path = np.array([[25.0, 25.0]])
        with pytest.raises(ValueError, match="at least 2"):
            extract_slit_timeseries(cube, path)

    def test_out_of_range_slit_index_raises(self):
        cube = np.zeros((10, 50, 50))
        path = np.array([[10.0, 10.0], [20.0, 20.0]])
        with pytest.raises(ValueError, match="out of range"):
            extract_slit_timeseries(cube, path, slit_index=5)

    def test_negative_slit_width_raises(self):
        cube = np.zeros((10, 50, 50))
        path = np.array([[10.0, 10.0], [20.0, 20.0]])
        with pytest.raises(ValueError, match="slit_width"):
            extract_slit_timeseries(cube, path, slit_width=-1)

    def test_out_of_bounds_loop_path_warns(self):
        cube = np.zeros((10, 50, 50))
        path = np.array([[100.0, 100.0], [200.0, 200.0]])
        with pytest.warns(UserWarning, match="outside the image"):
            extract_slit_timeseries(cube, path, jitter_correct=False)


class TestModelValidation:
    def test_two_regime_zero_tau_raises(self):
        t = np.linspace(0, 100, 50)
        with pytest.raises(ValueError, match="tau"):
            two_regime(t, A=100, P=60, phi=0, tau_g=0, tau_d=200, c0=0, c1=0)
        with pytest.raises(ValueError, match="tau"):
            two_regime(t, A=100, P=60, phi=0, tau_g=100, tau_d=0, c0=0, c1=0)

    def test_models_handle_overflow_gracefully(self):
        from oscfit.models import exponential, growing, gaussian
        t = np.array([0.0, 1e6])
        r1 = exponential(t, A=100, P=60, phi=0, tau=1e-6, c0=0, c1=0)
        assert np.all(np.isfinite(r1))
        r2 = growing(t, A=100, P=60, phi=0, tau=1e-6, c0=0, c1=0)
        assert np.all(np.isfinite(r2))
        r3 = gaussian(t, A=100, P=60, phi=0, tau_g=1e-6, c0=0, c1=0)
        assert np.all(np.isfinite(r3))


class TestSigmaZeroHandling:
    def test_perfect_signal_no_nan_chi2(self):
        t = np.arange(0, 600, 3.0)
        s = 100.0 * np.sin(2 * np.pi * t / 120)
        result = fit_oscillation(t, s, model="decayless", mcmc=False)
        chi2 = result.reduced_chi2()
        bic = result.bic()
        assert np.isfinite(chi2)
        assert np.isfinite(bic)
