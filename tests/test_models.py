"""Tests for oscillation models and synthetic data generators."""

import numpy as np
import pytest

from oscfit import models
from oscfit.synthetic import make_signal, make_xt_map, make_image_cube, INSTRUMENTS


class TestDecayless:
    def test_zero_time(self):
        result = models.decayless(0.0, A=100, P=60, phi=0, c0=0, c1=0)
        assert result == pytest.approx(0.0)

    def test_quarter_period(self):
        P = 100.0
        result = models.decayless(P / 4, A=50, P=P, phi=0, c0=0, c1=0)
        assert result == pytest.approx(50.0, rel=1e-10)

    def test_full_period(self):
        P = 80.0
        result = models.decayless(P, A=100, P=P, phi=0, c0=0, c1=0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_linear_trend(self):
        t = np.array([0.0, 10.0, 20.0])
        result = models.decayless(t, A=0, P=100, phi=0, c0=5, c1=2)
        np.testing.assert_allclose(result, [5.0, 25.0, 45.0])

    def test_array_input(self):
        t = np.linspace(0, 200, 100)
        result = models.decayless(t, A=50, P=100, phi=0, c0=0, c1=0)
        assert result.shape == (100,)
        assert np.max(result) == pytest.approx(50.0, abs=1.0)
        assert np.min(result) == pytest.approx(-50.0, abs=1.0)


class TestExponential:
    def test_initial_amplitude(self):
        result = models.exponential(0.0, A=100, P=60, phi=np.pi / 2, tau=200, c0=0, c1=0)
        assert result == pytest.approx(100.0, rel=1e-10)

    def test_decays_over_time(self):
        t = np.array([0.0, 200.0, 400.0])
        tau = 200.0
        envelope = 100.0 * np.exp(-t / tau)
        result = models.exponential(t, A=100, P=60, phi=np.pi / 2, tau=tau, c0=0, c1=0)
        assert np.abs(result[1]) < np.abs(result[0])
        assert np.abs(result[2]) < np.abs(result[1])

    def test_large_tau_approaches_decayless(self):
        t = np.linspace(0, 300, 100)
        tau = 1e10
        damped = models.exponential(t, A=50, P=100, phi=0, tau=tau, c0=0, c1=0)
        undamped = models.decayless(t, A=50, P=100, phi=0, c0=0, c1=0)
        np.testing.assert_allclose(damped, undamped, atol=1e-5)


class TestGaussian:
    def test_initial_amplitude(self):
        result = models.gaussian(0.0, A=100, P=60, phi=np.pi / 2, tau_g=200, c0=0, c1=0)
        assert result == pytest.approx(100.0, rel=1e-10)

    def test_symmetric_envelope(self):
        t = np.linspace(0, 400, 200)
        result = models.gaussian(t, A=100, P=50, phi=0, tau_g=100, c0=0, c1=0)
        envelope = 100.0 * np.exp(-t ** 2 / (2 * 100.0 ** 2))
        assert np.max(np.abs(result)) <= np.max(envelope) + 1e-10


class TestTwoRegime:
    def test_continuity_at_switch(self):
        tau_g, tau_d = 150.0, 300.0
        t_switch = tau_g ** 2 / tau_d
        dt = 0.01
        t = np.array([t_switch - dt, t_switch, t_switch + dt])
        result = models.two_regime(t, A=100, P=50, phi=0, tau_g=tau_g, tau_d=tau_d, c0=0, c1=0)
        assert abs(result[0] - result[1]) < 2.0
        assert abs(result[1] - result[2]) < 2.0

    def test_initial_gaussian_regime(self):
        tau_g, tau_d = 200.0, 500.0
        t_switch = tau_g ** 2 / tau_d
        t_early = np.linspace(0, t_switch * 0.5, 50)
        two_reg = models.two_regime(t_early, A=100, P=60, phi=0, tau_g=tau_g, tau_d=tau_d, c0=0, c1=0)
        gauss = models.gaussian(t_early, A=100, P=60, phi=0, tau_g=tau_g, c0=0, c1=0)
        np.testing.assert_allclose(two_reg, gauss, atol=1e-10)


class TestPeriodVarying:
    def test_zero_alpha_matches_gaussian(self):
        t = np.linspace(0, 300, 100)
        pv = models.period_varying(t, A=80, P0=100, phi=0, tau_g=500, alpha=0.0, c0=0, c1=0)
        ga = models.gaussian(t, A=80, P=100, phi=0, tau_g=500, c0=0, c1=0)
        np.testing.assert_allclose(pv, ga, atol=1e-10)

    def test_period_increases_with_positive_alpha(self):
        t = np.linspace(0, 500, 200)
        result = models.period_varying(t, A=50, P0=80, phi=0, tau_g=1e6, alpha=0.001, c0=0, c1=0)
        zero_crossings = np.where(np.diff(np.sign(result)))[0]
        if len(zero_crossings) >= 4:
            half_periods_early = np.diff(t[zero_crossings[:3]])
            half_periods_late = np.diff(t[zero_crossings[-3:]])
            assert np.mean(half_periods_late) > np.mean(half_periods_early)


class TestGrowing:
    def test_amplitude_grows(self):
        t = np.array([0.0, 100.0, 200.0])
        tau = 150.0
        result = models.growing(t, A=50, P=50, phi=np.pi / 2, tau=tau, c0=0, c1=0)
        assert np.abs(result[1]) > np.abs(result[0])
        assert np.abs(result[2]) > np.abs(result[1])


class TestModelsRegistry:
    def test_all_models_registered(self):
        expected = {"decayless", "exponential", "gaussian", "two_regime", "period_varying", "growing"}
        assert set(models.MODELS.keys()) == expected

    def test_param_counts_match_functions(self):
        import inspect
        for name, info in models.MODELS.items():
            sig = inspect.signature(info["func"])
            n_params_actual = len(sig.parameters) - 1  # subtract 't'
            assert info["n_params"] == n_params_actual, f"{name}: expected {info['n_params']}, got {n_params_actual}"
            assert len(info["param_names"]) == n_params_actual

    def test_all_models_callable_with_arrays(self):
        t = np.linspace(0, 300, 100)
        for name, info in models.MODELS.items():
            defaults = {
                "A": 50.0, "P": 80.0, "P0": 80.0, "phi": 0.0,
                "c0": 0.0, "c1": 0.0, "tau": 200.0, "tau_g": 150.0,
                "tau_d": 300.0, "alpha": 1e-4,
            }
            params = [defaults[p] for p in info["param_names"]]
            result = info["func"](t, *params)
            assert result.shape == t.shape, f"{name} output shape mismatch"
            assert np.all(np.isfinite(result)), f"{name} produced non-finite values"


class TestMakeSignal:
    def test_default_decayless(self):
        data = make_signal(model="decayless", seed=42)
        assert data["time"].shape == data["signal"].shape
        assert data["model"] == "decayless"
        assert "A" in data["params"]

    def test_snr_controls_noise(self):
        high = make_signal(snr=100, seed=42)
        low = make_signal(snr=3, seed=42)
        assert np.std(high["noise"]) < np.std(low["noise"])

    def test_clean_matches_model(self):
        data = make_signal(model="decayless", snr=0, seed=42)
        np.testing.assert_allclose(data["signal"], data["clean"])

    def test_gaps_produce_nans(self):
        data = make_signal(gaps=[(50, 80)], seed=42)
        gap_mask = (data["time"] >= 50) & (data["time"] <= 80)
        assert np.all(np.isnan(data["signal"][gap_mask]))
        assert np.all(np.isfinite(data["signal"][~gap_mask]))

    @pytest.mark.parametrize("model_name", models.MODELS.keys())
    def test_all_models_generate(self, model_name):
        data = make_signal(model=model_name, seed=42)
        assert len(data["time"]) > 10
        finite_mask = np.isfinite(data["signal"])
        assert np.sum(finite_mask) > 10


class TestMakeXtMap:
    def test_shape(self, synthetic_xt):
        xt = synthetic_xt
        assert xt["xt_map"].shape == (200, 80)
        assert xt["time"].shape == (200,)
        assert xt["x"].shape == (80,)

    def test_center_near_expected(self, synthetic_xt):
        xt = synthetic_xt
        centers = np.argmax(xt["xt_clean"], axis=1).astype(float)
        assert np.std(centers) > 0.5  # oscillation present
        assert np.mean(centers) == pytest.approx(40.0, abs=5.0)

    def test_noise_model(self):
        xt = make_xt_map(loop_intensity=1000, background=200, seed=42)
        residual = xt["xt_map"] - xt["xt_clean"]
        bright_mask = xt["xt_clean"] > 500
        dim_mask = xt["xt_clean"] < 250
        assert np.std(residual[bright_mask]) > np.std(residual[dim_mask])


class TestMakeImageCube:
    def test_shape(self, synthetic_cube):
        cube = synthetic_cube
        assert cube["cube"].shape == (60, 128, 128)
        assert cube["time"].shape == (60,)

    def test_loop_visible(self, synthetic_cube):
        cube = synthetic_cube
        frame = cube["cube"][0]
        assert np.max(frame) > np.median(frame) * 2

    def test_default_loop_is_semicircle(self, synthetic_cube):
        path = synthetic_cube["loop_path"]
        assert path.shape[1] == 2
        assert len(path) == 80

    def test_displacement_matches_period(self, synthetic_cube):
        d = synthetic_cube["true_displacement"]
        t = synthetic_cube["time"]
        zero_crossings = np.where(np.diff(np.sign(d)))[0]
        if len(zero_crossings) >= 2:
            half_period = t[zero_crossings[1]] - t[zero_crossings[0]]
            assert half_period == pytest.approx(40.0, abs=10.0)  # P=80, half=40
