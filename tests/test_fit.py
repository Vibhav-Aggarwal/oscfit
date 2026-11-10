"""Tests for MCMC oscillation fitting."""

import numpy as np
import pytest

from oscfit.fit import fit_oscillation
from oscfit.synthetic import make_signal


class TestDecaylessFit:
    def test_recovers_parameters_no_mcmc(self, decayless_signal):
        d = decayless_signal
        result = fit_oscillation(d["time"], d["signal"], model="decayless", mcmc=False)
        P_true = d["params"]["P"]
        A_true = d["params"]["A"]
        assert result.period[0] == pytest.approx(P_true, rel=0.1)
        assert result.amplitude[0] == pytest.approx(A_true, rel=0.2)

    def test_recovers_parameters_with_mcmc(self):
        d = make_signal(
            model="decayless",
            params={"A": 100.0, "P": 120.0, "phi": 0.5, "c0": 200.0, "c1": 0.0},
            duration=720.0, cadence=3.0, snr=20.0, seed=42,
        )
        result = fit_oscillation(
            d["time"], d["signal"], model="decayless",
            mcmc=True, n_walkers=16, n_steps=1500, n_burn=500,
        )
        assert result.period[0] == pytest.approx(120.0, rel=0.1)
        assert result.amplitude[0] == pytest.approx(100.0, rel=0.2)
        assert result.samples is not None
        assert result.samples.shape[1] == 5

    def test_velocity_amplitude(self):
        d = make_signal(
            model="decayless",
            params={"A": 50.0, "P": 100.0, "phi": 0.0, "c0": 0.0, "c1": 0.0},
            duration=500.0, cadence=3.0, snr=30.0, seed=42,
        )
        result = fit_oscillation(d["time"], d["signal"], model="decayless", mcmc=False)
        V, dV = result.velocity_amplitude
        V_expected = 2 * np.pi * 50.0 / 100.0
        assert V == pytest.approx(V_expected, rel=0.15)


class TestExponentialFit:
    def test_recovers_damping(self, exponential_signal):
        d = exponential_signal
        result = fit_oscillation(d["time"], d["signal"], model="exponential", mcmc=False)
        assert result.period[0] == pytest.approx(100.0, rel=0.15)
        assert result.params["tau"] > 0


class TestFitOutputs:
    def test_reduced_chi2_reasonable(self, decayless_signal):
        d = decayless_signal
        result = fit_oscillation(d["time"], d["signal"], model="decayless", mcmc=False)
        assert 0.1 < result.reduced_chi2() < 20.0

    def test_bic_aic(self, decayless_signal):
        d = decayless_signal
        result = fit_oscillation(d["time"], d["signal"], model="decayless", mcmc=False)
        assert np.isfinite(result.bic())
        assert np.isfinite(result.aic())

    def test_to_dict(self, decayless_signal):
        d = decayless_signal
        result = fit_oscillation(d["time"], d["signal"], model="decayless", mcmc=False)
        row = result.to_dict()
        assert "P (s)" in row
        assert "A (km)" in row
        assert "±" in row["P (s)"]

    def test_summary(self, decayless_signal):
        d = decayless_signal
        result = fit_oscillation(d["time"], d["signal"], model="decayless", mcmc=False)
        s = result.summary()
        assert "decayless" in s
        assert "χ²_red" in s

    def test_repr(self, decayless_signal):
        d = decayless_signal
        result = fit_oscillation(d["time"], d["signal"], model="decayless", mcmc=False)
        r = repr(result)
        assert "OscFitResult" in r

    def test_model_curve(self, decayless_signal):
        d = decayless_signal
        result = fit_oscillation(d["time"], d["signal"], model="decayless", mcmc=False)
        curve = result.model_curve()
        assert curve.shape == result.signal.shape


class TestAutoDetection:
    def test_period_auto_detected(self):
        d = make_signal(
            model="decayless",
            params={"A": 100.0, "P": 80.0, "phi": 0.0, "c0": 0.0, "c1": 0.0},
            duration=500.0, cadence=3.0, snr=20.0, seed=42,
        )
        result = fit_oscillation(d["time"], d["signal"], model="decayless",
                                 period_hint=None, mcmc=False)
        assert result.period[0] == pytest.approx(80.0, rel=0.15)


class TestConvergence:
    def test_convergence_check(self):
        d = make_signal(
            model="decayless",
            params={"A": 80.0, "P": 100.0, "phi": 0.0, "c0": 100.0, "c1": 0.0},
            duration=600.0, cadence=3.0, snr=25.0, seed=42,
        )
        result = fit_oscillation(
            d["time"], d["signal"], model="decayless",
            mcmc=True, n_walkers=16, n_steps=2000, n_burn=500,
        )
        conv = result.convergence_check()
        assert "converged" in conv

    def test_percentiles(self):
        d = make_signal(
            model="decayless",
            params={"A": 80.0, "P": 100.0, "phi": 0.0, "c0": 100.0, "c1": 0.0},
            duration=600.0, cadence=3.0, snr=25.0, seed=42,
        )
        result = fit_oscillation(
            d["time"], d["signal"], model="decayless",
            mcmc=True, n_walkers=16, n_steps=1500, n_burn=500,
        )
        p = result.percentiles("P")
        assert p is not None
        assert len(p) == 3
        assert p[0] < p[1] < p[2]


@pytest.mark.parametrize("model_name", ["decayless", "exponential", "gaussian", "growing"])
class TestAllModels:
    def test_fit_without_mcmc(self, model_name):
        d = make_signal(model=model_name, duration=600, cadence=3, snr=15, seed=42)
        result = fit_oscillation(d["time"], d["signal"], model=model_name, mcmc=False)
        assert result.model_name == model_name
        assert result.reduced_chi2() < 50.0
