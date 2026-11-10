"""Tests for seismology derivation and model comparison."""

import numpy as np
import pytest

from oscfit.seismology import derive_parameters, MU_0, M_P, MU_E
from oscfit.compare import compare_models, _prior_bounds
from oscfit.synthetic import make_signal


class TestSeismology:
    def test_kink_speed(self):
        result = derive_parameters(P=100, A=50, L=15)
        C_k_expected = 2 * 15e6 / 100  # m/s
        assert result.C_k == pytest.approx(C_k_expected)

    def test_velocity_amplitude(self):
        result = derive_parameters(P=100, A=50, L=15)
        V_expected = 2 * np.pi * 50e3 / 100
        assert result.V == pytest.approx(V_expected)

    def test_magnetic_field_order_of_magnitude(self):
        result = derive_parameters(P=150, A=80, L=20, n_e=1e15, zeta=2)
        assert 0.1 < result.B_gauss < 50.0

    def test_energy_flux_positive(self):
        result = derive_parameters(P=100, A=100, L=15)
        assert result.E_flux > 0

    def test_analytical_errors(self):
        result = derive_parameters(
            P=100, A=50, L=15,
            P_err=5, A_err=10, L_err=2,
        )
        assert result.B_err is not None
        assert result.B_err > 0
        assert result.C_k_err > 0
        assert result.V_err > 0

    def test_monte_carlo_errors(self):
        rng = np.random.default_rng(42)
        samples = np.column_stack([
            rng.normal(50, 5, 1000),
            rng.normal(100, 3, 1000),
            rng.normal(0, 0.5, 1000),
            rng.normal(200, 10, 1000),
            rng.normal(0, 1, 1000),
        ])
        result = derive_parameters(P=100, A=50, L=15, samples=samples)
        assert result.B_err is not None
        assert result.B_err > 0

    def test_density_contrast(self):
        r1 = derive_parameters(P=100, A=50, L=15, zeta=2)
        r2 = derive_parameters(P=100, A=50, L=15, zeta=5)
        assert r1.B_gauss != r2.B_gauss

    def test_to_dict(self):
        result = derive_parameters(P=100, A=50, L=15)
        d = result.to_dict()
        assert "B (G)" in d
        assert "Ck (km/s)" in d

    def test_summary(self):
        result = derive_parameters(P=100, A=50, L=15, P_err=5, A_err=10, L_err=2)
        s = result.summary()
        assert "Kink speed" in s
        assert "Magnetic field" in s

    def test_repr(self):
        result = derive_parameters(P=100, A=50, L=15)
        assert "SeismologyResult" in repr(result)


class TestValidateAgainstShrivastav:
    """Validate against published values from Shrivastav et al. (2024)."""

    def test_row1(self):
        result = derive_parameters(P=252, A=176, L=24.5, n_e=1e15, zeta=2)
        assert result.C_k_km_s == pytest.approx(194, rel=0.05)
        assert result.B_gauss == pytest.approx(2.6, rel=0.05)

    def test_row2(self):
        result = derive_parameters(P=31, A=50, L=29.4, n_e=1e15, zeta=2)
        assert result.C_k_km_s == pytest.approx(1897, rel=0.05)
        assert result.B_gauss == pytest.approx(25.3, rel=0.05)

    def test_row3(self):
        result = derive_parameters(P=31, A=86, L=37.1, n_e=1e15, zeta=2)
        assert result.C_k_km_s == pytest.approx(2394, rel=0.05)
        assert result.B_gauss == pytest.approx(31.9, rel=0.05)

    def test_row4(self):
        result = derive_parameters(P=95, A=68, L=7.7, n_e=1e15, zeta=2)
        assert result.C_k_km_s == pytest.approx(162, rel=0.05)
        assert result.B_gauss == pytest.approx(2.2, rel=0.05)

    def test_row10(self):
        result = derive_parameters(P=25, A=60, L=28.9, n_e=1e15, zeta=2)
        assert result.C_k_km_s == pytest.approx(2312, rel=0.05)
        assert result.B_gauss == pytest.approx(30.8, rel=0.05)

    def test_velocity_amplitudes(self):
        """Cross-check V = 2πA/P against published velocity amplitudes."""
        cases = [
            (252, 176, 4.4),
            (95, 68, 4.5),
            (25, 60, 15.1),
            (308, 355, 7.2),
            (25, 94, 23.6),
        ]
        for P, A, V_pub in cases:
            V_calc = 2 * np.pi * A / P  # km/s (A is already km, P is s)
            assert V_calc == pytest.approx(V_pub, abs=0.1)


class TestPriorBounds:
    def test_decayless_bounds(self):
        t = np.arange(200) * 3.0
        s = np.sin(2 * np.pi * t / 100)
        bounds = _prior_bounds("decayless", t, s, 100)
        assert len(bounds) == 5

    def test_exponential_bounds(self):
        t = np.arange(200) * 3.0
        s = np.sin(2 * np.pi * t / 100)
        bounds = _prior_bounds("exponential", t, s, 100)
        assert len(bounds) == 6


class TestModelComparison:
    def test_prefers_decayless_for_decayless_signal(self):
        d = make_signal(
            model="decayless",
            params={"A": 100, "P": 80, "phi": 0, "c0": 200, "c1": 0},
            duration=500, cadence=3, snr=25, seed=42,
        )
        result = compare_models(
            d["time"], d["signal"],
            models_to_compare=["decayless", "exponential"],
            n_live=100, dlogz=1.0,
        )
        assert result.preferred_model == "decayless"

    def test_bayes_factor(self):
        d = make_signal(model="decayless", duration=500, cadence=3, snr=25, seed=42)
        result = compare_models(
            d["time"], d["signal"],
            models_to_compare=["decayless", "exponential"],
            n_live=100, dlogz=1.0,
        )
        K = result.bayes_factor("decayless", "exponential")
        assert np.isfinite(K)

    def test_ranking(self):
        d = make_signal(model="decayless", duration=500, cadence=3, snr=25, seed=42)
        result = compare_models(
            d["time"], d["signal"],
            models_to_compare=["decayless", "exponential"],
            n_live=100, dlogz=1.0,
        )
        ranked = result.ranking()
        assert len(ranked) == 2
        assert ranked[0][0] == result.preferred_model

    def test_summary(self):
        d = make_signal(model="decayless", duration=500, cadence=3, snr=25, seed=42)
        result = compare_models(
            d["time"], d["signal"],
            models_to_compare=["decayless", "exponential"],
            n_live=100, dlogz=1.0,
        )
        s = result.summary()
        assert "Preferred" in s
