"""Coronal seismology: derive physical parameters from oscillation fits.

Converts fitted oscillation parameters (period, amplitude, loop length)
into physical quantities (magnetic field, Alfvén speed, energy flux)
using MHD kink mode theory. Supports Monte Carlo error propagation
from MCMC posterior samples.

References
----------
.. [1] Roberts, Edwin & Benz 1984, ApJ 279, 857
.. [2] Nakariakov & Ofman 2001, A&A 372, L53
.. [3] Van Doorsselaere et al. 2008, ApJ 676, L73
.. [4] Goossens et al. 2009, A&A 503, 213
"""

import numpy as np

MU_0 = 4.0 * np.pi * 1e-7       # vacuum permeability (T·m/A)
M_P = 1.6726e-27                  # proton mass (kg)
MU_E = 1.18                       # mean mass per electron, fully ionized H+He plasma
                                  # (matches Shrivastav et al. 2024 convention;
                                  #  ρ = n_e * μ_e * m_p)
KM_TO_M = 1e3
MM_TO_M = 1e6


def _validate_inputs(P, A, L, n_e, zeta):
    """Validate physical inputs to seismology calculations."""
    if not np.isfinite(P) or P <= 0:
        raise ValueError(f"Period P must be positive and finite, got {P}")
    if not np.isfinite(A) or A < 0:
        raise ValueError(f"Amplitude A must be non-negative and finite, got {A}")
    if not np.isfinite(L) or L <= 0:
        raise ValueError(f"Loop length L must be positive and finite, got {L}")
    if not np.isfinite(n_e) or n_e <= 0:
        raise ValueError(f"Electron density n_e must be positive, got {n_e}")
    if not np.isfinite(zeta) or zeta <= 1:
        raise ValueError(
            f"Density contrast zeta must be > 1 (loop denser than ambient), got {zeta}"
        )


def derive_parameters(P, A, L, P_err=None, A_err=None, L_err=None,
                      n_e=1e15, zeta=2.0, samples=None,
                      energy_flux_total=True):
    """Derive physical parameters from oscillation measurements.

    Parameters
    ----------
    P : float
        Period (s). Must be > 0.
    A : float
        Displacement amplitude (km). Must be >= 0.
    L : float
        Loop length (Mm). Must be > 0.
    P_err, A_err, L_err : float, optional
        Uncertainties on P, A, L.
    n_e : float
        Electron number density (m⁻³). Default: 10⁹ cm⁻³ = 10¹⁵ m⁻³.
    zeta : float
        Density contrast ratio ρ₀/ρ_e. Must be > 1. Default: 2.0.
    samples : ndarray, optional
        MCMC samples with columns [A, P, ...]. If provided, uses
        Monte Carlo error propagation instead of analytical formulas.
    energy_flux_total : bool
        If True, return total wave energy flux (kinetic + magnetic,
        assuming equipartition): F = ½(ρ₀+ρ_e)V²C_k. If False, return
        kinetic energy flux only: F = ¼(ρ₀+ρ_e)V²C_k.
        Default True matches Shrivastav et al. (2024) convention.

    Returns
    -------
    SeismologyResult
    """
    _validate_inputs(P, A, L, n_e, zeta)

    rho_0 = n_e * MU_E * M_P
    rho_e = rho_0 / zeta

    L_m = L * MM_TO_M
    A_m = A * KM_TO_M

    C_k = 2.0 * L_m / P
    V = 2.0 * np.pi * A_m / P
    V_A = C_k * np.sqrt((1.0 + 1.0 / zeta) / 2.0)
    B = V_A * np.sqrt(MU_0 * rho_0)
    e_flux_coeff = 0.5 if energy_flux_total else 0.25
    E_flux = e_flux_coeff * (rho_0 + rho_e) * V ** 2 * C_k

    result = SeismologyResult(
        C_k=C_k, V_A=V_A, B=B, V=V, E_flux=E_flux,
        rho_0=rho_0, rho_e=rho_e, n_e=n_e, zeta=zeta,
    )

    if samples is not None and len(samples) > 0:
        result = _monte_carlo_errors(result, samples, L, L_err, n_e, zeta)
    elif P_err is not None:
        result = _analytical_errors(result, P, A, L, P_err, A_err, L_err, n_e, zeta)

    return result


class SeismologyResult:
    """Container for derived physical parameters."""

    def __init__(self, C_k, V_A, B, V, E_flux, rho_0, rho_e, n_e, zeta):
        self.C_k = C_k
        self.C_k_err = None
        self.V_A = V_A
        self.V_A_err = None
        self.B = B                # Tesla
        self.B_err = None
        self.V = V                # m/s
        self.V_err = None
        self.E_flux = E_flux      # W/m²
        self.E_flux_err = None
        self.rho_0 = rho_0
        self.rho_e = rho_e
        self.n_e = n_e
        self.zeta = zeta

    @property
    def B_gauss(self):
        """Magnetic field in Gauss (1 T = 10⁴ G)."""
        return self.B * 1e4

    @property
    def B_gauss_err(self):
        return self.B_err * 1e4 if self.B_err is not None else None

    @property
    def C_k_km_s(self):
        """Kink speed in km/s."""
        return self.C_k / KM_TO_M

    @property
    def V_km_s(self):
        """Velocity amplitude in km/s."""
        return self.V / KM_TO_M

    @property
    def V_km_s_err(self):
        return self.V_err / KM_TO_M if self.V_err is not None else None

    def to_dict(self):
        """Export in Shrivastav et al. (2024) compatible format."""
        return {
            "Ck (km/s)": f"{self.C_k_km_s:.0f}",
            "B (G)": f"{self.B_gauss:.1f}",
            "V (km/s)": f"{self.V_km_s:.1f}",
            "E_flux (W/m2)": f"{self.E_flux:.1f}",
        }

    def summary(self):
        lines = ["Coronal Seismology Results"]
        lines.append(f"  Density: n_e = {self.n_e:.1e} m⁻³, ζ = {self.zeta:.1f}")
        lines.append(f"  ρ₀ = {self.rho_0:.2e} kg/m³, ρ_e = {self.rho_e:.2e} kg/m³")
        lines.append(f"  Kink speed:      C_k = {self.C_k_km_s:.0f} km/s"
                      + (f" ± {self.C_k_err / KM_TO_M:.0f}" if self.C_k_err else ""))
        lines.append(f"  Alfvén speed:    V_A = {self.V_A / KM_TO_M:.0f} km/s"
                      + (f" ± {self.V_A_err / KM_TO_M:.0f}" if self.V_A_err else ""))
        lines.append(f"  Magnetic field:  B   = {self.B_gauss:.1f} G"
                      + (f" ± {self.B_gauss_err:.1f}" if self.B_gauss_err else ""))
        lines.append(f"  Velocity amp:    V   = {self.V_km_s:.1f} km/s"
                      + (f" ± {self.V_km_s_err:.1f}" if self.V_km_s_err else ""))
        lines.append(f"  Energy flux:     F   = {self.E_flux:.1f} W/m²"
                      + (f" ± {self.E_flux_err:.1f}" if self.E_flux_err else ""))
        return "\n".join(lines)

    def __repr__(self):
        return (f"SeismologyResult(B={self.B_gauss:.1f} G, "
                f"C_k={self.C_k_km_s:.0f} km/s, "
                f"V={self.V_km_s:.1f} km/s, "
                f"F={self.E_flux:.1f} W/m²)")


def _monte_carlo_errors(result, samples, L, L_err, n_e, zeta):
    """Monte Carlo error propagation from MCMC posterior samples."""
    rng = np.random.default_rng(42)

    A_samples = samples[:, 0]
    P_samples = samples[:, 1]

    if L_err is not None and L_err > 0:
        L_samples = rng.normal(L, L_err, len(P_samples))
        L_samples = np.clip(L_samples, L * 0.5, L * 1.5)
    else:
        L_samples = np.full(len(P_samples), L)

    rho_0 = n_e * MU_E * M_P
    rho_e = rho_0 / zeta

    L_m = L_samples * MM_TO_M
    A_m = A_samples * KM_TO_M

    C_k = 2.0 * L_m / P_samples
    V = 2.0 * np.pi * A_m / P_samples
    V_A = C_k * np.sqrt((1.0 + 1.0 / zeta) / 2.0)
    B = V_A * np.sqrt(MU_0 * rho_0)
    E_flux = 0.5 * (rho_0 + rho_e) * V ** 2 * C_k

    result.C_k = np.median(C_k)
    result.C_k_err = np.std(C_k)
    result.V = np.median(V)
    result.V_err = np.std(V)
    result.V_A = np.median(V_A)
    result.V_A_err = np.std(V_A)
    result.B = np.median(B)
    result.B_err = np.std(B)
    result.E_flux = np.median(E_flux)
    result.E_flux_err = np.std(E_flux)

    return result


def _analytical_errors(result, P, A, L, P_err, A_err, L_err, n_e, zeta):
    """Analytical error propagation (first-order)."""
    L_m = L * MM_TO_M
    A_m = A * KM_TO_M

    if L_err is None:
        L_err = L * 0.1
    if A_err is None:
        A_err = A * 0.1
    L_err_m = L_err * MM_TO_M
    A_err_m = A_err * KM_TO_M

    frac_P = P_err / P
    frac_A = A_err_m / A_m if A_m > 0 else 0
    frac_L = L_err_m / L_m if L_m > 0 else 0

    result.C_k_err = result.C_k * np.sqrt(frac_L ** 2 + frac_P ** 2)
    result.V_err = result.V * np.sqrt(frac_A ** 2 + frac_P ** 2)
    result.V_A_err = result.V_A * np.sqrt(frac_L ** 2 + frac_P ** 2)
    result.B_err = result.B * np.sqrt(frac_L ** 2 + frac_P ** 2)
    result.E_flux_err = result.E_flux * np.sqrt(
        4 * frac_A ** 2 + frac_L ** 2 + 9 * frac_P ** 2
    )

    return result
