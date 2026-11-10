"""Oscillation models for coronal kink mode analysis.

References
----------
.. [1] Anfinogentov et al. 2015, A&A 583, A136 (decayless model)
.. [2] Nakariakov et al. 1999, Science 285, 862 (exponential damping)
.. [3] Pascoe et al. 2016a, A&A 585, L6 (Gaussian damping)
.. [4] Pascoe et al. 2016b, A&A 589, A136 (two-regime damping)
.. [5] Shrivastav et al. 2024, A&A 689, A295 (period-varying model)
"""

import numpy as np


def decayless(t, A, P, phi, c0, c1):
    r"""Decayless kink oscillation with linear trend.

    .. math::
        \xi(t) = A \sin\!\left(\frac{2\pi t}{P} + \varphi\right) + c_0 + c_1 t

    Parameters
    ----------
    t : array_like
        Time (s).
    A : float
        Displacement amplitude (km). Must be > 0.
    P : float
        Period (s). Must be > 0.
    phi : float
        Phase (rad).
    c0 : float
        Constant offset (km).
    c1 : float
        Linear trend slope (km/s).
    """
    return A * np.sin(2.0 * np.pi * t / P + phi) + c0 + c1 * t


def exponential(t, A, P, phi, tau, c0, c1):
    r"""Exponentially damped kink oscillation.

    .. math::
        \xi(t) = A \exp\!\left(-\frac{t}{\tau_d}\right)
                 \sin\!\left(\frac{2\pi t}{P} + \varphi\right) + c_0 + c_1 t

    The damping time relates to the density contrast ratio
    :math:`\zeta = \rho_0/\rho_e` and inhomogeneous layer width
    :math:`\epsilon = l/R` via
    :math:`\tau_d/P = 4/(\pi^2 \epsilon \kappa)` where
    :math:`\kappa = (\zeta - 1)/(\zeta + 1)` [2].

    Parameters
    ----------
    t : array_like
        Time (s).
    A : float
        Initial displacement amplitude (km).
    P : float
        Period (s).
    phi : float
        Phase (rad).
    tau : float
        Exponential damping time (s). Must be > 0.
    c0 : float
        Constant offset (km).
    c1 : float
        Linear trend slope (km/s).
    """
    exponent = np.clip(-t / tau, -700, 700)
    return A * np.exp(exponent) * np.sin(2.0 * np.pi * t / P + phi) + c0 + c1 * t


def gaussian(t, A, P, phi, tau_g, c0, c1):
    r"""Gaussian-damped kink oscillation (resonant absorption onset).

    .. math::
        \xi(t) = A \exp\!\left(-\frac{t^2}{2\tau_g^2}\right)
                 \sin\!\left(\frac{2\pi t}{P} + \varphi\right) + c_0 + c_1 t

    Valid for the initial phase of resonant absorption where
    :math:`\tau_g/P = 2/(\kappa \sqrt{\pi \epsilon})` for a sinusoidal
    density profile [3], with :math:`\kappa = (\zeta-1)/(\zeta+1)`
    and :math:`\epsilon = l/R` the inhomogeneous layer width.

    Parameters
    ----------
    t : array_like
        Time (s).
    A : float
        Initial displacement amplitude (km).
    P : float
        Period (s).
    phi : float
        Phase (rad).
    tau_g : float
        Gaussian damping time (s). Must be > 0.
    c0 : float
        Constant offset (km).
    c1 : float
        Linear trend slope (km/s).
    """
    exponent = np.clip(-t ** 2 / (2.0 * tau_g ** 2), -700, 700)
    return (
        A * np.exp(exponent)
        * np.sin(2.0 * np.pi * t / P + phi)
        + c0 + c1 * t
    )


def two_regime(t, A, P, phi, tau_g, tau_d, c0, c1):
    r"""Two-regime damping: Gaussian onset transitioning to exponential.

    .. math::
        \xi(t) = \begin{cases}
            A \exp(-t^2/2\tau_g^2)\,\sin(2\pi t/P + \varphi) + c_0 + c_1 t
                & t \le t_s \\
            A_s \exp(-(t-t_s)/\tau_d)\,\sin(2\pi t/P + \varphi) + c_0 + c_1 t
                & t > t_s
        \end{cases}

    The switch time is :math:`t_s = \tau_g^2/\tau_d` and the amplitude at
    the switch is :math:`A_s = A\,\exp(-t_s^2/2\tau_g^2)` [4].

    Parameters
    ----------
    t : array_like
        Time (s).
    A : float
        Initial displacement amplitude (km).
    P : float
        Period (s).
    phi : float
        Phase (rad).
    tau_g : float
        Gaussian damping time for initial phase (s).
    tau_d : float
        Exponential damping time for later phase (s).
    c0 : float
        Constant offset (km).
    c1 : float
        Linear trend slope (km/s).
    """
    t = np.asarray(t, dtype=np.float64)
    if tau_d <= 0 or tau_g <= 0:
        raise ValueError(f"tau_g and tau_d must be > 0, got tau_g={tau_g}, tau_d={tau_d}")
    t_switch = tau_g ** 2 / tau_d
    A_switch = A * np.exp(-np.clip(t_switch ** 2 / (2.0 * tau_g ** 2), 0, 700))

    sinusoid = np.sin(2.0 * np.pi * t / P + phi)
    trend = c0 + c1 * t

    gauss_exp = np.clip(-t ** 2 / (2.0 * tau_g ** 2), -700, 0)
    exp_exp = np.clip(-(t - t_switch) / tau_d, -700, 700)

    envelope = np.where(
        t <= t_switch,
        A * np.exp(gauss_exp),
        A_switch * np.exp(exp_exp),
    )
    return envelope * sinusoid + trend


def period_varying(t, A, P0, phi, tau_g, alpha, c0, c1):
    r"""Period-varying oscillation with Gaussian envelope.

    The instantaneous period is :math:`P(t) = P_0 (1 + \alpha t)`. The phase
    is the integral of the instantaneous angular frequency
    :math:`\omega(t) = 2\pi/P(t)`:

    .. math::
        \Phi(t) = \int_0^t \frac{2\pi}{P_0(1+\alpha t')}\,dt'
                = \frac{2\pi}{P_0 \alpha} \ln(1 + \alpha t)

    giving:

    .. math::
        \xi(t) = A \exp\!\left(-\frac{t^2}{2\tau_g^2}\right)
                 \sin\!\left(\Phi(t) + \varphi\right) + c_0 + c_1 t

    Models oscillations where coronal rain modifies the loop mass
    and hence the period over time [5]. For :math:`\alpha = 0` the
    phase reduces to :math:`2\pi t/P_0`.

    Parameters
    ----------
    t : array_like
        Time (s).
    A : float
        Initial displacement amplitude (km).
    P0 : float
        Initial period (s). Must be > 0.
    phi : float
        Phase (rad).
    tau_g : float
        Gaussian envelope time (s). Use large value for negligible damping.
    alpha : float
        Period drift rate (s⁻¹). Constrained to ``|alpha * t_max| < 1``
        to keep the instantaneous period positive.
    c0 : float
        Constant offset (km).
    c1 : float
        Linear trend slope (km/s).
    """
    t = np.asarray(t, dtype=np.float64)
    arg = 1.0 + alpha * t
    arg = np.clip(arg, 0.1, None)  # keep instantaneous period > 0.1 P0

    if abs(alpha) < 1e-15:
        phase = 2.0 * np.pi * t / P0 + phi
    else:
        phase = (2.0 * np.pi / (P0 * alpha)) * np.log(arg) + phi

    envelope = A * np.exp(-np.clip(t ** 2 / (2.0 * tau_g ** 2), 0, 700))
    return envelope * np.sin(phase) + c0 + c1 * t


def growing(t, A, P, phi, tau, c0, c1):
    r"""Growing (negatively damped) kink oscillation.

    .. math::
        \xi(t) = A \exp\!\left(+\frac{t}{\tau}\right)
                 \sin\!\left(\frac{2\pi t}{P} + \varphi\right) + c_0 + c_1 t

    Observed in some loops where energy injection exceeds dissipation.

    Parameters
    ----------
    t : array_like
        Time (s).
    A : float
        Initial displacement amplitude (km).
    P : float
        Period (s).
    phi : float
        Phase (rad).
    tau : float
        Growth time (s). Must be > 0.
    c0 : float
        Constant offset (km).
    c1 : float
        Linear trend slope (km/s).
    """
    exponent = np.clip(t / tau, -700, 700)
    return A * np.exp(exponent) * np.sin(2.0 * np.pi * t / P + phi) + c0 + c1 * t


MODELS = {
    "decayless": {
        "func": decayless,
        "param_names": ["A", "P", "phi", "c0", "c1"],
        "n_params": 5,
        "description": "Decayless sinusoid + linear trend",
    },
    "exponential": {
        "func": exponential,
        "param_names": ["A", "P", "phi", "tau", "c0", "c1"],
        "n_params": 6,
        "description": "Exponentially damped kink oscillation",
    },
    "gaussian": {
        "func": gaussian,
        "param_names": ["A", "P", "phi", "tau_g", "c0", "c1"],
        "n_params": 6,
        "description": "Gaussian-damped kink oscillation (resonant absorption onset)",
    },
    "two_regime": {
        "func": two_regime,
        "param_names": ["A", "P", "phi", "tau_g", "tau_d", "c0", "c1"],
        "n_params": 7,
        "description": "Two-regime damping: Gaussian → exponential transition",
    },
    "period_varying": {
        "func": period_varying,
        "param_names": ["A", "P0", "phi", "tau_g", "alpha", "c0", "c1"],
        "n_params": 7,
        "description": "Period-varying oscillation with Gaussian envelope",
    },
    "growing": {
        "func": growing,
        "param_names": ["A", "P", "phi", "tau", "c0", "c1"],
        "n_params": 6,
        "description": "Growing (negatively damped) kink oscillation",
    },
}
