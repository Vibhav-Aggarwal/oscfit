"""Oscillation fitting with MCMC uncertainty estimation.

Provides least-squares initial fit via scipy.optimize.curve_fit,
followed by emcee ensemble MCMC sampling for robust posterior
estimation. Supports all 6 models from models.py.

References
----------
.. [1] Foreman-Mackey et al. 2013, PASP 125, 306 (emcee)
.. [2] Anfinogentov et al. 2021, ApJS 252, 11 (SoBAT methodology)
"""

import numpy as np
from scipy.optimize import curve_fit
from oscfit.models import MODELS
from oscfit.detect import detect_periods


def fit_oscillation(time, signal, sigma=None, model="decayless",
                    period_hint=None, mcmc=True, n_walkers=32,
                    n_steps=3000, n_burn=1000, progress=False, seed=None):
    """Fit an oscillation model to a displacement time series.

    Parameters
    ----------
    time : array_like
        Time array (s). Must be monotonically increasing.
    signal : array_like
        Displacement signal (km or pixels). NaNs filtered out.
    sigma : array_like or float, optional
        Measurement uncertainties. If None, estimated from fit residuals.
    model : str
        Model name from ``oscfit.models.MODELS``.
    period_hint : float, optional
        Approximate period (s). If None, auto-detected via FFT.
    mcmc : bool
        Run MCMC after least-squares for uncertainty estimation.
    n_walkers : int
        Number of emcee walkers (must be even, >= 2 * n_params).
    n_steps : int
        MCMC steps per walker.
    n_burn : int
        Burn-in steps to discard.
    progress : bool
        Show emcee progress bar.
    seed : int, optional
        Random seed for MCMC. None = use system entropy (non-reproducible).

    Returns
    -------
    OscFitResult

    Raises
    ------
    ValueError
        If model unknown, time not monotonic, insufficient data, or
        signal is constant.
    """
    if model not in MODELS:
        raise ValueError(
            f"Unknown model '{model}'. Available: {list(MODELS.keys())}"
        )

    time = np.asarray(time, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)

    if time.shape != signal.shape:
        raise ValueError(
            f"time and signal must have same shape, got {time.shape} vs {signal.shape}"
        )

    finite = np.isfinite(signal) & np.isfinite(time)
    n_params = MODELS[model]["n_params"]

    if np.sum(finite) < n_params + 2:
        raise ValueError(
            f"Need at least {n_params + 2} finite samples for model '{model}', "
            f"got {int(np.sum(finite))}"
        )

    t = time[finite] - time[finite][0]
    s = signal[finite]

    if not np.all(np.diff(time[finite]) > 0):
        raise ValueError("time array must be strictly monotonically increasing")

    signal_range = float(np.max(s) - np.min(s))
    if signal_range < 1e-12 * max(abs(float(np.mean(s))), 1.0):
        raise ValueError(
            "Signal is constant or near-constant; no oscillation to fit"
        )

    if sigma is not None:
        sigma = np.asarray(sigma, dtype=np.float64)
        if sigma.ndim == 0:
            sig = np.full_like(s, float(sigma))
        else:
            sig = sigma[finite]
    else:
        sig = None

    model_info = MODELS[model]
    func = model_info["func"]
    param_names = model_info["param_names"]
    n_params = model_info["n_params"]

    if period_hint is None:
        det = detect_periods(t, s, method="fft")
        if det.dominant_period is not None:
            period_hint = det.dominant_period
        else:
            period_hint = (t[-1] - t[0]) / 4.0

    p0, bounds = _initial_guess(model, t, s, period_hint)

    try:
        popt, pcov = curve_fit(
            func, t, s, p0=p0, bounds=bounds, sigma=sig,
            absolute_sigma=(sig is not None), maxfev=10000,
        )
    except RuntimeError:
        popt = np.array(p0)
        pcov = np.eye(n_params) * np.inf

    residuals = s - func(t, *popt)
    if sig is None:
        noise_est = max(
            float(np.std(residuals)),
            float(np.std(s)) * 1e-3,
            1e-12,
        )
        sig = np.full_like(s, noise_est)
    else:
        sig = np.clip(sig, 1e-12, None)

    result = OscFitResult(
        model_name=model,
        param_names=param_names,
        params={k: v for k, v in zip(param_names, popt)},
        param_errors={
            k: (np.sqrt(pcov[i, i])
                if np.isfinite(pcov[i, i]) and pcov[i, i] >= 0
                else np.inf)
            for i, k in enumerate(param_names)
        },
        model_func=func,
        time=time[finite],
        t_norm=t,
        signal=s,
        sigma=sig,
        residuals=residuals,
    )

    if mcmc:
        n_walkers = max(n_walkers, 2 * n_params + 2)
        if n_walkers % 2 != 0:
            n_walkers += 1
        _run_mcmc(result, t, s, sig, model, n_walkers, n_steps, n_burn,
                  progress, seed)

    return result


class OscFitResult:
    """Container for oscillation fit results with posterior samples."""

    def __init__(self, model_name, param_names, params, param_errors,
                 model_func, time, t_norm, signal, sigma, residuals):
        self.model_name = model_name
        self.param_names = param_names
        self.params = params
        self.param_errors = param_errors
        self.model_func = model_func
        self.time = time
        self.t_norm = t_norm
        self.signal = signal
        self.sigma = sigma
        self.residuals = residuals
        self.samples = None
        self.sampler = None
        self.log_evidence = None

    @property
    def period(self):
        key = "P" if "P" in self.params else "P0"
        return self.params[key], self.param_errors[key]

    @property
    def amplitude(self):
        return self.params["A"], self.param_errors["A"]

    @property
    def velocity_amplitude(self):
        A, dA = self.amplitude
        P, dP = self.period
        if A <= 0 or P <= 0:
            return 0.0, 0.0
        V = 2.0 * np.pi * A / P
        dV = V * np.sqrt((dA / A) ** 2 + (dP / P) ** 2)
        return V, dV

    def model_curve(self, t=None):
        if t is None:
            t = self.t_norm
        pvals = [self.params[k] for k in self.param_names]
        return self.model_func(t, *pvals)

    def reduced_chi2(self):
        ndof = len(self.signal) - len(self.param_names)
        if ndof <= 0:
            return np.inf
        return np.sum((self.residuals / self.sigma) ** 2) / ndof

    def bic(self):
        """Bayesian Information Criterion."""
        n = len(self.signal)
        k = len(self.param_names)
        chi2 = np.sum((self.residuals / self.sigma) ** 2)
        return chi2 + k * np.log(n)

    def aic(self):
        """Akaike Information Criterion."""
        k = len(self.param_names)
        chi2 = np.sum((self.residuals / self.sigma) ** 2)
        return chi2 + 2 * k

    def percentiles(self, param, q=(16, 50, 84)):
        """Get posterior percentiles for a parameter from MCMC samples."""
        if self.samples is None:
            return None
        idx = self.param_names.index(param)
        return np.percentile(self.samples[:, idx], q)

    def to_dict(self):
        """Export parameters in Shrivastav et al. (2024) format."""
        A, dA = self.amplitude
        P, dP = self.period
        V, dV = self.velocity_amplitude
        return {
            "P (s)": f"{P:.0f} ± {dP:.0f}",
            "A (km)": f"{A:.0f} ± {dA:.0f}",
            "V (km/s)": f"{V:.1f} ± {dV:.1f}",
            "chi2_red": f"{self.reduced_chi2():.2f}",
            "model": self.model_name,
        }

    def convergence_check(self):
        """Check MCMC convergence via autocorrelation time."""
        if self.sampler is None:
            return {"converged": False, "reason": "No MCMC run"}
        try:
            tau = self.sampler.get_autocorr_time(quiet=True)
            n_eff = self.samples.shape[0] / np.max(tau)
            converged = n_eff > 50
            return {
                "converged": converged,
                "autocorr_time": tau,
                "n_effective": n_eff,
                "n_samples": self.samples.shape[0],
            }
        except Exception:
            return {"converged": False, "reason": "Could not estimate autocorrelation"}

    def summary(self):
        lines = [f"Model: {self.model_name}"]
        lines.append(f"  {'param':>8s}  {'value':>12s}  {'error':>12s}")
        lines.append(f"  {'─' * 8}  {'─' * 12}  {'─' * 12}")
        for name in self.param_names:
            val = self.params[name]
            err = self.param_errors[name]
            lines.append(f"  {name:>8s}  {val:>12.4f}  {err:>12.4f}")
        V, dV = self.velocity_amplitude
        lines.append(f"  {'V_amp':>8s}  {V:>12.4f}  {dV:>12.4f}  km/s")
        lines.append(f"  χ²_red = {self.reduced_chi2():.3f}")
        lines.append(f"  BIC = {self.bic():.1f}, AIC = {self.aic():.1f}")
        if self.samples is not None:
            conv = self.convergence_check()
            lines.append(f"  MCMC converged: {conv.get('converged', 'unknown')}")
        return "\n".join(lines)

    def __repr__(self):
        A, dA = self.amplitude
        P, dP = self.period
        return (f"OscFitResult(model={self.model_name}, "
                f"P={P:.1f}±{dP:.1f}s, A={A:.1f}±{dA:.1f}km, "
                f"χ²={self.reduced_chi2():.2f})")


def _initial_guess(model, t, s, P_hint):
    """Generate initial parameter guesses and bounds for each model."""
    A_hint = (np.max(s) - np.min(s)) / 2.0
    A_hint = max(A_hint, float(np.std(s)), 1e-6)
    c0_hint = np.mean(s)
    duration = max(t[-1] - t[0], 1.0)
    c1_hint = (s[-1] - s[0]) / duration

    A_max = max(A_hint * 10.0, A_hint + 1.0)
    P_lo, P_hi = max(P_hint * 0.3, 1e-6), max(P_hint * 3.0, P_hint * 0.3 + 1.0)

    if model == "decayless":
        p0 = [A_hint, P_hint, 0.0, c0_hint, c1_hint]
        bounds = ([0, P_lo, -np.pi, -np.inf, -np.inf],
                  [A_max, P_hi, np.pi, np.inf, np.inf])

    elif model == "exponential":
        tau_hint = duration
        p0 = [A_hint, P_hint, 0.0, tau_hint, c0_hint, c1_hint]
        bounds = ([0, P_lo, -np.pi, P_hint * 0.5, -np.inf, -np.inf],
                  [A_max, P_hi, np.pi, duration * 10, np.inf, np.inf])

    elif model == "gaussian":
        tau_g_hint = duration * 0.5
        p0 = [A_hint, P_hint, 0.0, tau_g_hint, c0_hint, c1_hint]
        bounds = ([0, P_lo, -np.pi, P_hint * 0.5, -np.inf, -np.inf],
                  [A_max, P_hi, np.pi, duration * 5, np.inf, np.inf])

    elif model == "two_regime":
        tau_g_hint = duration * 0.3
        tau_d_hint = duration * 0.8
        p0 = [A_hint, P_hint, 0.0, tau_g_hint, tau_d_hint, c0_hint, c1_hint]
        bounds = ([0, P_lo, -np.pi, P_hint, P_hint, -np.inf, -np.inf],
                  [A_max, P_hi, np.pi, duration * 3, duration * 10, np.inf, np.inf])

    elif model == "period_varying":
        tau_g_hint = duration * 2
        p0 = [A_hint, P_hint, 0.0, tau_g_hint, 0.0, c0_hint, c1_hint]
        bounds = ([0, P_lo, -np.pi, P_hint, -0.01, -np.inf, -np.inf],
                  [A_max, P_hi, np.pi, duration * 10, 0.01, np.inf, np.inf])

    elif model == "growing":
        tau_hint = duration * 2
        p0 = [A_hint, P_hint, 0.0, tau_hint, c0_hint, c1_hint]
        bounds = ([0, P_lo, -np.pi, P_hint * 0.5, -np.inf, -np.inf],
                  [A_max, P_hi, np.pi, duration * 10, np.inf, np.inf])
    else:
        raise ValueError(f"Unknown model: {model}")

    return p0, bounds


def _run_mcmc(result, t, s, sigma, model, n_walkers, n_steps, n_burn, progress, seed):
    """Run emcee MCMC for posterior sampling."""
    import emcee

    model_info = MODELS[model]
    func = model_info["func"]
    param_names = model_info["param_names"]
    ndim = len(param_names)

    p0_values = np.array([result.params[k] for k in param_names])
    period_key = "P" if "P" in result.params else "P0"
    _, bounds = _initial_guess(model, t, s, result.params[period_key])

    lo = np.array(bounds[0])
    hi = np.array(bounds[1])

    def log_prior(theta):
        for i, (val, l, h) in enumerate(zip(theta, lo, hi)):
            if not (l <= val <= h):
                return -np.inf
        return 0.0

    def log_likelihood(theta):
        try:
            model_vals = func(t, *theta)
        except (ValueError, RuntimeError):
            return -np.inf
        if not np.all(np.isfinite(model_vals)):
            return -np.inf
        return -0.5 * np.sum(((s - model_vals) / sigma) ** 2 + np.log(sigma ** 2))

    def log_probability(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    rng = np.random.default_rng(seed)
    scatter = np.abs(p0_values) * 0.005 + 1e-8
    pos = p0_values + scatter * rng.standard_normal((n_walkers, ndim))

    for i in range(ndim):
        pos[:, i] = np.clip(pos[:, i], lo[i] + 1e-10, hi[i] - 1e-10)

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_probability)
    sampler.run_mcmc(pos, n_steps, progress=progress)

    samples = sampler.get_chain(discard=n_burn, flat=True)

    result.samples = samples
    result.sampler = sampler

    for i, name in enumerate(param_names):
        q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
        result.params[name] = q50
        result.param_errors[name] = (q84 - q16) / 2.0

    popt_mcmc = [result.params[k] for k in param_names]
    result.residuals = s - func(t, *popt_mcmc)
