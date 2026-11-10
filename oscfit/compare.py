"""Bayesian model comparison via nested sampling.

Computes Bayesian evidence for each oscillation model using dynesty,
then ranks models by Bayes factor. This determines whether the data
prefer a decayless, damped, Gaussian-damped, or more complex model.

References
----------
.. [1] Speagle 2020, MNRAS 493, 3132 (dynesty)
.. [2] Pascoe et al. 2017, A&A 600, A78 (model comparison for kink oscillations)
"""

import numpy as np
from oscfit.models import MODELS


def compare_models(time, signal, sigma=None, models_to_compare=None,
                   n_live=200, dlogz=0.5):
    """Compare oscillation models via Bayesian evidence.

    Parameters
    ----------
    time : array_like
        Time array (s).
    signal : array_like
        Displacement signal.
    sigma : array_like or float, optional
        Measurement uncertainties. If None, estimated from data.
    models_to_compare : list of str, optional
        Model names to compare. Default: decayless, exponential, gaussian.
    n_live : int
        Number of live points for nested sampling.
    dlogz : float
        Stopping criterion (evidence tolerance).

    Returns
    -------
    ComparisonResult
    """
    import dynesty

    time = np.asarray(time, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)

    finite = np.isfinite(signal)
    t = time[finite] - time[finite][0]
    s = signal[finite]

    if sigma is None:
        from oscfit.fit import fit_oscillation
        quick = fit_oscillation(t + time[finite][0], s, model="decayless", mcmc=False)
        sigma = max(float(np.std(quick.residuals)), float(np.std(s)) * 1e-3, 1e-12)

    if np.isscalar(sigma):
        sigma_val = max(float(sigma), 1e-12)
        sig = np.full_like(s, sigma_val)
    else:
        sig = np.asarray(sigma, dtype=np.float64)[finite]
        sig = np.clip(sig, 1e-12, None)

    if models_to_compare is None:
        models_to_compare = ["decayless", "exponential", "gaussian"]

    from oscfit.detect import detect_periods
    det = detect_periods(t, s, method="fft")
    P_hint = det.dominant_period if det.dominant_period else (t[-1] - t[0]) / 4

    results = {}
    for model_name in models_to_compare:
        info = MODELS[model_name]
        ndim = info["n_params"]

        bounds = _prior_bounds(model_name, t, s, P_hint)

        def prior_transform(u, _bounds=bounds):
            theta = np.empty(len(u))
            for i, (lo, hi) in enumerate(_bounds):
                theta[i] = lo + u[i] * (hi - lo)
            return theta

        log_norm = -0.5 * np.sum(np.log(2.0 * np.pi * sig ** 2))

        def log_likelihood(theta, _func=info["func"], _log_norm=log_norm):
            model_vals = _func(t, *theta)
            if not np.all(np.isfinite(model_vals)):
                return -1e300
            return -0.5 * np.sum(((s - model_vals) / sig) ** 2) + _log_norm

        sampler = dynesty.NestedSampler(
            log_likelihood, prior_transform, ndim,
            nlive=n_live, bound='multi', sample='auto',
        )
        sampler.run_nested(dlogz=dlogz, print_progress=False)
        res = sampler.results

        results[model_name] = {
            "log_evidence": res.logz[-1],
            "log_evidence_err": res.logzerr[-1],
            "samples": res.samples,
            "weights": np.exp(res.logwt - res.logz[-1]),
            "n_params": ndim,
            "dynesty_results": res,
        }

    return ComparisonResult(results, models_to_compare)


class ComparisonResult:
    """Container for model comparison results."""

    def __init__(self, results, model_names):
        self.results = results
        self.model_names = model_names

    @property
    def preferred_model(self):
        """Model with highest Bayesian evidence."""
        best = max(self.results.items(), key=lambda x: x[1]["log_evidence"])
        return best[0]

    def bayes_factor(self, model_a, model_b):
        """Compute Bayes factor K = 2 * ln(Z_a / Z_b).

        Interpretation (Kass & Raftery 1995):
        K < 2: not worth mentioning
        2 < K < 6: positive evidence for model_a
        6 < K < 10: strong evidence
        K > 10: very strong evidence
        """
        lnZ_a = self.results[model_a]["log_evidence"]
        lnZ_b = self.results[model_b]["log_evidence"]
        return 2.0 * (lnZ_a - lnZ_b)

    def ranking(self):
        """Rank models by evidence, return list of (name, log_Z, log_Z_err)."""
        items = [
            (name, r["log_evidence"], r["log_evidence_err"])
            for name, r in self.results.items()
        ]
        return sorted(items, key=lambda x: x[1], reverse=True)

    def summary(self):
        lines = ["Model Comparison (Bayesian Evidence)"]
        lines.append(f"{'Model':>20s}  {'ln(Z)':>10s}  {'±':>8s}  {'ΔK vs best':>10s}")
        lines.append(f"{'─' * 20}  {'─' * 10}  {'─' * 8}  {'─' * 10}")

        ranked = self.ranking()
        best_lnZ = ranked[0][1]
        for name, lnZ, lnZ_err in ranked:
            K = 2.0 * (best_lnZ - lnZ)
            marker = " ←" if name == self.preferred_model else ""
            lines.append(f"{name:>20s}  {lnZ:>10.1f}  {lnZ_err:>8.1f}  {K:>10.1f}{marker}")

        lines.append(f"\nPreferred: {self.preferred_model}")
        return "\n".join(lines)

    def __repr__(self):
        return f"ComparisonResult(preferred={self.preferred_model}, n_models={len(self.model_names)})"


def _prior_bounds(model, t, s, P_hint):
    """Uniform prior bounds for each parameter."""
    A_max = (np.max(s) - np.min(s)) * 3
    duration = t[-1] - t[0]
    c0_range = np.max(np.abs(s)) * 3
    c1_range = (np.max(s) - np.min(s)) / max(duration, 1) * 5

    base = [
        (0.1, A_max),
        (P_hint * 0.3, P_hint * 3),
        (-np.pi, np.pi),
    ]

    trend = [(-c0_range, c0_range), (-c1_range, c1_range)]

    if model == "decayless":
        return base + trend

    if model == "exponential":
        return base + [(P_hint * 0.5, duration * 10)] + trend

    if model == "gaussian":
        return base + [(P_hint * 0.5, duration * 5)] + trend

    if model == "two_regime":
        return base + [(P_hint, duration * 3), (P_hint, duration * 10)] + trend

    if model == "period_varying":
        return base + [(P_hint * 0.5, duration * 10), (-0.01, 0.01)] + trend

    if model == "growing":
        return base + [(P_hint * 0.5, duration * 10)] + trend

    return base + trend
