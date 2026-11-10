"""Period detection via FFT, wavelet, and Lomb-Scargle analysis.

Implements spectral analysis methods following Ireland et al. (2015)
guidelines: no pre-detrending before FFT, proper windowing, power-law
noise background fitting with Vaughan (2005) debiasing, and
Bonferroni-corrected significance testing.

References
----------
.. [1] Ireland et al. 2015, SoPh 290, 613
.. [2] Vaughan 2005, A&A 431, 391
.. [3] Torrence & Compo 1998, BAMS 79, 61
"""

import numpy as np
from scipy.signal import lombscargle
from scipy.optimize import curve_fit


def detect_periods(time, signal, method="fft", min_period=None, max_period=None,
                   confidence=0.99, detrend=False):
    """Detect significant oscillation periods in a time series.

    Parameters
    ----------
    time : array_like
        Time array (s).
    signal : array_like
        Displacement signal. NaNs are handled gracefully.
    method : str
        ``"fft"`` (uniform sampling), ``"lomb"`` (handles gaps),
        ``"wavelet"`` (time-frequency), or ``"all"`` (run all three).
    min_period : float, optional
        Minimum period to search (s). Default: 2 × cadence.
    max_period : float, optional
        Maximum period to search (s). Default: duration / 2.
    confidence : float
        Confidence level for significance testing (0-1).
    detrend : bool
        If True, subtract linear trend before analysis. Default False
        per Ireland et al. (2015) recommendation.

    Returns
    -------
    DetectionResult
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if method not in ("fft", "lomb", "wavelet", "all"):
        raise ValueError(f"method must be 'fft', 'lomb', 'wavelet', or 'all', got '{method}'")

    time = np.asarray(time, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)

    if time.shape != signal.shape:
        raise ValueError(
            f"time and signal must have same shape, got {time.shape} vs {signal.shape}"
        )

    finite = np.isfinite(signal) & np.isfinite(time)
    t_clean = time[finite]
    s_clean = signal[finite]

    if len(s_clean) < 10:
        return DetectionResult.empty(method)

    if not np.all(np.diff(t_clean) > 0):
        raise ValueError("time array must be strictly monotonically increasing")

    dt = np.median(np.diff(t_clean))
    duration = t_clean[-1] - t_clean[0]

    if dt <= 0 or duration <= 0:
        return DetectionResult.empty(method)

    if min_period is None:
        min_period = 2.0 * dt
    if max_period is None:
        max_period = duration / 2.0

    if min_period <= 0 or max_period <= min_period:
        raise ValueError(
            f"Invalid period range: min={min_period}, max={max_period}"
        )

    if detrend:
        s_clean = _detrend_linear(t_clean, s_clean)

    results = {}
    if method in ("fft", "all"):
        results["fft"] = _fft_analysis(t_clean, s_clean, dt, min_period, max_period, confidence)
    if method in ("lomb", "all"):
        results["lomb"] = _lomb_analysis(t_clean, s_clean, min_period, max_period, confidence)
    if method in ("wavelet", "all"):
        results["wavelet"] = _wavelet_analysis(t_clean, s_clean, dt, min_period, max_period, confidence)

    if method == "all":
        return _merge_results(results, confidence)
    return results[method]


class DetectionResult:
    """Container for period detection results."""

    __slots__ = (
        "periods", "powers", "significances", "frequencies",
        "power_spectrum", "noise_model", "confidence_level",
        "method",
    )

    def __init__(self, periods, powers, significances, frequencies,
                 power_spectrum, noise_model, confidence_level, method):
        self.periods = np.asarray(periods)
        self.powers = np.asarray(powers)
        self.significances = np.asarray(significances)
        self.frequencies = frequencies
        self.power_spectrum = power_spectrum
        self.noise_model = noise_model
        self.confidence_level = confidence_level
        self.method = method

    @property
    def dominant_period(self):
        if len(self.periods) == 0:
            return None
        return self.periods[0]

    @classmethod
    def empty(cls, method="fft"):
        return cls(
            periods=np.array([]),
            powers=np.array([]),
            significances=np.array([]),
            frequencies=np.array([]),
            power_spectrum=np.array([]),
            noise_model=None,
            confidence_level=0,
            method=method,
        )

    def __repr__(self):
        if len(self.periods) == 0:
            return f"DetectionResult(no significant periods, method={self.method})"
        p = self.periods[0]
        return f"DetectionResult(dominant_period={p:.1f}s, n_peaks={len(self.periods)}, method={self.method})"


def _detrend_linear(t, s):
    coeffs = np.polyfit(t - t[0], s, 1)
    return s - np.polyval(coeffs, t - t[0])


def _split_cosine_bell(N, p=0.4):
    """Split cosine bell window (Auto-NUWT standard, p=0.4)."""
    w = np.ones(N)
    n_taper = int(N * p / 2)
    if n_taper < 1:
        return w
    taper = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_taper) / n_taper))
    w[:n_taper] = taper
    w[-n_taper:] = taper[::-1]
    return w


def _fft_analysis(t, signal, dt, min_period, max_period, confidence):
    """FFT with split cosine bell windowing and power-law noise fitting."""
    N = len(signal)
    window = _split_cosine_bell(N)
    window_correction = np.sum(window ** 2) / N

    signal_w = (signal - np.mean(signal)) * window
    fft_vals = np.fft.rfft(signal_w)
    freqs = np.fft.rfftfreq(N, d=dt)
    power = (2.0 / (N * window_correction)) * np.abs(fft_vals) ** 2

    valid = freqs > 0
    freqs = freqs[valid]
    power = power[valid]

    freq_mask = (1.0 / freqs >= min_period) & (1.0 / freqs <= max_period)
    if not np.any(freq_mask):
        return DetectionResult.empty("fft")

    noise_params = _fit_noise_background(freqs, power)
    noise_level = _power_law_noise(freqs, *noise_params)

    n_independent = np.sum(freq_mask)
    chi2_threshold = -np.log(1.0 - confidence ** (1.0 / n_independent)) * 2.0
    significance_threshold = noise_level * chi2_threshold

    peaks = _find_spectral_peaks(freqs, power, freq_mask, significance_threshold)

    periods = 1.0 / freqs[peaks] if len(peaks) > 0 else np.array([])
    peak_powers = power[peaks] if len(peaks) > 0 else np.array([])
    sigs = (peak_powers / noise_level[peaks]) if len(peaks) > 0 else np.array([])

    order = np.argsort(peak_powers)[::-1]

    return DetectionResult(
        periods=periods[order],
        powers=peak_powers[order],
        significances=sigs[order],
        frequencies=freqs,
        power_spectrum=power,
        noise_model=noise_params,
        confidence_level=confidence,
        method="fft",
    )


def _lomb_analysis(t, signal, min_period, max_period, confidence):
    """Lomb-Scargle periodogram for unevenly sampled / gapped data."""
    t_norm = t - t[0]
    s_norm = signal - np.mean(signal)

    freq_min = 1.0 / max_period
    freq_max = 1.0 / min_period
    n_freq = max(500, int(10 * (freq_max - freq_min) * (t_norm[-1] - t_norm[0])))
    freqs = np.linspace(freq_min, freq_max, n_freq)
    angular_freqs = 2.0 * np.pi * freqs

    power = lombscargle(t_norm, s_norm, angular_freqs, normalize=True)

    noise_params = _fit_noise_background(freqs, power)
    noise_level = _power_law_noise(freqs, *noise_params)

    n_independent = len(freqs)
    chi2_threshold = -np.log(1.0 - confidence ** (1.0 / n_independent)) * 2.0
    significance_threshold = noise_level * chi2_threshold

    freq_mask = np.ones(len(freqs), dtype=bool)
    peaks = _find_spectral_peaks(freqs, power, freq_mask, significance_threshold)

    periods = 1.0 / freqs[peaks] if len(peaks) > 0 else np.array([])
    peak_powers = power[peaks] if len(peaks) > 0 else np.array([])
    sigs = (peak_powers / noise_level[peaks]) if len(peaks) > 0 else np.array([])

    order = np.argsort(peak_powers)[::-1]

    return DetectionResult(
        periods=periods[order],
        powers=peak_powers[order],
        significances=sigs[order],
        frequencies=freqs,
        power_spectrum=power,
        noise_model=noise_params,
        confidence_level=confidence,
        method="lomb",
    )


def _wavelet_analysis(t, signal, dt, min_period, max_period, confidence):
    """Morlet wavelet analysis following Torrence & Compo (1998)."""
    N = len(signal)
    s_norm = signal - np.mean(signal)

    omega0 = 6.0
    n_scales = min(200, N)
    periods_test = np.geomspace(min_period, max_period, n_scales)
    scales = periods_test * (omega0 + np.sqrt(2 + omega0 ** 2)) / (4.0 * np.pi)

    s_fft = np.fft.fft(s_norm)
    k = np.arange(N)
    k = np.where(k <= N // 2, k, k - N)
    angular_freq = 2.0 * np.pi * k / (N * dt)

    global_power = np.zeros(n_scales)
    for i, scale in enumerate(scales):
        norm = (2.0 * np.pi * scale) ** 0.5 * (np.pi ** -0.25)
        psi_hat = norm * np.exp(-0.5 * (scale * angular_freq - omega0) ** 2)
        psi_hat[0] = 0
        psi_hat[angular_freq < 0] = 0
        wavelet_coeffs = np.fft.ifft(s_fft * np.conj(psi_hat))
        power_2d = np.abs(wavelet_coeffs) ** 2 / scale
        cone_mask = np.ones(N, dtype=bool)
        edge = int(np.sqrt(2) * scale)
        if edge > 0 and edge < N // 2:
            cone_mask[:edge] = False
            cone_mask[-edge:] = False
        global_power[i] = np.mean(power_2d[cone_mask]) if np.any(cone_mask) else np.mean(power_2d)

    freqs_test = 1.0 / periods_test
    noise_params = _fit_noise_background(freqs_test[::-1], global_power[::-1])
    noise_level = _power_law_noise(freqs_test, *noise_params)

    n_independent = n_scales
    chi2_threshold = -np.log(1.0 - confidence ** (1.0 / n_independent)) * 2.0
    significance_threshold = noise_level * chi2_threshold

    freq_mask = np.ones(n_scales, dtype=bool)
    peaks = _find_spectral_peaks(freqs_test, global_power, freq_mask, significance_threshold)

    periods = periods_test[peaks] if len(peaks) > 0 else np.array([])
    peak_powers = global_power[peaks] if len(peaks) > 0 else np.array([])
    sigs = (peak_powers / noise_level[peaks]) if len(peaks) > 0 else np.array([])

    order = np.argsort(peak_powers)[::-1]

    return DetectionResult(
        periods=periods[order],
        powers=peak_powers[order],
        significances=sigs[order],
        frequencies=freqs_test,
        power_spectrum=global_power,
        noise_model=noise_params,
        confidence_level=confidence,
        method="wavelet",
    )


def _power_law_noise(freq, N_c, alpha, N_w):
    """Power-law + white noise background model.

    S(f) = N_c * f^(-alpha) + N_w
    """
    return N_c * freq ** (-alpha) + N_w


def _fit_noise_background(freqs, power):
    """Fit power-law noise background with Vaughan (2005) debiasing.

    The log-periodogram is biased LOW by the Euler-Mascheroni constant
    (γ ≈ 0.5772) in natural log, which equals 0.25068 in log10:
        E[log10 I(f)] = log10 S(f) - 0.25068
    To debias, ADD this constant to log10(power) before fitting [Vaughan 2005].
    """
    mask = freqs > 0
    f = freqs[mask]
    p = power[mask]

    positive = p > 0
    if np.sum(positive) < 5:
        return (1.0, 1.0, np.median(power))

    VAUGHAN_BIAS_LOG10 = 0.25068  # = γ / ln(10) where γ is Euler-Mascheroni
    log_f = np.log10(f[positive])
    log_p = np.log10(p[positive]) + VAUGHAN_BIAS_LOG10

    try:
        coeffs = np.polyfit(log_f, log_p, 1)
        alpha_init = -coeffs[0]
        Nc_init = 10 ** coeffs[1]
    except (np.linalg.LinAlgError, ValueError):
        alpha_init = 1.0
        Nc_init = np.median(p)

    Nw_init = np.median(p[-len(p) // 4:])

    try:
        popt, _ = curve_fit(
            _power_law_noise, f, p,
            p0=[Nc_init, alpha_init, Nw_init],
            bounds=([0, 0, 0], [np.inf, 10.0, np.inf]),
            maxfev=5000,
        )
        return tuple(popt)
    except (RuntimeError, ValueError):
        return (Nc_init, max(alpha_init, 0.1), max(Nw_init, 1e-10))


def _find_spectral_peaks(freqs, power, freq_mask, threshold):
    """Find peaks above significance threshold with local maximum check."""
    candidates = np.where(freq_mask & (power > threshold))[0]
    if len(candidates) == 0:
        if np.any(freq_mask):
            return np.array([np.argmax(power * freq_mask)])
        return np.array([], dtype=int)

    peaks = []
    for idx in candidates:
        left = power[idx - 1] if idx > 0 else 0
        right = power[idx + 1] if idx < len(power) - 1 else 0
        if power[idx] >= left and power[idx] >= right:
            peaks.append(idx)

    return np.array(peaks, dtype=int) if peaks else np.array([candidates[np.argmax(power[candidates])]])


def _merge_results(results, confidence):
    """Merge detections from multiple methods, preferring confirmed periods."""
    all_periods = []
    all_powers = []
    all_sigs = []

    for method, r in results.items():
        for p, pw, s in zip(r.periods, r.powers, r.significances):
            matched = False
            for i, existing_p in enumerate(all_periods):
                if existing_p > 0 and abs(p - existing_p) / existing_p < 0.15:
                    all_powers[i] += pw
                    all_sigs[i] += s
                    matched = True
                    break
            if not matched:
                all_periods.append(p)
                all_powers.append(pw)
                all_sigs.append(s)

    if not all_periods:
        return DetectionResult.empty("all")

    order = np.argsort(all_powers)[::-1]
    first = list(results.values())[0]

    return DetectionResult(
        periods=np.array(all_periods)[order],
        powers=np.array(all_powers)[order],
        significances=np.array(all_sigs)[order],
        frequencies=first.frequencies,
        power_spectrum=first.power_spectrum,
        noise_model=first.noise_model,
        confidence_level=confidence,
        method="all",
    )
