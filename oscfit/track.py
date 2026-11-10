"""Loop center tracking in time-distance maps.

Implements Gaussian sub-pixel fitting at each timestep to extract
the transverse displacement time series of a coronal loop, following
the Auto-NUWT methodology (Weberg et al. 2018, Morton et al. 2013).

References
----------
.. [1] Weberg et al. 2018, ApJ 852, 57
.. [2] Morton et al. 2013, ApJ 768, 17
"""

import numpy as np
from scipy.optimize import curve_fit


def track_loop_center(xt_map, x=None, noise_model=None, method="gaussian",
                      max_shift=None, min_amplitude=None, max_chi2=10.0,
                      min_width=0.5, max_width=None):
    """Track the transverse position of a loop in a time-distance map.

    Parameters
    ----------
    xt_map : ndarray, shape (nt, nx)
        Time-distance intensity map (DN).
    x : ndarray, shape (nx,), optional
        Spatial coordinate array. Default: pixel indices.
    noise_model : dict, optional
        Instrument noise: ``{"readout": float, "gain": float}``.
        If None, noise estimated from data.
    method : str
        ``"gaussian"`` (fit Gaussian + linear background) or
        ``"centroid"`` (intensity-weighted centroid, faster but less accurate).
    max_shift : float, optional
        Maximum allowed shift between consecutive timesteps (pixels).
        Default: nx / 4.
    min_amplitude : float, optional
        Minimum Gaussian amplitude for valid fit (DN).
        Default: 3 × estimated noise.
    max_chi2 : float
        Maximum reduced chi-squared for valid fit.
    min_width : float
        Minimum Gaussian sigma for valid fit (pixels).
    max_width : float, optional
        Maximum Gaussian sigma. Default: nx / 4.

    Returns
    -------
    TrackResult
    """
    xt_map = np.asarray(xt_map, dtype=np.float64)
    nt, nx = xt_map.shape

    if x is None:
        x = np.arange(nx, dtype=np.float64)
    else:
        x = np.asarray(x, dtype=np.float64)

    if max_shift is None:
        max_shift = nx / 4.0
    if max_width is None:
        max_width = nx / 4.0

    noise_sigma = _estimate_noise(xt_map, noise_model)

    if min_amplitude is None:
        min_amplitude = 3.0 * np.median(noise_sigma)

    if method == "centroid":
        return _track_centroid(xt_map, x, noise_sigma)

    centers = np.full(nt, np.nan)
    center_errors = np.full(nt, np.nan)
    amplitudes = np.full(nt, np.nan)
    widths = np.full(nt, np.nan)
    backgrounds = np.full(nt, np.nan)
    chi2_values = np.full(nt, np.nan)
    quality = np.zeros(nt, dtype=int)

    prev_center = float(x[np.argmax(np.mean(xt_map, axis=0))])

    for t in range(nt):
        profile = xt_map[t]
        if noise_sigma.ndim == 2:
            sigma_profile = noise_sigma[t]
        elif noise_sigma.ndim == 1:
            sigma_profile = noise_sigma[t]
        else:
            sigma_profile = noise_sigma

        result = _fit_gaussian_profile(
            x, profile, sigma_profile, prev_center,
            min_amplitude, min_width, max_width, max_chi2, max_shift
        )

        centers[t] = result["center"]
        center_errors[t] = result["center_error"]
        amplitudes[t] = result["amplitude"]
        widths[t] = result["width"]
        backgrounds[t] = result["background"]
        chi2_values[t] = result["chi2_red"]
        quality[t] = result["quality"]

        if result["quality"] >= 1:
            prev_center = result["center"]

    centers = _fill_gaps(centers, quality, max_gap_fraction=0.5)

    return TrackResult(
        centers=centers,
        center_errors=center_errors,
        amplitudes=amplitudes,
        widths=widths,
        backgrounds=backgrounds,
        chi2_values=chi2_values,
        quality=quality,
        x=x,
    )


class TrackResult:
    """Container for loop tracking results."""

    __slots__ = (
        "centers", "center_errors", "amplitudes", "widths",
        "backgrounds", "chi2_values", "quality", "x",
    )

    def __init__(self, centers, center_errors, amplitudes, widths,
                 backgrounds, chi2_values, quality, x):
        self.centers = centers
        self.center_errors = center_errors
        self.amplitudes = amplitudes
        self.widths = widths
        self.backgrounds = backgrounds
        self.chi2_values = chi2_values
        self.quality = quality
        self.x = x

    @property
    def displacement(self):
        """Displacement from mean position (pixels)."""
        valid = self.quality >= 1
        if not np.any(valid):
            return self.centers - np.nanmean(self.centers)
        mean_center = np.nanmean(self.centers[valid])
        return self.centers - mean_center

    @property
    def valid_fraction(self):
        return np.sum(self.quality >= 1) / len(self.quality)

    def to_km(self, plate_scale_km):
        """Convert displacement from pixels to km."""
        return self.displacement * plate_scale_km

    def __repr__(self):
        n_valid = np.sum(self.quality >= 1)
        n_total = len(self.centers)
        return f"TrackResult({n_valid}/{n_total} valid, valid_fraction={self.valid_fraction:.1%})"


def _gaussian_with_background(x, A, x_c, sigma, bg0, bg1):
    """Gaussian peak + linear background.

    I(x) = A * exp(-(x - x_c)^2 / (2*sigma^2)) + bg0 + bg1*x
    """
    return A * np.exp(-0.5 * ((x - x_c) / sigma) ** 2) + bg0 + bg1 * x


def _fit_gaussian_profile(x, profile, sigma, prev_center,
                          min_amplitude, min_width, max_width,
                          max_chi2, max_shift):
    """Fit a single spatial profile with Gaussian + linear background."""
    result = {
        "center": np.nan, "center_error": np.nan,
        "amplitude": np.nan, "width": np.nan,
        "background": np.nan, "chi2_red": np.nan,
        "quality": 0,
    }

    peak_idx = np.argmax(profile)
    A_init = profile[peak_idx] - np.median(profile)
    x_c_init = x[peak_idx]
    sigma_init = 2.0
    bg0_init = np.median(profile)
    bg1_init = 0.0

    if abs(x_c_init - prev_center) > max_shift:
        search_mask = np.abs(x - prev_center) < max_shift
        if np.any(search_mask):
            local_peak = np.argmax(profile * search_mask)
            x_c_init = x[local_peak]
            A_init = profile[local_peak] - np.median(profile)

    p0 = [A_init, x_c_init, sigma_init, bg0_init, bg1_init]
    bounds = (
        [0, x[0], min_width, -np.inf, -np.inf],
        [np.inf, x[-1], max_width, np.inf, np.inf],
    )

    try:
        sigma_arr = sigma if np.ndim(sigma) > 0 else np.full_like(profile, sigma)
        sigma_arr = np.clip(sigma_arr, 1e-10, None)

        popt, pcov = curve_fit(
            _gaussian_with_background, x, profile, p0=p0,
            sigma=sigma_arr, absolute_sigma=True, bounds=bounds,
            maxfev=3000,
        )
        perr = np.sqrt(np.diag(pcov))

        A, x_c, sig, bg0, bg1 = popt
        residuals = profile - _gaussian_with_background(x, *popt)
        ndof = len(x) - 5
        chi2_red = np.sum((residuals / sigma_arr) ** 2) / ndof if ndof > 0 else np.inf

        result["center"] = x_c
        result["center_error"] = perr[1]
        result["amplitude"] = A
        result["width"] = sig
        result["background"] = bg0
        result["chi2_red"] = chi2_red

        shift_ok = abs(x_c - prev_center) <= max_shift
        amp_ok = A >= min_amplitude
        width_ok = min_width <= sig <= max_width
        chi2_ok = chi2_red <= max_chi2
        error_ok = perr[1] < sig

        if shift_ok and amp_ok and width_ok and chi2_ok and error_ok:
            result["quality"] = 2
        elif shift_ok and amp_ok:
            result["quality"] = 1

    except (RuntimeError, ValueError):
        pass

    return result


def _track_centroid(xt_map, x, noise_sigma):
    """Simple intensity-weighted centroid tracking (fast fallback)."""
    nt = xt_map.shape[0]
    bg = np.percentile(xt_map, 25, axis=1, keepdims=True)
    intensity = np.clip(xt_map - bg, 0, None)

    total = np.sum(intensity, axis=1)
    valid = total > 0
    centers = np.full(nt, np.nan)
    centers[valid] = np.sum(intensity[valid] * x, axis=1) / total[valid]

    center_errors = np.full(nt, 1.0)
    quality = np.where(valid, 1, 0)

    return TrackResult(
        centers=centers,
        center_errors=center_errors,
        amplitudes=np.max(xt_map, axis=1) - np.median(xt_map, axis=1),
        widths=np.full(nt, np.nan),
        backgrounds=np.median(xt_map, axis=1),
        chi2_values=np.full(nt, np.nan),
        quality=quality,
        x=x,
    )


def _estimate_noise(xt_map, noise_model=None):
    """Estimate per-pixel noise from instrument model or data."""
    if noise_model is not None:
        readout = noise_model.get("readout", 2.0)
        gain = noise_model.get("gain", 7.29)
        variance = readout ** 2 + gain * np.clip(xt_map, 0, None)
        return np.sqrt(variance)

    residuals = xt_map - np.median(xt_map, axis=0, keepdims=True)
    return np.full(xt_map.shape[0], np.std(residuals) + 1e-10)


def _fill_gaps(centers, quality, max_gap_fraction=0.5):
    """Fill small gaps in tracking via linear interpolation."""
    valid = quality >= 1
    if np.sum(valid) < 2:
        return centers

    total_gap = np.sum(~valid)
    if total_gap / len(centers) > max_gap_fraction:
        return centers

    filled = centers.copy()
    valid_idx = np.where(valid)[0]
    invalid_idx = np.where(~valid)[0]

    if len(invalid_idx) == 0:
        return filled

    interp_vals = np.interp(invalid_idx, valid_idx, centers[valid_idx])
    filled[invalid_idx] = interp_vals
    return filled
