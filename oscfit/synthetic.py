"""Synthetic data generators for testing and validation.

Generates realistic coronal loop oscillation signals and time-distance
maps with configurable noise models matching EUI and AIA instruments.
"""

import numpy as np
from oscfit import models


# Instrument noise parameters (readout_noise_DN, gain_DN_per_photon)
INSTRUMENTS = {
    "eui_hrieuv": {"readout": 2.0, "gain": 7.29, "cadence": 3.0, "plate_scale_km": 210.0},
    "aia_171": {"readout": 1.18, "gain": 17.7, "cadence": 12.0, "plate_scale_km": 435.0},
    "aia_193": {"readout": 1.18, "gain": 17.7, "cadence": 12.0, "plate_scale_km": 435.0},
}


def make_signal(
    model="decayless",
    params=None,
    duration=600.0,
    cadence=3.0,
    snr=10.0,
    seed=None,
    gaps=None,
):
    """Generate a synthetic oscillation time series.

    Parameters
    ----------
    model : str
        Model name from ``oscfit.models.MODELS``.
    params : dict, optional
        Model parameters. If None, uses physically reasonable defaults.
    duration : float
        Total observation duration (s).
    cadence : float
        Time between samples (s).
    snr : float
        Signal-to-noise ratio (amplitude / noise std).
    seed : int, optional
        Random seed for reproducibility.
    gaps : list of tuple, optional
        List of (start_s, end_s) intervals to mask as NaN.

    Returns
    -------
    dict
        time : ndarray, time array (s).
        signal : ndarray, displacement with noise (km).
        clean : ndarray, noise-free signal (km).
        noise : ndarray, noise component (km).
        params : dict, true parameter values.
        model : str, model name.
    """
    rng = np.random.default_rng(seed)
    model_info = models.MODELS[model]
    func = model_info["func"]

    if params is None:
        params = _default_params(model, duration)

    time = np.arange(0, duration, cadence)
    param_values = [params[k] for k in model_info["param_names"]]
    clean = func(time, *param_values)

    noise_std = params["A"] / snr if snr > 0 else 0.0
    noise = rng.normal(0, noise_std, len(time)) if noise_std > 0 else np.zeros_like(time)
    signal = clean + noise

    if gaps:
        for g_start, g_end in gaps:
            mask = (time >= g_start) & (time <= g_end)
            signal[mask] = np.nan

    return {
        "time": time,
        "signal": signal,
        "clean": clean,
        "noise": noise,
        "params": dict(params),
        "model": model,
    }


def make_xt_map(
    nx=100,
    nt=200,
    loop_center=50.0,
    loop_width=4.0,
    loop_intensity=500.0,
    background=100.0,
    oscillation_amplitude=3.0,
    period=120.0,
    cadence=3.0,
    instrument="eui_hrieuv",
    seed=None,
):
    """Generate a synthetic time-distance (x-t) map of an oscillating loop.

    Creates a 2D array simulating what you'd see after extracting a
    perpendicular slit through a coronal loop: a Gaussian intensity
    profile whose center oscillates sinusoidally over time, with
    instrument-appropriate Poisson + readout noise.

    Parameters
    ----------
    nx : int
        Spatial extent of slit (pixels).
    nt : int
        Number of time steps.
    loop_center : float
        Mean position of loop center (pixels).
    loop_width : float
        Gaussian sigma of loop cross-section (pixels).
    loop_intensity : float
        Peak intensity of loop above background (DN).
    background : float
        Background intensity (DN).
    oscillation_amplitude : float
        Displacement amplitude (pixels).
    period : float
        Oscillation period (s).
    cadence : float
        Time between frames (s).
    instrument : str
        Instrument name for noise model.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        xt_map : ndarray [nt, nx], intensity map (DN).
        xt_clean : ndarray [nt, nx], noise-free map.
        time : ndarray [nt], time array (s).
        x : ndarray [nx], spatial array (pixels).
        true_center : ndarray [nt], true loop center position (pixels).
        true_params : dict, ground truth parameters.
    """
    rng = np.random.default_rng(seed)
    inst = INSTRUMENTS.get(instrument, INSTRUMENTS["eui_hrieuv"])

    time = np.arange(nt) * cadence
    x = np.arange(nx, dtype=float)

    center = loop_center + oscillation_amplitude * np.sin(2.0 * np.pi * time / period)

    xt_clean = np.zeros((nt, nx))
    for i in range(nt):
        xt_clean[i] = background + loop_intensity * np.exp(
            -0.5 * ((x - center[i]) / loop_width) ** 2
        )

    sigma_readout = inst["readout"]
    gain = inst["gain"]
    total_intensity = np.clip(xt_clean, 0, None)
    noise_var = sigma_readout ** 2 + gain * total_intensity
    noise = rng.normal(0, np.sqrt(noise_var))
    xt_map = np.clip(xt_clean + noise, 0, None)

    return {
        "xt_map": xt_map,
        "xt_clean": xt_clean,
        "time": time,
        "x": x,
        "true_center": center,
        "true_params": {
            "loop_center": loop_center,
            "loop_width": loop_width,
            "oscillation_amplitude": oscillation_amplitude,
            "period": period,
            "loop_intensity": loop_intensity,
            "background": background,
        },
    }


def make_image_cube(
    ny=256,
    nx=256,
    nt=100,
    loop_path=None,
    loop_width=3.0,
    loop_intensity=400.0,
    background=80.0,
    oscillation_amplitude=2.0,
    period=100.0,
    cadence=5.0,
    instrument="eui_hrieuv",
    seed=None,
):
    """Generate a synthetic 3D image cube with an oscillating coronal loop.

    Parameters
    ----------
    ny, nx : int
        Image dimensions (pixels).
    nt : int
        Number of frames.
    loop_path : ndarray of shape (N, 2), optional
        Loop spine coordinates [(y, x), ...]. If None, generates a
        semicircular loop.
    loop_width : float
        Gaussian width of the loop cross-section (pixels).
    loop_intensity : float
        Peak loop intensity above background (DN).
    background : float
        Background level (DN).
    oscillation_amplitude : float
        Transverse displacement amplitude (pixels).
    period : float
        Oscillation period (s).
    cadence : float
        Time between frames (s).
    instrument : str
        Instrument name for noise model.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        cube : ndarray [nt, ny, nx], image cube (DN).
        time : ndarray [nt], time array (s).
        loop_path : ndarray (N, 2), loop spine coordinates.
        true_displacement : ndarray [nt], true transverse displacement (pixels).
    """
    rng = np.random.default_rng(seed)
    inst = INSTRUMENTS.get(instrument, INSTRUMENTS["eui_hrieuv"])

    if loop_path is None:
        theta = np.linspace(0.2 * np.pi, 0.8 * np.pi, 80)
        radius = min(ny, nx) * 0.35
        cy, cx = ny / 2, nx / 2
        loop_path = np.column_stack([
            cy - radius * np.sin(theta),
            cx + radius * np.cos(theta),
        ])

    tangents = np.gradient(loop_path, axis=0)
    norms = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    norms /= np.linalg.norm(norms, axis=1, keepdims=True)

    time = np.arange(nt) * cadence
    displacement = oscillation_amplitude * np.sin(2.0 * np.pi * time / period)

    yy, xx = np.mgrid[:ny, :nx]
    cube = np.full((nt, ny, nx), background, dtype=np.float64)

    for t_idx in range(nt):
        shifted_path = loop_path + displacement[t_idx] * norms
        for j in range(len(shifted_path)):
            dy = yy - shifted_path[j, 0]
            dx = xx - shifted_path[j, 1]
            r2 = dy ** 2 + dx ** 2
            cube[t_idx] += loop_intensity * np.exp(-r2 / (2.0 * loop_width ** 2))

    sigma_readout = inst["readout"]
    gain = inst["gain"]
    noise_var = sigma_readout ** 2 + gain * np.clip(cube, 0, None)
    cube = np.clip(cube + rng.normal(0, np.sqrt(noise_var)), 0, None)

    return {
        "cube": cube,
        "time": time,
        "loop_path": loop_path,
        "true_displacement": displacement,
    }


def _default_params(model, duration):
    """Physically reasonable default parameters for each model."""
    base = {"A": 100.0, "P": 120.0, "phi": 0.5, "c0": 500.0, "c1": 0.1}

    if model == "decayless":
        return base

    if model == "exponential":
        return {**base, "tau": duration * 0.8}

    if model == "gaussian":
        return {**base, "tau_g": duration * 0.5}

    if model == "two_regime":
        tau_g = duration * 0.3
        tau_d = duration * 0.8
        return {**base, "tau_g": tau_g, "tau_d": tau_d}

    if model == "period_varying":
        return {**base, "P0": base["P"], "tau_g": duration * 2, "alpha": 1e-4}

    if model == "growing":
        return {**base, "tau": duration * 1.5}

    return base
