"""End-to-end oscillation analysis pipeline.

Connects all modules: extract → track → detect → fit → seismology.
"""

import numpy as np
from oscfit.extract import extract_slit_timeseries
from oscfit.track import track_loop_center
from oscfit.detect import detect_periods
from oscfit.fit import fit_oscillation
from oscfit.seismology import derive_parameters


def analyze_loop(cube, loop_path, loop_length_Mm, plate_scale_km=210.0,
                 slit_index=None, slit_width=5, slit_length=40,
                 model="decayless", mcmc=True, n_e=1e15, zeta=2.0,
                 cadence=None, noise_model=None, jitter_correct=True):
    """Run the full oscillation analysis pipeline on a loop.

    Parameters
    ----------
    cube : ndarray, shape (nt, ny, nx)
        Image cube (DN).
    loop_path : ndarray, shape (N, 2)
        Loop spine coordinates (y, x).
    loop_length_Mm : float
        Physical loop length (Mm).
    plate_scale_km : float
        Pixel scale (km/pixel).
    slit_index : int, optional
        Position along loop for slit. Default: midpoint.
    slit_width : int
        Slit averaging width (pixels).
    slit_length : int
        Slit extent (pixels).
    model : str
        Oscillation model to fit.
    mcmc : bool
        Run MCMC for uncertainties.
    n_e : float
        Electron density (m⁻³).
    zeta : float
        Density contrast ratio.
    cadence : float, optional
        Time cadence (s). If None, inferred as 1s per frame.
    noise_model : dict, optional
        Instrument noise model.
    jitter_correct : bool
        Apply jitter correction.

    Returns
    -------
    PipelineResult
    """
    nt = cube.shape[0]
    if cadence is None:
        cadence = 1.0
    time = np.arange(nt) * cadence

    extraction = extract_slit_timeseries(
        cube, loop_path, slit_index=slit_index,
        slit_width=slit_width, slit_length=slit_length,
        jitter_correct=jitter_correct,
    )

    tracking = track_loop_center(
        extraction["xt_map"], noise_model=noise_model,
    )

    if tracking.valid_fraction < 0.2:
        raise RuntimeError(
            f"Loop tracking failed: only {tracking.valid_fraction:.1%} of "
            f"frames had valid Gaussian fits (need >= 20%). "
            f"Check input data quality, slit placement, or noise model."
        )

    displacement_km = tracking.to_km(plate_scale_km)

    detection = detect_periods(time, displacement_km, method="fft")

    fit_result = fit_oscillation(
        time, displacement_km, model=model,
        period_hint=detection.dominant_period,
        mcmc=mcmc,
    )

    L_err = loop_length_Mm * 0.1
    P_val, P_err = fit_result.period
    A_val, A_err = fit_result.amplitude

    seismo = derive_parameters(
        P=P_val, A=A_val, L=loop_length_Mm,
        P_err=P_err, A_err=A_err, L_err=L_err,
        n_e=n_e, zeta=zeta,
        samples=fit_result.samples,
    )

    return PipelineResult(
        extraction=extraction,
        tracking=tracking,
        detection=detection,
        fit=fit_result,
        seismology=seismo,
        time=time,
        displacement_km=displacement_km,
    )


class PipelineResult:
    """Container for full pipeline results."""

    def __init__(self, extraction, tracking, detection, fit, seismology,
                 time, displacement_km):
        self.extraction = extraction
        self.tracking = tracking
        self.detection = detection
        self.fit = fit
        self.seismology = seismology
        self.time = time
        self.displacement_km = displacement_km

    def to_csv_row(self, loop_length_Mm=None):
        """Export as a single row matching Shrivastav et al. format."""
        fit_dict = self.fit.to_dict()
        seismo_dict = self.seismology.to_dict()
        row = {**fit_dict, **seismo_dict}
        if loop_length_Mm is not None:
            row["L (Mm)"] = f"{loop_length_Mm:.1f}"
        return row

    def summary(self):
        lines = ["═" * 60]
        lines.append("OSCFIT PIPELINE RESULTS")
        lines.append("═" * 60)
        lines.append(f"\nTracking: {self.tracking!r}")
        lines.append(f"Detection: {self.detection!r}")
        lines.append(f"\n{self.fit.summary()}")
        lines.append(f"\n{self.seismology.summary()}")
        lines.append("═" * 60)
        return "\n".join(lines)

    def __repr__(self):
        return f"PipelineResult(model={self.fit.model_name}, B={self.seismology.B_gauss:.1f} G)"
