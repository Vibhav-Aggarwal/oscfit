"""oscfit — End-to-end analysis of decayless kink oscillations in coronal loops.

A modern Python replacement for the IDL tools Auto-NUWT (loop tracking) and
SoBAT (Bayesian fitting), with research-grade physics and uncertainty
quantification. Validated against Shrivastav et al. (2024).

Quick start::

    from oscfit import fit_oscillation, derive_parameters
    result = fit_oscillation(time, displacement, model="decayless")
    seismo = derive_parameters(P=result.period[0], A=result.amplitude[0], L=15)
    print(seismo.summary())
"""

__version__ = "0.1.0"

from oscfit.models import MODELS
from oscfit.detect import detect_periods
from oscfit.fit import fit_oscillation, OscFitResult
from oscfit.compare import compare_models, ComparisonResult
from oscfit.seismology import derive_parameters, SeismologyResult
from oscfit.extract import extract_slit_timeseries, extract_slits_along_loop, correct_jitter
from oscfit.track import track_loop_center, TrackResult
from oscfit.pipeline import analyze_loop, PipelineResult
from oscfit.synthetic import make_signal, make_xt_map, make_image_cube, INSTRUMENTS

__all__ = [
    "__version__",
    "MODELS",
    "detect_periods",
    "fit_oscillation", "OscFitResult",
    "compare_models", "ComparisonResult",
    "derive_parameters", "SeismologyResult",
    "extract_slit_timeseries", "extract_slits_along_loop", "correct_jitter",
    "track_loop_center", "TrackResult",
    "analyze_loop", "PipelineResult",
    "make_signal", "make_xt_map", "make_image_cube", "INSTRUMENTS",
]
