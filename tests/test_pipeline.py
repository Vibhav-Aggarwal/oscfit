"""End-to-end pipeline integration test."""

import numpy as np
import pytest

from oscfit.pipeline import analyze_loop
from oscfit.synthetic import make_image_cube


class TestPipeline:
    def test_full_pipeline(self):
        data = make_image_cube(
            ny=128, nx=128, nt=80,
            loop_width=3.0, loop_intensity=400.0,
            oscillation_amplitude=2.0, period=100.0,
            cadence=5.0, seed=42,
        )
        result = analyze_loop(
            cube=data["cube"],
            loop_path=data["loop_path"],
            loop_length_Mm=15.0,
            plate_scale_km=210.0,
            cadence=5.0,
            model="decayless",
            mcmc=False,
            jitter_correct=False,
        )

        assert result.fit.model_name == "decayless"
        assert result.seismology.B_gauss > 0
        assert result.seismology.C_k > 0

    def test_pipeline_summary(self):
        data = make_image_cube(
            ny=128, nx=128, nt=60,
            period=80, cadence=5, seed=42,
        )
        result = analyze_loop(
            cube=data["cube"],
            loop_path=data["loop_path"],
            loop_length_Mm=10.0,
            cadence=5.0,
            mcmc=False,
            jitter_correct=False,
        )
        s = result.summary()
        assert "OSCFIT PIPELINE RESULTS" in s

    def test_pipeline_csv_row(self):
        data = make_image_cube(
            ny=128, nx=128, nt=60,
            period=80, cadence=5, seed=42,
        )
        result = analyze_loop(
            cube=data["cube"],
            loop_path=data["loop_path"],
            loop_length_Mm=10.0,
            cadence=5.0,
            mcmc=False,
            jitter_correct=False,
        )
        row = result.to_csv_row(loop_length_Mm=10.0)
        assert "P (s)" in row
        assert "B (G)" in row
        assert "L (Mm)" in row

    def test_pipeline_repr(self):
        data = make_image_cube(
            ny=128, nx=128, nt=60,
            period=80, cadence=5, seed=42,
        )
        result = analyze_loop(
            cube=data["cube"],
            loop_path=data["loop_path"],
            loop_length_Mm=10.0,
            cadence=5.0,
            mcmc=False,
            jitter_correct=False,
        )
        assert "PipelineResult" in repr(result)
