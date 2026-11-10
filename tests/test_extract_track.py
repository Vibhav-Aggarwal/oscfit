"""Tests for slit extraction and loop tracking modules."""

import numpy as np
import pytest

from oscfit.extract import extract_slit_timeseries, extract_slits_along_loop, _compute_normal
from oscfit.track import track_loop_center, _gaussian_with_background


class TestComputeNormal:
    def test_horizontal_path(self):
        path = np.array([[50.0, 10.0], [50.0, 20.0], [50.0, 30.0]])
        normal = _compute_normal(path, 1)
        np.testing.assert_allclose(np.abs(normal), [1.0, 0.0], atol=1e-10)

    def test_vertical_path(self):
        path = np.array([[10.0, 50.0], [20.0, 50.0], [30.0, 50.0]])
        normal = _compute_normal(path, 1)
        np.testing.assert_allclose(np.abs(normal), [0.0, 1.0], atol=1e-10)

    def test_unit_length(self):
        path = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
        normal = _compute_normal(path, 1)
        assert np.linalg.norm(normal) == pytest.approx(1.0, abs=1e-10)

    def test_endpoint_handling(self):
        path = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        n_start = _compute_normal(path, 0)
        n_end = _compute_normal(path, 2)
        assert np.linalg.norm(n_start) == pytest.approx(1.0, abs=1e-10)
        assert np.linalg.norm(n_end) == pytest.approx(1.0, abs=1e-10)


class TestExtractSlit:
    def test_output_shape(self, synthetic_cube):
        cube = synthetic_cube["cube"]
        path = synthetic_cube["loop_path"]
        result = extract_slit_timeseries(cube, path, slit_length=30, jitter_correct=False)
        assert result["xt_map"].shape == (cube.shape[0], 30)

    def test_slit_captures_loop(self, synthetic_cube):
        cube = synthetic_cube["cube"]
        path = synthetic_cube["loop_path"]
        result = extract_slit_timeseries(cube, path, slit_length=40, jitter_correct=False)
        xt = result["xt_map"]
        assert np.max(xt) > np.median(xt) * 1.5

    def test_multiple_slits(self, synthetic_cube):
        cube = synthetic_cube["cube"]
        path = synthetic_cube["loop_path"]
        results = extract_slit_timeseries(
            cube, path, slit_index=[20, 40, 60],
            slit_length=30, jitter_correct=False,
        )
        assert len(results) == 3
        for r in results:
            assert r["xt_map"].shape[0] == cube.shape[0]

    def test_extract_slits_along_loop(self, synthetic_cube):
        cube = synthetic_cube["cube"]
        path = synthetic_cube["loop_path"]
        results = extract_slits_along_loop(
            cube, path, n_slits=5, slit_length=30, jitter_correct=False,
        )
        assert len(results) == 5


class TestGaussianModel:
    def test_peak_at_center(self):
        x = np.arange(50, dtype=float)
        y = _gaussian_with_background(x, A=100, x_c=25, sigma=3, bg0=10, bg1=0)
        assert np.argmax(y) == 25
        assert y[25] == pytest.approx(110.0, abs=0.1)

    def test_background_slope(self):
        x = np.arange(50, dtype=float)
        y = _gaussian_with_background(x, A=0, x_c=25, sigma=3, bg0=10, bg1=2)
        np.testing.assert_allclose(y, 10 + 2 * x)


class TestTrackLoopCenter:
    def test_recovers_oscillation(self, synthetic_xt):
        xt = synthetic_xt
        result = track_loop_center(xt["xt_map"], xt["x"])
        true_center = xt["true_center"]
        valid = result.quality >= 1
        assert np.sum(valid) > len(true_center) * 0.7

        measured = result.centers[valid]
        expected = true_center[valid]
        rms = np.sqrt(np.mean((measured - expected) ** 2))
        assert rms < 2.0  # sub-2-pixel accuracy

    def test_displacement_is_oscillatory(self, synthetic_xt):
        xt = synthetic_xt
        result = track_loop_center(xt["xt_map"], xt["x"])
        d = result.displacement
        valid = result.quality >= 1
        if np.sum(valid) > 20:
            zero_crossings = np.sum(np.diff(np.sign(d[valid])) != 0)
            assert zero_crossings >= 2

    def test_centroid_method(self, synthetic_xt):
        xt = synthetic_xt
        result = track_loop_center(xt["xt_map"], xt["x"], method="centroid")
        assert result.valid_fraction > 0.9

    def test_to_km(self, synthetic_xt):
        xt = synthetic_xt
        result = track_loop_center(xt["xt_map"], xt["x"])
        d_km = result.to_km(plate_scale_km=210.0)
        d_px = result.displacement
        np.testing.assert_allclose(d_km, d_px * 210.0)

    def test_noise_model_used(self, synthetic_xt):
        xt = synthetic_xt
        noise = {"readout": 2.0, "gain": 7.29}
        result = track_loop_center(xt["xt_map"], xt["x"], noise_model=noise)
        assert result.valid_fraction > 0.5

    def test_repr(self, synthetic_xt):
        xt = synthetic_xt
        result = track_loop_center(xt["xt_map"], xt["x"])
        r = repr(result)
        assert "TrackResult" in r
        assert "valid" in r


class TestTrackEdgeCases:
    def test_flat_profile(self):
        xt = np.ones((50, 30)) * 100.0
        result = track_loop_center(xt)
        assert isinstance(result.centers, np.ndarray)

    def test_single_bright_pixel(self):
        xt = np.ones((20, 30)) * 50.0
        xt[:, 15] = 500.0
        result = track_loop_center(xt)
        valid = result.quality >= 1
        if np.any(valid):
            assert np.nanmean(result.centers[valid]) == pytest.approx(15.0, abs=2.0)
