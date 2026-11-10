"""Slit extraction from image cubes for time-distance analysis.

Extracts perpendicular intensity slits along a loop spine from a 3D
image cube [nt, ny, nx], producing a time-distance map [nt, n_slit].
Includes sub-pixel interpolation, multi-pixel averaging, and jitter
correction via FFT cross-correlation.

References
----------
.. [1] Mandal et al. 2022, A&A 664, A28 (jitter correction)
"""

import numpy as np
from scipy.ndimage import map_coordinates


def extract_slit_timeseries(cube, loop_path, slit_index=None, slit_width=5,
                            slit_length=40, jitter_correct=True, ref_frame=0):
    """Extract time-distance maps from an image cube along a loop.

    Parameters
    ----------
    cube : ndarray, shape (nt, ny, nx)
        Image cube (intensity in DN).
    loop_path : ndarray, shape (N, 2)
        Loop spine coordinates as (y, x) pairs.
    slit_index : int or array_like, optional
        Index/indices along the loop path at which to place slits.
        If None, places a single slit at the midpoint.
    slit_width : int
        Number of parallel slits to average for SNR (must be odd).
    slit_length : int
        Length of each perpendicular slit (pixels, centered on spine).
    jitter_correct : bool
        Apply FFT cross-correlation jitter correction.
    ref_frame : int
        Reference frame index for jitter correction.

    Returns
    -------
    dict
        xt_map : ndarray, shape (nt, slit_length)
            Time-distance intensity map, averaged across slit_width.
        slit_coords : ndarray, shape (slit_length, 2)
            (y, x) coordinates of the slit on the image.
        slit_normal : ndarray, shape (2,)
            Unit normal vector of the slit.
        shifts : ndarray, shape (nt, 2) or None
            Measured jitter shifts (dy, dx) per frame.
    """
    cube = np.asarray(cube, dtype=np.float64)
    loop_path = np.asarray(loop_path, dtype=np.float64)

    if cube.ndim != 3:
        raise ValueError(f"cube must be 3D (nt, ny, nx), got shape {cube.shape}")
    if loop_path.ndim != 2 or loop_path.shape[1] != 2:
        raise ValueError(f"loop_path must be (N, 2), got shape {loop_path.shape}")
    if len(loop_path) < 2:
        raise ValueError(f"loop_path needs at least 2 points, got {len(loop_path)}")
    if slit_width < 1:
        raise ValueError(f"slit_width must be >= 1, got {slit_width}")
    if slit_length < 3:
        raise ValueError(f"slit_length must be >= 3, got {slit_length}")

    nt, ny, nx = cube.shape

    if slit_width % 2 == 0:
        slit_width += 1

    if jitter_correct:
        shifts = correct_jitter(cube, ref_frame=ref_frame)
    else:
        shifts = None

    if slit_index is None:
        slit_index = len(loop_path) // 2

    if np.isscalar(slit_index):
        slit_index = [slit_index]

    for idx in slit_index:
        if not (0 <= idx < len(loop_path)):
            raise ValueError(
                f"slit_index {idx} out of range [0, {len(loop_path) - 1}]"
            )

    if np.any(loop_path[:, 0] < 0) or np.any(loop_path[:, 0] >= ny) \
            or np.any(loop_path[:, 1] < 0) or np.any(loop_path[:, 1] >= nx):
        import warnings
        warnings.warn(
            "loop_path contains points outside the image bounds; "
            "extracted intensities at those positions will be zero",
            stacklevel=2,
        )

    results = []
    for idx in slit_index:
        normal = _compute_normal(loop_path, idx)
        center = loop_path[idx]
        slit_coords = _make_slit(center, normal, slit_length)

        xt_map = np.zeros((nt, slit_length))
        half_w = slit_width // 2
        tangent = np.array([-normal[1], normal[0]])

        for t in range(nt):
            frame = cube[t]
            slit_sum = np.zeros(slit_length)
            for w_offset in range(-half_w, half_w + 1):
                offset_coords = slit_coords + w_offset * tangent
                if shifts is not None:
                    offset_coords = offset_coords - shifts[t]
                y_coords = offset_coords[:, 0]
                x_coords = offset_coords[:, 1]
                values = map_coordinates(frame, [y_coords, x_coords],
                                         order=1, mode='constant', cval=0.0)
                slit_sum += values
            xt_map[t] = slit_sum / slit_width

        results.append({
            "xt_map": xt_map,
            "slit_coords": slit_coords,
            "slit_normal": normal,
            "shifts": shifts,
        })

    if len(results) == 1:
        return results[0]
    return results


def extract_slits_along_loop(cube, loop_path, n_slits=10, spacing=None,
                             slit_width=5, slit_length=40, jitter_correct=True):
    """Extract multiple evenly-spaced perpendicular slits along a loop.

    Parameters
    ----------
    cube : ndarray, shape (nt, ny, nx)
        Image cube.
    loop_path : ndarray, shape (N, 2)
        Loop spine coordinates (y, x).
    n_slits : int
        Number of slits to place along the loop.
    spacing : float, optional
        Spacing between slits in path-index units. Overrides n_slits.
    slit_width : int
        Width for averaging.
    slit_length : int
        Length of each slit (pixels).
    jitter_correct : bool
        Apply jitter correction.

    Returns
    -------
    list of dict
        One extraction result per slit.
    """
    N = len(loop_path)
    margin = max(3, N // 20)

    if spacing is not None:
        indices = np.arange(margin, N - margin, spacing).astype(int)
    else:
        indices = np.linspace(margin, N - margin - 1, n_slits).astype(int)

    return extract_slit_timeseries(
        cube, loop_path, slit_index=indices,
        slit_width=slit_width, slit_length=slit_length,
        jitter_correct=jitter_correct,
    )


def correct_jitter(cube, ref_frame=0):
    """Measure and return frame-to-frame jitter via FFT cross-correlation.

    Uses ``skimage.registration.phase_cross_correlation`` for sub-pixel
    accuracy (Mandal et al. 2022 method).

    Parameters
    ----------
    cube : ndarray, shape (nt, ny, nx)
        Image cube.
    ref_frame : int
        Index of reference frame.

    Returns
    -------
    shifts : ndarray, shape (nt, 2)
        Measured (dy, dx) shifts per frame relative to reference.
    """
    from skimage.registration import phase_cross_correlation

    nt = cube.shape[0]
    reference = cube[ref_frame]
    shifts = np.zeros((nt, 2))

    for i in range(nt):
        if i == ref_frame:
            continue
        shift, _, _ = phase_cross_correlation(
            reference, cube[i], upsample_factor=10
        )
        shifts[i] = shift

    return shifts


def _compute_normal(loop_path, index):
    """Compute unit normal vector at a point on the loop spine."""
    N = len(loop_path)
    if index == 0:
        tangent = loop_path[1] - loop_path[0]
    elif index == N - 1:
        tangent = loop_path[-1] - loop_path[-2]
    else:
        tangent = loop_path[index + 1] - loop_path[index - 1]

    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm < 1e-12:
        return np.array([0.0, 1.0])

    tangent = tangent / tangent_norm
    normal = np.array([-tangent[1], tangent[0]])
    return normal


def _make_slit(center, normal, length):
    """Generate slit pixel coordinates centered on a point.

    Returns
    -------
    coords : ndarray, shape (length, 2)
        (y, x) coordinates along the slit.
    """
    half = length / 2.0
    offsets = np.linspace(-half, half, length)
    coords = center + offsets[:, np.newaxis] * normal
    return coords
