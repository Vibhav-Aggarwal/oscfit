# oscfit

[![CI](https://github.com/Vibhav-Aggarwal/oscfit/actions/workflows/ci.yml/badge.svg)](https://github.com/Vibhav-Aggarwal/oscfit/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Powered by SunPy](https://img.shields.io/badge/powered%20by-SunPy-orange.svg)](https://sunpy.org/)
[![Research Paper](https://img.shields.io/badge/paper-PDF%20Manuscript-red.svg)](paper/manuscript.pdf)
[![JOSS Status](https://img.shields.io/badge/JOSS-In%20Review-brightgreen.svg)](https://joss.theoj.org/papers/84a9ed30d656b021760e68758362501c)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**End-to-end analysis of decayless kink oscillations in coronal loops.**

A modern Python replacement for the unmaintained IDL tools Auto-NUWT (loop tracking,
dead since 2019) and SoBAT (Bayesian fitting, last push 2020). Built to fill a real
gap: no Python package exists for end-to-end EUV coronal loop oscillation analysis —
every research group writes ad-hoc scripts. Validated against published, peer-reviewed
results.

## Why this exists

The corona-heating problem is one of solar physics' biggest open questions. Decayless
kink oscillations in coronal loops are a leading candidate energy source. Measuring
them is slow and manual: in 10+ years of SDO data, only ~223 oscillations have been
catalogued by hand (Nechaeva et al. 2019). `oscfit` automates the pipeline from raw
FITS image cube to seismology-derived magnetic field.

## Pipeline

```
FITS cube → [extract] → x-t map → [track] → displacement(t) → [detect] → periods
    → [fit] → parameters ± uncertainties → [compare] → best model
    → [seismology] → B, V_A, energy flux → CSV/FITS output
```

## Modules

| Module | What it does |
|--------|-------------|
| `models.py` | 6 oscillation models (decayless, exponential, Gaussian, two-regime, period-varying, growing) |
| `synthetic.py` | Signal / x-t map / image cube generators with EUI & AIA noise models |
| `detect.py` | FFT + Lomb-Scargle + Morlet wavelet, power-law noise background, Bonferroni significance |
| `extract.py` | Perpendicular slit extraction from FITS cubes, FFT jitter correction |
| `track.py` | Gaussian sub-pixel center tracking (Auto-NUWT port), quality filtering |
| `fit.py` | scipy curve_fit + emcee MCMC, convergence diagnostics, BIC/AIC |
| `compare.py` | dynesty nested sampling, Bayesian evidence, Bayes factors |
| `seismology.py` | Physics derivation (C_k, V_A, B, energy flux) with MC + analytical errors |
| `pipeline.py` | End-to-end orchestrator |
| `cli.py` | Command-line interface |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

```python
from oscfit import fit_oscillation, derive_parameters

# Fit an oscillation in a displacement time series
result = fit_oscillation(time, displacement_km, model="decayless", mcmc=True)
print(result.summary())

# Derive the loop's magnetic field
seismo = derive_parameters(P=result.period[0], A=result.amplitude[0],
                           L=15.0, n_e=1e15, zeta=2.0)
print(f"B = {seismo.B_gauss:.1f} G")
```

CLI:
```bash
oscfit seismo --period 252 --amplitude 176 --length 24.5
oscfit fit signal.csv --model decayless --mcmc
```

## Validation

145 tests, 91% coverage. Validated against **three independent published catalogs**:

| Catalog | N | Result |
|---------|---|--------|
| **Shrivastav et al. 2024** (Solar Orbiter/EUI) | 105 | B matches to **0.5%**, C_k to 0.1%, full per-loop B reproduction |
| **Nechaeva et al. 2019** (SDO/AIA, decaying) | 221 | C_k distribution matches (median 1407 vs 1330 km/s), all within band |
| **Anfinogentov et al. 2015** (SDO/AIA, decayless) | 10 | C_k matches published range |

Spans decaying + decayless oscillations, two instruments, three research groups.

See `examples/backtest_shrivastav.py` and `examples/backtest_independent.py`.

## Key physics (research-grade, vetted against literature)

```
Kink speed:    C_k = 2L/P                          (fundamental standing mode)
Velocity amp:  V = 2πA/P
Alfvén speed:  V_A = C_k·√((1 + 1/ζ)/2)
Magnetic field: B = V_A·√(μ₀·ρ₀)
Energy flux:   F = ½(ρ₀+ρ_e)·V²·C_k                (total; or ¼ for kinetic only)
Density:       ρ₀ = n_e·μ_e·m_p,  μ_e = 1.18 (fully ionized H+He)
```

## Validation Note

Shrivastav et al. (2025) Table 2 B-values are self-consistent with ρ ≈ 2.12×10⁻¹²
kg/m³ (n_e ≈ 1.07×10¹⁵ m⁻³), rather than the ρ = 1.67×10⁻¹² stated in the paper text (which
would yield B ~11% lower). A minor internal inconsistency oscfit surfaced during automated verification.

## Scientific Paper & Citation

If you use `oscfit` in your research, please cite the accompanying manuscript and software:

```bibtex
@article{Aggarwal2026_oscfit,
  author    = {{Aggarwal}, Vibhav},
  title     = {{oscfit}: An Automated Bayesian Framework and Seismological Inversion Pipeline for Decayless Kink Oscillations in Solar Coronal Loops},
  journal   = {arXiv preprint / Journal of Open Source Software},
  year      = {2026},
  url       = {https://github.com/Vibhav-Aggarwal/oscfit}
}
```

* **Preprint PDF (Elsevier 2-Column)**: [`paper/aggarwal-2026-oscfit-bayesian-coronal-seismology-v1.pdf`](paper/aggarwal-2026-oscfit-bayesian-coronal-seismology-v1.pdf)
* **Preprint PDF (AASTeX/arXiv Style)**: [`paper/manuscript.pdf`](paper/manuscript.pdf)
* **JOSS Paper Draft**: [`paper/paper.md`](paper/paper.md) (Journal of Open Source Software submission)
* **Figure Generation Script**: [`paper/generate_figures.py`](paper/generate_figures.py)

## Author
**Vibhav Aggarwal** ([@Vibhav-Aggarwal](https://github.com/Vibhav-Aggarwal)) • [vibhav.aggarwal2@gmail.com](mailto:vibhav.aggarwal2@gmail.com) • ORCID: [0009-0000-7686-7119](https://orcid.org/0009-0000-7686-7119)

## License
BSD-3-Clause
