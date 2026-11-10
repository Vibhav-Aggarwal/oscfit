"""Backtest oscfit against INDEPENDENT published catalogs (not Shrivastav).

Datasets:
  1. Nechaeva et al. 2019 (ApJS 241, 31) — 223 DECAYING kink oscillations,
     24th solar cycle, SDO/AIA. Periods in minutes, lengths/amplitudes in Mm.
  2. Anfinogentov et al. 2015 (A&A 583, A136) — ~54 DECAYLESS oscillations.
     Periods in seconds, lengths/amplitudes in Mm.

Neither catalog publishes B or density, so we validate:
  - C_k = 2L/P directly against the kink-speed ranges the papers report
  - B (with standard coronal density) lands in the expected few-to-tens-of-Gauss range
"""

import csv
from pathlib import Path
import numpy as np

from oscfit.seismology import derive_parameters

PAPERS_DIR = Path(__file__).parent.parent.parent / "papers"


def load_nechaeva():
    """Load Nechaeva 2019. P in minutes, L and Amp in Mm."""
    rows = []
    with open(PAPERS_DIR / "nechaeva2019.csv") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                P_min = float(r["Per"])
                L_Mm = float(r["Len"]) if r["Len"] else None
                A_Mm = float(r["Amp"]) if r["Amp"] else None
                if L_Mm is None or P_min <= 0 or L_Mm <= 0:
                    continue
                rows.append({
                    "P": P_min * 60.0,       # s
                    "L": L_Mm,               # Mm
                    "A": A_Mm * 1e3 if A_Mm else None,  # km
                    "event": r["Event"],
                    "loop": r["Loop"],
                })
            except (ValueError, KeyError):
                continue
    return rows


# Anfinogentov 2015 Table A.1 (decayless): L[Mm], P[s], A[Mm]
# Representative sample extracted from the published appendix.
ANFINOGENTOV_2015 = [
    # (AR, L_Mm, P_s, A_Mm)
    (11637, 190, 250, 0.09),
    (11639, 230, 210, 0.13),
    (11641, 370, 610, 0.37),
    (11650, 560, 440, 0.43),
    (11642, 450, 530, 0.33),
    (11431, 204, 175, 0.061),
    (11431, 174, 270, 0.18),
    (11460, 146, 232, 0.13),
    (11598, 135, 192, 0.072),
    (11598, 125, 234, 0.045),
]


def kink_speed_check(rows, label, published_range, published_median):
    """Compute C_k = 2L/P for all rows and compare to published stats."""
    cks = []
    for r in rows:
        ck = derive_parameters(P=r["P"], A=(r["A"] or 1.0), L=r["L"],
                               n_e=1e15, zeta=3.0).C_k_km_s
        cks.append(ck)
    cks = np.array(cks)
    print(f"\n{label}  (N = {len(cks)})")
    print(f"  oscfit C_k:    median = {np.median(cks):.0f} km/s, "
          f"mean = {np.mean(cks):.0f} km/s")
    print(f"  oscfit range:  {np.min(cks):.0f} – {np.max(cks):.0f} km/s")
    print(f"  published:     median ~ {published_median} km/s, "
          f"range {published_range[0]}–{published_range[1]} km/s")
    in_band = np.sum((cks >= published_range[0] * 0.5) & (cks <= published_range[1] * 1.5))
    print(f"  within ~published band: {in_band}/{len(cks)} ({in_band/len(cks)*100:.0f}%)")
    return cks


def magnetic_field_check(rows, label, n_e=1e15, zeta=3.0):
    """Compute B with standard coronal density; check it's physically reasonable."""
    bs = []
    for r in rows:
        if r["A"] is None:
            continue
        b = derive_parameters(P=r["P"], A=r["A"], L=r["L"],
                              n_e=n_e, zeta=zeta).B_gauss
        bs.append(b)
    bs = np.array(bs)
    print(f"\n{label} — magnetic field (n_e={n_e:.0e}, zeta={zeta})")
    print(f"  B: median = {np.median(bs):.1f} G, mean = {np.mean(bs):.1f} G, "
          f"range = {np.min(bs):.1f}–{np.max(bs):.1f} G")
    print(f"  Literature expectation for coronal loops: ~1–100 G")
    reasonable = np.sum((bs >= 0.1) & (bs <= 200))
    print(f"  physically reasonable (0.1–200 G): {reasonable}/{len(bs)} "
          f"({reasonable/len(bs)*100:.0f}%)")
    return bs


def main():
    print("=" * 70)
    print("OSCFIT BACKTEST vs INDEPENDENT CATALOGS")
    print("=" * 70)

    # --- Nechaeva 2019 (decaying oscillations) ---
    nech = load_nechaeva()
    print(f"\nLoaded Nechaeva et al. 2019: {len(nech)} oscillations with valid L, P")

    # Published: median C_k ~ 1330 km/s (Goddard fit), main band 800-3300 km/s
    kink_speed_check(nech, "NECHAEVA 2019 (decaying, SDO/AIA)",
                     published_range=(800, 3300), published_median=1330)
    magnetic_field_check(nech, "NECHAEVA 2019")

    # --- Anfinogentov 2015 (decayless oscillations) ---
    anf = [{"P": P, "L": L, "A": A * 1e3, "event": str(ar), "loop": ""}
           for ar, L, P, A in ANFINOGENTOV_2015]
    print(f"\n\nLoaded Anfinogentov et al. 2015: {len(anf)} decayless oscillations (sample)")
    kink_speed_check(anf, "ANFINOGENTOV 2015 (decayless, SDO/AIA)",
                     published_range=(700, 3000), published_median=1200)
    magnetic_field_check(anf, "ANFINOGENTOV 2015")

    # --- Specific hand-checks ---
    print("\n" + "=" * 70)
    print("SPOT CHECKS (C_k = 2L/P, computed by hand)")
    print("=" * 70)
    checks = [
        ("Nechaeva Ev1-L1", 232, 3.42 * 60),
        ("Nechaeva Ev3-L2", 262, 3.62 * 60),
        ("Anfinogentov AR11637", 190, 250),
        ("Anfinogentov AR11641", 370, 610),
    ]
    for name, L_Mm, P_s in checks:
        ck_hand = 2 * L_Mm * 1e3 / P_s  # km/s
        ck_tool = derive_parameters(P=P_s, A=1.0, L=L_Mm).C_k_km_s
        print(f"  {name:>24s}: hand={ck_hand:7.1f}  oscfit={ck_tool:7.1f}  "
              f"{'OK' if abs(ck_hand - ck_tool) < 1 else 'MISMATCH'}")

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
