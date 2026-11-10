"""Backtest oscfit against the full Shrivastav et al. (2024) dataset.

Downloads Arpit's published oscillation parameters (105 active region + 42
quiet sun) and runs them through oscfit's seismology module, comparing
output against published Ck and B values.
"""

import json
import subprocess
import numpy as np
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from oscfit import derive_parameters, fit_oscillation, make_signal


REPO = "ArpitkShrivastav/Oscillation-Parameters-Short-Loops"


def download_csv(filename):
    out = subprocess.run(
        ["gh", "api", f"repos/{REPO}/contents/{filename}", "--jq", ".content"],
        capture_output=True, text=True, check=True,
    )
    import base64
    return base64.b64decode(out.stdout).decode("utf-8")


def parse_ar(text):
    """Active region data: No, L(Mm), P(s), A(km), V(km/s), Ck(km/s), B(G), Width(Mm)."""
    rows = []
    for line in text.strip().split("\n")[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            P_str, P_err_str = parts[2].split("±")
            A_str, A_err_str = parts[3].split("±")
            V_str, V_err_str = parts[4].split("±")
            rows.append({
                "id": int(parts[0]),
                "L": float(parts[1]),
                "P": float(P_str), "P_err": float(P_err_str),
                "A": float(A_str), "A_err": float(A_err_str),
                "V": float(V_str), "V_err": float(V_err_str),
                "Ck": float(parts[5]),
                "B": float(parts[6]),
                "W": float(parts[7]),
                "region": "AR",
            })
        except (ValueError, IndexError):
            continue
    return rows


def parse_qs(text):
    """Quiet Sun: No, Date, X, Y, Start, End, L(Mm), P(s), A(km)."""
    rows = []
    for line in text.strip().split("\n")[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 9:
            continue
        try:
            P_str, P_err_str = parts[7].split("±")
            A_str, A_err_str = parts[8].split("±")
            L = float(parts[6])
            P = float(P_str)
            A = float(A_str)
            rows.append({
                "id": int(parts[0]),
                "L": L,
                "P": P, "P_err": float(P_err_str),
                "A": A, "A_err": float(A_err_str),
                "V": 2 * np.pi * A / P,  # derived
                "V_err": 2 * np.pi * float(A_err_str) / P,
                "Ck": 2 * L * 1e3 / P,  # km/s (L in Mm, *1e3 = km, /P in s)
                "B": None,  # not published for QS
                "W": None,
                "region": "QS",
            })
        except (ValueError, IndexError):
            continue
    return rows


def backtest_seismology(rows):
    """Run all rows through derive_parameters and compare."""
    results = []
    for row in rows:
        try:
            r = derive_parameters(
                P=row["P"], A=row["A"], L=row["L"],
                P_err=row["P_err"], A_err=row["A_err"], L_err=row["L"] * 0.1,
                n_e=1e15, zeta=2.0,
            )
            results.append({
                **row,
                "Ck_calc": r.C_k_km_s,
                "Ck_calc_err": r.C_k_err / 1e3 if r.C_k_err else None,
                "B_calc": r.B_gauss,
                "B_calc_err": r.B_gauss_err,
                "V_calc": r.V_km_s,
                "E_flux_calc": r.E_flux,
            })
        except ValueError as e:
            print(f"  Row {row['id']} failed: {e}")
    return results


def summary_stats(results, key_pub, key_calc):
    pub_vals = np.array([r[key_pub] for r in results if r.get(key_pub) is not None])
    calc_vals = np.array([r[key_calc] for r in results if r.get(key_pub) is not None])
    if len(pub_vals) == 0:
        return None
    abs_err = np.abs(calc_vals - pub_vals)
    rel_err = abs_err / pub_vals * 100
    return {
        "n": len(pub_vals),
        "mean_rel_err": np.mean(rel_err),
        "median_rel_err": np.median(rel_err),
        "max_rel_err": np.max(rel_err),
        "std_rel_err": np.std(rel_err),
        "within_1pct": np.sum(rel_err < 1.0),
        "within_5pct": np.sum(rel_err < 5.0),
        "within_10pct": np.sum(rel_err < 10.0),
        "pearson_r": np.corrcoef(pub_vals, calc_vals)[0, 1],
    }


def make_plots(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    ar_results = [r for r in results if r["region"] == "AR"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ck_pub = [r["Ck"] for r in ar_results]
    ck_calc = [r["Ck_calc"] for r in ar_results]
    axes[0].scatter(ck_pub, ck_calc, alpha=0.6, s=30, c="steelblue", edgecolors="k", linewidth=0.5)
    lim = max(max(ck_pub), max(ck_calc))
    axes[0].plot([0, lim], [0, lim], "r--", lw=1, label="y = x")
    axes[0].set_xlabel("Published $C_k$ (km/s)")
    axes[0].set_ylabel("oscfit $C_k$ (km/s)")
    axes[0].set_title(f"Kink Speed (N={len(ar_results)})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    b_pub = [r["B"] for r in ar_results]
    b_calc = [r["B_calc"] for r in ar_results]
    axes[1].scatter(b_pub, b_calc, alpha=0.6, s=30, c="orangered", edgecolors="k", linewidth=0.5)
    lim = max(max(b_pub), max(b_calc))
    axes[1].plot([0, lim], [0, lim], "r--", lw=1, label="y = x")
    axes[1].set_xlabel("Published B (G)")
    axes[1].set_ylabel("oscfit B (G)")
    axes[1].set_title(f"Magnetic Field (N={len(ar_results)})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    v_pub = [r["V"] for r in ar_results]
    v_calc = [r["V_calc"] for r in ar_results]
    axes[2].scatter(v_pub, v_calc, alpha=0.6, s=30, c="seagreen", edgecolors="k", linewidth=0.5)
    lim = max(max(v_pub), max(v_calc))
    axes[2].plot([0, lim], [0, lim], "r--", lw=1, label="y = x")
    axes[2].set_xlabel("Published V (km/s)")
    axes[2].set_ylabel("oscfit V (km/s)")
    axes[2].set_title(f"Velocity Amplitude (N={len(ar_results)})")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("oscfit Backtest vs Shrivastav et al. (2024) — Active Region Data",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "backtest_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, label, unit in zip(
        axes,
        [("Ck", "Ck_calc"), ("B", "B_calc"), ("V", "V_calc")],
        ["C_k", "B", "V"],
        ["km/s", "G", "km/s"],
    ):
        pub = np.array([r[key[0]] for r in ar_results])
        calc = np.array([r[key[1]] for r in ar_results])
        rel_err = (calc - pub) / pub * 100
        ax.hist(rel_err, bins=30, color="steelblue", edgecolor="k", alpha=0.7)
        ax.axvline(0, color="red", lw=1, ls="--")
        ax.set_xlabel(f"Relative error in {label} (%)")
        ax.set_ylabel("Count")
        mean_err = np.mean(rel_err)
        ax.set_title(f"{label}: mean = {mean_err:+.2f}%")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Relative Error Distribution", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "backtest_errors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def test_fit_recovery(rows, n_sample=20, seed=42):
    """For a sample of published parameters, synthesize signal, fit, and recover."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(rows), min(n_sample, len(rows)), replace=False)
    recoveries = []
    for idx in indices:
        row = rows[idx]
        try:
            d = make_signal(
                model="decayless",
                params={"A": row["A"], "P": row["P"], "phi": rng.uniform(0, 2 * np.pi),
                        "c0": 0.0, "c1": 0.0},
                duration=max(5 * row["P"], 300.0),
                cadence=3.0,
                snr=15.0,
                seed=int(idx),
            )
            fit = fit_oscillation(d["time"], d["signal"], model="decayless", mcmc=False)
            recoveries.append({
                "id": row["id"],
                "P_true": row["P"],
                "P_recovered": fit.params["P"],
                "P_err": fit.param_errors["P"],
                "A_true": row["A"],
                "A_recovered": fit.params["A"],
                "A_err": fit.param_errors["A"],
            })
        except (ValueError, RuntimeError) as e:
            print(f"  Row {row['id']} fit failed: {e}")
    return recoveries


def main():
    print("=" * 70)
    print("OSCFIT BACKTEST AGAINST SHRIVASTAV ET AL. (2024)")
    print("=" * 70)

    print("\n[1/4] Downloading data from GitHub...")
    ar_text = download_csv("Table2_Oscillation_parameters_active_regions.csv")
    qs_text = download_csv("Table2_Oscillation_parameters_Quiet_Sun_Coronal_holes.csv")

    ar_rows = parse_ar(ar_text)
    qs_rows = parse_qs(qs_text)
    print(f"  Active Region rows: {len(ar_rows)}")
    print(f"  Quiet Sun rows:     {len(qs_rows)}")
    print(f"  Total:              {len(ar_rows) + len(qs_rows)}")

    print("\n[2/4] Running seismology derivation on all rows...")
    all_results = backtest_seismology(ar_rows + qs_rows)
    ar_results = [r for r in all_results if r["region"] == "AR"]
    print(f"  Successfully processed: {len(all_results)}")

    print("\n[3/4] Statistical comparison vs published Active Region values:")
    print("-" * 70)
    for key_pub, key_calc, label in [
        ("Ck", "Ck_calc", "Kink speed C_k"),
        ("B", "B_calc", "Magnetic field B"),
        ("V", "V_calc", "Velocity amplitude V"),
    ]:
        stats = summary_stats(ar_results, key_pub, key_calc)
        if stats is None:
            continue
        print(f"\n  {label} (N = {stats['n']}):")
        print(f"    Mean   relative error:  {stats['mean_rel_err']:.2f}%")
        print(f"    Median relative error:  {stats['median_rel_err']:.2f}%")
        print(f"    Max    relative error:  {stats['max_rel_err']:.2f}%")
        print(f"    Std    relative error:  {stats['std_rel_err']:.2f}%")
        print(f"    Within 1% : {stats['within_1pct']:>3d} / {stats['n']}")
        print(f"    Within 5% : {stats['within_5pct']:>3d} / {stats['n']}")
        print(f"    Within 10%: {stats['within_10pct']:>3d} / {stats['n']}")
        print(f"    Pearson r : {stats['pearson_r']:.6f}")

    print("\n[4/4] MCMC fit recovery on synthetic signals using published params...")
    recoveries = test_fit_recovery(ar_rows + qs_rows, n_sample=20)
    p_errs = []
    a_errs = []
    p_within_1sig = 0
    a_within_1sig = 0
    for r in recoveries:
        p_err_pct = abs(r["P_recovered"] - r["P_true"]) / r["P_true"] * 100
        a_err_pct = abs(r["A_recovered"] - r["A_true"]) / r["A_true"] * 100
        p_errs.append(p_err_pct)
        a_errs.append(a_err_pct)
        if abs(r["P_recovered"] - r["P_true"]) < r["P_err"]:
            p_within_1sig += 1
        if abs(r["A_recovered"] - r["A_true"]) < r["A_err"]:
            a_within_1sig += 1
    print(f"  Synthetic signals fit:  {len(recoveries)}")
    print(f"  Median P recovery error: {np.median(p_errs):.2f}%")
    print(f"  Median A recovery error: {np.median(a_errs):.2f}%")
    print(f"  P within reported error: {p_within_1sig}/{len(recoveries)}")
    print(f"  A within reported error: {a_within_1sig}/{len(recoveries)}")

    output_dir = Path(__file__).parent / "backtest_output"
    print(f"\nGenerating plots in {output_dir}/")
    make_plots(all_results, output_dir)

    with open(output_dir / "backtest_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved {len(all_results)} rows to backtest_results.json")

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
