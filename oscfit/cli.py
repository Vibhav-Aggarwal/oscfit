"""Command-line interface for oscfit.

Usage:
    oscfit fit <signal.csv> [--model MODEL] [--mcmc] [--output FILE]
    oscfit version

The CSV signal file must have two columns: time (s), displacement (km).
"""

import argparse
import sys
import numpy as np

from oscfit import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="oscfit",
        description="Coronal loop oscillation analysis",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fit_p = sub.add_parser("fit", help="Fit an oscillation model to a CSV time series")
    fit_p.add_argument("input", help="CSV file with columns: time, signal")
    fit_p.add_argument("--model", default="decayless",
                       choices=["decayless", "exponential", "gaussian",
                                "two_regime", "period_varying", "growing"])
    fit_p.add_argument("--period-hint", type=float, default=None,
                       help="Approximate period (s) — auto-detected if omitted")
    fit_p.add_argument("--mcmc", action="store_true",
                       help="Run MCMC for uncertainty estimation")
    fit_p.add_argument("--seed", type=int, default=None,
                       help="MCMC random seed for reproducibility")
    fit_p.add_argument("--output", default=None,
                       help="Output CSV file for fitted parameters")

    seismo_p = sub.add_parser("seismo",
                              help="Derive physical parameters from oscillation fit")
    seismo_p.add_argument("--period", type=float, required=True, help="Period (s)")
    seismo_p.add_argument("--amplitude", type=float, required=True,
                          help="Displacement amplitude (km)")
    seismo_p.add_argument("--length", type=float, required=True,
                          help="Loop length (Mm)")
    seismo_p.add_argument("--n-e", type=float, default=1e15,
                          help="Electron density (m^-3), default 1e15")
    seismo_p.add_argument("--zeta", type=float, default=2.0,
                          help="Density contrast ratio, default 2.0")

    sub.add_parser("version", help="Print oscfit version and exit")

    args = parser.parse_args(argv)

    if args.command == "version":
        print(f"oscfit {__version__}")
        return 0

    if args.command == "fit":
        return _cmd_fit(args)

    if args.command == "seismo":
        return _cmd_seismo(args)

    return 1


def _cmd_fit(args):
    from oscfit.fit import fit_oscillation

    data = np.loadtxt(args.input, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] < 2:
        print(f"ERROR: input must be CSV with at least 2 columns (time, signal)",
              file=sys.stderr)
        return 2

    time, signal = data[:, 0], data[:, 1]

    try:
        result = fit_oscillation(
            time, signal,
            model=args.model,
            period_hint=args.period_hint,
            mcmc=args.mcmc,
            seed=args.seed,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3

    print(result.summary())

    if args.output:
        row = result.to_dict()
        with open(args.output, "w") as f:
            f.write(",".join(row.keys()) + "\n")
            f.write(",".join(str(v) for v in row.values()) + "\n")
        print(f"\nWrote {args.output}")

    return 0


def _cmd_seismo(args):
    from oscfit.seismology import derive_parameters

    try:
        result = derive_parameters(
            P=args.period, A=args.amplitude, L=args.length,
            n_e=args.n_e, zeta=args.zeta,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3

    print(result.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
