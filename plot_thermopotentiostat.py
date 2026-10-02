#!/usr/bin/env python3
"""
Plot the thermopotentiostat trajectory (paper Figure 7b), stitching together
one or more sequential run directories (e.g. a run and its restart):
  - left  axis (blue) : electrode charge  n_electrode / e-   <- Q.dat
  - right axis (black): potential  Phi - Phi_PZC / V         <- phi.dat
  - horizontal line   : target potential phi0

Usage
-----
    python plot_thermopotentiostat.py [RUN_DIR ...] [options]

Give the run directories in chronological order. Each must contain Q.dat and
phi.dat. If none are given, the current directory is used. Example for a run
with a restart nested inside it:

    python plot_thermopotentiostat.py  .  run2

The segments are concatenated into one continuous time series; the time axis
(step * POTIM) runs across all of them. POTIM and phi0 are read from the FIRST
run's INCAR / vasp_plugin.py unless overridden.
"""

import argparse
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")            # no display needed on a headless machine
import matplotlib.pyplot as plt


def load_column(path):
    """Load a one-number-per-line .dat file, skipping blank/garbage lines."""
    if not os.path.isfile(path):
        sys.exit(f"ERROR: could not find {path}")
    vals = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                vals.append(float(line.split()[0]))
            except ValueError:
                # header or non-numeric line -> skip
                continue
    if not vals:
        sys.exit(f"ERROR: no numeric data parsed from {path}")
    return np.asarray(vals)


def load_run(d):
    """Load and length-match Q.dat / phi.dat from a single run directory."""
    q = load_column(os.path.join(d, "Q.dat"))
    phi = load_column(os.path.join(d, "phi.dat"))
    n = min(len(q), len(phi))
    if len(q) != len(phi):
        print(f"  [{d}] NOTE: Q.dat={len(q)} lines, phi.dat={len(phi)}; "
              f"truncating to {n}.")
    return q[:n], phi[:n]


def grep_float(path, key):
    """Return the first float found after `key=` (INCAR) or `key =` (python)."""
    if not os.path.isfile(path):
        return None
    pat = re.compile(rf"{re.escape(key)}\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    with open(path) as fh:
        for line in fh:
            if line.strip().startswith("#"):     # ignore commented python lines
                continue
            m = pat.search(line)
            if m:
                return float(m.group(1))
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="*", default=["."],
                   help="run directories in chronological order "
                        "(each with Q.dat, phi.dat). Default: current dir.")
    p.add_argument("--potim", type=float, default=0.5,
                   help="MD timestep in fs, used if not found in INCAR (default 0.5)")
    p.add_argument("--phi0", type=float, default=None,
                   help="target potential for the horizontal line "
                        "(default: read phi0 from the first run's vasp_plugin.py)")
    p.add_argument("--tmax", type=float, default=None,
                   help="crop the x-axis to this many ps (e.g. 1.0 to match the paper)")
    p.add_argument("--smooth", type=int, default=1,
                   help="running-mean window (in steps) for display only (default 1 = none)")
    p.add_argument("--mark-restarts", action="store_true",
                   help="draw a faint vertical line at each restart boundary")
    p.add_argument("-o", "--out", default="thermopotentiostat.png",
                   help="output image filename (default thermopotentiostat.png)")
    args = p.parse_args()

    run_dirs = args.run_dirs if args.run_dirs else ["."]

    # ---- load and concatenate every segment in order ------------------------
    q_parts, phi_parts, boundaries = [], [], []
    total = 0
    for d in run_dirs:
        if not os.path.isdir(d):
            sys.exit(f"ERROR: run directory not found: {d}")
        q_d, phi_d = load_run(d)
        q_parts.append(q_d)
        phi_parts.append(phi_d)
        total += len(q_d)
        boundaries.append(total)        # step index where this segment ends
        print(f"  [{d}] {len(q_d)} steps")

    q = np.concatenate(q_parts)
    phi = np.concatenate(phi_parts)
    n = len(q)
    print(f"total: {n} steps across {len(run_dirs)} run(s)")

    # timestep: prefer the FIRST run's INCAR
    potim = grep_float(os.path.join(run_dirs[0], "INCAR"), "POTIM") or args.potim
    t = np.arange(n) * potim / 1000.0               # fs -> ps

    # target line: prefer phi0 from the FIRST run's plugin
    if args.phi0 is not None:
        phi0 = args.phi0
    else:
        phi0 = grep_float(os.path.join(run_dirs[0], "vasp_plugin.py"), "phi0")

    # optional smoothing (display only). Note: a running mean spans the restart
    # seam, which is fine since the series is physically continuous.
    def smooth(y, w):
        if w <= 1:
            return y
        k = np.ones(w) / w
        return np.convolve(y, k, mode="same")
    q_plot, phi_plot = smooth(q, args.smooth), smooth(phi, args.smooth)

    # ---- plot ---------------------------------------------------------------
    plt.rcParams.update({"font.size": 18, "axes.linewidth": 1.8,
                         "lines.linewidth": 1.6})
    fig, ax_left = plt.subplots(figsize=(8, 6))

    charge_color = "#00008B"   # dark blue, as in the paper
    pot_color = "black"

    # left axis: electrode charge (blue)
    ax_left.plot(t, q_plot, color=charge_color)
    ax_left.set_xlabel("Simulation time / ps")
    ax_left.set_ylabel(r"$n_{electrode}$ / e$^-$", color=charge_color)
    ax_left.tick_params(axis="y", colors=charge_color)
    ax_left.spines["left"].set_color(charge_color)

    # right axis: potential (black)
    ax_right = ax_left.twinx()
    ax_right.plot(t, phi_plot, color=pot_color)
    ax_right.set_ylabel(r"$\Phi - \Phi_{PZC}$ / V", color=pot_color)
    ax_right.tick_params(axis="y", colors=pot_color)

    # target potential line
    if phi0 is not None:
        ax_right.axhline(phi0, color="black", linewidth=1.5)
        print(f"target potential (horizontal line): phi0 = {phi0} V")
    else:
        print("WARNING: no phi0 found; skipping target line "
              "(pass --phi0 to draw it).")

    # optional restart markers (skip the final boundary = end of data)
    if args.mark_restarts and len(boundaries) > 1:
        for b in boundaries[:-1]:
            ax_left.axvline(b * potim / 1000.0, color="grey",
                            linewidth=1.0, linestyle=":", alpha=0.6)

    if args.tmax is not None:
        ax_left.set_xlim(0, args.tmax)
    else:
        ax_left.set_xlim(0, t[-1])

    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}  ({n} steps, POTIM={potim} fs, "
          f"t = {t[-1]:.3f} ps)")


if __name__ == "__main__":
    main()
