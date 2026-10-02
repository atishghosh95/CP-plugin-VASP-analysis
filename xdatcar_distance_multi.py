#!/usr/bin/env python3
"""
xdatcar_distance_multi.py
==========================

Overlay the same atom-pair distance-vs-time curve from several VASP AIMD
runs (e.g. different target potentials/fields) onto one plot, each read
from its own XDATCAR. General-purpose: works for any two atoms/species and
any set of run subdirectories, not hardcoded to Li/N or to 0.5/0.75/1.0.

Each subdirectory is expected to look like a normal VASP run directory
(XDATCAR + optionally INCAR for POTIM), e.g.:

    <basedir>/0.5/XDATCAR, INCAR, ...
    <basedir>/0.75/XDATCAR, INCAR, ...
    <basedir>/1.0/XDATCAR, INCAR, ...

------------------------------------------------------------------------
Selecting atoms
------------------------------------------------------------------------
--atom1 / --atom2 accept three forms (same as xdatcar_distance.py):

  "Li"        -> the 1st (and normally only) atom of species Li
  "O:3"       -> the 3rd atom of species O (1-based)
  "147"       -> global atom index (1-based)

------------------------------------------------------------------------
Usage
------------------------------------------------------------------------
Default case -- basedir contains 0.5/, 0.75/, 1.0/ subfolders, each with
its own XDATCAR + INCAR:

    python xdatcar_distance_multi.py --basedir . --atom1 Li --atom2 N

Custom subfolders/labels/colors, explicit output:

    python xdatcar_distance_multi.py \
        --basedir /path/to/far \
        --subdirs 0.5 0.75 1.0 \
        --labels "0.5 V" "0.75 V" "1.0 V" \
        --atom1 Li --atom2 N \
        --colors tab:blue tab:orange tab:green \
        --out li_n_multi.png \
        --dat-dir distance_data

If a subdirectory's INCAR doesn't have POTIM, pass --potim to apply the
same timestep to all of them, or edit that run's INCAR.
"""

import argparse
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# XDATCAR parsing (identical logic to xdatcar_distance.py)
# --------------------------------------------------------------------------

def read_xdatcar_header(f):
    _comment = f.readline()
    _scale = float(f.readline().split()[0])
    a1 = list(map(float, f.readline().split()[:3]))
    a2 = list(map(float, f.readline().split()[:3]))
    a3 = list(map(float, f.readline().split()[:3]))
    lattice = np.array([a1, a2, a3])
    species = f.readline().split()
    counts = list(map(int, f.readline().split()))
    natoms = sum(counts)
    return lattice, species, counts, natoms


def species_index_ranges(species, counts):
    ranges = {}
    cum = 0
    for sp, c in zip(species, counts):
        ranges[sp] = (cum, cum + c)
        cum += c
    return ranges


def resolve_atom_selector(sel, species, counts):
    sel = sel.strip()
    if sel.isdigit():
        idx = int(sel) - 1
        natoms = sum(counts)
        if not (0 <= idx < natoms):
            sys.exit(f"Atom index {sel} out of range (1..{natoms}).")
        return idx

    if ":" in sel:
        sym, n = sel.split(":", 1)
        n = int(n)
    else:
        sym, n = sel, 1

    ranges = species_index_ranges(species, counts)
    if sym not in ranges:
        sys.exit(f"Species '{sym}' not found in XDATCAR (available: "
                  f"{list(ranges.keys())}).")
    start, end = ranges[sym]
    count = end - start
    if not (1 <= n <= count):
        sys.exit(f"Requested {sym}:{n}, but there are only {count} atoms "
                  f"of species '{sym}'.")
    return start + (n - 1)


def read_potim_fs(rundir):
    incar = os.path.join(rundir, "INCAR")
    if not os.path.isfile(incar):
        return None
    with open(incar) as f:
        for line in f:
            line = line.split("#")[0].split("!")[0].strip()
            for token in line.split(";"):
                token = token.strip()
                if token.upper().startswith("POTIM"):
                    m = re.search(r"POTIM\s*=\s*([-+0-9.eEdD]+)", token, re.I)
                    if m:
                        return float(m.group(1).replace("d", "e").replace("D", "E"))
    return None


_SHIFTS = np.array([[i, j, k] for i in (-1, 0, 1)
                               for j in (-1, 0, 1)
                               for k in (-1, 0, 1)])


def min_image_distance(frac1, frac2, lattice):
    d_frac = (frac2 - frac1)[None, :] + _SHIFTS
    d_cart = d_frac @ lattice
    return np.linalg.norm(d_cart, axis=1).min()


def compute_distance_series(xdatcar_path, atom1_sel, atom2_sel, potim_fs,
                              skip=0, stride=1):
    with open(xdatcar_path) as f:
        lattice, species, counts, natoms = read_xdatcar_header(f)

    idx1 = resolve_atom_selector(atom1_sel, species, counts)
    idx2 = resolve_atom_selector(atom2_sel, species, counts)

    steps, dists = [], []
    with open(xdatcar_path) as f:
        for _ in range(7):
            f.readline()
        frame = 0
        while True:
            header_line = f.readline()
            if not header_line:
                break
            if not header_line.startswith("Direct configuration"):
                continue
            frame += 1
            frac1 = frac2 = None
            for i in range(natoms):
                line = f.readline()
                if i == idx1:
                    frac1 = np.array(list(map(float, line.split()[:3])))
                elif i == idx2:
                    frac2 = np.array(list(map(float, line.split()[:3])))
            if frac1 is None or frac2 is None:
                sys.exit(f"Failed to read coordinates for frame {frame} in "
                          f"{xdatcar_path}.")
            if frame <= skip:
                continue
            if (frame - skip - 1) % stride != 0:
                continue
            d = min_image_distance(frac1, frac2, lattice)
            steps.append(frame)
            dists.append(d)

    steps = np.array(steps)
    dists = np.array(dists)
    time_ps = steps * potim_fs / 1000.0
    return steps, time_ps, dists, species, counts


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Overlay atom-pair distance-vs-time curves from "
                    "several VASP AIMD runs (different subfolders) onto "
                    "one plot.")
    p.add_argument("--basedir", default=".",
                   help="Directory containing the run subfolders. Default: cwd.")
    p.add_argument("--subdirs", nargs="+", default=["0.5", "0.75", "1.0"],
                   help="Subfolder names under --basedir, each containing "
                        "its own XDATCAR (+ INCAR). Default: 0.5 0.75 1.0")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Legend labels, one per subdir, same order. "
                        "Default: use the subdir names themselves.")
    p.add_argument("--atom1", default="Li",
                   help="First atom selector (species, species:index, or "
                        "global index). Default: Li")
    p.add_argument("--atom2", default="N",
                   help="Second atom selector. Default: N")
    p.add_argument("--potim", type=float, default=None,
                   help="MD timestep in fs/ionic-step, applied to every "
                        "subdir. If omitted, each subdir's own INCAR is "
                        "read individually.")
    p.add_argument("--skip", type=int, default=0,
                   help="Leading configurations to drop from every run "
                        "(equilibration). Default 0.")
    p.add_argument("--stride", type=int, default=1,
                   help="Use every Nth configuration. Default 1 (all).")
    p.add_argument("--colors", nargs="+", default=None,
                   help="One matplotlib color per subdir. Default: "
                        "tab:blue, tab:orange, tab:green, ... cycle.")
    p.add_argument("--out", default="distance_multi.png",
                   help="Output plot filename. Default: distance_multi.png")
    p.add_argument("--dat-dir", default=None,
                   help="If given, save each subdir's raw (step, time_ps, "
                        "distance_A) series as <dat-dir>/<subdir>.dat")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Line opacity, useful if curves overlap heavily.")
    args = p.parse_args()

    default_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red",
                       "tab:purple", "tab:brown"]
    colors = args.colors or default_colors
    labels = args.labels or list(args.subdirs)
    if len(labels) != len(args.subdirs):
        sys.exit("--labels must have the same number of entries as --subdirs.")

    if args.dat_dir:
        os.makedirs(args.dat_dir, exist_ok=True)

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.linewidth"] = 1.4

    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    all_dists = []
    for i, sub in enumerate(args.subdirs):
        rundir = os.path.join(args.basedir, sub)
        xdatcar_path = os.path.join(rundir, "XDATCAR")
        if not os.path.isfile(xdatcar_path):
            sys.exit(f"Could not find XDATCAR in {rundir}")

        potim_fs = args.potim if args.potim is not None else read_potim_fs(rundir)
        if potim_fs is None:
            print(f"Warning: POTIM not found for '{sub}'; defaulting to "
                  f"1.0 fs/step.", file=sys.stderr)
            potim_fs = 1.0

        steps, time_ps, dists, species, counts = compute_distance_series(
            xdatcar_path, args.atom1, args.atom2, potim_fs,
            skip=args.skip, stride=args.stride)

        print(f"[{sub}] species: {list(zip(species, counts))}")
        print(f"[{sub}] frames used: {len(steps)}  POTIM={potim_fs} fs")
        print(f"[{sub}] distance (A): min={dists.min():.3f} max={dists.max():.3f} "
              f"mean={dists.mean():.3f} first={dists[0]:.3f} last={dists[-1]:.3f}")

        if args.dat_dir:
            out_path = os.path.join(args.dat_dir, f"{sub}.dat")
            np.savetxt(out_path, np.column_stack([steps, time_ps, dists]),
                       header="step  time_ps  distance_A")
            print(f"[{sub}] saved raw data to {out_path}")

        color = colors[i % len(colors)]
        ax.plot(time_ps, dists, color=color, lw=1.2, alpha=args.alpha,
                 label=labels[i])
        all_dists.append(dists)

    ax.set_xlabel("Simulation time / ps", fontweight="bold")
    ax.set_ylabel(f"{args.atom1}$-${args.atom2} distance / "
                  r"$\mathrm{\AA}$", fontweight="bold")
    ax.tick_params(width=1.4)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(True)
    leg = ax.legend(frameon=False, prop={"weight": "bold"})
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
