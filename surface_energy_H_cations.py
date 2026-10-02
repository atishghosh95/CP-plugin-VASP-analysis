"""
Cu(111) + H* (hcp) with a near-surface alkali cation (Li, Na, K) - CHE surface phase diagrams
Run from the surf-en folder:   python3 surface_energy_H_cations.py

Folder layout expected:
  only-h/clean/OUTCAR               clean Cu(111)            -> common reference E(Cu)
  only-h/H_xxML/OUTCAR              Cu + nH
  H-M/clean_M/OUTCAR                Cu + M, no H             -> mu_M = E(Cu+M) - E(Cu)
  H-M/H_xxML/OUTCAR                 Cu + nH + M

Equations (A = one-face area, H on top side only):
  dG(n, U)  = E(nH[+M]) - E(Cu) - mu_M - n/2 E(H2) + n*0.24 + n*e*U
  gamma(U)  = gamma_clean + dG(n, U) / A * 16.02           [J/m^2]
  transition n1 -> n2 at U = -dG_diff,
  dG_diff   = [E(n2) - E(n1)]/(n2 - n1) - 1/2 E(H2) + 0.24   (mu_M cancels)

Outputs:
  dG_diff_vs_coverage.png        always (needs no clean_M; 0->0.25 ML point only when clean_M exists)
  H-M/surface_energy_H_<M>.png   one per cation (provisional mu_M until H-M/clean_M/OUTCAR exists)
  surface_energy_H_all.png       most-stable (lowest) surface energy for H-only, Li, Na, K
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================== INPUT ==============================
A = 88.667            # Å^2
GAMMA_CLEAN = 1.30    # J/m^2, bare Cu(111)
E_H2 = -6.77          # eV
DZPE_TS = 0.24        # eV per H
U_RANGE = (-0.3, 0.3) # V_RHE
COV = [("0.25", 4), ("0.50", 8), ("0.75", 12), ("1.00", 16)]
CATIONS = ["Li", "Na", "K"]
COLORS = {"H only": "black", "Li": "#0072B2", "Na": "#E69F00", "K": "#009E73"}   # Okabe-Ito (CVD-safe)
RAMPS = {"Li": plt.cm.Blues, "Na": plt.cm.Oranges, "K": plt.cm.Greens}
# ===================================================================
CONV = 16.0218


def energy(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        v = re.findall(r"energy\(sigma->0\)\s*=\s*(-?\d+\.\d+)", f.read())
    return float(v[-1]) if v else None


def style(ax, title):
    ax.set_xlim(*U_RANGE)
    ax.set_xlabel(r"Potential (V$_{\mathrm{RHE}}$)")
    ax.set_ylabel(r"Surface energy (J/m$^2$)")
    ax.set_title(title)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(1.2)
    ax.tick_params(direction="in", top=True, right=True)


def main():
    E_cu = energy("only-h/clean/OUTCAR")
    systems = {"H only": {"ref": 0.0, "E": {n: energy(f"only-h/H_{c}ML/OUTCAR") for c, n in COV}}}
    for M in CATIONS:
        e_m = energy(f"H-{M}/clean_{M}/OUTCAR")
        E = {n: energy(f"H-{M}/H_{c}ML/OUTCAR") for c, n in COV}
        if e_m is not None:
            ref, prov = e_m - E_cu, False                                 # exact: mu_M = E(Cu+M) - E(Cu)
        else:
            # PROVISIONAL until H-M/clean_M/OUTCAR exists: assume the cation does not change the
            # 0 -> 0.25 ML step (supported by the identical 0.25 -> 0.50 ML step), i.e.
            # mu_M = E(Cu+4H+M) - E(Cu+4H)
            ref, prov = E[4] - systems["H only"]["E"][4], True
            print(f"{M}: clean_{M} OUTCAR missing -> PROVISIONAL mu_M = E(4H+{M}) - E(4H) = {ref:.3f} eV")
        systems[M] = {"ref": ref, "prov": prov, "E": E}

    # ---------- differential dG per added H (mu_M cancels except for the first step) ----------
    print("Differential dG per added H (eV)  = -(transition potential, V_RHE)")
    print(f"{'step':14s}" + "".join(f"{k:>10s}" for k in systems))
    steps = [(0, 4)] + [(COV[i][1], COV[i + 1][1]) for i in range(3)]
    diff = {k: [] for k in systems}
    for n1, n2 in steps:
        row = f"{n1/16:.2f}->{n2/16:.2f} ML "
        for k, s in systems.items():
            exact = not s.get("prov", False)
            e1 = (E_cu + s["ref"]) if (n1 == 0 and exact) else s["E"].get(n1)
            e2 = s["E"][n2]
            val = None if (e1 is None or e2 is None) else (e2 - e1) / (n2 - n1) - 0.5 * E_H2 + DZPE_TS
            diff[k].append(val)
            row += f"{'  pending' if val is None else f'{val:10.3f}'}"
        print(row)

    fig, ax = plt.subplots(figsize=(6, 4.4))
    xs = [(a + b) / 32 for a, b in steps]           # midpoint coverage of each step
    for k, vals in diff.items():
        pts = [(x, v) for x, v in zip(xs, vals) if v is not None]
        ax.plot(*zip(*pts), "o-", color=COLORS[k], lw=2, ms=7, label=k if k == "H only" else f"H + {k}")
    ax.axhline(0, color="0.5", lw=0.8, ls="--")
    ax.set_xlabel("H coverage (ML, midpoint of each 0.25 ML step)")
    ax.set_ylabel(r"$\Delta G$ per added H (eV)")
    ax.set_xticks(xs, ["0→0.25", "0.25→0.5", "0.5→0.75", "0.75→1"])
    ax.set_title("Cu(111): H binding vs coverage and cation")
    ax.legend(frameon=False, fontsize=9)
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)
    ax.tick_params(direction="in", top=True, right=True)
    fig.tight_layout(); fig.savefig("dG_diff_vs_coverage.png", dpi=300); plt.close(fig)
    print("Saved dG_diff_vs_coverage.png")

    # ---------- full surface-energy diagrams ----------
    U = np.linspace(*U_RANGE, 2001)
    envelopes = {}
    for k, s in systems.items():
        lines = {"bare": np.full_like(U, GAMMA_CLEAN)}
        for c, n in COV:
            dG = s["E"][n] - E_cu - s["ref"] - n * 0.5 * E_H2 + n * DZPE_TS
            lines[f"{c} ML"] = GAMMA_CLEAN + (dG + n * U) / A * CONV
        names = list(lines); stack = np.vstack([lines[x] for x in names])
        stable = stack.argmin(0); env = stack.min(0); envelopes[k] = env
        idx = np.where(np.diff(stable))[0]
        print(f"{k}: " + ", ".join(f"{names[stable[i]]}->{names[stable[i+1]]} at {U[i]:+.3f} V" for i in idx))
        if k == "H only":
            continue                                   # already plotted in only-h/
        fig, ax = plt.subplots(figsize=(6, 4.6))
        ax.axhline(GAMMA_CLEAN, color="black", lw=2, label="bare Cu(111)")
        for (c, _), col in zip(COV, RAMPS[k](np.linspace(0.35, 0.95, 4))):
            ax.plot(U, lines[f"{c} ML"], color=col, lw=2, label=f"{c} ML H* + {k}")
        ymin = min(env.min() - 0.1, GAMMA_CLEAN - 0.4); ymax = GAMMA_CLEAN + 0.25; span = ymax - ymin
        for j, i in enumerate(idx):
            ax.plot([U[i]] * 2, [ymin, env[i]], color="0.35", lw=1)
            ax.text(U[i] - 0.004, ymin + span * (0.03 if j % 2 == 0 else 0.30), names[stable[i]],
                    rotation=90, ha="right", va="bottom", fontsize=9)
        ax.plot(U, env, color="black", lw=0.8, ls=":", label="most stable")
        ax.set_ylim(ymin, ymax)
        style(ax, f"Cu(111) + H* (hcp) + {k}")
        ax.legend(fontsize=8, frameon=False, loc="lower right")
        out = f"H-{k}/surface_energy_H_{k}.png"
        fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
        print(f"Saved {out}")

    if len(envelopes) > 1:
        fig, ax = plt.subplots(figsize=(6, 4.6))
        for k, env in envelopes.items():
            ax.plot(U, env, color=COLORS[k], lw=2.2, label=k if k == "H only" else f"H + {k}")
        ax.axhline(GAMMA_CLEAN, color="0.6", lw=1, ls="--", label="bare Cu(111)")
        prov = any(systems[k].get("prov", False) for k in envelopes)
        style(ax, "Cu(111): most stable surface energy")
        ax.legend(fontsize=9, frameon=False, loc="lower right")
        fig.tight_layout(); fig.savefig("surface_energy_H_all.png", dpi=300); plt.close(fig)
        print("Saved surface_energy_H_all.png")


if __name__ == "__main__":
    main()
