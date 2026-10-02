"""
Cu(111) surface phase diagrams (computational hydrogen electrode) - H only, H + Li/Na/K, H + NO3
Run from the surf-en folder:   python3 surface_energy_all.py

Folders:
  only-h/clean/OUTCAR          clean Cu(111)                 -> E(Cu), common reference
  only-h/H_xxML/OUTCAR         Cu + nH
  H-M/clean_M/OUTCAR           Cu + M (no H)                 -> mu_M = E(Cu+M) - E(Cu)
  H-M/H_xxML/OUTCAR            Cu + nH + M      (M = Li, Na, K)
  H-NO3/H_xxML/OUTCAR          Cu + nH + NO3*

Reactions / equations (A = one-face area; adsorbates on the top side only):
  H+ + e- + *  -> H*                                  ΔG per H  = ΔE + 0.24 eV + eU
  HNO3(g) + *  -> NO3* + H+ + e-                     (oxidative: -eU)
  ΔG(nH)        = E(nH) - E(Cu) - n/2 E(H2) + 0.24n + n eU
  ΔG(nH + M)    = E(nH+M) - E(Cu) - mu_M - n/2 E(H2) + 0.24n + n eU
  ΔG(nH + NO3)  = E(nH+NO3) - E(Cu) - n/2 E(H2) - [E(HNO3) - 1/2 E(H2)] + 0.24n + NO3_CORR + (n-1) eU
  γ(U)          = γ_clean + ΔG / A × 16.02  [J/m^2];  stable phase = lowest γ at each U
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================== INPUT ==============================
A = 88.667             # Å^2, one face of the 8.763 x 10.119 Å cell
GAMMA_CLEAN = 1.30     # J/m^2, bare Cu(111)
U_RANGE = (-0.3, 0.3)  # V_RHE
COV = [("0.25", 4), ("0.50", 8), ("0.75", 12), ("1.00", 16)]
CATIONS = ["Li", "Na", "K"]

# Gas references (no slab): E, ZPE, TS in eV (user table)
GAS = {"H2":   {"E": -6.770252, "ZPE": 0.274367, "TS": 0.108078},
       "HNO3": {"E": -28.61574, "ZPE": 0.709611, "TS": 0.166417}}
E_H2, E_HNO3 = GAS["H2"]["E"], GAS["HNO3"]["E"]
DZPE_TS_H = 0.24       # eV per H*, relative to 1/2 H2 (standard CHE value)
# NO3 part treated with electronic energies only (no NO3* frequencies yet).
# When NO3* ZPE/TS are available set:
#   NO3_CORR = (ZPE-TS)_NO3* - (ZPE-TS)_HNO3 + 1/2 (ZPE-TS)_H2   using GAS above
NO3_CORR = 0.0

COLORS = {"H only": "black", "Li": "#0072B2", "Na": "#E69F00", "K": "#009E73", "NO3": "#CC79A7"}  # Okabe-Ito
RAMPS = {"Li": plt.cm.Blues, "Na": plt.cm.Oranges, "K": plt.cm.Greens, "NO3": plt.cm.Purples}
# ===================================================================
CONV = 16.0218


def outcar(path):
    """(energy of last COMPLETED ionic step, converged?) or (None, None)."""
    if not os.path.isfile(path):
        return None, None
    t = open(path).read()
    blocks = t.split("FREE ENERGIE OF THE ION-ELECTRON SYSTEM")[1:]
    if not blocks:
        return None, None
    e = float(re.search(r"energy\(sigma->0\)\s*=\s*(-?\d+\.\d+)", blocks[-1]).group(1))
    return e, "reached required accuracy" in t


def style(ax, title):
    ax.set_xlim(*U_RANGE)
    ax.set_xlabel(r"Potential (V$_{\mathrm{RHE}}$)")
    ax.set_ylabel(r"Surface energy (J/m$^2$)")
    ax.set_title(title)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(1.2)
    ax.tick_params(direction="in", top=True, right=True)


def envelope(U, lines):
    names = list(lines); stack = np.vstack([lines[x] for x in names])
    stable = stack.argmin(0)
    return names, stable, stack.min(0), np.where(np.diff(stable))[0]


def phase_plot(U, lines, env_info, styles, title, out, ytop=0.25, legend_loc="lower right"):
    names, stable, env, idx = env_info
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.axhline(GAMMA_CLEAN, color="black", lw=2, label="bare Cu(111)")
    for k, st in styles.items():
        ax.plot(U, lines[k], **st)
    ymin = min(env.min() - 0.1, GAMMA_CLEAN - 0.4); ymax = GAMMA_CLEAN + ytop; span = ymax - ymin
    for j, i in enumerate(idx):
        ax.plot([U[i]] * 2, [ymin, env[i]], color="0.35", lw=1)
        ax.text(U[i] - 0.004, ymin + span * (0.03 if j % 2 == 0 else 0.33), names[stable[i]],
                rotation=90, ha="right", va="bottom", fontsize=8.5)
    ax.plot(U, env, color="black", lw=0.8, ls=":", label="most stable")
    ax.set_ylim(ymin, ymax); style(ax, title)
    ax.legend(fontsize=7.5, frameon=False, loc=legend_loc)
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    print(f"Saved {out}")


def report(tag, env_info, U):
    names, stable, _, idx = env_info
    print(f"{tag}: " + ", ".join(f"{names[stable[i]]} -> {names[stable[i+1]]} at {U[i]:+.3f} V" for i in idx))


def main():
    U = np.linspace(*U_RANGE, 3001)
    E_cu, _ = outcar("only-h/clean/OUTCAR")
    bare = np.full_like(U, GAMMA_CLEAN)
    gam = lambda dG0, ne: GAMMA_CLEAN + (dG0 + ne * U) / A * CONV
    envs = {}

    # ---------------- H only ----------------
    EH = {n: outcar(f"only-h/H_{c}ML/OUTCAR")[0] for c, n in COV}
    H_lines = {f"{c} ML": gam(EH[n] - E_cu - n * 0.5 * E_H2 + n * DZPE_TS_H, n) for c, n in COV}
    info = envelope(U, {"bare": bare, **H_lines}); envs["H only"] = info[2]; report("H only", info, U)

    # ---------------- H + cation ----------------
    for M in CATIONS:
        e_m, _ = outcar(f"H-{M}/clean_{M}/OUTCAR")
        E = {n: outcar(f"H-{M}/H_{c}ML/OUTCAR")[0] for c, n in COV}
        if any(v is None for v in E.values()):
            print(f"{M}: missing H+{M} OUTCARs, skipped"); continue
        mu = (e_m - E_cu) if e_m is not None else E[4] - EH[4]   # provisional until clean_M exists
        if e_m is None:
            print(f"{M}: clean_{M} missing -> provisional mu_M = E(4H+{M}) - E(4H) = {mu:.3f} eV")
        lines = {f"{c} ML": gam(E[n] - E_cu - mu - n * 0.5 * E_H2 + n * DZPE_TS_H, n) for c, n in COV}
        info = envelope(U, {"bare": bare, **lines}); envs[M] = info[2]; report(M, info, U)
        st = {f"{c} ML": dict(color=col, lw=2, label=f"{c} ML H* + {M}")
              for (c, _), col in zip(COV, RAMPS[M](np.linspace(0.35, 0.95, 4)))}
        phase_plot(U, {"bare": bare, **lines}, info, st, f"Cu(111) + H* (hcp) + {M}",
                   f"H-{M}/surface_energy_H_{M}.png")

    # ---------------- H + NO3 ----------------
    no3_lines, no3_st = {}, {}
    cols = RAMPS["NO3"](np.linspace(0.45, 0.95, 4))
    print("H + NO3 (ΔG at 0 V_RHE; NO3 part electronic only):")
    for (c, n), col in zip(COV, cols):
        e, conv = outcar(f"H-NO3/H_{c}ML/OUTCAR")
        nions = None
        if e is not None:
            m = re.search(r"NIONS\s*=\s*(\d+)", open(f"H-NO3/H_{c}ML/OUTCAR").read())
            nions = int(m.group(1)) if m else None
        if e is None or nions != 96 + n + 4:
            print(f"  {c} ML: OUTCAR missing or wrong system (NIONS={nions}, expected {100 + n}) -> skipped")
            continue
        dG0 = (e - E_cu - n * 0.5 * E_H2 - (E_HNO3 - 0.5 * E_H2) + n * DZPE_TS_H + NO3_CORR)
        dG_no3 = dG0 - (EH[n] - E_cu - n * 0.5 * E_H2 + n * DZPE_TS_H)   # NO3 added onto the nH surface
        print(f"  {c} ML: ΔG = {dG0:+.3f} eV | NO3 on {c} ML H: ΔG = {dG_no3:+.3f} - eU "
              f"| {'converged' if conv else 'NOT converged (last ionic step used)'}")
        k = f"{c} ML + NO3"
        no3_lines[k] = gam(dG0, n - 1)
        no3_st[k] = dict(color=col, lw=2, ls="-" if conv else "--",
                         label=f"{c} ML H* + NO$_3$*" + ("" if conv else " (not conv.)"))
    if no3_lines:
        allL = {"bare": bare, **H_lines, **no3_lines}
        info = envelope(U, allL); envs["NO3"] = info[2]; report("H + NO3", info, U)
        st = {f"{c} ML": dict(color="0.75", lw=1.2, label="H* only" if c == "0.25" else None) for c, _ in COV}
        st.update(no3_st)
        phase_plot(U, allL, info, st, "Cu(111) + H* (hcp) + NO$_3$*", "H-NO3/surface_energy_H_NO3.png",
                   ytop=0.55, legend_loc="upper left")

    # ---------------- combined ----------------
    fig, ax = plt.subplots(figsize=(6, 4.6))
    lab = {"H only": "H only", "Li": "H + Li", "Na": "H + Na", "K": "H + K", "NO3": "H + NO$_3$"}
    for k, env in envs.items():
        ax.plot(U, env, color=COLORS[k], lw=2.2, label=lab[k])
    ax.axhline(GAMMA_CLEAN, color="0.6", lw=1, ls="--", label="bare Cu(111)")
    style(ax, "Cu(111): most stable surface energy")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig("surface_energy_H_all.png", dpi=300); plt.close(fig)
    print("Saved surface_energy_H_all.png")

    # ---------------- H-only plot (only-h folder) ----------------
    st = {f"{c} ML": dict(color=col, lw=2, label=f"{c} ML H*")
          for (c, _), col in zip(COV, plt.cm.Blues(np.linspace(0.35, 0.95, 4)))}
    phase_plot(U, {"bare": bare, **H_lines}, envelope(U, {"bare": bare, **H_lines}), st,
               "Cu(111) + H* (hcp)", "only-h/surface_energy_H_Cu111.png")


if __name__ == "__main__":
    main()
