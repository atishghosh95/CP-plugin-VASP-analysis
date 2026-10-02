"""
freeze_layers.py
----------------
Detect atomic layers (by z-coordinate) in a slab POSCAR, from bottom to top,
and write a new POSCAR with "Selective dynamics" tags freezing whichever
bottom layers you choose.

Handles slabs that wrap across the periodic z-boundary (common when the
slab is centered near z=0/c), by finding the vacuum gap and unwrapping
z-coordinates around it before sorting.

USAGE
-----
1. Edit the CONFIG block below (input file, how many/which layers to
   freeze, clustering tolerance).
2. Run:  python3 freeze_layers.py
3. First it PRINTS the detected layers (index 0 = bottom) so you can
   check the count/grouping matches your idea of a "layer" before you
   commit. Adjust Z_TOL and re-run if layers look merged/split wrong.
4. It then writes the new POSCAR with Selective dynamics.
"""

import numpy as np

# ============================== CONFIG ===============================
INPUT_POSCAR  = "POSCAR"
OUTPUT_POSCAR = "POSCAR_frozen"

# Clustering tolerance (Angstrom): atoms whose unwrapped z differs by less
# than this from the previous atom are grouped into the same layer.
# Increase if your layer count looks too high (adjacent planes merged
# that should be one physical layer); decrease if too low.
Z_TOL = 0.5

# Which layers (bottom-to-top, 0-indexed) to FREEZE.
# Option A: freeze the bottom N layers automatically -> set N_FREEZE
# Option B: freeze specific layer indices -> set FREEZE_LAYERS explicitly
#           (overrides N_FREEZE if not None)
N_FREEZE = 6
FREEZE_LAYERS = None   # e.g. [0, 1, 2] to freeze layers 0,1,2 explicitly

# Which degrees of freedom to freeze for frozen atoms (x, y, z).
# True = frozen (F), False = allowed to relax (T)
FREEZE_DOF = (True, True, True)
# =======================================================================


def read_poscar(path):
    with open(path) as f:
        lines = f.readlines()
    comment = lines[0].rstrip("\n")
    scale = float(lines[1].split()[0])
    lattice = np.array([[float(x) for x in lines[i].split()] for i in range(2, 5)]) * scale
    elements = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    natoms = sum(counts)

    line7 = lines[7].strip()
    selective = line7.lower().startswith("s")
    coord_idx = 8 if selective else 7
    coord_type = lines[coord_idx].strip()
    start = coord_idx + 1

    positions = np.array([[float(x) for x in lines[start + i].split()[:3]] for i in range(natoms)])
    is_direct = coord_type.lower().startswith("d")
    if is_direct:
        cart_positions = positions.dot(lattice)
    else:
        cart_positions = positions.copy()

    return {
        "comment": comment, "scale": scale, "lattice": lattice,
        "elements": elements, "counts": counts, "natoms": natoms,
        "positions": cart_positions,  # always cartesian internally
    }


def detect_layers(positions, lattice, z_tol=0.5):
    """Return list of layers bottom->top: each is (atom_indices, unwrapped_z_values)."""
    z = positions[:, 2].copy()
    c = lattice[2, 2]
    n = len(z)

    order = np.argsort(z)
    zs = z[order]
    gaps = np.diff(zs)
    wrap_gap = zs[0] + c - zs[-1]
    all_gaps = np.append(gaps, wrap_gap)
    split = np.argmax(all_gaps)  # largest gap = vacuum region

    # rotate ordering so the list starts right after the vacuum gap
    rot_order = np.concatenate([order[split + 1:], order[:split + 1]])
    zs_unwrapped = z[rot_order].copy()
    for i in range(1, n):
        if zs_unwrapped[i] < zs_unwrapped[i - 1] - 1e-6:
            zs_unwrapped[i:] += c

    layers = []
    cur_idx = [rot_order[0]]
    cur_z = [zs_unwrapped[0]]
    for i in range(1, n):
        if zs_unwrapped[i] - zs_unwrapped[i - 1] > z_tol:
            layers.append((cur_idx, cur_z))
            cur_idx, cur_z = [], []
        cur_idx.append(rot_order[i])
        cur_z.append(zs_unwrapped[i])
    layers.append((cur_idx, cur_z))
    return layers


def elements_for_indices(indices, counts, elements):
    labels = []
    for i in indices:
        cum = 0
        for e, c in zip(elements, counts):
            if i < cum + c:
                labels.append(e)
                break
            cum += c
    return labels


def print_layer_summary(layers, counts, elements):
    print(f"Detected {len(layers)} layers (index 0 = bottom):\n")
    print(f"{'Layer':>5} | {'#atoms':>6} | {'z range (A)':>18} | elements")
    print("-" * 60)
    for i, (idxs, zzs) in enumerate(layers):
        labels = elements_for_indices(idxs, counts, elements)
        comp = ", ".join(f"{e}:{labels.count(e)}" for e in sorted(set(labels)))
        print(f"{i:>5} | {len(idxs):>6} | {min(zzs):>8.3f}-{max(zzs):<8.3f} | {comp}")
    print()


def write_poscar(path, data, frozen_indices, freeze_dof):
    lattice = data["lattice"]
    natoms = data["natoms"]

    with open(path, "w") as f:
        f.write(data["comment"] + "\n")
        f.write("1.0\n")
        for row in lattice:
            f.write(f"  {row[0]:20.10f}{row[1]:20.10f}{row[2]:20.10f}\n")
        f.write("   " + "   ".join(data["elements"]) + "\n")
        f.write("   " + "   ".join(str(c) for c in data["counts"]) + "\n")
        f.write("Selective dynamics\n")
        f.write("Cartesian\n")
        frozen_set = set(frozen_indices)
        for i in range(natoms):
            x, y, z = data["positions"][i]
            if i in frozen_set:
                flag = ["F" if freeze_dof[k] else "T" for k in range(3)]
            else:
                flag = ["T", "T", "T"]
            f.write(f"  {x:20.10f}{y:20.10f}{z:20.10f}   {flag[0]}   {flag[1]}   {flag[2]}\n")


def main():
    data = read_poscar(INPUT_POSCAR)
    layers = detect_layers(data["positions"], data["lattice"], z_tol=Z_TOL)
    print_layer_summary(layers, data["counts"], data["elements"])

    if FREEZE_LAYERS is not None:
        freeze_layer_ids = FREEZE_LAYERS
    else:
        freeze_layer_ids = list(range(N_FREEZE))

    frozen_indices = []
    for li in freeze_layer_ids:
        frozen_indices.extend(layers[li][0])

    print(f"Freezing layers {freeze_layer_ids} -> {len(frozen_indices)} atoms frozen "
          f"(dof frozen = {freeze_dof_str()})")

    write_poscar(OUTPUT_POSCAR, data, frozen_indices, FREEZE_DOF)
    print(f"Wrote {OUTPUT_POSCAR}")


def freeze_dof_str():
    labels = ("x", "y", "z")
    return "".join(l for l, f in zip(labels, FREEZE_DOF) if f) or "none"


if __name__ == "__main__":
    main()
