"""
Kinetic Monte Carlo Water Solvation for Cu(111) + Pre-Adsorbed H (Single-Sided Slab)
======================================================================================

This is the single-sided counterpart to the RuO2(110):Mn double-sided script --
built for a plain Cu(111) slab (bare metal, bottom fused to the periodic image
below, only the top exposed to vacuum) that already has one adsorbed H atom
sitting on the surface (e.g. a reaction intermediate for HER).

By default this now fills the ENTIRE vacuum region at bulk water density
(KMC monolayer near the surface + Stage-2 bulk fill of everything above it),
matching the "solvate the whole vacuum, not just a thin layer" approach used
for the RuO2 case.

Key points specific to this file:
  - No Selective Dynamics in the input -> none is written on output either
    (if your input DID have it, the script still handles it generally).
  - The single pre-existing adsorbed H atom is treated as part of the fixed
    framework: it repels new water via the same clash checks as the Cu atoms,
    AND candidate KMC sites too close to it are pre-filtered out so the KMC
    simulation doesn't waste time trying to occupy an already-blocked site.
  - Vacuum is found generically (largest gap in sorted, periodic-wrapped z),
    same method as the RuO2 script, so this works whether the slab sits at
    z=0, mid-cell, or straddles the boundary.

Edit the USER PARAMETERS section below, then run:

    python3 solvate_cu_full_vacuum.py
"""

import os
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.spatial.transform import Rotation

# =====================================================================
# 1. USER PARAMETERS
# =====================================================================

INPUT_POSCAR   = "POSCAR"
OUTPUT_POSCAR  = "Cu111_solvated_CONTCAR"

# --- Stage 1: KMC monolayer on the exposed (top) face ---
ADSORPTION_HEIGHT = 2.3     # A, water O height above the top Cu layer
SITE_EXCLUSION_R  = 3.0     # A, minimum O-O spacing between adsorbed waters
ADSORBATE_EXCLUSION_R = 3.0 # A, minimum distance a candidate site must keep from any
                            # pre-existing non-Cu adsorbate (e.g. this structure's H atom)
K_ADS          = 1.0
E_BIND_EV      = 0.30
TEMPERATURE_K  = 300.0
MAX_KMC_STEPS  = 20_000
KMC_SEED       = 7

# --- Stage 2: fill the REST of the vacuum with bulk-density water ---
AUTO_FILL_BULK     = True   # fills the whole remaining vacuum at bulk density by default
N_BULK_WATERS      = 0      # only used if AUTO_FILL_BULK = False
BULK_TOP_MARGIN    = 2.0    # A of empty space to leave at the very top of the box (good practice)
BULK_MAX_ATTEMPTS  = 400_000
BULK_SEED          = 42

OH_BOND        = 0.9572
HOH_ANGLE_DEG  = 104.52

ELEMENT_RADII = {
    "Cu": 1.4, "Ru": 1.5, "Mn": 1.5, "C": 1.5, "N": 1.4, "O": 1.3, "H": 1.0,
    "S": 1.6, "F": 1.3, "Cl": 1.7, "B": 1.5, "P": 1.6,
}
DEFAULT_RADIUS = 1.5

# =====================================================================
# END OF USER PARAMETERS
# =====================================================================

print("Current working directory:", os.getcwd())
print("Looking for input file at:", os.path.abspath(INPUT_POSCAR))
if not os.path.isfile(INPUT_POSCAR):
    raise FileNotFoundError(f"Could not find '{INPUT_POSCAR}' in {os.getcwd()}.")


# ---------------------------------------------------------------------
# 2. Generic reader (handles optional Selective Dynamics either way)
# ---------------------------------------------------------------------

def read_poscar(path):
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f]
    comment = lines[0]
    scale = float(lines[1].split()[0])
    lat = np.array([[float(x) for x in lines[i].split()] for i in (2, 3, 4)])
    if scale < 0:
        raise ValueError("Negative (volume-based) scale factor is not supported.")
    lat = lat * scale

    tokens = lines[5].split()
    try:
        counts = [int(t) for t in tokens]
        species = [f"X{i+1}" for i in range(len(counts))]
        idx = 6
        print("Warning: VASP4-style POSCAR (no species line); elements auto-named X1, X2, ...")
    except ValueError:
        species = tokens
        counts = [int(t) for t in lines[6].split()]
        idx = 7

    natoms = sum(counts)
    line = lines[idx].strip()
    selective = line[:1].lower() == "s"
    if selective:
        idx += 1
        line = lines[idx].strip()
    is_direct = line[:1].lower() in ("d",)
    idx += 1

    coords, sd_flags = [], []
    for i in range(idx, idx + natoms):
        parts = lines[i].split()
        coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
        sd_flags.append(tuple(parts[3:6]) if selective else None)
    coords = np.array(coords)

    if not is_direct:
        coords = coords * scale
        coords = coords @ np.linalg.inv(lat)

    return {
        "comment": comment, "lattice": lat, "species": species, "counts": counts,
        "frac": coords, "selective": selective, "sd_flags": sd_flags,
    }


def dedup_struct(struct, tol=1e-4):
    species, counts, frac, sd_flags = struct["species"], struct["counts"], struct["frac"], struct["sd_flags"]
    new_frac_blocks, new_flag_blocks, new_counts = [], [], []
    total_removed, i = 0, 0
    for sp, n in zip(species, counts):
        block, flags = frac[i:i + n], sd_flags[i:i + n]
        rounded = np.round(block, 4)
        seen, keep_idx = {}, []
        for k, row in enumerate(map(tuple, rounded)):
            if row not in seen:
                seen[row] = k
                keep_idx.append(k)
        total_removed += n - len(keep_idx)
        new_frac_blocks.append(block[keep_idx])
        new_flag_blocks.append([flags[k] for k in keep_idx])
        new_counts.append(len(keep_idx))
        i += n
    if total_removed > 0:
        print(f"\n*** WARNING: {total_removed} exactly-duplicated atom(s) removed. Check your source file. ***")
    new_struct = dict(struct)
    new_struct["frac"] = np.vstack(new_frac_blocks)
    new_struct["sd_flags"] = [f for block in new_flag_blocks for f in block]
    new_struct["counts"] = new_counts
    return new_struct


struct = read_poscar(INPUT_POSCAR)
print("\n--- Framework read from", INPUT_POSCAR, "---")
print("Species:", struct["species"], " Counts:", struct["counts"], " Total:", sum(struct["counts"]))
print("Selective dynamics present:", struct["selective"])
struct = dedup_struct(struct)
print("Total after de-duplication check:", sum(struct["counts"]))

lat = struct["lattice"]
a_vec, b_vec, c_vec = lat[0], lat[1], lat[2]
c_len = np.linalg.norm(c_vec)
cart = struct["frac"] @ lat
area = np.linalg.norm(np.cross(a_vec, b_vec))

elem_list = []
for sp, n in zip(struct["species"], struct["counts"]):
    elem_list += [sp] * n
elem_arr = np.array(elem_list)
missing = sorted(set(elem_list) - set(ELEMENT_RADII))
if missing:
    print(f"Note: no radius for {missing}; using DEFAULT_RADIUS={DEFAULT_RADIUS}.")
fw_radii = np.array([ELEMENT_RADII.get(e, DEFAULT_RADIUS) for e in elem_list])


# ---------------------------------------------------------------------
# 3. Generic periodic-aware vacuum-gap detection + single/double-sided check
# ---------------------------------------------------------------------

zf = struct["frac"][:, 2]
order = np.argsort(zf)
zf_sorted = zf[order]
gaps = np.diff(zf_sorted)
wrap_gap = (zf_sorted[0] + 1.0) - zf_sorted[-1]
all_gaps = np.append(gaps, wrap_gap)
gap_idx = np.argmax(all_gaps)
vacuum_frac = all_gaps[gap_idx]
second_largest_frac = np.sort(all_gaps)[-2]

if gap_idx < len(gaps):
    lo_frac, hi_frac = zf_sorted[gap_idx], zf_sorted[gap_idx + 1]
else:
    lo_frac, hi_frac = zf_sorted[-1], zf_sorted[0] + 1.0

vacuum_thickness = vacuum_frac * c_len
second_largest_A = second_largest_frac * c_len
print(f"\nVacuum gap found: {vacuum_thickness:.3f} A "
      f"(next-largest internal gap: {second_largest_A:.3f} A)")

# heuristic: if the second-largest gap is comparable to normal interlayer spacing
# (a few A), the framework is one contiguous block -> single-sided slab.
# If it's also large (comparable to the vacuum itself), the framework is split
# into two separate chunks -> double-sided slab (see the RuO2 script for that case).
double_sided = second_largest_A > 0.3 * vacuum_thickness
top_face_z = lo_frac * c_len
print(f"Top (exposed) face at z = {top_face_z:.3f} A")
print("Slab type:", "double-sided (unexpected for this input -- check your structure!)"
      if double_sided else "single-sided (vacuum on top only)")


# ---------------------------------------------------------------------
# 4. Build the KMC hollow-site lattice from the top Cu layer,
#    pre-filtering out any site too close to a pre-existing adsorbate
# ---------------------------------------------------------------------

Cu_mask = elem_arr == "Cu"
cu_z_sorted = np.sort(cart[Cu_mask][:, 2])
# robustly find the top atomic layer: walk down from the max z while the gap
# between consecutive atoms stays much smaller than the interlayer spacing
top_cu_z = cu_z_sorted[-1]
layer_lo = top_cu_z
for i in range(len(cu_z_sorted) - 2, -1, -1):
    if top_cu_z - cu_z_sorted[i] < 1.0:   # well under the ~2 A interlayer spacing
        layer_lo = cu_z_sorted[i]
    else:
        break
top_layer_tol = (top_cu_z - layer_lo) + 0.05  # a little slack
top_xy_raw = cart[Cu_mask & (cart[:, 2] >= layer_lo - 1e-6)][:, :2]
top_xy = np.unique(np.round(top_xy_raw, 4), axis=0)
print(f"\nTop Cu layer z-range: [{layer_lo:.3f}, {top_cu_z:.3f}] A "
      f"({len(top_xy_raw)} atoms, {len(top_xy)} unique positions)")

a2, b2 = a_vec[:2], b_vec[:2]
shifts2d = [n1 * a2 + n2 * b2 for n1 in (-1, 0, 1) for n2 in (-1, 0, 1)]
pts_ext = np.vstack([top_xy + s for s in shifts2d])
tri = Delaunay(pts_ext)

nn_tree = cKDTree(top_xy)
nn_d, _ = nn_tree.query(top_xy, k=min(2, len(top_xy)))
nn_dist = np.median(nn_d[:, 1])
edge_cutoff = nn_dist * 1.5

centroids = []
for simplex in tri.simplices:
    p = pts_ext[simplex]
    edges = [np.linalg.norm(p[0] - p[1]), np.linalg.norm(p[1] - p[2]), np.linalg.norm(p[0] - p[2])]
    if max(edges) > edge_cutoff:
        continue
    centroids.append(p.mean(axis=0))
centroids = np.array(centroids)

a_len, b_len = np.linalg.norm(a_vec), np.linalg.norm(b_vec)
mask = ((centroids[:, 0] >= -1e-6) & (centroids[:, 0] < a_len - 1e-6) &
        (centroids[:, 1] >= -1e-6) & (centroids[:, 1] < b_len - 1e-6))
sites_xy = np.unique(np.round(centroids[mask], 3), axis=0)
print(f"Candidate hollow sites before adsorbate filtering: {len(sites_xy)}")

# pre-filter: remove sites too close (3D, at the intended adsorption z) to any
# pre-existing NON-Cu framework atom (e.g. the adsorbed H)
z_ads = top_cu_z + ADSORPTION_HEIGHT
other_atoms_mask = ~Cu_mask
if other_atoms_mask.any():
    other_cart = cart[other_atoms_mask]
    shifts_full3 = np.array([n1 * a_vec + n2 * b_vec for n1 in (-1, 0, 1) for n2 in (-1, 0, 1)])
    keep = []
    for i, (x, y) in enumerate(sites_xy):
        pt = np.array([x, y, z_ads])
        d = pt[None, None, :] - other_cart[:, None, :] - shifts_full3[None, :, :]
        dmin = np.linalg.norm(d, axis=2).min()
        if dmin >= ADSORBATE_EXCLUSION_R:
            keep.append(i)
    removed = len(sites_xy) - len(keep)
    sites_xy = sites_xy[keep]
    print(f"Removed {removed} site(s) too close to pre-existing adsorbate(s); "
          f"{len(sites_xy)} candidate sites remain.")

n_sites = len(sites_xy)
print(f"Water O height for adsorbed layer: z = {z_ads:.3f} A")


# ---------------------------------------------------------------------
# 5. Gillespie KMC monolayer fill
# ---------------------------------------------------------------------

KB = 8.617333e-5
K_DES = K_ADS * np.exp(-E_BIND_EV / (KB * TEMPERATURE_K))
print(f"\nk_ads = {K_ADS:.3g}, k_des = {K_DES:.3g}  (ratio {K_DES/K_ADS:.3g})")

shifts_full2d = np.array([n1 * a2 + n2 * b2 for n1 in (-1, 0, 1) for n2 in (-1, 0, 1)])


def run_kmc(sites_xy, seed):
    n = len(sites_xy)
    if n == 0:
        return np.array([], dtype=bool)
    rng_kmc = np.random.default_rng(seed)

    def min_site_dist(i, occupied_idx):
        if len(occupied_idx) == 0:
            return np.inf
        d = sites_xy[i][None, None, :] - sites_xy[occupied_idx][:, None, :] - shifts_full2d[None, :, :]
        return np.linalg.norm(d, axis=2).min()

    occ = np.zeros(n, dtype=bool)
    for step in range(MAX_KMC_STEPS):
        occ_idx = np.where(occ)[0]
        empty_idx = np.where(~occ)[0]
        eligible = [i for i in empty_idx if min_site_dist(i, occ_idx) >= SITE_EXCLUSION_R]
        n_e, n_o = len(eligible), len(occ_idx)
        total_rate = n_e * K_ADS + n_o * K_DES
        if total_rate == 0:
            break
        r = rng_kmc.random() * total_rate
        if r < n_e * K_ADS:
            occ[rng_kmc.choice(eligible)] = True
        else:
            occ[rng_kmc.choice(occ_idx)] = False
    return occ


occ = run_kmc(sites_xy, KMC_SEED)
monolayer_sites = sites_xy[occ] if n_sites else np.empty((0, 2))
print(f"KMC monolayer: {len(monolayer_sites)} / {n_sites} sites occupied")


# ---------------------------------------------------------------------
# 6. Build rigid water at each KMC site, with full clash-avoidance
# ---------------------------------------------------------------------

def water_local_geom(oh=OH_BOND, angle_deg=HOH_ANGLE_DEG):
    ang = np.deg2rad(angle_deg) / 2
    H1 = np.array([oh * np.cos(ang),  oh * np.sin(ang), 0.0])
    H2 = np.array([oh * np.cos(ang), -oh * np.sin(ang), 0.0])
    return H1, H2


H1_local, H2_local = water_local_geom()
O_RADIUS, H_RADIUS = ELEMENT_RADII["O"], ELEMENT_RADII["H"]
shifts_full3 = np.array([n1 * a_vec + n2 * b_vec for n1 in (-1, 0, 1) for n2 in (-1, 0, 1)])


def min_image_dists_full(point, ref_points):
    if len(ref_points) == 0:
        return np.array([np.inf])
    d = point[None, None, :] - ref_points[:, None, :] - shifts_full3[None, :, :]
    return np.linalg.norm(d, axis=2).min(axis=1)


rng_orient = np.random.default_rng(KMC_SEED + 1)
placed_O, placed_H = [], []


def place_water_at_site(x, y, z):
    """Fixed xy (from a KMC site): search orientations only, with a best-effort
    fallback (correctly initialized to -inf so it always records something,
    even at a severely crowded site)."""
    O_pos = np.array([x, y, z])
    best_upright, best_any, fallback, fallback_score = None, None, None, -np.inf

    for _ in range(500):
        rot = Rotation.random(random_state=rng_orient)
        H1 = O_pos + rot.apply(H1_local)
        H2 = O_pos + rot.apply(H2_local)

        po = np.array(placed_O) if placed_O else np.empty((0, 3))
        pH = np.array(placed_H).reshape(-1, 3) if placed_H else np.empty((0, 3))

        margins = []
        for pt, R in [(O_pos, O_RADIUS), (H1, H_RADIUS), (H2, H_RADIUS)]:
            d = min_image_dists_full(pt, cart); margins.append((d - (R + fw_radii)).min())
        d = min_image_dists_full(O_pos, po); margins.append((d - 2 * O_RADIUS).min() if len(d) else 1e9)
        d = min_image_dists_full(O_pos, pH); margins.append((d - (O_RADIUS + H_RADIUS)).min() if len(d) else 1e9)
        d = min_image_dists_full(H1, po);    margins.append((d - (H_RADIUS + O_RADIUS)).min() if len(d) else 1e9)
        d = min_image_dists_full(H2, po);    margins.append((d - (H_RADIUS + O_RADIUS)).min() if len(d) else 1e9)
        d = min_image_dists_full(H1, pH);    margins.append((d - 2 * H_RADIUS).min() if len(d) else 1e9)
        d = min_image_dists_full(H2, pH);    margins.append((d - 2 * H_RADIUS).min() if len(d) else 1e9)
        worst = min(margins)

        if worst > fallback_score:
            fallback_score, fallback = worst, (H1, H2)

        upright = H1[2] >= O_pos[2] - 0.3 and H2[2] >= O_pos[2] - 0.3
        if worst >= 0:
            best_any = (H1, H2)
            if upright:
                best_upright = (H1, H2)
                break

    chosen = best_upright or best_any or fallback
    if best_upright is None and best_any is None:
        print(f"  Note: site ({x:.2f},{y:.2f}) had no fully clash-free orientation in 500 tries "
              f"(margin {fallback_score:.3f} A); using least-bad found.")
    placed_O.append(O_pos)
    placed_H.append(list(chosen))


for (x, y) in monolayer_sites:
    place_water_at_site(x, y, z_ads)
n_monolayer = len(placed_O)
print(f"\nBuilt {n_monolayer} monolayer water molecules.")


# ---------------------------------------------------------------------
# 7. Stage 2: fill the REST of the vacuum with bulk-density water
#    (proper reject/retry: a new random position is tried on any clash,
#    rather than forcing a bad orientation at a fixed spot)
# ---------------------------------------------------------------------

bulk_z_lo = z_ads + 1.5
bulk_z_hi = c_len - BULK_TOP_MARGIN
remaining_thickness = max(bulk_z_hi - bulk_z_lo, 0.0)
remaining_volume = area * remaining_thickness

if AUTO_FILL_BULK:
    N_BULK_WATERS = max(int(round(remaining_volume / 29.9)), 0)
    print(f"\nAUTO_FILL_BULK enabled: remaining vacuum region is {remaining_thickness:.2f} A thick "
          f"({remaining_volume:.1f} A^3) -> targeting {N_BULK_WATERS} bulk-density waters.")

if N_BULK_WATERS > 0 and remaining_thickness > 0:
    print(f"\nStage 2: inserting up to {N_BULK_WATERS} additional bulk-like waters ...")
    rng_bulk = np.random.default_rng(BULK_SEED)
    bulk_attempts, bulk_accepted = 0, 0
    while bulk_accepted < N_BULK_WATERS and bulk_attempts < BULK_MAX_ATTEMPTS:
        bulk_attempts += 1
        u, v = rng_bulk.random(2)
        z = rng_bulk.uniform(bulk_z_lo, bulk_z_hi)
        xy = u * a_vec[:2] + v * b_vec[:2]
        O_pos = np.array([xy[0], xy[1], z])
        rot = Rotation.random(random_state=rng_bulk)
        H1 = O_pos + rot.apply(H1_local)
        H2 = O_pos + rot.apply(H2_local)

        po = np.array(placed_O) if placed_O else np.empty((0, 3))
        pH = np.array(placed_H).reshape(-1, 3) if placed_H else np.empty((0, 3))

        ok = True
        if np.any(min_image_dists_full(O_pos, cart) < (O_RADIUS + fw_radii)):
            ok = False
        if ok and np.any(min_image_dists_full(H1, cart) < (H_RADIUS + fw_radii)):
            ok = False
        if ok and np.any(min_image_dists_full(H2, cart) < (H_RADIUS + fw_radii)):
            ok = False
        if ok and len(po) and np.any(min_image_dists_full(O_pos, po) < 2 * O_RADIUS):
            ok = False
        if ok and len(pH) and np.any(min_image_dists_full(O_pos, pH) < (O_RADIUS + H_RADIUS)):
            ok = False
        if ok and len(po) and np.any(min_image_dists_full(H1, po) < (H_RADIUS + O_RADIUS)):
            ok = False
        if ok and len(po) and np.any(min_image_dists_full(H2, po) < (H_RADIUS + O_RADIUS)):
            ok = False
        if ok and len(pH) and np.any(min_image_dists_full(H1, pH) < 2 * H_RADIUS):
            ok = False
        if ok and len(pH) and np.any(min_image_dists_full(H2, pH) < 2 * H_RADIUS):
            ok = False

        if ok:
            placed_O.append(O_pos)
            placed_H.append([H1, H2])
            bulk_accepted += 1

    print(f"Stage 2: {bulk_accepted} / {N_BULK_WATERS} bulk waters placed ({bulk_attempts} attempts).")
    if bulk_accepted < N_BULK_WATERS:
        print("Target not fully reached -- near bulk-density jamming; try more BULK_MAX_ATTEMPTS "
              "or accept the count achieved.")

n_waters_total = len(placed_O)
print(f"\nTotal water molecules: {n_waters_total} "
      f"(monolayer: {n_monolayer}, bulk fill: {n_waters_total - n_monolayer})")


# ---------------------------------------------------------------------
# 8. Wrap into fractional coordinates per molecule
# ---------------------------------------------------------------------

inv_lat2d = np.linalg.inv(np.array([a_vec[:2], b_vec[:2]]).T)


def raw_uv(cart_xy):
    return inv_lat2d @ cart_xy


water_O_frac, water_H_frac = [], []
for O_pos, (H1, H2) in zip(placed_O, placed_H):
    uv_O_raw = raw_uv(O_pos[:2])
    floor_O = np.floor(uv_O_raw)
    uv_O = uv_O_raw - floor_O
    uv_H1 = raw_uv(H1[:2]) - floor_O
    uv_H2 = raw_uv(H2[:2]) - floor_O
    water_O_frac.append([uv_O[0], uv_O[1], (O_pos[2] / c_len) % 1.0])
    water_H_frac.append([uv_H1[0], uv_H1[1], (H1[2] / c_len) % 1.0])
    water_H_frac.append([uv_H2[0], uv_H2[1], (H2[2] / c_len) % 1.0])

water_O_frac = np.array(water_O_frac)
water_H_frac = np.array(water_H_frac)


# ---------------------------------------------------------------------
# 9. Merge and write
# ---------------------------------------------------------------------

def merge_and_write(struct, water_O_frac, water_H_frac, out_path):
    species, counts, frac = list(struct["species"]), list(struct["counts"]), struct["frac"]
    has_sd = struct["selective"]
    sd_flags = struct["sd_flags"]
    blocks, flag_blocks, i = {}, {}, 0
    for sp, n in zip(species, counts):
        blocks[sp] = frac[i:i + n]
        flag_blocks[sp] = sd_flags[i:i + n]
        i += n

    water_flags_O = [("T", "T", "T")] * len(water_O_frac)
    water_flags_H = [("T", "T", "T")] * len(water_H_frac)

    if "O" in blocks:
        blocks["O"] = np.vstack([blocks["O"], water_O_frac])
        flag_blocks["O"] = flag_blocks["O"] + water_flags_O
    else:
        species.append("O"); blocks["O"] = water_O_frac; flag_blocks["O"] = water_flags_O

    if "H" in blocks:
        blocks["H"] = np.vstack([blocks["H"], water_H_frac])
        flag_blocks["H"] = flag_blocks["H"] + water_flags_H
    else:
        species.append("H"); blocks["H"] = water_H_frac; flag_blocks["H"] = water_flags_H

    new_counts = [len(blocks[sp]) for sp in species]
    lines = [struct["comment"] if struct["comment"].strip() else "Solvated structure",
             "   1.00000000000000"]
    for row in struct["lattice"]:
        lines.append("    %19.16f %19.16f %19.16f" % tuple(row))
    lines.append("  " + "  ".join(f"{sp:<4}" for sp in species))
    lines.append("  " + "  ".join(str(c) for c in new_counts))
    if has_sd:
        lines.append("Selective dynamics")
    lines.append("Direct")
    for sp in species:
        for row, flags in zip(blocks[sp], flag_blocks[sp]):
            if has_sd:
                lines.append("  %19.16f %19.16f %19.16f  %s  %s  %s" %
                              (row[0], row[1], row[2], flags[0], flags[1], flags[2]))
            else:
                lines.append("  %19.16f %19.16f %19.16f" % tuple(row))

    out_path_abs = os.path.abspath(out_path)
    with open(out_path_abs, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWritten: {out_path_abs}")
    print("Species:", species, " Counts:", new_counts, " Total atoms:", sum(new_counts))
    return species, new_counts, out_path_abs


final_species, final_counts, written_path = merge_and_write(struct, water_O_frac, water_H_frac, OUTPUT_POSCAR)
assert os.path.isfile(written_path)
print(f"Confirmed on disk ({os.path.getsize(written_path)} bytes)")


# ---------------------------------------------------------------------
# 10. Validate
# ---------------------------------------------------------------------

def validate(out_path):
    s = read_poscar(out_path)
    lat_v = s["lattice"]
    a_v, b_v = lat_v[0], lat_v[1]
    cart_v = s["frac"] @ lat_v
    elems_v = []
    for sp, n in zip(s["species"], s["counts"]):
        elems_v += [sp] * n
    elems_v = np.array(elems_v)
    radii_v = np.array([ELEMENT_RADII.get(e, DEFAULT_RADIUS) for e in elems_v])
    shifts_v = np.array([n1 * a_v + n2 * b_v for n1 in (-1, 0, 1) for n2 in (-1, 0, 1)])

    n_water_O = len(water_O_frac)
    o_block_start = sum(final_counts[:final_species.index("O")]) if "O" in final_species else 0
    h_block_start = sum(final_counts[:final_species.index("H")])
    o_orig = struct["counts"][struct["species"].index("O")] if "O" in struct["species"] else 0
    h_orig = struct["counts"][struct["species"].index("H")] if "H" in struct["species"] else 0
    water_O_idx = np.arange(o_block_start + o_orig, o_block_start + o_orig + n_water_O)
    water_H_idx = np.arange(h_block_start + h_orig, h_block_start + h_orig + 2 * n_water_O)

    bonded = set()
    for k in range(n_water_O):
        oi, h1i, h2i = water_O_idx[k], water_H_idx[2 * k], water_H_idx[2 * k + 1]
        for p in [(oi, h1i), (h1i, oi), (oi, h2i), (h2i, oi), (h1i, h2i), (h2i, h1i)]:
            bonded.add(p)

    water_atoms = set(water_O_idx.tolist()) | set(water_H_idx.tolist())
    worst_ratio, worst_pair = 1e9, None
    for i in water_atoms:
        d = cart_v[i][None, None, :] - cart_v[None, :, :] - shifts_v[:, None, :]
        dist = np.linalg.norm(d, axis=2)
        dist[:, i] = 1e9
        for bj in [bp[1] for bp in bonded if bp[0] == i]:
            dist[:, bj] = 1e9
        mind = dist.min()
        j = np.unravel_index(np.argmin(dist), dist.shape)[1]
        ratio = mind / (radii_v[i] + radii_v[j])
        if ratio < worst_ratio:
            worst_ratio, worst_pair = ratio, (i, j, mind, radii_v[i] + radii_v[j])

    print(f"\nWorst non-bonded contact involving water: idx {worst_pair[0]} ({elems_v[worst_pair[0]]}) "
          f"- idx {worst_pair[1]} ({elems_v[worst_pair[1]]}), dist={worst_pair[2]:.3f} A, "
          f"cutoff={worst_pair[3]:.3f} A, ratio={worst_ratio:.3f} "
          f"({'OK' if worst_ratio > 0.9 else 'CHECK before AIMD/DFT'})")

    bad_bonds = 0
    for k in range(n_water_O):
        o_pos = cart_v[water_O_idx[k]]
        h1_pos = cart_v[water_H_idx[2 * k]]
        h2_pos = cart_v[water_H_idx[2 * k + 1]]
        d1, d2 = np.linalg.norm(o_pos - h1_pos), np.linalg.norm(o_pos - h2_pos)
        if not (0.9 < d1 < 1.0 and 0.9 < d2 < 1.0):
            bad_bonds += 1
    print(f"Water molecules with abnormal O-H bond length: {bad_bonds} / {n_water_O}")


validate(OUTPUT_POSCAR)

print("\nDone. Notes:")
print(f"- Vacuum gap: {vacuum_thickness:.2f} A, fully solvated (monolayer: {n_monolayer}, "
      f"bulk: {n_waters_total - n_monolayer}).")
print("- Pre-existing adsorbed H was treated as fixed framework and excluded neighboring sites.")
print("- Run geometry optimization / on-the-fly MLFF equilibration before production AIMD.")
