# C_RCWA Method in Python - Reflection Kernel Refactor
"""
Author: Matlab Code - Dr. Benjamin Civiletti /
        Python Code - Hoang Trieu

First Updated: June 6, 2024
Last Updated: May 6, 2026

Pipeline stages (each independently callable, reads/writes disk):
  run_scout()       → results/scout_data.npz
  run_descent()     → results/descent_results.json
  run_phase_map()   → phase_vortex_dielectric.png, intensity_crater_dielectric.png
  run_convergence() → console table (no file output)
"""

from math import *
from numpy import pi, sin
import numpy as np
from scipy import linalg
import scipy
import os
import json
import time


# ============================================================
# PIPELINE CONTROL FLAGS  (edit these before each run)
# ============================================================
RUN_SCOUT       = False   # Stage 1 — 2D grid scout. Skip if scout_data.npz exists.
RUN_DESCENT     = False    # Stage 2 — Nelder-Mead from seeds.
RUN_PHASE_MAP   = False    # Stage 3 — Phase vortex + intensity crater plots.
RUN_CONVERGENCE = True    # Stage 4 — Fourier mode convergence check.

# If RUN_SCOUT = False, descent will load from this seeds file instead.
# Set to None to load seeds from results/scout_data.npz auto-detection.
MANUAL_SEEDS_FILE = "manual_seeds.json"

# Output directory for all saved results.
RESULTS_DIR = "results"


# ============================================================
# HELPER FUNCTIONS  (untouched from original)
# ============================================================

def gridC(numLayers, gratingPeriod, layerThick, xDiscreteSize, yDiscreteSize):
    """Calculate x/y grid points and layer boundaries."""
    xSpacingNum = gratingPeriod / xDiscreteSize
    xGrid = np.linspace(
        -gratingPeriod / 2 + xDiscreteSize / 2,
        gratingPeriod / 2 - xDiscreteSize / 2,
        num=int(xSpacingNum),
    )
    ySlice = layerThick / yDiscreteSize
    yGrid = []
    yBound = []
    lThick = []
    lThick.extend(layerThick)
    lThick.insert(0, 0)

    for i in range(numLayers):
        ySliceNum = int(ySlice[i])
        sBounds = [np.sum(lThick[: i + 1]), np.sum(lThick[: i + 2])]
        yGrid.extend(
            np.linspace(
                sBounds[0] + yDiscreteSize[i] / 2,
                sBounds[1] - yDiscreteSize[i] / 2,
                ySliceNum,
            )
        )
        yBound.append(sBounds[0])

    return xGrid, yGrid, np.array(yBound)


def create_Y_matrix(matrix1, matrix2, matrix3, matrix4):
    """Build the block matrix Y from four sub-matrices."""
    Y_top    = np.concatenate((matrix1, matrix2), axis=1)
    Y_bottom = np.concatenate((matrix3, matrix4), axis=1)
    Y        = np.concatenate((Y_top, Y_bottom), axis=0)
    return Y


def simpson(f, left, right, numSub):
    """Simpson integration rule for callable f(x) or sampled array."""
    h = (right - left) / numSub
    if callable(f):
        x = np.linspace(left, right, numSub + 1)
        y = f(x)
    elif isinstance(f, np.ndarray):
        y = f
    else:
        raise ValueError("Invalid input for f at the simpson function.")

    array1 = y[1:numSub:2]
    array2 = y[2:(numSub - 1):2]
    approxInt = (h / 3) * (y[0] + y[numSub] + 4 * np.sum(array1) + 2 * np.sum(array2))
    return approxInt


def bisection(f, left, right, TOL):
    """Bisection root finder on [left, right]."""
    n = 0
    if f(left) * f(right) >= 0:
        raise ValueError(
            "There is no root of f(x) in [" + str(left) + "," + str(right) + "]"
        )
    while (right - left) / 2 > TOL:
        mid = (left + right) / 2
        if f(mid) == 0:
            break
        if f(left) * f(mid) < 0:
            right = mid
        else:
            left = mid
        n += 1
    return (right + left) / 2, n


def ToeplitzM(m, f, L, sampleP):
    """Construct Toeplitz matrix of Fourier coefficients."""
    n = 2 * m - 1
    fftvec = np.zeros(n, dtype=complex)
    ind = (n - 1) / 2
    j = np.arange(-ind, ind + 1)
    if callable(f):
        for i in range(n):
            k = lambda x, ji=j[i]: f(x) * np.exp(-2 * pi * 1j * ji * x / L)
            fftvec[i] = (1 / L) * simpson(k, -L / 2, L / 2, sampleP)
    elif isinstance(f, np.ndarray):
        for i in range(n):
            x = np.linspace(-L / 2, L / 2, sampleP + 1)
            basis = np.exp(-2 * pi * 1j * j[i] * x / L)
            if len(basis) != len(f):
                raise ValueError("Lengths of basis and f do not match")
            fSample = f * basis
            fftvec[i] = (1 / L) * simpson(fSample, -L / 2, L / 2, sampleP)
    else:
        raise ValueError("Invalid input for f at ToeplitzM function")

    fftvec0pos = fftvec[m - 1:]
    fftvec0neg = np.flip(fftvec[:m])
    T = linalg.toeplitz(fftvec0pos, fftvec0neg)
    return T


def invMap(x, y, g, S, yGrid, H, gratingPeriod):
    """Inverse coordinate map helper (retained for compatibility — dead code, do not call)."""
    xHat = x
    C    = g(x, gratingPeriod)
    F    = lambda x: C * S(x) - x + y
    yHat, _ = bisection(F, 0, 2 * H, 10 ** -3)
    yGrid    = np.array(yGrid)
    index    = np.argmin(np.abs(yHat - yGrid))
    return xHat, yHat, index


# ============================================================
# PHYSICAL CONSTANTS
# ============================================================
epsilon0 = 8.854e-12
mu0      = 4 * pi * 1e-7
eta0     = np.sqrt(mu0 / epsilon0)
lambda0  = 600
k0       = 2 * pi / lambda0
omega    = k0 / eta0


# ============================================================
# CORE SOLVER
# ============================================================

def get_reflection_with_T0R(theta_val, slab_thickness, H_fixed=700.0, p_pol='p', fourierMode=5):
    """
    Return the complex zeroth-order reflection coefficient r₀ for a 1D C-RCWA solve.

    Parameters
    ----------
    theta_val      : float  — incidence angle in RADIANS
    slab_thickness : float  — dielectric slab thickness in nm  (BIC-relevant parameter)
    H_fixed        : float  — air buffer thickness in nm        (physically inert ≥ ~600 nm)
    p_pol          : 's' or 'p'
    fourierMode    : int    — number of Fourier modes ±N (default 5, i.e. 11 total)
    """
    if p_pol not in ('s', 'p'):
        raise ValueError("p_pol must be either 's' or 'p'.")

    m          = 2 * fourierMode + 1
    numLayers  = 2
    gratingPeriod  = 500
    yDiscreteSize  = np.ones(numLayers)
    xDiscreteSize  = 1
    layerThick     = np.array([H_fixed, slab_thickness], dtype=float)
    ep             = np.empty(numLayers, dtype=object)
    ep[0]          = 1 + 1j * 1e-9      # air (tiny imaginary part for stability)
    ep[1]          = 12.0 + 0j          # silicon-like dielectric
    #ep[1]          = 1 + 1j*1e-9
    sampleP        = 500

    # Grating profile  (amplitude=100 nm, width σ=50 nm, Gaussian)
    g      = lambda x, gP: 100 * np.exp(-(x / 50) ** 2)
    gPrime = lambda x, gP: -2 * x / 50 ** 2 * g(x, gP)

    # Piecewise asymmetric C-method coordinate transform
    # S(0)=0, S(H1)=1, S(H1+H2)=0  — C¹ globally, C² at interface
    H1 = H_fixed
    H2 = slab_thickness

    def S(y):
        return (
            np.where(
                y <= H1,
                0.5 * (1 - np.cos(pi * y / H1)),
                0.5 * (1 + np.cos(pi * (y - H1) / H2)),
            )
            * ((y >= 0) & (y <= H1 + H2))
        )

    def SPrime(y):
        return (
            np.where(
                y <= H1,
                (pi / (2 * H1)) * np.sin(pi * y / H1),
                -(pi / (2 * H2)) * np.sin(pi * (y - H1) / H2),
            )
            * ((y >= 0) & (y <= H1 + H2))
        )

    detDG = lambda x, y: SPrime(y) * g(x, gratingPeriod) + 1
    a11   = lambda x, y: np.abs(detDG(x, y))
    a21   = lambda x, y: -np.sign(detDG(x, y)) * S(y) * gPrime(x, gratingPeriod)
    a12   = lambda x, y: a21(x, y)
    a22   = lambda x, y: ((S(y) * gPrime(x, gratingPeriod)) ** 2 + 1) / np.abs(detDG(x, y))

    xGrid, yGrid, yBound = gridC(
        numLayers, gratingPeriod, layerThick, xDiscreteSize, yDiscreteSize
    )
    yGridN = np.size(yGrid)

    k_vector = np.arange(-fourierMode, fourierMode + 1, dtype=complex)
    k        = k0 * sin(theta_val) + 2 * pi * k_vector / gratingPeriod
    K        = np.diag(k)
    beta     = np.sqrt(k0 ** 2 - k ** 2)

    O = np.zeros((m, m))
    I = np.eye(m, m)

    Ye_p = create_Y_matrix(O, -np.diag(beta / k0), I,  O)
    Ye_m = create_Y_matrix(O,  np.diag(beta / k0), I,  O)
    Yh_p = create_Y_matrix(-np.diag(beta / k0), O, O, -I)
    Yh_m = create_Y_matrix( np.diag(beta / k0), O, O, -I)

    A = np.zeros((4 * fourierMode + 2, 1), dtype=complex)
    if p_pol == 's':
        A[fourierMode] = 1
    else:
        A[3 * fourierMode + 1] = 1

    Zs          = np.empty(yGridN + 1, dtype=object)
    Zs[yGridN]  = np.concatenate((Ye_p, Yh_p), axis=0)

    for i in range(yGridN - 1, -1, -1):
        check = yGrid[i] - yBound > 0
        layer = np.where(check)[0][-1]

        ep11 = lambda x, l=layer, yi=yGrid[i]: ep[l] * a11(x, yi)
        ep21 = lambda x, l=layer, yi=yGrid[i]: ep[l] * a21(x, yi)
        ep12 = ep21
        ep22 = lambda x, l=layer, yi=yGrid[i]: ep[l] * a22(x, yi)
        ep33 = lambda x, l=layer, yi=yGrid[i]: ep[l] * a11(x, yi)

        mu11 = lambda x, l=layer, yi=yGrid[i]: a11(x, yi)
        mu21 = lambda x, l=layer, yi=yGrid[i]: a21(x, yi)
        mu12 = mu21
        mu22 = lambda x, l=layer, yi=yGrid[i]: a22(x, yi)
        mu33 = lambda x, l=layer, yi=yGrid[i]: a11(x, yi)

        Tep11 = ToeplitzM(m, ep11, gratingPeriod, sampleP)
        Tep21 = ToeplitzM(m, ep21, gratingPeriod, sampleP)
        Tep12 = Tep21
        Tep22 = ToeplitzM(m, ep22, gratingPeriod, sampleP)
        Tep33 = ToeplitzM(m, ep33, gratingPeriod, sampleP)

        Tmu11 = (ep[layer] ** -1) * Tep11
        Tmu21 = (ep[layer] ** -1) * Tep21
        Tmu12 = Tmu21
        Tmu22 = (ep[layer] ** -1) * Tep22
        Tmu33 = (ep[layer] ** -1) * Tep33

        P11 = K @ linalg.solve(Tep22, Tep21)
        P14 = k0 * Tmu33 - (1 / k0) * K @ linalg.solve(Tep22, K)
        P22 = Tmu12 @ linalg.solve(Tmu22, K)
        P23 = -k0 * (Tmu11 - Tmu12 @ linalg.solve(Tmu22, Tmu21))
        P32 = -k0 * Tep33 + (1 / k0) * K @ linalg.solve(Tmu22, K)
        P33 = K @ linalg.solve(Tmu22, Tmu21)
        P41 = -k0 * (Tep12 @ linalg.solve(Tep22, Tep21) - Tep11)
        P44 = Tep12 @ linalg.solve(Tep22, K)

        P = np.block([
            [P11, O,   O,   P14],
            [O,   P22, P23, O  ],
            [O,   P32, P33, O  ],
            [P41, O,   O,   P44],
        ])

        Delta = 1 if (i == 0 or i == yGridN - 1) else yGrid[i] - yGrid[i - 1]

        D, G   = np.linalg.eig(P)
        D      = np.diag(D)
        Ddiag  = np.diag(D)
        idx    = np.argsort(np.imag(Ddiag))[::-1]
        Ddiag  = Ddiag[idx]
        D      = np.diag(Ddiag)
        G      = G[:, idx]

        W   = linalg.solve(G, Zs[i + 1])
        Wu  = W[:4 * fourierMode + 2, :]
        Wl  = W[4 * fourierMode + 2:, :]
        Dl  = D[4 * fourierMode + 2:, 4 * fourierMode + 2:]

        expmat1 = scipy.linalg.expm(-1j * Delta * Dl)
        expmat2 = scipy.linalg.expm( 1j * Delta * D[:4 * fourierMode + 2, :4 * fourierMode + 2])

        block_matrix = np.block([
            [np.eye(4 * fourierMode + 2)],
            [expmat1 @ Wl @ np.linalg.solve(Wu, expmat2)],
        ])
        Zs[i] = G @ block_matrix

    Z0u = Zs[0][:4 * fourierMode + 2, :]
    Z0l = Zs[0][4 * fourierMode + 2:, :]

    top    = np.block([[Z0u, -Ye_m], [Z0l, -Yh_m]])
    bottom = np.block([[Ye_p], [Yh_p]])
    X      = scipy.linalg.solve(top, bottom)
    T0R    = X @ A
    return T0R

def get_reflection(theta_val, slab_thickness, H_fixed=700.0, p_pol='p', fourierMode=5):
    T0R = get_reflection_with_T0R(theta_val, slab_thickness, H_fixed, p_pol, fourierMode)
    offset = 4 * fourierMode + 2
    if p_pol == 's':
        return T0R[offset + fourierMode, 0]
    else:
        return T0R[offset + 3 * fourierMode + 1, 0]


# ============================================================
# OBJECTIVE FUNCTION
# ============================================================

def objective_function(params, fourierMode=10):
    """Scalar |r₀|² for optimizer.  params = [theta_deg, slab_nm]."""
    theta_deg, slab_nm = params[0], params[1]
    r_0 = get_reflection(np.radians(theta_deg), slab_nm, fourierMode=fourierMode)
    return float(np.abs(r_0) ** 2)


# ============================================================
# STAGE 1 — 2D SCOUT
# ============================================================

def run_scout(
    theta_range=(0.0, 30.0), n_theta=30,
    slab_range=(200.0, 1200.0), n_slab=25,
    output_dir=RESULTS_DIR,
):
    """
    Evaluate |r₀|² on a 2D grid and save to scout_data.npz.

    Skips automatically if scout_data.npz already exists.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    out_npz = os.path.join(output_dir, "scout_data.npz")
    if os.path.exists(out_npz):
        print(f"[SCOUT] scout_data.npz already exists — skipping scout. Delete to re-run.")
        return

    os.makedirs(output_dir, exist_ok=True)

    theta_vals = np.linspace(*theta_range, n_theta)
    slab_vals  = np.linspace(*slab_range,  n_slab)

    Theta_grid, Slab_grid = np.meshgrid(theta_vals, slab_vals)
    intensity_map = np.zeros_like(Theta_grid)

    total = n_theta * n_slab
    current = 0
    t0 = time.time()

    for i, slab in enumerate(slab_vals):
        for j, theta in enumerate(theta_vals):
            current += 1
            print(f"  [{current}/{total}] theta={theta:.3f} deg, slab={slab:.1f} nm", end="", flush=True)
            try:
                intensity_map[i, j] = objective_function([theta, slab])
                print(f"  → {intensity_map[i,j]:.4e}")
            except Exception as e:
                print(f"  ERROR: {e}")
                intensity_map[i, j] = np.nan

    np.savez_compressed(
        out_npz,
        theta_vals=theta_vals,
        slab_vals=slab_vals,
        intensity_map=intensity_map,
    )
    print(f"\n[SCOUT] Saved {out_npz}  (shape intensity_map={intensity_map.shape})")
    print(f"[SCOUT] Runtime: {(time.time()-t0)/60:.1f} min")

    # Log-scale heatmap
    fig, ax = plt.subplots(figsize=(10, 7))
    clipped = np.clip(intensity_map, 1e-4, None)
    mesh = ax.pcolormesh(
        Theta_grid, Slab_grid, clipped,
        cmap='inferno', shading='auto',
        norm=LogNorm(vmin=1e-4, vmax=1.0),
    )
    plt.colorbar(mesh, ax=ax, label='|r₀|² (log scale)')
    ax.set_xlabel('Theta (degrees)')
    ax.set_ylabel('Slab Thickness (nm)')
    ax.set_title('Stage 1: 2D Scout — |r₀|² (log scale)')
    plt.tight_layout()
    out_png = os.path.join(output_dir, "stage1_scout.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[SCOUT] Saved {out_png}")


# ============================================================
# STAGE 2 — NELDER-MEAD DESCENT
# ============================================================

def run_descent(
    seeds=None,
    seeds_file=None,
    scout_npz=None,
    output_dir=RESULTS_DIR,
    theta_bounds=(0.0, 30.0),
    slab_bounds=(200.0, 1200.0),
):
    """
    Run Nelder-Mead from each seed and save results to descent_results.json.

    Seed priority (first one that succeeds):
      1. `seeds`      — list of dicts passed directly (highest priority)
      2. `seeds_file` — path to a JSON file with the same format
      3. `scout_npz`  — auto-detect candidates from a scout_data.npz file

    Each seed dict must have keys: 'theta' (deg), 'slab' (nm).
    Optional keys: 'r0_sq', 'type'.
    """
    import scipy.optimize as opt

    os.makedirs(output_dir, exist_ok=True)
    out_json = os.path.join(output_dir, "descent_results.json")

    # ── Load seeds ───────────────────────────────────────────────
    if seeds is not None:
        seed_list = seeds
        print(f"[DESCENT] Using {len(seed_list)} manually provided seeds.")

    elif seeds_file is not None and os.path.exists(seeds_file):
        with open(seeds_file) as f:
            data = json.load(f)
        # Accept either a bare list or the JSON with a 'known_candidates' key
        seed_list = data if isinstance(data, list) else data.get("known_candidates", data)
        print(f"[DESCENT] Loaded {len(seed_list)} seeds from {seeds_file}.")

    elif scout_npz is not None and os.path.exists(scout_npz):
        seed_list = _auto_seeds_from_scout(scout_npz)
        print(f"[DESCENT] Auto-detected {len(seed_list)} seeds from {scout_npz}.")

    else:
        raise FileNotFoundError(
            "run_descent: no seeds provided and no valid seeds_file or scout_npz found.\n"
            "Pass seeds= directly, or point seeds_file= / scout_npz= to existing files."
        )

    if len(seed_list) == 0:
        print("[DESCENT] No seeds found. Aborting descent.")
        return

    # ── Bounded objective ────────────────────────────────────────
    def bounded_obj(x):
        t, s = x[0], x[1]
        if not (theta_bounds[0] <= t <= theta_bounds[1] and
                slab_bounds[0]  <= s <= slab_bounds[1]):
            return 1.0
        return objective_function([t, s], fourierMode=10)

    # ── Run descent from each seed ───────────────────────────────
    descent_results = []

    for k, seed in enumerate(seed_list):
        t0 = float(seed['theta'])
        s0 = float(seed['slab'])
        v0 = float(seed.get('r0_sq', seed.get('r0', np.nan)))
        seed_type = seed.get('type', 'unknown')

        print(f"\n{'='*64}")
        print(f"Descent {k+1}/{len(seed_list)}  [{seed_type}]")
        print(f"  Seed: theta={t0:.4f} deg,  slab={s0:.2f} nm,  |r₀|²≈{v0:.3e}")
        print(f"{'='*64}")

        iters = [0]

        def _cb(xk, _iters=iters):
            _iters[0] += 1
            v = bounded_obj(xk)
            print(f"  iter {_iters[0]:4d}: theta={xk[0]:10.6f}  slab={xk[1]:10.4f}  "
                  f"|r₀|²={v:.6e}  log₁₀={np.log10(v+1e-30):.2f}")

        result = opt.minimize(
            bounded_obj,
            [t0, s0],
            method='Nelder-Mead',
            callback=_cb,
            options={'xatol': 1e-6, 'fatol': 1e-12, 'maxiter': 2000, 'disp': False},
        )

        tf, sf, vf = result.x[0], result.x[1], result.fun

        if vf < 1e-8:
            status = "STRONG_BIC"
        elif vf < 1e-3:
            status = "NEAR_BIC"
        else:
            status = "NO_BIC"

        print(f"  → theta={tf:.8f} deg,  slab={sf:.4f} nm,  |r₀|²={vf:.6e}  [{status}]")

        descent_results.append({
            "seed_theta":       t0,
            "seed_slab":        s0,
            "seed_type":        seed_type,
            "final_theta":      tf,
            "final_slab":       sf,
            "r0_squared":       float(vf),
            "log10_r0_squared": float(np.log10(vf + 1e-30)),
            "status":           status,
            "optimizer_msg":    result.message,
            "n_iter":           iters[0],
            "fourier_mode":     5,
        })

    # ── Summary table ────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("DESCENT SUMMARY")
    print(f"{'='*64}")
    hdr = f"{'#':>3}  {'seed_θ':>8}  {'seed_slab':>10}  {'final_θ':>12}  {'final_slab':>11}  {'|r₀|²':>10}  status"
    print(hdr)
    print("-" * len(hdr))
    for k, dr in enumerate(descent_results):
        print(f"{k+1:>3}  {dr['seed_theta']:>8.4f}  {dr['seed_slab']:>10.2f}  "
              f"{dr['final_theta']:>12.6f}  {dr['final_slab']:>11.4f}  "
              f"{dr['r0_squared']:>10.3e}  {dr['status']}")

    best = min(descent_results, key=lambda x: x['r0_squared'])
    print(f"\nBest: theta={best['final_theta']:.6f} deg,  "
          f"slab={best['final_slab']:.4f} nm,  |r₀|²={best['r0_squared']:.4e}")

    # ── Save ─────────────────────────────────────────────────────
    with open(out_json, 'w') as f:
        json.dump(descent_results, f, indent=2)
    print(f"[DESCENT] Saved {out_json}")

    return descent_results


def _auto_seeds_from_scout(npz_path, intensity_thresh_dark=0.01,
                            intensity_thresh_fano=0.3, adjacency=2):
    """Extract BIC candidate seeds from a saved scout_data.npz."""
    data         = np.load(npz_path)
    theta_vals   = data['theta_vals']
    slab_vals    = data['slab_vals']
    intensity_map = data['intensity_map']

    seeds = []
    for i in range(len(slab_vals)):
        for j in range(len(theta_vals)):
            if intensity_map[i, j] >= intensity_thresh_dark:
                continue
            nearby_fano = any(
                0 <= j + dj < len(theta_vals) and
                intensity_map[i, j + dj] > intensity_thresh_fano
                for dj in range(-adjacency, adjacency + 1) if dj != 0
            )
            if nearby_fano:
                seeds.append({
                    'theta': float(theta_vals[j]),
                    'slab':  float(slab_vals[i]),
                    'r0_sq': float(intensity_map[i, j]),
                    'type':  'FW_auto',
                })
        # Symmetry-protected: near-dark at theta~0
        if intensity_map[i, 0] < 0.05:
            seeds.append({
                'theta': 0.5,
                'slab':  float(slab_vals[i]),
                'r0_sq': float(intensity_map[i, 0]),
                'type':  'symm_protected_auto',
            })
    return seeds


# ============================================================
# STAGE 3 — PHASE MAP + WINDING NUMBER
# ============================================================

def calculate_winding_number(phase_grid, contour_radius=None):
    n = phase_grid.shape[0]
    c = n // 2
    r = contour_radius if contour_radius is not None else c - 1

    top    = phase_grid[c - r,         c - r : c + r    ]
    right  = phase_grid[c - r : c + r, c + r            ]
    bottom = phase_grid[c + r,         c + r : c - r : -1]
    left   = phase_grid[c + r : c - r : -1, c - r       ]
    contour = np.concatenate([top, right, bottom, left])

    diffs   = np.diff(np.concatenate([contour, contour[:1]]))
    wrapped = np.angle(np.exp(1j * diffs))
    q       = np.round(np.sum(wrapped) / (2 * np.pi))

    print(f"\n=== TOPOLOGICAL CHECK ===")
    print(f"Winding number q = {q:.4f}  ->  nearest integer: {int(q)}")
    return int(q)


def run_phase_map(
    theta_center=None,
    slab_center=None,
    descent_json=None,
    output_dir=RESULTS_DIR,
    grid_size=15,
    theta_half_width=0.5,
    slab_half_width=5.0,
):
    """
    Generate phase vortex and intensity crater plots around a BIC candidate.

    Center priority:
      1. (theta_center, slab_center) passed directly
      2. Best result from descent_json
    """
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # ── Resolve center ───────────────────────────────────────────
    if theta_center is None or slab_center is None:
        if descent_json is None:
            descent_json = os.path.join(output_dir, "descent_results.json")
        if not os.path.exists(descent_json):
            raise FileNotFoundError(
                f"run_phase_map: no center provided and {descent_json} not found."
            )
        with open(descent_json) as f:
            results = json.load(f)
        best = min(results, key=lambda x: x['r0_squared'])
        theta_center = best['final_theta']
        slab_center  = best['final_slab']
        print(f"[PHASE MAP] Using best descent result: "
              f"theta={theta_center:.6f} deg,  slab={slab_center:.4f} nm")

    # ── Grid ─────────────────────────────────────────────────────
    theta_vals = np.linspace(theta_center - theta_half_width,
                             theta_center + theta_half_width, grid_size)
    slab_vals  = np.linspace(slab_center  - slab_half_width,
                             slab_center  + slab_half_width,  grid_size)

    Theta_grid, Slab_grid = np.meshgrid(theta_vals, slab_vals)
    Phase_grid     = np.zeros_like(Theta_grid)
    Intensity_grid = np.zeros_like(Theta_grid)

    total = grid_size ** 2
    print(f"[PHASE MAP] Starting {grid_size}×{grid_size} = {total} point sweep...")
    t0 = time.time()

    for i in range(grid_size):
        for j in range(grid_size):
            r_0 = get_reflection(np.radians(theta_vals[j]), slab_vals[i])
            Phase_grid[i, j]     = np.angle(r_0)
            Intensity_grid[i, j] = np.abs(r_0) ** 2
        elapsed = (time.time() - t0) / 60
        remaining = elapsed / (i + 1) * (grid_size - i - 1)
        print(f"  Row {i+1}/{grid_size} done  ({elapsed:.1f} min elapsed, "
              f"~{remaining:.1f} min remaining)")

    print(f"[PHASE MAP] Sweep complete in {(time.time()-t0)/60:.1f} min")

    # ── Winding number ───────────────────────────────────────────
    q = calculate_winding_number(Phase_grid)

    # ── Plot 1: Phase vortex ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(Theta_grid, Slab_grid, Phase_grid,
                         cmap='twilight', shading='auto')
    plt.colorbar(mesh, ax=ax, label='Phase angle (radians)')
    ax.set_xlabel('Theta (degrees)')
    ax.set_ylabel('Slab Thickness (nm)')
    ax.set_title(f'Phase Vortex — True BIC  (q = {q})')
    ax.plot(theta_center, slab_center, 'w*', markersize=12,
            markeredgecolor='k', label='BIC center')
    ax.legend()
    plt.tight_layout()
    out1 = os.path.join(output_dir, "phase_vortex_dielectric.png")
    plt.savefig(out1, dpi=300)
    plt.close()
    print(f"[PHASE MAP] Saved {out1}")

    # ── Plot 2: Intensity crater ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(Theta_grid, Slab_grid, Intensity_grid,
                         cmap='viridis', shading='auto')
    plt.colorbar(mesh, ax=ax, label='|r₀|²')
    ax.set_xlabel('Theta (degrees)')
    ax.set_ylabel('Slab Thickness (nm)')
    ax.set_title('Intensity Crater around True BIC')
    ax.plot(theta_center, slab_center, 'r*', markersize=12,
            markeredgecolor='k', label='BIC center')
    ax.legend()
    plt.tight_layout()
    out2 = os.path.join(output_dir, "intensity_crater_dielectric.png")
    plt.savefig(out2, dpi=300)
    plt.close()
    print(f"[PHASE MAP] Saved {out2}")

    return q


# ============================================================
# STAGE 4 — FOURIER MODE CONVERGENCE CHECK
# ============================================================

def run_convergence(
    theta_center=None,
    slab_center=None,
    descent_json=None,
    output_dir=RESULTS_DIR,
    modes=(5, 10, 15),
):
    """
    Evaluate |r₀|² at increasing Fourier mode counts at the best BIC candidate.

    A genuine BIC should be stable under increasing M.
    A numerical artifact will shift or disappear.

    Center priority: same as run_phase_map.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Resolve center ───────────────────────────────────────────
    if theta_center is None or slab_center is None:
        if descent_json is None:
            descent_json = os.path.join(output_dir, "descent_results.json")
        if not os.path.exists(descent_json):
            raise FileNotFoundError(
                f"run_convergence: no center provided and {descent_json} not found."
            )
        with open(descent_json) as f:
            results = json.load(f)
        best = min(results, key=lambda x: x['r0_squared'])
        theta_center = best['final_theta']
        slab_center  = best['final_slab']

    print(f"\n[CONVERGENCE] Checking at theta={theta_center:.6f} deg, slab={slab_center:.4f} nm")
    print(f"{'Mode (±N)':>10}  {'Total modes':>12}  {'|r₀|²':>14}  {'log₁₀|r₀|²':>14}")
    print("-" * 56)

    rows = []
    for M in modes:
        r0  = get_reflection(np.radians(theta_center), slab_center, fourierMode=M)
        val = float(np.abs(r0) ** 2)
        rows.append({'M': M, 'total_modes': 2*M+1, 'r0_squared': val})
        print(f"{M:>10}  {2*M+1:>12}  {val:>14.6e}  {np.log10(val+1e-30):>14.4f}")

    # Assess stability
    vals = [r['r0_squared'] for r in rows]
    if len(vals) >= 2:
        relative_change = abs(vals[-1] - vals[0]) / (vals[0] + 1e-30)
        if relative_change < 0.5:
            verdict = "STABLE — consistent with a genuine BIC"
        else:
            verdict = f"UNSTABLE (relative change = {relative_change:.1%}) — may be numerical artifact"
        print(f"\n[CONVERGENCE] Verdict: {verdict}")

    # Save
    out_json = os.path.join(output_dir, "convergence_results.json")
    with open(out_json, 'w') as f:
        json.dump({
            'theta_deg': theta_center,
            'slab_nm':   slab_center,
            'rows':      rows,
        }, f, indent=2)
    print(f"[CONVERGENCE] Saved {out_json}")

    return rows


# ============================================================
# MANUAL SEEDS FILE  (written once, read by run_descent)
# ============================================================

def write_manual_seeds(path=MANUAL_SEEDS_FILE):
    """
    Write the known BIC candidate seeds from the console log to disk.
    Call this once; afterwards run_descent will read it automatically.
    """
    seeds = [
        {"theta": 13.45,   "slab": 783.0,   "r0_sq": 3.2e-4, "type": "FW_candidate"},
        {"theta": 0.5,     "slab": 992.0,   "r0_sq": 3.7e-4, "type": "symm_protected_candidate"},
        {"theta": 10.3448, "slab": 408.33,  "r0_sq": 9.91e-3, "type": "FW_tight"},
        {"theta": 0.5,     "slab": 783.33,  "r0_sq": 7.91e-3, "type": "symm_protected"},
        {"theta": 8.2759,  "slab": 700.00,  "r0_sq": 1.53e-2, "type": "FW"},
        {"theta": 18.6207, "slab": 1158.33, "r0_sq": 1.45e-2, "type": "FW"},
        {"theta": 3.1034,  "slab": 866.67,  "r0_sq": 4.01e-2, "type": "FW"},
        {"theta": 12.4138, "slab": 908.33,  "r0_sq": 7.52e-2, "type": "FW"},
    ]
    with open(path, 'w') as f:
        json.dump(seeds, f, indent=2)
    print(f"[SEEDS] Wrote {len(seeds)} seeds to {path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Stage 1: Scout ───────────────────────────────────────────
    if RUN_SCOUT:
        run_scout(
            theta_range=(0.0, 30.0), n_theta=30,
            slab_range=(200.0, 1200.0), n_slab=25,
        )

    # ── Stage 2: Descent ─────────────────────────────────────────
    if RUN_DESCENT:
        # Write seeds file if it doesn't exist yet
        if not os.path.exists(MANUAL_SEEDS_FILE):
            write_manual_seeds(MANUAL_SEEDS_FILE)

        scout_npz = os.path.join(RESULTS_DIR, "scout_data.npz")

        descent_results = run_descent(
            seeds_file=MANUAL_SEEDS_FILE,           # use manual seeds (already have them)
            scout_npz=scout_npz if os.path.exists(scout_npz) else None,  # fallback
        )

    # ── Stage 3: Phase map ───────────────────────────────────────
    if RUN_PHASE_MAP:
        # Reads best result from descent_results.json automatically.
        # Override by passing theta_center=..., slab_center=... explicitly.
        run_phase_map()

    # ── Stage 4: Convergence ─────────────────────────────────────
    if RUN_CONVERGENCE:
        run_convergence(modes=(10, 15, 20))