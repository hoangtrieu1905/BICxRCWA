#!/usr/bin/env python3
"""
Phase 1 Diagnostic: Is H_val a physically meaningful parameter or an inert
air buffer?

Reads rcwa_v2.py parameters, runs a 1D scan of |r_0|^2 vs H_val at a fixed
theta (9.47 degrees, near the Fano ridge), and prints a verdict.

Usage:
    python h_diagnostic.py          # 50-point scan (~20 min)
    python h_diagnostic.py --fast   # 10-point scan (~4 min)
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from rcwa_v2 import get_reflection


def report_code_parameters():
    print("--- Code Inspection Report (rcwa_v2.py) ---")
    print()
    print("  layerThick = np.array([H_val, 700], dtype=float)")
    print("    Layer 0  ep[0] = 1 + 1j*10^-9  ~= 1  (air / vacuum)")
    print("    Layer 1  ep[1] = 12.0 + 0j           (dielectric slab, 700 nm)")
    print()
    print("  Semi-infinite boundary conditions (Rayleigh half-spaces):")
    print("    beta = sqrt(k0^2 - k^2)    <- NO epsilon factor in this expression")
    print("    Ye_p / Yh_p use beta/k0    <- eps_semi_infinite_above = 1 (vacuum)")
    print("    Ye_m / Yh_m use beta/k0    <- eps_semi_infinite_below = 1 (vacuum)")
    print()
    print("  KEY OBSERVATION:")
    print("    ep[0] ~= 1  AND  eps of both Rayleigh half-spaces = 1.")
    print("    The H_val air layer sits between two identical eps=1 regions.")
    print("    In exact theory, H_val should be INERT (a gauge degree of freedom,")
    print("    not a physical resonance parameter).")
    print("-" * 60)


def run_scan(n_points):
    theta_deg = 9.47
    theta_rad = np.radians(theta_deg)
    H_values  = np.linspace(100.0, 1500.0, n_points)

    print(f"\nRunning {n_points}-point 1D scan:")
    print(f"  Fixed   theta = {theta_deg} deg  (near Fano ridge from prior scout)")
    print(f"  Fixed   slab  = 700 nm")
    print(f"  Varying H_val in [{H_values[0]:.0f}, {H_values[-1]:.0f}] nm\n")

    intensities = np.empty(n_points)
    for k, H_val in enumerate(H_values):
        r0 = get_reflection(theta_rad, float(H_val))
        intensities[k] = float(np.abs(r0) ** 2)
        print(f"  [{k+1:2d}/{n_points}]  H = {H_val:7.1f} nm   |r_0|^2 = {intensities[k]:.6f}")

    return H_values, intensities


def plot_and_verdict(H_values, intensities, n_points):
    # ── save plot ─────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.plot(H_values, intensities, marker='o', markersize=4,
             linewidth=1.5, color='steelblue')
    plt.xlabel('H_val (nm)  [air-buffer thickness below grating]', fontsize=12)
    plt.ylabel('|r0|^2', fontsize=12)
    plt.title(f'Phase 1 Diagnostic: |r0|^2 vs H_val  '
              f'(theta = 9.47°, slab = 700 nm,  n = {n_points})',
              fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("h_diagnostic.png", dpi=300)
    plt.close()
    print("\n-> Saved: h_diagnostic.png")

    # ── statistics ────────────────────────────────────────────────────────────
    I_min   = float(np.min(intensities))
    I_max   = float(np.max(intensities))
    I_range = I_max - I_min
    I_rel   = I_range / (I_max + 1e-15)
    min_idx = int(np.argmin(intensities))

    print("\n--- Analysis ---")
    print(f"  |r_0|^2  min  : {I_min:.6e}   at  H = {H_values[min_idx]:.1f} nm")
    print(f"  |r_0|^2  max  : {I_max:.6e}")
    print(f"  Peak-to-peak  : {I_range:.6e}   ({100.0 * I_rel:.2f}% of max)")

    # ── verdict ───────────────────────────────────────────────────────────────
    print()
    # Case 1: genuine deep minimum — STOP
    if I_min < 1e-4 and I_rel > 0.5:
        print("VERDICT: H has a genuine deep minimum.")
        print(f"  Minimum |r_0|^2 = {I_min:.4e}  at  H = {H_values[min_idx]:.1f} nm")
        print("  STOP — do not proceed to Phase 2. "
              "Report the minimum value and location.")
        return

    # Case 2: oscillations > 10% — check for Fabry-Perot periodicity
    if I_rel > 0.10:
        try:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(intensities, prominence=0.05 * I_max)
            if len(peaks) >= 2:
                spacings = np.diff(H_values[peaks])
                mean_period = float(np.mean(spacings))
                lambda_half = 300.0  # nm (lambda=600 nm in vacuum, eps=1)
                print(f"  Found {len(peaks)} oscillation peaks;  "
                      f"mean spacing = {mean_period:.0f} nm  "
                      f"(lambda/2 in vacuum = {lambda_half:.0f} nm).")
                print("VERDICT: H is a Fabry-Perot spacer — it matters physically "
                      "but is not a BIC-controlling parameter. Proceed to Phase 2.")
                return
        except ImportError:
            pass
        print("VERDICT: H shows >10% variation (possibly Fabry-Perot or numerical).")
        print("  Proceed to Phase 2.")
        return

    # Case 3: essentially flat — H is inert
    print("VERDICT: H is approximately inert. Proceed to Phase 2.")


if __name__ == "__main__":
    fast   = "--fast" in sys.argv
    n_pts  = 10 if fast else 50

    print("=" * 60)
    print("PHASE 1 DIAGNOSTIC: Is H_val physically meaningful?")
    print(f"  Mode: {'FAST (10 points, ~4 min)' if fast else 'Full (50 points, ~20 min)'}")
    print("=" * 60)
    print()

    report_code_parameters()

    H_values, intensities = run_scan(n_pts)
    plot_and_verdict(H_values, intensities, n_pts)
