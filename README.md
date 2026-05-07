# BICxRCWA

Bound-state-in-the-continuum (BIC) search for a 1D dielectric grating using a coordinate-transformed RCWA (C-RCWA) solver in Python.

## Current Status (May 2026)

**No physical BIC is confirmed yet.**

- The previously reported near-zero point at `(θ≈13.42°, slab≈783.95 nm)` is a **ghost BIC** caused by low Fourier truncation (`M=5`).
- That point gives `|r₀|²≈1.5e-14` at `M=5` but jumps to nonzero values at higher modes (`M=10,15`), so it is not physical.
- The codebase now includes a mandatory convergence stage to reject such artifacts.

## Physical Setup

- Wavelength: `λ₀ = 600 nm`
- Period: `L = 500 nm`
- Grating profile: Gaussian, amplitude `100 nm`, width `σ = 50 nm`
- Permittivity: air `ε = 1 + i·1e-9`, dielectric `ε = 12`
- Geometry:
  - `H_fixed` = air buffer thickness (default `700 nm`, fixed)
  - `slab_thickness` = dielectric slab thickness (optimization variable)

Coordinate convention:

- Incident plane wave propagates upward from `y < 0`
- `y = 0`: lower boundary
- `y = H_fixed`: grating interface
- Reflection is evaluated at the bottom boundary
- Transmission exits from the top boundary

## Important Solver Fixes Already Applied

`rcwa_v2.py` includes the key corrections:

1. **Piecewise clamped asymmetric `S(y)` and `S'(y)`** (prevents out-of-support oscillation artifacts).
2. **Optimization variable fixed** from air-buffer height to **`slab_thickness`**.
3. **Lambda closure bug fixed** in the layer loop of `get_reflection`.
4. **Fourier mode exposed** in `get_reflection(..., fourierMode=...)` and objective path.

Do not revert these.

## Pipeline (rcwa_v2.py)

The main workflow is split into independently callable stages:

1. `run_scout()` → `results/scout_data.npz`, `results/stage1_scout.png`
2. `run_descent()` → `results/descent_results.json`
3. `run_phase_map()` → `results/phase_vortex_dielectric.png`, `results/intensity_crater_dielectric.png`
4. `run_convergence()` → `results/convergence_results.json`

At the top of `rcwa_v2.py`, toggle:

- `RUN_SCOUT`
- `RUN_DESCENT`
- `RUN_PHASE_MAP`
- `RUN_CONVERGENCE`

and then run:

```bash
python rcwa_v2.py
```

## Installation

Requires Python 3.8+:

```bash
pip install numpy scipy matplotlib
```

## Mandatory Convergence Protocol

Any BIC candidate must be checked at increasing Fourier mode count before being reported:

```python
for M in [10, 15, 20]:
    r = get_reflection(np.radians(theta_best), slab_best, fourierMode=M)
    print(f"M={M:2d}: |r₀|² = {abs(r)**2:.3e}")
```

Interpretation:

- **Physical candidate**: remains near zero (or shifts smoothly while remaining very small).
- **Ghost candidate**: jumps by orders of magnitude between `M` and `M+5`.

For this narrow grating (`σ/L≈0.1`), use **at least `M=10`** for scouting.

## Recommended Next Run

1. Re-run scout at `M=10` (the old `M=5` scout is not reliable).
2. Keep a manageable grid, e.g. 20×20 for initial scan.
3. Spot-check promising seeds at `M=15` before descent.
4. Run descent only for seeds that survive step 3.
5. Re-run convergence on best descent output.
6. Run phase/winding map only after convergence passes.

## Repository Contents (key files)

- `rcwa_v2.py` — main C-RCWA solver and 4-stage BIC search pipeline
- `manual_seeds.json` — manual candidate seeds (historical, mostly from `M=5`)
- `results/` — generated scout/descent/phase/convergence outputs
- `bic_animation_2.py`, `bic_animation_3.py` — field/animation scripts
- `h_diagnostic.py`, `sensitivity_test.py` — diagnostic scripts

## Notes

- `manual_seeds.json` comes from a low-mode scout and should not be trusted without high-mode recheck.
- Known problematic slab row near `283.33 nm` can violate energy conservation (`T0_Reflection > 1`) and should be excluded from analysis.
- Heatmaps should be plotted on a **log scale** for meaningful BIC candidate visibility.

## Credits

- MATLAB original: Dr. Benjamin Civiletti
- Python implementation and project development: Hoang Trieu
