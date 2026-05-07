# BIC Wave Propagation Animation
"""
Animates Re[H_y(r) * exp(-i*omega*t)] to show wave crests moving
through the grating structure.

Side-by-side: OFF-BIC (reflected wave visible) vs BIC (wave trapped).

This is physically rigorous — RCWA gives the time-harmonic field H(r),
and the full time-dependent solution is Re[H(r) * exp(-iwt)].

Note: This code is actually flipped. Plane wave 
coming from below and propagating upwards. This is just a convention and doesn't affect the physics. 
The "air" region is at the bottom and the "dielectric" region is at the top. 
The BIC is still "perfectly" (well, not really...) trapped with no reflection back downwards.

Author: Hoang Trieu
Date Updated: May 2, 2026
"""

from math import *
from numpy import pi, sin, cos
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import TwoSlopeNorm
from scipy import linalg
import scipy
import time

plt.close('all')

# =====================================================================
# HELPER FUNCTIONS (identical to optimization code)
# =====================================================================

def gridC(numLayers, gratingPeriod, layerThick, xDiscreteSize, yDiscreteSize):
    xSpacingNum = gratingPeriod / xDiscreteSize
    xGrid = np.linspace(
        -gratingPeriod/2 + xDiscreteSize/2,
        gratingPeriod/2 - xDiscreteSize/2,
        num=int(xSpacingNum))
    ySlice = layerThick / yDiscreteSize
    yGrid = []
    yBound = []
    lThick = list(layerThick)
    lThick.insert(0, 0)
    for i in range(numLayers):
        ySliceNum = int(ySlice[i])
        sBounds = [np.sum(lThick[:i+1]), np.sum(lThick[:i+2])]
        yGrid.extend(np.linspace(
            sBounds[0] + yDiscreteSize[i]/2,
            sBounds[1] - yDiscreteSize[i]/2,
            ySliceNum))
        yBound.append(sBounds[0])
    return xGrid, yGrid, np.array(yBound)


def create_Y_matrix(m1, m2, m3, m4):
    return np.concatenate([
        np.concatenate([m1, m2], axis=1),
        np.concatenate([m3, m4], axis=1)
    ], axis=0)


def simpson(f, left, right, numSub):
    h = (right - left) / numSub
    if callable(f):
        x = np.linspace(left, right, numSub + 1)
        y = f(x)
    elif isinstance(f, np.ndarray):
        y = f
    else:
        raise ValueError("Invalid input")
    a1 = y[1:numSub:2]
    a2 = y[2:(numSub-1):2]
    return (h/3) * (y[0] + y[numSub] + 4*np.sum(a1) + 2*np.sum(a2))


def ToeplitzM(m, f, L, sampleP):
    n = 2*m - 1
    fftvec = np.zeros(n, dtype=complex)
    ind = (n-1) / 2
    j = np.arange(-ind, ind + 1)
    if callable(f):
        for i in range(n):
            k = lambda x: f(x) * np.exp(-2*pi*1j*j[i]*x/L)
            fftvec[i] = (1/L) * simpson(k, -L/2, L/2, sampleP)
    elif isinstance(f, np.ndarray):
        for i in range(n):
            ind = (n-1) / 2
            j = np.arange(-ind, ind + 1)
            x = np.linspace(-L/2, L/2, sampleP + 1)
            basis = np.exp(-2*pi*1j*j[i]*x/L)
            fSample = f * basis
            fftvec[i] = (1/L) * simpson(fSample, -L/2, L/2, sampleP)
    fftvec0pos = fftvec[m-1:]
    fftvec0neg = np.flip(fftvec[:m])
    return linalg.toeplitz(fftvec0pos, fftvec0neg)


# =====================================================================
# FIXED PARAMETERS
# =====================================================================

epsilon0 = 8.854e-12
mu0 = 4 * pi * 1e-7
eta0 = np.sqrt(mu0 / epsilon0)
lambda0 = 600
k0 = 2 * pi / lambda0

H_fixed = 700.0          # air buffer thickness (nm) — fixed, physically inert
slab_thickness = 783.9477  # dielectric slab thickness (nm) — BIC-relevant
fourierMode = 5
m_fourier = 2 * fourierMode + 1
numLayers = 2
gratingPeriod = 500
yDiscreteSize = np.ones(numLayers)
xDiscreteSize = 1
sampleP = 500

layerThick = np.array([H_fixed, slab_thickness], dtype=float)
total_height = H_fixed + slab_thickness

ep = np.empty(numLayers, dtype=object)
ep[0] = 1 + 1j * 1e-9   # air layer (lower, [0, H_fixed])
ep[1] = 12.0 + 0j        # dielectric slab (upper, [H_fixed, H_fixed+slab_thickness])

g = lambda x, gP: 100 * np.exp(-(x / 50)**2)
gPrime = lambda x, gP: -2 * x / 50**2 * g(x, gP)

# Piecewise asymmetric C-method coordinate transform (Bug 1 fix — do not revert)
H1 = H_fixed
H2 = slab_thickness

def S_func(y):
    return (np.where(y <= H1,
                     0.5 * (1 - np.cos(pi * y / H1)),
                     0.5 * (1 + np.cos(pi * (y - H1) / H2)))
            * ((y >= 0) & (y <= H1 + H2)))

def SPrime(y):
    return (np.where(y <= H1,
                     (pi / (2 * H1)) * np.sin(pi * y / H1),
                     -(pi / (2 * H2)) * np.sin(pi * (y - H1) / H2))
            * ((y >= 0) & (y <= H1 + H2)))

detDG = lambda x, y: SPrime(y) * g(x, gratingPeriod) + 1

a11 = lambda x, y: np.abs(detDG(x, y))
a21 = lambda x, y: -np.sign(detDG(x, y)) * S_func(y) * gPrime(x, gratingPeriod)
a12 = lambda x, y: a21(x, y)
a22 = lambda x, y: ((S_func(y) * gPrime(x, gratingPeriod))**2 + 1) / np.abs(detDG(x, y))


# =====================================================================
# RCWA SOLVER (returns complex field for time animation)
# =====================================================================

def solve_complex_field(theta_rad):
    """Run RCWA and return the COMPLEX Hyhat field (not magnitude)."""

    (xGrid, yGrid, yBound) = gridC(numLayers, gratingPeriod, layerThick,
                                    xDiscreteSize, yDiscreteSize)
    yGridN = np.size(yGrid)
    xGridN = np.size(xGrid)

    k_vector = np.arange(-fourierMode, fourierMode+1, dtype=np.complex128)
    k = k0 * sin(theta_rad) + 2*pi*k_vector / gratingPeriod
    K = np.diag(k)
    beta = np.sqrt(k0**2 - k**2)

    O = np.zeros((m_fourier, m_fourier))
    I = np.eye(m_fourier)

    Ye_p = create_Y_matrix(O, -np.diag(beta/k0), I, O)
    Ye_m = create_Y_matrix(O,  np.diag(beta/k0), I, O)
    Yh_p = create_Y_matrix(-np.diag(beta/k0), O, O, -I)
    Yh_m = create_Y_matrix( np.diag(beta/k0), O, O, -I)

    A = np.zeros((4*fourierMode + 2, 1), dtype=complex)
    A[3*fourierMode + 1] = 1  # p-pol

    Zs = np.empty(yGridN + 1, dtype=object)
    Zs[yGridN] = np.concatenate((Ye_p, Yh_p), axis=0)

    Wus_local = np.empty(yGridN, dtype=object)
    Dus_local = np.empty(yGridN, dtype=object)

    for i in range(yGridN - 1, -1, -1):
        check = (yGrid[i] - yBound > 0)
        layer = np.where(check)[0][-1]

        ep11 = lambda x, l=layer, yi=yGrid[i]: ep[l] * a11(x, yi)
        ep21 = lambda x, l=layer, yi=yGrid[i]: ep[l] * a21(x, yi)
        ep22_f = lambda x, l=layer, yi=yGrid[i]: ep[l] * a22(x, yi)
        ep33 = lambda x, l=layer, yi=yGrid[i]: ep[l] * a11(x, yi)
        mu11 = lambda x, l=layer, yi=yGrid[i]: a11(x, yi)
        mu21 = lambda x, l=layer, yi=yGrid[i]: a21(x, yi)
        mu22_f = lambda x, l=layer, yi=yGrid[i]: a22(x, yi)
        mu33 = lambda x, l=layer, yi=yGrid[i]: a11(x, yi)

        Tep11 = ToeplitzM(m_fourier, ep11, gratingPeriod, sampleP)
        Tep21 = ToeplitzM(m_fourier, ep21, gratingPeriod, sampleP)
        Tep12 = Tep21
        Tep22 = ToeplitzM(m_fourier, ep22_f, gratingPeriod, sampleP)
        Tep33 = ToeplitzM(m_fourier, ep33, gratingPeriod, sampleP)
        Tmu11 = (ep[layer]**(-1)) * Tep11
        Tmu21 = (ep[layer]**(-1)) * Tep21
        Tmu12 = Tmu21
        Tmu22 = (ep[layer]**(-1)) * Tep22
        Tmu33 = (ep[layer]**(-1)) * Tep33

        P11 = K @ linalg.solve(Tep22, Tep21)
        P14 = k0*Tmu33 - (1/k0)*K @ linalg.solve(Tep22, K)
        P22 = Tmu12 @ linalg.solve(Tmu22, K)
        P23 = -k0*(Tmu11 - Tmu12 @ linalg.solve(Tmu22, Tmu21))
        P32 = -k0*Tep33 + (1/k0)*K @ linalg.solve(Tmu22, K)
        P33 = K @ linalg.solve(Tmu22, Tmu21)
        P41 = -k0*(Tep12 @ linalg.solve(Tep22, Tep21) - Tep11)
        P44 = Tep12 @ linalg.solve(Tep22, K)

        P = np.block([[P11, O, O, P14],
                      [O, P22, P23, O],
                      [O, P32, P33, O],
                      [P41, O, O, P44]])

        Delta = 1 if (i == 0 or i == yGridN - 1) else yGrid[i] - yGrid[i-1]

        D_eig, G = np.linalg.eig(P)
        D_eig = np.diag(D_eig)
        Ddiag = np.diag(D_eig)
        ind = np.argsort(np.imag(Ddiag))[::-1]
        Ddiag = Ddiag[ind]
        D_eig = np.diag(Ddiag)
        G = G[:, ind]

        W = linalg.solve(G, Zs[i+1])
        Wu = W[:4*fourierMode+2, :]
        Wus_local[i] = Wu
        Wl = W[4*fourierMode+2:, :]
        Du = D_eig[:4*fourierMode+2, :4*fourierMode+2]
        Dus_local[i] = Du
        Dl = D_eig[4*fourierMode+2:, 4*fourierMode+2:]

        block_matrix = np.block([
            [np.eye(4*fourierMode+2)],
            [scipy.linalg.expm(-1j*Delta*Dl) @ Wl @
             np.linalg.solve(Wu, scipy.linalg.expm(1j*Delta*Du))]
        ])
        Zs[i] = G @ block_matrix

    # Solve
    Z0u = Zs[0][:4*fourierMode+2, :]
    Z0l = Zs[0][4*fourierMode+2:, :]
    top = np.block([[Z0u, -Ye_m], [Z0l, -Yh_m]])
    bottom = np.block([[Ye_p], [Yh_p]])
    X = scipy.linalg.solve(top, bottom)
    T0R = X @ A

    # Reflection coefficient
    offset = 4 * fourierMode + 2
    r0 = T0R[offset + 3*fourierMode + 1, 0]

    # Propagate
    Ts = np.empty(yGridN + 1, dtype=object)
    Ts[0] = T0R[:4*fourierMode+2, :]
    f_arr = np.empty(yGridN + 1, dtype=object)

    for i in range(yGridN):
        Delta = 1 if (i == 0 or i == yGridN - 1) else yGrid[i] - yGrid[i-1]
        Ts[i+1] = linalg.solve(Wus_local[i],
                    scipy.linalg.expm(1j*Delta*Dus_local[i])) @ Ts[i]
    for i in range(yGridN + 1):
        f_arr[i] = Zs[i] @ Ts[i]

    
    # Reconstruct COMPLEX Hy in hat-space
    basis = lambda x, n: np.exp(1j * n * x)
    xGrid_c = xGrid.astype(np.complex128)
    k_c = np.complex128(k)

    Hyhat = np.zeros((yGridN, xGridN), dtype=np.complex128)
    for i in range(yGridN):
        for j in range(xGridN):
            Hyhat[i, j] = basis(xGrid_c[j], k_c) @ f_arr[i+1][6*fourierMode+3:, 0]
    
    '''
    # Reconstruct COMPLEX Hy in hat-space (then subtract incident wave in air region)
    basis = lambda x, n: np.exp(1j * n * x)
    xGrid_c = xGrid.astype(np.complex128)
    k_c = np.complex128(k)

    Hyhat = np.zeros((yGridN, xGridN), dtype=np.complex128)
    
    # Identify the index of the 0th order mode (the incident wave)
    zero_order_idx = fourierMode 
    k_x0 = k[zero_order_idx]
    beta_0 = beta[zero_order_idx]

    for i in range(yGridN):
        for j in range(xGridN):
            # 1. Compute the total field at this grid point
            total_field = basis(xGrid_c[j], k_c) @ f_arr[i+1][6*fourierMode+3:, 0]
            
            # 2. Subtract the incident wave IF we are in the Air region
            if yGrid[i] <= H:
                # Mathematical expression of your incident plane wave
                incident_wave = np.exp(1j * (k_x0 * xGrid_c[j] + beta_0 * yGrid[i]))
                Hyhat[i, j] = total_field - incident_wave
            else:
                # Inside the dielectric, leave the total field as is
                Hyhat[i, j] = total_field
    '''
    return Hyhat, r0, yGrid, xGrid


# =====================================================================
# COMPUTE FIELDS FOR TWO CASES
# =====================================================================

theta_BIC = 13.42159675   # STRONG_BIC: |r₀|² = 1.495935e-14
theta_OFF = 8.57          # well away from BIC — strong reflection

print("=" * 60)
print("COMPUTING FIELDS FOR TIME-DEPENDENT ANIMATION")
print("=" * 60)

print(f"\n1. Computing OFF-BIC field (theta = {theta_OFF}°)...")
t0 = time.time()
Hy_off, r0_off, yGrid, xGrid = solve_complex_field(np.radians(theta_OFF))
print(f"   |r0|^2 = {np.abs(r0_off)**2:.4e}  ({time.time()-t0:.1f}s)")

print(f"\n2. Computing BIC field (theta = {theta_BIC}°)...")
t0 = time.time()
Hy_bic, r0_bic, yGrid, xGrid = solve_complex_field(np.radians(theta_BIC))
print(f"   |r0|^2 = {np.abs(r0_bic)**2:.4e}  ({time.time()-t0:.1f}s)")

# =====================================================================
# ANIMATE: Re[H(r) * exp(-i*omega*t)]
# =====================================================================

n_time_frames = 48  # frames per period (smooth animation)
omega_t_values = np.linspace(0, 2*pi, n_time_frames, endpoint=False)

# Precompute all frames
print(f"\nPrecomputing {n_time_frames} time frames...")
frames_off = []
frames_bic = []
for wt in omega_t_values:
    phase = np.exp(-1j * wt)
    frames_off.append(np.real(Hy_off * phase))
    frames_bic.append(np.real(Hy_bic * phase))

# Find global color limits for consistent scale
all_vals = np.array(frames_off + frames_bic)
vmax = np.percentile(np.abs(all_vals), 97)
vmin = -vmax

print(f"Color scale: [{vmin:.2f}, {vmax:.2f}]")

# =====================================================================
# CROP TO AIR REGION & SET AIR-SPECIFIC COLOR SCALE
# =====================================================================

yGrid_arr = np.array(yGrid)
air_mask = yGrid_arr < H_fixed

air_frames_off = [f[air_mask, :] for f in frames_off]
air_frames_bic = [f[air_mask, :] for f in frames_bic]
yGrid_air = yGrid_arr[air_mask]

vmax_air = np.percentile(np.abs(np.array(air_frames_off)), 99)
vmin_air = -vmax_air
print(f"Air-region color scale: [{vmin_air:.2f}, {vmax_air:.2f}]")

# =====================================================================
# CREATE FIGURE
# =====================================================================

print("Rendering animation...")

fig, (ax_off, ax_bic) = plt.subplots(1, 2, figsize=(16, 6))
cmap = 'RdBu_r'

# --- Left panel: OFF-BIC (air only) ---
cm_off = ax_off.pcolormesh(xGrid, yGrid_air, air_frames_off[0],
                            cmap=cmap, shading='auto',
                            vmin=vmin_air, vmax=vmax_air)
ax_off.set_xlim([-gratingPeriod/2, gratingPeriod/2])
ax_off.set_ylim([0, H_fixed])
ax_off.set_xlabel(r'$\hat{x}$ (nm)', fontsize=14)
ax_off.set_ylabel(r'$\hat{y}$ (nm)', fontsize=14)
ax_off.set_title(f'OFF-BIC: $\\theta = {theta_OFF}°$\n'
                 f'$|r_0|^2 = {np.abs(r0_off)**2:.3f}$',
                 fontsize=15, fontweight='bold', color='darkred')

# Grating at top edge
ax_off.axhline(y=H_fixed-1, color='black', linewidth=3, linestyle='-', alpha=0.8)
ax_off.text(0, H_fixed - 15, 'GRATING', fontsize=11, color='black',
            fontweight='bold', ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

ax_off.annotate('', xy=(-200, 30), xytext=(-200, H_fixed - 40),
                arrowprops=dict(arrowstyle='->', color='red',
                               lw=3, mutation_scale=20))
ax_off.text(-185, H_fixed*0.45, 'REFLECTED\nWAVE', fontsize=11, color='red',
            fontweight='bold', alpha=0.9)

# --- Right panel: BIC (air only) ---
cm_bic = ax_bic.pcolormesh(xGrid, yGrid_air, air_frames_bic[0],
                            cmap=cmap, shading='auto',
                            vmin=vmin_air, vmax=vmax_air)
ax_bic.set_xlim([-gratingPeriod/2, gratingPeriod/2])
ax_bic.set_ylim([0, H_fixed])
ax_bic.set_xlabel(r'$\hat{x}$ (nm)', fontsize=14)
ax_bic.set_ylabel(r'$\hat{y}$ (nm)', fontsize=14)
ax_bic.set_title(f'BIC: $\\theta = {theta_BIC:.4f}°$\n'
                 f'$|r_0|^2 = {np.abs(r0_bic)**2:.2e}$',
                 fontsize=15, fontweight='bold', color='darkgreen')

ax_bic.axhline(y=H_fixed-1, color='black', linewidth=3, linestyle='-', alpha=0.8)
ax_bic.text(0, H_fixed - 15, 'GRATING', fontsize=11, color='black',
            fontweight='bold', ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

ax_bic.text(0, H_fixed*0.45, 'Hopefully no\nreflection...', fontsize=14,
            color='green', fontweight='bold', alpha=0.9, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Shared colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(cm_bic, cax=cbar_ax)
cbar.set_label(r'Re[$H_y \cdot e^{-i\omega t}$]', fontsize=13)

# Time indicator
time_text = fig.text(0.45, 0.02,
                     r'$\omega t = 0$',
                     fontsize=16, ha='center', fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='lightyellow',
                               alpha=0.9))

plt.subplots_adjust(left=0.06, right=0.90, top=0.85, bottom=0.10,
                    wspace=0.15)


def update(frame):
    cm_off.set_array(air_frames_off[frame].ravel())
    cm_bic.set_array(air_frames_bic[frame].ravel())

    wt = omega_t_values[frame]
    time_text.set_text(f'$\\omega t = {wt:.2f}$ rad '
                       f'({wt/(2*pi)*100:.0f}% of period)')
    return cm_off, cm_bic, time_text


anim = FuncAnimation(fig, update, frames=n_time_frames,
                     interval=80, blit=False, repeat=True)

# Save GIF
print("Saving GIF...")
writer = PillowWriter(fps=12)
anim.save('BIC_wave_propagation_air.gif', writer=writer, dpi=120)
print("Saved: BIC_wave_propagation_air.gif")

try:
    from matplotlib.animation import FFMpegWriter
    writer_mp4 = FFMpegWriter(fps=15, bitrate=3000)
    anim.save('BIC_wave_propagation_air.mp4', writer=writer_mp4, dpi=150)
    print("Saved: BIC_wave_propagation_air.mp4")
except Exception as e:
    print(f"MP4 skipped: {e}")

plt.show()
print("\nDone!")

'''
Including everything 

# =====================================================================
# CREATE FIGURE
# =====================================================================

print("Rendering animation...")

fig, (ax_off, ax_bic) = plt.subplots(1, 2, figsize=(16, 8))

# Diverging colormap: blue = negative, white = zero, red = positive
# This shows wave crests and troughs clearly
cmap = 'RdBu_r'

# --- Left panel: OFF-BIC ---
cm_off = ax_off.pcolormesh(xGrid, yGrid, frames_off[0],
                            cmap=cmap, shading='auto',
                            vmin=vmin, vmax=vmax)
ax_off.axhline(y=H, color='black', linewidth=2, linestyle='--', alpha=0.8)
ax_off.set_xlim([-gratingPeriod/2, gratingPeriod/2])
ax_off.set_ylim([0, total_height])
ax_off.set_xlabel(r'$\hat{x}$ (nm)', fontsize=14)
ax_off.set_ylabel(r'$\hat{y}$ (nm)', fontsize=14)
ax_off.set_title(f'OFF-BIC: $\\theta = {theta_OFF}°$\n'
                 f'$|r_0|^2 = {np.abs(r0_off)**2:.3f}$',
                 fontsize=15, fontweight='bold', color='darkred')

# Region labels
ax_off.text(-240, H*0.4, 'AIR', fontsize=16, color='black',
            fontweight='bold', alpha=0.5,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
ax_off.text(-240, H + 400, 'DIELECTRIC', fontsize=14, color='black',
            fontweight='bold', alpha=0.5,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

# Arrow showing reflected wave
ax_off.annotate('', xy=(-200, 50), xytext=(-200, H - 30),
                arrowprops=dict(arrowstyle='->', color='red',
                               lw=3, mutation_scale=20))
ax_off.text(-190, H*0.3, 'REFLECTED\nWAVE', fontsize=10, color='red',
            fontweight='bold', alpha=0.8)

# --- Right panel: BIC ---
cm_bic = ax_bic.pcolormesh(xGrid, yGrid, frames_bic[0],
                            cmap=cmap, shading='auto',
                            vmin=vmin, vmax=vmax)
ax_bic.axhline(y=H, color='black', linewidth=2, linestyle='--', alpha=0.8)
ax_bic.set_xlim([-gratingPeriod/2, gratingPeriod/2])
ax_bic.set_ylim([0, total_height])
ax_bic.set_xlabel(r'$\hat{x}$ (nm)', fontsize=14)
ax_bic.set_ylabel(r'$\hat{y}$ (nm)', fontsize=14)
ax_bic.set_title(f'BIC: $\\theta = {theta_BIC}°$\n'
                 f'$|r_0|^2 = {np.abs(r0_bic)**2:.2e}$',
                 fontsize=15, fontweight='bold', color='darkgreen')

# Region labels
ax_bic.text(-240, H*0.4, 'AIR', fontsize=16, color='black',
            fontweight='bold', alpha=0.5,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
ax_bic.text(-240, H + 400, 'DIELECTRIC', fontsize=14, color='black',
            fontweight='bold', alpha=0.5,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

# Arrow showing NO reflected wave
ax_bic.text(-220, H*0.3, 'NO\nREFLECTION!', fontsize=12,
            color='green', fontweight='bold', alpha=0.8,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Shared colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(cm_bic, cax=cbar_ax)
cbar.set_label(r'Re[$H_y(\mathbf{r}) \cdot e^{-i\omega t}$]',
               fontsize=13)

# Time indicator
time_text = fig.text(0.45, 0.02,
                     r'$\omega t = 0$',
                     fontsize=16, ha='center',
                     fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='lightyellow',
                               alpha=0.9))

plt.subplots_adjust(left=0.06, right=0.90, top=0.88, bottom=0.08,
                    wspace=0.15)


def update(frame):
    """Update both panels for time step."""
    cm_off.set_array(frames_off[frame].ravel())
    cm_bic.set_array(frames_bic[frame].ravel())

    wt = omega_t_values[frame]
    time_text.set_text(f'$\\omega t = {wt:.2f}$ rad '
                       f'({wt/(2*pi)*100:.0f}% of period)')

    return cm_off, cm_bic, time_text


anim = FuncAnimation(fig, update, frames=n_time_frames,
                     interval=80, blit=False, repeat=True)

# Save GIF
print("Saving GIF...")
writer = PillowWriter(fps=12)
anim.save('BIC_wave_propagation.gif', writer=writer, dpi=120)
print("Saved: BIC_wave_propagation.gif")

# Try MP4
try:
    from matplotlib.animation import FFMpegWriter
    writer_mp4 = FFMpegWriter(fps=15, bitrate=3000)
    anim.save('BIC_wave_propagation.mp4', writer=writer_mp4, dpi=150)
    print("Saved: BIC_wave_propagation.mp4")
except Exception as e:
    print(f"MP4 skipped: {e}")

plt.show()
print("\nDone!")
'''