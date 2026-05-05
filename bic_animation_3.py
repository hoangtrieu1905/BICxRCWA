# BIC Wave Propagation Animation — H Sensitivity
"""
Shows how deviating from H_BIC breaks the BIC condition.

Both panels use the same angle (theta_BIC). The only difference is H:
  Left  — H = H_off  (wrong grating height  → strong reflection)
  Right — H = H_BIC  (optimized height       → BIC, near-zero reflection)

Author: Hoang Trieu
Date Updated: May 4, 2026
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

H = 1104.372816
fourierMode = 5
m_fourier = 2 * fourierMode + 1
numLayers = 2
gratingPeriod = 500
yDiscreteSize = np.ones(numLayers)
xDiscreteSize = 1
sampleP = 500

ep = np.empty(numLayers, dtype=object)
ep[0] = 1 + 1j * 1e-9
ep[1] = 12.0 + 0j

g = lambda x, gP: 50 * np.exp(-(x / 50)**2)
gPrime = lambda x, gP: -2 * x / 50**2 * g(x, gP)


# =====================================================================
# RCWA SOLVER (returns complex field for time animation)
# =====================================================================

def solve_complex_field(theta_rad, H_optimized):
    """Run RCWA and return the COMPLEX Hyhat field (not magnitude)."""
    S_func = lambda y: 0.5 * (1 + np.cos(pi / H_optimized * (y - H_optimized)))
    SPrime = lambda y: -(pi / (2*H_optimized)) * np.sin(pi / H_optimized * (y - H_optimized))
    detDG  = lambda x, y: SPrime(y) * g(x, gratingPeriod) + 1
    a11 = lambda x, y: np.abs(detDG(x, y))
    a21 = lambda x, y: -np.sign(detDG(x, y)) * S_func(y) * gPrime(x, gratingPeriod)
    a12 = lambda x, y: a21(x, y)
    a22 = lambda x, y: ((S_func(y) * gPrime(x, gratingPeriod))**2 + 1) / np.abs(detDG(x, y))

    layerThick = np.array([H_optimized, 700], dtype=float)
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

        ep11 = lambda x: ep[layer] * a11(x, yGrid[i])
        ep21 = lambda x: ep[layer] * a21(x, yGrid[i])
        ep22_f = lambda x: ep[layer] * a22(x, yGrid[i])
        ep33 = lambda x: ep[layer] * a11(x, yGrid[i])
        mu11 = lambda x: a11(x, yGrid[i])
        mu21 = lambda x: a21(x, yGrid[i])
        mu22_f = lambda x: a22(x, yGrid[i])
        mu33 = lambda x: a11(x, yGrid[i])

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

    return Hyhat, r0, yGrid, xGrid


# =====================================================================
# COMPUTE FIELDS — same theta, two different H values
# =====================================================================

theta_BIC = 16.024928          # angle of the BIC resonance
H_BIC = 1104.372816           # optimized grating height — achieves BIC
H_off = 700                     # wrong height — breaks BIC condition

print("=" * 60)
print("H SENSITIVITY: same theta, two different grating heights")
print("=" * 60)

print(f"\n1. BIC     — theta = {theta_BIC}°,  H = {H_BIC} nm...")
t0 = time.time()
Hy_bic, r0_bic, yGrid_bic, xGrid_bic = solve_complex_field(np.radians(theta_BIC), H_BIC)
print(f"   |r0|^2 = {np.abs(r0_bic)**2:.4e}  ({time.time()-t0:.1f}s)")

print(f"\n2. OFF-BIC — theta = {theta_BIC}°,  H = {H_off} nm...")
t0 = time.time()
Hy_off, r0_off, yGrid_off, xGrid_off = solve_complex_field(np.radians(theta_BIC), H_off)
print(f"   |r0|^2 = {np.abs(r0_off)**2:.4e}  ({time.time()-t0:.1f}s)")

# =====================================================================
# ANIMATE: Re[H(r) * exp(-i*omega*t)]
# =====================================================================

n_time_frames = 48
omega_t_values = np.linspace(0, 2*pi, n_time_frames, endpoint=False)

print(f"\nPrecomputing {n_time_frames} time frames...")
frames_bic = []
frames_off = []
for wt in omega_t_values:
    phase = np.exp(-1j * wt)
    frames_bic.append(np.real(Hy_bic * phase))
    frames_off.append(np.real(Hy_off * phase))

# Each frame list is internally uniform; compute vmax from them separately
vmax = max(np.percentile(np.abs(np.array(frames_bic)), 97),
           np.percentile(np.abs(np.array(frames_off)), 97))
vmin = -vmax
print(f"Color scale: [{vmin:.2f}, {vmax:.2f}]")

# Use the larger total height for a consistent y-axis across both panels
total_height_bic = H_BIC + 900
total_height_off = H_off + 900
ylim_top = max(total_height_bic, total_height_off)

# =====================================================================
# CREATE FIGURE
# =====================================================================

print("Rendering animation...")

fig, (ax_off, ax_bic) = plt.subplots(1, 2, figsize=(16, 8))
cmap = 'RdBu_r'

# --- Left panel: OFF-BIC (H = H_off) ---
cm_off = ax_off.pcolormesh(xGrid_off, yGrid_off, frames_off[0],
                            cmap=cmap, shading='auto',
                            vmin=vmin, vmax=vmax)
ax_off.axhline(y=H_off, color='black', linewidth=2, linestyle='--', alpha=0.8)
ax_off.set_xlim([-gratingPeriod/2, gratingPeriod/2])
ax_off.set_ylim([0, ylim_top])
ax_off.set_xlabel(r'$\hat{x}$ (nm)', fontsize=14)
ax_off.set_ylabel(r'$\hat{y}$ (nm)', fontsize=14)
ax_off.set_title(f'OFF-BIC:  H = {H_off} nm\n'
                 f'$\\theta = {theta_BIC}°$,   '
                 f'$|r_0|^2 = {np.abs(r0_off)**2:.3f}$',
                 fontsize=15, fontweight='bold', color='darkred')

ax_off.text(-240, H_off * 0.4, 'AIR', fontsize=16, color='black',
            fontweight='bold', alpha=0.5,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
ax_off.text(-240, H_off + 400, 'DIELECTRIC', fontsize=14, color='black',
            fontweight='bold', alpha=0.5,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

ax_off.annotate('', xy=(-200, 50), xytext=(-200, H_off - 30),
                arrowprops=dict(arrowstyle='->', color='red',
                               lw=3, mutation_scale=20))
ax_off.text(-190, H_off * 0.3, 'REFLECTED\nWAVE', fontsize=10, color='red',
            fontweight='bold', alpha=0.8)

# --- Right panel: BIC (H = H_BIC) ---
cm_bic = ax_bic.pcolormesh(xGrid_bic, yGrid_bic, frames_bic[0],
                            cmap=cmap, shading='auto',
                            vmin=vmin, vmax=vmax)
ax_bic.axhline(y=H_BIC, color='black', linewidth=2, linestyle='--', alpha=0.8)
ax_bic.set_xlim([-gratingPeriod/2, gratingPeriod/2])
ax_bic.set_ylim([0, ylim_top])
ax_bic.set_xlabel(r'$\hat{x}$ (nm)', fontsize=14)
ax_bic.set_ylabel(r'$\hat{y}$ (nm)', fontsize=14)
ax_bic.set_title(f'BIC:  H = {H_BIC:.2f} nm\n'
                 f'$\\theta = {theta_BIC}°$,   '
                 f'$|r_0|^2 = {np.abs(r0_bic)**2:.2e}$',
                 fontsize=15, fontweight='bold', color='darkgreen')

ax_bic.text(-240, H_BIC * 0.4, 'AIR', fontsize=16, color='black',
            fontweight='bold', alpha=0.5,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
ax_bic.text(-240, H_BIC + 400, 'DIELECTRIC', fontsize=14, color='black',
            fontweight='bold', alpha=0.5,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

ax_bic.text(-220, H_BIC * 0.3, 'NO\nREFLECTION!', fontsize=12,
            color='green', fontweight='bold', alpha=0.8,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Shared colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(cm_bic, cax=cbar_ax)
cbar.set_label(r'Re[$H_y(\mathbf{r}) \cdot e^{-i\omega t}$]', fontsize=13)

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
    cm_off.set_array(frames_off[frame].ravel())
    cm_bic.set_array(frames_bic[frame].ravel())

    wt = omega_t_values[frame]
    time_text.set_text(f'$\\omega t = {wt:.2f}$ rad '
                       f'({wt/(2*pi)*100:.0f}% of period)')
    return cm_off, cm_bic, time_text


anim = FuncAnimation(fig, update, frames=n_time_frames,
                     interval=80, blit=False, repeat=True)

print("Saving GIF...")
writer = PillowWriter(fps=12)
anim.save('BIC_H_sensitivity.gif', writer=writer, dpi=120)
print("Saved: BIC_H_sensitivity.gif")

try:
    from matplotlib.animation import FFMpegWriter
    writer_mp4 = FFMpegWriter(fps=15, bitrate=3000)
    anim.save('BIC_H_sensitivity.mp4', writer=writer_mp4, dpi=150)
    print("Saved: BIC_H_sensitivity.mp4")
except Exception as e:
    print(f"MP4 skipped: {e}")

plt.show()
print("\nDone!")
