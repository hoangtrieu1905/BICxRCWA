# C_RCWA Method in Python - Reflection Kernel Refactor
"""
Author: Matlab Code - Dr. Benjamin Civiletti /
        Python Code - Hoang Trieu

First Updated: June 6, 2024
Last Updated: May 2, 2026
"""

from math import *
from numpy import pi, sin
import numpy as np
from scipy import linalg
import scipy


"""Helper Functions"""
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
    Y_top = np.concatenate((matrix1, matrix2), axis=1)
    Y_bottom = np.concatenate((matrix3, matrix4), axis=1)
    Y = np.concatenate((Y_top, Y_bottom), axis=0)
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
    array2 = y[2 : (numSub - 1) : 2]
    sumArray1 = np.sum(array1)
    sumArray2 = np.sum(array2)
    approxInt = (h / 3) * (y[0] + y[numSub] + 4 * sumArray1 + 2 * sumArray2)
    return approxInt


def bisection(f, left, right, TOL):
    """Bisection root finder on [left, right]."""
    n = 0
    if f(left) * f(right) >= 0:
        raise ValueError("There is no root of f(x) in [" + str(left) + "," + str(right) + "]")

    while (right - left) / 2 > TOL:
        mid = (left + right) / 2
        if f(mid) == 0:
            break

        if f(left) * f(mid) < 0:
            right = mid
        else:
            left = mid
        n += 1
    root = (right + left) / 2
    return root, n


def ToeplitzM(m, f, L, sampleP):
    """Construct Toeplitz matrix of Fourier coefficients."""
    n = 2 * m - 1
    fftvec = np.zeros(n, dtype=complex)
    ind = (n - 1) / 2
    j = np.arange(-ind, ind + 1)
    if callable(f):
        for i in range(n):
            k = lambda x: f(x) * np.exp(-2 * pi * 1j * j[i] * x / L)
            fftvec[i] = (1 / L) * simpson(k, -L / 2, L / 2, sampleP)
    elif isinstance(f, np.ndarray):
        for i in range(n):
            ind = (n - 1) / 2
            j = np.arange(-ind, ind + 1)
            x = np.linspace(-L / 2, L / 2, sampleP + 1)
            basis = np.exp(-2 * pi * 1j * j[i] * x / L)
            if len(basis) != len(f):
                raise ValueError("Lengths of basis and f do not match")
            fSample = f * basis
            fftvec[i] = (1 / L) * simpson(fSample, -L / 2, L / 2, sampleP)
    else:
        raise ValueError("Invalid input for f at ToeplitzM function")

    fftvec0pos = fftvec[m - 1 :]
    fftvec0neg = np.flip(fftvec[:m])
    T = linalg.toeplitz(fftvec0pos, fftvec0neg)
    return T


def invMap(x, y, g, S, yGrid, H, gratingPeriod):
    """Inverse coordinate map helper (retained for compatibility)."""
    xHat = x
    C = g(x, gratingPeriod)
    F = lambda x: C * S(x) - x + y
    yHat, _ = bisection(F, 0, 2 * H, 10**-3)
    yGrid = np.array(yGrid)
    index = np.argmin(np.abs(yHat - yGrid))
    return xHat, yHat, index


"""Physical Constants"""
epsilon0 = 8.854 * 10 ** (-12)
mu0 = 4 * pi * 10 ** (-7)
eta0 = np.sqrt(mu0 / epsilon0)
lambda0 = 600
k0 = 2 * pi / lambda0
omega = k0 / eta0


def get_reflection(theta_val, slab_thickness, H_fixed=700.0, p_pol='p'):
    # theta_val:       incidence angle in radians
    # slab_thickness:  dielectric layer thickness in nm (the BIC-relevant param)
    # H_fixed:         air buffer below grating -- must be >> grating amplitude (100 nm);
    #                  physically inert once H_fixed >= ~600 nm (confirmed by h_diagnostic.py)
    if p_pol not in ('s', 'p'):
        raise ValueError("p_pol must be either 's' or 'p'.")

    # Solver defaults retained from the original script.
    fourierMode = 5
    m = 2 * fourierMode + 1
    numLayers = 2
    gratingPeriod = 500
    yDiscreteSize = np.ones(numLayers)
    xDiscreteSize = 1
    layerThick = np.array([H_fixed, slab_thickness], dtype=float)
    ep = np.empty(numLayers, dtype=object)
    ep[0] = 1 + 1j * 10 ** (-9)
    ep[1] = 12.0 + 0j
    sampleP = 500
    method = 'C'

    # Grating profile and derivative.
    g = lambda x, gratingPeriod: 100 * np.exp(-(x / 50) ** 2)
    gPrime = lambda x, gratingPeriod: -2 * x / 50 ** 2 * g(x, gratingPeriod)

    # Old single-layer transform (retired):
    #S = lambda y: 0.5 * (1 + np.cos(pi / H_fixed * (y - H_fixed)))
    #SPrime = lambda y: -(pi / (2 * H_fixed)) * (np.sin(pi / H_fixed * (y - H_fixed)))
    
    # Coordinate transform: H1 = fixed air buffer, H2 = variable slab thickness
    H1 = H_fixed
    H2 = slab_thickness

    def S(y):
        return np.where(y <= H1, 
                        0.5 * (1 - np.cos(pi * y / H1)), 
                        0.5 * (1 + np.cos(pi * (y - H1) / H2))) * ((y >= 0) & (y <= H1 + H2))

    def SPrime(y):
        return np.where(y <= H1, 
                        (pi / (2 * H1)) * np.sin(pi * y / H1), 
                        -(pi / (2 * H2)) * np.sin(pi * (y - H1) / H2)) * ((y >= 0) & (y <= H1 + H2))
    detDG = lambda x, y: SPrime(y) * g(x, gratingPeriod) + 1

    a11 = lambda x, y: np.abs(detDG(x, y))
    a21 = lambda x, y: -np.sign(detDG(x, y)) * S(y) * gPrime(x, gratingPeriod)
    a12 = lambda x, y: a21(x, y)
    a22 = lambda x, y: ((S(y) * gPrime(x, gratingPeriod)) ** 2 + 1) / np.abs(detDG(x, y))

    xGrid, yGrid, yBound = gridC(
        numLayers, gratingPeriod, layerThick, xDiscreteSize, yDiscreteSize
    )
    yGridN = np.size(yGrid)

    gR = lambda x, gratingPeriod: -g(x, gratingPeriod) + H_fixed

    k_vector = np.arange(-fourierMode, fourierMode + 1, dtype=complex)
    k = k0 * sin(theta_val) + 2 * pi * k_vector / gratingPeriod
    K = np.diag(k)
    beta = np.sqrt(k0 ** 2 - k ** 2)

    O = np.zeros((m, m))
    I = np.eye(m, m)

    Ye_p = create_Y_matrix(O, -np.diag(beta / k0), I, O)
    Ye_m = create_Y_matrix(O, np.diag(beta / k0), I, O)
    Yh_p = create_Y_matrix(-np.diag(beta / k0), O, O, -I)
    Yh_m = create_Y_matrix(np.diag(beta / k0), O, O, -I)

    A = np.zeros((4 * fourierMode + 2, 1), dtype=complex)
    if p_pol == 's':
        A[fourierMode] = 1
    else:
        A[3 * fourierMode + 1] = 1

    Zs = np.empty(yGridN + 1, dtype=object)
    Zs[yGridN] = np.concatenate((Ye_p, Yh_p), axis=0)

    for i in range(yGridN - 1, -1, -1):
        check = yGrid[i] - yBound > 0
        layer = np.where(check)[0][-1]

        if method == 'C':
            ep11 = lambda x: ep[layer] * a11(x, yGrid[i])
            ep21 = lambda x: ep[layer] * a21(x, yGrid[i])
            ep12 = lambda x: ep[layer] * a12(x, yGrid[i])
            ep22 = lambda x: ep[layer] * a22(x, yGrid[i])
            ep33 = lambda x: ep[layer] * a11(x, yGrid[i])

            mu11 = lambda x: a11(x, yGrid[i])
            mu21 = lambda x: a21(x, yGrid[i])
            mu12 = lambda x: a12(x, yGrid[i])
            mu22 = lambda x: a22(x, yGrid[i])
            mu33 = lambda x: a11(x, yGrid[i])

            Tep11 = ToeplitzM(m, ep11, gratingPeriod, sampleP)
            Tep21 = ToeplitzM(m, ep21, gratingPeriod, sampleP)
            Tep12 = Tep21
            Tep22 = ToeplitzM(m, ep22, gratingPeriod, sampleP)
            Tep33 = ToeplitzM(m, ep33, gratingPeriod, sampleP)

            Tmu11 = (ep[layer] ** (-1)) * Tep11
            Tmu21 = (ep[layer] ** (-1)) * Tep21
            Tmu12 = Tmu21
            Tmu22 = (ep[layer] ** (-1)) * Tep22
            Tmu33 = (ep[layer] ** (-1)) * Tep33

            P11 = K @ (linalg.solve(Tep22, Tep21))
            P14 = k0 * Tmu33 - (1 / k0) * K @ (linalg.solve(Tep22, K))
            P22 = Tmu12 @ (linalg.solve(Tmu22, K))
            P23 = -k0 * (Tmu11 - Tmu12 @ (linalg.solve(Tmu22, Tmu21)))
            P32 = -k0 * Tep33 + (1 / k0) * K @ (linalg.solve(Tmu22, K))
            P33 = K @ (linalg.solve(Tmu22, Tmu21))
            P41 = -k0 * (Tep12 @ (linalg.solve(Tep22, Tep21)) - Tep11)
            P44 = Tep12 @ linalg.solve(Tep22, K)

            P = np.block(
                [
                    [P11, O, O, P14],
                    [O, P22, P23, O],
                    [O, P32, P33, O],
                    [P41, O, O, P44],
                ]
            )

        elif method == 'R':
            ind = np.where(gR(xGrid, gratingPeriod) < yGrid[i])[0]

            if np.size(ind) == np.size(xGrid, 0):
                ep11 = lambda x: ep[layer]
                ep22 = lambda x: ep11(x)
                ep33 = lambda x: ep11(x)
            elif (np.size(ind) != np.size(xGrid, 0)) and np.size(ind) != 0:
                xL = xGrid[ind[0]]
                xR = xGrid[ind[np.size(ind) - 1]]
                ep11 = lambda x: (x < xL) * ep[0] + (x > xR) * ep[0] + ((x >= xL) & (x <= xR)) * ep[1]
                ep22 = lambda x: ep11(x)
                ep33 = lambda x: ep11(x)
            elif ind.size == 0:
                ep11 = lambda x: ep[layer]
                ep22 = lambda x: ep11(x)
                ep33 = lambda x: ep11(x)

            Tep11 = ToeplitzM(m, ep11, gratingPeriod, sampleP)
            Tep21 = O
            Tep12 = O
            Tep22 = Tep11
            Tep33 = Tep11

            Tmu11 = I
            Tmu21 = O
            Tmu12 = O
            Tmu22 = I
            Tmu33 = I

            P14 = k0 * Tmu33 - (1 / k0) * K @ (linalg.solve(Tep22, K))
            P32 = -k0 * Tep33 + (1 / k0) * K @ (linalg.solve(Tmu22, K))
            P41 = -k0 * (Tep12 @ (linalg.solve(Tep22, Tep21)) - Tep11)
            P23 = -k0 * (Tmu11 - Tmu12 @ (linalg.solve(Tmu22, Tmu21)))

            P = np.block([[O, O, O, P14], [O, O, P23, O], [O, P32, O, O], [P41, O, O, O]])

        if i == 0 or i == yGridN - 1:
            Delta = 1
        else:
            Delta = yGrid[i] - yGrid[i - 1]

        D, G = np.linalg.eig(P)
        D = np.diag(D)

        Ddiag = np.diag(D)
        ind = np.argsort(np.imag(Ddiag))[::-1]
        Ddiag = Ddiag[ind]
        D = np.diag(Ddiag)
        G = G[:, ind]
        W = linalg.solve(G, Zs[i + 1])
        Wu = W[0 : 4 * fourierMode + 2, :]
        Wl = W[4 * fourierMode + 2 :, :]
        Dl = D[4 * fourierMode + 2 :, 4 * fourierMode + 2 :]
        expmat1 = scipy.linalg.expm(-1j * Delta * Dl)
        expmat2 = scipy.linalg.expm(1j * Delta * D[0 : 4 * fourierMode + 2, 0 : 4 * fourierMode + 2])

        block_matrix = np.block(
            [
                [np.eye(4 * fourierMode + 2)],
                [expmat1 @ Wl @ np.linalg.solve(Wu, expmat2)],
            ]
        )
        Zs[i] = G @ block_matrix

    Z0u = Zs[0][0 : 4 * fourierMode + 2, :]
    Z0l = Zs[0][4 * fourierMode + 2 :, :]

    top = np.block([[Z0u, -Ye_m], [Z0l, -Yh_m]])
    bottom = np.block([[Ye_p], [Yh_p]])
    X = scipy.linalg.solve(top, bottom)
    T0R = X @ A

    if theta_val == 0.0:
        diagnostic_offset = 4 * fourierMode + 2
        t_index = 3 * fourierMode + 1
        r_index = diagnostic_offset + 3 * fourierMode + 1
        print("\n=== MATRIX DIAGNOSTIC (Theta = 0.0) ===")
        print(f"Calculated T0_Transmission at Index {t_index}: {np.abs(T0R[t_index, 0])**2:.6e}")
        print(f"Calculated T0_Reflection   at Index {r_index}: {np.abs(T0R[r_index, 0])**2:.6f}")
        print("=======================================\n")
    # ---------------------------------------
    offset = 4 * fourierMode + 2
    if p_pol == 's':
        return T0R[offset + fourierMode, 0]
    else:
        # For p-pol, specular reflection is exactly at index 38
        return T0R[offset + 3 * fourierMode + 1, 0]

def objective_function(params):
    theta_deg = params[0]
    slab_val  = params[1]
    theta_rad = np.radians(theta_deg)
    r_0 = get_reflection(theta_rad, slab_val)
    return np.abs(r_0)**2

'''
Old Scouting code
def run_two_stage_sweeps():
    """Run two 1D parameter sweeps using objective_function.

    Stage 1: fix slab=700 nm, sweep theta in [0, 15] deg with 15 points.
    Stage 2: prompt for theta, then fix theta and sweep slab in [300, 700] nm with 20 points.
    """
    import matplotlib.pyplot as plt

    # Stage 1: theta sweep at fixed slab thickness.
    slab_fixed = 700.0
    theta_values = np.linspace(0.0, 15.0, 15)
    theta_intensities = np.array([objective_function([theta, slab_fixed]) for theta in theta_values])

    best_idx = int(np.argmin(theta_intensities))
    best_theta_auto = float(theta_values[best_idx])
    best_intensity_auto = float(theta_intensities[best_idx])

    print("\nStage 1 complete (slab = 700 nm).")
    print(f"Best theta from sweep: {best_theta_auto:.4f} deg (intensity = {best_intensity_auto:.6e})")

    plt.figure(figsize=(8, 5))
    plt.plot(theta_values, theta_intensities, marker='o', linewidth=1.5)
    plt.xlabel('Theta (deg)')
    plt.ylabel('|r_0|^2')
    plt.title('Stage 1: Theta Sweep at Slab = 700 nm')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)

    # Pause for user input before stage 2.
    user_text = input(
        f"Enter your chosen theta in degrees (press Enter to use {best_theta_auto:.4f}): "
    ).strip()
    if user_text == "":
        theta_fixed = best_theta_auto
    else:
        theta_fixed = float(user_text)

    print(f"\nUsing theta = {theta_fixed:.4f} deg for Stage 2.")

    # Stage 2: slab sweep at fixed theta.
    slab_values = np.linspace(300.0, 700.0, 20)
    slab_intensities = np.array([objective_function([theta_fixed, slab_val]) for slab_val in slab_values])

    best_slab_idx = int(np.argmin(slab_intensities))
    best_slab = float(slab_values[best_slab_idx])
    best_slab_intensity = float(slab_intensities[best_slab_idx])

    print(f"Best slab from sweep: {best_slab:.4f} nm (intensity = {best_slab_intensity:.6e})")

    plt.figure(figsize=(8, 5))
    plt.plot(slab_values, slab_intensities, marker='o', linewidth=1.5)
    plt.xlabel('Slab Thickness (nm)')
    plt.ylabel('|r_0|^2')
    plt.title(f'Stage 2: Slab Sweep at Theta = {theta_fixed:.4f} deg')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def run_optimization():
    """Run Nelder-Mead optimization to perfectly trap the BIC."""
    import scipy.optimize as opt
    import time

    # x0 = [theta_deg, slab_thickness_nm] -- update after running run_overnight_bic_hunt
    x0 = [8.5714, 447.3684]
    
    print(f"Starting Nelder-Mead optimization from x0 = {x0}...")
    print("This may take some time depending on how many steps it needs. Grab a coffee!\n")
    
    start_time = time.time()
    
    # Run the continuous optimization
    result = opt.minimize(
        objective_function, 
        x0, 
        method='Nelder-Mead', 
        options={
            'xatol': 1e-4,   # Tolerance for coordinate step size
            'fatol': 1e-6,   # Tolerance for intensity minimum
            'disp': True     # Print convergence messages
        }
    )
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*40)
    print("=== FINAL OPTIMIZATION RESULTS ===")
    print("="*40)
    print(result)
    print("="*40)
    
    if result.success:
        print(f"\nSUCCESS! You have trapped the BIC.")
        print(f"Optimal Theta: {result.x[0]:.5f} deg")
        print(f"Optimal H:     {result.x[1]:.5f} nm")
        print(f"Final Reflection Intensity: {result.fun:.6e}")
    
    print(f"\nTotal optimization time: {elapsed / 60:.2f} minutes")
'''

# def plot_phase_vortex():
#     """Generate a highly zoomed-in 2D map of the complex phase and intensity."""
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import time

#     # Center exactly on the Nelder-Mead optimal coordinate
#     theta_center = 8.1245
#     H_center = 451.88

#     # Create a zoomed-in grid (+/- 0.5 degrees and +/- 5 nm)
#     grid_size = 15
#     theta_vals = np.linspace(theta_center - 0.5, theta_center + 0.5, grid_size)
#     H_vals = np.linspace(H_center - 5.0, H_center + 5.0, grid_size)

#     Theta_grid, H_grid = np.meshgrid(theta_vals, H_vals)
#     Phase_grid = np.zeros_like(Theta_grid)
#     Intensity_grid = np.zeros_like(Theta_grid)

#     print(f"Starting 2D phase sweep ({grid_size}x{grid_size} = {grid_size**2} points)...")
#     print("This will take approximately 75 minutes. Let it run in the background!\n")

#     start_time = time.time()
    
#     for i in range(grid_size):
#         for j in range(grid_size):
#             # The physics engine strictly requires radians!
#             theta_rad = np.radians(theta_vals[j])
#             H_val = H_vals[i]

#             # Get the raw complex amplitude
#             r_0 = get_reflection(theta_rad, H_val)

#             # Store both the phase angle and the intensity
#             Phase_grid[i, j] = np.angle(r_0)
#             Intensity_grid[i, j] = np.abs(r_0)**2

#         print(f"Row {i+1}/{grid_size} completed...")

#     elapsed = (time.time() - start_time) / 60
#     print(f"\nSweep finished in {elapsed:.2f} minutes.")

#     # --- PLOT 1: The Topological Phase Vortex ---
#     plt.figure(figsize=(8, 6))
#     # 'twilight' is a cyclic colormap perfect for phase wrapping (-pi to pi)
#     mesh1 = plt.pcolormesh(Theta_grid, H_grid, Phase_grid, cmap='twilight', shading='auto')
#     plt.colorbar(mesh1, label='Phase Angle (radians)')
#     plt.xlabel('Theta (degrees)')
#     plt.ylabel('H (nm)')
#     plt.title('Topological Phase Vortex around Quasi-BIC')
#     # Plot a white star at the Nelder-Mead center
#     plt.plot(theta_center, H_center, 'w*', markersize=12, markeredgecolor='k', label='Optimal Point')
#     plt.legend()
#     plt.tight_layout()
#     plt.show(block=False)

#     # --- PLOT 2: The Intensity Dip ---
#     plt.figure(figsize=(8, 6))
#     mesh2 = plt.pcolormesh(Theta_grid, H_grid, Intensity_grid, cmap='viridis', shading='auto')
#     plt.colorbar(mesh2, label='Reflection Intensity |r_0|^2')
#     plt.xlabel('Theta (degrees)')
#     plt.ylabel('H (nm)')
#     plt.title('Reflection Intensity Minimum')
#     plt.plot(theta_center, H_center, 'r*', markersize=12, markeredgecolor='k', label='Optimal Point')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

def calculate_winding_number(phase_grid):
    """
    Computes the topological winding number (q) by integrating the phase 
    differences along a closed square contour around the center singularity.
    """
    import numpy as np

    # Find the center index of the grid
    # (Assuming the grid is square and the singularity is in the dead center)
    center_idx = phase_grid.shape[0] // 2
    
    # Define a small square contour 'radius' around the center (e.g., 2 pixels out)
    r = 2 
    
    # Extract the phase values along the top, right, bottom, and left edges of the square
    top_edge = phase_grid[center_idx - r, center_idx - r : center_idx + r]
    right_edge = phase_grid[center_idx - r : center_idx + r, center_idx + r]
    bottom_edge = phase_grid[center_idx + r, center_idx + r : center_idx - r : -1]
    left_edge = phase_grid[center_idx + r : center_idx - r : -1, center_idx - r]
    
    # Concatenate edges to form one continuous closed loop
    contour_phases = np.concatenate([top_edge, right_edge, bottom_edge, left_edge])
    
    # Calculate the discrete differences between adjacent points
    phase_diffs = np.diff(contour_phases)
    
    # Close the loop by calculating the difference between the last and first point
    loop_closure = contour_phases[0] - contour_phases[-1]
    phase_diffs = np.append(phase_diffs, loop_closure)
    
    # The critical step: Unwrap the phase differences to handle the branch cut!
    # This forces all differences to exist strictly between -pi and pi.
    wrapped_diffs = np.angle(np.exp(1j * phase_diffs))
    
    # Compute the winding number (Sum of wrapped differences divided by 2*pi)
    q = np.sum(wrapped_diffs) / (2 * np.pi)
    
    print(f"\n=== TOPOLOGICAL PROOF ===")
    print(f"Calculated Winding Number (q): {q:.4f}")
    print(f"Nearest Integer Charge: {int(np.round(q))}")
    
    return int(np.round(q))

def plot_phase_vortex(theta_center, slab_center):
    """Generate a highly zoomed-in 2D map of the complex phase and intensity, saving to disk."""
    import numpy as np
    import matplotlib.pyplot as plt
    import time

    # Create a zoomed-in grid (+/- 0.5 degrees and +/- 5 nm)
    grid_size = 15
    theta_vals = np.linspace(theta_center - 0.5, theta_center + 0.5, grid_size)
    slab_vals  = np.linspace(slab_center - 5.0,  slab_center + 5.0,  grid_size)

    Theta_grid, Slab_grid = np.meshgrid(theta_vals, slab_vals)
    Phase_grid     = np.zeros_like(Theta_grid)
    Intensity_grid = np.zeros_like(Theta_grid)

    print(f"Starting 2D phase sweep ({grid_size}x{grid_size} = {grid_size**2} points)...")
    print("This will take approximately 75 minutes. Saving plots to disk when finished!\n")

    start_time = time.time()

    for i in range(grid_size):
        for j in range(grid_size):
            theta_rad = np.radians(theta_vals[j])
            slab_val  = slab_vals[i]
            r_0 = get_reflection(theta_rad, slab_val)
            Phase_grid[i, j]     = np.angle(r_0)
            Intensity_grid[i, j] = np.abs(r_0)**2

        print(f"Row {i+1}/{grid_size} completed...")

    elapsed = (time.time() - start_time) / 60
    print(f"\nSweep finished in {elapsed:.2f} minutes.")

    # --- CALCULATE TOPOLOGICAL CHARGE ---
    calculate_winding_number(Phase_grid)

    # --- PLOT 1: The Topological Phase Vortex ---
    plt.figure(figsize=(8, 6))
    mesh1 = plt.pcolormesh(Theta_grid, Slab_grid, Phase_grid, cmap='twilight', shading='auto')
    plt.colorbar(mesh1, label='Phase Angle (radians)')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Slab Thickness (nm)')
    plt.title('Topological Phase Vortex around True BIC')
    plt.plot(theta_center, slab_center, 'w*', markersize=12, markeredgecolor='k', label='Optimal Point')
    plt.legend()
    plt.tight_layout()
    plt.savefig("phase_vortex_dielectric.png", dpi=300)
    print("-> Saved: phase_vortex_dielectric.png")
    plt.close()

    # --- PLOT 2: The Intensity Dip ---
    plt.figure(figsize=(8, 6))
    mesh2 = plt.pcolormesh(Theta_grid, Slab_grid, Intensity_grid, cmap='viridis', shading='auto')
    plt.colorbar(mesh2, label='Reflection Intensity |r_0|^2')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Slab Thickness (nm)')
    plt.title('Reflection Intensity Minimum (Dielectric)')
    plt.plot(theta_center, slab_center, 'r*', markersize=12, markeredgecolor='k', label='Optimal Point')
    plt.legend()
    plt.tight_layout()
    plt.savefig("intensity_crater_dielectric.png", dpi=300)
    print("-> Saved: intensity_crater_dielectric.png")
    plt.close()

def run_overnight_bic_hunt():
    import numpy as np
    import scipy.optimize as opt
    import matplotlib.pyplot as plt
    import time
    
    print("=== COMMENCING 2D TRUE-BIC PIPELINE ===\n")
    start_time = time.time()

    # --- STAGE 1: 2D Coarse Grid Scout ---
    print("Stage 1: Scouting 2D Parameter Space...")
    theta_vals = np.linspace(0.0, 30.0, 30)
    slab_vals  = np.linspace(200.0, 1200.0, 25)

    Theta_grid, Slab_grid = np.meshgrid(theta_vals, slab_vals)
    Intensity_grid = np.zeros_like(Theta_grid)

    total_iters = len(slab_vals) * len(theta_vals)
    current_iter = 0
    for i in range(len(slab_vals)):
        for j in range(len(theta_vals)):
            current_iter += 1
            print(f"Scouting grid point {current_iter}/{total_iters}: "
                  f"Theta = {theta_vals[j]:.4f}, Slab = {slab_vals[i]:.4f} nm...",
                  end="", flush=True)
            try:
                Intensity_grid[i, j] = objective_function([theta_vals[j], slab_vals[i]])
                print(f" Done. Intensity: {Intensity_grid[i, j]:.4e}")
            except Exception as e:
                print(f" ERROR: {e}")
                raise

    # ── Save scout data ─────────────────────────────────────────
    import os, json
    import numpy as np

    os.makedirs("results", exist_ok=True)

    # Intensity_grid is already a 2D array with shape (N_slab, N_theta) — no reshape needed
    _intensity_array = Intensity_grid

    np.savez_compressed(
        "results/scout_data.npz",
        theta_vals     = np.array(theta_vals),
        slab_vals      = np.array(slab_vals),
        intensity_map  = _intensity_array,
    )
    print(f"[SAVE] Scout data saved to results/scout_data.npz")
    print(f"[SAVE] Shape: theta={len(theta_vals)}, "
          f"slab={len(slab_vals)}, "
          f"intensity={_intensity_array.shape}")
    print(f"[SAVE] Intensity range: "
          f"min={_intensity_array.min():.4e}, "
          f"max={_intensity_array.max():.4f}")
    print(f"[SAVE] Global minimum at: "
          f"theta={theta_vals[np.unravel_index(_intensity_array.argmin(), _intensity_array.shape)[1]]:.4f} deg, "
          f"slab={slab_vals[np.unravel_index(_intensity_array.argmin(), _intensity_array.shape)[0]]:.2f} nm")

    # Also save human-readable JSON for quick inspection
    _meta = {
        "theta_min": float(np.array(theta_vals).min()),
        "theta_max": float(np.array(theta_vals).max()),
        "n_theta":   int(len(theta_vals)),
        "slab_min":  float(np.array(slab_vals).min()),
        "slab_max":  float(np.array(slab_vals).max()),
        "n_slab":    int(len(slab_vals)),
        "intensity_global_min": float(_intensity_array.min()),
        "intensity_global_max": float(_intensity_array.max()),
        "min_location_theta":   float(theta_vals[
            np.unravel_index(_intensity_array.argmin(),
                             _intensity_array.shape)[1]]),
        "min_location_slab":    float(slab_vals[
            np.unravel_index(_intensity_array.argmin(),
                             _intensity_array.shape)[0]]),
    }
    with open("results/scout_meta.json", "w") as f:
        json.dump(_meta, f, indent=2)
    print(f"[SAVE] Metadata saved to results/scout_meta.json")
    # ── End save block ───────────────────────────────────────────

    # ================================================================
    # STEP 2a — Summarize the intensity map
    # intensity_map[i, j]: i indexes slab_vals, j indexes theta_vals
    # ================================================================
    from matplotlib.colors import LogNorm
    import json

    intensity_map = Intensity_grid  # alias: shape (N_slab, N_theta)

    min_idx = np.unravel_index(np.argmin(intensity_map), intensity_map.shape)
    max_idx = np.unravel_index(np.argmax(intensity_map), intensity_map.shape)
    min_val   = intensity_map[min_idx]
    max_val   = intensity_map[max_idx]
    min_theta = theta_vals[min_idx[1]]
    min_slab  = slab_vals[min_idx[0]]
    max_theta = theta_vals[max_idx[1]]
    max_slab  = slab_vals[max_idx[0]]

    print(f"\nScout complete. Intensity map shape: (N_slab, N_theta) = ({len(slab_vals)}, {len(theta_vals)})")
    print(f"Global min: {min_val:.4e} at theta={min_theta:.2f} deg, slab={min_slab:.1f} nm")
    print(f"Global max: {max_val:.4f} at theta={max_theta:.2f} deg, slab={max_slab:.1f} nm")

    # ================================================================
    # STEP 2b — Find Fano ridge points (intensity > 0.5)
    # ================================================================
    fano_ridge_points = []
    for i in range(len(slab_vals)):
        for j in range(len(theta_vals)):
            if intensity_map[i, j] > 0.5:
                fano_ridge_points.append((i, j))

    print(f"\nFano ridge: found {len(fano_ridge_points)} grid points with intensity > 0.5")

    slab_to_thetas = {}
    for (i, j) in fano_ridge_points:
        s = slab_vals[i]
        if s not in slab_to_thetas:
            slab_to_thetas[s] = []
        slab_to_thetas[s].append(theta_vals[j])
    for s in sorted(slab_to_thetas.keys()):
        theta_list = sorted(slab_to_thetas[s])
        theta_str = ", ".join(f"{t:.2f}" for t in theta_list)
        print(f"  Slab={s:.1f} nm: Fano peaks at theta = [{theta_str}]")

    # ================================================================
    # STEP 2c — Find BIC candidate seeds
    # A grid point is a BIC candidate if:
    #   1. intensity < 0.01  (dark)
    #   2. at least one neighbor within ±2 theta steps (same slab row) has intensity > 0.3
    # ================================================================
    bic_candidates = []

    for i in range(len(slab_vals)):
        for j in range(len(theta_vals)):
            if intensity_map[i, j] >= 0.01:
                continue
            has_nearby_fano = False
            for dj in [-2, -1, 1, 2]:
                jj = j + dj
                if 0 <= jj < len(theta_vals) and intensity_map[i, jj] > 0.3:
                    has_nearby_fano = True
                    break
            if has_nearby_fano:
                bic_candidates.append({
                    'theta': float(theta_vals[j]),
                    'slab':  float(slab_vals[i]),
                    'intensity': float(intensity_map[i, j]),
                    'type': 'FW'
                })

    # Symmetry-protected BIC candidates: theta=0 rows that are dark
    # Seed at theta=0.5 to give the optimizer room to move in both theta directions
    for i in range(len(slab_vals)):
        if intensity_map[i, 0] < 0.05:
            bic_candidates.append({
                'theta': 0.5,
                'slab':  float(slab_vals[i]),
                'intensity': float(intensity_map[i, 0]),
                'type': 'symmetry-protected'
            })

    print(f"\nBIC candidate seeds found: {len(bic_candidates)} total")
    for k, cand in enumerate(bic_candidates):
        label = "FW candidate" if cand['type'] == 'FW' else "symmetry-protected candidate"
        print(f"  Seed {k+1}: theta={cand['theta']:.4f} deg, slab={cand['slab']:.2f} nm, "
              f"|r0|^2={cand['intensity']:.4e}")
        print(f"  [{label}]")

    # Fallback: if no candidates found, seed at steepest intensity drop in theta direction
    # (most negative dI/dtheta), NOT at max intensity — avoids landing on a Fano peak
    if len(bic_candidates) == 0:
        print("\nWARNING: No BIC candidates found with adjacency criterion.")
        print("Falling back to steepest intensity DROP in theta direction (not max intensity).")
        _, dI_dTheta = np.gradient(intensity_map, slab_vals, theta_vals)
        fallback_idx = np.unravel_index(np.argmin(dI_dTheta), dI_dTheta.shape)
        bic_candidates.append({
            'theta': float(theta_vals[fallback_idx[1]]),
            'slab':  float(slab_vals[fallback_idx[0]]),
            'intensity': float(intensity_map[fallback_idx]),
            'type': 'fallback-gradient'
        })
        print(f"  Fallback seed: theta={bic_candidates[-1]['theta']:.4f} deg, "
              f"slab={bic_candidates[-1]['slab']:.2f} nm")

    # ================================================================
    # TASK 3 — Nelder-Mead descent from each BIC candidate seed
    # ================================================================
    def bounded_objective(x):
        theta_deg, slab_nm = x[0], x[1]
        if not (0.0 <= theta_deg <= 30.0 and 200.0 <= slab_nm <= 1200.0):
            return 1.0
        return objective_function([theta_deg, slab_nm])

    descent_results = []
    N_seeds = len(bic_candidates)

    for seed_idx, cand in enumerate(bic_candidates):
        t0 = cand['theta']
        s0 = cand['slab']
        v0 = cand['intensity']

        print(f"\n{'='*64}")
        print(f"Starting descent {seed_idx+1}/{N_seeds} from theta={t0:.4f}, slab={s0:.2f}")
        print(f"Initial |r0|^2 = {v0:.4e}")
        print(f"{'='*64}")

        iteration_counter = [0]

        def _make_callback(sidx, counter):
            def callback(xk):
                counter[0] += 1
                val = bounded_objective(xk)
                print(f"  [Descent {sidx}] Iter {counter[0]:4d}: "
                      f"theta={xk[0]:10.6f} deg, slab={xk[1]:10.4f} nm, "
                      f"|r0|^2={val:.6e}  log10={np.log10(val+1e-30):.2f}")
            return callback

        result = opt.minimize(
            bounded_objective,
            [t0, s0],
            method='Nelder-Mead',
            options={'xatol': 1e-6, 'fatol': 1e-12, 'maxiter': 2000, 'disp': False},
            callback=_make_callback(seed_idx + 1, iteration_counter)
        )

        tf = result.x[0]
        sf = result.x[1]
        vf = result.fun

        print(f"Descent {seed_idx+1} converged:")
        print(f"  Final theta    = {tf:.8f} degrees")
        print(f"  Final slab     = {sf:.4f} nm")
        print(f"  Final |r0|^2  = {vf:.6e}")
        print(f"  log10(|r0|^2) = {np.log10(vf+1e-30):.2f}")
        print(f"  Optimizer msg  = {result.message}")

        if vf < 1e-8:
            status = "*** STRONG BIC CANDIDATE — topological check recommended ***"
        elif vf < 1e-3:
            status = "Promising — near-BIC, may need finer grid or more modes"
        else:
            status = "Not a BIC — likely a regular reflection minimum"
        print(f"  STATUS: {status}")

        descent_results.append({
            'seed_theta': t0,
            'seed_slab':  s0,
            'final_theta': tf,
            'final_slab':  sf,
            'r0_squared': float(vf),
            'log10_r0_squared': float(np.log10(vf + 1e-30)),
            'status': status,
            'optimizer_message': result.message
        })

    # ================================================================
    # TASK 4 — Final summary table + save to bic_candidates.json
    # ================================================================
    print(f"\n{'='*64}")
    print("FINAL BIC HUNT SUMMARY")
    print(f"{'='*64}")
    print("  # |   Seed theta |  Seed slab |  Final theta |  Final slab | |r0|^2    | Status")
    print("  --|--------------|------------|--------------|-------------|----------|-------")

    for k, dr in enumerate(descent_results):
        if dr['r0_squared'] < 1e-8:
            st = "BIC"
        elif dr['r0_squared'] < 1e-3:
            st = "near-BIC"
        else:
            st = "no"
        print(f"  {k+1} | {dr['seed_theta']:12.4f} | {dr['seed_slab']:10.2f} | "
              f"{dr['final_theta']:12.6f} | {dr['final_slab']:11.4f} | "
              f"{dr['r0_squared']:.3e} | {st}")

    best = min(descent_results, key=lambda x: x['r0_squared'])
    print(f"\nBest result: theta={best['final_theta']:.6f} deg, "
          f"slab={best['final_slab']:.4f} nm, |r0|^2={best['r0_squared']:.4e}")

    json_output = []
    for dr in descent_results:
        json_output.append({
            "theta_deg":          dr['final_theta'],
            "slab_nm":            dr['final_slab'],
            "r0_squared":         dr['r0_squared'],
            "log10_r0_squared":   dr['log10_r0_squared'],
            "seed_theta":         dr['seed_theta'],
            "seed_slab":          dr['seed_slab'],
            "fourier_mode":       5
        })
    with open("bic_candidates.json", "w") as f:
        json.dump(json_output, f, indent=2)
    print("-> Saved: bic_candidates.json")

    # ================================================================
    # TASK 5 — Save scout heatmap with Fano ridges and BIC markers
    # ================================================================
    fano_thetas_plot = np.array([theta_vals[j] for (i, j) in fano_ridge_points]) if fano_ridge_points else np.array([])
    fano_slabs_plot  = np.array([slab_vals[i]  for (i, j) in fano_ridge_points]) if fano_ridge_points else np.array([])

    seed_thetas_plot = np.array([c['theta'] for c in bic_candidates])
    seed_slabs_plot  = np.array([c['slab']  for c in bic_candidates])

    final_thetas_plot = np.array([dr['final_theta'] for dr in descent_results])
    final_slabs_plot  = np.array([dr['final_slab']  for dr in descent_results])

    fig, ax = plt.subplots(figsize=(10, 7))
    clipped = np.clip(intensity_map, 1e-4, None)
    mesh = ax.pcolormesh(
        Theta_grid, Slab_grid, clipped,
        cmap='inferno', shading='auto',
        norm=LogNorm(vmin=1e-4, vmax=1.0)
    )
    plt.colorbar(mesh, ax=ax, label='|r_0|^2 (log scale)')

    if len(fano_ridge_points) > 0:
        ax.scatter(fano_thetas_plot, fano_slabs_plot,
                   c='white', s=10, alpha=0.4, zorder=3, label='Fano ridge')
    if len(bic_candidates) > 0:
        ax.scatter(seed_thetas_plot, seed_slabs_plot,
                   marker='*', c='red', s=150, zorder=5, label='BIC seeds')
    if len(descent_results) > 0:
        ax.scatter(final_thetas_plot, final_slabs_plot,
                   marker='o', c='yellow', s=200, zorder=6,
                   edgecolors='black', linewidths=0.8, label='Converged BICs')

    ax.set_xlabel('Theta (degrees)')
    ax.set_ylabel('Slab Thickness (nm)')
    ax.set_title('Stage 1: 2D Scout — Fano Ridges (white) and BIC Seeds (red★)')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig("stage1_bic_scout.png", dpi=300)
    plt.close()
    print("-> Saved: stage1_bic_scout.png")

    optimal_theta = best['final_theta']
    optimal_slab  = best['final_slab']
    elapsed = (time.time() - start_time) / 3600
    print(f"\n=== PIPELINE COMPLETE. Total runtime: {elapsed:.2f} hours ===")

    return optimal_theta, optimal_slab


if __name__ == "__main__":
    #run_two_stage_sweeps()
    #run_optimization()
    optimal_theta, optimal_slab = run_overnight_bic_hunt()

    # Run ONLY the phase map around a known BIC candidate (update these values after each hunt):
    # optimal_theta = 1.51490625
    # optimal_slab  = 267.00795001

    if optimal_theta is not None and optimal_slab is not None:
        print("Running targeted phase map to calculate winding number...")
        plot_phase_vortex(optimal_theta, optimal_slab)
    else:
        print("Failed to find optimal parameters. Skipping phase map.")