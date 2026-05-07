import numpy as np
from math import pi, sin
import scipy
from scipy import linalg
from rcwa_v2 import (
    gridC,
    create_Y_matrix,
    ToeplitzM,
    k0,
    get_reflection
)

theta_BIC = np.radians(1.51490625)
H_BIC = 267.00795001

def get_reflection_with_S(theta_inc, H_val, S, SPrime, fourier_mode=5, p_pol='p'):
    if p_pol not in ('s', 'p'):
        raise ValueError("p_pol must be either 's' or 'p'.")

    fourierMode = fourier_mode
    m = 2 * fourierMode + 1
    numLayers = 2
    gratingPeriod = 500
    yDiscreteSize = np.ones(numLayers)
    xDiscreteSize = 1
    layerThick = np.array([H_val, 700], dtype=float)
    ep = np.empty(numLayers, dtype=object)
    ep[0] = 1 + 1j * 10 ** (-9)
    ep[1] = 12.0 + 0j
    sampleP = 500
    method = 'C'

    g = lambda x, gratingPeriod: 50 * np.exp(-(x / 50) ** 2)
    gPrime = lambda x, gratingPeriod: -2 * x / 50 ** 2 * g(x, gratingPeriod)

    detDG = lambda x, y: SPrime(y) * g(x, gratingPeriod) + 1

    a11 = lambda x, y: np.abs(detDG(x, y))
    a21 = lambda x, y: -np.sign(detDG(x, y)) * S(y) * gPrime(x, gratingPeriod)
    a12 = lambda x, y: a21(x, y)
    a22 = lambda x, y: ((S(y) * gPrime(x, gratingPeriod)) ** 2 + 1) / np.abs(detDG(x, y))

    xGrid, yGrid, yBound = gridC(
        numLayers, gratingPeriod, layerThick, xDiscreteSize, yDiscreteSize
    )
    yGridN = np.size(yGrid)

    gR = lambda x, gratingPeriod: -g(x, gratingPeriod) + H_val

    k_vector = np.arange(-fourierMode, fourierMode + 1, dtype=complex)
    k_arr = k0 * sin(theta_inc) + 2 * pi * k_vector / gratingPeriod
    K = np.diag(k_arr)
    beta = np.sqrt(k0 ** 2 - k_arr ** 2)

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

    offset = 4 * fourierMode + 2
    if p_pol == 's':
        return T0R[offset + fourierMode, 0]
    else:
        return T0R[offset + 3 * fourierMode + 1, 0]

def test_refactor():
    print("--- Step B: Verify Refactor ---")
    S_orig = lambda y: 0.5 * (1 + np.cos(pi / H_BIC * (y - H_BIC)))
    SPrime_orig = lambda y: -(pi / (2 * H_BIC)) * (np.sin(pi / H_BIC * (y - H_BIC)))

    r_orig = get_reflection(theta_BIC, H_BIC, p_pol='p')
    r_new = get_reflection_with_S(theta_BIC, H_BIC, S_orig, SPrime_orig, fourier_mode=5, p_pol='p')
    
    int_orig = np.abs(r_orig)**2
    int_new = np.abs(r_new)**2
    
    print(f"Original |r_0|^2: {int_orig:.15e}")
    print(f"Refactor |r_0|^2: {int_new:.15e}")
    
    if not np.isclose(int_orig, int_new, rtol=1e-12):
        print("ERROR: Refactor does not match original!")
        return False
    print("Refactor verified.\n")
    return True

def define_variants():
    # Variant A
    S_a = lambda y: 0.5 * (1 + np.cos(np.pi / H_BIC * (y - H_BIC)))
    SPrime_a = lambda y: -(np.pi / (2 * H_BIC)) * np.sin(np.pi / H_BIC * (y - H_BIC))

    # Variant B
    S_b = lambda y: np.where((y >= 0) & (y <= 2*H_BIC),
                             0.5 * (1 + np.cos(np.pi / H_BIC * (y - H_BIC))),
                             0.0)
    SPrime_b = lambda y: np.where((y >= 0) & (y <= 2*H_BIC),
                                  -(np.pi / (2 * H_BIC)) * np.sin(np.pi / H_BIC * (y - H_BIC)),
                                  0.0)

    # Variant C
    H1 = H_BIC
    H2 = 700.0

    def S_c(y):
        y = np.asarray(y, dtype=float)
        out = np.zeros_like(y)
        mask_low  = (y >= 0) & (y <= H1)
        out[mask_low]  = 0.5 * (1 - np.cos(np.pi * y[mask_low] / H1))
        mask_high = (y >  H1) & (y <= H1 + H2)
        out[mask_high] = 0.5 * (1 + np.cos(np.pi * (y[mask_high] - H1) / H2))
        return out if out.shape else out.item()

    def SPrime_c(y):
        y = np.asarray(y, dtype=float)
        out = np.zeros_like(y)
        mask_low  = (y >= 0) & (y <= H1)
        out[mask_low]  =  (np.pi / (2*H1)) * np.sin(np.pi * y[mask_low] / H1)
        mask_high = (y >  H1) & (y <= H1 + H2)
        out[mask_high] = -(np.pi / (2*H2)) * np.sin(np.pi * (y[mask_high] - H1) / H2)
        return out if out.shape else out.item()

    return (S_a, SPrime_a), (S_b, SPrime_b), (S_c, SPrime_c)

def sanity_check(variants):
    print("--- Step C: Sanity Check S Definitions ---")
    (S_a, _), (S_b, _), (S_c, _) = variants
    test_pts = [0, H_BIC, 2*H_BIC, H_BIC + 700]
    
    print(f"{'y':<15} | {'Variant (a)':<15} | {'Variant (b)':<15} | {'Variant (c)':<15}")
    print("-" * 65)
    for y in test_pts:
        val_a = S_a(y) if np.isscalar(y) else S_a(np.array([y]))[0]
        val_b = S_b(y) if np.isscalar(y) else S_b(np.array([y]))[0]
        val_c = S_c(y) if np.isscalar(y) else S_c(np.array([y]))[0]
        if not np.isscalar(val_a): val_a = val_a.item()
        if not np.isscalar(val_b): val_b = val_b.item()
        if not np.isscalar(val_c): val_c = val_c.item()
        print(f"{y:<15.4f} | {val_a:<15.4f} | {val_b:<15.4f} | {val_c:<15.4f}")
    print()

def run_forward_evals(variants):
    print("--- Step D: Run Evaluations ---")
    (S_a, SPrime_a), (S_b, SPrime_b), (S_c, SPrime_c) = variants
    
    r_a = get_reflection_with_S(theta_BIC, H_BIC, S_a, SPrime_a)
    r_b = get_reflection_with_S(theta_BIC, H_BIC, S_b, SPrime_b)
    r_c = get_reflection_with_S(theta_BIC, H_BIC, S_c, SPrime_c)
    
    int_a = np.abs(r_a)**2
    int_b = np.abs(r_b)**2
    int_c = np.abs(r_c)**2
    
    print(f"{'Variant':<35} {'|r_0|^2':<15} {'log10(|r_0|^2)':<15}")
    print("-" * 65)
    print(f"{'(a) unclamped cosine (current)':<35} {int_a:<15.2e} {np.log10(int_a):<15.1f}")
    print(f"{'(b) clamped cosine':<35} {int_b:<15.2e} {np.log10(int_b):<15.1f}")
    print(f"{'(c) piecewise asymmetric':<35} {int_c:<15.2e} {np.log10(int_c):<15.1f}")
    print()
    return int_a, int_b, int_c

def interpret(int_a, int_b, int_c):
    print("--- Step E: Interpretation ---")
    threshold_small = 1e-10
    threshold_large = 1e-3
    
    is_a_small = int_a <= threshold_small
    is_b_large = int_b >= threshold_large
    is_c_large = int_c >= threshold_large
    
    if is_a_small and int_b <= threshold_small and int_c <= threshold_small:
        print("CASE 1 - All three values are <= 1e-10")
        print("BIC is robust to the modeling choice. Bug exists but doesn't bite at the optimum. Re-optimization optional.")
    elif is_a_small and (is_b_large or is_c_large):
        print("CASE 2 - (a) is small but (b) and/or (c) are >= 1e-3")
        print("BIC is an artifact of the unclamped S. Re-optimization required from scratch with the corrected S. Previous results not reliable.")
    elif (int_b <= threshold_small and int_c <= threshold_small) and not is_a_small:
        print("CASE 3 - (b) and (c) are both small but (a) is small only by coincidence")
        print("BIC is genuine but the original code's number is misleading. Cite (b) or (c) going forward.")
    else:
        print("CASE 4 - None are small or intermediate case.")
        print("Something else is broken. Stop and reread the audit.")

if __name__ == "__main__":
    if test_refactor():
        variants = define_variants()
        sanity_check(variants)
        int_a, int_b, int_c = run_forward_evals(variants)
        interpret(int_a, int_b, int_c)

