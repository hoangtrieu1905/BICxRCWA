# C_RCWA Method in Python - Version 0.1 (2024)
'''
Author: Matlab Code - Dr. Benjamin Civiletti /
        Python Code - Hoang Trieu
Last edited: 2/19/2026

Note: If running from VS code, choose the Python 3.11.5 - python-binary-search interpreter to run the code.
'''
from math import *
from numpy import pi, sin, cos
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg
import pandas as pd
import scipy

plt.close('all')  #close all plots before starting

def gridC(numLayers, gratingPeriod, layerThick, xDiscreteSize, yDiscreteSize): #checked


    xSpacingNum = gratingPeriod/xDiscreteSize
    xGrid = np.linspace(-gratingPeriod/2 + xDiscreteSize/2, gratingPeriod/2 - xDiscreteSize/2, num = int(xSpacingNum))
    ySlice = layerThick/yDiscreteSize
    yGrid = []
    yBound = []
    lThick = []
    lThick.extend(layerThick)
    lThick.insert(0,0)

    for i in range(numLayers):
        ySliceNum = int(ySlice[i])
        sBounds = [np.sum(lThick[:i+1]), np.sum(lThick[:i+2])]
        yGrid.extend(np.linspace(sBounds[0] + yDiscreteSize[i]/2, sBounds[1] - yDiscreteSize[i]/2, ySliceNum))
        yBound.append(sBounds[0])

    return xGrid, yGrid, np.array(yBound)

def create_Y_matrix(matrix1, matrix2, matrix3, matrix4): #checked

    Y_top = np.concatenate((matrix1, matrix2), axis=1)
    Y_bottom = np.concatenate((matrix3, matrix4), axis=1)
    Y = np.concatenate((Y_top, Y_bottom), axis=0)
    return Y

def simpson(f, left, right, numSub): #unchecked

    h = (right-left)/numSub
    if callable(f):
        x = np.linspace(left, right, numSub + 1)
        y = f(x)
    elif isinstance(f, np.ndarray):
        y = f
    else:
        raise ValueError("Invalid input for f at the simpson function.")
    
    #Since Matlab and Numpy handles complexes differently, the error occurs
    #However, it's small enough not to be worried about
    array1 = y[1:numSub:2]
    array2 = y[2:(numSub-1):2]
    sumArray1 = np.sum(array1)
    sumArray2 = np.sum(array2)
    approxInt = (h/3) * (y[0] + y[numSub] + 4*sumArray1 + 2*sumArray2)
    return approxInt

def bisection(f, left, right, TOL):

    n = 0
    if f(left)*f(right) >= 0:
        raise ValueError("There is no root of f(x) in [" + str(left) + "," + str(right) + "]")

    while ((right - left)/2 > TOL):
        mid = (left + right)/2
        if f(mid) == 0:
            break

        if f(left)*f(mid) < 0: #root in [a,c]
            right = mid
        else:
            left = mid
        n += 1
    root = (right + left)/2
    return (root, n)

def ToeplitzM(m, f, L, sampleP): #Checked

    n = 2*m - 1
    fftvec = np.zeros(n, dtype= complex)
    ind = (n-1)/2
    j = np.arange(-ind, ind + 1)
    if callable(f):
        for i in range(n):
            k = lambda x: f(x)*np.exp(-2*pi*1j*j[i]*x/L)
            fftvec[i] = (1/L)*simpson(k, -L/2, L/2, sampleP)
    elif isinstance(f, np.ndarray):
        for i in range(n):
            ind = (n-1)/2
            j = np.arange(-ind, ind + 1)
            x = np.linspace(-L/2, L/2, sampleP + 1)
            basis = np.exp(-2*pi*1j*j[i]*x/L)
            if len(basis) != len(f):
                print(len(basis))
                print(len(f))
                raise ValueError("Lengths of basis and f do not match")
            fSample = f*basis
            fftvec[i] = (1/L)*simpson(fSample, -L/2, L/2, sampleP)
    else:
        raise ValueError("Invalid input for f at ToeplitzM function")

    fftvec0pos = fftvec[m-1:]
    fftvec0neg = np.flip(fftvec[:m])
    T = linalg.toeplitz(fftvec0pos, fftvec0neg)
    return T

def invMap(x, y, g, S, yGrid, H, gratingPeriod):
    xHat = x
    C = g(x, gratingPeriod)
    F = lambda x: C*S(x) - x + y
    yHat, n = bisection(F, 0, 2*H, 10**-3)
    yGrid = np.array(yGrid)
    index = np.argmin(np.abs(yHat - yGrid))
    return xHat, yHat, index

'''Physical Constants
#Only edit the wavelength lambda0 and incidence-angle theta'''

epsilon0 = 8.854*10**(-12)
mu0 = 4*pi*10**(-7)
eta0 = np.sqrt(mu0/epsilon0)
lambda0 = 600
k0 = 2*pi/lambda0
omega = k0/eta0
theta = 0


''' # Parameter in Python vs Matlab
% fourierMode(M)        - Fourier mode parameter 
% numLayers  (N)        - Total number of layers 
% gratingPeriod (L)     - Period of grating 
% yDiscreteParam (h)    - Discretization parameter in y 
% xDiscreteSize (xs)    - Discretization size in x 
% yDiscreteSize (ys)    - Discretization size in x 
% layerThick (lthick)   - (Nx1) array of layer thickness 
% ep                    - (Nx1) cell array of relative epsilon 
% g                     - Function for the grating 
% gPrime                - Function for the derivative of grating 
% sampleP               - Number of subintervals for simpson's rule (even) 
% p                     - Polarization State (either 'p' or 's')
% method                - Chooses the solver (either 'C', 'R', or 'CR') 
'''

fourierMode = 5
m = 2*fourierMode + 1 
numLayers = 2
gratingPeriod = 500
yDiscreteParam = 1
yDiscreteSize = np.ones(numLayers)
xDiscreteSize = 1
layerThick = [500, 900] #List
ep = np.empty(numLayers, dtype=object)
ep[0] = 1 + 1j*10**(-9)
ep[1] = -15 + 1j*4
#ep[1] = 1 + 1j*10**(-9)
sampleP = 500
p = 'p'
method = 'C'

g = lambda x, gratingPeriod: 100*np.cos(2*pi*x/gratingPeriod)
gPrime = lambda x, gratingPeriod: -200*pi/gratingPeriod*np.sin(2*pi*x/gratingPeriod)


'''Define the coordinate transform'''
H = 900
S = lambda y: 1/2*(1+np.cos(pi/(H) * (y-H)))
SPrime = lambda y: -(pi/(2*H))*(np.sin(pi/(H) * (y-H)))
detDG = lambda x, y: SPrime(y)*g(x, gratingPeriod)+1 #DG is the Jacobian

#Define elements in the Jacobian Matrix
a11 = lambda x, y: np.abs(detDG(x, y))
a21 = lambda x, y: -np.sign(detDG(x,y))*S(y)*gPrime(x, gratingPeriod)
a12 = lambda x, y: a21(x,y)
a22 = lambda x, y: ((S(y)*gPrime(x, gratingPeriod))**2 + 1)/ np.abs(detDG(x,y))

'''Calculate the grid points'''
'''
Grid points are the points where we calculate the fields
Calculate the spacing in the x and y -direction '''
(xGrid, yGrid, yBound) = gridC(numLayers, gratingPeriod, layerThick, xDiscreteSize, yDiscreteSize)
yGridN = np.size(yGrid)
xGridN = np.size(xGrid)

#print(xGrid, yGrid, yBound)

'''Plot the grating profile'''

fig_grating, ax_grating = plt.subplots()

plt.title("Grating Profile")
plt.xlabel("x")
plt.ylabel("y")
g_of_xGrid = []
for i in range(len(xGrid)):
    g_of_xGrid.append(g(xGrid[i], gratingPeriod))
ax_grating.axis('equal')
ax_grating.plot(xGrid, g_of_xGrid)
plt.show()

gR = lambda x, gratingPeriod: -g(x,gratingPeriod) + H

'''Define the boundary Condition'''

k_vector = np.arange(-fourierMode, fourierMode+1, dtype=np.complex128)
k = k0*sin(theta) + 2*pi*k_vector/gratingPeriod
K = np.diag(k)
beta = np.sqrt(k0**2-k**2)

O = np.zeros((m,m))
I = np.eye(m,m)

Ye_p = create_Y_matrix(O, -np.diag(beta/k0), I, O)
Ye_m = create_Y_matrix(O, np.diag(beta/k0), I, O)
Yh_p = create_Y_matrix(-np.diag(beta/k0), O, O, -I)
Yh_m = create_Y_matrix(np.diag(beta/k0), O, O, -I)
#df = pd.DataFrame(Ye_p)
#print(df)

'''Set the polarization state'''
A = np.zeros((4*fourierMode + 2, 1))
if p == 's': #(3.53)
    A[fourierMode] = 1 #index is 1 less than the code in matlab
elif p == 'p':
    A[3*fourierMode + 1] = 1

'''Reallocate some cells for the RCWA Algorithm'''
#Cell = ndarray with objects. Types can be different
Zs = np.empty(yGridN + 1, dtype=object)
Ts = np.empty(yGridN + 1, dtype=object)
Wus = np.empty(yGridN, dtype=object)
Dus = np.empty(yGridN + 1, dtype=object)
f = np.empty(yGridN + 1, dtype=object)

Zs[yGridN] = np.concatenate((Ye_p, Yh_p), axis = 0)

#print(pd.DataFrame(yGrid))
# print(np.shape(yGrid))
'''Start the RCWA Algorithm'''
for i in range(yGridN - 1, -1, -1):
    #Find the current layer
    print("i = "+ str(i))
    # if (yGrid[i] - yBound <= 0).any():
    #     check = (yGrid[i] - yBound > 0)
    #     print(check)
    #     print("Break at i = " + str(i))
    #     break
    # else:
    check = (yGrid[i] - yBound > 0)
    print(check)
    layer = np.where(check)[0][-1] #[0]: returns the indices where the condition is True [-1]: last indices
    print("Layer: "  + str(layer))

    #print(layer)

    if callable(ep[layer]):
        epsilon = ep[layer]
        T = ToeplitzM(m, epsilon, gratingPeriod, sampleP)
    else:
        epsilon = np.ones((1, sampleP + 1))
        epsilon = epsilon*ep[layer]
        T = np.eye(m)*ep[layer]

    if method == 'C': #C-RCWA
        #Define epsilon and mu
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



        P11 = K@(linalg.solve(Tep22, Tep21))
        P14 = k0*Tmu33 - (1/k0)*K@(linalg.solve(Tep22, K))
        P22 = Tmu12@(linalg.solve(Tmu22, K))
        P23 = -k0*(Tmu11 - Tmu12@(linalg.solve(Tmu22, Tmu21)))
        P32 = -k0*Tep33 + (1/k0) * K @ (linalg.solve(Tmu22, K))
        P33 = K@(linalg.solve(Tmu22, Tmu21))
        P41 = -k0*(Tep12@(linalg.solve(Tep22, Tep21)) - Tep11)
        P44 = Tep12@linalg.solve(Tep22, K)

        # Construct the P matrix using square brackets and commas
        P = np.block([[P11, O, O, P14],
              [O, P22, P23, O],
              [O, P32, P33, O],
              [P41, O, O, P44]])



    elif method == 'R':
        ind = np.where(gR(xGrid,gratingPeriod) < yGrid[i])[0]
        
        if np.size(ind) == np.size(xGrid, 0): #below grating
            ep11 = lambda x: ep[layer]
            ep22 = lambda x: ep11(x)
            ep33 = lambda x: ep11(x)
        elif (np.size(ind) != np.size(xGrid, 0)) and np.size(ind) !=0:
            xL = xGrid[ind[0]]
            xR = xGrid[ind[np.size(ind) -1]]
            ep11 = lambda x: (x < xL) * ep[0] + (x > xR) * ep[0] + \
                ((x >= xL) & (x <= xR)) * ep[1]
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
        Tmu21= O
        Tmu12 = O
        Tmu22= I
        Tmu33 = I
        
        P14 = k0*Tmu33 - (1/k0)*K@(linalg.solve(Tep22, K))
        P32 = -k0*Tep33 + (1/k0) * K @ (linalg.solve(Tmu22, K))
        P41 = -k0*(Tep12@(linalg.solve(Tep22, Tep21)) - Tep11)
        P23 = -k0*(Tmu11 - Tmu12@(linalg.solve(Tmu22, Tmu21)))
        
        P = np.block([[O, O, O, P14], [O, O, P23, O], [O, P32, O, O], \
                      [P41, O, O, O]])
    
    if i==0 or i== yGridN - 1:
        Delta = 1
    else:
        Delta = yGrid[i] - yGrid[i-1]
    
    # Compute eigenvalues and eigenvectors
    D, G = np.linalg.eig(P)
    
    # Convert eigenvalues to diagonal matrix to match MATLAB's behavior
    D = np.diag(D)

    Ddiag = np.diag(D)
    ind = np.argsort(np.imag(Ddiag))[::-1]
    Ddiag= Ddiag[ind]
    D = np.diag(Ddiag)
    G = G[:, ind]
    W = linalg.solve(G, Zs[i+1])
    Wu = W[0:4*fourierMode + 2, :]
    Wus[i] = Wu
    Wl = W[4*fourierMode + 2:, :]
    Du = D[0:4*fourierMode + 2, 0:4*fourierMode + 2]
    Dus[i] = Du
    Dl = D[4*fourierMode + 2:, 4*fourierMode+2:]
    expmat1 = scipy.linalg.expm(-1j * Delta * Dl)
    expmat2 = scipy.linalg.expm(1j*Delta*Du)
    
    # Construct the block matrix with the correct dimensions
    block_matrix = np.block([[np.eye(4*fourierMode + 2)], \
                             [expmat1 @ Wl @ np.linalg.solve(Wu, expmat2)]])
    
    # Perform matrix multiplication with the corrected block matrix
    Zs[i] = G @ block_matrix


#Solve for T0
Z0u = Zs[0][0:4*fourierMode + 2, :]
Z0l = Zs[0][4*fourierMode+2:,:]

top = np.block([[Z0u, -Ye_m], [Z0l, -Yh_m]])
bottom = np.block([[Ye_p], [Yh_p]])
X = scipy.linalg.solve(top, bottom)
T0R = X@A

Ts[0] = T0R[0:4*fourierMode + 2, :]

for i in range(yGridN):
    if i == 0 or i == yGridN - 1:
        Delta =1 
    else: 
        Delta = yGrid[i] - yGrid[i-1] #Page 29
    Ts[i+1] = (linalg.solve(Wus[i],scipy.linalg.expm(1j * Delta * Dus[i])))@Ts[i] #3.60
    
#Solve for f 
for i in range(yGridN + 1):
    f[i] = Zs[i]@Ts[i] #This is relation (3.56)
    
#Reconstruct the solution and plot 
basis = lambda x, n: np.exp(1j*n*x)

Ex = np.zeros((yGridN, xGridN), dtype=np.complex128)
Ey = np.zeros((yGridN, xGridN), dtype=np.complex128)
Hx = np.zeros((yGridN, xGridN), dtype=np.complex128)
Hy = np.zeros((yGridN, xGridN), dtype=np.complex128)

if method == 'C':
    Exhat = np.zeros((yGridN, xGridN), dtype=np.complex128)
    Eyhat = np.zeros((yGridN, xGridN), dtype=np.complex128)
    Hxhat = np.zeros((yGridN, xGridN), dtype=np.complex128)
    Hyhat = np.zeros((yGridN, xGridN), dtype=np.complex128)
    
    #Cast complex to preserve complex when doing matrix multiplication 
    xGrid_complex = xGrid.astype(np.complex128)
    k_complex = np.complex128(k)
    
    for i in range(yGridN):
        for j in range(xGridN):
            Exhat[i, j] = basis(xGrid_complex[j], k_complex)@f[i+1][0:2*fourierMode + 1, 0]
            Eyhat[i, j] = basis(xGrid_complex[j], k_complex)@f[i+1][2*fourierMode + 1:4*fourierMode + 2, 0]
            Hxhat[i, j] = basis(xGrid_complex[j], k_complex)@f[i+1][4*fourierMode + 2:6*fourierMode + 3, 0]
            Hyhat[i, j] = basis(xGrid_complex[j], k_complex)@f[i+1][6*fourierMode + 3:, 0]

    for i in range(yGridN):
        for j in range(xGridN):
            xHat, yHat, index= invMap(xGrid[j], yGrid[i], g, S, yGrid, H, gratingPeriod)
            Ex[i, j] = Exhat[index, j]
            Ey[i, j] = Eyhat[index, j]
            Hx[i, j] = Hxhat[index, j]
            Hy[i, j] = Hyhat[index, j]
elif method == 'R':
    for i in range(yGridN):
        for j in range(xGridN):
            Ex[i, j] = basis(xGrid[j], k)@f[i+1][0:2*fourierMode + 1, 0] #this basis is exp(i\alpha_n x_1) in page 33
            Ey[i, j] = basis(xGrid[j], k)@f[i+1][2*fourierMode + 1:4*fourierMode + 2, 0]
            Hx[i, j] = basis(xGrid[j], k)@f[i+1][4*fourierMode + 2:6*fourierMode + 3, 0]
            Hy[i, j] = basis(xGrid[j], k)@f[i+1][6*fourierMode + 3:, 0]
            

if p == 's' and method == 'C':
    # Create figure and axes
    fig1, axs1 = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figsize as needed
    
    # Plot for Eyhat
    axs1[0].pcolormesh(xGrid, yGrid, np.abs(Eyhat))
    axs1[0].axis('equal')
    axs1[0].set_xlim([-gratingPeriod/2, gratingPeriod/2])
    axs1[0].set_ylim([0, 2*H])
    axs1[0].set_xlabel(r'$\mathbf{\hat{x}}$', fontsize=20)
    axs1[0].set_ylabel(r'$\mathbf{\hat{y}}$', fontsize=20)
    axs1[0].set_title(r'$\left|\hat{E}_{y} \right|$', fontsize=20, fontweight='bold')
    axs1[0].tick_params(axis='both', which='major', labelsize=16)
    axs1[0].shading='interp'
    colormap = axs1[0].pcolormesh(xGrid, yGrid, np.abs(Eyhat), cmap='jet')
    plt.colorbar(colormap, ax=axs1[0])
    axs1[0].plot(xGrid, H*np.ones_like(xGrid), color='white', linewidth=2)
    
    # Plot for Ey
    axs1[1].pcolormesh(xGrid, yGrid, np.abs(Ey))
    axs1[1].axis('equal')
    axs1[1].set_xlim([-gratingPeriod/2, gratingPeriod/2])
    axs1[1].set_ylim([0, 2*H])
    axs1[1].set_xlabel(r'$\mathbf{x}$', fontsize=20)
    axs1[1].set_ylabel(r'$\mathbf{x}$', fontsize=20)
    axs1[1].set_title(r'$\left|E_{y} \right|$', fontsize=20)
    axs1[1].tick_params(axis='both', which='major', labelsize=16)
    axs1[1].shading='interp'
    colormap = axs1[1].pcolormesh(xGrid, yGrid, np.abs(Ey), cmap='jet')
    plt.colorbar(colormap, ax=axs1[1])
    axs1[1].plot(xGrid, -g(xGrid, gratingPeriod) + H, color='white', linewidth=2)
    
    # Adjust layout
    plt.tight_layout()
    
    # Set figure size and save or display the plot
    fig1.set_size_inches(12, 6)
    
    # Create figure and axes
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figsize as needed
    
    # Plot for Hxhat
    axs2[0].pcolormesh(xGrid, yGrid, np.abs(Hxhat))
    axs2[0].axis('equal')
    axs2[0].set_xlim([-gratingPeriod/2, gratingPeriod/2])
    axs2[0].set_ylim([0, 2*H])
    axs2[0].set_xlabel(r'$\mathbf{\hat{x}}$', fontsize=20)
    axs2[0].set_ylabel(r'$\mathbf{\hat{y}}$', fontsize=20)
    axs2[0].set_title(r'$\left|\hat{H}_{x} \right|$', fontsize=20, fontweight='bold')
    axs2[0].tick_params(axis='both', which='major', labelsize=16)
    axs2[0].shading='interp'
    colormap = axs2[0].pcolormesh(xGrid, yGrid, np.abs(Hxhat), cmap='jet')
    plt.colorbar(colormap, ax=axs2[0])
    axs2[0].plot(xGrid, H*np.ones_like(xGrid), color='white', linewidth=2)
    
    # Plot for Hx
    axs2[1].pcolormesh(xGrid, yGrid, np.abs(Hx))
    axs2[1].axis('equal')
    axs2[1].set_xlim([-gratingPeriod/2, gratingPeriod/2])
    axs2[1].set_ylim([0, 2*H])
    axs2[1].set_xlabel(r'$\mathbf{x}$', fontsize=20)
    axs2[1].set_ylabel(r'$\mathbf{x}$', fontsize=20)
    axs2[1].set_title(r'$\left|H_{x} \right|$', fontsize=20)
    axs2[1].tick_params(axis='both', which='major', labelsize=16)
    axs2[1].shading='interp'
    colormap = axs2[1].pcolormesh(xGrid, yGrid, np.abs(Hx), cmap='jet')
    plt.colorbar(colormap, ax=axs2[1])
    axs2[1].plot(xGrid, -g(xGrid, gratingPeriod) + H, color='white', linewidth=2)
    
    # Adjust layout
    plt.tight_layout()
    
    # Set figure size and save or display the plot
    fig2.set_size_inches(12, 6)
    plt.show()
elif p == 'p' and method == 'C':
    # Create figure and axes
    fig1, axs1 = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figsize as needed
    
    # Plot for Hyhat
    axs1[0].pcolormesh(xGrid, yGrid, np.abs(Hyhat))
    axs1[0].axis('equal')
    axs1[0].set_xlim([-gratingPeriod/2, gratingPeriod/2])
    axs1[0].set_ylim([0, 2*H])
    axs1[0].set_xlabel(r'$\mathbf{\hat{x}}$', fontsize=20)
    axs1[0].set_ylabel(r'$\mathbf{\hat{y}}$', fontsize=20)
    axs1[0].set_title(r'$\left|\hat{H}_{y} \right|$', fontsize=20, fontweight='bold')
    axs1[0].tick_params(axis='both', which='major', labelsize=16)
    axs1[0].shading='interp'
    colormap = axs1[0].pcolormesh(xGrid, yGrid, np.abs(Hyhat), cmap='jet')
    plt.colorbar(colormap, ax=axs1[0])
    axs1[0].plot(xGrid, H*np.ones_like(xGrid), color='white', linewidth=2)
    
    # Plot for Hy
    axs1[1].pcolormesh(xGrid, yGrid, np.abs(Hy))
    axs1[1].axis('equal')
    axs1[1].set_xlim([-gratingPeriod/2, gratingPeriod/2])
    axs1[1].set_ylim([0, 2*H])
    axs1[1].set_xlabel(r'$\mathbf{x}$', fontsize=20)
    axs1[1].set_ylabel(r'$\mathbf{x}$', fontsize=20)
    axs1[1].set_title(r'$\left|H_{y} \right|$', fontsize=20)
    axs1[1].tick_params(axis='both', which='major', labelsize=16)
    axs1[1].shading='interp'
    colormap = axs1[1].pcolormesh(xGrid, yGrid, np.abs(Hy), cmap='jet')
    plt.colorbar(colormap, ax=axs1[1])
    axs1[1].plot(xGrid, -g(xGrid, gratingPeriod) + H, color='white', linewidth=2)
    
    # Adjust layout
    plt.tight_layout()
    
    # Set figure size and save or display the plot
    fig1.set_size_inches(12, 6)
    
    # Create figure and axes
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figsize as needed
    
    #Plot for Exhat
    axs2[0].pcolormesh(xGrid, yGrid, np.abs(Exhat))
    axs2[0].axis('equal')
    axs2[0].set_xlim([-gratingPeriod/2, gratingPeriod/2])
    axs2[0].set_ylim([0, 2*H])
    axs2[0].set_xlabel(r'$\mathbf{\hat{x}}$', fontsize=20)
    axs2[0].set_ylabel(r'$\mathbf{\hat{y}}$', fontsize=20)
    axs2[0].set_title(r'$\left|\hat{E}_{x} \right|$', fontsize=20, fontweight='bold')
    axs2[0].tick_params(axis='both', which='major', labelsize=16)
    axs2[0].shading='interp'
    colormap = axs2[0].pcolormesh(xGrid, yGrid, np.abs(Exhat), cmap='jet')
    plt.colorbar(colormap, ax=axs2[0])
    axs2[0].plot(xGrid, H*np.ones_like(xGrid), color='white', linewidth=2)
    
    # Plot for Ex
    axs2[1].pcolormesh(xGrid, yGrid, np.abs(Ex))
    axs2[1].axis('equal')
    axs2[1].set_xlim([-gratingPeriod/2, gratingPeriod/2])
    axs2[1].set_ylim([0, 2*H])
    axs2[1].set_xlabel(r'$\mathbf{x}$', fontsize=20)
    axs2[1].set_ylabel(r'$\mathbf{x}$', fontsize=20)
    axs2[1].set_title(r'$\left|E_{x} \right|$', fontsize=20)
    axs2[1].tick_params(axis='both', which='major', labelsize=16)
    axs2[1].shading='interp'
    colormap = axs2[1].pcolormesh(xGrid, yGrid, np.abs(Ex), cmap='jet')
    plt.colorbar(colormap, ax=axs2[1])
    axs2[1].plot(xGrid, -g(xGrid, gratingPeriod) + H, color='white', linewidth=2)
    
    # Adjust layout
    plt.tight_layout()
    
    # Set figure size and save or display the plot
    fig2.set_size_inches(12, 6)
    plt.show()

elif p == 'p' and method == 'R':
    # Create figure and axes
    fig1, axs1 = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figsize as needed
    
    # Plot for Hyhat
    axs1[0].pcolormesh(xGrid, yGrid, np.abs(Hy))
    axs1[0].axis('equal')
    axs1[0].set_xlim([-gratingPeriod/2, gratingPeriod/2])
    axs1[0].set_ylim([0, 2*H])
    axs1[0].set_xlabel(r'$\mathbf{x}$', fontsize=20)
    axs1[0].set_ylabel(r'$\mathbf{y}$', fontsize=20)
    axs1[0].set_title(r'$\left|H_{y} \right|$', fontsize=20, fontweight='bold')
    axs1[0].tick_params(axis='both', which='major', labelsize=16)
    axs1[0].shading='interp'
    colormap = axs1[0].pcolormesh(xGrid, yGrid, np.abs(Hy), cmap='jet')
    plt.colorbar(colormap, ax=axs1[0])
    axs1[0].plot(xGrid, -g(xGrid, gratingPeriod) + H, color='white', linewidth=2)
    
    # Plot for Hy
    axs1[1].pcolormesh(xGrid, yGrid, np.abs(Ex))
    axs1[1].axis('equal')
    axs1[1].set_xlim([-gratingPeriod/2, gratingPeriod/2])
    axs1[1].set_ylim([0, 2*H])
    axs1[1].set_xlabel(r'$\mathbf{x}$', fontsize=20)
    axs1[1].set_ylabel(r'$\mathbf{x}$', fontsize=20)
    axs1[1].set_title(r'$\left|E_{x} \right|$', fontsize=20)
    axs1[1].tick_params(axis='both', which='major', labelsize=16)
    axs1[1].shading='interp'
    colormap = axs1[1].pcolormesh(xGrid, yGrid, np.abs(Ex), cmap='jet')
    plt.colorbar(colormap, ax=axs1[1])
    axs1[1].plot(xGrid, -g(xGrid, gratingPeriod) + H, color='white', linewidth=2)
    
    # Adjust layout
    plt.tight_layout()
    
    
    # Set figure size and save or display the plot
    fig1.set_size_inches(12, 6)
    plt.show()
else:
    # Create figure and axes
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figsize as needed
    
    # Plot for Exhat
    axs2[0].pcolormesh(xGrid, yGrid, np.abs(Ey))
    axs2[0].axis('equal')
    axs2[0].set_xlim([-gratingPeriod/2, gratingPeriod/2])
    axs2[0].set_ylim([0, 2*H])
    axs2[0].set_xlabel(r'$\mathbf{x}$', fontsize=20)
    axs2[0].set_ylabel(r'$\mathbf{y}$', fontsize=20)
    axs2[0].set_title(r'$\left|E_{y} \right|$', fontsize=20, fontweight='bold')
    axs2[0].tick_params(axis='both', which='major', labelsize=16)
    axs2[0].shading='interp'
    colormap = axs2[0].pcolormesh(xGrid, yGrid, np.abs(Ey), cmap='jet')
    plt.colorbar(colormap, ax=axs2[0])
    axs1[0].plot(xGrid, -g(xGrid, gratingPeriod) + H, color='white', linewidth=2)
    
    # Plot for Ex
    axs2[1].pcolormesh(xGrid, yGrid, np.abs(Hx))
    axs2[1].axis('equal')
    axs2[1].set_xlim([-gratingPeriod/2, gratingPeriod/2])
    axs2[1].set_ylim([0, 2*H])
    axs2[1].set_xlabel(r'$\mathbf{x}$', fontsize=20)
    axs2[1].set_ylabel(r'$\mathbf{x}$', fontsize=20)
    axs2[1].set_title(r'$\left|H_{x} \right|$', fontsize=20)
    axs2[1].tick_params(axis='both', which='major', labelsize=16)
    axs2[1].shading='interp'
    colormap = axs2[1].pcolormesh(xGrid, yGrid, np.abs(Hx), cmap='jet')
    plt.colorbar(colormap, ax=axs2[1])
    axs2[1].plot(xGrid, -g(xGrid, gratingPeriod) + H, color='white', linewidth=2)
    
    # Adjust layout
    plt.tight_layout()
    
    # Set figure size and save or display the plot
    fig2.set_size_inches(12, 6)
    plt.show()