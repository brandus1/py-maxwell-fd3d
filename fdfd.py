#!/usr/bin/env python3
"""
3D FDFD Example with:
 - Lossy materials (sigma != 0)
 - PML boundaries
 - mu_r = 1 everywhere
 - A user-supplied 3D integer array indexing materials
 - A point dipole source

Dependencies: numpy, scipy, your pyfd3d code base (fd3d.py, pml.py, etc.)
"""

import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
# ---------------------------------------------------
# 1) Example "materials" dictionary or JSON
#    For each ID, we have: name, epsilon_r, sigma (S/m)
#    (You can load this from a real JSON file or define inline)
# ---------------------------------------------------
materials_dict = {
    0: {"name": "Air",       "epsilon_r": 1.0,    "sigma": 0.0},
    1: {"name": "Metal",     "epsilon_r": 1.0,    "sigma": 1e7},  # big sigma => very lossy
    2: {"name": "SiO2",      "epsilon_r": 1.46**2,"sigma": 0.0},
    3: {"name": "CustomMat", "epsilon_r": 5.0,    "sigma": 1e3},
}

# ---------------------------------------------------
# 2) A 3D integer array specifying each voxel's material ID
#    Example: shape (nx, ny, nz). We'll just make a toy domain
# ---------------------------------------------------
nx, ny, nz = 60, 40, 50
material_id_array = np.zeros((nx, ny, nz), dtype=int)

# Suppose we place a "metal" bar in the center
material_id_array[nx//4 : 3*nx//4, ny//2 : , nz//3 : 2*nz//3] = 1

# Another region of "CustomMat"
material_id_array[   : nx//5, 0:ny//2, nz//4 : nz//2] = 3

# everything else is 0 => "Air"

# ---------------------------------------------------
# 3) Simulation domain, resolution, PML etc.
# ---------------------------------------------------
L0    = 1e-6  # length scale for converting your geometry units (e.g. 1 µm)
wvlen = 1.0   # in L0 units => 1 micron if L0=1e-6
Npml  = [10,10,10]  # thickness of PML in each dimension

# Physical domain extents in L0
xrange = (0, 6.0)    # 6 microns in x
yrange = (0, 4.0)    # 4 microns in y
zrange = (0, 5.0)    # 5 microns in z

dx = (xrange[1]-xrange[0]) / nx
dy = (yrange[1]-yrange[0]) / ny
dz = (zrange[1]-zrange[0]) / nz

# ---------------------------------------------------
# 4) Build the permittivity arrays (eps_xx, eps_yy, eps_zz)
#    including conduction sigma -> complex permittivity
# ---------------------------------------------------
eps0_SI = 8.854e-12  # in SI
c0_SI   = 3e8

def build_permittivity_3d(material_id_array, materials_dict, wvlen, L0):
    """
    Returns eps_xx, eps_yy, eps_zz as 3D arrays of shape = material_id_array.shape
    Each voxel's complex epsilon is: eps_0 * ( eps_r - j sigma/(omega eps_0) ).
    """
    nx, ny, nz = material_id_array.shape
    # Angular frequency in scaled units:
    # In real SI: omega_SI = 2π c0 / (wvlen_SI)
    # We incorporate the chosen L0 so that wvlen = wvlen_SI / L0
    # => wvlen_SI = wvlen * L0
    # => omega = 2π c0 / (wvlen * L0)
    omega = 2.0*np.pi*c0_SI / (wvlen * L0)

    eps_xx = np.zeros((nx, ny, nz), dtype=complex)
    eps_yy = np.zeros((nx, ny, nz), dtype=complex)
    eps_zz = np.zeros((nx, ny, nz), dtype=complex)

    for idx in np.unique(material_id_array):
        mat_info   = materials_dict[idx]
        eps_r_here = mat_info["epsilon_r"]
        sigma_here = mat_info["sigma"]  # S/m
        # Complex relative permittivity:
        #   eps_r_eff = ( eps_r - j*sigma / (omega eps0_SI) )
        # Then multiply by eps0_SI * L0 if using that scale, or just store
        # the dimensionless relative perm if everything else is consistent.
        # Typically, in FDFD code, we do e = e0*( e_r - j*sigma/(omega e0) ).
        # That is e = e_r * e0 - j*sigma/(omega). 
        # But if your fd3d code expects the dimensionless quantity (like eps_r), we do:
        eps_r_complex = eps_r_here - 1j*(sigma_here / (omega * eps0_SI))

        # Fill all voxels that have integer `idx` with that complex value:
        mask = (material_id_array == idx)
        eps_xx[mask] = eps_r_complex
        eps_yy[mask] = eps_r_complex
        eps_zz[mask] = eps_r_complex

    return eps_xx, eps_yy, eps_zz

eps_xx, eps_yy, eps_zz = build_permittivity_3d(material_id_array, materials_dict, wvlen, L0)

# Bundle into dictionary as your fd3d code expects:
eps_r_tensor = {
    "eps_xx": eps_xx,
    "eps_yy": eps_yy,
    "eps_zz": eps_zz,
}

# ---------------------------------------------------
# 5) Define a simple source Jx, Jy, Jz
#    e.g. a point dipole in the center
# ---------------------------------------------------
nxC, nyC, nzC = nx//2, ny//2, nz//2
Jx = np.zeros((nx, ny, nz), dtype=complex)
Jy = np.zeros((nx, ny, nz), dtype=complex)
Jz = np.zeros((nx, ny, nz), dtype=complex)

# Put a single dipole along z in the center:
Jz[nxC, nyC, nzC] = 1.0

Jdict = {
    "Jx": Jx,
    "Jy": Jy,
    "Jz": Jz
}

# ---------------------------------------------------
# 6) Call your FDFD assembly routine to get (A, b)
#    We assume your code is something like:
#      A,b,_ = curlcurlE(L0, wvlen, xrange, yrange, zrange,
#                        eps_r_tensor, Jdict, Npml, s=-1)
# ---------------------------------------------------
from pyfd3d.fd3d import curlcurlE

A, b, Ch = curlcurlE(
    L0        = L0,
    wvlen     = wvlen,
    xrange    = xrange,
    yrange    = yrange,
    zrange    = zrange,
    eps_r_tensor_dict = eps_r_tensor,
    JCurrentVector    = Jdict,
    Npml      = Npml,
    s         = -1,
    nonuniform= None  # or pass a dict if you want nonuniform mesh
)

print("Matrix A shape =", A.shape, "  b shape =", b.shape)

# ---------------------------------------------------
# 7) Solve the linear system A E = b
# ---------------------------------------------------
residual_history = []

# Create a new figure for residual plot
plt.figure(figsize=(10, 6))
ax = plt.gca()
(line,) = ax.plot([], [], 'b-')

ax.set_xlabel("Iteration")
ax.set_ylabel("Residual Norm")
ax.set_yscale("log")  # log scale for residual
ax.set_title("BiCGSTAB Convergence (log scale)")

def iteration_callback(xk):
    """
    Called at each BiCGSTAB iteration, with xk = current solution guess.
    We'll compute the residual norm, store it, and update the live plot.
    """
    r = b - A @ xk
    r_norm = np.linalg.norm(r)
    residual_history.append(r_norm)
    print(f"Iteration {len(residual_history)}: residual = {r_norm:.3e}")
    # Update the line's data:
    xdata = np.arange(1, len(residual_history) + 1)
    line.set_xdata(xdata)
    line.set_ydata(residual_history)

    ax.relim()          # Recompute plot limits
    ax.autoscale_view() # Autoscale axes
    plt.draw()
    plt.pause(0.01)     # Small pause so GUI updates

# Optional initial guess
x0 = np.zeros_like(b, dtype=complex)

# Solve using BiCGSTAB with rtol (and optionally atol).
E_solution, info = spla.bicgstab(
    A, b,
    x0=x0,
    rtol=1e-8,
    maxiter=2000,
    callback=iteration_callback
)

if info == 0:
    print("BiCGSTAB converged!")
else:
    print(f"BiCGSTAB ended with info={info}")

# Close the residual plot figure
plt.close()

# E_solution is a length-3*N vector => the Ex,Ey,Ez fields in a single array
# You can reshape them back to (nx,ny,nz):
Ex_sol = E_solution[0 : nx*ny*nz ].reshape((nx,ny,nz), order='F')
Ey_sol = E_solution[nx*ny*nz : 2*nx*ny*nz ].reshape((nx,ny,nz), order='F')
Ez_sol = E_solution[2*nx*ny*nz :               ].reshape((nx,ny,nz), order='F')

# ---------------------------------------------------
# 8) Post-process or visualize the fields
# ---------------------------------------------------
# Create a new figure for field visualization
plt.figure(figsize=(10, 6))
midZ = nz//2
plt.imshow(np.abs(Ez_sol[:,:,midZ].T), origin='lower', cmap='inferno')
plt.colorbar(label='|Ez| at z-slice')
plt.title('Central z-slice of Ez')
plt.show()