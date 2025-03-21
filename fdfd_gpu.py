#!/usr/bin/env python3
"""
3D FDFD Example with GPU acceleration:
 - Uses CuPy for GPU-accelerated computation
 - Lossy materials (sigma != 0)
 - PML boundaries
 - mu_r = 1 everywhere
 - A user-supplied 3D integer array indexing materials
 - A point dipole source

Dependencies: numpy, scipy, cupy, your pyfd3d code base with GPU support
"""

import numpy as np
import time
import matplotlib.pyplot as plt

try:
    import cupy as cp
    from cupyx.scipy.sparse.linalg import bicgstab as cp_bicgstab
    GPU_AVAILABLE = True
except ImportError:
    print("CuPy not available. To use GPU acceleration, install CuPy with: pip install cupy-cuda11x")
    GPU_AVAILABLE = False

# ---------------------------------------------------
# 1) Example "materials" dictionary or JSON
#    For each ID, we have: name, epsilon_r, sigma (S/m)
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
# 6) Run both CPU and GPU versions for comparison
# ---------------------------------------------------
from pyfd3d.fd3d import curlcurlE, curlcurlE_gpu

def run_cpu_version():
    print("\nRunning CPU version...")
    start_time = time.time()
    
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
        nonuniform= None
    )
    
    assembly_time = time.time() - start_time
    print(f"CPU Assembly time: {assembly_time:.3f} seconds")
    print(f"Matrix A shape = {A.shape}, b shape = {b.shape}")
    
    # Solve with BiCGSTAB
    import scipy.sparse.linalg as spla
    
    residual_history_cpu = []
    
    def iteration_callback(xk):
        r = b - A @ xk
        r_norm = np.linalg.norm(r)
        residual_history_cpu.append(r_norm)
        iter_count = len(residual_history_cpu)
        print(f"CPU Iteration {iter_count}: residual = {r_norm:.3e}")
    
    x0 = np.zeros_like(b, dtype=complex)
    
    solve_start_time = time.time()
    E_solution, info = spla.bicgstab(
        A, b, x0=x0, atol=1e-8, maxiter=2000,
        callback=iteration_callback
    )
    solve_time = time.time() - solve_start_time
    
    if info == 0:
        print(f"CPU BiCGSTAB converged in {len(residual_history_cpu)} iterations!")
    else:
        print(f"CPU BiCGSTAB ended with info={info}")
        
    print(f"CPU Solve time: {solve_time:.3f} seconds")
    print(f"CPU Total time: {assembly_time + solve_time:.3f} seconds")
    
    # Reshape the solution
    Ex_sol = E_solution[0 : nx*ny*nz].reshape((nx, ny, nz), order='F')
    Ey_sol = E_solution[nx*ny*nz : 2*nx*ny*nz].reshape((nx, ny, nz), order='F')
    Ez_sol = E_solution[2*nx*ny*nz :].reshape((nx, ny, nz), order='F')
    
    return Ex_sol, Ey_sol, Ez_sol, residual_history_cpu, assembly_time, solve_time

def run_gpu_version():
    if not GPU_AVAILABLE:
        print("GPU version skipped (CuPy not available)")
        return None, None, None, [], 0, 0
    
    print("\nRunning GPU version...")
    start_time = time.time()
    
    # GPU method that solves the system on the GPU
    A, b, Ch, E_solution = curlcurlE_gpu(
        L0        = L0,
        wvlen     = wvlen,
        xrange    = xrange,
        yrange    = yrange,
        zrange    = zrange,
        eps_r_tensor_dict = eps_r_tensor,
        JCurrentVector    = Jdict,
        Npml      = Npml,
        s         = -1,
        nonuniform= None,
        solve_system=True,
        tol=1e-8,
        max_iter=2000
    )
    
    total_time = time.time() - start_time
    print(f"GPU Total time: {total_time:.3f} seconds")
    
    # Reshape the solution
    Ex_sol = E_solution[0 : nx*ny*nz].reshape((nx, ny, nz), order='F')
    Ey_sol = E_solution[nx*ny*nz : 2*nx*ny*nz].reshape((nx, ny, nz), order='F')
    Ez_sol = E_solution[2*nx*ny*nz :].reshape((nx, ny, nz), order='F')
    
    # For simplicity, we don't track assembly vs solve time separately here
    # since it's done inside the curlcurlE_gpu function
    return Ex_sol, Ey_sol, Ez_sol, [], total_time, 0

# ---------------------------------------------------
# 7) Run both versions and compare
# ---------------------------------------------------
if __name__ == "__main__":
    print(f"Problem size: {nx} x {ny} x {nz} = {nx*ny*nz} voxels")

    # Run CPU version
    try:
        Ex_cpu, Ey_cpu, Ez_cpu, residual_history_cpu, cpu_assembly_time, cpu_solve_time = run_cpu_version()
        cpu_total_time = cpu_assembly_time + cpu_solve_time
    except Exception as e:
        print(f"CPU version failed with error: {e}")
        cpu_total_time = float('inf')
        Ex_cpu, Ey_cpu, Ez_cpu = None, None, None
    
    # Run GPU version if available
    try:
        Ex_gpu, Ey_gpu, Ez_gpu, residual_history_gpu, gpu_total_time, _ = run_gpu_version()
    except Exception as e:
        print(f"GPU version failed with error: {e}")
        gpu_total_time = float('inf')
        Ex_gpu, Ey_gpu, Ez_gpu = None, None, None
    
    # Print speedup comparison
    if cpu_total_time != float('inf') and gpu_total_time != float('inf'):
        speedup = cpu_total_time / gpu_total_time if GPU_AVAILABLE else 0
        print(f"\nSpeedup (CPU/GPU): {speedup:.2f}x")
    
    # Compare results if both methods succeeded
    if Ex_cpu is not None and Ex_gpu is not None:
        print("\nComparing results:")
        
        # Calculate relative differences
        Ex_diff = np.linalg.norm(Ex_cpu - Ex_gpu) / np.linalg.norm(Ex_cpu)
        Ey_diff = np.linalg.norm(Ey_cpu - Ey_gpu) / np.linalg.norm(Ey_cpu)
        Ez_diff = np.linalg.norm(Ez_cpu - Ez_gpu) / np.linalg.norm(Ez_cpu)
        
        print(f"Relative difference in Ex: {Ex_diff:.3e}")
        print(f"Relative difference in Ey: {Ey_diff:.3e}")
        print(f"Relative difference in Ez: {Ez_diff:.3e}")
    
    # ---------------------------------------------------
    # 8) Visualize the results
    # ---------------------------------------------------
    # Determine which solution to visualize
    if Ex_gpu is not None and GPU_AVAILABLE:
        Ex, Ey, Ez = Ex_gpu, Ey_gpu, Ez_gpu
        title_prefix = "GPU"
    elif Ex_cpu is not None:
        Ex, Ey, Ez = Ex_cpu, Ey_cpu, Ez_cpu
        title_prefix = "CPU"
    else:
        print("No valid solutions to visualize")
        exit(1)
        
    # Plot the field magnitude in a central slice
    midZ = nz//2
    
    plt.figure(figsize=(10, 8))
    
    plt.subplot(221)
    plt.imshow(np.abs(Ex[:,:,midZ].T), origin='lower', cmap='inferno')
    plt.colorbar(label='|Ex| at z-slice')
    plt.title(f'{title_prefix} |Ex| at central z-slice')
    
    plt.subplot(222)
    plt.imshow(np.abs(Ey[:,:,midZ].T), origin='lower', cmap='inferno')
    plt.colorbar(label='|Ey| at z-slice')
    plt.title(f'{title_prefix} |Ey| at central z-slice')
    
    plt.subplot(223)
    plt.imshow(np.abs(Ez[:,:,midZ].T), origin='lower', cmap='inferno')
    plt.colorbar(label='|Ez| at z-slice')
    plt.title(f'{title_prefix} |Ez| at central z-slice')
    
    # Plot total field magnitude
    plt.subplot(224)
    E_mag = np.sqrt(np.abs(Ex[:,:,midZ])**2 + np.abs(Ey[:,:,midZ])**2 + np.abs(Ez[:,:,midZ])**2)
    plt.imshow(E_mag.T, origin='lower', cmap='inferno')
    plt.colorbar(label='|E| at z-slice')
    plt.title(f'{title_prefix} Total |E| at central z-slice')
    
    plt.tight_layout()
    plt.savefig('fdfd_results.png', dpi=150)
    plt.show() 