import numpy as np
import scipy.sparse as sp
import json
from pyfd3d.derivatives import createDws
from pyfd3d.pml import S_create_3D
from pyfd3d.utils import bwdmean

def fdfd_lossy(
    L0: float,                  # Scaling parameter (e.g., 1e-6 for microns)
    wvlen: float,               # Wavelength in units of L0
    xrange: np.ndarray,         # [xmin, xmax] in units of L0
    yrange: np.ndarray,         # [ymin, ymax]
    zrange: np.ndarray,         # [zmin, zmax]
    material_indices: np.ndarray, # 3D array of integer indices
    materials_json_path: str,   # Path to materials JSON file
    JCurrentVector: dict,       # Current density {'Jx': ..., 'Jy': ..., 'Jz': ...}
    Npml: tuple,                # (Nx_pml, Ny_pml, Nz_pml)
    s: float = -1               # Beltrami-Laplace parameter
):
    """
    3D FDFD simulation with lossy materials, PML boundaries, and μ=1.
    
    Args:
        L0: Distance scaling factor (e.g., 1e-6 for microns).
        wvlen: Wavelength in units of L0.
        xrange, yrange, zrange: Simulation domain bounds.
        material_indices: 3D NumPy array of integers mapping to materials.
        materials_json_path: Path to JSON file with material properties.
        JCurrentVector: Dict with Jx, Jy, Jz current density arrays.
        Npml: Tuple of PML thicknesses in x, y, z directions.
        s: Beltrami-Laplace scaling factor (default -1).
    
    Returns:
        A: System matrix (sparse).
        b: Source vector.
        Ch: Curl operator for H-field recovery.
    """
    
    # Physical constants (scaled by L0)
    eps0 = 8.85e-12 * L0  # Permittivity of free space
    mu0 = 4 * np.pi * 1e-7 * L0  # Permeability of free space (μ_r = 1)
    eta0 = np.sqrt(mu0 / eps0)  # Impedance of free space
    c0 = 1 / np.sqrt(eps0 * mu0)  # Speed of light
    omega = 2 * np.pi * c0 / wvlen  # Angular frequency (rad/s)

    # Load materials from JSON
    with open(materials_json_path, 'r') as f:
        materials = json.load(f)
    
    # Validate material indices
    N = material_indices.shape
    if N != (int(np.diff(xrange)[0] / (xrange[1] - xrange[0]) * N[0]), 
             int(np.diff(yrange)[0] / (yrange[1] - yrange[0]) * N[1]), 
             int(np.diff(zrange)[0] / (zrange[1] - zrange[0]) * N[2])):
        raise ValueError("Material indices shape does not match domain grid.")
    
    # Construct complex permittivity (ε = ε_0 * (ε_r - jσ/ω))
    unique_indices = np.unique(material_indices)
    eps_r = np.zeros(N, dtype=complex)
    for idx in unique_indices:
        if str(idx) not in materials:
            raise KeyError(f"Material index {idx} not found in materials JSON.")
        mat = materials[str(idx)]
        eps_r[material_indices == idx] = mat['e_R'] - 1j * mat['sigma'] / (omega * eps0)

    # Assuming isotropic materials for simplicity (eps_xx = eps_yy = eps_zz)
    eps_xx = eps_r.copy()
    eps_yy = eps_r.copy()
    eps_zz = eps_r.copy()

    # Edge smoothing for Yee grid
    eps_xx = bwdmean(eps_xx, 'x')
    eps_yy = bwdmean(eps_yy, 'y')
    eps_zz = bwdmean(eps_zz, 'z')

    # Grid parameters
    M = np.prod(N)
    L = np.array([np.diff(xrange)[0], np.diff(yrange)[0], np.diff(zrange)[0]])
    dL = L / N

    # Permittivity and inverse permittivity tensors
    Tepx = sp.spdiags(eps0 * eps_xx.flatten(order='F'), 0, M, M)
    Tepy = sp.spdiags(eps0 * eps_yy.flatten(order='F'), 0, M, M)
    Tepz = sp.spdiags(eps0 * eps_zz.flatten(order='F'), 0, M, M)

    iTepx = sp.spdiags(1 / (eps0 * eps_xx.flatten(order='F')), 0, M, M)
    iTepy = sp.spdiags(1 / (eps0 * eps_yy.flatten(order='F')), 0, M, M)
    iTepz = sp.spdiags(1 / (eps0 * eps_zz.flatten(order='F')), 0, M, M)

    TepsSuper = sp.block_diag((Tepx, Tepy, Tepz))
    iTepsSuper = sp.block_diag((iTepx, iTepy, iTepz))

    # Magnetic permeability (μ = μ_0 since μ_r = 1)
    iTmuSuper = (1 / mu0) * sp.identity(3 * M)
    TmuSuper = mu0 * sp.identity(3 * M)

    # PML operators
    Sxfi, Sxbi, Syfi, Sybi, Szfi, Szbi = S_create_3D(omega, dL, N, Npml, eps0, eta0)

    # Derivative operators with PML
    Dxf = Sxfi @ createDws('x', 'f', dL, N)
    Dxb = Sxbi @ createDws('x', 'b', dL, N)
    Dyf = Syfi @ createDws('y', 'f', dL, N)
    Dyb = Sybi @ createDws('y', 'b', dL, N)
    Dzf = Szfi @ createDws('z', 'f', dL, N)
    Dzb = Szbi @ createDws('z', 'b', dL, N)

    # Curl operators
    Ce = sp.bmat([[None, -Dzf, Dyf],
                  [Dzf, None, -Dxf],
                  [-Dyf, Dxf, None]])
    Ch = sp.bmat([[None, -Dzb, Dyb],
                  [Dzb, None, -Dxb],
                  [-Dyb, Dxb, None]])

    # Grad-div (Beltrami-Laplace) operator
    gd00 = Dxf @ iTepx @ Dxb @ Tepx
    gd01 = Dxf @ iTepx @ (Dyb @ Tepy)
    gd02 = Dxf @ iTepx @ (Dzb @ Tepz)
    gd10 = Dyf @ iTepy @ (Dxb @ Tepx)
    gd11 = Dyf @ iTepy @ (Dyb @ Tepy)
    gd12 = Dyf @ iTepy @ (Dzb @ Tepz)
    gd20 = Dzf @ iTepz @ (Dxb @ Tepx)
    gd21 = Dzf @ iTepz @ (Dyb @ Tepy)
    gd22 = Dzf @ iTepz @ (Dzb @ Tepz)

    GradDiv = sp.bmat([[gd00, gd01, gd02],
                       [gd10, gd11, gd12],
                       [gd20, gd21, gd22]])

    # System matrix A
    WAccelScal = sp.identity(3 * M) @ iTmuSuper
    A = Ch @ iTmuSuper @ Ce + s * WAccelScal @ GradDiv - omega**2 * TepsSuper

    # Source vector b
    Jx = JCurrentVector['Jx'].flatten(order='F')
    Jy = JCurrentVector['Jy'].flatten(order='F')
    Jz = JCurrentVector['Jz'].flatten(order='F')
    J = np.concatenate((Jx, Jy, Jz), axis=0)
    
    b = -1j * omega * J
    JCorrection = (1j / omega) * (s * GradDiv @ WAccelScal) @ iTepsSuper @ J
    b = b + JCorrection

    return A, b, Ch

# Example usage
if __name__ == "__main__":
    # Define simulation parameters
    L0 = 1e-6  # Microns
    wvlen = 1.55  # Wavelength in microns
    xrange = np.array([-2, 2])
    yrange = np.array([-2, 2])
    zrange = np.array([-2, 2])
    N = (50, 50, 50)
    Npml = (10, 10, 10)

    # Dummy material indices (0: air, 1: silicon)
    material_indices = np.zeros(N, dtype=int)
    material_indices[20:30, 20:30, 20:30] = 1  # Silicon block

    # Materials JSON
    materials = {
        "0": {"name": "air", "e_R": 1.0, "sigma": 0.0},
        "1": {"name": "silicon", "e_R": 11.7, "sigma": 0.01}
    }
    with open("materials.json", "w") as f:
        json.dump(materials, f)

    # Dummy dipole source at center
    Jx = np.zeros(N, dtype=complex)
    Jy = np.zeros(N, dtype=complex)
    Jz = np.zeros(N, dtype=complex)
    Jz[25, 25, 25] = 1.0  # z-polarized dipole
    JCurrentVector = {'Jx': Jx, 'Jy': Jy, 'Jz': Jz}

    # Run simulation
    A, b, Ch = fdfd_lossy(L0, wvlen, xrange, yrange, zrange, material_indices, 
                          "materials.json", JCurrentVector, Npml)
    print("System matrix A shape:", A.shape)
    print("Source vector b shape:", b.shape)