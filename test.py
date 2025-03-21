import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import bicgstab, LinearOperator
import matplotlib.pyplot as plt
import json

# Assuming the FDFD solver is in pyfd3d.fdfd
from pyfd3d.fdfd import fdfd_lossy

def test_fdfd_simulation():
    """Test a 3D FDFD simulation with GMRES and a Jacobi preconditioner."""
    # Simulation parameters
    L0 = 1e-6  # Scaling factor (microns)
    wvlen = 1.55  # Wavelength (microns)
    omega = 2 * np.pi * (3e8 / (wvlen * L0))  # Angular frequency (rad/s)

    # Domain: 4x4x4 micron grid, 50x50x50 points
    xrange = np.array([-2, 2])
    yrange = np.array([-2, 2])
    zrange = np.array([-2, 2])
    N = (50, 50, 50)
    Npml = (10, 10, 10)  # PML layers

    # Materials: air (0) and silicon (1)
    material_indices = np.zeros(N, dtype=int)
    material_indices[20:30, 20:30, 20:30] = 1  # Silicon block
    materials = {
        "0": {"name": "air", "e_R": 1.0, "sigma": 0.0},
        "1": {"name": "silicon", "e_R": 11.7, "sigma": 0.01}
    }
    with open("materials.json", "w") as f:
        json.dump(materials, f)

    # Dipole source at center
    Jx = np.zeros(N, dtype=complex)
    Jy = np.zeros(N, dtype=complex)
    Jz = np.zeros(N, dtype=complex)
    center_idx = (25, 25, 25)
    Jz[center_idx] = 1.0
    JCurrentVector = {'Jx': Jx, 'Jy': Jy, 'Jz': Jz}

    # Run FDFD simulation
    A, b, Ch = fdfd_lossy(L0, wvlen, xrange, yrange, zrange, material_indices,
                          "materials.json", JCurrentVector, Npml)

    # Convert A to CSR format for faster operations
    A = A.tocsr()

    # Jacobi preconditioner: inverse of diagonal elements
    diag_A = A.diagonal()
    # Avoid division by zero by setting small values to 1
    diag_A[np.abs(diag_A) < 1e-10] = 1.0
    M_jacobi = sp.diags(1.0 / diag_A, 0)
    M = LinearOperator(A.shape, lambda x: M_jacobi @ x)

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

    # Solve with GMRES
    tol = 1e-6
    max_iter = 2000
    E, info = bicgstab(A, b, M=M, rtol=tol, maxiter=max_iter, callback=iteration_callback)
    if info != 0:
        print(f"GMRES failed to converge: info = {info}")

    # Reshape electric field
    M = np.prod(N)
    Ex = E[:M].reshape(N, order='F')
    Ey = E[M:2*M].reshape(N, order='F')
    Ez = E[2*M:].reshape(N, order='F')

    # Validate
    if np.abs(Ez[center_idx]) <= 1e-10:
        raise AssertionError("Ez at source is too small!")

    plt.figure(figsize=(10, 6))
    midZ = N[2] // 2
    plt.imshow(np.abs(Ez[:,:,midZ].T), origin='lower', cmap='inferno')
    plt.colorbar(label='|Ez| at z-slice')
    plt.title('Central z-slice of Ez')
    plt.show()
    print("Test passed: Non-zero Ez near source.")
    print(f"Ez at source position: {Ez[center_idx]:.3e}")

if __name__ == "__main__":
    test_fdfd_simulation()