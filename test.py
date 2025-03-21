import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import bicgstab, LinearOperator
import matplotlib.pyplot as plt
import json

# Assuming the FDFD solver is in pyfd3d.fdfd
from pyfd3d.fdfd import fdfd_lossy

def test_fdfd_simulation():
    """Test a 3D FDFD simulation with BiCGSTAB and a Jacobi preconditioner, with live plotting."""
    # Simulation parameters
    L0 = 1e-6  # meters
    f = 434e6  # Hz
    c = 3e8   # m/s
    wvlen = c / f  # ≈ 0.691 m
    omega = 2 * np.pi * f

    # Domain: 4x4x4 micron grid, 50x50x50 points
    xrange = np.array([-2, 2])
    yrange = np.array([-2, 2])
    zrange = np.array([-2, 2])
    N = (50, 50, 50)
    Npml = (10, 10, 10)  # PML layers

    # Materials: air (0) and silicon (1)
    material_indices = np.zeros(N, dtype=int)
    material_indices[20:30, :, 20:30] = 1  # Waveguide along z-axis# material_indices[20:30, 20:30, 20:30] = 1  # Silicon block
    materials = {
        "0": {"name": "air", "e_R": 1.0, "sigma": 0.0},
        "1": {"name": "dielectric", "e_R": 2.8, "sigma": 0.0}
        # "1": {"name": "silicon", "e_R": 2.8, "sigma": 1}
        # "1": {"name": "silicon", "e_R": 2.8, "sigma": 7.5e-4}

    }
    with open("materials.json", "w") as f:
        json.dump(materials, f)

    # Dipole source at center
    Jx = np.zeros(N, dtype=complex)
    Jy = np.zeros(N, dtype=complex)
    Jz = np.zeros(N, dtype=complex)
    center_idx = (N[0]//2, N[1]//2, N[2]//2)
    Jz[center_idx] = 1.0
    JCurrentVector = {'Jx': Jx, 'Jy': Jy, 'Jz': Jz}

    # Run FDFD simulation
    A, b, Ch = fdfd_lossy(L0, wvlen, xrange, yrange, zrange, material_indices,
                          "materials.json", JCurrentVector, Npml)

    # Convert A to CSR format for faster operations
    A = A.tocsr()

    # Jacobi preconditioner: inverse of diagonal elements
    diag_A = A.diagonal()
    diag_A[np.abs(diag_A) < 1e-10] = 1.0  # Avoid division by zero
    M_jacobi = sp.diags(1.0 / diag_A, 0)
    M = LinearOperator(A.shape, lambda x: M_jacobi @ x)

    # --- Setup for live plotting ---
    M_total = np.prod(N)  # Total number of grid points (needed for reshaping)
    midZ = N[2] // 2      # Middle z-slice for visualization
    update_interval = 10  # Update image every 10 iterations to reduce overhead
    iter_count = 0        # Iteration counter
    residual_history = []

    # Figure 1: Residual plot (as in your original code)
    fig_res, ax_res = plt.subplots(figsize=(10, 6))
    line, = ax_res.plot([], [], 'b-')
    ax_res.set_xlabel("Iteration")
    ax_res.set_ylabel("Residual Norm")
    ax_res.set_yscale("log")
    ax_res.set_title("BiCGSTAB Convergence (log scale)")

    # Figure 2: Live Ez field plot
    fig_ez, ax_ez = plt.subplots(figsize=(10, 6))
    Ez_slice = np.zeros((N[0], N[1]), dtype=complex)  # Placeholder for Ez slice
    im = ax_ez.imshow(np.abs(Ez_slice.T), origin='lower', cmap='inferno')
    plt.colorbar(im, ax=ax_ez, label='|Ez| at z-slice')
    
    # Add material overlay to live plot
    material_slice = material_indices[:, :, midZ]
    material_contour = ax_ez.contour(material_slice.T, levels=[0.5], colors='white', linewidths=2, alpha=0.5)
    
    ax_ez.set_title('Central z-slice of Ez (live update)')
    ax_ez.set_xlabel('x')
    ax_ez.set_ylabel('y')

    # Enable interactive mode for live updates
    plt.ion()

    def iteration_callback(xk):
        nonlocal iter_count, material_contour
        iter_count += 1
        # Compute residual
        r = b - A @ xk
        r_norm = np.linalg.norm(r)
        residual_history.append(r_norm)
        print(f"Iteration {iter_count}: residual = {r_norm:.3e}")

        # Update residual plot
        xdata = np.arange(1, len(residual_history) + 1)
        line.set_xdata(xdata)
        line.set_ydata(residual_history)
        ax_res.relim()
        ax_res.autoscale_view()
        fig_res.canvas.draw()
        fig_res.canvas.flush_events()

        # Update Ez image every 'update_interval' iterations
        if iter_count % update_interval == 0:
            Ez_k = xk[2*M_total:].reshape(N, order='F')  # Extract Ez from xk
            Ez_slice = Ez_k[:, :, midZ]                   # Get central z-slice
            im.set_data(np.abs(Ez_slice.T))               # Update image data
            im.set_clim(vmin=0, vmax=np.max(np.abs(Ez_slice)))  # Adjust color scale
            # Update material contour
            if hasattr(material_contour, 'collections'):
                for c in material_contour.collections:
                    c.remove()
            material_contour = ax_ez.contour(material_slice.T, levels=[0.5], colors='white', linewidths=2, alpha=0.5)
            fig_ez.canvas.draw()
            fig_ez.canvas.flush_events()

        plt.pause(0.01)  # Allow GUI to update

    # Solve with BiCGSTAB
    tol = 1e-6
    max_iter = 6000
    E, info = bicgstab(A, b, M=M, rtol=tol, maxiter=max_iter, callback=iteration_callback)
    if info != 0:
        print(f"BiCGSTAB failed to converge: info = {info}")
    print(f"BiCGSTAB converged: info = {info}")

    # Reshape final electric field
    Ex = E[:M_total].reshape(N, order='F')
    Ey = E[M_total:2*M_total].reshape(N, order='F')
    Ez = E[2*M_total:].reshape(N, order='F')

    # Validate
    if np.abs(Ez[center_idx]) <= 1e-10:
        raise AssertionError("Ez at source is too small!")

    # Final plot (turn off interactive mode)
    plt.ioff()
    fig_final, ax_final = plt.subplots(figsize=(10, 6))
    Ez_slice = Ez[:, :, midZ]  # Get central z-slice
    im = ax_final.imshow(np.abs(Ez_slice.T), origin='lower', cmap='inferno')
    plt.colorbar(im, ax=ax_final, label='|Ez| at z-slice')
    
    # Add material overlay
    material_slice = material_indices[:, :, midZ]
    ax_final.contour(material_slice.T, levels=[0.5], colors='white', linewidths=2, alpha=0.5)
    
    ax_final.set_title('Central z-slice of Ez (final)')
    ax_final.set_xlabel('x')
    ax_final.set_ylabel('y')
    plt.show()

    print("Test passed: Non-zero Ez near source.")
    print(f"Ez at source position: {Ez[center_idx]:.3e}")

if __name__ == "__main__":
    test_fdfd_simulation()