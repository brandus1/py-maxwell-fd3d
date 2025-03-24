import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

###############################################################################
# 1) Build small second-order 1D SBP difference operators: D+ (plus grid),
#    along with the mass matrix P. (Same approach as in earlier demos.)
###############################################################################
def sbp_1d_2nd_order(N, dx):
    """
    Build second-order SBP operators on a 1D domain with (N+2) points
    (indices i=0..N+1). We return:
      Dp: (N+2)x(N+2) derivative matrix (SBP version of d/dx on 'plus' grid)
      Pmat: diagonal mass matrix (N+2)x(N+2)
    """
    P = np.ones(N+2)
    Pmat = np.diag(P)

    Q = np.zeros((N+2, N+2))
    # interior points: central difference
    for i in range(1, N+1):
        Q[i, i-1] = -0.5
        Q[i, i+1] =  0.5
    # boundary rows: one-sided
    Q[0,0]   = -1.0; Q[0,1]   =  1.0
    Q[N+1,N] = -1.0; Q[N+1,N+1] =  1.0

    invP = np.diag(1.0/P)
    Dp   = (1.0/dx) * (invP @ Q)
    return Dp, Pmat

###############################################################################
# 2) Build 3D SBP operators from the 1D operators via Kronecker products,
#    but we'll just store the 1D operators and do the 3D derivative by looping.
###############################################################################
def build_3d_ops(Nx, Ny, Nz, dx, dy, dz):
    Dx_plus_1d, Px_1d = sbp_1d_2nd_order(Nx, dx)
    Dy_plus_1d, Py_1d = sbp_1d_2nd_order(Ny, dy)
    Dz_plus_1d, Pz_1d = sbp_1d_2nd_order(Nz, dz)
    ops = {
        "Dx": Dx_plus_1d, "Dy": Dy_plus_1d, "Dz": Dz_plus_1d,
        "Px": Px_1d,      "Py": Py_1d,      "Pz": Pz_1d
    }
    return ops

###############################################################################
# 3) Helpers for partial derivatives in x, y, z for 3D arrays
###############################################################################
def dfdx_3d(f, D, Nx, Ny, Nz):
    out = np.zeros_like(f)
    for j in range(Ny+2):
        for k in range(Nz+2):
            out[:, j, k] = D @ f[:, j, k]
    return out

def dfdy_3d(f, D, Nx, Ny, Nz):
    out = np.zeros_like(f)
    for i in range(Nx+2):
        for k in range(Nz+2):
            out[i, :, k] = D @ f[i, :, k]
    return out

def dfdz_3d(f, D, Nx, Ny, Nz):
    out = np.zeros_like(f)
    for i in range(Nx+2):
        for j in range(Ny+2):
            out[i, j, :] = D @ f[i, j, :]
    return out

###############################################################################
# 4) Update equations for E and H
###############################################################################
def update_e_fields(Ex, Ey, Ez, Hx, Hy, Hz, ops, dt, Nx, Ny, Nz, cE):
    dHy_dz = dfdz_3d(Hy, ops["Dz"], Nx, Ny, Nz)
    dHz_dy = dfdy_3d(Hz, ops["Dy"], Nx, Ny, Nz)
    dHz_dx = dfdx_3d(Hz, ops["Dx"], Nx, Ny, Nz)
    dHx_dz = dfdz_3d(Hx, ops["Dz"], Nx, Ny, Nz)
    dHx_dy = dfdy_3d(Hx, ops["Dy"], Nx, Ny, Nz)
    dHy_dx = dfdx_3d(Hy, ops["Dx"], Nx, Ny, Nz)

    Ex_new = Ex + dt*cE*( dHy_dz - dHz_dy )
    Ey_new = Ey + dt*cE*( dHz_dx - dHx_dz )
    Ez_new = Ez + dt*cE*( dHx_dy - dHy_dx )

    # Impose PEC: forcibly zero out E at boundary
    Ex_new[ 0, : , :] = 0; Ex_new[-1,: , :] = 0
    Ex_new[ : , 0 , :] = 0; Ex_new[ : ,-1 , :] = 0
    Ex_new[ : , : , 0] = 0; Ex_new[ : , : ,-1] = 0

    Ey_new[ 0, : , :] = 0; Ey_new[-1,: , :] = 0
    Ey_new[ : , 0 , :] = 0; Ey_new[ : ,-1 , :] = 0
    Ey_new[ : , : , 0] = 0; Ey_new[ : , : ,-1] = 0

    Ez_new[ 0, : , :] = 0; Ez_new[-1,: , :] = 0
    Ez_new[ : , 0 , :] = 0; Ez_new[ : ,-1 , :] = 0
    Ez_new[ : , : , 0] = 0; Ez_new[ : , : ,-1] = 0

    return Ex_new, Ey_new, Ez_new

def update_h_fields(Ex, Ey, Ez, Hx, Hy, Hz, ops, dt, Nx, Ny, Nz, cH):
    dEz_dy = dfdy_3d(Ez, ops["Dy"], Nx, Ny, Nz)
    dEy_dz = dfdz_3d(Ey, ops["Dz"], Nx, Ny, Nz)
    dEx_dz = dfdz_3d(Ex, ops["Dz"], Nx, Ny, Nz)
    dEz_dx = dfdx_3d(Ez, ops["Dx"], Nx, Ny, Nz)
    dEy_dx = dfdx_3d(Ey, ops["Dx"], Nx, Ny, Nz)
    dEx_dy = dfdy_3d(Ex, ops["Dy"], Nx, Ny, Nz)

    Hx_new = Hx - dt*cH*( dEz_dy - dEy_dz )
    Hy_new = Hy - dt*cH*( dEx_dz - dEz_dx )
    Hz_new = Hz - dt*cH*( dEy_dx - dEx_dy )

    # For PEC, there's typically no direct constraint on H at the boundary, so do nothing special here.
    return Hx_new, Hy_new, Hz_new

###############################################################################
# 5) Generator function that runs the SBP-SAT FDTD time steps, yields Ez slice
###############################################################################
def fdtd_3d_sbp_sat_pec(Nx=20, Ny=20, Nz=20, Nsteps=100):
    """
    A generator that yields Ez-slice data after each time step.
    Nx,Ny,Nz: number of interior cells (not counting the +2 boundary layers).
    Nsteps: number of time steps to run.
    We'll yield a 2D array of Ez at x=mid-plane each iteration.
    """
    dx = dy = dz = 1e-3

    ops = build_3d_ops(Nx, Ny, Nz, dx, dy, dz)

    # Free-space constants
    eps0 = 8.854187817e-12
    mu0  = 4*np.pi*1e-7
    c0   = 1.0/np.sqrt(eps0*mu0)

    # stable dt
    dt = 0.99 * min(dx, dy, dz)/(np.sqrt(3)*c0)

    # Allocate field arrays (Nx+2) x (Ny+2) x (Nz+2)
    Ex = np.zeros((Nx+2, Ny+2, Nz+2))
    Ey = np.zeros((Nx+2, Ny+2, Nz+2))
    Ez = np.zeros((Nx+2, Ny+2, Nz+2))
    Hx = np.zeros((Nx+2, Ny+2, Nz+2))
    Hy = np.zeros((Nx+2, Ny+2, Nz+2))
    Hz = np.zeros((Nx+2, Ny+2, Nz+2))

    # coefficients used in updates
    cE = 1.0/eps0
    cH = 1.0/mu0

    # Indices of domain center
    i0 = (Nx+2)//2
    j0 = (Ny+2)//2
    k0 = (Nz+2)//2

    for n in range(Nsteps):
        # Update H from E
        Hx, Hy, Hz = update_h_fields(Ex, Ey, Ez, Hx, Hy, Hz, ops, dt, Nx, Ny, Nz, cH)

        # Update E from H
        Ex, Ey, Ez = update_e_fields(Ex, Ey, Ez, Hx, Hy, Hz, ops, dt, Nx, Ny, Nz, cE)

        # Add a simple Gaussian source in Ez at center
        t0 = 20.0
        spread = 6.0
        pulse = np.exp(-0.5*(((n - t0)/spread)**2))
        Ez[i0, j0, k0] += pulse

        # yield the slice x=mid-plane (a 2D array in y,z)
        xmid = i0
        Ez_slice = Ez[xmid,:,:]
        yield Ez_slice

###############################################################################
# 6) Build a matplotlib animation of the Ez mid-plane slice
###############################################################################
def animate_sbp_sat_fdtd_3d():
    Nx, Ny, Nz = 20, 20, 20
    Nsteps = 120  # number of frames in animation

    fig, ax = plt.subplots()
    ax.set_title("Ez slice at x=mid-plane")

    # Use generator
    field_generator = fdtd_3d_sbp_sat_pec(Nx, Ny, Nz, Nsteps)

    # Initialize first frame
    Ez_slice_init = next(field_generator)
    im = ax.imshow(Ez_slice_init, origin='lower', 
                   cmap='RdBu', animated=True, 
                   vmin=-0.2, vmax=0.2)  # adjust vmin/vmax as needed
    plt.colorbar(im, ax=ax)

    def init():
        im.set_data(Ez_slice_init)
        return [im]

    def update(frame):
        # frame is the next Ez_slice
        im.set_data(frame)
        return [im]

    # Build an animation using the generator as frames
    ani = animation.FuncAnimation(
        fig, update, frames=field_generator, 
        init_func=init, blit=True, interval=50
    )

    plt.show()

if __name__ == "__main__":
    animate_sbp_sat_fdtd_3d()