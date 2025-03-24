import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

###############################################################################
# 1) 1D SBP Operators (Second-Order)
###############################################################################
def sbp_1d_2nd_order(N, dx):
    """
    Build second-order SBP operators on a 1D domain with (N+2) points
    (i=0..N+1). Returns:
      Dp: (N+2)x(N+2) derivative matrix (SBP version of d/dx on 'plus' grid)
      Pmat: diagonal mass matrix (N+2)x(N+2)
    """
    P = np.ones(N+2)       # simplistic uniform weights
    Pmat = np.diag(P)
    Q = np.zeros((N+2, N+2))

    # interior: centered difference
    for i in range(1, N+1):
        Q[i, i-1] = -0.5
        Q[i, i+1] =  0.5

    # boundaries: one-sided
    Q[0,0]    = -1.0; Q[0,1]    =  1.0
    Q[N+1,N]  = -1.0; Q[N+1,N+1]=  1.0

    invP = np.diag(1.0/P)
    Dp   = (1.0/dx)*(invP @ Q)
    return Dp, Pmat

###############################################################################
# 2) For 3D, we do a uniform mesh in y,z but have two blocks in x:
#    BLOCK 0: Nx_coarse cells => dx_coarse
#    BLOCK 1: Nx_fine   cells => dx_fine = dx_coarse / ratio
###############################################################################
def build_3d_ops(Nx, dx, Ny, dy, Nz, dz):
    """
    Build 1D SBP operators for x, y, z. We'll do second-order for each dimension.
    Nx,dx => for that block's x dimension
    We'll store in a dict: "Dx","Px","Dy","Py","Dz","Pz".
    """
    Dx_plus_1d, Px_1d = sbp_1d_2nd_order(Nx, dx)
    Dy_plus_1d, Py_1d = sbp_1d_2nd_order(Ny, dy)
    Dz_plus_1d, Pz_1d = sbp_1d_2nd_order(Nz, dz)
    return {
       "Dx": Dx_plus_1d, "Px": Px_1d,
       "Dy": Dy_plus_1d, "Py": Py_1d,
       "Dz": Dz_plus_1d, "Pz": Pz_1d,
    }

def dfdx_3d(f, D, Nx, Ny, Nz):
    """Approx partial derivative wrt x for a shape (Nx+2, Ny+2, Nz+2)."""
    out = np.zeros_like(f)
    for j in range(Ny+2):
        for k in range(Nz+2):
            out[:, j, k] = D @ f[:, j, k]
    return out

def dfdy_3d(f, D, Nx, Ny, Nz):
    """Partial derivative wrt y."""
    out = np.zeros_like(f)
    for i in range(Nx+2):
        for k in range(Nz+2):
            out[i, :, k] = D @ f[i, :, k]
    return out

def dfdz_3d(f, D, Nx, Ny, Nz):
    """Partial derivative wrt z."""
    out = np.zeros_like(f)
    for i in range(Nx+2):
        for j in range(Ny+2):
            out[i, j, :] = D @ f[i, j, :]
    return out

###############################################################################
# 3) Time-update for E/H in each block (like standard SBP-SAT FDTD).
#    We'll do a direct "PEC" on the outer boundary of each block,
#    plus we'll add an interface coupling for the block0/block1 boundary in x.
###############################################################################
def update_e_fields_block(E, H, ops, dt, Nx, Ny, Nz, cE):
    """
    E = (Ex, Ey, Ez); H = (Hx, Hy, Hz).
    ops = {Dx, Px, Dy, Py, Dz, Pz}.
    Nx,Ny,Nz => number of interior cells for this block.
    cE = 1/eps0.
    Returns updated (Ex, Ey, Ez).
    """
    Ex, Ey, Ez = E
    Hx, Hy, Hz = H

    dHy_dz = dfdz_3d(Hy, ops["Dz"], Nx, Ny, Nz)
    dHz_dy = dfdy_3d(Hz, ops["Dy"], Nx, Ny, Nz)
    dHz_dx = dfdx_3d(Hz, ops["Dx"], Nx, Ny, Nz)
    dHx_dz = dfdz_3d(Hx, ops["Dz"], Nx, Ny, Nz)
    dHx_dy = dfdy_3d(Hx, ops["Dy"], Nx, Ny, Nz)
    dHy_dx = dfdx_3d(Hy, ops["Dx"], Nx, Ny, Nz)

    Ex_new = Ex + dt*cE*( dHy_dz - dHz_dy )
    Ey_new = Ey + dt*cE*( dHz_dx - dHx_dz )
    Ez_new = Ez + dt*cE*( dHx_dy - dHy_dx )

    # Outer PEC boundaries => forcibly zero E
    # x=0, x=Nx+1
    Ex_new[ 0,:,:] = 0; Ex_new[-1,:,:] = 0
    Ey_new[ 0,:,:] = 0; Ey_new[-1,:,:] = 0
    Ez_new[ 0,:,:] = 0; Ez_new[-1,:,:] = 0
    # y=0, y=Ny+1
    Ex_new[:, 0,:] = 0; Ex_new[:,-1,:] = 0
    Ey_new[:, 0,:] = 0; Ey_new[:,-1,:] = 0
    Ez_new[:, 0,:] = 0; Ez_new[:,-1,:] = 0
    # z=0, z=Nz+1
    Ex_new[:,:, 0] = 0; Ex_new[:,:, -1] = 0
    Ey_new[:,:, 0] = 0; Ey_new[:,:, -1] = 0
    Ez_new[:,:, 0] = 0; Ez_new[:,:, -1] = 0

    return (Ex_new, Ey_new, Ez_new)

def update_h_fields_block(E, H, ops, dt, Nx, Ny, Nz, cH):
    """
    E = (Ex, Ey, Ez); H = (Hx, Hy, Hz).
    cH = 1/mu0.
    Returns updated (Hx, Hy, Hz).
    """
    Ex, Ey, Ez = E
    Hx, Hy, Hz = H

    dEz_dy = dfdy_3d(Ez, ops["Dy"], Nx, Ny, Nz)
    dEy_dz = dfdz_3d(Ey, ops["Dz"], Nx, Ny, Nz)
    dEx_dz = dfdz_3d(Ex, ops["Dz"], Nx, Ny, Nz)
    dEz_dx = dfdx_3d(Ez, ops["Dx"], Nx, Ny, Nz)
    dEy_dx = dfdx_3d(Ey, ops["Dx"], Nx, Ny, Nz)
    dEx_dy = dfdy_3d(Ex, ops["Dy"], Nx, Ny, Nz)

    Hx_new = Hx - dt*cH*( dEz_dy - dEy_dz )
    Hy_new = Hy - dt*cH*( dEx_dz - dEz_dx )
    Hz_new = Hz - dt*cH*( dEy_dx - dEx_dy )

    return (Hx_new, Hy_new, Hz_new)

###############################################################################
# 4) SAT interface coupling between block0 (coarse) and block1 (fine) in x-direction
#    We'll do a simple face-to-face interface. We must:
#      - Interpolate fields from coarse face to fine face
#      - Add penalty terms in E or H update
#    For brevity, we do a minimal approach with 2:1 ratio, no sub-time stepping.
###############################################################################

def sat_interface(E0, H0, Nx0, E1, H1, Nx1, ratio=2):
    """
    Minimal coupling for tangential E continuity, tangential H continuity, etc.
    Assume we are coupling block0's right boundary (i=Nx0+1) to block1's left boundary (i=0).
    ratio=2 => 2:1 spacing
    For a more rigorous method, you'd incorporate the SBP mass matrices (P) and sign the penalty
    exactly as in the theory.  Here, we do a simple average up/down to enforce continuity.
    """

    # E0, H0 are (Ex0, Ey0, Ez0), etc.
    Ex0, Ey0, Ez0 = E0
    Hx0, Hy0, Hz0 = H0
    Ex1, Ey1, Ez1 = E1
    Hx1, Hy1, Hz1 = H1

    # The "coarse" block boundary is i= Nx0+1 (the last column in x).
    # The "fine" block boundary is i=0 (the first column in x).
    # We do a naive average: the fine boundary nodes match the coarse boundary. 
    # Because ratio=2, each coarse cell in y,z corresponds to 2 fine cells. 
    # We'll do a quick 2D loop in y,z to exchange fields. 
    # For actual SBP-SAT, you'd add penalty terms to E,H updates. We'll do an approximate approach.

    # shape of block0 in y,z => (Ny0+2, Nz0+2)
    # shape of block1 in y,z => (2*(Ny0+2?), 2*(Nz0+2?)) if we also refined y,z. 
    # But for simplicity, let's keep y,z the same in both. Then ratio only in x. 
    # So "fine" has Nx1 = ratio*Nx0 in x, but Ny1=Ny0, Nz1=Nz0.
    # => we can match y=0..Ny0+1, z=0..Nz0+1 directly if ratio is only in x dimension.

    # iC = Nx0+1 in block0, iF = 0 in block1
    iC = Nx0+1
    iF = 0
    Ny = Ey0.shape[1]-2  # or block0.Ny
    Nz = Ez0.shape[2]-2

    # We'll do a simple average to enforce E continuity:
    #   E_coarse(iC, j, k) = E_fine(iF, j, k) = (E_coarse + E_fine)/2
    # For each tangential component. 
    # For normal component, you'd do the subgridding logic carefully. 
    # This is a partial demonstration.

    for j in range(Ny+2):
        for k in range(Nz+2):
            # average Ex => normal to interface, might differ. We'll do a direct average anyway
            eavg = 0.5*(Ex0[iC, j, k] + Ex1[iF, j, k])
            Ex0[iC, j, k] = eavg
            Ex1[iF, j, k] = eavg

            eavg = 0.5*(Ey0[iC, j, k] + Ey1[iF, j, k])
            Ey0[iC, j, k] = eavg
            Ey1[iF, j, k] = eavg

            eavg = 0.5*(Ez0[iC, j, k] + Ez1[iF, j, k])
            Ez0[iC, j, k] = eavg
            Ez1[iF, j, k] = eavg

            # same for H
            havg = 0.5*(Hx0[iC, j, k] + Hx1[iF, j, k])
            Hx0[iC, j, k] = havg
            Hx1[iF, j, k] = havg

            havg = 0.5*(Hy0[iC, j, k] + Hy1[iF, j, k])
            Hy0[iC, j, k] = havg
            Hy1[iF, j, k] = havg

            havg = 0.5*(Hz0[iC, j, k] + Hz1[iF, j, k])
            Hz0[iC, j, k] = havg
            Hz1[iF, j, k] = havg

    # Return the possibly updated fields
    return (Ex0,Ey0,Ez0),(Hx0,Hy0,Hz0),(Ex1,Ey1,Ez1),(Hx1,Hy1,Hz1)

###############################################################################
# 5) Main subgridding driver + animation
###############################################################################
def subgridding_3d_demo():
    """
    Two-block SBP-SAT FDTD with a 2:1 ratio in x dimension.
    We'll do a uniform y,z in both blocks, a point source in block0,
    and watch wave cross into block1.
    """
    #  block0: Nx0=10, dx0=1e-3
    #  block1: Nx1=20, dx1=dx0/2
    Nx0 = 10
    dx0 = 1e-3
    ratio = 2
    Nx1 = ratio*Nx0
    dx1 = dx0/ratio

    # unify y,z for both blocks
    Ny  = 10
    Nz  = 10
    dy  = 1e-3
    dz  = 1e-3

    # Build ops
    ops0 = build_3d_ops(Nx0, dx0, Ny, dy, Nz, dz)
    ops1 = build_3d_ops(Nx1, dx1, Ny, dy, Nz, dz)

    # arrays for each block
    # shape => (Nx+2, Ny+2, Nz+2)
    Ex0 = np.zeros((Nx0+2, Ny+2, Nz+2))
    Ey0 = np.zeros((Nx0+2, Ny+2, Nz+2))
    Ez0 = np.zeros((Nx0+2, Ny+2, Nz+2))
    Hx0 = np.zeros((Nx0+2, Ny+2, Nz+2))
    Hy0 = np.zeros((Nx0+2, Ny+2, Nz+2))
    Hz0 = np.zeros((Nx0+2, Ny+2, Nz+2))

    Ex1 = np.zeros((Nx1+2, Ny+2, Nz+2))
    Ey1 = np.zeros((Nx1+2, Ny+2, Nz+2))
    Ez1 = np.zeros((Nx1+2, Ny+2, Nz+2))
    Hx1 = np.zeros((Nx1+2, Ny+2, Nz+2))
    Hy1 = np.zeros((Nx1+2, Ny+2, Nz+2))
    Hz1 = np.zeros((Nx1+2, Ny+2, Nz+2))

    # Physical constants
    eps0 = 8.854187817e-12
    mu0  = 4.0e-7*np.pi
    c0   = 1.0/np.sqrt(eps0*mu0)

    # single dt (not local time stepping => we pick the stricter of the two)
    # coarse dt limit: dt0 ~ dx0/(c0 sqrt(3))
    # fine   dt limit: dt1 ~ dx1/(c0 sqrt(3)) => dt1= dt0/ ratio => smaller
    # We'll pick dt = dt1 (the smaller) for safety
    dt_coarse = 0.99 * dx0/(c0*np.sqrt(3))
    dt_fine   = 0.99 * dx1/(c0*np.sqrt(3))
    dt = min(dt_coarse, dt_fine)  # effectively dt_fine

    cE = 1.0/eps0
    cH = 1.0/mu0

    # For animation
    Nsteps = 200
    frames_Ez_slice = []

    # center of block0
    i0_cx = (Nx0+2)//2
    j0_cy = (Ny+2)//2
    k0_cz = (Nz+2)//2

    # We'll look at mid-plane z => zmid
    zmid = (Nz+2)//2

    for n in range(Nsteps):
        # 1) Update H in each block
        Hx0,Hy0,Hz0 = update_h_fields_block(
            (Ex0, Ey0, Ez0), (Hx0, Hy0, Hz0), ops0, dt, Nx0, Ny, Nz, cH)
        Hx1,Hy1,Hz1 = update_h_fields_block(
            (Ex1, Ey1, Ez1), (Hx1, Hy1, Hz1), ops1, dt, Nx1, Ny, Nz, cH)

        # 2) Apply interface SAT/coupling
        #    (Here done by naive averaging to enforce approximate continuity.)
        (Ex0,Ey0,Ez0),(Hx0,Hy0,Hz0),(Ex1,Ey1,Ez1),(Hx1,Hy1,Hz1) = sat_interface(
            (Ex0,Ey0,Ez0),(Hx0,Hy0,Hz0), Nx0,
            (Ex1,Ey1,Ez1),(Hx1,Hy1,Hz1), Nx1, ratio=ratio)

        # 3) Update E in each block
        Ex0,Ey0,Ez0 = update_e_fields_block(
            (Ex0, Ey0, Ez0), (Hx0, Hy0, Hz0), ops0, dt, Nx0, Ny, Nz, cE)
        Ex1,Ey1,Ez1 = update_e_fields_block(
            (Ex1, Ey1, Ez1), (Hx1, Hy1, Hz1), ops1, dt, Nx1, Ny, Nz, cE)

        # 4) Re-apply interface coupling for E as well
        (Ex0,Ey0,Ez0),(Hx0,Hy0,Hz0),(Ex1,Ey1,Ez1),(Hx1,Hy1,Hz1) = sat_interface(
            (Ex0,Ey0,Ez0),(Hx0,Hy0,Hz0), Nx0,
            (Ex1,Ey1,Ez1),(Hx1,Hy1,Hz1), Nx1, ratio=ratio)

        # 5) Add a point source in block0, Ez0
        t0 = 20.0
        spread = 6.0
        pulse = np.exp(-0.5*(((n - t0)/spread)**2))
        Ez0[i0_cx, j0_cy, k0_cz] += pulse

        # 6) Store a slice for animation:
        #    We'll merge the two blocks side-by-side into one bigger array for plotting.
        #    The total x-extent is Nx0+2 + Nx1+2 - 2 for the interface double-counting => Nx0+Nx1+2
        #    We'll form a 2D slice at z=zmid in y vs x.
        NxT = (Nx0+2) + (Nx1+2) 
        slice2d = np.zeros( (NxT, Ny+2) )

        # 2) Copy block0’s data (12 columns):
        slice2d[0 : Nx0+2, :] = Ez0[:,:,zmid]  # no transpose

        # 3) Copy block1’s data (22 columns):
        slice2d[ Nx0+2 : , :] = Ez1[:,:,zmid]  # no transpose

        # Transpose so that we get a shape that is (Ny+2) in vertical axis, (NxT) in horizontal, 
        # but let's just keep (NxT, Ny+2) for imshow with 'origin=lower' 
        # We'll store as is so that x is the first dimension, y is second dimension.

        frames_Ez_slice.append(slice2d.copy())

    # Now animate frames_Ez_slice with matplotlib
    fig, ax = plt.subplots()
    ax.set_title("Subgridding 3D SBP-SAT FDTD: Ez slice at z=mid")
    im = ax.imshow(frames_Ez_slice[0], origin='lower', cmap='RdBu', 
                   vmin=-0.2, vmax=0.2, animated=True)
    plt.colorbar(im, ax=ax)
    def init():
        im.set_data(frames_Ez_slice[0])
        return [im]

    def update(frame_idx):
        im.set_data(frames_Ez_slice[frame_idx])
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames_Ez_slice), init_func=init, blit=True, interval=50
    )
    plt.show()

if __name__ == "__main__":
    subgridding_3d_demo()