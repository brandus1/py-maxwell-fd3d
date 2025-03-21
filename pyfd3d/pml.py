import numpy as np
import scipy.sparse as sp
try:
    import cupy as cp
    from cupyx.scipy.sparse import spdiags as cp_spdiags
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# EPSILON_0 = 8.85*10**-12
# MU_0 = 4*np.pi*10**-7
# ETA_0 = np.sqrt(MU_0/EPSILON_0)

def sig_w(l, dw, eta_0, m=3, lnR=-30):
    # helper for S()
    sig_max = -(m + 1) * lnR / (2 * eta_0 * dw)
    return sig_max * (l / dw)**m


def S(l, dw, omega, epsilon_0, eta_0):
    # helper for create_sfactor()
    return 1 - 1j * sig_w(l, dw, eta_0) / (omega * epsilon_0)


def create_sfactor(s, omega, dL, N, N_pml, epsilon_0, eta_0):
    # used to help construct the S matrices for the PML creation
    '''
        eta_0: if dL is in units of micromeneters, eta_0 must be in units of micrometers 
        epsilon_0: 
    '''
    sfactor_vecay = np.ones(N, dtype=np.complex128)
    if N_pml < 1:
        return sfactor_vecay

    dw = N_pml * dL

    for i in range(N):
        if s == 'f':
            if i <= N_pml:
                sfactor_vecay[i] = S(dL * (N_pml - i + 0.5), dw, omega, epsilon_0, eta_0)
            elif i > N - N_pml:
                sfactor_vecay[i] = S(dL * (i - (N - N_pml) - 0.5), dw, omega, epsilon_0, eta_0)
        if s == 'b':
            if i <= N_pml:
                sfactor_vecay[i] = S(dL * (N_pml - i + 1), dw, omega, epsilon_0, eta_0)
            elif i > N - N_pml:
                sfactor_vecay[i] = S(dL * (i - (N - N_pml) - 1), dw, omega, epsilon_0, eta_0)
    return sfactor_vecay


def create_sfactor_gpu(s, omega, dL, N, N_pml, epsilon_0, eta_0):
    if not GPU_AVAILABLE:
        raise ImportError("CuPy is not available. Cannot use GPU acceleration.")
        
    sfactor_vecay = cp.ones(N, dtype=cp.complex128)
    if N_pml < 1:
        return sfactor_vecay

    dw = N_pml * dL

    # Using CuPy's element-wise operations instead of loop
    i_array = cp.arange(N)
    
    if s == 'f':
        # First PML region
        mask1 = i_array <= N_pml
        l_values1 = dL * (N_pml - i_array[mask1] + 0.5)
        sig_values1 = -(3 + 1) * (-30) / (2 * eta_0 * dw) * (l_values1 / dw)**3
        sfactor_vecay[mask1] = 1 - 1j * sig_values1 / (omega * epsilon_0)
        
        # Second PML region
        mask2 = i_array > (N - N_pml)
        l_values2 = dL * (i_array[mask2] - (N - N_pml) - 0.5)
        sig_values2 = -(3 + 1) * (-30) / (2 * eta_0 * dw) * (l_values2 / dw)**3
        sfactor_vecay[mask2] = 1 - 1j * sig_values2 / (omega * epsilon_0)
    
    if s == 'b':
        # First PML region
        mask1 = i_array <= N_pml
        l_values1 = dL * (N_pml - i_array[mask1] + 1)
        sig_values1 = -(3 + 1) * (-30) / (2 * eta_0 * dw) * (l_values1 / dw)**3
        sfactor_vecay[mask1] = 1 - 1j * sig_values1 / (omega * epsilon_0)
        
        # Second PML region
        mask2 = i_array > (N - N_pml)
        l_values2 = dL * (i_array[mask2] - (N - N_pml) - 1)
        sig_values2 = -(3 + 1) * (-30) / (2 * eta_0 * dw) * (l_values2 / dw)**3
        sfactor_vecay[mask2] = 1 - 1j * sig_values2 / (omega * epsilon_0)
    
    return sfactor_vecay


def create_sc_pml(omega, dL, N, Npml, epsilon_0, eta_0):
    dx, dy, dz = dL
    Nx, Ny, Nz = N
    Nx_pml, Ny_pml, Nz_pml = Npml
    M = np.prod(N);
    
    sxf = create_sfactor('f', omega, dx, Nx, Nx_pml, epsilon_0, eta_0)
    syf = create_sfactor('f', omega, dy, Ny, Ny_pml, epsilon_0, eta_0)
    szf = create_sfactor('f', omega, dz, Nz, Nz_pml, epsilon_0, eta_0)
    
    sxb= create_sfactor('b', omega, dx, Nx, Nx_pml, epsilon_0, eta_0)
    syb= create_sfactor('b', omega, dy, Ny, Ny_pml, epsilon_0, eta_0)
    szb= create_sfactor('b', omega, dz, Nz, Nz_pml, epsilon_0, eta_0)
    
    #now we create the matrix (i.e. repeat sxf Ny times repeat Syf Nx times)
    [Sxf, Syf, Szf] = np.meshgrid(sxf, syf, szf, indexing = 'ij');
    [Sxb, Syb, Szb] = np.meshgrid(sxb, syb, szb, indexing = 'ij');
    
    return Sxf, Syf, Szf, Sxb, Syb, Szb

def create_sc_pml_gpu(omega, dL, N, Npml, epsilon_0, eta_0):
    if not GPU_AVAILABLE:
        raise ImportError("CuPy is not available. Cannot use GPU acceleration.")
        
    dx, dy, dz = dL
    Nx, Ny, Nz = N
    Nx_pml, Ny_pml, Nz_pml = Npml
    
    sxf = create_sfactor_gpu('f', omega, dx, Nx, Nx_pml, epsilon_0, eta_0)
    syf = create_sfactor_gpu('f', omega, dy, Ny, Ny_pml, epsilon_0, eta_0)
    szf = create_sfactor_gpu('f', omega, dz, Nz, Nz_pml, epsilon_0, eta_0)
    
    sxb = create_sfactor_gpu('b', omega, dx, Nx, Nx_pml, epsilon_0, eta_0)
    syb = create_sfactor_gpu('b', omega, dy, Ny, Ny_pml, epsilon_0, eta_0)
    szb = create_sfactor_gpu('b', omega, dz, Nz, Nz_pml, epsilon_0, eta_0)
    
    # Create meshgrid on GPU
    xx, yy, zz = cp.meshgrid(cp.arange(Nx), cp.arange(Ny), cp.arange(Nz), indexing='ij')
    
    # Use broadcasting to create the 3D PML arrays
    Sxf = sxf[:, cp.newaxis, cp.newaxis] + cp.zeros_like(xx, dtype=cp.complex128)
    Syf = syf[cp.newaxis, :, cp.newaxis] + cp.zeros_like(xx, dtype=cp.complex128)
    Szf = szf[cp.newaxis, cp.newaxis, :] + cp.zeros_like(xx, dtype=cp.complex128)
    
    Sxb = sxb[:, cp.newaxis, cp.newaxis] + cp.zeros_like(xx, dtype=cp.complex128)
    Syb = syb[cp.newaxis, :, cp.newaxis] + cp.zeros_like(xx, dtype=cp.complex128)
    Szb = szb[cp.newaxis, cp.newaxis, :] + cp.zeros_like(xx, dtype=cp.complex128)
    
    return Sxf, Syf, Szf, Sxb, Syb, Szb

def S_create_3D(omega, dL, N, Npml, epsilon_0, eta_0):
    dx, dy, dz = dL
    Nx, Ny, Nz = N
    Nx_pml, Ny_pml, Nz_pml = Npml
    M = np.prod(N);
    
    Sxf, Syf, Szf, Sxb, Syb, Szb =  create_sc_pml(omega, dL, N, Npml, epsilon_0, eta_0);
    
    #Sxf(:) converts from n x n t0 n^2 x 1
    Sxfi=sp.spdiags(1/Sxf.flatten(order = 'F'),0,M,M);
    Sxbi=sp.spdiags(1/Sxb.flatten(order = 'F'),0,M,M);
    Syfi=sp.spdiags(1/Syf.flatten(order = 'F'),0,M,M);
    Sybi=sp.spdiags(1/Syb.flatten(order = 'F'),0,M,M);
    Szfi=sp.spdiags(1/Szf.flatten(order = 'F'),0,M,M);
    Szbi=sp.spdiags(1/Szb.flatten(order = 'F'),0,M,M);
    
    return Sxfi, Sxbi, Syfi, Sybi, Szfi, Szbi

def S_create_3D_gpu(omega, dL, N, Npml, epsilon_0, eta_0):
    if not GPU_AVAILABLE:
        raise ImportError("CuPy is not available. Cannot use GPU acceleration.")
        
    dx, dy, dz = dL
    Nx, Ny, Nz = N
    M = int(cp.prod(cp.array(N)).item())
    
    Sxf, Syf, Szf, Sxb, Syb, Szb = create_sc_pml_gpu(omega, dL, N, Npml, epsilon_0, eta_0)
    
    # Create sparse diagonal matrices on GPU
    Sxfi = cp_spdiags(1/Sxf.flatten(order='F'), 0, M, M)
    Sxbi = cp_spdiags(1/Sxb.flatten(order='F'), 0, M, M)
    Syfi = cp_spdiags(1/Syf.flatten(order='F'), 0, M, M)
    Sybi = cp_spdiags(1/Syb.flatten(order='F'), 0, M, M)
    Szfi = cp_spdiags(1/Szf.flatten(order='F'), 0, M, M)
    Szbi = cp_spdiags(1/Szb.flatten(order='F'), 0, M, M)
    
    return Sxfi, Sxbi, Syfi, Sybi, Szfi, Szbi



                  
    
    