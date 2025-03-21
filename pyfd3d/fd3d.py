import numpy as np
import scipy.sparse as sp
from .derivatives import *
from .pml import *
from typing import *
from .utils import *
from .nonuniform_grid import *

try:
    import cupy as cp
    from cupyx.scipy.sparse import bmat as cp_bmat
    from cupyx.scipy.sparse import identity as cp_identity
    from cupyx.scipy.sparse import block_diag as cp_block_diag
    from cupyx.scipy.sparse import spdiags as cp_spdiags
    from cupyx.scipy.sparse.linalg import bicgstab as cp_bicgstab
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def curlcurlE(
    L0: float, #L0 scaling parameter for distance units, usually 1e-6
    wvlen: float, # wvlen in units of L0
    xrange: np.array, #(xmin, xmax) in units of L0
    yrange: np.array, #(xmin, xmax) in units of L0
    zrange: np.array, #(xmin, xmax) in units of L0
    eps_r_tensor_dict: dict, 
    JCurrentVector: dict,
    Npml, 
    s = -1,
    nonuniform = None, #dictionary containing all scaling operators for the forward and backward difference ops
):
    
    # normal SI parameters
    eps0 = 8.85*10**-12*L0;
    mu0 = 4*np.pi*10**-7*L0; 
    eta0 = np.sqrt(mu0/eps0);
    c0 = 1/np.sqrt(eps0*mu0);  # speed of light in 
    omega = 2*np.pi*c0/(wvlen);  # angular frequency in rad/sec
    
    eps_xx = eps_r_tensor_dict['eps_xx']
    eps_yy = eps_r_tensor_dict['eps_yy']
    eps_zz = eps_r_tensor_dict['eps_zz']
    
    ## edge smoothing of eps_xx, eps_yy, eps_zz
    eps_xx = bwdmean(eps_xx, 'x')
    eps_yy = bwdmean(eps_yy, 'y')
    eps_zz = bwdmean(eps_zz, 'z')
    
     
    N = eps_xx.shape;
    M = np.prod(N);
    
    L = np.array([np.diff(xrange)[0], np.diff(yrange)[0], np.diff(zrange)[0]]);
    dL = L/N
    
    Tepz = sp.spdiags(eps0*eps_zz.flatten(order = 'F'), 0, M,M);
    Tepx = sp.spdiags(eps0*eps_xx.flatten(order = 'F'), 0, M,M);
    Tepy = sp.spdiags(eps0*eps_yy.flatten(order = 'F'), 0, M,M);

    iTepz = sp.spdiags(1/(eps0*eps_zz.flatten(order = 'F')), 0, M,M);
    iTepx = sp.spdiags(1/(eps0*eps_xx.flatten(order = 'F')), 0, M,M);
    iTepy = sp.spdiags(1/(eps0*eps_yy.flatten(order = 'F')), 0, M,M);
    
    iTepsSuper = sp.block_diag((iTepx, iTepy, iTepz));
    TepsSuper = sp.block_diag((Tepx, Tepy, Tepz));

    iTmuSuper = (1/mu0)*sp.identity(3*M)
    TmuSuper = (mu0)*sp.identity(3*M)
    
    ## generate PML parameters
    # Sxf = sp.identity(3*M);
    Sxfi, Sxbi, Syfi, Sybi, Szfi, Szbi = S_create_3D(omega, dL, N, Npml, eps0, eta0) #sp.identity(M);
    
    ## CREATE DERIVATIVES
    Dxf = Sxfi@createDws('x', 'f', dL, N); 
    Dxb = Sxbi@createDws('x', 'b', dL, N); 

    Dyf = Syfi@createDws('y', 'f', dL, N);
    Dyb = Sybi@createDws('y', 'b', dL, N); 
    
    Dzf = Szfi@createDws('z', 'f', dL, N); 
    Dzb = Szbi@createDws('z', 'b', dL, N); 
    
    
    #curlE and curlH
    Ce = sp.bmat([[None, -Dzf, Dyf], 
                  [Dzf, None, -Dxf], 
                  [-Dyf, Dxf, None]])
    Ch = sp.bmat([[None, -Dzb, Dyb], 
                  [Dzb, None, -Dxb], 
                  [-Dyb, Dxb, None]])
    
    ##graddiv, aka beltrami-laplace from Wonseok's paper
    gd00 = Dxf@iTepx@Dxb@Tepx
    gd01 = Dxf@iTepx@(Dyb@Tepy)
    gd02 = Dxf@iTepx@(Dzb@Tepz)
    
    gd10 = Dyf@iTepy@(Dxb@Tepx)
    gd11 = Dyf@iTepy@(Dyb@Tepy)
    gd12 = Dyf@iTepy@(Dzb@Tepz)
    
    gd20 = Dzf@iTepz@(Dxb@Tepx)
    gd21 = Dzf@iTepz@(Dyb@Tepy)
    gd22 = Dzf@iTepz@(Dzb@Tepz)

    GradDiv = sp.bmat([[gd00, gd01, gd02],
                       [gd10, gd11, gd12],
                       [gd20, gd21, gd22]]);
    
    WAccelScal = sp.identity(3*M)@iTmuSuper;
    A = Ch@iTmuSuper@Ce + s*WAccelScal@GradDiv - omega**2*TepsSuper;
    
    ## source setup
    Jx = JCurrentVector['Jx'].flatten(order = 'F')
    Jy = JCurrentVector['Jy'].flatten(order = 'F')
    Jz = JCurrentVector['Jz'].flatten(order = 'F')
    
    J = np.concatenate((Jx,Jy,Jz), axis = 0);
    print(J.shape)
    
    b = -1j*omega*J; 
    JCorrection = (1j/omega) * (s*GradDiv@WAccelScal)@iTepsSuper@J;
    b = b+JCorrection;
    
    return A,b, Ch # last arg let's you recover H fields

def curlcurlE_gpu(
    L0: float, #L0 scaling parameter for distance units, usually 1e-6
    wvlen: float, # wvlen in units of L0
    xrange: np.array, #(xmin, xmax) in units of L0
    yrange: np.array, #(xmin, xmax) in units of L0
    zrange: np.array, #(xmin, xmax) in units of L0
    eps_r_tensor_dict: dict, 
    JCurrentVector: dict,
    Npml, 
    s = -1,
    nonuniform = None, #dictionary containing all scaling operators for the forward and backward difference ops
    solve_system = False, # Whether to solve the system on GPU
    tol = 1e-8, # Tolerance for the iterative solver if solve_system=True
    max_iter = 2000, # Max iterations for the solver if solve_system=True
):
    if not GPU_AVAILABLE:
        raise ImportError("CuPy is not available. Cannot use GPU acceleration.")
    
    # normal SI parameters
    eps0 = 8.85*10**-12*L0
    mu0 = 4*np.pi*10**-7*L0
    eta0 = np.sqrt(mu0/eps0)
    c0 = 1/np.sqrt(eps0*mu0)  # speed of light in 
    omega = 2*np.pi*c0/(wvlen)  # angular frequency in rad/sec
    
    # Copy permittivity tensors to GPU
    eps_xx = cp.asarray(eps_r_tensor_dict['eps_xx'])
    eps_yy = cp.asarray(eps_r_tensor_dict['eps_yy'])
    eps_zz = cp.asarray(eps_r_tensor_dict['eps_zz'])
    
    ## edge smoothing of eps_xx, eps_yy, eps_zz (implement bwdmean for GPU)
    def bwdmean_gpu(eps, axis):
        if axis == 'x':
            return 0.5 * (eps + cp.roll(eps, 1, axis=0))
        elif axis == 'y':
            return 0.5 * (eps + cp.roll(eps, 1, axis=1))
        elif axis == 'z':
            return 0.5 * (eps + cp.roll(eps, 1, axis=2))
    
    eps_xx = bwdmean_gpu(eps_xx, 'x')
    eps_yy = bwdmean_gpu(eps_yy, 'y')
    eps_zz = bwdmean_gpu(eps_zz, 'z')
    
    N = eps_xx.shape
    M = int(cp.prod(cp.array(N)).item())
    
    # Calculate domain sizes and discretization
    L = cp.array([cp.diff(cp.asarray(xrange))[0], cp.diff(cp.asarray(yrange))[0], cp.diff(cp.asarray(zrange))[0]])
    dL = L/cp.array(N)
    
    # Create permittivity and inverse permittivity diagonal matrices
    Tepz = cp_spdiags(eps0*eps_zz.flatten(order='F'), 0, M, M)
    Tepx = cp_spdiags(eps0*eps_xx.flatten(order='F'), 0, M, M)
    Tepy = cp_spdiags(eps0*eps_yy.flatten(order='F'), 0, M, M)

    iTepz = cp_spdiags(1/(eps0*eps_zz.flatten(order='F')), 0, M, M)
    iTepx = cp_spdiags(1/(eps0*eps_xx.flatten(order='F')), 0, M, M)
    iTepy = cp_spdiags(1/(eps0*eps_yy.flatten(order='F')), 0, M, M)
    
    iTepsSuper = cp_block_diag((iTepx, iTepy, iTepz))
    TepsSuper = cp_block_diag((Tepx, Tepy, Tepz))

    iTmuSuper = (1/mu0)*cp_identity(3*M)
    TmuSuper = (mu0)*cp_identity(3*M)
    
    ## generate PML parameters using GPU implementation
    Sxfi, Sxbi, Syfi, Sybi, Szfi, Szbi = S_create_3D_gpu(omega, dL.get(), N, Npml, eps0, eta0)
    
    ## CREATE DERIVATIVES using GPU implementation
    Dxf = Sxfi@createDws_gpu('x', 'f', dL.get(), N)
    Dxb = Sxbi@createDws_gpu('x', 'b', dL.get(), N)

    Dyf = Syfi@createDws_gpu('y', 'f', dL.get(), N)
    Dyb = Sybi@createDws_gpu('y', 'b', dL.get(), N)
    
    Dzf = Szfi@createDws_gpu('z', 'f', dL.get(), N)
    Dzb = Szbi@createDws_gpu('z', 'b', dL.get(), N)
    
    # curl operators
    Ce = cp_bmat([[None, -Dzf, Dyf], 
                  [Dzf, None, -Dxf], 
                  [-Dyf, Dxf, None]])
    Ch = cp_bmat([[None, -Dzb, Dyb], 
                  [Dzb, None, -Dxb], 
                  [-Dyb, Dxb, None]])
    
    # graddiv terms
    gd00 = Dxf@iTepx@Dxb@Tepx
    gd01 = Dxf@iTepx@(Dyb@Tepy)
    gd02 = Dxf@iTepx@(Dzb@Tepz)
    
    gd10 = Dyf@iTepy@(Dxb@Tepx)
    gd11 = Dyf@iTepy@(Dyb@Tepy)
    gd12 = Dyf@iTepy@(Dzb@Tepz)
    
    gd20 = Dzf@iTepz@(Dxb@Tepx)
    gd21 = Dzf@iTepz@(Dyb@Tepy)
    gd22 = Dzf@iTepz@(Dzb@Tepz)

    GradDiv = cp_bmat([[gd00, gd01, gd02],
                       [gd10, gd11, gd12],
                       [gd20, gd21, gd22]])
    
    WAccelScal = cp_identity(3*M)@iTmuSuper
    A = Ch@iTmuSuper@Ce + s*WAccelScal@GradDiv - omega**2*TepsSuper
    
    ## source setup - copy to GPU
    Jx_gpu = cp.asarray(JCurrentVector['Jx'].flatten(order='F'))
    Jy_gpu = cp.asarray(JCurrentVector['Jy'].flatten(order='F'))
    Jz_gpu = cp.asarray(JCurrentVector['Jz'].flatten(order='F'))
    
    J = cp.concatenate((Jx_gpu, Jy_gpu, Jz_gpu), axis=0)
    print(J.shape)
    
    b = -1j*omega*J
    JCorrection = (1j/omega) * (s*GradDiv@WAccelScal)@iTepsSuper@J
    b = b + JCorrection
    
    if solve_system:
        # Solve the system on GPU using CuPy's bicgstab
        # Create callback to track convergence
        residual_history = []
        
        def callback(x):
            r = b - A @ x
            r_norm = cp.linalg.norm(r).get()
            residual_history.append(r_norm)
            print(f"Iteration {len(residual_history)}: residual = {r_norm:.3e}")
        
        x0 = cp.zeros_like(b, dtype=cp.complex128)
        E_solution, info = cp_bicgstab(A, b, x0=x0, tol=tol, maxiter=max_iter, callback=callback)
        
        # Return the solution as numpy arrays (copying back from GPU)
        if info == 0:
            print("BiCGSTAB converged!")
        else:
            print(f"BiCGSTAB ended with info={info}")
            
        # Return numpy arrays for CPU processing
        return cp.asnumpy(A), cp.asnumpy(b), cp.asnumpy(Ch), E_solution.get()
    
    # Return the sparse matrices and source term
    return A, b, Ch


def curlcurlE_nu(
    L0: float, #L0 scaling parameter for distance units, usually 1e-6
    wvlen: float, # wvlen in units of L0
    dx: float,
    dy: float,
    dz: float,
    dx_scale,
    dy_scale,
    dz_scale,
    eps_r_tensor_dict: dict, 
    JCurrentVector: dict,
    Npml, 
    s = -1,
):
    
    # normal SI parameters
    eps0 = 8.85*10**-12*L0;
    mu0 = 4*np.pi*10**-7*L0; 
    eta0 = np.sqrt(mu0/eps0);
    c0 = 1/np.sqrt(eps0*mu0);  # speed of light in 
    omega = 2*np.pi*c0/(wvlen);  # angular frequency in rad/sec
    
    eps_xx = eps_r_tensor_dict['eps_xx']
    eps_yy = eps_r_tensor_dict['eps_yy']
    eps_zz = eps_r_tensor_dict['eps_zz']
    
    ## edge smoothing of eps_xx, eps_yy, eps_zz
    eps_xx = bwdmean(eps_xx, 'x')
    eps_yy = bwdmean(eps_yy, 'y')
    eps_zz = bwdmean(eps_zz, 'z')
    
    dL = np.array([dx, dy, dz])
    N = eps_xx.shape;
    M = np.prod(N);
    
    Tepz = sp.spdiags(eps0*eps_zz.flatten(order = 'F'), 0, M,M);
    Tepx = sp.spdiags(eps0*eps_xx.flatten(order = 'F'), 0, M,M);
    Tepy = sp.spdiags(eps0*eps_yy.flatten(order = 'F'), 0, M,M);

    iTepz = sp.spdiags(1/(eps0*eps_zz.flatten(order = 'F')), 0, M,M);
    iTepx = sp.spdiags(1/(eps0*eps_xx.flatten(order = 'F')), 0, M,M);
    iTepy = sp.spdiags(1/(eps0*eps_yy.flatten(order = 'F')), 0, M,M);
    
    iTepsSuper = sp.block_diag((iTepx, iTepy, iTepz));
    TepsSuper = sp.block_diag((Tepx, Tepy, Tepz));

    iTmuSuper = (1/mu0)*sp.identity(3*M)
    TmuSuper = (mu0)*sp.identity(3*M)
    
    ## generate PML parameters
    # Sxf = sp.identity(3*M);
    Sxfi, Sxbi, Syfi, Sybi, Szfi, Szbi = S_create_3D(omega, dL, N, Npml, eps0, eta0) #sp.identity(M);
    
    ## non-uniform scaling
    dx_scale_0 = np.ones(N[0])
    dy_scale_0 = np.ones(N[1])
    dz_scale_0 = np.ones(N[2])
    
    ## 
    Fsxi, Fsyi, Fszi, Fsxi_conj, Fsyi_conj, Fszi_conj = non_uniform_scaling_operator(dx_scale, dy_scale, dz_scale)
    
    ## CREATE DERIVATIVES
    Dxf = Sxfi@Fsxi@createDws('x', 'f', dL, N)
    Dxb = Sxbi@Fsxi_conj@createDws('x', 'b', dL, N)

    Dyf = Syfi@Fsyi@createDws('y', 'f', dL, N)
    Dyb = Sybi@Fsyi_conj@createDws('y', 'b', dL, N)
    
    Dzf = Szfi@Fszi@createDws('z', 'f', dL, N)
    Dzb = Szbi@Fszi_conj@createDws('z', 'b', dL, N)
    
    
    #curlE and curlH
    Ce = sp.bmat([[None, -Dzf, Dyf], 
                  [Dzf, None, -Dxf], 
                  [-Dyf, Dxf, None]])
    Ch = sp.bmat([[None, -Dzb, Dyb], 
                  [Dzb, None, -Dxb], 
                  [-Dyb, Dxb, None]])
    
    ##graddiv, aka beltrami-laplace from Wonseok's paper
    gd00 = Dxf@iTepx@Dxb@Tepx
    gd01 = Dxf@iTepx@(Dyb@Tepy)
    gd02 = Dxf@iTepx@(Dzb@Tepz)
    
    gd10 = Dyf@iTepy@(Dxb@Tepx)
    gd11 = Dyf@iTepy@(Dyb@Tepy)
    gd12 = Dyf@iTepy@(Dzb@Tepz)
    
    gd20 = Dzf@iTepz@(Dxb@Tepx)
    gd21 = Dzf@iTepz@(Dyb@Tepy)
    gd22 = Dzf@iTepz@(Dzb@Tepz)

    GradDiv = sp.bmat([[gd00, gd01, gd02],
                       [gd10, gd11, gd12],
                       [gd20, gd21, gd22]]);
    
    WAccelScal = sp.identity(3*M)@iTmuSuper;
    A = Ch@iTmuSuper@Ce + s*WAccelScal@GradDiv - omega**2*TepsSuper;
    
    ## source setup
    Jx = JCurrentVector['Jx'].flatten(order = 'F')
    Jy = JCurrentVector['Jy'].flatten(order = 'F')
    Jz = JCurrentVector['Jz'].flatten(order = 'F')
    
    J = np.concatenate((Jx,Jy,Jz), axis = 0);
    
    b = -1j*omega*J; 
    JCorrection = (1j/omega) * (s*GradDiv@WAccelScal)@iTepsSuper@J;
    b = b+JCorrection;
    
    return A,b, Ch # last arg let's you recover H fields

    
