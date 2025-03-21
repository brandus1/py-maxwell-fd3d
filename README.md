# py-maxwell-fd3d
Solving Maxwell's equations via A python implementation of the 3D curl-curl E-field equations. The primary purpose of this code is to expose the underlying techniques for generating finite differences in a relatively transparent way (so no classes or complicated interfaces). This code contains additional work to engineer the eignspectrum for better convergence with iterative solvers (using the Beltrami-Laplace operator). You can control this in the main function through the input parameter $s = {0,-1,1}$

There is also a preconditioners to render the system matrix symmetric.

## Single Function Entry Point
The only function one really has to worry about is the one in fd3d.py This allows you to generate the matrix A and source vector b. Beyond that point, it is up to you to decide how to solve the linear system. There are some examples using scipy.sparse's qmr and bicg-stab in the notebooks but more likely than not, faster implementations exist elsewhere so it is important to have access to the underlying system matrix and right hand side. 

# important notes about implementation
1. Note that arrays are ordered using column-major (or Fortan) ordering whereas numpy is natively row-major or C ordering. You will see this in operations like reshape where I specify ordering (x.reshape(ordering = 'F')). It will also appear in meshgrid operations (use indexing = 'ij'). 

2. Symmetric Systems: There is a preconditioner to make the linear system matrix symmetric. This is advantageous both for iterative AND direct methods

# Numerical Solution to the Linear System
Solving the 3D linear system of the curl-curl equation is not easy. 


## Iterative Solvers
Unfortunately, given that iterative solvers don't have the same kind of robustness as factorization (i.e. iterative solvers need to converge which isn't always gauranteed) combined with the fact that FDFD for Maxwell's equations are typically indefinite, iterative solving of equations is a bit more of an art than not. For different systems, solvers may converge reasonably or may not. 

For now, the solvers I've tried in scipy's sparse.linalg library are QMR and BICG-STAB and LGMRES. BICG-STAB and QMR are usually your go-to solvers but I've noticed some cases where LGMRES performs better.

External solvers include packages like petsc or suitesparse (but I'm still looking for good python interfaces for any external solvers).

## Direct Solvers
Direct solvers are robust but are incredibly memory inefficient, particulary for the curl-curl equations in 3D. If you want to experiment with solvers, try packages which support a bunch-kauffman factorization for a complex symmetric matrix (reduces memory by 50%) and also use block low rank compression (i.e. MUMPS). Note that existing python interfaces to MUMPS are incomplete, they only support real valued matrices, so finding a way to use these might require you to do some digging or exporting the system matrix for use in an external solver.

As a general note, for a reordering like nested dissection, we now that the fill-in scales as around O(n^(4/3)). So, if you want to simulate a 200x100x100 grid, that's around 6 million DOF and the fill-in will be on the order of 1 billion nonzeros. Compare that with 2D, where nested dissection only fill in as nlog(n).

## General issues with using scipy.sparse
In general, scipy's sparse solvers are not ideal in terms of computational efficiency at tackling large 3D problems

1. So far, it appears that using scipy's iterative solvers, the case of the finite width photonic crystal slab has some issues with converging, even with the beltrami-laplace operator (s=-1). scipy's lgmres and gcrotmk seems to work better, but are a lot slower than bicgstab or qmr. Note that $s=-1$ is useful in that it helps convert a lot of cases from completely non-converging to converging, but the convergence may still be slow.

3. Not easy to implement modified versions of ILU preconditioning with scipy.sparse solvers, particularly block preconditioning.

## Proposed external package: Petsc and petsc4py


# Examples

1. Dipole in Vacuum (vacuum.ipynb): a radiating dipole(s) (there are two, one in x and y) in vacuum with the domain truncated by a PML.

2. Plane Wave (plane-wave test, unidirectional plane wave source)

3. Photonic Crystal Slab: lgmres

4. 3D waveguide (with a point dipole source)

![Alt text](./img/vacuum_slices.png?raw=true "Title")

![Alt text](./img/phc_slab_slices.png?raw=true "Title")

![Alt text](./img/cylindrical_waveguide_Ex.png?raw=true "Title")

![Alt text](./img/3d_waveguide_abs_slice.png?raw=true "Title")



### Recommended Visualization in 3D: Plotly
see some of the examples below

# Modal Sources
A nice way to verify any mode-solver in FDFD is to check whether or not it actually excites the pure mode when used as a source in an FDFD simulation.

For now, note that my other set of codes, eigenwell has a number of mode solvers. The type of mode solver you want assumes a $k_z$ out of plane and then allows 2 dimensional variation in the other two directions (Note that the TE-TM mode solvers do not apply for the 3D waveguide case!)


# Nonuniform Grid
We implement a simple continuously graded nonuniform grid. This can be helpful with higher-index structures and resolving intrinsic field discontinuities across interfaces, which is inevitable in 3D structures.

# GPU Acceleration

The code now includes GPU acceleration using CuPy, which can significantly speed up calculations for large problems. GPU support is implemented for:

1. Core finite difference operations
2. PML construction
3. Linear system assembly
4. Linear system solution

## Requirements

To use GPU acceleration, you need:
- CUDA-capable NVIDIA GPU
- CuPy installed (`pip install cupy-cuda11x` - replace with appropriate version for your CUDA installation)

## Using GPU Acceleration

The GPU-accelerated versions are implemented as separate functions alongside the CPU implementations:

- `createDws_gpu()` - GPU-accelerated finite difference operators
- `S_create_3D_gpu()` - GPU-accelerated PML construction
- `curlcurlE_gpu()` - Main GPU-accelerated function for assembling and optionally solving the system

You can run the example GPU script to compare performance:

```bash
python fdfd_gpu.py
```

This script compares CPU vs GPU performance and shows the speedup. For large problems, you can expect significant performance improvements, especially for the solve phase.

## Performance Considerations

- For small problems, the overhead of transferring data to and from the GPU may outweigh the benefits
- For large problems (> 100,000 voxels), GPU acceleration can provide 5-20x faster execution
- The solve_system parameter in `curlcurlE_gpu()` lets you choose whether to solve on the GPU or CPU

## Example Usage

```python
from pyfd3d.fd3d import curlcurlE_gpu

# Assemble and solve on GPU
A, b, Ch, E_solution = curlcurlE_gpu(
    # ... standard parameters ...
    solve_system=True,
    tol=1e-8,
    max_iter=2000
)

# Or just assemble on GPU and solve later
A, b, Ch = curlcurlE_gpu(
    # ... standard parameters ...
    solve_system=False
)
```
