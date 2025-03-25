import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
from numba import jit

# Simulation Parameters (unchanged)
N_x, N_y, N_z = 40, 40, 40
h = 0.1
dt = 0.05
c = 1.0
eps0 = 1.0
mu0 = 1.0

# CFL Check (unchanged)
cfl = (c * dt / h) * np.sqrt(3)
print(f"CFL Number: {cfl:.4f}")
if cfl <= 1.0:
    print("CFL condition is respected (CFL <= 1). Simulation should be stable.")
else:
    print("WARNING: CFL condition is violated (CFL > 1). Simulation may be unstable.")

# Fine grid parameters (2:1 ratio)
h_f = h / 2                 # Fine spatial step
dt_f = dt / 2               # Fine time step
N_xf, N_yf, N_zf = 10, 10, 10  # Fine grid sizes (covers x=0.5 to 1.0, etc.)
x_f_start, y_f_start, z_f_start = 5, 5, 5  # Coarse indices where fine grid begins
x_f_end = x_f_start + N_xf
y_f_end = y_f_start + N_yf
z_f_end = z_f_start + N_zf

# Coarse Field Arrays
E_x = np.zeros((N_x, N_y + 1, N_z + 1))
E_y = np.zeros((N_x + 1, N_y, N_z + 1))
E_z = np.zeros((N_x + 1, N_y + 1, N_z))
H_x = np.zeros((N_x + 1, N_y, N_z))
H_y = np.zeros((N_x, N_y + 1, N_z))
H_z = np.zeros((N_x, N_y, N_z + 1))

# Fine Field Arrays
E_x_f = np.zeros((N_xf, N_yf + 1, N_zf + 1))
E_y_f = np.zeros((N_xf + 1, N_yf, N_zf + 1))
E_z_f = np.zeros((N_xf + 1, N_yf + 1, N_zf))
H_x_f = np.zeros((N_xf + 1, N_yf, N_zf))
H_y_f = np.zeros((N_xf, N_yf + 1, N_zf))
H_z_f = np.zeros((N_xf, N_yf, N_zf + 1))

# PML Parameters for Coarse Grid
Npml = 8  # PML thickness (small for demonstration; typically 8-10 for better absorption)
m = 3     # Polynomial grading order
eta = np.sqrt(mu0 / eps0)  # Impedance (1.0 in this normalized system)
sigma_max = (0.8 * (m + 1)) / (eta * h)  # Maximum conductivity

# Define sigma arrays
sigma_x = np.zeros(N_x + 1)
for i in range(Npml):
    sigma_x[i] = sigma_max * ((Npml - i - 0.5) / Npml) ** m
for i in range(N_x - Npml + 1, N_x + 1):
    sigma_x[i] = sigma_max * ((i - (N_x - Npml) + 0.5) / Npml) ** m

sigma_y = np.zeros(N_y + 1)
for j in range(Npml):
    sigma_y[j] = sigma_max * ((Npml - j - 0.5) / Npml) ** m
for j in range(N_y - Npml + 1, N_y + 1):
    sigma_y[j] = sigma_max * ((j - (N_y - Npml) + 0.5) / Npml) ** m

sigma_z = np.zeros(N_z + 1)
for k in range(Npml):
    sigma_z[k] = sigma_max * ((Npml - k - 0.5) / Npml) ** m
for k in range(N_z - Npml + 1, N_z + 1):
    sigma_z[k] = sigma_max * ((k - (N_z - Npml) + 0.5) / Npml) ** m

# CPML coefficients
b_x = np.exp(-sigma_x * dt / eps0)
a_x = b_x - 1  # Simplified for kappa=1, alpha=0
b_y = np.exp(-sigma_y * dt / eps0)
a_y = b_y - 1
b_z = np.exp(-sigma_z * dt / eps0)
a_z = b_z - 1

# PML Psi Variables for Coarse Grid
psi_E_x_y = np.zeros((N_x, N_y + 1, N_z + 1))
psi_E_x_z = np.zeros((N_x, N_y + 1, N_z + 1))
psi_E_y_x = np.zeros((N_x + 1, N_y, N_z + 1))
psi_E_y_z = np.zeros((N_x + 1, N_y, N_z + 1))
psi_E_z_x = np.zeros((N_x + 1, N_y + 1, N_z))
psi_E_z_y = np.zeros((N_x + 1, N_y + 1, N_z))

psi_H_x_y = np.zeros((N_x + 1, N_y, N_z))
psi_H_x_z = np.zeros((N_x + 1, N_y, N_z))
psi_H_y_x = np.zeros((N_x, N_y + 1, N_z))
psi_H_y_z = np.zeros((N_x, N_y + 1, N_z))
psi_H_z_x = np.zeros((N_x, N_y, N_z + 1))
psi_H_z_y = np.zeros((N_x, N_y, N_z + 1))

# Initial Condition: Gaussian pulse in coarse E_z
i_center, j_center, k_center = N_x // 2, N_y // 2, N_z // 2
E_z[i_center, j_center, k_center] = 1.0

# Update Functions

@jit(nopython=True)
def update_E_fields_standard(E_x, E_y, E_z, H_x, H_y, H_z, h, dt, Nx, Ny, Nz):
    dt_eps0 = dt / eps0
    h_inv = 1.0 / h
    for i in range(Nx):
        for j in range(1, Ny):
            for k in range(1, Nz):
                dHz_dy = (H_z[i, j, k] - H_z[i, j - 1, k]) * h_inv
                dHy_dz = (H_y[i, j, k] - H_y[i, j, k - 1]) * h_inv
                E_x[i, j, k] += dt_eps0 * (dHz_dy - dHy_dz)
    for i in range(1, Nx):
        for j in range(Ny):
            for k in range(1, Nz):
                dHx_dz = (H_x[i, j, k] - H_x[i, j, k - 1]) * h_inv
                dHz_dx = (H_z[i, j, k] - H_z[i - 1, j, k]) * h_inv
                E_y[i, j, k] += dt_eps0 * (dHx_dz - dHz_dx)
    for i in range(1, Nx):
        for j in range(1, Ny):
            for k in range(Nz):
                dHy_dx = (H_y[i, j, k] - H_y[i - 1, j, k]) * h_inv
                dHx_dy = (H_x[i, j, k] - H_x[i, j - 1, k]) * h_inv
                E_z[i, j, k] += dt_eps0 * (dHy_dx - dHx_dy)

@jit(nopython=True)
def update_H_fields_standard(E_x, E_y, E_z, H_x, H_y, H_z, h, dt, Nx, Ny, Nz):
    dt_mu0 = dt / mu0
    h_inv = 1.0 / h
    for i in range(Nx + 1):
        for j in range(Ny):
            for k in range(Nz):
                dEy_dz = (E_y[i, j, k + 1] - E_y[i, j, k]) * h_inv
                dEz_dy = (E_z[i, j + 1, k] - E_z[i, j, k]) * h_inv
                H_x[i, j, k] += dt_mu0 * (dEy_dz - dEz_dy)
    for i in range(Nx):
        for j in range(Ny + 1):
            for k in range(Nz):
                dEz_dx = (E_z[i + 1, j, k] - E_z[i, j, k]) * h_inv
                dEx_dz = (E_x[i, j, k + 1] - E_x[i, j, k]) * h_inv
                H_y[i, j, k] += dt_mu0 * (dEz_dx - dEx_dz)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz + 1):
                dEx_dy = (E_x[i, j + 1, k] - E_x[i, j, k]) * h_inv
                dEy_dx = (E_y[i + 1, j, k] - E_y[i, j, k]) * h_inv
                H_z[i, j, k] += dt_mu0 * (dEx_dy - dEy_dx)

@jit(nopython=True)
def update_E_fields_PML(E_x, E_y, E_z, H_x, H_y, H_z, h, dt, psi_E_x_y, psi_E_x_z, psi_E_y_x, psi_E_y_z, psi_E_z_x, psi_E_z_y, b_x, b_y, b_z, a_x, a_y, a_z, Nx, Ny, Nz):
    dt_eps0 = dt / eps0
    h_inv = 1.0 / h
    for i in range(Nx):
        for j in range(1, Ny):
            for k in range(1, Nz):
                dHz_dy = (H_z[i, j, k] - H_z[i, j - 1, k]) * h_inv
                dHy_dz = (H_y[i, j, k] - H_y[i, j, k - 1]) * h_inv
                psi_E_x_y[i, j, k] = b_y[j] * psi_E_x_y[i, j, k] + a_y[j] * dHz_dy
                psi_E_x_z[i, j, k] = b_z[k] * psi_E_x_z[i, j, k] + a_z[k] * dHy_dz
                E_x[i, j, k] += dt_eps0 * (dHz_dy - dHy_dz + psi_E_x_y[i, j, k] - psi_E_x_z[i, j, k])
    for i in range(1, Nx):
        for j in range(Ny):
            for k in range(1, Nz):
                dHx_dz = (H_x[i, j, k] - H_x[i, j, k - 1]) * h_inv
                dHz_dx = (H_z[i, j, k] - H_z[i - 1, j, k]) * h_inv
                psi_E_y_z[i, j, k] = b_z[k] * psi_E_y_z[i, j, k] + a_z[k] * dHx_dz
                psi_E_y_x[i, j, k] = b_x[i] * psi_E_y_x[i, j, k] + a_x[i] * dHz_dx
                E_y[i, j, k] += dt_eps0 * (dHx_dz - dHz_dx + psi_E_y_z[i, j, k] - psi_E_y_x[i, j, k])
    for i in range(1, Nx):
        for j in range(1, Ny):
            for k in range(Nz):
                dHy_dx = (H_y[i, j, k] - H_y[i - 1, j, k]) * h_inv
                dHx_dy = (H_x[i, j, k] - H_x[i, j - 1, k]) * h_inv
                psi_E_z_x[i, j, k] = b_x[i] * psi_E_z_x[i, j, k] + a_x[i] * dHy_dx
                psi_E_z_y[i, j, k] = b_y[j] * psi_E_z_y[i, j, k] + a_y[j] * dHx_dy
                E_z[i, j, k] += dt_eps0 * (dHy_dx - dHx_dy + psi_E_z_x[i, j, k] - psi_E_z_y[i, j, k])

@jit(nopython=True)
def update_H_fields_PML(E_x, E_y, E_z, H_x, H_y, H_z, h, dt, psi_H_x_y, psi_H_x_z, psi_H_y_x, psi_H_y_z, psi_H_z_x, psi_H_z_y, b_x, b_y, b_z, a_x, a_y, a_z, Nx, Ny, Nz):
    dt_mu0 = dt / mu0
    h_inv = 1.0 / h
    for i in range(Nx + 1):
        for j in range(Ny):
            for k in range(Nz):
                dEy_dz = (E_y[i, j, k + 1] - E_y[i, j, k]) * h_inv
                dEz_dy = (E_z[i, j + 1, k] - E_z[i, j, k]) * h_inv
                psi_H_x_z[i, j, k] = b_z[k] * psi_H_x_z[i, j, k] + a_z[k] * dEy_dz
                psi_H_x_y[i, j, k] = b_y[j] * psi_H_x_y[i, j, k] + a_y[j] * dEz_dy
                H_x[i, j, k] += dt_mu0 * (dEy_dz - dEz_dy + psi_H_x_z[i, j, k] - psi_H_x_y[i, j, k])
    for i in range(Nx):
        for j in range(Ny + 1):
            for k in range(Nz):
                dEz_dx = (E_z[i + 1, j, k] - E_z[i, j, k]) * h_inv
                dEx_dz = (E_x[i, j, k + 1] - E_x[i, j, k]) * h_inv
                psi_H_y_x[i, j, k] = b_x[i] * psi_H_y_x[i, j, k] + a_x[i] * dEz_dx
                psi_H_y_z[i, j, k] = b_z[k] * psi_H_y_z[i, j, k] + a_z[k] * dEx_dz
                H_y[i, j, k] += dt_mu0 * (dEz_dx - dEx_dz + psi_H_y_x[i, j, k] - psi_H_y_z[i, j, k])
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz + 1):
                dEx_dy = (E_x[i, j + 1, k] - E_x[i, j, k]) * h_inv
                dEy_dx = (E_y[i + 1, j, k] - E_y[i, j, k]) * h_inv
                psi_H_z_y[i, j, k] = b_y[j] * psi_H_z_y[i, j, k] + a_y[j] * dEx_dy
                psi_H_z_x[i, j, k] = b_x[i] * psi_H_z_x[i, j, k] + a_x[i] * dEy_dx
                H_z[i, j, k] += dt_mu0 * (dEx_dy - dEy_dx + psi_H_z_y[i, j, k] - psi_H_z_x[i, j, k])

# Interpolation Function (linear, coarse to fine)
def interpolate_coarse_to_fine(coarse_field, start_idx, N_f, axis):
    fine_field = np.zeros(N_f)
    for i in range(N_f):
        coarse_i = start_idx + i // 2
        frac = (i % 2) * 0.5
        if coarse_i + 1 < coarse_field.shape[axis]:
            fine_field[i] = (1 - frac) * coarse_field[coarse_i] + frac * coarse_field[coarse_i + 1]
        else:
            fine_field[i] = coarse_field[coarse_i]
    return fine_field

# Interface Update for All Six Faces
def update_interface():
    sigma = 1 / h  # Penalty parameter for SAT terms
    
    # Interpolate coarse to fine for all six faces
    # Left (x=0.5)
    for j in range(N_yf + 1):
        for k in range(N_zf):
            E_z_f[0, j, k] = interpolate_coarse_to_fine(E_z[x_f_start, j + y_f_start, :], z_f_start + k // 2, 1, 0)[0]
    # Right (x=1.0)
    for j in range(N_yf + 1):
        for k in range(N_zf):
            E_z_f[N_xf, j, k] = interpolate_coarse_to_fine(E_z[x_f_end, j + y_f_start, :], z_f_start + k // 2, 1, 0)[0]
    # Front (y=0.5)
    for i in range(N_xf + 1):
        for k in range(N_zf):
            E_z_f[i, 0, k] = interpolate_coarse_to_fine(E_z[i + x_f_start, y_f_start, :], z_f_start + k // 2, 1, 0)[0]
    # Back (y=1.0)
    for i in range(N_xf + 1):
        for k in range(N_zf):
            E_z_f[i, N_yf, k] = interpolate_coarse_to_fine(E_z[i + x_f_start, y_f_end, :], z_f_start + k // 2, 1, 0)[0]
    # Bottom (z=0.5)
    for i in range(N_xf + 1):
        for j in range(N_yf + 1):
            E_z_f[i, j, 0] = interpolate_coarse_to_fine(E_z[i + x_f_start, j + y_f_start, :], z_f_start, 1, 0)[0]
    # Top (z=1.0)
    for i in range(N_xf + 1):
        for j in range(N_yf + 1):
            E_z_f[i, j, N_zf-1] = interpolate_coarse_to_fine(E_z[i + x_f_start, j + y_f_start, :], z_f_end, 1, 0)[0]

    # SAT terms for continuity at all interfaces
    # Left (x=0.5)
    H_y[x_f_start, y_f_start:y_f_start + N_yf, z_f_start:z_f_start + N_zf] -= dt * sigma * (
        E_z[x_f_start, y_f_start:y_f_start + N_yf, z_f_start:z_f_start + N_zf] - E_z_f[0, :N_yf, :N_zf]
    )
    # Right (x=1.0)
    H_y[x_f_end - 1, y_f_start:y_f_start + N_yf, z_f_start:z_f_start + N_zf] += dt * sigma * (
        E_z[x_f_end, y_f_start:y_f_start + N_yf, z_f_start:z_f_start + N_zf] - E_z_f[N_xf, :N_yf, :N_zf]
    )
    # Front (y=0.5)
    H_z[x_f_start:x_f_start + N_xf, y_f_start, z_f_start:z_f_start + N_zf] -= dt * sigma * (
        E_x[x_f_start:x_f_start + N_xf, y_f_start, z_f_start:z_f_start + N_zf] - E_x_f[:N_xf, 0, :N_zf]
    )
    # Back (y=1.0)
    H_z[x_f_start:x_f_start + N_xf, y_f_end - 1, z_f_start:z_f_start + N_zf] += dt * sigma * (
        E_x[x_f_start:x_f_start + N_xf, y_f_end, z_f_start:z_f_start + N_zf] - E_x_f[:N_xf, N_yf, :N_zf]
    )
    # Bottom (z=0.5)
    H_x[x_f_start:x_f_start + N_xf, y_f_start:y_f_start + N_yf, z_f_start] -= dt * sigma * (
        E_y[x_f_start:x_f_start + N_xf, y_f_start:y_f_start + N_yf, z_f_start] - E_y_f[:N_xf, :N_yf, 0]
    )
    # Top (z=1.0)
    H_x[x_f_start:x_f_start + N_xf, y_f_start:y_f_start + N_yf, z_f_end - 1] += dt * sigma * (
        E_y[x_f_start:x_f_start + N_xf, y_f_start:y_f_start + N_yf, z_f_end] - E_y_f[:N_xf, :N_yf, N_zf-1]
    )

# Simulation Loop
n_steps = 5000
E_z_history = []

for n in tqdm(range(n_steps), desc="Simulation Progress"):
    # Update coarse grid with PML
    update_E_fields_PML(E_x, E_y, E_z, H_x, H_y, H_z, h, dt,
                        psi_E_x_y, psi_E_x_z, psi_E_y_x, psi_E_y_z, psi_E_z_x, psi_E_z_y,
                        b_x, b_y, b_z, a_x, a_y, a_z, N_x, N_y, N_z)
    update_H_fields_PML(E_x, E_y, E_z, H_x, H_y, H_z, h, dt,
                        psi_H_x_y, psi_H_x_z, psi_H_y_x, psi_H_y_z, psi_H_z_x, psi_H_z_y,
                        b_x, b_y, b_z, a_x, a_y, a_z, N_x, N_y, N_z)
    
    # Update fine grid (two steps due to 2:1 time step ratio)
    for _ in range(2):
        update_E_fields_standard(E_x_f, E_y_f, E_z_f, H_x_f, H_y_f, H_z_f, h_f, dt_f, N_xf, N_yf, N_zf)
        update_H_fields_standard(E_x_f, E_y_f, E_z_f, H_x_f, H_y_f, H_z_f, h_f, dt_f, N_xf, N_yf, N_zf)
    
    # Interface coupling for all six faces
    update_interface()
    
    # Store for visualization
    E_z_slice = E_z[N_x // 2, :, :].copy()
    E_z_history.append(E_z_slice[:-1, :])

# Animation
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(E_z_history[0], cmap='RdBu', vmin=-0.005, vmax=0.005, interpolation='bicubic', aspect='equal')
ax.set_title(f'E_z Field Progression (x = {N_x//2})')
ax.set_xlabel('y')
ax.set_ylabel('z')
fig.colorbar(im, label='E_z (V/m)')

def animate(n):
    im.set_array(E_z_history[n])
    ax.set_title(f'E_z Field Progression (x = {N_x//2}), Step {n}')
    return [im]

anim = animation.FuncAnimation(fig, animate, frames=n_steps, interval=100, blit=True)
plt.show()