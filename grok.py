import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Simulation Parameters
N_x, N_y, N_z = 20, 20, 20  # Coarse grid sizes
h = 0.1                     # Coarse spatial step (meters)
dt = 0.05                   # Coarse time step (seconds)
c = 1.0                     # Speed of light
eps0 = 1.0                  # Electric permittivity
mu0 = 1.0                   # Magnetic permeability

# Fine grid parameters (2:1 ratio)
h_f = h / 2                 # Fine spatial step
dt_f = dt / 2               # Fine time step
N_xf, N_yf, N_zf = 10, 10, 10  # Fine grid sizes (covers x=0.5 to 1.0, etc.)
x_f_start, y_f_start, z_f_start = 5, 5, 5  # Coarse indices where fine grid begins

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

# Initial Condition: Gaussian pulse in coarse E_z
i_center, j_center, k_center = N_x // 2, N_y // 2, N_z // 2
E_z[i_center, j_center, k_center] = 1.0

# Update Functions (same as yours, adapted for grid)
def update_E_fields(grid, E_x, E_y, E_z, H_x, H_y, H_z, h, dt):
    Nx, Ny, Nz = grid
    for i in range(Nx):
        for j in range(1, Ny):
            for k in range(1, Nz):
                dHz_dy = (H_z[i, j, k] - H_z[i, j - 1, k]) / h
                dHy_dz = (H_y[i, j, k] - H_y[i, j, k - 1]) / h
                E_x[i, j, k] += (dt / eps0) * (dHz_dy - dHy_dz)
    for i in range(1, Nx):
        for j in range(Ny):
            for k in range(1, Nz):
                dHx_dz = (H_x[i, j, k] - H_x[i, j, k - 1]) / h
                dHz_dx = (H_z[i, j, k] - H_z[i - 1, j, k]) / h
                E_y[i, j, k] += (dt / eps0) * (dHx_dz - dHz_dx)
    for i in range(1, Nx):
        for j in range(1, Ny):
            for k in range(Nz):
                dHy_dx = (H_y[i, j, k] - H_y[i - 1, j, k]) / h
                dHx_dy = (H_x[i, j, k] - H_x[i, j - 1, k]) / h
                E_z[i, j, k] += (dt / eps0) * (dHy_dx - dHx_dy)

def update_H_fields(grid, E_x, E_y, E_z, H_x, H_y, H_z, h, dt):
    Nx, Ny, Nz = grid
    for i in range(Nx + 1):
        for j in range(Ny):
            for k in range(Nz):
                dEy_dz = (E_y[i, j, k + 1] - E_y[i, j, k]) / h if k < Nz - 1 else (0 - E_y[i, j, k]) / h
                dEz_dy = (E_z[i, j + 1, k] - E_z[i, j, k]) / h if j < Ny - 1 else (0 - E_z[i, j, k]) / h
                H_x[i, j, k] += (dt / mu0) * (dEy_dz - dEz_dy)
                if k == 0: H_x[i, j, k] += dt * 1.0 * E_y[i, j, 0]
                if k == Nz - 1: H_x[i, j, k] += dt * (-1.0) * E_y[i, j, Nz]
                if j == 0: H_x[i, j, k] += dt * 1.0 * E_z[i, 0, k]
                if j == Ny - 1: H_x[i, j, k] += dt * (-1.0) * E_z[i, Ny, k]
    for i in range(Nx):
        for j in range(Ny + 1):
            for k in range(Nz):
                dEz_dx = (E_z[i + 1, j, k] - E_z[i, j, k]) / h if i < Nx - 1 else (0 - E_z[i, j, k]) / h
                dEx_dz = (E_x[i, j, k + 1] - E_x[i, j, k]) / h if k < Nz - 1 else (0 - E_x[i, j, k]) / h
                H_y[i, j, k] += (dt / mu0) * (dEz_dx - dEx_dz)
                if i == 0: H_y[i, j, k] += dt * 1.0 * E_z[0, j, k]
                if i == Nx - 1: H_y[i, j, k] += dt * (-1.0) * E_z[Nx, j, k]
                if k == 0: H_y[i, j, k] += dt * 1.0 * E_x[i, j, 0]
                if k == Nz - 1: H_y[i, j, k] += dt * (-1.0) * E_x[i, j, Nz]
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz + 1):
                dEx_dy = (E_x[i, j + 1, k] - E_x[i, j, k]) / h if j < Ny - 1 else (0 - E_x[i, j, k]) / h
                dEy_dx = (E_y[i + 1, j, k] - E_y[i, j, k]) / h if i < Nx - 1 else (0 - E_y[i, j, k]) / h
                H_z[i, j, k] += (dt / mu0) * (dEx_dy - dEy_dx)
                if i == 0: H_z[i, j, k] += dt * 1.0 * E_y[0, j, k]
                if i == Nx - 1: H_z[i, j, k] += dt * (-1.0) * E_y[Nx, j, k]
                if j == 0: H_z[i, j, k] += dt * 1.0 * E_x[i, 0, k]
                if j == Ny - 1: H_z[i, j, k] += dt * (-1.0) * E_x[i, Ny, k]

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

# Interface Update
def update_interface():
    # Coarse to fine (e.g., E_z at x=0.5 boundary, i=5)
    for j in range(N_yf + 1):
        for k in range(N_zf):
            E_z_f[0, j, k] = interpolate_coarse_to_fine(E_z[5, j + y_f_start, :], z_f_start + k // 2, 1, 0)[0]
    # Add SAT terms (simplified, penalty to enforce continuity)
    sigma = 1 / h
    H_y[x_f_start, y_f_start:y_f_start + N_yf, z_f_start:z_f_start + N_zf] -= dt * sigma * (
    E_z[x_f_start, y_f_start:y_f_start + N_yf, z_f_start:z_f_start + N_zf] - 
    E_z_f[0, :N_yf, :]
)

# Simulation Loop
n_steps = 100
E_z_history = []

for n in range(n_steps):
    # Update coarse grid
    update_E_fields((N_x, N_y, N_z), E_x, E_y, E_z, H_x, H_y, H_z, h, dt)
    update_H_fields((N_x, N_y, N_z), E_x, E_y, E_z, H_x, H_y, H_z, h, dt)
    
    # Update fine grid (two steps)
    for _ in range(2):
        update_E_fields((N_xf, N_yf, N_zf), E_x_f, E_y_f, E_z_f, H_x_f, H_y_f, H_z_f, h_f, dt_f)
        update_H_fields((N_xf, N_yf, N_zf), E_x_f, E_y_f, E_z_f, H_x_f, H_y_f, H_z_f, h_f, dt_f)
    
    # Interface coupling
    update_interface()
    
    # Store for visualization
    E_z_slice = E_z[N_x // 2, :, :].copy()
    E_z_history.append(E_z_slice[:-1, :])

# Animation
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(E_z_history[0], cmap='RdBu', vmin=-0.1, vmax=0.1, interpolation='bicubic', aspect='equal')
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