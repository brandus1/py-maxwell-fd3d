import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
from numba import jit

# Simulation Parameters
N_x, N_y, N_z = 100, 100, 100  # Coarse grid sizes
h = .04
dt = 0.005                   # Coarse time step (seconds)
c = 1.0                     # Speed of light
eps0 = 1.0                 # Electric permittivity
mu0 = 1.0                   # Magnetic permeability

# Coarse Field Arrays
E_x = np.zeros((N_x, N_y + 1, N_z + 1))
E_y = np.zeros((N_x + 1, N_y, N_z + 1))
E_z = np.zeros((N_x + 1, N_y + 1, N_z))
H_x = np.zeros((N_x + 1, N_y, N_z))
H_y = np.zeros((N_x, N_y + 1, N_z))
H_z = np.zeros((N_x, N_y, N_z + 1))


# Initial Condition: Gaussian pulse in coarse E_z
sigma = 2.0  # Width in grid units, e.g., 2 grid points
for i in range(N_x + 1):
    for j in range(N_y + 1):
        for k in range(N_z):
            dist_sq = (i - N_x // 2)**2 + (j - N_y // 2)**2 + (k - N_z // 2)**2
            E_z[i, j, k] = np.exp(-dist_sq / (2 * sigma**2))
# Initial Condition: Point source in coarse E_z
# i_center, j_center, k_center = N_x // 2, N_y // 2, N_z // 2
# E_z[i_center, j_center, k_center] = 1.0

# Update Functions (optimized with Numba)
@jit(nopython=True)
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

@jit(nopython=True)
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

# Simulation Loop
n_steps = 1000
E_z_history = []

for n in tqdm(range(n_steps), desc="Simulation Progress"):
    # Update coarse grid
    update_E_fields((N_x, N_y, N_z), E_x, E_y, E_z, H_x, H_y, H_z, h, dt)
    update_H_fields((N_x, N_y, N_z), E_x, E_y, E_z, H_x, H_y, H_z, h, dt)
    
    # Store for visualization
    E_z_slice = E_z[N_x // 2, :, :].copy()
    E_z_history.append(E_z_slice[:-1, :])

# Animation
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(E_z_history[0], cmap='RdBu', vmin=-0.05, vmax=0.05, interpolation='bicubic', aspect='equal')
ax.set_title(f'E_z Field Progression (x = {N_x//2})')
ax.set_xlabel('y')
ax.set_ylabel('z')
fig.colorbar(im, label='E_z (V/m)')

time_text = ax.text(0.02, 0.95, 'Step: 0', transform=ax.transAxes)

def animate(n):
    im.set_array(E_z_history[n])
    time_text.set_text(f'Step: {n}')
    return [im, time_text]

anim = animation.FuncAnimation(fig, animate, frames=n_steps, interval=0, blit=True)
plt.show()