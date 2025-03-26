import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
from numba import jit

# Constants
mu0 = 4 * np.pi * 1e-7    # Magnetic permeability
eps0 = 8.854e-12          # Electric permittivity
c = 1 / np.sqrt(mu0 * eps0)  # Speed of light
h = 0.01                  # Spatial step size (m)
dt = h / (2 * c)          # Time step size (s), adjusted for stability
N_x, N_y, N_z = 50, 50, 50  # Grid size

# Field arrays (standard Yee grid)
E_x = np.zeros((N_x, N_y + 1, N_z + 1))  # Edges along x
E_y = np.zeros((N_x + 1, N_y, N_z + 1))  # Edges along y
E_z = np.zeros((N_x + 1, N_y + 1, N_z))  # Edges along z
H_x = np.zeros((N_x + 1, N_y, N_z))      # Faces normal to x
H_y = np.zeros((N_x, N_y + 1, N_z))      # Faces normal to y
H_z = np.zeros((N_x, N_y, N_z + 1))      # Faces normal to z

# Function to update E-fields with SBP-SAT considerations
@jit(nopython=True)
def update_E_fields(E_x, E_y, E_z, H_x, H_y, H_z, dt, eps0, h, N_x, N_y, N_z):
    # Update E_x
    for i in range(N_x):
        for j in range(N_y + 1):
            for k in range(N_z + 1):
                # dH_z / dy
                if j == 0:
                    dHz_dy = (H_z[i, j, k] - 0) / h  # One-sided at j=0
                elif j == N_y:
                    dHz_dy = (0 - H_z[i, j - 1, k]) / h  # One-sided at j=N_y
                else:
                    dHz_dy = (H_z[i, j, k] - H_z[i, j - 1, k]) / h
                # dH_y / dz
                if k == 0:
                    dHy_dz = (H_y[i, j, k] - 0) / h  # One-sided at k=0
                elif k == N_z:
                    dHy_dz = (0 - H_y[i, j, k - 1]) / h  # One-sided at k=N_z
                else:
                    dHy_dz = (H_y[i, j, k] - H_y[i, j, k - 1]) / h
                E_x[i, j, k] += (dt / eps0) * (dHz_dy - dHy_dz)

    # Update E_y
    for i in range(N_x + 1):
        for j in range(N_y):
            for k in range(N_z + 1):
                # dH_x / dz
                if k == 0:
                    dHx_dz = (H_x[i, j, k] - 0) / h
                elif k == N_z:
                    dHx_dz = (0 - H_x[i, j, k - 1]) / h
                else:
                    dHx_dz = (H_x[i, j, k] - H_x[i, j, k - 1]) / h
                # dH_z / dx
                if i == 0:
                    dHz_dx = (H_z[i, j, k] - 0) / h
                elif i == N_x:
                    dHz_dx = (0 - H_z[i - 1, j, k]) / h
                else:
                    dHz_dx = (H_z[i, j, k] - H_z[i - 1, j, k]) / h
                E_y[i, j, k] += (dt / eps0) * (dHx_dz - dHz_dx)

    # Update E_z
    for i in range(N_x + 1):
        for j in range(N_y + 1):
            for k in range(N_z):
                # dH_y / dx
                if i == 0:
                    dHy_dx = (H_y[i, j, k] - 0) / h
                elif i == N_x:
                    dHy_dx = (0 - H_y[i - 1, j, k]) / h
                else:
                    dHy_dx = (H_y[i, j, k] - H_y[i - 1, j, k]) / h
                # dH_x / dy
                if j == 0:
                    dHx_dy = (H_x[i, j, k] - 0) / h
                elif j == N_y:
                    dHx_dy = (0 - H_x[i, j - 1, k]) / h
                else:
                    dHx_dy = (H_x[i, j, k] - H_x[i, j - 1, k]) / h
                E_z[i, j, k] += (dt / eps0) * (dHy_dx - dHx_dy)

# Function to update H-fields with SAT terms for PEC
@jit(nopython=True)
def update_H_fields(H_x, H_y, H_z, E_x, E_y, E_z, dt, mu0, h, N_x, N_y, N_z):
    # Update H_x
    for i in range(N_x + 1):
        for j in range(N_y):
            for k in range(N_z):
                dEz_dy = (E_z[i, j + 1, k] - E_z[i, j, k]) / h
                dEy_dz = (E_y[i, j, k + 1] - E_y[i, j, k]) / h
                H_x[i, j, k] -= (dt / mu0) * (dEz_dy - dEy_dz)
                # SAT terms for PEC
                if k == 0:
                    H_x[i, j, k] +- dt * 1.0 * E_y[i, j, 0]  # Enforce E_y = 0 at z=0
                if k == N_z - 1:
                    H_x[i, j, k] -= dt * 1.0 * E_y[i, j, N_z]  # Enforce E_y = 0 at z=h*N_z

    # Update H_y
    for i in range(N_x):
        for j in range(N_y + 1):
            for k in range(N_z):
                dEx_dz = (E_x[i, j, k + 1] - E_x[i, j, k]) / h
                dEz_dx = (E_z[i + 1, j, k] - E_z[i, j, k]) / h
                H_y[i, j, k] -= (dt / mu0) * (dEx_dz - dEz_dx)
                # SAT terms for PEC
                if k == 0:
                    H_y[i, j, k] -= dt * 1.0 * E_x[i, j, 0]  # Enforce E_x = 0 at z=0
                if k == N_z - 1:
                    H_y[i, j, k] += dt * 1.0 * E_x[i, j, N_z]  # Enforce E_x = 0 at z=h*N_z

    # Update H_z
    for i in range(N_x):
        for j in range(N_y):
            for k in range(N_z + 1):
                dEy_dx = (E_y[i + 1, j, k] - E_y[i, j, k]) / h
                dEx_dy = (E_x[i, j + 1, k] - E_x[i, j, k]) / h
                H_z[i, j, k] -= (dt / mu0) * (dEy_dx - dEx_dy)
                # SAT terms for PEC
                if j == 0:
                    H_z[i, j, k] += dt * 1.0 * E_x[i, 0, k]  # Enforce E_x = 0 at y=0
                if j == N_y - 1:
                    H_z[i, j, k] -= dt * 1.0 * E_x[i, N_y, k]  # Enforce E_x = 0 at y=h*N_y

# Initial condition: Gaussian pulse in E_z at the center
x0, y0, z0 = N_x // 2, N_y // 2, N_z // 2
sigma = h
for i in range(N_x + 1):
    for j in range(N_y + 1):
        for k in range(N_z):
            r = np.sqrt((i - x0)**2 + (j - y0)**2 + (k - z0)**2) * h
            E_z[i, j, k] = np.exp(-r**2 / (2 * sigma**2))

# Simulation Loop
n_steps = 1000
E_z_history = []

for n in tqdm(range(n_steps), desc="Simulation Progress"):
    # Update coarse grid
    update_E_fields(E_x, E_y, E_z, H_x, H_y, H_z, dt, eps0, h, N_x, N_y, N_z)
    update_H_fields(H_x, H_y, H_z, E_x, E_y, E_z, dt, mu0, h, N_x, N_y, N_z)
    
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