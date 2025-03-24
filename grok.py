import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Simulation Parameters
N_x, N_y, N_z = 20, 20, 20  # Number of cells in x, y, z directions
h = 0.1                     # Spatial step size (meters)
dt = 0.05                   # Time step size (seconds), satisfies CFL: dt < h / (c * sqrt(3))
c = 1.0                     # Speed of light (normalized units)
eps0 = 1.0                  # Electric permittivity (normalized units)
mu0 = 1.0                   # Magnetic permeability (normalized units)

# Initialize Field Arrays
# Electric field components (staggered grid)
E_x = np.zeros((N_x, N_y + 1, N_z + 1))  # E_x at (i+0.5, j, k)
E_y = np.zeros((N_x + 1, N_y, N_z + 1))  # E_y at (i, j+0.5, k)
E_z = np.zeros((N_x + 1, N_y + 1, N_z))  # E_z at (i, j, k+0.5)

# Magnetic field components (staggered grid)
H_x = np.zeros((N_x + 1, N_y, N_z))      # H_x at (i, j+0.5, k+0.5)
H_y = np.zeros((N_x, N_y + 1, N_z))      # H_y at (i+0.5, j, k+0.5)
H_z = np.zeros((N_x, N_y, N_z + 1))      # H_z at (i+0.5, j+0.5, k)

# Set Initial Condition: Gaussian pulse in E_z at the center
i_center, j_center, k_center = N_x // 2, N_y // 2, N_z // 2
E_z[i_center, j_center, k_center] = 1.0  # Amplitude of the pulse

# Function to Update Electric Fields
def update_E_fields():
    """
    Updates E-field components using standard FDTD-like differences for interior points.
    Boundary points are updated naturally, with SAT terms in H-field updates enforcing PEC conditions.
    """
    # Update E_x: ∂H_z/∂y - ∂H_y/∂z
    for i in range(N_x):
        for j in range(1, N_y):
            for k in range(1, N_z):
                dHz_dy = (H_z[i, j, k] - H_z[i, j - 1, k]) / h
                dHy_dz = (H_y[i, j, k] - H_y[i, j, k - 1]) / h
                E_x[i, j, k] += (dt / eps0) * (dHz_dy - dHy_dz)

    # Update E_y: ∂H_x/∂z - ∂H_z/∂x
    for i in range(1, N_x):
        for j in range(N_y):
            for k in range(1, N_z):
                dHx_dz = (H_x[i, j, k] - H_x[i, j, k - 1]) / h
                dHz_dx = (H_z[i, j, k] - H_z[i - 1, j, k]) / h
                E_y[i, j, k] += (dt / eps0) * (dHx_dz - dHz_dx)

    # Update E_z: ∂H_y/∂x - ∂H_x/∂y
    for i in range(1, N_x):
        for j in range(1, N_y):
            for k in range(N_z):
                dHy_dx = (H_y[i, j, k] - H_y[i - 1, j, k]) / h
                dHx_dy = (H_x[i, j, k] - H_x[i, j - 1, k]) / h
                E_z[i, j, k] += (dt / eps0) * (dHy_dx - dHx_dy)

# Function to Update Magnetic Fields with SAT Terms
def update_H_fields():
    """
    Updates H-field components using standard FDTD differences and adds SAT terms
    to enforce PEC boundary conditions (tangential E = 0) weakly.
    """
    # Update H_x: ∂E_y/∂z - ∂E_z/∂y
    for i in range(N_x + 1):
        for j in range(N_y):
            for k in range(N_z):
                dEy_dz = (E_y[i, j, k + 1] - E_y[i, j, k]) / h if k < N_z - 1 else (0 - E_y[i, j, k]) / h
                dEz_dy = (E_z[i, j + 1, k] - E_z[i, j, k]) / h if j < N_y - 1 else (0 - E_z[i, j, k]) / h
                H_x[i, j, k] += (dt / mu0) * (dEy_dz - dEz_dy)
                # SAT terms for PEC boundaries
                if k == 0:          # Bottom (z=0), E_y tangential
                    H_x[i, j, k] += dt * 1.0 * E_y[i, j, 0]
                if k == N_z - 1:    # Top (z=Lz), E_y tangential
                    H_x[i, j, k] += dt * (-1.0) * E_y[i, j, N_z]
                if j == 0:          # Front (y=0), E_z tangential
                    H_x[i, j, k] += dt * 1.0 * E_z[i, 0, k]
                if j == N_y - 1:    # Back (y=Ly), E_z tangential
                    H_x[i, j, k] += dt * (-1.0) * E_z[i, N_y, k]

    # Update H_y: ∂E_z/∂x - ∂E_x/∂z
    for i in range(N_x):
        for j in range(N_y + 1):
            for k in range(N_z):
                dEz_dx = (E_z[i + 1, j, k] - E_z[i, j, k]) / h if i < N_x - 1 else (0 - E_z[i, j, k]) / h
                dEx_dz = (E_x[i, j, k + 1] - E_x[i, j, k]) / h if k < N_z - 1 else (0 - E_x[i, j, k]) / h
                H_y[i, j, k] += (dt / mu0) * (dEz_dx - dEx_dz)
                # SAT terms for PEC boundaries
                if i == 0:          # Left (x=0), E_z tangential
                    H_y[i, j, k] += dt * 1.0 * E_z[0, j, k]
                if i == N_x - 1:    # Right (x=Lx), E_z tangential
                    H_y[i, j, k] += dt * (-1.0) * E_z[N_x, j, k]
                if k == 0:          # Bottom (z=0), E_x tangential
                    H_y[i, j, k] += dt * 1.0 * E_x[i, j, 0]
                if k == N_z - 1:    # Top (z=Lz), E_x tangential
                    H_y[i, j, k] += dt * (-1.0) * E_x[i, j, N_z]

    # Update H_z: ∂E_x/∂y - ∂E_y/∂x
    for i in range(N_x):
        for j in range(N_y):
            for k in range(N_z + 1):
                dEx_dy = (E_x[i, j + 1, k] - E_x[i, j, k]) / h if j < N_y - 1 else (0 - E_x[i, j, k]) / h
                dEy_dx = (E_y[i + 1, j, k] - E_y[i, j, k]) / h if i < N_x - 1 else (0 - E_y[i, j, k]) / h
                H_z[i, j, k] += (dt / mu0) * (dEx_dy - dEy_dx)
                # SAT terms for PEC boundaries
                if i == 0:          # Left (x=0), E_y tangential
                    H_z[i, j, k] += dt * 1.0 * E_y[0, j, k]
                if i == N_x - 1:    # Right (x=Lx), E_y tangential
                    H_z[i, j, k] += dt * (-1.0) * E_y[N_x, j, k]
                if j == 0:          # Front (y=0), E_x tangential
                    H_z[i, j, k] += dt * 1.0 * E_x[i, 0, k]
                if j == N_y - 1:    # Back (y=Ly), E_x tangential
                    H_z[i, j, k] += dt * (-1.0) * E_x[i, N_y, k]

# Run Simulation
n_steps = 100              # Number of time steps
E_z_history = []           # Store E_z slices for animation

for n in range(n_steps):
    update_E_fields()      # Update electric fields
    update_H_fields()      # Update magnetic fields with SAT terms
    # Record E_z at x = N_x//2 plane for visualization
    E_z_slice = E_z[N_x // 2, :, :].copy()
    # Trim the E_z slice to match the dimensions required by pcolormesh
    E_z_history.append(E_z_slice[:-1, :])

# Set Up Animation
fig, ax = plt.subplots(figsize=(8, 6))
# Use imshow with interpolation for smoother visualization
im = ax.imshow(E_z_history[0], cmap='RdBu', vmin=-.1, vmax=.1, 
               interpolation='bicubic', aspect='equal')
ax.set_title(f'E_z Field Progression (x = {N_x//2})')
ax.set_xlabel('y')
ax.set_ylabel('z')
fig.colorbar(im, label='E_z (V/m)')

# Animation Function
def animate(n):
    im.set_array(E_z_history[n])
    ax.set_title(f'E_z Field Progression (x = {N_x//2}), Step {n}')
    return [im]

# Create and Display Animation
anim = animation.FuncAnimation(fig, animate, frames=n_steps, interval=100, blit=True)
plt.show()