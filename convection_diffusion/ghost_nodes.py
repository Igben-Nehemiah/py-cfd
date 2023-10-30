import numpy as np
import matplotlib.pyplot as plt


L = 1.0  # Length of the rod
nx = 10  # Number of cells (excluding ghost cells)
dx = L / nx  # Cell size

# Thermal conductivity
k = 1.0

# Define the temperature source term (e.g., a constant heat source)
Q = 1.0

# Boundary conditions
T_left = 100.0  # Temperature at the left boundary
T_right = 200.0  # Temperature at the right boundary

# Add ghost cells at the boundaries
nx += 2
x = np.linspace(-dx / 2, L + dx / 2, nx)

# Initialize temperature array (including ghost cells)
T = np.zeros(nx)

# Set initial guess (could be all zeros)
T[1:-1] = 0.5 * (T_left + T_right)

# Perform an iterative solution (e.g., Gauss-Seidel)
max_iterations = 1000
tolerance = 1e-6

for iteration in range(max_iterations):
    T_new = np.copy(T)

    for i in range(1, nx - 1):
        # Calculate heat flux at cell faces
        q_left = -k * (T[i] - T[i - 1]) / dx
        q_right = -k * (T[i + 1] - T[i]) / dx

        # Update temperature at cell center using finite volume method
        T_new[i] = T[i] + (q_left - q_right + Q * dx) / (k * dx)

    # Apply boundary conditions to ghost cells
    T_new[0] = 2 * T_left - T_new[1]
    T_new[-1] = 2 * T_right - T_new[-2]

    # Check for convergence
    if np.allclose(T, T_new, atol=tolerance):
        break

    T = T_new


plt.plot(x, T)
plt.xlabel("Position (m)")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Distribution in a Rod")
plt.grid(True)
plt.show()
