import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


# Function to interpolate the velocity field at particle positions
def interpolate_velocity(positions, field):
    velocities = griddata((x.ravel(), y.ravel()), field.reshape(nx * ny, 2),
                          positions, method='linear')
    return velocities


# Define the grid dimensions (nx, ny)
nx = 10  # Number of grid points in the x-direction
ny = 10  # Number of grid points in the y-direction

# Create a grid of x and y coordinates
x, y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))

# Define a random vector field as a numpy array with shape (nx, ny, 2)
# The shape (nx, ny, 2) represents two components: x and y components of velocity
# vector_field = np.random.rand(nx, ny, 2)
vector_field = np.load('particle_tracing/velocity_field.npy')

# Normalize the vector field to make it smoother (optional)
vector_magnitude = np.sqrt(vector_field[:, :, 0]**2 + vector_field[:, :, 1]**2)
# vector_field[:, :, 0] /= vector_magnitude
# vector_field[:, :, 1] /= vector_magnitude

# Initialize particle positions
num_particles = 10
# Random initial positions within [0, 1]
initial_positions = np.random.rand(num_particles, 2)

# Parameters
time_step = 0.02
num_steps = 100

# Particle tracing loop
particle_positions = [initial_positions]
for _ in range(num_steps):
    current_positions = particle_positions[-1]

    # Interpolate vector field values at particle positions
    # Interpolate vector field values at particle positions
    velocities = interpolate_velocity(current_positions, vector_field)

    # Euler's method
    new_positions = current_positions + velocities * time_step
    particle_positions.append(new_positions)

# Extract x and y coordinates for plotting
x_coords = np.array([pos[:, 0] for pos in particle_positions])
y_coords = np.array([pos[:, 1] for pos in particle_positions])

# Plot the vector field
plt.figure(figsize=(8, 6))
plt.quiver(x, y, vector_field[:, :, 0], vector_field[:, :, 1], scale=10)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Vector Field")

# Plot the trajectories of particles
plt.figure(figsize=(8, 6))
for i in range(num_particles):
    plt.plot(x_coords[:, i], y_coords[:, i])

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Particle Tracing in the Vector Field")
plt.show()
