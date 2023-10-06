import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def calculate_velocity(x, y):
    vx = -y
    vy = x
    return vx, vy


grid_size_x, grid_size_y = 40, 40
x = np.linspace(-5, 5, grid_size_x)
y = np.linspace(-5, 5, grid_size_y)

X, Y = np.meshgrid(x, y)

vx, vy = calculate_velocity(X, Y)

# This will be used for the discrete case
velocity_field = np.stack((vx, vy), axis=-1)


# Initialise particle positions
num_particles = 100
# Random initial positions within [0, 1]
initial_positions = np.random.uniform(-2, 2, (num_particles, 2))

# Parameters
time_step = 0.02
num_steps = 1000

# Particle tracing loop
particle_positions = [initial_positions]
for _ in range(num_steps):
    current_positions = particle_positions[-1]

    # Interpolate vector field values at particle positions
    velocities = np.array([calculate_velocity(x, y)
                          for x, y in current_positions])

    # Euler's method
    new_positions = current_positions + velocities * time_step
    particle_positions.append(new_positions)

grid_options = {
    'visible': True,
    'color': 'y',
    'linestyle': '--',
    'linewidth': 1
}
options = {
    'color': 'Teal',
    'scale': 50
}
# Extract x and y coordinates for plotting
x_coords = np.array([pos[:, 0] for pos in particle_positions])
y_coords = np.array([pos[:, 1] for pos in particle_positions])

fig, ax = plt.subplots(facecolor='k')

ax.set_facecolor('black')
ax.grid(**grid_options)
paths = ax.scatter(x_coords[0], y_coords[0])
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

ax.set_xticks([-2, -1, 0, 1, 2])

ax.quiver(x, y, velocity_field[:, :, 0],
          velocity_field[:, :, 1], **options)


def update(frame):
    ax.clear()
    ax.quiver(x, y, velocity_field[:, :, 0],
              velocity_field[:, :, 1], **options)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.scatter(x_coords[frame], y_coords[frame], color=['Teal'], marker='.')
    ax.grid(**grid_options)
    return paths,


anim = FuncAnimation(fig, update, num_steps - 1,
                     interval=1, cache_frame_data=False)
plt.show()
