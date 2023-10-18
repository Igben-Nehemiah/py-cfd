import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation

from vector_field import VectorField, GridRange
from .particle_tracer import ParticleTracer

Pos = tuple[float, float, float] 

def get_positions_from_radius(radius: float, n_points: int, x_offset=0, y_offset=0, z_offset=0) -> list[Pos]:
    angle = 2*np.pi/n_points
    positions: list[Pos] = []

    for i in range(n_points):
        x = radius*np.cos(i*angle) + x_offset
        y = radius*np.sin(i*angle) + y_offset
        z = radius
        positions.append((x, y, z))
    
    return positions

field = VectorField(grid_ranges=[GridRange(-5,5,30), GridRange(-5,5,30), GridRange(-5,5,30)])
field.set_from_function(lambda x, y, z: (x, y, z))

positions = get_positions_from_radius(1, 10)
pt = ParticleTracer(field=field, initial_positions=positions, dt=0.05, n_time_steps=100)

pt.trace()

def update(num, ax):
    ax.quiver(*field.mesh_grids,
                    *field.mesh_grids_values)


fig = plt.figure()
ax = fig.add_subplot(projection="3d")

# Setting the axes properties
ax.set(xlim3d=(-5, 5), xlabel='X')
ax.set(ylim3d=(-5, 5), ylabel='Y')
ax.set(zlim3d=(-5, 5), zlabel='Z')

ani = animation.FuncAnimation(fig, update, 10, fargs=(ax,))

plt.show()

