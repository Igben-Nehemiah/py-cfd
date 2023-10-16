import numpy as np
from particle_advection.particle_path_plotter import Plotter
from particle_advection.particle_tracer import ParticleTracer
from particle_advection.plot_config import PlotConfig
from particle_advection.vector_fields_functions import divergence_field
from vector_field import VectorField, GridRange
from random import uniform


# Helper functions
Pos = tuple[float, float] 

def get_positions_from_radius(radius: float, n_points: int, x_offset=0, y_offset=0) -> list[Pos]:
    angle = 2*np.pi/n_points
    positions: list[Pos] = []

    for i in range(n_points):
        x = radius*np.cos(i*angle) + x_offset
        y = radius*np.sin(i*angle) + y_offset
        positions.append((x, y))
    
    return positions


def get_positions(start_pos: tuple[float, float], end_pos: tuple[float, float], n_points: float):
    dx, dy = (end_pos[0] - start_pos[0])/n_points , (end_pos[1] - start_pos[1])/n_points
    positions: list[Pos] = []
    
    for i in range(0, n_points):
        x = start_pos[0] + i*dx
        y = start_pos[1] + i*dy
        positions.append((x, y))
    
    return positions


def get_random_positions(x_range: tuple[float, float], y_range: tuple[float, float], n_points):

    positions: list[Pos] = []

    for i in range(n_points):
        rand_x = uniform(x_range[0], x_range[1])
        rand_y = uniform(y_range[0], y_range[1])
        positions.append((rand_x, rand_y))

    return positions

field1 = VectorField([GridRange(-12, 12, 30), GridRange(-6, 6, 30)])
field1.set_from_function(lambda x,y: (2*x*y, 1-x**2-y**2))
field1.desc = r"$\vec{F} = 2xy\hat{i} + (1-x^2-y^2)\hat{j}$"
field1_initial_positions = get_positions_from_radius(0.5, 10) + get_positions((-5,6), (5,6), 20)

# This one shows a positive divergence
field2 = VectorField([GridRange(-12, 12, 30), GridRange(-6, 6, 30)])
field2.set_from_function(divergence_field)
field2.desc = r"$\vec{F} = 2x\hat{i} + 2y\hat{j}$"
field2_initial_positions = get_positions_from_radius(2, 12) + get_positions_from_radius(1, 12) + get_positions_from_radius(0.2, 12)

f3_xrange = GridRange(-12, 12, 30)
f3_yrange = GridRange(-6, 6, 30)
field3 = VectorField([f3_xrange, f3_yrange])
field3.set_from_function(lambda x, y: (2*y**2+x-4, np.cos(x)))
field3.desc = r"$\vec{F} = (2y^2+x-4)\hat{i} + cos(x)\hat{j}$"
field3_initial_positions = get_positions_from_radius(2, 10) + get_positions_from_radius(2.5, 10)

# This one shows a non-zero curl
field4 = VectorField([GridRange(-12, 12, 30), GridRange(-6, 6, 30)])
field4.set_from_function(lambda x, y: (-y, x))
field4.desc = r"$\vec{F} = -y\hat{i} + x\hat{j}$"
field4_initial_positions = get_random_positions((-5, 5), (-5,5), 20)

fields = [field1, field2, field3, field4]

initial_positions_list = [field1_initial_positions, field2_initial_positions, 
                            field3_initial_positions, field4_initial_positions]

tracers = [ParticleTracer(field=field, initial_positions=initial_positions_list[i], dt=0.02, n_time_steps=200) for i, field in enumerate(fields)]

plot_cfg = PlotConfig(size=(12,6),
    title_size=20,
    label_size=16)

plotter = Plotter(plot_cfg, tracers)
plotter.plot(animate=True, show_path_lines=[True, False], save=True)