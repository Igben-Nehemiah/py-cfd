import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from boundary_condition import BoundaryCondition
from heat_diffusion_solver import HeatDiffusionSolver
from rectangular_plate import RectangularPlate


dt = 0.2
time_interval = 40

plate1 = RectangularPlate(0.5, 0.5, 8850, 389, 385)
plate1.discretise(50, 50)
plate1.temperature = 100  # initial temperature

plate2 = plate1.copy()
plate3 = plate1.copy()


def calculate():
    solver = HeatDiffusionSolver(dt, time_interval)
    solver.set_boundary_condition(
        east_bc=BoundaryCondition('Dirichlet', 100),
        west_bc=BoundaryCondition('Dirichlet', 0),
        north_bc=BoundaryCondition('Dirichlet', 0),
        south_bc=BoundaryCondition('Dirichlet', 0)
    )

    # Solve for plate1
    solver.plate = plate1
    solver.solve()
    plate1_time_temp_dist = solver.time_temp_dist.copy()

    # Solve for plate2 with different boundary conditions
    solver.west_bc = BoundaryCondition('Dirichlet', 100)
    solver.east_bc = BoundaryCondition('Dirichlet', 0)
    solver.plate = plate2
    solver.solve()
    plate2_time_temp_dist = solver.time_temp_dist.copy()

    # Superposition of the solutions of plate1 and plate 2
    superposition_sol = np.copy(plate1_time_temp_dist)
    superposition_sol[:] += plate2_time_temp_dist[:]

    # Solve for plate3 with the boundary conditions of plate1 and plate2 combined

    solver.set_boundary_condition(
        east_bc=BoundaryCondition('Dirichlet', 100),
        west_bc=BoundaryCondition('Dirichlet', 100),
        north_bc=BoundaryCondition('Dirichlet', 0),
        south_bc=BoundaryCondition('Dirichlet', 0)
    )
    solver.plate = plate3
    solver.solve()
    plate3_time_temp_dist = solver.time_temp_dist.copy()

    return plate1_time_temp_dist, plate2_time_temp_dist, plate3_time_temp_dist, superposition_sol


def plot(time_temp_dists, nt_steps, animate=False):
    # Common options for colormesh plots
    options = {
        'cmap': plt.cm.jet,
        'vmin': 0,
        'vmax': 100
    }

    def update(n):
        # Set the figure title based on the current time
        title.set_text(f'Temperature at t = {n*dt:2f} s')

        # Update the colormesh plots in each subplot
        mesh1 = ax1.pcolormesh(time_temp_dists[0][n], **options)
        mesh2 = ax2.pcolormesh(time_temp_dists[1][n], **options)
        mesh3 = ax3.pcolormesh(time_temp_dists[2][n], **options)
        mesh4 = ax4.pcolormesh(time_temp_dists[3][n], **options)

        return mesh1, mesh2, mesh3, mesh4, title

    fig = plt.figure(constrained_layout=True)
    spec = fig.add_gridspec(3, 2)

    # Add subplots to the figure
    ax1 = fig.add_subplot(spec[0, 0])
    ax1.set_title('Plate 1')

    ax2 = fig.add_subplot(spec[0, 1])
    ax2.set_title('Plate 2')

    ax3 = fig.add_subplot(spec[2, :])
    ax3.set_title('Superposition')

    ax4 = fig.add_subplot(spec[1, :])
    ax4.set_title('Plate 3')

    for ax in fig.axes:
        ax.set_xticks([])
        ax.set_yticks([])

    title = fig.suptitle('Temperature at t = 0.00 seconds', fontsize=12)

    # Create a colorbar for all subplots
    cbar = fig.colorbar(ax1.pcolormesh(time_temp_dists[0][0], **options), ax=[ax1, ax2, ax3, ax4],
                        orientation='vertical', shrink=0.8)
    cbar.set_label('Temperature')

    if animate:
        anim = FuncAnimation(fig, update, frames=nt_steps, repeat=False)
        anim.save("heat_equation_solution.mp4")
        # plt.show()
    else:
        # If not animating, update the plot for the final time step and display
        update(time_temp_dists[0].shape[0] - 1)
        plt.show()


if __name__ == '__main__':
    p1, p2, p3, s = calculate()
    plot([p1, p2, s, p3], p1.shape[0] - 1, False)
