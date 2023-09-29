from copy import deepcopy
from typing import Literal
import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Plate:
    def __init__(self, width: float, height: float, rho: float, cp: float, k: float):
        self.width = width
        self.height = height
        self.rho = rho
        self.cp = cp
        self.k = k

    def discretise(self, nx_nodes: int, ny_nodes):
        self.nx_nodes = nx_nodes
        self.ny_nodes = ny_nodes
        self.dx = self.width/nx_nodes
        self.dy = self.height/ny_nodes
        self._temperature = np.empty((nx_nodes, ny_nodes))

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, temp: NDArray[np.float64] | float):
        if self._temperature is None:
            raise ValueError("Discretise plate before setting temperature")

        if isinstance(temp, (int, float)):
            self._temperature[:, :] = temp
            return

        if temp.shape != self._temperature.shape:
            raise ValueError("Temperature shapes do not match")

        self._temperature = temp

    @property
    def critical_time_step(self):
        return self.rho*self.cp*self.dx**2*self.dy**2/(2*self.k*(self.dx**2 + self.dy**2))

    def copy(self):
        return deepcopy(self)


class BoundaryCondition:
    def __init__(self, bc_type: Literal['Nuemann', 'Dirichlet'], value: float):
        self.value = value
        self.bc_type = bc_type


class FlatPlateDiffusionSolver:
    def __init__(self, plate: Plate, dt: float, time_interval: float):
        self.dt = dt
        self.plate = plate
        self.time_interval = time_interval
        self.n_time_steps = int(self.time_interval / dt) + 1

        # This will hold the temperature of the plate at different times
        self._time_temp_dist = np.empty(
            (self.n_time_steps, self.plate.nx_nodes, self.plate.ny_nodes))

    def set_boundary_condition(self, west_bc: BoundaryCondition,
                               east_bc: BoundaryCondition,
                               north_bc: BoundaryCondition,
                               south_bc: BoundaryCondition):
        self.west_bc = west_bc
        self.east_bc = east_bc
        self.north_bc = north_bc
        self.south_bc = south_bc

    def solve(self):
        if self.dt > self.plate.critical_time_step:
            raise ArithmeticError(
                f'The time step is greater than the critical time step {self.plate.critical_time_step} ')

        # Set initial condition
        self._time_temp_dist[0] = self.plate.temperature

        dx, dy = self.plate.dx, self.plate.dy
        k, cp, rho = self.plate.k, self.plate.cp, self.plate.rho

        T_W, T_E, T_N, T_S = self.west_bc.value, self.east_bc.value, self.north_bc.value, self.south_bc.value

        a_E = a_W = k*dy/dx
        a_N = a_S = k*dx/dy
        a_Po = rho*cp*dx*dy/self.dt
        a_P = a_Po

        t_step = 0  # Current time step

        while (t_step < self.n_time_steps - 1):
            t_step += 1

            # Previous time temperature nodes
            T_prev = self.plate.temperature

            # Current time temperature nodes
            T = self._time_temp_dist[t_step]

            # Internal Nodes
            T[1:-1, 1:-1] = (a_E*T_prev[1:-1, 2:] + a_W*T_prev[1:-1, 0:-2] + a_N*T_prev[0:-2, 1:-1] +
                             a_S*T_prev[2:, 1:-1] + (a_Po - (a_W + a_E + a_N + a_S))*T_prev[1:-1, 1:-1])/a_P

            # Western Face
            S_u = 2*k*dy*(T_W - T_prev[1:-1, 0])/dx
            T[1:-1, 0] = (a_E*T_prev[1:-1, 1] + a_N*T_prev[0:-2, 0] +
                          a_S*T_prev[2:, 0] + (a_Po - (a_E + a_N + a_S))*T_prev[1:-1, 0] + S_u)/a_P

            # Eastern Face
            S_u = 2*k*dy*(T_E - T_prev[1:-1, -1])/dx
            T[1:-1, -1] = (a_W*T_prev[1:-1, -2] + a_N*T_prev[0:-2, -1] +
                           a_S*T_prev[2:, -1] + (a_Po - (a_W + a_N + a_S))*T_prev[1:-1, -1] + S_u)/a_P

            # Northern Face
            S_u = 2*k*dx*(T_N - T_prev[0, 1:-1])/dy
            T[0, 1:-1] = (a_E*T_prev[0, 2:] + a_W*T_prev[0, 0:-2] +
                          a_S*T_prev[0, 1:-1] + (a_Po - (a_E + a_W + a_S))*T_prev[0, 1:-1] + S_u)/a_P

            # Southern Face
            S_u = 2*k*dx*(T_S - T_prev[-1, 1:-1])/dy
            T[-1, 1:-1] = (a_E*T_prev[-1, 2:] + a_W*T_prev[-1, 0:-2] +
                           a_N*T_prev[-1, 1:-1] + (a_Po - (a_E + a_W + a_N))*T_prev[-1, 1:-1] + S_u)/a_P

            # North-West Corner
            S_u = 2*k*dy*(T_W - T_prev[0, 0])/dx + \
                2*k*dx*(T_N - T_prev[0,  0])/dy
            T[0, 0] = (a_E*T_prev[0, 1] + a_S*T_prev[1, 0] + (a_Po - (a_E + a_S)) *
                       T_prev[0, 0] + S_u)/a_P

            # South-West Corner
            S_u = 2*k*dy*(T_W - T_prev[-1, 0])/dx + \
                2*k*dx*(T_S - T_prev[-1,  0])/dy
            T[-1, 0] = (a_E*T_prev[-1, 1] + a_N*T_prev[-2, 0] + (a_Po - (a_E + a_N)) *
                        T_prev[-1, 0] + S_u)/a_P

            # North-East Corner
            S_u = 2*k*dy*(T_E - T_prev[0, -1])/dx + \
                2*k*dx*(T_N - T_prev[0,  -1])/dy
            T[0, -1] = (a_W*T_prev[0, -2] + a_S*T_prev[1, -1] + (a_Po - (a_W + a_S)) *
                        T_prev[0, -1] + S_u)/a_P

            # South_East Corner
            S_u = 2*k*dy*(T_E - T_prev[-1, -1])/dx + \
                2*k*dx*(T_S - T_prev[-1,  -1])/dy
            T[-1, -1] = (a_W*T_prev[-1, -2] + a_N*T_prev[-2, -1] + (a_Po - (a_W + a_N)) *
                         T_prev[-1, -1] + S_u)/a_P

            self.plate.temperature = T

    def plot_heatmap(self, animated=False):
        if animated:
            anim = animation.FuncAnimation(
                plt.figure(), self._animate, interval=1, frames=solver.n_time_steps, repeat=False)
            anim.save('anim.gif')
        else:
            self._plot(self._time_temp_dist[-1], self.n_time_steps)
            plt.show()

    def _animate(self, time):
        self._plot(self._time_temp_dist[time], time)

    def _plot(self, temp_dist: NDArray[np.float64], n: int):
        plt.clf()

        plt.title(f"Temperature at t = {n*self.dt:.2f} s")

        plt.pcolormesh(temp_dist, cmap=plt.cm.jet, vmin=0, vmax=200)
        plt.colorbar()

        return plt

    @property
    def time_temp_dist(self):
        return self._time_temp_dist

    @time_temp_dist.setter
    def time_temp_dist(self, time_temp_dist: NDArray[np.float64]):
        self._time_temp_dist = time_temp_dist


plate = Plate(0.5, 0.5, 8850, 389, 385)
plate.discretise(100, 100)
plate.temperature = 200  # initial temperature
plate2 = plate.copy()
plate3 = plate.copy()

solver = FlatPlateDiffusionSolver(plate, 0.05, 40)
solver.set_boundary_condition(
    east_bc=BoundaryCondition('Dirichlet', 200),
    west_bc=BoundaryCondition('Dirichlet', 0),
    north_bc=BoundaryCondition('Dirichlet', 0),
    south_bc=BoundaryCondition('Dirichlet', 0)
)
solver.solve()
plate_time_temp_dist = solver.time_temp_dist.copy()
solver.plot_heatmap()


solver.plate = plate2
solver.west_bc = BoundaryCondition('Dirichlet', 200)
solver.east_bc = BoundaryCondition('Dirichlet', 0)
solver.solve()
plate2_time_temp_dist = solver.time_temp_dist.copy()
solver.plot_heatmap()

# Superposition
superposition = np.copy(plate_time_temp_dist)
superposition[:] = superposition[:]
superposition[:] += plate2_time_temp_dist[:]

solver.time_temp_dist = superposition
solver.plot_heatmap()

# Combined
solver.set_boundary_condition(
    east_bc=BoundaryCondition('Dirichlet', 200),
    west_bc=BoundaryCondition('Dirichlet', 200),
    north_bc=BoundaryCondition('Dirichlet', 0),
    south_bc=BoundaryCondition('Dirichlet', 0)
)

solver.plate = plate3
solver.solve()
solver.plot_heatmap(True)
