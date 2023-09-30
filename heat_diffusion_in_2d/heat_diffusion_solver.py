import numpy as np
from numpy.typing import NDArray
from rectangular_plate import RectangularPlate
from boundary_condition import BoundaryCondition


class HeatDiffusionSolver:
    '''
    A class for solving heat diffusion problems on a rectangular plate.

    This class encapsulates the solver for simulating heat diffusion on a rectangular
    plate over a specified time interval.

    Parameters
    ----------
        - dt (float): The time step size for the simulation.
        - time_interval (float): The total time interval for the simulation.

    Attributes
    ----------
        - dt (float): The time step size for the simulation.
        - plate (RectangularPlate): The rectangular plate on which heat diffusion is simulated.
        - time_interval (float): The total time interval for the simulation.
        - n_time_steps (int): The number of time steps calculated based on the time interval.

    Example
    -------
        >>> # Create a RectangularPlate object and set its properties.
        >>> plate = RectangularPlate(width=0.5, height=0.5, rho=8850, cp=389, k=385)
        >>> #Create a HeatDiffusionSolver instance with a time step of 0.01 and a total time interval of 400.0.
        >>> solver = HeatDiffusionSolver(dt=0.01, time_interval=400.0)
        >>> solver.plate = plate
        >>> # Use the solver to simulate heat diffusion on the plate.
        >>> solver.solve()
    '''

    def __init__(self, dt: float, time_interval: float):
        '''
        Initialise a HeatDiffusionSolver instance.

        Parameters
        ----------
            - dt (float): The time step size for the simulation.
            - time_interval (float): The total time interval for the simulation.
        '''
        self.dt = dt
        self.time_interval = time_interval
        self.n_time_steps = int(self.time_interval / dt) + 1

    def set_boundary_condition(self, west_bc: BoundaryCondition,
                               east_bc: BoundaryCondition,
                               north_bc: BoundaryCondition,
                               south_bc: BoundaryCondition):
        '''
        Set boundary conditions for the rectangular plate.

        This method allows you to specify boundary conditions for the rectangular plate
        along its west, east, north, and south sides.

        Parameters
        ----------
            - west_bc (BoundaryCondition): The boundary condition for the west side of the plate.
            - east_bc (BoundaryCondition): The boundary condition for the east side of the plate.
            - north_bc (BoundaryCondition): The boundary condition for the north side of the plate.
            - south_bc (BoundaryCondition): The boundary condition for the south side of the plate.
        '''
        self.west_bc = west_bc
        self.east_bc = east_bc
        self.north_bc = north_bc
        self.south_bc = south_bc

    def __critical_time_step(self):
        return self.plate.rho*self.plate.cp*self.plate.dx**2*self.plate.dy**2/(2*self.plate.k*(self.plate.dx**2 + self.plate.dy**2))

    def solve(self):
        '''
        Solve the 2D unsteady-state heat equation using the finite volume method.

        This method iteratively calculates the temperature distribution on a 2D plate
        over multiple time steps. The finite volume method is used to update the
        temperature at each node.

        Raises
        ------
            ArithmeticError: If the time step is greater than the critical time step.
            ValueError: If the plate is not set.

        Notes
        -----
            - The plate properties (dimensions, thermal conductivity, heat capacity, density)
            and boundary conditions should be set before calling this method.
            - The initial temperature distribution is set based on the plate's initial state.
            - The plate should be set on the solver.

        Returns
        -------
            None

        Example
        -------
            >>> # Create a Plate object and set its properties and boundary conditions.
            >>> plate = Plate(0.5, 0.5, 8850, 389, 385)
            >>> plate.discretise(100, 100)
            >>> plate.temperature = 200  # Initial condition

            >>> # Set boundary conditions
            >>> west_condition = BoundaryCondition(lambda x: 50, type='Dirichlet')
            >>> east_condition = BoundaryCondition(lambda x: 50, type='Dirichlet')
            >>> north_condition = BoundaryCondition(lambda x: 0, type='Dirichlet')
            >>> south_condition = BoundaryCondition(lambda x: 50, type='Dirichlet')

            >>> # Initialise Solver
            >>> solver = HeatDiffusionSolver(0.05, 400)
            >>> solver.set_boundary_condition(west_condition, east_condition, north_condition, south_condition)
            >>> solver.plate = plate
            >>> # Solve the heat equation over a specified number of time steps.
            >>> solver.solve()
        '''
        if self.plate is None:
            raise ValueError("Solver's plate has not been set")

        if self.dt > self.__critical_time_step():
            raise ArithmeticError(
                f'The time step is greater than the critical time step {self.__critical_time_step()} ')

        # This will hold the temperature distribution of the plate at different times
        self._time_temp_dist = np.empty(
            (self.n_time_steps, self.plate.nx_nodes, self.plate.ny_nodes))

        # Set initial condition
        self._time_temp_dist[0] = self.plate.temperature

        dx, dy = self.plate.dx, self.plate.dy
        k, cp, rho = self.plate.k, self.plate.cp, self.plate.rho

        w_nodal_points = e_nodal_points = np.arange(
            dy/2, self.plate.height, dy)
        n_nodal_points = s_nodal_points = np.arange(
            dx/2, self.plate.width, dx)

        T_W, T_E, T_N, T_S = self.west_bc.func(w_nodal_points), self.east_bc.func(
            e_nodal_points), self.north_bc.func(n_nodal_points), self.south_bc.func(s_nodal_points)

        aE = aW = k * dy / dx
        aN = aS = k * dx / dy
        aPo = rho * cp * dx * dy / self.dt
        aP = aPo

        t_step = 0  # Current time step

        while (t_step < self.n_time_steps - 1):
            t_step += 1

            # Previous time temperature nodes
            T_prev = self.plate.temperature

            # Current time temperature nodes
            T = self._time_temp_dist[t_step]

            # Internal Nodes
            T[1:-1, 1:-1] = (aE*T_prev[1:-1, 2:] + aW*T_prev[1:-1, 0:-2] + aN*T_prev[0:-2, 1:-1] +
                             aS*T_prev[2:, 1:-1] + (aPo - (aW + aE + aN + aS))*T_prev[1:-1, 1:-1])/aP

            # Western Face
            S_u = 2*k*dy*(T_W[1:-1] - T_prev[1:-1, 0])/dx
            T[1:-1, 0] = (aE*T_prev[1:-1, 1] + aN*T_prev[0:-2, 0] +
                          aS*T_prev[2:, 0] + (aPo - (aE + aN + aS))*T_prev[1:-1, 0] + S_u)/aP

            # Eastern Face
            S_u = 2*k*dy*(T_E[1:-1] - T_prev[1:-1, -1])/dx
            T[1:-1, -1] = (aW*T_prev[1:-1, -2] + aN*T_prev[0:-2, -1] +
                           aS*T_prev[2:, -1] + (aPo - (aW + aN + aS))*T_prev[1:-1, -1] + S_u)/aP

            # Northern Face
            S_u = 2*k*dx*(T_N[1:-1] - T_prev[0, 1:-1])/dy
            T[0, 1:-1] = (aE*T_prev[0, 2:] + aW*T_prev[0, 0:-2] +
                          aS*T_prev[0, 1:-1] + (aPo - (aE + aW + aS))*T_prev[0, 1:-1] + S_u)/aP

            # Southern Face
            S_u = 2*k*dx*(T_S[1:-1] - T_prev[-1, 1:-1])/dy
            T[-1, 1:-1] = (aE*T_prev[-1, 2:] + aW*T_prev[-1, 0:-2] +
                           aN*T_prev[-1, 1:-1] + (aPo - (aE + aW + aN))*T_prev[-1, 1:-1] + S_u)/aP

            # North-West Corner
            S_u = 2*k*dy*(T_W[-1] - T_prev[0, 0])/dx + \
                2*k*dx*(T_N[0] - T_prev[0,  0])/dy
            T[0, 0] = (aE*T_prev[0, 1] + aS*T_prev[1, 0] + (aPo - (aE + aS)) *
                       T_prev[0, 0] + S_u)/aP

            # South-West Corner
            S_u = 2*k*dy*(T_W[0] - T_prev[-1, 0])/dx + \
                2*k*dx*(T_S[0] - T_prev[-1,  0])/dy
            T[-1, 0] = (aE*T_prev[-1, 1] + aN*T_prev[-2, 0] + (aPo - (aE + aN)) *
                        T_prev[-1, 0] + S_u)/aP

            # North-East Corner
            S_u = 2*k*dy*(T_E[-1] - T_prev[0, -1])/dx + \
                2*k*dx*(T_N[-1] - T_prev[0,  -1])/dy
            T[0, -1] = (aW*T_prev[0, -2] + aS*T_prev[1, -1] + (aPo - (aW + aS)) *
                        T_prev[0, -1] + S_u)/aP

            # South-East Corner
            S_u = 2*k*dy*(T_E[0] - T_prev[-1, -1])/dx + \
                2*k*dx*(T_S[-1] - T_prev[-1,  -1])/dy
            T[-1, -1] = (aW*T_prev[-1, -2] + aN*T_prev[-2, -1] + (aPo - (aW + aN)) *
                         T_prev[-1, -1] + S_u)/aP

            self.plate.temperature = T

    @property
    def time_temp_dist(self):
        '''
        The temperature distribution in time at every nodal point in the plate.
        '''
        return self._time_temp_dist

    @time_temp_dist.setter
    def time_temp_dist(self, time_temp_dist: NDArray[np.float64]):
        '''
        Parameters
        ----------
            - time_temp_dist: (NDArray[np.float64 | float]): The temperature distribution in time at every nodal point.
        '''
        self._time_temp_dist = time_temp_dist
