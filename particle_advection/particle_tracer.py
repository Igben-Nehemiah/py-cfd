import numpy as np
from vector_field import VectorField
from particle_advection.trace_method import TraceMethod



class ParticleTracer:
    """
    A utility for tracing the movement of particles through a vector field over time using various numerical integration methods.

    Parameters
    ----------
    - field: VectorField
        The vector field to trace particles in.
    - initial_positions: array-like
        The initial positions of the particles.
    - dt: float
        The time step for tracing.
    - n_time_steps: int
        The number of time steps to simulate.

    Example
    -------
    ```
    # Create vector field
    field = VectorField([GridRange(-2, 2, 5), GridRange(-2, 2, 5)])
    field.set_from_function(lambda x, y: (x**2, y**2))

    # Initialise ParticleTracer with created vector field
    tracer = ParticleTracer(field=field, initial_positions=[(0.5, 0.5), (0.5, 0.6)], dt=0.01, n_time_steps=100)

    positions_evolution = tracer.trace()
    ```
    """

    def __init__(self, field: VectorField, initial_positions, dt, n_time_steps):
        """
        Initialise a ParticleTracer instance.

        Parameters
        ----------
        - field: VectorField
            The vector field to trace particles in.
        - initial_positions: array-like
            The initial positions of the particles.
        - dt: float
            The time step for tracing.
        - n_time_steps: int
            The number of time steps to simulate.

        Example
        -------
        tracer = ParticleTracer(field_instance, initial_positions, 0.01, 100)
        """
        self.dt = dt
        self.field = field
        self.positions = np.array(initial_positions)
        self.n_time_steps = n_time_steps
        self.positions_evolution = np.empty(
            (n_time_steps + 1, ) + self.positions.shape)
        self.positions_evolution[0] = self.positions.copy()

    def trace(self, method: TraceMethod = TraceMethod.EULER):
        """
        Trace particles using a specified method.

        Parameters
        ----------
        - method: TraceMethod (optional)
            The tracing method (default: Euler).

        Example
        -------
        tracer.trace(TraceMethod.RK4)
        """
        match method:
            case TraceMethod.EULER:
                self.__trace_euler()
            case TraceMethod.MODIFIED_EULER:
                self.__trace_modified_euler()
            case TraceMethod.RK4:
                self.__trace_rk4()
            case _:
                raise ValueError("Invalid trace method")

    def __trace_euler(self):
        """
        Trace particles using the Euler method.
        """
        self.__initialise_positions()

        for i in range(1, self.n_time_steps + 1):
            v0 = self.field.get_vectors_at_positions(self.positions)
            self.positions += self.dt * v0
            self.positions_evolution[i] = self.positions

    def __trace_modified_euler(self):
        """
        Trace particles using the Modified Euler method.
        """
        self.__initialise_positions()

        for i in range(1, self.n_time_steps + 1):
            # Calculate the vector at the current positions
            v0 = self.field.get_vectors_at_positions(self.positions)

            # Calculate intermediate positions using a full Euler step
            p1 = self.positions + self.dt * v0

            # Calculate the vector at the intermediate positions
            v1 = self.field.get_vectors_at_positions(p1)

            # Update the positions for the next iteration using the modified Euler formula
            self.positions += 0.5 * self.dt * (v0 + v1)

            # Store the updated positions
            self.positions_evolution[i] = self.positions

    def __trace_rk4(self):
        """
        Trace particles using the Runge-Kutta 4th order method.
        """
        self.__initialise_positions()

        for i in range(1, self.n_time_steps + 1):
            # Calculate the vector at the current positions
            v0 = self.field.get_vectors_at_positions(self.positions)

            # Calculate k1
            k1 = self.dt * v0

            # Calculate p1, v1, and k2
            p1 = self.positions + 0.5 * k1
            v1 = self.field.get_vectors_at_positions(p1)
            k2 = self.dt * v1

            # Calculate p2, v2, and k3
            p2 = self.positions + 0.5 * k2
            v2 = self.field.get_vectors_at_positions(p2)
            k3 = self.dt * v2

            # Calculate p3, v3, and k4
            p3 = self.positions + k3
            v3 = self.field.get_vectors_at_positions(p3)
            k4 = self.dt * v3

            # Update the positions using a weighted sum of k1, k2, k3, and k4
            self.positions += (1/6) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Store the updated positions
            self.positions_evolution[i] = self.positions

    def __initialise_positions(self):
        self.positions = self.positions_evolution[0].copy()
