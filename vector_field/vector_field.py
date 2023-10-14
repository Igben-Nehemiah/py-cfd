import numpy as np
from scipy.interpolate import RegularGridInterpolator
from grid_range import GridRange
from typing import Sequence, Callable
import inspect


class VectorField:
    """
    Represents a vector field in a grid and provides methods for interpolation.

    Parameters
    ----------
        grid_ranges (List[GridRange]): A list of GridRange objects defining grid dimensions.

    Attributes
    ----------
        mesh_points (List[np.ndarray]): List of grid points along each dimension.
        mesh_grids (Tuple[np.ndarray]): Mesh grids created from grid points.
        interpolators (List[RegularGridInterpolator]): Interpolators for each grid.

    Example
    -------
    ```
        # Create a VectorField with two grid dimensions, each ranging from 0 to 5 with 6 points.
        vf = VectorField([GridRange(0, 5, 6), GridRange(0, 5, 6)])

        # Set the VectorField from a function that computes vector values.
        vf.set_from_function(lambda args: (args[0]**2, args[1]**2))

        # Get vectors at specified points.
        vectors = vf.get_vectors_at_points([(1, 2), (2, 4), (2, 2), (3, 1)])
    ```
    """

    def __init__(self, grid_ranges: list[GridRange]):
        # Generate grid points from grid ranges
        self.mesh_points = [np.linspace(*grid_range.as_tuple())
                            for grid_range in grid_ranges]

        # Create mesh grids from the grid points
        self.mesh_grids = np.meshgrid(*self.mesh_points, indexing='ij')

        # Initialize interpolators
        self.interpolators = None

    def set_from_function(self, func: Callable[[tuple[float, ...]], tuple[float, ...]]) -> None:
        """
        Set the VectorField from a function that computes vector values.

        Parameters
        ----------
            func (Callable[[Tuple[float, ...]], Tuple[float, ...]): A function that computes vector values.

        Raises
        ------
            ValueError: If the function doesn't accept the same number of arguments as grid dimensions.
        """
        # Check if the function can accept the same number of arguments as the grid dimensions
        required_args = len(self.mesh_points)
        func_args = len(inspect.signature(func).parameters)
        if func_args != required_args:
            raise ValueError("Function should accept {} arguments, but it accepts {}.".format(
                required_args, func_args))

        # Calculate grid values using the provided function
        self.mesh_grids_values = np.array(func(*self.mesh_grids))

        # Sum the grid values to create a combined grid
        self.combined_grid_values = np.sum(self.mesh_grids_values, axis=0)

        # Create interpolators for each grid
        self.interpolators = [RegularGridInterpolator(
            tuple(self.mesh_points), grid_values) for grid_values in self.mesh_grids_values]

    def get_vectors_at_points(self, points: Sequence[tuple[float, ...]]) -> Sequence[tuple[float, ...]]:
        """
        Interpolate vectors at specified points within the grid.

        Parameters
        ----------
            points (Sequence[Tuple[float, ...]]): A sequence of points for which to interpolate vectors.

        Returns
        -------
            Sequence[Tuple[float, ...]: A sequence of interpolated vectors at the specified points.

        Raises
        ------
            ValueError: If the dimensionality of the specified points doesn't match the grid dimensions.
        """
        # Check the points for dimensionality
        if any(len(point) != len(self.mesh_points) for point in points):
            raise ValueError(
                "Point dimension does not match the grid dimension.")

        # Interpolate vectors at the specified points using the interpolators
        interpolated_vectors = [interpolator(
            points) for interpolator in self.interpolators]

        # Transpose the result to have one tuple per point
        return tuple(map(tuple, np.array(interpolated_vectors).T))
