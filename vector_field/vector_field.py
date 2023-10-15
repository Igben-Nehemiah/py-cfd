import numpy as np
from scipy.interpolate import RegularGridInterpolator, interpn
from vector_field.grid_range import GridRange
from typing import Sequence, Callable
import inspect


class VectorField:
    """
    Represents a vector field in a grid and provides methods for interpolation.

    Parameters
    ----------
        - grid_ranges (List[GridRange]): A list of GridRange objects defining grid dimensions.

    Attributes
    ----------
        - mesh_positions (List[np.ndarray]): List of grid positions along each dimension.
        - mesh_grids (Tuple[np.ndarray]): Mesh grids created from grid positions.
        - grid_ranges (List[GridRange]): A list of GridRange objects defining grid dimensions.

    Example
    -------
    ```
    # Create a VectorField with two grid dimensions, each ranging from 0 to 5 with 6 positions.
    vf = VectorField([GridRange(0, 5, 6), GridRange(0, 5, 6)])

    # Set the VectorField from a function that computes vector values.
    vf.set_from_function(lambda x, y: (x**2, y**2))

    # Get vectors at specified positions.
    vectors = vf.get_vectors_at_positions([(1, 2), (2, 4), (2, 2), (3, 1)])
    ```
    """

    def __init__(self, grid_ranges: list[GridRange]):
        # Generate grid positions from grid ranges
        self.mesh_positions = [np.linspace(*grid_range.as_tuple())
                               for grid_range in grid_ranges]

        self.grid_ranges = grid_ranges

        # Create mesh grids from the grid positions
        self.mesh_grids = np.meshgrid(*self.mesh_positions, indexing='ij')

    @property
    def dim(self):
        return len(self.mesh_grids)

    def set_grid_values(self, mesh_grids_values: np.ndarray) -> None:
        """
        Set the VectorField's grid values directly from a collection of mesh grid values.

        Parameters
        ----------
            - mesh_grids_values (np.ndarray): A collection of mesh grid values for each dimension.

        Raises
        ------
            ValueError: If the provided mesh grid values do not match the number of grid dimensions.
        """
        if len(mesh_grids_values) != len(self.mesh_positions):
            raise ValueError(
                "The provided mesh grid values do not match the number of grid dimensions.")

        # Set the mesh grid values
        self.mesh_grids_values = np.array(mesh_grids_values)

    def set_from_function(self, func: Callable[[tuple[float, ...]], tuple[float, ...]]) -> None:
        """
        Set the VectorField from a function that computes vector values.

        Parameters
        ----------
            - func (Callable[[Tuple[float, ...]], Tuple[float, ...]): A function that computes vector values.

        Raises
        ------
            ValueError: If the function doesn't accept the same number of arguments as grid dimensions.
        """
        # Check if the function can accept the same number of arguments as the grid dimensions
        required_args = len(self.mesh_positions)
        func_args = len(inspect.signature(func).parameters)
        if func_args != required_args:
            raise ValueError("Function should accept {} arguments, but it accepts {}.".format(
                required_args, func_args))

        # Calculate grid values using the provided function
        self.mesh_grids_values = np.array(func(*self.mesh_grids))

    def get_vectors_at_positions(self, positions: Sequence[np.array]) -> Sequence[np.array]:
        """
        Interpolate vectors at specified positions within the grid.

        Parameters
        ----------
            - positions (Sequence[np.ndarray]): A sequence of NumPy arrays representing positions for which to interpolate vectors.

        Returns
        -------
            Sequence[np.ndarray]: A sequence of NumPy arrays containing the interpolated vectors at the specified positions.

        Raises
        ------
            ValueError: If the dimensionality of the specified positions doesn't match the grid dimensions.
        """
        # Check the positions for dimensionality
        if any(len(position) != len(self.mesh_positions) for position in positions):
            raise ValueError(
                "position dimension does not match the grid dimension.")

        # Interpolate vectors at the specified positions using the interpolators
        interpolated_vectors = [interpn(
            self.mesh_positions, values, positions, bounds_error=False) for values in self.mesh_grids_values]

        # Transpose the result to have one NDArray per position
        return np.array(interpolated_vectors).T
