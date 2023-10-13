import numpy as np
from typing import Sequence, Callable


class GridRange:
    def __init__(self, start: float, end: float, n_points: int):
        self.start = start
        self.end = end
        self.n_points = n_points

    def as_tuple(self) -> tuple[float, float, int]:
        return self.start, self.end, self.n_points


class VectorField:
    def __init__(self, grid_ranges: list[GridRange]):
        self.mesh_grids = np.meshgrid(
            *[np.linspace(*grid_range.as_tuple()) for grid_range in grid_ranges])

    def set_from_function(self, func: Callable[[tuple[float, ...]], tuple[float, ...]]) -> None:
        # TODO: Check func arguments and return values
        vectorized_func = np.vectorize(func)

        self.mesh_grids_values = vectorized_func(*self.mesh_grids)
        self.combined_grid_values = np.sum(self.mesh_grids_values, axis=0)


vf = VectorField([GridRange(0, 5, 5), GridRange(0, 5, 5)])
vf.set_from_function(lambda x, y: (x**2, y**2))
