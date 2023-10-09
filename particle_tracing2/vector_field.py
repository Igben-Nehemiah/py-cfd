import numpy as np
from typing import Tuple


class VectorField:
    def __init__(self, shape: Tuple) -> None:
        self.shape = shape
        self.n_components = len(self.shape)
        self.field = np.zeros(shape + (self.n_components,), dtype=np.float64)

    def set_vector(self, position, vector):
        indices = tuple(position)
        self.field[indices] = vector

    def set_from_function(self, func):
        for indices in np.ndindex(self.shape):
            vector = func(*indices)

            if len(vector) != self.n_components:
                raise ValueError()

            self.field[indices] = vector

    def get_vector(self, position):
        indices = tuple(position)
        return self.field[indices]

    def scale_field(self, scaler):
        self.field *= scaler

    def normalise_field(self):
        magnitudes = np.linalg.norm(self.field, axis=-1)
        magnitudes[magnitudes == 0] = 1
        self.field /= magnitudes[..., np.newaxis]

    def __str__(self) -> str:
        return f'VectorField instance with shape: {self.shape}, n_components: {self.n_components}'

    def __repr__(self) -> str:
        return f'VectorField({self.shape})'
