from typing import Literal, Callable
import numpy as np
from numpy.typing import NDArray


class BoundaryCondition:
    '''
    A simple model of a boundary condition.

    Attributes
    ----------
        - bc_type (Literal['Nuemann', 'Dirichlet']): The type of boundary condition.
        - func (Callable[[NDArray[np.float64]], NDArray[np.float64]]): The function describing the boundary condition.
    '''

    def __init__(self, bc_type: Literal['Nuemann', 'Dirichlet'], func: Callable[[float], float]):
        '''
        Initialise a new boundary condition with the type and value.

        Parameters
        ----------
            - bc_type (Literal['Nuemann', 'Dirichlet']): The type of boundary condition.
            - func (Callable[[float], float]): The function describing the boundary condition.
        '''
        self.func = np.vectorize(func)
        self.bc_type = bc_type
