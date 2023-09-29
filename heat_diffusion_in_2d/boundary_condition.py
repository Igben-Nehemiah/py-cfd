from typing import Literal


class BoundaryCondition:
    '''
    A simple model of a boundary condition.

    Attributes
    ----------
        - bc_type (Literal['Nuemann', 'Dirichlet']): The type of boundary condition.
        - value (float): The value of the function at this boundary condition.
    '''

    def __init__(self, bc_type: Literal['Nuemann', 'Dirichlet'], value: float):
        '''
        Initialise a new boundary condition with the type and value.

        Parameters
        ----------
            - bc_type (Literal['Nuemann', 'Dirichlet']): The type of boundary condition.
            - value (float): The value of the function at this boundary condition.
        '''
        self.value = value
        self.bc_type = bc_type
