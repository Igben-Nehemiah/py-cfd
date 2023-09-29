import numpy as np
from numpy.typing import NDArray
from copy import deepcopy


class RectangularPlate:
    '''
    A simple model of a rectangular and flat plate.

    Attributes
    ----------
        - width (float): The width of the plate in m.
        - height (float): The heigth of the plate in m.
        - rho (float): The density of the plate in kg/m3.
        - cp (float): The specific heat capacity of the plate in J/kg/K.
        - k (float): The thermal conductivity of the plate in W/m/K.

    Properties
    ----------
        - temperature (NDArray[np.float64]): The temperature distribution at every nodal point in the plate.

    Methods
    -------
        - dicretise(nx_nodes, ny_nodes): Discretises the plate into nx_nodes and ny_nodes number of x and y nodes respectively.
        - copy(): Copies plate's properties and returns copied plate.

    Example
    -------
        >>> plate = RectangularPlate(width=0.5, height=0.5, rho=8850, cp=389, k=385)
        >>> plate.discretise(100, 100)
    '''

    def __init__(self, width: float, height: float, rho: float, cp: float, k: float):
        ''''
        Initialise a RectangularPlate instance with the given data.

        Parameters
        ----------
            - width (float): The width of the plate in m.
            - height (float): The heigth of the plate in m.
            - rho (float): The density of the plate in kg/m3.
            - cp (float): The specific heat capacity of the plate in J/kg/K.
            - k (float): The thermal conductivity of the plate in W/m/K.              
        '''
        self.width = width
        self.height = height
        self.rho = rho
        self.cp = cp
        self.k = k

    def discretise(self, nx_nodes: int, ny_nodes: int):
        ''''
        Discretises the plate into nx_nodes and ny_nodes number of x and y nodes respectively.

        Parameters
        ----------
            - nx_nodes (int): Number of x nodes.
            - ny_nodes (int): Number of y nodes.
        '''
        self.nx_nodes = nx_nodes
        self.ny_nodes = ny_nodes
        self.dx = self.width/nx_nodes
        self.dy = self.height/ny_nodes
        self._temperature = np.empty((nx_nodes, ny_nodes))

    @property
    def temperature(self):
        '''
        The temperature distribution at every nodal point in the plate.
        '''
        return self._temperature

    @temperature.setter
    def temperature(self, temp: NDArray[np.float64] | float):
        '''
        Parameters
        ----------
            - temp (NDArray[np.float64 | float]): The temperature distribution at every nodal point or the uniform temperature across
                                                all nodal points.

        Raises
        ------
            ValueError: If the plate has not been discretised or when the temperature shapes do not match.

        '''
        if self._temperature is None:
            raise ValueError("Discretise plate before setting temperature")

        if isinstance(temp, (int, float)):
            self._temperature[:, :] = temp
            return

        if temp.shape != self._temperature.shape:
            raise ValueError("Temperature shapes do not match")

        self._temperature = temp

    def copy(self):
        '''
        Copies the instance of a RectangularPlate.

        Returns
        -------
            FlatPlate: A deep copy of the instance.
        '''
        return deepcopy(self)
