class GridRange:
    """
    Represents a range within a grid with specified start, end, and number of points.

    Parameters
    ----------
        start (float): The start value of the range.
        end (float): The end value of the range.
        n_points (int): The number of points within the range.

    Returns
    -------
        GridRange: A GridRange object.

    Example
    -------
    ```
        # Create a GridRange for a range from 0.0 to 1.0 with 5 points.
        range_0_to_1 = GridRange(0.0, 1.0, 5)

        # Retrieve the range as a tuple.
        range_tuple = range_0_to_1.as_tuple()
        # range_tuple is now (0.0, 1.0, 5).
    ```
    """

    def __init__(self, start: float, end: float, n_points: int):
        self.start = start
        self.end = end
        self.n_points = n_points

    def as_tuple(self) -> tuple[float, float, int]:
        """
        Returns the grid range as a tuple of start, end, and number of points.

        Returns
        -------
            tuple[float, float, int]: A tuple representing the grid range.
        """
        return self.start, self.end, self.n_points
