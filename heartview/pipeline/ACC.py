import numpy as np
import pandas as pd

def compute_magnitude(x, y, z):
    """Compute the magnitude of a 3-axis accelerometer signal.

    Parameters
    ----------
    x : array_like
        An array containing the x-axis accelerometer data.
    y : array_like
        An array containing the y-axis accelerometer data.
    z : array_like
        An array containing the z-axis accelerometer data.

    Returns
    -------
    magnitude : pd.Series
        A series of acceleration magnitude values.
    """
    square_roots = np.sqrt([x, y, z]).T
    magnitude = pd.DataFrame(square_roots).apply(
        lambda x: x ** 2).sum(axis = 1)
    return magnitude