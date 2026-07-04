"""Small, generic array helpers shared across modules with no natural common base class."""


def vector_to_1D_array(array):
    """
    Turns arrays that are shape (n,) into (n, 1) arrays

    Parameters
    ----------
    array: np.ndarray
        Array of n dimensions

    Returns
    -------
    array: np.ndarray
        If n > 1, return original array. Otherwise, return 2D array with shape (-1,n)
    """
    # If array is not 2D, give it shape (len(array), 1)
    if not len(array.shape) > 1:
        array = array.reshape(-1, 1)
    return array
