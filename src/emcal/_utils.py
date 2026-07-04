"""Small, generic array helpers shared across modules with no natural common base class."""

import numpy as np


def blockwise_sse(y_vals, exp_y_vals, n_blocks, len_x):
    """
    Reshapes y_vals into n_blocks rows of len_x columns and computes the per-block
    residual against exp_y_vals and the summed-squared-error per block.

    Parameters
    ----------
    y_vals: np.ndarray
        Values to reshape into blocks, shape (n_blocks * len_x,)
    exp_y_vals: np.ndarray
        Experimental values to compare each block against, shape (len_x,)
    n_blocks: int
    len_x: int

    Returns
    -------
    sse: np.ndarray
        Summed squared error per block, shape (n_blocks,)
    block_errors: np.ndarray
        Raw (unsquared) per-block residuals, shape (n_blocks, len_x)
    """
    y_resh = y_vals.reshape(n_blocks, len_x)
    block_errors = y_resh - exp_y_vals[np.newaxis, :]
    sse = np.sum(block_errors**2, axis=1)
    return sse, block_errors


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
