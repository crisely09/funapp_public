"""Meijer G Approximant."""

import numpy as np

from funapp.tools import factorial

def borel_trans_coeffs(zn):
    r"""Get the coefficients of the Borel transformed series.

    Parameters
    ----------
    zn : np.ndarray
        Array with Taylor/Maclaurin series to be transformed.

    Returns
    -------
    bn : Array with the Borel transform coefficients :math:`b_n = z_n / n!`
    """
    bn = np.zeros(zn.shape)
    for i, z in enumerate(zn):
        bn[i] = z/factorial(i)
    return bn

