"""Chebyshev scaled model from a Pade approximant."""

import numpy as np
from scipy import linalg, optimize

from funapp.base import BaseApproximant


class ChebyshevApproximant(BaseApproximant):
    """
    Attributes
    ----------
    _m, _n : int
        Order of numerator and denominator series
        of the Pade approximant.
    _numcheb, _dencheb : np.polynomial.chebyshev.chebval
        Chebyshev polynomials for the model.
    _xlen : int
        Length of the array with points.
    _der : int or list(int)
        If derivatives are provided,`_der` stores the values in a different
        array.
    _pders : list
        List of all derivatives, up to order _m.
    _qders : list
        List of all derivatives, up to order _n.
    eps : float
        The little change to be made for the derivative approximation.

    Methods
    -------
    __call__
    _evaluate
    _derivate

    """

    def __init__(self, x, y, m, n):
        """Initialize base class for Pade Approximants.

        Parameters
        ----------
        x : np.ndarray
            Points where the function was evaluated.
        y : np.ndarray
            Values of the function, and/or function
            derivatives evaluated at `x`.
        m, n : int
            Order of numerator and denominator series
            of the Pade approximant.

        """
        # Check arrays and initialize base class
        BaseApproximant.__init__(self, x, y)
        if not isinstance(m, int):
            raise TypeError('m must be integer.')
        if not isinstance(n, int):
            raise TypeError('m must be integer.')
        self._m = m
        self._n = n

    def __call__(self, x):
        """Evaluate the Chebyshev approximant at the points `x`.

        Arguments
        ---------
        x : array like
            Array with points where to evaluate the approximant

        Returns
        -------
        Array with the values of the approximant at each point given..

        """
        if isinstance(x, list):
            x = np.array(x, dtype=float)
        elif not isinstance(x, np.ndarray):
            raise TypeError('x must be given as a list or as np.ndarray.')
        result = self._evaluate(x)
        return result

