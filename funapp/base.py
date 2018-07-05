"""
Base classes for function approximants.
"""

import numpy as np
import scipy as sp
from scipy import linalg as sl
from funapp.tools import check_input, clean_arrays


__all__ = ['BaseApproximant',]


class BaseApproximant(object):
    """
    Base class of function approximants.

    Interpolate/extrapolate univariate functions.

    BaseApproximant(`x`, `y`)

    `x` and `y` are arrays with values used to approximate
    some function f: ``y = f(x)``.

    Including derivatives as input:
    If `x` has one element repeted in the list, the values of `y` at the 
    same position are assumed to be the derivatives of the function at
    that point, i.e. if x[m] = 1.5 and x[m+1] = 1.5, then y[m] = f(x) and
    y[m+1] = f'(x), etc.


    Attributes
    ----------

    _xlen: int
        Length of input points where the function was evaluated.
    _der: int or list(int)
        If derivatives are provided,`_der` stores the values in a
        different array.

    Methods
    -------

    __call__(x, der)
    _get_nder(x, y)
    _evaluate(xi)
    _derivate(nder)

    """
    __slots__ = ('_xlen', '_der', '_x', '_y')

    def __init__(self, x, y):
        """Initialize the class.

        Parameters:
        -----------
        x: np.ndarray((N,), dtype=float)
            Points where the function is evaluated.
        y: np.ndarray((N,), dtype=float)
            Value of the function at points x.

        """
        # Check the sizes of the arrays and store local variables.
        x = np.ravel(x)
        y = np.ravel(y)
        check_input(x, y)
        x, y = clean_arrays(x, y)
        self._xlen = len(x)
        self._get_ders(x, y)
        self._prune_arrays(x, y)

    def __call__(self, xi, der=0):
        """
        Evaluate the approximant and derivatives.

        Parameters
        ----------
        xi: array like
            Array of points where to evaluate the approximant
        der: int or list(int)
            Number, or list of numbers of the derivative(s) to extract,
            evaluation of the function corresponds to the 0th derivative.

        Returns
        -------
        y: array, shape (len(xi), len(der))
            Values of the approximant and derivatives at each point of xi.

        """
        # Make xi 1D-array
        xi = np.ravel(xi)

        # Evaluate function and derivatives
        if der == 0:
            y = self._evaluate(xi)
        else:
            y = self._derivate(xi, der)
        return y

    def _prune_arrays(self, x, y):
        """Remove derivatives from arrays.

        Parameters
        ----------
        x : np.ndarray
            Points where the function is evaluated.
        y : np.ndarray
            Value of the function at points x.

        """
        same = []; xused = []
        x = list(x); y = list(y)
        same = [i for i in range(len(x)) if x.count(x[i]) > 1]
        repv = set()
        rep = []
        for i in same:
            if x[i] not in repv:
                repv.add(x[i])
            else:
                rep.append(i)
        self._x = np.delete(x, rep)
        self._y = np.delete(y, rep)

    def _get_ders(self, x, y):
        """
        Check if any derivative was provided as input and save the
        location in `_der`.

        `_der` has the following form
        [x, [yi, ...], ] where yi is the value of the ith-derivative of f(x),
        excluding the 0th derivative.

        """
        ders = [] ; xused = []
        count = 0
        for xj in x:
            if xj not in xused:
                # Get all repited values
                unique = np.where(abs(x-xj)<1e-10)[0]
                # Remove 0th derivative
                if len(unique) > 1:
                    count += len(y[unique[1:]])
                    ders.append([xj, list(y[unique[1:]])])
            xused.append(xj)
        self._ders = ders
        self._dlen = count

    def _evaluate(self, xi):
        """
        Actual evaluation of the function approximant.

        Parameters
        ----------
        xi : array-like
            Points where the approximant will be evaluated.
        """
        raise NotImplementedError()

    def _derivate(self, xi, der):
        """
        Evaluate the Pade Approximant derivatives at `x`.

        Parameters
        ----------
        x : array-like
            Points where the approximant will be evaluated.
        der: int
            Order of the derivative(s) to extract.

        Returns
        Array with values of the derivatives of x.
        """
        raise NotImplementedError()
