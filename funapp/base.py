"""
Base classes for function approximants.
"""

import numpy as np
import scipy as sp
from scipy import linalg as sl

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
    _ider: int or list(int)
        If derivatives are provided,`_ider` stores the location of the `x`
        value.

    Methods
    -------

    __call__(x, der)
    _get_nder(x, y)
    _evaluate(xi, der)
    _derivate(der)

    """
    __slots__ = ('_xlen', '_ider', '_x', '_y')

    def __init__(self, x, y):
        # Check the sizes of the arrays and store local variables.
        x = np.ravel(x)
        y = np.ravel(y)
        x, y = self._clean_arrays(x, y)
        self._xlen = len(x)
        self._get_ider(x)
        self._x = x
        self._y = y

    def __call__(self, xi, der=0):
        """
        Evaluate the approximant and derivatives

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
        y = self._evaluate(xi, der)
        return y

    def _clean_arrays(self, x, y):
        """
        Remove any repeated value from arrays.
        """
        same = []; xused = []
        x = list(x); y = list(y)
        l = zip(x, y)
        same = [[i,l[i]] for i in range(len(l)) if l.count(l[i]) > 1]
        seen = []; rep = []
        for i, s in enumerate(same):
            if s[1] not in seen:
                rep.append(same[i][0])
                seen.append(s[1])
        xnew = np.delete(x, rep)
        ynew = np.delete(y, rep)
        return xnew, ynew

    def _get_ider(self, x):
        """
        Check if any derivative was provided as input and save the
        location in `_ider` 

        `_ider` has the following form
        [x, [yi, ...], ] where yi is the position in y that
        corresponds to the value of the ith-derivative of f(x),
        excluding the 0th derivative.
        """
        ider = [] ; xused = []
        for xj in x:
            if xj not in xused:
                # Get all repited values
                unique = np.where(abs(x-xj)<1e-10)[0]
                # Remove 0th derivative
                if len(unique) > 1:
                    ider.append([xj, list(unique[1:])])
            xused.append(xj)
        self._ider = ider

    def _evaluate(self, xi, der):
        """
        Actual evaluation of the function approximant.
        """
        raise NotImplementedError()

    def _derivate(self, nder):
        """
        Evaluate derivatives of the function approximant.
        """
        raise NotImplementedError()
