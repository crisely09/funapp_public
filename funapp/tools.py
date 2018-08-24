"""Useful tools."""

import numpy as np
import itertools
from scipy import optimize

__all__ = ['check_input', 'clean_arrays']


def check_input(x, y):
    """Check type and length of arrays.

    Parameters
    ----------
    x : list or np.ndarray((N,), dtype=float)
        Points where the function is evaluated. 
    y : list or np.ndarray((N), dtype=float) 
        Value of the function at point x.

    Raises
    ------
    TypeError : if x and/or y are not np.ndarray(dtype=float).
    ValueError : when the arrays are not of the same length.

    """
    if isinstance(x, list):
        x = np.ndarray(x, dtype=float)
    elif not isinstance(x, np.ndarray):
        raise TypeError('x must be np.ndarray.')
    if isinstance(y, list):
        x = np.ndarray(x, dtype=float)
    elif not isinstance(y, np.ndarray):
        raise TypeError('y must be np.ndarray.')
    if x.size != y.size:
        raise ValueError('Arrays x and y should have the same size.')


def clean_arrays(x, y):
     """Remove any repeated value from arrays.
     
     Parameters
     ----------
     x : np.ndarray
         Points where the function is evaluated.
     y : np.ndarray
         Value of the function at points x.
     
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


def factorial(n):
    """Evaluate the factorial of n."""
    if n <= 0:
        return 1
    else:
        return n*factorial(n-1)


def choose_function(k, n):
    """Evaluate k choose n."""
    result = factorial(k)/factorial(k-n)/factorial(n)
    return result


def taylor(ders, n, eps, sgn):
    r"""Evaluate the Taylor series up to n-term.

    :math:`F(x \pm \epsilon) = F(x) \pm \epsilon F'(x) + 1/2 \epsilon^2 F''(x) + ...`

    Parameters
    ----------
    ders :  array-like, float
        Values of the function derivatives, including 0th derivative.
    n :  int
        Order of the Taylor expansion
    eps : float
        The increment made to the point for this approximation.
    sgn: int
        Sign of used in the expansion for :math:`\epsilon`.
        Options are {1, -1}

    """
    if isinstance(ders, list):
        ders = np.array(ders, dtype=float)
    elif not isinstance(ders, np.ndarray):
        raise TypeError('ders should be given as list or np.ndarray')
    if not isinstance(n, int):
        raise TypeError('n must be integer.')
    if n != len(ders)-1:
        raise ValueError('The number of derivatives does not match with the order of the Taylor series.')
    if not isinstance(eps, float):
        raise TypeError('eps must be float.')
    if sgn not in (-1, 1):
        raise ValueError('sgn should be either 1 or -1.')
    result = ders[0]
    for i in range(1, n+1):
        result += (sgn**(i) * eps**(i) * ders[i])/(factorial(i))
    return result


def taylor_coeffs(ders, n):
    r"""Return Taylor series coefficients up to n-term.

    The Taylor series centered at 0 (Maclaurin series):
    :math:`F(x) = F(0) + F'(0)x + 1/2 F''(0) x^2 + ...`

    As the derivatives are given already evaluated at some point, this
    funtion serves only to organize the coefficients {F(0), F'(0), 1/2 F''(0)}
    in a list/array.

    Parameters
    ----------
    ders :  array-like, float
        Values of the function derivatives, including 0th derivative.
    n :  int
        Order of the Taylor expansion

    """
    if isinstance(ders, list):
        ders = np.array(ders, dtype=float)
    elif not isinstance(ders, np.ndarray):
        raise TypeError('ders should be given as list or np.ndarray')
    if not isinstance(n, int):
        raise TypeError('n must be integer.')
    if n != len(ders)-1:
        raise ValueError('The number of derivatives does not match with the order of the Taylor series.')
    coeffs = np.zeros(n+1)
    for i in range(n+1):
        coeffs[i] = (ders[i])/(factorial(i))
    return coeffs


def fit_pseudosemivariance(xdata, ydata):
    r"""Fit a semivariance approximation from the errors of the model and the distance between points.

    We define our function as:
    :math:`\gamma_{s_0,a}(|\Delta x|) = s_0 [1-\exp(-a(\Delta x))]`
    using :math:`\frac{1}{2}(\Delta y_{kl})^2 vs. |\Delta x_{kl}|`,
    where :math:`\Delta y_{kl} = y_k - y_l`, :math:`\Delta x_{kl} = x_k - x_l.

    Parameters
    ----------
    xdata : array-like, float
        The points where the function is evaluated.
    ydata : array-like, float
        Function values at xdata.

    Returns
    -------
    szero, sexponent : float
        Parameters of the semivariance.
    """
    if isinstance(xdata, list):
        xdata = np.array(xdata, dtype=float)
    elif not isinstance(xdata, np.ndarray):
        raise TypeError('xdata should be given as list or np.ndarray')
    if isinstance(ydata, list):
        ydata = np.array(ydata, dtype=float)
    elif not isinstance(ydata, np.ndarray):
        raise TypeError('ydata should be given as list or np.ndarray')
    # Make arrays with distances between points and function values
    distance_y = [0.5*(k - l)**2 for (k, l) in itertools.combinations_with_replacement(ydata, 2)]
    distance_x = [abs(k - l) for (k, l) in itertools.combinations_with_replacement(xdata, 2)]

    def semivariance(x, szero, sexponent):
        """Function defining the semivariance for optimizer"""
        return szero*(1 - np.exp(-sexponent * (x**2)))
    result = optimize.curve_fit(semivariance, distance_x, distance_y)
    szero, sexponent = result[0]
    return szero, sexponent


def variance_column(xdata, point, szero, sexponent):
    r"""Construct column for the variance matrix.

    The column is made out of the elements:
    :math:`c_{k point}, where {k} are the indices of the existing points xdata, and point
    is the other point, :math:`c_{k point} = s_0 e^{-a (x_k - point)^2}`.
    A last point is added, for k+1, where we assing a value of 1.

    Paratemers
    ----------
    xdata : array-like, float
        Array of points to evaluate variance.
    point : float
        External point to which the variance is evaluated.
    szero : float
        Parameter :math:`s_0` obtained from the semivariance fit.
    sexponent :  float
        Parameter :math:`a` in the exponent of the semivariance fit.

    Returns
    -------
    column : np.ndarray((k+1, 1), dtype=float)
        The column :math:`c_{k point}`
    """
    lcolumn = len(xdata)
    column = np.zeros(lcolumn)
    for i in range(lcolumn):
        column[i] = szero*np.exp(-sexponent*((xdata[i] - point)**2))
    column = np.append(column, 1)
    return column


def get_variance_matrix(xdata, szero, sexponent):
    """Construct variance matrix using fit from semivariance.

    The matrix elements are:
    :math:`c_{ij} = c_{s_0, a}(|x_i-x_j|) = s_0 \left(\exp(-a(x_i -x_j)^2)\right)`
    The special cases are :math:`c_{im}`, where m = n+1, being n the total number of points
    given. So :math:`c_{im}`, and :math:`c_{mi}=1`, and :math:`c_{mm}=0`

    Paratemers
    ----------
    xdata : array-like, float
        Array of points to evaluate variance.
    point : float
        External point to which the variance is evaluated.
    szero : float
        Parameter :math:`s_0` obtained from the semivariance fit.
    sexponent :  float
        Parameter :math:`a` in the exponent of the semivariance fit.

    Returns
    -------
    cmatrix : np.ndarray((k+1,), dtype=float)
        Each column is :math:`c_{k point}`
    """
    length = len(xdata)
    cmatrix = np.zeros((length+1, length+1))
    # Fill per column
    for i in range(length):
        cmatrix[:,i] = variance_column(xdata, xdata[i], szero, sexponent)
    # Last column is all ones, but the last term, that is zero
    cmatrix[:,-1] = np.ones(length+1)
    cmatrix[-1,-1] = 0.0
    return cmatrix


def estimate_standard_deviation(xdata, newpoint, szero, sexponent):
    r""" Estimate the standard deviation of a new point.

    The approximation of the standard deviation is defined as:
    :math:`\sigma(x_new) = \sqrt \right(ws^T C ws - 2 c_{k x_new}^T ws + c_{x_new x_new} \left)`

    Paratemers
    ----------
    xdata : array-like, float
        Array of points to evaluate variance.
    newpoint : float
        External point to which the variance is evaluated.
    szero : float
        Parameter :math:`s_0` obtained from the semivariance fit.
    sexponent :  float
        Parameter :math:`a` in the exponent of the semivariance fit.

    Returns
    -------
    sigma : float
        The estimated standard deviation.
    """
    ldata = len(xdata)
    # Build the variance matrix and variance vector of the new point
    cmatrix = get_variance_matrix(xdata, szero, sexponent)
    vnew = variance_column(xdata, newpoint, szero, sexponent)
    # Solve equations for the weigths
    result = np.linalg.solve(cmatrix, vnew)
    # Take out the last element
    ws = result[:ldata]
    n_cmatrix = cmatrix[:ldata,:ldata]
    # Computation of sigma
    ws_cmatrix_ws = np.dot(ws, np.dot(n_cmatrix, ws))
    cnew = szero
    sigma = np.sqrt(ws_cmatrix_ws - 2*np.dot(vnew[:ldata], ws) + cnew)
    return sigma


def finite_difference(function, point, eps=1e-3):
    """Extremetly simple finite difference approxiation."""
    der = (function(point+eps) - function(point))/eps
    return der
