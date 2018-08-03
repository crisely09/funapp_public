"""Meijer G Approximant."""


import numpy as np
from mpmath import *
mp.pretty = True

from funapp.tools import factorial


def borel_trans_coeffs(zn):
    r"""Get the coefficients of the Borel transformed series.

    Parameters
    ----------
    zn : np.ndarray (n,)
        Array with Taylor/Maclaurin series to be transformed.

    Returns
    -------
    bn : Array with the Borel transform coefficients :math:`b_n = z_n / n!`

    """
    if not isinstance(zn, np.ndarray):
        raise TypeError('Coefficients must be given in a numpy ndarray')
    bn = np.zeros(zn.shape)
    for i, z in enumerate(zn):
        bn[i] = z/factorial(i)
    return bn


def consecutive_ratios_odd(bn):
    """Get the consecutive ratios of Borel transform coefficients. """
    ratios_num = len(bn) - 1
    rn = np.array([bn[i+1]/bn[i] for i in range(ratios_num)])
    return rn


def rational_function_for_ratios(rn):
    """Fit the consecutive ratios of Borel transform coefficients into a rational function.
    
    Parameters
    ----------
    rn : np.ndarray
        The consecutive ratios.
    
    """
    if not isinstance(rn, np.ndarray):
        raise TypeError('Ratios must be given in a numpy ndarray')
    ratios_num = len(rn)
    l = ratios_num / 2
    npoints = ratios_num
    bvector = np.zeros(ratios_num)
    matrix = np.zeros((ratios_num, npoints))
    for i, r in enumerate(rn):
        # qm terms
        for j in range(l):
            nterm = (i)**(j+1)
            matrix[i, j] = r * nterm
        # pm terms
        for j in range(l, npoints):
            matrix[i, j] = -(i)**(j-l)
        bvector[i] = - r
    result = np.linalg.lstsq(matrix, bvector)[0]
    qs = result[:l]
    ps = result[l:]
    return ps, qs


def aux_little_series(coeffs, n, linit, lfin):
    """Auxiliary function for the rational function tests."""
    result = 0.
    for i, l in enumerate(range(linit, lfin+1)):
        result += coeffs[i]*((n+1)**l)
    return result


def find_roots(poly_coeffs):
    """Construct a poly1d object and give back the roots.

    Parameters
    ----------
    poly_coeffs : np.ndarray
        The coefficients of a polynomial expansion.
        Assumes the order decreases through the array.

    """
    # Reverse order of coefficients to use poly1d
    p = np.poly1d(poly_coeffs)
    return p.roots


def gamma_products(xs, l):
    """Calculate products:
    
    .. math::
        \Pi_{i=1}^l \Gamma (x_i)
    """
    result = 1
    for i in range(l):
        result *= gamma(xs[i])
    return result


def meijerg_approx_low(xs, ys, pl, ql, points):
    r"""Return the value of the Meijer Approximant at some point(s).

    The Maijer approximant is a representation of the Laplace Transform:
    .. math::
        Z_{B,N}(g) = \int^\infnty_0 e^\tau B_N(g \tau) d\tau
    with the Meijer-G function:
    .. math::
        Z_{B,N}(g) = \frac{\Pi_{i=1}^1 \Gamma (-y_i)}{\Pi_{i=1}^1 \Gamma(-x_i)}
                     G^{l+2\,,1}_{l+1\,,l+2} 
                     \left( \left. \begin{matrix} 1, -y_1, \dots, - y_l \\
                     1, 1, - x_1 , \dots - x_l \end{matrix}\; \right| -\frac{q_l}{p_l g}\right)

    Parameters:
    -----------
    xs : np.ndarray
        Roots of the numerator of the rational function for the ratios
        of the Borel transform coefficients
    ys : np.ndarray
        Roots of the denominator of the rational function for the ratios
        of the Borel transform coefficients.
    pl : float
        Last term coefficient of the numerator series of the rational function.
    ql : float
        Last term coefficient of the denominator series of the rational function.
    points : np.ndarray
        Array with points where to evaluate the approximant.

    Returns:
    --------
    result : np.ndarray
    """
    # Make the input lists for the Meijer G function in mpmath
    xs = xs[1:]
    # Originals:
    lista = [[1], [-y for y in ys]]
    listb = [[1, 1]+[-x for x in xs], []]
    #print "lista ", lista
    #print "listb ", listb
    lpoints = len(points)
    result = np.zeros(lpoints, dtype=np.complex_)
    # Compute the ratio of products of Gamma functions
    const = gamma_products(-ys, len(ys))/gamma_products(-xs, len(xs))
    #print "constant ", const
    z = -ql/pl
    #print 'inside', z
    for i in range(lpoints):
        ztmp = z/points[i]
        result[i] = const * meijerg(lista, listb, ztmp)
    return result


def meijerg_approximant():
    """ """
