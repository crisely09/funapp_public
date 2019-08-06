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


def meijerg_approx_low(xs, ys, pl, ql, points, maxterms=None):
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

    Parameters
    ----------
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

    Returns
    -------
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
        if maxterms is None:
            result[i] = const * meijerg(lista, listb, ztmp)
        else:
            result[i] = const * meijerg(lista, listb, ztmp, maxterms=maxterms)
    return result


def cma_solver(objective, params, **kwargs):
    """Minimize a function with the Covariance Matrix Adaptation Evolution Strategy.

    This is a simplified version of the solver in fanpy: `wfns.solver.equation.cma`.

    Parameters
    ----------
    objective : callable
        The function to be minimized.
    params : list or np.ndarray
        Parameters of the objective function to be optimized.
    kwargs : dict
        Keyword argumets for `cma.fmin`. The defaults are taken from fanpy solver: `wfns.solver.equation.cma`
        By default, 'sigma0' is set to 0.01 and 'options' to `{'ftarget': None, 'timeout': np.inf,
        'tolfun': 1e-11, 'verb_filenameprefix': 'outcmaes', 'verb_log': 0}`.
        The 'sigma0' is the initial standard deviation. The optimum is expected to be within
        `3*sigma0` of the initial guess.
        The 'ftarget' is the termination condition for the function value upper limit.
        The 'timeout' is the termination condition for the time. It is provided in seconds and must
        be provided as a string.
        The 'tolfun' is the termination condition for the change in function value.
        The 'verb_filenameprefix' is the prefix of the logger files that will be written to disk.
        The 'verb_log' is the verbosity of the logger files that will be written to disk. `0` means
        that no logs will be saved.
        See `cma.evolution_strategy.cma_default_options` for more options.
    
    Returns
    -------
    Dictionary with the following keys and values:
    success : bool
        True if optimization succeeded.
    optvalue : list
        Returned value of the `cma.fmin`.
    params : np.ndarray
        Parameters at the end of the optimization.
    message : str
        Termination reason.
    """
    import cma

    if kwargs == {}:
        kwargs = {'sigma0': 0.5, 'options': {'ftarget': 1e-8, 'tolfun': 1e-8,
                                              'maxiter':15, 'verb_filenameprefix': 'outcmaes',
                                              'verb_log': 0}}

    results = cma.fmin(objective, params, **kwargs)

    output = {}
    output['success'] = results[-3] != {}
    output['params'] = results[0]
    output['optvalue'] = results[1]

    if output['success']:
        output['message'] = ('Following termination conditions are satisfied:' +
                             ''.join(' {0}: {1},'.format(key, val)
                                     for key, val in results[-3].items()))
        output['message'] = output['message'][:-1] + '.'
    else:
        output['message'] = 'Optimization did not succeed.'

    return output
