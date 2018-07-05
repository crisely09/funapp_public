"""Useful tools."""

import numpy as np

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
