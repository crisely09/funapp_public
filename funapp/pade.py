"""
Family of Pade Approximants.
"""

import numpy as np
import scipy as sp
from scipy import linalg as sl


__all__ = ['BasicPadeApproximant', 'GeneralPadeApproximant',]

class PadeApproximent(BaseApproximant):
    """
    Attributes
    ----------
    _m, _n : int
        Order of numerator and denominator series
        of the Padé approximant.
    _p, _q : np.poly1d
        Polynomials of the Pade approximant.
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

    def __init__(x, y, m, n):
        """Initialize Class for General Pade Approximants.

        Parameters
        ----------
        x : np.ndarray
            Points where the function was evaluated.
        y : np.ndarray
            Values of the function, and/or function
            derivatives evaluated at `x`.
        m, n : int
            Order of numerator and denominator series
            of the Padé approximant.

        """
        # Check arrays and initialize base class
        BaseApproximant.__init__(x, y)
        if m + n <= self._xlen:
            raise ValueError("To construct a [%d/%d] approximant at least\
                    %d points are needed" % (m, n, m+n))
        self._m = m
        self._n = n

    def __call__(self, x, der=0):
        """Evaluate the Padé approximant and its derivatives at the points `x`.

        Arguments
        ---------
        x : array like
            Array with points where to evaluate the approximant
        der : int or list(int)
            Number, or list of numbers of the derivative(s) to extract,
            Note: evaluation of the function corresponds to the 0th derivative.

        Returns
        -------
        List with arrays containing derivatives, each order in a different array.

        """
        if isinstance(x, list):
            x = np.array(x, dtype=float)
        elif not isinstance(x, np.ndarray):
            raise TypeError('x must be given as a list or as np.ndarray.')
        result = []
        if not isinstance(der, list):
            if not isinstance(der, int):
                raise TypeError('der should be either an integer or a list of integers.')
            der = [der]
        for d in der:
            if not isinstance(d, int):
                raise TypeError('The elements of der should be int.')
            if d == 0:
                result.append(self._evaluate(x))
            else:
                result.append(self._derivate(x, d))
        return result

    def store_derivatives(self):
        """Save derivatives of polynomials."""
        # Store all derivatives or p and q
        self._pders = []
        for i in range(self._m):
            self._pders.append(self._p.deriv(i+1))
        self._qders = []
        for i in range(self._n):
            self._qders.append(self._q.deriv(i+1))

    def _evaluate(self, x):
        """
        Evaluate the Pade Approximant at the set of points `x`.

        Parameters
        ----------
        x : array-like
            Points where the approximant will be evaluated.

        Returns
        -------
        Array with values of the approximant evaluated at x.

        """
        return self._p(x)/self._q(x)

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
        if not isinstance(m, int):
            raise TypeError('The order of derivative, m, should be int.')
        if m == 0:
            raise ValueError('For the evaluation of the function use the call method instead.')
        numerator = 0.0
        denominator = 1.0
        der_deffs = []
        a = 1
        d = 1
        c = 1.0
        der_terms.append([a, [], d, c])
        for i in range(m):
            new_terms = []
            for term in der_terms:
                add_prelated(term, new_terms)
                add_qnumterms(term, new_terms)
                add_qdenterms(term, new_terms)
            clean_terms(new_terms)
            der_terms = new_terms

        # Evaluate Terms
        # Create array to store values
        ders = np.zeros(x.shape)
        # Evaluate term by term
        for term in der_terms:
            tmp = np.zeros(x.shape)
            # Separate values first
            a = term[0]
            qterms = term[1]
            d = term[2]
            c = term[3]
            # add P terms
            if a > 1:
                tmp += self._pders[a-1](xi)
            else:
                tmp += self._p(xi)
            # Multiply by Q terms in numerator
            for t in qterms:
                order, power = t
                if order > 1:
                    valueq = self._qders[order-1](xi)
                else:
                    valueq = self._q(xi)
                if power != 1:
                    valueq = valueq**power
                else:
                    tmp *= valueq
            # Divide by Q terms in denominator
            if d == 1:
                tmp /= self._q(xi)
            else:
                tmp /= (self._q(xi))**d
            # Multiply by constant c
            tmp *= c
            ders += tmp
        return ders


def add_prelated(term, new_terms):
    """Add generated from derivative of P(x) terms.

    Parameters
    ----------
    term : list
        List with information about the term to be derivated.
    new_terms : list
        The list where the new terms will be appended.

    """
    # The only term that changes is a.
    a = term[0] + 1
    qterms = term[1]
    d = term[2]
    c = term[3]
    # Add new term to the list
    new_terms.append([a, qterms, d, c])


def add_qdenterms(term, new_terms):
    """Add generated from derivative of Q(x) terms in the denominator.

    Parameters
    ----------
    term : list
        List with information about the term to be derivated.
    new_terms : list
        The list where the new terms will be appended.

    """
    # We modify the constant, the power of Q, and add Q' to the
    # numerator.
    a = term[0]
    qterms = term[1].append([1, 1]) # derivative order and power of the term
    d = term[2] + 1
    c = term[3] * (-term[2])
    new_terms.append([a, qterms, d, c])


def add_qnumterms(term, new_terms):
    """Add generated from derivative of Q(x) terms in the numerator.

    Parameters
    ----------
    term : list
        List with information about the term to be derivated.
    new_terms : list
        The list where the new terms will be appended.

    """
    # Use chain rule for Q terms.
    a = term[0]
    qterms = term[1]
    d = term[2]
    for i, qterm in enumerate(qterms):
        n, m = qterm
        # Chain rule
        if m > 1:
            c = term[3] * m
            qterm[1] = m - 1
            new_terms.append([a, [qterm, [qterm[0] + 1, 1]], d, c])
        # Simple derivative
        else:
            new_terms.append([a, [qterm[0] + 1, 1], d, c])


def clean_terms(new_terms):
    """Clean list by summing same order terms.

    Parameters
    ----------
    new_terms : list
        The list where the new terms will be appended.

    """
    for i, term in enumerate(new_terms):
        for j in range(i, len(new_terms)):
            other = new_terms[j]
            # Check P and Q simple
            if term[0] == other[0] and term[2] == other[2]:
                # Check Q terms in numerator
                if term[1] == other[1]:
                    # Identify repeated terms and add them up
                    if j not in same:
                        term[3] += other[3]
                        # Save indices of the repeated term to be
                        # deleted later
                        same.append(j)
    # Use only the unique terms
    new = [new_terms[i] for i in range(len(new_terms)) if i not in same]
    return new


class BasicPadeApproximant(PadeApproximant):
    """Class for the generate the basic Pade approximant from a Taylor series.

    Parameters
    ----------
    x : np.ndarray
        Points where the function was evaluated.
    y : np.ndarray
        Values of the function, and/or function
        derivatives evaluated at `x`.
    ds : np.ndarray
        Taylor series coefficients.
    m: int
        Order of numerator and denominator series
        of the Padé approximant.

    """
    def __init__(self, x, y, ds, m):
        self._dslen = len(ds)
        if m + n <= self._xlen:
            raise ValueError("To construct a [%d/%d] approximant at least\
                    %d points are needed" % (m, n, m+n))
        if m + n <= self._dslen:
            raise ValueError("To construct a [%d/%d] approximant at least\
                    %d Taylor coefficients are needed" % (m, n, m+n))
        PadeApproximant.__init__(x, y, m, n)

        # Build approximant using poly1d objects
        # with the scipy method
        from scipy import misc
        p, q = misc.pade(ds, m)
        self._p = p
        self._q = q

        # Save derivatives
        self.store_derivatives()


class GeneralPadeApproximant(PadeApproximant):
    """Class for Pade Approximants generated without Taylor Series coefficients.

    Attributes
    ----------
    _m, _n : int
        Order of numerator and denominator series
        of the Padé approximant.
    _xlen : int
        Length of the array with points.
    _der : int or list(int)
        If derivatives are provided,`_der` stores the values in a different
        array.
    eps : float
        The little change to be made for the derivative approximation.

    Methods
    -------
    __call__
    _get_matrix_form
    _sort_coeffs
    _evaluate
    _derivate

    Example
    -------
    >>> x = [0., 1.5, 3.4]
    >>> y = [-0.47, -0.48, -0.498]
    >>> m = 3
    >>> n = 3
    >>> padegen = GeneralPadeApproximant(x, y, m, n)

    Then we evaluate the function and its first derivative on a new set
    of points xnew
    
    >>> xnew = [2.0, 5.0]
    >>> newdata = padegen(xnew, der=[0,1])
    """

    def __init__(self, x, y, m, n, eps=1e-3):
        """Initialize Class for General Pade Approximants.

        Parameters
        ----------
        x : np.ndarray
            Points where the function was evaluated.
        y : np.ndarray
            Values of the function, and/or function
            derivatives evaluated at `x`.
        m, n : list with orders
            Order of numerator and denominator series
            of the Padé approximant.

        """
        # Check arrays and initialize base class
        PadeApproximant.__init__(x, y, m, n)
        self.eps = eps
        # get matrix form
        matrix = self._get_matrix_form()
        zeros = np.zeros(self._xlen)

        # Solve for the coefficients
        result = np.linalg.solve(matrix, zeros)
        ps, qs = self._sort_coeffs(result)

        # Build approximant using poly1d objects
        self._p = np.poly1d(ps)
        self._q = np.poly1d(qs)

        # Save derivatives
        self.store_derivatives()

    def _get_matrix_form(self):
        r"""Construct the matrix to use for solving the linear equations.

        We use a linear solver giving a matrix A which rows are:
        :math:`{1, F(x_k)x_k, -x_k, F(x_k)x_k^2, -x_k^2, ..., F(x_k)x_k^n, -x_k^m}`
        and solve: A x = 0,
        A includes the derivatives in the approximation:
        :math:`F(x + \epsilon)Q(x + \epsilon) \approx P(x + \epsilon)`
        where
        :math:`F(x + \epsilon) \approx F(x) \pm \frac{1}{2}\epsilon F'(x) + ...`

        """

        # Fill out the matrix for the function values
        lenpars = self._m + self._n
        matrix = np.zeros((lenpars, self._lenx))
        for i, point in enumerate(self._x):
            # Set values related with the numerator P_m(x)
            matrix[i,0] = - 1.0
            prefac = point
            for j in range(1, self._m - 1):
                matrix[i, j] = -prefac
                prefac *= point
            # Set values related with the denominator Q_n(x)
            matrix[i,self._m] = self._y[i]
            prefac = point*self._y[i]
            for j in range(self._m + 1, lenpars - 1):
                matrix[i, j] = prefac
                prefac *= point

        # If derivatives provided, get the extra matrix elements
        # Fill out the matrix for the derivatives
        if self._ders:
            # New starting point
            k = len(self._x)
            for der in self._ders:
                derlist.append(self._y[np.where(der[0]==self._x)])
                higher_der = [dval for dval in der[1]]
                derlist += higher_der
                lder = len(derlist)
                for i in range(lder):
                    # Use the space in the matrix after the points
                    sgn = (-1)**i
                    if (i+1)%2 == 0:
                        eps += eps
                    newx = der[0] + sgn*eps
                    newy = taylor(derlist, lder-1, eps, sgn)
                    # Set values related with the numerator P_m(x)
                    matrix[k, 0] = - 1.0
                    prefac = newx
                    for j in range(1, self._m - 1):
                        matrix[k, j] = -prefac
                        prefac *= newx
                    # Set values related with the denominator Q_n(x)
                    matrix[k, self._m] = newy
                    prefac = newx*newy
                    for j in range(self._m + 1, lenpars - 1):
                        matrix[k, j] = prefac
                    k += 1
        return matrix

    def _sort_coeffs(self, coeffs):
        """Sort the coefficients of the Pade Approximant.

        Arguments
        ---------
        coeffs : np.ndarray(self._xlen)
            Results from the linear solver.

        Returns
        -------
        ps : np.ndarray(m)
            Coefficients for the numerator of the approximant.
        qs : np.ndarray(n)
            Coefficients for the denominator of the approximant.

        """
        ps = coeffs[:,len(self._x)]
        qs = coeffs[len(self._x):]
