"""
Family of Pade Approximants.
"""

import numpy as np
from scipy import linalg as sl
from scipy import optimize

from funapp.base import BaseApproximant
from funapp.tools import taylor

__all__ = ['PadeApproximant', 'BasicPadeApproximant', 'GeneralPadeApproximant', 'LSQPadeApproximant',
           'LSQSelPadeApproximant', 'ChebyshevApproximant', 'ChebyshevSelPadeApproximant',]

class PadeApproximant(BaseApproximant):
    """
    Attributes
    ----------
    _m, _n : int
        Order of numerator and denominator series
        of the Pade approximant.
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
    _store_derivatives
    _sort_coeffs_all

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

    def __call__(self, x, der=0):
        """Evaluate the Pade approximant and its derivatives at the points `x`.

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

    def _store_derivatives(self):
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
        xi : array-like
            Points where the approximant will be evaluated.
        der: int
            Order of the derivative(s) to extract.

        Returns
        Array with values of the derivatives of x.
        """
        xi = np.ravel(xi)
        if not isinstance(der, int):
            raise TypeError('The order of derivative, der, should be int.')
        if der == 0:
            raise ValueError('For the evaluation of the function use the call method instead.')
        numerator = 0.0
        denominator = 1.0
        der_terms = []
        a = 0
        d = 1
        c = 1.0
        # Generate lists that contains information of each derivative term
        # For each term has 4 components:
        # 1) the derivative order of the numerator
        # 2) the power and derivative order of terms generated form the derivative
        #    of the denominator
        # 3) the derivative order of the denominator
        # 4) the coefficient (constant in front of the term)
        #
        der_terms.append([a, [], d, c])
        for i in range(der):
            new_terms = []
            for term in der_terms:
                add_prelated(term, new_terms)
                add_qnumterms(term, new_terms)
                add_qdenterms(term, new_terms)
            new_terms = clean_terms(new_terms)
            der_terms = new_terms

        # Evaluate Terms
        # Create array to store values
        ders = np.zeros(xi.shape)
        # Evaluate term by term
        for term in der_terms:
            tmp = np.zeros(xi.shape)
            # Separate values first
            a = term[0]
            qterms = term[1]
            d = term[2]
            c = term[3]
            # add P terms
            if a > 0:
                if (a-1) < len(self._pders):
                    tmp += self._pders[a-1](xi)
            else:
                tmp += self._p(xi)
            # Multiply by Q terms in numerator
            for t in qterms:
                order, power = t
                if order > 0:
                    if (order-1) < len(self._qders):
                        valueq = self._qders[order-1](xi)
                    else:
                        valueq = 0.0
                else:
                    valueq = self._q(xi)
                if power != 1:
                    valueq = valueq**power
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
    qterms = term[1][:]
    qterms.append([1, 1]) # derivative order and power of the term
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

    Note: The new terms are added at the begging of the new list of qterms.

    """
    # Use chain rule for Q terms.
    a = term[0]
    qterms = term[1]
    d = term[2]
    c = term[3]
    for i, qterm in enumerate(qterms):
        order, power = qterm
        # Chain rule
        if power > 1:
            c *= power
            new = [[order, power - 1], [order + 1, 1]]
            extra = [t for t in qterms if qterm != t]
            if len(extra) > 0:
                for ex in extra:
                    new.append(ex)
            new_terms.append([a, new, d, c])
        # Simple derivative
        else:
            new = [[order + 1, 1]]
            extra = [t for t in qterms if qterm != t]
            if len(extra) > 0:
                for ex in extra:
                    new.append(ex)
            new_terms.append([a, new, d, c])


def clean_terms(new_terms):
    """Clean list by summing same order terms.

    Parameters
    ----------
    new_terms : list
        The list where the new terms will be appended.

    """
    same = []
    # First clean all qterms, multiply terms with same order
    new_q = []
    sames = []
    for i, term in enumerate(new_terms):
        new_q.append([])
        qterms = term[1][:]
        #sames.append([])
        for l in range(len(qterms)):
            try:
                q = qterms[l][:]
                for k in range(l+1, len(qterms)):
                    qother = qterms[k][:]
                    if q[0] == qother[0]:
                        #if k not in sames[i]:
                        q[1] += qother[1]
                        del(qterms[l])

                new_q[i].append(q[:])
            except IndexError:
                pass
    for i in range(len(new_terms)):
        new_terms[i][1] = new_q[i]
    for i, term in enumerate(new_terms):
        # Sum terms with same orders
        for j in range(i+1, len(new_terms)):
            other = new_terms[j]
            # Check P and Q simple
            if term[0] == other[0] and term[2] == other[2]:
                # Check Q terms in numerator
                if term[1] == other[1]:
                    # Identify repeated terms and add them up
                    if i not in same:
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
        of the Pade approximant.

    """
    def __init__(self, x, y, ds, m):
        PadeApproximant.__init__(self, x, y, m, m)
        self._dslen = len(ds)

        # Build approximant using poly1d objects
        # with the scipy method
        from scipy import misc
        p, q = misc.pade(ds, m)
        self._p = p
        self._q = q

        # Save derivatives
        self._store_derivatives()


class GeneralPadeApproximant(PadeApproximant):
    """Class for Pade Approximants generated without Taylor Series coefficients.

    Attributes
    ----------
    _m, _n : int
        Order of numerator and denominator series
        of the Pade approximant.
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
    _get_matrix_all
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

        With general, we mean that we don't have a Taylor series, nor the derivatives
        and the approximant is generated just from the given points and the 
        function evaluated at those points.

        Parameters
        ----------
        x : np.ndarray
            Points where the function was evaluated.
        y : np.ndarray
            Values of the function, and/or function
            derivatives evaluated at `x`.
        m, n : list with orders
            Order of numerator and denominator series
            of the Pade approximant.

        """
        # Check arrays and initialize base class
        PadeApproximant.__init__(self, x, y, m, n)
        if m + n > self._xlen:
            raise ValueError("To construct a [%d/%d] approximant at least\
                    %d points are needed" % (m, n, m+n))
        self.eps = eps
        # get matrix form
        matrix, bvector = self._get_matrix_all()
        

        # Solve for the coefficients
        result = np.linalg.lstsq(matrix, bvector)[0]
        # Split the results for Pm and Qn, also reverse order
        # because the poly1d polynomials are order from high order
        # to low order (opposite as the Pade series)
        ps, qs = self._sort_coeffs_all(result)
        ps = ps[::-1]
        qs = qs[::-1]
        # Qn is missing the first term, b0 = 1, so append it now.
        qs = np.append(qs, np.array([1.0]))

        # Build approximant using poly1d objects
        self._p = np.poly1d(ps)
        self._q = np.poly1d(qs)

        # Save derivatives
        self._store_derivatives()

    def _get_matrix_all(self):
        r"""Construct the matrix with all the series components.

        We use a linear solver giving a matrix A which rows are:
        :math:`{-1, -x_k, -x_k^2, ..., -x_k^m, F(x_k)x_k, F(x_k)x_k^2, F(x_k)x_k^n}`
        and solve: A x = b,
        A includes the derivatives in the approximation:
        :math:`F(x + \epsilon)Q(x + \epsilon) \approx P(x + \epsilon)`
        where
        :math:`F(x + \epsilon) \approx F(x) \pm \frac{1}{2}\epsilon F'(x) + ...`
        and :math:`b=-F(x)`.

        """

        # Fill out the matrix for the function values
        lenpars = self._m + self._n + 1
        npoints = self._xlen
        matrix = np.zeros((npoints, lenpars))
        b = np.zeros(npoints)
        for i, point in enumerate(self._x):
            # Set values related with the numerator P_m(x)
            # NOTE: As Pm series has a_0, we have m+1 terms
            matrix[i,0] =  -1.0
            prefac = point
            for j in range(1, self._m+1):
                matrix[i, j] = -prefac
                prefac *= point
            # Set values related with the denominator Q_n(x)
            # NOTE: b_0 = 1, so we only have n terms
            prefac = point*self._y[i]
            for j in range(self._m+1, lenpars):
                matrix[i, j] = prefac
                prefac *= point
            b[i] = - self._y[i]

        # If derivatives provided, get the extra matrix elements
        # Fill out the matrix for the derivatives
        if self._ders:
            # New starting point
            k = len(self._x)
            eps = self.eps
            for der in self._ders:
                derlist = []
                derlist.append(self._y[np.where(der[0]==self._x)])
                higher_der = [dval for dval in der[1]]
                derlist += higher_der
                lder = len(derlist)-1
                for i in range(lder):
                    # Use the space in the matrix after the points
                    sgn = (-1)**i
                    if (i+1)%2 == 0:
                        eps += eps
                    newx = der[0] + sgn*eps
                    newy = taylor(derlist, lder, eps, sgn)
                    # Set values related with the numerator P_m(x)
                    matrix[k, 0] =  -1.0
                    prefac = newx
                    for j in range(1, self._m+1):
                        matrix[k, j] = -prefac
                        prefac *= newx
                    # Set values related with the denominator Q_n(x)
                    b[k] = - newy
                    prefac = newx*newy
                    for j in range(self._m+1, lenpars):
                        matrix[k, j] = prefac
                    k += 1
        return matrix, b

    def _sort_coeffs_all(self, coeffs):
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
        ps = coeffs[:self._m+1]
        qs = coeffs[self._m+1:]
        return ps, qs

# TODO: check elements in the lists are in increasing order
class SelectivePadeApproximant(PadeApproximant):
    """Class for Pade Approximants generated without Taylor Series coefficients.

    Attributes
    ----------
    _m, _n : int
        Highest order of numerator and denominator series
        of the Pade approximant.
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
    _get_matrix_select
    _sort_coeffs
    _evaluate
    _derivate

    Example
    -------
    >>> x = [0., 1.5, 3.4]
    >>> y = [-0.47, -0.48, -0.498]
    >>> m = [0, 4]
    >>> n = [1, 2, 3, 4]
    >>> padegen = SelectivePadeApproximant(x, y, m, n)

    Then we evaluate the function and its first derivative on a new set
    of points xnew

    >>> xnew = [2.0, 5.0]
    >>> newdata = padegen(xnew, der=[0,1])
    """

    def __init__(self, x, y, ml, nl, eps=1e-3):
        """Initialize Class for General Pade Approximants.

        With general, we mean that we don't have a Taylor series, nor the derivatives
        and the approximant is generated just from the given points and the
        function evaluated at those points.

        Parameters
        ----------
        x : np.ndarray
            Points where the function was evaluated.
        y : np.ndarray
            Values of the function, and/or function
            derivatives evaluated at `x`.
        ml, nl : list of int
            Orders of numerator and denominator series
            of the Pade approximant.

        """
        # Check parameters
        if not isinstance(ml, (list, int)):
            raise TypeError('ml should be given as list of integers')
        if not isinstance(nl, (list, int)):
            raise TypeError('nl should be given as list of integers')
        mlen = len(ml)
        # Because the default already contains the order zero for the denominator
        # if zero is in the list, take it off
        if 0 in nl:
            nl.remove(0)
        nlen = len(nl)
        # Initialize base class
        PadeApproximant.__init__(self, x, y, max(ml), max(nl))
        # Check number of points provided
        if mlen + nlen > self._xlen:
            raise ValueError("To construct a [%d/%d] approximant at least\
                    %d points are needed" % (mlen, nlen, mlen+nlen))
        self.eps = eps

        # get matrix form
        matrix, bvector = self._get_matrix_select(ml, nl, mlen, nlen)

        # Solve for the coefficients
        result = np.linalg.lstsq(matrix, bvector)[0]
        # Split the results for Pm and Qn, also reverse order
        # because the poly1d polynomials are order from high order
        # to low order (opposite as the Pade series)
        nums, denoms = self._sort_coeffs_selective(result, mlen)
        ps = []; qs = []
        # Add zeros por the powers not included in ml and nl
        for i in range(self._m+1):
            if i in ml:
                ps.append(nums[ml.index(i)])
            else:
                ps.append(0)
        # Qn is missing the first term, b0 = 1, so append it now.
        qs.append(1.0)
        for j in range(1 ,self._n+1):
            if j in nl:
                qs.append(denoms[nl.index(j)])
            else:
                qs.append(0)
        ps = ps[::-1]
        qs = qs[::-1]

        # Build approximant using poly1d objects
        self._p = np.poly1d(ps)
        self._q = np.poly1d(qs)

        # Save derivatives
        self._store_derivatives()


    def _get_matrix_select(self, ml, nl, mlen, nlen):
        r"""Construct the matrix for some of the series components.

        We use a linear solver giving a matrix A which rows are:
        :math:`{1, F(x_k)x_k, -x_k, F(x_k)x_k^2, -x_k^2, ..., F(x_k)x_k^n, -x_k^m}`
        and solve: A x = b,
        A includes the derivatives in the approximation:
        :math:`F(x + \epsilon)Q(x + \epsilon) \approx P(x + \epsilon)`
        where
        :math:`F(x + \epsilon) \approx F(x) \pm \frac{1}{2}\epsilon F'(x) + ...`
        and b = - F(x)

        Parameters
        ----------
        ml :  list of int
            The powers to include in the numerator series
        nl : list of int
            Powers to include in the denominator series

        Returns
        -------
        matrix : np.ndarray((npoints, lenpars))
            Matrix to use for solving the linear equations
        b : np.ndarray(npoints)
            b vector for the linear equations
        """
        # Fill out the matrix for the function values
        lenpars = mlen +  nlen + 1
        npoints = self._xlen
        matrix = np.zeros((npoints, lenpars))
        b = np.zeros(npoints)
        for i, point in enumerate(self._x):
            # Set values related with the numerator P_m(x)
            # NOTE: As Pm series has a_0, we have m+1 terms
            prefac = point
            for j, mpower in enumerate(ml):
                matrix[i, j] = - (prefac**mpower)
            # Set values related with the denominator Q_n(x)
            # NOTE: b_0 = 1, so we only have n terms
            prefac = self._y[i]
            for j, npower in enumerate(nl):
                matrix[i, mlen+j] = prefac * (point**npower)
            b[i] = - self._y[i]

        # If derivatives provided, get the extra matrix elements
        # Fill out the matrix for the derivatives
        if self._ders:
            # New starting point
            k = len(self._x)
            eps = self.eps
            for der in self._ders:
                derlist = []
                derlist.append(self._y[np.where(der[0]==self._x)])
                higher_der = [dval for dval in der[1]]
                derlist += higher_der
                lder = len(derlist)-1
                for i in range(lder):
                    # Use the space in the matrix after the points
                    sgn = (-1)**i
                    if (i+1)%2 == 0:
                        eps += eps
                    newx = der[0] + sgn*eps
                    newy = taylor(derlist, lder, eps, sgn)
                    # Set values related with the numerator P_m(x)
                    prefac = newx
                    for j, mpower in enumerate(ml):
                        matrix[k, j] = - (prefac**mpower)
                    # Set values related with the denominator Q_n(x)
                    b[k] = - newy
                    prefac = newy
                    for j, npower in enumerate(nl):
                        matrix[k, mlen+j] = prefac * (newx**npower)
                    k += 1
        return matrix, b

    def _sort_coeffs_selective(self, coeffs, mlen):
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
        ps = coeffs[:mlen]
        qs = coeffs[mlen:]
        return ps, qs


class LSQPadeApproximant(PadeApproximant):
    """Class for Pade Approximants generated by LSQP fitting.

    Attributes
    ----------
    _m, _n : int
        Highest order of numerator and denominator series
        of the Pade approximant.
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
    _evaluate
    _derivate
    """
    def __init__(self, x, y, m, n, **kwargs):
        """Initialize Class for General Pade Approximants.

        With general, we mean that we don't have a Taylor series, nor the derivatives
        and the approximant is generated just from the given points and the
        function evaluated at those points.

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
        kwargs :
            Key-word arguments for the LSQP optimizer.

        """
        # Check arrays and initialize base class
        PadeApproximant.__init__(self, x, y, m, n)
        p, q = self.fit_pade_lsq()
        self._p = p
        self._q = q

        # Save derivatives
        self._store_derivatives()

    def _pade_forfit(self, allcoeffs):
        numcoeffs = allcoeffs[:self._m]
        dencoeffs = allcoeffs[self._m:]
        dencoeffs = np.append(dencoeffs, np.ones(1))
        p = np.poly1d(numcoeffs)
        q = np.poly1d(dencoeffs)
        error = np.power(np.linalg.norm(p(self._x)/q(self._x) - self._y), 2)
        return error

    def fit_pade_lsq(self):
        numcoeffs = np.ones(self._m)
        dencoeffs = np.ones(self._n-1)
        pars = np.append(numcoeffs, dencoeffs)
        result = optimize.minimize(self._pade_forfit, pars, tol=1e-12)
        optpars = result.x
        pf = np.poly1d(optpars[:self._m])
        qf = np.poly1d(np.append(optpars[self._m:], 1))
        return pf, qf


class LSQSelPadeApproximant(PadeApproximant):
    """Class for Pade Approximants generated by LSQP fitting.

    Attributes
    ----------
    _m, _n : int
        Highest order of numerator and denominator series
        of the Pade approximant.
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
    _evaluate
    _derivate
    """
    def __init__(self, x, y, ml, nl, **kwargs):
        """Initialize Class for General Pade Approximants.

        With general, we mean that we don't have a Taylor series, nor the derivatives
        and the approximant is generated just from the given points and the
        function evaluated at those points.

        Parameters
        ----------
        x : np.ndarray
            Points where the function was evaluated.
        y : np.ndarray
            Values of the function, and/or function
            derivatives evaluated at `x`.
        ml, nl : list with orders
            Order of numerator and denominator series
            of the Pade approximant.
        kwargs :
            Key-word arguments for the LSQP optimizer.

        """
        # Check parameters
        if not isinstance(ml, (list, int)):
            raise TypeError('ml should be given as list of integers')
        if not isinstance(nl, (list, int)):
            raise TypeError('nl should be given as list of integers')
        # Because the default already contains the order zero for the denominator
        # if zero is in the list, take it off
        if 0 in nl:
            nl.remove(0)
        self.mlen = len(ml)
        self.nlen = len(nl)
        # Initialize base class
        PadeApproximant.__init__(self, x, y, max(ml), max(nl))
        # Save the list to construct series later
        self.ml = list(self._m - np.array(ml))
        self.nl = list(self._n - np.array(nl))
        p, q = self.fit_pade_lsq()
        self._p = p
        self._q = q

        # Save derivatives
        self._store_derivatives()

    def _get_coeffs(self, allcoeffs):
        """Get coefficients for the polynomials in order."""
        numcoeffs = np.zeros(self._m+1)
        dencoeffs = np.zeros(self._n+1)
        numcoeffs[self.ml] = allcoeffs[:self.mlen]
        dencoeffs[self.nl] = allcoeffs[self.mlen:]
        dencoeffs[-1] = 1.0
        return numcoeffs, dencoeffs

    def _pade_forfit(self, allcoeffs):
        """ Internal function for the LSQS optimization.

        Parameters
        ----------
        allcoeffs : array-like, float
            The coefficients of the Pade.
        """
        lpars = len(allcoeffs)
        numcoeffs, dencoeffs = self._get_coeffs(allcoeffs)
        p = np.poly1d(numcoeffs)
        q = np.poly1d(dencoeffs)
        error = 0
        for i in range(len(self._x)):
            error += np.power(self._y[i] - (p(self._x[i])/q(self._x[i])), 2)
        error /= lpars+1
        error = np.sqrt(error)
        #error2 = np.power(np.linalg.norm(p(self._x)/q(self._x) - self._y), 2)
        return error

    def fit_pade_lsq(self):
        numcoeffs = np.ones(self.mlen)
        dencoeffs = np.ones(self.nlen)
        pars = np.append(numcoeffs, dencoeffs)
        result = optimize.minimize(self._pade_forfit, pars, tol=1e-10)
        optpars = result.x
        numcoeffs, dencoeffs = self._get_coeffs(optpars)
        pf = np.poly1d(numcoeffs)
        qf = np.poly1d(dencoeffs)
        return pf, qf


class LSQPadeApproximant(PadeApproximant):
    """Class for Pade Approximants generated by LSQP fitting.

    Attributes
    ----------
    _m, _n : int
        Highest order of numerator and denominator series
        of the Pade approximant.
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
    _evaluate
    _derivate
    """
    def __init__(self, x, y, m, n, **kwargs):
        """Initialize Class for General Pade Approximants.

        With general, we mean that we don't have a Taylor series, nor the derivatives
        and the approximant is generated just from the given points and the
        function evaluated at those points.

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
        kwargs :
            Key-word arguments for the LSQP optimizer.

        """
        # Check arrays and initialize base class
        PadeApproximant.__init__(self, x, y, m, n)
        p, q = self.fit_pade_lsq()
        self._p = p
        self._q = q

        # Save derivatives
        self._store_derivatives()

    def _pade_forfit(self, allcoeffs):
        numcoeffs = allcoeffs[:self._m]
        dencoeffs = allcoeffs[self._m:]
        dencoeffs = np.append(dencoeffs, np.ones(1))
        p = np.poly1d(numcoeffs)
        q = np.poly1d(dencoeffs)
        error = np.power(np.linalg.norm(p(self._x)/q(self._x) - self._y), 2)
        return error

    def fit_pade_lsq(self):
        numcoeffs = np.ones(self._m)
        dencoeffs = np.ones(self._n-1)
        pars = np.append(numcoeffs, dencoeffs)
        result = optimize.minimize(self._pade_forfit, pars, tol=1e-12)
        optpars = result.x
        pf = np.poly1d(optpars[:self._m])
        qf = np.poly1d(np.append(optpars[self._m:], 1))
        return pf, qf


class ChebyshevApproximant(BaseApproximant):
    """
    Attributes
    ----------
    _m, _n : int
        Order of numerator and denominator series
        of the Pade approximant.
    _ps, _qs : np.array
        Coefficients of the numerator and denominator of the approximant.
    _xlen : int
        Length of the array with points.
    _der : int or list(int)
        If derivatives are provided,`_der` stores the values in a different
        array.

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
            highest order of numerator and denominator series
            of the Pade approximant.

        """
        # Check arrays and initialize base class
        BaseApproximant.__init__(self, x, y)
        if not isinstance(m, int):
            raise TypeError('m must be integer.')
        if not isinstance(n, int):
            raise TypeError('m must be integer.')
        if self._ders:
            raise NotImplementedError('At the moment derivatives are not allowed.')
        self._m = m
        self._n = n
        self._xmax = max(self._x)

    def __call__(self, x, der=0):
        """Evaluate the Pade approximant and its derivatives at the points `x`.

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
                raise NotImplementedError
        return result

    def _store_derivatives(self):
        """Save derivatives of polynomials."""
        raise NotImplementedError

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
        result = np.polynomial.chebyshev.chebval(x/self._xmax, self._ps)
        result /= (1 + np.polynomial.chebyshev.chebval(x/self._xmax, self._qs))
        return result


    def _derivate(self, xi, der):
        """
        Evaluate the Pade Approximant derivatives at `x`.

        Parameters
        ----------
        xi : array-like
            Points where the approximant will be evaluated.
        der: int
            Order of the derivative(s) to extract.

        Returns
        Array with values of the derivatives of x.
        """
        raise NotImplementedError


class ChebyshevSelPadeApproximant(ChebyshevApproximant):
    """Class for scaled Chebyshev approximants generated without Taylor Series coefficients.

    Attributes
    ----------
    _m, _n : int
        Highest order of numerator and denominator series
        of the Pade approximant.
    _xlen : int
        Length of the array with points.
    _der : int or list(int)
        If derivatives are provided,`_der` stores the values in a different
        array.

    Methods
    -------
    __call__
    _get_matrix_select
    _sort_coeffs
    _evaluate
    _derivate

    Example
    -------
    >>> x = [0., 1.5, 3.4]
    >>> y = [-0.47, -0.48, -0.498]
    >>> m = [0, 4]
    >>> n = [1, 2, 3, 4]
    >>> padegen = chebyshevSelPadeApproximant(x, y, m, n)

    Then we evaluate the function and its first derivative on a new set
    of points xnew

    >>> xnew = [2.0, 5.0]
    >>> newdata = padegen(xnew)
    """

    def __init__(self, x, y, ml, nl):
        """Initialize Class for scaled Chebyshev Pade Approximants.

        With general, we mean that we don't have a Taylor series, nor the derivatives
        and the approximant is generated just from the given points and the
        function evaluated at those points.

        Parameters
        ----------
        x : np.ndarray
            Points where the function was evaluated.
        y : np.ndarray
            Values of the function, and/or function
            derivatives evaluated at `x`.
        ml, nl : list of int
            Orders of numerator and denominator series
            of the Pade approximant.

        """
        # Check parameters
        if not isinstance(ml, (list, int)):
            raise TypeError('ml should be given as list of integers')
        if not isinstance(nl, (list, int)):
            raise TypeError('nl should be given as list of integers')
        mlen = len(ml)
        # Because the default already contains the order zero for the denominator
        # if zero is in the list, take it off
        if 0 in nl:
            nl.remove(0)
        nlen = len(nl)
        # Initialize base class
        ChebyshevApproximant.__init__(self, x, y, max(ml), max(nl))
        # Check number of points provided
        if mlen + nlen > self._xlen:
            raise ValueError("To construct a [%d/%d] approximant at least\
                    %d points are needed" % (mlen, nlen, mlen+nlen))
        # Save the lists
        self.ml = ml
        self.nl = nl

        # Create model
        self.update_approximant()

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
        ctmp = np.zeros(self._m - 1)
        lastnum = self._ps[-1]
        ctmp[-1] = self._m*lastnum
        result = np.polynomial.chebyshev.chebval(x/self._xmax, self._ps)
        result += np.polynomial.chebyshev.chebval(x/self._xmax, ctmp)
        result /= (1 + np.polynomial.chebyshev.chebval(x/self._xmax, self._qs))
        return result

    def update_approximant(self):
        """From the current information update the approximant coefficients.
        """
        mlen = len(self.ml)
        nlen = len(self.nl)
        # get matrix form
        matrix, bvector = self._get_matrix_select(self.ml, self.nl, mlen, nlen)

        # Solve for the coefficients
        result = np.linalg.lstsq(matrix, bvector)[0]
        # Split the results for Pm and Qn, also reverse order
        # because the poly1d polynomials are order from high order
        # to low order (opposite as the Pade series)
        nums, denoms = self._sort_coeffs_selective(result, mlen)
        ps, qs = self.complete_coefficients(nums, denoms)
        # Instead of using polynomials we save the coefficients
        self._ps = ps
        self._qs = qs

    def complete_coefficients(self, nums, denoms):
        """Complete the list of coefficients to evaluate the model."""
        # Add zeros por the powers not included in ml and nl
        ps = np.zeros(self._m+1)
        ps[self.ml] = nums
        qs = np.zeros(self._n+1)
        for i, n in enumerate(self.nl):
            qs[n] = denoms[i]
        return ps, qs

    def add_point(self, xnew, ynew):
        """Add new point to the fit and update the model.

        Parameters
        ----------
        xnew : float
            Point to be added.
        ynew : float
            The value of the function at the new x.

        """
        self._x = np.append(self._x, xnew)
        self._y = np.append(self._y, ynew)
        self._xlen = len(self._x)
        self.update_approximant()

    def add_order(self, neworder, option):
        """Add an order to the numerator series.

        Parameters
        ----------
        neworder :  int
            The new order of Chebyshev polynomial to be added.
        option : str
            Where is the order added, 'numerator' or 'denominator'

        """
        if option == 'numerator':
            self.ml.append(neworder)
            self.ml.sort()
            self._m = max(self.ml)
        elif option == 'denominator':
            self.nl.append(neworder)
            self.nl.sort()
            self._n = max(self.nl)
        else:
            raise ValueError("option valid values are:'numerator' or 'denominator'")
        self.update_approximant()


    def _get_matrix_select(self, ml, nl, mlen, nlen, eps=None):
        r"""Construct the matrix for some of the series components.

        We use a linear solver giving a matrix A which rows are:
        :math:`{1, F(x_k)x_k, -x_k, F(x_k)x_k^2, -x_k^2, ..., F(x_k)x_k^n, -x_k^m}`
        and solve: A x = b,
        A includes the derivatives in the approximation:
        :math:`F(x + \epsilon)Q(x + \epsilon) \approx P(x + \epsilon)`
        where
        :math:`F(x + \epsilon) \approx F(x) \pm \frac{1}{2}\epsilon F'(x) + ...`
        and b = - F(x)

        Parameters
        ----------
        ml :  list of int
            The powers to include in the numerator series
        nl : list of int
            Powers to include in the denominator series
        eps : array like
            Precision of the reference points

        Returns
        -------
        matrix : np.ndarray((npoints, lenpars))
            Matrix to use for solving the linear equations
        b : np.ndarray(npoints)
            b vector for the linear equations
        """
        # Fill out the matrix for the function values
        lenpars = mlen +  nlen
        npoints = self._xlen
        if eps is not None:
            if len(eps) != npoints:
                raise ValueError("eps length should be the same as points used in the fit")
        else:
            eps = np.ones(npoints)
        matrix = np.zeros((npoints, lenpars))
        b = np.zeros(npoints)
        for i, point in enumerate(self._x):
            # Set values related with the numerator P_m(x)
            # NOTE: As Pm series has a_0, we have m+1 terms
            for j, mpower in enumerate(ml):
                cs = np.zeros(mpower+1)
                cs[-1] = 1.0
                if mpower == self._m:
                    # Add nT_{n-2} term
                    ctmp = np.zeros(mpower-1)
                    ctmp[-1] = self._m
                    tmp = np.polynomial.chebyshev.chebval(point/self._xmax, ctmp)
                    tmp += np.polynomial.chebyshev.chebval(point/self._xmax, cs)
                    matrix[i, j] = - (1./eps[i]) * tmp
                else:
                    matrix[i, j] = - (1./eps[i])*np.polynomial.chebyshev.chebval(point/self._xmax, cs)
            # Set values related with the denominator Q_n(x)
            # NOTE: b_0 = 1, so we only have n terms
            prefac = self._y[i]
            for j, npower in enumerate(nl):
                # start with an array of 2 elements, so it starts at T1
                cs = np.zeros(npower+1)
                cs[-1] = 1.0
                matrix[i, mlen+j] = (1./eps[i])*prefac*np.polynomial.chebyshev.chebval(point/self._xmax, cs)
            b[i] = - (1./eps[i])*self._y[i]
        return matrix, b

    def _sort_coeffs_selective(self, coeffs, mlen):
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
        ps = coeffs[:mlen]
        qs = coeffs[mlen:]
        return ps, qs
