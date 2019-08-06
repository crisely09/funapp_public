"""
Family of Pade Approximants in 2 dimensions.
"""

import numpy as np
from scipy import linalg as sl
from scipy import optimize

from funapp.base import BaseApproximant
from funapp.tools import taylor, chebypoly


class Chebyshev2DApproximant(object):
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
    """

    def __init__(self, x, y, zmatrix, x_maxorder, y_maxorder):
        """Initialize base class for Pade Approximants.

        Parameters
        ----------
        x, y : np.ndarray
            Points where the function was evaluated in two axis, x and y.
        ymatrix : np.ndarray((lenx, leny), dtype=float)
            Values of the function at (`x`, `y`).
        x_maxorder, y_maxorder : int
            highest order of numerator and denominator series
            of the Pade approximant (NOTE:numerator and denominator have the
            same higher order term).

        """
        # Check arrays and initialize base class
        if not isinstance(x_maxorder, int):
            raise TypeError('x_maxorder must be integer.')
        if not isinstance(y_maxorder, int):
            raise TypeError('y_maxorder must be integer.')
        if self._ders:
            raise NotImplementedError('At the moment derivatives are not allowed.')
        # Check the sizes of the arrays and store local variables.
        if zmatrix.shape != (len(x), len(y)):
            raise ValueError('zmatrix should have a shape: (len(x), len(y)).')
        x = np.ravel(x)
        y = np.ravel(y)
        self._prune_arrays(x, y, zmatrix)
        self._xmaxorder = x_maxorder
        self._ymaxorder = y_maxorder
        self._xmax = max(self._x)
        self._ymax = max(self._y)

    def _prune_arrays(self, x, y, zmatrix):
        """Remove derivatives from arrays.

        Parameters
        ----------
        x, y : np.ndarray
            Points where the function is evaluated.
        zmatrix : np.ndarray
            Value of the function at points (x,y).

        """
        xsame = []; xused = []
        ysame = []; yused = []
        x = list(x); y = list(y)
        xsame = [i for i in range(len(x)) if x.count(x[i]) > 1]
        ysame = [i for i in range(len(y)) if y.count(y[i]) > 1]
        xrepv = set()
        xrep = []
        for i in xsame:
            if x[i] not in xrepv:
                xrepv.add(x[i])
            else:
                xrep.append(i)
        yrepv = set()
        yrep = []
        for i in ysame:
            if y[i] not in yrepv:
                yrepv.add(y[i])
            else:
                yrep.append(i)
        # Clean first x and y
        self._x = np.delete(x, xrep)
        self._y = np.delete(y, yrep)
        # Clean zmatrix
        self._zmatrix = np.delete(zmatriz, xrep, 0)
        self._zmatrix = np.delete(zmatriz, yrep, 1)

    def __call__(self, x, y):
        """Evaluate the Pade approximant at the points `x`.

        Arguments
        ---------
        x, y : array like
            Array with points where to evaluate the approximant

        Returns
        -------
        List with arrays containing derivatives, each order in a different array.

        """
        if isinstance(x, list):
            x = np.array(x, dtype=float)
        elif not isinstance(x, np.ndarray):
            raise TypeError('x must be given as a list or as np.ndarray.')
        if isinstance(y, list):
            y = np.array(y, dtype=float)
        elif not isinstance(y, np.ndarray):
            raise TypeError('y must be given as a list or as np.ndarray.')
        return self._evaluate(x, y)

    def _evaluate(self, x, y):
        """
        Evaluate the Pade Approximant at the set of points `x`.

        Parameters
        ----------
        x, y : array-like
            Points where the approximant will be evaluated.

        Returns
        -------
        Array with values of the approximant evaluated at (x,y).

        """
        lastnum = self._ps[-1]
        lastc = np.zeros(self._xmaxorder+1)
        lastc[-1] = lastnum
        cn = self._ps[:-1]
        nresult = chebypoly(x/self._xmax, cn)*chebypoly(y/self._ymax, cn)
        # p_n * T_{n-2}*n
        ctmp = np.zeros(self._maxorder - 1)
        ctmp[-1] = self._xmaxorder*lastnum
        nresult += self.factor*chebypoly(y/self._ymax, self._ps)*chebypoly(x/self._xmax, ctmp)
        # p_n * T_{n}
        nresult += self.factor*chebypoly(x/self._xmax, lastc)
        # 1 + q_n T_{}
        lastd = np.zeros(self._n+1)
        lastd[-1] = self._qs[-1]
        cd = self._qs[:self._n]
        dresult = 1 + chebypoly(x/self._xmax, cd)
        dresult += self.factor*chebypoly(x/self._xmax, lastd)
        return nresult/dresult


class ChebyshevMy2DPadeApproximant(Chebyshev2DApproximant):
    """Class for scaled Chebyshev approximants generated without Taylor Series coefficients.
    This particular class works for an specific 2D Energy model.

    Attributes
    ----------
    _k, _l : int
        Highest order of numerator and denominator series
        of the Pade approximant in each dimension.
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
    >>> kindex = 4
    >>> lindex = 3
    >>> padegen = ChebyshevMy2DPadeApproximant(x, y, k, l)

    Then we evaluate the function on a new set
    of points xnew

    >>> xnew = [2.0, 5.0]
    >>> newdata = padegen(xnew)
    """

    def __init__(self, x, y, zmatrix, kindex, lindex, factor=1.0):
        """Initialize Class for scaled Chebyshev Pade Approximants.

        With general, we mean that we don't have a Taylor series, nor the derivatives
        and the approximant is generated just from the given points and the
        function evaluated at those points.

        Parameters
        ----------
        x, y : np.ndarray
            Points where the function was evaluated.
        zmatrix : np.ndarray
            Values of the function, and/or function
            derivatives evaluated at (`x`, `y`).
        kindex, lindex : int
            Maximum orders of numerator and denominator series
            of the Pade approximant for each dimension.

        """
        # Check parameters
        if not isinstance(kindex, int):
            raise TypeError('kindex should be given as integer')
        if kindex < 4:
            raise ValueError("kindex should be greater or equal to 4, for this model.")
        if not isinstance(l, int):
            raise TypeError('lindex should be given as integer')
        # Make lists for mu terms (k-index) 
        ml = range(kindex+1)
        for i in range(3):
            ml.pop(-2)
        mlen = len(ml)
        nl = range(1, kindex+1)
        nlen = len(nl)
        # Male list for epsilon terms (l-index)
        ol = range(lindex)
        olen = len(ol)
        # Initialize base class
        Chebyshev2DApproximant.__init__(self, x, y, zmatrix, kindex, lindex)
        # Check number of points provided
        if mlen + nlen > self._xlen:
            raise ValueError("To construct a [%d/%d] approximant at least\
                    %d points are needed" % (mlen, nlen, mlen+nlen))
        # Save the lists
        self.ml = ml
        self.nl = nl
        self.ol = ol
        self.factor = factor

        # Create model
        self.update_approximant()

    def update_approximant(self):
        """From the current information update the approximant coefficients.
        """
        # get matrix form
        matrix, bvector = self._get_matrix_select()
        u, s, vh = np.linalg.svd(matrix)
        #print "singular values ", s

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


    def _get_matrix_select(self, eps=None):
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
        eps : array like
            Precision of the reference points

        Returns
        -------
        matrix : np.ndarray((npoints, lenpars))
            Matrix to use for solving the linear equations
        b : np.ndarray(npoints)
            b vector for the linear equations
        """
        ml = self.ml
        nl = self.nl
        mlen = len(self.ml)
        nlen = len(self.nl)
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
                    tmp = chebypoly(point/self._xmax, ctmp)
                    tmp += chebypoly(point/self._xmax, cs)
                    matrix[i, j] = - self.factor*(1./eps[i]) * tmp
                else:
                    matrix[i, j] = - (1./eps[i])*chebypoly(point/self._xmax, cs)
            # Set values related with the denominator Q_n(x)
            # NOTE: b_0 = 1, so we only have n terms
            prefac = self._y[i]
            for j, npower in enumerate(nl):
                # start with an array of 2 elements, so it starts at T1
                cs = np.zeros(npower+1)
                cs[-1] = 1.0
                matrix[i, mlen+j] = (1./eps[i])*prefac*chebypoly(point/self._xmax, cs)
                if npower == self._n:
                    matrix[i, mlen+j] *= self.factor
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

    def residuals(self, pars, lnum=None, nonzeros_num=None, nonzeros_den=None, eps=None):
        """Compute the residuals of the model.

        Used for the optimization of parameters with LASSO or LSQS.

        Parameters
        ----------
        pars : array, float
            The current parameters, at each iteration of the optimization.
        """
        if lnum is None:
            lnum = len(self.ml)
        if nonzeros_num is None:
            raise ValueError("nonzeros_num has to be provided.")
        if nonzeros_den is None:
            raise ValueError("nonzeros_den has to be provided.")
        if eps is None:
            raise ValueError("eps has to be provided.")
        self.ml = list(nonzeros_num)
        self.nl = list(nonzeros_den)
        # Instead of using polynomials we save the coefficients
        ps = np.zeros(self._m+1)
        qs = np.zeros(self._n+1)
        # Optimize the coefficients using the least-squares non-linear solver
        ps[nonzeros_num] = pars[:lnum]
        qs[nonzeros_den] = pars[lnum:]
        self._ps = ps
        self._qs = qs
        self.update_approximant()
        vmodel = self._evaluate(self._x)[0]
        return (self._y - vmodel)/eps

