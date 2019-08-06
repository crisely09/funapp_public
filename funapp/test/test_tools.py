"""Tests for the set of tools in FunApp."""

import numpy as np
import itertools
from numpy.testing import assert_raises
import matplotlib.pyplot as plt

from funapp.tools import factorial, taylor, taylor_coeffs
from funapp.tools import fit_pseudosemivariance, variance_column, get_variance_matrix
from funapp.tools import estimate_standard_deviation



def test_factorial():
    """Test the factorial function."""
    assert factorial(3.0) == 6.0
    assert factorial(4.0) == 24.0
    assert factorial(0.0) == 1.0

#test_factorial()

def test_taylor():
    """Check the Taylor expansion constructor."""
    def func(points):
        """Test function."""
        return points**3

    def first_der(points):
        """Derivative of function."""
        return 3.0 * points**2.0

    def second_der(points):
        """Second derivative of function."""
        return 6.0 * points

    points = np.array([1.2, 1.4, 2.5])
    fnvals = func(points)
    ders1 = first_der(points)
    ders2 = second_der(points)
    ders = [fnvals[0], ders1[0], ders2[0]]
    eps = 1e-2
    sgn = 1
    # Check parameters
    assert_raises(TypeError, taylor, 's', 2, eps, sgn)
    assert_raises(TypeError, taylor, ders, 2.0, eps, sgn)
    assert_raises(ValueError, taylor, [fnvals[0], ders1[0]], 2, eps, sgn)
    assert_raises(TypeError, taylor, ders, 2, 1, sgn)
    assert_raises(ValueError, taylor, ders, 2, eps, 0)


    result0 = fnvals[0] + eps*ders1[0] + 0.5*(eps**2)*ders2[0]
    result1 = taylor(ders, 2, eps, 1)
    print result0
    print result1
    assert abs(result0 - result1) < 1e-6

#test_taylor()

def test_taylor_coeffs():
    """Check the Taylor expansion constructor."""
    def func(points):
        """Test function."""
        return points**3

    def first_der(points):
        """Derivative of function."""
        return 3.0 * points**2.0

    def second_der(points):
        """Second derivative of function."""
        return 6.0 * points

    points = np.array([1.2, 1.4, 2.5])
    fnvals = func(points)
    ders1 = first_der(points)
    ders2 = second_der(points)
    ders = [fnvals[0], ders1[0], ders2[0]]
    # Check parameters
    assert_raises(TypeError, taylor_coeffs, 's', 2)
    assert_raises(TypeError, taylor_coeffs, ders, 2.0)
    assert_raises(ValueError, taylor_coeffs, [fnvals[0], ders1[0]], 2)

    point = np.array([1.])
    ders = [func(point), first_der(point), second_der(point)]
    coeffs = taylor_coeffs(ders, 2)
    assert np.allclose(coeffs, [1.0, 3.0, 3.0])

#test_taylor_coeffs()

def test_fit_pseudosemivariance():
    """Check the function for fitting the semivariance."""
    # Check input
    assert_raises(TypeError, fit_pseudosemivariance, 'acb', [0,1,2])
    assert_raises(TypeError, fit_pseudosemivariance, [0,1,2], 'hola')
    xdata = np.arange(0.1, 1, 0.1)
    np.random.seed(1729)
    y_noise = 0.005 * np.random.normal(size=xdata.size)
    ydata = np.exp(-xdata**2) + y_noise
    szero, aexp = fit_pseudosemivariance(xdata, ydata)
    distance_y = [0.5*(k - l)**2 for (k, l) in itertools.combinations_with_replacement(ydata, 2)]
    distance_x = [abs(k - l) for (k, l) in itertools.combinations_with_replacement(xdata, 2)]
    xfit = np.arange(min(distance_x), max(distance_x), 0.1)
    fit = szero*(1-np.exp(-aexp*(xfit**2)))
    plt.plot(distance_x, distance_y, 'o', label='Semivariance')
    plt.plot(xfit, fit, label='Fit')
    plt.show()
    # Plot looks OK

#test_fit_pseudosemivariance()

def test_variance_column():
    """Check function to evaluate each column of the variance matrix."""
    xdata = np.arange(0.1, 1, 0.1)
    szero = 0.25
    aexp = 0.125
    point = 1.1
    result0 = [0.220624, 0.225927, 0.230779, 0.235147, 0.238999, 0.242308, 0.24505,
               0.247203, 0.248753, 1.0]
    result1 = variance_column(xdata, point, szero, aexp)
    assert np.allclose(result1, result0)

test_variance_column()


def test_variance_matrix():
    """Test function to construct variance matrix."""
    xdata = np.arange(0.1, 1, 0.1)
    szero = 0.25
    aexp = 0.125
    matrix = get_variance_matrix(xdata, szero, aexp)
    result0 = np.array([[0.25, 0.249688, 0.248753, 0.247203, 0.24505, 0.242308, 0.238999, 0.235147, 0.230779],
               [0.249688, 0.25, 0.249688, 0.248753, 0.247203, 0.24505, 0.242308, 0.238999, 0.235147],
               [0.248753, 0.249688, 0.25, 0.249688, 0.248753, 0.247203, 0.24505, 0.242308, 0.238999],
               [0.247203, 0.248753, 0.249688, 0.25, 0.249688, 0.248753, 0.247203, 0.24505, 0.242308],
               [0.24505, 0.247203, 0.248753, 0.249688, 0.25, 0.249688, 0.248753, 0.247203, 0.24505],
               [0.242308, 0.24505, 0.247203, 0.248753, 0.249688, 0.25, 0.249688, 0.248753, 0.247203],
               [0.238999, 0.242308, 0.24505, 0.247203, 0.248753, 0.249688, 0.25, 0.249688, 0.248753],
               [0.235147, 0.238999, 0.242308, 0.24505, 0.247203, 0.248753, 0.249688, 0.25, 0.249688],
               [0.230779, 0.235147, 0.238999, 0.242308, 0.24505, 0.247203, 0.248753, 0.249688, 0.25]])
    assert matrix[-1, -1] == 0.0
    assert (matrix[-1, :len(xdata)] == 1.0).all()
    assert np.allclose(matrix[:len(xdata), :len(xdata)], result0)

test_variance_matrix()

def test_estimate_standard_deviation():
    """Check the function to estimate the standard deviation."""
    xdata = np.arange(0.1, 1, 0.1)
    point = 1.1
    szero = 0.25
    aexp = 0.125
    cmatrix = get_variance_matrix(xdata, szero, aexp)
    vnew = variance_column(xdata, point, szero, aexp)
    # Solve equations for the weigths
    result = np.linalg.solve(cmatrix, vnew)
    # Take out the last element
    ws = result[:len(xdata)]
    n_cmatrix = cmatrix[:len(xdata),:len(xdata)]
    # Computation of sigma
    ws_cmatrix_ws = np.dot(ws, np.dot(n_cmatrix, ws))
    cnew = szero
    result0 = np.sqrt(ws_cmatrix_ws - 2*np.dot(vnew[:len(xdata)], ws) + cnew)
    result1 = estimate_standard_deviation(xdata, point, szero, aexp)
    assert abs(result0 - result1) < 1e-11

test_estimate_standard_deviation()
