"""Tests for the set of tools in FunApp."""

import numpy as np
from numpy.testing import assert_raises
from funapp.tools import factorial, taylor



def test_factorial():
    """Test the factorial function."""
    assert factorial(3.0) == 6.0
    assert factorial(4.0) == 24.0
    assert factorial(0.0) == 1.0

test_factorial()

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

test_taylor()
