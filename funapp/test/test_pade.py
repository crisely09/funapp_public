"""Tests for PadeApproximant classes."""


import numpy as np
from nose.tools import assert_raises

from funapp.base import BaseApproximant
from funapp.pade import PadeApproximant, BasicPadeApproximant, GeneralPadeApproximant
from funapp.pade import SelectivePadeApproximant
from funapp.pade import add_prelated, add_qdenterms, add_qnumterms, clean_terms


def test_add_prelated():
    """Test function to add P(x) terms to the derivative.
    """
    # Initial terms
    iterms = [[1, [[1,1]], 2, 1.0], [2, [[2, 1], [3, 2]], 1, 2.0]]
    new_terms = []

    add_prelated(iterms[0], new_terms)
    assert new_terms[0] == [2, [[1, 1]], 2, 1.0]
    add_prelated(iterms[1], new_terms)
    assert new_terms[1] == [3, [[2, 1], [3, 2]], 1, 2.0]

#test_add_prelated()

def test_add_qdenterms():
    """Test function to add Q(x) terms in denominator to the derivative.
    """
    # Initial terms
    iterms = [[1, [[1,1]], 2, 1.0], [2, [[2, 1], [3, 2]], 1, 2.0]]
    new_terms = []

    add_qdenterms(iterms[0], new_terms)
    assert new_terms[0] == [1, [[1, 1], [1, 1]], 3, -2.0]
    add_qdenterms(iterms[1], new_terms)
    assert new_terms[1] == [2, [[2, 1], [3, 2], [1, 1]], 2, -2.0]

#test_add_qdenterms()

def test_add_qnumterms():
    """Test function to add Q(x) terms in numerator to the derivative.
    """
    # Initial terms
    iterms = [[1, [[1,1]], 2, 1.0], [2, [[2, 1], [3, 2]], 1, 2.0]]
    new_terms = []

    add_qnumterms(iterms[0], new_terms)
    assert new_terms[0] == [1, [[2, 1]], 2, 1.0]
    add_qnumterms(iterms[1], new_terms)
    assert new_terms[1] == [2, [[3, 1], [3, 2]], 1, 2.0]
    assert new_terms[2] == [2, [[3, 1], [4, 1], [2, 1]], 1, 4.0]

#test_add_qnumterms()

def test_clean_terms0():
    """Test function that cleans repeated terms.
    """
    # Initial terms
    iterms = [[1, [[1,1]], 2, 1.0], [2, [[2, 1], [3, 2]], 1, 2.0]]
    new_terms = []
    add_qdenterms(iterms[0], new_terms)
    cleaned = clean_terms(new_terms)
    assert cleaned == [[1, [[1, 2]], 3, -2.0]]

#test_clean_terms0()

def test_clean_terms1():
    """Test function that cleans repeated terms.
    """
    # Initial terms
    iterms = [[1, [[1,1]], 2, 1.0], [2, [[2, 1], [3, 2]], 1, 2.0]]
    new_terms = []
    add_prelated(iterms[0], new_terms)
    add_qdenterms(iterms[0], new_terms)
    add_qnumterms(iterms[0], new_terms)
    cleaned = clean_terms(new_terms)
    assert cleaned == [[2, [[1, 1]], 2, 1.0], [1, [[1, 2]], 3, -2.0], [1, [[2, 1]], 2, 1.0]]

#test_clean_terms1()

def test_clean_terms2():
    """Test function that cleans repeated terms."""
    # Initial terms
    iterms = [[1, [[1,1]], 2, 1.0], [2, [[2, 1], [3, 2]], 1, 2.0]]
    new_terms = []
    for i in range(len(iterms)):
        add_prelated(iterms[i], new_terms)
        add_qdenterms(iterms[i], new_terms)
        add_qnumterms(iterms[i], new_terms)
    cleaned = clean_terms(new_terms)
    assert cleaned == [[2, [[1, 1]], 2, 1.0], [1, [[1, 2]], 3, -2.0], [1, [[2, 1]], 2, 1.0],
                       [3, [[2,1], [3,2]], 1, 2.0], [2, [[2, 1], [3, 2], [1, 1]], 2, -2.0],
                       [2, [[3, 3]], 1, 2.0], [2, [[3, 1], [4, 1], [2, 1]], 1, 4.0]]

#test_clean_terms2()

def test_basicpade0():
    """Test class attributes."""
    x = [0., 1.2, 3.5, 3.5, 3.5, 4.7, 7., 7., 7., 9.0]
    y = [1.3, 1.4, 2.3, 3.4, 3.4, 4.5, 5.4, 5.6, 5.5, 6.7]


    assert_raises(TypeError, PadeApproximant, x, y, 1.0, 1)
    assert_raises(TypeError, PadeApproximant, x, y, 1, 1.0)
    pade = PadeApproximant(x, y, 1, 2)
    assert pade._xlen == len(x)-1
    assert np.allclose(pade._x, [0., 1.2, 3.5, 4.7, 7.0, 9.0], rtol=1e-4)
    assert np.allclose(pade._y, [1.3, 1.4, 2.3, 4.5, 5.4, 6.7], rtol=1e-4)
    assert pade._m == 1
    assert pade._n == 2

#test_basicpade0()

def function_tests(varx):
    r"""Function to test the approximant.
    
    :math:f(x) = e^{-(x^2)/4}
    """
    result = np.exp(-(varx**2.0)/4.0)
    return result

def derivative_tests(varx):
    r"""Derivative of the function for tests.

    :math: f'(x) = - 1/2 x  e^{-(x^2)/4}
    """
    result = -0.5*varx*np.exp(-(varx**2.0)/4.0)
    return result

def test_generalpade0():
    """Test class methods."""
    x = np.array([0.1, 1.2, 2.2, 3.5, 4.7, 5.3])
    y = function_tests(x)
    x = np.append(x, np.array(5.3))
    y = np.append(y, derivative_tests(5.3))
    pade = GeneralPadeApproximant(x, y, 2, 3)
    assert abs(pade([2.9]) - function_tests(2.9)) < 1e-2
    assert abs(pade([1.0], 1)[0] - (-0.3401)) < 1e-2
    assert abs(pade([1.0], [1, 2])[0] - (-0.3401)) < 1e-2
    assert abs(pade([1.0], [1, 2])[1] - (-0.22444)) < 1e-2

#test_generalpade0()

def test_generalpade_derivatives():
    """Test the derivatives when m!=n."""
    x = np.array([0.1, 1.2, 2.2, 3.5, 4.7, 5.3, 6.0])
    y = function_tests(x)

    # First check m > n
    pade = GeneralPadeApproximant(x, y, 5, 1)
    ders = pade(np.array([1.0]), range(3))
    # Check p-derivatives
    assert np.allclose(pade._pders[0], [pade._p[i]*(i) for i in range(1,6)[::-1]])
    assert np.allclose(pade._pders[1], [pade._p[i]*(i*(i-1)) for i in range(2,6)[::-1]])
    assert np.allclose(pade._pders[2], [pade._p[i]*(i*(i-1)*(i-2)) for i in range(3,6)[::-1]])
    # Check q-derivatives
    assert np.allclose(pade._qders[0], [pade._q[i]*(i) for i in range(1,2)[::-1]])
    der0 = pade._p(1.0)/pade._q(1.0)
    der1 = (pade._pders[0](1.0)/pade._q(1.0)) - ((pade._p(1.0)/pade._q(1.0))*(pade._qders[0](1.0)/pade._q(1.0)))
    der2 = (pade._pders[1](1.0)/pade._q(1.0)) - 2.0*(pade._pders[0](1.0)*pade._qders[0](1.0)/(pade._q(1.0)**2.0))
    der2 += 2.0*(pade._p(1.0)*(pade._qders[0](1.0)**2.0)/(pade._q(1.0)**3.0))
    assert np.allclose(der0, ders[0])
    assert np.allclose(der1, ders[1])
    assert np.allclose(der2, ders[2])

    # Now check m < n
    pade = GeneralPadeApproximant(x, y, 1, 5)
    ders = pade(np.array([1.0]), range(3))
    # Check p-derivatives
    assert np.allclose(pade._qders[0], [pade._q[i]*(i) for i in range(1,6)[::-1]])
    assert np.allclose(pade._qders[1], [pade._q[i]*(i*(i-1)) for i in range(2,6)[::-1]])
    assert np.allclose(pade._qders[2], [pade._q[i]*(i*(i-1)*(i-2)) for i in range(3,6)[::-1]])
    # Check q-derivatives
    assert np.allclose(pade._pders[0], [pade._p[i]*(i) for i in range(1,2)[::-1]])
    der0 = pade._p(1.0)/pade._q(1.0)
    der1 = (pade._pders[0](1.0)/pade._q(1.0)) - ((pade._p(1.0)/pade._q(1.0))*(pade._qders[0](1.0)/pade._q(1.0)))
    der2 = (-pade._p(1.0)*pade._qders[1](1.0))/(pade._q(1.0)**2.0) - 2.0*(pade._pders[0](1.0)*pade._qders[0](1.0)/(pade._q(1.0)**2.0))
    der2 += 2.0*(pade._p(1.0)*(pade._qders[0](1.0)**2.0)/(pade._q(1.0)**3.0))
    assert np.allclose(der0, ders[0])
    assert np.allclose(der1, ders[1])
    assert np.allclose(der2, ders[2])

test_generalpade_derivatives()


def test_selectivepade0():
    """Test class methods."""
    x = np.array([0.1, 1.2, 2.2, 3.5, 4.7, 5.3])
    y = function_tests(x)
    x = np.append(x, np.array(5.3))
    y = np.append(y, derivative_tests(5.3))
    ml = [0, 3]
    nl = [1, 2, 3]
    pade = SelectivePadeApproximant(x, y, ml, nl)
    assert abs(pade([2.2]) - function_tests(2.2)) < 1e-2
    assert abs(pade([3.5]) - function_tests(3.5)) < 1e-2
    assert abs(pade([5.0], 1)[0] - (-0.01067)) < 1e-2

#test_selectivepade0()
