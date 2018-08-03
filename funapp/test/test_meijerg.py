"""Tests for Meijer Approximant and auxiliary functions."""


import numpy as np
from nose.tools import assert_raises

from funapp.meijer import borel_trans_coeffs, consecutive_ratios_odd, rational_function_for_ratios
from funapp.meijer import find_roots, gamma_products, meijerg_approx_low, aux_little_series
from funapp.meijer import cma_solver


def test_borel_trans_coeffs():
    """Test the function to get the Borel transform coefficients."""
    coeffs = np.arange(0.4, 1.6, 0.2)
    borel = borel_trans_coeffs(coeffs)
    result0 = [0.4, 0.6, 0.4, 0.166667, 0.05, 0.0116667, 0.00222222]
    assert np.allclose(borel, result0)

test_borel_trans_coeffs()

def test_consecutive_ratios_odd():
    """Test consecutive ratios of Borel coefficients for the case with a series of odd order.
    """
    coeffs = np.arange(0.4, 1.6, 0.2)
    borel = borel_trans_coeffs(coeffs)
    ratios = consecutive_ratios_odd(borel)
    result0 = [0.6/0.4, 0.4/0.6, 0.166667/0.4, 0.05/0.166667, 0.0116667/0.05, 0.00222222/0.0116667]
    assert np.allclose(ratios, result0)

test_consecutive_ratios_odd()

def test_rational_function_for_ratios():
    """Test the rational function fit for the consecutive ratios of Borel coefficients."""
    ratios = np.array([-1./8., -35./96., -11./24.])
    assert_raises(TypeError, rational_function_for_ratios, [1, 2, 3])
    ps, qs = rational_function_for_ratios(ratios)
    # Make polynomials to check prediction of ratios
    assert np.allclose(ps, [-1./8., -113./216])
    assert np.allclose(qs, [7./9])

test_rational_function_for_ratios()

def test_find_roots():
    """Test of the wrapper function to find the roots of a polynomial."""
    ratios = np.array([-1./8., -35./96., -11./24.])
    assert_raises(TypeError, rational_function_for_ratios, [1, 2, 3])
    ps, qs = rational_function_for_ratios(ratios)
    # Add the constant 1. to the qm coefficients
    ps = ps[::-1]
    qs = qs[::-1]
    qs = np.append(qs, np.array(1.))
    rootsp = find_roots(ps)
    rootsq = find_roots(qs)
    assert np.allclose(rootsp, [- ps[1]/ps[0]])
    assert np.allclose(rootsq, [-1/qs[0]])

test_find_roots()

def test_gamma_product():
    """Test for the Gamma function products"""
    y = np.array([9/7.])
    x = np.array([27/113.])
    result01 = gamma_products(y, 1)
    result02 = gamma_products(x, 1)
    assert abs(result01 - 0.89974717650283) < 1e-9
    assert abs(result02 - 3.803298638983) < 1e-9

test_gamma_product()

def test_meijerg():
    """Test Meijer-G approximant, comparing with results from the paper of Mera 2018."""
    ratios = np.array([-1./8., -35./96., -11./24.])
    ps, qs = rational_function_for_ratios(ratios)
    # Add the constant 1. to the qm coefficients
    ps = ps[::-1]
    qs = qs[::-1]
    qs = np.append(qs, np.array(1.))
    rootsp = find_roots(ps)
    rootsq = find_roots(qs)
    xvector = np.array([1, rootsp[0]])
    yvector = np.array([rootsq[0]])
    pl = ps[0]
    ql = qs[0]
    points = np.array([-1+0j, -10+0j, -100+0j])
    zs = meijerg_approx_low(xvector, yvector, pl, ql, points)
    print zs
    assert (zs - [1.133285+0.144952j, 0.744345+0.436317j, 0.386356+0.321210j] < 1e-6).all()
#test_meijerg()

def test_example2():
    """Test Meijer-G approximant for equation 37."""
    coeffs = np.array([1/2., 3/4., -21/8., 333/16., -30885./128, 916731/256.])
    borel = borel_trans_coeffs(coeffs)
    print borel
    ratios = consecutive_ratios_odd(borel)
    print ratios
    ps, qs = rational_function_for_ratios(ratios)
    print "ps, qs", ps, qs
    # Add the constant 1. to the qm coefficients
    ps = ps[::-1]
    qs = qs[::-1]
    qs = np.append(qs, np.array(1.))
    rootsp = find_roots(ps)
    rootsq = find_roots(qs)
    xvector = np.append(np.array([1]), rootsp)
    yvector = rootsq
    print "xvector", xvector
    print "yvector", yvector
    pl = ps[0]
    ql = qs[0]
    points = np.array([1., 2., 50.])
    zs = meijerg_approx_low(xvector, yvector, pl, ql, points)
    print zs/2
#test_example2()

def test_example3():
    coeffs = np.array([1, -1, 2.667, -4.667])
    borel = borel_trans_coeffs(coeffs)
    ratios = consecutive_ratios_odd(borel)
    ps, qs = rational_function_for_ratios(ratios)
    # Add the constant 1. to the qm coefficients
    ps = ps[::-1]
    qs = qs[::-1]
    qs = np.append(qs, np.array(1.))
    rootsp = find_roots(ps)
    rootsq = find_roots(qs)
    xvector = np.append(np.array([1]), rootsp)
    yvector = rootsq
    pl = ps[0]
    ql = qs[0]
    points = np.array([0.6736])
    zs = meijerg_approx_low(xvector, yvector, pl, ql, points)
    assert abs(1 - zs) < 1e-3

test_example3()

def test_example4():
    coeffs = np.array([1, -0.667, 0.556, -2.056])
    borel = borel_trans_coeffs(coeffs)
    ratios = consecutive_ratios_odd(borel)
    ps, qs = rational_function_for_ratios(ratios)
    # Add the constant 1. to the qm coefficients
    ps = ps[::-1]
    qs = qs[::-1]
    qs = np.append(qs, np.array(1.))
    rootsp = find_roots(ps)
    rootsq = find_roots(qs)
    xvector = np.append(np.array([1]), rootsp)
    yvector = rootsq
    pl = ps[0]
    ql = qs[0]
    points = np.array([0.5381])
    zs = meijerg_approx_low(xvector, yvector, pl, ql, points)
    print "zs", zs
    v = 1/(2. + zs)
    result0 = 0.5921
    print v, result0
#test_example4()

def test_example5():
    coeffs = np.array([1, 6, 210, 13860])
    borel = borel_trans_coeffs(coeffs)
    ratios = consecutive_ratios_odd(borel)
    ps, qs = rational_function_for_ratios(ratios)
    # Add the constant 1. to the qm coefficients
    ps = ps[::-1]
    qs = qs[::-1]
    qs = np.append(qs, np.array(1.))
    rootsp = find_roots(ps)
    rootsq = find_roots(qs)
    xvector = np.append(np.array([1]), rootsp)
    yvector = rootsq
    pl = ps[0]
    ql = qs[0]
    points = np.array([1, 10, 100])
    zs = meijerg_approx_low(xvector, yvector, pl, ql, points)
    result0 = [0.473794 + 0.368724j, 0.255694 + 0.228610j, 0.144490 + 0.133539j]
    assert (abs(zs - result0) < 1e-6).all()
test_example5()

def test_example6():
    coeffs = np.array([1, 1/2., 9/8., 75/16.])
    borel = borel_trans_coeffs(coeffs)
    ratios = consecutive_ratios_odd(borel)
    ps, qs = rational_function_for_ratios(ratios)
    # Add the constant 1. to the qm coefficients
    ps = ps[::-1]
    qs = qs[::-1]
    qs = np.append(qs, np.array(1.))
    rootsp = find_roots(ps)
    rootsq = find_roots(qs)
    xvector = np.append(np.array([1]), rootsp)
    yvector = rootsq
    pl = ps[0]
    ql = qs[0]
    points = np.array([1, 100])
    zs = meijerg_approx_low(xvector, yvector, pl, ql, points)
    result0 = [0.990312240887789089 + 0.481308237536857j, 0.13677671640883210679 + 0.23483780883795517888j]
    assert (abs(zs - result0) < 1e-7).all
test_example6()


def check_cma():
    """Check if cma module is available."""
    try:
        import cma
    except ModuleNotFoundError:
        return False
    else:
        return True


def test_cma():
    from scipy.optimize import rosen
    x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
    results = cma_solver(rosen, x0)
    assert results['success']
    assert np.allclose(results['optvalue'], 0)
    assert np.allclose(results['params'], np.ones(5))
    assert results['message'] == 'Following termination conditions are satisfied: tolfun: 1e-11.'
test_cma()
