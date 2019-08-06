"""Tests for Meijer Approximant and auxiliary functions."""


import numpy as np
from scipy import misc
import mpmath
from nose.tools import assert_raises

from funapp.meijer import borel_trans_coeffs, consecutive_ratios_odd, rational_function_for_ratios
from funapp.meijer import find_roots, gamma_products, meijerg_approx_low, aux_little_series
from funapp.meijer import cma_solver
from funapp.pade import GeneralPadeApproximant, BasicPadeApproximant
from funapp.tools import factorial, taylor_coeffs


def test_borel_trans_coeffs():
    """Test the function to get the Borel transform coefficients."""
    coeffs = np.arange(0.4, 1.6, 0.2)
    borel = borel_trans_coeffs(coeffs)
    result0 = [0.4, 0.6, 0.4, 0.166667, 0.05, 0.0116667, 0.00222222]
    assert np.allclose(borel, result0)

#test_borel_trans_coeffs()

def test_consecutive_ratios_odd():
    """Test consecutive ratios of Borel coefficients for the case with a series of odd order.
    """
    coeffs = np.arange(0.4, 1.6, 0.2)
    borel = borel_trans_coeffs(coeffs)
    ratios = consecutive_ratios_odd(borel)
    result0 = [0.6/0.4, 0.4/0.6, 0.166667/0.4, 0.05/0.166667, 0.0116667/0.05, 0.00222222/0.0116667]
    assert np.allclose(ratios, result0)

#test_consecutive_ratios_odd()

def test_rational_function_for_ratios():
    """Test the rational function fit for the consecutive ratios of Borel coefficients."""
    ratios = np.array([-1./8., -35./96., -11./24.])
    assert_raises(TypeError, rational_function_for_ratios, [1, 2, 3])
    ps, qs = rational_function_for_ratios(ratios)
    # Make polynomials to check prediction of ratios
    assert np.allclose(ps, [-1./8., -113./216])
    assert np.allclose(qs, [7./9])

#test_rational_function_for_ratios()

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

#test_find_roots()

def test_gamma_product():
    """Test for the Gamma function products"""
    y = np.array([9/7.])
    x = np.array([27/113.])
    result01 = gamma_products(y, 1)
    result02 = gamma_products(x, 1)
    assert abs(result01 - 0.89974717650283) < 1e-9
    assert abs(result02 - 3.803298638983) < 1e-9

#test_gamma_product()

def test_meijerg_ex():
    """Test a Meijer-G approximant of a Meijer-G function."""
    # e^x = 1 + x + 1/2 x^2 + 1/6 x^3 + 1/24 x^4 + ...
    tcoeffs = np.array([1, 1, 1./2., 1./6., 1./24.])
    borel = borel_trans_coeffs(tcoeffs)
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
    points = np.array([1.2, -1.4])
    result0 = np.exp(points)
    zs = meijerg_approx_low(xvector, yvector, pl, ql, points)
    assert (abs(result0 - np.real(zs)) < 1e-7).all()
#test_meijerg_ex()


def test_maijerg_ex_frompade():
    """Test a Meijer-G approximant of a Meijer-G function."""
    # e^x = 1 + x + 1/2 x^2 + 1/6 x^3 + 1/24 x^4 + ...
    tcoeffs = np.array([1, 1, 1./2., 1./6., 1./24.+ 1./120., 1/factorial(6.), 1/factorial(7.)])
    p, q = misc.pade(tcoeffs, 3)
    points = np.array([0.8, 0.5, 0.2, -0.5, -0.8, -1.0])
    y = np.exp(points)
    basic_pade = BasicPadeApproximant(points, y, tcoeffs, 3)
    tcoeffs_pade = taylor_coeffs(basic_pade(np.array([0.]), range(4)), 3)
    borel = borel_trans_coeffs(tcoeffs_pade)
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
    #print tcoeffs_pade
    zs = meijerg_approx_low(xvector, yvector, pl, ql, points)
    #print "from pade ", p(points)/q(points)
    #print "result 0", y
    #print "result 1 ", zs
    assert (abs(y - np.real(zs)) < 1e-2).all()
#test_maijerg_ex_frompade()


def optimize_pars_meijerg(xvec, yvec, pl, ql, points, vals):
    from scipy import optimize
    xlen = len(xvec)
    ylen = len(yvec)
    plql = np.array([pl, ql])
    pars = np.append(xvec, yvec)
    pars = np.append(pars, plql)
    def opt_meijer(pars):
        xvector = pars[:xlen]
        yvector = pars[xlen:xlen+ylen]
        #print "xvec ", xvector
        #print "yvec ", yvector
        pl = pars[-2]
        ql = pars[-1]
        #print "pl ", pl
        #print "ql ", ql
        try:
            zs = meijerg_approx_low(xvector, yvector, pl, ql, points, maxterms=1000)
            if (zs != np.inf).any() or (zs != np.nan).any():
                error = np.power(np.linalg.norm(vals - np.real(zs)), 2)
                print "Error: ", error
                print "zs ", np.real(zs)
                return error
            else:
                return 1000.00
        except mpmath.libmp.libhyper.NoConvergence:
            return 100.00

    #result = cma_solver(opt_meijer, pars)
    #good_pars = result['params']
    #optz = result['optvalue']
    #return good_pars, optz
    good_pars = optimize.minimize(opt_meijer, pars)
    return good_pars


    

def test_maijerg_ex_recursive():
    """Test a Meijer-G approximant of a Meijer-G function."""
    # e^-x = 1 - x + 1/2 x^2 - 1/6 x^3 + 1/24 x^4 - ...
    # First construct Meijer-G from Taylor expansion
    tcoeffs = np.array([1, -1, 1./2., -1./6., 1./24.])
    borel = borel_trans_coeffs(tcoeffs)
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
    points = np.array([0.01, 0.1, 0.15, 0.21, 0.23, 0.29, 0.31, 0.32, 0.3, 0.25])
    result0 = np.exp(-points)
    print result0
    zs = meijerg_approx_low(xvector, yvector, pl, ql, points)
    print "initial Meijer-G ", zs

    # Then make a Pade out of the Meijer-G results
    gen_pade = GeneralPadeApproximant(points, np.real(zs), 4, 6)
    tcoeffs_pade = taylor_coeffs(gen_pade(np.array([0.]), range(6)), 5)
    borel1 = borel_trans_coeffs(tcoeffs_pade)
    ratios1 = consecutive_ratios_odd(borel)
    ps1, qs1 = rational_function_for_ratios(ratios)
    # Add the constant 1. to the qm coefficients
    ps1 = ps1[::-1]
    qs1 = qs1[::-1]
    qs1 = np.append(qs, np.array(1.))
    rootsp1 = find_roots(ps1)
    rootsq1 = find_roots(qs1)
    xvector1 = np.append(np.array([1]), rootsp1)
    yvector1 = rootsq1
    pl1 = ps1[0]
    ql1 = qs1[0]
    #print tcoeffs_pade
    zs1 = meijerg_approx_low(xvector1, yvector1, pl1, ql1, points)
    print "True result", result0
    print "from pade ", gen_pade(points)
    print "result 1 ", np.real(zs1)
    #y, optz = optimize_pars_meijerg(xvector1, yvector1, pl1, ql1, points, result0)
    #print "result value", optz
    y = optimize_pars_meijerg(xvector1, yvector1, pl1, ql1, points, result0)
    print "result params", y
    print "True result", result0
test_maijerg_ex_recursive()


def test_meijerg_1o1x():
    """Test a Meijer-G approximant of a Meijer-G function."""
    # 1/1-x = 1+ x + x^2 + x^3 + x^4 + ...
    tcoeffs = np.array([1., 1., 1., 1., 1., 1., 1.])
    borel = borel_trans_coeffs(tcoeffs)
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
    points = np.array([0.25, -0.41])
    result0 = 1. /(1. - points)
    zs = meijerg_approx_low(xvector, yvector, pl, ql, points)
    assert (abs(result0 - np.real(zs)) < 1e-7).all()
#test_meijerg_1o1x()


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
    #print zs
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

#test_example3()

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
#test_example5()

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
#test_example6()


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
#test_cma()
