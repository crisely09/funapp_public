"""Tests for BaseApproximant class."""


import numpy as np
from funapp.base import BaseApproximant


def test_base0():
    """
    Test class attributes.
    """
    x = [0., 1.2, 3.5, 3.5, 3.5, 4.7, 7., 7., 7., 9.0]
    y = [1.3, 1.4, 2.3, 3.4, 3.4, 4.5, 5.4, 5.6, 5.5, 6.7]

    base = BaseApproximant(x, y)

    assert base._xlen == len(x)-1
    assert np.allclose(base._x, [0., 1.2, 3.5, 4.7, 7.0, 9.0], rtol=1e-4)
    assert np.allclose(base._y, [1.3, 1.4, 2.3, 4.5, 5.4, 6.7], rtol=1e-4)
    assert base._der == [[3.5, [3.4]], [7.0, [5.6, 5.5]]]
    assert base._der == [[3.5, [3.4]], [7.0, [5.6, 5.5]]]

test_base0()

def test_base1():
    """
    Test class attributes.
    """
    x = [0., 1.2, 3.5, 3.5, 3.5, 3.5, 3.5, 4.7, 7., 7., 7., 9.0]
    y = [1.3, 1.4, 2.3, 3.4, 3.4, 3.5, 3.5, 4.5, 5.4, 5.6, 5.5, 6.7]

    base = BaseApproximant(x, y)

    assert base._xlen == len(x)-2
    assert base._der == [[3.5, [3.4, 3.5]], [7.0, [5.6, 5.5]]]
    assert np.allclose(base._x, [0., 1.2, 3.5, 4.7, 7.0, 9.0], rtol=1e-4)
    assert np.allclose(base._y, [1.3, 1.4, 2.3, 4.5, 5.4, 6.7], rtol=1e-4)

test_base1()

def test_base2():
    """
    Test class attributes.
    """
    x = [0., 1.2, 3.5, 4.7, 7., 9.0]
    y = [1.3, 1.4, 2.3, 3.5, 5.4, 6.7]

    base = BaseApproximant(x, y)

    assert base._xlen == len(x)
    assert base._der == []
    assert np.allclose(base._x, [0., 1.2, 3.5, 4.7, 7.0, 9.0], rtol=1e-4)
    assert np.allclose(base._y, [1.3, 1.4, 2.3, 3.5, 5.4, 6.7], rtol=1e-4)

test_base2()
