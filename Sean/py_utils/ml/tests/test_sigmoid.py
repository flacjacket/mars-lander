"""
Test sigmoid and sigmoid gradient
"""

from py_utils.ml.sigmoid import sigmoid, sigmoid_gradient

import numpy as np

epsilon = 0.00001


def _approx(x, y):
    return np.all(np.abs(x - y) < epsilon)


def test_sigmoid():
    assert _approx(sigmoid(np.ndarray([0])), np.ndarray([0.5]))

    x = np.array([-1, 0, 1])
    y = np.array([0.268941421369,
                  0.5,
                  0.731058578630])
    assert _approx(sigmoid(x), y)


def test_sigmoid_gradient():
    assert _approx(sigmoid_gradient(np.ndarray([0])), np.ndarray([0.25]))

    x = np.array([-1, -0.5, 0, 0.5, 1])
    y = np.array([0.196612, 0.235004, 0.250000, 0.235004, 0.196612])
    assert _approx(sigmoid_gradient(x), y)
