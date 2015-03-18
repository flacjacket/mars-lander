"""
Definitions of sigmoid and its gradiet
"""

import numpy as np
cimport numpy as np


cpdef inline np.ndarray sigmoid(np.ndarray z):
    return 1 / (1 + np.exp(-z))


cpdef inline np.ndarray sigmoid_gradient(np.ndarray z):
    return sigmoid(z) * (1 - sigmoid(z))
