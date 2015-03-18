import numpy as np
cimport numpy as np

cpdef inline np.ndarray sigmoid(np.ndarray z)
cpdef inline np.ndarray sigmoid_gradient(np.ndarray z)