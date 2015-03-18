import cython

import numpy as np
cimport numpy as np

from .sigmoid cimport sigmoid, sigmoid_gradient


###############################################################################
# Feedforward
###############################################################################



@cython.boundscheck(False)
@cython.wraparound(False)
cdef _compute_layer0(np.ndarray df_input,
                     np.ndarray[double, ndim=1, mode='c'] theta,
                     long [:] theta_dim,
                     np.ndarray[double, ndim=1, mode='c'] a,
                     np.ndarray[double, ndim=1, mode='c'] z):
    cdef np.ndarray[double, ndim=2] theta_mat = \
        theta[theta_dim[0]:theta_dim[1]].reshape((theta_dim[2], theta_dim[3]))
    #np.dot(df_input, theta_mat[1:])
    z[:] = (np.dot(df_input, theta_mat[1:]) + theta_mat[0]).flatten()
        #theta[theta_dim[0]:theta_dim[1]:theta_dim[3]]
    a[:] = sigmoid(z)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _compute_layer(np.ndarray[double, ndim=2, mode='c'] df_input,
                    np.ndarray[double, ndim=1, mode='c'] theta,
                    long [:] theta_dim,
                    np.ndarray[double, ndim=1, mode='c'] a,
                    np.ndarray[double, ndim=1, mode='c'] z):
    cdef np.ndarray[double, ndim=2] theta_mat = \
        theta[theta_dim[0]:theta_dim[1]].reshape((theta_dim[2], theta_dim[3]))
    z[:] = (np.dot(df_input, theta_mat[1:]) + theta_mat[0]).flatten()
    a[:] = sigmoid(z)


#@cython.boundscheck(False)
@cython.wraparound(False)
def _feedfwd(np.ndarray df_input,
             np.ndarray[double, ndim=1, mode='c'] theta,
             long [:, :] theta_dims):
        # We build the intermediate values as we go
        cdef long feed_len = np.sum(theta_dims[:, 3]) * df_input.shape[0]
        cdef np.ndarray[long, ndim=1] ind = np.cumsum(theta_dims[:, 3]) * df_input.shape[0]
        cdef np.ndarray[double, ndim=1] a = np.empty(feed_len)
        cdef np.ndarray[double, ndim=1] z = np.empty(feed_len)
        print(ind)

        # feed forward from the input
        _compute_layer0(df_input, theta, theta_dims[0], a[0:ind[0]], z[0:ind[0]])

        # feed forward the rest of the neural network
        cdef int i
        for i in range(1, theta_dims.shape[0]):
            _compute_layer(a[ind[i-2]:ind[i-1]].reshape((df_input.shape[0], -1)), theta, theta_dims[i],
                           a[ind[i-1]:ind[i]], z[ind[i-1]:ind[i]])

        return a, z


###############################################################################
# Cost
###############################################################################


@cython.boundscheck(False)
@cython.wraparound(False)
def _cost(np.ndarray df_input,
          np.ndarray[unsigned char, ndim=2, mode='c'] df_output,
          np.ndarray theta,
          long [:, :] theta_dims,
          double _lambda):
    cdef int m = df_output.shape[0]

    # compute the regularization term
    cdef double t = sum(np.sum(theta[dim[0]+dim[3]:dim[1]]**2) for dim in theta_dims)
    # compute the cost with regularization
    cdef double j = -np.sum(df_output * np.log(df_input) + (1 - df_output) * np.log(1 - df_input)) / m + \
        _lambda * t / (2 * m)

    return j


###############################################################################
# Gradient
###############################################################################


#@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[double, ndim=2] _compute_grad(np.ndarray delta,
                                                     np.ndarray a,
                                                     np.ndarray theta_trunc,
                                                     int m, double _lambda):
    return (np.dot(a.T, delta) + _lambda * theta_trunc) / m


#@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[double, ndim=2] _compute_grad0(np.ndarray delta,
                                                      np.ndarray df_input,
                                                      np.ndarray theta_trunc,
                                                      int m,
                                                      double _lambda):
    return _compute_grad(delta, df_input, theta_trunc, m, _lambda)


def _grad(a, z,
          np.ndarray df_input,
          np.ndarray[unsigned char, ndim=2] df_output,
          np.ndarray[double, ndim=1] theta,
          np.ndarray[long, ndim = 2] theta_dims,
          double _lambda):
    # Number of training examples
    cdef int m = df_output.shape[0]

    cdef np.ndarray[double, ndim=1] grads = np.zeros_like(theta)
    # 'final' delta is difference between computed and expected
    cdef np.ndarray[double, ndim=2] delta = a[-1] - df_output

    # compute the gradients back
    cdef np.ndarray[long, ndim=1] dim
    cdef np.ndarray[double, ndim=2] th_trunc
    cdef long i
    for i in range(len(theta_dims)-1, 0, -1):
        dim = theta_dims[i]
        th_trunc = theta[dim[0]+dim[3]:dim[1]].reshape((dim[2]-1, dim[3]))
        # Compute the gradient at the -i layer
        grads[dim[0]+dim[3]:dim[1]] = _compute_grad(delta, a[i-1], th_trunc, m, _lambda).flatten()

        # Compute the delta on the next iteration
        delta = np.dot(delta, th_trunc.T) * sigmoid_gradient(z[i-1])

    dim = theta_dims[0]
    th_trunc = theta[dim[0]+dim[3]:dim[1]].reshape((dim[2]-1, dim[3]))
    grads[dim[0]+dim[3]:dim[1]] = _compute_grad0(delta, df_input, th_trunc, m, _lambda).flatten()

    return grads
