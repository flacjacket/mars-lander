"""
Neural network class
"""

from __future__ import division

import numpy as np
from .layer import Layer
from .sigmoid import sigmoid_gradient


class NeuralNetwork(object):
    def __init__(self, input_size, hidden_sizes, output_size=1):
        """Neural network class

        Create a neural network that contains a single hidden layer

        Parameters
        ==========

            input_size: int
                number of inputs to the neural network

            hidden_size: int or list of ints
                number of elements in each of the hidden layers

            output_size: int
                number of output nodes, defaults to 1, so is only required for
                multi-classification

        """
        # Turn a single hidden layer call into a list
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        # don't set a regularization initially
        self._lambda = 0

        # Randomly initialize the edge weights
        _froms = np.array([input_size] + list(hidden_sizes))
        _tos = np.array(list(hidden_sizes) + [output_size])

        self.layers = [Layer(_from, _to) for _from, _to in zip(_froms, _tos)]

        cumsum = np.cumsum((_froms + 1) * _tos)
        self.theta_dim = list(zip(np.append([0], cumsum), cumsum))

    def gen_theta(self, epsilon_init=0.12, store=False):
        return np.hstack(layer.gen_theta(epsilon_init, store=store).flatten() for layer in self.layers)

    def load_theta_flat(self, theta):
        for dim, layer in zip(self.theta_dim, self.layers):
            layer.theta = theta[dim[0]:dim[1]].reshape(layer.shape)

    def load_theta(self, *thetas):
        for theta, layer in zip(thetas, self.layers):
            layer.theta = theta

    def set_lambda(self, new_lambda):
        self._lambda = new_lambda

    def _cost_unregularized(self, output_layer, df_output):
        return -np.sum(df_output * np.log(output_layer) +
                       (1 - df_output) * np.log(1 - output_layer))

    def cost(self, df_input, df_output, thetaTs):
        """Compute the cost of the current neural network

        Computes the cost for the given neural network on the set of input data
        compared to the given output

        Parameters
        ==========

            df_input: numpy array
                (N x input_size) numpy array of the input data

            df_output: numpy array
                (N x output_size) numpy array of the desired output

        Returns
        =======

            cost: float
                cost of the current current neural network on the given data

        """
        m = df_output.shape[0]

        t = 0
        a = df_input
        # Feed the input forward
        for layer, dim in zip(self.layers, self.theta_dim):
            thetaT = thetaTs[dim[0]:dim[1]].reshape(layer.shapeT)
            a = layer.feedfwd(a, thetaT)
            t += layer.regularization(thetaT)

        # Compute and return the cost
        return (self._cost_unregularized(a, df_output) +
                t * self._lambda / 2) / m

    def backpropagate(self, df_input, df_output, thetas):
        """Compute the cost and the back propagate the gradient

        Computes the cost for the given neural network on the set of input data
        compared to the given output

        Parameters
        ==========

            df_input: numpy array
                (N x input_size) numpy array of the input data

            df_output: numpy array
                (N x output_size) numpy array of the desired output

        Returns
        =======

            cost: float
                cost of the current current neural network on the given data

            grad: list of numpy arrays or numpy array
                list gradients on the theta matrices (or unrolled gradient of
                all the theta matrices)

        """
        m = df_output.shape[0]
        grad = np.empty_like(thetas)
        thetas = [thetas[dim[0]:dim[1]].reshape(layer.shapeT)
                  for layer, dim in zip(self.layers, self.theta_dim)]

        t = 0
        a = df_input
        # Feed the input forward, also storing the activations
        for layer, theta in zip(self.layers, thetas):
            a = layer.feedfwd(a, theta, store=True)
            t += layer.regularization(theta)

        # Get the cost out as well
        j = (self._cost_unregularized(a, df_output) + t * self._lambda / 2) / m

        # Backpropagate the gradient
        delta = a - df_output
        for layer, dim, prev_theta in zip(self.layers[-2::-1], self.theta_dim[::-1], thetas[::-1]):
            grad[dim[0]:dim[0]+delta.shape[1]] = np.sum(delta, axis=0)
            grad[dim[0]+delta.shape[1]:dim[1]] = (np.dot(layer.a.T, delta) + self._lambda * prev_theta[1:]).flatten()
            delta = np.dot(delta, prev_theta.T)[:, 1:] * sigmoid_gradient(layer.z)

        dim = self.theta_dim[0]
        grad[0:delta.shape[1]] = np.sum(delta, axis=0)
        grad[delta.shape[1]:dim[1]] = (np.dot(df_input.T, delta) + self._lambda * thetas[0][1:]).flatten()

        return j, grad / m

    def predict(self, df_input, thetaTs):
        """Predict the output for given input parameters"""
        a = df_input
        # Feed the input forward
        for layer, dim in zip(self.layers, self.theta_dim):
            thetaT = thetaTs[dim[0]:dim[1]].reshape(layer.shapeT)
            a = layer.feedfwd(a, thetaT)

        if self.theta_dim[-1, 3] == 1:
            a = np.round(a).astype(int)
        else:
            a = np.argmax(a, axis=1)

        return a
