"""
Neural network class
"""

import numpy as np
from .nn_cy import _feedfwd, _cost, _grad


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
        map_from = np.array([input_size] + list(hidden_sizes)) + 1
        map_to = np.array(list(hidden_sizes) + [output_size])

        theta_ind = np.cumsum(map_to * map_from)
        theta_ind0 = theta_ind.copy()
        theta_ind0[1:] = theta_ind0[:-1]
        theta_ind0[0] = 0

        self.theta_dim = np.ascontiguousarray(
                np.vstack([theta_ind0, theta_ind, map_from, map_to]).T
        )

    def gen_theta(self, epsilon_init=0.12):
        theta_len = self.theta_dim[-1, 1]
        return 2 * epsilon_init * np.random.rand(theta_len) - epsilon_init

    def set_lambda(self, new_lambda):
        self._lambda = new_lambda

    def cost(self, df_input, df_output, theta):
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
        a = _feedfwd(df_input, theta, self.theta_dim)[0][-1]
        return _cost(a, df_output, theta, self.theta_dim, self._lambda)

    def backpropagate(self, df_input, df_output, theta):
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
        # Compute the forward propagation parameters
        a, z = _feedfwd(df_input, theta, self.theta_dim)
        # Get the cost out as well
        j = _cost(a[-1], df_output, theta, self.theta_dim, self._lambda)
        # Compute the gradient
        grad = _grad(a, z, df_input, df_output, theta, self.theta_dim, self._lambda)

        return j, grad

    def predict(self, df_input, theta):
        """Predict the output for given input parameters"""
        df_output = _feedfwd(df_input, theta, self.theta_dim)[0][-1]

        if self.theta_dim[-1, 3] == 1:
            df_output = np.round(df_output).astype(int)
        else:
            df_output = np.argmax(df_output, axis=1)

        return df_output
