"""
Test neural network
"""

from py_utils.ml.nn import NeuralNetwork

import numpy as np

epsilon = 0.000001

cost_npzfile = "py_utils/ml/tests/cost.npz"


def test_neuralnetwork():
    # Check creation of the neural networks and the manipulation of values
    # Single hidden layer
    input_size = 20
    hidden_size = 10

    nn = NeuralNetwork(input_size, hidden_size)

    theta2_index = hidden_size * (input_size + 1)
    end_index = theta2_index + (hidden_size + 1)
    theta_dim = np.array([[0, theta2_index, (input_size + 1), hidden_size],
                          [theta2_index, end_index, (hidden_size + 1), 1]])

    assert nn.theta_dim.shape == (2, 4)
    assert np.all(nn.theta_dim == theta_dim)

    theta = nn.gen_theta()
    assert theta.size == hidden_size * (input_size + 1) + (hidden_size + 1)

    # Multiple hidden layers
    input_size = 20
    hidden_sizes = [10, 10, 10]

    nn = NeuralNetwork(input_size, hidden_sizes)

    theta2_index = hidden_sizes[0] * (input_size + 1)
    theta3_index = theta2_index + hidden_sizes[1] * (hidden_sizes[0] + 1)
    theta4_index = theta3_index + hidden_sizes[2] * (hidden_sizes[1] + 1)
    end_index = theta4_index + (hidden_sizes[2] + 1)
    theta_dim = np.array([[0, theta2_index, hidden_sizes[0], (input_size + 1)],
                          [theta2_index, theta3_index, hidden_sizes[1], (hidden_sizes[0] + 1)],
                          [theta3_index, theta4_index, hidden_sizes[2], (hidden_sizes[1] + 1)],
                          [theta4_index, end_index, 1, (hidden_sizes[2] + 1)]])

    theta_size = hidden_sizes[0] * (input_size + 1) + \
        hidden_sizes[1] * (hidden_sizes[0] + 1) + \
        hidden_sizes[2] * (hidden_sizes[1] + 1) + \
        (hidden_sizes[0] + 1)

    assert nn.theta_dim.shape == (4, 4)
    #assert np.all(nn.theta_dim == theta_dim)

    theta = nn.gen_theta()
    assert theta.size == theta_size

    # Multiple hidden layers and multi-classification output layer
    input_size = 20
    hidden_sizes = [10, 10, 10]
    output_size = 3

    nn = NeuralNetwork(input_size, hidden_sizes, output_size)

    theta2_index = hidden_sizes[0] * (input_size + 1)
    theta3_index = theta2_index + hidden_sizes[1] * (hidden_sizes[0] + 1)
    theta4_index = theta3_index + hidden_sizes[2] * (hidden_sizes[1] + 1)
    end_index = theta4_index + output_size * (hidden_sizes[2] + 1)
    theta_dim = np.array([[0, theta2_index, hidden_sizes[0], (input_size + 1)],
                          [theta2_index, theta3_index, hidden_sizes[1], (hidden_sizes[0] + 1)],
                          [theta3_index, theta4_index, hidden_sizes[2], (hidden_sizes[1] + 1)],
                          [theta4_index, end_index, output_size, (hidden_sizes[2] + 1)]])

    theta_size = hidden_sizes[0] * (input_size + 1) + \
        hidden_sizes[1] * (hidden_sizes[0] + 1) + \
        hidden_sizes[2] * (hidden_sizes[1] + 1) + \
        output_size * (hidden_sizes[0] + 1)

    assert nn.theta_dim.shape == (4, 4)
    #assert np.all(nn.theta_dim == theta_dim)

    theta = nn.gen_theta()
    assert theta.size == theta_size

    # Check that lambda can be varied
    assert nn._lambda == 0
    nn.set_lambda(1)
    assert nn._lambda == 1


def test_cost():
    cost_noreg = 0.287629
    cost_reg = 0.576051
    # Create a neural network with known values of theta1 and theta2 and check
    # the cost computed
    data = np.load(cost_npzfile)
    theta1 = data['theta1']
    theta2 = data['theta2']

    assert theta1.shape == (25, 401)
    assert theta2.shape == (10, 26)

    nn = NeuralNetwork(400, 25, 10)
    theta = np.ascontiguousarray(np.hstack([theta1.T.flatten(), theta2.T.flatten()]))

    # Load the data for the neural network
    df_x = data['x']
    df_y = data['y']
    df_y = np.eye(10, dtype=np.uint8)[df_y]

    assert abs(nn.cost(df_x, df_y, theta) - cost_noreg) < epsilon

    nn.set_lambda(3)

    assert abs(nn.cost(df_x, df_y, theta) - cost_reg) < epsilon


def test_backprop():
    cost_reg = 0.576051
    # Create a neural network with known values of theta1 and theta2 and check
    # the cost computed
    data = np.load(cost_npzfile)
    theta1 = data['theta1']
    theta2 = data['theta2']

    nn = NeuralNetwork(400, 25, 10)
    theta = np.ascontiguousarray(np.hstack([theta1.T.flatten(), theta2.T.flatten()]))

    # Load the data for the neural network
    df_x = data['x']
    df_y = data['y']
    df_y = np.eye(10, dtype=np.uint8)[df_y]

    nn.set_lambda(3)

    assert abs(nn.cost(df_x, df_y, theta) - cost_reg) < epsilon

    alpha = 0.5
    _, grad = nn.backpropagate(df_x, df_y, theta)
    theta -= grad * alpha

    assert abs(nn.cost(df_x, df_y, theta) - cost_reg) > epsilon
    assert nn.cost(df_x, df_y, theta) < cost_reg
