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
    theta_dim = [(0, theta2_index), (theta2_index, end_index)]

    assert nn.theta_dim == theta_dim

    assert len(nn.layers) == 2
    assert nn.layers[0].shape == (hidden_size, input_size + 1)
    assert nn.layers[1].shape == (1, hidden_size + 1)

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
    theta_dim = [(0, theta2_index),
                 (theta2_index, theta3_index),
                 (theta3_index, theta4_index),
                 (theta4_index, end_index)]

    theta_size = hidden_sizes[0] * (input_size + 1) + \
        hidden_sizes[1] * (hidden_sizes[0] + 1) + \
        hidden_sizes[2] * (hidden_sizes[1] + 1) + \
        (hidden_sizes[0] + 1)

    assert nn.theta_dim == theta_dim

    assert len(nn.layers) == 4
    assert nn.layers[0].shape == (hidden_sizes[0], input_size + 1)
    assert nn.layers[1].shape == (hidden_sizes[1], hidden_sizes[0] + 1)
    assert nn.layers[2].shape == (hidden_sizes[2], hidden_sizes[1] + 1)
    assert nn.layers[3].shape == (1, hidden_sizes[2] + 1)

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
    theta_dim = [(0, theta2_index),
                 (theta2_index, theta3_index),
                 (theta3_index, theta4_index),
                 (theta4_index, end_index)]

    theta_size = hidden_sizes[0] * (input_size + 1) + \
        hidden_sizes[1] * (hidden_sizes[0] + 1) + \
        hidden_sizes[2] * (hidden_sizes[1] + 1) + \
        output_size * (hidden_sizes[0] + 1)

    assert nn.theta_dim == theta_dim

    assert len(nn.layers) == 4
    assert nn.layers[0].shape == (hidden_sizes[0], input_size + 1)
    assert nn.layers[1].shape == (hidden_sizes[1], hidden_sizes[0] + 1)
    assert nn.layers[2].shape == (hidden_sizes[2], hidden_sizes[1] + 1)
    assert nn.layers[3].shape == (output_size, hidden_sizes[2] + 1)

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
    thetaT = np.hstack([theta1.T.flatten(), theta2.T.flatten()])

    assert theta1.shape == (25, 401)
    assert theta2.shape == (10, 26)

    # Load the data for the neural network
    df_x = data['x']
    df_y = data['y']
    df_y = np.eye(10, dtype=np.uint8)[df_y]

    # Test with theta pased in
    nn = NeuralNetwork(400, 25, 10)

    cost = nn.cost(df_x, df_y, thetaT)
    assert abs(cost - cost_noreg) < epsilon
    nn.set_lambda(3)
    cost = nn.cost(df_x, df_y, thetaT)
    assert abs(cost - cost_reg) < epsilon


def test_backprop():
    cost_reg = 0.576051
    # Create a neural network with known values of theta1 and theta2 and check
    # the cost computed
    data = np.load(cost_npzfile)
    theta1 = data['theta1']
    theta2 = data['theta2']
    theta = np.hstack([theta1.T.flatten(), theta2.T.flatten()])

    # Load the data for the neural network
    df_x = data['x']
    df_y = data['y']
    df_y = np.eye(10, dtype=np.uint8)[df_y]

    # Test with theta pased in
    nn1 = NeuralNetwork(400, 25, 10)
    nn1.set_lambda(3)

    assert abs(nn1.cost(df_x, df_y, theta) - cost_reg) < epsilon

    alpha = 0.5
    _, grad = nn1.backpropagate(df_x, df_y, theta)

    e = 1e-4
    N = 50
    for i in range(len(theta) // N):
        d = np.zeros_like(theta)
        d[i * N] = e
        cost1 = nn1.cost(df_x, df_y, theta - d)
        cost2 = nn1.cost(df_x, df_y, theta + d)
        assert abs(grad[i * N] - (cost2 - cost1) / (2 * e)) < 1e-10
    theta -= grad * alpha

    assert abs(nn1.cost(df_x, df_y, theta) - cost_reg) > epsilon
    assert nn1.cost(df_x, df_y, theta) < cost_reg

#    # Test with theta loaded normally
#    nn2 = NeuralNetwork(400, 25, 10)
#    nn2.load_theta(theta1, theta2)
#    nn2.set_lambda(3)
#
#    alpha = 0.5
#    _, grad = nn2.backpropagate(df_x, df_y)
#    theta -= grad * alpha
#
#    assert abs(nn2.cost(df_x, df_y, theta) - cost_reg) > epsilon
#    assert nn2.cost(df_x, df_y, theta) < cost_reg
#
#    # Test with theta loaded flat
#    nn3 = NeuralNetwork(400, 25, 10)
#    nn3.load_theta_flat(theta)
#    nn3.set_lambda(3)
#
#    alpha = 0.5
#    _, grad = nn3.backpropagate(df_x, df_y)
#    theta -= grad * alpha
#
#    assert abs(nn3.cost(df_x, df_y) - cost_reg) > epsilon
#    assert nn3.cost(df_x, df_y) < cost_reg
