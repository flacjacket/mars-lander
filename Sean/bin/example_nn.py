"""
Test script that will train a neural network using example data
"""

import os
import sys

import numpy as np
import scipy.optimize as opt

file_path = os.path.abspath(__file__)
file_dir = os.path.dirname(file_path)
jpl_dir = os.path.split(file_dir)[0]

sys.path.append(jpl_dir)

cost_npzfile = os.path.join(jpl_dir, "py_utils", "ml", "tests", "cost.npz")


def cost_grad(x, nn, df_x, df_y):
    return nn.backpropagate(df_x, df_y, x)


def main():
    from py_utils.ml.nn import NeuralNetwork

    data = np.load(cost_npzfile)

    nn = NeuralNetwork(400, 25, 10)
    nn.set_lambda(1)

    # Load the example inputs and outputs
    df_x = data['x']
    df_y = data['y']
    df_ymat = np.eye(10)[df_y]

    x0 = nn.gen_theta()

    j = nn.cost(df_x, df_ymat, x0)
    print("Initial cost:              %.5e" % j)

    args = (nn, df_x, df_ymat)
    x, f, d = opt.fmin_l_bfgs_b(cost_grad, x0, args=args, maxiter=100)

    print("Cost after %d iterations: %.5e" % (d['nit'], f))

    prediction = nn.predict(df_x, x)
    percent = np.mean(prediction == df_y) * 100

    print("Training accuracy: %.3f%%" % percent)

if __name__ == "__main__":
    main()
