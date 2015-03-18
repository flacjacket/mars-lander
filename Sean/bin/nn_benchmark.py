#!/usr/bin python

from __future__ import print_function

import os
import timeit

file_path = os.path.abspath(__file__)
file_dir = os.path.dirname(file_path)
jpl_dir = os.path.split(file_dir)[0]

terrain_dir = os.path.join(jpl_dir, "training data", "terrainS0C0R10_100")
terrain_file = os.path.join(terrain_dir, "terrainS0C0R10_100.pgm")
solution_file = os.path.join(terrain_dir, "terrainS0C0R10_100.invHazard.pgm")

cost_npz = os.path.join(jpl_dir, "py_utils", "ml", "tests", "cost.npz")


###############################################################################
# Image creation and access
###############################################################################


setup_image = """\
import sys
sys.path.append(r'%s')
from py_utils.images.pgm import read_pgm
from py_utils.images.image import Image
image = Image(read_pgm(r'%s'))
""" % (jpl_dir, terrain_file)


# Benchmark access of a single point in an Image
def bench_image():
    print("Accessing point in image... ",  end='')

    stmt = "image[200, 300]"

    n = 10000
    time = timeit.timeit(stmt=stmt, setup=setup_image, number=n)

    print("%.5e s" % (time / n))


###############################################################################
# Pre-built Image-based neural network
###############################################################################


setup_nnbuild = """\
import numpy as np
import sys
sys.path.append(r'%s')
from py_utils.images.pgm import read_pgm
from py_utils.images.image import Image
input_image = Image(read_pgm(r'%s'))
output_image = read_pgm(r'%s')
df_input = np.empty([958, input_image.size], dtype=np.uint8)
df_output = np.empty([958, 1], dtype=np.uint8)""" % (jpl_dir, terrain_file, solution_file)

setup_nnbuild_ffwd = setup_nnbuild + """
from py_utils.ml.nn import NeuralNetwork
for n in range(958):
    df_input[n, :] = input_image[21, 21+n]
    df_output[n, 0] = output_image[21, 21+n]
nn = NeuralNetwork(input_image.size, 20)
x0 = nn.gen_theta()
"""


# Bench mark construcing the inputs to a NN from an Image
def bench_nnbuild():
    print("Building neural network input (1 row)...", end='')

    stmt = """\
for n in range(958):
    df_input[n, :] = input_image[21, 21+n]
    df_output[n, 0] = output_image[21, 21+n]"""

    n = 10
    time = timeit.timeit(stmt=stmt, setup=setup_nnbuild, number=n)

    print("%.5e s" % (time / n))


def bench_nnbuild_ffwd():
    print("Neural network feed forward (1 row)... ",  end='')

    stmt = "nn.cost(df_input, df_output, x0)"

    n = 100
    time = timeit.timeit(stmt=stmt, setup=setup_nnbuild_ffwd, number=n)

    print("%.5e s" % (time / n))


###############################################################################
# Example neural network
###############################################################################


setup_nnexample = """\
import numpy as np
import scipy.optimize as opt
import sys
sys.path.append(r'%s')
from py_utils.ml.nn import NeuralNetwork
cost_grad = lambda x, nn, df_x, df_y: nn.backpropagate(df_x, df_y, x)

nn = NeuralNetwork(400, 25, 10)
nn.set_lambda(1)

# Load the example inputs and outputs
data = np.load(r'%s')
df_x = data['x']
df_y = data['y']
df_y = np.eye(10, dtype=np.uint8)[df_y]

x0 = nn.gen_theta()
args = (nn, df_x, df_y)""" % (jpl_dir, cost_npz)


def bench_nn_ffwd():
    print("Exmple neural network cost... ",  end='')

    stmt = "nn.cost(df_x, df_y, x0)"
    n = 100
    time = timeit.timeit(stmt=stmt, setup=setup_nnexample, number=n)

    print("%.5e s" % (time / n))


def bench_nn_grad():
    print("Exmple neural network gradient... ",  end='')

    stmt = "nn.backpropagate(df_x, df_y, x0)"
    n = 10
    time = timeit.timeit(stmt=stmt, setup=setup_nnexample, number=n)

    print("%.5e s" % (time / n))


def bench_nn_learn():
    print("Exmple neural network leaning (100 iterations)... ",  end='')

    stmt = "opt.fmin_l_bfgs_b(cost_grad, x0, args=args, maxiter=100)"
    n = 1
    time = timeit.timeit(stmt=stmt, setup=setup_nnexample, number=n)

    print("%.5e s" % (time / n))


###############################################################################


def main():
    bench_image()

    bench_nn_ffwd()
    bench_nn_grad()
    bench_nn_learn()

    bench_nnbuild()
    bench_nnbuild_ffwd()

if __name__ == '__main__':
    main()
