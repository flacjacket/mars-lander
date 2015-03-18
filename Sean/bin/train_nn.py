"""
Train on the first image
"""

import os
import sys

import numpy as np
import scipy.optimize as opt

file_path = os.path.abspath(__file__)
file_dir = os.path.dirname(file_path)
jpl_dir = os.path.split(file_dir)[0]

sys.path.append(jpl_dir)

terrain_dir = os.path.join(jpl_dir, "training data", "terrainS0C0R10_100")
terrain_file = os.path.join(terrain_dir, "terrainS0C0R10_100.pgm")
train_file = os.path.join(terrain_dir, "terrainS0C0R10_100.invHazard.pgm")

training_output = os.path.join(jpl_dir, "trained.pgm")
test_output = os.path.join(jpl_dir, "test.pgm")

nrows = ncols = 1000
offset = 21

train_rows = 200
n_train = (ncols - 2 * offset) * train_rows  # (nrows - 2 * offset)
n_hidden = [100, 100]


def cost_grad(x, nn, df_x, df_y):
    nn.load_theta(x)
    return nn.backpropagate(df_x, df_y, unroll=True)


def main():
    from py_utils.images.pgm import read_pgm, write_pgm
    from py_utils.images.image import Image
    from py_utils.ml.nn import NeuralNetwork

    terrain = read_pgm(terrain_file)
    train_full = read_pgm(train_file)

    print("Read terrain files")

    image = Image(terrain)

    df_x = np.empty((n_train, image.size), dtype=np.uint8)
    df_y = np.empty((n_train, 1), dtype=np.uint8)

    for n in range(n_train):
        row = offset + (n // (ncols - 2 * offset))
        col = offset + (n % (ncols - 2 * offset))
        df_x[n] = image[row, col]
        df_y[n] = train_full[row, col] == 0xff

    print("Constructed input array")

    nn = NeuralNetwork(image.size, n_hidden)
    nn.set_lambda(0.1)

    x = nn.unroll_theta()
    args = (nn, df_x, df_y)

    x, f, d = opt.fmin_l_bfgs_b(cost_grad, x, args=args, maxiter=9, iprint=1)

    nn.load_theta(x)

    prediction = nn.predict(df_x)
    percent = np.mean(prediction == df_y) * 100

    print("Training accuracy: %.3f%%" % percent)

    prediction = prediction.reshape((-1, ncols - 2 * offset)) * 0xff
    output = np.zeros((nrows, ncols), dtype=np.uint8)
    output[offset:offset+train_rows, offset:ncols-offset] = prediction
    write_pgm(output, training_output)

    #for row in range(nrows - 2 * offset):
    #    for col in range(ncols - 2 * offset):
    #        df = image[row+offset, col+offset].reshape((1, -1))
    #        output[row+offset, col+offset] = nn.predict(df)[0, 0] * 0xff
    #write_pgm(output, test_output)

if __name__ == "__main__":
    main()
