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
root_dir = os.path.split(jpl_dir)[0]

sys.path.append(jpl_dir)

terrain_dir = os.path.join(root_dir, "training data", "terrainS0C0R10_100")
input_image = os.path.join(terrain_dir, "terrainS0C0R10_100.pgm")
input_height = os.path.join(terrain_dir, "terrainS0C0R10_100_dem.raw")
train_file = os.path.join(terrain_dir, "terrainS0C0R10_100.invHazard.pgm")

output_training = os.path.join(root_dir, "trained.pgm")
output_theta = os.path.join(root_dir, "theta.npz")

nrows = ncols = 1000
offset = 21

train_rows = 200
n_train = (ncols - 2 * offset) * train_rows  # (nrows - 2 * offset)
n_hidden = [100, 100]


def cost_grad(x, nn, df_x, df_y):
    return nn.backpropagate(df_x, df_y, x)


def main():
    from py_utils.images.pgm import read_pgm, write_pgm
    from py_utils.images.raw import read_raw
    from py_utils.images.image import Image
    from py_utils.ml.nn import NeuralNetwork

    height = read_raw(input_height, 500, 500)
    train_full = read_pgm(train_file)

    print("Read terrain files")

    image = Image(height, r=9)

    df_x = np.empty((n_train, image.size), dtype=np.float32)
    df_y = np.empty((n_train, 1), dtype=np.uint8)

    for n in range(n_train):
        row = offset + (n // (ncols - 2 * offset))
        col = offset + (n % (ncols - 2 * offset))
        df_x[n] = image[row // 2, col // 2]
        df_y[n] = train_full[row, col] == 0xff

    print("Constructed input array")

    nn = NeuralNetwork(image.size, n_hidden)
    nn.set_lambda(0.01)

    x = nn.gen_theta()
    args = (nn, df_x, df_y)

    x, f, d = opt.fmin_l_bfgs_b(cost_grad, x, args=args, maxiter=9, iprint=1)

    prediction = nn.predict(df_x, x)
    percent = np.mean(prediction == df_y) * 100

    print("Training accuracy: %.3f%%" % percent)

    prediction = prediction.reshape((-1, ncols - 2 * offset)) * 0xff
    output = np.zeros((nrows, ncols), dtype=np.uint8)
    output[offset:offset+train_rows, offset:ncols-offset] = prediction

    write_pgm(output, output_training)
    np.savez_compressed(output_theta, theta=x)

    for row in range(nrows - 2 * offset):
        for col in range(ncols - 2 * offset):
            df = image[row+offset, col+offset].reshape((1, -1))
            output[row+offset, col+offset] = nn.predict(df)[0, 0] * 0xff
            print(output[row+offset, col+offset])
    #write_pgm(output, test_output)

if __name__ == "__main__":
    main()
