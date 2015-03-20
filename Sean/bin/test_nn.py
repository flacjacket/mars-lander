import os
import sys

import numpy as np

file_path = os.path.abspath(__file__)
file_dir = os.path.dirname(file_path)
jpl_dir = os.path.split(file_dir)[0]
root_dir = os.path.split(jpl_dir)[0]

sys.path.append(jpl_dir)

terrain_dir = os.path.join(root_dir, "training data", "terrainS0C0R10_100")
input_image = os.path.join(terrain_dir, "terrainS0C0R10_100.pgm")
input_height = os.path.join(terrain_dir, "terrainS0C0R10_100_dem.raw")
train_file = os.path.join(terrain_dir, "terrainS0C0R10_100.invHazard.pgm")

output_theta = os.path.join(root_dir, "theta.npz")
output_training = os.path.join(root_dir, "trained.pgm")
output_test = os.path.join(root_dir, "test.pgm")

nrows = ncols = 1000
offset = 21
n_train = (ncols - 2 * offset) * (nrows - 2 * offset)

n_hidden = [100, 100]


def main():
    from py_utils.images.pgm import write_pgm
    from py_utils.images.raw import read_raw
    from py_utils.images.image import Image
    from py_utils.ml.nn import NeuralNetwork

    height = read_raw(input_height, 500, 500)
    image = Image(height, r=9)

    nn = NeuralNetwork(image.size, n_hidden)
    theta = np.load(output_theta)['theta']

    df = np.empty((n_train, image.size), dtype=np.float32)
    output = np.zeros((nrows, ncols), dtype=np.uint8)

    for n in range(n_train):
        row = offset + (n // (ncols - 2 * offset))
        col = offset + (n % (ncols - 2 * offset))
        df[n] = image[row // 2, col // 2]

    output[offset:nrows-offset, offset:ncols-offset] = \
            nn.predict(df, theta).reshape((nrows - 2 * offset, ncols - 2 * offset)) * 0xff

    write_pgm(output, output_test)


if __name__ == '__main__':
    main()
