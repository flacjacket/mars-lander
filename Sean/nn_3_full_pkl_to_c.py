from __future__ import print_function

from nn_1_full_preprocess import output_dir, n_features, pickle_x_train, pickle_y_train, pickle_x_test, pickle_y_test, SCR
from nn_2_S0C0R10_train import nn_save_best

from pylearn2.utils import serial

import numpy as np
import os


def run_net(X, y, w, b):
    # RegularizedLinear 1
    X = np.dot(X, w[0])
    X += b[0]
    X[X < 0] = 0

    # RegularizedLinear 2
    X = np.dot(X, w[1])
    X += b[1]
    X[X < 0] = 0

    # RegularizedLinear 3
    X = np.dot(X, w[2])
    X += b[2]
    X[X < 0] = 0

    # Softmax
    X = np.dot(X, w[3])
    X += b[3]
    X = X.argmax(axis=1)

    return X


def main():
    for slope, crater, roughness in SCR:
        # Load and check the model
        print("Loading model")
        model = serial.load(nn_save_best.format(slope=slope, crater=crater, roughness=roughness))

        assert len(model.layers) == 4, "Assuming model has 3 layers (rect lin, rect lin, softmax)"

        assert model.layers[0].__class__.__name__ == "RectifiedLinear"
        assert model.layers[1].__class__.__name__ == "RectifiedLinear"
        assert model.layers[2].__class__.__name__ == "RectifiedLinear"
        assert model.layers[3].__class__.__name__ == "Softmax"

        # Extract the matrices out of the layers
        b = []
        w = []

        prev_size = n_features

        for i, layer in enumerate(model.layers):
            bias = layer.b.get_value().astype(np.float32)
            try:
                # RectifiedLinear uses trainsformer
                weights = layer.transformer.get_params()[0].get_value().astype(np.float32)
            except AttributeError:
                # Softmax uses W
                weights = layer.W.get_value().astype(np.float32)

            # Quick check dimensions
            assert weights.shape[0] == prev_size, "Weights have incorrect size, should be {}, got {}".format(prev_size, weights.shape[0])
            assert weights.shape[1] == bias.shape[0], "Bias has incorrect size, should be {}, got {}".format(weights.shape[1], bias.shape[0])
            prev_size = weights.shape[1]

            print("Saving layer {}, {} by {} elements".format(i, weights.shape[0], weights.shape[1]))
            bias.tofile(os.path.join(output_dir, 'b{}.raw'.format(i)).format(slope=slope, crater=crater, roughness=roughness))
            weights.tofile(os.path.join(output_dir, 'w{}.raw'.format(i)).format(slope=slope, crater=crater, roughness=roughness))

            b.append(bias)
            w.append(weights)

        print("Checking performance of net...")

        # Let's check that the values are reasonable, the training data should have good results
        X = serial.load(pickle_x_train.format(slope=slope, crater=crater, roughness=roughness))
        y = serial.load(pickle_y_train.format(slope=slope, crater=crater, roughness=roughness)).flatten()

        X = run_net(X, y, w, b)

        print("Percent training data correct: {:.1%}".format(np.sum(X == y) / y.size))

        tot = np.sum(y)
        cor = np.sum(X[y == 1])
        print("Safe:   {} / {} ({:.1%})".format(cor, tot, cor / tot))
        tot = len(y) - tot
        cor = tot - np.sum(X[y == 0])
        print("Unsafe: {} / {} ({:.1%})".format(cor, tot, cor / tot))

        # Also, check the test data
        X = serial.load(pickle_x_test.format(slope=slope, crater=crater, roughness=roughness))
        y = serial.load(pickle_y_test.format(slope=slope, crater=crater, roughness=roughness)).flatten()

        X = run_net(X, y, w, b)

        print("Percent test data correct: {:.1%}".format(np.sum(X == y) / y.size))

        tot = np.sum(y)
        cor = np.sum(X[y == 1])
        print("Safe:   {} / {} ({:.1%})".format(cor, tot, cor / tot))
        tot = len(y) - tot
        cor = tot - np.sum(X[y == 0])
        print("Unsafe: {} / {} ({:.1%})".format(cor, tot, cor / tot))


if __name__ == "__main__":
    main()
