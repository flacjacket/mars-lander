from __future__ import print_function

from easy_1_preprocess import output_dir, n_features, pickle_x_train, pickle_y_train, pickle_x_test, pickle_y_test
from easy_2_train import nn_save_best

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

    # Softmax
    X = np.dot(X, w[2])
    X += b[2]
    X = X.argmax(axis=1)

    return X


def main():
    # Load and check the model
    print("Loading model")
    model = serial.load(nn_save_best)

    assert len(model.layers) == 3, "Assuming model has 3 layers (rect lin, rect lin, softmax)"

    assert model.layers[0].__class__.__name__ == "RectifiedLinear"
    assert model.layers[1].__class__.__name__ == "RectifiedLinear"
    assert model.layers[2].__class__.__name__ == "Softmax"

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
        bias.tofile(os.path.join(output_dir, 'b{}.raw'.format(i)))
        weights.tofile(os.path.join(output_dir, 'w{}.raw'.format(i)))

        b.append(bias)
        w.append(weights)

    print("Checking performance of net...")

    # Let's check that the values are reasonable, the training data should have good results
    X = serial.load(pickle_x_train)
    y = serial.load(pickle_y_train).flatten()

    X = run_net(X, y, w, b)

    print("Percent training data correct: {:.1%}".format(np.sum(X == y) / y.size))

    # Also, check the test data
    X = serial.load(pickle_x_test)
    y = serial.load(pickle_y_test).flatten()

    X = run_net(X, y, w, b)

    print("Percent test data correct: {:.1%}".format(np.sum(X == y) / y.size))


if __name__ == "__main__":
    main()
