from train_nn_easy import nn_save_best, output_dir
from train_nn_easy import pickle_x_train, pickle_y_train
from train_nn_easy import pickle_x_test, pickle_y_test

from pylearn2.utils import serial

import numpy as np
import os

# Load and check the model
print("Loading model")
model = serial.load(nn_save_best)

assert len(model.layers) == 3, "Assuming model has 3 layers (rect lin, rect lin, softmax)"

assert model.layers[0].__class__.__name__ == "RectifiedLinear"
assert model.layers[1].__class__.__name__ == "RectifiedLinear"
assert model.layers[2].__class__.__name__ == "Softmax"

# Extract the matrices out of the layers
bias_list = []
weights_list = []
for i, layer in enumerate(model.layers):
    bias = layer.b.get_value().astype(np.double)
    try:
        # RectifiedLinear uses trainsformer
        weights = layer.transformer.get_params()[0].get_value().astype(np.double)
    except AttributeError:
        # Softmax uses W
        weights = layer.W.get_value().astype(np.double)

    print("Saving layer {}".format(i))
    bias.tofile(os.path.join(output_dir, 'b{}.raw'.format(i)))
    weights.tofile(os.path.join(output_dir, 'w{}.raw'.format(i)))

    bias_list.append(bias)
    weights_list.append(weights)

# Let's check that the values are reasonable, the training data should have good results
X = serial.load(pickle_x_train)
y = serial.load(pickle_y_train).flatten()

for b, w in zip(bias_list, weights_list):
    X = np.dot(X, w)
    X += b
    X[X < 0] = 0

X = X.argmax(axis=1)

print("Percent test data correct: {:.1%}".format(np.sum(X == y) / y.size))

# Also, check the test data
X = serial.load(pickle_x_test)
y = serial.load(pickle_y_test).flatten()

for b, w in zip(bias_list, weights_list):
    X = np.dot(X, w)
    X += b
    X[X < 0] = 0

X = X.argmax(axis=1)

print("Percent test data correct: {:.1%}".format(np.sum(X == y) / y.size))
