import numpy as np
import os
import sys

from pylearn2.utils import serial

this_dir = os.path.split(os.path.abspath(__file__))[0]
jpl_dir = os.path.split(this_dir)[0]

sys.path.append(jpl_dir)

from py_utils.datasets import INPUT_SHAPE, BUFFER, R, data_image, data_labels
from py_utils.pgm import read_pgm


n_samples = (INPUT_SHAPE - 2 * BUFFER)**2
n_train = (INPUT_SHAPE - 2 * BUFFER) * 400
n_test = (INPUT_SHAPE - 2 * BUFFER) * 200

n_features = (2 * R + 1)**2

X = np.zeros((n_train, n_features), dtype=np.float16)
X_test = np.zeros((n_test, n_features), dtype=np.float16)

y = np.zeros((n_train, 1), dtype=int)
y_test = np.zeros((n_test, 1), dtype=int)

data = read_pgm(data_image[1])
labels = read_pgm(data_labels[1])

norm_data = np.zeros_like(data, dtype=np.float16)
norm_data[:] = data / 255


print("Training on {} samples ({:.1%})".format(n_train, n_train / n_samples))
print("Testing on {} samples ({:.1%})".format(n_test, n_test / n_samples))


def get_data(n):
    i = n // (INPUT_SHAPE - 2 * BUFFER)
    j = n % (INPUT_SHAPE - 2 * BUFFER)
    return norm_data[BUFFER+i-R:BUFFER+i+R+1, BUFFER+j-R:BUFFER+j+R+1]


def get_label(n):
    i = n // (INPUT_SHAPE - 2 * BUFFER)
    j = n % (INPUT_SHAPE - 2 * BUFFER)
    return labels[BUFFER+i, BUFFER+j] == 0xff


for i in range(n_train):
    X[i, :] = get_data(i).flatten()
    y[i, 0] = get_label(i)

for i in range(n_test):
    X_test[i, :] = get_data(i + n_train).flatten()
    y_test[i, 0] = get_label(i + n_train)

print("With {:.1%} safe training examples and {:.1%} safe test examples".format(np.sum(y) / n_train, np.sum(y_test) / n_test))


serial.save("X_train.pkl", X)
serial.save("X_test.pkl", X_test)
serial.save('y_train.pkl', y)
serial.save('y_test.pkl', y_test)
