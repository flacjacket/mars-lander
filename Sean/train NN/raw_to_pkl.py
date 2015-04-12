import numpy as np

from pylearn2.utils import serial

"""
Takes nn_out_safe.raw and nn_out_unsafe.raw files and outputs the pkl files for
NN training
"""

n_features = 35**2

df_safe = np.fromfile("nn_out_safe.raw", dtype=np.float32).reshape((-1, n_features))
df_unsafe = np.fromfile("nn_out_unsafe.raw", dtype=np.float32).reshape((-1, n_features))

n_safe = df_safe.shape[0]
n_unsafe = df_unsafe.shape[0]

print("Loaded {} safe and {} unsafe training data points".format(n_safe, n_unsafe))

n_ref = min(n_safe, n_unsafe)
n_train = int(n_ref * 0.75)
n_test = n_ref - n_train

print("Training on {} of each, testing on {} of each".format(n_train, n_test))

X_train = np.vstack([df_safe[:n_train, :],
                     df_unsafe[:n_train, :]])
X_test = np.vstack([df_safe[n_train:n_train+n_test, :],
                    df_unsafe[n_train:n_train+n_test, :]])

y_train = np.vstack([np.ones((n_train, 1), dtype=int),
                     np.zeros((n_train, 1), dtype=int)])
y_test = np.vstack([np.ones((n_test, 1), dtype=int),
                    np.zeros((n_test, 1), dtype=int)])

serial.save("X_train.pkl", X_train)
serial.save("X_test.pkl", X_test)
serial.save('y_train.pkl', y_train)
serial.save('y_test.pkl', y_test)
