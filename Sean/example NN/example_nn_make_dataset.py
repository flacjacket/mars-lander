import numpy as np
from pylearn2.utils import serial

data = np.load('cost.npz')

sel = np.random.choice(5000, 5000, replace=False)
train = sel[:4000]
test = sel[4000:]

serial.save('X_train.pkl', data['x'][train, :])
serial.save('X_test.pkl', data['x'][test, :])
serial.save('y_train.pkl', data['y'][train].reshape((-1, 1)))
serial.save('y_test.pkl', data['y'][test].reshape((-1, 1)))
