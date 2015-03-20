import numpy as np
from .sigmoid import sigmoid


class Layer(object):
    def __init__(self, from_size, to_size):
        self.shape = (to_size, from_size + 1)
        self.shapeT = (from_size + 1, to_size)
        self.a = None
        self.z = None

    def gen_theta(self, epsilon_init=0.12, store=False):
        theta = np.random.rand(*self.shape)
        if store:
            self.theta = theta
        return theta

    def feedfwd(self, df_input, thetaT, store=False):
        assert thetaT.shape == self.shapeT
        z = np.dot(df_input, thetaT[1:]) + thetaT[0]

        a = sigmoid(z)
        if store:
            # self.a = a
            self.z = z

        return a

    def regularization(self, thetaT):
        assert thetaT.shape == self.shapeT
        return np.sum(thetaT[1:]**2)
