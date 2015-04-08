# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:20:23 2015

@author: seanvig2
"""

import numpy as np
import os

from pylearn2.datasets import dense_design_matrix
from py_utils.pgm import read_pgm
from py_utils.raw import read_raw

if os.name == 'nt':
    jpl_dir = os.path.normpath(r'C:\Users\seanvig2\Desktop\Source\mars-lander')
else:
    jpl_dir = os.path.normpath(os.path.expanduser('~/mars-lander'))

difficulty = {
    1: 'S0C0R10',
    2: 'S4C0R10',
    3: 'S4C4R10',
    4: 'S4C4R20'
}

train_dir = os.path.join(jpl_dir, 'training data')

data_height = {
    n: os.path.join(
        train_dir, 'terrain{0}_100', 'terrain{0}_100_500by500_dem.raw'
    ).format(d) for n, d in difficulty.items()
}

data_labels = {
    n: os.path.join(
        train_dir, 'terrain{0}_100', 'terrain{0}_100.invHazard.pgm'
    ).format(d) for n, d in difficulty.items()
}

R = 17
BUFFER = 20

INPUT_SHAPE = 1000
OUTPUT_SHAPE = INPUT_SHAPE - 2 * BUFFER

RAW_SHAPE = 500
RAW_BUFFER = BUFFER // 2

N_FEAT = 1 + 2 * ((R + 1) // 2)


class HeightDataset(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, which_set, hardness=1):

        print("Using dataset:", difficulty[hardness])

        h = read_raw(data_height[hardness], RAW_SHAPE, RAW_SHAPE)

        i = 0
        X = np.empty(((RAW_SHAPE-2*RAW_BUFFER)**2, N_FEAT**2), dtype=np.float32)
        for xi in range(RAW_BUFFER, RAW_SHAPE-RAW_BUFFER):
            for yi in range(RAW_BUFFER, RAW_SHAPE-RAW_BUFFER):
                X[i, :] = \
                    h[xi-(R+1)//2:xi+(R+3)//2, yi-(R+1)//2:yi+(R+3)//2].flatten()
                i += 1

        assert i == (RAW_SHAPE-2*RAW_BUFFER)**2

        X = np.hstack([X, X, X, X])
        X = X.reshape(OUTPUT_SHAPE**2, N_FEAT**2)

        y = read_pgm(data_labels[hardness])
        y = y[BUFFER:INPUT_SHAPE-BUFFER, BUFFER:INPUT_SHAPE-BUFFER].flatten()

        assert X.shape[0] == y.shape[0]

        n_train = int(OUTPUT_SHAPE**2 * 0.7)
        if which_set == 'train':
            print("Training data")
            X = X[:n_train, :]
            y = y[:n_train]
        elif which_set == 'valid':
            print("Validation data")
            X = X[n_train:, :]
            y = y[n_train:]
        else:
            raise ValueError("Invalid set:", which_set)

        super(HeightDataset, self).__init__(
            X=X,
            y=y.flatten().reshape(-1, 1)
        )
