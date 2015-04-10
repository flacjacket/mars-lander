from __future__ import print_function, division

import numpy as np
import os

from py_utils.pgm import read_pgm
from py_utils.raw import read_raw

this_dir = os.path.split(os.path.abspath(__file__))[0]
jpl_root = os.path.split(os.path.split(this_dir)[0])[0]

difficulty = {
    1: 'S0C0R10',
    2: 'S4C0R10',
    3: 'S4C4R10',
    4: 'S4C4R20'
}

train_dir = os.path.join(jpl_root, 'training data')

data_height = {
    n: os.path.join(
        train_dir, 'terrain{0}_100', 'terrain{0}_100_500by500_dem.raw'
    ).format(d) for n, d in difficulty.items()
}

data_image = {
    n: os.path.join(
        train_dir, 'terrain{0}_100', 'terrain{0}_100.pgm'
    ).format(d) for n, d in difficulty.items()
}

data_labels = {
    n: os.path.join(
        train_dir, 'terrain{0}_100', 'terrain{0}_100.invHazard.pgm'
    ).format(d) for n, d in difficulty.items()
}

R = 17
BUFFER = 21

INPUT_SHAPE = 1000
OUTPUT_SHAPE = INPUT_SHAPE - 2 * BUFFER

RAW_SHAPE = 500
RAW_BUFFER = BUFFER // 2

N_FEAT = 1 + 2 * ((R + 1) // 2)
