import re
import numpy as np


def read_raw(filename, height, width):
    with open(filename, 'rb') as f:
        input_buf = f.read()

    return np.frombuffer(
        input_buf, dtype=np.float32, count=height * width
    ).copy().reshape((height, width))
