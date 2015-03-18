from py_utils.images.access_image import access_image

import numpy as np

# Construct a 10x10 matrix
df = np.arange(100, dtype=np.uint8).reshape((10, 10))


def test_access_image():
    x = 2
    y = 3
    col = np.array([0, 1, 2])
    row = np.array([-1, 0, 1])
    output = np.array([13, 22, 23, 24, 31, 32, 33, 34, 35], dtype=np.uint8)

    assert df[2, 3] == 23

    assert np.all(access_image(df, col, row, x, y, output.size) == output)
