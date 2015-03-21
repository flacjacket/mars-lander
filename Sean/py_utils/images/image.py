import numpy as np

from .access_image import access_image
from .access_height import access_height


class Image(object):
    def __init__(self, df, r=17):
        self._image = df

        row_ind = np.arange(r, -1, -1).astype(int)
        col_ind = np.ceil(np.sqrt(r**2 - (row_ind - 0.5)**2) - 0.5).astype(int)

        assert np.all((row_ind + 0.5)**2 + (col_ind + 0.5)**2 > r**2)
        assert np.all((row_ind - 0.5)**2 + (col_ind - 0.5)**2 < r**2)

        self.col_ind = np.append(col_ind, col_ind[-1:0:-1])
        self.row_ind = np.append(-row_ind, row_ind[-1:0:-1])

        self.size = 2 * np.sum(self.col_ind) + self.col_ind.size

    def __getitem__(self, key):
        x, y = key

        assert isinstance(x, int) and isinstance(y, int)

        if self._image.dtype == np.uint8:
            return access_image(self._image, self.col_ind, self.row_ind,
                                x, y, self.size)
        elif self._image.dtype == np.float32:
            return access_height(self._image, self.col_ind, self.row_ind,
                                 x, y, self.size)
