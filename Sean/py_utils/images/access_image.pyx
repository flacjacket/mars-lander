import cython

import numpy as np
cimport numpy as np

DTYPEu = np.uint8


@cython.boundscheck(False)
@cython.wraparound(False)
def access_image(unsigned char [:, ::1]  df, long [:] col, long [:] row,
                 int x, int y, int size):
    cdef unsigned char [:] result = np.empty(size, dtype=DTYPEu)

    cdef int ind = 0
    cdef int c, r
    for c, r in zip(col, row):
        result[ind:ind+2*c+1] = df[x+r, y-c:y+c+1]
        ind += 2*c + 1

    return result
