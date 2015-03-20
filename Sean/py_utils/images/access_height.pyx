import cython

import numpy as np
cimport numpy as np

DTYPE = np.float32


#@cython.boundscheck(False)
#@cython.wraparound(False)
def access_height(float [:, ::1]  df, long [:] col, long [:] row,
                  int x, int y, int size):
    cdef float [:] result = np.empty(size, dtype=DTYPE)

    cdef int i1, i2 = 0
    cdef int c, r
    for c, r in zip(col, row):
        i1, i2 = i2, i2+2*c+1
        result[i1:i2] = df[x+r, y-c:y+c+1]

    return result
