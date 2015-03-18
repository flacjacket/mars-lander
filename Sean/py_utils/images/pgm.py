import re
import numpy as np


def read_pgm(filename, typestring=None):
    """Return image data from a raw PGM file as numpy array.

    Reads in the specified PGM file [1]_ and returns a NumPy array.

    Parameters
    ==========

        filename: string
            Name of file to be read in

        typestring: string
            Typestring used to set dtype of array, assumes big endian, unsigned
            ints, unless specified [2]_

    References
    ==========

    .. [1] http://netpbm.sourceforge.net/doc/pgm.html
    .. [2] http://docs.scipy.org/doc/numpy/reference/arrays.interface.html#__array_interface__

    """
    with open(filename, 'rb') as f:
        input_buf = f.read()

    try:
        header, width, height, maxval = re.search(
            b"(^P5\s+(?:#.*[\r\n])*"       # Match PGM magic number
            b"(\d+)\s+(?:#.*[\r\n])*"      # Match width
            b"(\d+)\s+(?:#.*[\r\n])*"      # Match height
            b"(\d+)\s+(?:#.*[\r\n]\s)*)",  # Match maxval
            input_buf
        ).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)

    width, height, maxval = map(int, [width, height, maxval])

    # Assume unsigned, big endian
    if typestring is None:
        if maxval < 256:
            typestring = 'u1'
        else:
            typestring = '>u2'

    return np.frombuffer(
        input_buf, dtype=typestring, count=width * height, offset=len(header)
    ).copy().reshape((height, width))


def write_pgm(df, filename):
    """Write the given data frame to a raw PGM file

    Writes out the input NymPy array to a PGM file [1]_.

    Parameters
    ==========

        df: numpy array
            Array to be converted to PGM

        filename: string
            Location of file to write out PGM file

    References
    ==========

    .. [1] http://netpbm.sourceforge.net/doc/pgm.html

    """
    width, height = df.shape
    if df.max() > 0xff:
        maxval = 0xffff
    else:
        maxval = 0xff

    # generate header
    header = [
        b'P5',
        ' '.join(map(str, df.shape)).encode(),
        str(maxval).encode()
    ]

    data = (bytes(row.data) for row in df)

    with open(filename, 'wb') as f:
        f.write(b'\n'.join(header))
        f.write(b'\n')

        for row in data:
            f.write(row)
