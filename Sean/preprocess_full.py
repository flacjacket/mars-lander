from __future__ import division, print_function

from py_utils.preprocessing import check_full

from py_utils.compare_output import compare_output
from py_utils.datasets import data_height, data_labels, jpl_dir
from py_utils.pgm import read_pgm, write_pgm
from py_utils.raw import read_raw

import os
import time

import numpy as np


def main():

    output_dir = os.path.join(jpl_dir, "preprocessing outputs")

    for i in range(1, 5):
        print("Training data set %d" % i)
        d = data_height[i]
        v = data_labels[i]
        df = read_raw(d, 500, 500)
        df_verify = read_pgm(v)
        df_out = np.zeros_like(df_verify)

        start = time.time()
        df_out[:] = check_full(df)
        end = time.time()
        write_pgm(df_out, os.path.join(output_dir, 'out%d_full.pgm' % i))
        compare_output(df_out, df_verify,
                       os.path.join(output_dir, 'out%d_full.png' % i))
        print('Time: %.2f s' % (end - start))
        print()


if __name__ == '__main__':
    main()
