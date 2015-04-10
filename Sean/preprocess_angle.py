from __future__ import print_function, division

from py_utils.preprocessing import check_angle

from py_utils.compare_output import compare_output
from py_utils.datasets import data_height, data_labels, jpl_root
from py_utils.pgm import read_pgm, write_pgm
from py_utils.raw import read_raw

import os
import time

import numpy as np


def main():
    output_dir = os.path.join(jpl_root, "preprocessing outputs")

    for i in range(1, 5):
        print("Training data set %d" % i)
        d = data_height[i]
        v = data_labels[i]
        df = read_raw(d, 500, 500)
        df_verify = read_pgm(v)
        df_out = np.zeros_like(df_verify)

        start = time.time()
        df_out[:] = check_angle(df)
        end = time.time()
        write_pgm(df_out, os.path.join(output_dir, 'out%d_angle.pgm' % i))
        compare_output(df_out, df_verify,
                       os.path.join(output_dir, 'out%d_angle.png' % i))
        print('Time: %.2f s' % (end - start))
        print()


if __name__ == '__main__':
    main()
