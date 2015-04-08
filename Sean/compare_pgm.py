#!/usr/bin python

import argparse

from py_utils.compare_output import compare_output
from py_utils.pgm import read_pgm

SAFE = 0xff
UNSAFE = 0x00


def main():
    parser = argparse.ArgumentParser(
        description='Compare PGM files for generated safe landing zones to '
                    'known safe landing zones'
    )
    parser.add_argument(
        '-g', '--gen', metavar='FILE', help='Generated PGM file', required=True
    )
    parser.add_argument(
        '-c', '--cmp', metavar='FILE', help='Reference PGM file', required=True
    )
    parser.add_argument('output', help='Output comparison png file')

    args = parser.parse_args()

    df_gen = read_pgm(args.gen)
    df_cmp = read_pgm(args.cmp)

    compare_output(df_gen, df_cmp, args.output)


if __name__ == '__main__':
    main()
