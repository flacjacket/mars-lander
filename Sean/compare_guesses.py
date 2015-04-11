#!/usr/bin python

import argparse

from py_utils.compare_output import compare_guesses
from py_utils.pgm import read_pgm

SAFE = 0xff
UNSAFE = 0x00


def main():
    parser = argparse.ArgumentParser(
        description='Compare PGM files for known safe/unsafe landing zones '
                    'with some unknown to known solution'
    )
    parser.add_argument(
        '-g', '--gen', metavar='FILE', help='Generated PGM file', required=True
    )
    parser.add_argument(
        '-c', '--cmp', metavar='FILE', help='Reference PGM file', required=True
    )
    parser.add_argument(
        '-f', '--fill', metavar='FILE', help='Filler PGM file', required=True
    )
    parser.add_argument('output', help='Output comparison png file')

    args = parser.parse_args()

    df_gen = read_pgm(args.gen)
    df_cmp = read_pgm(args.cmp)
    df_fill = read_pgm(args.fill)

    compare_guesses(df_gen, df_cmp, df_fill, args.output)


if __name__ == '__main__':
    main()
