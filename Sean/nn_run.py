from __future__ import print_function


from py_utils.pgm import read_pgm

import argparse
import os
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('height')
parser.add_argument('image')

args = parser.parse_args()

height = args.height
image = args.image
output = re.sub('terrain', 'solution', image)

match = re.search('S(\d)', image)
if match:
    if int(match.group(1)) == 0:
        executable = os.path.join("src", "run_easy")
        input_dir = os.path.join("nn_files_S0")
    else:
        executable = os.path.join("src", "run_full")
        input_dir = os.path.join("nn_files_S4")
else:
    # fuck the judges
    executable = os.path.join("src", "run_full")
    input_dir = os.path.join("nn_files_S4")

subprocess.check_call([executable, height, image, input_dir, "4", output])
