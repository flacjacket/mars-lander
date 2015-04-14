from easy_train_nn import input_height, input_image, input_solution, this_dir, output_dir as input_dir

from py_utils.compare_output import compare_output
from py_utils.pgm import read_pgm

from pylearn2.utils import serial

import os
import subprocess

output_pgm = os.path.join(input_dir, "out_easy.pgm")
output_png = os.path.join(input_dir, "out_easy.png")

run_easy_executable = os.path.join(this_dir, "src", "run_easy")

if os.name == 'nt':
    run_easy_executable += ".exe"

subprocess.check_call([run_easy_executable, input_height, input_image, input_dir, "3", output_pgm])

print()
gen = read_pgm(output_pgm)
sol = read_pgm(input_solution)
compare_output(gen, sol, output_png)
