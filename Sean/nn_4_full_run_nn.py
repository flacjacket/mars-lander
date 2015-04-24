from __future__ import print_function

from nn_1_full_preprocess import input_height, input_image, input_solution, this_dir, SCR, output_dir as input_dir

from py_utils.compare_output import compare_output
from py_utils.pgm import read_pgm

import os
import subprocess

output_pgm = os.path.join(input_dir, "out.pgm")
output_png = os.path.join(input_dir, "out.png")

run_easy_executable = os.path.join(this_dir, "src", "run_easy")
run_full_executable = os.path.join(this_dir, "src", "run_full")

if os.name == 'nt':
    run_easy_executable += ".exe"
    run_full_executable += ".exe"


def main():
    for slope, crater, roughness in SCR[3:]:
        if slope == 0:
            subprocess.check_call([run_easy_executable,
                                   input_height.format(slope=slope, crater=crater, roughness=roughness),
                                   input_image.format(slope=slope, crater=crater, roughness=roughness),
                                   input_dir.format(slope=slope, crater=crater, roughness=roughness),
                                   "4",
                                   output_pgm.format(slope=slope, crater=crater, roughness=roughness)])
        else:
            subprocess.check_call([run_full_executable,
                                   input_height.format(slope=slope, crater=crater, roughness=roughness),
                                   input_image.format(slope=slope, crater=crater, roughness=roughness),
                                   input_dir.format(slope=slope, crater=crater, roughness=roughness),
                                   "4",
                                   output_pgm.format(slope=slope, crater=crater, roughness=roughness)])

        print()
        gen = read_pgm(output_pgm.format(slope=slope, crater=crater, roughness=roughness))
        sol = read_pgm(input_solution.format(slope=slope, crater=crater, roughness=roughness))
        compare_output(gen, sol, output_png.format(slope=slope, crater=crater, roughness=roughness))


if __name__ == "__main__":
    main()
