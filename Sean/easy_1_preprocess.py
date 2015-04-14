from __future__ import print_function

from py_utils.pgm import read_pgm
from py_utils.compare_output import compare_guesses

import numpy as np
import os
import subprocess

from pylearn2.config import yaml_parse
from pylearn2.utils import serial

this_dir = os.path.split(os.path.abspath(__file__))[0]

training_dir = os.path.abspath(os.path.join(this_dir, "..", "training data", "terrainS0C0R10_100"))
input_height = os.path.join(training_dir, "terrainS0C0R10_100_500by500_dem.raw")
input_image = os.path.join(training_dir, "terrainS0C0R10_100.pgm")
input_solution = os.path.join(training_dir, "terrainS0C0R10_100.invHazard.pgm")

for check_file in [input_height, input_image, input_solution]:
    if not os.path.exists(check_file):
        raise ValueError("File does not exist:", check_file)

output_dir = os.path.join(this_dir, "nn_files_easy")
output_pgm = os.path.join(output_dir, "out_easy_preprocessing.pgm")
output_png = os.path.join(output_dir, "out_easy_preprocessing.png")
output_nn = os.path.join(output_dir, "nn_out")
output_nn_safe = output_nn + "_safe.raw"
output_nn_unsafe = output_nn + "_unsafe.raw"

pickle_x_train = os.path.join(output_dir, "X_train.pkl")
pickle_x_test = os.path.join(output_dir, "X_test.pkl")
pickle_y_train = os.path.join(output_dir, "y_train.pkl")
pickle_y_test = os.path.join(output_dir, "y_test.pkl")

n_features = 35**2

gen_easy_executable = os.path.abspath(os.path.join(this_dir, "src", "gen_easy"))

if os.name == 'nt':
    gen_easy_executable += ".exe"

def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ##########################################################################
    # Generate NN input
    ##########################################################################

    subprocess.check_call([gen_easy_executable, input_height, input_image,
                           input_solution, output_pgm, output_nn])

    print()
    gen = read_pgm(output_pgm)
    sol = read_pgm(input_solution)
    fill = read_pgm(input_image)
    compare_guesses(gen, sol, fill, output_png)
    print()

    ##########################################################################
    # Pickle NN input
    ##########################################################################

    df_safe = np.fromfile(output_nn_safe, dtype=np.float32).reshape((-1, n_features))
    df_unsafe = np.fromfile(output_nn_unsafe, dtype=np.float32).reshape((-1, n_features))

    n_safe = df_safe.shape[0]
    n_unsafe = df_unsafe.shape[0]

    print("Loaded {} safe and {} unsafe training data points".format(n_safe, n_unsafe))

    n_train_safe = int(n_safe * 0.9)
    n_train_unsafe = int(n_unsafe * 0.9)

    n_test_safe = n_safe - n_train_safe
    n_test_unsafe = n_unsafe - n_train_unsafe

    print("Training on {} safe and {} unsafe".format(n_train_safe, n_train_unsafe))
    print("Testing on {} safe and {} unsafe".format(n_test_safe, n_test_unsafe))

    print("Saving data... ", end="")
    X_train = np.vstack([df_safe[:n_train_safe, :],
                        df_unsafe[:n_train_unsafe, :]])
    X_test = np.vstack([df_safe[n_train_safe:, :],
                        df_unsafe[n_train_unsafe:, :]])

    y_train = np.vstack([np.ones((n_train_safe, 1), dtype=int),
                         np.zeros((n_train_unsafe, 1), dtype=int)])
    y_test = np.vstack([np.ones((n_test_safe, 1), dtype=int),
                        np.zeros((n_test_unsafe, 1), dtype=int)])

    serial.save(pickle_x_train, X_train)
    serial.save(pickle_x_test, X_test)
    serial.save(pickle_y_train, y_train)
    serial.save(pickle_y_test, y_test)
    print("done")


if __name__ == "__main__":
    main()
