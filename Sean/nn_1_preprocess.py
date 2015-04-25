from __future__ import print_function

from py_utils.pgm import read_pgm
from py_utils.compare_output import compare_guesses

import numpy as np
import os
import subprocess

from pylearn2.utils import serial

this_dir = os.path.split(os.path.abspath(__file__))[0]

training_dir = os.path.abspath(os.path.join(this_dir, "..", "training data", "terrainS{slope}C{crater}R{roughness}_100"))
input_height = os.path.join(training_dir, "terrainS{slope}C{crater}R{roughness}_100_500by500_dem.raw")
input_image = os.path.join(training_dir, "terrainS{slope}C{crater}R{roughness}_100.pgm")
input_solution = os.path.join(training_dir, "terrainS{slope}C{crater}R{roughness}_100.invHazard.pgm")

#for check_file in [input_height, input_image, input_solution]:
#    if not os.path.exists(check_file):
#        raise ValueError("File does not exist:", check_file)

output_dir = os.path.join(this_dir, "nn_files_S{slope}")
output_pgm = os.path.join(output_dir, "out_preprocessing_S{slope}C{crater}R{roughness}.pgm")
output_png = os.path.join(output_dir, "out_preprocessing_S{slope}C{crater}R{roughness}.png")
output_nn = os.path.join(output_dir, "nn_outS{slope}C{crater}R{roughness}")
output_nn_safe = output_nn + "_safe.raw"
output_nn_unsafe = output_nn + "_unsafe.raw"

pickle_x_train = os.path.join(output_dir, "X_train.pkl")
pickle_x_test = os.path.join(output_dir, "X_test.pkl")
pickle_y_train = os.path.join(output_dir, "y_train.pkl")
pickle_y_test = os.path.join(output_dir, "y_test.pkl")

n_features = 35**2

gen_easy_executable = os.path.abspath(os.path.join(this_dir, "src", "gen_easy"))
gen_full_executable = os.path.abspath(os.path.join(this_dir, "src", "gen_full"))

if os.name == 'nt':
    gen_easy_executable += ".exe"
    gen_full_executable += ".exe"

SCR = [[0, 0, 10],
       [4, 0, 10],
       [4, 4, 10],
       [4, 4, 20]]

def main():
    for slope, crater, roughness in SCR:
        out = output_dir.format(slope=slope, crater=crater, roughness=roughness)
        if not os.path.exists(out):
            os.makedirs(out)

        ######################################################################
        # Generate NN input
        ######################################################################

        if slope == 0:
            subprocess.check_call([gen_easy_executable,
                                   input_height.format(slope=slope, crater=crater, roughness=roughness),
                                   input_image.format(slope=slope, crater=crater, roughness=roughness),
                                   input_solution.format(slope=slope, crater=crater, roughness=roughness),
                                   output_pgm.format(slope=slope, crater=crater, roughness=roughness),
                                   output_nn.format(slope=slope, crater=crater, roughness=roughness)])
        else:
            subprocess.check_call([gen_full_executable,
                                   input_height.format(slope=slope, crater=crater, roughness=roughness),
                                   input_image.format(slope=slope, crater=crater, roughness=roughness),
                                   input_solution.format(slope=slope, crater=crater, roughness=roughness),
                                   output_pgm.format(slope=slope, crater=crater, roughness=roughness),
                                   output_nn.format(slope=slope, crater=crater, roughness=roughness)])

        print()
        gen = read_pgm(output_pgm.format(slope=slope, crater=crater, roughness=roughness))
        sol = read_pgm(input_solution.format(slope=slope, crater=crater, roughness=roughness))
        fill = read_pgm(input_image.format(slope=slope, crater=crater, roughness=roughness))
        compare_guesses(gen, sol, fill, output_png.format(slope=slope, crater=crater, roughness=roughness))
        print()

    ######################################################################
    # Pickle NN input
    ######################################################################

    scr = SCR[0]
    s, c, r = scr
    df_safe = np.fromfile(output_nn_safe.format(slope=s, crater=c, roughness=r), dtype=np.float32).reshape((-1, n_features))
    df_unsafe = np.fromfile(output_nn_unsafe.format(slope=s, crater=c, roughness=r), dtype=np.float32).reshape((-1, n_features))

    n_safe = df_safe.shape[0]
    n_unsafe = df_unsafe.shape[0]

    print("Loaded {} safe and {} unsafe training data points".format(n_safe, n_unsafe))

    print("Saving data... ", end="")
    X_train = np.vstack([
        df_safe[:int(0.9 * n_safe), :],
        df_unsafe[:int(0.9 * n_unsafe), :]
    ])
    X_test = np.vstack([
        df_safe[int(0.9 * n_safe):, :],
        df_unsafe[int(0.9 * n_unsafe):, :]
    ])

    y_train = np.vstack([
        np.ones((int(0.9 * n_safe), 1), dtype=int),
        np.zeros((int(0.9 * n_unsafe), 1), dtype=int),
    ])
    y_test = np.vstack([
        np.ones((n_safe - int(0.9 * n_safe), 1), dtype=int),
        np.zeros((n_unsafe - int(0.9 * n_unsafe), 1), dtype=int),
    ])

    serial.save(pickle_x_train.format(slope=slope), X_train)
    serial.save(pickle_x_test.format(slope=slope), X_test)
    serial.save(pickle_y_train.format(slope=slope), y_train)
    serial.save(pickle_y_test.format(slope=slope), y_test)
    print("done")

    ######################################################################
    # Pickle NN input
    ######################################################################

    scr = SCR[1:]
    slope = scr[0][0]
    df_safe = [np.fromfile(output_nn_safe.format(slope=s, crater=c, roughness=r), dtype=np.float32).reshape((-1, n_features)) for s, c, r in scr]
    df_unsafe = [np.fromfile(output_nn_unsafe.format(slope=s, crater=c, roughness=r), dtype=np.float32).reshape((-1, n_features)) for s, c, r in scr]

    n_safe = [df.shape[0] for df in df_safe]
    n_unsafe = [df.shape[0] for df in df_unsafe]

    print("Loaded {} safe and {} unsafe training data points".format(sum(n_safe), sum(n_unsafe)))

    tot_train = sum(int(0.4 * n) for n in n_safe) + sum(int(0.4 * n) for n in n_unsafe)
    tot_test = sum(int(0.1 * n) for n in n_safe) + sum(int(0.1 * n) for n in n_unsafe)
    print("Saving {} training and {} test points".format(tot_train, tot_test))
    print("Saving data... ", end="")

    X_train = np.empty((tot_train, df_safe[0].shape[1]))
    X_test = np.empty((tot_train, df_safe[0].shape[1]))
    y_train = np.empty((tot_train, 1))
    y_test = np.empty((tot_train, 1))

    i_train = 0
    i_test = 0
    for df, n in zip(df_safe, n_safe):
        X_train[i_train:i_train+int(0.4 * n), :] = df[:int(0.4 * n)]
        X_test[i_test:i_test+int(0.1 * n), :] = df[int(0.4*n):int(0.4*n)+int(0.1 * n)]
        y_train[i_train:i_train+int(0.4 * n), :] = 1
        y_test[i_test:i_test+int(0.1 * n), :] = 1
        i_train += int(0.4 * n)
        i_test += int(0.1 * n)

    for df, n in zip(df_unsafe, n_unsafe):
        X_train[i_train:i_train+int(0.4 * n), :] = df[:int(0.4 * n)]
        X_test[i_test:i_test+int(0.1 * n), :] = df[int(0.4*n):int(0.4*n)+int(0.1 * n)]
        y_train[i_train:i_train+int(0.4 * n), :] = 0
        y_test[i_test:i_test+int(0.1 * n), :] = 0
        i_train += int(0.4 * n)
        i_test += int(0.1 * n)

    serial.save(pickle_x_train.format(slope=slope), X_train)
    serial.save(pickle_x_test.format(slope=slope), X_test)
    serial.save(pickle_y_train.format(slope=slope), y_train)
    serial.save(pickle_y_test.format(slope=slope), y_test)
    print("done")


if __name__ == "__main__":
    main()
