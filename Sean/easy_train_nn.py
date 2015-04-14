from __future__ import print_function

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
output_pgm = os.path.join(output_dir, "out_easy.pgm")
output_nn = os.path.join(output_dir, "nn_out")
output_nn_safe = output_nn + "_safe.raw"
output_nn_unsafe = output_nn + "_unsafe.raw"

pickle_x_train = os.path.join(output_dir, "X_train.pkl")
pickle_x_test = os.path.join(output_dir, "X_test.pkl")
pickle_y_train = os.path.join(output_dir, "y_train.pkl")
pickle_y_test = os.path.join(output_dir, "y_test.pkl")

nn_save = os.path.join(output_dir, "nn_train.pkl")
nn_save_best = os.path.join(output_dir, "nn_train_best.pkl")

n_features = 35**2

gen_easy_executable = os.path.abspath(os.path.join(this_dir, "src", "gen_easy"))

if os.name == 'nt':
    gen_easy_executable += ".exe"

yaml = """\
!obj:pylearn2.train.Train {{
    dataset: &train !obj:pylearn2.datasets.dense_design_matrix.DenseDesignMatrix {{
        X: !pkl: '{x_train}',
        y: !pkl: '{y_train}',
        y_labels: 2,
    }},

    model: !obj:pylearn2.models.mlp.MLP {{
        layers : [
            !obj:pylearn2.models.mlp.RectifiedLinear {{
                layer_name: 'h0',
                dim: 500,
                sparse_init: 15,
                # Rather than using weight decay, we constrain the norms of the weight vectors
                max_col_norm: 1.
            }},
            !obj:pylearn2.models.mlp.RectifiedLinear {{
                layer_name: 'h1',
                dim: 500,
                sparse_init: 15,
                # Rather than using weight decay, we constrain the norms of the weight vectors
                max_col_norm: 1.
            }},
            !obj:pylearn2.models.mlp.Softmax {{
                layer_name: 'y',
                n_classes: 2,
                irange: 0,
                init_bias_target_marginals: *train
            }},
        ],
        nvis: {n_features},
    }},

    # We train using batch gradient descent
    algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {{
        batch_size: 1000,

        #learning_rate: 1e-1,
        #learning_rule: !obj:pylearn2.training_algorithms.learning_rule.Momentum {{
        #    init_momentum: 0.5,
        #}},

        # We monitor how well we're doing during training on a validation set
        monitoring_dataset: {{
            'train' : *train,
            'valid' : !obj:pylearn2.datasets.dense_design_matrix.DenseDesignMatrix {{
                X: !pkl: '{x_test}',
                y: !pkl: '{y_test}',
                y_labels: 2,
            }},
        }},

        # We stop after 50 epochs
        termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter {{
            max_epochs: 50,
        }},
    }},

    # We save the model whenever we improve on the validation set classification error
    extensions: [
        !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {{
             channel_name: 'valid_y_misclass',
             save_path: '{save_file_best}'
        }},
        # http://daemonmaker.blogspot.com/2014/12/monitoring-experiments-in-pylearn2.html
        !obj:pylearn2.train_extensions.live_monitoring.LiveMonitoring {{}},
        # Not sure what this does...
        #!obj:pylearn2.training_algorithms.sgd.LinearDecayOverEpoch {{
        #    start: 5,
        #    saturate: 100,
        #    decay_factor: .01
        #}}
    ],

    save_path: '{save_file}',
    save_freq: 1,
}}
""".format(
    n_features=n_features,
    save_file=nn_save,
    save_file_best=nn_save_best,
    x_train=pickle_x_train,
    x_test=pickle_x_test,
    y_train=pickle_y_train,
    y_test=pickle_y_test
)


def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ##########################################################################
    # Generate NN input
    ##########################################################################

    subprocess.check_call([gen_easy_executable, input_height, input_image,
                           input_solution, output_pgm, output_nn])

    ##########################################################################
    # Pickle NN input
    ##########################################################################

    df_safe = np.fromfile(output_nn_safe, dtype=np.float32).reshape((-1, n_features))
    df_unsafe = np.fromfile(output_nn_unsafe, dtype=np.float32).reshape((-1, n_features))

    n_safe = df_safe.shape[0]
    n_unsafe = df_unsafe.shape[0]

    print("Loaded {} safe and {} unsafe training data points".format(n_safe, n_unsafe))

    n_ref = min(n_safe, n_unsafe)
    n_train = int(n_ref * 0.75)
    n_test = n_ref - n_train

    print("Training on {} of each, testing on {} of each".format(n_train, n_test))

    X_train = np.vstack([df_safe[:n_train, :],
                        df_unsafe[:n_train, :]])
    X_test = np.vstack([df_safe[n_train:n_train+n_test, :],
                        df_unsafe[n_train:n_train+n_test, :]])

    y_train = np.vstack([np.ones((n_train, 1), dtype=int),
                         np.zeros((n_train, 1), dtype=int)])
    y_test = np.vstack([np.ones((n_test, 1), dtype=int),
                        np.zeros((n_test, 1), dtype=int)])

    print("Saving data... ", end="")
    serial.save(pickle_x_train, X_train)
    serial.save(pickle_x_test, X_test)
    serial.save(pickle_y_train, y_train)
    serial.save(pickle_y_test, y_test)
    print("done")

    ##########################################################################
    # Train NN
    ##########################################################################

    train = yaml_parse.load(yaml)
    train.main_loop()


if __name__ == "__main__":
    main()
