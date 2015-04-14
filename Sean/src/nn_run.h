/*********************************************************************
 * nn_run.h
 ********************************************************************/

#ifndef _NN_RUN_H_
#define _NN_RUN_H_

#include "nn_common.h"


namespace nn {
    int read_layer(const char *fname, std::vector<std::vector<float>> &layer_list, int prev_size);
    int generate_input(std::vector<unsigned char> &solution, std::vector<unsigned char> &image, std::vector<float> &nn_input, std::vector<int> &locs);
    std::vector<unsigned char> generate_output(std::vector<float> &nn_input, int n_examples, std::vector<std::vector<float>> &weights, std::vector<std::vector<float>> &biases);
    void apply_output(std::vector<unsigned char> &solution, std::vector<unsigned char> &nn_output, std::vector<int> &locs);
}

#endif
