/*********************************************************************
 * nn_run.h
 ********************************************************************/

#ifndef _NN_RUN_H_
#define _NN_RUN_H_

#include "nn_common.h"

#define NN_CUTOFF 1.

namespace nn {
    int read_layer(const char *fname, std::vector<std::vector<float>> &layer_list, int prev_size);

    int generate_input(std::vector<unsigned char> &solution, std::vector<unsigned> &locs);
    void generate_solution(std::vector<unsigned char> &solution, std::vector<unsigned> &locs, std::vector<unsigned char> &image, std::vector<std::vector<float>> &weights, std::vector<std::vector<float>> &biases);
}

#endif
