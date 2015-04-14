/*********************************************************************
 * nn_run.h
 ********************************************************************/

#ifndef _NN_RUN_H_
#define _NN_RUN_H_

#include "nn_common.h"


namespace nn {
    int read_layer(const char *fname, std::vector<std::vector<float>> &layer_list, int prev_size);
    void generate_solution(std::vector<unsigned char> &output, std::vector<unsigned char> &image,
                           std::vector<std::vector<float>> &weights, std::vector<std::vector<float>> &biases);
}

#endif
