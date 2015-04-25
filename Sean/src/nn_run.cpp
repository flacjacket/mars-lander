/*********************************************************************
* nn_run.cpp
*
* Run neural network data
*********************************************************************/

#include <cmath>
#include <fstream>
#include <vector>
#include <openblas/cblas.h>

#include "nn_run.h"

#include "data_params.h"
#include "error.h"
#include "preprocess_common.h"

/* Read in a layer
 *
 * Reads in the layer, consisting of a either a bias vector or a weights
 * matrix, from the given file, appends the read data to the given list of
 * vectors.  Returns the size of the layer (given the previous layer has size
 * prev_size).
 */
int nn::read_layer(const char *fname, std::vector<std::vector<float>> &layer_list, int prev_size) {
    unsigned file_size, next_size;

    // Open the file
    std::ifstream f(fname, std::ios::in | std::ios::binary);
    if (!f.is_open()) {
        error("(nn::read_layer) Unable to read layer from %s", fname);
    }

    // Get the layer size
    f.seekg(0, std::ios::end);
    file_size = f.tellg();
    f.seekg(0, std::ios::beg);
    next_size = (file_size / sizeof(float)) / prev_size;

    // Check that the file is correctly sized
    if (next_size * prev_size * sizeof(float) != file_size) {
        error("(nn::read_layer) Layer file does not have acceptible size, got %d, previous layer %d", file_size, prev_size);
    }

    // Read in the layer
    std::vector<float> layer(prev_size * next_size);
    f.read((char*) &layer[0], file_size);
    layer_list.push_back(layer);

    // Error handling
    if (!f) {
        error("(nn::read_layer) Error reading %s", fname);
    }

    // Close file
    f.close();

    return next_size;
}


/* Feed forward a layer
 *
 * Feeds the layer input forward.
 */
static inline void feed_fwd(float *input, float* output, float* weights, float* bias, unsigned n_inputs, unsigned layer_from, unsigned layer_to) {
    // BLAS takes ~5.7 sec

    // Do the matrix multiplication
    // output = weights * input
    // C = alpha*A*B + beta*C  (cblas formula)
    // dim C: M x N
    // dim A: M x K
    // dim B: K x N
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n_inputs, layer_to, layer_from,  // M, N, K
                1., input, layer_from,           // alpha, A, leading dim A
                weights, layer_to,               // B, leading dim B
                0., output, layer_to);           // beta, C, leading dim C

    // Apply the bias vector to the input
    // output = 1 * bias + output
    for (unsigned i = 0; i < n_inputs; i++) {
        cblas_saxpy(layer_to, 1., bias, 1, output + i * layer_to, 1);
    }
}


/* Run a rectified linear layer
 *
 * Runs the given vector through the given rectified linear layer.
 */
static inline void apply_rectified_linear(float *input, float *output, float *weights, float *bias, unsigned n_inputs, unsigned layer_from, unsigned layer_to) {
    // Feed it forward
    feed_fwd(input, output, weights, bias, n_inputs, layer_from, layer_to);
    // Rectify it
    for (unsigned i = 0; i < n_inputs * layer_to; i++) {
        if (output[i] < 0) {
            output[i] = 0;
        }
    }
}


/* Run a softmax layer
 *
 * Runs the given vector through the given softmax layer.
 */
static inline void apply_softmax(float *input, float *layer, unsigned char* output, float* weights, float* bias, unsigned n_inputs, unsigned layer_from, unsigned layer_to) {
    // First, we have to feed forward
    feed_fwd(input, layer, weights, bias, n_inputs, layer_from, layer_to);

    // Now, determine correct output
    for (unsigned i = 0; i < n_inputs; i++) {
        // if second column > first column, it is safe
        output[i] = ((exp(layer[2 * i]) / exp(layer[2 * i + 1])) < NN_CUTOFF) ? SAFE : UNSAFE;
    }
}


/* Generate input to the neural network
 *
 * Constructs the vector to feed into the net
 */
int nn::generate_input(unsigned char *solution, unsigned n_solutions, std::vector<unsigned> &locs) {
    int n_inputs = 0;

    for (unsigned i = 0; i < n_solutions; i++) {
        // If known safe or unsafe, continue
        if (solution[i] == SAFE || solution[i] == UNSAFE) {
            continue;
        }

        // Register the addition of the input
        locs.push_back(i);
        n_inputs++;
    }

    return n_inputs;
}

// Process NN in mini-batches to minimize memory footprint
#define N_BATCH 4096

void nn::generate_solution(
        unsigned char *solution, std::vector<unsigned> &locs, std::vector<unsigned char> &image,
        std::vector<std::vector<float>> &weights, std::vector<std::vector<float>> &biases)
{
    std::vector<unsigned>::iterator loc = locs.begin();
    unsigned i;

    unsigned n_examples = locs.size();
    unsigned size_layer1 = biases[0].size();
    unsigned size_layer2 = biases[1].size();
    unsigned size_layer3 = biases[2].size();
    unsigned size_output = biases[3].size();

    float* nn_input = (float*) malloc(N_BATCH * NN_FEAT * sizeof(float));
    float* nn_layer1 = (float*) malloc(N_BATCH * size_layer1 * sizeof(float));
    float* nn_layer2 = (float*) malloc(N_BATCH * size_layer2 * sizeof(float));
    float* nn_layer3 = (float*) malloc(N_BATCH * size_layer3 * sizeof(float));
    float *nn_layer_output = (float*) malloc(N_BATCH * size_output * sizeof(float));
    unsigned char* nn_output = (unsigned char*) malloc(N_BATCH * size_output);

    for (i = 0; i < (n_examples - 1) / N_BATCH; i++) {
        float* input_it = nn_input;
        for (unsigned j = 0; j < N_BATCH; j++) {
            // Copy the neural net input over
            std::vector<unsigned char>::iterator img_it = image.begin() + *(loc + j) - (NN_WINDOW / 2) * (1 + NROWS);
            for (unsigned r = 0; r < NN_WINDOW; r++) {
                for (unsigned c = 0; c < NN_WINDOW; c++) {
                    *(input_it++) = NORMALIZE(*img_it);
                    img_it++;
                }
                img_it += NROWS - NN_WINDOW;
            }
        }

        // Run the NN
        // Rectified Linear Layer
        apply_rectified_linear(nn_input, nn_layer1, &weights[0][0], &biases[0][0], N_BATCH, NN_FEAT, size_layer1);
        // Rectified Linear Layer
        apply_rectified_linear(nn_layer1, nn_layer2, &weights[1][0], &biases[1][0], N_BATCH, size_layer1, size_layer2);
        // Rectified Linear Layer
        apply_rectified_linear(nn_layer2, nn_layer3, &weights[2][0], &biases[2][0], N_BATCH, size_layer2, size_layer3);
        // Softmax Layer
        apply_softmax(nn_layer3, nn_layer_output, nn_output, &weights[3][0], &biases[3][0], N_BATCH, size_layer3, size_output);

        // Assign the outputs to the proper solution locations
        for (unsigned j = 0; j < N_BATCH; j++) {
            solution[*loc] = nn_output[j];
            loc++;
        }
    }

    // Run the last batch
    n_examples -= i * N_BATCH;

    float *input_it = nn_input;
    for (unsigned j = 0; j < n_examples; j++) {
        // Copy the neural net input over
        std::vector<unsigned char>::iterator img_it = image.begin() + *(loc + j) - (NN_WINDOW / 2) * (1 + NROWS);
        for (unsigned r = 0; r < NN_WINDOW; r++) {
            for (unsigned c = 0; c < NN_WINDOW; c++) {
                *(input_it++) = NORMALIZE(*img_it);
                img_it++;
            }
            img_it += NROWS - NN_WINDOW;
        }
    }

    // Run the NN
    // Rectified Linear Layer
    apply_rectified_linear(nn_input, nn_layer1, &weights[0][0], &biases[0][0], n_examples, NN_FEAT, size_layer1);
    // Rectified Linear Layer
    apply_rectified_linear(nn_layer1, nn_layer2, &weights[1][0], &biases[1][0], n_examples, size_layer1, size_layer2);
    // Rectified Linear Layer
    apply_rectified_linear(nn_layer2, nn_layer3, &weights[2][0], &biases[2][0], n_examples, size_layer2, size_layer3);
    // Softmax Layer
    apply_softmax(nn_layer3, nn_layer_output, nn_output, &weights[3][0], &biases[3][0], n_examples, size_layer3, size_output);

    // Assign the outputs to the proper solution locations
    for (unsigned j = 0; j < n_examples; j++) {
        solution[*loc] = nn_output[j];
        loc++;
    }

    free(nn_input);
    free(nn_layer1);
    free(nn_layer2);
    free(nn_layer3);
    free(nn_layer_output);
    free(nn_output);
}
