/*********************************************************************
* nn_run.cpp
*
* Run neural network data
*********************************************************************/

#include <fstream>
#include <iostream>
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
    int file_size, next_size;

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
static inline std::vector<float> feed_fwd(std::vector<float> input, std::vector<float> weights, std::vector<float> bias) {
    std::size_t layer_from, layer_to;

    // Intuit the size the layer maps from and maps to from the bias vector, check that everything is consistent
    layer_to = bias.size();
    layer_from = weights.size() / layer_to;
    if (layer_to * layer_from != weights.size()) {
        error("(feed_fwd) Improper layer bias and weights dimenionsions, got %d and %d (respectively)", bias.size(), weights.size());
    }

    // Also, check that the input data has the same size
    if (layer_from != input.size()) {
        error("(feed_fwd) Improper number of data points fed into layer, got %d, should be size %d", input.size(), layer_from);
    }

    //std::cout << layer_from << " " << layer_to << std::endl;

    // Allocate the output vector
    std::vector<float> output(layer_to);
    //std::cout << 1 << std::endl;

    // Do the matrix multiplication
    // output = weights * input
    // using the CBLAS formula:
    // y = alpha*A*x + beta*y,
    // dim A: M x N
    //cblas_sgemv(CblasRowMajor, CblasNoTrans,
    //        layer_to, layer_from,        // M, N
    //        1., &weights[0], layer_from, // alpha, A, leading dim A
    //        &input[0], 1,                // x, inc x
    //        0., &output[0], 1);          // beta, y, inc y

    // Apply the bias vector to the input
    // output = 1 * bias + output
    //std::cout << 2 << std::endl;
    //cblas_saxpy(layer_to, 1., &bias[0], 1, &output[0], 1);

    return output;
}


/* Run a rectified linear layer
 *
 * Runs the given vector through the given rectified linear layer.
 */
static inline std::vector<float> apply_rectified_linear(std::vector<float> input, std::vector<float> weights, std::vector<float> bias) {
    std::vector<float> output = feed_fwd(input, weights, bias);

    return output;
}


/* Run a softmax layer
 *
 * Runs the given vector through the given softmax layer.
 */
// TODO


/* Generate solution from selection and the given NN parameters
 *
 * Replaces the unknown parts of the output with the fed forward solution.
 */
void nn::generate_solution(std::vector<unsigned char> &output, std::vector<unsigned char> &image, std::vector<std::vector<float>> &weights, std::vector<std::vector<float>> &biases) {
    std::vector<float> nn_input(NN_FEAT);
    std::vector<float> nn_output;

    for (std::size_t i = 0; i < output.size(); i++) {
        // If known safe or unsafe, continue
        if (output[i] == SAFE || output[i] == UNSAFE) {
            continue;
        }

        // Copy the neural net input over
        for (int r = 0; r < NN_WINDOW; r++) {
            for (int c = 0; c < NN_WINDOW; c++) {
                nn_input[r*NN_WINDOW+c] = NORMALIZE(image[i+(r-NN_WINDOW/2) * NROWS + (c-NN_WINDOW/2)]);
            }
        }

        // Rectified Linear Layer
        nn_output = apply_rectified_linear(nn_input, weights[0], biases[0]);
        // Rectified Linear Layer
        //nn_input = apply_rectified_linear(nn_input, weights[1], biases[1]);
        // Softmax Layer
        //nn_input = apply_softmax(nn_input, weights[1], biases[1]);
    }
}
