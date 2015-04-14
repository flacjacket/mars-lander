/*********************************************************************
* nn_run.cpp
*
* Run neural network data
*********************************************************************/

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
static inline std::vector<float> feed_fwd(std::vector<float> input, std::vector<float> weights, std::vector<float> bias, unsigned n_inputs) {
    std::size_t layer_from, layer_to;

    // Intuit the size the layer maps from and maps to from the bias vector, check that everything is consistent
    layer_to = bias.size();
    layer_from = weights.size() / layer_to;
    if (layer_to * layer_from != weights.size()) {
        error("(feed_fwd) Improper layer bias and weights dimenionsions, got %d and %d (respectively)", bias.size(), weights.size());
    }

    // Also, check that the input data has the same size
    if (layer_from * n_inputs != input.size()) {
        error("(feed_fwd) Improper number of data points fed into layer, got %d, should be size %d", input.size(), layer_from * n_inputs);
    }

    // Allocate the output vector
    std::vector<float> output(layer_to * n_inputs);

    // Do the matrix multiplication
    // output = weights * input
    // C = alpha*A*B + beta*C  (cblas formula)
    // dim C: M x N
    // dim A: M x K
    // dim B: K x N
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        n_inputs, layer_to, layer_from,  // M, N, K
        1., &input[0], layer_from,       // alpha, A, leading dim A
        &weights[0], layer_to,           // B, leading dim B
        0., &output[0], layer_to);       // beta, C, leading dim C

    // Apply the bias vector to the input
    // output = 1 * bias + output
    for (unsigned i = 0; i < n_inputs; i++) {
        cblas_saxpy(layer_to, 1., &bias[0], 1, &output[i * layer_to], 1);
    }

    return output;
}


/* Run a rectified linear layer
 *
 * Runs the given vector through the given rectified linear layer.
 */
static inline std::vector<float> apply_rectified_linear(std::vector<float> input, std::vector<float> weights, std::vector<float> bias, int n_inputs) {
    std::vector<float> output = feed_fwd(input, weights, bias, n_inputs);

    // rectify it
    for (unsigned i = 0; i < output.size(); i++) {
        if (output[i] < 0) {
            output[i] = 0;
        }
    }

    return output;
}


/* Run a softmax layer
 *
 * Runs the given vector through the given softmax layer.
 */
static inline std::vector<unsigned char> apply_softmax(std::vector<float> input, std::vector<float> weights, std::vector<float> bias, int n_inputs) {
    std::vector<unsigned char> output(n_inputs);
    std::vector<float> layer = feed_fwd(input, weights, bias, n_inputs);

    for (int i = 0; i < n_inputs; i++) {
        // if second column > first column, it is safe
        output[i] = (layer[2 * i] < layer[2 * i + 1]) ? SAFE : UNSAFE;
    }

    return output;
}


/* Generate input to the neural network
 *
 * Constructs the vector to feed into the net
 */
int nn::generate_input(std::vector<unsigned char> &solution, std::vector<unsigned char> &image, std::vector<float> &nn_input, std::vector<int> &locs) {
    int n_inputs = 0;

    for (unsigned i = 0; i < solution.size(); i++) {
        // If known safe or unsafe, continue
        if (solution[i] == SAFE || solution[i] == UNSAFE) {
            continue;
        }

        // Register the addition of the input
        locs.push_back(i);
        n_inputs++;
    }

    // Resize the input
    nn_input.resize(n_inputs* NN_FEAT);

    int loc;
    for (unsigned i = 0; i < locs.size(); i++) {
        loc = locs[i];
        // Copy the neural net input over
        for (int r = 0; r < NN_WINDOW; r++) {
            for (int c = 0; c < NN_WINDOW; c++) {
                nn_input[i*NN_FEAT + r*NN_WINDOW + c] = NORMALIZE(image[loc + (r - NN_WINDOW / 2) * NROWS + (c - NN_WINDOW / 2)]);
            }
        }
    }

    return n_inputs;
}


/* Actually run the net
 */
std::vector<unsigned char> nn::generate_output(std::vector<float> &nn_input, int n_examples, std::vector<std::vector<float>> &weights, std::vector<std::vector<float>> &biases) {
    std::vector<float> nn_layer;
    // Run the NN
    // Rectified Linear Layer
    nn_layer = apply_rectified_linear(nn_input, weights[0], biases[0], n_examples);
    // Rectified Linear Layer
    nn_layer = apply_rectified_linear(nn_layer, weights[1], biases[1], n_examples);
    // Softmax Layer
    return apply_softmax(nn_layer, weights[2], biases[2], n_examples);
}


/* Apply the solution of the net to the image
 */
void nn::apply_output(std::vector<unsigned char> &solution, std::vector<unsigned char> &nn_output, std::vector<int> &locs) {
    for (unsigned i = 0; i < locs.size(); i++) {
        solution[locs[i]] = nn_output[i];
    }
}