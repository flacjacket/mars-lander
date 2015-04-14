/*********************************************************************
* nn_run.cpp
*
* Run neural network data
*********************************************************************/

#include <fstream>
#include <vector>

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
    int file_size, layer_size;

    // Open the file
    std::ifstream f(fname, std::ios::in | std::ios::binary);
    if (!f.is_open()) {
        error("(nn::read_layer) Unable to read layer from %s", fname);
    }

    // Get the layer size
    f.seekg(0, std::ios::end);
    file_size = f.tellg();
    f.seekg(0, std::ios::beg);
    layer_size = (file_size / sizeof(float)) / prev_size;

    // Check that the file is correctly sized
    if (layer_size * prev_size * sizeof(float) != file_size) {
        error("(nn::read_layer) Layer file does not have acceptible size, got %d, previous layer %d", file_size, prev_size);
    }

    // Read in the layer
    std::vector<float> layer(file_size);
    f.read((char*) &layer[0], file_size);
    layer_list.push_back(layer);

    // Error handling
    if (!f) {
        error("(nn::read_layer) Error reading %s", fname);
    }

    // Close file
    f.close();

    return layer_size;
}


/* Run a regularized linear layer
 *
 * Runs the given vector through the given rectified linear layer.
 */
// TODO


/* Run a softmax layer
 *
 * Runs the given vector through the given softmax layer.
 */
// TODO
