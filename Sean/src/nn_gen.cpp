/*********************************************************************
* nn_gen.cpp
*
* Generate and save neural network data
*********************************************************************/

#include <fstream>
#include <string>
#include <vector>

#include "nn_gen.h"

#include "data_params.h"
#include "error.h"
#include "preprocess_common.h"

#define NORMALIZE(x) ((float) ((x) - 159) / 255.)

/*
 * Take NROWS x NCOLS selection and an NROWS x NCOLS image and generate the corresponding NN input
 */
void nn::from_pgm_labeled(std::vector<unsigned char> &selection, std::vector<unsigned char> &solution, std::vector<unsigned char> &image,
                          std::vector<float> &output_safe, std::vector<float> &output_unsafe) {
    std::vector<int> ind_safe, ind_unsafe;
    std::size_t n_safe = 0, n_unsafe = 0;

    // First, get a count and pre-allocate output
    // Don't deal with BUFFER, these should all be unsafe anyways
    for (int i = 0; i < NROWS * NCOLS; i++) {
        if (selection[i] == FEED_TO_NET) {
            if (solution[i] == SAFE) {
                n_safe++;
                ind_safe.push_back(i);
            } else if (solution[i] == UNSAFE) {
                n_unsafe++;
                ind_unsafe.push_back(i);
            } else {
                error("(nn::from_pgm_labeled) Index neither safe nor unsafe");
            }
        }
    }

    output_safe.reserve(n_safe * NN_FEAT);
    output_unsafe.reserve(n_unsafe * NN_FEAT);

    int loc;
    // Generate all the safe NN input
    for (auto i = ind_safe.begin(); i < ind_safe.end(); i++) {
        loc = *i;
        for (int r = -NN_WINDOW / 2; r <= NN_WINDOW / 2; r++) {
            for (int c = -NN_WINDOW / 2; c <= NN_WINDOW / 2; c++) {
                output_safe.push_back(NORMALIZE(image[loc + r*NROWS + c]));
            }
        }
    }
    // Generate all the unsafe NN input
    for (auto i = ind_unsafe.begin(); i < ind_unsafe.end(); i++) {
        loc = *i;
        for (int r = -NN_WINDOW / 2; r <= NN_WINDOW / 2; r++) {
            for (int c = -NN_WINDOW / 2; c <= NN_WINDOW / 2; c++) {
                output_unsafe.push_back(NORMALIZE(image[loc + r*NROWS + c]));
            }
        }
    }
}


void nn::write_file(const char *fname, std::vector<float> &output_safe, std::vector<float> &output_unsafe) {
    std::ofstream f;
    std::string safe_name, unsafe_name;

    safe_name = std::string(fname) + "_safe.raw";
    unsafe_name = std::string(fname) + "_unsafe.raw";

    /*************************************************************************
     * Safe file output
     ************************************************************************/

    // Open file
    f.open(safe_name.c_str(), std::ios::out | std::ios::binary);
    // Write file
    if (f.is_open()) {
        f.write((const char*)&output_safe[0], output_safe.size() * sizeof(float));
    } else {
        error("(nn::write_file) Can't open file named '%s' for writing safe file", safe_name.c_str());
    }
    // Check file status
    if (!f) {
        error("(nn::write_file) Error writing safe file");
    }
    // Close file
    f.close();

    /*************************************************************************
     * Unsafe file output
     ************************************************************************/

    // Open file
    f.open(unsafe_name.c_str(), std::ios::out | std::ios::binary);
    // Write file
    if (f.is_open()) {
        f.write((const char*)&output_unsafe[0], output_unsafe.size() * sizeof(float));
    } else {
        error("(nn::write_file) Can't open file named '%s' for writing unsafe file", unsafe_name.c_str());
    }
    // Check file status
    if (!f) {
        error("(nn::write_file) Error writing unsafe file");
    }
    // Close file
    f.close();
}