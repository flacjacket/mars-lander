#include <chrono>
#include <iostream>
#include <vector>

#include "data_params.h"

#include "error.h"
#include "pgm.h"
#include "preprocess_easy.h"
#include "raw.h"

#define TIME_IT(tp1, tp2, call) \
    tp1 = std::chrono::system_clock::now(); call; tp2 = std::chrono::system_clock::now(); print_tp(tp1, tp2)

/*
 * Generate a Neural Network input for the given input height files
 */

void print_tp(std::chrono::system_clock::time_point tp1, std::chrono::system_clock::time_point tp2) {
    std::chrono::milliseconds diff = std::chrono::duration_cast<std::chrono::milliseconds>(tp2 - tp1);
    std::cout << "Took " << diff.count() << " ms" << std::endl;
}

int main(int argc, char* argv[]) {
    std::vector<float> input_height;
    std::vector<unsigned char> input_image;
    std::vector<unsigned char> output_pgm, output_nn;

    std::chrono::system_clock::time_point tp1, tp2;

    if (argc != 4) {
        std::cerr << "Usage:  gen_easy INPUT_HEIGHT INPUT_IMAGE OUTPUT_PGM OUTPUT_NN" << std::endl;
        error("Need 4 args, got %d", argc - 1);
    }

    // Get the data
    std::cout << "Reading height data from " << argv[1] << std::endl;
    input_height = raw::read_file(argv[1], NROWS_HEIGHT*NCOLS_HEIGHT);

    // Preprocess the data
    std::cout << "Preprocessing data" << std::endl;
    TIME_IT(tp1, tp2, output_pgm = preprocess_easy(input_height));

    // Fix up the edges
    std::cout << "Generating PGM" << std::endl;
    TIME_IT(tp1, tp2, output_pgm = preprocess_gen_pgm(output_pgm));

    // Save the output
    std::cout << "Saving PGM to " << argv[3] << std::endl;
    TIME_IT(tp1, tp2, pgm::write_file(argv[3], output_pgm, NROWS, NCOLS));

    // Read the image
    std::cout << "Reading image data from " << argv[2] << std::endl;
    TIME_IT(tp1, tp2, input_image = pgm::read_file(argv[2], NROWS, NCOLS));

    // Get the NN output
    std::cout << "Generating NN input" << std::endl;
    //TIME_IT(tp1, tp2, output_nn = nn_from_pgm(output));

    // Save NN input input
    std::cout << "Saving NN to " << argv[4] << std::endl;
    //TIME_IT(tp1, tp2, nn_write_file(output_nn));

    /* This block checks the reading of PGM files against the exported PGM
    // Read that output back in
    std::cout << "Re-reading data" << std::endl;
    input = pgmReadFile("out.pgm", NROWS, NCOLS);

    std::cout << "Checking data" << std::endl;
    auto i = input.begin();
    for (auto o = output.begin(); o < output.end(); o++, i++) {
        if (*o != *i) {
            std::cout << "Mis-match!" << std::endl;
        }
    }
    */
}
