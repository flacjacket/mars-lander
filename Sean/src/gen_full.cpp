#include <chrono>
#include <iostream>
#include <vector>

#include "data_params.h"

#include "error.h"
#include "nn_gen.h"
#include "pgm.h"
#include "preprocess_full.h"
#include "raw.h"

#define TIME_IT(tp1, tp2, call) \
    tp1 = std::chrono::system_clock::now(); call; tp2 = std::chrono::system_clock::now(); print_tp(tp1, tp2);

void print_tp(std::chrono::system_clock::time_point tp1, std::chrono::system_clock::time_point tp2) {
    std::chrono::milliseconds diff = std::chrono::duration_cast<std::chrono::milliseconds>(tp2 - tp1);
    std::cout << "Took " << diff.count() << " ms" << std::endl;
}

int main(int argc, char* argv[]) {
    std::vector<float> input_height;
    std::vector<unsigned char> input_image, input_solution;
    std::vector<unsigned char> output_pgm;
    std::vector<float> output_nn_safe, output_nn_unsafe;

    std::chrono::system_clock::time_point tp1, tp2;

    if (argc != 6) {
        std::cerr << "Usage:  gen_easy INPUT_HEIGHT INPUT_IMAGE INPUT_SOLUTION OUTPUT_PGM OUTPUT_NN" << std::endl;
        error("Need 5 args, got %d", argc - 1);
    }

    /*************************************************************************
     * Data reading
     ************************************************************************/

    // Read the heights
    std::cout << "Reading height data from " << argv[1] << std::endl;
    data = raw::read_file(argv[1], NROWS_HEIGHT*NCOLS_HEIGHT);

    // Read the image
    std::cout << "Reading image data from " << argv[2] << std::endl;
    input_image = pgm::read_file(argv[2], NROWS, NCOLS);

    // Read the solution
    std::cout << "Reading solution data from " << argv[3] << std::endl;
    input_solution = pgm::read_file(argv[3], NROWS, NCOLS);
    std::cout << std::endl;

    /*************************************************************************
     * Data processing
     ************************************************************************/

    // Preprocess the data
    std::cout << "Preprocessing data" << std::endl;
    TIME_IT(tp1, tp2, output = preprocess_full(input_height));

    // Extrapolate to 1000x1000
    std::cout << "Generating PGM" << std::endl;
    TIME_IT(tp1, tp2, output_pgm = preprocess_gen_pgm(output_pgm));

    // Get the NN output
    std::cout << "Generating NN input" << std::endl;
    TIME_IT(tp1, tp2, nn::from_pgm_labeled(output_pgm, input_solution, input_image, output_nn_safe, output_nn_unsafe));
    std::cout << std::endl;

    /*************************************************************************
     * Data saving
     ************************************************************************/

    // Save the PGM
    std::cout << "Saving data PGM to " << argv[4] << std::endl;
    pgm::write_file(argv[4], output_pgm, NROWS, NCOLS);

    // Save NN input
    std::cout << "Saving NN to " << argv[5] << "_safe.raw and " << argv[5] << "_unsafe.raw" << std::endl;
    nn::write_file(argv[5], output_nn_safe, output_nn_unsafe);
}
