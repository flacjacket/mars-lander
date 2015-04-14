#include <chrono>
#include <iostream>
#include <string>
#include <vector>

// ... I hate Windows, c.f. pgm.cpp
#include <cstdlib>

#include "data_params.h"

#include "error.h"
#include "nn_run.h"
#include "pgm.h"
#include "preprocess_easy.h"
#include "raw.h"

#define TIME_IT(tp1, tp2, call) \
    tp1 = std::chrono::system_clock::now(); call; tp2 = std::chrono::system_clock::now(); print_tp(tp1, tp2)

static void print_tp(std::chrono::system_clock::time_point tp1, std::chrono::system_clock::time_point tp2) {
    std::chrono::milliseconds diff = std::chrono::duration_cast<std::chrono::milliseconds>(tp2 - tp1);
    std::cout << "Time: " << diff.count() << " ms" << std::endl;
}


/*
 * Run a pre-generated NN from the given inputs, saving the solution
 */
int main(int argc, char* argv[]) {
    std::vector<float> input_height;
    std::vector<unsigned char> input_image;

    std::vector<unsigned char> output_preprocess;
    std::vector<unsigned char> output_nn;

    std::chrono::system_clock::time_point tp1, tp2;

    std::vector<std::vector<float>> biases, weights;
    int prev_layer_size;
    int n_layers;

    if (argc != 6) {
        std::cerr << "Usage: run_easy INPUT_HEIGHT INPUT_IMAGE INPUT_DIR N_LAYERS OUTPUT_SOLUTION" << std::endl;
        error("Needed 5 args, got %d", argc - 1);
    }

    /*************************************************************************
     * Data reading
     ************************************************************************/

    // Reading the heights
    std::cout << "Reading height data from " << argv[1] << std::endl;
    input_height = raw::read_file(argv[1], NROWS_HEIGHT*NCOLS_HEIGHT);

    // Read the image
    std::cout << "Reading image data from " << argv[2] << std::endl;
    input_image = pgm::read_file(argv[2], NROWS, NCOLS);

    // Yada, yada, yada, I hate Windows, WHY CAN'T I USE std::stoi and std::to_string?????????
    prev_layer_size = NN_FEAT;
    n_layers = strtol(argv[4], NULL, 10);
    for (int i = 0; i < n_layers; i++) {
        char buffer[50];
        // Read weights
        sprintf(buffer, "%s/w%d.raw", argv[3], i);
        std::cout << "Reading weights from " << buffer;
        prev_layer_size = nn::read_layer(buffer, biases, prev_layer_size);

        std::cout << ", size " << prev_layer_size << std:: endl;

        // Read biases
        sprintf(buffer, "%s/b%d.raw", argv[3], i);
        std::cout << "Reading biases from " << buffer << std::endl;
        int bias_size = nn::read_layer(buffer, biases, prev_layer_size);

        if (bias_size != 1) {
            error("(main) Bias must have sive 1, got %d", bias_size);
        }

    }

    if (prev_layer_size != 2) {
        error("(main) Final layer must be softmax with size 2");
    }

    /*************************************************************************
     * Data preprocessing
     ************************************************************************/

    // Preprocess the data
    std::cout << "Preprocessing data" << std::endl;
    TIME_IT(tp1, tp2, output_preprocess = preprocess_easy(input_height));

    // Extrapolate to 1000x1000
    std::cout << "Generating PGM" << std::endl;
    TIME_IT(tp1, tp2, output_preprocess = preprocess_gen_pgm(output_preprocess));

    /*************************************************************************
     * Neural Net run
     ************************************************************************/

    // TODO

    /*************************************************************************
     * Data saving
     ************************************************************************/

    // Save the PGM
    std::cout << "Saving PGM to " << argv[5] << std::endl;
    //pgm::write_file(argv[5], output_nn, NROWS, NCOLS);
}
