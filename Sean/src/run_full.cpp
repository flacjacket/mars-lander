#include <chrono>
#include <iostream>
#include <string>
#include <vector>

// ... I hate Windows, c.f. pgm.cpp
#include <cstdio>
#include <cstdlib>

#include "data_params.h"

#include "error.h"
#include "nn_run.h"
#include "pgm.h"
#include "preprocess_full.h"
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
    std::vector<unsigned char> solution;

    // input and output from net
    std::vector<float> nn_input;
    std::vector<unsigned char> nn_output;
    // layer wieghts and biases
    std::vector<std::vector<float>> weights, biases;
    // locations of points to feed to net
    std::vector<unsigned> nn_locs;

    std::chrono::system_clock::time_point tp1, tp2;


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

    {
        int prev_layer_size, next_layer_size, n_layers;
        prev_layer_size = NN_FEAT;
        // Figure out number of layers
        // Yada, yada, yada, I hate Windows, WHY CAN'T I USE std::stoi and std::to_string?????????
        n_layers = strtol(argv[4], NULL, 10);
        // Now read in that many weights and biases
        for (int i = 0; i < n_layers; i++) {
            char buffer[50];
            // Read weights
            sprintf(buffer, "%s/w%d.raw", argv[3], i);
            std::cout << "Reading weights from " << buffer;
            next_layer_size = nn::read_layer(buffer, weights, prev_layer_size);

            std::cout << ", " << next_layer_size << " x " << prev_layer_size << std:: endl;

            // Read biases
            sprintf(buffer, "%s/b%d.raw", argv[3], i);
            std::cout << "Reading biases from " << buffer << std::endl;
            int bias_size = nn::read_layer(buffer, biases, next_layer_size);

            if (bias_size != 1) {
                error("(main) Bias must have sive 1, got %d", bias_size);
            }

            prev_layer_size = next_layer_size;
        }

        if (prev_layer_size != 2) {
            error("(main) Final layer must be softmax with size 2");
        }
    }
    std::cout << std::endl;

    /*************************************************************************
     * Data preprocessing
     ************************************************************************/

    // Preprocess the data
    std::cout << "Preprocessing data" << std::endl;
    TIME_IT(tp1, tp2, solution = preprocess_full(input_height));

    // Extrapolate to 1000x1000
    std::cout << "Generating PGM" << std::endl;
    TIME_IT(tp1, tp2, solution = preprocess_gen_pgm(solution));
    std::cout << std::endl;

    /*************************************************************************
     * Neural Net run
     ************************************************************************/

    // Generate the NN input
    std::cout << "Find NN input locations" << std::endl;
    TIME_IT(tp1, tp2, nn::generate_input(solution, nn_locs));

    // Generate the NN output
    std::cout << "Generating NN solution" << std::endl;
    TIME_IT(tp1, tp2, nn::generate_solution(solution, nn_locs, input_image, weights, biases));
    std::cout << std::endl;

    /*************************************************************************
     * Data saving
     ************************************************************************/

    // Save the PGM
    std::cout << "Saving PGM to " << argv[5] << std::endl;
    pgm::write_file(argv[5], solution, NROWS, NCOLS);
}
