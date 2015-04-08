#include <array>
#include <chrono>
#include <iostream>

#include "height_params.h"
#include "pgm.h"
#include "preprocess.h"
#include "readraw.h"

#define TIME_IT(tp1, tp2, call) \
    tp1 = std::chrono::system_clock::now(); call; tp2 = std::chrono::system_clock::now();

void print_tp(std::chrono::system_clock::time_point tp1, std::chrono::system_clock::time_point tp2) {
    std::chrono::milliseconds diff = std::chrono::duration_cast<std::chrono::milliseconds>(tp2 - tp1);
    std::cout << "Took " << diff.count() << " ms" << std::endl;
}


int main() {
    std::array<float, NROWS*NCOLS> data;
    std::array<unsigned char, 4*NROWS*NCOLS> output;

    std::chrono::system_clock::time_point tp1, tp2;

    // Get the data
    std::cout << "Inputing data" << std::endl;
    TIME_IT(tp1, tp2, read_raw("raw.dem", data));
    print_tp(tp1, tp2);

    // Preprocess the data
    std::cout << "Preprocessing data" << std::endl;
    //preprocess_angle(data, output);
    TIME_IT(tp1, tp2, preprocess_angle(data, output));
    print_tp(tp1, tp2);

    // Save the output
    pgmWriteFile("out.pgm", &output[0], 2*NROWS, 2*NCOLS);
}
