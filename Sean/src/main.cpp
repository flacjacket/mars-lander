#include <chrono>
#include <iostream>
#include <vector>

#include "data_params.h"

#include "pgm.h"
#include "preprocess.h"
#include "readraw.h"

#define TIME_IT(tp1, tp2, call) \
    tp1 = std::chrono::system_clock::now(); call; tp2 = std::chrono::system_clock::now(); print_tp(tp1, tp2);

void print_tp(std::chrono::system_clock::time_point tp1, std::chrono::system_clock::time_point tp2) {
    std::chrono::milliseconds diff = std::chrono::duration_cast<std::chrono::milliseconds>(tp2 - tp1);
    std::cout << "Took " << diff.count() << " ms" << std::endl;
}

int main() {
    std::vector<float> data;
    std::vector<unsigned char> output, input;

    std::chrono::system_clock::time_point tp1, tp2;

    // Get the data
    std::cout << "Reading data " << std::endl;
    TIME_IT(tp1, tp2, data = read_raw("raw.dem", NROWS_HEIGHT*NCOLS_HEIGHT));

    // Preprocess the data
    std::cout << "Preprocessing data" << std::endl;
    TIME_IT(tp1, tp2, output = preprocess_angle(data));

    // Save the output
    std::cout << "Saving data" << std::endl;
    TIME_IT(tp1, tp2, pgmWriteFile("out.pgm", output, NROWS, NCOLS));

    // Read that output back in
    std::cout << "Re-reading data" << std::endl;
    TIME_IT(tp1, tp2, input = pgmReadFile("out.pgm", NROWS, NCOLS));

    std::cout << "Checking data" << std::endl;
    auto i = input.begin();
    for (auto o = output.begin(); o < output.end(); o++, i++) {
        if (*o != *i) {
            std::cout << "Mis-match!" << std::endl;
        }
    }
}
