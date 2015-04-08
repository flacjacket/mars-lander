#include <algorithm>
#include <cmath>
#include <openblas/cblas.h>

#include "readraw.h"

#define ANGLE (10.*3.1415/180.)

#define ZH 10
#define ZW 19

int main() {
    std::array<float, NROWS*NCOLS> data;
    std::array<unsigned char, NROWS*NCOLS> output;

    std::array<float, ZH*ZW> z_top;
    std::array<float, ZH*ZW> z_bot;
    std::array<float, ZH*ZW> dist;

    auto it = dist.begin();
    for (int c = -(ZH-1); c < ZH; c++) {
        for (int r = 0; r < ZH; r++) {
            *it = sqrt(c * c + r * r);
            it++;
        }
    }

    // Get the data
    data = read_raw("raw.dem");

    for (int i = 10; i < NROWS - 10; i++) {
        for (int j = 10; j < NCOLS - 10; j++) {
            // build the matrices of heights
            for (int k = 0; k < ZH; k++) {
                std::copy(&data[(i-(ZH-1)+k)*NCOLS + j - ZH+1],
                          &data[(i-(ZH-1)+k)*NCOLS + j + ZH],
                          &z_top[k * ZW]);
                std::reverse_copy(&data[(i+(ZH-1)-k)*NCOLS + j - ZH+1],
                                  &data[(i+(ZH-1)-k)*NCOLS + j + ZH],
                                  &z_bot[k * ZW]);
            }

            // z_bot = -1 * z_top + z_bot
            cblas_saxpy(ZH*ZW, -1, &z_top[0], 1, &z_bot[0], 1);

            // figure out if any cause it to be false
            auto z = z_bot.begin();
            for (auto d = dist.begin(); z < z_bot.end(); z++, d++) {
                if (abs(atan2(*z, *d)) > ANGLE) {
                    output[i*NCOLS + j] = 0x00;
                    break;
                }
            }
            if (z == z_bot.end()) {
                output[i*NCOLS + j] = 0xff;
            }
        }
    }
}
