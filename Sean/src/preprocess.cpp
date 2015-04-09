#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include <openblas/cblas.h>

#include "data_params.h"

#define SPACING 0.2

#define ANGLE (10.*3.1415/180.)

#define R_BASE 1.7
#define R_FOOT 0.25
#define R_MAX (R_BASE + 0.05)
#define R_MIN (R_BASE - 2 * R_FOOT - 0.05)

#define ZH 10
#define ZW 19

#define SETOUTPUT(output, i, j) \
    output[2*i*NCOLS + 2*j] = output[2*i*NCOLS + 2*j + 1] = output[(2*i + 1)*NCOLS + 2*j] = output[(2*i + 1)*NCOLS + 2*j + 1]

std::vector<unsigned char> preprocess_angle(std::vector<float> &data) {
    std::vector<unsigned char> output(NROWS*NCOLS);

    std::array<float, ZH*ZW> z_top;
    std::array<float, ZH*ZW> z_bot;

    std::array<float, ZH*ZW> dist;
    std::vector<int> d_loc;

    // Zero the data
    std::fill(output.begin(), output.end(), 0);

    // pre-compute locations given acceptable distances
    int i = 0;
    float d_sq;
    auto it = dist.begin();
    for (int r = 0; r < ZH; r++) {
        for (int c = -(ZH-1); c < ZH; c++) {
            d_sq = (c * c + r * r) * (SPACING * SPACING);
            if (d_sq >= R_MIN * R_MIN && d_sq <= R_MAX * R_MAX) {
                d_loc.push_back(i);
                *it = sqrt(d_sq);
            }
            i++;
            it++;
        }
    }

    for (int i = 10; i < NROWS_HEIGHT - 10; i++) {
        for (int j = 10; j < NCOLS_HEIGHT - 10; j++) {
            // build the matrices of heights
            for (int k = 0; k < ZH; k++) {
                std::reverse_copy(&data[(i - k)*NCOLS_HEIGHT + j - ZH + 1],
                                  &data[(i - k)*NCOLS_HEIGHT + j + ZH],
                                  &z_top[k * ZW]);
                std::copy(&data[(i + k)*NCOLS_HEIGHT + j - ZH + 1],
                          &data[(i + k)*NCOLS_HEIGHT + j + ZH],
                          &z_bot[k * ZW]);
            }

            // z_bot = -1 * z_top + z_bot
            cblas_saxpy(ZH*ZW, -1., &z_top[0], 1, &z_bot[0], 1);

            // figure out if any cause it to be false
            SETOUTPUT(output, i, j) = 0xff;
            for (auto d_ind = d_loc.begin(); d_ind < d_loc.end(); d_ind++) {
                if (fabs(atan2(z_bot[*d_ind], 2 * dist[*d_ind])) > ANGLE) {
                    SETOUTPUT(output, i, j) = 0x00;
                    break;
                }
            }
        }
    }

    return output;
}