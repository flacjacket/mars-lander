#include <algorithm> // std::fill, std::reverse_copy
#include <array>
#include <cmath>     // atan2, fabs
#include <vector>
#include <openblas/cblas.h>

#include "data_params.h"

// Use up to 10 deg
#define ANGLE (10.*3.1415/180.)

#define R_MAX (R_BASE + 0.05)
#define R_MIN (R_BASE - 2 * R_FOOT - 0.05)

#define ZH 10
#define ZW 19

#define SET_OUTPUT(output, i, j) \
    output[2*i*NCOLS + 2*j] = output[2*i*NCOLS + 2*j + 1] = output[(2*i + 1)*NCOLS + 2*j] = output[(2*i + 1)*NCOLS + 2*j + 1]


/*
 * Get the footpad locations to consider in the height data for a given window
 *
 * Pass in the inner and outer radii, and the vectors to populate with the
 * distances between the points and the indices that they reside in.
 */
static void footpad_dist_2point(double r_min, double r_max,
                                std::vector<float> &dist, std::vector<int> &d_loc) {
    int i = 0;
    float d_sq;

    for (int r = 0; r < ZH; r++) {
        for (int c = -(ZH-1); c < ZH; c++) {
            // Compute square distance for the given location
            d_sq = (c * c + r * r) * (SPACING_HEIGHT * SPACING_HEIGHT);
            // ... and compare it to the min and max square radii
            if (d_sq >= r_min * r_min && d_sq <= r_max * r_max) {
                // If it's acceptible, store the index and the distance
                d_loc.push_back(i);
                dist.push_back(2 * sqrt(d_sq));
            }
            i++;
        }
    }
}


std::vector<unsigned char> preprocess_angle(std::vector<float> &data) {
    std::vector<unsigned char> output(NROWS*NCOLS);
    unsigned char to_output;

    std::array<float, ZH*ZW> z_top;
    std::array<float, ZH*ZW> z_bot;

    std::vector<float> dist;
    std::vector<int> d_loc;


    // Zero the data
    std::fill(output.begin(), output.end(), 0);

    // pre-compute locations given acceptable distances
    footpad_dist_2point(R_MIN, R_MAX, dist, d_loc);

    for (int i = BUFFER/2; i < NROWS_HEIGHT - BUFFER/2; i++) {
        for (int j = BUFFER/2; j < NCOLS_HEIGHT - BUFFER/2; j++) {
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
            to_output = 0xff;
            auto dist_ind = dist.begin();
            for (auto z_ind = d_loc.begin(); z_ind < d_loc.end(); z_ind++, dist_ind++) {
                if (fabs(atan2(z_bot[*z_ind], *dist_ind)) > ANGLE) {
                    to_output = 0x00;
                    break;
                }
            }

            SET_OUTPUT(output, i, j) = to_output;
        }
    }

    return output;
}


/*
 * Fixes the edges so it matches the data sets we are given
 */
void fix_edges(std::vector<unsigned char> &output) {
    for (int i = BUFFER; i < NROWS - BUFFER; i++) {
        output[BUFFER*NCOLS + i] =                         // Top row
            output[i*NCOLS + BUFFER] =                     // left column
            output[i*NCOLS + NROWS - BUFFER - 1] =         // right column
            output[(NROWS - BUFFER - 1)*NCOLS + i] = 0x00; // Bottom row
    }
}
