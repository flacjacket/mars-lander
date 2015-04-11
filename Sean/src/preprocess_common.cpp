#include <cmath>     // sqrt
#include <vector>

#include "preprocess_common.h"
#include "data_params.h"


/*
 * Get the base locations to consider in the height data for a given window
 */
void base_loc(double r_min, double r_max, std::vector<int> &d_loc) {
    double d_sq;
    int ind = 0;

    for (int r = 1 - ZH; r < ZH; r++) {
        for (int c = 1 - ZH; c < ZH; c++) {
            // Compute square distance for the given location and compare it to the min and max square radii
            d_sq = (c * c + r * r) * (SPACING_HEIGHT * SPACING_HEIGHT);
            if (d_sq >= r_min * r_min && d_sq < r_max * r_max) {
                // If it's acceptible, store the index
                d_loc.push_back(ind);
            }
            ind++;
        }
    }
}

/*
 * Get the footpad locations to consider in the height data for a given window
 *
 * Also gives the distance both diametrically opposite, and 90 deg rotated from the point
 */
void footpad_dist_4point(double r_min, double r_max, std::vector<float> &dist, std::vector<int> &d_loc) {
    double d_sq;
    int ind = 0;

    for (int r = 0; r < ZH; r++) {
        for (int c = 1; c < ZH; c++) {
            // Compute square distance for the given location
            d_sq = (c * c + r * r) * (SPACING_HEIGHT * SPACING_HEIGHT);
            // ... and compare it to the min and max square radii
            if (d_sq >= r_min * r_min && d_sq < r_max * r_max) {
                // If it's acceptible, store the index and the distance
                d_loc.push_back(ind);
                dist.push_back(2 * sqrt(d_sq));
                //dist_short.push_back(sqrt(2 * (c + r) * abs(c - r)) * SPACING_HEIGHT);
            }
            ind++;
        }
    }
}


/*
 * Fixes the edges so it matches the data sets we are given
 */
void preprocess_fix_edges(std::vector<unsigned char> &output) {
    for (int i = BUFFER; i < NROWS - BUFFER; i++) {
        output[BUFFER*NCOLS + i] =                         // Top row
            output[i*NCOLS + BUFFER] =                     // left column
            output[i*NCOLS + NROWS - BUFFER - 1] =         // right column
            output[(NROWS - BUFFER - 1)*NCOLS + i] = 0x00; // Bottom row
    }
}
