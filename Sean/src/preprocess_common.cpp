#include <cmath>     // sqrt
#include <vector>

#include "preprocess_common.h"
#include "data_params.h"

#define SET_OUTPUT(output, i, j) \
    output[2*i*NCOLS + 2*j] = output[2*i*NCOLS + 2*j + 1] = output[(2*i + 1)*NCOLS + 2*j] = output[(2*i + 1)*NCOLS + 2*j + 1]


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
void footpad_dist_4point(double r_min, double r_max, std::vector<float> &dist, std::vector<unsigned> &d_loc) {
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
            }
            ind++;
        }
    }
}


/*
 * Turns the NROWS_HEIGHT x NCOLS_HEIGHT data vector to a NROWS x NCOLS vector
 * for pgm output
 */
std::vector<unsigned char> preprocess_gen_pgm(std::vector<unsigned char> &output) {
    std::vector<unsigned char> new_output(NROWS*NCOLS);

    // Zero the data
    std::fill(new_output.begin(), new_output.end(), UNSAFE);

    // Fill in with the passed data
    for (int i = BUFFER/2; i < NROWS_HEIGHT - BUFFER/2; i++) {
        for (int j = BUFFER/2; j < NCOLS_HEIGHT - BUFFER/2; j++) {
            SET_OUTPUT(new_output, i, j) = output[i*NCOLS_HEIGHT + j];
        }
    }

    // Fix the edges
    for (int i = BUFFER; i < NROWS - BUFFER; i++) {
        new_output[BUFFER*NCOLS + i] =                           // Top row
            new_output[i*NCOLS + BUFFER] =                       // left column
            new_output[i*NCOLS + NROWS - BUFFER - 1] =           // right column
            new_output[(NROWS - BUFFER - 1)*NCOLS + i] = UNSAFE; // Bottom row
    }

    return new_output;
}
