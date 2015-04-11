#include <algorithm> // std::fill, std::reverse_copy
#include <array>
#include <cmath>     // atan2, fabs
#include <vector>
#include <openblas/cblas.h>

#include "data_params.h"

// Use 10 deg as high safe cutoff
#define ANGLE_SAFE (10.*3.1415/180.)
// Use 12 deg as low unsafe cutoff
#define ANGLE_UNSAFE (13.*3.1415/180.)

// Use 0.29 m as high safe cutoff
#define HEIGHT_SAFE (HEIGHT_UNSAFE - 0.1)
// Use 0.39 m as low unsafe cutoff
#define HEIGHT_UNSAFE 0.39

#define ZH 11
#define ZW 21

#define COS_ATAN(x) (1. / sqrt((x) * (x) + 1))

#define SET_OUTPUT(output, i, j) \
    output[2*i*NCOLS + 2*j] = output[2*i*NCOLS + 2*j + 1] = output[(2*i + 1)*NCOLS + 2*j] = output[(2*i + 1)*NCOLS + 2*j + 1]


/*
 * Get the base locations to consider in the height data for a given window
 */
static inline void base_loc(double r_min, double r_max, std::vector<int> &d_loc) {
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
static inline void footpad_dist_4point(double r_min, double r_max,
                                std::vector<float> &dist_long, std::vector<float> &dist_short,
                                std::vector<int> &d_loc) {
    double d_sq;
    int ind = 0;

    for (int r = 0; r < ZH; r++) {
        for (int c = 1; c < ZH; c++) {
            // Compute square distance for the given location
            d_sq = (c * c + r * r) * (SPACING_HEIGHT * SPACING_HEIGHT);
            // ... and compare it to the min and max square radii
            if (d_sq >= r_min * r_min && d_sq <= r_max * r_max) {
                // If it's acceptible, store the index and the distance
                d_loc.push_back(ind);
                dist_long.push_back(2 * sqrt(d_sq));
                dist_short.push_back(sqrt(2 * (c + r) * abs(c - r)) * SPACING_HEIGHT);
            }
            ind++;
        }
    }
}


std::vector<unsigned char> preprocess_full(std::vector<float> &data) {
    std::vector<unsigned char> output(NROWS*NCOLS);
    unsigned char to_output;

    std::array<float, ZH*(ZH-1)> z_nw;
    std::array<float, ZH*(ZH-1)> z_ne;
    std::array<float, ZH*(ZH-1)> z_se;
    std::array<float, ZH*(ZH-1)> z_sw;

    std::vector<float> dist_long;
    std::vector<float> dist_short;
    std::vector<int> d_loc;

    double cos_theta, cos_phi1, cos_phi2;
    float z1, z2, dz1, dz2;

    // Zero the data
    std::fill(output.begin(), output.end(), 0);

    // pre-compute locations given acceptable distances
    // Distances to guarantee unsafe
    footpad_dist_4point(R_BASE - 2 * R_FOOT, R_BASE, dist_long, dist_short, d_loc);
    // Distances to guarantee safe
    // TODO

    for (int i = BUFFER/2; i < NROWS_HEIGHT - BUFFER/2; i++) {
        for (int j = BUFFER/2; j < NCOLS_HEIGHT - BUFFER/2; j++) {
            // build the matrices of heights
            // TODO: This can be improved by re-using data
            for (int k = 0; k < ZH; k++) {
                std::reverse_copy(&data[(i - k)*NCOLS_HEIGHT + j - ZH + 1],
                                  &data[(i - k)*NCOLS_HEIGHT + j],
                                  &z_nw[k * (ZH - 1)]);
                std::copy(&data[(i + k)*NCOLS_HEIGHT + j + 1],
                          &data[(i + k)*NCOLS_HEIGHT + j + ZH],
                          &z_se[k * (ZH - 1)]);

                // NE and SW are not contiguous mem copies, run another loop
                for (int l = 0; l < ZH - 1; l++) {
                    z_ne[k * (ZH - 1) + l] = data[(i - l - 1)*NCOLS_HEIGHT + j + k];
                    z_sw[k * (ZH - 1) + l] = data[(i + l + 1)*NCOLS_HEIGHT + j - k];
                }
            }

            // Diametrically opposite delta z
            // z_se = -1 * z_nw + z_se
            cblas_saxpy(ZH*(ZH - 1), -1., &z_nw[0], 1, &z_se[0], 1);
            // z_sw = -1 * z_ne + z_sw
            cblas_saxpy(ZH*(ZH - 1), -1., &z_ne[0], 1, &z_sw[0], 1);

            // figure out if any cause it to be unsafe
            to_output = 0xff;
            //auto dist_long_ind = dist_long.begin();
            //auto dist_short_ind = dist_short.begin();
            for (auto z_ind = d_loc.begin(); z_ind < d_loc.end(); z_ind++/*, dist_long_ind++, dist_short_ind++*/) {
                /*
                 * Find guaranteed unsafe
                 * Need to use full diameter to guaratee is unsafe
                 */
                // Determine tilting configuration
                z1 = z_nw[*z_ind];
                dz1 = z_se[*z_ind];
                z2 = z_ne[*z_ind];
                dz2 = z_sw[*z_ind];
                if (z1 + dz1 / 2 > z2 + dz2 / 2) {
                    // Primary landing feet NW/SE
                    //cos_theta = cos(atan2(dz1, 2 * R_BASE));
                    cos_theta = COS_ATAN(dz1 / (2 * R_BASE));
                    // Midpoint of primary to either secondary
                    cos_phi1 = COS_ATAN((z2 - (z1 + dz1 / 2)) / R_BASE);
                    cos_phi2 = COS_ATAN((z2 + dz2 - (z1 + dz1 / 2)) / R_BASE);
                } else {
                    // Primary landing feet NE/SW
                    cos_theta = COS_ATAN(dz2 / (2 * R_BASE));
                    // Midpoint of primary to either secondary
                    cos_phi1 = COS_ATAN((z1 - (z2 + dz2 / 2)) / R_BASE);
                    cos_phi2 = COS_ATAN((z1 + dz1 - (z2 + dz2 / 2)) / R_BASE);
                }

                // Check tilting each direction
                if (fabs(acos(cos_theta * cos_phi1)) > ANGLE_UNSAFE || fabs(acos(cos_theta * cos_phi2)) > ANGLE_UNSAFE) {
                    to_output = 0x00;
                    break;
                }
            }

            SET_OUTPUT(output, i, j) = to_output;
            continue;
        }
    }

    return output;
}


std::vector<unsigned char> preprocess_easy(std::vector<float> &data) {
    std::vector<unsigned char> output(NROWS*NCOLS);
    std::vector<int> d_loc_unsafe, d_loc_safe;

    std::array<float, ZW*ZW> z;
    float cur_z, min_z, max_z;

    // Zero the data
    std::fill(output.begin(), output.end(), 0);

    // Get all of the base locations
    // Check known unsafe
    base_loc(0, R_BASE, d_loc_unsafe);
    // Check known safe
    base_loc(R_BASE, R_BASE + 2 * SPACING_HEIGHT, d_loc_safe);

    for (int i = BUFFER/2; i < NROWS_HEIGHT - BUFFER/2; i++) {
        for (int j = BUFFER/2; j < NCOLS_HEIGHT - BUFFER/2; j++) {
            // build the matrices of heights
            // TODO: This can be improved by re-using data
            for (int k = 1 - ZH; k < ZH; k++) {
                std::copy(&data[(i + k)*NCOLS_HEIGHT + j + 1 - ZH],
                    &data[(i + k)*NCOLS_HEIGHT + j + ZH],
                    &z[(k + ZH - 1) * ZW]);
            }

            // find min and max heights
            min_z = max_z = z[d_loc_unsafe[0]];
            for (auto ind = d_loc_unsafe.begin(); ind < d_loc_unsafe.end(); ind++) {
                cur_z = z[*ind];
                if (cur_z < min_z) {
                    min_z = cur_z;
                }
                if (cur_z > max_z) {
                    max_z = cur_z;
                }
            }

            // Here we can say if KNOWN UNSAFE
            if (max_z - min_z > HEIGHT_UNSAFE) {
                goto is_unsafe;
            }

            for (auto ind = d_loc_safe.begin(); ind < d_loc_safe.end(); ind++) {
                cur_z = z[*ind];
                if (cur_z < min_z) {
                    min_z = cur_z;
                }
                if (cur_z > max_z) {
                    max_z = cur_z;
                }
            }

            // Here we can say if KNOWN UNSAFE
            if (max_z - min_z < HEIGHT_SAFE) {
                goto is_safe;
            }

            // Otherwise we have to punt
            SET_OUTPUT(output, i, j) = 0xff / 2;
            continue;

is_safe:
            SET_OUTPUT(output, i, j) = 0xff;
            continue;

is_unsafe:
            // SET_OUTPUT(output, i, j) = 0x00;
            continue;
        }
    }

    return output;
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
