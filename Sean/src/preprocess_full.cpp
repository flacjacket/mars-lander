#include <algorithm> // std::fill, std::reverse_copy
#include <array>
#include <cmath>     // atan2, fabs
#include <vector>
#include <openblas/cblas.h>

#include "preprocess_common.h"
#include "data_params.h"

#define COS_ATAN(x) (1. / sqrt((x) * (x) + 1))


std::vector<unsigned char> preprocess_full(std::vector<float> &data) {
    std::vector<unsigned char> output(NROWS*NCOLS);

    std::array<float, ZH*(ZH-1)> z_nw;
    std::array<float, ZH*(ZH-1)> z_ne;
    std::array<float, ZH*(ZH-1)> z_se;
    std::array<float, ZH*(ZH-1)> z_sw;

    std::vector<float> dist_unsafe; // , dist_safe;
    std::vector<int> dloc_unsafe; // , d_loc_safe;

    double theta, phi1, phi2;
    //double tilt_min, tilt_check;
    float z1, z2, dz1, dz2;

    // Zero the data
    std::fill(output.begin(), output.end(), 0);

    // pre-compute locations given acceptable distances
    // Distances to guarantee unsafe
    footpad_dist_4point(R_BASE - 2 * R_FOOT, R_BASE, dist_unsafe, dloc_unsafe);
    // Distances to guarantee safe
    // TODO
    //footpad_dist_4point(R_BASE - SPACING, R_BASE, dist_safe, dloc_safe);

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
            //auto dist_ind = dist_unsafe.begin();
            //tilt_min = 0.;
            for (auto z_ind = dloc_unsafe.begin(); z_ind < dloc_unsafe.end(); z_ind++) {
                /*
                 * Find guaranteed unsafe
                 * Need to use full diameter to guaratee is unsafe
                 */
                z1 = z_nw[*z_ind];
                dz1 = z_se[*z_ind];
                z2 = z_ne[*z_ind];
                dz2 = z_sw[*z_ind];

                // Check for unsafe tilting configuration
                if (z1 + dz1 / 2 > z2 + dz2 / 2) {
                    // Primary landing feet NW/SE
                    theta = atan(dz1 / (2 * R_BASE));
                    // Midpoint of primary to either secondary
                    phi1 = atan((z2 - (z1 + dz1 / 2)) / R_BASE);
                    phi2 = atan((z2 + dz2 - (z1 + dz1 / 2)) / R_BASE);
                } else {
                    // Primary landing feet NE/SW
                    theta = atan(dz2 / (2 * R_BASE));
                    // Midpoint of primary to either secondary
                    phi1 = atan((z1 - (z2 + dz2 / 2)) / R_BASE);
                    phi2 = atan((z1 + dz1 - (z2 + dz2 / 2)) / R_BASE);
                }

                /**************************************************************
                 * Check unsafe
                 *************************************************************/

                // Check tilting each direction
                if (fabs(acos(cos(theta) * cos(phi1))) > ANGLE_UNSAFE || fabs(acos(cos(theta) * cos(phi2))) > ANGLE_UNSAFE) {
                    goto is_unsafe;
                }

                /**************************************************************
                 * Check safe
                 *************************************************************/

                // If already not safe, continue
                /*if (tilt_min > ANGLE_SAFE) {
                    continue;
                }

                // Check for safe tilting configuration
                if (z1 + dz1 / 2 > z2 + dz2 / 2) {
                    // Primary landing feet NW/SE
                    theta = atan(dz1 / (2 * *dist_long_ind));
                    // Midpoint of primary to either secondary
                    phi1 = atan((z2 - (z1 + dz1 / 2)) / *dist_long_ind);
                    phi2 = atan((z2 + dz2 - (z1 + dz1 / 2)) / *dist_long_ind);
                } else {
                    // Primary landing feet NE/SW
                    theta = atan(dz2 / (2 * *dist_long_ind));
                    // Midpoint of primary to either secondary
                    phi1 = atan((z1 - (z2 + dz2 / 2)) / *dist_long_ind);
                    phi2 = atan((z1 + dz1 - (z2 + dz2 / 2)) / *dist_long_ind);
                }

                tilt_check = std::max(fabs(acos(cos(theta) * cos(phi1))),
                        fabs(acos(cos(theta) * cos(phi2))));

                if (tilt_check > tilt_min) {
                    tilt_min = tilt_check;
                }*/
            }

            /*if (tilt_min < ANGLE_SAFE) {
                goto is_safe;
            }*/

            /******************************************************************
             * Safety unknown
             *****************************************************************/

            output[NCOLS_HEIGHT*i + j] = SAFE;
            continue;

is_safe:
            output[NCOLS_HEIGHT*i + j] = SAFE;
            continue;

is_unsafe:
            // SET_OUTPUT(output, i, j) = UNSAFE;
            continue;
        }
    }

    return output;
}
