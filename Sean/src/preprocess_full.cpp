#include <algorithm> // std::fill, std::reverse_copy
#include <cmath>     // atan2, fabs
#include <vector>
#include <openblas/cblas.h>

#include "preprocess_common.h"
#include "data_params.h"

#define COS_ATAN(x) (1. / sqrt((x) * (x) + 1))


std::vector<unsigned char> preprocess_full(std::vector<float> &data) {
    std::vector<unsigned char> output(NROWS*NCOLS);

    float *x_ne = (float*) malloc(12 * ZH * (ZH - 1) * sizeof(float));
    float *x_nw = x_ne + 3 * ZH * (ZH - 1);
    float *x_sw = x_nw + 3 * ZH * (ZH - 1);
    float *x_se = x_sw + 3 * ZH * (ZH - 1);

    float *y_ne = x_ne + ZH * (ZH - 1);
    float *y_nw = x_nw + ZH * (ZH - 1);
    float *y_sw = x_sw + ZH * (ZH - 1);
    float *y_se = x_se + ZH * (ZH - 1);

    float *z_nw = y_nw + ZH * (ZH - 1);
    float *z_ne = y_ne + ZH * (ZH - 1);
    float *z_se = y_se + ZH * (ZH - 1);
    float *z_sw = y_sw + ZH * (ZH - 1);

    float *x_rot_ne = (float*) malloc(12 * ZH * (ZH - 1) * sizeof(float));
    float *x_rot_nw = x_rot_ne + 3 * ZH * (ZH - 1);
    float *x_rot_sw = x_rot_nw + 3 * ZH * (ZH - 1);
    float *x_rot_se = x_rot_sw + 3 * ZH * (ZH - 1);

    std::vector<float> dist_unsafe;
    std::vector<unsigned> dloc_unsafe;

    float x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4;
    float theta, phi;
    float R[9];

    // Zero the data
    std::fill(output.begin(), output.end(), 0);

    // pre-compute locations given acceptable distances
    // Distances to guarantee unsafe
    footpad_dist_4point(R_BASE - 2 * R_FOOT, R_BASE, dist_unsafe, dloc_unsafe);

    // determine x and y for each quadrant
    int ind = 0;
    for (int i = 0; i < ZH; i++) {
        for (int j = 0; j < ZH - 1; j++) {
            x_ne[ind] = j * SPACING_HEIGHT;
            y_ne[ind] = (i + 1) * SPACING_HEIGHT;

            x_nw[ind] = i * SPACING_HEIGHT;
            y_nw[ind] = -(j + 1) * SPACING_HEIGHT;

            x_sw[ind] = -j * SPACING_HEIGHT;
            y_sw[ind] = -(i + 1) * SPACING_HEIGHT;

            x_se[ind] = (j + 1) * SPACING_HEIGHT;
            y_se[ind] = -i * SPACING_HEIGHT;

            ind++;
        }
    }

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
            //cblas_saxpy(ZH*(ZH - 1), -1., &z_nw[0], 1, &z_se[0], 1);
            // z_sw = -1 * z_ne + z_sw
            //cblas_saxpy(ZH*(ZH - 1), -1., &z_ne[0], 1, &z_sw[0], 1);

            // figure out if any cause it to be unsafe
            for (auto z_ind = dloc_unsafe.begin(); z_ind < dloc_unsafe.end(); z_ind++) {
                unsigned ind = *z_ind;

                // Compute unsafe tilt
                x1 = x_ne[ind]; x2 = x_nw[ind]; x3 = x_sw[ind]; x4 = x_se[ind];
                y1 = y_ne[ind]; y2 = y_nw[ind]; y3 = y_sw[ind]; y4 = y_se[ind];
                z1 = z_ne[ind]; z2 = z_nw[ind]; z3 = z_sw[ind]; z4 = z_se[ind];

                // Check for unsafe tilting configuration
                if (z1 + z3 > z2 + z4) {
                    // Primary balancing in 1st/3rd quadrants
                    // Rotation of base to X-axis
                    theta = atan(-y1 / x1);
                    // Primary landing feet NW/SE
                    phi = atan((z3 - z1) / R_BASE);
                    // Midpoint of primary to either secondary
                    //beta1 = atan((z2 - (z1 + dz1 / 2)) / R_BASE);
                    //beta2 = atan((z2 + dz2 - (z1 + dz1 / 2)) / R_BASE);
                } else {
                    // Primary balancing in 2nd/4th quadrants
                    // Rotation of base to X-axis
                    theta = atan(-y4 / x4);
                    // Primary landing feet NE/SW
                    phi = atan((z2 - z4) / R_BASE);
                    // Midpoint of primary to either secondary
                    //beta1 = atan((z1 - (z2 + dz2 / 2)) / R_BASE);
                    //beta2 = atan((z1 + dz1 - (z2 + dz2 / 2)) / R_BASE);
                }

                R[0] = cos(theta);
                R[1] = -sin(theta);
                R[2] = 0;
                R[3] = cos(phi) * sin(theta);
                R[4] = cos(phi) * cos(theta);
                R[5] = -sin(phi);
                R[6] = sin(phi) * sin(theta);
                R[7] = sin(phi) * cos(theta);
                R[8] = cos(phi);

                /**************************************************************
                 * Check unsafe tilt
                 *************************************************************/

                // Check tilting each direction
                /*if (fabs(acos(cos(alpha) * cos(beta1))) > ANGLE_UNSAFE || fabs(acos(cos(alpha) * cos(beta2))) > ANGLE_UNSAFE) {
                    goto is_unsafe;
                }*/

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
            // Output already zeroed
            // output[NCOLS_HEIGHT*i + j] = UNSAFE;
            continue;
        }
    }

    free(x_ne);
    free(x_rot_ne);

    return output;
}
