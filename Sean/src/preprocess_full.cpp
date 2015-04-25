#include <algorithm> // std::fill, std::reverse_copy
#include <cmath>     // atan2, fabs
#include <vector>
#include <openblas/cblas.h>

#include "preprocess_common.h"
#include "data_params.h"

#include <iostream>

#define COS_ATAN(x) (1. / sqrt((x) * (x) + 1))
#define SIN_ATAN(x) ((x) / sqrt((x) * (x) + 1))


std::vector<unsigned char> preprocess_full(std::vector<float> &data) {
    std::vector<unsigned char> output(NROWS*NCOLS);

    float *x_ne = (float*) malloc(12 * ZH * (ZH - 1) * sizeof(float));
    float *x_nw = x_ne + ZH * (ZH - 1);
    float *x_sw = x_ne + 2 * ZH * (ZH - 1);
    float *x_se = x_ne + 3 * ZH * (ZH - 1);

    float *y_ne = x_ne + 4 * ZH * (ZH - 1);
    float *y_nw = x_ne + 5 * ZH * (ZH - 1);
    float *y_sw = x_ne + 6 * ZH * (ZH - 1);
    float *y_se = x_ne + 7 * ZH * (ZH - 1);

    float *z_ne = x_ne + 8 * ZH * (ZH - 1);
    float *z_nw = x_ne + 9 * ZH * (ZH - 1);
    float *z_sw = x_ne + 10 * ZH * (ZH - 1);
    float *z_se = x_ne + 11 * ZH * (ZH - 1);

    float *x_rot_ne = (float*) malloc(12 * ZH * (ZH - 1) * sizeof(float));
    float *z_rot_ne = x_rot_ne + 8 * ZH * (ZH - 1);
    float *z_rot_nw = x_rot_ne + 9 * ZH * (ZH - 1);
    float *z_rot_sw = x_rot_ne + 10 * ZH * (ZH - 1);
    float *z_rot_se = x_rot_ne + 11 * ZH * (ZH - 1);

    std::vector<float> dist_unsafe;
    std::vector<unsigned> dloc_unsafe, dloc_base;

    float max_z, max_angle;
    float cos_theta, sin_theta, cos_phi, sin_phi, cos_chi1, cos_chi2;
    float z1, z2, z3, z4;
    float *z_tilt1, *z_tilt2, *z_primary1, *z_primary2;
    float R[9];

    // Zero the data
    std::fill(output.begin(), output.end(), 0);

    // pre-compute locations given acceptable distances
    // Distances to guarantee unsafe
    footpad_dist_4point(R_BASE - 2 * R_FOOT, R_BASE, dist_unsafe, dloc_unsafe);
    base_loc_4point(0, R_BASE, dloc_base);

    // determine x and y for each quadrant
    int ind = 0;
    for (int i = 0; i < ZH; i++) {
        for (int j = 0; j < ZH - 1; j++) {
            x_ne[ind] = i * SPACING_HEIGHT;
            y_ne[ind] = (j + 1) * SPACING_HEIGHT;

            x_nw[ind] = -(j + 1) * SPACING_HEIGHT;
            y_nw[ind] = i * SPACING_HEIGHT;

            x_sw[ind] = -i * SPACING_HEIGHT;
            y_sw[ind] = -(j + 1) * SPACING_HEIGHT;

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

            // figure out if any cause it to be unsafe
            max_z = max_angle = 0;

            for (auto z_ind = dloc_unsafe.begin(); z_ind < dloc_unsafe.end(); z_ind++) {
                unsigned ind = *z_ind;

                // Compute unsafe tilt
                z1 = z_ne[ind]; z2 = z_nw[ind]; z3 = z_sw[ind]; z4 = z_se[ind];

                // Check for unsafe tilting configuration
                if (z1 + z3 > z2 + z4) {
                    // Midpoint of primary feet
                    //z_mp = (z1 + z3) / 2;
                    // Primary balancing in 1st/3rd quadrants
                    z_primary1 = z_rot_ne + ind;
                    z_primary2 = z_rot_sw + ind;
                    // Tilting happens in 2nd/4th quadrants
                    z_tilt1 = z_rot_nw + ind;
                    z_tilt2 = z_rot_se + ind;
                    // Rotation (about z axis) of primary feet to x-axis
                    cos_theta = COS_ATAN(-y_ne[ind] / x_ne[ind]);
                    sin_theta = SIN_ATAN(-y_ne[ind] / x_ne[ind]);
                    // Rotation (about y axis) of primary landing feet flat
                    cos_phi = COS_ATAN((z1 - z3) / (2 * R_BASE));
                    sin_phi = SIN_ATAN((z1 - z3) / (2 * R_BASE));
                } else {
                    // Midpoint of primary feet
                    //z_mp = (z2 + z4) / 2;
                    // Primary balancing in 2nd/4th quadrants
                    z_primary1 = z_rot_nw + ind;
                    z_primary2 = z_rot_se + ind;
                    // Tilting happens in 1st/3rd quadrants
                    z_tilt1 = z_rot_ne + ind;
                    z_tilt2 = z_rot_sw + ind;
                    // Rotation (about z axis) of primary feet to x-axis
                    cos_theta = COS_ATAN(-y_se[ind] / x_se[ind]);
                    sin_theta = SIN_ATAN(-y_se[ind] / x_se[ind]);
                    // Rotation (about y axis) of primary landing feet flat
                    cos_phi = COS_ATAN((z4 - z2) / (2 * R_BASE));
                    sin_phi = SIN_ATAN((z4 - z2) / (2 * R_BASE));
                }

                R[0] = cos_phi * cos_theta;
                R[1] = -cos_phi * sin_theta;
                R[2] = sin_phi;
                R[3] = sin_theta;
                R[4] = cos_theta;
                R[5] = 0;
                R[6] = -sin_phi * cos_theta;
                R[7] = sin_phi * sin_theta;
                R[8] = cos_phi;

                // x_rot_ne (3 x (4*ZH*(ZH-1)) = R (3x3) * x_ne (3 x (4*ZH*(ZH-1))
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    3, 4 * ZH * (ZH - 1), 3,           // M, N, K
                    1., R, 3,                          // alpha, A, leading dim A
                    x_ne, 4 * ZH * (ZH - 1),           // B, leading dim B
                    0., x_rot_ne, 4 * ZH * (ZH - 1));  // beta, C, leading dim C

                cos_chi1 = COS_ATAN(((*z_primary1 + *z_primary2) / 2  - *z_tilt1) / R_BASE);
                cos_chi2 = COS_ATAN((*z_tilt2 - (*z_primary1 + *z_primary2) / 2) / R_BASE);

                /**************************************************************
                 * Check unsafe tilt
                 *************************************************************/

                // Check tilting each direction
                if (fabs(acos(cos_phi * cos_chi1)) > ANGLE_UNSAFE || fabs(acos(cos_phi * cos_chi2)) > ANGLE_UNSAFE) {
                    goto is_unsafe;
                }

                for (auto base_ind = dloc_base.begin(); base_ind < dloc_base.end(); base_ind++) {
                    if (z_ne[*base_ind] - (*z_primary1 + *z_primary2) / 2 > HEIGHT_UNSAFE + 0.05 || 
                        z_nw[*base_ind] - (*z_primary1 + *z_primary2) / 2 > HEIGHT_UNSAFE + 0.05 ||
                        z_sw[*base_ind] - (*z_primary1 + *z_primary2) / 2 > HEIGHT_UNSAFE + 0.05 ||
                        z_se[*base_ind] - (*z_primary1 + *z_primary2) / 2 > HEIGHT_UNSAFE + 0.05) {
                        goto is_unsafe;
                    }

                    if (z_ne[*base_ind] - (*z_primary1 + *z_primary2) / 2 < max_z) {
                        max_z = z_ne[*base_ind] - (*z_primary1 + *z_primary2) / 2;
                    }
                    if (z_ne[*base_ind] - (*z_primary1 + *z_primary2) / 2 < max_z) {
                        max_z = z_ne[*base_ind] - (*z_primary1 + *z_primary2) / 2;
                    }
                    if (z_ne[*base_ind] - (*z_primary1 + *z_primary2) / 2 < max_z) {
                        max_z = z_ne[*base_ind] - (*z_primary1 + *z_primary2) / 2;
                    }
                    if (z_ne[*base_ind] - (*z_primary1 + *z_primary2) / 2 < max_z) {
                        max_z = z_ne[*base_ind] - (*z_primary1 + *z_primary2) / 2;
                    }
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

            output[NCOLS_HEIGHT*i + j] = FEED_TO_NET;
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
