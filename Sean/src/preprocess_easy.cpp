#include <algorithm> // std::fill, std::reverse_copy
#include <array>
#include <cmath>     // atan2, fabs
#include <vector>

#include "preprocess_common.h"
#include "data_params.h"


/*
 * Only check the heights on the easiest dataset
 */
std::vector<unsigned char> preprocess_easy(std::vector<float> &data) {
    std::vector<unsigned char> output(NROWS_HEIGHT*NCOLS_HEIGHT);
    std::vector<int> d_loc_unsafe, d_loc_safe;

    std::array<float, ZW*ZW> z;
    float cur_z, min_z, max_z;

    // Zero the data
    std::fill(output.begin(), output.end(), 0);

    // Get all of the base locations
    // ...for checking known unsafe
    base_loc(0, R_BASE, d_loc_unsafe);
    // ...for checking known safe
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

            /******************************************************************
             * Check unsafe
             *****************************************************************/

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

            /******************************************************************
             * Check safe
             *****************************************************************/

            for (auto ind = d_loc_safe.begin(); ind < d_loc_safe.end(); ind++) {
                cur_z = z[*ind];
                if (cur_z < min_z) {
                    min_z = cur_z;
                }
                if (cur_z > max_z) {
                    max_z = cur_z;
                }
            }
            if (max_z - min_z < HEIGHT_SAFE) {
                goto is_safe;
            }

            /******************************************************************
             * Safety unknown
             *****************************************************************/
            output[NCOLS_HEIGHT*i + j] = FEED_TO_NET;
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
