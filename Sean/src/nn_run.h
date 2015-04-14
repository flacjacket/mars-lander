/*********************************************************************
 * nn_run.h
 ********************************************************************/

#ifndef _NN_RUN_H_
#define _NN_RUN_H_

#include "nn_common.h"

/**********
* used for generating and saveing NN data input
*/

namespace nn {
    int read_layer(const char *fname, std::vector<std::vector<float>> &layer_list, int prev_size);
}

#endif
