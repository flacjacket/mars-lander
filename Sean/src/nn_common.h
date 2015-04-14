/*********************************************************************
 * nn_common.h
 ********************************************************************/

#ifndef _NN_COMMON_H
#define _NN_COMMON_H

// Dimension of window
#define NN_WINDOW 35

// Number of features
#define NN_FEAT (NN_WINDOW * NN_WINDOW)

// How NN input is normalized
#define NORMALIZE(x) ((float) ((x) - 159) / 255.)

#endif
