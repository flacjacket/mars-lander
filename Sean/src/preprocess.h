/*********************************************************************
 * preprocess.h
 ********************************************************************/

#ifndef _PREPROCESS_H
#define _PREPROCESS_H

#include "height_params.h"

void preprocess_angle(std::array<float, NROWS*NCOLS> &data, std::array<unsigned char, 4 * NROWS*NCOLS>& output);

#endif // _PREPROCESS_H
