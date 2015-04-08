/*********************************************************************
 * preprocess.h
 ********************************************************************/

#ifndef _PREPROCESS_H
#define _PREPROCESS_H

#include <array>

#include "readraw.h"

std::array<unsigned char, 4*NROWS*NCOLS>
preprocess_angle(std::array<float, NROWS*NCOLS> data);

#endif // _PREPROCESS_H
