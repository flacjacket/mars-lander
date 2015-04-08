/*********************************************************************
 * readraw.h
 ********************************************************************/

#ifndef _READRAW_H
#define _READRAW_H

#include "height_params.h"

void read_raw(const char *filename, std::array<float, NROWS*NCOLS> &data);

#endif // _READRAW_H