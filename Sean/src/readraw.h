/*********************************************************************
 * readraw.h
 ********************************************************************/

#ifndef _READRAW_H
#define _READRAW_H

#define NROWS 500
#define NCOLS 500

std::array<float, NROWS*NCOLS> read_raw(const char *filename);

#endif // _READRAW_H
