#ifndef _DATA_PARAMS_H
#define _DATA_PARAMS_H

// The buffer width on the final output data set
#define BUFFER 20 // actually 21, but for C indexing, say 20

// Radii of the base and the foot
#define R_BASE 1.7
#define R_FOOT 0.25

// PGM data parameters
#define NROWS 1000
#define NCOLS 1000

#define SPACING 0.1

// DEM data parameters
#define NROWS_HEIGHT 500
#define NCOLS_HEIGHT 500

#define SPACING_HEIGHT 0.2

#define IS_BIGENDIAN 1

#endif // _DATA_PARAMS_H
