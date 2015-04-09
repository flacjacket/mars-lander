#include <fstream>
#include <iterator>
#include <vector>

#include "data_params.h"
#include "error.h"

/*
 * swap the endian-ness of the data
 */
static inline void endian_swap(float *longone) {
    unsigned char temp;
    struct long_bytes {
        unsigned char byte1;
        unsigned char byte2;
        unsigned char byte3;
        unsigned char byte4;
    } *longptr;

    longptr = (struct long_bytes *) longone;

#define SWAP(i, j, tmp) tmp = i; i = j; j = tmp
    SWAP(longptr->byte1, longptr->byte4, temp);
    SWAP(longptr->byte2, longptr->byte3, temp);
#undef SWAP
}

/*
 * read in the raw data
 */
std::vector<float> read_raw(const char *filename, std::vector<float>::size_type size) {
    // Define the vector to return of the requested size
    std::vector<float> data(size);

    // Open the file
    std::ifstream f(filename, std::ios::in | std::ios::binary);

    if (f.is_open()) {
        // Try to read in all the data
        f.read((char*) &data[0], size * sizeof(float));

        // Error handling
        if (!f) {
            f.close();
            error("Error reading, only %d bytes read", f.gcount());
        }
    } else {
        error("(read_raw) Unable to open - %s", filename);
    }

    // Close file
    f.close();

#if IS_BIGENDIAN
	// data is given in big endian, so we need to swap it to little endian
    for (unsigned i = 0; i < size; i++) {
        endian_swap(&data[i]);
    }
    return data;
#endif
}
