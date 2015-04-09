#include <fstream>
#include <iterator>
#include <vector>

#include "data_params.h"
#include "error.h"

static void endian_swap(float *longone)
{
    struct long_bytes {
        char byte1;
        char byte2;
        char byte3;
        char byte4;
    } *longptr;
    unsigned char temp;

    longptr = (struct long_bytes *) longone;
    temp = longptr->byte1;
    longptr->byte1 = longptr->byte4;
    longptr->byte4 = temp;
    temp = longptr->byte2;
    longptr->byte2 = longptr->byte3;
    longptr->byte3 = temp;
}

std::vector<float> read_raw(const char *filename, std::vector<float>::size_type size) {
    std::vector<float> data(size);

    std::ifstream f(filename, std::ios::in | std::ios::binary);

    if (f.is_open()) {
        f.read((char*) &data[0], size * sizeof(float));

        if (!f) {
            f.close();
            error("Error reading, only %d bytes read", f.gcount());
        }
    } else {
        error("(read_raw) Unable to open - %s", filename);
    }

    f.close();

#if IS_BIGENDIAN
	// data is given in big endian, so we need to swap it to little endian
    for (unsigned i = 0; i < size; i++) {
        endian_swap(&data[i]);
    }
    return data;
#endif
}
