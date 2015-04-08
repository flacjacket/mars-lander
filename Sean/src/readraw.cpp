#include <array>
#include <fstream>

#include "readraw.h"
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

std::array<float, NROWS*NCOLS> read_raw(const char *filename) {
    std::array<float, NROWS*NCOLS> data;
    std::ifstream f (filename, std::ios::in | std::ios::binary);

    if (f.is_open()) {
        f.read((char*) &data[0], NROWS*NCOLS*sizeof(float));
        if (!f) {
            f.close();
            error("Error reading, only %d bytes read", f.gcount());
        }
    } else {
        error("Missing file - %s", filename);
    }
    f.close();

	// data is given in big endian, so we need to swap it to little endian
    for (int i = 0; i < NCOLS*NROWS; i++) {
        endian_swap(&data[i]);
    }

    return data;
}
