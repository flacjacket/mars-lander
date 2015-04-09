/*********************************************************************
 * pnmio.c
 *
 * Various routines to manipulate PNM files.
 *********************************************************************/


/* Standard includes */
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "error.h"

#define BUFSIZE 80

/*********************************************************************/

static std::string _getNextString(std::ifstream &f) {
    std::string output;
    char line[BUFSIZE];
    std::size_t i;

    do {
        // Read the next line
        f.read(line, BUFSIZE);

        // Find the next newline
        output = std::string(line, BUFSIZE);
        i = output.find("\n");

        if (i == std::string::npos) {
            error("(_getNextString) Unable to find newline");
        }

        // Shrink the string down to the right size and move the read
        output.resize(i);
        f.seekg(i - BUFSIZE + 1, std::ios_base::cur);
    } while (output[0] == '#');

    return output;
}


/*********************************************************************
 * pnmReadHeader
 */


int pnmReadHeader(std::ifstream &f, std::size_t N) {
    int maxval, magic;
    unsigned nrows, ncols;
    std::string line;
	
    // Read magic number
    line = _getNextString(f);
    if (line[0] != 'P') {
        error("(pnmReadHeader) Magic number does not begin with 'P', but with a '%c'", line[0]);
    }
    sscanf(line.c_str(), "P%d", &magic);
	
    // Read size, skipping comments
    line = _getNextString(f);
    sscanf(line.c_str(), "%u %u", &ncols, &nrows);
    // Some sanity checks on the dimension
    if (nrows * ncols != N) {
        error("(pnmReadHeader) The dimensions %d x %d do not give size %d", nrows, ncols, N);
    }
	
    // Read maxval, skipping comments
    line = _getNextString(f);
    maxval = atoi(line.c_str());
    //fread(line, 1, 1, fp); // Read newline which follows maxval
	
    if (maxval != 255) {
        warning("(pnmReadHeader) Maxval is not 255, but %d", maxval);
    }

    return magic;
}


/*********************************************************************
 * pgmReadHeader
 */
 
void pgmReadHeader(std::ifstream &f, std::size_t N) {
    int magic;

    magic = pnmReadHeader(f, N);
    if (magic != 5) {
        error("(pgmReadHeader) Magic number is not 'P5', but 'P%d'", magic);
    }
}


/*********************************************************************
 * pgmRead
 */
 
std::vector<unsigned char> pgmRead(std::ifstream &f, unsigned int nrows, unsigned int ncols) {
    std::vector<unsigned char> img(nrows * ncols);

    // Read header
    pgmReadHeader(f, nrows * ncols);

    // Read binary image data
    f.read((char*) &img[0], nrows * ncols);

    return img;
}


/*********************************************************************
 * pgmReadFile
 */


std::vector<unsigned char> pgmReadFile(const char *fname, unsigned int nrows, unsigned int ncols) {
    std::vector<unsigned char> data;
    // Open file
    std::ifstream f(fname, std::ios::in | std::ios::binary);

    // Read file
    if (f.is_open()) {
        data = pgmRead(f, nrows, ncols);
    } else {
        error("(pgmReadFile) Can't open file named '%s' for reading\n", fname);
    }

    // Close file
    f.close();
    return data;
}


/*********************************************************************
 * pgmWrite
 */


void pgmWrite(std::ofstream &f, std::vector<unsigned char> &img, unsigned int nrows, unsigned int ncols) {
    int i;
    char buf[BUFSIZE];

    // Write header
    i = sprintf(buf, "%u %u\n", nrows, ncols);
    f.write("P5\n", 3);
    f.write(buf, i);
    f.write("255\n", 4);

    // Write binary data
    f.write((const char*) &img[0], nrows * ncols);

    // Check file status
    if (!f) {
        error("(pgmWrite) Error writing data\n");
    }
}


/*********************************************************************
 * pgmWriteFile
 */


void pgmWriteFile(const char *fname, std::vector<unsigned char> &img, unsigned int nrows, unsigned int ncols) {
    // Open file
    std::ofstream f(fname, std::ios::out | std::ios::binary);

    // Write file
    if (f.is_open()) {
        pgmWrite(f, img, nrows, ncols);
    } else {
        error("(pgmWriteFile) Can't open file named '%s' for writing\n", fname);
    }

    // Close file
    f.close();
}