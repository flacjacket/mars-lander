/*********************************************************************
 * pgm.cpp
 *
 * Various routines to manipulate PGM (i.e. PNM) files
 *********************************************************************/

#include <fstream>
#include <string>
#include <vector>

// So Windows C++11 is totally fucked, in that functions like std::stoi and
// std::to_string don't exist, because of something broken with GLIBCXX ??
// I tried including the headers like in the SO link below, but couldn't get it
// working, so I'm punting and using C
// http://stackoverflow.com/questions/8542221/stdstoi-doesnt-exist-in-g-4-6-1-on-mingw
#include <cstdlib>

#include "error.h"

#define BUFSIZE 80

/*********************************************************************/

static std::string _getNextString(std::ifstream &f) {
    std::string output;
    char line[BUFSIZE];
    std::size_t ind;

    do {
        // Read the next line
        f.read(line, BUFSIZE);

        // Find the next newline
        output = std::string(line, BUFSIZE);
        ind = output.find("\n");

        if (ind == std::string::npos) {
            error("(_getNextString) Unable to find newline");
        }

        // Shrink the string down to the right size and move the read
        output.resize(ind);
        f.seekg(ind - BUFSIZE + 1, std::ios_base::cur);
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
    // I HATE WINDOWS
    // magic = std::stoi(line.substr(1, std::string::npos));
    magic = strtol(line.c_str(), NULL, 10);

    // Read size, for both dimensions
    line = _getNextString(f);
    {
        std::size_t ind = line.find(" ");
        if (ind == std::string::npos) {
            error("(pnmReadHeader) Error reading dimension");
        }
        // I HATE WINDOWS
        // ncols = std::stoi(line.substr(0, ind));
        // nrows = std::stoi(line.substr(ind+1, std::string::npos));
        ncols = strtol(line.substr(0, ind).c_str(), NULL, 10);
        nrows = strtol(line.substr(ind+1, std::string::npos).c_str(), NULL, 10);
    }
    // Some sanity checks on the dimension
    if (nrows * ncols != N) {
        error("(pnmReadHeader) The dimensions %d x %d do not give size %d", nrows, ncols, N);
    }

    // Read maxval, skipping comments
    line = _getNextString(f);
    // I HATE WINDOWS
    // maxval = std::stoi(line);
    maxval = strtol(line.c_str(), NULL, 10);

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
        error("(pgmReadFile) Can't open file named '%s' for reading", fname);
    }

    // Close file
    f.close();
    return data;
}


/*********************************************************************
 * pgmWrite
 */


void pgmWrite(std::ofstream &f, std::vector<unsigned char> &img, unsigned int nrows, unsigned int ncols) {
    // "%u %u\n", nrows, ncols
    // I HATE WINDOWS
    // std::string buf = std::to_string(nrows) + " " + std::to_string(ncols) + "\n";
    char buffer [50];
    int buf_size = sprintf(buffer, "%u %u\n", nrows, ncols);

    // Write header
    f.write("P5\n", 3);
    // I HATE WINDOWS
    // f.write(buf.c_str(), buf.size());
    f.write(buffer, buf_size);
    f.write("255\n", 4);

    // Write binary data
    f.write((const char*) &img[0], nrows * ncols);

    // Check file status
    if (!f) {
        error("(pgmWrite) Error writing data");
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
        error("(pgmWriteFile) Can't open file named '%s' for writing", fname);
    }

    // Close file
    f.close();
}
