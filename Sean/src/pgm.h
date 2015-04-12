/*********************************************************************
 * pgm.h
 *********************************************************************/

#ifndef _PGM_H_
#define _PGM_H_

/**********
 * used for reading from/writing to files
 */

namespace pgm {
    std::vector<unsigned char> read_file(const char *fname, unsigned int nrows, unsigned int ncols);
    void write_file(const char *fname, std::vector<unsigned char> &img, unsigned int nrows, unsigned int ncols);

    std::vector<unsigned char> read(std::ifstream &f, unsigned int nrows, unsigned int ncols);
    void write(std::ofstream &f, std::vector<unsigned char> &img, unsigned int nrows, unsigned int ncols);
    void read_header(std::ifstream &f, std::size_t N);
    int pnm_read_header(std::ifstream &f, std::size_t N);
}

#endif
