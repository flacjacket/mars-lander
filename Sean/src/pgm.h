/*********************************************************************
 * pgm.h
 *********************************************************************/

#ifndef _PGM_H_
#define _PGM_H_

/**********
 * used for reading from/writing to files
 */

std::vector<unsigned char> pgmReadFile(const char *fname, unsigned int nrows, unsigned int ncols);

void pgmWriteFile(const char *fname, std::vector<unsigned char> &img, unsigned int nrows, unsigned int ncols);

#endif
