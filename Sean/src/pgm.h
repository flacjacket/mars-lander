/*********************************************************************
 * pgm.h
 *********************************************************************/

#ifndef _PGM_H_
#define _PGM_H_

/**********
 * used for reading from/writing to files
 */

//template<size_t N>
//void pgmReadFile(const char *fname, std::array<unsigned char, N> &img);

template<std::size_t N>
extern void pgmWriteFile(const char *fname, std::array<unsigned char, N> &img, unsigned int n_cols);


#endif
