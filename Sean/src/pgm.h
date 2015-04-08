/*********************************************************************
 * pgm.h
 *********************************************************************/

#ifndef _PGM_H_
#define _PGM_H_

#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
/**********
 * With pgmReadFile and pgmRead, setting img to NULL causes memory
 * to be allocated
 */

/**********
 * used for reading from/writing to files
 */
unsigned char* pgmReadFile(const char *fname, unsigned char *img, int *ncols, int *nrows);
void pgmWriteFile(const char *fname, unsigned char *img, int ncols, int nrows);

/**********
 * used for communicating with stdin and stdout
 */
unsigned char* pgmRead(FILE *fp, unsigned char *img, int *ncols, int *nrows);
void pgmWrite(FILE *fp, unsigned char *img, int ncols, int nrows);

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif
