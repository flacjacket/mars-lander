/*********************************************************************
 * pnmio.h
 *********************************************************************/

#ifndef _PNMIO_H_
#define _PNMIO_H_

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
unsigned char* pgmReadFile(
  char *fname,
  unsigned char *img,
  int *ncols, 
  int *nrows);
void pgmWriteFile(
  char *fname,
  unsigned char *img,
  int ncols,
  int nrows);
void ppmWriteFileRGB(
  char *fname,
  unsigned char *redimg,
  unsigned char *greenimg,
  unsigned char *blueimg,
  int ncols,
  int nrows);

int  ppmReadFile(
  char *fname,
  unsigned char **r, 
  unsigned char **g, 
  unsigned char **b,
  int *ncols, 
  int *nrows);
/**********
 * used for communicating with stdin and stdout
 */
unsigned char* pgmRead(
  FILE *fp,
  unsigned char *img,
  int *ncols, int *nrows);
void pgmWrite(
  FILE *fp,
  unsigned char *img,
  int ncols,
  int nrows);
void ppmWrite(
  FILE *fp,
  unsigned char *redimg,
  unsigned char *greenimg,
  unsigned char *blueimg,
  int ncols,
  int nrows);
void pnmReadHeader(
  FILE *fp, 
  int *magic, 
  int *ncols, int *nrows, 
  int *maxval);
int getsubsetPGM(FILE *fp, int fcols, int frows, long head_length, unsigned char *subimg, 
			  int cols, int rows, int start_col, int end_col, int start_row, int end_row);
int write_short_img_pgm(char *filename,  unsigned short *simg, int cols, int rows);
int write_int_img_pgm(char *filename,  int *gx2, int cols, int rows);
int write_float_img_pgm(char *filename, float *gx2, int cols, int rows);
#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif
