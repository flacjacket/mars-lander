/*********************************************************************
 * pnmio.c
 *
 * Various routines to manipulate PNM files.
 *********************************************************************/


/* Standard includes */
#include <stdio.h>   /* FILE  */
#include <stdlib.h>  /* malloc(), atoi() */

/* Our includes */
#include "error.h"
#include "pgm.h"
#define LENGTH 80


/*********************************************************************/

static void _getNextString(
  FILE *fp,
  char *line)
{
  int i;

  line[0] = '\0';

  while (line[0] == '\0')  {
    fscanf(fp, "%s", line);
    i = -1;
    do  {
      i++;
      if (line[i] == '#')  {
        line[i] = '\0';
        while (fgetc(fp) != '\n') ;
      }
    }  while (line[i] != '\0');
  }
}


/*********************************************************************
 * pnmReadHeader
 */

void pnmReadHeader(
  FILE *fp, 
  int *magic, 
  int *ncols, int *nrows, 
  int *maxval)
{
  char line[LENGTH];
	
  /* Read magic number */
  _getNextString(fp, line);
  if (line[0] != 'P')
    error("(pnmReadHeader) Magic number does not begin with 'P', "
             "but with a '%c'", line[0]);
  sscanf(line, "P%d", magic);
	
  /* Read size, skipping comments */
  _getNextString(fp, line);
  *ncols = atoi(line);
  _getNextString(fp, line);
  *nrows = atoi(line);
  if (*ncols < 0 || *nrows < 0 || *ncols > 900000 || *nrows >900000)
    error("(pnmReadHeader) The dimensions %d x %d are unacceptable",
             *ncols, *nrows);
	
  /* Read maxval, skipping comments */
  _getNextString(fp, line);
  *maxval = atoi(line);
  fread(line, 1, 1, fp); /* Read newline which follows maxval */
	
  if (*maxval != 255)
    warning("(pnmReadHeader) Maxval is not 255, but %d", *maxval);
}


/*********************************************************************
 * pgmReadHeader
 */

void pgmReadHeader(
  FILE *fp, 
  int *magic, 
  int *ncols, int *nrows, 
  int *maxval)
{
  pnmReadHeader(fp, magic, ncols, nrows, maxval);
  if (*magic != 5)
    error("(pgmReadHeader) Magic number is not 'P5', but 'P%d'", *magic);
}


/*********************************************************************
 * ppmReadHeader
 */

void ppmReadHeader(
  FILE *fp, 
  int *magic, 
  int *ncols, int *nrows, 
  int *maxval)
{
  pnmReadHeader(fp, magic, ncols, nrows, maxval);
  if (*magic != 6)
    error("(ppmReadHeader) Magic number is not 'P6', but 'P%d'", *magic);
}


/*********************************************************************
 * pgmReadHeaderFile
 */

void pgmReadHeaderFile(
  char *fname, 
  int *magic, 
  int *ncols, int *nrows, 
  int *maxval)
{
  FILE *fp;

  /* Open file */
  if ( (fp = fopen(fname, "rb")) == NULL)
    error("(pgmReadHeaderFile) Can't open file named '%s' for reading\n", fname);

  /* Read header */
  pgmReadHeader(fp, magic, ncols, nrows, maxval);

  /* Close file */
  fclose(fp);
}


/*********************************************************************
 * ppmReadHeaderFile
 */

void ppmReadHeaderFile(
  char *fname, 
  int *magic, 
  int *ncols, int *nrows, 
  int *maxval)
{
  FILE *fp;

  /* Open file */
  if ( (fp = fopen(fname, "rb")) == NULL)
    error("(ppmReadHeaderFile) Can't open file named '%s' for reading\n", fname);

  /* Read header */
  ppmReadHeader(fp, magic, ncols, nrows, maxval);

  /* Close file */
  fclose(fp);
}


/*********************************************************************
 * pgmRead
 *
 * NOTE:  If img is NULL, memory is allocated.
 */

unsigned char* pgmRead(
  FILE *fp,
  unsigned char *img,
  int *ncols, int *nrows)
{
  unsigned char *ptr;
  int magic, maxval;
  int i;

  /* Read header */
  pgmReadHeader(fp, &magic, ncols, nrows, &maxval);

  /* Allocate memory, if necessary, and set pointer */
  if (img == NULL)  {
    ptr = (unsigned char *) malloc(*ncols * *nrows * sizeof(char));
    if (ptr == NULL)  
      error("(pgmRead) Memory not allocated");
  }
  else
    ptr = img;

  /* Read binary image data */
  {
    unsigned char *tmpptr = ptr;
    for (i = 0 ; i < *nrows ; i++)  {
      fread(tmpptr, *ncols, 1, fp);
      tmpptr += *ncols;
    }
  }

  return ptr;
}


/*********************************************************************
 * pgmReadFile
 *
 * NOTE:  If img is NULL, memory is allocated.
 */

unsigned char* pgmReadFile(
  char *fname,
  unsigned char *img,
  int *ncols, int *nrows)
{
  unsigned char *ptr;
  FILE *fp;

  /* Open file */
  if ( (fp = fopen(fname, "rb")) == NULL)
    error("(pgmReadFile) Can't open file named '%s' for reading\n", fname);

  /* Read file */
  ptr = pgmRead(fp, img, ncols, nrows);

  /* Close file */
  fclose(fp);

  return ptr;
}


/*********************************************************************
 * pgmWrite
 */

void pgmWrite(
  FILE *fp,
  unsigned char *img, 
  int ncols, 
  int nrows)
{
  int i;

  /* Write header */
  fprintf(fp, "P5\n");
  fprintf(fp, "%d %d\n", ncols, nrows);
  fprintf(fp, "255\n");

  /* Write binary data */
  for (i = 0 ; i < nrows ; i++)  {
    fwrite(img, ncols, 1, fp);
    img += ncols;
  }
}


/*********************************************************************
 * pgmWriteFile
 */

void pgmWriteFile(
  char *fname, 
  unsigned char *img, 
  int ncols, 
  int nrows)
{
  FILE *fp;

  /* Open file */
  if ( (fp = fopen(fname, "wb")) == NULL)
    error("(pgmWriteFile) Can't open file named '%s' for writing\n", fname);

  /* Write to file */
  pgmWrite(fp, img, ncols, nrows);

  /* Close file */
  fclose(fp);
}


/*********************************************************************
 * ppmWrite
 */

void ppmWrite(
  FILE *fp,
  unsigned char *redimg,
  unsigned char *greenimg,
  unsigned char *blueimg,
  int ncols, 
  int nrows)
{
  int i, j;

  /* Write header */
  fprintf(fp, "P6\n");
  fprintf(fp, "%d %d\n", ncols, nrows);
  fprintf(fp, "255\n");

  /* Write binary data */
  for (j = 0 ; j < nrows ; j++)  {
    for (i = 0 ; i < ncols ; i++)  {
      fwrite(redimg, 1, 1, fp); 
      fwrite(greenimg, 1, 1, fp);
      fwrite(blueimg, 1, 1, fp);
      redimg++;  greenimg++;  blueimg++;
    }
  }
}


/*********************************************************************
 * ppmWriteFileRGB
 */

void ppmWriteFileRGB(
  char *fname, 
  unsigned char *redimg,
  unsigned char *greenimg,
  unsigned char *blueimg,
  int ncols, 
  int nrows)
{
  FILE *fp;

  /* Open file */
  if ( (fp = fopen(fname, "wb")) == NULL)
    error("(ppmWriteFileRGB) Can't open file named '%s' for writing\n", fname);

  /* Write to file */
  ppmWrite(fp, redimg, greenimg, blueimg, ncols, nrows);

  /* Close file */
  fclose(fp);
}


/******************************************************************************
********************************   PPM_READ   *********************************
*******************************************************************************

    This function reads in a PPM or PGM image from a file. It allocates the
    memory internally to hold the image. It is the caller's responsibility to
    free the memory when it is no longer needed. Note that all of the image
    bands will be placed into a single allocated block; free only the red
    pointer. Note that if the input image is monochrome, then all three
    pointers will point to the same address. This function returns 0 for
    success and -1 for failure. */

int  ppmReadFile(
  char *fname,
  unsigned char **r, 
  unsigned char **g, 
  unsigned char **b,
  int *ncols, 
  int *nrows)
{
    FILE *fp;
	unsigned char *rpt, *gpt, *bpt;
    int magic, npixels, maxval, i;
	unsigned char *rgb, *rgbpt;
  /* Open file */
  if ( (fp = fopen(fname, "rb")) == NULL)
    error("(ppmWriteFileRGB) Can't open file named '%s' for readingng\n", fname);

  ppmReadHeader(fp, &magic, ncols, nrows, &maxval);
  
  npixels = *nrows * *ncols;
    
  if(*r == NULL)
  {
     if ((*r = (unsigned char *)malloc(npixels*3)) == NULL) {
	 error("Error allocating %d bytes for PGM file: %s\n", npixels, fname);
	 }
	 *g = *r + npixels;
	 *b = *g + npixels;
  }
  
  rgb = (unsigned char *)malloc(npixels*3);
  fread(rgb, npixels*3, 1, fp);
  	rpt = *r;
	gpt = *g;
	bpt = *b;
    rgbpt = rgb;
  for(i = 0; i < npixels; ++i)
  {
     *(rpt++)=*(rgbpt++);
	 *(gpt++)=*(rgbpt++);
	 *(bpt++)=*(rgbpt++);
  }
  free(rgb);
  fclose(fp);
  return 1;
}


int getsubsetPGM(FILE *fp, int fcols, int frows, long head_length, unsigned char *subimg, 
			  int cols, int rows, int start_col, int end_col, int start_row, int end_row)
{	
	int i , j;
	long offset;
	char *onerow;
	onerow = (  char *)malloc(sizeof(char)*fcols);
	offset = head_length + fcols*start_row;
	fseek(fp, offset, SEEK_SET);
	for(i = 0; i < rows; ++i)
	{
		fread(onerow, sizeof(char), fcols, fp);
		for(j = 0; j < cols; ++j)
		{
			subimg[i*cols +j] = onerow[start_col +j];
		}
	}
	free(onerow);
	return 1;
}

int write_short_img_pgm(char *filename,  unsigned short *simg, int cols, int rows)
{
	unsigned char *img;
	int i, j;
	int imax, imin;
	imax =0;
	imin = 1<<15;
	img = (unsigned char *)malloc(sizeof(char)*cols*rows);
	for(i = 0; i < cols*rows; ++i)
	{
		if(simg[i] > imax) imax = simg[i];
		
		if(simg[i] < imin) imin = simg[i];
	}
	for(i = 0; i < cols*rows; ++i)
	{
		img[i] = (simg[i] - imin)*250/(imax-imin);
		//if(abs(gx2[i]) < 1000)
		//{
			//img[i] = 0;
		//}
	}
	pgmWriteFile(filename, img, cols, rows);
	free(img);
	return 1;
}


int write_int_float_pgm(char *filename,  float *gx2, int cols, int rows)
{
	unsigned char *img;
	int i, j;
	float imax, imin;
	imax =0;
	imin =100000000;
	img = (unsigned char *)malloc(sizeof(char)*cols*rows);
	for(i = 0; i < cols*rows; ++i)
	{
		if(gx2[i] > imax) imax = gx2[i];
		
		if(gx2[i] < imin ) imin = gx2[i];
	}
	imin = 0;

	for(i = 0; i < cols*rows; ++i)
	{
		img[i] = (gx2[i] - imin)*250/(imax-imin);
		//if(abs(gx2[i]) < 1000)
		//{
			//img[i] = 0;
		//}
	}
	pgmWriteFile(filename, img, cols, rows);
	free(img);
	return 1;
}


int write_int_img_pgm(char *filename,  int *gx2, int cols, int rows)
{
	unsigned char *img;
	int i, j;
	int imax, imin;
	imax =0;
	imin =100000000;
	img = (unsigned char *)malloc(sizeof(char)*cols*rows);
	for(i = 0; i < cols*rows; ++i)
	{
		if(gx2[i] > imax) imax = gx2[i];
		
		if(gx2[i] < imin && gx2[i] != 0) imin = gx2[i];
	}
	for(i = 0; i < cols*rows; ++i)
	{
		img[i] = (gx2[i] - imin)*250/(imax-imin);
		//if(abs(gx2[i]) < 1000)
		//{
			//img[i] = 0;
		//}
	}
	pgmWriteFile(filename, img, cols, rows);
	free(img);
	return 1;
}
