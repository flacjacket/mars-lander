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
#define LENGTH 80


/*********************************************************************/

static void _getNextString(FILE *fp, char *line) {
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

void pnmReadHeader(FILE *fp, int *magic, int *ncols, int *nrows, int *maxval) {
    char line[LENGTH];
	
    /* Read magic number */
    _getNextString(fp, line);
    if (line[0] != 'P')
        error("(pnmReadHeader) Magic number does not begin with 'P', but with a '%c'", line[0]);
    sscanf(line, "P%d", magic);
	
    /* Read size, skipping comments */
    _getNextString(fp, line);
    *ncols = atoi(line);
    _getNextString(fp, line);
    *nrows = atoi(line);
    if (*ncols < 0 || *nrows < 0 || *ncols > 900000 || *nrows >900000)
        error("(pnmReadHeader) The dimensions %d x %d are unacceptable", *ncols, *nrows);
	
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

void pgmReadHeader(FILE *fp, int *magic, int *ncols, int *nrows, int *maxval) {
    pnmReadHeader(fp, magic, ncols, nrows, maxval);
    if (*magic != 5)
        error("(pgmReadHeader) Magic number is not 'P5', but 'P%d'", *magic);
}


/*********************************************************************
 * pgmRead
 *
 * NOTE:  If img is NULL, memory is allocated.
 */

unsigned char* pgmRead(FILE *fp, unsigned char *img, int *ncols, int *nrows) {
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
    } else
        ptr = img;

    /* Read binary image data */
    unsigned char *tmpptr = ptr;
    for (i = 0 ; i < *nrows ; i++)  {
        fread(tmpptr, *ncols, 1, fp);
        tmpptr += *ncols;
    }

    return ptr;
}


/*********************************************************************
 * pgmReadFile
 *
 * NOTE:  If img is NULL, memory is allocated.
 */

unsigned char* pgmReadFile(const char *fname, unsigned char *img, int *ncols, int *nrows) {
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

void pgmWrite(FILE *fp, unsigned char *img, int ncols, int nrows) {
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

void pgmWriteFile(const char *fname, unsigned char *img, int ncols, int nrows) {
  FILE *fp;

  /* Open file */
  if ( (fp = fopen(fname, "wb")) == NULL)
    error("(pgmWriteFile) Can't open file named '%s' for writing\n", fname);

  /* Write to file */
  pgmWrite(fp, img, ncols, nrows);

  /* Close file */
  fclose(fp);
}