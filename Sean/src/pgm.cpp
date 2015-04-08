/*********************************************************************
 * pnmio.c
 *
 * Various routines to manipulate PNM files.
 *********************************************************************/


/* Standard includes */
#include <array>
#include <fstream>
#include <cstdio>
#include <cstdlib>  /* malloc(), atoi() */

#include "height_params.h"
#include "error.h"
#define BUFSIZE 80


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
    char line[BUFSIZE];
	
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

template<size_t N>
void pgmReadFile(const char *fname, std::array<unsigned char, N> &img) {
    FILE *fp;

    /* Open file */
    if ( (fp = fopen(fname, "rb")) == NULL)
        error("(pgmReadFile) Can't open file named '%s' for reading\n", fname);

    /* Read file */
    pgmRead(fp, &img[0], 500, 500);

    /* Close file */
    fclose(fp);
}


/*********************************************************************
 * pgmWrite
 */

void pgmWrite(std::ofstream &f, unsigned char *img, int ncols, int nrows) {
  int i;
  char buf[BUFSIZE];

  /* Write header */
  i = sprintf(buf, "%d %d\n", ncols, nrows);
  f.write("P5\n", 3);
  f.write(buf, i);
  f.write("255\n", 4);

  /* Write binary data */
  f.write((const char*)img, ncols * ncols);
  /*for (i = 0 ; i < nrows ; i++)  {
    f.write((const char*)img, ncols);
    img += ncols;
  }*/
}


/*********************************************************************
 * pgmWriteFile
 */

template<std::size_t N>
void pgmWriteFile(const char *fname, std::array<unsigned char, N> &img, unsigned int n_cols) {
    std::ofstream f(fname, std::ios::out | std::ios::binary);

    if (f.is_open()) {
        pgmWrite(f, &img[0], n_cols, N / n_cols);
    } else {
        error("(pgmWriteFile) Can't open file named '%s' for writing\n", fname);
    }

    f.close();
}

template void pgmWriteFile<4*NROWS*NCOLS>(const char *fname,
                                          std::array<unsigned char, 4*NROWS*NCOLS> &img,
                                          unsigned int n_cols);