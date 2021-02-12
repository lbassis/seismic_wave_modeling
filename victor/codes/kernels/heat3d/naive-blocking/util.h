#ifndef _PROBE_H_
#define _PROBE_H_

#include "common.h"


void nrerror(char error_text[]);

float ***d3tensor(long nrl, long nrh, long ncl, long nch, long ndl, long ndh);

/*
  This initializes the array A to be all 1's.  
  This is nearly superfluous (could use memset), but
  provides convenience and consistency nonetheless...
 */
void StencilInit(int nx,int ny,int nz, /* size of the array */
		 float *A,float *B,float *C); /* the array to initialize to 1's */


#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define FINITE_ADD(nx,ny,k,j,i, wst) ((prev[Index3D (nx, ny, k+wst, j, i)] + prev[Index3D (nx, ny, k-wst, j, i)]))

int IMAX(int a,int b);

float seconds_per_tick();

#endif
