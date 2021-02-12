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
		 float *A); /* the array to initialize to 1's */



int IMAX(int a,int b);
void clear_cache();

float seconds_per_tick();

//O3D
float staggardv4 (float, float, float, float, float, float,
        float, float, float, float,
        float, float, float, float,
        float, float, float, float );

float staggards4 (float, float, float, float, float, float, float,
        float, float, float, float,
        float, float, float, float,
        float, float, float, float );

float staggardt4 (float, float, float, float, float,
        float, float, float, float,
        float, float, float, float );



#endif
