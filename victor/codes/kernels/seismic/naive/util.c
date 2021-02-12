/*
	Stencil Probe utilities
	Helper functions for the probe.
*/
#include <stdio.h>
#include <stdlib.h>
#include "util.h"
#include "cycle.h"
/////////////////////////////
#include "common.h"
#define NR_END 0







void nrerror(char error_text[])
/* Numerical Recipes standard error handler */
{
        fprintf(stderr,"Numerical Recipes run-time error...\n");
        fprintf(stderr,"%s\n",error_text);
        fprintf(stderr,"...now exiting to system...\n");
        exit(1);
}




float ***d3tensor(long nrl, long nrh, long ncl, long nch, long ndl, long ndh)
/* allocate a float 3tensor with range t[nrl..nrh][ncl..nch][ndl..ndh] */
{
        long i,j,nrow=nrh-nrl+1,ncol=nch-ncl+1,ndep=ndh-ndl+1;
        float ***t;

        /* allocate pointers to pointers to rows */
        t=(float ***) malloc((size_t)((nrow+NR_END)*sizeof(float**)));
        if (!t) nrerror("allocation failure 1 in f3tensor()");
        t += NR_END;
        t -= nrl;

        /* allocate pointers to rows and set pointers to them */
        t[nrl]=(float **) malloc((size_t)((nrow*ncol+NR_END)*sizeof(float*)));
        if (!t[nrl]) nrerror("allocation failure 2 in f3tensor()");
        t[nrl] += NR_END;
        t[nrl] -= ncl;

        /* allocate rows and set pointers to them */
        t[nrl][ncl]=(float *) malloc((size_t)((nrow*ncol*ndep+NR_END)*sizeof(float)));
        if (!t[nrl][ncl]) nrerror("allocation failure 3 in f3tensor()");
        t[nrl][ncl] += NR_END;
        t[nrl][ncl] -= ndl;

        for(j=ncl+1;j<=nch;j++) t[nrl][j]=t[nrl][j-1]+ndep;
        for(i=nrl+1;i<=nrh;i++) {
                t[i]=t[i-1]+ncol;
                t[i][ncl]=t[i-1][ncl]+ncol*ndep;
                for(j=ncl+1;j<=nch;j++) t[i][j]=t[i][j-1]+ndep;
        }

        /* return pointer to array of pointers to rows */
        return t;
}














int IMAX(int a,int b)
{
	if ( a > b ) 
	{
	return a;
	}else{
	return b;
	}
}

/*
  This initializes the array A to be all 1's.  
  This is nearly superfluous (could use memset), but
  provides convenience and consistency nonetheless...
 */
void StencilInit(int nx,int ny,int nz, /* size of the array */
		 float *A){ /* the array to initialize to 1s */
  long last = nx*ny*nz;
  long ii;
  int i,j,k;
   int id;
#if (PINIT)
#pragma omp parallel  default(shared) private(i,j,k)
{
#pragma omp  for
#endif
          for ( k =0; k < nz; k++){
          for ( j = 0; j < ny; j++){
          for ( i = 0; i < nx; i++){

	A[Index3D (nx, ny, i, j, k)] = (i + j + k)*0.0001 ;

}
#if (PINIT)
}
#endif
}

}
}

/*
  Function to clear the cache, preventing data items in cache
  from making subsequent trials run faster.
*/
void clear_cache()
{
  int i;
  float* tarray, accum;

  tarray = (float*) malloc(sizeof(float)*1310720);
  for (i=0,accum=0; i<1310719; i++)
    tarray[i] = 1.0;

}


