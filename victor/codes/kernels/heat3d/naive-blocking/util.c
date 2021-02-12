/*
	Stencil Probe utilities
	Helper functions for the probe.
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "util.h"
#include "common.h"
#include "dim3d.h"
#include "math.h"
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
		 float *A,float *B, float *C)

{ /* the array to initialize to 1s */
  int i,j,k,s;
  float val;
#if (PINIT)
#pragma omp parallel  default(shared) private(i,j,k)
{
#pragma omp  for
#endif
          for ( i = 0; i < nx; i++){
          for ( j = 0; j < ny; j++){
	  for ( k = 0; k < nz; k++){

	A[Index3D (nx, ny, k, j, i)] = 0.0f ;
	B[Index3D (nx, ny, k, j, i)] = 0.0f ;
        C[Index3D (nx, ny, k, j, i)] = 2250000.0f*DT*DT;

	}
	}
	}
#if (PINIT)
 }
#endif
        val = 1.0f;
        for(s=5; s>=0; s--)
        {
          for ( i = (nx/2-s); i < (nx/2+s); i++){
          for ( j = (ny/4-s); j < (ny/4+s); j++){
          for ( k = (nz/4-s); k < (nz/4+s); k++){

	    A[Index3D (nx, ny, k, j, i)] = val ;

            }
            }
            }
	 val *= 10.0f;

       }



}


void reference_implementation(float* next,float* prev,float* vel,float* coeff, int nx, int ny, int nz)
{

  int i, j, k, t,wst;
  float value;



	for (i = SIZE_STENCIL; i < (nx - SIZE_STENCIL); i++) {
	for (j = SIZE_STENCIL; j < (ny - SIZE_STENCIL); j++) {
	for (k = SIZE_STENCIL; k < (nz - SIZE_STENCIL); k++) {

	value = 0.0f;
	value += prev[Index3D (nx, ny, k, j, i)];
	for (wst = 1; wst <=SIZE_STENCIL; wst++) 
	{
		value += coeff[wst]*(prev[Index3D (nx, ny, k+wst, j, i)] + prev[Index3D (nx, ny, k-wst, j, i)]);
		value += coeff[wst]*(prev[Index3D (nx, ny, k, j+wst, i)] + prev[Index3D (nx, ny, k, j-wst, i)]);
		value += coeff[wst]*(prev[Index3D (nx, ny, k, j, i-wst)] + prev[Index3D (nx, ny, k, j, i+wst)]);
	}

	next[Index3D (nx, ny, k, j, i)] = 2.0f*prev[Index3D (nx, ny, k, j, i)] - next[Index3D (nx, ny, k, j, i)] + value*vel[Index3D (nx, ny, k-1, j, i)] ;
	}
	}
	}



}//end


bool within_epsilon(float* output, float *reference, const int dimx, const int dimy, const int dimz, const int radius)
{
  bool retval = true;
  int iz,iy,ix;
  float difference;

      for(ix=0; ix<dimx; ix++) {
    	for(iy=0; iy<dimy; iy++) {
  	   for(iz=0; iz<dimz; iz++) {

	if( ix>=radius && ix<(dimx-radius) && iy>=radius && iy<(dimy-radius) && iz>=radius && iz<(dimz-radius) ) {
	  difference = (fabsf(reference[Index3D (dimx, dimy, iz, iy, ix)]) - fabs(output[Index3D (dimx, dimy, iz, iy, ix)]))/(reference[Index3D (dimx, dimy, iz, iy, ix)]) ;

// 0.1 pourcent en valeur relative
	  if( difference > 0.001 ) {
	    retval = false;
            printf("%d %d %d %f \n",iz,iy,ix,difference);
	    printf(" ERROR i-j-k: (%d,%d,%d)\t%f instead of %f\n", ix,iy,iz, reference[Index3D (dimx, dimy, iz, iy, ix)] , output[Index3D (dimx, dimy, iz, iy, ix)]);
	    return false;
	  }
	}
      }
    }
  }
  return retval;
}
















