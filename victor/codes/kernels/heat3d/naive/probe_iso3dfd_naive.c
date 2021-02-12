#include <stdio.h>
#include "common.h"
#include "util.h"
#include "dim3d.h"
#define un 1.0




void ComputeStencil(float* prev,float* next,float* vel,float* coeff, int nx, int ny, int nz, int i, int j, int k)
{

  int wst;
  float value;


        
        value = 0.0f;
        value += prev[Index3D (nx, ny, k, j, i)]*coeff[0];
        for (wst = 1; wst <=SIZE_STENCIL; wst++)
        {

                value += coeff[wst]*(prev[Index3D (nx, ny, k+wst, j, i)] + prev[Index3D (nx, ny, k-wst, j, i)]);
                value += coeff[wst]*(prev[Index3D (nx, ny, k, j+wst, i)] + prev[Index3D (nx, ny, k, j-wst, i)]);
                value += coeff[wst]*(prev[Index3D (nx, ny, k, j, i+wst)] + prev[Index3D (nx, ny, k, j, i-wst)]);

        }


                next[Index3D (nx, ny, k, j, i)] = prev[Index3D (nx, ny, k, j, i)] +  value ;
}



void StencilProbeit(float* prev,float* next,float* vel,float* coeff, int nx, int ny, int nz, int tx, int ty, int tz)
{

  int i, j, k, t,wst;
  float value;


//#private(i,j,k) shared(coeff,prev,next)
#pragma omp  for schedule(runtime)
	for (i = SIZE_STENCIL; i < (nx - SIZE_STENCIL); i++) 
	for (j = SIZE_STENCIL; j < (ny - SIZE_STENCIL); j++) 
	for (k = SIZE_STENCIL; k < (nz - SIZE_STENCIL); k++) 


	ComputeStencil(prev,next,vel,coeff,nx,ny,nz,i,j,k);

}//end


void StencilProbe(float* prev,float* next,float* vel,float* coeff, int nx, int ny, int nz, int tx, int ty, int tz, int timesteps)
{

  int t;


#pragma omp parallel  default(shared) private(t)
{
  for (t = 0; t < timesteps; t++)

{
#ifdef PROUT
	printf("Timesteps %d %d  \n",t,t+1);
#endif
        StencilProbeit(prev,next,vel,coeff,nx,ny,nz,tx,ty,tz);
        StencilProbeit(next,prev,vel,coeff,nx,ny,nz,tx,ty,tz);

}

}//dt
}//end

