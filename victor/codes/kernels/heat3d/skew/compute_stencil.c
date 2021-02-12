
#include <sys/time.h>
#include <stdio.h>
#include "common.h"
#include "dim3d.h"
#include "util.h"



void compute_stencil(float* prev,float* next,float* vel,float* coeff,int blockMin_2_x, int  blockMax_2_x, int blockMin_2_y, int  blockMax_2_y, int blockMin_z, int blockMax_z,int nx,int ny, int nz)

{

  int i, j, k, t,wst;
  float value; 

        	const float c0=coeff[0], c1=coeff[1], c2=coeff[2], c3=coeff[3], c4=coeff[4];
        #if ( SIZE_STENCIL == 8 )
		const float c5=coeff[5], c6=coeff[6], c7=coeff[7], c8=coeff[8];
	#endif


for (i=blockMin_2_x; i < blockMax_2_x; i++) {
for (j=blockMin_2_y; j < blockMax_2_y; j++) {
for (k=blockMin_z; k < blockMax_z; k++) {

							value = 0.0f;
#ifdef UNROLL

							value = prev[Index3D (nz, ny, k, j, i)]*c0
									+ c1 * (prev[Index3D (nz, ny, k+1, j, i)] + prev[Index3D (nz, ny, k-1, j, i)])
                                                                        + c1 * (prev[Index3D (nz, ny, k, j+1, i)] + prev[Index3D (nz, ny, k, j-1, i)])
                                                                        + c1 * (prev[Index3D (nz, ny, k, j, i+1)] + prev[Index3D (nz, ny, k, j, i-1)])
;

#else
	value += prev[Index3D (nz, ny, k, j, i)]*coeff[0];
	for (wst = 1; wst <=SIZE_STENCIL; wst++) 
	{

		
	
		value += coeff[wst]*(prev[Index3D (nz, ny, k+wst, j, i)] + prev[Index3D (nz, ny, k-wst, j, i)]);
		value += coeff[wst]*(prev[Index3D (nz, ny, k, j+wst, i)] + prev[Index3D (nz, ny, k, j-wst, i)]);
		value += coeff[wst]*(prev[Index3D (nz, ny, k, j, i+wst)] + prev[Index3D (nz, ny, k, j, i-wst)]);
		
	}
#endif

		next[Index3D (nz, ny, k, j, i)] = prev[Index3D (nz, ny, k, j, i)] + value ;

	}
	}
	}

}	// fin sous-routine




