#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "common.h"
#include "util.h"
#include "dim3d.h"
#define un 1.0






void StencilProbeit(float* prev,float* next,float* vel,float* coeff, int nx, int ny, int nz, int tx, int ty, int tz,int nb_block_x,int nb_block_y,int nb_block_z,int* tab_i1,int* tab_i2,int* tab_j1,int* tab_j2,int* tab_k1,int* tab_k2)

{

  int i, j, k, t,wst;
  float value;
  int i1,i2,j1,j2,k1,k2;
  int ii,jj,kk;


        	const float c0=coeff[0], c1=coeff[1], c2=coeff[2], c3=coeff[3], c4=coeff[4];
        #if ( SIZE_STENCIL == 8 )
		const float c5=coeff[5], c6=coeff[6], c7=coeff[7], c8=coeff[8];
	#endif


#if (BLOCKING1)
#pragma omp  for  private(i,j,k,kk,jj,ii,k1,k2,i1,i2,j1,j2) schedule(runtime)
for (ii=0;ii<nb_block_x;ii++) {
	i1 = SIZE_STENCIL + tx*ii;
	i2 = MIN(i1 + tx,nx-SIZE_STENCIL);
for (jj=0;jj<nb_block_y;jj++) {
	j1 = SIZE_STENCIL + ty*jj;
	j2 = MIN(j1 + ty,ny-SIZE_STENCIL);
for (kk=0;kk<nb_block_z;kk++) {
	k1 = SIZE_STENCIL + tz*kk;
	k2 = MIN(k1 + tz,nz-SIZE_STENCIL);
#endif

#if (BLOCKING2)
#pragma omp  for private(i,j,k,kk,jj,ii,k1,k2,i1,i2,j1,j2) schedule(runtime)
for (ii=0;ii<nb_block_x;ii++) {
for (jj=0;jj<nb_block_y;jj++) {
for (kk=0;kk<nb_block_z;kk++) {
	i1 = tab_i1[ii];
	i2 = tab_i2[ii];
	j1 = tab_j1[jj];
	j2 = tab_j2[jj];
	k1 = tab_k1[kk];
	k2 = tab_k2[kk];

#endif




#ifdef PROUT
	printf("%d %d %d %d %d %d \n",i1,i2,j1,j2,k1,k2);
#endif


	for (i = i1; i < i2; i++) {
	for (j = j1; j < j2; j++) {
	for (k = k1; k < k2; k++) {
						value = 0.0f;
	




#ifdef UNROLL

							value = prev[Index3D (nx, ny, k, j, i)]*c0
									+ c1 * (prev[Index3D (nx, ny, k+1, j, i)] + prev[Index3D (nx, ny, k-1, j, i)])
                                                                        + c1 * (prev[Index3D (nx, ny, k, j+1, i)] + prev[Index3D (nx, ny, k, j-1, i)])
                                                                        + c1 * (prev[Index3D (nx, ny, k, j, i+1)] + prev[Index3D (nx, ny, k, j, i-1)])

;





#else
value += prev[Index3D (nx, ny, k, j, i)]*coeff[0];
	for (wst = 1; wst <=SIZE_STENCIL; wst++) 
	{
	
		value += coeff[wst]*(prev[Index3D (nx, ny, k+wst, j, i)] + prev[Index3D (nx, ny, k-wst, j, i)]);
		value += coeff[wst]*(prev[Index3D (nx, ny, k, j+wst, i)] + prev[Index3D (nx, ny, k, j-wst, i)]);
		value += coeff[wst]*(prev[Index3D (nx, ny, k, j, i+wst)] + prev[Index3D (nx, ny, k, j, i-wst)]);
		
	}
#endif

		next[Index3D (nx, ny, k, j, i)] = prev[Index3D (nx, ny, k, j, i)] + value ;

	}
	}
	}


}
}
}

}//end


void StencilProbe(float* prev,float* next,float* vel,float* coeff, int nx, int ny, int nz, int tx, int ty, int tz, int timesteps)
{

  int t;
  int nb_block_x,nb_block_y,nb_block_z;
  int* tab_k1;
  int *tab_k2;
  int *tab_j1;
  int *tab_j2;
  int *tab_i1;
  int *tab_i2;

  int ii,jj,kk;

//	Definition Block

	 nb_block_z = (nz-2*SIZE_STENCIL) / tz;
	 if (nb_block_z*tz < (nz-2*SIZE_STENCIL-1) ) nb_block_z ++;
	 nb_block_y = (ny-2*SIZE_STENCIL) / ty;
	 if (nb_block_y*ty < (ny-2*SIZE_STENCIL-1) ) nb_block_y ++;
	 nb_block_x = (nx-2*SIZE_STENCIL) / tx;
	 if (nb_block_x*tx < (nx-2*SIZE_STENCIL-1) ) nb_block_x ++;


#if (BLOCKING1)
	 tab_k1=(int*)malloc(sizeof(int)*1);
	 tab_k2=(int*)malloc(sizeof(int)*1);
	 tab_j1=(int*)malloc(sizeof(int)*1);
	 tab_j2=(int*)malloc(sizeof(int)*1);
	 tab_i1=(int*)malloc(sizeof(int)*1);
	 tab_i2=(int*)malloc(sizeof(int)*1);
#endif

#if (BLOCKING2)
	 tab_k1=(int*)malloc(sizeof(int)*nb_block_z);
	 tab_k2=(int*)malloc(sizeof(int)*nb_block_z);
	 tab_j1=(int*)malloc(sizeof(int)*nb_block_y);
	 tab_j2=(int*)malloc(sizeof(int)*nb_block_y);
	 tab_i1=(int*)malloc(sizeof(int)*nb_block_x);
	 tab_i2=(int*)malloc(sizeof(int)*nb_block_x);


for (ii=0;ii<nb_block_x;ii++) {
	tab_i1[ii] =  SIZE_STENCIL + tx*ii;
	tab_i2[ii] = MIN(tab_i1[ii]  + tx,nx-SIZE_STENCIL);
for (jj=0;jj<nb_block_y;jj++) {
	tab_j1[jj] =  SIZE_STENCIL + ty*jj;
	tab_j2[jj] =  MIN(tab_j1[jj]  + ty,ny-SIZE_STENCIL);
for (kk=0;kk<nb_block_z;kk++) {
	tab_k1[kk] =  SIZE_STENCIL + tz*kk;
	tab_k2[kk] =   MIN(tab_k1[kk]  + tz,nz-SIZE_STENCIL);

}
}
}
#endif




#pragma omp parallel  default(shared) private(t)
{
  for (t = 0; t < timesteps; t+=2)

{
#ifdef PROUT
	printf("Timesteps %d %d  \n",t,t+1);
#endif
        StencilProbeit(prev,next,vel,coeff,nx,ny,nz,tx,ty,tz,nb_block_x,nb_block_y,nb_block_z,tab_i1,tab_i2,tab_j1,tab_j2,tab_k1,tab_k2);
        StencilProbeit(next,prev,vel,coeff,nx,ny,nz,tx,ty,tz,nb_block_x,nb_block_y,nb_block_z,tab_i1,tab_i2,tab_j1,tab_j2,tab_k1,tab_k2);

}

}//dt
}//end

