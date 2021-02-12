//	version naive
/****************************************/
#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "util.h"
#include "cycle.h"
/* run.h has the run parameters */
#include "run.h"
#include "timer.h"
#include <inttypes.h>


int omp_thread_count();


void StencilProbe(
	float* vx0,
        float* vy0,
        float* vz0,
        float* txx0,
        float* tyy0,
        float* tzz0,
        float* txy0,
        float* txz0,
        float* tyz0,
        float* fx,
        float* fy,
        float* fz,
	int nx, int ny, int nz,
        int tx, int ty, int tz, int timesteps);



int main(int argc,char *argv[])
{

	float* vx0;
        float* vy0;
        float* vz0;
        float* txx0;
        float* tyy0;
        float* tzz0;
        float* txy0;
        float* txz0;
        float* tyz0;
        float* fx;
        float* fy;
        float* fz;

  int nx,ny,nz,tx,ty,tz,timesteps;
  int i,j,k;
 
  float spt;
  int block;

  uint64_t start, exec_time;








  /* parse command line options */
  
  nx = atoi(argv[1]);
  ny = atoi(argv[2]);
  nz = atoi(argv[3]);
  tx = 1;
  ty = 1;
  tz = 1;
  timesteps = atoi(argv[4]);
//  printf("%dx%dx%d, blocking: %dx%dx%d, timesteps: %d\n",
//	 nx,ny,nz,tx,ty,tz,timesteps);
  
  
  
  /* allocate arrays */ 

 vx0=(float*)malloc(sizeof(float)*nx*ny*nz);
 vy0=(float*)malloc(sizeof(float)*nx*ny*nz);
 vz0=(float*)malloc(sizeof(float)*nx*ny*nz);
 txx0=(float*)malloc(sizeof(float)*nx*ny*nz);
 tyy0=(float*)malloc(sizeof(float)*nx*ny*nz);
 tzz0=(float*)malloc(sizeof(float)*nx*ny*nz);
 txy0=(float*)malloc(sizeof(float)*nx*ny*nz);
 txz0=(float*)malloc(sizeof(float)*nx*ny*nz);
 tyz0=(float*)malloc(sizeof(float)*nx*ny*nz);
 fx=(float*)malloc(sizeof(float)*nx*ny*nz);
 fy=(float*)malloc(sizeof(float)*nx*ny*nz);
 fz=(float*)malloc(sizeof(float)*nx*ny*nz);



    /* initialize arrays to all ones */

    StencilInit(nx,ny,nz,vx0);
    StencilInit(nx,ny,nz,vy0);
    StencilInit(nx,ny,nz,vz0);
    StencilInit(nx,ny,nz,txx0);
    StencilInit(nx,ny,nz,tyy0);
    StencilInit(nx,ny,nz,tzz0);
    StencilInit(nx,ny,nz,txy0);
    StencilInit(nx,ny,nz,txz0);
    StencilInit(nx,ny,nz,tyz0);
    StencilInit(nx,ny,nz,fx);
    StencilInit(nx,ny,nz,fy);
    StencilInit(nx,ny,nz,fz);


	 start = get_time();
    
    /* stencil function */ 
    StencilProbe(
	vx0, 
	vy0,
        vz0,
        txx0,
        tyy0,
        tzz0,
        txy0,
        txz0,
        tyz0,
	fx,
	fy,
	fz,
	nx, ny, nz, tx, ty, tz, timesteps);


	exec_time = diff_time(start, get_time());    	


//printf("#call of pair of stencil(prev/next) :%d \n",timesteps/2);
 
printf ("%d;%d;%d;%d;%d;" "%" PRIu64, nx, ny, nz, timesteps, omp_thread_count(), exec_time);
//printf ("%d " "%" PRIu64, omp_thread_count(), exec_time/1000/1000);

/*
for (k = 2; k < nz - 2; k++) {
for (j = 2; j < ny - 2; j++) {
for (i = 2; i < nx - 2; i++) {
*/
#ifdef PROUT
	i=13;
	j=15;
	k=37;
	printf("\n");
       printf("i=%d - j=%d - k=%d - %f - %f \n",i,j,k,vx0[Index3D (nx, ny, i, j, k)],txx0[Index3D (nx, ny, i, j, k)]);
#endif
/*
          }
          }
          }
*/

  
  /* free arrays */
 free(vx0);
 free(vy0);
 free(vz0);
 free(fx);
 free(fy);
 free(fz);
 free(txx0);
 free(tyy0);
 free(tzz0);
 free(txy0);
 free(txz0);
 free(tyz0);




}

int omp_thread_count() {
        int n = 0;
    #pragma omp parallel reduction(+:n)
        n += 1;
        return n;
}
