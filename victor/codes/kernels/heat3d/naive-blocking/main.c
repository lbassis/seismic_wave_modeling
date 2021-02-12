//	version naive
/****************************************/
#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "util.h"
#include "timer.h"
#include <inttypes.h>
#include "dim3d.h"

#if defined(PAPI)
#include "papi.h"
#endif

int omp_thread_count();


void StencilProbe(
	float* prev,
        float* next,
        float* vel,
	float *coeff,
	int nx, int ny, int nz,
        int tx, int ty, int tz, int timesteps);


int main(int argc,char *argv[])
{

	float* prev;
        float* next;
        float* vel;
	float* ref;
	float *coeff;

  int nx,ny,nz,tx,ty,tz,timesteps;
  int i,j,k;
 
  uint64_t start, exec_time;
  float elapsed_time=0.0f,throughput_mpoints=0.0f, mflops=0.0f;
  float normalized_time;


  FILE *  fp1;

if (argc <= 7){
    printf("\nUSAGE:\n%s <grid x> <grid y> <grid z> <block x> <block y> <block z> <timesteps>\n", argv[0]);
    return EXIT_FAILURE;
}

#if defined(PAPI)
int events[2] = {PAPI_L3_TCM, PAPI_L3_TCA}, ret;
unsigned long long values[2];


if (PAPI_num_counters() < 2) {
      fprintf(stderr, "No hardware counters here, or PAPI not supported.\n");
      exit(1);
 }
#endif

  /* parse command line options */
  
  nx = atoi(argv[1]);
  ny = atoi(argv[2]);
  nz = atoi(argv[3]);
  tx = atoi(argv[4]);
  ty = atoi(argv[5]);
  tz = atoi(argv[6]);
  timesteps = atoi(argv[7]);
  
  
  
  /* allocate arrays */ 

 prev=(float*)malloc(sizeof(float)*nx*ny*nz);
 next=(float*)malloc(sizeof(float)*nx*ny*nz);
 vel=(float*)malloc(sizeof(float)*nx*ny*nz);
 coeff=(float*)malloc(sizeof(float)*(SIZE_STENCIL+1));



    /* initialize arrays to all ones */


 if (SIZE_STENCIL == 1 )
	   {
		           coeff[0] = -2.847222222f;
		           coeff[1] = 1.6f;
	     }


   if (SIZE_STENCIL == 2 )
	     {
		             coeff[0] = -2.847222222f;
		             coeff[1] = 1.6f;
		             coeff[2] = -0.2f;
	       }

  if (SIZE_STENCIL == 4 )
  {
	coeff[0] = -2.847222222f;
	coeff[1] = 1.6f;
	coeff[2] = -0.2f;
	coeff[3] = 2.53968e-2;
	coeff[4] = -1.785714e-3;
  }

  if (SIZE_STENCIL == 8 )
  {
	coeff[0] = -3.0548446f;
	coeff[1] =  1.7777778f;
	coeff[2] = -3.1111111e-1;
	coeff[3] =  7.572087e-2;
	coeff[4] = -1.76767677e-2;
	coeff[5] = 3.480962e-3;
 	coeff[6] = -5.180005e-4;
	coeff[7] =  5.074287e-5;
	coeff[8] = -2.42812e-6;
 }

	coeff[0] = (3.0f*coeff[0]) / (DXYZ*DXYZ);
	for (i=1; i<=SIZE_STENCIL;i++)
		coeff[i] = coeff[i] / (DXYZ*DXYZ);


/*************************************************************/
    StencilInit(nx,ny,nz,prev,next,vel);
#if defined(OUTPUTINIT)

        fp1 = fopen("output-naive-init", "w");
        for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
        for (k = 0; k < nz; k++) {
        fprintf(fp1, "%d %d %d %e %e %e \n",k,j,i,prev[Index3D (nx, ny, k, j, i)],next[Index3D (nx, ny, k, j, i)],vel[Index3D (nx, ny, k, j, i)]);
        }
        }
        }
        fclose(fp1);

#endif

//fp1 = fopen("cache-misses-naive-blocking.csv", "a");
//fprintf(fp1, "L1, L2, L3\n");

/***************************************************************/
    start = get_time();
#if defined(PAPI)
if ((ret = PAPI_start_counters(events, 2)) != PAPI_OK) {
      fprintf(stderr, "PAPI failed to start counters: %s\n", PAPI_strerror(ret));
      exit(1);
   }
#endif

    /* stencil function */ 
    StencilProbe(prev,next,vel,coeff,nx, ny, nz, tx, ty, tz, timesteps);

#if defined(PAPI)
if ((ret = PAPI_read_counters(values, 2)) != PAPI_OK) {
     fprintf(stderr, "PAPI failed to read counters: %s\n", PAPI_strerror(ret));
     exit(1);
   }
#endif

    exec_time = diff_time(start, get_time());    	


/******************************************************************/
 
	elapsed_time = exec_time/1e6;
  	normalized_time = elapsed_time/timesteps;   
  	throughput_mpoints = ((nx-2*SIZE_STENCIL)*(ny-2*SIZE_STENCIL)*(nz-2*SIZE_STENCIL))/(normalized_time*1e6f);
  	mflops = (7.0f*SIZE_STENCIL + 2.0f)* throughput_mpoints;


#if defined(PAPI)
  printf("L3_TCM,%llu,L3_TCA,%llu,Time,%f,Gflops,%f\n", values[0], values[1], elapsed_time, mflops/1e3f);
#else
  printf ("%d;%d;%d;%d;%d;%f;" "% \n" PRIu64, nx, ny, nz, timesteps, omp_thread_count(), mflops,exec_time);
#endif


//	printf ("%d;%d;%d;%d;%d;%f;" "%" PRIu64, nx, ny, nz, timesteps, omp_thread_count(),mflops, exec_time);

//printf("\n%lld, %lld, %lld\n", values[0], values[1], values[2]);
   //fprintf(fp1, "%lld, %lld, %lld\n", values[0], values[1], values[2]);
//   printf("%lld, %lld, %f\n", values[0], values[1], elapsed_time);
 
//fclose(fp1);
/*
  	printf("-------------------------------\n");
	printf("#call of pair of stencil(prev/next) :%d \n",timesteps/2);
	printf ("Threads - time : " "%d " "%" PRIu64 "\n", omp_thread_count(), exec_time);
  	printf("time:       %8.2f sec\n", elapsed_time);
  	printf("throughput: %8.2f MPoints/s\n", throughput_mpoints );
  	printf("flops:      %8.2f GFlops\n", mflops/1e3f );
*/
//	printf ("%d " "%" PRIu64, omp_thread_count(), exec_time);

/************************************************/
k=13;
j=15;
i=37;
//printf("\n");
//printf(" k=%d -  j=%d - i=%d - %e - %e \n",k,j,i,prev[Index3D (nx, ny, k, j, i)],next[Index3D (nx, ny, k, j, i)]);

/*****************************************************/




#if defined(OUTPUTFINAL)
	
	fp1 = fopen("output-naive-final", "w");
        for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
        for (k = 0; k < nz; k++) {
	fprintf(fp1, "%d %d %d %e %e \n",k,j,i,prev[Index3D (nx, ny, k, j, i)],next[Index3D (nx, ny, k, j, i)]);
	}
	}
	}
	fclose(fp1);
#endif




  
  /* free arrays */
 free(prev);
 free(next);
 free(vel);

}

int omp_thread_count() {
        int n = 0;
    #pragma omp parallel reduction(+:n)
        n += 1;
        return n;
}
