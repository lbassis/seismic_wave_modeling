//	Version time skewing 
//	Version Juin 2014
// 	A valider l utilisation avec nx / ny / nz de taille differentes
//	0 - (nx-1)
//	reuse doit etre pair

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

////////////////////////////
int omp_thread_count();
/*
void StencilProbe(float* vx0,float* vy0,float* vz0,float* txx0,float* tyy0,float* tzz0,float* txy0,float* txz0,float* tyz0,
        float* fx,float* fy,float* fz,
	int nx, int ny, int nz, int tx, int ty, int tz, int timesteps,int reuse);
*/
void StencilProbe(
	float* prev,
        float* next,
        float* vel,
	float *coeff,
	int nx, int ny, int nz,
        int tx, int ty, int tz, int timesteps,int reuse);



/********************************************************************/
//uint64_t  start_thd[128], exec_time_thd[128];

int main(int argc,char *argv[])
{

	float* prev;
        float* next;
        float* vel;
	float* ref;
	float *coeff;

  int nx,ny,nz,tx,ty,tz,timesteps,timesteps_input,reuse,flag_z;
  int i,j,k;
  float spt;
  FILE *  fp_in15;  

//	IMBRIC
  int t_2_x,t_2_y;
  FILE *  fp_in16;
  uint64_t start, exec_time;
  float elapsed_time=0.0f,throughput_mpoints=0.0f, mflops=0.0f;
  float normalized_time;

  FILE *  fp1;

#if defined(PAPI)  
int events[2] = {PAPI_L3_TCM, PAPI_L3_TCA}, ret;
unsigned long long values[2];

if (PAPI_num_counters() < 2) {
      fprintf(stderr, "No hardware counters here, or PAPI not supported.\n");
      exit(1);
   }
#endif


/**********************************************************************************************************************************/
 
if (argc <= 6) 
{
    printf("\nUSAGE:\n%s <grid x> <grid y> <grid z> <reuse> <flag_z>  <timesteps>\n", argv[0]);
    return EXIT_FAILURE;
}
  nx = atoi(argv[1]);
  ny = atoi(argv[2]);
  nz = atoi(argv[3]);

//	parametre de skewing dans la direction horizontale
  reuse = atoi(argv[4]);
//	flag_z = 0 Pas de skewing horizontal
//	flag_z = 1 skewing horizontal
  flag_z = atoi(argv[5]);
  timesteps_input = atoi(argv[6]);


/******************	Control **************************/
//	nx / ny / nz / paires
//	reuse == 0
// 2
if (reuse < 1) 
{
  printf("Reuse > 1 afin de garantir calcul successif vitesse / contrainte \n");
  exit(0);
}

if ((timesteps_input%reuse)!=0)
{
  printf("Timesteps doit etre un multiple de Reuse \n");
  exit(0);
}

//	2*SIZE_STENCIL x reuse + 1
  tx = 2*SIZE_STENCIL*reuse+1;
  ty = 2*SIZE_STENCIL*reuse+1;


  if (tx > nx ) 
  {
	printf("Taille du parametre de skewing (2xSIZE_STENCILxreuse+1) : tx=%d nx=%d \n",tx,nx);
	exit(0);
  }
 
  if (tx > nx ) 
  {
	printf("Taille du parametre de skewing (2xSIZE_STENCILxreuse+1) : tx=%d nx=%d \n",tx,nx);
	exit(0);
  }

  if (flag_z == 1 ) tz =  (SIZE_STENCIL)*reuse+1;
  if (flag_z == 0 ) tz = nz;

/******************	Control ****************/
//2
  timesteps =  timesteps_input / (reuse/1);


/**********************************************************************************************************************************/
  
  /* allocate arrays */ 



#if MALLOC_STD

 prev=(float*)malloc(sizeof(float)*nx*ny*nz);
 next=(float*)malloc(sizeof(float)*nx*ny*nz);
 vel=(float*)malloc(sizeof(float)*nx*ny*nz);
#endif

#if MALLOC_OPTI
        float *prev_base = (float*)_mm_malloc( (nx*ny*nz+16+MASK_ALLOC_OFFSET(0 ))*sizeof(float), CACHELINE_BYTES);
        float *next_base = (float*)_mm_malloc( (nx*ny*nz+16+MASK_ALLOC_OFFSET(16))*sizeof(float), CACHELINE_BYTES);
        float *vel_base  = (float*)_mm_malloc( (nx*ny*nz+16+MASK_ALLOC_OFFSET(32))*sizeof(float), CACHELINE_BYTES);

// Align working vectors offsets 
      prev = &prev_base[16 +ALIGN_HALO_FACTOR +MASK_ALLOC_OFFSET(0 )];
      next = &next_base[16 +ALIGN_HALO_FACTOR +MASK_ALLOC_OFFSET(16)];
      vel  = &vel_base [16 +ALIGN_HALO_FACTOR +MASK_ALLOC_OFFSET(32)];

#endif



 coeff=(float*)malloc(sizeof(float)*(SIZE_STENCIL+1));

  
/**********************************************************************************************************************************/



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

    StencilInit(nx,ny,nz,prev,next,vel);

#if defined(OUTPUTINIT)

        fp1 = fopen("output-skew-init", "w");
        for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
        for (k = 0; k < nz; k++) {

        fprintf(fp1, "%d %d %d %e %e %e \n",k,j,i,prev[Index3D (nx, ny, k, j, i)],next[Index3D (nx, ny, k, j, i)],vel[Index3D (nx, ny, k, j, i)]   );
        }
        }
        }
	fclose(fp1);

#endif

//fp1 = fopen("cache-misses-skew.csv", "a");
//fprintf(fp1, "L1, L2, L3\n");

/**********************************************************************************************************************************/
	start = get_time();


#if defined(PAPI)  
if ((ret = PAPI_start_counters(events, 2)) != PAPI_OK) {
      fprintf(stderr, "PAPI failed to start counters: %s\n", PAPI_strerror(ret));
      exit(1);
   }
#endif

	StencilProbe(prev,next,vel,coeff,nx, ny, nz, tx, ty, tz, timesteps,reuse);

#if defined(PAPI)  
if ((ret = PAPI_read_counters(values, 2)) != PAPI_OK) {
     fprintf(stderr, "PAPI failed to read counters: %s\n", PAPI_strerror(ret));
     exit(1);
   }
#endif
	exec_time = diff_time(start, get_time());


/**********************************************************************************************************************************/

 
	elapsed_time = exec_time/1e6;
  	normalized_time = elapsed_time/(reuse*timesteps);   
  	throughput_mpoints = ((nx-2*SIZE_STENCIL)*(ny-2*SIZE_STENCIL)*(nz-2*SIZE_STENCIL))/(normalized_time*1e6f);
  	mflops = (7.0f*SIZE_STENCIL + 2.0f)* throughput_mpoints;



#if defined(PAPI)
  printf("L3_TCA,%llu,L3_TCM,%llu,Time,%f,Gflops,%f\n", values[0], values[1], elapsed_time, mflops/1e3f);
#else
  printf ("%d;%d;%d;%d;%d;%f;" "% \n" PRIu64, nx, ny, nz, timesteps, omp_thread_count(), mflops,exec_time);
#endif

//printf ("%d;%d;%d;%d;%d;%d;%f;" "%" PRIu64, nx, ny, nz, timesteps, omp_thread_count(), reuse, mflops, exec_time);

//         printf("%lld, %lld, %f\n", values[0], values[1], elapsed_time);
   //fprintf(fp1, "%lld, %lld, %lld\n", values[0], values[1], values[2]);
 
 //fclose(fp1);
/*
  	printf("-------------------------------\n");
	printf("#call of pair of stencil(prev/next) :%d \n",timesteps);
	printf("Timesteps=%d - Reuse=%d \n",timesteps,reuse);
	printf ("Threads - time : " "%d " "%" PRIu64 "\n", omp_thread_count(), exec_time);
  	printf("time:       %8.2f sec\n", elapsed_time);
  	printf("throughput: %8.2f MPoints/s\n", throughput_mpoints );
  	printf("flops:      %8.2f GFlops\n", mflops/1e3f );
*/


/*************************************************/

//printf ("Temps total d execution %d " "%" PRIu64, omp_thread_count(), exec_time);
//printf("\n");
/*
for (i=0;i<omp_thread_count();i++)
{
printf ("%d " "%" PRIu64, i, exec_time_thd[i]);
printf("\n");
}
*/
/************************************************/
k=13;
j=15;
i=37;
//printf("\n");
//printf(" k=%d -  j=%d - i=%d - %e - %e \n",k,j,i,prev[Index3D (nx, ny, k, j, i)],next[Index3D (nx, ny, k, j, i)]);

/*****************************************************/
#if defined(OUTPUTFINAL)


	fp1 = fopen("output-skew-final", "w");
        for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
        for (k = 0; k < nz; k++) {

        fprintf(fp1, "%d %d %d %e %e \n",k,j,i,prev[Index3D (nx, ny, k, j, i)],next[Index3D (nx, ny, k, j, i)]);
        }
        }
        }

        fclose(fp1);
#endif















/**********************************************************************************************************************************/
  /* free arrays */
#ifdef MALLOC_STD
 free(prev);
 free(next);
 free(vel);
#endif



}







/*******************************************************************************************************************************************/
int omp_thread_count() 
{
        int n = 0;
    #pragma omp parallel reduction(+:n)
        n += 1;
        return n;
}

