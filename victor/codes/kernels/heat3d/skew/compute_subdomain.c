#include <sys/time.h>
#include <stdio.h>
#include "common.h"
#include "util.h"
#include "dim3d.h"
#include "timer.h"
#include <omp.h>


#define MAX(x,y) (x > y ? x : y)
FILE *  fp_in16;  

//uint64_t start_thd[128], exec_time_thd[128];

/****************************************************************************/
void compute_subdomain(
float* prev,float* next,float* vel,float* coeff,
int ii,int jj,
int neg_x_slope,int neg_y_slope,int pos_x_slope,int pos_y_slope,
int tx,int ty,
int t_2_x,int t_2_y,int tz,
int nx,int ny,int nz,
int timesteps,int flag,int flag_x_fin,int flag_y_fin,
int* tab_blockMin_z, int* tab_blockMax_z,int* tab_blockMin_x,int* tab_blockMax_x,int* tab_blockMin_y,int* tab_blockMax_y)

/*******************************************************************************/
{

//extern uint64_t start_thd[128], exec_time_thd[128];

#define un 1.0
int neg_2_x_slope,pos_2_x_slope,ii2,blockMin_2_x,blockMax_2_x;
int inc_2_x,debut_2_x,fin_2_x;
int ref_Min_x,ref_Max_x;
int neg_2_y_slope,pos_2_y_slope,jj2,blockMin_2_y,blockMax_2_y;
int inc_2_y,debut_2_y,fin_2_y;
int ref_Min_y,ref_Max_y;
int kk,neg_z_slope,pos_z_slope;

int id=omp_get_thread_num();

int i, j, k, t,i_z;
int blockMin_x, blockMin_y, blockMin_z;
int blockMax_x, blockMax_y, blockMax_z;








/*******************************************************************************************/
	i_z=0;
	for (kk= SIZE_STENCIL; kk < (nz- SIZE_STENCIL); kk+=tz) {
/*******************************************************************************************/
/*
	neg_z_slope =  SIZE_STENCIL;
	pos_z_slope = -SIZE_STENCIL;
	if (kk ==  SIZE_STENCIL)   neg_z_slope = 0;
	if (kk >= nz-tz-1)   pos_z_slope = 0;
*/


//	On considere reuse pair
/*******************************************************************************************/
	for (t=0; t < timesteps; t++) {
	i_z++;

/*******************************************************************************************/

/*
    blockMin_x = MAX( SIZE_STENCIL, ii + t * neg_x_slope );
  if (flag_x_fin == 1) {
  	blockMax_x = nx- SIZE_STENCIL;
  }else{
	blockMax_x =  MAX( SIZE_STENCIL, ii + (tx-1) + t * pos_x_slope+1) ;
 }


    blockMin_y = MAX( SIZE_STENCIL, jj + t * neg_y_slope   );
  if ( flag_y_fin == 1) {
       	blockMax_y = ny -SIZE_STENCIL;
  }else{
	blockMax_y = MAX(SIZE_STENCIL, jj + (ty -1) + t * pos_y_slope+1);
  }

*/

/*
blockMin_2_x = tab_blockMin_x[t];
blockMax_2_x = tab_blockMax_x[t];
blockMin_2_y = tab_blockMin_y[t];
blockMax_2_y = tab_blockMax_y[t];
*/
/*
blockMin_2_x = blockMin_x;
blockMax_2_x = blockMax_x;
blockMin_2_y = blockMin_y;
blockMax_2_y = blockMax_y;
*/



//	Direction Z
/*
blockMin_z = tab_blockMin_z[i_z];
blockMax_z = tab_blockMax_z[i_z];
*/

/*
  blockMin_z = MAX( SIZE_STENCIL, kk - t * neg_z_slope);
  blockMax_z = MAX( SIZE_STENCIL, kk + tz  + t * pos_z_slope);
  if ( pos_z_slope == 0  ) blockMax_z = nz- SIZE_STENCIL;
*/

#ifdef PROUTALL
	printf("Stencil numero %d  \n",flag);
	printf("ii=%d - t=%d - blockmin_x=%d - blockmax_x=%d \n",ii,t,blockMin_2_x,blockMax_2_y-1);
	printf("jj=%d - t=%d - blockmin_y=%d - blockmax_y=%d \n",jj,t,blockMin_2_y,blockMax_2_y-1);
	printf("kk=%d - t=%d - blockmin_z=%d - blockmax_z=%d \n",kk,t,blockMin_z,blockMax_z-1);
#endif
/*******************************************************************************************/


/*********************************************************************/
	start_thd[id] = get_time();
/************************************************************************/
        if ( (t%2) == 0 )
        {
	compute_stencil(prev,next,vel,coeff,
	tab_blockMin_x[t], tab_blockMax_x[t], tab_blockMin_y[t], tab_blockMax_y[t], tab_blockMin_z[i_z], tab_blockMax_z[i_z], nx, ny, nz);
  	}



        if ( (t%2) != 0 )
        {
	compute_stencil(next,prev,vel,coeff,
	tab_blockMin_x[t], tab_blockMax_x[t], tab_blockMin_y[t], tab_blockMax_y[t], tab_blockMin_z[i_z], tab_blockMax_z[i_z], nx, ny, nz);
	}

/***************************************************************************/
	exec_time_thd[id] += diff_time(start_thd[id], get_time());	
/****************************************************************************/	

}	//	Fin DT
}	//	fin direction  unit-stride

}	// fin sous-routine




