
//////////////////////////////////////////
//	Noyau de calcul
/*	on calcule de SIZE_STENCIL/2 a nx -  1 - SIZE_STENCIL/2 

        Allocation de 0 a nx - 1 */
///////////////////////////////////////////

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "util.h"
#include <omp.h>
#include "dim3d.h"

#define TABSIZE1 100000
#define TABSIZE2 1000


#define MAX(x,y) (x > y ? x : y)
FILE *  fp_in16;  

void StencilProbe(float* prev,float* next,float* vel,float* coeff,int nx, int ny, int nz, int tx, int ty, int tz, int timesteps, int reuse){
  int t_2_x,t_2_y;
  int ref_ty,ref_tx;
  int neg_x_slope, pos_x_slope;
  int neg_y_slope, pos_y_slope;
  int neg_z_slope, pos_z_slope;
  int blockMin_x, blockMin_y, blockMin_z;
  int blockMax_x, blockMax_y, blockMax_z;
  int flag;
  int flag_x_fin,flag_y_fin;
  int ii, jj, kk, i, j, k;
  int borne_x,borne_y;
  int ii1,jj1;
  int borne_x_o,borne_y_o,borne_x_f,borne_y_f;
  int tx_old,ty_old,stride ;
  void *res;
  int dt;
  float ds;
  float rho,lamx,mux,muy,muz,muxyz;
  int i_z,i_x,i_y,dt2,t;
  int* tab1_i;
  int* tab1_j;
  int* tab2_i;
  int* tab2_j;
  int* tab3_i;
  int* tab3_j;
  int* tab4_i;
  int* tab4_j;
  int* tab_blockMin_z;
  int* tab_blockMax_z;
  int* tab_blockMin_x;
  int* tab_blockMax_x;
  int* tab_blockMin_y;
  int* tab_blockMax_y;
  int* tab_blockMin2_x;
  int* tab_blockMax2_x;
  int* tab_blockMin2_y;
  int* tab_blockMax2_y;
  int* tab_blockMin3_x;
  int* tab_blockMax3_x;
  int* tab_blockMin3_y;
  int* tab_blockMax3_y;
  int* tab_blockMin4_x;
  int* tab_blockMax4_x;
  int* tab_blockMin4_y;
  int* tab_blockMax4_y;
  int tab_mapping[TABSIZE2][TABSIZE2];
  int tab_mapping2[TABSIZE2][TABSIZE2];
  int tab_mapping3[TABSIZE2][TABSIZE2];
  int tab_mapping4[TABSIZE2][TABSIZE2];

  tab_blockMin_z=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMax_z=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMin_x=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMax_x=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMin_y=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMax_y=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMin2_x=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMax2_x=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMin2_y=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMax2_y=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMin3_x=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMax3_x=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMin3_y=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMax3_y=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMin4_x=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMax4_x=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMin4_y=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMax4_y=(int*)malloc(sizeof(int)*TABSIZE1);

/****************************** Allocation des tableaux *****************************************/
  int id;
  ref_tx=tx;
  ref_ty=ty;

/*****************************	Allocation des tableaux ***************************************/
  tab1_i=(int*)malloc(sizeof(int)*TABSIZE1);
  tab1_j=(int*)malloc(sizeof(int)*TABSIZE1);
  tab2_i=(int*)malloc(sizeof(int)*TABSIZE1);
  tab2_j=(int*)malloc(sizeof(int)*TABSIZE1);
  tab3_i=(int*)malloc(sizeof(int)*TABSIZE1);
  tab3_j=(int*)malloc(sizeof(int)*TABSIZE1);
  tab4_i=(int*)malloc(sizeof(int)*TABSIZE1);
  tab4_j=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMin_z=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMax_z=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMin_x=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMax_x=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMin_y=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMax_y=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMin2_x=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMax2_x=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMin2_y=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMax2_y=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMin3_x=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMax3_x=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMin3_y=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMax3_y=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMin4_x=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMax4_x=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMin4_y=(int*)malloc(sizeof(int)*TABSIZE1);
  tab_blockMax4_y=(int*)malloc(sizeof(int)*TABSIZE1);

/****************************** Allocation des tableaux *****************************************/
//	Type 1 : X=flag1 // Y=flag=1 ( X=ferme/ Y=ferme)
//	Type 2 : X=flag1 // Y =flag2 (X=ferme / Y=ouvert)
//	Type 3 : X=flag2 // Y= flag2 (X=ouvert//Y=ferme)
//	Type 4 : X=flag2 // Y=flag2 (X=ouvert//Youvert)
/*******************************************************************************************/
/*	Condition	*/
//	La direction z  est local ( non threadé )
//	Les directions x/y creent les threads
//	On prend 5xtmesteps par securite (c est a priori 4 )
/*
	if ( ( tx < 4*timesteps +1 )  ||  ( ty < 4*timesteps +1 ) || (tz < 2*timesteps +1) )
	{
	printf ("Il faut que la taille initiale du sous domaine permette le skewing dans les directions +/- sur ts les dt \n");
	exit(0);
	}

	printf("%d %d %d \n",tx,timesteps,tx-5*timesteps);
*/
/*	Fin condition	*/
/*******************************************************************************************/

  stride=1;
  borne_x_f = ((nx-2*SIZE_STENCIL)/(tx+1))  ;
  borne_x_o = borne_x_f +1 ;
  borne_y_f = ((ny-2*SIZE_STENCIL)/(ty+1))  ;
  borne_y_o = borne_y_f + 1 ;
  tx_old=tx;
  ty_old=ty;

/*******************************************************************************************/
#ifdef PROUTALL
  printf("X_o=%d//X_f=%d//Y_o=%d//Y_f=%d \n", borne_x_o,borne_x_f,borne_y_o,borne_y_f);
  printf ("tx_old = %d - ty_old=%d \n",tx_old,ty_old);
#endif
/*******************************************************************************************/

  tab1_i[1] = SIZE_STENCIL + 1;
  tab1_j[1] = SIZE_STENCIL + 1;
  for (ii1=2; ii1 <= borne_x_f ; ii1+=1)
        tab1_i[ii1] = tab1_i[ii1-1] + tx + 1;
  for (jj1=2; jj1 <= borne_y_f ; jj1+=1)
        tab1_j[jj1] = tab1_j[jj1-1] + ty + 1;

/*******************************************************************************************/
  tab2_i[1] = SIZE_STENCIL + 1;
  tab2_j[1] = SIZE_STENCIL;
  for (ii1=2; ii1 <= borne_x_f ; ii1+=1)
        tab2_i[ii1] = tab2_i[ii1-1] + tx + 1;
  for (jj1=2; jj1 <= borne_y_o; jj1+=1)
         tab2_j[jj1] = tab2_j[jj1-1] + ty + 1;

/*******************************************************************************************/
  tab3_i[1] = SIZE_STENCIL;
  tab3_j[1] = SIZE_STENCIL + 1;
  for (ii1=2; ii1 <= borne_x_o ; ii1+=1)
	tab3_i[ii1] = tab3_i[ii1-1] + tx + 1;
  for (jj1=2; jj1 <= borne_y_f ; jj1+=1) 
	tab3_j[jj1] = tab3_j[jj1-1] + ty + 1;

/*******************************************************************************************/
  tab4_i[1] =  SIZE_STENCIL;
  tab4_j[1]=   SIZE_STENCIL;
  for (ii1=2; ii1 <= borne_x_o ; ii1+=1)
        tab4_i[ii1] = tab4_i[ii1-1] + tx + 1;
  for (jj1=2; jj1 <= borne_y_o ; jj1+=1)
         tab4_j[jj1] = tab4_j[jj1-1] + ty + 1;
  i_z = 0;
/***************	Borne tuile Z ******************************************************/
/*******************************************************************************************/
  for (kk= SIZE_STENCIL; kk < (nz- SIZE_STENCIL); kk+=tz) {
/*******************************************************************************************/
	neg_z_slope =  SIZE_STENCIL;
	pos_z_slope = -SIZE_STENCIL;
	if (kk ==  SIZE_STENCIL)   neg_z_slope = 0;
	if (kk >= nz-tz-1)   pos_z_slope = 0;
/*******************************************************************************************/
	for (dt=0; dt < reuse; dt++) {
/*******************************************************************************************/
		i_z ++;
//	Direction Z
		tab_blockMin_z[i_z] = MAX( SIZE_STENCIL, kk - dt * neg_z_slope);
		tab_blockMax_z[i_z] = MAX( SIZE_STENCIL, kk + tz  + dt * pos_z_slope);
		if ( pos_z_slope == 0  ) tab_blockMax_z[i_z] = nz-SIZE_STENCIL;
#ifdef PROUTALL
		printf("%d %d %d %d %d \n", tab_blockMin_z[i_z] , tab_blockMax_z[i_z] ,dt,reuse,i_z);
#endif
	}
  }
/****************************************************************************************/

  #pragma omp parallel default(shared) \
  firstprivate(ii,jj,flag_x_fin,flag_y_fin,tab_blockMin_x,tab_blockMax_x,tab_blockMin_y,tab_blockMax_y,tab_blockMin2_x,tab_blockMax2_x,tab_blockMin2_y,\
		  tab_blockMax2_y,tab_blockMin3_x,tab_blockMax3_x,tab_blockMin3_y,tab_blockMax3_y,tab_blockMin4_x) \
  private(dt,neg_x_slope,pos_x_slope,neg_y_slope,pos_y_slope,ii1,jj1,dt2,i_x,i_y,tab_mapping,tab_mapping2,tab_mapping3,tab_mapping4) 
  {
/*******************************************************************************************/
//	timesteps multiple de reuse
  for (dt=0; dt < timesteps; dt+=1) {
/*******************************************************************************************/
#ifdef PROUTALL
#pragma omp single
	printf("dt = %d - timesteps = %d - reuse = %d  \n",dt,timesteps,reuse);
#endif
/*******************************************************************************************/
//	Type 1 -  
//	Ferme
/*****************************************************************************************/
	flag = 1;
	flag_x_fin=0;
	flag_y_fin=0;
	tx=tx_old;
	ty=ty_old;
	pos_x_slope = -SIZE_STENCIL;
	pos_y_slope = -SIZE_STENCIL;
/**********************************************************************************/
//	boucle ii1 // jj1 sans openmp -- partage  memoire
//	calcul MinX[ii1] etc..
/**********************************************************************************/
	i_x = 0;
	i_y = 0;
	if (dt == 0) {
		for (ii1=1; ii1 <= borne_x_f ; ii1+=1) {
			for (jj1=1; jj1 <= borne_y_f ; jj1+=1) {
				neg_x_slope =  SIZE_STENCIL;
				ii = tab1_i[ii1];
				if (ii ==  SIZE_STENCIL)   neg_x_slope = 0;
				neg_y_slope =  SIZE_STENCIL;
				jj = tab1_j[jj1];
				if (jj ==  SIZE_STENCIL)  neg_y_slope = 0;
				tab_mapping[ii1][jj1] = i_x+1;
/*******************************************************************************************/
				for (dt2=0; dt2 < reuse; dt2++) {
					i_x++;
					i_y++;
/*******************************************************************************************/
					tab_blockMin_x[i_x] =  MAX( SIZE_STENCIL, ii + dt2 * neg_x_slope );
					if (flag_x_fin == 1) {
						tab_blockMax_x[i_x] = nx-SIZE_STENCIL;
  					} else {
						tab_blockMax_x[i_x] =  MAX(SIZE_STENCIL, ii + (tx-1) + dt2 * pos_x_slope+1) ;
 					}
					tab_blockMin_y[i_y] = MAX(SIZE_STENCIL, jj + dt2 * neg_y_slope   );
  					if ( flag_y_fin == 1) {
						tab_blockMax_y[i_y] = ny-SIZE_STENCIL;
					} else {
						tab_blockMax_y[i_y] = MAX(SIZE_STENCIL, jj + (ty -1) + dt2 * pos_y_slope+1);
  					}
#ifdef PROUTALL
					printf("ii=%d - i_x=%d - t=%d - blockmin_x=%d - blockmax_x=%d \n",ii,i_x,dt2,tab_blockMin_x[i_x],tab_blockMax_x[i_x]-1);
#endif
				}	// for reuse
			}
		}	//	ii1 - jj1
	}	//	if dt == 0
	i_x = 0;
	i_y = 0;
/*********************************************************************************************/
	#pragma omp for schedule(runtime) collapse(2)
	for (ii1=1; ii1 <= borne_x_f ; ii1+=1) {
      		for (jj1=1; jj1 <= borne_y_f ; jj1+=1) {
			neg_x_slope =  SIZE_STENCIL;
			ii = tab1_i[ii1];
			if (ii ==  SIZE_STENCIL)   neg_x_slope = 0;
			neg_y_slope =  SIZE_STENCIL;
			jj = tab1_j[jj1];
			if (jj ==  SIZE_STENCIL)  neg_y_slope = 0;
//printf("COMPUTE - ii1=%d - jj1=%d - mapping=%d -min=%d Max=%d  \n",ii1,jj1,tab_mapping[ii1][jj1],tab_blockMin_x[tab_mapping[ii1][jj1]],tab_blockMax_x[tab_mapping[ii1][jj1]]);
			compute_subdomain(prev,next,vel,coeff,
					ii,jj,neg_x_slope,neg_y_slope,pos_x_slope,pos_y_slope,
					tx,ty,t_2_x,t_2_y,tz,nx,ny,nz,reuse,flag,flag_x_fin,flag_y_fin,
					tab_blockMin_z,tab_blockMax_z,&tab_blockMin_x[tab_mapping[ii1][jj1]],
					&tab_blockMax_x[tab_mapping[ii1][jj1]],&tab_blockMin_y[tab_mapping[ii1][jj1]],&tab_blockMax_y[tab_mapping[ii1][jj1]]);
		}	//fin block
	}	//fin block

//}	//end OMP
#ifdef PROUTALL
	printf ("Fin part I \n");
#endif
/*******************************************************************************************/
//	Type 2
//	Ferme - Ouvert
/*******************************************************************************************/
	ty=stride;
	flag = 2;
	pos_x_slope = -SIZE_STENCIL;
	pos_y_slope =  SIZE_STENCIL;
	flag_x_fin=0;
/**********************************************************************************/
//	boucle ii1 // jj1 sans openmp -- partage  memoire
//	calcul MinX[ii1] etc..
/**********************************************************************************/
	i_x = 0;
	i_y = 0;
	if (dt == 0) {
		for (ii1=1; ii1 <= borne_x_f ; ii1+=1) {
			for (jj1=1; jj1 <=borne_y_o; jj1+=1) {	
				neg_x_slope =  SIZE_STENCIL;
				ii = tab2_i[ii1];
				if (ii ==  SIZE_STENCIL)   neg_x_slope = 0;
				neg_y_slope = -SIZE_STENCIL;
				jj = tab2_j[jj1];
				if (jj ==  SIZE_STENCIL)  neg_y_slope = 0;
				flag_y_fin=0;
				if (jj1 == borne_y_o)  flag_y_fin=1;
				tab_mapping2[ii1][jj1] = i_x+1;
/*******************************************************************************************/
				for (dt2=0; dt2 < reuse; dt2++) {
					i_x++;
					i_y++;
/*******************************************************************************************/
					tab_blockMin2_x[i_x] =  MAX( SIZE_STENCIL, ii + dt2 * neg_x_slope );
					if (flag_x_fin == 1) {
						tab_blockMax2_x[i_x] = nx-SIZE_STENCIL;
  					} else {
						tab_blockMax2_x[i_x] =  MAX(SIZE_STENCIL, ii + (tx-1) + dt2 * pos_x_slope+1) ;
 					}
					tab_blockMin2_y[i_y] = MAX(SIZE_STENCIL, jj + dt2 * neg_y_slope   );
					if ( flag_y_fin == 1) {
						tab_blockMax2_y[i_y] = ny-SIZE_STENCIL;
					} else {
						tab_blockMax2_y[i_y] = MAX(SIZE_STENCIL, jj + (ty -1) + dt2 * pos_y_slope+1);
  					}

#ifdef PROUTALL
					printf("ii=%d - i_x=%d - t=%d - blockmin_x=%d - blockmax_x=%d \n",ii,i_x,dt2,tab_blockMin2_x[i_x],tab_blockMax2_x[i_x]-1);
#endif
				}	// for reuse
			}
		}	//	ii1 - jj1
	}	//	if dt == 0
	i_x = 0;
	i_y = 0;
/*********************************************************************************************/
	#pragma omp for schedule(runtime) collapse(2)
	for (ii1=1; ii1 <= borne_x_f ; ii1+=1) {
		for (jj1=1; jj1 <=borne_y_o; jj1+=1) {	
			neg_x_slope =  SIZE_STENCIL;
			ii = tab2_i[ii1];
			if (ii ==  SIZE_STENCIL)   neg_x_slope = 0;
			neg_y_slope = -SIZE_STENCIL;
			jj = tab2_j[jj1];
			if (jj ==  SIZE_STENCIL)  neg_y_slope = 0;
			flag_y_fin=0;
			if (jj1 == borne_y_o)  flag_y_fin=1;
			compute_subdomain(prev,next,vel,coeff,
				ii,jj,neg_x_slope,neg_y_slope,pos_x_slope,pos_y_slope,
				tx,ty,t_2_x,t_2_y,tz,nx,ny,nz,reuse,flag,flag_x_fin,flag_y_fin,tab_blockMin_z,tab_blockMax_z,
				&tab_blockMin2_x[tab_mapping2[ii1][jj1]],&tab_blockMax2_x[tab_mapping2[ii1][jj1]],
				&tab_blockMin2_y[tab_mapping2[ii1][jj1]],&tab_blockMax2_y[tab_mapping2[ii1][jj1]]);
		}	//fin block
	}	//fin block
//} // Fin OpenMP
	ty=ty_old;
#ifdef PROUTALL
	printf("Fin part II \n");
#endif
/*******************************************************************************************/
//	Type 3
//	ouvert / ferme
/*******************************************************************************************/
	tx = stride;
	flag = 3;
	neg_x_slope = -SIZE_STENCIL;
	pos_x_slope =  SIZE_STENCIL;
	pos_y_slope = -SIZE_STENCIL;
	flag_y_fin=0;
/**********************************************************************************/
//	boucle ii1 // jj1 sans openmp -- partage  memoire
//	calcul MinX[ii1] etc..
/**********************************************************************************/
	i_x = 0;
	i_y = 0;
	if (dt == 0) {
		for (ii1=1; ii1 <= borne_x_o ; ii1+=1) {
			for (jj1=1; jj1 <= borne_y_f ; jj1+=1) {
				ii=tab3_i[ii1];
				neg_y_slope = SIZE_STENCIL;
				flag_x_fin=0;
				jj=tab3_j[jj1];
				if (jj ==  SIZE_STENCIL)  neg_y_slope = 0;
				if (ii1 == borne_x_o) flag_x_fin=1;
				tab_mapping3[ii1][jj1] = i_x+1;
/*******************************************************************************************/
				for (dt2=0; dt2 < reuse; dt2++) {
					i_x++;
					i_y++;
/*******************************************************************************************/
					tab_blockMin3_x[i_x] =  MAX( SIZE_STENCIL, ii + dt2 * neg_x_slope );
					if (flag_x_fin == 1) {
					  	tab_blockMax3_x[i_x] = nx-SIZE_STENCIL;
					} else {
						tab_blockMax3_x[i_x] =  MAX(SIZE_STENCIL, ii + (tx-1) + dt2 * pos_x_slope+1) ;
 					}
					tab_blockMin3_y[i_y] = MAX(SIZE_STENCIL, jj + dt2 * neg_y_slope   );
					if ( flag_y_fin == 1) {
						tab_blockMax3_y[i_y] = ny-SIZE_STENCIL;
					} else {
						tab_blockMax3_y[i_y] = MAX(SIZE_STENCIL, jj + (ty -1) + dt2 * pos_y_slope+1);
  					}
#ifdef PROUTALL
					printf("ii=%d - i_x=%d - t=%d - blockmin_x=%d - blockmax_x=%d \n",ii,i_x,dt2,tab_blockMin3_x[i_x],tab_blockMax3_x[i_x]-1);
#endif
				}	// for reuse
			}
		}	//	ii1 - jj1
	}	//	if dt == 0
	i_x = 0;
	i_y = 0;
/*********************************************************************************************/
	#pragma omp for schedule(runtime) collapse(2)
	for (ii1=1; ii1 <= borne_x_o ; ii1+=1) {
		for (jj1=1; jj1 <= borne_y_f ; jj1+=1) {
			ii=tab3_i[ii1];
			neg_y_slope = SIZE_STENCIL;
			flag_x_fin=0;
			jj=tab3_j[jj1];
			if (jj ==  SIZE_STENCIL)  neg_y_slope = 0;
			if (ii1 == borne_x_o) flag_x_fin=1;
			compute_subdomain(prev,next,vel,coeff,ii,jj,
					neg_x_slope,neg_y_slope,pos_x_slope,pos_y_slope,tx,ty,t_2_x,t_2_y,tz,
					nx,ny,nz,reuse,flag,flag_x_fin,flag_y_fin,tab_blockMin_z,tab_blockMax_z,
					&tab_blockMin3_x[tab_mapping3[ii1][jj1]],&tab_blockMax3_x[tab_mapping3[ii1][jj1]],
					&tab_blockMin3_y[tab_mapping3[ii1][jj1]],&tab_blockMax3_y[tab_mapping3[ii1][jj1]]);
		}	//fin block
	}	//fin block
//} // fin OpenMP
	tx=tx_old;
#ifdef PROUTALL
	printf("Fin part III \n"),
#endif
/*******************************************************************************************/
//	Type 4
//	ouvert / ouvert
/*******************************************************************************************/
	tx=stride;
	ty=stride;
	neg_x_slope = -SIZE_STENCIL;
	pos_x_slope =  SIZE_STENCIL;
	pos_y_slope =  SIZE_STENCIL;
	flag = 4;
/**********************************************************************************/
//	boucle ii1 // jj1 sans openmp -- partage  memoire
//	calcul MinX[ii1] etc..
/**********************************************************************************/
	i_x = 0;
	i_y = 0;
	if (dt == 0) {
		for (ii1=1; ii1 <=borne_x_o; ii1+=1) {
			for (jj1=1; jj1 <= borne_y_o; jj1+=1) {
				ii = tab4_i[ii1];
				neg_y_slope = -SIZE_STENCIL;
				jj = tab4_j[jj1];
				if (jj ==  SIZE_STENCIL)  neg_y_slope = 0;
				flag_x_fin=0;
				flag_y_fin=0;
				if (ii1 == borne_x_o) flag_x_fin=1;
				if (jj1 == borne_y_o) flag_y_fin=1;
				tab_mapping4[ii1][jj1] = i_x+1;
/*******************************************************************************************/
				for (dt2=0; dt2 < reuse; dt2++) {
					i_x++;
					i_y++;
/*******************************************************************************************/
					tab_blockMin4_x[i_x] =  MAX( SIZE_STENCIL, ii + dt2 * neg_x_slope );
					if (flag_x_fin == 1) {
						tab_blockMax4_x[i_x] = nx-SIZE_STENCIL;
					} else {
						tab_blockMax4_x[i_x] =  MAX(SIZE_STENCIL, ii + (tx-1) + dt2 * pos_x_slope+1) ;
 					}
					tab_blockMin4_y[i_y] = MAX(SIZE_STENCIL, jj + dt2 * neg_y_slope   );
					if ( flag_y_fin == 1) {
						tab_blockMax4_y[i_y] = ny-SIZE_STENCIL;
					} else {
						tab_blockMax4_y[i_y] = MAX(SIZE_STENCIL, jj + (ty -1) + dt2 * pos_y_slope+1);
  					}
#ifdef PROUTALL
					printf("ii=%d - i_x=%d - t=%d - blockmin_x=%d - blockmax_x=%d \n",ii,i_x,dt2,tab_blockMin4_x[i_x],tab_blockMax4_x[i_x]-1);
#endif
				}	// for reuse
			}
		}	//	ii1 - jj1
	}	//	if dt == 0
	i_x = 0;
	i_y = 0;
/*********************************************************************************************/
	#pragma omp for schedule(runtime) collapse(2)
	for (ii1=1; ii1 <=borne_x_o; ii1+=1) {
		for (jj1=1; jj1 <= borne_y_o; jj1+=1) {
			ii = tab4_i[ii1];
			neg_y_slope = -SIZE_STENCIL;
			jj = tab4_j[jj1];
			if (jj ==  SIZE_STENCIL)  neg_y_slope = 0;
			flag_x_fin=0;
			flag_y_fin=0;
			if (ii1 == borne_x_o) flag_x_fin=1;
			if (jj1 == borne_y_o) flag_y_fin=1;
			compute_subdomain(prev,next,vel,coeff,ii,jj,neg_x_slope,neg_y_slope,pos_x_slope,pos_y_slope,
				tx,ty,t_2_x,t_2_y,tz,nx,ny,nz,reuse,flag,flag_x_fin,flag_y_fin,tab_blockMin_z,tab_blockMax_z,
				&tab_blockMin4_x[tab_mapping4[ii1][jj1]],&tab_blockMax4_x[tab_mapping4[ii1][jj1]],
				&tab_blockMin4_y[tab_mapping4[ii1][jj1]],&tab_blockMax4_y[tab_mapping4[ii1][jj1]]);
		}	//fin block
	}	// fin block
//} // end openMP
#ifdef PROUTALL
	printf("fin part IV \n");
#endif
/*******************************************************************************************/
  }//end for dt
  }//end openMP
/*******************************************************************************************/

/*****************************	Allocation des tableaux ***************************************/
  free(tab1_i);
  free(tab1_j);
  free(tab2_i);
  free(tab2_j);
  free(tab3_i);
  free(tab3_j);
  free(tab4_i);
  free(tab4_j);
  free(tab_blockMin_z);
  free(tab_blockMax_z);
  free(tab_blockMin_x);
  free(tab_blockMax_x);
  free(tab_blockMin_y);
  free(tab_blockMax_y);
  free(tab_blockMin2_x);
  free(tab_blockMax2_x);
  free(tab_blockMin2_y);
  free(tab_blockMax2_y);
  free(tab_blockMin3_x);
  free(tab_blockMax3_x);
  free(tab_blockMin3_y);
  free(tab_blockMax3_y);
  free(tab_blockMin4_x);
  free(tab_blockMax4_x);
  free(tab_blockMin4_y);
  free(tab_blockMax4_y);
}	// fin main
