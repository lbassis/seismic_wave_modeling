#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "../include/nrutil.h"
#include "../include/struct.h"
#include "../include/inlineFunctions.h"
#include "../include/options.h"
#include "../include/alloAndInit.h"
#include "../include/new_allocation.h"

int allocateVelocities(struct VELOCITY **velocities, struct PARAMETERS PRM) {

  const int XMIN = PRM.xMin;
  const int XMAX = PRM.xMax;
  const int YMIN = PRM.yMin;
  const int YMAX = PRM.yMax;
  const int ZMIN = PRM.zMin;
  const int ZMAX = PRM.zMax;
  const int ZMAX0 = PRM.zMax0;
  const int DELTA = PRM.delta;
  const int BLOCK_SIZE = PRM.block_size;

  int i, j;
  int nb_blocks_x = ceil((float)(XMAX - XMIN)/BLOCK_SIZE);
  int nb_blocks_y = ceil((float)(YMAX - YMIN)/BLOCK_SIZE);

  *velocities = malloc(nb_blocks_x * nb_blocks_y * sizeof(struct VELOCITY)); // vetor de velocities

  for (i = 0; i < nb_blocks_y; i++) {
    for (j = 0; j < nb_blocks_x; j++) {
      (*velocities)[i*nb_blocks_x + j].x = myd3tensor0(-1, BLOCK_SIZE + 2, -1, BLOCK_SIZE + 2, ZMIN - DELTA, ZMAX0);
      (*velocities)[i*nb_blocks_x + j].z = myd3tensor0(-1, BLOCK_SIZE + 2, -1, BLOCK_SIZE + 2, ZMIN - DELTA, ZMAX0);
      (*velocities)[i*nb_blocks_x + j].y = myd3tensor0(-1, BLOCK_SIZE + 2, -1, BLOCK_SIZE + 2, ZMIN - DELTA, ZMAX0);
    }
  }

  printf("o total eh %d e foi ate o indice %d\n", nb_blocks_x*nb_blocks_y, (i-1)*nb_blocks_x+j-1);
  return (EXIT_SUCCESS);
}

int allocateSources(struct SOURCE **sources, struct PARAMETERS PRM) {

  const int XMIN = PRM.xMin;
  const int XMAX = PRM.xMax;
  const int YMIN = PRM.yMin;
  const int YMAX = PRM.yMax;
  const int ZMIN = PRM.zMin;
  const int ZMAX = PRM.zMax;
  const int ZMAX0 = PRM.zMax0;
  const int DELTA = PRM.delta;
  const int BLOCK_SIZE = PRM.block_size;

  int i, j;
  int nb_blocks_x = ceil((float)(XMAX - XMIN)/BLOCK_SIZE);
  int nb_blocks_y = ceil((float)(YMAX - YMIN)/BLOCK_SIZE);

  *sources = malloc(nb_blocks_x * nb_blocks_y * sizeof(struct SOURCE));

  for (i = 0; i < nb_blocks_y; i++) {
    for (j = 0; j < nb_blocks_x; j++) {
      ReadSrc(&((*sources)[i*nb_blocks_x + j]), PRM);
      (*sources)[i*nb_blocks_x + j].fx = myd3tensor0(1, BLOCK_SIZE, 1, BLOCK_SIZE, ZMIN - DELTA, ZMAX0);
      (*sources)[i*nb_blocks_x + j].fy = myd3tensor0(1, BLOCK_SIZE, 1, BLOCK_SIZE, ZMIN - DELTA, ZMAX0);
      (*sources)[i*nb_blocks_x + j].fz = myd3tensor0(1, BLOCK_SIZE, 1, BLOCK_SIZE, ZMIN - DELTA, ZMAX0);
    }
  }

  return (EXIT_SUCCESS);
}

int allocateStress(struct STRESS **stresses, struct PARAMETERS PRM) {

  const int XMIN = PRM.xMin;
  const int XMAX = PRM.xMax;
  const int YMIN = PRM.yMin;
  const int YMAX = PRM.yMax;
  const int ZMIN = PRM.zMin;
  const int ZMAX = PRM.zMax;
  const int ZMAX0 = PRM.zMax0;
  const int DELTA = PRM.delta;
  const int BLOCK_SIZE = PRM.block_size;

  int i, j;
  int nb_blocks_x = ceil((float)(XMAX - XMIN)/BLOCK_SIZE);
  int nb_blocks_y = ceil((float)(YMAX - YMIN)/BLOCK_SIZE);

  *stresses = malloc(nb_blocks_x * nb_blocks_y * sizeof(struct STRESS));

  for (i = 0; i < nb_blocks_y; i++) {
    for (j = 0; j < nb_blocks_x; j++) {
      (*stresses)[i*nb_blocks_x + j].xx = myd3tensor0(-1, BLOCK_SIZE + 2, -1, BLOCK_SIZE + 2, ZMIN - DELTA, ZMAX0);
      (*stresses)[i*nb_blocks_x + j].yy = myd3tensor0(-1, BLOCK_SIZE + 2, -1, BLOCK_SIZE + 2, ZMIN - DELTA, ZMAX0);
      (*stresses)[i*nb_blocks_x + j].zz = myd3tensor0(-1, BLOCK_SIZE + 2, -1, BLOCK_SIZE + 2, ZMIN - DELTA, ZMAX0);
      (*stresses)[i*nb_blocks_x + j].xy = myd3tensor0(-1, BLOCK_SIZE + 2, -1, BLOCK_SIZE + 2, ZMIN - DELTA, ZMAX0);
      (*stresses)[i*nb_blocks_x + j].xz = myd3tensor0(-1, BLOCK_SIZE + 2, -1, BLOCK_SIZE + 2, ZMIN - DELTA, ZMAX0);
      (*stresses)[i*nb_blocks_x + j].yz = myd3tensor0(-1, BLOCK_SIZE + 2, -1, BLOCK_SIZE + 2, ZMIN - DELTA, ZMAX0);
    }
  }

  return (EXIT_SUCCESS);
}

int allocateABC(struct ABSORBING_BOUNDARY_CONDITION **ABCs, struct PARAMETERS PRM) {

  const int XMIN = PRM.xMin;
  const int XMAX = PRM.xMax;
  const int YMIN = PRM.yMin;
  const int YMAX = PRM.yMax;
  const int ZMIN = PRM.zMin;
  const int ZMAX = PRM.zMax;
  const int ZMAX0 = PRM.zMax0;
  const int DELTA = PRM.delta;
  const int BLOCK_SIZE = PRM.block_size;

  /* others */
  int i, i2, j, j2, k, imp, jmp, block_index;
  int nb_blocks_x = ceil((float)(XMAX - XMIN + 2 * DELTA + 3)/BLOCK_SIZE);
  int nb_blocks_y = ceil((float)(YMAX - YMIN + 2 * DELTA + 3)/BLOCK_SIZE);
  enum typePlace place;

  *ABCs = malloc(nb_blocks_x * nb_blocks_y * sizeof(struct ABSORBING_BOUNDARY_CONDITION));

  for (i = 0; i < nb_blocks_y; i++) {
    for (j = 0; j < nb_blocks_x; j++) {
      block_index = i*nb_blocks_x + j;

      /* Velocity */
      (*ABCs)[block_index].nPower = NPOWER;
      (*ABCs)[block_index].npmlv = 0;
      (*ABCs)[block_index].ipml = i3tensor(-1, BLOCK_SIZE + 2, -1, BLOCK_SIZE + 2, ZMIN - DELTA, ZMAX0);

      for (imp = -1; imp <= BLOCK_SIZE + 2; imp++) {
	i2 = PRM.imp2i_array[imp];
	for (jmp = -1; jmp <= BLOCK_SIZE + 2; jmp++) {
	  j2 = PRM.jmp2j_array[jmp];
	  for (k = ZMIN - DELTA; k <= ZMAX0; k++) {
	    (*ABCs)[block_index].ipml[imp][jmp][k] = -1;

	    place = WhereAmI(i2, j2, k, PRM);
	    if (place == ABSORBINGLAYER || place == FREEABS) {
	      (*ABCs)[block_index].npmlv += 1;
	      (*ABCs)[block_index].ipml[imp][jmp][k] = (*ABCs)[block_index].npmlv;
	    }
	  }
	}
      }
      //printf("\nNumber of points in the CPML : %li\n", (*ABCs)[block_index].npmlv);

      (*ABCs)[block_index].phivxx = mydvector0(1, (*ABCs)[block_index].npmlv);
      (*ABCs)[block_index].phivxy = mydvector0(1, (*ABCs)[block_index].npmlv);
      (*ABCs)[block_index].phivxz = mydvector0(1, (*ABCs)[block_index].npmlv);
      (*ABCs)[block_index].phivyx = mydvector0(1, (*ABCs)[block_index].npmlv);
      (*ABCs)[block_index].phivyy = mydvector0(1, (*ABCs)[block_index].npmlv);
      (*ABCs)[block_index].phivyz = mydvector0(1, (*ABCs)[block_index].npmlv);
      (*ABCs)[block_index].phivzx = mydvector0(1, (*ABCs)[block_index].npmlv);
      (*ABCs)[block_index].phivzy = mydvector0(1, (*ABCs)[block_index].npmlv);
      (*ABCs)[block_index].phivzz = mydvector0(1, (*ABCs)[block_index].npmlv);

      /* Stress */
      (*ABCs)[block_index].npmlt = (*ABCs)[block_index].npmlv;

      (*ABCs)[block_index].phitxxx = mydvector0(1, (*ABCs)[block_index].npmlt);
      (*ABCs)[block_index].phitxyy = mydvector0(1, (*ABCs)[block_index].npmlt);
      (*ABCs)[block_index].phitxzz = mydvector0(1, (*ABCs)[block_index].npmlt);
      (*ABCs)[block_index].phitxyx = mydvector0(1, (*ABCs)[block_index].npmlt);
      (*ABCs)[block_index].phityyy = mydvector0(1, (*ABCs)[block_index].npmlt);
      (*ABCs)[block_index].phityzz = mydvector0(1, (*ABCs)[block_index].npmlt);
      (*ABCs)[block_index].phitxzx = mydvector0(1, (*ABCs)[block_index].npmlt);
      (*ABCs)[block_index].phityzy = mydvector0(1, (*ABCs)[block_index].npmlt);
      (*ABCs)[block_index].phitzzz = mydvector0(1, (*ABCs)[block_index].npmlt);
    }
  }
  return (EXIT_SUCCESS);
}

int allocateMedium(struct MEDIUM *MDM, struct PARAMETERS PRM) {

  const int ZMIN = PRM.zMin;
  const int ZMAX0 = PRM.zMax0;
  const int DELTA = PRM.delta;

  MDM->k2ly0 = ivector(ZMIN - DELTA, ZMAX0);
  MDM->k2ly2 = ivector(ZMIN - DELTA, ZMAX0);

  return (EXIT_SUCCESS);
}

int AllocateFields2(struct VELOCITY **velocities,
		    struct STRESS **stresses,
		    struct ANELASTICITY *ANL,
		    struct ABSORBING_BOUNDARY_CONDITION *ABC,
		    struct MEDIUM *MDM,
		    struct SOURCE **sources, struct PARAMETERS PRM) {

  allocateVelocities(velocities, PRM);
  allocateSources(sources, PRM);
  allocateStress(stresses, PRM);
  allocateABC(ABC, PRM);
  allocateMedium(MDM, PRM);

  return (EXIT_SUCCESS);
}
