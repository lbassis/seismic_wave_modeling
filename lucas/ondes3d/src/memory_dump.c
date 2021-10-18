#include <stdio.h>

#include "../include/struct.h"
#include "../include/memory_dump.h"

void dump_tensors(struct VELOCITY *velocity, struct STRESS *stress, struct ABSORBING_BOUNDARY_CONDITION *abc, struct SOURCE *src, struct PARAMETERS PRM, FILE *out) {

  const int MPMX = PRM.mpmx;
  const int MPMY = PRM.mpmy;
  const int ZMIN = PRM.zMin;
  const int ZMAX0 = PRM.zMax0;
  const int DELTA = PRM.delta;

  int i, j, k;

  // velocity, stress and ipml have the same size. we will dump everything together just to check it
  for (i = -1; i <= MPMX; i++) {
    for (j = -1; j <= MPMY; j++) {
      for (k = ZMIN - DELTA; k <= ZMAX0; k++) {

	fprintf(out, "%f %f %f %f %f %f %f %f %f %d\n", i3access(velocity->x, -1, PRM.block_size + 2, -1, PRM.block_size + 2, PRM.zMin - PRM.delta, PRM.zMax0, i, j, k),
    i3access(velocity->y, -1, PRM.block_size + 2, -1, PRM.block_size + 2, PRM.zMin - PRM.delta, PRM.zMax0, i, j, k),
    i3access(velocity->z, -1, PRM.block_size + 2, -1, PRM.block_size + 2, PRM.zMin - PRM.delta, PRM.zMax0, i, j, k),
    i3access(stress->xx, -1, PRM.block_size + 2, -1, PRM.block_size + 2, PRM.zMin - PRM.delta, PRM.zMax0, i, j, k),
		i3access(stress->yy, -1, PRM.block_size + 2, -1, PRM.block_size + 2, PRM.zMin - PRM.delta, PRM.zMax0, i, j, k),
    i3access(stress->zz, -1, PRM.block_size + 2, -1, PRM.block_size + 2, PRM.zMin - PRM.delta, PRM.zMax0, i, j, k),
    i3access(stress->xy, -1, PRM.block_size + 2, -1, PRM.block_size + 2, PRM.zMin - PRM.delta, PRM.zMax0, i, j, k),
    i3access(stress->xz, -1, PRM.block_size + 2, -1, PRM.block_size + 2, PRM.zMin - PRM.delta, PRM.zMax0, i, j, k),
    i3access(stress->yz, -1, PRM.block_size + 2, -1, PRM.block_size + 2, PRM.zMin - PRM.delta, PRM.zMax0, i, j, k),
    i3access(abc->ipml, -1, PRM.block_size + 2, -1, PRM.block_size + 2, PRM.zMin - PRM.delta, PRM.zMax0, i, j, k)
        );
      }
    }
  }

  // source has a different size
  for (i = 1; i <= MPMX; i++) {
    for (j = 1; j <= MPMY; j++) {
      for (k = ZMIN - DELTA; k <= ZMAX0; k++) {
	fprintf(out, "%d", i3access(abc->ipml, -1, PRM.block_size + 2, -1, PRM.block_size + 2, PRM.zMin - PRM.delta, PRM.zMax0, i, j, k));
      }
    }
  }
  printf("\n");
}

void dump_vectors(struct ABSORBING_BOUNDARY_CONDITION *abc, struct MEDIUM *mdm, struct PARAMETERS PRM, FILE *out) {

  const int ZMIN = PRM.zMin;
  const int ZMAX0 = PRM.zMax0;
  const int DELTA = PRM.delta;

  int i;
/*
  for (i = 1; i <= abc->npmlv; i++) {
    fprintf(out, "%f %f %f %f %f %f %f %f %f\n", abc->phivxx[i], abc->phivxy[i], abc->phivxz[i], abc->phivyx[i],
	    abc->phivyy[i], abc->phivyz[i], abc->phivzx[i], abc->phivzy[i], abc->phivzz[i]);
  }
*/
  for (i = 1; i <= abc->npmlt; i++) {
    fprintf(out, "%f %f %f %f %f %f %f %f %f\n", abc->phitxxx[i], abc->phitxyy[i], abc->phitxzz[i], abc->phitxyx[i],
	    abc->phityyy[i], abc->phityzz[i], abc->phitxzx[i], abc->phityzy[i], abc->phitzzz[i]);
  }

  for (i = ZMIN - DELTA; i <= ZMAX0; i++) {
    fprintf(out, "%d %d\n", mdm->k2ly0[i], mdm->k2ly2[i]);
  }
}

void dump_integers(struct ABSORBING_BOUNDARY_CONDITION *abc, FILE *out) {

  fprintf(out, "%d %d\n", abc->npmlv, abc->npmlt);
}


void dump(struct VELOCITY *velocity, struct STRESS *stress, struct ABSORBING_BOUNDARY_CONDITION *abc, struct SOURCE *src, struct MEDIUM *mdm, struct PARAMETERS PRM) {

  FILE *out = fopen("memory_dump.txt", "w");

  dump_tensors(velocity, stress, abc, src, PRM, out);
  dump_vectors(abc, mdm, PRM, out);
  dump_integers(abc, out);

  fclose(out);
}
