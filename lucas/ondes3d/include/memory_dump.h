#include <stdio.h>

#include "struct.h"
#include "new_nrutil.h"

void dump_tensors(struct VELOCITY *velocity, struct STRESS *stress, struct ABSORBING_BOUNDARY_CONDITION *abc, struct SOURCE *src, struct PARAMETERS PRM, FILE *out);
void dump_vectors(struct ABSORBING_BOUNDARY_CONDITION *abc, struct MEDIUM *mdm, struct PARAMETERS PRM, FILE *out);
void dump_integers(struct ABSORBING_BOUNDARY_CONDITION *abc, FILE *out);
void dump(struct VELOCITY *velocity, struct STRESS *stress, struct ABSORBING_BOUNDARY_CONDITION *abc, struct SOURCE *src, struct MEDIUM *mdm, struct PARAMETERS PRM);
