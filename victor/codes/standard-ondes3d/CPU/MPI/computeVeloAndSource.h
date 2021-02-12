#ifndef COMPUTEVELOCITY_H_
#define COMPUTEVELOCITY_H_
#include <stdarg.h>
#include <math.h>
#include <stdio.h>
#include "nrutil.h"
#include "struct.h"
int computeSeisMoment(struct SOURCE *SRC, double time,struct  PARAMETERS PRM );

void computeVelocity(
					 /* Main Outputs (we also increment Absorbing Layer functions)*/
					 struct VELOCITY *v0,
					 struct ABSORBING_BOUNDARY_CONDITION *ABC,
					 /* INPUTS */
					 struct  STRESS t0,
					 struct MEDIUM MDM,
					 struct PARAMETERS PRM,
					 struct ANELASTICITY ANL, /* we only use ANOTHER anelasticity method here */
					 struct SOURCE SRC,

					 int mpmx_begin, int mpmx_end, /* local computed domain */
					 int mpmy_begin, int mpmy_end,

					 /* source (VELO method) */
					 int l 	/* time step */
						);

void computeSource (struct VELOCITY *v0,
                    struct PARAMETERS PRM,
                    struct MEDIUM MDM,
                    struct SOURCE SRC,
                    int l
                   );

#endif	/* COMPUTEVELOCITY_H_ */
