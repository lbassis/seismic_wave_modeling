#ifndef COMPUTESTRESS_H_
#define COMPUTESTRESS_H_
void ComputeStress(	
				   /* Main Outputs (we also increment Absorbing Layer functions)*/
				   struct STRESS *t0,
				   /* INPUTS */
				   struct VELOCITY v0,
				   struct MEDIUM MDM,
				   struct PARAMETERS PRM,
				   struct ABSORBING_BOUNDARY_CONDITION ABC,
				   struct ANELASTICITY ANL,

				   int mpmx_begin, int mpmx_end, /* local computed domain */
				   int mpmy_begin, int mpmy_end
					);

/* ********* */
/* Functions */
/* ********* */
/* KRISTEK and MOCZO Related */
/* GOAL :
   find the right Y for the cell, and its neighbours */
int ChooseY(			/* OUTPUTS */
                          double *yMuMe, double *yMuNeighP, double *yMuNeighN,
                          double *yKapMe, double *yKapNeighP, double *yKapNeighN,
                          int i, int j, int k,   
                          struct ANELASTICITY ANL,
                          struct MEDIUM MDM
                          );

#endif	/* COMPUTESTRESS_H_ */
