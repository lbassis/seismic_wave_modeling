#include "struct.h"

int InitializeABC2(struct ABSORBING_BOUNDARY_CONDITION **ABCs,
		   struct MEDIUM *MDM,
		   struct ANELASTICITY *ANL, struct PARAMETERS PRM);
static void CompABCCoef2(	/* outputs */
			   double *dump, double *alpha, double *kappa,
			   /* inputs */
			   int imp,
			   double abscissa_in_PML,
			   struct ABSORBING_BOUNDARY_CONDITION ABC,
			   struct PARAMETERS PRM);
