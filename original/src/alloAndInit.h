#ifndef ALLOANDINIT_H_
#define ALLOANDINIT_H_
#include "struct.h"
#include "alloAndInit_LayerModel.h"

int InitPartDomain(struct PARAMETERS *PRM, struct OUTPUTS *OUT);
int AllocateFields(struct VELOCITY *v0,
		   struct STRESS *t0,
		   struct ANELASTICITY *ANL,
		   struct ABSORBING_BOUNDARY_CONDITION *ABC,
		   struct MEDIUM *MDM,
		   struct SOURCE *SRC, struct PARAMETERS PRM);
/* NB : other allocations are made when reading files or at their initialisations */

int InitializeCOMM(struct COMM_DIRECTION *NORTH,
		   struct COMM_DIRECTION *SOUTH,
		   struct COMM_DIRECTION *EAST,
		   struct COMM_DIRECTION *WEST,
		   /* inputs */
		   int nnorth,
		   int nsouth,
		   int neast, int nwest, struct PARAMETERS PRM);

int InitializeABC(struct ABSORBING_BOUNDARY_CONDITION *ABC,
		  struct MEDIUM *MDM,
		  struct ANELASTICITY *ANL, const struct PARAMETERS PRM);

int InitializeGeol(struct OUTPUTS *OUT,
		   const struct MEDIUM MDM, const struct PARAMETERS PRM);


int InitializeDayBradley(struct MEDIUM *MDM,
			 struct ANELASTICITY *ANL, struct PARAMETERS PRM);


int InitializeKManelas(struct ANELASTICITY *ANL,
		       struct MEDIUM *MDM, double dt);



int DeallocateAll(int STATION_STEP,
		  struct ANELASTICITY *ANL,
		  struct ABSORBING_BOUNDARY_CONDITION *ABC,
		  struct SOURCE *SRC,
		  struct MEDIUM *MDM,
		  struct STRESS *t0,
		  struct VELOCITY *v0,
		  struct OUTPUTS *OUT,
		  struct COMM_DIRECTION *NORTH,
		  struct COMM_DIRECTION *SOUTH,
		  struct COMM_DIRECTION *EAST,
		  struct COMM_DIRECTION *WEST, struct PARAMETERS *PRM);

int InitializeOutputs(int STATION_STEP,
		      struct OUTPUTS *OUT, struct PARAMETERS PRM);

/*==== small internal functions ====*/
/* for allocateAndInitializeKMinterface */
int depth2k(int pos, double depth, double DS);
double k2depth(int pos, int k, double ds);
double Amp(double q0, struct PARAMETERS *PRM, double f0);

/* for InitializeABC */
static void CompABCCoef(	/* outputs */
			   double *dump, double *alpha, double *kappa,
			   /* inputs */
			   int imp,
			   double abscissa_in_PML,
			   struct ABSORBING_BOUNDARY_CONDITION ABC,
			   struct PARAMETERS PRM);

/* for initializeKManelas */
double CompYlkap(double Ylalpha, double Ylbeta, double kap, double mu,
		 double rho);
int inv(double **M, double **invM, int XMAX);	/* from Numerical recipies */


#endif				/* ALLOANDINIT_H_ */
