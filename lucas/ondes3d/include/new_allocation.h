#include "struct.h"

int allocateVelocities(struct VELOCITY **velocities, struct PARAMETERS PRM);
int allocateSources(struct SOURCE **sources, struct PARAMETERS PRM);
int allocateStress(struct STRESS **stresses, struct PARAMETERS PRM);
int allocateABC(struct ABSORBING_BOUNDARY_CONDITION **ABCs, struct PARAMETERS PRM);
int allocateMedium(struct MEDIUM *MDM, struct PARAMETERS PRM);

int AllocateFields2(struct VELOCITY **velocities,
		    struct STRESS **stresses,
		    struct ANELASTICITY *ANL,
		    struct ABSORBING_BOUNDARY_CONDITION *ABC,
		    struct MEDIUM *MDM,
		    struct SOURCE **sources, struct PARAMETERS PRM);
