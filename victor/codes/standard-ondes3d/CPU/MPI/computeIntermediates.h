#ifndef _COMPUTE_INTERMEDIATES_H
#define _COMPUTE_INTERMEDIATES_H
void ComputeIntermediates(	
						  /* Outputs */
						  struct ABSORBING_BOUNDARY_CONDITION *ABC,
						  struct ANELASTICITY *ANL,
						  
						  /* Parameters */
						  struct VELOCITY v0, /* Velocity */
						  struct PARAMETERS PRM,
						  struct MEDIUM MDM,
						  
						  const int mpmx_begin,const int mpmx_end, /* local computed domain */
						  const int mpmy_begin,const int mpmy_end 
                          );


#endif /* _COMPUTE_INTERMEDIATES_H */
