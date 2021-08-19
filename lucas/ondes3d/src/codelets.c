#include <starpu.h>

#include "../include/struct.h"
#include "../include/computeIntermediates.h"
#include "../include/computeStress.h"
#include "../include/codelets.h"


void compute_intermediates_func(void *buffers[], void *cl_arg) {
  struct starpu_vector_interface *abc_handle = buffers[0];
  struct starpu_vector_interface *anl_handle = buffers[1];

  int mpmx_begin, mpmx_end, mpmy_begin, mpmy_end;
  struct VELOCITY v0;
  struct PARAMETERS prm;
  struct MEDIUM mdm;
  struct ABSORBING_BOUNDARY_CONDITION *abc;
  struct ANELASTICITY *anl;

  starpu_codelet_unpack_args(cl_arg, &v0, &prm, &mdm, &mpmx_begin, &mpmx_end, &mpmy_begin, &mpmy_end);

  abc = (struct ABSORBING_BOUNDARY_CONDITION *)STARPU_VARIABLE_GET_PTR(abc_handle);
  anl = (struct ANELASTICITY *)STARPU_VARIABLE_GET_PTR(anl_handle);

  ComputeIntermediates(abc, anl, v0, prm, mdm, mpmx_begin, mpmx_end, mpmy_begin, mpmy_end);
}

void compute_stress_func(void *buffers[], void *cl_arg) {
  struct starpu_vector_interface *stress_handle = buffers[0];
  struct starpu_vector_interface *abc_handle = buffers[1];
  struct starpu_vector_interface *anl_handle = buffers[2];

  int mpmx_begin, mpmx_end, mpmy_begin, mpmy_end;
  struct VELOCITY v0;
  struct PARAMETERS prm;
  struct MEDIUM mdm;
  struct ABSORBING_BOUNDARY_CONDITION *abc;
  struct ANELASTICITY *anl;
  struct STRESS *t0;

  starpu_codelet_unpack_args(cl_arg, &v0, &prm, &mdm, &mpmx_begin, &mpmx_end, &mpmy_begin, &mpmy_end);

  t0 = (struct STRESS *)STARPU_VARIABLE_GET_PTR(stress_handle);
  abc = (struct ABSORBING_BOUNDARY_CONDITION *)STARPU_VARIABLE_GET_PTR(abc_handle);
  anl = (struct ANELASTICITY *)STARPU_VARIABLE_GET_PTR(anl_handle);

  ComputeStress(t0, v0, mdm, prm, *abc, *anl, mpmx_begin, mpmx_end, mpmy_begin, mpmy_end);
}
