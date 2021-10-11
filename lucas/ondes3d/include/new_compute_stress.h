#ifndef STRESS_H_
#define STRESS_H_

#include <starpu.h>

void compute_stress_task(void *buffers[], void *cl_arg);

#endif
