#ifndef VELOCITY_H_
#define VELOCITY_H_

#include <starpu.h>

void compute_velo_task(void *buffers[], void *cl_arg);
void compute_velo_k1(void *buffers[], void *cl_arg);
void compute_velo_k2(void *buffers[], void *cl_arg);

#endif
