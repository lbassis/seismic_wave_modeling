#ifndef SEIS_MOMENT_H_
#define SEIS_MOMENT_H_

#include <starpu.h>

void seis_moment_task(void *buffers[], void *cl_arg);

#endif
