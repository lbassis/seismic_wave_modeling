#include <starpu.h>

void seis_moment_task(void *buffers[], void *cl_arg);

struct starpu_codelet seis_moment_cl = {
					.cpu_funcs = {seis_moment_task},
					.nbuffers = 16,
					.modes = {STARPU_W, STARPU_W, STARPU_W, STARPU_R,
						  STARPU_R, STARPU_R, STARPU_R, STARPU_R,
						  STARPU_R, STARPU_R, STARPU_R, STARPU_R,
						  STARPU_R, STARPU_R, STARPU_R, STARPU_R},
};


