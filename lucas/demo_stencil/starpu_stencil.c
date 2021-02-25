#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <starpu.h>

#define SIZE 10 // 9x9 + padding
#define EPSILON 0.001
#define STENCIL_MAX_STEPS 10000


/* =========================== */
/* --- Auxiliary functions --- */
/* =========================== */

/** init stencil values to 0, borders to non-zero */
static void stencil_init(float **prev_vector, float **next_vector) {
  int i, j;
  for(i = 0; i < SIZE; i++) {
    for(j = 0; j < SIZE; j++) {
      (*prev_vector)[i*n+j] = 0.0;
      (*next_vector)[i*n+j] = 0.0;
    }
  }
  for(i = 0; i < SIZE; i++) {
    (*prev_vector)[i*n] = i;
    (*prev_vector)[i*n+n-1] = SIZE - i;
    (*next_vector)[i*n] = i;
    (*next_vector)[i*n+n-1] = SIZE - i;

  }
  for(j = 0; j < SIZE; j++) {
    (*prev_vector)[j] = j;
    (*prev_vector)[(n-1)*n+j] = SIZE - j;
    (*next_vector)[j] = j;
    (*next_vector)[(n-1)*n+j] = SIZE - j;
  }
}

/** return 1 if computation has converged */
static int stencil_test_convergence(int current_buffer, float *prev_vector, float *next_vector) {
  int prev_buffer = (current_buffer - 1 + STENCIL_NBUFFERS) % STENCIL_NBUFFERS;
  int i, j;
  for(i = 1; i < SIZE-1; i++) {
    for(j = 1; j < SIZE-1; j++) {
      if (current_buffer == 0) {
	if(fabs(prev_vector[i*n+j] - next_vector[i*n+j]) > EPSILON) {
	  return 0;
	}
      }
      else {
	if(fabs(next_vector[i*n+j] - prev_vector[i*n+j]) > EPSILON) {
	  return 0;
	}		
      }
    }
  }
  return 1;
}

/** display a part of the stencil values */
static void stencil_display(float *vector, int x0, int x1, int y0, int y1) {
  int i, x;
  for(i = y0; i <= y1; i++) {
    for(j = x0; j <= x1; j++) {
      printf("%8.5g ", vector[i*n+j]);
    }
    printf("\n");
  }
}


/* =========================== */
/* --- StarPU functions --- */
/* =========================== */

struct starpu_codelet stencil_step = {
				      .cpu_funcs = {step_func, NULL},
				      .nbuffers = 3, // alpha, prev_vector, next_vector
				      .modes = {STARPU_R, STARPU_R, STARPU_W},
};

void step_func(void *buffers[], void *cl_arg) {
  struct starpu_vector_interface *prev_vector_handle = buffers[0];
  struct starpu_vector_interface *next_vector_handle = buffers[1];

  unsigned n = STARPU_VECTOR_GET_NX(prev_vector_handle);
  
  float *prev_vector = (float *)STARPU_VECTOR_GET_PTR(prev_vector_handle);
  float *next_vector = (float *)STARPU_VECTOR_GET_PTR(next_vector_handle);
  float alpha;
  
  starpu_codelet_unpack_args(cl_arg, &alpha);

  for (int i = 1; i < n-1; i++) {
    for (int j = 1; j < n-1; j++) {
      next_vector[i*n+j] =
	alpha * prev_vector[(i-1)*n+j] +
	alpha * prev_vector[(i+1)*n+j] +
	alpha * prev_vector[i*n+j-1] +
	alpha * prev_vector[i*n+j+1] +
	(1.0 - 4.0 * alpha) * prev_vector[i*n+j];
    }
  }
}


/* =========================== */
/* ---         Main        --- */
/* =========================== */

int main(int argc, char**argv) {

  int current_buffer = 0;
  double alpha = 0.02;
  float prev_vector[SIZE*SIZE], next_vector[SIZE*SIZE];
  starpu_data_handle_t prev_vector_handle, next_vector_handle;

  int ret = starpu_init(NULL);
  if (ret != 0) {
    printf("Erro inicializando StarPU\n");
    return -1;
  }

  stencil_init(&prev_vector, &next_vector);

  // actually we will need one handle for each part ie 9 handles for each 3x3 block for each vector
  starpu_vector_data_register(&prev_vector_handle, 0, (uintptr_t)prev_vector, SIZE, sizeof(prev_vector[0]));
  starpu_vector_data_register(&next_vector_handle, 0, (uintptr_t)next_vector, SIZE, sizeof(next_vector[0]));

  for(int s = 0; s < STENCIL_MAX_STEPS; s++) {
    // 9 tasks for treating 3x3

    int prev_buffer = current_buffer;
    int next_buffer = (current_buffer + 1) % 2;

    // the handles will depend on the current buffer
      starpu_insert_task(&stencil_step,
			 STARPU_R, prev_vector_handle,
			 STARPU_W, next_vector_handle,
			 STARPU_VALUE, &alpha, sizeof(alpha),
			 0);
      starpu_task_wait_for_all();
      if(stencil_test_convergence()) {
	  //printf("# steps = %d\n", s);
	  break;
      }

      current_buffer = next_buffer;
  }
  
  starpu_data_unregister(prev_vector_handle);
  starpu_data_unregister(next_vector_handle);

  starpu_shutdown();

  //stencil_display(current_buffer, 0, n-1, 0, n-1);
  return 0;
}

