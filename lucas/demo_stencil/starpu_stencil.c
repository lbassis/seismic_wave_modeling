#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <starpu.h>

#define SIZE 5
//#define BLOCK_SIZE 3
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
      (*prev_vector)[i*SIZE+j] = 0.0;
      (*next_vector)[i*SIZE+j] = 0.0;
    }
  }
  for(i = 0; i < SIZE; i++) {
    (*prev_vector)[i*SIZE] = i;
    (*prev_vector)[i*SIZE+SIZE-1] = SIZE - i;
    (*next_vector)[i*SIZE] = i;
    (*next_vector)[i*SIZE+SIZE-1] = SIZE - i;

  }
  for(j = 0; j < SIZE; j++) {
    (*prev_vector)[j] = j;
    (*prev_vector)[(SIZE-1)*SIZE+j] = SIZE - j;
    (*next_vector)[j] = j;
    (*next_vector)[(SIZE-1)*SIZE+j] = SIZE - j;
  }
}

/** return 1 if computation has converged */
static int stencil_test_convergence(int current_buffer, float *prev_vector, float *next_vector) {
  int i, j;
  for(i = 1; i < SIZE-1; i++) {
    for(j = 1; j < SIZE-1; j++) {
      if (current_buffer == 0) {
	if(fabs(prev_vector[i*SIZE+j] - next_vector[i*SIZE+j]) > EPSILON) {
	  return 0;
	}
      }
      else {
	if(fabs(next_vector[i*SIZE+j] - prev_vector[i*SIZE+j]) > EPSILON) {
	  return 0;
	}		
      }
    }
  }
  return 1;
}

/** display a part of the stencil values */
static void stencil_display(float *vector, int x0, int x1, int y0, int y1) {
  int i, j;
  for(i = y0; i <= y1; i++) {
    for(j = x0; j <= x1; j++) {
      printf("%8.5g ", vector[i*SIZE+j]);
    }
    printf("\n");
  }
}


/* =========================== */
/* --- StarPU functions --- */
/* =========================== */

void step_func(void *buffers[], void *cl_arg) {
  struct starpu_vector_interface *prev_vector_handle = buffers[0];
  struct starpu_vector_interface *next_vector_handle = buffers[1];

  //unsigned n = STARPU_VECTOR_GET_NX(prev_vector_handle);
  
  float *prev_vector = (float *)STARPU_VECTOR_GET_PTR(prev_vector_handle);
  float *next_vector = (float *)STARPU_VECTOR_GET_PTR(next_vector_handle);
  float alpha;
  
  starpu_codelet_unpack_args(cl_arg, &alpha);

  for (int i = 1; i < SIZE-1; i++) {
    for (int j = 1; j < SIZE-1; j++) {
    next_vector[i*SIZE+j] =
	alpha * prev_vector[(i-1)*SIZE+j] +
	alpha * prev_vector[(i+1)*SIZE+j] +
	alpha * prev_vector[i*SIZE+j-1] +
	alpha * prev_vector[i*SIZE+j+1] +
	(1.0 - 4.0 * alpha) * prev_vector[i*SIZE+j];
    }
  }
}

struct starpu_codelet stencil_step = {
				      .cpu_funcs = {step_func},
				      .nbuffers = 2, // prev_vector, next_vector
				      .modes = {STARPU_R, STARPU_W},
};

/* =========================== */
/* ---         Main        --- */
/* =========================== */

int main(int argc, char**argv) {

  int current_buffer = 0;
  float alpha = 0.02;
  float *prev_vector = malloc(sizeof(float)*SIZE*SIZE);
  float *next_vector = malloc(sizeof(float)*SIZE*SIZE);
  starpu_data_handle_t prev_vector_handle, next_vector_handle;
    
  int ret = starpu_init(NULL);
  if (ret != 0) {
    printf("Erro inicializando StarPU\n");
    return -1;
  }

  stencil_init(&prev_vector, &next_vector);
  stencil_display(prev_vector, 0, SIZE-1, 0, SIZE-1);

 
  starpu_vector_data_register(&prev_vector_handle, 0, (uintptr_t)prev_vector, SIZE*SIZE, sizeof(prev_vector[0]));
  starpu_vector_data_register(&next_vector_handle, 0, (uintptr_t)next_vector, SIZE*SIZE, sizeof(next_vector[0]));

  for(int s = 0; s < STENCIL_MAX_STEPS; s++) {

    int prev_buffer = current_buffer;
    int next_buffer = (current_buffer + 1) % 2;

    printf("step %d\n", s);
    // the handles will depend on the current buffer
    if (current_buffer == 0) {
      starpu_insert_task(&stencil_step,
			 STARPU_R, prev_vector_handle,
			 STARPU_W, next_vector_handle,
			 STARPU_VALUE, &alpha, sizeof(alpha),
			 0);
    }
    else {
      starpu_insert_task(&stencil_step,
			 STARPU_R, next_vector_handle,
			 STARPU_W, prev_vector_handle,
			 STARPU_VALUE, &alpha, sizeof(alpha),
			 0);
    }

      starpu_task_wait_for_all();
      if(stencil_test_convergence(current_buffer, prev_vector, next_vector)) {
      	  printf("# steps = %d\n", s);
      	  break;
      }

      current_buffer = next_buffer;
  }
  
  starpu_data_unregister(prev_vector_handle);
  starpu_data_unregister(next_vector_handle);

  starpu_shutdown();

  printf("result:\n");
  if (current_buffer == 0) {
      stencil_display(next_vector, 0, SIZE-1, 0, SIZE-1);
  }
  else {
    stencil_display(prev_vector, 0, SIZE-1, 0, SIZE-1);
  }

  return 0;
}

