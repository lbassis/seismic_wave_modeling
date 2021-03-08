// para quinta 11/03 14h
// reservar tupis

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <starpu.h>
#include "stencil_tasks.h"

#define SIZE 9
#define BLOCK_SIZE 3
#define N_BLOCKS (SIZE/BLOCK_SIZE)*(SIZE/BLOCK_SIZE)
#define EPSILON 0.001
#define STENCIL_MAX_STEPS 10000

void step_func(void *buffers[], void *cl_arg);
static void stencil_init(float **prev_vector, float **next_vector);
static void vector_to_blocks(float *vector, float ***blocks);
static int stencil_test_convergence(int current_buffer, float *prev_vector, float *next_vector);
static void stencil_display(float *vector, int x0, int x1, int y0, int y1);
void insert_block_task(int i, int j, int n_blocks, starpu_data_handle_t *currently_reading, starpu_data_handle_t *currently_writing, float alpha);

/* =========================== */
/* ---         Main        --- */
/* =========================== */

int main(int argc, char**argv) {

  int current_buffer = 0;
  float alpha = 0.02;
  float *prev_vector = malloc(sizeof(float)*SIZE*SIZE);
  float *next_vector = malloc(sizeof(float)*SIZE*SIZE);
  float **prev_blocks, **next_blocks;
  starpu_data_handle_t prev_vector_handles[(SIZE/BLOCK_SIZE)*(SIZE/BLOCK_SIZE)];
  starpu_data_handle_t next_vector_handles[(SIZE/BLOCK_SIZE)*(SIZE/BLOCK_SIZE)];
  starpu_data_handle_t *currently_reading, *currently_writing;
  
  int ret = starpu_init(NULL);
  if (ret != 0) {
    printf("Erro inicializando StarPU\n");
    return -1;
  }

  stencil_init(&prev_vector, &next_vector);
  vector_to_blocks(prev_vector, &prev_blocks);
  vector_to_blocks(next_vector, &next_blocks);
  //stencil_display(prev_vector, 0, SIZE-1, 0, SIZE-1);

  for (int i = 0; i < N_BLOCKS; i++) {
    starpu_vector_data_register(&prev_vector_handles[i], 0, (uintptr_t)prev_blocks[i], BLOCK_SIZE*BLOCK_SIZE, sizeof(prev_blocks[i][0]));
    starpu_vector_data_register(&next_vector_handles[i], 0, (uintptr_t)next_blocks[i], BLOCK_SIZE*BLOCK_SIZE, sizeof(next_blocks[i][0]));
    printf("registrou %p e %p (%d %d)\n", &prev_vector_handles[i], &next_vector_handles[i]);
  }
  for(int s = 0; s < STENCIL_MAX_STEPS; s++) {
  
    int prev_buffer = current_buffer;
    int next_buffer = (current_buffer + 1) % 2;
  
    // the handles will depend on the current buffer
    for (int i = 0; i < N_BLOCKS; i++) {
      for (int j = 0; j < N_BLOCKS; j++) {
	if (current_buffer == 0) {
	  currently_reading = prev_vector_handles;
	  currently_writing = next_vector_handles;
	}
	else {
	  currently_reading = next_vector_handles;
	  currently_writing = prev_vector_handles;
	}
	printf("vai inserir\n");
	insert_block_task(i, j, N_BLOCKS, currently_reading, currently_writing, alpha);
      }
    }
    starpu_task_wait_for_all();
    // precisa usar os handles!!!
    if (stencil_test_convergence(current_buffer, prev_vector, next_vector)) {
      printf("# steps = %d\n", s);
      break;
    }
  
    current_buffer = next_buffer;     
  }
  
  //acquire e release pra ler/escrever
  //starpu_data_unregister(prev_vector_handles);
  //starpu_data_unregister(next_vector_handles);
  
  //starpu_shutdown();
  
  //printf("result:\n");
  //if (current_buffer == 0) {
  //    stencil_display(next_vector, 0, SIZE-1, 0, SIZE-1);
  //}
  //else {
  //  stencil_display(prev_vector, 0, SIZE-1, 0, SIZE-1);
  //}

  return 0;
}

/* =========================== */
/* --- StarPU functions --- */
/* =========================== */

typedef struct {
  int position;
  float alpha;
} params;

enum starpu_data_access_mode modes[STARPU_NMAXBUFS+1] =
{
 STARPU_R, STARPU_W, STARPU_R, STARPU_R, STARPU_R, STARPU_R
};

void step_func(void *buffers[], void *cl_arg) {
  //struct starpu_vector_interface *prev_vector_handle = buffers[0];
  //struct starpu_vector_interface *next_vector_handle = buffers[1];
  //
  ////unsigned n = STARPU_VECTOR_GET_NX(prev_vector_handle);
  //
  //float *prev_vector = (float *)STARPU_VECTOR_GET_PTR(prev_vector_handle);
  //float *next_vector = (float *)STARPU_VECTOR_GET_PTR(next_vector_handle);
  //float alpha;
  //
  //starpu_codelet_unpack_args(cl_arg, &alpha);
  //
  //for (int i = 1; i < SIZE-1; i++) {
  //  for (int j = 1; j < SIZE-1; j++) {
  //  next_vector[i*SIZE+j] =
  //	alpha * prev_vector[(i-1)*SIZE+j] +
  //	alpha * prev_vector[(i+1)*SIZE+j] +
  //	alpha * prev_vector[i*SIZE+j-1] +
  //	alpha * prev_vector[i*SIZE+j+1] +
  //	(1.0 - 4.0 * alpha) * prev_vector[i*SIZE+j];
  //  }
  //}
  params *params = cl_arg;
  printf("veio %d e %f\n", params->position, params->alpha);
}

struct starpu_codelet stencil_step = {
				      .cpu_funcs = {step_func},
				      .nbuffers = 6,
				      .dyn_modes = modes,
};

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

static void vector_to_blocks(float *vector, float ***blocks) {
  *blocks = malloc(N_BLOCKS*sizeof(float*));

  for (int i = 0; i < SIZE; i++) {
    (*blocks)[i] = malloc(BLOCK_SIZE*BLOCK_SIZE*sizeof(float));
  }

  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {

      int target_block_row = i/BLOCK_SIZE;
      int target_block_col = j/BLOCK_SIZE;
      int target_block_index = target_block_row*(SIZE/BLOCK_SIZE)+target_block_col;

      int block_i = i%BLOCK_SIZE;
      int block_j = j%BLOCK_SIZE;

      (*blocks)[target_block_index][block_i*(SIZE/BLOCK_SIZE)+block_j] = vector[i*SIZE+j];
    }
  }
}

// da pra por o teste dentro da task usando um int como booleano pra dizer se convergiu
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
  
void insert_block_task(int i, int j, int n_blocks, starpu_data_handle_t *currently_reading, starpu_data_handle_t *currently_writing, float alpha) {
  starpu_data_handle_t **neighborhood = malloc(sizeof(starpu_data_handle_t)*4);
  struct starpu_task *task = starpu_task_create();
  params params;
  int neighboors = 0;

  params.position = 1;
  params.alpha = alpha;

  if (i != 0) {
    printf("adiciona o %p\n", &(currently_reading[(i-1)*n_blocks+j]));
    neighborhood[neighboors] = &(currently_reading[(i-1)*n_blocks+j]);
    neighboors++;
  }
  if (j != n_blocks) {
    printf("adiciona o %p\n",  &(currently_reading[i*n_blocks+j+1]));
    neighborhood[neighboors] = &(currently_reading[i*n_blocks+j+1]);
    neighboors++;
  }
  if (i != n_blocks) {
    neighborhood[neighboors] = &(currently_reading[(i+1)*n_blocks+j]);
    neighboors++;
  }
  if (j != 0) {
    neighborhood[neighboors] = &(currently_reading[i*n_blocks+j-1]);
    neighboors++;
  }

  task->cl = &stencil_step;
  task->cl_arg = &params;
  task->cl_arg_size = sizeof(params);
  task->cl->nbuffers = 2 + neighboors;
  task->dyn_handles = malloc(task->cl->nbuffers * sizeof(starpu_data_handle_t));
  
  task->dyn_handles[0] = currently_reading[i*n_blocks+j];
  task->dyn_handles[1] = currently_writing[i*n_blocks+j];

  for (int k = 0; k < neighboors; k++) {
    task->dyn_handles[k+2] = *(neighborhood[k]);
  }

  starpu_task_submit(task);
}
