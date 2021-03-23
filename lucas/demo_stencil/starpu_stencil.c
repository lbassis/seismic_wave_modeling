// para quinta 25/03 14h
// reservar tupis

// comparar resultados com as outras versoes


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
#define STENCIL_MAX_STEPS 1//0000

void step_func(void *buffers[], void *cl_arg);
static void stencil_init(float **prev_vector, float **next_vector);
static void vector_to_blocks(float *vector, float ***blocks);
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
  stencil_display(prev_vector, 0, SIZE-1, 0, SIZE-1);

  for (int i = 0; i < N_BLOCKS; i++) {
    starpu_vector_data_register(&(prev_vector_handles[i]), 0, (uintptr_t)prev_blocks[i], BLOCK_SIZE*BLOCK_SIZE, sizeof(prev_blocks[i][0]));
    starpu_vector_data_register(&(next_vector_handles[i]), 0, (uintptr_t)next_blocks[i], BLOCK_SIZE*BLOCK_SIZE, sizeof(next_blocks[i][0]));

    //printf("bloco %d:\n", i);
    //for (int l = 0; l < BLOCK_SIZE; l++) {
    //  for (int m = 0; m < BLOCK_SIZE; m++) {
    //	printf("%f ", prev_blocks[i][l*BLOCK_SIZE+m]);
    //  }
    //  printf("\n");
    //}
	
  }

  for(int s = 0; s < STENCIL_MAX_STEPS; s++) {
  
    int prev_buffer = current_buffer;
    int next_buffer = (current_buffer + 1) % 2;
  
    // the handles will depend on the current buffer
    for (int i = 0; i < SIZE/BLOCK_SIZE; i++) {
      for (int j = 0; j < SIZE/BLOCK_SIZE; j++) {
	if (current_buffer == 0) {
	  currently_reading = prev_vector_handles;
	  currently_writing = next_vector_handles;
	}
	else {
	  currently_reading = next_vector_handles;
	  currently_writing = prev_vector_handles;
	}
	insert_block_task(i, j, SIZE/BLOCK_SIZE, currently_reading, currently_writing, alpha);
	
      }
    }
    starpu_task_wait_for_all();
    // check if everyone converged
  
    current_buffer = next_buffer;     
  }

  for (int i = 0; i < N_BLOCKS; i++) {
    starpu_data_unregister(prev_vector_handles[i]);
    starpu_data_unregister(next_vector_handles[i]);
  }

  starpu_shutdown();

  printf("result:\n");
  if (current_buffer == 1) {
    for (int k = 0; k < N_BLOCKS; k++) {
      printf("block %d\n", k);
      stencil_display(next_blocks[k], 0, BLOCK_SIZE-1, 0, BLOCK_SIZE-1);
      printf("\n");
    }
  }
  else {
    for (int k = 0; k < N_BLOCKS; k++) {
      printf("block %d\n", k);
      stencil_display(prev_blocks[k], 0, BLOCK_SIZE-1, 0, BLOCK_SIZE-1);
      printf("\n");
    }

  }
  return 0;
}

/* =========================== */
/* --- StarPU functions --- */
/* =========================== */

typedef struct {
  int i;
  int j;
  int neighboors;
  float alpha;
} params_t;


void step_func(void *buffers[], void *cl_arg) {
  struct starpu_vector_interface *prev_vector_handle = buffers[0];
  struct starpu_vector_interface *next_vector_handle = buffers[1];
  struct starpu_vector_interface *temp_handle;
  float *prev_vector, *next_vector, *full_vector, **neighborhood, alpha;
  int i, j, neighboors, last_neighboor = 0;

  starpu_codelet_unpack_args(cl_arg, &i, &j, &neighboors, &alpha);

  full_vector = calloc((BLOCK_SIZE+2)*(BLOCK_SIZE+2), sizeof(float));
  prev_vector = (float *)STARPU_VECTOR_GET_PTR(prev_vector_handle);
  next_vector = (float *)STARPU_VECTOR_GET_PTR(next_vector_handle);

  neighborhood = malloc(neighboors*sizeof(float));

  for (int k = 0; k < neighboors; k++) {
    temp_handle = buffers[2+k];
    neighborhood[k] = (float *)STARPU_VECTOR_GET_PTR(temp_handle);

    //printf("vizinho %d\n", k);
    //for (int l = 0; l < BLOCK_SIZE; l++) {
    //  for (int m = 0; m < BLOCK_SIZE; m++) {
    //	printf("%f ", neighborhood[k][l*BLOCK_SIZE+m]);
    //  }
    //  printf("\n");
    //}
  }
  
  
  // fill a bigger block with all the necessary info
  for (int k = 1; k < BLOCK_SIZE+1; k++) {
    for (int l = 1; l < BLOCK_SIZE+1; l++) {
      full_vector[k*(BLOCK_SIZE+2)+l] = prev_vector[(k-1)*BLOCK_SIZE+l-1];
    }
  }
  
  if (i > 0) { // it has a northern neighboor
    for (int k = 1; k < BLOCK_SIZE+1; k++) {
      full_vector[k] = neighborhood[last_neighboor][BLOCK_SIZE*(BLOCK_SIZE-1)+k-1];
    }
    last_neighboor++;
  }
  if (j < (SIZE/BLOCK_SIZE)-1) { // it has a eastern neighboor
    for (int k = 1; k < BLOCK_SIZE+1; k++) {
      full_vector[(BLOCK_SIZE+2)*k+BLOCK_SIZE+1] = neighborhood[last_neighboor][BLOCK_SIZE*(k-1)];
    }
    last_neighboor++;
  }
  if (i < (SIZE/BLOCK_SIZE)-1) { // it has a southern neighboor
    for (int k = 1; k < BLOCK_SIZE+1; k++) {
      full_vector[(BLOCK_SIZE+2)*(BLOCK_SIZE+1)+k] = neighborhood[last_neighboor][k-1];
    }
    last_neighboor++;
  }
  if (j > 0) { // it has a western neighboor
    for (int k = 1; k < BLOCK_SIZE+1; k++) {
      full_vector[(BLOCK_SIZE+2)*k] = neighborhood[last_neighboor][(k-1)*BLOCK_SIZE+BLOCK_SIZE-1];
    }
  }
  
  //printf("bloco %d ficou\n", i*SIZE/BLOCK_SIZE+j);
  //for (int k = 0; k < 5; k++) {
  //  for (int l = 0; l < 5; l++) {
  //    printf("%f ", full_vector[k*5+l]);
  //  }
  //  printf("\n");
  //}


  // o 0 (next) tem que pegar as posicoes 1, 7, 11, 5 e 6 do full vector
  // calculate the stencil 
  for (int k = 1; k < BLOCK_SIZE+1; k++) {
    for (int l = 1; l < BLOCK_SIZE+1; l++) {
      //printf("%d vem dos vizinhos %d, %d, %d, %d e ele mesmo %d\n", (k-1)*BLOCK_SIZE+l-1, (k-1)*(BLOCK_SIZE+2)+l, k*(BLOCK_SIZE+2)+l+1, (k+1)*(BLOCK_SIZE+2)+l, k*(BLOCK_SIZE+2)+l-1, k*(BLOCK_SIZE+2)+l);
      next_vector[(k-1)*BLOCK_SIZE+l-1] = alpha * full_vector[(k-1)*(BLOCK_SIZE+2)+l] + // north
  	alpha * full_vector[k*(BLOCK_SIZE+2)+l+1] + // east
  	alpha * full_vector[(k+1)*(BLOCK_SIZE+2)+l] +
	alpha * full_vector[k*(BLOCK_SIZE+2)+l-2] + // west
  	(1.0 - 4.0 * alpha) * full_vector[k*(BLOCK_SIZE+2)+l];
    } 
  }
    // 0.02*1 + 0.02*1 + 1 - 0.08
  //printf("bloco %d ficou\n", i*SIZE/BLOCK_SIZE+j);
  //for (int k = 0; k < 3; k++) {
  //  for (int l = 0; l < 3; l++) {
  //    printf("%f ", next_vector[k*3+l]);
  //  }
  //  printf("\n");
  //}
  
  
  free(full_vector);
  //return converges
}

enum starpu_data_access_mode modes[STARPU_NMAXBUFS+1] = {STARPU_R, STARPU_W, STARPU_R, STARPU_R, STARPU_R, STARPU_R};

struct starpu_codelet stencil_step = {
				      .cpu_funcs = {step_func},
				      .nbuffers = STARPU_VARIABLE_NBUFFERS,
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

      //printf("pegando a posicao %d do original que eh %f e botandn oa posicao %d do block %d\n", i*SIZE+j, vector[i*SIZE+j], block_i*(SIZE/BLOCK_SIZE)+block_j, target_block_index);
      (*blocks)[target_block_index][block_i*(SIZE/BLOCK_SIZE)+block_j] = vector[i*SIZE+j];
    }
  }
}

/** display a part of the stencil values */
static void stencil_display(float *vector, int x0, int x1, int y0, int y1) {
  int i, j;
  for(i = y0; i <= y1; i++) {
    for(j = x0; j <= x1; j++) {
      printf("%8.5g ", vector[i*(x1-x0+1)+j]);
    }
    printf("\n");
  }
}
  
void insert_block_task(int i, int j, int n_blocks, starpu_data_handle_t *currently_reading, starpu_data_handle_t *currently_writing, float alpha) {
  struct starpu_data_descr *descriptors;
  starpu_data_handle_t **neighborhood = malloc(sizeof(starpu_data_handle_t)*4);
  int neighboors = 0;

  if (i != 0) {
    neighborhood[neighboors] = &(currently_reading[(i-1)*n_blocks+j]);
    neighboors++;
  }
  if (j != n_blocks-1) {
    neighborhood[neighboors] = &(currently_reading[i*n_blocks+j+1]);
    neighboors++;
  }
  if (i != n_blocks-1) {
    neighborhood[neighboors] = &(currently_reading[(i+1)*n_blocks+j]);
    neighboors++;
  }
  if (j != 0) {
    neighborhood[neighboors] = &(currently_reading[i*n_blocks+j-1]);
    neighboors++;
  }

  descriptors = malloc((2+neighboors) * sizeof(struct starpu_data_descr));

  descriptors[0].handle = currently_reading[i*n_blocks+j];
  descriptors[0].mode = STARPU_R;
  descriptors[1].handle = currently_writing[i*n_blocks+j];
  descriptors[1].mode = STARPU_W;
  for (int k = 0; k < neighboors; k++) {
    descriptors[k+2].handle = (*neighborhood[k]);
    descriptors[k+2].mode = STARPU_R;
  }

starpu_task_insert(&stencil_step,
		     STARPU_VALUE, &i, sizeof(i),
		     STARPU_VALUE, &j, sizeof(j),
		     STARPU_VALUE, &neighboors, sizeof(neighboors),
		     STARPU_VALUE, &alpha, sizeof(alpha),
		     STARPU_DATA_MODE_ARRAY, descriptors, neighboors+2,
		     0);

}
