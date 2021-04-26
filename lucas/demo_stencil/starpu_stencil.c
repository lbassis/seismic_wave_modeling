// para quinta 25/03 14h
// reservar tupis

// comparar resultados com as outras versoes


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <starpu.h>

#define SIZE 40
#define EPSILON 0.001
#define STENCIL_MAX_STEPS 1//0000

int __n_blocks_x, __n_blocks_y, __n_blocks;
int __block_size_x, __block_size_y;

void step_func(void *buffers[], void *cl_arg);
static void stencil_init(float **prev_vector, float **next_vector);
static void vector_to_blocks(float *vector, float ***blocks);
static void stencil_display(float *vector, int x0, int x1, int y0, int y1);
void insert_block_task(int i, int j, starpu_data_handle_t *currently_reading, starpu_data_handle_t *currently_writing, float alpha);
void read_topo();

void read_topo() {

  FILE *input = fopen("topo.in", "r");
  fscanf(input, "%d\t%d", &__n_blocks_x, &__n_blocks_y);
  fclose(input);

  __n_blocks = __n_blocks_x * __n_blocks_y;
  __block_size_x = SIZE/__n_blocks_x;
  __block_size_y = SIZE/__n_blocks_y;
  printf("%d %d %d %d %d\n", __n_blocks_x, __n_blocks_y, __n_blocks, __block_size_x, __block_size_y);

}

/* =========================== */
/* ---         Main        --- */
/* =========================== */

int main(int argc, char**argv) {

  read_topo();
  printf("%dx%d\n", __n_blocks_x, __n_blocks_y);
  int current_buffer = 0;
  float alpha = 0.02;
  float *prev_vector = malloc(sizeof(float)*SIZE*SIZE);
  float *next_vector = malloc(sizeof(float)*SIZE*SIZE);
  float **prev_blocks, **next_blocks;
  starpu_data_handle_t prev_vector_handles[__n_blocks];
  starpu_data_handle_t next_vector_handles[__n_blocks];
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

  for (int i = 0; i < __n_blocks; i++) {
    starpu_vector_data_register(&(prev_vector_handles[i]), 0, (uintptr_t)prev_blocks[i], __block_size_x*__block_size_y, sizeof(prev_blocks[i][0]));
    starpu_vector_data_register(&(next_vector_handles[i]), 0, (uintptr_t)next_blocks[i], __block_size_x*__block_size_y, sizeof(next_blocks[i][0]));

    //printf("bloco %d:\n", i);
    //for (int l = 0; l < BLOCK_SIZE; l++) {
    //  for (int m = 0; m < BLOCK_SIZE; m++) {
    //	printf("%f ", prev_blocks[i][l*BLOCK_SIZE+m]);
    //  }
    //  printf("\n");
    //}
	
  }

  struct timespec t1, t2;
  clock_gettime(CLOCK_MONOTONIC, &t1);

  for(int s = 0; s < STENCIL_MAX_STEPS; s++) {
  
    int prev_buffer = current_buffer;
    int next_buffer = (current_buffer + 1) % 2;
  
    // the handles will depend on the current buffer
    for (int i = 0; i < __n_blocks_x; i++) {
      for (int j = 0; j < __n_blocks_y; j++) {
	if (current_buffer == 0) {
	  currently_reading = prev_vector_handles;
	  currently_writing = next_vector_handles;
	}
	else {
	  currently_reading = next_vector_handles;
	  currently_writing = prev_vector_handles;
	}
	insert_block_task(i, j, currently_reading, currently_writing, alpha);
	
      }
    }
    starpu_task_wait_for_all();
    // check if everyone converged
  
    current_buffer = next_buffer;     
  }

  for (int i = 0; i < __n_blocks; i++) {
    starpu_data_unregister(prev_vector_handles[i]);
    starpu_data_unregister(next_vector_handles[i]);
  }

  starpu_shutdown();
  clock_gettime(CLOCK_MONOTONIC, &t2);
  const double t_usec = (t2.tv_sec - t1.tv_sec) * 1000000.0 + (t2.tv_nsec - t1.tv_nsec) / 1000.0;
  printf("starpu;%f\n", t_usec/1000);
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
  int i = 0, j = 0, neighboors, last_neighboor = 0;
  int k_begin, k_end, l_begin, l_end;
  int block_size_x, block_size_y;

  k_begin = 1;
  k_end = block_size_x+1;
  l_begin = 1;
  l_end = block_size_y+1;
  
  starpu_codelet_unpack_args(cl_arg, &i, &j, &neighboors, &alpha, &block_size_x, &block_size_y);
  printf("bloco %d\n", i+j*2);
  full_vector = calloc((block_size_x+2)*(block_size_y+2), sizeof(float));
  prev_vector = (float *)STARPU_VECTOR_GET_PTR(prev_vector_handle);
  next_vector = (float *)STARPU_VECTOR_GET_PTR(next_vector_handle);

  neighborhood = malloc(neighboors*sizeof(float*));

  for (int k = 0; k < neighboors; k++) {
    temp_handle = buffers[2+k];
    neighborhood[k] = (float *)STARPU_VECTOR_GET_PTR(temp_handle);
  }
  
  // fill a bigger block with all the necessary info
  //for (int k = 1; k < block_size_x+1; k++) {
  //  for (int l = 1; l < block_size_y+1; l++) {
  //    printf("acessando a posicao %d\n", (k-1)*(block_size_x+2)+l);
  //    printf("e o prev vector na posicao %d\n", (k-1)*block_size_x+l-1);
  //    full_vector[k*(block_size_x+2)+l] = prev_vector[(l-1)*block_size_x+k-1];
  //  }
  //}
  for (int k = 0; k < block_size_y; k++) {
    for (int l = 0; l < block_size_x; l++) {
      full_vector[(k+1)*block_size_x+l+1] = prev_vector[k*block_size_x+l];
    }
  }
  printf("passou\n");
  
  if (i > 0) { // it has a northern neighboor -> ultima linha do vizinho na primeira linha do full
    for (int k = 0; k < block_size_x; k++) {
      full_vector[k+1] = neighborhood[last_neighboor][(block_size_y-1)*block_size_x+k];
    }
    last_neighboor++;
  }

  if (j < __n_blocks_x-1) { // primeira coluna do vizinho na ultima coluna do full
    for (int k = 0; k < block_size_y; k++) {
      full_vector[(k+1)*(block_size_x+2)+block_size_y] = neighborhood[last_neighboor][k*block_size_x];
    }
    last_neighboor++;
  }
  
  if (i < __n_blocks_y-1) { //primeira linha do vizinho na ultima linha do full
    for (int k = 0; k < block_size_x; k++) {
      full_vector[(block_size_x+2)*block_size_y+(k+1)] = 1;//neighborhood[last_neighboor][k];
    }
    last_neighboor++;
  }
  if (j > 0) { // ultima coluna do vizinho na primeira coluna do full
    for (int k = 0; k < block_size_y; k++) {
      //full_vector[(block_size_x+2)*(k+1)] = 1;//neighborhood[last_neighboor][block_size_x*k+block_size_x-1];
      //printf("full acessando %d e vizinho acessando %d\n", (block_size_x+2)*(k+1), block_size_x*k+block_size_x-1);
    }
  }

  printf("bloco %d,%d com %d vizinhos\n", i, j, neighboors);
  k_begin = 1;
  l_begin = 1;
  k_end = block_size_x+1;
  l_end = block_size_y+1;
  
  if (i == 0) {
    k_begin = 2;
  }
  if (j == __n_blocks_y-1) {
    l_end = block_size_y;
  }
  if (i == __n_blocks_x-1) {
    k_end = block_size_x;
  }
  if (j == 0) {
    l_begin = 2;
  }
  // calculate the stencil
  int k_next = 0;
  int l_next = 0;
  //for (int k = k_begin; k < k_end; k++) {
  //  for (int l = l_begin; l < l_end; l++) {
  //    //next_vector[(k-1)*block_size_x+l-1] = alpha * full_vector[(k-1)*(block_size_x+2)+l] + // north
  //    next_vector[k_next*block_size_x+l_next] = alpha * full_vector[(k-1)*(block_size_x+2)+l] + // north
  //	alpha * full_vector[k*(block_size_x+2)+l+1] + // east
  //	alpha * full_vector[(k+1)*(block_size_x+2)+l] + // south
  //	alpha * full_vector[k*(block_size_x+2)+l-1] + // west
  //	(1.0 - 4.0 * alpha) * full_vector[k*(block_size_x+2)+l];
  //    l_next++;
  //  }
  //  l_next = 0;
  //  k_next++;
  //}  
  printf("bloco %d,%d acabou\n", i, j);  
  free(full_vector);
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
  *blocks = malloc(__n_blocks*sizeof(float*));

  for (int i = 0; i < __n_blocks; i++) {
    (*blocks)[i] = malloc(__block_size_x*__block_size_y*sizeof(float));
  }

  for (int i = 0; i < __n_blocks_x; i++) {
    for (int j = 0; j < __n_blocks_y; j++) {

      int target_block_row = i/__block_size_x;
      int target_block_col = j/__block_size_y;
      int target_block_index = target_block_row*__n_blocks_x+target_block_col;

      int block_i = i%__block_size_x;
      int block_j = j%__block_size_y;

      (*blocks)[target_block_index][block_i*__n_blocks_x+block_j] = vector[i*__n_blocks_x+j];
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
  
void insert_block_task(int i, int j, starpu_data_handle_t *currently_reading, starpu_data_handle_t *currently_writing, float alpha) {
  struct starpu_data_descr *descriptors;
  starpu_data_handle_t **neighborhood = malloc(sizeof(starpu_data_handle_t)*4);
  int neighboors = 0;

  if (i != 0) {
    neighborhood[neighboors] = &(currently_reading[(i-1)*__n_blocks_x+j]);
    neighboors++;
  }
  if (j != __n_blocks_y-1) {
    neighborhood[neighboors] = &(currently_reading[i*__n_blocks_x+j+1]);
    neighboors++;
  }
  if (i != __n_blocks_x-1) {
    neighborhood[neighboors] = &(currently_reading[(i+1)*__n_blocks_x+j]);
    neighboors++;
  }
  if (j != 0) {
    neighborhood[neighboors] = &(currently_reading[i*__n_blocks_x+j-1]);
    neighboors++;
  }

  descriptors = malloc((2+neighboors) * sizeof(struct starpu_data_descr));

  descriptors[0].handle = currently_reading[i*__n_blocks_x+j];
  descriptors[0].mode = STARPU_R;
  descriptors[1].handle = currently_writing[i*__n_blocks_x+j];
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
		   STARPU_VALUE, &__block_size_x, sizeof(__block_size_x),
		   STARPU_VALUE, &__block_size_y, sizeof(__block_size_y),
		   STARPU_DATA_MODE_ARRAY, descriptors, neighboors+2,
		   0);
}
