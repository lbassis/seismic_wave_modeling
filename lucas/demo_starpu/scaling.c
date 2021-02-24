#include <starpu.h>
#include <stdio.h>

#define SIZE 5

void scal_func(void *buffers[], void *cl_arg) {
  struct starpu_vector_interface *vector_handle = buffers[0];

  unsigned n    = STARPU_VECTOR_GET_NX(vector_handle);
  float *vector = (float *)STARPU_VECTOR_GET_PTR(vector_handle);
  
  float factor;
  starpu_codelet_unpack_args(cl_arg, &factor);

  for (int i = 0; i < n; i++) {
    vector[i] *= factor;
  }
}

struct starpu_codelet scal_cl = {
				 .cpu_funcs = {scal_func, NULL},
				 .nbuffers = 1,
				 .modes = {STARPU_RW},
};

int main() {

  float factor = 3.;
  float *vector;
  starpu_data_handle_t vector_handle;

  int ret = starpu_init(NULL);

  if (ret != 0) {
    printf("Erro inicializando StarPU\n");
    return -1;
  }

  vector = malloc(sizeof(int)*SIZE);
  printf("Vetor original:\n");
  for (int i = 0; i < SIZE; i++) {
    vector[i] = i;
    printf("%f \n", vector[i]);
  }

  starpu_vector_data_register(&vector_handle, 0, (uintptr_t)vector, SIZE, sizeof(vector[0]));
  starpu_insert_task(&scal_cl,
		     STARPU_RW, vector_handle,
		     STARPU_VALUE, &factor, sizeof(factor),
		     0);
  starpu_task_wait_for_all();

  starpu_data_unregister(vector_handle);

  printf("\nVetor * %f\n", factor);
  for (int i = 0; i < SIZE; i++) {
    printf("%f \n", vector[i]);
  }

  free(vector);
  starpu_shutdown();

}
