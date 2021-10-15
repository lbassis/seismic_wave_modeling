#include <stdio.h>
#include <stdlib.h>
#include <starpu.h>

#include "../include/IO.h"
#include "../include/alloAndInit.h"
#include "../include/options.h"

#include "../include/memory_dump.h"


int VerifFunction(int exitStatus, const char *msg, struct PARAMETERS PRM) {
  if (exitStatus != EXIT_SUCCESS) {
    fprintf(stderr, "%-50s [ERROR] with cpu %i \n", msg, PRM.me);
  }
}

int main(int argc, char *argv[]) {

  int ret, l, imode, np = 1;
  int mpmx_begin, mpmx_end, mpmy_begin, mpmy_end;

  int nb_blocks_x, nb_blocks_y, block_size;
  int i, j, block_index;
  double time;

  struct PARAMETERS PRM = { 0 };
  struct MEDIUM MDM = { 0 };
  struct ANELASTICITY ANL = { 0 };
  struct OUTPUTS OUT = { 0 };
  struct SOURCE SRC = { 0 };
  struct VELOCITY v0 = { 0 };
  struct STRESS t0 = { 0 };
  struct ABSORBING_BOUNDARY_CONDITION ABC =  { 0 };

  if (argc < 2) {
    printf("Insira o tamanho do bloco a ser utilizado\n");
    return -1;
  }

  else {
    PRM.block_size = atoi(argv[1]);
    printf("Usando blocos de tamanho %dx%d\n", PRM.block_size, PRM.block_size);
  }

  ret = starpu_init(NULL);
  if (ret != 0) {
    printf("Erro inicializando StarPU\n");
    return -1;
  }

  PRM.px = 1;
  PRM.py = 1;
  VerifFunction(ReadPrmFile(&PRM, &MDM, &ABC, &ANL, &OUT, PRMFILE), "read parameter file ", PRM);

  VerifFunction(InitPartDomain(&PRM, &OUT), "split domain MPI", PRM);

  if (model == GEOLOGICAL) {
    VerifFunction(ReadGeoFile(&MDM, PRM), "read geological file", PRM);
  }

  VerifFunction(ReadStation(&OUT, PRM, MDM), "read station file ", PRM);
  VerifFunction(AllocateFields(&v0, &t0, &ANL, &ABC, &MDM, &SRC, PRM), "allocate Fields ", PRM);
  if (model == LAYER) {
    VerifFunction(InitLayerModel(&MDM, &ANL, PRM), "initialize layer model", PRM);
  }

  /* in the absorbing layers */
  VerifFunction(InitializeABC(&ABC, &MDM, &ANL, PRM), "initilize absorbing boundaries ", PRM);

  /* Allocation output */
  //VerifFunction(InitializeOutputs(STATION_STEP, &OUT, PRM),
	//	" MapSeismograms", PRM);
  VerifFunction(EXIT_SUCCESS, "Beginning of the iteration", PRM);

  main_loop(&SRC, &ABC, &MDM, &t0, &v0, &PRM);

  printf("acabou\n");
  starpu_shutdown();
}
