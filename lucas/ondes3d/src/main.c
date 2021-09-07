#include <stdio.h>
#include <stdlib.h>
//#include <starpu.h>

#include "../include/IO.h"
#include "../include/alloAndInit.h"
#include "../include/options.h"
#include "../include/computeVeloAndSource.h"
#include "../include/computeIntermediates.h"
#include "../include/computeStress.h"
#include "../include/codelets.h"
#include "../include/memory_dump.h"

//struct starpu_codelet intermediates_cl = {
//  .name = {"intermediate"},
//  .cpu_funcs = {compute_intermediates_func, NULL},
//  .nbuffers = 2,
//  .modes = {STARPU_RW, STARPU_RW},
//};
//
//struct starpu_codelet stress_cl = {
//  .name = {"stress"},
//  .cpu_funcs = {compute_stress_func, NULL},
//  .nbuffers = 3,
//  .modes = {STARPU_RW, STARPU_R, STARPU_R},
//};

int VerifFunction(int exitStatus, const char *msg, struct PARAMETERS PRM) {
  if (exitStatus != EXIT_SUCCESS) {
    fprintf(stderr, "%-50s [ERROR] with cpu %i \n", msg, PRM.me);
  }
}

/* imode : to increase the velocity of the computation, we begin by computing
   the values of ksil at the boundaries of the px * py parts of the array
   Afterwise, we can compute the values  in the middle */
void calculate_mappings(int imode, int MPMX, int MPMY, int *mpmx_begin, int *mpmx_end, int *mpmy_begin, int *mpmy_end) {
  switch(imode) {
  case 1:
    *mpmx_begin = 1;
    *mpmx_end = 3;
    *mpmy_begin = 1;
    *mpmy_end = MPMY;
    break;
  case 2:
    *mpmx_begin = MPMX - 2;
    *mpmx_end = MPMX;
    *mpmy_begin = 1;
    *mpmy_end = MPMY;
    break;
  case 3:
    *mpmy_begin = 1;
    *mpmy_end = 3;
    *mpmx_begin = 4;
    *mpmx_end = MPMX - 3;
    break;
  case 4:
    *mpmy_begin = MPMY - 2;
    *mpmy_end = MPMY;
    *mpmx_begin = 4;
    *mpmx_end = MPMX - 3;
    break;
  case 5:
    *mpmx_begin = 4;
    *mpmx_end = MPMX - 3;
    *mpmy_begin = 4;
    *mpmy_end = MPMY - 3;
  }      
}

/*void insert_compute_intermediates_task(starpu_data_handle_t abc_handle, starpu_data_handle_t anl_handle, struct VELOCITY v0, struct PARAMETERS prm,
				       struct MEDIUM mdm, int mpmx_begin, int mpmx_end, int mpmy_begin, int mpmy_end) {

  //starpu_data_handle_t abc_handle, anl_handle;
  //starpu_variable_data_register(&abc_handle, STARPU_MAIN_RAM, (uintptr_t)abc, sizeof(abc));
  //starpu_variable_data_register(&anl_handle, STARPU_MAIN_RAM, (uintptr_t)anl, sizeof(anl));

  starpu_insert_task(&intermediates_cl,
		     STARPU_RW, abc_handle,
		     STARPU_RW, anl_handle,
		     STARPU_VALUE, &v0, sizeof(v0),
		     STARPU_VALUE, &prm, sizeof(prm),
		     STARPU_VALUE, &mdm, sizeof(mdm),
		     STARPU_VALUE, &mpmx_begin, sizeof(mpmx_begin),
		     STARPU_VALUE, &mpmx_end, sizeof(mpmx_end),
		     STARPU_VALUE, &mpmy_begin, sizeof(mpmy_begin),
		     STARPU_VALUE, &mpmy_end, sizeof(mpmy_end),
		     0);
}

void insert_compute_stress(starpu_data_handle_t stress_handle, starpu_data_handle_t abc_handle, starpu_data_handle_t anl_handle, struct VELOCITY v0, struct MEDIUM MDM,
			   struct PARAMETERS PRM,int mpmx_begin, int mpmx_end, int mpmy_begin, int mpmy_end) {
  //starpu_data_handle_t stress_handle;
  //starpu_variable_data_register(&stress_handle, STARPU_MAIN_RAM, (uintptr_t)t0, sizeof(t0));

  starpu_insert_task(&stress_cl,
		     STARPU_RW, stress_handle,
		     STARPU_R, abc_handle,
		     STARPU_R, anl_handle,
		     STARPU_VALUE, &v0, sizeof(v0),
		     STARPU_VALUE, &PRM, sizeof(PRM),
		     STARPU_VALUE, &MDM, sizeof(MDM),
		     STARPU_VALUE, &mpmx_begin, sizeof(mpmx_begin),
		     STARPU_VALUE, &mpmx_end, sizeof(mpmx_end),
		     STARPU_VALUE, &mpmy_begin, sizeof(mpmy_begin),
		     STARPU_VALUE, &mpmy_end, sizeof(mpmy_end),
		     0);
}
*/
int main(int argc, char *argv[]) {

  int ret, l, imode, np = 1;
  int mpmx_begin, mpmx_end, mpmy_begin, mpmy_end;
  double time;
  
  struct PARAMETERS PRM = { 0 };
  struct MEDIUM MDM = { 0 };
  struct ABSORBING_BOUNDARY_CONDITION ABC = { 0 };
  struct ANELASTICITY ANL = { 0 };
  struct OUTPUTS OUT = { 0 };

  struct SOURCE *SRC = NULL;
  struct VELOCITY *v0 = NULL;
  struct STRESS *t0 = NULL;

  if (argc < 2) {
    printf("Insira o tamanho do bloco a ser utilizado\n");
    return -1;
  }

  else {
    PRM.block_size = atoi(argv[1]);
    printf("Usando blocos de tamanho %dx%d\n", PRM.block_size, PRM.block_size);
  }

  /*ret = starpu_init(NULL);
  if (ret != 0) {
    printf("Erro inicializando StarPU\n");
    return -1;
    }*/
  PRM.px = 1;
  PRM.py = 1;
  VerifFunction(ReadPrmFile(&PRM, &MDM, &ABC, &ANL, &OUT, PRMFILE), "read parameter file ", PRM);
  printf("aqui o nlayer ta %d\n", MDM.nLayer);
  VerifFunction(ReadSrc(&SRC, PRM), "read sources file ", PRM);
  printf("depois do readsrc o nlayer ta %d\n", MDM.nLayer);
  VerifFunction(InitPartDomain(&PRM, &OUT), "split domain MPI", PRM);
  printf("depois do initpartdomain o nlayer ta %d\n", MDM.nLayer);
  
  printf("antes do readgeofile ta %d\n", MDM.nLayer);
  if (model == GEOLOGICAL) {
    VerifFunction(ReadGeoFile(&MDM, PRM), "read geological file", PRM);
  }

  VerifFunction(ReadStation(&OUT, PRM, MDM), "read station file ", PRM);
  printf("aqui o nlayer ta %d\n", MDM.nLayer);
  VerifFunction(AllocateFields2(&v0, &t0, &ANL, &ABC, &MDM, &SRC, PRM), "allocate Fields ", PRM);

  if (model == LAYER) {
    VerifFunction(InitLayerModel(&MDM, &ANL, PRM), "initialize layer model", PRM);
  }

  printf("passou!\n");

  /** initialize fields **/
  /* inside the domain */
  if (ANLmethod == DAYandBRADLEY) {
    VerifFunction(InitializeDayBradley(&MDM, &ANL, PRM), "initilize Day and BRADLEY", PRM);
  } else if (ANLmethod == KRISTEKandMOCZO) {
    VerifFunction(InitializeKManelas(&ANL, &MDM, PRM.dt), "initilize KRISTEK and MOCZO anelasticity", PRM);
  }
  
  /* in the absorbing layers */
  VerifFunction(InitializeABC(&ABC, &MDM, &ANL, PRM), "initilize absorbing boundaries ", PRM);
  
  if (model == GEOLOGICAL) {
#ifdef OUT_HOGE
    /* Checking the geological model : we write in a binary file */
    VerifFunction(OutGeol(MDM, OUT, PRM, HOGE), "check geological model", PRM);
#endif
    /* Computing the height of the stations */
    VerifFunction(InitializeGeol(&OUT, MDM, PRM), "initialize height station", PRM);
  }

  /* Allocation output */
  VerifFunction(InitializeOutputs(STATION_STEP, &OUT, PRM),
		" MapSeismograms", PRM);
  VerifFunction(EXIT_SUCCESS, "Beginning of the iteration", PRM);


  /*starpu_data_handle_t abc_handle, anl_handle, stress_handle;
  starpu_variable_data_register(&abc_handle, STARPU_MAIN_RAM, (uintptr_t)&ABC, sizeof(ABC));
  starpu_variable_data_register(&anl_handle, STARPU_MAIN_RAM, (uintptr_t)&ANL, sizeof(ANL));
  starpu_variable_data_register(&stress_handle, STARPU_MAIN_RAM, (uintptr_t)&t0, sizeof(t0));
  */
  
  
  for (l = 1; l <= PRM.tMax; l++) {
    time = PRM.dt * l;
    if (source == HISTFILE)
      computeSeisMoment(&SRC, time, PRM);

    /* Calculation */
    /* === First step : t = l + 1/2 for stress === */
    /* computation of intermediates :
       Phiv (CPML), t??? (PML), ksi (Day & Bradley), ksil (Kristek and Moczo) */
    //for (imode = 1; imode <= 5; imode++) {
    //  calculate_mappings(imode, PRM.mpmx, PRM.mpmy, &mpmx_begin, &mpmx_end, &mpmy_begin, &mpmy_end);
    //  //insert_compute_intermediates_task(abc_handle, anl_handle, v0, PRM, MDM, mpmx_begin, mpmx_end, mpmy_begin, mpmy_end);
    //  ComputeIntermediates(&ABC, &ANL, v0, PRM, MDM, mpmx_begin, mpmx_end, mpmy_begin, mpmy_end);
    //}
    //ComputeIntermediates(&ABC, &ANL, v0, PRM, MDM, 1, PRM.mpmx, 1, PRM.mpmy);

    // fazer o ABC e o ANL como data handles em R
    //for (imode = 1; imode <= 5; imode++) {
    //  calculate_mappings(imode, PRM.mpmx, PRM.mpmy, &mpmx_begin, &mpmx_end, &mpmy_begin, &mpmy_end);
    //  //insert_compute_stress(stress_handle, abc_handle, anl_handle, v0, MDM, PRM, mpmx_begin, mpmx_end, mpmy_begin, mpmy_end);
    //  ComputeStress(&t0, v0, MDM, PRM, ABC, ANL, mpmx_begin, mpmx_end, mpmy_begin, mpmy_end);
    //}

    //ComputeStress(&t0, v0, MDM, PRM, ABC, ANL, 1, PRM.mpmx, 1, PRM.mpmy);
    
    //dump(&v0, &t0, &ABC, &SRC, &MDM, PRM);
    exit(0);
  }

  
  //starpu_shutdown();
}
