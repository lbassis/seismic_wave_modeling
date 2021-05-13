#include <stdio.h>
#include <stdlib.h>

#include "../include/IO.h"
#include "../include/alloAndInit.h"
#include "../include/options.h"

int VerifFunction(int exitStatus, const char *msg, struct PARAMETERS PRM) {
  if (exitStatus != EXIT_SUCCESS) {
    fprintf(stderr, "%-50s [ERROR] with cpu %i \n", msg, PRM.me);
  }
}

void init_distribution_coords(struct PARAMETERS *prm) {
  prm->me = 0;
  prm->px = 1;
  prm->py = 1;
  prm->coords[0] = 0;
  prm->coords[1] = 0;
}

void calculate_mappings(int imode, int MPMX, int MPMY, int *mpmx_begin, int *mpmx_end, int *mpmy_begin, int *mpmy_end) {
  switch(imode) {
  case 1:
    &mpmx_begin = 1;
    &mpmx_end = 3;
    &mpmy_begin = 1;
    &mpmy_end = MPMY;
    break;
  case 2:
    &mpmx_begin = MPMX - 2;
    &mpmx_end = MPMX;
    &mpmy_begin = 1;
    &mpmy_end = MPMY;
    break;
  case 3:
    &mpmy_begin = 1;
    &mpmy_end = 3;
    &mpmx_begin = 4;
    &mpmx_end = MPMX - 3;
    break;
  case 4:
    &mpmy_begin = MPMY - 2;
    &mpmy_end = MPMY;
    &mpmx_begin = 4;
    &mpmx_end = MPMX - 3;
    break;
  case 5:
    &mpmx_begin = 4;
    &mpmx_end = MPMX - 3;
    &mpmy_begin = 4;
    &mpmy_end = MPMY - 3;
  }      
}

int main() {

  int np = 1;
  struct SOURCE SRC = { 0 };
  struct PARAMETERS PRM = { 0 };
  struct MEDIUM MDM = { 0 };
  struct ABSORBING_BOUNDARY_CONDITION ABC = { 0 };
  struct ANELASTICITY ANL = { 0 };
  struct OUTPUTS OUT = { 0 };

  struct VELOCITY v0 = { 0 };
  struct STRESS t0 = { 0 };

  init_distribution_coords(&PRM);
  
  VerifFunction(ReadPrmFile(&PRM, &MDM, &ABC, &ANL, &OUT, PRMFILE), "read parameter file ", PRM);
  VerifFunction(ReadSrc(&SRC, PRM), "read sources file ", PRM);
  VerifFunction(InitPartDomain(&PRM, &OUT), "split domain MPI", PRM);

  if (model == GEOLOGICAL) {
    VerifFunction(ReadGeoFile(&MDM, PRM), "read geological file", PRM);
  }

  VerifFunction(ReadStation(&OUT, PRM, MDM), "read station file ", PRM);
  VerifFunction(AllocateFields(&v0, &t0, &ANL, &ABC, &MDM, &SRC, PRM), "allocate Fields ", PRM);

  if (model == LAYER) {
    VerifFunction(InitLayerModel(&MDM, &ANL, PRM), "initialize layer model", PRM);
  }

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
}
