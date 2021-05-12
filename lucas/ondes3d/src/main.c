#include <stdio.h>
#include <stdlib.h>

#include "../include/IO.h"
#include "../include/options.h"

int VerifFunction(int exitStatus, const char *msg, struct PARAMETERS PRM) {
  if (exitStatus != EXIT_SUCCESS) {
    fprintf(stderr, "%-50s [ERROR] with cpu %i \n", msg, PRM.me);
  }
}

int main() {

  int fStatus;

  struct PARAMETERS PRM = { 0 };
  struct MEDIUM MDM = { 0 };
  struct ABSORBING_BOUNDARY_CONDITION ABC = { 0 };
  struct ANELASTICITY ANL = { 0 };
  struct OUTPUTS OUT = { 0 };

  fStatus = ReadPrmFile(&PRM, &MDM, &ABC, &ANL, &OUT, PRMFILE);
  VerifFunction(fStatus, "read parameter file ", PRM);
}
