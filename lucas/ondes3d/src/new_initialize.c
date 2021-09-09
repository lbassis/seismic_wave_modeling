#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "../include/nrutil.h"
#include "../include/struct.h"
#include "../include/inlineFunctions.h"
#include "../include/options.h"
#include "../include/alloAndInit.h"
#include "../include/new_initialize.h"


/* **********************
   PURPOSE : initialize CPML :
   step 1- extend coefficients to Absorbing and FreeAbs Borders ( FreeSurface is already computed )
   IDEA : dertermine the nearest "inside" cell,
   PML cell coeff= nearest "inside" cell
   step 2- compute PML/CPML coefficients

   ************************* */
int InitializeABC2(struct ABSORBING_BOUNDARY_CONDITION **ABCs,
		   struct MEDIUM *MDM,
		   struct ANELASTICITY *ANL, struct PARAMETERS PRM)
{
  /* mapping */
  const int XMIN = PRM.xMin;
  const int XMAX = PRM.xMax;
  const int YMIN = PRM.yMin;
  const int YMAX = PRM.yMax;
  const int ZMIN = PRM.zMin;
  const int ZMAX = PRM.zMax;
  const int ZMAX0 = PRM.zMax0;
  const int DELTA = PRM.delta;

  const int MPMX = PRM.mpmx;
  const int MPMY = PRM.mpmy;

  const double DS = PRM.ds;
    
  const int BLOCK_SIZE = PRM.block_size;
    
  /* step 1 */
  enum typePlace place;	/* What type of cell  */
  int i, j, k;		/* Global coordinates of the cell */
  int imp, jmp;		/* local  coordinates  */

  int iN, jN, kN;		/* Global coordinates of the Nearest cell */
  int impN, jmpN;		/* local versions */

  int icpu, jcpu;		/* coordinates of the cpu */

  /* step 2 */
  double xoriginleft, xoriginright, yoriginfront, yoriginback,
    zoriginbottom, zorigintop, xval, yval, zval, abscissa_in_PML,
    abscissa_normalized;

  int i_block, j_block, block_index;
  int nb_blocks_x = ceil((float)(XMAX - XMIN + 2 * DELTA + 3)/BLOCK_SIZE);
  int nb_blocks_y = ceil((float)(YMAX - YMIN + 2 * DELTA + 3)/BLOCK_SIZE);

   
  for (i_block = 0; i_block < nb_blocks_y; i_block++) {
    for (j_block = 0; j_block < nb_blocks_x; j_block++) {
      block_index = i_block*nb_blocks_x + j_block;

      /* ****************************************************** */
      /* Definition of the vectors used in the PML/CPML formulation */
      /* ****************************************************** */
      (*ABCs)[block_index].dumpx = dvector(1, BLOCK_SIZE);
      (*ABCs)[block_index].dumpx2 = dvector(1, BLOCK_SIZE);
      (*ABCs)[block_index].dumpy = dvector(1, BLOCK_SIZE);
      (*ABCs)[block_index].dumpy2 = dvector(1, BLOCK_SIZE);

      (*ABCs)[block_index].dumpz = dvector(ZMIN - DELTA, ZMAX0);
      (*ABCs)[block_index].dumpz2 = dvector(ZMIN - DELTA, ZMAX0);

      if (ABCmethod == CPML) {
	/* We use kappa, alpha even if we are not in CPML (ie : regular domain )
	 * In that case, they are chosen not to modify the derivatives,
	 * that is to say :
	 * dump = 0., kappa=1; alpha=0;
	 */
	(*ABCs)[block_index].kappax = dvector(1, BLOCK_SIZE);
	(*ABCs)[block_index].alphax = dvector(1, BLOCK_SIZE);
	(*ABCs)[block_index].kappax2 = dvector(1, BLOCK_SIZE);
	(*ABCs)[block_index].alphax2 = dvector(1, BLOCK_SIZE);
	(*ABCs)[block_index].kappay = dvector(1, BLOCK_SIZE);
	(*ABCs)[block_index].alphay = dvector(1, BLOCK_SIZE);
	(*ABCs)[block_index].kappay2 = dvector(1, BLOCK_SIZE);
	(*ABCs)[block_index].alphay2 = dvector(1, BLOCK_SIZE);

	(*ABCs)[block_index].kappaz = dvector(ZMIN - DELTA, ZMAX0);
	(*ABCs)[block_index].alphaz = dvector(ZMIN - DELTA, ZMAX0);
	(*ABCs)[block_index].kappaz2 = dvector(ZMIN - DELTA, ZMAX0);
	(*ABCs)[block_index].alphaz2 = dvector(ZMIN - DELTA, ZMAX0);
      }
      /*** Compute PML coefficients  ***/
      /* We compute the PML domain 
	 /* NB : when ABCmethod == PML, CompABCCoef will ignore alphai, kappai arguments */

      /*** initialize oefficients like you were in regular domain ***/
      for (imp = 1; imp <= BLOCK_SIZE; imp++) {
	(*ABCs)[block_index].dumpx[imp] = 0.0;
	(*ABCs)[block_index].dumpx2[imp] = 0.0;
	if (ABCmethod == CPML) {
	  (*ABCs)[block_index].kappax[imp] = 1.0;
	  (*ABCs)[block_index].kappax2[imp] = 1.0;
	  (*ABCs)[block_index].alphax[imp] = 0.0;
	  (*ABCs)[block_index].alphax2[imp] = 0.0;
	}
      }
      for (jmp = 1; jmp <= BLOCK_SIZE; jmp++) {
	(*ABCs)[block_index].dumpy[jmp] = 0.0;
	(*ABCs)[block_index].dumpy2[jmp] = 0.0;
	if (ABCmethod == CPML) {
	  (*ABCs)[block_index].kappay[jmp] = 1.0;
	  (*ABCs)[block_index].kappay2[jmp] = 1.0;
	  (*ABCs)[block_index].alphay[jmp] = 0.0;
	  (*ABCs)[block_index].alphay2[jmp] = 0.0;
	}
      }

      for (k = ZMIN - DELTA; k <= ZMAX0; k++) {
	(*ABCs)[block_index].dumpz[k] = 0.0;
	(*ABCs)[block_index].dumpz2[k] = 0.0;
	if (ABCmethod == CPML) {
	  (*ABCs)[block_index].kappaz[k] = 1.0;
	  (*ABCs)[block_index].kappaz2[k] = 1.0;
	  (*ABCs)[block_index].alphaz[k] = 0.0;
	  (*ABCs)[block_index].alphaz2[k] = 0.0;
	}
      }

      /* For the x axis */
      xoriginleft = XMIN * DS;
      xoriginright = XMAX * DS;
      for (imp = 1; imp <= BLOCK_SIZE; imp++) {
	i = PRM.imp2i_array[imp];
	xval = DS * (i - 1);

	if (i <= XMIN + 1) {	/* For the left side */
	  abscissa_in_PML = xoriginleft - xval;
	  CompABCCoef2((*ABCs)[block_index].dumpx, (*ABCs)[block_index].alphax, (*ABCs)[block_index].kappax,
		      imp, abscissa_in_PML, (*ABCs)[block_index], PRM);

	  abscissa_in_PML = xoriginleft - (xval + DS / 2.0);
	  CompABCCoef2((*ABCs)[block_index].dumpx2, (*ABCs)[block_index].alphax2, (*ABCs)[block_index].kappax2,
		      imp, abscissa_in_PML, (*ABCs)[block_index], PRM);
	}

	if (i >= XMAX + 1) {	/* For the right side */
	  abscissa_in_PML = xval - xoriginright;
	  CompABCCoef2((*ABCs)[block_index].dumpx, (*ABCs)[block_index].alphax, (*ABCs)[block_index].kappax,
		      imp, abscissa_in_PML, (*ABCs)[block_index], PRM);

	  abscissa_in_PML = xval + DS / 2.0 - xoriginright;
	  CompABCCoef2((*ABCs)[block_index].dumpx2, (*ABCs)[block_index].alphax2, (*ABCs)[block_index].kappax2,
		      imp, abscissa_in_PML, (*ABCs)[block_index], PRM);
	}

	if (ABCmethod == CPML) {	/* CPML */
	  if ((*ABCs)[block_index].alphax[imp] < 0.0)
	    (*ABCs)[block_index].alphax[imp] = 0.0;
	  if ((*ABCs)[block_index].alphax2[imp] < 0.0)
	    (*ABCs)[block_index].alphax2[imp] = 0.0;
	}

      }				/* end of imp */

      /* For the y axis */

      yoriginfront = YMIN * DS;
      yoriginback = YMAX * DS;

      for (jmp = 1; jmp <= BLOCK_SIZE; jmp++) {
	j = PRM.jmp2j_array[jmp];
	yval = DS * (j - 1);

	if (j <= YMIN + 1) {	/* For the front side */
	  abscissa_in_PML = yoriginfront - yval;
	  CompABCCoef2((*ABCs)[block_index].dumpy, (*ABCs)[block_index].alphay, (*ABCs)[block_index].kappay,
		      jmp, abscissa_in_PML, (*ABCs)[block_index], PRM);

	  abscissa_in_PML = yoriginfront - (yval + DS / 2.0);
	  CompABCCoef2((*ABCs)[block_index].dumpy2, (*ABCs)[block_index].alphay2, (*ABCs)[block_index].kappay2,
		      jmp, abscissa_in_PML, (*ABCs)[block_index], PRM);
	}
	if (j >= YMAX + 1) {	/* For the back side */
	  abscissa_in_PML = yval - yoriginback;
	  CompABCCoef2((*ABCs)[block_index].dumpy, (*ABCs)[block_index].alphay, (*ABCs)[block_index].kappay2,
		      jmp, abscissa_in_PML, (*ABCs)[block_index], PRM);

	  abscissa_in_PML = yval + DS / 2.0 - yoriginback;
	  CompABCCoef2((*ABCs)[block_index].dumpy2, (*ABCs)[block_index].alphay2, (*ABCs)[block_index].kappay2,
		      jmp, abscissa_in_PML, (*ABCs)[block_index], PRM);
	}
	if (ABCmethod == CPML) {	/* CPML */
	  if ((*ABCs)[block_index].alphay[jmp] < 0.0)
	    (*ABCs)[block_index].alphay[jmp] = 0.0;
	  if ((*ABCs)[block_index].alphay2[jmp] < 0.0)
	    (*ABCs)[block_index].alphay2[jmp] = 0.0;
	}

      }				/* end of jmp */

      /* For the z axis */
      /* NB : Free Surface means not compute the top side
       *
       */
      /* For the bottom side */
      zoriginbottom = ZMIN * DS;
      for (k = ZMIN - DELTA; k <= ZMIN + 1; k++) {
	zval = DS * (k - 1);
	abscissa_in_PML = zoriginbottom - zval;
	CompABCCoef2((*ABCs)[block_index].dumpz, (*ABCs)[block_index].alphaz, (*ABCs)[block_index].kappaz,
		    k, abscissa_in_PML, (*ABCs)[block_index], PRM);

	abscissa_in_PML = zoriginbottom - (zval + DS / 2.0);
	CompABCCoef2((*ABCs)[block_index].dumpz2, (*ABCs)[block_index].alphaz2, (*ABCs)[block_index].kappaz2,
		    k, abscissa_in_PML, (*ABCs)[block_index], PRM);
      }				/* end for k */

      /* For the top side */
      if (surface == ABSORBING) {	/* absorbing layer above z = ZMAX */
	zorigintop = ZMAX * DS;
	for (k = ZMAX + 1; k <= ZMAX0; k++) {
	  zval = DS * (k - 1);
	  abscissa_in_PML = zval - zorigintop;
	  CompABCCoef2((*ABCs)[block_index].dumpz, (*ABCs)[block_index].alphaz, (*ABCs)[block_index].kappaz,
		      k, abscissa_in_PML, (*ABCs)[block_index], PRM);

	  abscissa_in_PML = zval + DS / 2.0 - zorigintop;
	  CompABCCoef2((*ABCs)[block_index].dumpz2, (*ABCs)[block_index].alphaz2, (*ABCs)[block_index].kappaz2,
		      k, abscissa_in_PML, (*ABCs)[block_index], PRM);

	  if (ABCmethod == CPML) {	/* CPML */
	    if ((*ABCs)[block_index].alphaz[k] < 0.0)
	      (*ABCs)[block_index].alphaz[k] = 0.0;
	    if ((*ABCs)[block_index].alphaz2[k] < 0.0)
	      (*ABCs)[block_index].alphaz2[k] = 0.0;
	  }
	}			/* end of k */
      }				/* end surface == ABSORBING (top side) */
    }
  }

  return EXIT_SUCCESS;
}				/* end of CPML initialisations */

static void CompABCCoef2(	/* outputs */
			   double *dump, double *alpha, double *kappa,
			   /* inputs */
			   int imp,
			   double abscissa_in_PML,
			   struct ABSORBING_BOUNDARY_CONDITION ABC,
			   struct PARAMETERS PRM)
{
    double abscissa_normalized;
    if (abscissa_in_PML >= 0.0) {
	abscissa_normalized = abscissa_in_PML / (PRM.delta * PRM.ds);
	dump[imp] = ABC.dump0 * pow(abscissa_normalized, ABC.nPower);

	if (ABCmethod == CPML) {	/* CPML */
	    kappa[imp] =
		1.0 + (ABC.kappa0 - 1.0) * pow(abscissa_normalized,
					       ABC.nPower);
	    alpha[imp] = ABC.alpha0 * (1.0 - abscissa_normalized);
	}
    }
}				/* end function */


