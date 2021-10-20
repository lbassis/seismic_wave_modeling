#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "../include/new_nrutil.h"
#include "../include/struct.h"
#include "../include/inlineFunctions.h"
#include "../include/options.h"
#include "../include/alloAndInit.h"
/* ==================================== */
/* ALLOCATE and INITIALIZE PARTITIONNING OF DOMAIN */
/* ==================================== */
int InitPartDomain(struct PARAMETERS *PRM, struct OUTPUTS *OUT)
{
  /* mapping */
  const int XMIN = PRM->xMin;
  const int XMAX = PRM->xMax;
  const int YMIN = PRM->yMin;
  const int YMAX = PRM->yMax;
  const int ZMIN = PRM->zMin;
  const int ZMAX = PRM->zMax;
  const int ZMAX0 = PRM->zMax0;
  const int DELTA = PRM->delta;

  int MPMX;
  int MPMY;
  const int PX = PRM->px;
  const int PY = PRM->py;

  const double DS = PRM->ds;
  const double DT = PRM->dt;
  int i, j, k, icpu, jcpu;


  PRM->mpmx_tab = ivector(0, PX - 1);
  PRM->mpmy_tab = ivector(0, PY - 1);

  /* the domain goes from XMIN-DELTA to XMAX+DELTA+2
     XMIN-DELTA = CL  XMAX+DELTA+2 = CL
     hence the global size is XMAX+DELTA+2-XMIN+DELTA+1
     we part this domain */

  /* D�coupage : 1rst way */

  /**************************************************************************/
  //#if (DECOUP1)

  /* Here the +1 allow us to be sure to have  mpmx*np > global size of computed domain
     This implies that an outsize last domain and useless operations */
  /*
    PRM->mpmx = (XMAX - XMIN + 2*DELTA + 3 )/PX + 1;
    if ( mpmx <= 10 ){
    printf (" Reduire le nombre de processeurs utilises \n");
    exit(0);
    }
    for ( i = 0; i <= (PX-1); i++){ PRM->mpmx_tab[i] = mpmx ; }

    PRM->mpmy = (YMAX - YMIN + 2*DELTA + 3 )/PY + 1;
    if ( mpmy <= 10 ){
    printf (" Reduire le nombre de processeurs utilises \n");
    exit(0);
    }

    for ( i = 0; i <= (PY-1); i++){ PRM->mpmy_tab[i] = mpmy ; }
  */
  //#endif /* fin du premier d�coupage */

  /* D�coupage : 2nd way */

  /************************************************************************/
  //#if (DECOUP2)

  /* Ici on essaie d ajuster au plus juste les tailles de mpmx --> variables
     On decoupe de facon pessimiste et on ajuste progressivement
     Reste le probleme du cout specifique des CPMLs qui n est pas aborde */
  double mpmx_tmp1;
  double mpmy_tmp1;
  int difference;


  mpmx_tmp1 = (double) (XMAX - XMIN + 2 * DELTA + 3) / (double) PX;
  PRM->mpmx = floor(mpmx_tmp1);
  MPMX = PRM->mpmx;

  for (i = 0; i <= (PX - 1); i++) {
    PRM->mpmx_tab[i] = MPMX;
  }

  if (PX * MPMX < (XMAX - XMIN + 2 * DELTA + 3)) {
    difference = (XMAX - XMIN + 2 * DELTA + 3) - PX * MPMX;
    for (i = 1; i <= difference; i++)
      PRM->mpmx_tab[i] = MPMX + 1;
  }

  if (MPMX <= 10) {
    printf(" Reduce the number of processes used \n");
    exit(0);
  }

  mpmy_tmp1 = (double) (YMAX - YMIN + 2 * DELTA + 3) / (double) PY;
  PRM->mpmy = (int) floor(mpmy_tmp1);
  MPMY = PRM->mpmy;

  for (i = 0; i <= (PY - 1); i++)
    PRM->mpmy_tab[i] = MPMY;

  if (PY * MPMY < (YMAX - YMIN + 2 * DELTA + 3)) {
    difference = (YMAX - YMIN + 2 * DELTA + 3) - PY * MPMY;
    for (i = 1; i <= difference; i++)
      PRM->mpmy_tab[i] = MPMY + 1;
  }

  if (MPMY <= 10) {
    printf(" Reduce the number of processes used \n");
    exit(0);
  }
  //#endif /* fin du deuxi�me d�coupage */
  /********************************************************************/
  /* Fin du d�coupage */

  /* Allocate output matrix */

  int mpmx_max = 0;
  int mpmy_max = 0;
  for (i = 0; i < PX; i++) {
    if (PRM->mpmx_tab[i] > mpmx_max) {
      mpmx_max = PRM->mpmx_tab[i];
    }
  }
  for (i = 0; i < PY; i++) {
    if (PRM->mpmy_tab[i] > mpmy_max) {
      mpmy_max = PRM->mpmy_tab[i];
    }
  }

  OUT->test_size = IMAX(mpmx_max, mpmy_max);
  OUT->test_size = IMAX(OUT->test_size, (ZMAX0) - (ZMIN - DELTA) + 1);

  OUT->test_size = OUT->test_size * OUT->test_size + 1;
  OUT->snapBuff = mydvector0(1, OUT->test_size);

  /*  Verify the Partitionning */
  k = 0;
  for (i = 0; i < PX; i++) {
    k = k + PRM->mpmx_tab[i];
  }
  if (k < (XMAX - XMIN + 2 * DELTA + 3)) {
    printf(" Issue %i in the partitionning", 1);
    exit(0);
  }

  k = 0;
  for (i = 0; i < PY; i++) {
    k = k + PRM->mpmy_tab[i];
  }
  if (k < (YMAX - YMIN + 2 * DELTA + 3)) {
    printf(" Issue %i in the partitionning", 2);
    exit(0);
  }

  OUT->total_prec_x = 0;
  if (PRM->coords[0] == 0)
    OUT->total_prec_x = 0;
  if (PRM->coords[0] != 0) {
    for (i = 0; i < PRM->coords[0]; i++)
      OUT->total_prec_x += PRM->mpmx_tab[i];
  }

  OUT->total_prec_y = 0;
  if (PRM->coords[1] == 0)
    OUT->total_prec_y = 0;
  if (PRM->coords[1] != 0) {
    for (i = 0; i < PRM->coords[1]; i++)
      OUT->total_prec_y += PRM->mpmy_tab[i];
  }

  /* update mpmx and mpmy  */
  PRM->mpmx = PRM->mpmx_tab[PRM->coords[0]];
  PRM->mpmy = PRM->mpmy_tab[PRM->coords[1]];
  MPMX = PRM->mpmx;
  MPMY = PRM->mpmy;

  /* i2imp_array largement surdimmensionne pour supporter DECOUP1 */

  icpu = XMIN - DELTA;
  PRM->i2imp_array = ivector(XMIN - DELTA, XMAX + 2 * DELTA + 2);
  for (j = 1; j <= PX; j++) {
    for (i = 1; i <= PRM->mpmx_tab[j - 1]; i++) {
      ivector_access(PRM->i2imp_array, XMIN - DELTA, XMAX + 2 * DELTA + 2, icpu) = i;
      //PRM->i2imp_array[icpu - (XMIN - DELTA)] = i;
      icpu++;
    }
  }


  /* j2jmp_array largement surdimmensionne pour supporter DECOUP1 */

  jcpu = YMIN - DELTA;
  PRM->j2jmp_array = ivector(YMIN - DELTA, YMAX + 2 * DELTA + 2);

  for (j = 1; j <= PY; j++) {
    for (i = 1; i <= PRM->mpmy_tab[j - 1]; i++) {
      //PRM->j2jmp_array[jcpu] = i;
      ivector_access(PRM->j2jmp_array, YMIN - DELTA, YMAX + 2 * DELTA + 2, jcpu) = i;
      jcpu++;
    }
  }

  /* On veut s affranchir des anciennes fonctions imp2i */

  icpu = XMIN - DELTA;
  PRM->imp2i_array = ivector(-1, MPMX + 2);
  for (i = -1; i <= MPMX + 2; i++) {
    //PRM->imp2i_array[i] = XMIN - DELTA + OUT->total_prec_x + i - 1;
    ivector_access(PRM->imp2i_array, -1, MPMX + 2, i) = XMIN - DELTA + OUT->total_prec_x + i - 1;
  }

  jcpu = YMIN - DELTA;
  PRM->jmp2j_array = ivector(-1, MPMY + 2);
  for (i = -1; i <= MPMY + 2; i++)
    //PRM->jmp2j_array[i] = YMIN - DELTA + OUT->total_prec_y + i - 1;
    ivector_access(PRM->jmp2j_array, -1, MPMY + 2, i) = YMIN - DELTA + OUT->total_prec_y + i - 1;

  /* On veut s affranchir des anciennes fonctions i2icpu
     En fait i2icpu ne doit pas renvoyer le rang mais les coordonnees abs et ord
     sinon cela n a pas de sens en 2D */

  /* Ok ici en considerant icpu est abscisse */
  int idebut;
  int jdebut;

  icpu = 0;
  k = 0;
  PRM->i2icpu_array = ivector(XMIN - DELTA, XMAX + 2 * DELTA + 2);
  idebut = XMIN - DELTA;

  for (j = 0; j <= (PX - 1); j++) {
    for (i = 1; i <= PRM->mpmx_tab[j]; i++) {
      //PRM->i2icpu_array[idebut] = j;
      ivector_access(PRM->i2icpu_array, XMIN - DELTA, XMAX + 2 * DELTA + 2, idebut) = j;
      idebut++;
    }
  }

  /* Ordonnee */

  icpu = 0;
  k = 0;
  PRM->j2jcpu_array = ivector(YMIN - DELTA, YMAX + 2 * DELTA + 2);
  jdebut = YMIN - DELTA;

  for (j = 0; j <= (PY - 1); j++) {
    for (i = 1; i <= PRM->mpmy_tab[j]; i++) {
      //PRM->j2jcpu_array[jdebut] = j;
      ivector_access(PRM->j2jcpu_array, YMIN - DELTA, YMAX + 2 * DELTA + 2, jdebut) = j;
      jdebut++;
    }
  }

  PRM->nmaxx = (MPMY + 2 + 2) * (ZMAX0 - ZMIN + DELTA + 1) * 2;
  PRM->nmaxy = (MPMX + 2 + 2) * (ZMAX0 - ZMIN + DELTA + 1) * 2;


  return (EXIT_SUCCESS);

}				/* end Part Domain */


/* ================ */
/* ALLOCATE FIELDS  */
/* ================ */
int AllocateFields(struct VELOCITY *v0,
		   struct STRESS *t0,
		   struct ANELASTICITY *ANL,
		   struct ABSORBING_BOUNDARY_CONDITION *ABC,
		   struct MEDIUM *MDM,
		   struct SOURCE *SRC, struct PARAMETERS PRM)
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

  int nrows = PRM.mpmx + 4;
  int ncols = PRM.mpmy + 4;
  int n_blocks_x = ncols/PRM.block_size;
  int n_blocks_y = nrows/PRM.block_size;
  PRM.n_blocks_x = n_blocks_x;
  PRM.n_blocks_y = n_blocks_y;

  /* others */
  int i, j, k, imp, jmp;
  enum typePlace place;

#if (VERBOSE > 2)
  fprintf(stderr, "## VELOCITY\n ");
#endif
  /* V0 */
  v0->x = myd3tensor0(-1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
  v0->y = myd3tensor0(-1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
  v0->z = myd3tensor0(-1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);


  /* 1 - MPMX since no need of communication cells */
#if DEBUG_ALLO
  fprintf(stderr, "\n ## SOURCE\n ");
#endif
  if (source == HISTFILE) {
    SRC->fx = myd3tensor0(1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0);
    SRC->fy = myd3tensor0(1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0);
    SRC->fz = myd3tensor0(1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0);
  }				/* end of source = 1 */


#if (VERBOSE > 2)
  fprintf(stderr, "\n ## STRESS\n");
#endif
  /* T0 */
  t0->xx = myd3tensor0(-1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
  t0->yy = myd3tensor0(-1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
  t0->zz = myd3tensor0(-1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
  t0->xy = myd3tensor0(-1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
  t0->xz = myd3tensor0(-1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
  t0->yz = myd3tensor0(-1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);

#if (VERBOSE > 2)
  fprintf(stderr, "## BORDERS\n ");
#endif
  /* Velocity */
  ABC->nPower = NPOWER;

  ABC->ipml = i3tensor(-1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);

  ABC->phiv = (struct phiv_s*)malloc(PRM.n_blocks_y * PRM.n_blocks_x * sizeof(struct phiv_s));
  int depth = PRM.zMax0 - (PRM.zMin - PRM.delta) + 1;
  for (int i_block = 0; i_block < PRM.n_blocks_y; i_block++) {
    for (int j_block = 0; j_block < PRM.n_blocks_x; j_block++) {
      ABC->npmlv = 0;
      ABC->phiv[i_block * PRM.n_blocks_x + j_block].base_ptr = calloc(9 * PRM.block_size * PRM.block_size * depth, sizeof(double));
      ABC->phiv[i_block * PRM.n_blocks_x + j_block].size = 9 * PRM.block_size * PRM.block_size * depth;
      ABC->phiv[i_block * PRM.n_blocks_x + j_block].offset = PRM.block_size * PRM.block_size * depth;
      COMPUTE_ADDRESS_PHIV_S(ABC->phiv[i_block * PRM.n_blocks_x + j_block]);
      for(imp = -1 + PRM.block_size * i_block; imp < PRM.block_size * (i_block+1); imp++ ){
        i = ivector_access(PRM.imp2i_array, -1, MPMX + 2, imp);
        for(jmp = -1 + PRM.block_size * j_block; jmp < PRM.block_size * (j_block+1); jmp++ ){
          j = ivector_access(PRM.jmp2j_array, -1, MPMY + 2, jmp);
          for (k = ZMIN - DELTA; k <= ZMAX0; k++) {
            i3access(ABC->ipml, -1, MPMX+2, -1, MPMY+2, ZMIN-DELTA, ZMAX0, imp, jmp, k) = -1;
            place = WhereAmI(i, j, k, PRM);
            if (place == ABSORBINGLAYER || place == FREEABS) {
              ABC->npmlv += 1;
              int* ppp = &i3access(ABC->ipml, -1, MPMX+2, -1, MPMY+2, ZMIN-DELTA, ZMAX0, imp, jmp, k);
              *ppp = ABC->npmlv;

            }
          }
        }
      }
    }
  }


  if (PRM.me == 0) {
    if (ABCmethod == CPML)
      printf("\nNumber of points in the CPML : %li\n", ABC->npmlv);
    if (ABCmethod == PML)
      printf("\nNumber of points in the PML : %li\n", ABC->npmlt);

  }

  /* Stress */
  ABC->npmlt = ABC->npmlv;

  if (ABCmethod == CPML) {

    ABC->phit = (struct phit_s*)malloc(PRM.n_blocks_y * PRM.n_blocks_x * sizeof(struct phit_s));
    for (int i_block = 0; i_block < PRM.n_blocks_y; i_block++) {
      for (int j_block = 0; j_block < PRM.n_blocks_x; j_block++) {
	ABC->phit[i_block * PRM.n_blocks_x + j_block].base_ptr = calloc(9 * PRM.block_size * PRM.block_size * depth, sizeof(double));
	ABC->phit[i_block * PRM.n_blocks_x + j_block].size = 9 * PRM.block_size * PRM.block_size * depth;
	ABC->phit[i_block * PRM.n_blocks_x + j_block].offset = PRM.block_size * PRM.block_size * depth;
      }
    }
  }				/* end of if PML */

#if (VERBOSE > 2)
  fprintf(stderr, "## MEDIUM ");
#endif
  if (model == LAYER) {
    MDM->k2ly0 = ivector(ZMIN - DELTA, ZMAX0);
    MDM->k2ly2 = ivector(ZMIN - DELTA, ZMAX0);
  }

  return (EXIT_SUCCESS);
}				/* end allocateFields */


/* =================================== */
/* INITIALIZE COMMUNICATIONS BUFFERS   */
/* =================================== */
int InitializeCOMM(struct COMM_DIRECTION *NORTH,	/* no segfault */
		   struct COMM_DIRECTION *SOUTH,
		   struct COMM_DIRECTION *EAST,
		   struct COMM_DIRECTION *WEST,
		   /* inputs */
		   int nnorth,
		   int nsouth, int neast, int nwest, struct PARAMETERS PRM)
{
  enum typePlace place;
  int i;
  /* mapping */
  const int PX = PRM.px;
  const int PY = PRM.py;
  const int MPMX = PRM.mpmx;
  const int MPMY = PRM.mpmy;
  const int NMAXX = PRM.nmaxx;
  const int NMAXY = PRM.nmaxy;

  /* construct COM_DIRECTION (execpt channels) */
  /* north */
  NORTH->nmax = NMAXY;
  NORTH->rank = nnorth;

  NORTH->iMinS = -1;
  NORTH->iMaxS = MPMX + 2;
  NORTH->jMinS = MPMY - 1;
  NORTH->jMaxS = MPMY;
  NORTH->bufV0S = mydvector0(0, (3 * NMAXY) - 1);
  NORTH->bufT0S = mydvector0(0, (6 * NMAXY) - 1);

  NORTH->iMinR = -1;
  NORTH->iMaxR = MPMX + 2;
  NORTH->jMinR = MPMY + 1;
  NORTH->jMaxR = MPMY + 2;
  NORTH->bufV0R = mydvector0(0, (3 * NMAXY) - 1);
  NORTH->bufT0R = mydvector0(0, (6 * NMAXY) - 1);

  /* South */
  SOUTH->nmax = NMAXY;
  SOUTH->rank = nsouth;

  SOUTH->iMinS = -1;
  SOUTH->iMaxS = MPMX + 2;
  SOUTH->jMinS = 1;
  SOUTH->jMaxS = 2;
  SOUTH->bufV0S = mydvector0(0, (3 * NMAXY) - 1);
  SOUTH->bufT0S = mydvector0(0, (6 * NMAXY) - 1);

  SOUTH->iMinR = -1;
  SOUTH->iMaxR = MPMX + 2;
  SOUTH->jMinR = -1;
  SOUTH->jMaxR = 0;
  SOUTH->bufV0R = mydvector0(0, (3 * NMAXY) - 1);
  SOUTH->bufT0R = mydvector0(0, (6 * NMAXY) - 1);

  if (ANLmethod == KRISTEKandMOCZO) {
    SOUTH->bufKsiS = mydvector0(0, (6 * NMAXY) - 1);
    SOUTH->bufKsiR = mydvector0(0, (6 * NMAXY) - 1);
  }

  /* EAST */
  EAST->nmax = NMAXX;
  EAST->rank = neast;

  EAST->iMinS = MPMX - 1;
  EAST->iMaxS = MPMX;
  EAST->jMinS = -1;
  EAST->jMaxS = MPMY + 2;
  EAST->bufV0S = mydvector0(0, (3 * NMAXX) - 1);
  EAST->bufT0S = mydvector0(0, (6 * NMAXX) - 1);

  EAST->iMinR = MPMX + 1;
  EAST->iMaxR = MPMX + 2;
  EAST->jMinR = -1;
  EAST->jMaxR = MPMY + 2;
  EAST->bufV0R = mydvector0(0, (3 * NMAXX) - 1);
  EAST->bufT0R = mydvector0(0, (6 * NMAXX) - 1);

  if (ANLmethod == KRISTEKandMOCZO) {
    EAST->bufKsiR = mydvector0(0, (6 * NMAXX) - 1);
    EAST->bufKsiS = mydvector0(0, (6 * NMAXX) - 1);
  }

  /* WEST */
  WEST->nmax = NMAXX;
  WEST->rank = nwest;

  WEST->iMinS = 1;
  WEST->iMaxS = 2;
  WEST->jMinS = -1;
  WEST->jMaxS = MPMY + 2;
  WEST->bufV0S = mydvector0(0, (3 * NMAXX) - 1);
  WEST->bufT0S = mydvector0(0, (6 * NMAXX) - 1);

  WEST->iMinR = -1;
  WEST->iMaxR = 0;
  WEST->jMinR = -1;
  WEST->jMaxR = MPMY + 2;
  WEST->bufV0R = mydvector0(0, (3 * NMAXX) - 1);
  WEST->bufT0R = mydvector0(0, (6 * NMAXX) - 1);

  if (ANLmethod == KRISTEKandMOCZO) {
    WEST->bufKsiR = mydvector0(0, (6 * NMAXX) - 1);
    WEST->bufKsiS = mydvector0(0, (6 * NMAXX) - 1);
  }

  /** choose channels **/

  /* stress */
  i = 0;
  if (PRM.coords[0] % 2 == 0) {
    EAST->channelT0S = 3 + i * 8;
    WEST->channelT0R = 4 + i * 8;

    WEST->channelT0S = 7 + i * 8;
    EAST->channelT0R = 8 + i * 8;
  } else if (PRM.coords[0] % 2 == 1) {
    WEST->channelT0R = 3 + i * 8;
    EAST->channelT0S = 4 + i * 8;

    EAST->channelT0R = 7 + i * 8;
    WEST->channelT0S = 8 + i * 8;
  }

  if (PRM.coords[1] % 2 == 0) {
    NORTH->channelT0S = 2 + i * 8;
    SOUTH->channelT0R = 1 + i * 8;

    SOUTH->channelT0S = 5 + i * 8;
    NORTH->channelT0R = 6 + i * 8;
  } else if (PRM.coords[1] % 2 == 1) {
    SOUTH->channelT0R = 2 + i * 8;
    NORTH->channelT0S = 1 + i * 8;

    NORTH->channelT0R = 5 + i * 8;
    SOUTH->channelT0S = 6 + i * 8;
  }

  /* velocity */
  i = i + 1;
  if (PRM.coords[0] % 2 == 0) {
    EAST->channelV0S = 3 + i * 8;
    WEST->channelV0R = 4 + i * 8;

    WEST->channelV0S = 7 + i * 8;
    EAST->channelV0R = 8 + i * 8;
  } else if (PRM.coords[0] % 2 == 1) {
    WEST->channelV0R = 3 + i * 8;
    EAST->channelV0S = 4 + i * 8;

    EAST->channelV0R = 7 + i * 8;
    WEST->channelV0S = 8 + i * 8;
  }

  if (PRM.coords[1] % 2 == 0) {
    NORTH->channelV0S = 2 + i * 8;
    SOUTH->channelV0R = 1 + i * 8;

    SOUTH->channelV0S = 5 + i * 8;
    NORTH->channelV0R = 6 + i * 8;
  } else if (PRM.coords[1] % 2 == 1) {
    SOUTH->channelV0R = 2 + i * 8;
    NORTH->channelV0S = 1 + i * 8;

    NORTH->channelV0R = 5 + i * 8;
    SOUTH->channelV0S = 6 + i * 8;
  }

  if (ANLmethod == KRISTEKandMOCZO) {	/* ksil */
    i = i + 1;
    if (PRM.coords[0] % 2 == 0) {
      EAST->channelKsiS = 3 + i * 8;
      WEST->channelKsiR = 4 + i * 8;

      WEST->channelKsiS = 7 + i * 8;
      EAST->channelKsiR = 8 + i * 8;
    } else if (PRM.coords[0] % 2 == 1) {
      WEST->channelKsiR = 3 + i * 8;
      EAST->channelKsiS = 4 + i * 8;

      EAST->channelKsiR = 7 + i * 8;
      WEST->channelKsiS = 8 + i * 8;
    }

    if (PRM.coords[1] % 2 == 0) {
      NORTH->channelKsiS = 2 + i * 8;
      SOUTH->channelKsiR = 1 + i * 8;

      SOUTH->channelKsiS = 5 + i * 8;
      NORTH->channelKsiR = 6 + i * 8;
    } else if (PRM.coords[1] % 2 == 1) {
      SOUTH->channelKsiR = 2 + i * 8;
      NORTH->channelKsiS = 1 + i * 8;

      NORTH->channelKsiR = 5 + i * 8;
      SOUTH->channelKsiS = 6 + i * 8;
    }
  }



  return EXIT_SUCCESS;
}				/* end InitCOMM */

/* **********************
   PURPOSE : initialize CPML :
   step 1- extend coefficients to Absorbing and FreeAbs Borders ( FreeSurface is already computed )
   IDEA : dertermine the nearest "inside" cell,
   PML cell coeff= nearest "inside" cell
   step 2- compute PML/CPML coefficients

   ************************* */
int InitializeABC(struct ABSORBING_BOUNDARY_CONDITION *ABC,
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

  /* ****************************************************** */
  /* Definition of the vectors used in the PML/CPML formulation */
  /* ****************************************************** */
  ABC->dumpx = dvector(1, MPMX);
  printf("dumpx com tamanho %d\n", (MPMX - 1 + 1 + NR_END));
  ABC->dumpx2 = dvector(1, MPMX);
  ABC->dumpy = dvector(1, MPMY);
  ABC->dumpy2 = dvector(1, MPMY);

  ABC->dumpz = dvector(ZMIN - DELTA, ZMAX0);
  ABC->dumpz2 = dvector(ZMIN - DELTA, ZMAX0);

  if (ABCmethod == CPML) {
    /* We use kappa, alpha even if we are not in CPML (ie : regular domain )
     * In that case, they are chosen not to modify the derivatives,
     * that is to say :
     * dump = 0., kappa=1; alpha=0;
     */
    ABC->kappax = dvector(1, MPMX);
    ABC->alphax = dvector(1, MPMX);
    ABC->kappax2 = dvector(1, MPMX);
    ABC->alphax2 = dvector(1, MPMX);
    ABC->kappay = dvector(1, MPMY);
    ABC->alphay = dvector(1, MPMY);
    ABC->kappay2 = dvector(1, MPMY);
    ABC->alphay2 = dvector(1, MPMY);

    ABC->kappaz = dvector(ZMIN - DELTA, ZMAX0);
    ABC->alphaz = dvector(ZMIN - DELTA, ZMAX0);
    ABC->kappaz2 = dvector(ZMIN - DELTA, ZMAX0);
    ABC->alphaz2 = dvector(ZMIN - DELTA, ZMAX0);
  }
  /*** Compute PML coefficients  ***/
  /* We compute the PML domain
     /* NB : when ABCmethod == PML, CompABCCoef will ignore alphai, kappai arguments */

  /*** initialize oefficients like you were in regular domain ***/
  for (imp = 1; imp <= MPMX; imp++) {
    ivector_access(ABC->dumpx, 1, MPMX, imp) = 0.0;
    ivector_access(ABC->dumpx2, 1, MPMX, imp) = 0.0;
    if (ABCmethod == CPML) {
      ivector_access(ABC->kappax, 1, MPMX, imp) = 1.0;
      ivector_access(ABC->kappax2, 1, MPMX, imp) = 1.0;
      ivector_access(ABC->alphax, 1, MPMX, imp) = 0.0;
      ivector_access(ABC->alphax2, 1, MPMX, imp) = 0.0;
    }
  }
  for (jmp = 1; jmp <= MPMY; jmp++) {
    ivector_access(ABC->dumpy, 1, MPMY, jmp) = 0.0;
    ivector_access(ABC->dumpy2, 1, MPMY, jmp) = 0.0;
    if (ABCmethod == CPML) {
      ivector_access(ABC->kappay, 1, MPMY, jmp) = 1.0;
      ivector_access(ABC->kappay2, 1, MPMY, jmp) = 1.0;
      ivector_access(ABC->alphay, 1, MPMY, jmp) = 0.0;
      ivector_access(ABC->alphay2, 1, MPMY, jmp) = 0.0;
    }
  }

  for (k = ZMIN - DELTA; k <= ZMAX0; k++) {
    ivector_access(ABC->dumpz, ZMIN-DELTA, ZMAX0, k) = 0.0;
    ivector_access(ABC->dumpz2, ZMIN-DELTA, ZMAX0, k) = 0.0;
    if (ABCmethod == CPML) {
      ivector_access(ABC->kappaz, ZMIN-DELTA, ZMAX0, k) = 1.0;
      ivector_access(ABC->kappaz2, ZMIN-DELTA, ZMAX0, k) = 1.0;
      ivector_access(ABC->alphaz, ZMIN-DELTA, ZMAX0, k) = 0.0;
      ivector_access(ABC->alphaz2, ZMIN-DELTA, ZMAX0, k) = 0.0;
    }
  }

  /* For the x axis */
  xoriginleft = XMIN * DS;
  xoriginright = XMAX * DS;
  for (imp = 1; imp <= MPMX; imp++) {
    //i = PRM.imp2i_array[imp];
    i = ivector_access(PRM.imp2i_array, -1, MPMX + 2, imp);
    xval = DS * (i - 1);

    if (i <= XMIN + 1) {	/* For the left side */
      abscissa_in_PML = xoriginleft - xval;
      CompABCCoef(ABC->dumpx, ABC->alphax, ABC->kappax,
		  imp, abscissa_in_PML, 1, PRM.mpmx, *ABC, PRM);

      abscissa_in_PML = xoriginleft - (xval + DS / 2.0);
      CompABCCoef(ABC->dumpx2, ABC->alphax2, ABC->kappax2,
		  imp, abscissa_in_PML, 1, PRM.mpmx, *ABC, PRM);
    }

    if (i >= XMAX + 1) {	/* For the right side */
      abscissa_in_PML = xval - xoriginright;
      CompABCCoef(ABC->dumpx, ABC->alphax, ABC->kappax,
		  imp, abscissa_in_PML, 1, PRM.mpmx, *ABC, PRM);

      abscissa_in_PML = xval + DS / 2.0 - xoriginright;
      CompABCCoef(ABC->dumpx2, ABC->alphax2, ABC->kappax2,
		  imp, abscissa_in_PML, 1, PRM.mpmx, *ABC, PRM);
    }

    if (ABCmethod == CPML) {	/* CPML */
      if (ivector_access(ABC->alphax, 1, MPMX, imp) < 0.0)
	ivector_access(ABC->alphax, 1, MPMX, imp) = 0.0;
      if (ivector_access(ABC->alphax2, 1, MPMX, imp) < 0.0)
        ivector_access(ABC->alphax2, 1, MPMX, imp) = 0.0;
    }

  }				/* end of imp */

  /* For the y axis */

  yoriginfront = YMIN * DS;
  yoriginback = YMAX * DS;

  for (jmp = 1; jmp <= MPMY; jmp++) {
    //j = PRM.jmp2j_array[jmp];
    j = ivector_access(PRM.jmp2j_array, -1, MPMY + 2, jmp);
    yval = DS * (j - 1);

    if (j <= YMIN + 1) {	/* For the front side */
      abscissa_in_PML = yoriginfront - yval;
      CompABCCoef(ABC->dumpy, ABC->alphay, ABC->kappay,
		  jmp, abscissa_in_PML, 1, PRM.mpmy, *ABC, PRM);

      abscissa_in_PML = yoriginfront - (yval + DS / 2.0);
      CompABCCoef(ABC->dumpy2, ABC->alphay2, ABC->kappay2,
		  jmp, abscissa_in_PML, 1, PRM.mpmy, *ABC, PRM);
    }
    if (j >= YMAX + 1) {	/* For the back side */
      abscissa_in_PML = yval - yoriginback;
      CompABCCoef(ABC->dumpy, ABC->alphay, ABC->kappay2,
		  jmp, abscissa_in_PML, 1, PRM.mpmy, *ABC, PRM);

      abscissa_in_PML = yval + DS / 2.0 - yoriginback;
      CompABCCoef(ABC->dumpy2, ABC->alphay2, ABC->kappay2,
		  jmp, abscissa_in_PML, 1, PRM.mpmy, *ABC, PRM);
    }
    if (ABCmethod == CPML) {	/* CPML */
      if (ivector_access(ABC->alphay, 1, MPMY, jmp) < 0.0)
	ivector_access(ABC->alphay, 1, MPMY, jmp) = 0.0;
      if (ivector_access(ABC->alphay2, 1, MPMY, jmp) < 0.0)
        ivector_access(ABC->alphay2, 1, MPMY, jmp) = 0.0;
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
    CompABCCoef(ABC->dumpz, ABC->alphaz, ABC->kappaz,
		k, abscissa_in_PML, PRM.zMin - PRM.delta, PRM.zMax0, *ABC, PRM);

    abscissa_in_PML = zoriginbottom - (zval + DS / 2.0);
    CompABCCoef(ABC->dumpz2, ABC->alphaz2, ABC->kappaz2,
		k, abscissa_in_PML, PRM.zMin - PRM.delta, PRM.zMax0, *ABC, PRM);
  }				/* end for k */

  /* For the top side */
  if (surface == ABSORBING) {	/* absorbing layer above z = ZMAX */
    zorigintop = ZMAX * DS;
    for (k = ZMAX + 1; k <= ZMAX0; k++) {
      zval = DS * (k - 1);
      abscissa_in_PML = zval - zorigintop;
      CompABCCoef(ABC->dumpz, ABC->alphaz, ABC->kappaz,
		  k, abscissa_in_PML, PRM.zMin - PRM.delta, PRM.zMax0, *ABC, PRM);

      abscissa_in_PML = zval + DS / 2.0 - zorigintop;
      CompABCCoef(ABC->dumpz2, ABC->alphaz2, ABC->kappaz2,
		  k, abscissa_in_PML, PRM.zMin - PRM.delta, PRM.zMax0, *ABC, PRM);

      if (ABCmethod == CPML) {	/* CPML */
	if (ABC->alphaz[k] < 0.0)
	  ABC->alphaz[k] = 0.0;
	if (ABC->alphaz2[k] < 0.0)
	  ABC->alphaz2[k] = 0.0;
      }
    }			/* end of k */
  }				/* end surface == ABSORBING (top side) */


  return EXIT_SUCCESS;

}				/* end of CPML initialisations */

/* SMALL FUNCTIONS related */
static void CompABCCoef(	/* outputs */
			double *dump, double *alpha, double *kappa,
			/* inputs */
			int imp,
			double abscissa_in_PML,
			int min_index,
			int max_index,
			struct ABSORBING_BOUNDARY_CONDITION ABC,
			struct PARAMETERS PRM)
{
  double abscissa_normalized;
  if (abscissa_in_PML >= 0.0) {
    abscissa_normalized = abscissa_in_PML / (PRM.delta * PRM.ds);

    ivector_access(dump, min_index, max_index, imp) = ABC.dump0 * pow(abscissa_normalized, ABC.nPower);

    if (ABCmethod == CPML) {	/* CPML */
      ivector_access(kappa, min_index, max_index, imp) =
	1.0 + (ABC.kappa0 - 1.0) * pow(abscissa_normalized,
				       ABC.nPower);
      ivector_access(alpha, min_index, max_index, imp) = ABC.alpha0 * (1.0 - abscissa_normalized);
    }
  }
}				/* end function */

//int InitializeOutputs(int STATION_STEP,
//		      struct OUTPUTS *OUT, struct PARAMETERS PRM)
//{
//    /* ===Seismograms related === */
//    int i, j, k;
//    int i2, j2, k2;
//    int ir;
//    int icpu, jcpu;
//    int icpu2, jcpu2;
//    int icpuEnd, jcpuEnd;
//    /* mapping */
//    const int PX = PRM.px;
//
//    OUT->mapping_seis = imatrix(0, OUT->iObs - 1, 1, 9);
//    OUT->seis_output =
//	myd3tensor0(0, STATION_STEP - 1, 0, OUT->iObs - 1, 1, 9);
//    OUT->seis_buff = mydvector0(0, STATION_STEP - 1);
//    /* mapping cpu X direction (coords[0]) then y direction (coords[1])
//       rank = coords[0] + coords[1]*px */
//
//    /* For info :
//     * Vx component  : i, j
//     * Vy component  : i2, j2
//     * Vz component  : i2, j
//     * Tii component : i2, j
//     * Txy component : i, j2
//     * Txz component : i, j
//     * Tyz component : i2, j2
//     */
//
//    for (ir = 0; ir < OUT->iObs; ir++) {
//	if (OUT->ista[ir] == 1) {
//
//	    i = OUT->ixobs[ir];
//	    j = OUT->iyobs[ir];
//	    icpu = PRM.i2icpu_array[i];
//	    jcpu = PRM.j2jcpu_array[j];
//
//	    if (OUT->xobswt[ir] >= 0.5) {
//		i2 = i;
//	    } else {
//		i2 = i - 1;
//	    }
//	    if (OUT->yobswt[ir] >= 0.5) {
//		j2 = j;
//	    } else {
//		j2 = j - 1;
//	    }
//	    icpu2 = PRM.i2icpu_array[i2];
//	    jcpu2 = PRM.j2jcpu_array[j2];
//
//	    /* Vx component */
//	    OUT->mapping_seis[ir][1] = icpu + jcpu * PX;
//
//	    /* Vy component */
//	    OUT->mapping_seis[ir][2] = icpu2 + jcpu2 * PX;
//
//	    /* Vz component */
//	    OUT->mapping_seis[ir][3] = icpu2 + jcpu * PX;
//
//	    /* Tii component */
//
//	    OUT->mapping_seis[ir][4] = icpu2 + jcpu * PX;
//	    OUT->mapping_seis[ir][5] = icpu2 + jcpu * PX;
//	    OUT->mapping_seis[ir][6] = icpu2 + jcpu * PX;
//
//	    /* Txy component */
//	    OUT->mapping_seis[ir][7] = icpu2 + jcpu2 * PX;
//
//	    /* Txz component */
//	    OUT->mapping_seis[ir][8] = icpu + jcpu * PX;
//
//	    /* Tyz component */
//	    OUT->mapping_seis[ir][9] = icpu2 + jcpu2 * PX;
//
//	}			/* end of if ista */
//    }				/* end of ir */
//
//    /* ===Snapshots related === */
//    if (snapType == ODISPL || snapType == OBOTH) {
//	const int ZMIN = PRM.zMin;
//	const int ZMAX0 = PRM.zMax0;
//	const int DELTA = PRM.delta;
//	const int MPMX = PRM.mpmx;
//	const int MPMY = PRM.mpmy;
//
//	OUT->Uxy = myd3tensor0(1, 3, -1, MPMX + 2, -1, MPMY + 2);
//	OUT->Uxz = myd3tensor0(1, 3, -1, MPMX + 2, ZMIN - DELTA, ZMAX0);
//	OUT->Uyz = myd3tensor0(1, 3, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
//    }
//    return (EXIT_SUCCESS);
//}				/* end function */




int DeallocateAll(int STATION_STEP,
		  struct ANELASTICITY *ANL,
		  struct ABSORBING_BOUNDARY_CONDITION *ABC,
		  struct SOURCE *SRC,
		  struct MEDIUM *MDM,
		  struct STRESS *t0,
		  struct VELOCITY *v0,
		  struct OUTPUTS *OUT,
		  struct COMM_DIRECTION *NORTH,
		  struct COMM_DIRECTION *SOUTH,
		  struct COMM_DIRECTION *EAST,
		  struct COMM_DIRECTION *WEST, struct PARAMETERS *PRM)
{
  int step;

  /* mapping */
  const int XMIN = PRM->xMin;
  const int XMAX = PRM->xMax;
  const int YMIN = PRM->yMin;
  const int YMAX = PRM->yMax;
  const int ZMIN = PRM->zMin;
  const int ZMAX = PRM->zMax;
  const int ZMAX0 = PRM->zMax0;
  const int DELTA = PRM->delta;
  const int MPMX = PRM->mpmx;
  const int MPMY = PRM->mpmy;

  const int NLAYER = MDM->nLayer;
  const int NLAYERINT = MDM->nLayer - 1;

  const int NPMLV = ABC->npmlv;
  const int NPMLT = ABC->npmlt;

  struct COMM_DIRECTION *com;

  /*** anelasticity ***/
  if (ANLmethod == ANOTHER) {
    free(ANL->q0);
    free(ANL->amp);

    if (model == LAYER) {
      free(ANL);
    }
  }				/* end anelasticities */


  /*** Velocity ***/
  free(v0->x);
  free(v0->y);
  free(v0->z);

  /*** Stress ***/
  free(t0->xx);
  free(t0->yy);
  free(t0->zz);
  free(t0->xy);
  free(t0->xz);
  free(t0->yz);

  /*** Source ***/
  free(SRC->ixhypo);
  free(SRC->iyhypo);
  free(SRC->izhypo);
  free(SRC->insrc);

  if (source == HISTFILE) {

    free(SRC->strike);
    free(SRC->dip);
    free(SRC->rake);
    free(SRC->slip);
    free(SRC->xweight);
    free(SRC->yweight);
    free(SRC->zweight);

    free(SRC->vel);

    free(SRC->fx);
    free(SRC->fy);
    free(SRC->fz);
  }


  /*** MEDIUM  ***/
  free(MDM->laydep);
  free(MDM->rho0);
  free(MDM->mu0);
  free(MDM->kap0);

  if (model == LAYER) {
    free(MDM->rho2);
    free(MDM->mu2);
    free(MDM->kap2);
  }				/* end if model */

  /*** Absorbing Boundary Condition ***/
  free(ABC->ipml);

  free(ABC->dumpx);
  free(ABC->dumpx2);
  free(ABC->dumpy);
  free(ABC->dumpy2);
  free(ABC->dumpz);
  free(ABC->dumpz2);

  if (ABCmethod == CPML) {
    for (int i_block = 0; i_block < PRM->n_blocks_y; i_block++) {
      for (int j_block = 0; j_block < PRM->n_blocks_x; j_block++) {
        free(ABC->phiv[i_block * PRM->n_blocks_x + j_block].base_ptr);
      }
    }
    free(ABC->phiv);


    free(ABC->phitxxx);
    free(ABC->phitxyy);
    free(ABC->phitxzz);

    free(ABC->phitxyx);
    free(ABC->phityyy);
    free(ABC->phityzz);

    free(ABC->phitxzx);
    free(ABC->phityzy);
    free(ABC->phitzzz);

    free(ABC->kappax);
    free(ABC->kappax2);
    free(ABC->kappay);
    free(ABC->kappay2);
    free(ABC->kappaz);
    free(ABC->kappaz2);

    free(ABC->alphax);
    free(ABC->alphax2);
    free(ABC->alphay);
    free(ABC->alphay2);
    free(ABC->alphaz);
    free(ABC->alphaz2);
  }
  /*** Communications ***/
  for (step = 1; step <= 4; step++) {
    /* mapping */
    if (step == 1)
      com = NORTH;
    if (step == 2)
      com = SOUTH;
    if (step == 3)
      com = EAST;
    if (step == 4)
      com = WEST;

    free(com->bufV0S);
    free(com->bufV0R);

    free(com->bufT0S);
    free(com->bufT0R);

    /* remove mapping */
    com = NULL;
  }


  /*** parameters ***/
  free(PRM->imp2i_array);
  free(PRM->jmp2j_array);

  free(PRM->i2imp_array);
  free(PRM->j2jmp_array);

  free(PRM->i2icpu_array);
  free(PRM->j2jcpu_array);

  /*** OUTPUTS ***/
  /* seismogramms */
  free(OUT->ixobs);
  free(OUT->iyobs);
  free(OUT->izobs);

  free(OUT->xobs);
  free(OUT->yobs);
  free(OUT->zobs);

  free(OUT->nobs);
  free(OUT->ista);

  free(OUT->xobswt);
  free(OUT->yobswt);
  free(OUT->zobswt);

  free(OUT->mapping_seis);
  free(OUT->seis_output);
  free(OUT->seis_buff);

  /* velocity planes */
  free(OUT->snapBuff);
  if (snapType == ODISPL || snapType == OBOTH) {
    free(OUT->Uxy);
    free(OUT->Uxz);
    free(OUT->Uyz);
  }
  OUT->Vxglobal = NULL;	/* already free */
  OUT->Vyglobal = NULL;
  OUT->Vzglobal = NULL;
  /* partition domain related */
  free(PRM->mpmx_tab);
  free(PRM->mpmy_tab);


  return EXIT_SUCCESS;
}
