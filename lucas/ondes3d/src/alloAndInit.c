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
	    PRM->i2imp_array[icpu] = i;
	    icpu++;
	}
    }


    /* j2jmp_array largement surdimmensionne pour supporter DECOUP1 */

    jcpu = YMIN - DELTA;
    PRM->j2jmp_array = ivector(YMIN - DELTA, YMAX + 2 * DELTA + 2);

    for (j = 1; j <= PY; j++) {
	for (i = 1; i <= PRM->mpmy_tab[j - 1]; i++) {
	    PRM->j2jmp_array[jcpu] = i;
	    jcpu++;
	}
    }

    /* On veut s affranchir des anciennes fonctions imp2i */

    icpu = XMIN - DELTA;
    PRM->imp2i_array = ivector(-1, MPMX + 2);
    for (i = -1; i <= MPMX + 2; i++) {
	PRM->imp2i_array[i] = XMIN - DELTA + OUT->total_prec_x + i - 1;
    }

    jcpu = YMIN - DELTA;
    PRM->jmp2j_array = ivector(-1, MPMY + 2);
    for (i = -1; i <= MPMY + 2; i++)
	PRM->jmp2j_array[i] = YMIN - DELTA + OUT->total_prec_y + i - 1;

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
	    PRM->i2icpu_array[idebut] = j;

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
	    PRM->j2jcpu_array[jdebut] = j;
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
    ABC->npmlv = 0;
    ABC->ipml = i3tensor(-1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);

    for (imp = -1; imp <= MPMX + 2; imp++) {
	i = PRM.imp2i_array[imp];
	for (jmp = -1; jmp <= MPMY + 2; jmp++) {
	    j = PRM.jmp2j_array[jmp];
	    for (k = ZMIN - DELTA; k <= ZMAX0; k++) {
	      TENSOR_POSITION(ABC->ipml, imp, jmp, k, -1, -1, MPMY+2, ZMIN-DELTA, ZMAX0) = -1;

		place = WhereAmI(i, j, k, PRM);
		if (place == ABSORBINGLAYER || place == FREEABS) {
		    ABC->npmlv += 1;
		    TENSOR_POSITION(ABC->ipml, imp, jmp, k, -1, -1, MPMY+2, ZMIN-DELTA, ZMAX0) = ABC->npmlv;
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

    if (ABCmethod == CPML) {

	ABC->phivxx = mydvector0(1, ABC->npmlv);
	ABC->phivxy = mydvector0(1, ABC->npmlv);
	ABC->phivxz = mydvector0(1, ABC->npmlv);

	ABC->phivyx = mydvector0(1, ABC->npmlv);
	ABC->phivyy = mydvector0(1, ABC->npmlv);
	ABC->phivyz = mydvector0(1, ABC->npmlv);

	ABC->phivzx = mydvector0(1, ABC->npmlv);
	ABC->phivzy = mydvector0(1, ABC->npmlv);
	ABC->phivzz = mydvector0(1, ABC->npmlv);

    }				/* end of if PML */

    /* Stress */
    ABC->npmlt = ABC->npmlv;

    if (ABCmethod == CPML) {

	ABC->phitxxx = mydvector0(1, ABC->npmlt);
	ABC->phitxyy = mydvector0(1, ABC->npmlt);
	ABC->phitxzz = mydvector0(1, ABC->npmlt);
	ABC->phitxyx = mydvector0(1, ABC->npmlt);
	ABC->phityyy = mydvector0(1, ABC->npmlt);
	ABC->phityzz = mydvector0(1, ABC->npmlt);
	ABC->phitxzx = mydvector0(1, ABC->npmlt);
	ABC->phityzy = mydvector0(1, ABC->npmlt);
	ABC->phitzzz = mydvector0(1, ABC->npmlt);

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

    if (model == GEOLOGICAL) {

	for (imp = -1; imp <= MPMX + 2; imp++) {
	    i = PRM.imp2i_array[imp];
	    for (jmp = -1; jmp <= MPMY + 2; jmp++) {
		j = PRM.jmp2j_array[jmp];
		for (k = ZMIN - DELTA; k <= ZMAX0; k++) {
		    place = WhereAmI(i, j, k, PRM);

		    if (place != REGULAR) {


			/* ========================== */
			/* Find the nearest cell */
			/* ========================== */
			/* Warnings:
			   - domain "inside" computed is :
			   [XMIN+2,XMAX]x[YMIN+2,YMAX]x[ZMIN+2,ZMAX]
			 */
			iN = i;
			jN = j;
			kN = k;
			/* EAST */
			if (i < XMIN + 2) {
			    iN = XMIN + 2;
			}
			/* WEST */
			if (i > XMAX) {
			    iN = XMAX;
			}
			/* SOUTH */
			if (j < YMIN + 2) {
			    jN = YMIN + 2;
			}
			/* NORTH */
			if (j > YMAX) {
			    jN = YMAX;
			}
			/* DOWN */
			if (k < ZMIN + 2) {
			    kN = ZMIN + 2;
			}
			/* UP */
			if (k > ZMAX) {
			    kN = ZMAX;
			}


			/* ======================== */
			/* GLOBAL TO LOCAL INDEX    */
			/* ======================== */
			/* Warning :
			 * each cpu have 2 cells of communications for each direction. [-1,0] and [mpm{x/y}+1,mpm{x/y}+2]
			 * Meanwhile, i2cpu_array/i2imp_array give only 1 possibility ( the other one may be the cells of communication ).
			 * So if it don't match exactly, we try to use those cells, even if it is not quite robust.
			 * ( the 10 minimum cells for each CPU should prevents some cases for some "DELTA")
			 * A more robust method should be to make communications.
			 */
			/* X AXIS */
			icpu = PRM.i2icpu_array[iN];
			if (PRM.coords[0] == icpu) {	/* I am the right  cpu */
			    impN = PRM.i2imp_array[iN];
			} else {	/* Try to correct */
			    if (iN == PRM.imp2i_array[-1]) {
				impN = -1;
			    } else if (iN == PRM.imp2i_array[0]) {
				impN = 0;
			    } else if (iN == PRM.imp2i_array[MPMX + 1]) {
				impN = MPMX + 1;
			    } else if (iN == PRM.imp2i_array[MPMX + 2]) {
				impN = MPMX + 2;
			    } else {
				printf
				    ("Extend to CPML : wrong CPU X for i=%i imp=%i me=%i\n",
				     i, imp, PRM.me);
				impN = imp;
				exit(EXIT_FAILURE);
			    }
			}	/* end try to correct */

			/* Y AXIS */
			jcpu = PRM.j2jcpu_array[jN];
			if (PRM.coords[1] == jcpu) {	/* I am the right  cpu */
			    jmpN = PRM.j2jmp_array[jN];
			} else {	/* Try to correct */
			    if (jN == PRM.jmp2j_array[-1]) {
				jmpN = -1;
			    } else if (jN == PRM.jmp2j_array[0]) {
				jmpN = 0;
			    } else if (jN == PRM.jmp2j_array[MPMY + 1]) {
				jmpN = MPMY + 1;
			    } else if (jN == PRM.jmp2j_array[MPMY + 2]) {
				jmpN = MPMY + 2;
			    } else {
				printf
				    ("Extend to CPML : wrong CPU Y for j=%i jmp=%i me=%i\n",
				     j, jmp, PRM.me);
				jmpN = jmp;
				exit(EXIT_FAILURE);
			    }
			}	/* end try to correct */
			/* ========== */
			/* ALLOCATE   */
			/* ========== */
			//MDM->imed[imp][jmp][k] = MDM->imed[impN][jmpN][kN];
      i3access(MDM->imed, -1, PRM.mpmx + 2, -1, PRM.mpmy + 2, PRM.zMin - PRM.delta, PRM.zMax0, imp, jmp, k) = \
      i3access(MDM->imed, -1, PRM.mpmx + 2, -1, PRM.mpmy + 2, PRM.zMin - PRM.delta, PRM.zMax0, impN, jmpN, kN);
		    }		/* end if no regular */

		}		/* end for k */
	    }			/* end for jmp */
	}			/* end for imp */

    }				/* end if model */

    /* ****************************************************** */
    /* Definition of the vectors used in the PML/CPML formulation */
    /* ****************************************************** */
    ABC->dumpx = dvector(1, MPMX);
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
	ABC->dumpx[imp] = 0.0;
	ABC->dumpx2[imp] = 0.0;
	if (ABCmethod == CPML) {
	    ABC->kappax[imp] = 1.0;
	    ABC->kappax2[imp] = 1.0;
	    ABC->alphax[imp] = 0.0;
	    ABC->alphax2[imp] = 0.0;
	}
    }
    for (jmp = 1; jmp <= MPMY; jmp++) {
	ABC->dumpy[jmp] = 0.0;
	ABC->dumpy2[jmp] = 0.0;
	if (ABCmethod == CPML) {
	    ABC->kappay[jmp] = 1.0;
	    ABC->kappay2[jmp] = 1.0;
	    ABC->alphay[jmp] = 0.0;
	    ABC->alphay2[jmp] = 0.0;
	}
    }

    for (k = ZMIN - DELTA; k <= ZMAX0; k++) {
	ABC->dumpz[k] = 0.0;
	ABC->dumpz2[k] = 0.0;
	if (ABCmethod == CPML) {
	    ABC->kappaz[k] = 1.0;
	    ABC->kappaz2[k] = 1.0;
	    ABC->alphaz[k] = 0.0;
	    ABC->alphaz2[k] = 0.0;
	}
    }

    /* For the x axis */
    xoriginleft = XMIN * DS;
    xoriginright = XMAX * DS;
    for (imp = 1; imp <= MPMX; imp++) {
	i = PRM.imp2i_array[imp];
	xval = DS * (i - 1);

	if (i <= XMIN + 1) {	/* For the left side */
	    abscissa_in_PML = xoriginleft - xval;
	    CompABCCoef(ABC->dumpx, ABC->alphax, ABC->kappax,
			imp, abscissa_in_PML, *ABC, PRM);

	    abscissa_in_PML = xoriginleft - (xval + DS / 2.0);
	    CompABCCoef(ABC->dumpx2, ABC->alphax2, ABC->kappax2,
			imp, abscissa_in_PML, *ABC, PRM);
	}

	if (i >= XMAX + 1) {	/* For the right side */
	    abscissa_in_PML = xval - xoriginright;
	    CompABCCoef(ABC->dumpx, ABC->alphax, ABC->kappax,
			imp, abscissa_in_PML, *ABC, PRM);

	    abscissa_in_PML = xval + DS / 2.0 - xoriginright;
	    CompABCCoef(ABC->dumpx2, ABC->alphax2, ABC->kappax2,
			imp, abscissa_in_PML, *ABC, PRM);
	}

	if (ABCmethod == CPML) {	/* CPML */
	    if (ABC->alphax[imp] < 0.0)
		ABC->alphax[imp] = 0.0;
	    if (ABC->alphax2[imp] < 0.0)
		ABC->alphax2[imp] = 0.0;
	}

    }				/* end of imp */

    /* For the y axis */

    yoriginfront = YMIN * DS;
    yoriginback = YMAX * DS;

    for (jmp = 1; jmp <= MPMY; jmp++) {
	j = PRM.jmp2j_array[jmp];
	yval = DS * (j - 1);

	if (j <= YMIN + 1) {	/* For the front side */
	    abscissa_in_PML = yoriginfront - yval;
	    CompABCCoef(ABC->dumpy, ABC->alphay, ABC->kappay,
			jmp, abscissa_in_PML, *ABC, PRM);

	    abscissa_in_PML = yoriginfront - (yval + DS / 2.0);
	    CompABCCoef(ABC->dumpy2, ABC->alphay2, ABC->kappay2,
			jmp, abscissa_in_PML, *ABC, PRM);
	}
	if (j >= YMAX + 1) {	/* For the back side */
	    abscissa_in_PML = yval - yoriginback;
	    CompABCCoef(ABC->dumpy, ABC->alphay, ABC->kappay2,
			jmp, abscissa_in_PML, *ABC, PRM);

	    abscissa_in_PML = yval + DS / 2.0 - yoriginback;
	    CompABCCoef(ABC->dumpy2, ABC->alphay2, ABC->kappay2,
			jmp, abscissa_in_PML, *ABC, PRM);
	}
	if (ABCmethod == CPML) {	/* CPML */
	    if (ABC->alphay[jmp] < 0.0)
		ABC->alphay[jmp] = 0.0;
	    if (ABC->alphay2[jmp] < 0.0)
		ABC->alphay2[jmp] = 0.0;
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
		    k, abscissa_in_PML, *ABC, PRM);

	abscissa_in_PML = zoriginbottom - (zval + DS / 2.0);
	CompABCCoef(ABC->dumpz2, ABC->alphaz2, ABC->kappaz2,
		    k, abscissa_in_PML, *ABC, PRM);
    }				/* end for k */

    /* For the top side */
    if (surface == ABSORBING) {	/* absorbing layer above z = ZMAX */
	zorigintop = ZMAX * DS;
	for (k = ZMAX + 1; k <= ZMAX0; k++) {
	    zval = DS * (k - 1);
	    abscissa_in_PML = zval - zorigintop;
	    CompABCCoef(ABC->dumpz, ABC->alphaz, ABC->kappaz,
			k, abscissa_in_PML, *ABC, PRM);

	    abscissa_in_PML = zval + DS / 2.0 - zorigintop;
	    CompABCCoef(ABC->dumpz2, ABC->alphaz2, ABC->kappaz2,
			k, abscissa_in_PML, *ABC, PRM);

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






int InitializeGeol(struct OUTPUTS *OUT,
		   struct MEDIUM MDM, struct PARAMETERS PRM)
{
    int ir, i1, k;
    int imp, jmp, icpu, jcpu;
    /* mapping */
    const int XMIN = PRM.xMin;
    const int XMAX = PRM.xMax;
    const int YMIN = PRM.yMin;
    const int YMAX = PRM.yMax;
    const int ZMIN = PRM.zMin;
    const int ZMAX = PRM.zMax;
    const int ZMAX0 = PRM.zMax0;
    const int DELTA = PRM.delta;
    const double DS = PRM.ds;
    const int NVOID = MDM.numVoid;
    const int NSEA = MDM.numSea;

    /* if geological, compute height of the stations  */
    if (model == GEOLOGICAL) {	/* the geological model is read in a file */

	if (surface == ABSORBING) {	/* absorbing layer above z = ZMAX */

	    if (PRM.me == 0) {
		printf("\n Stations coordinates :\n");
	    }

	    for (ir = 0; ir < OUT->iObs; ir++) {

		icpu = PRM.i2icpu_array[OUT->ixobs[ir]];
		jcpu = PRM.j2jcpu_array[OUT->iyobs[ir]];

		imp = PRM.i2imp_array[OUT->ixobs[ir]];
		jmp = PRM.j2jmp_array[OUT->iyobs[ir]];

		if (PRM.coords[0] == icpu) {
		    if (PRM.coords[1] == jcpu) {

			OUT->izobs[ir] = ZMAX0;
			for (k = ZMAX0; k >= ZMIN - DELTA; k--) {
			    //i1 = MDM.imed[imp][jmp][k];
          i1 = i3access(MDM.imed, -1, PRM.mpmx + 2, -1, PRM.mpmy + 2, PRM.zMin - PRM.delta, PRM.zMax0, imp, jmp, k);
			    if (sea == NOSEA && i1 != NVOID) {
				OUT->izobs[ir] = k;
				break;
			    }
			    if (sea == SEA && i1 != NVOID && i1 != NSEA) {
				OUT->izobs[ir] = k;
				break;
			    }
			}

			/*  OUT->zobs[ir] = (OUT->izobs[ir]-1)*DS + PRM.z0; */
			OUT->zobswt[ir] = 0.0;

			printf("Station %d (%d) : \n", ir + 1,
			       OUT->ista[ir]);
			printf("global position : %d %d %d\n",
			       OUT->ixobs[ir], OUT->iyobs[ir],
			       OUT->izobs[ir]);
			printf("weights : %f %f %f\n", OUT->xobswt[ir],
			       OUT->yobswt[ir], OUT->zobswt[ir]);

		    }		/* end of jcpu */
		}		/* end of icpu */

	    }			/* end of ir (stations) */

	}			/* end of surface = 0 */

    }				/* end of model = 1 */

    return EXIT_SUCCESS;
}				/* end function  */

/* =================================== */
/* INITIALIZE DAY and BRADLEY          */
/* =================================== */
int InitializeDayBradley(struct MEDIUM *MDM,
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
    const double DT = PRM.dt;
    /* others */
    int i, j, k, imp, jmp;
    int step, stepMax;
    int ly0, ly;

    int nLayeR;
    double *qP;
    double *qS;
    double *mU;
    double *kaP;
    double *rhO;

    const double Tm = ANL->tm;
    const double TM = ANL->tM;
    const double W0 = ANL->w0;
    const double PI = PRM.pi;

    if (ANLmethod != DAYandBRADLEY) {
	return EXIT_FAILURE;
    }
    if (model == GEOLOGICAL) {
	stepMax = 1;
    } else if (model == LAYER) {
	stepMax = 2;
    }

    for (step = 0; step < stepMax; step++) {
	/* map */
	if (step == 1) {	/* Interfaces */
	    nLayeR == MDM->nLayer2;
	    qS = (*ANL).Qs2;
	    qP = (*ANL).Qp2;
	    mU = (*MDM).mu2;
	    kaP = (*MDM).kap2;
	    rhO = (*MDM).rho2;
	} else if (step == 0) {	/* layers */
	    nLayeR = MDM->nLayer;
	    qS = (*ANL).Qs0;
	    qP = (*ANL).Qp0;
	    mU = (*MDM).mu0;
	    kaP = (*MDM).kap0;
	    rhO = (*MDM).rho0;
	}

	for (ly = 0; ly < nLayeR; ly++) {
	    double c1, c2;
	    double mu_w, kap_w;
	    mu_w = mU[ly];
	    c1 = (1. -
		  (1. / (PI * qS[ly])) *
		  log((TM * TM + W0 * W0 * Tm * Tm * TM * TM) / (Tm * Tm +
								 W0 * W0 *
								 Tm * Tm *
								 TM *
								 TM)));
	    c2 = (2. / (PI * qS[ly])) * atan(W0 * (TM - Tm) /
					     (1. + W0 * W0 * Tm * TM));

	    mU[ly] = mu_w / sqrt(c1 * c1 + c2 * c2);


	    kap_w = kaP[ly];
	    c1 = (1. -
		  (1. / (PI * qP[ly])) *
		  log((TM * TM + W0 * W0 * Tm * Tm * TM * TM) / (Tm * Tm +
								 W0 * W0 *
								 Tm * Tm *
								 TM *
								 TM)));
	    c2 = ((2. / (PI * qP[ly])) *
		  atan(W0 * (TM - Tm) / (1. + W0 * W0 * Tm * TM)));

	    kaP[ly] =
		(kap_w + 4. / 3. * mu_w) / sqrt(c1 * c1 + c2 * c2) -
		4. * mU[ly] / 3.;
	}

	/* remove map */
	nLayeR == 0;
	qS = NULL;
	qP = NULL;
	mU = NULL;
	kaP = NULL;
    }				/* end for pos */

    return EXIT_SUCCESS;
}				/* end Initialize DAY and BRADLEY */



/*  functions : Ylapha/beta to Ylkap and inverse a matrix */
double CompYlkap(double Ylalpha, double Ylbeta, double kap, double mu,
		 double rho)
{
    double alpha_2, beta_2;
    alpha_2 = (kap + 4. / 3. * mu) / rho;
    beta_2 = mu / rho;

    return (alpha_2 * Ylalpha - 4. / 3. * beta_2 * Ylbeta) / (alpha_2 -
							      4. / 3. *
							      beta_2);
}


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

	free(ABC->phivxx);
	free(ABC->phivyy);
	free(ABC->phivzz);

	free(ABC->phivxy);
	free(ABC->phivyx);

	free(ABC->phivxz);
	free(ABC->phivzx);

	free(ABC->phivyz);
	free(ABC->phivzy);

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
