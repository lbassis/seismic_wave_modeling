#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "../include/nrutil.h"
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

    /* Découpage : 1rst way */

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
//#endif /* fin du premier découpage */

    /* Découpage : 2nd way */

/************************************************************************/
//#if (DECOUP2)

    /* Ici on essaie d ajuster au plus juste les tailles de mpmx --> variables
       On decoupe de facon pessimiste et on ajuste progressivement
       Reste le probleme du cout specifique des CPMLs qui n est pas aborde */
    double mpmx_tmp1;
    double mpmy_tmp1;
    int difference;


    mpmx_tmp1 = (double) (XMAX - XMIN + 2 * DELTA + 3) / (double) PX;
    printf("o px eh %d\n", PX);
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
//#endif /* fin du deuxième découpage */
/********************************************************************/
    /* Fin du découpage */

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
		ABC->ipml[imp][jmp][k] = -1;

		place = WhereAmI(i, j, k, PRM);
		if (place == ABSORBINGLAYER || place == FREEABS) {
		    ABC->npmlv += 1;
		    ABC->ipml[imp][jmp][k] = ABC->npmlv;
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

    if (ABCmethod == PML) {
	ABC->vxx = mydvector0(1, ABC->npmlv);
	ABC->vxy = mydvector0(1, ABC->npmlv);
	ABC->vxz = mydvector0(1, ABC->npmlv);
	ABC->vyx = mydvector0(1, ABC->npmlv);
	ABC->vyy = mydvector0(1, ABC->npmlv);
	ABC->vyz = mydvector0(1, ABC->npmlv);
	ABC->vzx = mydvector0(1, ABC->npmlv);
	ABC->vzy = mydvector0(1, ABC->npmlv);
	ABC->vzz = mydvector0(1, ABC->npmlv);


    } else if (ABCmethod == CPML) {

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

    if (ABCmethod == PML) {
	ABC->txxx = mydvector0(1, ABC->npmlt);
	ABC->tyyx = mydvector0(1, ABC->npmlt);
	ABC->tzzx = mydvector0(1, ABC->npmlt);
	ABC->txyx = mydvector0(1, ABC->npmlt);
	ABC->txzx = mydvector0(1, ABC->npmlt);
	ABC->txxy = mydvector0(1, ABC->npmlt);
	ABC->tyyy = mydvector0(1, ABC->npmlt);
	ABC->tzzy = mydvector0(1, ABC->npmlt);
	ABC->txyy = mydvector0(1, ABC->npmlt);
	ABC->tyzy = mydvector0(1, ABC->npmlt);
	ABC->txxz = mydvector0(1, ABC->npmlt);
	ABC->tyyz = mydvector0(1, ABC->npmlt);
	ABC->tzzz = mydvector0(1, ABC->npmlt);
	ABC->tyzz = mydvector0(1, ABC->npmlt);
	ABC->txzz = mydvector0(1, ABC->npmlt);
    } else if (ABCmethod == CPML) {

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
    fprintf(stderr, "\n ## ANELASTICITIES\n");
#endif
    if (ANLmethod == DAYandBRADLEY) {
	ANL->ksixx =
	    myd3tensor0(-1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
	ANL->ksiyy =
	    myd3tensor0(-1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
	ANL->ksizz =
	    myd3tensor0(-1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
	ANL->ksixy =
	    myd3tensor0(-1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
	ANL->ksixz =
	    myd3tensor0(-1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
	ANL->ksiyz =
	    myd3tensor0(-1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);

	ANL->fxx =
	    myd3tensor0(-1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
	ANL->fyy =
	    myd3tensor0(-1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
	ANL->fzz =
	    myd3tensor0(-1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
	ANL->fxy =
	    myd3tensor0(-1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
	ANL->fxz =
	    myd3tensor0(-1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
	ANL->fyz =
	    myd3tensor0(-1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);

    } else if (ANLmethod == KRISTEKandMOCZO) {
	ANL->ksilxx =
	    myd3tensor0(-1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
	ANL->ksilyy =
	    myd3tensor0(-1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
	ANL->ksilzz =
	    myd3tensor0(-1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
	ANL->ksilxy =
	    myd3tensor0(-1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
	ANL->ksilxz =
	    myd3tensor0(-1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
	ANL->ksilyz =
	    myd3tensor0(-1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
    }				/* end of ANLmethod == KRISTEKandMOCZO */

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

    if (ANLmethod == KRISTEKandMOCZO) {
	NORTH->bufKsiS = mydvector0(0, (6 * NMAXY) - 1);
	NORTH->bufKsiR = mydvector0(0, (6 * NMAXY) - 1);
    }

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
			MDM->imed[imp][jmp][k] = MDM->imed[impN][jmpN][kN];

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
			    i1 = MDM.imed[imp][jmp][k];
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




/* =================================================== */
/* INITIALIZE KRISTEK and MOCZO anelasticity method    */
/* =================================================== */
int InitializeKManelas(struct ANELASTICITY *ANL,
		       struct MEDIUM *MDM, double dt)
{
    /* Ref : 
     * Seismic-Wave Propagation in Viscoelastic media , 
     * J.Kristek and P. Moczo,
     * Bulletin of seismological society of america, vol. 93, No 5 pp 2273-2280, october 2003
     */
    /* Warning :
     * the real system is not linear, but we consider it is.
     */
    /*  assert ( ANLmethod == KRISTEKandMOCZO ); */

    const int n = 4;
    const int kmax = 7;		/* = 2*n-1 */

    int interfaceStep, intStepMax;
    int step;

    /* for mapping */
    int NLAYER;
    double *RHO, *KAP, *MU;
    double *Qp, *Qs;
    double **Ylkap;
    double **Ylmu;

    /* others */
    int i;
    int l, k, l1, l2, ly;
    double val, dlogw;
    double y[5], wl[5];
    double ydum[5], q[8], wk[8];
    double **a, **adum, **inv_adum;
    double *d, *c, *ytk, *ytm;
    double kapdum, mudum;

    /* ALGORITHM */
    /* 0 - some allocations
       1- compute pulsations : wl and wk
       2- depending on the "interface";
       for ( interfaceStep )
       allocation of Yl and mappings ;
       for each layer
       compute Ylalpha, Ylbeta ( step )
       Ylaplha/beta -> Ylmu/kap;
       compute new kap, mu and Ylmu/kap;
       end 
       end 
     */

    /* allocate */
    ANL->wl = mydvector0(1, 4);
    a = dmatrix(1, kmax, 1, n);

    adum = dmatrix(1, n, 1, n);
    inv_adum = dmatrix(1, n, 1, n);

    /* Compute wl and wk */
    wl[1] = ANL->wmin;
    wl[n] = ANL->wmax;
    dlogw = (log10(ANL->wmax) - log10(ANL->wmin)) / (n - 1);
    for (l = 2; l < n; l++) {
	wl[l] = pow(10., log10(ANL->wmin) + dlogw * (l - 1));
    }

    wk[1] = wl[1];
    wk[kmax] = wl[n];
    dlogw = (log10(ANL->wmax) - log10(ANL->wmin)) / (kmax - 1);
    for (k = 2; k < kmax; k++) {
	wk[k] = pow(10., log10(ANL->wmin) + dlogw * (k - 1));
    }

    ANL->wl[1] = wl[1];
    ANL->wl[2] = wl[2];
    ANL->wl[3] = wl[3];
    ANL->wl[4] = wl[4];
#if (VERBOSE > 2 )
    printf("wl 1:%e 2:%e  3:%e 4:%e \n ",
	   ANL->wl[1], ANL->wl[2], ANL->wl[3], ANL->wl[4]);
#endif

  /*=== Compute yl and other coefficients ===*/
    if (model == GEOLOGICAL) {
	intStepMax = 1;
    } else if (model == USUAL) {
	intStepMax = 2;
    }

    for (interfaceStep = 0; interfaceStep < intStepMax; interfaceStep++) {
    /***  allocate and map ***/
	if (interfaceStep == 0) {	/* layers */
	    NLAYER = MDM->nLayer;
	    MU = MDM->mu0;
	    KAP = MDM->kap0;
	    RHO = MDM->rho0;
	    Qp = ANL->Qp0;
	    Qs = ANL->Qs0;

	    ANL->ylkap = mydmatrix0(1, 4, 0, NLAYER - 1);
	    ANL->ylmu = mydmatrix0(1, 4, 0, NLAYER - 1);

	    Ylkap = ANL->ylkap;
	    Ylmu = ANL->ylmu;

	} else if (interfaceStep == 1) {	/* interfaces */
	    NLAYER = MDM->nLayer2;
	    MU = MDM->mu2;
	    KAP = MDM->kap2;
	    RHO = MDM->rho2;
	    Qp = ANL->Qp2;
	    Qs = ANL->Qs2;

	    ANL->ylkap2 = mydmatrix0(1, 4, 0, NLAYER - 1);
	    ANL->ylmu2 = mydmatrix0(1, 4, 0, NLAYER - 1);

	    Ylkap = ANL->ylkap2;
	    Ylmu = ANL->ylmu2;
	}

      /**** Compute Ylalpha Ylbeta ***/
	for (ly = 0; ly <= NLAYER - 1; ly++) {

	    for (step = 1; step <= 2; step++) {	/* Compute: step1=>Ylalpha(Qp), step2=>Ylmu(Qs) */

		/* initialise */
		for (k = 1; k <= kmax; k++) {	/* we compute on Q^-1 */
		    if (step == 1) {
			q[k] = 1. / Qp[ly];
		    } else if (step == 2) {
			q[k] = 1. / Qs[ly];
		    }

		    for (l = 1; l <= n; l++) {	/* construct matrix */
			a[k][l] =
			    (wk[k] / wl[l] + q[k]) / (1 +
						      (wk[k] / wl[l]) *
						      (wk[k] / wl[l]));
		    }
		}

		/* Linear inverse problem (Over-determined case) */
		/* A Y = Q -> Y = (A^t A)^(-1) A^t Q */
		/* We compute A^t A */
		for (l1 = 1; l1 <= n; l1++) {
		    for (l2 = 1; l2 <= n; l2++) {
			adum[l1][l2] = 0.0;
			for (k = 1; k <= kmax; k++) {
			    adum[l1][l2] += a[k][l1] * a[k][l2];
			}
		    }
		}

		/* Inverse A^t A */
		for (l1 = 1; l1 <= n; l1++) {
		    for (l2 = 1; l2 <= n; l2++) {
			if (l1 == l2)
			    inv_adum[l1][l2] = 1.;
			if (l1 != l2)
			    inv_adum[l1][l2] = 0.;
		    }
		}
		if (inv(adum, inv_adum, n) != EXIT_SUCCESS) {
		    return EXIT_FAILURE;
		}

		/* We compute A^t y */
		for (l1 = 1; l1 <= n; l1++) {
		    ydum[l1] = 0.0;
		    for (k = 1; k <= kmax; k++) {
			ydum[l1] += a[k][l1] * q[k];
		    }
		}

		for (l = 1; l <= n; l++) {
		    y[l] = 0.0;
		    for (k = 1; k <= n; k++) {
			y[l] += inv_adum[l][k] * ydum[k];
		    }
		}

#if (VERBOSE > 2 )
		/* verify result */
		for (l = 1; l <= n; l++) {
		    printf("interfacestep %i step : %i l = %i  yl = %g\n",
			   interfaceStep, step, l, y[l]);
		}

		for (k = 1; k <= kmax; k++) {
		    double verif = 0.;
		    for (l = 1; l <= n; l++) {

			verif += a[k][l] * y[l];
		    }
		    verif = (verif - q[k]) / q[k];
		    printf("residu %i  = %g \n", k, verif);
		}
#endif

		if (step == 1) {	/* Qp */
		    Ylkap[1][ly] = y[1];
		    Ylkap[2][ly] = y[2];
		    Ylkap[3][ly] = y[3];
		    Ylkap[4][ly] = y[4];
		} else if (step == 2) {	/* Qs */
		    Ylmu[1][ly] = y[1];
		    Ylmu[2][ly] = y[2];
		    Ylmu[3][ly] = y[3];
		    Ylmu[4][ly] = y[4];
		}

	    }			/* end for step ( compute Ylaplha, Ylbeta ) */



      /*=== Ylalpha, Ylbeta => Ylkap , Ylmu ===*/
	    for (i = 1; i <= 4; i++) {
		Ylkap[i][ly] =
		    CompYlkap(Ylkap[i][ly], Ylmu[i][ly], KAP[ly], MU[ly],
			      RHO[ly]);
	    }

      /*=== compute de kappa~ ,mu~  ===*/
	    double dum;
	    dum = 0.;
	    for (i = 1; i <= 4; i++) {
		dum += ((wl[i] * dt) / (2. - wl[i] * dt)) * Ylkap[i][ly];
	    }
	    kapdum = KAP[ly] * (1. + dum);


	    dum = 0.;
	    for (i = 1; i <= 4; i++) {
		dum += ((wl[i] * dt) / (2. - wl[i] * dt)) * Ylmu[i][ly];
	    }
	    mudum = MU[ly] * (1. + dum);

      /*===  Ylkap, Ylmu => Ylkap~ , Ylmu~ ===*/
	    for (i = 1; i <= 4; i++) {
		Ylmu[i][ly] *= (2. / (2. - wl[i] * dt)) * MU[ly];
	    }
	    for (i = 1; i <= 4; i++) {
		Ylkap[i][ly] *= (2. / (2. - wl[i] * dt)) * KAP[ly];
	    }

	    KAP[ly] = kapdum;
	    MU[ly] = mudum;

	}			/* end for ly */

	/* remove mapping */
	RHO = NULL;
	MU = NULL;
	KAP = NULL;
	Qp = NULL;
	Qs = NULL;

	Ylkap = (double **) NULL;
	Ylmu = (double **) NULL;

    }				/* end for interfaceStep */

    /* free local memory */
    free_dmatrix(a, 1, n, 1, n);
    free_dmatrix(adum, 1, n, 1, n);

    return EXIT_SUCCESS;
}

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

int inv(double **M, double **invM, int XMAX)
{				/* from Numerical recipies */
    int i, j, k, n, l, ind;
    double max;
    double **auxM, **E, **aux;

    auxM = dmatrix(1, XMAX, 1, XMAX);
    E = dmatrix(1, XMAX, 1, XMAX);
    aux = dmatrix(1, XMAX, 1, XMAX);

    for (i = 1; i <= XMAX; i++) {
	for (j = 1; j <= XMAX; j++) {
	    auxM[i][j] = M[i][j];
	}
    }

    for (n = 1; n <= XMAX - 1; n++) {

	max = 0.0;
	ind = n;
	for (k = n; k <= XMAX; k++) {
	    if (abs(M[k][n]) > abs(max)) {
		max = M[k][n];
		ind = k;
	    }
	}

	if (ind != n) {

	    for (i = 1; i <= XMAX; i++) {
		for (j = 1; j <= XMAX; j++) {
		    if (i == j)
			E[i][j] = 1.;
		    if (i != j)
			E[i][j] = 0.;
		}
	    }

	    E[ind][ind] -= 1.;
	    E[n][n] -= 1.;
	    E[ind][n] += 1.;
	    E[n][ind] += 1.;

	    for (i = 1; i <= XMAX; i++) {
		for (j = 1; j <= XMAX; j++) {
		    aux[i][j] = 0.;
		    for (l = 1; l <= XMAX; l++) {
			aux[i][j] += E[i][l] * auxM[l][j];
		    }
		}
	    }
	    for (i = 1; i <= XMAX; i++) {
		for (j = 1; j <= XMAX; j++) {
		    auxM[i][j] = aux[i][j];
		}
	    }

	    for (i = 1; i <= XMAX; i++) {
		for (j = 1; j <= XMAX; j++) {
		    aux[i][j] = 0.;
		    for (l = 1; l <= XMAX; l++) {
			aux[i][j] += E[i][l] * invM[l][j];
		    }
		}
	    }
	    for (i = 1; i <= XMAX; i++) {
		for (j = 1; j <= XMAX; j++) {
		    invM[i][j] = aux[i][j];
		}
	    }

	}

	for (k = n + 1; k <= XMAX; k++) {

	    for (i = 1; i <= XMAX; i++) {
		for (j = 1; j <= XMAX; j++) {
		    if (i == j)
			E[i][j] = 1.;
		    if (i != j)
			E[i][j] = 0.;
		}
	    }

	    E[k][n] -= auxM[k][n] / auxM[n][n];

	    for (i = 1; i <= XMAX; i++) {
		for (j = 1; j <= XMAX; j++) {
		    aux[i][j] = 0.;
		    for (l = 1; l <= XMAX; l++) {
			aux[i][j] += E[i][l] * auxM[l][j];
		    }
		}
	    }
	    for (i = 1; i <= XMAX; i++) {
		for (j = 1; j <= XMAX; j++) {
		    auxM[i][j] = aux[i][j];
		}
	    }

	    for (i = 1; i <= XMAX; i++) {
		for (j = 1; j <= XMAX; j++) {
		    aux[i][j] = 0.;
		    for (l = 1; l <= XMAX; l++) {
			aux[i][j] += E[i][l] * invM[l][j];
		    }
		}
	    }
	    for (i = 1; i <= XMAX; i++) {
		for (j = 1; j <= XMAX; j++) {
		    invM[i][j] = aux[i][j];
		}
	    }

	}
    }

    for (n = XMAX; n >= 2; n--) {

	for (k = n - 1; k >= 1; k--) {

	    for (i = 1; i <= XMAX; i++) {
		for (j = 1; j <= XMAX; j++) {
		    if (i == j)
			E[i][j] = 1.;
		    if (i != j)
			E[i][j] = 0.;
		}
	    }

	    E[k][n] -= auxM[k][n] / auxM[n][n];

	    for (i = 1; i <= XMAX; i++) {
		for (j = 1; j <= XMAX; j++) {
		    aux[i][j] = 0.;
		    for (l = 1; l <= XMAX; l++) {
			aux[i][j] += E[i][l] * auxM[l][j];
		    }
		}
	    }
	    for (i = 1; i <= XMAX; i++) {
		for (j = 1; j <= XMAX; j++) {
		    auxM[i][j] = aux[i][j];
		}
	    }

	    for (i = 1; i <= XMAX; i++) {
		for (j = 1; j <= XMAX; j++) {
		    aux[i][j] = 0.;
		    for (l = 1; l <= XMAX; l++) {
			aux[i][j] += E[i][l] * invM[l][j];
		    }
		}
	    }
	    for (i = 1; i <= XMAX; i++) {
		for (j = 1; j <= XMAX; j++) {
		    invM[i][j] = aux[i][j];
		}
	    }

	}
    }

    for (n = 1; n <= XMAX; n++) {

	for (i = 1; i <= XMAX; i++) {
	    for (j = 1; j <= XMAX; j++) {
		if (i == j)
		    E[i][j] = 1.;
		if (i != j)
		    E[i][j] = 0.;
	    }
	}

	E[n][n] = 1. / auxM[n][n];

	for (i = 1; i <= XMAX; i++) {
	    for (j = 1; j <= XMAX; j++) {
		aux[i][j] = 0.;
		for (l = 1; l <= XMAX; l++) {
		    aux[i][j] += E[i][l] * auxM[l][j];
		}
	    }
	}
	for (i = 1; i <= XMAX; i++) {
	    for (j = 1; j <= XMAX; j++) {
		auxM[i][j] = aux[i][j];
	    }
	}

	for (i = 1; i <= XMAX; i++) {
	    for (j = 1; j <= XMAX; j++) {
		aux[i][j] = 0.;
		for (l = 1; l <= XMAX; l++) {
		    aux[i][j] += E[i][l] * invM[l][j];
		}
	    }
	}
	for (i = 1; i <= XMAX; i++) {
	    for (j = 1; j <= XMAX; j++) {
		invM[i][j] = aux[i][j];
	    }
	}

    }
    return EXIT_SUCCESS;
}




int InitializeOutputs(int STATION_STEP,
		      struct OUTPUTS *OUT, struct PARAMETERS PRM)
{
    /* ===Seismograms related === */
    int i, j, k;
    int i2, j2, k2;
    int ir;
    int icpu, jcpu;
    int icpu2, jcpu2;
    int icpuEnd, jcpuEnd;
    /* mapping */
    const int PX = PRM.px;

    OUT->mapping_seis = imatrix(0, OUT->iObs - 1, 1, 9);
    OUT->seis_output =
	myd3tensor0(0, STATION_STEP - 1, 0, OUT->iObs - 1, 1, 9);
    OUT->seis_buff = mydvector0(0, STATION_STEP - 1);
    /* mapping cpu X direction (coords[0]) then y direction (coords[1])
       rank = coords[0] + coords[1]*px */

    /* For info :
     * Vx component  : i, j 
     * Vy component  : i2, j2 
     * Vz component  : i2, j 
     * Tii component : i2, j
     * Txy component : i, j2
     * Txz component : i, j 
     * Tyz component : i2, j2 
     */

    for (ir = 0; ir < OUT->iObs; ir++) {
	if (OUT->ista[ir] == 1) {

	    i = OUT->ixobs[ir];
	    j = OUT->iyobs[ir];
	    icpu = PRM.i2icpu_array[i];
	    jcpu = PRM.j2jcpu_array[j];

	    if (OUT->xobswt[ir] >= 0.5) {
		i2 = i;
	    } else {
		i2 = i - 1;
	    }
	    if (OUT->yobswt[ir] >= 0.5) {
		j2 = j;
	    } else {
		j2 = j - 1;
	    }
	    icpu2 = PRM.i2icpu_array[i2];
	    jcpu2 = PRM.j2jcpu_array[j2];

	    /* Vx component */
	    OUT->mapping_seis[ir][1] = icpu + jcpu * PX;

	    /* Vy component */
	    OUT->mapping_seis[ir][2] = icpu2 + jcpu2 * PX;

	    /* Vz component */
	    OUT->mapping_seis[ir][3] = icpu2 + jcpu * PX;

	    /* Tii component */

	    OUT->mapping_seis[ir][4] = icpu2 + jcpu * PX;
	    OUT->mapping_seis[ir][5] = icpu2 + jcpu * PX;
	    OUT->mapping_seis[ir][6] = icpu2 + jcpu * PX;

	    /* Txy component */
	    OUT->mapping_seis[ir][7] = icpu2 + jcpu2 * PX;

	    /* Txz component */
	    OUT->mapping_seis[ir][8] = icpu + jcpu * PX;

	    /* Tyz component */
	    OUT->mapping_seis[ir][9] = icpu2 + jcpu2 * PX;

	}			/* end of if ista */
    }				/* end of ir */

    /* ===Snapshots related === */
    if (snapType == ODISPL || snapType == OBOTH) {
	const int ZMIN = PRM.zMin;
	const int ZMAX0 = PRM.zMax0;
	const int DELTA = PRM.delta;
	const int MPMX = PRM.mpmx;
	const int MPMY = PRM.mpmy;

	OUT->Uxy = myd3tensor0(1, 3, -1, MPMX + 2, -1, MPMY + 2);
	OUT->Uxz = myd3tensor0(1, 3, -1, MPMX + 2, ZMIN - DELTA, ZMAX0);
	OUT->Uyz = myd3tensor0(1, 3, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
    }
    return (EXIT_SUCCESS);
}				/* end function */




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
    if (ANLmethod == DAYandBRADLEY) {
	free_d3tensor(ANL->ksixx, -1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA,
		      ZMAX0);
	free_d3tensor(ANL->ksiyy, -1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA,
		      ZMAX0);
	free_d3tensor(ANL->ksizz, -1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA,
		      ZMAX0);
	free_d3tensor(ANL->ksixy, -1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA,
		      ZMAX0);
	free_d3tensor(ANL->ksixz, -1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA,
		      ZMAX0);
	free_d3tensor(ANL->ksiyz, -1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA,
		      ZMAX0);

	free_d3tensor(ANL->fxx, -1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA,
		      ZMAX0);
	free_d3tensor(ANL->fyy, -1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA,
		      ZMAX0);
	free_d3tensor(ANL->fzz, -1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA,
		      ZMAX0);
	free_d3tensor(ANL->fxy, -1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA,
		      ZMAX0);
	free_d3tensor(ANL->fxz, -1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA,
		      ZMAX0);
	free_d3tensor(ANL->fyz, -1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA,
		      ZMAX0);

	free_dvector(ANL->Qp0, 0, NLAYER - 1);
	free_dvector(ANL->Qs0, 0, NLAYER - 1);
	if (model == LAYER) {
	    free_dvector(ANL->Qp2, 0, NLAYERINT - 1);
	    free_dvector(ANL->Qs2, 0, NLAYERINT - 1);
	}
    } else if (ANLmethod == KRISTEKandMOCZO) {
	free_d3tensor(ANL->ksilxx, -1, MPMX + 2, -1, MPMY + 2,
		      ZMIN - DELTA, ZMAX0);
	free_d3tensor(ANL->ksilyy, -1, MPMX + 2, -1, MPMY + 2,
		      ZMIN - DELTA, ZMAX0);
	free_d3tensor(ANL->ksilzz, -1, MPMX + 2, -1, MPMY + 2,
		      ZMIN - DELTA, ZMAX0);
	free_d3tensor(ANL->ksilxy, -1, MPMX + 2, -1, MPMY + 2,
		      ZMIN - DELTA, ZMAX0);
	free_d3tensor(ANL->ksilxz, -1, MPMX + 2, -1, MPMY + 2,
		      ZMIN - DELTA, ZMAX0);
	free_d3tensor(ANL->ksilyz, -1, MPMX + 2, -1, MPMY + 2,
		      ZMIN - DELTA, ZMAX0);

	free_dmatrix(ANL->ylmu, 1, 4, 0, NLAYER - 1);
	free_dmatrix(ANL->ylkap, 1, 4, 0, NLAYER - 1);
	free_dvector(ANL->wl, 1, 4);

	free_dvector(ANL->Qp0, 0, NLAYER - 1);
	free_dvector(ANL->Qs0, 0, NLAYER - 1);

	if (model == LAYER) {
	    free_dmatrix(ANL->ylmu2, 1, 4, 0, NLAYERINT - 1);
	    free_dmatrix(ANL->ylkap2, 1, 4, 0, NLAYERINT - 1);

	    free_dvector(ANL->Qp2, 0, NLAYERINT - 1);
	    free_dvector(ANL->Qs2, 0, NLAYERINT - 1);
	}

    } else if (ANLmethod == ANOTHER) {
	free_dvector(ANL->q0, 0, NLAYER - 1);
	free_dvector(ANL->amp, 0, NLAYER - 1);

	if (model == LAYER) {
	    free_dvector(ANL->amp2, 0, NLAYERINT - 1);
	}
    }				/* end anelasticities */


      /*** Velocity ***/
    free_d3tensor(v0->x, -1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
    free_d3tensor(v0->y, -1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
    free_d3tensor(v0->z, -1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);

  /*** Stress ***/
    free_d3tensor(t0->xx, -1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
    free_d3tensor(t0->yy, -1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
    free_d3tensor(t0->zz, -1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
    free_d3tensor(t0->xy, -1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
    free_d3tensor(t0->xz, -1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
    free_d3tensor(t0->yz, -1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);


  /*** Source ***/
    free_ivector(SRC->ixhypo, 0, SRC->iSrc - 1);
    free_ivector(SRC->iyhypo, 0, SRC->iSrc - 1);
    free_ivector(SRC->izhypo, 0, SRC->iSrc - 1);
    free_ivector(SRC->insrc, 0, SRC->iSrc - 1);

    if (source == HISTFILE) {

	free_dvector(SRC->strike, 0, SRC->iSrc - 1);
	free_dvector(SRC->dip, 0, SRC->iSrc - 1);
	free_dvector(SRC->rake, 0, SRC->iSrc - 1);
	free_dvector(SRC->slip, 0, SRC->iSrc - 1);
	free_dvector(SRC->xweight, 0, SRC->iSrc - 1);
	free_dvector(SRC->yweight, 0, SRC->iSrc - 1);
	free_dvector(SRC->zweight, 0, SRC->iSrc - 1);

	free_dmatrix(SRC->vel, 0, SRC->iSrc - 1, 0, SRC->iDur - 1);

	free_d3tensor(SRC->fx, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0);
	free_d3tensor(SRC->fy, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0);
	free_d3tensor(SRC->fz, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0);
    }


      /*** MEDIUM  ***/
    if (model == GEOLOGICAL) {
	free_i3tensor(MDM->imed, -1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA,
		      ZMAX0);
    }

    free_dvector(MDM->laydep, 0, NLAYER - 1);
    free_dvector(MDM->rho0, 0, NLAYER - 1);
    free_dvector(MDM->mu0, 0, NLAYER - 1);
    free_dvector(MDM->kap0, 0, NLAYER - 1);

    if (model == LAYER) {
	free_dvector(MDM->rho2, 0, NLAYER - 1);
	free_dvector(MDM->mu2, 0, NLAYER - 1);
	free_dvector(MDM->kap2, 0, NLAYER - 1);
    }				/* end if model */

      /*** Absorbing Boundary Condition ***/
    free_i3tensor(ABC->ipml, -1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA,
		  ZMAX0);

    free_dvector(ABC->dumpx, 1, MPMX);
    free_dvector(ABC->dumpx2, 1, MPMX);
    free_dvector(ABC->dumpy, 1, MPMY);
    free_dvector(ABC->dumpy2, 1, MPMY);
    free_dvector(ABC->dumpz, ZMIN - DELTA, ZMAX0);
    free_dvector(ABC->dumpz2, ZMIN - DELTA, ZMAX0);

    if (ABCmethod == CPML) {

	free_dvector(ABC->phivxx, 1, NPMLV);
	free_dvector(ABC->phivyy, 1, NPMLV);
	free_dvector(ABC->phivzz, 1, NPMLV);

	free_dvector(ABC->phivxy, 1, NPMLV);
	free_dvector(ABC->phivyx, 1, NPMLV);

	free_dvector(ABC->phivxz, 1, NPMLV);
	free_dvector(ABC->phivzx, 1, NPMLV);

	free_dvector(ABC->phivyz, 1, NPMLV);
	free_dvector(ABC->phivzy, 1, NPMLV);

	free_dvector(ABC->phitxxx, 1, NPMLT);
	free_dvector(ABC->phitxyy, 1, NPMLT);
	free_dvector(ABC->phitxzz, 1, NPMLT);

	free_dvector(ABC->phitxyx, 1, NPMLT);
	free_dvector(ABC->phityyy, 1, NPMLT);
	free_dvector(ABC->phityzz, 1, NPMLT);

	free_dvector(ABC->phitxzx, 1, NPMLT);
	free_dvector(ABC->phityzy, 1, NPMLT);
	free_dvector(ABC->phitzzz, 1, NPMLT);

	free_dvector(ABC->kappax, 1, MPMX);
	free_dvector(ABC->kappax2, 1, MPMX);
	free_dvector(ABC->kappay, 1, MPMY);
	free_dvector(ABC->kappay2, 1, MPMY);
	free_dvector(ABC->kappaz, ZMIN - DELTA, ZMAX0);
	free_dvector(ABC->kappaz2, ZMIN - DELTA, ZMAX0);

	free_dvector(ABC->alphax, 1, MPMX);
	free_dvector(ABC->alphax2, 1, MPMX);
	free_dvector(ABC->alphay, 1, MPMY);
	free_dvector(ABC->alphay2, 1, MPMY);
	free_dvector(ABC->alphaz, ZMIN - DELTA, ZMAX0);
	free_dvector(ABC->alphaz2, ZMIN - DELTA, ZMAX0);

    } else if (ABCmethod == PML) {
	free_dvector(ABC->vxx, 1, NPMLV);
	free_dvector(ABC->vyy, 1, NPMLV);
	free_dvector(ABC->vzz, 1, NPMLV);
	free_dvector(ABC->vxy, 1, NPMLV);
	free_dvector(ABC->vxz, 1, NPMLV);
	free_dvector(ABC->vyz, 1, NPMLV);

	free_dvector(ABC->txxx, 1, NPMLT);
	free_dvector(ABC->txxy, 1, NPMLT);
	free_dvector(ABC->txxz, 1, NPMLT);

	free_dvector(ABC->tyyx, 1, NPMLT);
	free_dvector(ABC->tyyy, 1, NPMLT);
	free_dvector(ABC->tyyz, 1, NPMLT);

	free_dvector(ABC->tzzx, 1, NPMLT);
	free_dvector(ABC->tzzy, 1, NPMLT);
	free_dvector(ABC->tzzz, 1, NPMLT);

	free_dvector(ABC->txyx, 1, NPMLT);
	free_dvector(ABC->txyy, 1, NPMLT);

	free_dvector(ABC->txzx, 1, NPMLT);
	free_dvector(ABC->txzz, 1, NPMLT);

	free_dvector(ABC->tyzy, 1, NPMLT);
	free_dvector(ABC->tyzz, 1, NPMLT);
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

	free_dvector(com->bufV0S, 0, 3 * com->nmax - 1);
	free_dvector(com->bufV0R, 0, 3 * com->nmax - 1);

	free_dvector(com->bufT0S, 0, 6 * com->nmax - 1);
	free_dvector(com->bufT0R, 0, 6 * com->nmax - 1);

	if (ANLmethod == KRISTEKandMOCZO) {
	    free_dvector(com->bufKsiS, 0, 6 * com->nmax - 1);
	    free_dvector(com->bufKsiR, 0, 6 * com->nmax - 1);
	}
	/* remove mapping */
	com = NULL;
    }


  /*** parameters ***/
    free_ivector(PRM->imp2i_array, -1, MPMX + 2);
    free_ivector(PRM->jmp2j_array, -1, MPMY + 2);

    free_ivector(PRM->i2imp_array, XMIN - DELTA, XMAX + 2 * DELTA + 2);
    free_ivector(PRM->j2jmp_array, YMIN - DELTA, YMAX + 2 * DELTA + 2);

    free_ivector(PRM->i2icpu_array, XMIN - DELTA, XMAX + 2 * DELTA + 2);
    free_ivector(PRM->j2jcpu_array, YMIN - DELTA, YMAX + 2 * DELTA + 2);

  /*** OUTPUTS ***/
    /* seismogramms */
    free_ivector(OUT->ixobs, 0, OUT->iObs - 1);
    free_ivector(OUT->iyobs, 0, OUT->iObs - 1);
    free_ivector(OUT->izobs, 0, OUT->iObs - 1);

    free_dvector(OUT->xobs, 0, OUT->iObs - 1);
    free_dvector(OUT->yobs, 0, OUT->iObs - 1);
    free_dvector(OUT->zobs, 0, OUT->iObs - 1);

    free_ivector(OUT->nobs, 0, OUT->iObs - 1);
    free_ivector(OUT->ista, 0, OUT->iObs - 1);

    free_dvector(OUT->xobswt, 0, OUT->iObs - 1);
    free_dvector(OUT->yobswt, 0, OUT->iObs - 1);
    free_dvector(OUT->zobswt, 0, OUT->iObs - 1);

    free_imatrix(OUT->mapping_seis, 0, OUT->iObs - 1, 1, 9);
    free_d3tensor(OUT->seis_output, 0, STATION_STEP - 1, 0, OUT->iObs - 1,
		  1, 9);
    free_dvector(OUT->seis_buff, 0, STATION_STEP - 1);

    /* velocity planes */
    free_dvector(OUT->snapBuff, 1, OUT->test_size);
    if (snapType == ODISPL || snapType == OBOTH) {
	free_d3tensor(OUT->Uxy, 1, 3, -1, MPMX + 2, -1, MPMY + 2);
	free_d3tensor(OUT->Uxz, 1, 3, -1, MPMX + 2, ZMIN - DELTA, ZMAX0);
	free_d3tensor(OUT->Uyz, 1, 3, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);
    }
    OUT->Vxglobal = NULL;	/* already free */
    OUT->Vyglobal = NULL;
    OUT->Vzglobal = NULL;
    /* partition domain related */
    free_ivector(PRM->mpmx_tab, 0, PRM->px - 1);
    free_ivector(PRM->mpmy_tab, 0, PRM->py - 1);


    return EXIT_SUCCESS;
}


/* ################# TEST FUNCTIONS ################################# */
#ifdef TEST
int testInitializeKManelas(void)
{
    struct ANELASTICITY ANL;
    struct MEDIUM MDM;

    const double qp = 0.2, qs = 0.02, wmin = 0.1, wmax = 10.;
    const int n = 4;
    const int NLAYER = 2;
    double *Yl1mu, *Yl2mu, *Yl3mu, *Yl4mu,
	*Yl1kap, *Yl2kap, *Yl3kap, *Yl4kap,
	*kapmat, *mumat, *Qp0, *Qs0, *rho0;
    double wl1, wl2, wl3, wl4, dt;
    int l, ly;
    /* cstes */
    /*  qs=0.02; /\* constant Q-inverse *\/ */
    /*   qp=0.2; /\* constant Q-inverse *\/ */

    ANL.wmin = wmin;
    ANL.wmax = wmax;
    dt = 0.0045;

    /* Definition */
    MDM.nLayer = NLAYER;
    MDM.kap0 = dvector(0, NLAYER - 1);
    MDM.mu0 = dvector(0, NLAYER - 1);
    MDM.rho0 = dvector(0, NLAYER - 1);
    ANL.Qp0 = dvector(0, NLAYER - 1);
    ANL.Qs0 = dvector(0, NLAYER - 1);

    for (ly = 0; ly <= NLAYER - 1; ly++) {
	MDM.kap0[ly] = 1600 * (1125 * 1125 - 4 * 625 * 625 / 3);
	MDM.mu0[ly] = 1600 * 625 * 625;
	MDM.rho0[ly] = 2700.;
	ANL.Qp0[ly] = 5;
	ANL.Qs0[ly] = 50;
    }
    ANL.Qp0[0] = 10;
    ANL.Qs0[0] = 100;
    printf("mu  %15.3g kap %15.3f \n", MDM.mu0[0], MDM.kap0[0]);

    InitializeKManelas(&ANL, &MDM, dt, USUAL);


    /* sorties */
    for (ly = 0; ly <= NLAYER - 1; ly++) {
	printf("\n===== layer  %i ====== \n", ly);

	printf("wl1 %15.3f yl1mu  %15.3f yl1kap %15.3f \n", ANL.wl1,
	       ANL.yl1mu[ly], ANL.yl1kap[ly]);
	printf("wl2 %15.3f yl2mu  %15.3f yl2kap %15.3f \n", ANL.wl2,
	       ANL.yl2mu[ly], ANL.yl2kap[ly]);
	printf("wl3 %15.3f yl3mu  %15.3f yl3kap %15.3f \n", ANL.wl3,
	       ANL.yl3mu[ly], ANL.yl3kap[ly]);
	printf("wl4 %15.3f yl4mu  %15.3f yl4kap %15.3f \n", ANL.wl4,
	       ANL.yl4mu[ly], ANL.yl4kap[ly]);
	printf("mu  %15.3f kap %15.3f \n", MDM.mu0[ly], MDM.kap0[ly]);
    }

}
#endif				/* TEST */
