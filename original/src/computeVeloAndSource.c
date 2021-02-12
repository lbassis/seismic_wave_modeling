/*
   Compute Velocity field ( add source part )
   TODO : FREEABS computation
*/
#ifdef LUCAS_INSTRUMENTATION
#include <scorep/SCOREP_User.h>
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "struct.h"

#ifdef OMP
#include <omp.h>
#endif


#include "inlineFunctions.h"
#include "options.h"

#include "computeVeloAndSource.h"

/* OMP */
extern int numThreads_Level_1;

int computeSeisMoment(struct SOURCE *SRC, double time,
		      struct PARAMETERS PRM)
{
#ifdef LUCAS_INSTRUMENTATION
    SCOREP_USER_REGION_DEFINE(computeSeisMoment)
	SCOREP_USER_REGION_BEGIN(computeSeisMoment, "seisMoment",
				 SCOREP_USER_REGION_TYPE_COMMON)
#endif
    int it, is, iw;
    int i, j, k;
    double pxx, pyy, pzz, pxy, pyz, pxz;
    double mo, weight;
    int icpu, jcpu;
    int imp, jmp;
    int jcpu1, jcpu2, jcpu3;
    int jmp1, jmp2, jmp3;

    const double DS = PRM.ds;
    const double DT = PRM.dt;

    /* === Increment of seismic moment === */
    /* Like this, the result is not always the same when you change the compiler or the machine.
       This other method was proposed by Faiza but it does not seem to work when dt < dtbiem
       Need reviewing !!!
       it = (int) (dt/dtbiem);
       it *= l;
     */
    it = ceil(time / SRC->dtbiem);
    if (source == HISTFILE) {
	if (it < SRC->iDur) {
	    for (is = 0; is < SRC->iSrc; is++) {
		if (SRC->insrc[is] == 1) {
		    mo = SRC->vel[is][it] * DT;
		    pxx =
			radxx(SRC->strike[is], SRC->dip[is],
			      SRC->rake[is]);
		    pyy =
			radyy(SRC->strike[is], SRC->dip[is],
			      SRC->rake[is]);
		    pzz =
			radzz(SRC->strike[is], SRC->dip[is],
			      SRC->rake[is]);
		    pxy =
			radxy(SRC->strike[is], SRC->dip[is],
			      SRC->rake[is]);
		    pyz =
			radyz(SRC->strike[is], SRC->dip[is],
			      SRC->rake[is]);
		    pxz =
			radxz(SRC->strike[is], SRC->dip[is],
			      SRC->rake[is]);

		    for (iw = 0; iw < 8; iw++) {
			weight = 1.0;
			if ((iw % 2) == 0) {
			    i = SRC->ixhypo[is];
			    weight = (1.0 - SRC->xweight[is]);
			} else {
			    i = SRC->ixhypo[is] + 1;
			    weight = SRC->xweight[is];
			}
			if ((iw % 4) <= 1) {
			    j = SRC->iyhypo[is];
			    weight = weight * (1.0 - SRC->yweight[is]);
			} else {
			    j = SRC->iyhypo[is] + 1;
			    weight = weight * SRC->yweight[is];
			}
			if (iw < 4) {
			    k = SRC->izhypo[is];
			    weight = weight * (1.0 - SRC->zweight[is]);
			} else {
			    k = SRC->izhypo[is] + 1;
			    weight = weight * SRC->zweight[is];
			}

			icpu = PRM.i2icpu_array[i];
			imp = PRM.i2imp_array[i];

			jcpu1 = PRM.j2jcpu_array[j - 1];
			jcpu2 = PRM.j2jcpu_array[j];
			jcpu3 = PRM.j2jcpu_array[j + 1];

			jmp1 = PRM.j2jmp_array[j - 1];
			jmp2 = PRM.j2jmp_array[j];
			jmp3 = PRM.j2jmp_array[j + 1];

			if (PRM.coords[0] == icpu) {

			    if (PRM.coords[1] == jcpu3)
				SRC->fx[imp][jmp3][k] +=
				    0.5 * mo * pxy * weight;
			    if (PRM.coords[1] == jcpu1)
				SRC->fx[imp][jmp1][k] -=
				    0.5 * mo * pxy * weight;
			    if (PRM.coords[1] == jcpu2)
				SRC->fx[imp][jmp2][k + 1] +=
				    0.5 * mo * pxz * weight;
			    if (PRM.coords[1] == jcpu2)
				SRC->fx[imp][jmp2][k - 1] -=
				    0.5 * mo * pxz * weight;
			    if (PRM.coords[1] == jcpu2)
				SRC->fy[imp][jmp2][k] +=
				    0.5 * mo * pxy * weight;
			    if (PRM.coords[1] == jcpu1)
				SRC->fy[imp][jmp1][k] +=
				    0.5 * mo * pxy * weight;
			    if (PRM.coords[1] == jcpu2)
				SRC->fy[imp][jmp2][k] +=
				    0.5 * mo * pyy * weight;
			    if (PRM.coords[1] == jcpu1)
				SRC->fy[imp][jmp1][k] -=
				    0.5 * mo * pyy * weight;
			    if (PRM.coords[1] == jcpu2)
				SRC->fy[imp][jmp2][k + 1] +=
				    0.125 * mo * pyz * weight;
			    if (PRM.coords[1] == jcpu1)
				SRC->fy[imp][jmp1][k + 1] +=
				    0.125 * mo * pyz * weight;
			    if (PRM.coords[1] == jcpu2)
				SRC->fy[imp][jmp2][k - 1] -=
				    0.125 * mo * pyz * weight;
			    if (PRM.coords[1] == jcpu1)
				SRC->fy[imp][jmp1][k - 1] -=
				    0.125 * mo * pyz * weight;
			    if (PRM.coords[1] == jcpu2)
				SRC->fz[imp][jmp2][k] +=
				    0.5 * mo * pxz * weight;
			    if (PRM.coords[1] == jcpu2)
				SRC->fz[imp][jmp2][k - 1] +=
				    0.5 * mo * pxz * weight;
			    if (PRM.coords[1] == jcpu3)
				SRC->fz[imp][jmp3][k] +=
				    0.125 * mo * pyz * weight;
			    if (PRM.coords[1] == jcpu3)
				SRC->fz[imp][jmp3][k - 1] +=
				    0.125 * mo * pyz * weight;
			    if (PRM.coords[1] == jcpu1)
				SRC->fz[imp][jmp1][k] -=
				    0.125 * mo * pyz * weight;
			    if (PRM.coords[1] == jcpu1)
				SRC->fz[imp][jmp1][k - 1] -=
				    0.125 * mo * pyz * weight;
			    if (PRM.coords[1] == jcpu2)
				SRC->fz[imp][jmp2][k] +=
				    0.5 * mo * pzz * weight;
			    if (PRM.coords[1] == jcpu2)
				SRC->fz[imp][jmp2][k - 1] -=
				    0.5 * mo * pzz * weight;
			}

			icpu = PRM.i2icpu_array[i - 1];
			imp = PRM.i2imp_array[i - 1];

			if (PRM.coords[0] == icpu) {

			    if (PRM.coords[1] == jcpu2)
				SRC->fy[imp][jmp2][k] -=
				    0.5 * mo * pxy * weight;
			    if (PRM.coords[1] == jcpu1)
				SRC->fy[imp][jmp1][k] -=
				    0.5 * mo * pxy * weight;
			    if (PRM.coords[1] == jcpu2)
				SRC->fy[imp][jmp2][k] +=
				    0.5 * mo * pyy * weight;
			    if (PRM.coords[1] == jcpu1)
				SRC->fy[imp][jmp1][k] -=
				    0.5 * mo * pyy * weight;
			    if (PRM.coords[1] == jcpu2)
				SRC->fy[imp][jmp2][k + 1] +=
				    0.125 * mo * pyz * weight;
			    if (PRM.coords[1] == jcpu1)
				SRC->fy[imp][jmp1][k + 1] +=
				    0.125 * mo * pyz * weight;
			    if (PRM.coords[1] == jcpu2)
				SRC->fy[imp][jmp2][k - 1] -=
				    0.125 * mo * pyz * weight;
			    if (PRM.coords[1] == jcpu1)
				SRC->fy[imp][jmp1][k - 1] -=
				    0.125 * mo * pyz * weight;
			    if (PRM.coords[1] == jcpu2)
				SRC->fx[imp][jmp2][k] -=
				    0.5 * mo * pxx * weight;
			    if (PRM.coords[1] == jcpu3)
				SRC->fz[imp][jmp3][k] +=
				    0.125 * mo * pyz * weight;
			    if (PRM.coords[1] == jcpu3)
				SRC->fz[imp][jmp3][k - 1] +=
				    0.125 * mo * pyz * weight;
			    if (PRM.coords[1] == jcpu1)
				SRC->fz[imp][jmp1][k] -=
				    0.125 * mo * pyz * weight;
			    if (PRM.coords[1] == jcpu1)
				SRC->fz[imp][jmp1][k - 1] -=
				    0.125 * mo * pyz * weight;
			    if (PRM.coords[1] == jcpu2)
				SRC->fz[imp][jmp2][k] +=
				    0.5 * mo * pzz * weight;
			    if (PRM.coords[1] == jcpu2)
				SRC->fz[imp][jmp2][k - 1] -=
				    0.5 * mo * pzz * weight;
			    if (PRM.coords[1] == jcpu2)
				SRC->fz[imp][jmp2][k] -=
				    0.5 * mo * pxz * weight;
			    if (PRM.coords[1] == jcpu2)
				SRC->fz[imp][jmp2][k - 1] -=
				    0.5 * mo * pxz * weight;
			}

			icpu = PRM.i2icpu_array[i + 1];
			imp = PRM.i2imp_array[i + 1];

			if (PRM.coords[0] == icpu) {

			    if (PRM.coords[1] == jcpu2)
				SRC->fx[imp][jmp2][k] +=
				    0.5 * mo * pxx * weight;
			}


		    }		/* end of iw (weighting) */
		}		/* end of if SRC.insrc */
	    }			/* end of is (each source) */
	}			/* end of if it */

    }				/* end of source = 1 */
#ifdef LUCAS_INSTRUMENTATION
    SCOREP_USER_REGION_END(computeSeisMoment)
#endif
	return (EXIT_SUCCESS);
}				/* end increment Seismic moment */


void computeVelocity(		/* Status : USED */
			/* Main Outputs (we also increment Absorbing Layer functions) */
			struct VELOCITY *v0,
			struct ABSORBING_BOUNDARY_CONDITION *ABC,
			/* INPUTS */
			struct STRESS t0, struct MEDIUM MDM, struct PARAMETERS PRM, struct ANELASTICITY ANL,	/* we only use ANOTHER anelasticity method here */
			struct SOURCE SRC, int mpmx_begin, int mpmx_end,	/* local computed domain */
			int mpmy_begin, int mpmy_end,
			/* source (VELO method) */
			int l	/* time step */
    )
{
#ifdef LUCAS_INSTRUMENTATION
    SCOREP_USER_REGION_DEFINE(computeVelocity)
	SCOREP_USER_REGION_BEGIN(computeVelocity, "velocity",
				 SCOREP_USER_REGION_TYPE_COMMON)
#endif
	/* approximations of a value in the corner of the cube */
    int ly0, ly2;		/* layer xy (+0) or z (+ds/2) */
    double kapxy, kapxz, muxy, muxz,	/* rigidity and mu */
     kapx, kapy, mux, muy;
    double rhoxy, rhoxz;	/* density  */


    /*  */
    double bx, by, bz;		/* inverses of rho */
    double vp, vpxy, vpxz;	/* approximations of vp on the corner of the cube */

    /*  */
    enum typePlace place;	/* What type of cell  */
    long int npml;		/* index in Absorbing Layer */
    int i, j, k;		/* local position of the cell */
    int imp, jmp;		/* global position of the cell (NB : kmp=k) */
    /* intermediates */
    double xdum, ydum, zdum;
    double phixdum, phiydum, phizdum;

    /* source == VELO */
    int is;			/* index of the source */
    int icpu, jcpu;		/* coordiantes of the cpu */
    double rho;			/* density at [imp][jmp][k] */

    /* For mapping */
    double DS, DT, PI;		/* PARAMETERS */


    /* Verify Paramters */
    assert((interface == USUAL) || (interface == KMINTERFACE));
    assert((source == HISTFILE) || (source == VELO));
    assert(ABCmethod == CPML || ABCmethod == PML);
    assert(surface == ABSORBING || surface == FREE);
    assert(ANLmethod == ELASTIC || ANLmethod == ANOTHER ||
	   ANLmethod == DAYandBRADLEY || ANLmethod == KRISTEKandMOCZO);

    /* mapping */
    DS = PRM.ds;
    DT = PRM.dt;


#ifdef OMP
#pragma omp parallel default( shared) \
private( ly0, ly2) \
private( kapxy, kapxz, kapx, kapy ) \
private( muxy, muxz, mux, muy ) \
private( rhoxy,rhoxz ) \
private( bx, by, bz ) \
private( vp, vpxy, vpxz ) \
private( place ) \
private( npml )	\
private( i, j, k ) \
private( imp, jmp ) \
private( xdum, ydum, zdum ) \
private( phixdum, phiydum, phizdum ) num_threads(numThreads_Level_1)
    {
#pragma omp for schedule ( runtime) collapse(2)
#endif
	for (i = mpmx_begin; i <= mpmx_end; i++) {
	    for (j = mpmy_begin; j <= mpmy_end; j++) {
		jmp = PRM.jmp2j_array[j];
		imp = PRM.imp2i_array[i];
		for (k = PRM.zMin - PRM.delta; k <= PRM.zMax0; k++) {

		    /* INITIALISATIONS */
		    place = WhereAmI(imp, jmp, k, PRM);

		    if (place == OUTSIDE) {
			continue;
		    } else if (place == LIMIT) {
			v0->x[i][j][k] = 0.;
			v0->y[i][j][k] = 0.;
			v0->z[i][j][k] = 0.;
			continue;
		    } else if (place == ABSORBINGLAYER || place == FREEABS) {
			npml = ABC->ipml[i][j][k];
		    }

      /*=====================================================*\
		  ELASTIC PART :

		  (nothing to do for regular domain, or only FreeSurface domain)
		  structure :
		  + PML/CPML
		  -- common initialisations
		  -- REGULAR & ABSORBING LAYER common part
		  -- FREESURFACE & FREEABS common part

		  -- ABSORBING LAYER & FREEABS special part
		  \*=====================================================*/
      /*******************************************/
		    /* COMMON INITIALISATIONS                  */
      /*******************************************/
		    if (model == GEOLOGICAL) {
			int med1, med2, med3, med4;

			med1 = MDM.imed[i][j][k];
			med2 = MDM.imed[i][j + 1][k];
			med3 = MDM.imed[i + 1][j][k];
			med4 = MDM.imed[i + 1][j + 1][k];

			rhoxy = 0.25 * (MDM.rho0[med1] + MDM.rho0[med2] +
					MDM.rho0[med3] + MDM.rho0[med4]);
			bx = 1. / MDM.rho0[med1];
			by = 1.0 / rhoxy;

			med1 = MDM.imed[i][j][k];
			med2 = MDM.imed[i][j][k + 1];
			med3 = MDM.imed[i + 1][j][k];
			med4 = MDM.imed[i + 1][j][k + 1];
			rhoxz = 0.25 * (MDM.rho0[med1] + MDM.rho0[med2] +
					MDM.rho0[med3] + MDM.rho0[med4]);
			bz = 1.0 / rhoxz;
		    } else if (model == LAYER) {
			ly0 = MDM.k2ly0[k];
			ly2 = MDM.k2ly2[k];

			bx = 1.0 / MDM.rho0[ly0];
			by = 1.0 / MDM.rho0[ly0];

			rhoxz = MDM.rho2[ly2];
			bz = 1.0 / rhoxz;
		    }		/* end if model */


		    /*******************************************/
		    /* REGULAR & ABSORBING LAYER               */
		    /*******************************************/
		    if (place == REGULAR || place == ABSORBINGLAYER) {
			/* Computation of Vx,Vy and Vz */
			v0->x[i][j][k] += staggardv4(bx,
						     ABC->kappax[i],
						     ABC->kappay[j],
						     ABC->kappaz[k], DT,
						     DS,
						     t0.xx[i - 1][j][k],
						     t0.xx[i][j][k],
						     t0.xx[i - 2][j][k],
						     t0.xx[i + 1][j][k],
						     t0.xy[i][j - 1][k],
						     t0.xy[i][j][k],
						     t0.xy[i][j - 2][k],
						     t0.xy[i][j + 1][k],
						     t0.xz[i][j][k - 1],
						     t0.xz[i][j][k],
						     t0.xz[i][j][k - 2],
						     t0.xz[i][j][k + 1],
						     place, ABCmethod);

			v0->y[i][j][k] += staggardv4(by,
						     ABC->kappax2[i],
						     ABC->kappay2[j],
						     ABC->kappaz[k], DT,
						     DS, t0.xy[i][j][k],
						     t0.xy[i + 1][j][k],
						     t0.xy[i - 1][j][k],
						     t0.xy[i + 2][j][k],
						     t0.yy[i][j][k],
						     t0.yy[i][j + 1][k],
						     t0.yy[i][j - 1][k],
						     t0.yy[i][j + 2][k],
						     t0.yz[i][j][k - 1],
						     t0.yz[i][j][k],
						     t0.yz[i][j][k - 2],
						     t0.yz[i][j][k + 1],
						     place, ABCmethod);

			v0->z[i][j][k] += staggardv4(bz,
						     ABC->kappax2[i],
						     ABC->kappay[j],
						     ABC->kappaz2[k], DT,
						     DS, t0.xz[i][j][k],
						     t0.xz[i + 1][j][k],
						     t0.xz[i - 1][j][k],
						     t0.xz[i + 2][j][k],
						     t0.yz[i][j - 1][k],
						     t0.yz[i][j][k],
						     t0.yz[i][j - 2][k],
						     t0.yz[i][j + 1][k],
						     t0.zz[i][j][k],
						     t0.zz[i][j][k + 1],
						     t0.zz[i][j][k - 1],
						     t0.zz[i][j][k + 2],
						     place, ABCmethod);
		    }		/* end REGULAR and ABSORBINGLAYER */

		    /* ********************************** */
		    /* FREE SURFACE & FREEABS COMMON PART */
		    /* ********************************** */
		    if (place == FREESURFACE || place == FREEABS) {

			/* Description :
			 * # NB : no free surface for geological model so this part is only use with model == lAYER
			 * # What we need/compute :
			 * for k=1, full v0
			 * for k=2, only v0->x, v0->y (for t0(k=0) computation)
			 *
			 * # Details For k=1 :
			 *  For K=1
			 *  v0->x, v0->y are computed like usual.
			 *  v0->z need special treatment
			 */

			/* k=1 */
	  /*-----*/
			if (k == 1) {
			    /* v0->x, v0->y (copied & pasted) */
			    v0->x[i][j][k] += staggardv4(bx,
							 ABC->kappax[i],
							 ABC->kappay[j],
							 ABC->kappaz[k],
							 DT, DS,
							 t0.xx[i -
							       1][j][k],
							 t0.xx[i][j][k],
							 t0.xx[i -
							       2][j][k],
							 t0.xx[i +
							       1][j][k],
							 t0.xy[i][j -
								  1][k],
							 t0.xy[i][j][k],
							 t0.xy[i][j -
								  2][k],
							 t0.xy[i][j +
								  1][k],
							 t0.xz[i][j][k -
								     1],
							 t0.xz[i][j][k],
							 t0.xz[i][j][k -
								     2],
							 t0.xz[i][j][k +
								     1],
							 place, ABCmethod);

			    v0->y[i][j][k] += staggardv4(by,
							 ABC->kappax2[i],
							 ABC->kappay2[j],
							 ABC->kappaz[k],
							 DT, DS,
							 t0.xy[i][j][k],
							 t0.xy[i +
							       1][j][k],
							 t0.xy[i -
							       1][j][k],
							 t0.xy[i +
							       2][j][k],
							 t0.yy[i][j][k],
							 t0.yy[i][j +
								  1][k],
							 t0.yy[i][j -
								  1][k],
							 t0.yy[i][j +
								  2][k],
							 t0.yz[i][j][k -
								     1],
							 t0.yz[i][j][k],
							 t0.yz[i][j][k -
								     2],
							 t0.yz[i][j][k +
								     1],
							 place, ABCmethod);
			    /* v0->z */
			    /* expression is obtained considering :
			     *  0- Elastic Formulation approximation (no anelasticity part taken into account)
			     *  1- a 2nd order of approximation for vz
			     *  2- we are searching vz(k=1) as dz(vz)|z=0 consistent with the followings expressions :
			     *
			     *  (dt(tzz)|z=0) = 0 = ( (kappa-2/3 mu)*( dx(vx) + dy(vy) ) + (kappa + 4/3 mu) * dz(vz) )|z=0
			     *
			     *  and 2nd Order of deviration
			     *
			     *  dz(vz)|z=0 = (vz(z=+3/2*DS) - vz(DS/2))/dz
			     *
			     *  so vz(k=1) can be determined with vy, vx, and vz(k=0).
			     */
			    kapx = MDM.kap0[ly0];
			    mux = MDM.mu0[ly0];

			    kapy = MDM.kap0[ly0];
			    muy = MDM.mu0[ly0];

			    v0->z[i][j][k] = v0->z[i][j][k - 1]
				- (kapx - 2. / 3. * mux) / (kapx +
							    4. / 3. *
							    mux) *
				(v0->x[i + 1][j][k] - v0->x[i][j][k])
				- (kapy - 2. / 3. * mux) / (kapy +
							    4. / 3. *
							    muy) *
				(v0->y[i][j][k] - v0->y[i][j - 1][k]);

			}	/* end k = 1 */

			/* k=2 */
	      /*-----*/
			/*
			 * 2nd order approximation.
			 * Details :
			 * Cf."Simulating Seismic Wave propagation in 3D elastic Media Using staggered-Grid Finite Difference"
			 *  [ Robert W.Graves, p. 1099 ]
			 *   Bulletin of the Seismological Society of America Vol 4. August 1996
			 */
			if (k == 2) {
			    v0->x[i][j][k] = v0->x[i][j][k - 1]
				- (v0->z[i][j][k - 1] -
				   v0->z[i - 1][j][k - 1])
				- (v0->x[i][j][k - 1] -
				   v0->x[i][j][k - 2] + v0->z[i][j][k -
								    2] -
				   v0->z[i - 1][j][k - 2]);

			    v0->y[i][j][k] = v0->y[i][j][k - 1]
				- (v0->z[i][j + 1][k - 1] -
				   v0->z[i][j][k - 1])
				- (v0->y[i][j][k - 1] -
				   v0->y[i][j][k - 2] + v0->z[i][j + 1][k -
									2]
				   - v0->z[i][j][k - 2]);

			}

		    }		/* end FREE SURFACE and FREEABS COMMON PART */


		    /* ***************************** */
		    /* ABSORBING LAYER & FREEABS part */
		    /* ***************************** */
		    /* NB : we also compute phit here
		     * For FREEABS,
		     * we only need to atenuate for k=1; the rest is already computed with symetries.
		     * what only differs is phitzzz = 0. since dz(tzz)==0 at Free Surface
		     */
		    if ((place == ABSORBINGLAYER) ||
			(place == FREEABS && k == PRM.zMax0 - 1)) {
			/* initialize */
			if (model == GEOLOGICAL) {
			    int med1, med2, med3, med4;

			    med1 = MDM.imed[i][j][k];
			    vp = RhoMuKap2Vp(MDM.rho0[med1], MDM.mu0[med1], MDM.kap0[med1]);	/* vx */

			    med2 = MDM.imed[i][j + 1][k];	/* vy */
			    med3 = MDM.imed[i + 1][j][k];
			    med4 = MDM.imed[i + 1][j + 1][k];

			    rhoxy =
				(MDM.rho0[med1] + MDM.mu0[med2] +
				 MDM.mu0[med3] + MDM.mu0[med4]) * 0.25;
			    muxy =
				averageInverseDouble4(MDM.mu0[med1],
						      MDM.mu0[med2],
						      MDM.mu0[med3],
						      MDM.mu0[med4],
						      model);
			    kapxy =
				averageInverseDouble4(MDM.kap0[med1],
						      MDM.kap0[med2],
						      MDM.kap0[med3],
						      MDM.kap0[med4],
						      model);

			    med2 = MDM.imed[i][j][k + 1];	/* vz */
			    med3 = MDM.imed[i + 1][j][k];
			    med4 = MDM.imed[i + 1][j][k + 1];

			    rhoxz =
				(MDM.rho0[med1] + MDM.mu0[med2] +
				 MDM.mu0[med3] + MDM.mu0[med4]) * 0.25;
			    muxz =
				averageInverseDouble4(MDM.mu0[med1],
						      MDM.mu0[med2],
						      MDM.mu0[med3],
						      MDM.mu0[med4],
						      model);
			    kapxz =
				averageInverseDouble4(MDM.kap0[med1],
						      MDM.kap0[med2],
						      MDM.kap0[med3],
						      MDM.kap0[med4],
						      model);

			} else if (model == LAYER) {

			    vp = RhoMuKap2Vp(MDM.rho0[ly0], MDM.mu0[ly0], MDM.kap0[ly0]);	/* vx */

			    muxy = MDM.mu0[ly0];	/* vy */
			    kapxy = MDM.kap0[ly0];
			    rhoxy = MDM.rho0[ly0];

			    muxz = MDM.mu2[ly2];	/* vz */
			    kapxz = MDM.kap2[ly2];
			}	/* end if model */
			vpxy = RhoMuKap2Vp(rhoxy, muxy, kapxy);	/* vy */
			vpxz = RhoMuKap2Vp(rhoxz, muxz, kapxz);	/* vz */

			/* Calculation of vx */
			if (ABCmethod == CPML) {
			    phixdum = ABC->phitxxx[npml];
			    phiydum = ABC->phitxyy[npml];
			    phizdum = ABC->phitxzz[npml];

			    ABC->phitxxx[npml] =
				CPML4(vp, ABC->dumpx[i], ABC->alphax[i],
				      ABC->kappax[i], phixdum, DS, DT,
				      t0.xx[i - 1][j][k], t0.xx[i][j][k],
				      t0.xx[i - 2][j][k],
				      t0.xx[i + 1][j][k]);
			    ABC->phitxyy[npml] =
				CPML4(vp, ABC->dumpy[j], ABC->alphay[j],
				      ABC->kappay[j], phiydum, DS, DT,
				      t0.xy[i][j - 1][k], t0.xy[i][j][k],
				      t0.xy[i][j - 2][k],
				      t0.xy[i][j + 1][k]);
			    ABC->phitxzz[npml] =
				CPML4(vp, ABC->dumpz[k], ABC->alphaz[k],
				      ABC->kappaz[k], phizdum, DS, DT,
				      t0.xz[i][j][k - 1], t0.xz[i][j][k],
				      t0.xz[i][j][k - 2],
				      t0.xz[i][j][k + 1]);
			    v0->x[i][j][k] +=
				bx * DT * (ABC->phitxxx[npml] +
					   ABC->phitxyy[npml] +
					   ABC->phitxzz[npml]);
			} else if (ABCmethod == PML) {
			    xdum = ABC->vxx[npml];
			    ydum = ABC->vxy[npml];
			    zdum = ABC->vxz[npml];

			    ABC->vxx[npml] =
				PMLdump4(bx, DT, DS, vp * ABC->dumpx[i],
					 xdum, t0.xx[i - 1][j][k],
					 t0.xx[i][j][k],
					 t0.xx[i - 2][j][k],
					 t0.xx[i + 1][j][k]);
			    ABC->vxy[npml] =
				PMLdump4(bx, DT, DS, vp * ABC->dumpy[j],
					 ydum, t0.xy[i][j - 1][k],
					 t0.xy[i][j][k],
					 t0.xy[i][j - 2][k],
					 t0.xy[i][j + 1][k]);
			    ABC->vxz[npml] =
				PMLdump4(bx, DT, DS, vp * ABC->dumpz[k],
					 zdum, t0.xz[i][j][k - 1],
					 t0.xz[i][j][k],
					 t0.xz[i][j][k - 2],
					 t0.xz[i][j][k + 1]);
			    v0->x[i][j][k] =
				ABC->vxx[npml] + ABC->vxy[npml] +
				ABC->vxz[npml];
			}

			/* Calculation of vy */
			if (ABCmethod == CPML) {
			    phixdum = ABC->phitxyx[npml];
			    phiydum = ABC->phityyy[npml];
			    phizdum = ABC->phityzz[npml];

			    ABC->phitxyx[npml] =
				CPML4(vpxy, ABC->dumpx2[i],
				      ABC->alphax2[i], ABC->kappax2[i],
				      phixdum, DS, DT, t0.xy[i][j][k],
				      t0.xy[i + 1][j][k],
				      t0.xy[i - 1][j][k],
				      t0.xy[i + 2][j][k]);
			    ABC->phityyy[npml] =
				CPML4(vpxy, ABC->dumpy2[j],
				      ABC->alphay2[j], ABC->kappay2[j],
				      phiydum, DS, DT, t0.yy[i][j][k],
				      t0.yy[i][j + 1][k],
				      t0.yy[i][j - 1][k],
				      t0.yy[i][j + 2][k]);
			    ABC->phityzz[npml] =
				CPML4(vpxy, ABC->dumpz[k], ABC->alphaz[k],
				      ABC->kappaz[k], phizdum, DS, DT,
				      t0.yz[i][j][k - 1], t0.yz[i][j][k],
				      t0.yz[i][j][k - 2],
				      t0.yz[i][j][k + 1]);

			    v0->y[i][j][k] +=
				by * DT * (ABC->phitxyx[npml] +
					   ABC->phityyy[npml] +
					   ABC->phityzz[npml]);
			} else if (ABCmethod == PML) {
			    xdum = ABC->vyx[npml];
			    ydum = ABC->vyy[npml];
			    zdum = ABC->vyz[npml];
			    ABC->vyx[npml] =
				PMLdump4(by, DT, DS, vpxy * ABC->dumpx2[i],
					 xdum, t0.xy[i][j][k],
					 t0.xy[i + 1][j][k],
					 t0.xy[i - 1][j][k],
					 t0.xy[i + 2][j][k]);
			    ABC->vyy[npml] =
				PMLdump4(by, DT, DS, vpxy * ABC->dumpy2[j],
					 ydum, t0.yy[i][j][k],
					 t0.yy[i][j + 1][k],
					 t0.yy[i][j - 1][k],
					 t0.yy[i][j + 2][k]);
			    ABC->vyz[npml] =
				PMLdump4(by, DT, DS, vpxy * ABC->dumpz[k],
					 zdum, t0.yz[i][j][k - 1],
					 t0.yz[i][j][k],
					 t0.yz[i][j][k - 2],
					 t0.yz[i][j][k + 1]);
			    v0->y[i][j][k] =
				ABC->vyx[npml] + ABC->vyy[npml] +
				ABC->vyz[npml];
			}

			/* Calculation of vz */
			if (ABCmethod == CPML) {
			    phixdum = ABC->phitxzx[npml];
			    phiydum = ABC->phityzy[npml];
			    phizdum = ABC->phitzzz[npml];

			    ABC->phitxzx[npml] =
				CPML4(vpxz, ABC->dumpx2[i],
				      ABC->alphax2[i], ABC->kappax2[i],
				      phixdum, DS, DT, t0.xz[i][j][k],
				      t0.xz[i + 1][j][k],
				      t0.xz[i - 1][j][k],
				      t0.xz[i + 2][j][k]);
			    ABC->phityzy[npml] =
				CPML4(vpxz, ABC->dumpy[j], ABC->alphay[j],
				      ABC->kappay[j], phiydum, DS, DT,
				      t0.yz[i][j - 1][k], t0.yz[i][j][k],
				      t0.yz[i][j - 2][k],
				      t0.yz[i][j + 1][k]);
			    if (place == ABSORBINGLAYER) {
				ABC->phitzzz[npml] =
				    CPML4(vpxz, ABC->dumpz2[k],
					  ABC->alphaz2[k], ABC->kappaz2[k],
					  phizdum, DS, DT, t0.zz[i][j][k],
					  t0.zz[i][j][k + 1],
					  t0.zz[i][j][k - 1],
					  t0.zz[i][j][k + 2]);
			    } else if (place == FREEABS && k == PRM.zMax0 - 1) {	/* phitzzz = 0. since dz(tzz)==0 at Free Surface */
				ABC->phitzzz[npml] = 0.;
			    }

			    v0->z[i][j][k] +=
				bz * DT * (ABC->phitxzx[npml] +
					   ABC->phityzy[npml] +
					   ABC->phitzzz[npml]);
			} else if (ABCmethod == PML) {
			    xdum = ABC->vzx[npml];
			    ydum = ABC->vzy[npml];
			    zdum = ABC->vzz[npml];
			    ABC->vzx[npml] =
				PMLdump4(bz, DT, DS, vpxz * ABC->dumpx2[i],
					 xdum, t0.xz[i][j][k],
					 t0.xz[i + 1][j][k],
					 t0.xz[i - 1][j][k],
					 t0.xz[i + 2][j][k]);
			    ABC->vzy[npml] =
				PMLdump4(bz, DT, DS, vpxz * ABC->dumpy[j],
					 ydum, t0.yz[i][j - 1][k],
					 t0.yz[i][j][k],
					 t0.yz[i][j - 2][k],
					 t0.yz[i][j + 1][k]);
			    if (place == ABSORBINGLAYER) {
				ABC->vzz[npml] =
				    PMLdump4(bz, DT, DS,
					     vpxz * ABC->dumpz2[k], zdum,
					     t0.zz[i][j][k],
					     t0.zz[i][j][k + 1],
					     t0.zz[i][j][k - 1],
					     t0.zz[i][j][k + 2]);
			    } else if (place == FREEABS && k == PRM.zMax0 - 1) {	/* phitzzz = 0. since dz(tzz)==0 at Free Surface */
				ABC->vzz[npml] = 0.;
			    }
			    v0->z[i][j][k] =
				ABC->vzx[npml] + ABC->vzy[npml] +
				ABC->vzz[npml];
			}
			/* end of Calculation of Vz */
		    }		/*         end of ( place == ABSORBINGLAYER  ) */



		   /*=====================================================*
		 * ANELASTIC PART : ANOTHER anelasticity method
		 *
		 *
		 *=====================================================*/
		    /* INFOS:
		     * Only "another anelasticity method" modify the velocity
		     * other method only modify the stress part, so we do nothing there
		     *
		     */
		    if (ANLmethod == ANOTHER) {
			if (model == GEOLOGICAL) {
			} else if (model == LAYER) {
			    v0->x[i][j][k] = ANL.amp[ly0] * v0->x[i][j][k];
			    v0->y[i][j][k] = ANL.amp[ly0] * v0->y[i][j][k];
			    if (ly0 != ly2) {
				v0->z[i][j][k] =
				    ANL.amp2[ly2] * v0->z[i][j][k];
			    } else {
				v0->z[i][j][k] =
				    ANL.amp[ly0] * v0->z[i][j][k];
			    }

			}	/* end model */

		    }		/* end of if ANLmethod */



		    /*=========================================*
		 * Add Source PART                histfile *
		 *=========================================*/
		    /* Ajout des Sources.hist */
		    if (source == HISTFILE) {

			v0->x[i][j][k] += bx * SRC.fx[i][j][k] * DT / DS;
			v0->y[i][j][k] += by * SRC.fy[i][j][k] * DT / DS;
			v0->z[i][j][k] += bz * SRC.fz[i][j][k] * DT / DS;
		    }		/* end of if source */


		}		/* end for k */
	    }			/* end for j */
	}			/* end for i */
#ifdef OMP
    }
#endif
#ifdef LUCAS_INSTRUMENTATION
    SCOREP_USER_REGION_END(computeVelocity)
#endif
}				/* endof ComputeVelocity */

  /*================================================*
   * Add Source PART                Velocity Vector *
   *================================================*/
void computeSource(struct VELOCITY *v0,
		   struct PARAMETERS PRM,
		   struct MEDIUM MDM, struct SOURCE SRC, int l)
{
#ifdef LUCAS_INSTRUMENTATION
    SCOREP_USER_REGION_DEFINE(computeSource)
	SCOREP_USER_REGION_BEGIN(computeSource, "source",
				 SCOREP_USER_REGION_TYPE_COMMON)
#endif
/* Variable declaration */
    int is, imp, jmp, k, icpu, jcpu, ly0;
    int med1, med2, med3, med4;
    double rho, rhoxy;
    double DT;			/* time step */
    double PI;			/* pi */

/* Mapping */
    DT = PRM.dt;
    PI = PRM.pi;

    /* We add the velocity vector which stands for the source here */
    for (is = 0; is < SRC.iSrc; is++) {
	icpu = PRM.i2icpu_array[SRC.ixhypo[is]];
	imp = PRM.i2imp_array[SRC.ixhypo[is]];

	jcpu = PRM.j2jcpu_array[SRC.iyhypo[is]];
	jmp = PRM.j2jmp_array[SRC.iyhypo[is]];

	k = SRC.izhypo[is];

	if (PRM.coords[0] == icpu && PRM.coords[1] == jcpu) {
	    if (model == GEOLOGICAL) {
		int med1, med2, med3, med4;

		med1 = MDM.imed[imp][jmp][k];
		med2 = MDM.imed[imp][jmp + 1][k];
		med3 = MDM.imed[imp + 1][jmp][k];
		med4 = MDM.imed[imp + 1][jmp + 1][k];

		rho = MDM.rho0[med1];
		rhoxy = 0.25 * (MDM.rho0[med1] + MDM.rho0[med2] +
				MDM.rho0[med3] + MDM.rho0[med4]);

	    } else if (model == LAYER) {
		ly0 = MDM.k2ly0[k];
		rho = MDM.rho0[ly0];
		rhoxy = MDM.rho0[ly0];
	    }

	    v0->x[imp][jmp][k] +=
		-sin(135. * PI / 180.) * 10.e6 * 2.0 * PI * PI * 7. * 7. *
		((l - 1) * DT -
		 1.2 / 7.) * exp(-PI * PI * 7. * 7. * ((l - 1) * DT -
						       1.2 / 7.) * ((l -
								     1) *
								    DT -
								    1.2 /
								    7.)) *
		DT / rho;
	    v0->y[imp][jmp][k] +=
		-cos(135. * PI / 180.) * 10.e6 * 2.0 * PI * PI * 7. * 7. *
		((l - 1) * DT -
		 1.2 / 7.) * exp(-PI * PI * 7. * 7. * ((l - 1) * DT -
						       1.2 / 7.) * ((l -
								     1) *
								    DT -
								    1.2 /
								    7.)) *
		DT / rhoxy;

	}			/* end of if icpu && jcpu */
    }				/* end of is */
#ifdef LUCAS_INSTRUMENTATION
    SCOREP_USER_REGION_END(computeSource)
#endif
}				/* end of function computeSource */
