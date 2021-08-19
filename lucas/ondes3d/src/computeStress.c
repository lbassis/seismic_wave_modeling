/* 
   Compute Stress field
   TO DEBUG : FREEABS
*/
#ifdef LUCAS_INSTRUMENTATION
#include <scorep/SCOREP_User.h>
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


#ifdef OMP
#include <omp.h>
#endif


#include "../include/struct.h"
#include "../include/computeStress.h"
#include "../include/inlineFunctions.h"
#include "../include/options.h"
/* ######################### Local inline Functions ###################################### */
/* ## PML/CPML ## */


/* ## DayAndBradley anelasticity ## */


/* ## KRISTEK and Moczo anelasticity ## */


/* OMP */
extern int numThreads_Level_1;


/* ######################### the Function ###################################### */
void ComputeStress(
		      /* Main Outputs (we also increment Absorbing Layer functions) */
		      struct STRESS *t0,
		      /* INPUTS */
		      struct VELOCITY v0, struct MEDIUM MDM, struct PARAMETERS PRM, struct ABSORBING_BOUNDARY_CONDITION ABC, struct ANELASTICITY ANL, int mpmx_begin, int mpmx_end,	/* local computed domain */
		      int mpmy_begin, int mpmy_end)
{
#ifdef LUCAS_INSTRUMENTATION
    SCOREP_USER_REGION_DEFINE(computeStress)
	SCOREP_USER_REGION_BEGIN(computeStress, "stress",
				 SCOREP_USER_REGION_TYPE_COMMON)
#endif
	/* approximations of a value in the corner of the cube */
    double kapxyz,		/* rigidity and mu */
     kapxy, kapxz, kapx, kapy, kapz, muxy, muxz, mux, muy, muz, muxyz;

    /* PML */
    double b1, b2;
    double xdum, ydum, zdum;


    int ly0, ly2;		/* layer index */

    /*  */
    enum typePlace place;	/* What type of cell  */
    long int npml;		/* index in Absorbing Layer */
    int i, j, k;		/* local position of the cell */
    int imp, jmp;		/* global position of the cell (NB : kmp=k) */

    /* For mapping */
    double DS, DT;		/* PARAMETERS */


    double ***KSILxx, ***KSILyy, ***KSILzz,	/* ANELASTICITY : K&M  */
    ***KSILxy, ***KSILxz, ***KSILyz;

    /* Verify Paramters */
    assert(interface == USUAL || interface == KMINTERFACE);
    assert(ABCmethod == CPML || ABCmethod == PML);
    assert(surface == ABSORBING || surface == FREE);
    assert(ANLmethod == ELASTIC || ANLmethod == ANOTHER ||
	   ANLmethod == DAYandBRADLEY || ANLmethod == KRISTEKandMOCZO);

    /* mapping */
    DS = PRM.ds;
    DT = PRM.dt;
    if (ANLmethod == KRISTEKandMOCZO) {
	KSILxx = ANL.ksilxx;
	KSILyy = ANL.ksilyy;
	KSILzz = ANL.ksilzz;
	KSILxy = ANL.ksilxy;
	KSILxz = ANL.ksilxz;
	KSILyz = ANL.ksilyz;
    }








#ifdef OMP
#pragma omp parallel default( shared ) 	\
private( kapx ) 			\
private( mux, muy, muz, muxyz ) 	\
private( b1, b2 ) 			\
private( ly0, ly2 )			\
private( place, npml ) 			\
private( i, j, k ) 			\
private( imp, jmp ) num_threads(numThreads_Level_1)
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

		    /* jump "Not computed area" */
		    if ((place == OUTSIDE) || (place == LIMIT)) {
			continue;
		    }
		    /* find the right npml number */
		    if ((place == ABSORBINGLAYER) || (place == FREEABS)) {
			npml = ABC.ipml[i][j][k];
		    }
		    /* medium */
		    if (model == GEOLOGICAL) {
			int med1, med2;
			med1 = MDM.imed[i][j][k];
			/* x corner */
			med1 = MDM.imed[i][j][k];
			med2 = MDM.imed[i + 1][j][k];
			mux =
			    averageInverseDouble2(MDM.mu0[med1],
						  MDM.mu0[med2], model);
			kapx =
			    averageInverseDouble2(MDM.kap0[med1],
						  MDM.kap0[med2], model);
			/* y corner */
			med2 = MDM.imed[i][j + 1][k];
			muy =
			    averageInverseDouble2(MDM.mu0[med1],
						  MDM.mu0[med2], model);

			/* z corner */
			med1 = MDM.imed[i][j][k];
			med2 = MDM.imed[i][j][k + 1];
			muz =
			    averageInverseDouble2(MDM.mu0[med1],
						  MDM.mu0[med2], model);

			/* xyz corner */
			muxyz =
			    CornerXYZ_GeolInverse(MDM.mu0, i, j, k, MDM);

		    } else if (model == LAYER) {
			/* Warning : k2ly0 & k2ly2
			   give 0 if k in FREEABS or if depth(k) > laydep[0] in general */
			ly0 = MDM.k2ly0[k];
			ly2 = MDM.k2ly2[k];

			kapx = MDM.kap0[ly0];
			mux = MDM.mu0[ly0];
			muy = MDM.mu0[ly0];

			muz = MDM.mu2[ly2];
			muxyz = MDM.mu2[ly2];
		    }		/* end if model */

		    /*=====================================================*\
		  ELASTIC PART : 

		  structure :
		  -- common elastic initialisation
		  -- REGULAR & ABSORBING LAYER common part
		  -- ABSORBING LAYER special part
		  -- FREESURFACE & FREEABS common part
		  -- FREEABS special part

		  \*=====================================================*/
		    if ((place == REGULAR) || (place == ABSORBINGLAYER)) {
		  /*********************************************/
			/* REGULAR & ABSORBING LAYER initialisations */
		  /*********************************************/

			/* NB : 
			 * We Modify the derivative in CPML using kappa and alpha.
			 * In regular domain or when PML is used, we do not modify derivatives, 
			 * that is to say :
			 * kapCPML = 1.
			 * aCPML = 0.,
			 * 
			 * So kapCPML and aCMPL are ingnored in Regular domain and PML formulation,
			 * there is no need to make a special case for them
			 */
			/* Source : An improved PML for the wave equation
			   - Dimitri Komatitsch & Roland Martin, [Geophsysics 2007] */

			/* Computation of Stress T  */
			t0->xx[i][j][k] += staggards4(kapx, mux,
						      ABC.kappax2[i],
						      ABC.kappay[j],
						      ABC.kappaz[k], DT,
						      DS, v0.x[i][j][k],
						      v0.x[i + 1][j][k],
						      v0.x[i - 1][j][k],
						      v0.x[i + 2][j][k],
						      v0.y[i][j - 1][k],
						      v0.y[i][j][k],
						      v0.y[i][j - 2][k],
						      v0.y[i][j + 1][k],
						      v0.z[i][j][k - 1],
						      v0.z[i][j][k],
						      v0.z[i][j][k - 2],
						      v0.z[i][j][k + 1],
						      place, ABCmethod);

			t0->yy[i][j][k] += staggards4(kapx, mux,
						      ABC.kappay[j],
						      ABC.kappax2[i],
						      ABC.kappaz[k], DT,
						      DS,
						      v0.y[i][j - 1][k],
						      v0.y[i][j][k],
						      v0.y[i][j - 2][k],
						      v0.y[i][j + 1][k],
						      v0.x[i][j][k],
						      v0.x[i + 1][j][k],
						      v0.x[i - 1][j][k],
						      v0.x[i + 2][j][k],
						      v0.z[i][j][k - 1],
						      v0.z[i][j][k],
						      v0.z[i][j][k - 2],
						      v0.z[i][j][k + 1],
						      place, ABCmethod);

			t0->zz[i][j][k] += staggards4(kapx, mux,
						      ABC.kappaz[k],
						      ABC.kappax2[i],
						      ABC.kappay[j], DT,
						      DS,
						      v0.z[i][j][k - 1],
						      v0.z[i][j][k],
						      v0.z[i][j][k - 2],
						      v0.z[i][j][k + 1],
						      v0.x[i][j][k],
						      v0.x[i + 1][j][k],
						      v0.x[i - 1][j][k],
						      v0.x[i + 2][j][k],
						      v0.y[i][j - 1][k],
						      v0.y[i][j][k],
						      v0.y[i][j - 2][k],
						      v0.y[i][j + 1][k],
						      place, ABCmethod);

			t0->xy[i][j][k] += staggardt4(muy,
						      ABC.kappax[i],
						      ABC.kappay2[j], DT,
						      DS,
						      v0.y[i - 1][j][k],
						      v0.y[i][j][k],
						      v0.y[i - 2][j][k],
						      v0.y[i + 1][j][k],
						      v0.x[i][j][k],
						      v0.x[i][j + 1][k],
						      v0.x[i][j - 1][k],
						      v0.x[i][j + 2][k],
						      place, ABCmethod);

			t0->xz[i][j][k] += staggardt4(muz,
						      ABC.kappax[i],
						      ABC.kappaz2[k], DT,
						      DS,
						      v0.z[i - 1][j][k],
						      v0.z[i][j][k],
						      v0.z[i - 2][j][k],
						      v0.z[i + 1][j][k],
						      v0.x[i][j][k],
						      v0.x[i][j][k + 1],
						      v0.x[i][j][k - 1],
						      v0.x[i][j][k + 2],
						      place, ABCmethod);

			t0->yz[i][j][k] += staggardt4(muxyz,
						      ABC.kappay2[j],
						      ABC.kappaz2[k], DT,
						      DS, v0.z[i][j][k],
						      v0.z[i][j + 1][k],
						      v0.z[i][j - 1][k],
						      v0.z[i][j + 2][k],
						      v0.y[i][j][k],
						      v0.y[i][j][k + 1],
						      v0.y[i][j][k - 1],
						      v0.y[i][j][k + 2],
						      place, ABCmethod);
		    }

		    /* ****************************** */
		    /*  ABSORBING LAYER special part  */
		    /* ****************************** */
		    if (place == ABSORBINGLAYER) {
			if (ABCmethod == CPML) {
			    b1 = kapx + 4. * mux / 3.;
			    b2 = kapx - 2. * mux / 3.;

			    t0->xx[i][j][k] +=
				DT * b2 * (ABC.phivyy[npml] +
					   ABC.phivzz[npml]) +
				DT * b1 * ABC.phivxx[npml];
			    t0->yy[i][j][k] +=
				DT * b2 * (ABC.phivxx[npml] +
					   ABC.phivzz[npml]) +
				DT * b1 * ABC.phivyy[npml];
			    t0->zz[i][j][k] +=
				DT * b2 * (ABC.phivxx[npml] +
					   ABC.phivyy[npml]) +
				DT * b1 * ABC.phivzz[npml];

			    t0->xy[i][j][k] +=
				DT * muy * (ABC.phivyx[npml] +
					    ABC.phivxy[npml]);
			    t0->xz[i][j][k] +=
				DT * muz * (ABC.phivzx[npml] +
					    ABC.phivxz[npml]);
			    t0->yz[i][j][k] +=
				DT * muxyz * (ABC.phivzy[npml] +
					      ABC.phivyz[npml]);

			} else if (ABCmethod == PML) {
			    t0->xx[i][j][k] =
				ABC.txxx[npml] + ABC.txxy[npml] +
				ABC.txxz[npml];
			    t0->yy[i][j][k] =
				ABC.tyyx[npml] + ABC.tyyy[npml] +
				ABC.tyyz[npml];
			    t0->zz[i][j][k] =
				ABC.tzzx[npml] + ABC.tzzy[npml] +
				ABC.tzzz[npml];

			    t0->xy[i][j][k] =
				ABC.txyx[npml] + ABC.txyy[npml];
			    t0->xz[i][j][k] =
				ABC.txzx[npml] + ABC.txzz[npml];
			    t0->yz[i][j][k] =
				ABC.tyzy[npml] + ABC.tyzz[npml];
			}
		    }
		    /* ************************************** */
		    /*  FREESURFACE and FREEABS common part   */
		    /* ************************************** */
		    /* # Description :
		     * -- k=0 
		     * we keep 4th order approximation of t0 for k=0 (just before the free surface),
		     * so we need values of the 2 above cells : k=1 (fully), k=2(partially).
		     * Partially, since only t0->xz(k=0) and t0->yz(k=0) depend on v0.x(k=2) and v0.y(k=2) 
		     * Details of those v0.x,v0.y can be found in the Function ComputeVelocity
		     * -- k=1 (freeSurface)
		     * 
		     * # From local index to z coordinates
		     * for k=1, t0->xx,t0->yy,t0->zz, t0->xy -> z=0; 
		     *          t0->xz,t0->yz            -> z=DS/2.
		     * for k=2, t0->xx,t0->yy,t0->zz, t0->xy -> z=DS 
		     *          t0->xz,t0->yz            -> z=3*DS/2. 
		     *          we only need to compute t0->zz, t0->xz and t0->yz 
		     *          (for v0.z(k=0), v0.x(k=1), v0.y(k=1) respectively)
		     * # Equations
		     * Cf."Simulating Seismic Wave propagation in 3D elastic Media Using staggered-Grid Finite Difference"
		     *  [ Robert W.Graves, p. 1099 ]
		     *   Bulletin of the Seismological Society of America Vol 4. August 1996
		     */
		    if (place == FREESURFACE || place == FREEABS) {
			if (k == 1) {
			    /* imposed values */
			    t0->xz[i][j][1] = -t0->xz[i][j][0];
			    t0->yz[i][j][1] = -t0->yz[i][j][0];
			    t0->zz[i][j][1] = 0.;

			    /* for t0->xx, t0->yy, we consider elastic formulation 
			     * giving dt(tzz)|z=0 = 0 => vz,t0->xx and t0->yy formulation
			     * Cf. eqn 98 to 100 p.13, Technical Note : Formulation of Finite Difference Method [ Hideo Aochi ]
			     */
			    b1 = 4. * mux * (kapx + mux / 3.) / (kapx +
								 4. / 3. *
								 mux);
			    b2 = 2. * mux * (kapx -
					     2. / 3. * mux) / (kapx +
							       4. / 3. *
							       mux);

			    t0->xx[i][j][k] +=
				b1 * DT * ((9. / 8.) *
					   (v0.x[i + 1][j][k] -
					    v0.x[i][j][k])
					   -
					   (1. / 24.) *
					   (v0.x[i + 2][j][k] -
					    v0.x[i -
						 1][j][k])) / (DS *
							       ABC.
							       kappax2[i])
				+
				b2 * DT * ((9. / 8.) *
					   (v0.y[i][j][k] -
					    v0.y[i][j - 1][k])
					   -
					   (1. / 24.) *
					   (v0.y[i][j + 1][k] -
					    v0.y[i][j -
						    2][k])) / (DS *
							       ABC.
							       kappay[j]);

			    t0->yy[i][j][k] +=
				b1 * DT * ((9. / 8.) *
					   (v0.y[i][j][k] -
					    v0.y[i][j - 1][k])
					   -
					   (1. / 24.) *
					   (v0.y[i][j + 1][k] -
					    v0.y[i][j -
						    2][k])) / (DS *
							       ABC.
							       kappay[j])
				+
				b2 * DT * ((9. / 8.) *
					   (v0.x[i + 1][j][k] -
					    v0.x[i][j][k])
					   -
					   (1. / 24.) *
					   (v0.x[i + 2][j][k] -
					    v0.x[i -
						 1][j][k])) / (DS *
							       ABC.
							       kappax2[i]);

			    /* t0->xy computed like usual */
			    t0->xy[i][j][k] +=
				staggardt4(muy, un, un, DT, DS,
					   v0.y[i - 1][j][k],
					   v0.y[i][j][k],
					   v0.y[i - 2][j][k],
					   v0.y[i + 1][j][k],
					   v0.x[i][j][k],
					   v0.x[i][j + 1][k],
					   v0.x[i][j - 1][k],
					   v0.x[i][j + 2][k], place,
					   ABCmethod);
			}	/* end if k=1 */
			if (k == 2) {
			    /* imposed values */
			    /* (other values are not needed) */
			    t0->xz[i][j][2] = -t0->xz[i][j][-1];
			    t0->yz[i][j][2] = -t0->yz[i][j][-1];
			    t0->zz[i][j][2] = -t0->zz[i][j][0];
			}
		    }		/* end FreeSurface and FreeAbs common part */

		    /*********************** */
		    /* FREEABS special part  */
		    /*********************** */
		    if (place == FREEABS) {
			/* Nothing to do for imposed values 
			 *  (since k=0 and k=-1  already contains PML/CPML) 
			 * what's left is : for k=1, t0->xx, t0->yy and t0->xy
			 */
			if (k == 1) {
			    if (ABCmethod == CPML) {
				//CTUL 07.12
				//b1 = kapx + 4.*mux/3.;
				//b2 = kapx - 2.*mux/3.;
				b1 = 4. * mux * (kapx + mux / 3.) / (kapx +
								     4. /
								     3. *
								     mux);
				b2 = 2. * mux * (kapx -
						 2. / 3. * mux) / (kapx +
								   4. /
								   3. *
								   mux);


				t0->xx[i][j][k] +=
				    DT * b2 * (ABC.phivyy[npml] +
					       ABC.phivzz[npml]) +
				    DT * b1 * ABC.phivxx[npml];
				t0->yy[i][j][k] +=
				    DT * b2 * (ABC.phivxx[npml] +
					       ABC.phivzz[npml]) +
				    DT * b1 * ABC.phivyy[npml];
				t0->xy[i][j][k] +=
				    DT * muy * (ABC.phivyx[npml] +
						ABC.phivxy[npml]);
			    } else if (ABCmethod == PML) {
				t0->xx[i][j][k] =
				    ABC.txxx[npml] + ABC.txxy[npml] +
				    ABC.txxz[npml];
				t0->yy[i][j][k] =
				    ABC.tyyx[npml] + ABC.tyyy[npml] +
				    ABC.tyyz[npml];
				t0->xy[i][j][k] =
				    ABC.txyx[npml] + ABC.txyy[npml];
			    }

			}	/* end if k == 1 */
		    }		/* end FREEABS special part */

		    /*=====================================================*	\
		  ANELASTIC PART : 
	    
		  structure for each method :
		  -- common initialisations
		  -- REGULAR & ABSORBING LAYER common part
		  -- ABSORBING LAYER special part
		  -- FREESURFACE & FREEABS common part
		  -- FREEABS special part
	    
		  \*=====================================================*/
		    /*******************************************/
		    /* Add Anelastic Contribution              */
		    /*******************************************/
		    /* NB: 
		     * Anelastic coefficient are already taking into account CPML special treatment
		     * Thus we do not need to make special cases for them, here.
		     * For more information about CPML special treatments and anelasticity, Cf. the function Compute2ermediates 
		     */
#ifndef NOANELASTICITY

		    /* ANOTHER Attenuation method */
		    /* -------------------------- */
		    if (ANLmethod == ANOTHER) {
			if (model == GEOLOGICAL) {
			    exit(EXIT_FAILURE);
			} else if (model == LAYER) {

			    t0->xx[i][j][k] =
				ANL.amp[ly0] * t0->xx[i][j][k];
			    t0->yy[i][j][k] =
				ANL.amp[ly0] * t0->yy[i][j][k];
			    t0->zz[i][j][k] =
				ANL.amp[ly0] * t0->zz[i][j][k];
			    t0->xy[i][j][k] =
				ANL.amp[ly0] * t0->xy[i][j][k];
			    if (ly0 != ly2) {
				t0->xz[i][j][k] =
				    ANL.amp2[ly2] * t0->xz[i][j][k];
				t0->yz[i][j][k] =
				    ANL.amp2[ly2] * t0->yz[i][j][k];
			    } else {
				t0->xz[i][j][k] =
				    ANL.amp[ly0] * t0->xz[i][j][k];
				t0->yz[i][j][k] =
				    ANL.amp[ly0] * t0->yz[i][j][k];
			    }
			}	/* end of if model */

			/* DAY and BRADLEY method     */
			/* -------------------------- */
			/* Description :
			 * PML/CPML, FREEABS specificities are already made in computeIntermediates()
			 */
		    } else if (ANLmethod == DAYandBRADLEY) {
			if (place == REGULAR || place == ABSORBINGLAYER) {
			    t0->xx[i][j][k] += -DT * ANL.ksixx[i][j][k];
			    t0->yy[i][j][k] += -DT * ANL.ksiyy[i][j][k];
			    t0->zz[i][j][k] += -DT * ANL.ksizz[i][j][k];
			    t0->xy[i][j][k] += -DT * ANL.ksixy[i][j][k];
			    t0->xz[i][j][k] += -DT * ANL.ksixz[i][j][k];
			    t0->yz[i][j][k] += -DT * ANL.ksiyz[i][j][k];
			} else if (place == FREESURFACE
				   || place == FREEABS) {
			    if (k == 1) {
				/* imposed values (maintain antisymetric values ) */
				/* Before anelastic part :
				   /*  t0->xz[i][j][1] = - t0->xz[i][j][0]; */
				/*  t0->yz[i][j][1] = - t0->yz[i][j][0]; */
				/*  t0->zz[i][j][1] = 0. ;             */
				t0->xz[i][j][k] +=
				    +DT * ANL.ksixz[i][j][0];
				t0->yz[i][j][k] +=
				    +DT * ANL.ksiyz[i][j][0];

				/* other values (classic) */
				t0->xx[i][j][k] +=
				    -DT * ANL.ksixx[i][j][k];
				t0->yy[i][j][k] +=
				    -DT * ANL.ksiyy[i][j][k];
				t0->xy[i][j][k] +=
				    -DT * ANL.ksixy[i][j][k];
			    }
			    if (k == 2) {
				/* imposed values (maintain antisymetric values ) */
				/*  (other values are not needed) */
				/*        t0->xz[i][j][2] = - t0->xz[i][j][-1]; */
				/*        t0->yz[i][j][2] = - t0->yz[i][j][-1]; */
				/*        t0->zz[i][j][2] = - t0->zz[i][j][0]; */
				t0->xz[i][j][k] +=
				    +DT * ANL.ksixz[i][j][-1];
				t0->yz[i][j][k] +=
				    +DT * ANL.ksiyz[i][j][-1];
				t0->zz[i][j][2] +=
				    +DT * ANL.ksizz[i][j][0];

			    }
			}	/* end FreeSurface & FreeAbs  */

			/* KRISTEK and MOCZO method */
			/* -------------------------- */
		    } else if (ANLmethod == KRISTEKandMOCZO) {

			/* Kristek and Moczo anelasticity */
			double yMuMe[1], yKapMe[1];
			double yMuNeighP[3], yMuNeighN[3];	/* neighbors : x,y,z directions ( positive and negative ) */
			double yKapNeighP[3], yKapNeighN[3];

			double partTraceKsil;	/* intermediates */
			double traceme, tracexP, tracexN, traceyP, traceyN, tracezN, tracezP;	/* traces of me and my neighbors
												   in a direction ( positive or negative ) */
			/* Free Surface formulation :
			 * 1- only some KSIL are computed : KSILxx,KSILyy,KSILzz, and KSILxy
			 * (Cf Function computeIntermediates for details) 
			 *
			 * 2- approximation of sum part (averaging the KSIL of neighbours)
			 *    in z direction is only made on the coefficient below (k=0)
			 *    since KSIL(k=2) does not exist
			 *
			 * 3- FREEABS special formulation is inside KSIL computation (Cf. computeIntermediates())
			 */
			/* here we jump the limit case */
			if ((place == FREESURFACE || place == FREEABS)
			    && (k == PRM.zMax0)) {
			    continue;
			}

			/* find the right yIndimu/kap */
			ChooseY(yMuMe, yMuNeighP, yMuNeighN,	/* OUTPUTS */
				yKapMe, yKapNeighP, yKapNeighN, i, j, k,	/* inputs */
				ANL, MDM);

			/* add anelasticity part */
			traceme =
			    KSILxx[i][j][k] + KSILyy[i][j][k] +
			    KSILzz[i][j][k];

			tracexP =
			    KSILxx[i + 1][j][k] + KSILyy[i + 1][j][k] +
			    KSILzz[i + 1][j][k];
			tracexN =
			    KSILxx[i - 1][j][k] + KSILyy[i - 1][j][k] +
			    KSILzz[i - 1][j][k];

			traceyP =
			    KSILxx[i][j + 1][k] + KSILyy[i][j + 1][k] +
			    KSILzz[i][j + 1][k];
			traceyN =
			    KSILxx[i][j - 1][k] + KSILyy[i][j - 1][k] +
			    KSILzz[i][j - 1][k];

			tracezN =
			    KSILxx[i][j][k - 1] + KSILyy[i][j][k - 1] +
			    KSILzz[i][j][k - 1];
			if (place == REGULAR || place == ABSORBINGLAYER) {
			    tracezP =
				KSILxx[i][j][k + 1] + KSILyy[i][j][k + 1] +
				KSILzz[i][j][k + 1];
			} else if (place == FREE || place == FREEABS) {	/* just a workaround since in k+1 => not in the domain */
			    tracezP = tracezN;
			}


			partTraceKsil =
			    (yKapMe[0] - 2. / 3. * yMuMe[0]) * traceme +
			    0.5 *
			    ((yKapNeighP[0] -
			      2. / 3. * yMuNeighP[0]) * tracexP +
			     (yKapNeighN[0] -
			      2. / 3. * yMuNeighN[0]) * tracexN) +
			    0.5 *
			    ((yKapNeighP[1] -
			      2. / 3. * yMuNeighP[1]) * traceyP +
			     (yKapNeighN[1] -
			      2. / 3. * yMuNeighN[1]) * traceyN) +
			    0.5 *
			    ((yKapNeighP[2] -
			      2. / 3. * yMuNeighP[2]) * tracezP +
			     (yKapNeighN[2] -
			      2. / 3. * yMuNeighN[2]) * tracezN);

			t0->xx[i][j][k] += -DT * (partTraceKsil +
						  2. * SumYlKsil(KSILxx,
								 yMuMe,
								 yMuNeighP,
								 yMuNeighN,
								 i, j, k));

			t0->yy[i][j][k] += -DT * (partTraceKsil +
						  2. * SumYlKsil(KSILyy,
								 yMuMe,
								 yMuNeighP,
								 yMuNeighN,
								 i, j, k));

			t0->zz[i][j][k] += -DT * (partTraceKsil +
						  2. * SumYlKsil(KSILzz,
								 yMuMe,
								 yMuNeighP,
								 yMuNeighN,
								 i, j, k));


			t0->xy[i][j][k] +=
			    -DT * 2. * SumYlKsil(KSILxy, yMuMe, yMuNeighP,
						 yMuNeighN, i, j, k);

			if (place != FREESURFACE && place != FREEABS) {
			    t0->xz[i][j][k] +=
				-DT * 2. * SumYlKsil(KSILxz, yMuMe,
						     yMuNeighP, yMuNeighN,
						     i, j, k);
			    t0->yz[i][j][k] +=
				-DT * 2. * SumYlKsil(KSILyz, yMuMe,
						     yMuNeighP, yMuNeighN,
						     i, j, k);
			}	/* end no freesurface */

		    }		/* end Kristek & Moczo anelasticity method */
#endif				/* ifndef NOANELASTICITY */

		}		/* end for k */
	    }			/* end for j */
	}			/* end for i */

#ifdef OMP
    }
#endif
#ifdef LUCAS_INSTRUMENTATION
    SCOREP_USER_REGION_END(computeStress)
#endif
}				/* endof ComputeStress */


/* ********* */
/* Functions */
/* ********* */
/* KRISTEK and MOCZO Related */
/* GOAL : 
   find the right Y for the cell, and its neighbours */
int ChooseY(			/* OUTPUTS */
	       double *yMuMe, double *yMuNeighP, double *yMuNeighN,
	       double *yKapMe, double *yKapNeighP, double *yKapNeighN,
	       int i, int j, int k,
	       struct ANELASTICITY ANL, struct MEDIUM MDM)
{

    if (model == LAYER) {
	int ly0, ly2;
	ly0 = MDM.k2ly0[k];
	ly2 = MDM.k2ly2[k];

	/* me */
	yMuMe[0] = ANL.ylmu[IndexL(i, j, k)][ly0];
	yKapMe[0] = ANL.ylkap[IndexL(i, j, k)][ly0];
	/* x */
	yMuNeighP[0] = ANL.ylmu[IndexL(i + 1, j, k)][ly0];
	yMuNeighN[0] = ANL.ylmu[IndexL(i - 1, j, k)][ly0];

	yKapNeighP[0] = ANL.ylkap[IndexL(i + 1, j, k)][ly0];
	yKapNeighN[0] = ANL.ylkap[IndexL(i - 1, j, k)][ly0];

	/* y */
	yMuNeighP[1] = ANL.ylmu[IndexL(i, j + 1, k)][ly0];
	yMuNeighN[1] = ANL.ylmu[IndexL(i, j - 1, k)][ly0];

	yKapNeighP[1] = ANL.ylkap[IndexL(i, j + 1, k)][ly0];
	yKapNeighN[1] = ANL.ylkap[IndexL(i, j - 1, k)][ly0];

	/* z */
	if (ly0 != ly2) {	/* interfaces */
	    yMuNeighP[2] = ANL.ylmu2[IndexL(i, j, k + 1)][ly2];
	    yMuNeighN[2] = ANL.ylmu2[IndexL(i, j, k - 1)][ly2];

	    yKapNeighP[2] = ANL.ylkap2[IndexL(i, j, k + 1)][ly2];
	    yKapNeighN[2] = ANL.ylkap2[IndexL(i, j, k - 1)][ly2];

	} else {		/* layers */
	    yMuNeighP[2] = ANL.ylmu2[IndexL(i, j, k + 1)][ly0];
	    yMuNeighN[2] = ANL.ylmu2[IndexL(i, j, k - 1)][ly0];

	    yKapNeighP[2] = ANL.ylkap2[IndexL(i, j, k + 1)][ly0];
	    yKapNeighN[2] = ANL.ylkap2[IndexL(i, j, k - 1)][ly0];
	}

    } else if (model == GEOLOGICAL) {
	int medP, medN;
	/* me */
	medP = MDM.imed[i][j][k];
	yMuMe[0] = ANL.ylmu[IndexL(i, j, k)][medP];
	yKapMe[0] = ANL.ylkap[IndexL(i, j, k)][medP];
	/* x */
	medP = MDM.imed[i + 1][j][k];
	medN = MDM.imed[i - 1][j][k];
	yMuNeighP[0] = ANL.ylmu[IndexL(i + 1, j, k)][medP];
	yMuNeighN[0] = ANL.ylmu[IndexL(i - 1, j, k)][medN];

	yKapNeighP[0] = ANL.ylkap[IndexL(i + 1, j, k)][medP];
	yKapNeighN[0] = ANL.ylkap[IndexL(i - 1, j, k)][medN];

	/* y */
	medP = MDM.imed[i][j + 1][k];
	medN = MDM.imed[i][j - 1][k];
	yMuNeighP[1] = ANL.ylmu[IndexL(i, j + 1, k)][medP];
	yMuNeighN[1] = ANL.ylmu[IndexL(i, j - 1, k)][medN];

	yKapNeighP[1] = ANL.ylkap[IndexL(i, j + 1, k)][medP];
	yKapNeighN[1] = ANL.ylkap[IndexL(i, j - 1, k)][medN];

	/* z */
	medP = MDM.imed[i][j][k + 1];
	medN = MDM.imed[i][j][k - 1];
	yMuNeighP[2] = ANL.ylmu2[IndexL(i, j, k + 1)][medP];
	yMuNeighN[2] = ANL.ylmu2[IndexL(i, j, k - 1)][medN];

	yKapNeighP[2] = ANL.ylkap2[IndexL(i, j, k + 1)][medP];
	yKapNeighN[2] = ANL.ylkap2[IndexL(i, j, k - 1)][medN];
    }				/* end if model */

    return EXIT_SUCCESS;

}				/* End ChooseY */
