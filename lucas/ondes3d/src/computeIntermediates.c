/* 
   Compute Intermediates
*/
#ifdef LUCAS_INSTRUMENTATION
#include <scorep/SCOREP_User.h>
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "../include/struct.h"
#include "../include/inlineFunctions.h"
#include "../include/computeIntermediates.h"
#include "../include/options.h"


static inline double Q2A(double Q, double TM, double Tm, double W0,
			 double PI)
{				/*  to compute Qp & Qs of Day and Bradley anelasticity method */
    return (2. / (PI * Q)) * log(TM / Tm) / (1. -
					     (2. / (PI * Q)) * log(W0 *
								   Tm));
}

static inline int LocalComputeDayBradley(struct ANELASTICITY *ANL,
					 int i, int j, int k,
					 double vpx, double vpy,
					 double vpz, double vpxyz,
					 double mux, double muy,
					 double muz, double muxyz,
					 double kapx, double Apx,
					 double Asx, double Asy,
					 double Asz, double Asxyz,
					 double tk, double DS, double DT,
					 struct VELOCITY v0,
					 struct
					 ABSORBING_BOUNDARY_CONDITION *ABC,
					 long int npml,
					 enum typePlace place)
{
    /* FREE SURFACE */
    double dzV0z;		/* derivative approximation of Vz in z direction  */
    double a, b;		/* intermediates for FREEABS */

    double phixdum, phiydum, phizdum;	/* intermediates */
    double fxxB, fyyB, fzzB,	/* f before computation */
     fxyB, fxzB, fyzB;
    double a_z, b_z;		/* FreeAbs intermediate Coefficients */
    /* mapping */

    double Tm, TM, W0;		/* ANELASTICITY, Day & Bradley method */
    if (ANLmethod == DAYandBRADLEY) {
	Tm = ANL->tm;
	TM = ANL->tM;
	W0 = ANL->w0;
    }

    /* Regular & CPML common part */
    /* ------------------------- */
    if (place == REGULAR || place == ABSORBINGLAYER) {

	/* save previous values */
	fxxB = ANL->fxx[i][j][k];
	fyyB = ANL->fyy[i][j][k];
	fzzB = ANL->fzz[i][j][k];

	fxyB = ANL->fxy[i][j][k];
	fxzB = ANL->fxz[i][j][k];
	fyzB = ANL->fyz[i][j][k];

	/* compute fij */
	/* NB : DB_FKSIs4/t4 are adapted to REGULAR/CPML DOMAIN
	 * DB_FKSIs4/t4 will ignore PML coefficients if place != ABSORBINGLAYER
	 * We do not need to distinguish those cases
	 */
	if (place == REGULAR) {
	    /* xx, yy, zz */
	    ANL->fxx[i][j][k] = FKSIs4(mux, kapx, DS, Asx, Apx,
				       v0.x[i][j][k], v0.x[i + 1][j][k],
				       v0.x[i - 1][j][k],
				       v0.x[i + 2][j][k],
				       v0.y[i][j - 1][k], v0.y[i][j][k],
				       v0.y[i][j - 2][k],
				       v0.y[i][j + 1][k],
				       v0.z[i][j][k - 1], v0.z[i][j][k],
				       v0.z[i][j][k - 2], v0.z[i][j][k + 1]
		);

	    ANL->fyy[i][j][k] = FKSIs4(mux, kapx, DS, Asx, Apx,
				       v0.y[i][j - 1][k], v0.y[i][j][k],
				       v0.y[i][j - 2][k],
				       v0.y[i][j + 1][k], v0.x[i][j][k],
				       v0.x[i + 1][j][k],
				       v0.x[i - 1][j][k],
				       v0.x[i + 2][j][k],
				       v0.z[i][j][k - 1], v0.z[i][j][k],
				       v0.z[i][j][k - 2], v0.z[i][j][k + 1]
		);

	    ANL->fzz[i][j][k] = FKSIs4(mux, kapx, DS, Asx, Apx,
				       v0.z[i][j][k - 1], v0.z[i][j][k],
				       v0.z[i][j][k - 2],
				       v0.z[i][j][k + 1], v0.x[i][j][k],
				       v0.x[i + 1][j][k],
				       v0.x[i - 1][j][k],
				       v0.x[i + 2][j][k],
				       v0.y[i][j - 1][k], v0.y[i][j][k],
				       v0.y[i][j - 2][k], v0.y[i][j + 1][k]
		);
	    /* xy */
	    ANL->fxy[i][j][k] = FKSIt4(muy, DS, Asy,
				       v0.y[i - 1][j][k], v0.y[i][j][k],
				       v0.y[i - 2][j][k],
				       v0.y[i + 1][j][k], v0.x[i][j][k],
				       v0.x[i][j + 1][k],
				       v0.x[i][j - 1][k], v0.x[i][j + 2][k]
		);
	    /* xz */
	    ANL->fxz[i][j][k] = FKSIt4(muz, DS, Asz,
				       v0.z[i - 1][j][k], v0.z[i][j][k],
				       v0.z[i - 2][j][k],
				       v0.z[i + 1][j][k], v0.x[i][j][k],
				       v0.x[i][j][k + 1],
				       v0.x[i][j][k - 1], v0.x[i][j][k + 2]
		);
	    /* yz */
	    ANL->fyz[i][j][j] = FKSIt4(muxyz, DS, Asxyz,
				       v0.z[i][j][k], v0.z[i][j + 1][k],
				       v0.z[i][j - 1][k],
				       v0.z[i][j + 2][k], v0.y[i][j][k],
				       v0.y[i][j][k + 1],
				       v0.y[i][j][k - 1], v0.y[i][j][k + 2]
		);

	} else if (place == ABSORBINGLAYER) {

	    phixdum = ABC->phivxx[npml];
	    phiydum = ABC->phivyy[npml];
	    phizdum = ABC->phivzz[npml];
	    ANL->fxx[i][j][k] =
		FKSIs4CPML(mux, kapx, DS, DT, Asx, Apx, vpx, phixdum,
			   phiydum, phizdum, ABC->dumpx2[i],
			   ABC->alphax2[i], ABC->kappax2[i], ABC->dumpy[j],
			   ABC->alphay[j], ABC->kappay[j], ABC->dumpz[k],
			   ABC->alphaz[k], ABC->kappaz[k], v0.x[i][j][k],
			   v0.x[i + 1][j][k], v0.x[i - 1][j][k],
			   v0.x[i + 2][j][k], v0.y[i][j - 1][k],
			   v0.y[i][j][k], v0.y[i][j - 2][k],
			   v0.y[i][j + 1][k], v0.z[i][j][k - 1],
			   v0.z[i][j][k], v0.z[i][j][k - 2],
			   v0.z[i][j][k + 1]
		);

	    ANL->fyy[i][j][k] =
		FKSIs4CPML(mux, kapx, DS, DT, Asx, Apx, vpx, phiydum,
			   phixdum, phizdum, ABC->dumpy[j], ABC->alphay[j],
			   ABC->kappay[j], ABC->dumpx2[i], ABC->alphax2[i],
			   ABC->kappax2[i], ABC->dumpz[k], ABC->alphaz[k],
			   ABC->kappaz[k], v0.y[i][j - 1][k],
			   v0.y[i][j][k], v0.y[i][j - 2][k],
			   v0.y[i][j + 1][k], v0.x[i][j][k],
			   v0.x[i + 1][j][k], v0.x[i - 1][j][k],
			   v0.x[i + 2][j][k], v0.z[i][j][k - 1],
			   v0.z[i][j][k], v0.z[i][j][k - 2],
			   v0.z[i][j][k + 1]
		);

	    ANL->fzz[i][j][k] =
		FKSIs4CPML(mux, kapx, DS, DT, Asx, Apx, vpx, phizdum,
			   phixdum, phiydum, ABC->dumpz[k], ABC->alphaz[k],
			   ABC->kappaz[k], ABC->dumpx2[i], ABC->alphax2[i],
			   ABC->kappax2[i], ABC->dumpy[j], ABC->alphay[j],
			   ABC->kappay[j], v0.z[i][j][k - 1],
			   v0.z[i][j][k], v0.z[i][j][k - 2],
			   v0.z[i][j][k + 1], v0.x[i][j][k],
			   v0.x[i + 1][j][k], v0.x[i - 1][j][k],
			   v0.x[i + 2][j][k], v0.y[i][j - 1][k],
			   v0.y[i][j][k], v0.y[i][j - 2][k],
			   v0.y[i][j + 1][k]
		);
	    /* xy */
	    phixdum = ABC->phivyx[npml];
	    phiydum = ABC->phivxy[npml];
	    ANL->fxy[i][j][k] = FKSIt4CPML(muy, DS, DT, Asy, vpy,
					   phixdum, phiydum,
					   ABC->dumpx[i], ABC->alphax[i],
					   ABC->kappax[i], ABC->dumpy2[j],
					   ABC->alphay2[j],
					   ABC->kappay2[j],
					   v0.y[i - 1][j][k],
					   v0.y[i][j][k],
					   v0.y[i - 2][j][k],
					   v0.y[i + 1][j][k],
					   v0.x[i][j][k],
					   v0.x[i][j + 1][k],
					   v0.x[i][j - 1][k],
					   v0.x[i][j + 2][k]
		);
	    /* xz */
	    phixdum = ABC->phivzx[npml];
	    phizdum = ABC->phivxz[npml];
	    ANL->fxz[i][j][k] = FKSIt4CPML(muz, DS, DT, Asz, vpz,
					   phixdum, phizdum,
					   ABC->dumpx[i], ABC->alphax[i],
					   ABC->kappax[i], ABC->dumpz2[k],
					   ABC->alphaz2[k],
					   ABC->kappaz2[k],
					   v0.z[i - 1][j][k],
					   v0.z[i][j][k],
					   v0.z[i - 2][j][k],
					   v0.z[i + 1][j][k],
					   v0.x[i][j][k],
					   v0.x[i][j][k + 1],
					   v0.x[i][j][k - 1],
					   v0.x[i][j][k + 2]
		);
	    /* yz */
	    phiydum = ABC->phivzy[npml];
	    phizdum = ABC->phivyz[npml];
	    ANL->fyz[i][j][j] = FKSIt4CPML(muxyz, DS, DT, Asxyz, vpxyz,
					   phiydum, phizdum,
					   ABC->dumpy2[j], ABC->alphay2[j],
					   ABC->kappay2[j], ABC->dumpz2[k],
					   ABC->alphaz2[k],
					   ABC->kappaz2[k], v0.z[i][j][k],
					   v0.z[i][j + 1][k],
					   v0.z[i][j - 1][k],
					   v0.z[i][j + 2][k],
					   v0.y[i][j][k],
					   v0.y[i][j][k + 1],
					   v0.y[i][j][k - 1],
					   v0.y[i][j][k + 2]
		);

	}			/* end CPML */

	/* Compute ANL->ksi */
	ANL->ksixx[i][j][k] =
	    (1. / (tk / DT + 0.5)) * ((tk / DT - 0.5) *
				      ANL->ksixx[i][j][k] + 0.5 * (fxxB +
								   ANL->
								   fxx[i]
								   [j]
								   [k]));
	ANL->ksiyy[i][j][k] =
	    (1. / (tk / DT + 0.5)) * ((tk / DT - 0.5) *
				      ANL->ksiyy[i][j][k] + 0.5 * (fyyB +
								   ANL->
								   fyy[i]
								   [j]
								   [k]));
	ANL->ksizz[i][j][k] =
	    (1. / (tk / DT + 0.5)) * ((tk / DT - 0.5) *
				      ANL->ksizz[i][j][k] + 0.5 * (fzzB +
								   ANL->
								   fzz[i]
								   [j]
								   [k]));
	ANL->ksixy[i][j][k] =
	    (1. / (tk / DT + 0.5)) * ((tk / DT - 0.5) *
				      ANL->ksixy[i][j][k] + 0.5 * (fxyB +
								   ANL->
								   fxy[i]
								   [j]
								   [k]));
	ANL->ksixz[i][j][k] =
	    (1. / (tk / DT + 0.5)) * ((tk / DT - 0.5) *
				      ANL->ksixz[i][j][k] + 0.5 * (fxzB +
								   ANL->
								   fxz[i]
								   [j]
								   [k]));
	ANL->ksiyz[i][j][k] =
	    (1. / (tk / DT + 0.5)) * ((tk / DT - 0.5) *
				      ANL->ksiyz[i][j][k] + 0.5 * (fyzB +
								   ANL->
								   fyz[i]
								   [j]
								   [k]));

    }				/* End regular and absorbing layer */

    /* FREESURFACE & FREEABS    */
    /* ------------------------ */
    /* Description :
     * anelasticities only influence t0 computation (except another attenuation method).
     * Meanwhile, on Free Surface, only t0xx(k=1),t0yy(k=1),t0xy(k=1) are really computed
     * So every coefficient will be computed only for those part 
     *
     * Expressions are simply obtain from approximation of dz(v0.z) in an elastic medium :
     * DT(t0.zz)|z=0 = 0 = lam *(dx(v0.x) + dy(v0.y) + dz(v0.z)) + 2*mu * dz(v0.z);
     *
     */

      /***** freeSurface & freeabs common initialisations ******/
    if (place == FREESURFACE || place == FREEABS) {
	/* We only need : ksixx, ksiyy and ksixy for k = 1
	 *  and so fxx, fyy and fxy
	 */
	if (k == 1) {
	    /* compute coefficients */
	    /* approximate dz(v0.z) (4th order) */
	    dzV0z = (kapx - 2. / 3. * mux) / (kapx + 4. / 3. * mux) *
		(Diff4
		 (DS, v0.x[i][j][k], v0.x[i + 1][j][k], v0.x[i - 1][j][k],
		  v0.x[i + 2][j][k]) + Diff4(DS, v0.y[i][j - 1][k],
					     v0.y[i][j][k],
					     v0.y[i][j - 2][k],
					     v0.y[i][j + 1][k])
		);

	    /* save previous values */
	    fxxB = ANL->fxx[i][j][k];
	    fyyB = ANL->fyy[i][j][k];
	    fxyB = ANL->fxy[i][j][k];

	}			/* end k=1 */
    }				/* end common initialisations */

     /***** freeSurface computation ******/
    if (place == FREESURFACE && k == 1) {
	/* compute fij */
	/* Idea :
	 * result = Normal (without dz(V0.Z) part +
	 *          dz(V0.Z) part
	 */
	ANL->fxx[i][j][k] = FKSIs4(mux, kapx, DS, Asx, Apx,
				   v0.x[i][j][k], v0.x[i + 1][j][k],
				   v0.x[i - 1][j][k], v0.x[i + 2][j][k],
				   v0.y[i][j - 1][k], v0.y[i][j][k],
				   v0.y[i][j - 2][k], v0.y[i][j + 1][k],
				   0., 0., 0., 0.) +
	    ((kapx + 4. / 3. * mux) * Apx - 2. * mux * Asx) * dzV0z;

	ANL->fyy[i][j][k] = FKSIs4(mux, kapx, DS, Asx, Apx,
				   v0.y[i][j - 1][k], v0.y[i][j][k],
				   v0.y[i][j - 2][k], v0.y[i][j + 1][k],
				   v0.x[i][j][k], v0.x[i + 1][j][k],
				   v0.x[i - 1][j][k], v0.x[i + 2][j][k],
				   0., 0., 0., 0.) +
	    ((kapx + 4. / 3. * mux) * Apx - 2. * mux * Asx) * dzV0z;

	ANL->fxy[i][j][k] = FKSIt4(muy, DS, Asy,
				   v0.y[i - 1][j][k], v0.y[i][j][k],
				   v0.y[i - 2][j][k], v0.y[i + 1][j][k],
				   v0.x[i][j][k], v0.x[i][j + 1][k],
				   v0.x[i][j - 1][k], v0.x[i][j + 2][k]);

	/* Compute ANL->ksi */
	ANL->ksixx[i][j][k] =
	    (1. / (tk / DT + 0.5)) * ((tk / DT - 0.5) *
				      ANL->ksixx[i][j][k] + 0.5 * (fxxB +
								   ANL->
								   fxx[i]
								   [j]
								   [k]));
	ANL->ksiyy[i][j][k] =
	    (1. / (tk / DT + 0.5)) * ((tk / DT - 0.5) *
				      ANL->ksiyy[i][j][k] + 0.5 * (fyyB +
								   ANL->
								   fyy[i]
								   [j]
								   [k]));
	ANL->ksixy[i][j][k] =
	    (1. / (tk / DT + 0.5)) * ((tk / DT - 0.5) *
				      ANL->ksixy[i][j][k] + 0.5 * (fxyB +
								   ANL->
								   fxy[i]
								   [j]
								   [k]));

    /***** freeAbs computation ******/
    } else if (place == FREEABS && k == 1) {
	/* compute fij */
	/* xy */
	/* (copy&paste) */
	phixdum = ABC->phivyx[npml];
	phiydum = ABC->phivxy[npml];
	ANL->fxy[i][j][k] = FKSIt4CPML(muy, DS, DT, Asy, vpy,
				       phixdum, phiydum,
				       ABC->dumpx[i], ABC->alphax[i],
				       ABC->kappax[i], ABC->dumpy2[j],
				       ABC->alphay2[j], ABC->kappay2[j],
				       v0.y[i - 1][j][k], v0.y[i][j][k],
				       v0.y[i - 2][j][k],
				       v0.y[i + 1][j][k], v0.x[i][j][k],
				       v0.x[i][j + 1][k],
				       v0.x[i][j - 1][k], v0.x[i][j + 2][k]
	    );

	/* xx, yy, */
	/* Idea :
	 * result = Normal (without dz(V0.Z) part +
	 *          dz(V0.Z) part
	 */

	phixdum = ABC->phivxx[npml];
	phiydum = ABC->phivyy[npml];
	phizdum = ABC->phivzz[npml];
	/* without dz(V0.Z) Part */
	ANL->fxx[i][j][k] = FKSIs4CPML(mux, kapx, DS, DT, Asx, Apx, vpx,
				       phixdum, phiydum, phizdum,
				       ABC->dumpx2[i], ABC->alphax2[i],
				       ABC->kappax2[i], ABC->dumpy[j],
				       ABC->alphay[j], ABC->kappay[j],
				       ABC->dumpz[k], ABC->alphaz[k],
				       ABC->kappaz[k], v0.x[i][j][k],
				       v0.x[i + 1][j][k],
				       v0.x[i - 1][j][k],
				       v0.x[i + 2][j][k],
				       v0.y[i][j - 1][k], v0.y[i][j][k],
				       v0.y[i][j - 2][k],
				       v0.y[i][j + 1][k], 0., 0., 0., 0.);

	ANL->fyy[i][j][k] = FKSIs4CPML(mux, kapx, DS, DT, Asx, Apx, vpy,
				       phiydum, phixdum, phizdum,
				       ABC->dumpy[j], ABC->alphay[j],
				       ABC->kappay[j], ABC->dumpx2[i],
				       ABC->alphax2[i], ABC->kappax2[i],
				       ABC->dumpz[k], ABC->alphaz[k],
				       ABC->kappaz[k], v0.y[i][j - 1][k],
				       v0.y[i][j][k], v0.y[i][j - 2][k],
				       v0.y[i][j + 1][k], v0.x[i][j][k],
				       v0.x[i + 1][j][k],
				       v0.x[i - 1][j][k],
				       v0.x[i + 2][j][k], 0., 0., 0., 0.);

	/* add dz(V0z) part */
	/* xx */
	b_z =
	    exp(-(vpx * ABC->dumpz[k] / ABC->kappaz[k] + ABC->alphaz[k]) *
		DT);
	a_z = 0.0;
	if (abs(vpx * ABC->dumpz[k]) > 0.000001) {
	    a_z =
		vpx * ABC->dumpz[k] * (b_z -
				       1.0) / (ABC->kappaz[k] * (vpx *
								 ABC->
								 dumpz[k] +
								 ABC->
								 kappaz[k]
								 *
								 ABC->
								 alphaz
								 [k]));
	}
	ANL->fxx[i][j][k] +=
	    b_z * phizdum + (1. / ABC->kappaz[k] + a_z) * dzV0z;

	/* yy */
	b_z =
	    exp(-(vpy * ABC->dumpz[k] / ABC->kappaz[k] + ABC->alphaz[k]) *
		DT);
	a_z = 0.0;
	if (abs(vpy * ABC->dumpz[k]) > 0.000001) {
	    a_z =
		vpy * ABC->dumpz[k] * (b_z -
				       1.0) / (ABC->kappaz[k] * (vpy *
								 ABC->
								 dumpz[k] +
								 ABC->
								 kappaz[k]
								 *
								 ABC->
								 alphaz
								 [k]));
	}
	ANL->fyy[i][j][k] +=
	    b_z * phizdum + (1. / ABC->kappaz[k] + a_z) * dzV0z;


	/* Compute ksi */
	ANL->ksixx[i][j][k] =
	    (1. / (tk / DT + 0.5)) * ((tk / DT - 0.5) *
				      ANL->ksixx[i][j][k] + 0.5 * (fxxB +
								   ANL->
								   fxx[i]
								   [j]
								   [k]));
	ANL->ksiyy[i][j][k] =
	    (1. / (tk / DT + 0.5)) * ((tk / DT - 0.5) *
				      ANL->ksiyy[i][j][k] + 0.5 * (fyyB +
								   ANL->
								   fyy[i]
								   [j]
								   [k]));
	ANL->ksixy[i][j][k] =
	    (1. / (tk / DT + 0.5)) * ((tk / DT - 0.5) *
				      ANL->ksixy[i][j][k] + 0.5 * (fxyB +
								   ANL->
								   fxy[i]
								   [j]
								   [k]));

    }				/* end FREEABS & k=1 */

    return (EXIT_SUCCESS);
}				/* end compute DayBradley */



/* OMP */
extern int numThreads_Level_1;

void ComputeIntermediates(
			     /* Outputs */
			     struct ABSORBING_BOUNDARY_CONDITION *ABC,
			     struct ANELASTICITY *ANL,
			     /* Parameters */
			     struct VELOCITY v0,	/* Velocity */
			     struct PARAMETERS PRM, struct MEDIUM MDM, int mpmx_begin, int mpmx_end,	/* local computed domain */
			     int mpmy_begin, int mpmy_end)
{
#ifdef LUCAS_INSTRUMENTATION
    SCOREP_USER_REGION_DEFINE(computeIntermediates)
	SCOREP_USER_REGION_BEGIN(computeIntermediates, "intermediates",
				 SCOREP_USER_REGION_TYPE_COMMON)
#endif
	/* approximations of a value in the corner of the cube */
    double rhox, rhoy, rhoz, rhoxyz;	/* rho at +0 and +ds/2 */
    double kapxyz,		/* rigidity and mu */
     kapxy, kapxz, kapx, kapy, kapz, muxy, muxz, mux, muy, muz, muxyz;
    /*  */
    double vpx, vpy, vpz, vpxyz;	/* approximations of vp on the corner of the cube */

    /*  */
    enum typePlace place;	/* What type of cell  */
    long int npml;		/* index in Absorbing Layer */
    int i, j, k;		/* local position of the cell */
    int imp, jmp;		/* global position of the cell (NB : kmp=k) */

    /* PML */
    double b1, b2;
    double xdum, ydum, zdum;

    /* CPML */
    double phixdum, phiydum, phizdum;	/* intermediates */

    /* DAY and BRADLEY */
    int ly0, ly2;		/* layer of the cell */
    double Ap, As, As2;		/*  */
    double Apx, Asx, Asy, Asz, Asxyz;	/* on the corners */
    double tk;


    /* KRISTEK and MOCZO */
    double el, fl;		/* intermediates */

    /* FREE SURFACE */
    double dzV0z;		/* derivative approximation of Vz in z direction  */
    double a, b;		/* intermediates for FREEABS */

    /* For mapping */
    double DS, DT;		/* PRM */

    double Tm, TM, W0;		/* ANELASTICITY, Day & Bradley method */
    int med1, med2;

    /* Verify Parameters */
    assert(interface == USUAL || interface == KMINTERFACE);

    assert(ABCmethod == CPML || ABCmethod == PML);
    assert(surface == ABSORBING || surface == FREE);
    assert(ANLmethod == ELASTIC || ANLmethod == ANOTHER ||
	   ANLmethod == DAYandBRADLEY || ANLmethod == KRISTEKandMOCZO);

    if (ANLmethod == DAYandBRADLEY) {
	assert((ABCmethod == CPML));
    } else if (ANLmethod == KRISTEKandMOCZO) {
	assert((ABCmethod == CPML));
    }

    /* mapping */
    DS = PRM.ds;
    DT = PRM.dt;

    if (ANLmethod == DAYandBRADLEY) {
	Tm = ANL->tm;
	TM = ANL->tM;
	W0 = ANL->w0;
    }


#ifdef OMP
#pragma omp parallel default( shared) \
private( ly0, ly2) \
private( kapxy, kapxz, kapxyz, kapx, kapy,kapz ) \
private( muxy, muxz, muxyz, mux, muy,muz ) \
private( rhoxyz, rhox,rhoy,rhoz) \
private( b1,b2 ) \
private( vpx,vpy,vpz,vpxyz) \
private( place ) \
private( npml )	\
private( i, j, k ) \
private( imp, jmp ) \
private( xdum, ydum, zdum ) \
private ( a,b, dzV0z ) \
private (Ap, As, As2, Apx, Asx,Asy,Asz,Asxyz,tk) \
private( phixdum, phiydum, phizdum ) num_threads(numThreads_Level_1)
    {
#pragma omp for schedule ( runtime) collapse(2)
#endif
	/* loop */
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
			npml = ABC->ipml[i][j][k];
		    }
		    /* medium */
		    if (model == GEOLOGICAL) {

			med1 = MDM.imed[i][j][k];
			med2 = MDM.imed[i + 1][j][k];

			mux =
			    averageInverseDouble2(MDM.mu0[med1],
						  MDM.mu0[med2], model);
			kapx =
			    averageInverseDouble2(MDM.kap0[med1],
						  MDM.kap0[med2], model);

		    } else if (model == LAYER) {
			/* Warning : k2ly0 & k2ly2
			   give 0 if k in FREEABS or if depth(k) > laydep[0] in general */
			ly0 = MDM.k2ly0[k];
			ly2 = MDM.k2ly2[k];

			mux = MDM.mu0[ly0];
			kapx = MDM.kap0[ly0];
		    }

      /*=====================================================*\
		  ELASTIC PART : PML and CPML

		  (nothing to do for regular domain, or only FreeSurface domain)
		  structure :
		  + PML/CPML
		  -- common initialisations
		  -- ABSORBING LAYER 
		  -- FREEABS

		  \*=====================================================*/

		    /* ABSORBING Layer, FreeAbs common initialisations */
		    /* ---------------------- */
		    if (place == ABSORBINGLAYER || place == FREEABS) {
			/* Initialize corner coefficients */
			if (model == GEOLOGICAL) {

			    med1 = MDM.imed[i][j][k];
			    /* x corner */
			    med1 = MDM.imed[i][j][k];
			    med2 = MDM.imed[i + 1][j][k];
			    rhox = 0.5 * (MDM.rho0[med1] + MDM.rho0[med2]);

			    /* y corner */
			    med2 = MDM.imed[i][j + 1][k];
			    rhoy = 0.5 * (MDM.rho0[med1] + MDM.rho0[med2]);
			    muy =
				averageInverseDouble2(MDM.mu0[med1],
						      MDM.mu0[med2],
						      model);
			    kapy =
				averageInverseDouble2(MDM.kap0[med1],
						      MDM.kap0[med2],
						      model);

			    /* z corner */
			    med1 = MDM.imed[i][j][k];
			    med2 = MDM.imed[i][j][k + 1];
			    rhoz = 0.5 * (MDM.rho0[med1] + MDM.rho0[med2]);
			    muz =
				averageInverseDouble2(MDM.mu0[med1],
						      MDM.mu0[med2],
						      model);
			    kapz =
				averageInverseDouble2(MDM.kap0[med1],
						      MDM.kap0[med2],
						      model);

			    /* xyz corner */
			    rhoxyz =
				CornerXYZ_Geol(MDM.rho0, i, j, k, MDM);
			    kapxyz =
				CornerXYZ_GeolInverse(MDM.kap0, i, j, k,
						      MDM);
			    muxyz =
				CornerXYZ_GeolInverse(MDM.mu0, i, j, k,
						      MDM);

			    vpx = RhoMuKap2Vp(rhox, mux, kapx);
			    vpy = RhoMuKap2Vp(rhoy, muy, kapy);

			    vpz = RhoMuKap2Vp(rhoz, muz, kapz);
			    vpxyz = RhoMuKap2Vp(rhoxyz, muxyz, kapxyz);
			} else if (model == LAYER) {

			    muy = MDM.mu0[ly0];
			    kapy = MDM.kap0[ly0];
			    rhox = MDM.rho0[ly0];
			    rhoy = MDM.rho0[ly0];

			    rhoz = MDM.rho2[ly2];
			    rhoxyz = MDM.rho2[ly2];
			    muz = MDM.mu2[ly2];
			    muxyz = MDM.mu2[ly2];
			    kapz = MDM.kap2[ly2];
			    kapxyz = MDM.kap2[ly2];

			    vpx = RhoMuKap2Vp(rhox, mux, kapx);
			    vpy = vpx;

			    vpz = RhoMuKap2Vp(rhoz, muz, kapz);
			    vpxyz = vpz;
			}
		    }		/* end initialize corners coeff in PML/CPML */

		    /* ABSORBING LAYER      */
		    /* -------------------- */
		    if (place == ABSORBINGLAYER) {
			/* Z Coefficients */


			/* Compute  PHIV */
			/* ------------- */
			if (ABCmethod == CPML) {
			    /* txx, tyy, tzz */
			    phixdum = ABC->phivxx[npml];
			    phiydum = ABC->phivyy[npml];
			    phizdum = ABC->phivzz[npml];

			    ABC->phivxx[npml] =
				CPML4(vpx, ABC->dumpx2[i], ABC->alphax2[i],
				      ABC->kappax2[i], phixdum, DS, DT,
				      v0.x[i][j][k], v0.x[i + 1][j][k],
				      v0.x[i - 1][j][k],
				      v0.x[i + 2][j][k]);
			    ABC->phivyy[npml] =
				CPML4(vpx, ABC->dumpy[j], ABC->alphay[j],
				      ABC->kappay[j], phiydum, DS, DT,
				      v0.y[i][j - 1][k], v0.y[i][j][k],
				      v0.y[i][j - 2][k],
				      v0.y[i][j + 1][k]);
			    ABC->phivzz[npml] =
				CPML4(vpx, ABC->dumpz[k], ABC->alphaz[k],
				      ABC->kappaz[k], phizdum, DS, DT,
				      v0.z[i][j][k - 1], v0.z[i][j][k],
				      v0.z[i][j][k - 2],
				      v0.z[i][j][k + 1]);
			    /* txy */
			    phixdum = ABC->phivyx[npml];
			    phiydum = ABC->phivxy[npml];

			    ABC->phivyx[npml] =
				CPML4(vpy, ABC->dumpx[i], ABC->alphax[i],
				      ABC->kappax[i], phixdum, DS, DT,
				      v0.y[i - 1][j][k], v0.y[i][j][k],
				      v0.y[i - 2][j][k],
				      v0.y[i + 1][j][k]);
			    ABC->phivxy[npml] =
				CPML4(vpy, ABC->dumpy2[j], ABC->alphay2[j],
				      ABC->kappay2[j], phiydum, DS, DT,
				      v0.x[i][j][k], v0.x[i][j + 1][k],
				      v0.x[i][j - 1][k],
				      v0.x[i][j + 2][k]);
			    /* txz */
			    phixdum = ABC->phivzx[npml];
			    phizdum = ABC->phivxz[npml];

			    ABC->phivzx[npml] =
				CPML4(vpz, ABC->dumpx[i], ABC->alphax[i],
				      ABC->kappax[i], phixdum, DS, DT,
				      v0.z[i - 1][j][k], v0.z[i][j][k],
				      v0.z[i - 2][j][k],
				      v0.z[i + 1][j][k]);
			    ABC->phivxz[npml] =
				CPML4(vpz, ABC->dumpz2[k], ABC->alphaz2[k],
				      ABC->kappaz2[k], phizdum, DS, DT,
				      v0.x[i][j][k], v0.x[i][j][k + 1],
				      v0.x[i][j][k - 1],
				      v0.x[i][j][k + 2]);
			    /* tyz */
			    phiydum = ABC->phivzy[npml];
			    phizdum = ABC->phivyz[npml];

			    ABC->phivzy[npml] =
				CPML4(vpxyz, ABC->dumpy2[j],
				      ABC->alphay2[j], ABC->kappay2[j],
				      phiydum, DS, DT, v0.z[i][j][k],
				      v0.z[i][j + 1][k], v0.z[i][j - 1][k],
				      v0.z[i][j + 2][k]);
			    ABC->phivyz[npml] =
				CPML4(vpxyz, ABC->dumpz2[k],
				      ABC->alphaz2[k], ABC->kappaz2[k],
				      phizdum, DS, DT, v0.y[i][j][k],
				      v0.y[i][j][k + 1], v0.y[i][j][k - 1],
				      v0.y[i][j][k + 2]);

			    /* Compute  Txxx, etc. */
			    /* ------------------- */
			} else if (ABCmethod == PML) {
			    b1 = kapx + 4. * mux / 3.;
			    b2 = kapx - 2. * mux / 3.;

			    /* t0.xx */
			    xdum = ABC->txxx[npml];
			    ydum = ABC->txxy[npml];
			    zdum = ABC->txxz[npml];
			    ABC->txxx[npml] =
				PMLdump4(b1, DT, DS, vpx * ABC->dumpx2[i],
					 xdum, v0.x[i][j][k],
					 v0.x[i + 1][j][k],
					 v0.x[i - 1][j][k],
					 v0.x[i + 2][j][k]);
			    ABC->txxy[npml] =
				PMLdump4(b2, DT, DS, vpx * ABC->dumpy[j],
					 ydum, v0.y[i][j - 1][k],
					 v0.y[i][j][k], v0.y[i][j - 2][k],
					 v0.y[i][j + 1][k]);
			    ABC->txxz[npml] =
				PMLdump4(b2, DT, DS, vpx * ABC->dumpz[k],
					 zdum, v0.z[i][j][k - 1],
					 v0.z[i][j][k], v0.z[i][j][k - 2],
					 v0.z[i][j][k + 1]);

			    /* t0.yy */
			    xdum = ABC->tyyx[npml];
			    ydum = ABC->tyyy[npml];
			    zdum = ABC->tyyz[npml];
			    ABC->tyyx[npml] =
				PMLdump4(b2, DT, DS, vpx * ABC->dumpx2[i],
					 xdum, v0.x[i][j][k],
					 v0.x[i + 1][j][k],
					 v0.x[i - 1][j][k],
					 v0.x[i + 2][j][k]);
			    ABC->tyyy[npml] =
				PMLdump4(b1, DT, DS, vpx * ABC->dumpy[j],
					 ydum, v0.y[i][j - 1][k],
					 v0.y[i][j][k], v0.y[i][j - 2][k],
					 v0.y[i][j + 1][k]);
			    ABC->tyyz[npml] =
				PMLdump4(b2, DT, DS, vpx * ABC->dumpz[k],
					 zdum, v0.z[i][j][k - 1],
					 v0.z[i][j][k], v0.z[i][j][k - 2],
					 v0.z[i][j][k + 1]);
			    /* t0.zz */
			    xdum = ABC->tzzx[npml];
			    ydum = ABC->tzzy[npml];
			    zdum = ABC->tzzz[npml];
			    ABC->tzzx[npml] =
				PMLdump4(b2, DT, DS, vpx * ABC->dumpx2[i],
					 xdum, v0.x[i][j][k],
					 v0.x[i + 1][j][k],
					 v0.x[i - 1][j][k],
					 v0.x[i + 2][j][k]);
			    ABC->tzzy[npml] =
				PMLdump4(b2, DT, DS, vpx * ABC->dumpy[j],
					 ydum, v0.y[i][j - 1][k],
					 v0.y[i][j][k], v0.y[i][j - 2][k],
					 v0.y[i][j + 1][k]);
			    ABC->tzzz[npml] =
				PMLdump4(b1, DT, DS, vpx * ABC->dumpz[k],
					 zdum, v0.z[i][j][k - 1],
					 v0.z[i][j][k], v0.z[i][j][k - 2],
					 v0.z[i][j][k + 1]);

			    /* t0.xy */
			    xdum = ABC->txyx[npml];
			    ydum = ABC->txyy[npml];
			    ABC->txyx[npml] =
				PMLdump4(muy, DT, DS, vpy * ABC->dumpx[i],
					 xdum, v0.y[i - 1][j][k],
					 v0.y[i][j][k], v0.y[i - 2][j][k],
					 v0.y[i + 1][j][k]);
			    ABC->txyy[npml] =
				PMLdump4(muy, DT, DS, vpy * ABC->dumpy2[j],
					 ydum, v0.x[i][j][k],
					 v0.x[i][j + 1][k],
					 v0.x[i][j - 1][k],
					 v0.x[i][j + 2][k]);

			    /* t0.xz */
			    xdum = ABC->txzx[npml];
			    zdum = ABC->txzz[npml];
			    ABC->txzx[npml] =
				PMLdump4(muz, DT, DS, vpz * ABC->dumpx[i],
					 xdum, v0.z[i - 1][j][k],
					 v0.z[i][j][k], v0.z[i - 2][j][k],
					 v0.z[i + 1][j][k]);
			    ABC->txzz[npml] =
				PMLdump4(muz, DT, DS, vpz * ABC->dumpz2[k],
					 zdum, v0.x[i][j][k],
					 v0.x[i][j][k + 1],
					 v0.x[i][j][k - 1],
					 v0.x[i][j][k + 2]);
			    /* t0.yz */
			    ydum = ABC->tyzy[npml];
			    zdum = ABC->tyzz[npml];
			    ABC->tyzy[npml] =
				PMLdump4(muxyz, DT, DS,
					 vpxyz * ABC->dumpy2[j], ydum,
					 v0.z[i][j][k], v0.z[i][j + 1][k],
					 v0.z[i][j - 1][k],
					 v0.z[i][j + 2][k]);
			    ABC->tyzz[npml] =
				PMLdump4(muxyz, DT, DS,
					 vpxyz * ABC->dumpz2[k], zdum,
					 v0.y[i][j][k], v0.y[i][j][k + 1],
					 v0.y[i][j][k - 1],
					 v0.z[i][j][k + 2]);
			}	/* end if ABCmethod */

		    }		/* End compute Absorbing Layers */

		    /* FREEABS      */
		    /* ------------ */
		    /* We only need to compute t0xx(k=1),t0yy(k=1),t0xy(k=1) 
		     * So each coefficient will be computed only for those part 
		     *
		     * Expressions are simply obtain from approximation of dz(v0.z) in an elastic medium :
		     * DT(t0.zz)|z=0 = 0 = lam *(dx(v0.x) + dy(v0.y) + dz(v0.z)) + 2*mu * dz(v0.z);
		     *
		     */
		    if (place == FREEABS) {
			/* approximate dz(v0.z) (4th order) */
			dzV0z =
			    (kapx - 2. / 3. * mux) / (kapx +
						      4. / 3. * mux) *
			    (Diff4
			     (DS, v0.x[i][j][k], v0.x[i + 1][j][k],
			      v0.x[i - 1][j][k],
			      v0.x[i + 2][j][k]) + Diff4(DS,
							 v0.y[i][j - 1][k],
							 v0.y[i][j][k],
							 v0.y[i][j - 2][k],
							 v0.y[i][j + 1][k])
			    );

			if (ABCmethod == CPML) {
			    /* txx, tyy */
			    phixdum = ABC->phivxx[npml];
			    phiydum = ABC->phivyy[npml];
			    phizdum = ABC->phivzz[npml];
			    /* (copy&paste) */
			    ABC->phivxx[npml] =
				CPML4(vpx, ABC->dumpx2[i], ABC->alphax2[i],
				      ABC->kappax2[i], phixdum, DS, DT,
				      v0.x[i][j][k], v0.x[i + 1][j][k],
				      v0.x[i - 1][j][k],
				      v0.x[i + 2][j][k]);
			    ABC->phivyy[npml] =
				CPML4(vpx, ABC->dumpy[j], ABC->alphay[j],
				      ABC->kappay[j], phiydum, DS, DT,
				      v0.y[i][j - 1][k], v0.y[i][j][k],
				      v0.y[i][j - 2][k],
				      v0.y[i][j + 1][k]);
			    /* special */
			    b = exp(-
				    (vpx * ABC->dumpz[k] / ABC->kappaz[k] +
				     ABC->alphaz[k]) * DT);
			    a = 0.0;
			    if (abs(vpx * ABC->dumpz[k]) > 0.000001) {
				a = vpx * ABC->dumpz[k] * (b -
							   1.0) /
				    (ABC->kappaz[k] *
				     (vpx * ABC->dumpz[k] +
				      ABC->kappaz[k] * ABC->alphaz[k]));
			    }
			    ABC->phivzz[npml] = b * phizdum + a * (dzV0z);

			    /* txy ( copy&paste) */
			    phixdum = ABC->phivyx[npml];
			    phiydum = ABC->phivxy[npml];

			    ABC->phivyx[npml] =
				CPML4(vpy, ABC->dumpx[i], ABC->alphax[i],
				      ABC->kappax[i], phixdum, DS, DT,
				      v0.y[i - 1][j][k], v0.y[i][j][k],
				      v0.y[i - 2][j][k],
				      v0.y[i + 1][j][k]);
			    ABC->phivxy[npml] =
				CPML4(vpy, ABC->dumpy2[j], ABC->alphay2[j],
				      ABC->kappay2[j], phiydum, DS, DT,
				      v0.x[i][j][k], v0.x[i][j + 1][k],
				      v0.x[i][j - 1][k],
				      v0.x[i][j + 2][k]);
			} else if (ABCmethod == PML) {
			    /* for txxz, and tyyz
			       /* t0.xx */
			    xdum = ABC->txxx[npml];
			    ydum = ABC->txxy[npml];
			    zdum = ABC->txxz[npml];
			    ABC->txxx[npml] =
				PMLdump4(b1, DT, DS, vpx * ABC->dumpx2[i],
					 xdum, v0.x[i][j][k],
					 v0.x[i + 1][j][k],
					 v0.x[i - 1][j][k],
					 v0.x[i + 2][j][k]);
			    ABC->txxy[npml] =
				PMLdump4(b2, DT, DS, vpx * ABC->dumpy[j],
					 ydum, v0.y[i][j - 1][k],
					 v0.y[i][j][k], v0.y[i][j - 2][k],
					 v0.y[i][j + 1][k]);
			    ABC->txxz[npml] =
				(2. - DT * vpx * ABC->dumpz[k]) * zdum +
				b2 * dzV0z / (2. +
					      DT * vpx * ABC->dumpz[k]);

			    /* t0.yy */
			    xdum = ABC->tyyx[npml];
			    ydum = ABC->tyyy[npml];
			    zdum = ABC->tyyz[npml];
			    ABC->tyyx[npml] =
				PMLdump4(b2, DT, DS, vpx * ABC->dumpx2[i],
					 xdum, v0.x[i][j][k],
					 v0.x[i + 1][j][k],
					 v0.x[i - 1][j][k],
					 v0.x[i + 2][j][k]);
			    ABC->tyyy[npml] =
				PMLdump4(b1, DT, DS, vpx * ABC->dumpy[j],
					 ydum, v0.y[i][j - 1][k],
					 v0.y[i][j][k], v0.y[i][j - 2][k],
					 v0.y[i][j + 1][k]);
			    ABC->tyyz[npml] =
				(2. - DT * vpx * ABC->dumpz[k]) * zdum +
				b2 * dzV0z / (2. +
					      DT * vpx * ABC->dumpz[k]);

			    /* t0.xy */
			    xdum = ABC->txyx[npml];
			    ydum = ABC->txyy[npml];
			    ABC->txyx[npml] =
				PMLdump4(muy, DT, DS, vpy * ABC->dumpx[i],
					 xdum, v0.y[i - 1][j][k],
					 v0.y[i][j][k], v0.y[i - 2][j][k],
					 v0.y[i + 1][j][k]);
			    ABC->txyy[npml] =
				PMLdump4(muy, DT, DS, vpy * ABC->dumpy2[j],
					 ydum, v0.x[i][j][k],
					 v0.x[i][j + 1][k],
					 v0.x[i][j - 1][k],
					 v0.x[i][j + 2][k]);

			}
		    }		/* end FREEABS */
		    /* end ELASTIC PART */

		    /*=====================================================*\
			Compute ANELASTIC PART : DAY & BRADLEY
	  
			Goal : increment KSI and fxx,fyy,...
			structure :
			+ DAY & BRADLEY
			-- common initialisations
			-- Regular
			-- CPML
			-- FREE SURFACE
			-- FREEABS
			\*=====================================================*/

		    /* Warning :
		     * This part must be after phiv computation 
		     */

		    if (ANLmethod == DAYandBRADLEY) {

			if (surface == FREE && k == PRM.zMax0) {	/* jump not computed domain */
			    continue;
			}
			/*  initialisations */
			/* ---------------------- */
			/* compute coefficients */
			tk = exp(log(Tm) +
				 (2. *
				  (1. + (imp % 2) + 2. * (jmp % 2) +
				   4. * (k % 2)) - 1.) * (log(TM) -
							  log(Tm)) / 16.);

			/* Initialize corner coefficients */
			if (model == GEOLOGICAL) {
			    int med1, med2, med3, med4;
			    double Qpx, Qsx, Qsy, Qsz, Qsxyz;
			    med1 = MDM.imed[i][j][k];
			    /* x corner : fxx, fyy,fzz, fxy */
			    med1 = MDM.imed[i][j][k];
			    med2 = MDM.imed[i + 1][j][k];
			    rhox = 0.5 * (MDM.rho0[med1] + MDM.rho0[med2]);
			    Qpx =
				averageInverseDouble2(ANL->Qp0[med1],
						      ANL->Qp0[med2],
						      model);
			    Qsx =
				averageInverseDouble2(ANL->Qs0[med1],
						      ANL->Qs0[med2],
						      model);
			    Asx = Q2A(Qsx, TM, Tm, W0, PRM.pi);
			    Apx = Q2A(Qsx, TM, Tm, W0, PRM.pi);

			    /* y corner : fxy */
			    med2 = MDM.imed[i][j + 1][k];
			    rhoy = 0.5 * (MDM.rho0[med1] + MDM.rho0[med2]);
			    muy =
				averageInverseDouble2(MDM.mu0[med1],
						      MDM.mu0[med2],
						      model);
			    kapy =
				averageInverseDouble2(MDM.kap0[med1],
						      MDM.kap0[med2],
						      model);
			    Qsy =
				averageInverseDouble2(ANL->Qs0[med1],
						      ANL->Qs0[med2],
						      model);
			    Asy = Q2A(Qsy, TM, Tm, W0, PRM.pi);

			    /* z corner : xz */
			    med1 = MDM.imed[i][j][k];
			    med2 = MDM.imed[i][j][k + 1];
			    muz =
				averageInverseDouble2(MDM.mu0[med1],
						      MDM.mu0[med2],
						      model);
			    kapz =
				averageInverseDouble2(MDM.kap0[med1],
						      MDM.kap0[med2],
						      model);
			    Qsz =
				averageInverseDouble2(ANL->Qs0[med1],
						      ANL->Qs0[med2],
						      model);
			    Asz = Q2A(Qsz, TM, Tm, W0, PRM.pi);

			    /* xyz corner : fyz */
			    rhoxyz =
				CornerXYZ_Geol(MDM.rho0, i, j, k, MDM);
			    kapxyz =
				CornerXYZ_GeolInverse(MDM.kap0, i, j, k,
						      MDM);
			    muxyz =
				CornerXYZ_GeolInverse(MDM.mu0, i, j, k,
						      MDM);
			    Qsxyz =
				averageInverseDouble2(ANL->Qs0[med1],
						      ANL->Qs0[med2],
						      model);
			    Asxyz = Q2A(Qsxyz, TM, Tm, W0, PRM.pi);

			} else if (model == LAYER) {

			    muy = MDM.mu0[ly0];
			    kapy = MDM.kap0[ly0];
			    rhox = MDM.rho0[ly0];
			    rhoy = MDM.rho0[ly0];

			    As = (2. / (PRM.pi * ANL->Qs0[ly0])) * log(TM / Tm) / (1. - (2. / (PRM.pi * ANL->Qs0[ly0])) * log(W0 * Tm));	/* fxx, fyy,fzz, fxy */
			    Ap = (2. / (PRM.pi * ANL->Qp0[ly0])) * log(TM /
								       Tm)
				/ (1. -
				   (2. / (PRM.pi * ANL->Qp0[ly0])) *
				   log(W0 * Tm));

			    rhoz = MDM.rho2[ly2];
			    rhoxyz = MDM.rho2[ly2];
			    muz = MDM.mu2[ly2];
			    muxyz = MDM.mu2[ly2];
			    kapz = MDM.kap2[ly2];
			    kapxyz = MDM.kap2[ly2];

			    As2 = (2. / (PRM.pi * ANL->Qs2[ly2])) * log(TM / Tm) / (1. - (2. / (PRM.pi * ANL->Qs2[ly2])) * log(W0 * Tm));	/* fxz fyz */

			    Apx = Ap;
			    Asx = As;
			    Asy = As;
			    Asz = As2;
			    Asxyz = As2;
			}	/* end if model */
			vpx = RhoMuKap2Vp(rhox, mux, kapx);
			vpy = RhoMuKap2Vp(rhoy, muy, kapy);

			vpz = RhoMuKap2Vp(rhoz, muz, kapz);
			vpxyz = RhoMuKap2Vp(rhoxyz, muxyz, kapxyz);

			/* end common initializations */

			LocalComputeDayBradley(ANL,
					       i, j, k,
					       vpx, vpy, vpz, vpxyz,
					       mux, muy, muz, muxyz,
					       kapx,
					       Apx,
					       Asx, Asy, Asz, Asxyz,
					       tk,
					       DS, DT,
					       v0, ABC, npml, place);


		    }		/* end DAY and BRADLEY */

		   /*=====================================================*\
		  Compute ANELASTIC PART : KRISTEK & MOCZO
	      
		  Goal : increment KSIL
		  structure :
		  + KRISTEK and MOCZO
		  -- common initialisations
		  -- Regular & CPML common part 
		  -- CPML special part
		  -- FREE SURFACE & FREEABS common part
		  -- FREEABS special part
	      
		  \*=====================================================*/

		    /* Description :
		     * anelasticities only influence t0 computation (except another attenuation method).
		     * Meanwhile, on Free Surface, only t0xx(k=1),t0yy(k=1),t0xy(k=1) are really computed
		     * So every coefficient will be computed only for those part 
		     *
		     * Free Surface Expressions are simply obtain from approximation of dz(v0.z) in an elastic medium :
		     * DT(t0.zz)|z=0 = 0 = lam *(dx(v0.x) + dy(v0.y) + dz(v0.z)) + 2*mu * dz(v0.z);
		     *
		     */
		    if (ANLmethod == KRISTEKandMOCZO) {

			/* COMMON INITIALISATIONS      */
			/* -------------------------- */
			int index = IndexL(i, j, k);
			fl = (ANL->wl[index] * DT) / (2. +
						      ANL->wl[index] * DT);
			el = (2. - ANL->wl[index] * DT) / (2. +
							   ANL->wl[index] *
							   DT);

			/* Regular & CPML common part */
			/* -------------------------- */
			if (place == REGULAR || place == ABSORBINGLAYER) {
			    ANL->ksilxx[i][j][k] =
				el * ANL->ksilxx[i][j][k]
				+ 2. * fl * Diff4(DS, v0.x[i][j][k],
						  v0.x[i + 1][j][k],
						  v0.x[i - 1][j][k],
						  v0.x[i + 2][j][k]);

			    ANL->ksilyy[i][j][k] =
				el * ANL->ksilyy[i][j][k]
				+ 2. * fl * Diff4(DS, v0.y[i][j - 1][k],
						  v0.y[i][j][k],
						  v0.y[i][j - 2][k],
						  v0.y[i][j + 1][k]);

			    ANL->ksilzz[i][j][k] =
				el * ANL->ksilzz[i][j][k]
				+ 2. * fl * Diff4(DS, v0.z[i][j][k - 1],
						  v0.z[i][j][k],
						  v0.z[i][j][k - 2],
						  v0.z[i][j][k + 1]);

			    ANL->ksilxy[i][j][k] =
				el * ANL->ksilxy[i][j][k]
				+
				fl *
				(Diff4
				 (DS, v0.x[i][j][k], v0.x[i][j + 1][k],
				  v0.x[i][j - 1][k],
				  v0.x[i][j + 2][k]) + Diff4(DS,
							     v0.y[i -
								  1][j][k],
							     v0.y[i][j][k],
							     v0.y[i -
								  2][j][k],
							     v0.y[i +
								  1][j]
							     [k]));

			    ANL->ksilyz[i][j][k] =
				el * ANL->ksilyz[i][j][k]
				+
				fl *
				(Diff4
				 (DS, v0.y[i][j][k], v0.y[i][j][k + 1],
				  v0.y[i][j][k - 1],
				  v0.y[i][j][k + 2]) + Diff4(DS,
							     v0.z[i][j][k],
							     v0.z[i][j +
								     1][k],
							     v0.z[i][j -
								     1][k],
							     v0.z[i][j +
								     2]
							     [k]));

			    ANL->ksilxz[i][j][k] =
				el * ANL->ksilxz[i][j][k]
				+
				fl *
				(Diff4
				 (DS, v0.x[i][j][k], v0.x[i][j][k + 1],
				  v0.x[i][j][k - 1],
				  v0.x[i][j][k + 2]) + Diff4(DS,
							     v0.z[i -
								  1][j][k],
							     v0.z[i][j][k],
							     v0.z[i -
								  2][j][k],
							     v0.z[i +
								  1][j +
								     2]
							     [k]));
			}

			/* CPML special part */
			if ((place == ABSORBINGLAYER)
			    && (ABCmethod == CPML)) {
			    ANL->ksilxx[i][j][k] +=
				2. * fl * ABC->phivxx[npml];
			    ANL->ksilyy[i][j][k] +=
				2. * fl * ABC->phivyy[npml];
			    ANL->ksilzz[i][j][k] +=
				2. * fl * ABC->phivzz[npml];
			    ANL->ksilxy[i][j][k] +=
				fl * (ABC->phivyx[npml] +
				      ABC->phivxy[npml]);
			    ANL->ksilyz[i][j][k] +=
				fl * (ABC->phivyz[npml] +
				      ABC->phivzy[npml]);
			    ANL->ksilxz[i][j][k] +=
				fl * (ABC->phivxz[npml] +
				      ABC->phivzx[npml]);
			}

			/* FREE SURFACE & FREEABS common part */
			/* ---------------------------------- */
			/* we only need to compute :
			 * ksilxx, ksilyy, ksilzz,
			 * ksilxy for k=1
			 * 
			 * what differs from classic formulation 
			 * is only dzV0z approximation
			 */
			if (place == FREESURFACE || place == FREEABS) {
			    if (k == 1) {
				/* approximate dz(v0.z) (4th order) */
				dzV0z =
				    (kapx - 2. / 3. * mux) / (kapx +
							      4. / 3. *
							      mux) *
				    (Diff4
				     (DS, v0.x[i][j][k], v0.x[i + 1][j][k],
				      v0.x[i - 1][j][k],
				      v0.x[i + 2][j][k]) + Diff4(DS,
								 v0.
								 y[i][j -
								      1]
								 [k],
								 v0.
								 y[i][j]
								 [k],
								 v0.
								 y[i][j -
								      2]
								 [k],
								 v0.
								 y[i][j +
								      1]
								 [k])
				    );

				/* ksil computation */
				/* we just change dzV0z approximation */
				ANL->ksilxx[i][j][k] =
				    el * ANL->ksilxx[i][j][k]
				    + 2. * fl * Diff4(DS, v0.x[i][j][k],
						      v0.x[i + 1][j][k],
						      v0.x[i - 1][j][k],
						      v0.x[i + 2][j][k]);

				ANL->ksilyy[i][j][k] =
				    el * ANL->ksilyy[i][j][k]
				    + 2. * fl * Diff4(DS,
						      v0.y[i][j - 1][k],
						      v0.y[i][j][k],
						      v0.y[i][j - 2][k],
						      v0.y[i][j + 1][k]);

				ANL->ksilzz[i][j][k] =
				    el * ANL->ksilzz[i][j][k]
				    + 2. * fl * dzV0z;

				ANL->ksilxy[i][j][k] =
				    el * ANL->ksilxy[i][j][k]
				    +
				    fl *
				    (Diff4
				     (DS, v0.x[i][j][k], v0.x[i][j + 1][k],
				      v0.x[i][j - 1][k],
				      v0.x[i][j + 2][k]) + Diff4(DS,
								 v0.y[i -
								      1][j]
								 [k],
								 v0.
								 y[i][j]
								 [k],
								 v0.y[i -
								      2][j]
								 [k],
								 v0.y[i +
								      1][j]
								 [k]));
			    }	/* end if k = 1 */
			}

			/* FREEABS special part */
			/* -------------------- */
			if (place == FREEABS) {
			    ANL->ksilxx[i][j][k] +=
				2. * fl * ABC->phivxx[npml];
			    ANL->ksilyy[i][j][k] +=
				2. * fl * ABC->phivyy[npml];
			    ANL->ksilxy[i][j][k] +=
				fl * (ABC->phivyx[npml] +
				      ABC->phivxy[npml]);
			}
		    }		/* end Kristek and Moczo */



		}		/* end for k */
	    }			/* end for j */
	}			/* end for i */

#ifdef OMP
    }
#endif
#ifdef LUCAS_INSTRUMENTATION
    SCOREP_USER_REGION_END(computeIntermediates)
#endif
}				/* endof Compute2ermediates */
