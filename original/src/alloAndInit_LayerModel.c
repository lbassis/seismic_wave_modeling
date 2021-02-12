#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "nrutil.h"
#include "struct.h"
#include "inlineFunctions.h"
#include "options.h"
#include "alloAndInit.h"

#ifndef VERBOSE
static const int VERBOSE = 0;
#endif
#define POSITION

/* Goal : intialise layers 
   # Algorithm
   compute k2ly0 and k2ly2

   if ( interface == USUAL ) copy infos 
   else if ( interface == KMINTERFACE )
      create new medium & new ANL
   
      for every layers
       Getlayer & Get kMin kMax info for the layer
      
       compute his lower interface , 
       append himself and the lower interface to the new Medium & ANL
      done
    
      remplace old medium with new one
   fi
   
   

*/

/************************ my structs  */
struct typelayer {
    double laydep;
    double rho;
    double mu;
    double kap;
    double q;
    double amp;
    double Qp;
    double Qs;
};

/************************* local functions */
static int Initk2ly(struct MEDIUM *MDM, struct PARAMETERS PRM)
{
  /*** Infos ***/
    /* 
     * a 0 cell : -dz/2 +dz/2
     * a 2 cell : 0     +dz
     * z = (k-1)*dz
     * 
     * when there is multiple layer within a cell,
     * give the highest index of those layers.
     */
    int k;
    int ly, ly0, ly2;
    double z0;


    /* map */
    const int ZMIN = PRM.zMin;
    const int ZMAX = PRM.zMax;
    const int ZMAX0 = PRM.zMax0;
    const int DELTA = PRM.delta;
    const int NLAYER0 = MDM->nLayer;
    const int NLAYER2 = MDM->nLayer2;

    double *LAYDEP0, *LAYDEP2;
    LAYDEP0 = MDM->laydep;
    LAYDEP2 = MDM->laydep2;

    for (k = ZMAX0; k >= ZMIN - DELTA; k--) {
	z0 = (k - 1) * PRM.ds;
#ifndef POSITION
	/* here a point represent the volume around himself */
	double z_2;
	z_2 = (k - 1.5) * PRM.ds;
	/* k2ly0 */
	ly0 = 0;
	for (ly = 0; ly < NLAYER0; ly++) {
	    if (z_2 <= LAYDEP0[ly]) {
		ly0 = ly;
	    }
	}
	MDM->k2ly0[k] = ly0;

	/* k2ly2 */
	ly2 = 0;
	for (ly = 0; ly < NLAYER2; ly++) {
	    if (z0 <= LAYDEP2[ly]) {
		ly2 = ly;
	    }
	}
	MDM->k2ly2[k] = ly2;
#else
	/* here a point represent a position */
	/* NB : for KMINTERFACE we really need <= for the algorithm
	 */
	/* k2ly0 */
	ly0 = 0;
	for (ly = 0; ly < NLAYER0; ly++) {
	    if (interface == KMINTERFACE && z0 <= LAYDEP0[ly]) {
		ly0 = ly;
	    }
	    if (interface == LAYER && z0 < LAYDEP0[ly]) {
		ly0 = ly;
	    }
	}
	MDM->k2ly0[k] = ly0;

	/* k2ly2 */
	double z2;
	z2 = (k - 0.5) * PRM.ds;
	ly2 = 0;
	for (ly = 0; ly < NLAYER2; ly++) {
	    if (interface == KMINTERFACE && z2 <= LAYDEP2[ly]) {
		ly2 = ly;
	    }
	    if (interface == LAYER && z2 < LAYDEP2[ly]) {
		ly2 = ly;
	    }
	}
	MDM->k2ly2[k] = ly2;
#endif

	if (VERBOSE > 4 && PRM.me == 0) {
	    fprintf(stderr, "#### k=%i k2ly0=%i k2ly2=%i \n",
		    k, MDM->k2ly0[k], MDM->k2ly2[k]);
	}

    }
    return EXIT_SUCCESS;
}


static int PrintLayer(struct typelayer myLayer)
{
    fprintf(stderr,
	    "laydep=%g rho=%g mu=%g kap=%g Qp=%g Qs=%g amp=%g q=%g \n",
	    myLayer.laydep, myLayer.rho, myLayer.mu, myLayer.kap,
	    myLayer.Qp, myLayer.Qs, myLayer.amp, myLayer.q);
    return EXIT_SUCCESS;
}

static int GetkBkE(int *kB, int *kE, int myly, struct MEDIUM *MDM,
		   struct PARAMETERS PRM)
{
    /* kB : k Begin */
    if (myly == 0) {
	*kB = PRM.zMax0;
    } else {
	*kB = floor(MDM->laydep[myly] / PRM.ds) + 1;	/* k at the interface */
    }
    if (myly + 1 > MDM->nLayer - 1) {
	*kE = PRM.zMin - PRM.delta;
    } else {
	*kE = ((int) floor(MDM->laydep[myly + 1] / PRM.ds)) + 1;	/* k at the interface */
    }
    return EXIT_SUCCESS;
}


static int ExtractAlayer(struct typelayer *layer, int ly, int pos,
			 struct MEDIUM *MDM, struct ANELASTICITY *ANL)
{
    if (pos == 0) {
	layer->laydep = MDM->laydep[ly];
	layer->rho = MDM->rho0[ly];
	layer->mu = MDM->mu0[ly];
	layer->kap = MDM->kap0[ly];

	if (ANLmethod == ANOTHER) {
	    layer->q = ANL->q0[ly];
	    layer->amp = ANL->amp[ly];
	} else if (ANLmethod == KRISTEKandMOCZO
		   || ANLmethod == DAYandBRADLEY) {
	    layer->Qs = ANL->Qs0[ly];
	    layer->Qp = ANL->Qp0[ly];
	}
    } else if (pos == 2) {
	layer->laydep = MDM->laydep2[ly];
	layer->rho = MDM->rho2[ly];
	layer->mu = MDM->mu2[ly];
	layer->kap = MDM->kap2[ly];

	if (ANLmethod == ANOTHER) {
	    layer->q = ANL->q2[ly];
	    layer->amp = ANL->amp2[ly];
	} else if (ANLmethod == KRISTEKandMOCZO
		   || ANLmethod == DAYandBRADLEY) {
	    layer->Qs = ANL->Qs2[ly];
	    layer->Qp = ANL->Qp2[ly];
	}

    } else {
	return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

static int AppendLayer2MDM(struct MEDIUM *MDM, struct ANELASTICITY *ANL,
			   int pos, struct typelayer layer)
{
    int NLAYER = 0;

    if (pos != 0 && pos != 2) {
	return EXIT_FAILURE;
    }

    if (pos == 0) {
	MDM->nLayer += 1;
	NLAYER = MDM->nLayer;

	MDM->laydep = mydvectorRealloc(MDM->laydep, 0, NLAYER - 1);
	MDM->rho0 = mydvectorRealloc(MDM->rho0, 0, NLAYER - 1);
	MDM->mu0 = mydvectorRealloc(MDM->mu0, 0, NLAYER - 1);
	MDM->kap0 = mydvectorRealloc(MDM->kap0, 0, NLAYER - 1);

	MDM->laydep[NLAYER - 1] = layer.laydep;
	MDM->rho0[NLAYER - 1] = layer.rho;
	MDM->mu0[NLAYER - 1] = layer.mu;
	MDM->kap0[NLAYER - 1] = layer.kap;
	if (ANLmethod == ANOTHER) {
	    ANL->q0 = mydvectorRealloc(ANL->q0, 0, NLAYER - 1);
	    ANL->amp = mydvectorRealloc(ANL->amp, 0, NLAYER - 1);

	    ANL->q0[NLAYER - 1] = layer.q;
	    ANL->amp[NLAYER - 1] = layer.amp;
	} else if (ANLmethod == KRISTEKandMOCZO
		   || ANLmethod == DAYandBRADLEY) {
	    ANL->Qp0 = mydvectorRealloc(ANL->Qp0, 0, NLAYER - 1);
	    ANL->Qs0 = mydvectorRealloc(ANL->Qs0, 0, NLAYER - 1);

	    ANL->Qp0[NLAYER - 1] = layer.Qp;
	    ANL->Qs0[NLAYER - 1] = layer.Qs;
	}
    } else if (pos == 2) {
	MDM->nLayer2 += 1;
	NLAYER = MDM->nLayer2;

	MDM->laydep2 = mydvectorRealloc(MDM->laydep2, 0, NLAYER - 1);
	MDM->rho2 = mydvectorRealloc(MDM->rho2, 0, NLAYER - 1);
	MDM->mu2 = mydvectorRealloc(MDM->mu2, 0, NLAYER - 1);
	MDM->kap2 = mydvectorRealloc(MDM->kap2, 0, NLAYER - 1);

	MDM->laydep2[NLAYER - 1] = layer.laydep;
	MDM->rho2[NLAYER - 1] = layer.rho;
	MDM->mu2[NLAYER - 1] = layer.mu;
	MDM->kap2[NLAYER - 1] = layer.kap;
	if (ANLmethod == ANOTHER) {
	    ANL->q2 = mydvectorRealloc(ANL->q2, 0, NLAYER - 1);
	    ANL->amp2 = mydvectorRealloc(ANL->amp2, 0, NLAYER - 1);

	    ANL->q2[NLAYER - 1] = layer.q;
	    ANL->amp2[NLAYER - 1] = layer.amp;
	} else if (ANLmethod == KRISTEKandMOCZO
		   || ANLmethod == DAYandBRADLEY) {
	    ANL->Qp2 = mydvectorRealloc(ANL->Qp2, 0, NLAYER - 1);
	    ANL->Qs2 = mydvectorRealloc(ANL->Qs2, 0, NLAYER - 1);

	    ANL->Qp2[NLAYER - 1] = layer.Qp;
	    ANL->Qs2[NLAYER - 1] = layer.Qs;
	}
    }				/* end if pos */

    return EXIT_SUCCESS;
}

static int AddPart2Layer(struct typelayer *lOut, struct typelayer lIn,
			 double w)
{
    lOut->rho += w * lIn.rho;
    lOut->mu += w * 1. / lIn.mu;
    lOut->kap += w * 1. / lIn.kap;
    if (ANLmethod == ANOTHER) {
	lOut->q += w * 1. / lIn.q;
    } else if (ANLmethod == KRISTEKandMOCZO || ANLmethod == DAYandBRADLEY) {
	lOut->Qp += w * 1. / lIn.Qp;
	lOut->Qs += w * 1. / lIn.Qs;
    }
    return EXIT_SUCCESS;
}

static int EmptyLayer(struct typelayer *lOut)
{
    lOut->laydep = 0.;
    lOut->rho = 0.;
    lOut->mu = 0.;
    lOut->kap = 0.;
    if (ANLmethod == ANOTHER) {
	lOut->q = 0.;
    } else if (ANLmethod == KRISTEKandMOCZO || ANLmethod == DAYandBRADLEY) {
	lOut->Qp = 0.;
	lOut->Qs = 0.;
    }
}

/************************ real outside function  */
int InitLayerModel(struct MEDIUM *MDM,
		   struct ANELASTICITY *ANL, struct PARAMETERS PRM)
{
    int k;
    int ly, ly0, ly2;
    const int ZMIN = PRM.zMin;
    const int ZMAX = PRM.zMax;
    const int ZMAX0 = PRM.zMax0;
    const int DELTA = PRM.delta;

    struct MEDIUM newMDM = { 0 };
    struct ANELASTICITY newANL = { 0 };

    if (model != LAYER) {
	return EXIT_FAILURE;
    }
    if (interface != USUAL && interface != KMINTERFACE) {
	return EXIT_FAILURE;
    }

    MDM->nLayer2 = MDM->nLayer;
    MDM->laydep2 = MDM->laydep;

  /*** compute k2ly  ***/
    Initk2ly(MDM, PRM);

  /*** compute amp ***/
    if (ANLmethod == ANOTHER) {
	for (ly = 0; ly < MDM->nLayer; ly++) {
	    if (ANL->q0[ly] <= 0) {
		ANL->amp[ly] = 1.0;
	    } else {
		ANL->amp[ly] = exp(-PRM.pi * f0 * PRM.dt / ANL->q0[ly]);
	    }
	}
    }

    if (interface == USUAL) {
	/* duplicate datas to avoid complex deallocation,
	   memory use is negligeable since NLAYER is very small */
	MDM->laydep2 = NULL;
	MDM->laydep2 = mydvector0(0, MDM->nLayer);

	MDM->rho2 = mydvector0(0, MDM->nLayer);
	MDM->mu2 = mydvector0(0, MDM->nLayer);
	MDM->kap2 = mydvector0(0, MDM->nLayer);
	if (ANLmethod == ANOTHER) {
	    ANL->q2 = mydvector0(0, MDM->nLayer);
	    ANL->amp2 = mydvector0(0, MDM->nLayer);
	} else if (ANLmethod == KRISTEKandMOCZO
		   || ANLmethod == DAYandBRADLEY) {
	    ANL->Qp2 = mydvector0(0, MDM->nLayer);
	    ANL->Qs2 = mydvector0(0, MDM->nLayer);
	}

	for (ly = 0; ly < MDM->nLayer; ly++) {
	    MDM->laydep2[ly] = MDM->laydep[ly];

	    MDM->rho2[ly] = MDM->rho0[ly];
	    MDM->mu2[ly] = MDM->mu0[ly];
	    MDM->kap2[ly] = MDM->kap0[ly];
	    if (ANLmethod == ANOTHER) {
		ANL->q2[ly] = ANL->q0[ly];
		ANL->amp2[ly] = ANL->amp[ly];
	    } else if (ANLmethod == KRISTEKandMOCZO
		       || ANLmethod == DAYandBRADLEY) {
		ANL->Qs2[ly] = ANL->Qs0[ly];
		ANL->Qp2[ly] = ANL->Qp0[ly];
	    }
	}

    } else if (interface == KMINTERFACE) {
					 /*** Add an interface layer between each layer ***/
	int pos;
	for (pos = 0; pos <= 2; pos += 2) {	/* (z+0) xy then (z+ds/2) z positions */
	    struct typelayer LayerOri = { 0 };
	    struct typelayer Layertmp = { 0 };
	    struct typelayer Layer = { 0 };

	    int lyBeginInt, lyEndInt;
	    int lyOri;
	    int KB, KE;
	    double weight, zSup, zInf, prevZinf;
	    double depthE;	/* altitude at the bottom of the layer   */
	    int NextlyOriToWrite = 0;

	    for (lyOri = 0; lyOri < MDM->nLayer; lyOri++) {
		EmptyLayer(&LayerOri);
		ExtractAlayer(&LayerOri, lyOri, 0, MDM, ANL);
		GetkBkE(&KB, &KE, lyOri, MDM, PRM);

	/*** compute the interface layer ***/
		/* for pos= 0 : int_{-ds/2}^{+ds/2}, 
		   for pos= 2 : int_{0}^{+ds}, 
		 */
		EmptyLayer(&Layer);
		if (lyOri != 0) {
		    prevZinf = zInf;
		}
#ifndef POSITION
		if (pos == 0) {
		    lyBeginInt = MDM->k2ly0[KE + 1];
		    lyEndInt = MDM->k2ly0[KE];
		    zSup = (KE - 0.5) * PRM.ds;
		    zInf = (KE - 1.5) * PRM.ds;
		} else if (pos == 2) {
		    lyBeginInt = MDM->k2ly2[KE + 1];
		    lyEndInt = MDM->k2ly2[KE];
		    zSup = (KE) * PRM.ds;
		    zInf = (KE - 1) * PRM.ds;
		}
#else
		if (pos == 0) {
		    lyBeginInt = MDM->k2ly2[KE];
		    lyEndInt = MDM->k2ly2[KE - 1];
		    zSup = (KE - 0.5) * PRM.ds;
		    zInf = (KE - 1.5) * PRM.ds;
		} else if (pos == 2) {
		    lyBeginInt = MDM->k2ly0[KE + 1];
		    lyEndInt = MDM->k2ly0[KE];
		    zSup = (KE) * PRM.ds;
		    zInf = (KE - 1) * PRM.ds;
		}
#endif
		if (VERBOSE > 3 && PRM.me == 0) {
		    fprintf(stderr,
			    "### layerOri %i : pos=%i kB=%i kE=%i \n",
			    lyOri + 1, pos, KB, KE);
		    fprintf(stderr,
			    "### layerOri %i : pos=%i zSup=%g zInf=%g \n",
			    lyOri + 1, pos, zSup, zInf);
		    fprintf(stderr,
			    "### layerOri %i : pos=%i integrate from %i to %i \n",
			    lyOri + 1, pos, lyBeginInt + 1, lyEndInt + 1);
		}

		for (ly = lyBeginInt; ly <= lyEndInt; ly++) {
		    ExtractAlayer(&Layertmp, ly, 0, MDM, ANL);
		    if (ly != MDM->nLayer - 1) {
			depthE = MDM->laydep[ly + 1];
		    } else {
			depthE = zInf - 1.;
		    }		/* for last layer, make sure DMAX( zInf, depdhE )= zInf */

		    weight =
			1. / PRM.ds * (DMIN(zSup, MDM->laydep[ly]) -
				       DMAX(zInf, depthE));
		    AddPart2Layer(&Layer, Layertmp, weight);

		    if (VERBOSE > 3 && PRM.me == 0) {
			fprintf(stderr,
				"### LayerInt pos=%i : layerOri %i contribute with weight=%g \n",
				pos, ly + 1, weight);
		    }
		    assert(weight <= 1.);
		    assert(weight >= 0.);
		}
	/*** 
             1- write myself
             2 - write the interface below me
        ***/

		if (lyOri == NextlyOriToWrite) {
	  /** append me **/
		    if (lyOri != 0) {
			LayerOri.laydep = prevZinf;
		    }
		    AppendLayer2MDM(&newMDM, &newANL, pos, LayerOri);
		    if (VERBOSE > 3 && PRM.me == 0) {
			fprintf(stderr, "## layerOri %i is writing \n",
				lyOri + 1);
			PrintLayer(LayerOri);
		    }
	  /** append interface **/
		    if (lyOri != MDM->nLayer - 1) {	/* last layer do not have interface below */

			Layer.laydep = zSup;

			Layer.mu = 1. / Layer.mu;
			Layer.kap = 1. / Layer.kap;
			if (ANLmethod == ANOTHER) {
			    Layer.q = 1. / Layer.q;
			    if (Layer.q <= 0) {
				Layer.amp = 1.0;
			    } else {
				Layer.amp =
				    exp(-PRM.pi * f0 * PRM.dt / Layer.q);
			    }

			} else if (ANLmethod == KRISTEKandMOCZO
				   || ANLmethod == DAYandBRADLEY) {
			    Layer.Qp = 1. / Layer.Qp;
			    Layer.Qs = 1. / Layer.Qs;
			}
			AppendLayer2MDM(&newMDM, &newANL, pos, Layer);

			NextlyOriToWrite = lyEndInt;
			if (VERBOSE > 3 && PRM.me == 0) {
			    PrintLayer(Layer);
			}
		    }		/* end last layer do not have interface below */

		}		/* end I write */
	    }			/* end for lyOri */

	}			/*  end for pos */


    /*** update map k2ly ***/
	newMDM.k2ly0 = ivector(ZMIN - DELTA, ZMAX0);
	newMDM.k2ly2 = ivector(ZMIN - DELTA, ZMAX0);
	Initk2ly(&newMDM, PRM);

    /*** replace MDM with newMDM ***/
	int prevNLAYER0 = MDM->nLayer;
	free_dvector(MDM->laydep, 0, prevNLAYER0 - 1);
	free_dvector(MDM->rho0, 0, prevNLAYER0 - 1);
	free_dvector(MDM->mu0, 0, prevNLAYER0 - 1);
	free_dvector(MDM->kap0, 0, prevNLAYER0 - 1);
	free_ivector(MDM->k2ly0, ZMIN - DELTA, ZMAX0);
	free_ivector(MDM->k2ly2, ZMIN - DELTA, ZMAX0);

	MDM->nLayer = newMDM.nLayer;
	MDM->nLayer2 = newMDM.nLayer2;

	MDM->laydep = newMDM.laydep;
	MDM->k2ly0 = newMDM.k2ly0;
	MDM->k2ly2 = newMDM.k2ly2;

	MDM->rho0 = newMDM.rho0;
	MDM->mu0 = newMDM.mu0;
	MDM->kap0 = newMDM.kap0;

	MDM->laydep2 = newMDM.laydep2;
	MDM->rho2 = newMDM.rho2;
	MDM->mu2 = newMDM.mu2;
	MDM->kap2 = newMDM.kap2;

	if (ANLmethod == ANOTHER) {
	    free_dvector(ANL->q0, 0, prevNLAYER0 - 1);
	    free_dvector(ANL->amp, 0, prevNLAYER0 - 1);

	    ANL->q0 = newANL.q0;
	    ANL->amp = newANL.amp;
	    ANL->q2 = newANL.q2;
	    ANL->amp2 = newANL.amp2;

	} else if (ANLmethod == KRISTEKandMOCZO
		   || ANLmethod == DAYandBRADLEY) {
	    free_dvector(ANL->Qp0, 0, prevNLAYER0 - 1);
	    free_dvector(ANL->Qs0, 0, prevNLAYER0 - 1);

	    ANL->Qs0 = newANL.Qs0;
	    ANL->Qp0 = newANL.Qp0;
	    ANL->Qs2 = newANL.Qs2;
	    ANL->Qp2 = newANL.Qp2;
	}
    }				/* end if interface */

    /* give some infos to user */
    if (VERBOSE > 1 && PRM.me == 0) {
	int nmax = 0;
	struct typelayer Layertmp = { 0 };
	int pos;
	for (pos = 0; pos <= 2; pos += 2) {
	    if (pos == 0) {
		nmax = MDM->nLayer;
	    }
	    if (pos == 2) {
		nmax = MDM->nLayer2;
	    }

	    fprintf(stderr, "###### pos %i #######\n", pos);
	    for (ly = 0; ly < nmax; ly++) {
		ExtractAlayer(&Layertmp, ly, pos, MDM, ANL);
		PrintLayer(Layertmp);
	    }
	}
    }



    return EXIT_SUCCESS;
}				/* end function */
