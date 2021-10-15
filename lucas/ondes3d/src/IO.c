#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "../include/new_nrutil.h"
#include "../include/struct.h"
#include "../include/inlineFunctions.h"
#include "../include/options.h"
#include "../include/struct.h"
#include "../include/IO.h"


#if defined (OUTNETCDF) || defined (OUTNEMESIS)
#include <netcdf.h>
#endif
///////////////////////////////////////////
//      Read/Define  MPI Topology File
//////////////////////////////////////////


int ReadTopo(struct PARAMETERS *PRM, const char *topoFile, const int np)
{				/* no segfault */
    FILE *fp;
    char flname[STRL];
    int px, py;

    /* position */


#ifdef TOPO_MPI
    strcpy(flname, topoFile);
    fp = fopen(flname, "r");
    if (fp == NULL) {
	fprintf(stderr, "error when opening file: %s\n", flname);
	exit(EXIT_FAILURE);
    }


    fscanf(fp, "%d %d", &PRM->px, &PRM->py);

    if (np != (PRM->px * PRM->py)) {
	printf
	    ("Mismatch entre le nombre de processeurs demandes et la topologie MPI\n");
	printf("%d %d %d \n", PRM->px, PRM->py, np);
	printf("Arret du code");
	exit(0);
    }
    fclose(fp);
#else

    if (sqrt(np) == floor(sqrt(np))) {
	px = sqrt(np);
	py = sqrt(np);
    } else {
	px = np;
	py = 1;


	while ((px > py) && (px / py > 2)) {

	    if (px % 2 == 0) {
		py = 2 * py;
		px = px / 2;
	    } else {
		break;
	    }

	}

	if (px / py > 2) {
	    while (px > py) {

		if (px % 3 == 0) {
		    py = 3 * py;
		    px = px / 3;
		} else {
		    break;
		}

	    }
	}

	if (px / py > 2) {

	    while (px > py) {

		if (px % 5 == 0) {
		    py = 5 * py;
		    px = px / 5;
		} else {
		    break;
		}

	    }
	}
    }

#endif


    PRM->px = px;
    PRM->py = py;


#if 0
    /* maybe switched with automatical parameters
       advantages : no more topology file needed, no more errors form that
       issue : do not take into account XMIN/XMAX, YMIN/YMAX parameters,
       by allocating, px, py after reading PRM file.
     */
    int px, int py;
    int count;
    px = np;
    if (px % 2 == 1) {
	px = px - 1;
    }				/* to avoid case where px = np and py = 1, useful? */
    py = 1;

    while (px > py) {
	if (px % 2 == 0) {
	    py = 2 * py;
	    px = px / 2;
	} else if (px % 2 == 1) {
	    break;
	}
    }

    PRM->px = px;
    PRM->py = py;
#endif

    return (EXIT_SUCCESS);
}










/////////////////////////////////
//      Read Parameter File
/////////////////////////////////



int ReadPrmFile(struct PARAMETERS *PRM, struct MEDIUM *MDM,	/* no segfault */
		struct ABSORBING_BOUNDARY_CONDITION *ABC,
		struct ANELASTICITY *ANL,
		struct OUTPUTS *OUT, const char *prmFile)
{
    /* READ PRMFILE  */
    /* ============= */
    /* 3 parts in the file
     * 1-> limit of the domain
     * 2-> names of input files
     * 3-> parameters for each layers
     * 4-> other anelasticity parameters
     */
    char *filestr;
    char *cur;			/* current field */
    PRM->delta = DELTA_VAL;
    PRM->pi = acos(-1.0);
    ABC->nPower = NPOWER;
    ABC->reflect = REFLECT;

    filestr = File2str(prmFile);
    /* == FIRST PART == */
    /* steps */
    PRM->dt = FindDble(filestr, "dt");
    PRM->ds = FindDble(filestr, "ds");

    /* numbers */
    PRM->tMax = FindInt(filestr, "tmax");
    PRM->xMin = FindInt(filestr, "xMin");
    PRM->xMax = FindInt(filestr, "xMax");

    PRM->yMin = FindInt(filestr, "yMin");
    PRM->yMax = FindInt(filestr, "yMax");

    PRM->zMin = FindInt(filestr, "zMin");

    if (model == LAYER) {
	if (surface == ABSORBING) {
	    PRM->zMax = FindInt(filestr, "zMax");
	} else if (surface == FREE) {
	    PRM->zMax = 0;
	}
	/* construct zMax0 */
	if (surface == ABSORBING) {
	    PRM->zMax0 = PRM->zMax + PRM->delta + 2;
	} else if (surface == FREE) {
	    PRM->zMax0 = 2;
	}
    } else if (model == GEOLOGICAL) {
	PRM->zMax = FindInt(filestr, "zMax");
	/* imposed values */
	PRM->zMax0 = PRM->zMax + 2;
	MDM->numVoid = 0;
	MDM->numSea = 0;
	if (sea == SEA) {
	    MDM->numSea = 1;
	}

	/* verifs */
	if (PRM->xMin != 1) {
	    perror("error xMin != 1 ");
	    return EXIT_FAILURE;
	}
	if (PRM->yMin != 1) {
	    perror("error yMin != 1 ");
	    return EXIT_FAILURE;
	}
	if (PRM->zMin != 1) {
	    perror("error zMin != 1 ");
	    return EXIT_FAILURE;
	}

    }

    /* ouput plane */
    OUT->i0 = FindInt(filestr, "i0");
    OUT->j0 = FindInt(filestr, "j0");
    if (model == LAYER) {
	if (surface == ABSORBING) {
	    OUT->k0 = FindInt(filestr, "k0");
	} else if (surface == FREE) {
	    OUT->k0 = 0;
	}
    } else if (model == GEOLOGICAL) {
	OUT->k0 = FindInt(filestr, "k0");
    }

    /* special */
    ABC->dump0 =
	-(ABC->nPower +
	  1) * log(ABC->reflect) / (2.0 * PRM->delta * PRM->ds);
    if (ABCmethod == CPML) {
	ABC->fd = FindDble(filestr, "fd0");
	ABC->alpha0 = ABC->fd * PRM->pi;
	ABC->kappa0 = 1.0;
    }
    if (model == GEOLOGICAL) {
	PRM->x0 = FindDble(filestr, "x0");
	PRM->y0 = FindDble(filestr, "y0");
	PRM->z0 = FindDble(filestr, "z0");
    }

    /* == SECOND PART == */
#ifndef OUT_DIR
    cur = FindField(filestr, "dir");
    if (cur[strlen(cur) - 1] != '/') {
	strcat(cur, "/");
    }
    strcpy(PRM->dir, cur);
    free(cur);
#else
    strcpy(PRM->dir, QUOTEME(OUT_DIR));
    if (PRM->dir[strlen(PRM->dir) - 1] != '/') {
	strcat(PRM->dir, "/");
    }
#endif
    cur = FindField(filestr, "fsrcMap");
    strcpy(PRM->fsrcMap, cur);
    free(cur);

    cur = FindField(filestr, "fstatMap");
    strcpy(PRM->fstatMap, cur);
    free(cur);

    /* special */
    if (source == HISTFILE) {
	cur = FindField(filestr, "fsrcHist");
	strcpy(PRM->fsrcHist, cur);
	free(cur);
    }

    if (model == GEOLOGICAL) {
	cur = FindField(filestr, "fgeo");
	strcpy(PRM->fgeo, cur);
	free(cur);
    }


    /* == THIRD PART == */
    double vp, vs;
    double dum;
    char *layStr;
    char layName[STRL], iLay[STRL];
    int i;

    MDM->nLayer = FindInt(filestr, "nlayer");
    MDM->rho0 = mydvector0(0, MDM->nLayer - 1);
    MDM->mu0 = mydvector0(0, MDM->nLayer - 1);
    MDM->kap0 = mydvector0(0, MDM->nLayer - 1);

    if (model == LAYER) {
	MDM->laydep = mydvector0(0, MDM->nLayer - 1);
    } else if (model == GEOLOGICAL) {
	MDM->name_mat = calloc(MDM->nLayer, sizeof(char *));
	for (i = 0; i < MDM->nLayer; i++) {
	    MDM->name_mat[i] = calloc(STRL, sizeof(char));
	}
    }

    if (ANLmethod == ANOTHER) {
	ANL->q0 = mydvector0(0, MDM->nLayer - 1);
	ANL->amp = mydvector0(0, MDM->nLayer - 1);
    } else if (ANLmethod == DAYandBRADLEY || ANLmethod == KRISTEKandMOCZO) {
	ANL->Qp0 = mydvector0(0, MDM->nLayer - 1);
	ANL->Qs0 = mydvector0(0, MDM->nLayer - 1);
    }

    for (i = 0; i < MDM->nLayer; i++) {
	strcpy(layName, "layer");
	sprintf(iLay, "%i", i + 1);
	strcat(layName, iLay);
	layStr = FindField(filestr, layName);

	MDM->rho0[i] = FindDble(layStr, "rho");
	vp = FindDble(layStr, "vp");
	vs = FindDble(layStr, "vs");
	RhoVpVs2MuKap(MDM->rho0[i], vp, vs, &MDM->mu0[i], &MDM->kap0[i]);

	/* special */
	if (model == LAYER) {
	    MDM->laydep[i] = FindDble(layStr, "depth");
	    MDM->laydep[i] = 1000. * MDM->laydep[i];	/* convert [km] to [m]  */

	    if (i > 0) {
		if (MDM->laydep[i - 1] < MDM->laydep[i]) {
		    fprintf(stderr,
			    "error in layer's depth : \n layers %i and %i \n depth %g %g\n",
			    i, i + 1, MDM->laydep[i - 1], MDM->laydep[i]);
		    return (EXIT_FAILURE);
		}
	    }
	} else if (model == GEOLOGICAL) {
	    cur = FindField(layStr, "name");
	    strcpy(MDM->name_mat[i], cur);
	    free(cur);
	}

	if (ANLmethod == ANOTHER) {
	    ANL->q0[i] = FindDble(layStr, "q0");
	    if (ANL->q0[i] <= 0.) {
		ANL->amp[i] = 1.0;
	    } else {
		ANL->amp[i] = exp(-PRM->pi * f0 * PRM->dt / ANL->q0[i]);
	    }
	} else if (ANLmethod == DAYandBRADLEY ||
		   ANLmethod == KRISTEKandMOCZO) {
	    ANL->Qp0[i] = FindDble(layStr, "Qp");
	    ANL->Qs0[i] = FindDble(layStr, "Qs");
	}
	free(layStr);
    }

    /* Force void for z > 0. for free surface */
    if ((model == LAYER) && (surface == FREE)) {	/* we add a small part to avoid precision errors */
	MDM->laydep[0] = 0. + PRM->ds / 1000.;
    }


    /* == FOURTH PART == */
    if (ANLmethod == DAYandBRADLEY) {
	ANL->tm = FindDble(filestr, "taum");
	ANL->tM = FindDble(filestr, "tauM");
	ANL->w0 = FindDble(filestr, "w0");
	if (ANL->tm >= ANL->tM) {
	    if (PRM->me == 0)
		printf
		    ("Warning :error in Day And Bradley parameters : tm and tM \n try to correct it\n");
	    dum = ANL->tm;
	    ANL->tm = ANL->tM;
	    ANL->tM = dum;
	}
    } else if (ANLmethod == KRISTEKandMOCZO) {
	ANL->wmin = FindDble(filestr, "wmin");
	ANL->wmax = FindDble(filestr, "wmax");
	if (ANL->wmin >= ANL->wmax) {
	    if (PRM->me == 0)
		printf
		    ("Warning :error in Day And Bradley parameters : wmin and wmax \n try to correct it\n");
	    dum = ANL->wmin;
	    ANL->wmin = ANL->wmax;
	    ANL->wmax = dum;
	}
    }

    free(filestr);

    /* Inform user  */
    if (PRM->me == 0) {
	int ly;
	/* mapping */
	int X0 = PRM->x0;
	int Y0 = PRM->y0;
	int Z0 = PRM->z0;
	int XMIN = PRM->xMin;
	int XMAX = PRM->xMax;
	int YMIN = PRM->yMin;
	int YMAX = PRM->yMax;
	int ZMIN = PRM->zMin;
	int ZMAX = PRM->zMax;
	int ZMAX0 = PRM->zMax0;
	int DELTA = PRM->delta;
	int NLAYER = MDM->nLayer;
	double DS = PRM->ds;
	/* parameters infos */
	if (model == GEOLOGICAL)
	    printf("## GEOLOGICAL medium representation  ##\n");
	if (model == LAYER)
	    printf("## LAYER medium representation  ##\n");

	if (surface == FREE)
	    printf("## FREE SURFACE on top ##\n");
	if (surface == ABSORBING)
	    printf("## ABSORBING SURFACE on top ##\n");

	if (ABCmethod == CPML)
	    printf("## CPML absorbing layers ##\n");
	if (ABCmethod == PML)
	    printf("## PML absorbing layers ##\n");

	if (ANLmethod == ELASTIC)
	    printf("## Elastic medium ##\n");
	if (ANLmethod == ANOTHER)
	    printf("## simple anelasticity method ##\n");
	if (ANLmethod == DAYandBRADLEY)
	    printf("## Day and Bradley anelasticity  ##\n");
	if (ANLmethod == KRISTEKandMOCZO)
	    printf("## Kristek and Moczo anelasticity  ##\n");

	/* rest */
	printf("\nDimension of FDM order ... %i\n", 4);
	printf("\nParameter File ... %s\n", prmFile);

	if (model == GEOLOGICAL) {
	    printf("Geological model from ... %s\n", PRM->fgeo);
	}

	printf("Source Model based on ... %s\n", PRM->fsrcMap);

	if (source == HISTFILE) {
	    printf("Rupture History from ... %s\n", PRM->fsrcHist);
	}

	printf("Station Position at ... %s\n", PRM->fstatMap);

	printf("Output directory ... %s\n", PRM->dir);
	printf("\nspatial grid DS = %f[m]\n", PRM->ds);
	printf("time step dt = %f[s]\n", PRM->dt);

	if (model == GEOLOGICAL) {	/* the geological model is read in a file */
	    printf("\nVisualisation of plane (y,z) at X = %7.2f [km]\n",
		   ((OUT->i0 - 1) * DS + PRM->x0) / 1000.);
	    printf("Visualisation of plane (x,z) at Y = %7.2f [km]\n",
		   ((OUT->j0 - 1) * DS + PRM->y0) / 1000.);
	} else if (model == LAYER) {	/* the layers are read in .PRM */
	    printf("\nVisualisation of plane (y,z) at X = %7.2f [km]\n",
		   (OUT->i0 - 1) * DS / 1000.);
	    printf("Visualisation of plane (x,z) at Y = %7.2f [km]\n",
		   (OUT->j0 - 1) * DS / 1000.);
	}


	if (model == GEOLOGICAL) {	/* the geological model is read in a file */

	    printf("Visualisation of plane (x,y) at Z = %7.2f [km]\n",
		   ((OUT->k0 - 1) * DS + Z0) / 1000.);
	    printf("\nModel Region (%i:%i, %i:%i, %i:%i)\n", XMIN, XMAX,
		   YMIN, YMAX, ZMIN, ZMAX);
	    printf
		("        (%7.2f:%7.2f, %7.2f:%7.2f, %7.2f:%7.2f) [km]\n",
		 (X0 + XMIN * DS) / 1000., (X0 + XMAX * DS) / 1000.,
		 (Y0 + YMIN * DS) / 1000., (Y0 + YMAX * DS) / 1000.,
		 (Z0 + ZMIN * DS) / 1000., (Z0 + ZMAX * DS) / 1000.);

	} else if (model == LAYER) {	/* the layers are read in .PRM */
	    printf("Visualisation of plane (x,y) at Z = %7.2f [km]\n",
		   (OUT->k0 - 1) * DS / 1000.);
	    printf("\nModel Region (%i:%i, %i:%i, %i:%i)\n", XMIN, XMAX,
		   YMIN, YMAX, ZMIN, ZMAX);
	    printf
		("        (%7.2f:%7.2f, %7.2f:%7.2f, %7.2f:%7.2f) [km]\n",
		 XMIN * DS / 1000., XMAX * DS / 1000., YMIN * DS / 1000.,
		 YMAX * DS / 1000., ZMIN * DS / 1000., ZMAX * DS / 1000.);
	}			/* end of if model */

	if (ABCmethod == PML) {	/* PML */
	    printf
		("\nCPML absorbing boundary, dumping %f, width %d, ratio %f\n",
		 ABC->dump0, DELTA, ABC->reflect);
	} else if (ABCmethod == CPML) {	/* CPML */
	    printf
		("\nCPML absorbing boundary, dumping %f, width %d, ratio %f, frequency %f\n",
		 ABC->dump0, DELTA, ABC->reflect, ABC->fd);
	}


	printf("\nstructure model\n");
	for (ly = 0; ly < NLAYER; ly++) {
	    vp = RhoMuKap2Vp(MDM->rho0[ly], MDM->mu0[ly], MDM->kap0[ly]);
	    vs = RhoMu2Vs(MDM->rho0[ly], MDM->mu0[ly]);
	    if (model == LAYER) {
		if (ANLmethod == ANOTHER) {	/* another attenuation method */
		    printf("depth[km]   vp      vs      rho     q0\n");
		    printf("%7.2f %7.2f %7.2f %7.2f %7.5e\n",
			   MDM->laydep[ly] / 1000., vp, vs, MDM->rho0[ly],
			   ANL->q0[ly]);
		} else {
		    printf("depth[km]   vp      vs      rho\n");
		    printf("%7.2f %7.2f %7.2f %7.2f\n",
			   MDM->laydep[ly] / 1000., vp, vs, MDM->rho0[ly]);
		}		/* end of ANLmethod */
	    } else if (model == GEOLOGICAL) {
		if (ANLmethod == ANOTHER) {	/* another attenuation method */
		    printf("name   vp      vs      rho     q0\n");
		    printf("%s %7.2f %7.2f %7.2f %7.5e\n",
			   MDM->name_mat[ly], vp, vs, MDM->rho0[ly],
			   ANL->q0[ly]);
		} else {
		    printf("name   vp      vs      rho\n");
		    printf("%s %7.2f %7.2f %7.2f\n",
			   MDM->name_mat[ly], vp, vs, MDM->rho0[ly]);
		}		/* end of ANLmethod */
	    }
	}			/* end for ly */


	if (ANLmethod == DAYandBRADLEY) {	/* Day and Bradley method */
	    printf("anelasticity method : Day and Bradley method \n");
	    printf("Lowest relaxation time : %f\n", ANL->tm);
	    printf("Highest relaxation time : %f\n", ANL->tM);
	    printf("Reference frequency : %f\n", ANL->w0);
	    for (ly = 0; ly < MDM->nLayer; ly++) {
		printf("\nLayer number %i :\n", ly + 1);
		printf("Anelastic quality factor (P-waves) : %f\n",
		       ANL->Qp0[ly]);
		printf("Anelastic quality factor (S-waves) : %f\n",
		       ANL->Qs0[ly]);
	    }
	} else if (ANLmethod == KRISTEKandMOCZO) {	/* Kristek and Moczo method */
	    printf("anelasticity method : Kristek and Moczo method \n");
	    printf("Lowest pulsation : %f\n", ANL->wmin);
	    printf("Highest pulsation : %f\n", ANL->wmax);
	    for (ly = 0; ly < MDM->nLayer; ly++) {
		printf("\nLayer number %i :\n", ly + 1);
		printf("Anelastic quality factor (P-waves) : %f\n",
		       ANL->Qp0[ly]);
		printf("Anelastic quality factor (S-waves) : %f\n",
		       ANL->Qs0[ly]);
	    }
	}

    }				/* end of me = 0 */

    return (EXIT_SUCCESS);
}				/* end function ReadPrmFile */







///////////////////////////////////////////
/* Reading the source position */
///////////////////////////////////////////
int ReadSrc(struct SOURCE *SRC, struct PARAMETERS PRM)
{
    /* NB :
     * Contrary to previous version,
     * we do not store real coordinates of each sources since they are unused after
     */
    FILE *fp;
    char flname[STRL];
    double xhypo;
    double yhypo;
    double zhypo;
    int is, n;
    enum typePlace place;
    int X0 = PRM.x0;
    int Y0 = PRM.y0;
    int Z0 = PRM.z0;
    double DS = PRM.ds;
    int ISRC;

    /* position */
    strcpy(flname, PRM.fsrcMap);
    fp = fopen(flname, "r");
    if (fp == NULL) {
	fprintf(stderr, "error when opening file: %s\n", flname);
	return (EXIT_FAILURE);
    }
    fscanf(fp, "%d", &(SRC->iSrc));
    fscanf(fp, "%lf %lf %lf", &(SRC->xhypo0), &(SRC->yhypo0),
	   &(SRC->zhypo0));
    if (model == GEOLOGICAL) {	/* the geological model is read in a file */
	SRC->ixhypo0 = floor(SRC->xhypo0 - X0 / DS) + 1;
	SRC->iyhypo0 = floor(SRC->yhypo0 - Y0 / DS) + 1;
	SRC->izhypo0 = floor(SRC->zhypo0 - Z0 / DS) + 1;
    } else if (model == LAYER) {	/* the layers are read in .PRM */
	SRC->ixhypo0 = floor(SRC->xhypo0 / DS) + 1;
	SRC->iyhypo0 = floor(SRC->yhypo0 / DS) + 1;
	SRC->izhypo0 = floor(SRC->zhypo0 / DS) + 1;
    }				/* end of if model */

#if (VERBOSE > 2)
    if (PRM.me == 0) {
	printf("\nNUMBER OF SOURCE %d\n", SRC->iSrc);
	printf("Hypocenter ... (%f, %f, %f)\n", SRC->xhypo0, SRC->yhypo0,
	       SRC->zhypo0);
	printf(".............. (%i, %i, %i)\n", SRC->ixhypo0, SRC->iyhypo0,
	       SRC->izhypo0);
    }
#endif

    /* mapping */
    ISRC = SRC->iSrc;

    /* allocate */
    SRC->ixhypo = ivector(0, ISRC - 1);
    SRC->iyhypo = ivector(0, ISRC - 1);
    SRC->izhypo = ivector(0, ISRC - 1);
    SRC->insrc = ivector(0, ISRC - 1);

    if (source == HISTFILE) {	/* the source time function is read in .hist */
	SRC->strike = mydvector0(0, ISRC - 1);
	SRC->dip = mydvector0(0, ISRC - 1);
	SRC->rake = mydvector0(0, ISRC - 1);
	SRC->slip = mydvector0(0, ISRC - 1);
	SRC->xweight = mydvector0(0, ISRC - 1);
	SRC->yweight = mydvector0(0, ISRC - 1);
	SRC->zweight = mydvector0(0, ISRC - 1);
    }


    for (is = 0; is < SRC->iSrc; is++) {
	/* read */
	if (fscanf(fp, "%d %lf %lf %lf", &n, &xhypo, &yhypo, &zhypo)
	    != 4)
	    return EXIT_FAILURE;

	n = n - 1;		/* begining of indexes from 1 -> 0 */
	if (model == GEOLOGICAL) {

	    SRC->ixhypo[n] = floor((xhypo - X0) / DS) + 1;
	    SRC->iyhypo[n] = floor((yhypo - Y0) / DS) + 1;
	    SRC->izhypo[n] = floor((zhypo - Z0) / DS) + 1;

	    if (source == HISTFILE) {
		SRC->xweight[n] = (xhypo - X0) / DS - SRC->ixhypo[n] + 1;
		SRC->yweight[n] = (yhypo - Y0) / DS - SRC->iyhypo[n] + 1;
		SRC->zweight[n] = (zhypo - Z0) / DS - SRC->izhypo[n] + 1;
	    }

	} else if (model == LAYER) {
	    SRC->ixhypo[n] = floor((xhypo) / DS) + 1;
	    SRC->iyhypo[n] = floor((yhypo) / DS) + 1;
	    SRC->izhypo[n] = floor((zhypo) / DS) + 1;

	    if (source == HISTFILE) {
		SRC->xweight[n] = (xhypo) / DS - SRC->ixhypo[n] + 1;
		SRC->yweight[n] = (yhypo) / DS - SRC->iyhypo[n] + 1;
		SRC->zweight[n] = (zhypo) / DS - SRC->izhypo[n] + 1;
	    }

	}			/* end of if model */

#if ( VERBOSE > 2 )
	if (PRM.me == 0) {
	    printf("Source %i .... (%f, %f, %f)\n", n + 1, xhypo, yhypo,
		   zhypo);
	    printf(".............. (%i, %i, %i)\n", SRC->ixhypo[n],
		   SRC->iyhypo[n], SRC->izhypo[n]);
	}
#endif

	/* verify source is inside */
	SRC->insrc[n] = 1;
	place =
	    WhereAmI(SRC->ixhypo[n], SRC->iyhypo[n], SRC->izhypo[n], PRM);
	if (!(place == REGULAR || place == FREESURFACE)) {
	    if (PRM.me == 0) {
		printf
		    ("Warning: Source %d (%d, %d, %d) (%8.2f, %8.2f, %8.2f)km is not included\n",
		     n + 1, SRC->ixhypo[n], SRC->iyhypo[n], SRC->izhypo[n],
		     xhypo / 1000., yhypo / 1000., zhypo / 1000.);
	    }
	    SRC->insrc[n] = 0;
	}
    }				/* end for each source */

    fclose(fp);

#if (VERBOSE >1 )
    if (PRM.me == 0)
	fprintf(stderr, "%s [read] \n", flname);
#endif

    if (source == HISTFILE) {	/* the source time function is read in .hist */
	FILE *fp2;
	char buf[STRL];
	double mo;
	int it;
	int IDUR;
	const double PI = PRM.pi;

	/* read the first part */
	strcpy(flname, PRM.fsrcHist);
	fp2 = fopen(flname, "r");
	if (fp2 == NULL) {
	    perror("cannot open HistFile");
	    return EXIT_FAILURE;
	}

	fgets(buf, STRL - 1, fp2);	/* avoid the 2 first lines */
	fgets(buf, STRL - 1, fp2);

	fscanf(fp2, "%lf %lf", &(SRC->dsbiem), &(SRC->dtbiem));
	fscanf(fp2, "%d", &(IDUR));
	SRC->iDur = IDUR;

	if (PRM.me == 0) {
	    printf("\nSource duration %f sec\n", SRC->dtbiem * (IDUR - 1));
	    printf("fault segment %f m, %f s\n", SRC->dsbiem, SRC->dtbiem);
	}			/*  end if me == 0  */

	/* allocate */
	SRC->vel = dmatrix(0, ISRC - 1, 0, SRC->iDur - 1);
	/* read the rest */
	mo = 0.0;
	for (n = 0; n < ISRC; n++) {
	    if (fscanf(fp2, "%d", &is) != 1)
		return EXIT_FAILURE;
	    is = is - 1;
	    if (fscanf
		(fp2, "%lf %lf %lf", &(SRC->strike[is]), &(SRC->dip[is]),
		 &(SRC->rake[is]))
		!= 3)
		return EXIT_FAILURE;
#if (VERBOSE > 2 )
	    if (PRM.me == 0) {
		printf("%d ( %i %i %i ) : %f %f %f\n",
		       is + 1, SRC->ixhypo[is], SRC->iyhypo[is],
		       SRC->izhypo[is], SRC->strike[is], SRC->dip[is],
		       SRC->rake[is]);
	    }
#endif
	    SRC->strike[is] = SRC->strike[is] / 180. * PI;
	    SRC->dip[is] = SRC->dip[is] / 180. * PI;
	    SRC->rake[is] = SRC->rake[is] / 180. * PI;
	    SRC->slip[is] = 0.;

	    for (it = 0; it < SRC->iDur; it++) {
		if (1 != fscanf(fp2, "%lf", &(SRC->vel[is * (SRC->iDur - 1) + it])))
		    return EXIT_FAILURE;
		SRC->slip[is] += SRC->vel[is * (SRC->iDur - 1) + it] * SRC->dtbiem;
		SRC->vel[is * (SRC->iDur - 1) + it] =
		    (SRC->dsbiem / DS) * (SRC->dsbiem / DS) *
		    SRC->vel[is * (SRC->iDur - 1) + it] / DS;
	    }
	    SRC->mo += SRC->slip[is];
	}			/* end of is (source) */

	fclose(fp2);

	SRC->mo = SRC->dsbiem * SRC->dsbiem * SRC->mo;
	if (SRC->mo == 0.) {
	    return EXIT_FAILURE;
	}
	SRC->mw = (log10(SRC->mo) - 9.1) / 1.5;
	if (PRM.me == 0) {
	    printf("Mw = %f; Mo = %e [N m] \n", SRC->mw, SRC->mo);
	}

    }				/* end of source = HISTFILE */

    return (EXIT_SUCCESS);
}				/* end ReadSrc */



//////////////////////////////////////
//      Read Stations
/////////////////////////////////////



int ReadStation(struct OUTPUTS *OUT, struct PARAMETERS PRM,
		struct MEDIUM MDM)
{
    FILE *fp_in3;
    char flname[STRL];
    int ir;

    int IOBS;
    int icpu, jcpu, imp, jmp, k, i1;


    const int TMAX = PRM.tMax;
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
    double ori;

    /* Reading the positions of the stations */
    if ((model == LAYER) || (model == GEOLOGICAL && surface == FREE)) {
	if (PRM.me == 0) {
	    printf("\n Stations coordinates :\n");
	}
    }

    strcpy(flname, PRM.fstatMap);
    fp_in3 = fopen(flname, "r");
    if (fp_in3 == NULL) {
	perror("failed at fopen station file");
	return (EXIT_FAILURE);
    }

    fscanf(fp_in3, "%d", &(OUT->iObs));
    IOBS = OUT->iObs;		/* mapping */


    OUT->nobs = ivector(0, IOBS - 1);
    OUT->xobs = dvector(0, IOBS - 1);
    OUT->yobs = dvector(0, IOBS - 1);
    OUT->zobs = dvector(0, IOBS - 1);

    OUT->ixobs = ivector(0, IOBS - 1);
    OUT->iyobs = ivector(0, IOBS - 1);
    OUT->izobs = ivector(0, IOBS - 1);
    OUT->xobswt = dvector(0, IOBS - 1);
    OUT->yobswt = dvector(0, IOBS - 1);
    OUT->zobswt = dvector(0, IOBS - 1);
    OUT->ista = ivector(0, IOBS - 1);




    for (ir = 0; ir < IOBS; ir++) {

	if (model == LAYER) {	/* the layers are read in .PRM */
	    /* warning : after zobs is ignored when surface == FREE */
	    fscanf(fp_in3, "%d %lf %lf %lf", &(OUT->nobs[ir]),
		   &OUT->xobs[ir], &OUT->yobs[ir], &OUT->zobs[ir]);

	} else if (model == GEOLOGICAL) {	/* the geological model is read in a file */

	    fscanf(fp_in3, "%d %lf %lf", &(OUT->nobs[ir]), &OUT->xobs[ir],
		   &OUT->yobs[ir]);
	    OUT->zobs[ir] = 0.;
	}			/* end of if model */

	OUT->ista[ir] = 1;
	ori = 0.;

	if (model == GEOLOGICAL)
	    ori = PRM.x0;
	OUT->ixobs[ir] = floor((OUT->xobs[ir] - ori) / DS) + 1;
	OUT->xobswt[ir] =
	    ((OUT->xobs[ir] - ori) / DS - OUT->ixobs[ir] + 1);

	if (model == GEOLOGICAL)
	    ori = PRM.y0;
	OUT->iyobs[ir] = floor((OUT->yobs[ir] - ori) / DS) + 1;
	OUT->yobswt[ir] = (OUT->yobs[ir] - ori) / DS - OUT->iyobs[ir] + 1;


	if (model == GEOLOGICAL) {

	    if (OUT->xobs[ir] - PRM.x0 < XMIN * DS
		|| OUT->xobs[ir] - PRM.x0 > XMAX * DS
		|| OUT->yobs[ir] - PRM.y0 < YMIN * DS
		|| OUT->yobs[ir] - PRM.y0 > YMAX * DS) {
		OUT->ista[ir] = 0;
	    }



	    if (OUT->ista[ir] == 1) {
		icpu = PRM.i2icpu_array[OUT->ixobs[ir]];
		jcpu = PRM.j2jcpu_array[OUT->iyobs[ir]];


		imp = PRM.i2imp_array[OUT->ixobs[ir]];
		jmp = PRM.j2jmp_array[OUT->iyobs[ir]];

		if (PRM.coords[0] == icpu) {
		    if (PRM.coords[1] == jcpu) {


			OUT->izobs[ir] = ZMAX;

			for (k = ZMAX; k >= ZMIN - (PRM.delta); k--) {
			    i1 = i3access(MDM.imed, -1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0, imp, jmp, k);
			    if (i1 <= MDM.numSea)
				OUT->izobs[ir] = OUT->izobs[ir] - 1;
			}

			OUT->zobs[ir] =
			    (OUT->izobs[ir] - 1) * PRM.ds + PRM.z0;
			OUT->zobswt[ir] = 0.0;
		    }
		}

	    }


	}



	if (model == LAYER) {	/* the layers are read in .PRM */

	    OUT->izobs[ir] = floor(OUT->zobs[ir] / DS) + 1;
	    OUT->zobswt[ir] = (OUT->zobs[ir] / DS - OUT->izobs[ir] + 1);

	    if (OUT->xobs[ir] < XMIN * DS || OUT->xobs[ir] > XMAX * DS ||
		OUT->yobs[ir] < YMIN * DS || OUT->yobs[ir] > YMAX * DS ||
		OUT->zobs[ir] < ZMIN * DS || OUT->zobs[ir] > ZMAX * DS) {
		OUT->ista[ir] = 0;
	    }
#if ( VERBOSE > 2 )
	    if (PRM.me == 0) {
		printf("Station %d (%d) : %f %f %f\n", ir + 1,
		       OUT->ista[ir], OUT->xobs[ir], OUT->yobs[ir],
		       OUT->zobs[ir]);
		if (OUT->ista[ir] == 1) {
		    printf("global position : %d %d %d\n", OUT->ixobs[ir],
			   OUT->iyobs[ir], OUT->izobs[ir]);
		    printf("weights : %f %f %f\n", OUT->xobswt[ir],
			   OUT->yobswt[ir], OUT->zobswt[ir]);
		}
	    }
#endif
	} else if (model == GEOLOGICAL) {	/* the geological model is read in a file */

	    if (OUT->xobs[ir] - PRM.x0 < XMIN * DS
		|| OUT->xobs[ir] - PRM.x0 > XMAX * DS
		|| OUT->yobs[ir] - PRM.y0 < YMIN * DS
		|| OUT->yobs[ir] - PRM.y0 > YMAX * DS) {
		OUT->ista[ir] = 0;
	    }

	    if (surface == FREE) {	/* free surface at z = 0 */
#if (VERBOSE > 2)
		if (PRM.me == 0) {
		    printf("Station %d (%d) : %f %f %f\n", ir + 1,
			   OUT->ista[ir], OUT->xobs[ir], OUT->yobs[ir],
			   OUT->zobs[ir]);

		    if (OUT->ista[ir] == 1) {
			printf("global position : %d %d %d\n",
			       OUT->ixobs[ir], OUT->iyobs[ir],
			       OUT->izobs[ir]);
			printf("weights : %f %f %f\n", OUT->xobswt[ir],
			       OUT->yobswt[ir], OUT->zobswt[ir]);
		    }
		}
#endif
	    }

	}			/* end of if model */

    }				/* end of ir (stations) */

    fclose(fp_in3);

    return (EXIT_SUCCESS);
}				/* end ReadStation */





/////////////////////////////////
//      Read Geology File
/////////////////////////////////



int ReadGeoFile(struct MEDIUM *MDM, struct PARAMETERS PRM)
{
    FILE *fp_in0;
    int nx, ny, nz;
    char buf[STRL], buf2[STRL], prop_name[STRL];
    double X0dum, Y0dum, Z0dum;
    double dx, dy, dz;
    int flagi, flagj;		/* to know if you store the value in imed */

    int i, j, k, i1;
    int imp, jmp, icpu, jcpu;

    const int NUM_VOID = MDM->numVoid;
    const int NUM_SEA = MDM->numSea;

    const int XMAX = PRM.xMax;
    const int YMAX = PRM.yMax;
    const int ZMIN = PRM.zMin;
    const int ZMAX = PRM.zMax;
    const int ZMAX0 = PRM.zMax0;
    const int MPMX = PRM.mpmx;
    const int MPMY = PRM.mpmy;
    const int DELTA = PRM.delta;

    assert(PRM.xMin == 1);
    assert(PRM.yMin == 1);
    assert(PRM.zMin == 1);


    fp_in0 = fopen(PRM.fgeo, "r");
    if (fp_in0 == NULL) {
	perror("failed to open Geological file\n");
	return EXIT_FAILURE;
    }

    MDM->imed = i3tensor(-1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0);

    for (imp = -1; imp <= MPMX + 2; imp++) {
    	for (jmp = -1; jmp <= MPMY + 2; jmp++) {
    	    for (k = ZMIN - DELTA; k <= ZMAX0; k++) {
              i3access(MDM->imed, -1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0, imp, jmp, k) = 999;
    		      //MDM->imed[imp][jmp][k] = 999;	/* initial value */
    	    }
    	}
    }

    if (PRM.me == 0)
	printf("\nReading the geological model in file : %s\n", PRM.fgeo);

    fscanf(fp_in0, "%s %d", buf, &nx);
    fscanf(fp_in0, "%s %d", buf, &ny);
    fscanf(fp_in0, "%s %d", buf, &nz);

    if (PRM.me == 0)
	printf("Size of the geological model : %d %d %d\n", nx, ny, nz);

    if (nx != XMAX || ny != YMAX || nz > ZMAX) {
	fprintf(stderr, "Size of the model : %d %d %d\n", XMAX, YMAX,
		ZMAX);
	fprintf(stderr, "Size of the geological model : %d %d %d\n", nx,
		ny, nz);

	perror
	    ("failed: check the values of nx, ny and nz in geological model file");
	return EXIT_FAILURE;
    }

    if (PRM.me == 0)
	printf("Geological model Z+ = %d - Calculation model Z+ = %d\n",
	       nz, ZMAX);

    fscanf(fp_in0, "%s %lf", buf, &X0dum);
    fscanf(fp_in0, "%s %lf", buf, &Y0dum);
    fscanf(fp_in0, "%s %lf", buf, &Z0dum);
    fscanf(fp_in0, "%s %lf", buf, &dx);
    fscanf(fp_in0, "%s %lf", buf, &dy);
    fscanf(fp_in0, "%s %lf", buf, &dz);
    fscanf(fp_in0, "%s %s", buf, buf2);

    if (PRM.me == 0)
	printf
	    ("Origin coordinates in prm file : %7.2f %7.2f %7.2f - grid size : %7.2f\n",
	     PRM.x0, PRM.y0, PRM.z0, PRM.ds);

    if (PRM.me == 0)
	printf
	    ("Origin coordinates in geological model file : %7.2f %7.2f %7.2f - grid size : %7.2f %7.2f %7.2f\n",
	     X0dum, Y0dum, Z0dum, dx, dy, dz);

    if (dx != dy || dy != dz || abs(PRM.ds - dx) > 1.e-6)
	perror
	    ("failed: check the values of dx, dy and dz in geological model file");

  /*** read imed ***/
    for (k = 1; k <= nz; k++) {
	for (j = 1; j <= ny; j++) {
	    for (i = 1; i <= nx; i++) {

		fscanf(fp_in0, "%s", &prop_name);

	/** will you store it in imed ? **/
		/* y direction */
		flagj = 0;
		jcpu = PRM.j2jcpu_array[j];
		jmp = PRM.j2jmp_array[j];
		if (PRM.coords[1] == jcpu) {	/* common case */
		    flagj = 1;
		} else {	/* shared borders case */
		    if (PRM.jmp2j_array[-1] == j) {
			jmp = -1;
			flagj = 1;
		    }
		    if (PRM.jmp2j_array[0] == j) {
			jmp = 0;
			flagj = 1;
		    }
		    if (PRM.jmp2j_array[MPMY + 1] == j) {
			jmp = MPMY + 1;
			flagj = 1;
		    }
		    if (PRM.jmp2j_array[MPMY + 2] == j) {
			jmp = MPMY + 2;
			flagj = 1;
		    }
		}

		/* x direction */
		flagi = 0;
		icpu = PRM.i2icpu_array[i];
		imp = PRM.i2imp_array[i];
		if (PRM.coords[0] == icpu) {	/* common case */
		    flagi = 1;
		} else {	/* shared borders case */
		    if (PRM.imp2i_array[-1] == i) {
			imp = -1;
			flagi = 1;
		    }
		    if (PRM.imp2i_array[0] == i) {
			imp = 0;
			flagi = 1;
		    }
		    if (PRM.imp2i_array[MPMX + 1] == i) {
			imp = MPMX + 1;
			flagi = 1;
		    }
		    if (PRM.imp2i_array[MPMX + 2] == i) {
			imp = MPMX + 2;
			flagi = 1;
		    }

		}

	/** store **/
		if (flagi == 1 && flagj == 1) {
		    for (i1 = 0; i1 < MDM->nLayer; i1++) {
			if (strcmp(MDM->name_mat[i1], prop_name) == 0) {
			    //MDM->imed[imp][jmp][k] = i1;
          i3access(MDM->imed, -1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0, imp, jmp, k) = i1;
			}	/* end of if */

		    }		/* end of i1 */
		}		/* end of if flag == 1 */


	    }			/* end of i */
	}			/* end of j */
    }				/* end of k */
    fclose(fp_in0);

  /*** add air/void above ( numLayer = 0 ) ***/
    if (ZMAX > nz) {
	for (imp = -1; imp <= MPMX + 2; imp++) {
	    for (jmp = -1; jmp <= MPMY + 2; jmp++) {
		for (k = nz; k <= ZMAX; k++) {
		    //MDM->imed[imp][jmp][k] = NUM_VOID;
        i3access(MDM->imed, -1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0, imp, jmp, k) = NUM_VOID;
		}
	    }
	}
    }

	    /*** add sea(1) if void(0) and sea z<=0 ***/
    if (sea == SEA) {
	if (strcmp(MDM->name_mat[NUM_SEA], "eau") != 0) {
	    printf("warning sea is eau but eau is named : %s\n",
		   MDM->name_mat[NUM_SEA]);
	}

	enum typePlace place;
	for (imp = -1; imp <= MPMX + 2; imp++) {
	    for (jmp = -1; jmp <= MPMY + 2; jmp++) {
		for (k = ZMIN - DELTA; k <= ZMAX; k++) {
		    i = PRM.imp2i_array[imp];
		    j = PRM.jmp2j_array[jmp];
		    place = WhereAmI(i, j, k, PRM);
		    if (place == REGULAR && ((k - 1) * PRM.ds + PRM.z0 < 0.) &&	/* should be replaced by a function? */
			i3access(MDM->imed, -1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0, imp, jmp, k) == NUM_VOID) {

			i3access(MDM->imed, -1, MPMX + 2, -1, MPMY + 2, ZMIN - DELTA, ZMAX0, imp, jmp, k) = NUM_SEA;
		    }		/* end if */

		}
	    }
	}
    }				/* end if sea */
    return EXIT_SUCCESS;
}				/* end function read geological file */



///////////////////////////////////////////
//      Write Seismograms File - bin
/////////////////////////////////////////
#ifdef OUTBIN
int OutSeismogramsbin(struct OUTPUTS OUT, struct PARAMETERS PRM, int ir,
		      int l, const char *flname4, int station_step)
{
    FILE *fp4;
    int l1, lend;

#if ( VERBOSE > 2 )
    if (l == PRM.tMax)
	printf("%d %s\n", ir + 1, fp4);
#endif
    /* empty previous file  */
    if (l == station_step || (PRM.tMax < station_step && l == PRM.tMax)) {
	fp4 = fopen(flname4, "wb");
	fclose(fp4);
    }

    fp4 = fopen(flname4, "ab+");
    lend = station_step;
    if (l == PRM.tMax && PRM.tMax % station_step != 0) {
	lend = PRM.tMax % station_step;
    }

    fwrite(&OUT.seis_output[0][ir][1], sizeof(double), lend, fp4);
    fwrite(&OUT.seis_output[0][ir][2], sizeof(double), lend, fp4);
    fwrite(&OUT.seis_output[0][ir][3], sizeof(double), lend, fp4);







    fclose(fp4);
    return EXIT_SUCCESS;

}				/* end function */

#endif


///////////////////////////////////////////
//      Write Seismograms File - bin
/////////////////////////////////////////
#if defined (OUTSTD) || defined (OUTNEMESIS)
int OutSeismogramsbis(struct OUTPUTS OUT, struct PARAMETERS PRM, int ir,
		      int l, const char *flname4, int station_step)
{
    FILE *fp4;
    int l1, lend;

#if ( VERBOSE > 2 )
    if (l == PRM.tMax)
	printf("%d %s\n", ir + 1, flname4);
#endif

    fp4 = fopen(flname4, "a+");
    lend = station_step;
    if (l == PRM.tMax && PRM.tMax % station_step != 0) {
	lend = PRM.tMax % station_step;
    }
//  printf ("%d %d %d %d \n",l1,lend,ir,l);

    for (l1 = 0; l1 < lend; l1++) {
	fprintf(fp4, "%d %15e %15e %15e \n", l - lend + l1 + 1,
		OUT.seis_output[l1][ir][1], OUT.seis_output[l1][ir][2],
		OUT.seis_output[l1][ir][3]);
    }

    fclose(fp4);
    return EXIT_SUCCESS;

}


#endif




///////////////////////////////////////////
//      Write Seismograms File
//////////////////////////////////////////
#ifdef OUTSTD
int OutSeismograms(struct OUTPUTS OUT, struct PARAMETERS PRM, int ir,
		   int l, const char *flname4, int station_step)
{
    FILE *fp4;
    int l1, lend;

#if ( VERBOSE > 2 )
    if (l == PRM.tMax)
	printf("%d %s\n", ir + 1, flname4);
#endif
    /* empty previous file  */
    if (l == station_step || (PRM.tMax < station_step && l == PRM.tMax)) {
	fp4 = fopen(flname4, "w");
	fclose(fp4);
    }


    /* update header */
    fp4 = fopen(flname4, "r+");
    fprintf(fp4, "%d %f %f %f\n", OUT.nobs[ir], OUT.xobs[ir], OUT.yobs[ir],
	    OUT.zobs[ir]);
    fprintf(fp4, "%d %f\n", l, PRM.dt);
    fclose(fp4);

    /* update the end of file */
    fp4 = fopen(flname4, "a+");
    lend = station_step;
    if (l == PRM.tMax && PRM.tMax % station_step != 0) {
	lend = PRM.tMax % station_step;
    }
    for (l1 = 0; l1 < lend; l1++)
	fprintf(fp4, "%15e %15e %15e \n", OUT.seis_output[l1][ir][1],
		OUT.seis_output[l1][ir][2], OUT.seis_output[l1][ir][3]);


/*    fprintf(fp4, "%15e %15e %15e %15e %15e %15e %15e %15e %15e\n",
            OUT.seis_output[l1][ir][1], OUT.seis_output[l1][ir][2], OUT.seis_output[l1][ir][3],
            OUT.seis_output[l1][ir][4], OUT.seis_output[l1][ir][5], OUT.seis_output[l1][ir][6],
            OUT.seis_output[l1][ir][7], OUT.seis_output[l1][ir][8], OUT.seis_output[l1][ir][9]);*/


    fprintf(fp4, "%15e %15e %15e \n", OUT.seis_output[l1][ir][1],
	    OUT.seis_output[l1][ir][2], OUT.seis_output[l1][ir][3]);



    fclose(fp4);
    return EXIT_SUCCESS;

}				/* end function */
#endif



int IOsurfloc2glob1(int flag, int coord0, int coord1,
		    struct PARAMETERS PRM, struct OUTPUTS OUT)
{

    int i, j, XMIN, XMAX, YMIN, YMAX, imp, DELTA;

    XMIN = PRM.xMin;
    XMAX = PRM.xMax;
    YMIN = PRM.yMin;
    YMAX = PRM.yMax;
    DELTA = PRM.delta;

    OUT.total_prec_x = 0;
    for (j = 0; j < coord0; j++) {
	OUT.total_prec_x += PRM.mpmx_tab[j];
    }
    OUT.total_prec_y = 0;
    for (j = 0; j < coord1; j++) {
	OUT.total_prec_y += PRM.mpmy_tab[j];
    }
    imp = 0;
    for (i = 1; i <= PRM.mpmx_tab[coord0]; i++) {
	for (j = 1; j <= PRM.mpmy_tab[coord1]; j++) {
	    assert(imp < OUT.test_size);
	    assert(XMIN - DELTA - 1 + i + OUT.total_prec_x <
		   XMAX + DELTA + 3);
	    assert(YMIN - DELTA - 1 + j + OUT.total_prec_y <
		   YMAX + DELTA + 3);
	    imp++;
	    if (flag == 1)
		OUT.Vxglobal[XMIN - DELTA - 1 + i +
			     OUT.total_prec_x][YMIN - DELTA - 1 + j +
					       OUT.total_prec_y] =
		    OUT.snapBuff[imp];
	    if (flag == 2)
		OUT.Vyglobal[XMIN - DELTA - 1 + i +
			     OUT.total_prec_x][YMIN - DELTA - 1 + j +
					       OUT.total_prec_y] =
		    OUT.snapBuff[imp];
	    if (flag == 3)
		OUT.Vzglobal[XMIN - DELTA - 1 + i +
			     OUT.total_prec_x][YMIN - DELTA - 1 + j +
					       OUT.total_prec_y] =
		    OUT.snapBuff[imp];

	}			/* end of j */
    }				/* end of i */

}




int IOsurfloc2glob3(int flag, int coord0, int coord1,
		    struct PARAMETERS PRM, struct OUTPUTS OUT)
{

    int i, j, XMIN, XMAX, YMIN, YMAX, ZMIN, imp, DELTA, ZMAX0, k;

    XMIN = PRM.xMin;
    ZMIN = PRM.zMin;
    XMAX = PRM.xMax;
    YMIN = PRM.yMin;
    YMAX = PRM.yMax;
    DELTA = PRM.delta;
    ZMAX0 = PRM.zMax0;

    OUT.total_prec_y = 0;
    for (j = 0; j < coord1; j++) {
	OUT.total_prec_y = OUT.total_prec_y + PRM.mpmy_tab[j];
    }
    imp = 0;
    for (j = 1; j <= PRM.mpmy_tab[coord1]; j++) {
	for (k = ZMIN - DELTA; k <= ZMAX0; k++) {
	    assert(imp < OUT.test_size);
	    assert(YMIN - DELTA - 1 + j + OUT.total_prec_y <
		   YMAX + DELTA + 3);
	    imp++;
	    if (flag == 1)
		OUT.Vxglobal[YMIN - DELTA - 1 + j + OUT.total_prec_y][k] =
		    OUT.snapBuff[imp];
	    if (flag == 2)
		OUT.Vyglobal[YMIN - DELTA - 1 + j + OUT.total_prec_y][k] =
		    OUT.snapBuff[imp];
	    if (flag == 3)
		OUT.Vzglobal[YMIN - DELTA - 1 + j + OUT.total_prec_y][k] =
		    OUT.snapBuff[imp];

	}
    }





}




int IOsurfloc2glob2(int flag, int coord0, int coord1,
		    struct PARAMETERS PRM, struct OUTPUTS OUT)
{

    int i, j, XMIN, XMAX, YMIN, YMAX, ZMIN, imp, DELTA, ZMAX0, k;

    XMIN = PRM.xMin;
    ZMIN = PRM.zMin;
    XMAX = PRM.xMax;
    YMIN = PRM.yMin;
    YMAX = PRM.yMax;
    DELTA = PRM.delta;
    ZMAX0 = PRM.zMax0;



    OUT.total_prec_x = 0;
    for (j = 0; j < coord0; j++) {
	OUT.total_prec_x = OUT.total_prec_x + PRM.mpmx_tab[j];
    }
    imp = 0;
    for (i = 1; i <= PRM.mpmx_tab[coord0]; i++) {
	for (k = ZMIN - DELTA; k <= ZMAX0; k++) {
	    assert(imp < OUT.test_size);
	    assert(XMIN - DELTA - 1 + i + OUT.total_prec_x <
		   XMAX + DELTA + 3);
	    imp++;
	    if (flag == 1)
		OUT.Vxglobal[XMIN - DELTA - 1 + i + OUT.total_prec_x][k] =
		    OUT.snapBuff[imp];
	    if (flag == 2)
		OUT.Vyglobal[XMIN - DELTA - 1 + i + OUT.total_prec_x][k] =
		    OUT.snapBuff[imp];
	    if (flag == 3)
		OUT.Vzglobal[XMIN - DELTA - 1 + i + OUT.total_prec_x][k] =
		    OUT.snapBuff[imp];

	}
    }





}



int IOsurfhelper1(double ***v, int flag, int outvel, int outdisp,
		  struct PARAMETERS PRM, struct OUTPUTS OUT)
{

    int imp, MPMX, MPMY, i, j;

    MPMX = PRM.mpmx;
    MPMY = PRM.mpmy;



    imp = 0;
    for (i = 1; i <= MPMX; i++) {
	for (j = 1; j <= MPMY; j++) {
	    assert(imp < OUT.test_size);
	    imp++;

	    if (outvel == 1) {
		if (surface == ABSORBING) {
		    OUT.snapBuff[imp] = v[i][j][OUT.k0];
		} else if (surface == FREE) {
		    OUT.snapBuff[imp] = v[i][j][1];
		}
	    } else if (outdisp == 1) {

		//if (flag == 1)
		    //OUT.snapBuff[imp] = OUT.Uxy[1][i][j];
		//if (flag == 2)
		    //OUT.snapBuff[imp] = OUT.Uxy[2][i][j];
		//if (flag == 3)
		    //OUT.snapBuff[imp] = OUT.Uxy[3][i][j];

	    }

	}			/* end of j */
    }				/* end of i */

}



int IOsurfhelper2(double ***v, int flag, int outvel, int outdisp,
		  struct PARAMETERS PRM, struct OUTPUTS OUT)
{

    int imp, MPMX, MPMY, ZMIN, ZMAX0, DELTA, i, j, jmp_tmp, k;

    MPMX = PRM.mpmx;
    MPMY = PRM.mpmy;
    ZMIN = PRM.zMin;
    DELTA = PRM.delta;
    ZMAX0 = PRM.zMax0;
    jmp_tmp = PRM.j2jmp_array[OUT.j0];



    imp = 0;
    for (i = 1; i <= MPMX; i++) {
	for (k = ZMIN - DELTA; k <= ZMAX0; k++) {
	    assert(imp < OUT.test_size);
	    imp++;
	    if (outvel == 1) {
		OUT.snapBuff[imp] = v[i][jmp_tmp][k];
	    } else if (outdisp == 1) {
		//if (flag == 1)
		    //OUT.snapBuff[imp] = OUT.Uxz[1][i][k];
		//if (flag == 2)
		    //OUT.snapBuff[imp] = OUT.Uxz[2][i][k];
		//if (flag == 3)
		    //OUT.snapBuff[imp] = OUT.Uxz[3][i][k];


	    }
	}
    }



}




int IOsurfhelper3(double ***v, int flag, int outvel, int outdisp,
		  struct PARAMETERS PRM, struct OUTPUTS OUT)
{

    int imp, MPMX, MPMY, ZMIN, ZMAX0, DELTA, i, j, imp_tmp, k;

    MPMX = PRM.mpmx;
    MPMY = PRM.mpmy;
    ZMIN = PRM.zMin;
    DELTA = PRM.delta;
    ZMAX0 = PRM.zMax0;
    imp_tmp = PRM.i2imp_array[OUT.j0];



    imp = 0;
    for (j = 1; j <= MPMY; j++) {
	for (k = ZMIN - DELTA; k <= ZMAX0; k++) {
	    assert(imp < OUT.test_size);
	    imp++;
	    if (outvel == 1) {
		OUT.snapBuff[imp] = v[imp_tmp][j][k];
	    } else if (outdisp == 1) {
		//if (flag == 1)
		    //OUT.snapBuff[imp] = OUT.Uyz[1][j][k];
		//if (flag == 2)
		    //OUT.snapBuff[imp] = OUT.Uyz[2][j][k];
		//if (flag == 3)
		    //OUT.snapBuff[imp] = OUT.Uyz[3][j][k];

	    }
	}
    }



}








#if defined (OUTNETCDF)


////////////////////////////////////////////////////
//      Write OBS NECDF File
////////////////////////////////////////////////////

int OutObsNetcdf(char *flname, int l, int ir, int TMAX, double dt,
		 struct OUTPUTS OUT, int station_step)
{
    /* When we create netCDF variables and dimensions, we get back an
     * ID for each one. */

    int ncid4, ncid4bis, time_dimid, time_varid;
    int seisx_varid, seisy_varid, seisz_varid, seisxx_varid, seisyy_varid,
	seiszz_varid, seisxy_varid, seisyz_varid, seisxz_varid;
    int dimids[1];
    double time_actual_range[2];
    double seisx_missing_value[1], seisy_missing_value[1],
	seisz_missing_value[1], seisxx_missing_value[1],
	seisyy_missing_value[1], seiszz_missing_value[1],
	seisxy_missing_value[1], seisyz_missing_value[1],
	seisxz_missing_value[1];
    int cdf_node_offset[1];
    double *times;

    /* This is the data array we will write. */
    double *seisx_out, *seisy_out, *seisz_out, *seisxx_out, *seisyy_out,
	*seiszz_out, *seisxy_out, *seisxz_out, *seisyz_out;

    double xobs_out[1], yobs_out[1], zobs_out[1];

    /* Error handling and loop index. */
    int retval, l1, i;

    /* These settings tell netcdf to write one timestep of data. (The
     * setting of start[0] inside the loop below tells netCDF which
     * timestep to write.) */
    long count[1] = { station_step }, start[1] = { l - station_step };

    /* Title including the station's coordinates. */
    char title[150] = "Sismogrammes for stations coordinates: x = ",
	xcoord[20], ycoord[20], zcoord[20];

    times = dvector(0, (long) (TMAX - 1));

    seisx_out = dvector(0, (long) (TMAX - 1));
    seisy_out = dvector(0, (long) (TMAX - 1));
    seisz_out = dvector(0, (long) (TMAX - 1));

    seisxx_out = dvector(0, (long) (TMAX - 1));
    seisyy_out = dvector(0, (long) (TMAX - 1));
    seiszz_out = dvector(0, (long) (TMAX - 1));

    seisxy_out = dvector(0, (long) (TMAX - 1));
    seisyz_out = dvector(0, (long) (TMAX - 1));
    seisxz_out = dvector(0, (long) (TMAX - 1));

    xobs_out[0] = OUT.xobs[ir];
//xobs[ir];
    yobs_out[0] = OUT.yobs[ir];
    zobs_out[0] = OUT.zobs[ir];


    for (l1 = 0; l1 < STATION_STEP; l1++) {
//seisx[ir][l1];

	seisx_out[l1] = OUT.seis_output[l1][ir][1];
	seisy_out[l1] = OUT.seis_output[l1][ir][2];
	seisz_out[l1] = OUT.seis_output[l1][ir][3];
	seisxx_out[l1] = OUT.seis_output[l1][ir][4];
	seisyy_out[l1] = OUT.seis_output[l1][ir][5];
	seiszz_out[l1] = OUT.seis_output[l1][ir][6];
	seisxy_out[l1] = OUT.seis_output[l1][ir][7];
	seisxz_out[l1] = OUT.seis_output[l1][ir][8];
	seisyz_out[l1] = OUT.seis_output[l1][ir][9];
    }


/**************************************************************************/

    for (l1 = 0; l1 < STATION_STEP; l1++)
	times[l1] = dt * (l1 + 1);

    /* Create the file. */

    if (l == station_step) {
	if ((retval = nc_create(flname, NC_CLOBBER, &ncid4)))
	    ERR(retval);
    } else {
	if ((retval = nc_open(flname, NC_WRITE, &ncid4bis)))
	    ERR(retval);


	if ((retval = nc_inq_varid(ncid4bis, "time", &time_varid)))
	    ERR(retval);
	if ((retval = nc_inq_varid(ncid4bis, "seisx", &seisx_varid)))
	    ERR(retval);
	if ((retval = nc_inq_varid(ncid4bis, "seisy", &seisy_varid)))
	    ERR(retval);
	if ((retval = nc_inq_varid(ncid4bis, "seisz", &seisz_varid)))
	    ERR(retval);
	if ((retval = nc_inq_varid(ncid4bis, "seisxx", &seisxx_varid)))
	    ERR(retval);
	if ((retval = nc_inq_varid(ncid4bis, "seisyy", &seisyy_varid)))
	    ERR(retval);
	if ((retval = nc_inq_varid(ncid4bis, "seiszz", &seiszz_varid)))
	    ERR(retval);
	if ((retval = nc_inq_varid(ncid4bis, "seisxy", &seisxy_varid)))
	    ERR(retval);
	if ((retval = nc_inq_varid(ncid4bis, "seisyz", &seisyz_varid)))
	    ERR(retval);
	if ((retval = nc_inq_varid(ncid4bis, "seisxz", &seisxz_varid)))
	    ERR(retval);
    }


    /* Define the dimensions. NetCDF will hand back an ID for each. */
    if (l == station_step) {
	if ((retval = nc_def_dim(ncid4, "time", TMAX, &time_dimid)))
	    ERR(retval);

	/* The dimids array is used to pass the IDs of the dimensions of
	 * the variable. */
	dimids[0] = time_dimid;


	/* Define the variables. */

	if ((retval = nc_def_var(ncid4, "time", NC_DOUBLE, 1,
				 &time_dimid, &time_varid)))
	    ERR(retval);


	if ((retval = nc_def_var(ncid4, "seisx", NC_DOUBLE, 1,
				 dimids, &seisx_varid)))
	    ERR(retval);
	if ((retval = nc_def_var(ncid4, "seisy", NC_DOUBLE, 1,
				 dimids, &seisy_varid)))
	    ERR(retval);
	if ((retval = nc_def_var(ncid4, "seisz", NC_DOUBLE, 1,
				 dimids, &seisz_varid)))
	    ERR(retval);
	if ((retval = nc_def_var(ncid4, "seisxx", NC_DOUBLE, 1,
				 dimids, &seisxx_varid)))
	    ERR(retval);
	if ((retval = nc_def_var(ncid4, "seisyy", NC_DOUBLE, 1,
				 dimids, &seisyy_varid)))
	    ERR(retval);
	if ((retval = nc_def_var(ncid4, "seiszz", NC_DOUBLE, 1,
				 dimids, &seiszz_varid)))
	    ERR(retval);
	if ((retval = nc_def_var(ncid4, "seisxy", NC_DOUBLE, 1,
				 dimids, &seisxy_varid)))
	    ERR(retval);
	if ((retval = nc_def_var(ncid4, "seisyz", NC_DOUBLE, 1,
				 dimids, &seisyz_varid)))
	    ERR(retval);
	if ((retval = nc_def_var(ncid4, "seisxz", NC_DOUBLE, 1,
				 dimids, &seisxz_varid)))
	    ERR(retval);


	/* Assign attributes to variables. */

	if ((retval = nc_put_att_text(ncid4, time_varid, "long_name",
				      strlen("Time"), "Time")))
	    ERR(retval);
	if ((retval = nc_put_att_text(ncid4, time_varid, "units",
				      strlen("Second"), "Second")))
	    ERR(retval);
	time_actual_range[0] = 0.;
	time_actual_range[1] = TMAX * dt;
	if ((retval = nc_put_att_double(ncid4, time_varid, "actual_range",
					NC_DOUBLE, 2,
					&time_actual_range[0])))
	    ERR(retval);

	if ((retval = nc_put_att_text(ncid4, seisx_varid, "long_name",
				      strlen("Seismograph along X axis "),
				      "Seismograph along X axis")))
	    ERR(retval);
	if ((retval = nc_put_att_text(ncid4, seisy_varid, "long_name",
				      strlen("Seismograph along Y axis "),
				      "Seismograph along Y axis")))
	    ERR(retval);
	if ((retval = nc_put_att_text(ncid4, seisz_varid, "long_name",
				      strlen("Seismograph along Z axis "),
				      "Seismograph along Z axis")))
	    ERR(retval);
	if ((retval = nc_put_att_text(ncid4, seisxx_varid, "long_name",
				      strlen("Seismograph along XX axis "),
				      "Seismograph along XX axis")))
	    ERR(retval);
	if ((retval = nc_put_att_text(ncid4, seisyy_varid, "long_name",
				      strlen("Seismograph along YY axis "),
				      "Seismograph along YY axis")))
	    ERR(retval);
	if ((retval = nc_put_att_text(ncid4, seiszz_varid, "long_name",
				      strlen("Seismograph along ZZ axis "),
				      "Seismograph along ZZ axis")))
	    ERR(retval);
	if ((retval = nc_put_att_text(ncid4, seisxy_varid, "long_name",
				      strlen("Seismograph along XY axis "),
				      "Seismograph along XY axis")))
	    ERR(retval);
	if ((retval = nc_put_att_text(ncid4, seisyz_varid, "long_name",
				      strlen("Seismograph along YZ axis "),
				      "Seismograph along YZ axis")))
	    ERR(retval);
	if ((retval = nc_put_att_text(ncid4, seisxz_varid, "long_name",
				      strlen("Seismograph along XZ axis "),
				      "Seismograph along XZ axis")))
	    ERR(retval);

	if ((retval =
	     nc_put_att_double(ncid4, seisx_varid, "missing_value",
			       NC_DOUBLE, 1, seisx_missing_value)))
	    ERR(retval);
	if ((retval =
	     nc_put_att_double(ncid4, seisy_varid, "missing_value",
			       NC_DOUBLE, 1, seisy_missing_value)))
	    ERR(retval);
	if ((retval =
	     nc_put_att_double(ncid4, seisz_varid, "missing_value",
			       NC_DOUBLE, 1, seisz_missing_value)))
	    ERR(retval);
	if ((retval =
	     nc_put_att_double(ncid4, seisxx_varid, "missing_value",
			       NC_DOUBLE, 1, seisxx_missing_value)))
	    ERR(retval);
	if ((retval =
	     nc_put_att_double(ncid4, seisyy_varid, "missing_value",
			       NC_DOUBLE, 1, seisyy_missing_value)))
	    ERR(retval);
	if ((retval =
	     nc_put_att_double(ncid4, seiszz_varid, "missing_value",
			       NC_DOUBLE, 1, seiszz_missing_value)))
	    ERR(retval);
	if ((retval =
	     nc_put_att_double(ncid4, seisyz_varid, "missing_value",
			       NC_DOUBLE, 1, seisyz_missing_value)))
	    ERR(retval);
	if ((retval =
	     nc_put_att_double(ncid4, seisxy_varid, "missing_value",
			       NC_DOUBLE, 1, seisxy_missing_value)))
	    ERR(retval);
	if ((retval =
	     nc_put_att_double(ncid4, seisxz_varid, "missing_value",
			       NC_DOUBLE, 1, seisxz_missing_value)))
	    ERR(retval);

	if ((retval = nc_put_att_text(ncid4, seisx_varid, "units",
				      strlen("m/s²"), "m/s²")))
	    ERR(retval);
	if ((retval = nc_put_att_text(ncid4, seisy_varid, "units",
				      strlen("m/s²"), "m/s²")))
	    ERR(retval);
	if ((retval = nc_put_att_text(ncid4, seisz_varid, "units",
				      strlen("m/s²"), "m/s²")))
	    ERR(retval);
	if ((retval = nc_put_att_text(ncid4, seisxx_varid, "units",
				      strlen("m/s²"), "m/s²")))
	    ERR(retval);
	if ((retval = nc_put_att_text(ncid4, seisyy_varid, "units",
				      strlen("m/s²"), "m/s²")))
	    ERR(retval);
	if ((retval = nc_put_att_text(ncid4, seiszz_varid, "units",
				      strlen("m/s²"), "m/s²")))
	    ERR(retval);
	if ((retval = nc_put_att_text(ncid4, seisxy_varid, "units",
				      strlen("m/s²"), "m/s²")))
	    ERR(retval);
	if ((retval = nc_put_att_text(ncid4, seisyz_varid, "units",
				      strlen("m/s²"), "m/s²")))
	    ERR(retval);
	if ((retval = nc_put_att_text(ncid4, seisxz_varid, "units",
				      strlen("m/s²"), "m/s²")))
	    ERR(retval);

	/* Assign  global attributes */
	if ((retval =
	     nc_put_att_text(ncid4, NC_GLOBAL, "Conventions", 6,
			     "COARDS")))
	    ERR(retval);

	sprintf(xcoord, "%f", xobs_out[0]);
	strcat(title, xcoord);
	strcat(title, ", y = ");
	sprintf(ycoord, "%f", yobs_out[0]);
	strcat(title, ycoord);
	sprintf(zcoord, "%f", zobs_out[0]);
	strcat(title, ", z= ");
	strcat(title, zcoord);
	if ((retval =
	     nc_put_att_text(ncid4, NC_GLOBAL, "title", strlen(title),
			     title)))
	    ERR(retval);

	cdf_node_offset[0] = 0;
	if ((retval =
	     nc_put_att_int(ncid4, NC_GLOBAL, "node_offset", NC_INT, 1,
			    cdf_node_offset)))
	    ERR(retval);

	/* End define mode. This tells netCDF we are done defining
	 * metadata. */
	if ((retval = nc_enddef(ncid4)))
	    ERR(retval);
    }

    /*Write the pretend data to the file. */
    if (l == station_step) {
	if ((retval =
	     nc_put_vara_double(ncid4, time_varid, start, count,
				&times[0])))
	    ERR(retval);
	if ((retval =
	     nc_put_vara_double(ncid4, seisx_varid, start, count,
				&seisx_out[0])))
	    ERR(retval);
	if ((retval =
	     nc_put_vara_double(ncid4, seisy_varid, start, count,
				&seisy_out[0])))
	    ERR(retval);
	if ((retval =
	     nc_put_vara_double(ncid4, seisz_varid, start, count,
				&seisz_out[0])))
	    ERR(retval);
	if ((retval =
	     nc_put_vara_double(ncid4, seisxx_varid, start, count,
				&seisxx_out[0])))
	    ERR(retval);
	if ((retval =
	     nc_put_vara_double(ncid4, seisyy_varid, start, count,
				&seisyy_out[0])))
	    ERR(retval);
	if ((retval =
	     nc_put_vara_double(ncid4, seiszz_varid, start, count,
				&seiszz_out[0])))
	    ERR(retval);
	if ((retval =
	     nc_put_vara_double(ncid4, seisxy_varid, start, count,
				&seisxy_out[0])))
	    ERR(retval);
	if ((retval =
	     nc_put_vara_double(ncid4, seisyz_varid, start, count,
				&seisyz_out[0])))
	    ERR(retval);
	if ((retval =
	     nc_put_vara_double(ncid4, seisxz_varid, start, count,
				&seisxz_out[0])))
	    ERR(retval);
    }

    else {
	if ((retval =
	     nc_put_vara_double(ncid4bis, time_varid, start, count,
				&times[l - station_step])))
	    ERR(retval);
	if ((retval =
	     nc_put_vara_double(ncid4bis, seisx_varid, start, count,
				&seisx_out[l - station_step])))
	    ERR(retval);
	if ((retval =
	     nc_put_vara_double(ncid4bis, seisy_varid, start, count,
				&seisy_out[l - station_step])))
	    ERR(retval);
	if ((retval =
	     nc_put_vara_double(ncid4bis, seisz_varid, start, count,
				&seisz_out[l - station_step])))
	    ERR(retval);
	if ((retval =
	     nc_put_vara_double(ncid4bis, seisxx_varid, start, count,
				&seisxx_out[l - station_step])))
	    ERR(retval);
	if ((retval =
	     nc_put_vara_double(ncid4bis, seisyy_varid, start, count,
				&seisyy_out[l - station_step])))
	    ERR(retval);
	if ((retval =
	     nc_put_vara_double(ncid4bis, seiszz_varid, start, count,
				&seiszz_out[l - station_step])))
	    ERR(retval);
	if ((retval =
	     nc_put_vara_double(ncid4bis, seisxy_varid, start, count,
				&seisxy_out[l - station_step])))
	    ERR(retval);
	if ((retval =
	     nc_put_vara_double(ncid4bis, seisyz_varid, start, count,
				&seisyz_out[l - station_step])))
	    ERR(retval);
	if ((retval =
	     nc_put_vara_double(ncid4bis, seisxz_varid, start, count,
				&seisxz_out[l - station_step])))
	    ERR(retval);
    }

    /* Close the file. */
    if (l == station_step) {
	if ((retval = nc_close(ncid4)))
	    ERR(retval);
    } else {
	if ((retval = nc_close(ncid4bis)))
	    ERR(retval);
    }

    return EXIT_SUCCESS;

}

#endif


#if defined (OUTNETCDF) || defined (OUTNEMESIS)

////////////////////////////////////////////////////
//      Write Free surface NETCDF File - NETCDSURF
////////////////////////////////////////////////////

int OutSurfNetcdf(char *flname, int d, int XMIN, int XMAX, int YMIN,
		  int YMAX, int ZMIN, int ZMAX, double ds, double **vx_out,
		  double **vy_out, double **vz_out)
{

/* When we create netCDF variables and dimensions, we get back an
    * ID for each one. */
    int ncid, x_dimid, y_dimid, x_varid, y_varid, z_dimid, z_varid;
    int vx_varid, vy_varid, vz_varid;
    int dimids[3];
    double x_actual_range[2], y_actual_range[2], z_actual_range[2];
    double vx_missing_value[1], vy_missing_value[1], vz_missing_value[1];
    int cdf_node_offset[1];
    int NX, NY;

    double *xs, *ys, zs[1];

    /* This is the data array we will write. */
    double **vxt_out, **vyt_out, **vzt_out;

    /* Error handling and loop index. */
    int retval, x, y, xbis, ybis;

    NX = XMAX - XMIN + 2 * d + 1;
    NY = YMAX - YMIN + 2 * d + 1;
    xs = dvector(0, NX - 1);
    ys = dvector(0, NY - 1);
    vxt_out = dmatrix(0, NY - 1, 0, NX - 1);
    vyt_out = dmatrix(0, NY - 1, 0, NX - 1);
    vzt_out = dmatrix(0, NY - 1, 0, NX - 1);
    zs[0] = 0;

    for (x = XMIN - d + 1; x <= XMAX + d + 1; x++) {
	xbis = x - (XMIN - d + 1);
	xs[xbis] = (x - 1) * ds / 1000;
    }

    for (y = YMIN - d + 1; y <= YMAX + d + 1; y++) {
	ybis = y - (YMIN - d + 1);
	ys[ybis] = (y - 1) * ds / 1000;
    }

/*   for( x = XMIN-d+1; x <= XMAX+d+1  ; x++ ){
      xbis = x-(XMIN-d+1);
      for( y = YMIN-d+1 ; y <= YMAX+d+1; y++ ){
          ybis=y-(YMIN-d+1);
	  vx_out[xbis][ybis] = vx0[x][y][0];
          vy_out[xbis][ybis] = vy0[x][y][0];
          vz_out[xbis][ybis] = vz0[x][y][0];
          }
   }
*/

    /* For multidimensional arrays, the last dimension varies fastest.
       Thus, row-order rather than column order is used for matrices.
       We have to tranpose the matrices. */

    for (y = YMIN - d + 1; y <= YMAX + d + 1; y++) {
	ybis = y - (YMIN - d + 1);
	for (x = XMIN - d + 1; x <= XMAX + d + 1; x++) {
	    xbis = x - (XMIN - d + 1);
	    vxt_out[ybis][xbis] = vx_out[x][y];
	    vyt_out[ybis][xbis] = vy_out[x][y];
	    vzt_out[ybis][xbis] = vz_out[x][y];
	}
    }

    /* Create the file. */
    if ((retval = nc_create(flname, NC_CLOBBER, &ncid)))
	ERR(retval);

    /* Define the dimensions. NetCDF will hand back an ID for each. */
    if ((retval = nc_def_dim(ncid, "y", NY, &y_dimid)))
	ERR(retval);
    if ((retval = nc_def_dim(ncid, "x", NX, &x_dimid)))
	ERR(retval);
    if ((retval = nc_def_dim(ncid, "z", 1, &z_dimid)))
	ERR(retval);

    /* The dimids array is used to pass the IDs of the dimensions of
     * the variable. */
    dimids[0] = y_dimid;
    dimids[1] = x_dimid;
    dimids[2] = z_dimid;

    /* Define the variables. The type of the variable in this case is
     * NC_DOUBLE. */
    if ((retval = nc_def_var(ncid, "x", NC_DOUBLE, 1, &x_dimid, &x_varid)))
	ERR(retval);
    if ((retval = nc_def_var(ncid, "y", NC_DOUBLE, 1, &y_dimid, &y_varid)))
	ERR(retval);
    if ((retval = nc_def_var(ncid, "z", NC_DOUBLE, 1, &z_dimid, &z_varid)))
	ERR(retval);
    if ((retval = nc_def_var(ncid, "vx", NC_DOUBLE, 2, dimids, &vx_varid)))
	ERR(retval);
    if ((retval = nc_def_var(ncid, "vy", NC_DOUBLE, 2, dimids, &vy_varid)))
	ERR(retval);
    if ((retval = nc_def_var(ncid, "vz", NC_DOUBLE, 2, dimids, &vz_varid)))
	ERR(retval);

    /* Assign attributes to variables. */
    if ((retval = nc_put_att_text(ncid, x_varid, "long_name",
				  strlen("X"), "X")))
	ERR(retval);
    if ((retval = nc_put_att_text(ncid, x_varid, "units",
				  strlen("meter"), "meter")))
	ERR(retval);
    x_actual_range[0] = ((XMIN - d) * ds / 1000);
    x_actual_range[1] = ((XMAX + d) * ds / 1000);
    if ((retval = nc_put_att_double(ncid, x_varid, "actual_range",
				    NC_DOUBLE, 2, &x_actual_range[0])))
	ERR(retval);

    if ((retval = nc_put_att_text(ncid, y_varid, "long_name",
				  strlen("Y"), "Y")))
	ERR(retval);
    if ((retval = nc_put_att_text(ncid, y_varid, "units",
				  strlen("meter"), "meter")))
	ERR(retval);
    y_actual_range[0] = (YMIN - d) * ds / 1000;
    y_actual_range[1] = (YMAX + d) * ds / 1000;
    if ((retval = nc_put_att_double(ncid, y_varid, "actual_range",
				    NC_DOUBLE, 2, &y_actual_range[0])))
	ERR(retval);

    if ((retval = nc_put_att_text(ncid, z_varid, "long_name",
				  strlen("Z"), "Z")))
	ERR(retval);
    if ((retval = nc_put_att_text(ncid, z_varid, "units",
				  strlen("meter"), "meter")))
	ERR(retval);
    z_actual_range[0] = ((ZMIN - d) * ds / 1000);
    z_actual_range[1] = 0;
    if ((retval = nc_put_att_double(ncid, z_varid, "actual_range",
				    NC_DOUBLE, 2, &z_actual_range[0])))
	ERR(retval);
    if ((retval = nc_put_att_text(ncid, vx_varid, "long_name",
				  strlen("vitessex"), "vitessex")))
	ERR(retval);
    if ((retval = nc_put_att_text(ncid, vx_varid, "units",
				  strlen("meter/second"), "meter/second")))
	ERR(retval);
    if ((retval = nc_put_att_double(ncid, vx_varid, "missing_value",
				    NC_DOUBLE, 1, vx_missing_value)))
	ERR(retval);

    if ((retval = nc_put_att_text(ncid, vy_varid, "long_name",
				  strlen("vitessey"), "vitessey")))
	ERR(retval);
    if ((retval = nc_put_att_text(ncid, vy_varid, "units",
				  strlen("meter/second"), "meter/second")))
	ERR(retval);
    if ((retval = nc_put_att_double(ncid, vy_varid, "missing_value",
				    NC_DOUBLE, 1, vy_missing_value)))
	ERR(retval);
    if ((retval = nc_put_att_text(ncid, vz_varid, "long_name",
				  strlen("vitessez"), "vitessez")))
	ERR(retval);
    if ((retval = nc_put_att_text(ncid, vz_varid, "units",
				  strlen("meter/second"), "meter/second")))
	ERR(retval);
    if ((retval = nc_put_att_double(ncid, vz_varid, "missing_value",
				    NC_DOUBLE, 1, vz_missing_value)))
	ERR(retval);

    /* Assign  global attributes */
    if ((retval =
	 nc_put_att_text(ncid, NC_GLOBAL, "Conventions", 6, "COARDS")))
	ERR(retval);
    if ((retval =
	 nc_put_att_text(ncid, NC_GLOBAL, "title", 10, "v[x][y][0]")))
	ERR(retval);
    cdf_node_offset[0] = 0;
    if ((retval =
	 nc_put_att_int(ncid, NC_GLOBAL, "node_offset", NC_INT, 1,
			cdf_node_offset)))
	ERR(retval);

    /* End define mode. This tells netCDF we are done defining
     * metadata. */
    if ((retval = nc_enddef(ncid)))
	ERR(retval);

    /*Write the pretend data to the file. */

    if ((retval = nc_put_var_double(ncid, x_varid, &xs[0])))
	ERR(retval);
    if ((retval = nc_put_var_double(ncid, y_varid, &ys[0])))
	ERR(retval);
    if ((retval = nc_put_var_double(ncid, z_varid, &zs[0])))
	ERR(retval);
    if ((retval = nc_put_var_double(ncid, vx_varid, &vxt_out[0][0])))
	ERR(retval);
    if ((retval = nc_put_var_double(ncid, vy_varid, &vyt_out[0][0])))
	ERR(retval);
    if ((retval = nc_put_var_double(ncid, vz_varid, &vzt_out[0][0])))
	ERR(retval);


    /* Close the file. */
    if ((retval = nc_close(ncid)))
	ERR(retval);

}



#endif


int ERR(int retval)
{
}

////////////////////////////////////
//      Write Geological model
///////////////////////////////////


int OutGeol(struct MEDIUM MDM, struct OUTPUTS OUT, struct PARAMETERS PRM,
	    const char *flname)
{
    int i, j, k, imp, jmp;
    int kmax;
    int med;
    char flname5[STRL], flnameInfo[STRL];
    char number[STRL];
    FILE *fp5 = NULL;
    FILE *fpInfo = NULL;
    /* dummies */
    double xdum, ydum, zdum, vpdum;
    /* mapping */
    const double X0 = PRM.x0;
    const double Y0 = PRM.y0;
    const double Z0 = PRM.z0;
    const double DS = PRM.ds;

    if (model == GEOLOGICAL) {	/* the geological model is read in a file */

	strcpy(flname5, PRM.dir);
	strcat(flname5, flname);
	sprintf(number, "%2.2i%2.2i", PRM.coords[0], PRM.coords[1]);
	strcat(flname5, number);

	strcpy(flnameInfo, flname5);
	strcat(flnameInfo, "_info.txt");
	fpInfo = fopen(flnameInfo, "w");
	/* give infos for 3D viewers */
	fprintf(fpInfo, "%s %d %d %d %d\n", flname5,
		PRM.me, PRM.mpmx + 2 + 1 + 1, PRM.mpmy + 2 + 1 + 1,
		PRM.zMax0 - PRM.zMin + PRM.delta + 1);

	fclose(fpInfo);

	fp5 = fopen(flname5, "wb");
	kmax = 0;
	kmax = PRM.zMax0;

	for (k = PRM.zMin - PRM.delta; k <= kmax; k++) {
	    zdum = Z0 + (k - 1) * DS;

	    for (jmp = -1; jmp <= PRM.mpmy + 2; jmp++) {
		j = PRM.jmp2j_array[jmp];
		ydum = Y0 + (j - 1) * DS;

		for (imp = -1; imp <= PRM.mpmx + 2; imp++) {
		    i = PRM.imp2i_array[imp];
		    xdum = X0 + (i - 1) * DS;

		    //med = MDM.imed[imp][jmp][k];
        med = i3access(MDM.imed, -1, PRM.mpmx + 2, -1, PRM.mpmy + 2, PRM.zMin - PRM.delta, PRM.zMax0, imp, jmp, k);
		    vpdum = (double) (med);
		    fwrite(&vpdum, sizeof(double), 1, fp5);
		    vpdum =
			RhoMuKap2Vp(MDM.rho0[med], MDM.mu0[med],
				    MDM.kap0[med]);

		    fwrite(&vpdum, sizeof(double), 1, fp5);

		    vpdum = RhoMu2Vs(MDM.rho0[med], MDM.mu0[med]);
		    fwrite(&vpdum, sizeof(double), 1, fp5);
		}		/* end of imp */
	    }			/* end of jmp */
	}			/* end of k */
	fclose(fp5);

    }				/* end of model = 1 */

    return (EXIT_SUCCESS);
}				/* end outGeol */





///////////////////////////////////////////////
//      Write information related to the medium
///////////////////////////////////////////////

int PrintInfoMedium(struct ANELASTICITY ANL, struct MEDIUM MDM,
		    struct SOURCE SRC, struct PARAMETERS PRM)
{
    int ly = 0;
    double zdum = 0.;
    double mu = 0., kap = 0., rho = 0.;
    int ly0 = 0, ly2 = 0;

    if (PRM.me == 0) {
	/* classic layers */
	printf("%3s %8s %8s %8s %8s", "ly", "laydep", "rho0", "mu0",
	       "kap0");
	if (ANLmethod == ANOTHER) {
	    printf(" %8s", "amp");
	} else if (ANLmethod == KRISTEKandMOCZO
		   || ANLmethod == DAYandBRADLEY) {
	    printf(" %8s %8s", "Qp", "Qs");
	}
	printf("\n");

	for (ly = 0; ly < MDM.nLayer; ly++) {
//FD
//      printf("ZDUM %d %e %e \n",ly,zdum,MDM.laydep[ly]);
//      zdum=MDM.laydep[ly];
	    rho = MDM.rho0[ly];
	    mu = MDM.mu0[ly];
	    kap = MDM.kap0[ly];

	    printf("%3i %8.2e %8.2e %8.2e %8.2e ", ly + 1, zdum, rho, mu,
		   kap);
	    if (ANLmethod == ANOTHER) {
		printf(" %8.2e", ANL.amp[ly]);
	    } else if (ANLmethod == KRISTEKandMOCZO
		       || ANLmethod == DAYandBRADLEY) {
		printf(" %8.2e %8.2e", ANL.Qp0[ly], ANL.Qs0[ly]);
	    }
	    printf("\n");
	}

	if (model == LAYER) {	/* print z+ds/2 infos */
	    const double DS = PRM.ds;
	    printf("%3s %8s %8s %8s %8s", "ly2", "laydep2", "rho2", "mu2",
		   "kap2");
	    if (ANLmethod == ANOTHER) {
		printf(" %8s", "amp2");
	    } else if (ANLmethod == KRISTEKandMOCZO
		       || ANLmethod == DAYandBRADLEY) {
		printf(" %8s %8s", "Qp2", "Qs2");
	    }
	    printf("\n");

	    for (ly2 = 0; ly2 < MDM.nLayer2; ly2++) {
		zdum = MDM.laydep2[ly2];
		rho = MDM.rho2[ly2];
		mu = MDM.mu2[ly2];
		kap = MDM.kap2[ly2];

		printf("%3i %8.2e %8.2e %8.2e %8.2e ", ly2 + 1, zdum, rho,
		       mu, kap);
		if (ANLmethod == ANOTHER) {
		    printf(" %8.2e", ANL.q2[ly2]);
		} else if (ANLmethod == KRISTEKandMOCZO
			   || ANLmethod == DAYandBRADLEY) {
		    printf(" %8.2e %8.2e", ANL.Qp2[ly2], ANL.Qs2[ly2]);
		}
		printf("\n");
	    }			/* end for */

	}			/* end if model */

    }				/* end if ME=0 */

    return EXIT_SUCCESS;
}




/////////////////////////////////
//      Helper functions
/////////////////////////////////

char *FindField(const char *cs, const char *ct)
{
    int indexB = -1;
    int indexE = -1;
    size_t sizect = strlen(ct);
    char *strB = NULL;
    char *strE = NULL;
    char *out = NULL;
    char *ctBegin = NULL;
    char *ctEnd = NULL;

    ctBegin = malloc(strlen(ct) + 3);
    strcpy(ctBegin, "<");
    strcat(ctBegin, ct);
    strcat(ctBegin, ">");

    ctEnd = malloc(strlen(ct) + 4);
    strcpy(ctEnd, "</");
    strcat(ctEnd, ct);
    strcat(ctEnd, ">");

    if (cs != NULL && ct != NULL) {
	strB = strstr(cs, ctBegin);

	if (strB != NULL)
	    indexB = strB - cs + strlen(ctBegin);
	else {
	    fprintf(stderr, "cannot find : %s in %s", ctBegin, cs);
	    exit(EXIT_FAILURE);
	}

	strE = strstr(cs, ctEnd);
	if (strE != NULL)
	    indexE = strE - cs;
	else {
	    fprintf(stderr, "cannot find : %s in %s", ctEnd, cs);
	    exit(EXIT_FAILURE);
	}

    } else {
	assert(EXIT_SUCCESS == EXIT_FAILURE);
	return NULL;
    }
    out = malloc((indexE - indexB + 1) * sizeof(char));

    strncpy(out, &cs[indexB], (indexE - indexB + 1) * sizeof(char));
    out[indexE - indexB] = '\0';

    free(ctBegin);
    free(ctEnd);

    return out;
}

int FindInt(const char *strIn, const char *srch)
{				/* work */
    int out = 0;
    char *cur = NULL;
    cur = FindField(strIn, srch);
    if (cur != NULL) {
	out = atoi(cur);
    } else {
	fprintf(stderr, "Error in '%s' in file '%s', line %i", __func__,
		__FILE__, __LINE__);
	assert(EXIT_SUCCESS == EXIT_FAILURE);
	out = -1;
    }
    free(cur);
    return out;
}

double FindDble(const char *strIn, const char *srch)
{				/* work */
    double out = 0.;
    char *cur = NULL;
    cur = FindField(strIn, srch);
    if (cur != NULL) {
	out = atof(cur);
    } else {
	fprintf(stderr, "Error in '%s' in file '%s', line %i", __func__,
		__FILE__, __LINE__);
	assert(EXIT_SUCCESS == EXIT_FAILURE);
	out = -1.;
    }
    free(cur);
    return out;
}

char *File2str(const char *flname)
{
    FILE *fp = NULL;
    char *out = NULL;
    int size = 0;

    char cur[STRL];

    fp = fopen(flname, "r");

    if (fp == NULL) {
	fprintf(stderr, "error when opening file: %s", flname);
	exit(EXIT_FAILURE);
    }

    int i = 1;
    size = 1;
    out = malloc(i * STRL * sizeof(char));
    while (fscanf(fp, "%s", &cur) != EOF) {
	size += strlen(cur);
	if (size >= STRL - 2) {
	    i = i + 1;
	    out = realloc(out, i * STRL * sizeof(char));
	    if (out == NULL) {
		perror("Error while converting file to string");
	    }
	    size = strlen(out) % STRL + 1;
	}
	strncat(out, cur, strlen(cur));
    }
    fclose(fp);

    return out;
}


void force_big_endian(unsigned char *bytes)
{
    static int doneTest = 0;
    static int shouldSwap = 0;

    /* declare an int */
    int tmp1 = 1;

    /* the char pointer points to the tmp1 variable
       // so tmp2+1, +2, +3 points to the bytes of tmp1 */
    unsigned char *tmp2 = (unsigned char *) &tmp1;

    /* look if the endianness
       // of the machine has already been
       // tested */
    if (!doneTest) {
	if (*tmp2 != 0)
	    shouldSwap = 1;
	doneTest = 1;
    }

    if (shouldSwap) {
	unsigned char tmp;
	tmp = bytes[0];
	bytes[0] = bytes[3];
	bytes[3] = tmp;
	tmp = bytes[1];
	bytes[1] = bytes[2];
	bytes[2] = tmp;
    }
}

void write_float(FILE * fp, float val)
{
    force_big_endian((unsigned char *) &val);
    fwrite(&val, sizeof(float), 1, fp);
}
