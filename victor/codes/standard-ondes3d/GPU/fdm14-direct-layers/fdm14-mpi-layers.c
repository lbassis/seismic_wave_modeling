/* last revision : April 21st 2009 */
/* CPML from Komatitsch */
/* Parallelisation by Fabrice Dupros */
/* Free surface at k = 1 (stress imaging method) */
/* Moment force given by name_file.hist and name_file.map */
/* Read "3DVEL_ChuReg.txt" for 3D structure */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "nr.h"
#include "nrutil.h"
#include "myfdm5-3.h"
#include "mpi.h"
#include <assert.h>

#if  (TAU)
  #include <TAU.h>
#endif

#if ( TAUGLOBAL)
  #include <TAU.h>
#endif

#if (MISS)
  #include "papi.h"
#endif

#if (FLOPS)
  #include "papi.h"
#endif

#define PRM "./DATA/essai.prm"
#define TOPOLOGIE  "topologie.in"
#define BENCH  "bench.out"

#define delta 10
#define reflect 0.001
#define f0 1

#define SURFACE_STEP 25000

#define un 1.0

                /* Beginning of main */

int main ( int argc, char **argv )
{

/* Model parameters */

int     NDIM, i0, j0, k0, TMAX,
    XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX;
double  ds, dt;



double	*laydep, *vp0, *vs0, *rho0,	*q0;
int NLAYER,ly;
int     id, il, ix, idum, jdum, ix0, ixdum;
long    n1, n2;
/* end(HA) */


/* Source : position */

int     is, n, iw, ISRC, ixhypo0, iyhypo0, izhypo0;
int     *insrc, *ixhypo, *iyhypo, *izhypo;
double  xhypo0, yhypo0, zhypo0, weight;
double  *xhypo, *yhypo, *zhypo, *xweight, *yweight, *zweight;

/* Source : values */

int     IDUR;
double  pxx, pyy, pzz, pxy, pyz, pxz, dsbiem, dtbiem, mo, mw;
double  *strike, *dip, *rake, *slip;
double  **vel;

/* Velocity and stress */

double  ***vx0, ***vy0, ***vz0,
    ***txx0, ***tyy0, ***tzz0, ***txy0, ***txz0, ***tyz0;

/* Moment force */

double  ***fx, ***fy, ***fz;

/* Material parameters */

double  rhoxy, rhoxz,
    vpxy, vpxz, vpx, vpy, vpz, vpxyz,
	muxy, muxz, mux, muy, muz, muxyz,
	lamxy, lamxz, lamx, lamy, lamz, lamxyz;
double  ***rho, ***mu, ***lam, ***vp, ***vs;

/* Output */

int     ir, IOBS;
int     *nobs, *ista, *ixobs, *iyobs, *izobs;
double  w1, w2, w3;
double 	*xobs, *yobs, *zobs, *xobswt, *yobswt, *zobswt;
double  **seisx, **seisy, **seisz, **seisxx, **seisyy, **seiszz, **seisxy, **seisxz, **seisyz;

/* Memory variables for the CPML */

double  *phivxx, *phivxy, *phivxz, *phivyx, *phivyy, *phivyz, *phivzx, *phivzy, *phivzz,
	*phitxxx, *phitxyy, *phitxzz, *phitxyx, *phityyy, *phityzz, *phitxzx, *phityzy, *phitzzz;

/* Other variables for the CPML */

double  dump0, alpha0, kappa0, NPOWER, fd, phixdum, phiydum, phizdum,
    xoriginleft, xoriginright, yoriginfront, yoriginback, zoriginbottom, zorigintop,
    xval, yval, zval, abscissa_in_PML, abscissa_normalized;
double	*dumpx, *dumpy, *dumpz, *dumpx2, *dumpy2, *dumpz2,
    *alphax, *alphay, *alphaz, *alphax2, *alphay2, *alphaz2,
    *kappax, *kappay, *kappaz, *kappax2, *kappay2, *kappaz2;
long	npml, npmlv, npmlt;

/* File names */

char 	flname[50], number[5], flname1[80], flname2[80], flname3[80],
	flname4[80], flname6[80],
	outdir[50], srcfile1[80], srcfile2[80], srcfile3[80], buf[256],
	char1[30] = "surfacexy",
	char2[30] = "surfacexz",
	char3[30] = "surfaceyz",
	char4[30] = "obs",
	char6[30] = "hoge";

/* Files */

FILE * 	fp1;
FILE *  fp2;
FILE *  fp3;
FILE *  fp4;
FILE *  fp5;
FILE * 	fp6;
FILE *  fp_in0;
FILE *  fp_in1;
FILE *  fp_in2;
FILE *	fp_in3;
FILE *	fp_in4; /* added by HA */
FILE *	fp_in4b; /* added by HA */
FILE *  fp_in5;

/* Other variables */

int     i, j, k, l, l1, it;
double  pi, time, xdum, ydum, zdum, h, b1, b2;

/* Variables for the communication between the CPUs */

double  *sxbuf, *sybuf, *szbuf, *rxbuf, *rybuf, *rzbuf,
   *sxxbuf, *syybuf, *szzbuf, *sxybuf, *syzbuf, *sxzbuf,
   *rxxbuf, *ryybuf, *rzzbuf, *rxybuf, *ryzbuf, *rxzbuf;

double  *sxbuf2, *sybuf2, *szbuf2, *rxbuf2, *rybuf2, *rzbuf2,
   *sxxbuf2, *syybuf2, *szzbuf2, *sxybuf2, *syzbuf2, *sxzbuf2,
   *rxxbuf2, *ryybuf2, *rzzbuf2, *rxybuf2, *ryzbuf2, *rxzbuf2 ;

double  *sxbuf3, *sybuf3, *szbuf3, *rxbuf3, *rybuf3, *rzbuf3,
   *sxxbuf3, *syybuf3, *szzbuf3, *sxybuf3, *syzbuf3, *sxzbuf3,
   *rxxbuf3, *ryybuf3, *rzzbuf3, *rxybuf3, *ryzbuf3, *rxzbuf3;

double  *sxbuf4, *sybuf4, *szbuf4, *rxbuf4, *rybuf4, *rzbuf4,
   *sxxbuf4, *syybuf4, *szzbuf4, *sxybuf4, *syzbuf4, *sxzbuf4,
   *rxxbuf4, *ryybuf4, *rzzbuf4, *rxybuf4, *ryzbuf4, *rxzbuf4 ;

/* Other variables for the parallelisation */

double  ***seis_output;
int     **mapping_seis;

int     test_recv1, test_recv2, test_recv3;

double  *Vxtemp1, *Vytemp1, *Vztemp1;

double  **Vxglobal, **Vyglobal, **Vzglobal;

double  mpmx_tmp1;
double  mpmy_tmp1;

int     idebut, jdebut;

int     *mpmx_tab;
int     *mpmy_tab;

int     px, py;

int     total_prec_x, mpmx0;
int     total_prec_y, mpmy0;

int     *i2imp_array;
int     *imp2i_array;
int     *i2icpu_array;

int     *j2jmp_array;
int     *jmp2j_array;
int     *j2jcpu_array;

char	pname[MPI_MAX_PROCESSOR_NAME];

MPI_Status status;
MPI_Request req;
MPI_Comm comm2d, rc;

int     coords[2], proc_coords[2], coords_global[2][1024];
int     i1, i2;

int     nnord, nsud, nest, nouest;

MPI_Request sendreq[20];
MPI_Request recvreq[20];
MPI_Request sendreq2[20];
MPI_Request recvreq2[20];
MPI_Request sendreq3[20];
MPI_Request recvreq3[20];
MPI_Request sendreq4[20];
MPI_Request recvreq4[20];

MPI_Status recvstatus[20];
MPI_Status sendstatus[20];
MPI_Status recvstatus2[20];
MPI_Status sendstatus2[20];
MPI_Status recvstatus3[20];
MPI_Status sendstatus3[20];
MPI_Status recvstatus4[20];
MPI_Status sendstatus4[20];

int     np, resultlength, imp, jmp, imp_tmp, jmp_tmp, icpu, jcpu, nmaxx, nmaxy, imode;
int     mpmx_debut, mpmx_fin;
int     mpmy_debut, mpmy_fin;
int     mpmx, mpmx_int, mpmx_bord;
int     mpmy, mpmy_int, mpmy_bord;
int     jcpu1, jcpu2, jcpu3;
int     jmp1, jmp2, jmp3;

int     test_size;
int     difference;

int     mpmx_max, mpmy_max;

                /* PAPI */

#if (PAPI)
  float   real_time, proc_time, mflops;
  long_long flpops;
  double  perc;
  float   ireal_time, iproc_time, imflops;
  long_long iflpops;
  int     retval;
  int     EventSet = PAPI_NULL;
  int     EventCode ;
  long_long values[3];
#endif

                /* Timing */

double  timing1, timing2, timing3, timing4, timing_comm1, timing_comm2;
double  timing_bc1, timing_bc2, timing_total, timing_sum1, timing_sum2;
double  timing_bc1_max, timing_bc1_min, timing_bc2_min, timing_bc2_max;
double  timing_pml, timing_ds4, timing_dt4;

double  timing_bc_max, timing_bc_min, timing_comm_min, timing_comm_max, timing_bc_total, timing_comm_total;

double  tmp1;
double  tmp2;

int     my_rank = 0;

                /* Beginning of the program */
                /* Initialization for MPI */

  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &np);

  MPI_Comm_dup (MPI_COMM_WORLD, &comm2d);
  MPI_Comm_rank (comm2d, &my_rank);

  fp_in5 = fopen (TOPOLOGIE, "r");

  if ( fp_in5 == NULL ){
    perror ("failed at fopen 5");
    exit(1);
  }

  fscanf ( fp_in5, "%d %d", &px, &py);

  if ( np != (px*py) ){
	printf ("Mismatch entre le nombre de processeurs demandes et la topologie MPI\n");
	printf ("%d %d %d\n", px, py, np);
	printf ("Arret du code");
	exit(0);
  }

  fclose (fp_in5);

  coords[1] = my_rank/px;
  coords[0] = my_rank -coords[1]*px;

  nnord = my_rank + px;
  nsud =  my_rank - px;

  nest = my_rank + 1;
  nouest = my_rank - 1;

  if ( coords[1] == 0 ) nsud = -1 ;
  if ( coords[1] == py-1 ) nnord = -1 ;

  if ( coords[0] == 0 ) nouest = -1 ;
  if ( coords[0] == px-1 ) nest = -1 ;

  MPI_Get_processor_name (pname, &resultlength);

  MPI_Barrier (comm2d);

  if ( my_rank == 0 ){
    printf ("Topologie\n");
  }

  printf ("me:%d - est:%d - ouest:%d - nord:%d - sud:%d \n", my_rank, nest, nouest, nnord, nsud);
  printf ("me:%d - coords(0):%d - coords(1):%d \n", my_rank, coords[0], coords[1]);

  MPI_Barrier (comm2d);

                /* Initializations */

  NPOWER = 2.;

  timing_comm1 = 0.;
  timing_comm2 = 0.;
  timing_bc1 = 0.;
  timing_bc2 = 0.;
  timing_total = 0.;
  timing_pml = 0.;
  timing_dt4 = 0.;
  timing_ds4 = 0.;

                /* TAU */

  #if (TAU)
    TAU_PROFILE_TIMER (pml_sig, "pml_sig", "void (int,char **)", TAU_USER);
    TAU_PROFILE_TIMER (free_surface_sig," free_surface_sig", "void (int,char **)", TAU_USER);
    TAU_PROFILE_TIMER (interior_sig, "interior_sig", "void (int,char **)", TAU_USER);
    TAU_PROFILE_TIMER (exchange_sig, "exchange_sig", "void (int,char **)", TAU_USER);

    TAU_PROFILE_TIMER (pml_vit, "pml_vit", "void (int,char **)", TAU_USER);
    TAU_PROFILE_TIMER (free_surface_vit, "free_surface_vit", "void (int,char **)", TAU_USER);
    TAU_PROFILE_TIMER (interior_vit, "interior_vit", "void (int,char **)", TAU_USER);
    TAU_PROFILE_TIMER (exchange_vit, "exchange_vit", "void (int,char **)", TAU_USER);

    TAU_PROFILE_SET_NODE (my_rank)
  #endif

  #if (TAUGLOBAL)
    TAU_PROFILE_TIMER (compute_sig, "compute_sig", "void (int,char **)", TAU_USER);
    TAU_PROFILE_TIMER (exchange_sig, "exchange_sig", "void (int,char **)", TAU_USER);

    TAU_PROFILE_TIMER (compute_vit, "compute_vit", "void (int,char **)", TAU_USER);
    TAU_PROFILE_TIMER (exchange_vit, "exchange_vit", "void (int,char **)", TAU_USER);

    TAU_PROFILE_SET_NODE (my_rank)
  #endif

  #if (MISS)

                /* PAPI */

    if ( (retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT )
      printf ("ERROR Init\n");
    if ( (retval = PAPI_create_eventset(&EventSet)) != PAPI_OK )
      printf ("ERROR create\n");
    if ( (retval = PAPI_add_event(EventSet, PAPI_L3_DCA)) != PAPI_OK )
      printf ("ERROR add\n");
    if ( (retval = PAPI_add_event(EventSet, PAPI_L3_DCH)) != PAPI_OK )
      printf ("ERROR add\n");
    if ( (retval = PAPI_add_event(EventSet, PAPI_TOT_CYC)) != PAPI_OK )
      printf ("ERROR add\n");

  #endif

                /* We read the model parameters in meter, second */

  fp_in0 = fopen ( PRM, "r");

  if ( fp_in0 == NULL ){
    perror ("failed at fopen 0");
    exit(1);
  }

  fscanf ( fp_in0, "%d", &NDIM);
  fscanf ( fp_in0, "%d %d", &XMIN, &XMAX);
  fscanf ( fp_in0, "%d %d", &YMIN, &YMAX);
  fscanf ( fp_in0, "%d", &ZMIN);
  fscanf ( fp_in0, "%d", &TMAX);
  fscanf ( fp_in0, "%d %d", &i0, &j0);
  fscanf ( fp_in0, "%s", outdir);
  fscanf ( fp_in0, "%s", srcfile1);
  fscanf ( fp_in0, "%s", srcfile2);
  fscanf ( fp_in0, "%s", srcfile3);
  fscanf ( fp_in0, "%lf %lf %lf", &ds, &dt, &fd);
  fscanf ( fp_in0, "%d", &NLAYER);

  laydep = dvector(0, NLAYER-1);
  vp0 = dvector(0, NLAYER-1);
  vs0 = dvector(0, NLAYER-1);
  rho0 = dvector(0, NLAYER-1);
  q0  = dvector(0, NLAYER-1);

  for ( ly = 0; ly < NLAYER; ly++){
    fscanf ( fp_in0, "%lf %lf %lf %lf %lf",
	&laydep[ly], &vp0[ly], &vs0[ly], &rho0[ly], &q0[ly]);
  if ( my_rank == 0 ){
    printf ("%lf %lf %lf %lf %lf\n",
	laydep[ly], vp0[ly], vs0[ly], rho0[ly], q0[ly]);
    }
  }

  fclose (fp_in0);

  pi = acos(-1.0);

  dump0 = - (NPOWER + 1) * log(reflect) / (2.0 * delta * ds);
  alpha0 = fd*pi; /* alpha0 = pi*fd where fd is the dominant frequency of the source */
  kappa0 = 1.0;

  if ( my_rank == 0 ){

    printf ("\nDimension of FDM order ... %i\n", NDIM );
    printf ("\nParameter File ... %s\n", PRM);
    printf ("Source Model based on ... %s\n", srcfile1 );
    printf ("Rupture History from ... %s\n", srcfile2 );
    printf ("Station Position at ... %s\n", srcfile3 );
    printf ("Output directory ... %s\n", outdir );
    printf ("\nspatial grid ds = %f[m]\n", ds);
    printf ("time step dt = %f[s]\n", dt);
    printf ("\nVisualisation of plane (y,z) at X = %7.2f [km]\n", (i0-1)*ds/1000.);
    printf ("Visualisation of plane (x,z) at Y = %7.2f [km]\n", (j0-1)*ds/1000.);
    printf ("Visualisation of plane (x,y) at Z = %7.2f [km]\n", 0.);
    printf ("\nModel Region (%i:%i, %i:%i, %i:%i)\n",
      XMIN, XMAX, YMIN, YMAX, ZMIN, 0);

    printf ("        (%7.2f:%7.2f, %7.2f:%7.2f, %7.2f:%7.2f) [km]\n",
        XMIN*ds/1000., XMAX*ds/1000., YMIN*ds/1000.,
        YMAX*ds/1000., ZMIN*ds/1000., 0.);

    printf ("\nCPML absorbing boundary, dumping %f, width %d, ratio %f, frequency %f\n",
	    dump0, delta, reflect, fd);

    printf ("\nstructure model\n");
    printf ("will be read by a specific file later.\n");

    char cmd[150];
    sprintf(cmd, "test -d %s || mkdir %s",outdir,outdir);
    system(cmd);

  } /* end of my_rank = 0 */

				/* Reading the source position */

  strcpy (flname, srcfile1);
  fp_in1 = fopen (flname, "r");

  if ( fp_in1 == NULL ){
    perror ("failed at fopen 1");
    exit(1);
  }

  fscanf ( fp_in1, "%d", &ISRC );
  fscanf ( fp_in1, "%lf %lf %lf", &xhypo0, &yhypo0, &zhypo0);

  ixhypo  = ivector (0, ISRC-1);
  iyhypo  = ivector (0, ISRC-1);
  izhypo  = ivector (0, ISRC-1);
  insrc   = ivector (0, ISRC-1);
  xhypo   = dvector (0, ISRC-1);
  yhypo   = dvector (0, ISRC-1);
  zhypo   = dvector (0, ISRC-1);
  strike  = dvector (0, ISRC-1);
  dip     = dvector (0, ISRC-1);
  rake    = dvector (0, ISRC-1);
  slip	  = dvector (0, ISRC-1);
  xweight = dvector (0, ISRC-1);
  yweight = dvector (0, ISRC-1);
  zweight = dvector (0, ISRC-1);

  for ( is = 0;  is < ISRC; is++)
    fscanf ( fp_in1, "%d %lf %lf %lf", &n, &xhypo[is], &yhypo[is], &zhypo[is]);

  fclose (fp_in1);

  if ( my_rank == 0 ){
    printf("\nNumber of sources : %d\n", ISRC);
    printf("Hypocenter ... (%f, %f, %f)\n", xhypo0, yhypo0, zhypo0);
  }

  ixhypo0 = (int)(xhypo0/ds)+1;
  iyhypo0 = (int)(yhypo0/ds)+1;
  izhypo0 = (int)(zhypo0/ds)+1;

  if ( xhypo0 < 0.0 && xhypo0/ds != (int)(xhypo0/ds) ) ixhypo0 = ixhypo0 - 1;
  if ( yhypo0 < 0.0 && yhypo0/ds != (int)(yhypo0/ds) ) iyhypo0 = iyhypo0 - 1;
  if ( zhypo0 < 0.0 && zhypo0/ds != (int)(zhypo0/ds) ) izhypo0 = izhypo0 - 1;

  if ( my_rank == 0 ){
    printf (".............. (%i, %i, %i)\n", ixhypo0, iyhypo0, izhypo0);
  }

  for ( is = 0; is < ISRC; is++){

    ixhypo[is] = (int)(xhypo[is]/ds)+1;
    iyhypo[is] = (int)(yhypo[is]/ds)+1;
    izhypo[is] = (int)(zhypo[is]/ds)+1;

    if( xhypo[is] < 0.0 && xhypo[is]/ds != (int)(xhypo[is]/ds) ) ixhypo[is] = ixhypo[is]-1;
    if( yhypo[is] < 0.0 && yhypo[is]/ds != (int)(yhypo[is]/ds) ) iyhypo[is] = iyhypo[is]-1;
    if( zhypo[is] < 0.0 && zhypo[is]/ds != (int)(zhypo[is]/ds) ) izhypo[is] = izhypo[is]-1;

    xweight[is] = xhypo[is]/ds - ixhypo[is]+1;
	yweight[is] = yhypo[is]/ds - iyhypo[is]+1;
    zweight[is] = zhypo[is]/ds - izhypo[is]+1;


    if ( my_rank == 0 ){
      printf ("Source %i .... (%f, %f, %f)\n", is+1, xhypo[is], yhypo[is], zhypo[is] );
      printf (".............. (%i, %i, %i)\n", ixhypo[is], iyhypo[is], izhypo[is]);
    }

    insrc[is] = 1;

    if ( ixhypo[is] > XMAX+1 || ixhypo[is] < XMIN+1 || iyhypo[is] > YMAX+1 ||
         iyhypo[is] < YMIN+1 || izhypo[is] > 1 || izhypo[is] < ZMIN+1 ){

      if ( my_rank == 0 ){
        printf("Warning: Source %d (%d, %d, %d) (%f8.2, %f8.2, %f8.2)km is not included \n",
            is+1, ixhypo[is], iyhypo[is], izhypo[is],
            xhypo[is]/1000., yhypo[is]/1000., zhypo[is]/1000.);
      }
	  insrc[is] = 0;
    }

  } /* end of is (source) */

               /* Reading the source time function */

  strcpy (flname, srcfile2);
  fp_in2 = fopen (flname, "r");

  if ( fp_in2 == NULL ){
    perror ("failed at fopen 2");
    exit(1);
  }

  fgets (buf, 255, fp_in2);
  fgets (buf, 255, fp_in2);

  fscanf ( fp_in2, "%lf %lf", &dsbiem, &dtbiem );
  fscanf ( fp_in2, "%d", &IDUR );

  vel = dmatrix (0, ISRC-1, 0, IDUR-1);

  if ( my_rank == 0 ){
    printf ("\nSource duration %f sec\n", dtbiem*(IDUR-1));
    printf ("fault segment %f m, %f s\n", dsbiem, dtbiem);
  }

  mo = 0.0;

  for ( is = 0; is < ISRC; is++){

	fscanf ( fp_in2, "%d", &n);
    fscanf ( fp_in2, "%lf %lf %lf", &strike[is], &dip[is], &rake[is]);

    if ( my_rank == 0 ){
      printf ("%d ( %f %f %f ) : %f %f %f\n",
	      is+1, xhypo[is], yhypo[is], zhypo[is], strike[is], dip[is], rake[is]);
    }

    strike[is] = strike[is]/180.*pi;
    dip[is] = dip[is]/180.*pi;
    rake[is] = rake[is]/180.*pi;
    slip[is] = 0.;

    for ( it = 0; it < IDUR; it++ ){
      fscanf ( fp_in2, "%lf", &vel[is][it]);
      slip[is] += vel[is][it]*dtbiem;
      vel[is][it] = (dsbiem/ds) * (dsbiem/ds) * vel[is][it] / ds;
    }

    mo += slip[is];

  } /* end of is (source) */

  fclose (fp_in2);

  mo = dsbiem * dsbiem * mo;
  mw = (log10(mo) - 9.1)/1.5;

  if ( my_rank == 0 ){
    printf("Mw = %f; Mo = %e [N m] \n", mw,  mo);
  }

				/* Reading the positions of the stations */

  if ( my_rank == 0 ){
    printf("\n Stations coordinates :\n");
  }

  fp_in3 = fopen (srcfile3, "r");

  if ( fp_in3 == NULL ){
    perror ("failed at fopen 3");
    exit(1);
  }

  fscanf ( fp_in3, "%d", &IOBS);

  nobs   = ivector (0, IOBS-1);
  xobs   = dvector (0, IOBS-1);
  yobs   = dvector (0, IOBS-1);
  zobs   = dvector (0, IOBS-1);
  ixobs  = ivector (0, IOBS-1);
  iyobs  = ivector (0, IOBS-1);
  izobs  = ivector (0, IOBS-1);
  xobswt = dvector (0, IOBS-1);
  yobswt = dvector (0, IOBS-1);
  zobswt = dvector (0, IOBS-1);
  ista   = ivector (0, IOBS-1);

  for ( ir = 0; ir < IOBS; ir++){

    fscanf ( fp_in3, "%d %lf %lf %lf", &nobs[ir], &xobs[ir], &yobs[ir], &zobs[ir] );

    ista[ir] = 1;

    ixobs[ir] = double2int(xobs[ir]/ds)+1;
    xobswt[ir] = (xobs[ir]/ds - ixobs[ir]+1);

    iyobs[ir] = double2int(yobs[ir]/ds+1);
    yobswt[ir] = (yobs[ir]/ds - iyobs[ir]+1);

    izobs[ir] = double2int(zobs[ir]/ds)+1;
    zobswt[ir] = (zobs[ir]/ds - izobs[ir]+1);

    if ( xobs[ir] < XMIN*ds || xobs[ir] > XMAX*ds ||
         yobs[ir] < YMIN*ds || yobs[ir] > YMAX*ds ||
         zobs[ir] < ZMIN*ds || zobs[ir] > 0 ){

      ista[ir] = 0;
    }

    if ( my_rank == 0 ){
      printf ("%d %d %f %f %f\n", ir+1, ista[ir], xobs[ir], yobs[ir], zobs[ir]);
      printf ("......%d %d %d\n", ixobs[ir], iyobs[ir], izobs[ir]);
      printf ("......%f %f %f\n", xobswt[ir], yobswt[ir], zobswt[ir]);
    }

  } /* end of ir (stations) */

  fclose (fp_in3);

  seisx  = dmatrix (0, IOBS-1, 0, TMAX-1);
  seisy  = dmatrix (0, IOBS-1, 0, TMAX-1);
  seisz  = dmatrix (0, IOBS-1, 0, TMAX-1);
  seisxx = dmatrix (0, IOBS-1, 0, TMAX-1);
  seisyy = dmatrix (0, IOBS-1, 0, TMAX-1);
  seiszz = dmatrix (0, IOBS-1, 0, TMAX-1);
  seisxy = dmatrix (0, IOBS-1, 0, TMAX-1);
  seisxz = dmatrix (0, IOBS-1, 0, TMAX-1);
  seisyz = dmatrix (0, IOBS-1, 0, TMAX-1);

                /* MPI */

  mpmx_tab = ivector(0,px-1);
  mpmy_tab = ivector(0,py-1);

/* Le domaine va de XMIN-delta a XMAX+delta+2
   XMIN-delta == CL  XMAX+delta+2 = CL
   la taille globale est donc XMAX+delta+2-XMIN+delta+1
   On partitionne donc ce domaine */

                /* Découpage : première variante */

  #if (DECOUP1)

/* Ici le +1 permet d etre sur d avoir mpmx*np > taille totale du domaine de calcul
   ceci entraine un dernier domaine de calcul surdimensionne et des operations inutiles */

	mpmx = (XMAX - XMIN + 2*delta + 3 )/px + 1;
    if ( mpmx <= 10 ) {
      printf (" Reduire le nombre de processeurs utilises \n");
      exit(0);
    }
	mpmx0 = mpmx;
	mpmx_bord = mpmx;
	mpmx_int = mpmx;
    for (i=0;i<=(px-1);i++)
      mpmx_tab[i] = mpmx ;

	mpmy = (YMAX - YMIN + 2*delta + 3 )/py + 1;
    if ( mpmy <= 10 ) {
      printf (" Reduire le nombre de processeurs utilises \n");
      exit(0);
    }
	mpmy0 = mpmy;
	mpmy_bord = mpmy;
	mpmy_int = mpmy;
    for (i=0;i<=(py-1);i++)
      mpmy_tab[i] = mpmy ;

  #endif /* fin du premier découpage */

                /* Découpage : deuxième variante */

  #if (DECOUP2)

/* Ici on essaie d ajuster au plus juste les tailles de mpmx --> variables
   on decoupe de facon pessimiste et on ajuste progressivement
   Reste le probleme du cout specifique des PMLS qui n est pas aborde */

	mpmx_tmp1 = (double)(XMAX - XMIN + 2*delta + 3 )/(double)px;
	mpmx = (int)floor(mpmx_tmp1);
	mpmx_int = mpmx;
	mpmx0 = mpmx;
    mpmx_bord = mpmx;

    for (i=0;i<=(px-1);i++)
      mpmx_tab[i] = mpmx ;

	if ( px*mpmx < ( XMAX - XMIN + 2*delta + 3 ) ) {
	  difference = ( XMAX - XMIN + 2*delta + 3 ) - px*mpmx;
	  for (i=1;i<=difference;i++)
	    mpmx_tab[i] = mpmx +1 ;
	}

    if ( mpmx <= 10 ) {
      printf (" Reduire le nombre de processeurs utilises \n");
      exit(0);
    }

	mpmy_tmp1 = (double)(YMAX - YMIN + 2*delta + 3 )/(double)py;
	mpmy = (int)floor(mpmy_tmp1);
	mpmy_int = mpmy;
	mpmy0 = mpmy;
    mpmy_bord = mpmy;

    for (i=0;i<=(py-1);i++)
      mpmy_tab[i] = mpmy ;

    if ( py*mpmy < ( YMAX - YMIN + 2*delta + 3 ) ) {
 	  difference = ( YMAX - YMIN + 2*delta + 3 ) - py*mpmy;
	  for (i=1;i<=difference;i++)
		mpmy_tab[i] = mpmy +1 ;
	}

    if ( mpmy <= 10 ) {
      printf (" Reduire le nombre de processeurs utilises \n");
      exit(0);
    }

  #endif /* fin du deuxième découpage */

                /* Fin du découpage */

/* Allocation ici tableau pour output */

  mpmx_max = 0.0;
  mpmy_max = 0.0;

  for ( i=0; i<px; i++){
    if (mpmx_tab[i] > mpmx_max ){
	  mpmx_max = mpmx_tab[i];
	}
  }
  for ( i=0; i<py; i++){
	if (mpmy_tab[i] > mpmy_max ){
	  mpmy_max = mpmy_tab[i];
	}
  }

  test_size = IMAX(mpmx_max,mpmy_max);
  test_size = IMAX(test_size,1-(ZMIN-delta)+1);

  test_size = test_size*test_size+1;
  Vxtemp1 = dvector(1,test_size);

  imp=0;
  for ( i = 1; i <= mpmx; i++ ){
    for ( j = 1; j <= mpmy; j++ ){
      assert (imp < test_size) ;
      imp ++;
      Vxtemp1[imp] = 0.0;
    }
  }

  MPI_Barrier(comm2d);

/* Controle decoupage */

  k = 0;
  for (i=0;i<px;i++){
	k = k + mpmx_tab[i];
  }
  if ( k < ( XMAX-XMIN+2*delta+3) ){
	printf(" Probleme dans le decoupage");
	exit(0);
  }

  k = 0;
  for (i=0;i<py;i++){
	k = k + mpmy_tab[i];
  }
  if ( k < ( YMAX-YMIN+2*delta+3) ){
	printf(" Probleme dans le decoupage");
	exit(0);
  }

  total_prec_x = 0;
  if (coords[0] == 0 ) total_prec_x = 0;
  if (coords[0] != 0 ) {
	for (i=0;i<coords[0];i++)
	  total_prec_x = total_prec_x + mpmx_tab[i];
  }

  total_prec_y = 0;
  if (coords[1] == 0 ) total_prec_y = 0;
  if (coords[1] != 0 ) {
	for (i=0;i<coords[1];i++)
	  total_prec_y = total_prec_y + mpmy_tab[i];
  }

  mpmx = mpmx_tab[coords[0]];
  mpmy = mpmy_tab[coords[1]];

/* i2imp_array largement surdimmensionne pour supporter DECOUP1 */

  icpu = XMIN-delta;
  i2imp_array = ivector(XMIN-delta,XMAX+2*delta+2);

  for ( j = 1; j <= px; j++ ){
	for ( i = 1; i <= mpmx_tab[j-1]; i++ ){
	  i2imp_array[icpu] = i;
	  icpu++;
	}
  }

/* j2jmp_array largement surdimmensionne pour supporter DECOUP1 */

  jcpu = YMIN-delta;
  j2jmp_array = ivector(YMIN-delta,YMAX+2*delta+2);

  for ( j = 1; j <= py; j++ ){
	for ( i = 1; i <= mpmy_tab[j-1]; i++ ){
  	  j2jmp_array[jcpu] = i;
	  jcpu++;
	}
  }

/* On veut s affranchir des anciennes fonctions imp2i */

  icpu = XMIN-delta;
  imp2i_array = ivector(-1,mpmx+2);
  for ( i = -1; i <= mpmx+2; i++ )
	imp2i_array[i] = XMIN-delta + total_prec_x + i -1 ;

  jcpu = YMIN-delta;
  jmp2j_array = ivector(-1,mpmy+2);
  for ( i = -1; i <= mpmy+2; i++ )
	jmp2j_array[i] = YMIN-delta + total_prec_y + i -1 ;

/* On veut s affranchir des anciennes fonctions i2icpu
   en fait i2icpu ne doit pas renvoyer le rang mais les coordonnees abs et ord
   sinon cela n a pas de sens en 2D */

/* Ok ici en considerant icpu est abscisse */

  icpu = 0;
  k = 0;
  i2icpu_array = ivector(XMIN-delta, XMAX+2*delta+2);
  idebut = XMIN-delta;

  for ( j = 0; j <= (px-1); j++){
	for ( i = 1; i <= mpmx_tab[j]; i++){
 	  i2icpu_array[ idebut ] = j;
	  idebut ++;
	}
  }

/* Ordonnee */

  icpu = 0;
  k = 0;
  j2jcpu_array = ivector(YMIN-delta, YMAX+2*delta+2);
  jdebut = YMIN-delta;

  for ( j = 0; j <= (py-1); j++){
	for ( i = 1; i <= mpmy_tab[j]; i++){
 	  j2jcpu_array[ jdebut ] = j;
	  jdebut ++;
	}
  }

  nmaxx = (  mpmy + 2 + 2 ) * ( 1-ZMIN + delta + 1 ) * 2;
  nmaxy = (  mpmx + 2 + 2 ) * ( 1-ZMIN + delta + 1 ) * 2;

  MPI_Barrier(MPI_COMM_WORLD);

/* Ici allouer le max de nmaxx nmaxy pour les buffers de comms */
                /* East - West */

/* modification TU 25/05/2010
allocate_communication_buffer est transformée en double*  (void* initialement)
pour permettre la libération de la mémoire à la fin
*/
  double *buffer_s1;
  double *buffer_s2;
  double *buffer_s3;
  double *buffer_s4;
  double *buffer_r1;
  double *buffer_r2;
  double *buffer_r3;
  double *buffer_r4;

  buffer_s1 = allocate_communication_buffer (
				 &sxbuf, 0, nmaxx-1,
				 &sybuf, 0, nmaxx-1,
				 &szbuf, 0, nmaxx-1,
				 &sxxbuf, 0, nmaxx-1,
				 &syybuf, 0, nmaxx-1,
				 &szzbuf, 0, nmaxx-1,
				 &sxybuf, 0, nmaxx-1,
				 &syzbuf, 0, nmaxx-1,
				 &sxzbuf, 0, nmaxx-1
				 ) ;
  buffer_s2 = allocate_communication_buffer (
				 &sxbuf2, 0, nmaxx-1,
				 &sybuf2, 0, nmaxx-1,
				 &szbuf2, 0, nmaxx-1,
				 &sxxbuf2, 0, nmaxx-1,
				 &syybuf2, 0, nmaxx-1,
				 &szzbuf2, 0, nmaxx-1,
				 &sxybuf2, 0, nmaxx-1,
				 &syzbuf2, 0, nmaxx-1,
				 &sxzbuf2, 0, nmaxx-1
				 ) ;

  buffer_s3 = allocate_communication_buffer (
				 &rxbuf, 0, nmaxx-1,
				 &rybuf, 0, nmaxx-1,
				 &rzbuf, 0, nmaxx-1,
				 &rxxbuf, 0, nmaxx-1,
				 &ryybuf, 0, nmaxx-1,
				 &rzzbuf, 0, nmaxx-1,
				 &rxybuf, 0, nmaxx-1,
				 &ryzbuf, 0, nmaxx-1,
				 &rxzbuf, 0, nmaxx-1
				 ) ;

  buffer_s4 = allocate_communication_buffer (
				 &rxbuf2, 0, nmaxx-1,
				 &rybuf2, 0, nmaxx-1,
				 &rzbuf2, 0, nmaxx-1,
				 &rxxbuf2, 0, nmaxx-1,
				 &ryybuf2, 0, nmaxx-1,
				 &rzzbuf2, 0, nmaxx-1,
				 &rxybuf2, 0, nmaxx-1,
				 &ryzbuf2, 0, nmaxx-1,
				 &rxzbuf2, 0, nmaxx-1
				 ) ;

                /* North - South */

  buffer_r1 = allocate_communication_buffer (
				 &sxbuf3, 0, nmaxy-1,
				 &sybuf3, 0, nmaxy-1,
				 &szbuf3, 0, nmaxy-1,
				 &sxxbuf3, 0, nmaxy-1,
				 &syybuf3, 0, nmaxy-1,
				 &szzbuf3, 0, nmaxy-1,
				 &sxybuf3, 0, nmaxy-1,
				 &syzbuf3, 0, nmaxy-1,
				 &sxzbuf3, 0, nmaxy-1
				 ) ;

  buffer_r2 = allocate_communication_buffer (
				 &sxbuf4, 0, nmaxy-1,
				 &sybuf4, 0, nmaxy-1,
				 &szbuf4, 0, nmaxy-1,
				 &sxxbuf4, 0, nmaxy-1,
				 &syybuf4, 0, nmaxy-1,
				 &szzbuf4, 0, nmaxy-1,
				 &sxybuf4, 0, nmaxy-1,
				 &syzbuf4, 0, nmaxy-1,
				 &sxzbuf4, 0, nmaxy-1
				 ) ;

  buffer_r3 = allocate_communication_buffer (
				 &rxbuf3, 0, nmaxy-1,
				 &rybuf3, 0, nmaxy-1,
				 &rzbuf3, 0, nmaxy-1,
				 &rxxbuf3, 0, nmaxy-1,
				 &ryybuf3, 0, nmaxy-1,
				 &rzzbuf3, 0, nmaxy-1,
				 &rxybuf3, 0, nmaxy-1,
				 &ryzbuf3, 0, nmaxy-1,
				 &rxzbuf3, 0, nmaxy-1
				 ) ;

  buffer_r4 = allocate_communication_buffer (
				 &rxbuf4, 0, nmaxy-1,
				 &rybuf4, 0, nmaxy-1,
				 &rzbuf4, 0, nmaxy-1,
				 &rxxbuf4, 0, nmaxy-1,
				 &ryybuf4, 0, nmaxy-1,
				 &rzzbuf4, 0, nmaxy-1,
				 &rxybuf4, 0, nmaxy-1,
				 &ryzbuf4, 0, nmaxy-1,
				 &rxzbuf4, 0, nmaxy-1
				 ) ;

                /* Definition of the vectors and matrixes used in the program */

/* -1 - 0 comms
   mpmx+1 - mpmx+2 comms
   domaine de calcul ( CPML + physiques + BC ) 1 --> mpmx */

  vx0 = d3tensor(-1, mpmx+2, -1,mpmy+2, ZMIN-delta, 1);
  vy0 = d3tensor(-1, mpmx+2, -1,mpmy+2, ZMIN-delta, 1);
  vz0 = d3tensor(-1, mpmx+2, -1,mpmy+2, ZMIN-delta, 1);

  for ( imp = -1; imp <= mpmx +2; imp++ ){
    for ( jmp = -1; jmp <= mpmy +2; jmp++ ){
      for ( k = ZMIN-delta; k <= 1; k++ ){
	    vx0[imp][jmp][k] = 0.0;
	    vy0[imp][jmp][k] = 0.0;
	    vz0[imp][jmp][k] = 0.0;
      }
    }
  }

  npmlv = 0;

  for ( imp = 1; imp <= mpmx; imp++ ){
 	i = imp2i_array[imp];
    for ( jmp = 1; jmp <= mpmy; jmp++ ){
   	  j = jmp2j_array[jmp];
	  for ( k = ZMIN-delta+1; k <= 1; k++){
        if( i <= XMIN+2 || i >= XMAX || j <= YMIN+2 || j >= YMAX || k <= ZMIN+2 )
  	      npmlv += 1;
      }
    }
  }

  if ( my_rank == 0 ){
    printf( "\nNumber of points in the CPML : %ld\n", npmlv);
  }

  phivxx = dvector(1, npmlv);
  phivxy = dvector(1, npmlv);
  phivxz = dvector(1, npmlv);
  phivyx = dvector(1, npmlv);
  phivyy = dvector(1, npmlv);
  phivyz = dvector(1, npmlv);
  phivzx = dvector(1, npmlv);
  phivzy = dvector(1, npmlv);
  phivzz = dvector(1, npmlv);
  for ( npml = 1; npml <= npmlv; npml++ ){
    phivxx[npml] = 0;
    phivxy[npml] = 0;
    phivxz[npml] = 0;
    phivyx[npml] = 0;
    phivyy[npml] = 0;
    phivyz[npml] = 0;
    phivzx[npml] = 0;
    phivzy[npml] = 0;
    phivzz[npml] = 0;
  }

/* 1 - mpmx car pas besoin des tranches de comms */

  fx = d3tensor(1, mpmx, 1,mpmy, ZMIN-delta, 1);
  fy = d3tensor(1, mpmx, 1,mpmy, ZMIN-delta, 1);
  fz = d3tensor(1, mpmx, 1,mpmy, ZMIN-delta, 1);
  for ( imp = 1; imp <= mpmx; imp++ ){
    i = imp2i_array[imp];
    for ( jmp = 1; jmp <= mpmy; jmp++ ){
      j = jmp2j_array[jmp];
      for ( k = ZMIN-delta; k <= 1; k++ ){
	    fx[imp][jmp][k] = 0.0;
	    fy[imp][jmp][k] = 0.0;
	    fz[imp][jmp][k] = 0.0;
      }
    }
  }

  txx0 = d3tensor(-1,mpmx+2,-1,mpmy+2,ZMIN-delta,1);
  tyy0 = d3tensor(-1,mpmx+2,-1,mpmy+2,ZMIN-delta,1);
  tzz0 = d3tensor(-1,mpmx+2,-1,mpmy+2,ZMIN-delta,1);
  txy0 = d3tensor(-1,mpmx+2,-1,mpmy+2,ZMIN-delta,1);
  txz0 = d3tensor(-1,mpmx+2,-1,mpmy+2,ZMIN-delta,1);
  tyz0 = d3tensor(-1,mpmx+2,-1,mpmy+2,ZMIN-delta,1);

  for ( imp = -1; imp <= mpmx+2; imp++ ){
    for ( jmp = -1; jmp <= mpmy+2; jmp++ ){
      for ( k = ZMIN-delta; k <= 1; k++){
        txx0[imp][jmp][k] = 0.0;
        tyy0[imp][jmp][k] = 0.0;
        tzz0[imp][jmp][k] = 0.0;
        txy0[imp][jmp][k] = 0.0;
        txz0[imp][jmp][k] = 0.0;
        tyz0[imp][jmp][k] = 0.0;
      }
    }
  }

  npmlt = 0;

  for(imp = 1; imp <= mpmx; imp++){
 	i = imp2i_array[imp];
    for(jmp = 1; jmp <= mpmy; jmp++){
   	  j = jmp2j_array[jmp];
	  for ( k = ZMIN-delta+1; k <= 1; k++){
        if ( i <= XMIN+2 || i >= XMAX || j <= YMIN+2 || j >= YMAX || k <= ZMIN+2 )
  	      npmlt += 1;
      }
    }
  }

  phitxxx = dvector(1, npmlt);
  phitxyy = dvector(1, npmlt);
  phitxzz = dvector(1, npmlt);
  phitxyx = dvector(1, npmlt);
  phityyy = dvector(1, npmlt);
  phityzz = dvector(1, npmlt);
  phitxzx = dvector(1, npmlt);
  phityzy = dvector(1, npmlt);
  phitzzz = dvector(1, npmlt);
  for ( npml = 1; npml <= npmlt; npml++ ){
    phitxxx[npml] = 0.0;
    phitxyy[npml] = 0.0;
    phitxzz[npml] = 0.0;
    phitxyx[npml] = 0.0;
    phityyy[npml] = 0.0;
    phityzz[npml] = 0.0;
    phitxzx[npml] = 0.0;
    phityzy[npml] = 0.0;
    phitzzz[npml] = 0.0;
  }

/* Dimension -1 --> mpmx pour les besoins MPI du decoupage
   -1 et mpmx+2 inutiles */

  rho = d3tensor(-1,mpmx+2,-1,mpmy+2,ZMIN-delta,2);
  mu  = d3tensor(-1,mpmx+2,-1,mpmy+2,ZMIN-delta,2);
  lam = d3tensor(-1,mpmx+2,-1,mpmy+2,ZMIN-delta,2);
  vp  = d3tensor(-1,mpmx+2,-1,mpmy+2,ZMIN-delta,2);
  vs  = d3tensor(-1,mpmx+2,-1,mpmy+2,ZMIN-delta,2);

  for ( imp = -1; imp <= mpmx+2; imp++ ){
    for ( jmp = -1; jmp <= mpmy+2; jmp++ ){
      for ( k = ZMIN-delta; k <= 2; k++){
        rho[imp][jmp][k] = 0.0;
        mu[imp][jmp][k] = 0.0;
        lam[imp][jmp][k] = 0.0;
        vp[imp][jmp][k] = 0.0;
        vs[imp][jmp][k] = 0.0;
      }
    }
  }


/* reading 1D structure */

  for(imp = -1; imp <= mpmx+2; imp++){
    i = imp2i_array[imp];
        //TU 12.03.2012 the grid of fdm14 is shifted by 1 compared with fdm7-2
	xdum = (i-1) * ds;
	  for(jmp = -1; jmp <= mpmy+2; jmp++){
 		j = jmp2j_array[jmp];
	        //TU 12.03.2012 the grid of fdm14 is shifted by 1 compared with fdm7-2
        for ( k = ZMIN-delta; k <= 2; k++){
	  //TU 12.03.2012 the grid of fdm14 is shifted by 1 compared with fdm7-2
          zdum = (k-1) * ds/1000.;
          vp[imp][jmp][k]  = vp0[NLAYER-1];
          vs[imp][jmp][k]  = vs0[NLAYER-1];
          rho[imp][jmp][k] = rho0[NLAYER-1];

        for ( ly = 0; ly < NLAYER-1; ly++){
	      if ( zdum <= laydep[ly] && zdum > laydep[ly+1] ){
	        rho[imp][jmp][k] = rho0[ly];
	        vp[imp][jmp][k] = vp0[ly];
	        vs[imp][jmp][k] = vs0[ly];
	      }
	    }

	    if ( zdum > laydep[0] ){
	      vs[imp][jmp][k] = vs0[0];
	      vp[imp][jmp][k] = vp0[0];
	      rho[imp][jmp][k] = rho0[0];
	    }

          mu[imp][jmp][k]  = vs[imp][jmp][k]*vs[imp][jmp][k]*rho[imp][jmp][k];
          lam[imp][jmp][k] = vp[imp][jmp][k]*vp[imp][jmp][k]*rho[imp][jmp][k]
                - 2.0*mu[imp][jmp][k];
        } /* fin k*/
  } /* loop j */
} /* loop imp */


                /* Extension in the CPML layers */

/* 5 faces */

  for(imp = -1; imp <= mpmx+2; imp++){
    i = imp2i_array[imp];
    if ( (i>=XMIN+2) && ( i <= XMAX )) {

	  for(jmp = -1; jmp <= mpmy+2; jmp++){
 		j = jmp2j_array[jmp];
    	if ( (j>=YMIN+2) && ( j <= YMAX )) {

	 	  for ( k = ZMIN-delta; k <= ZMIN+1; k++){

		    vp[imp][jmp][k]  = vp[imp][jmp][ZMIN+2];
		    vs[imp][jmp][k]  = vs[imp][jmp][ZMIN+2];
		    rho[imp][jmp][k] = rho[imp][jmp][ZMIN+2];
		    lam[imp][jmp][k] = lam[imp][jmp][ZMIN+2];
		    mu[imp][jmp][k]  = mu[imp][jmp][ZMIN+2];

	      } /* end of k */
        } /* end of if j */
      }	/* end of jmp */
    } /* end of if i */
  } /* end of imp */

  for(jmp = -1; jmp <= mpmy+2; jmp++){
 	j = jmp2j_array[jmp];
	if ( (j>=YMIN+2) && ( j <= YMAX )) {

      for ( k = ZMIN+2; k <= 2; k++){

        for(imp = -1; imp <= mpmx+2; imp++){
 	      i = imp2i_array[imp];
          if ( (i>=XMIN-delta) && ( i <= XMIN+1 )) {
            icpu = i2icpu_array[XMIN+2];
            imp_tmp = i2imp_array[XMIN+2];
            if ( coords[0] == icpu ){

  		      vp[imp][jmp][k]  = vp[imp_tmp][jmp][k];
  		      vs[imp][jmp][k]  = vs[imp_tmp][jmp][k];
  		      rho[imp][jmp][k] = rho[imp_tmp][jmp][k];
  		      lam[imp][jmp][k] = lam[imp_tmp][jmp][k];
  		      mu[imp][jmp][k]  = mu[imp_tmp][jmp][k];

  	        } /* end of if icpu */
	      } /* end of if i */
		} /* end of imp */

        for(imp = -1; imp <= mpmx+2; imp++){
 	      i = imp2i_array[imp];
          if ( (i>=XMAX+1) && ( i <= XMAX + delta+2 )) {
            icpu = i2icpu_array[XMAX];
            imp_tmp = i2imp_array[XMAX];
            if ( coords[0] == icpu ){

  		      vp[imp][jmp][k]  = vp[imp_tmp][jmp][k];
  		      vs[imp][jmp][k]  = vs[imp_tmp][jmp][k];
  		      rho[imp][jmp][k] = rho[imp_tmp][jmp][k];
  		      lam[imp][jmp][k] = lam[imp_tmp][jmp][k];
  		      mu[imp][jmp][k]  = mu[imp_tmp][jmp][k];

  	        } /* end of if icpu */
          } /* end if i */
	    } /* end of imp */
      } /* end of k */
    } /* end of if j */
  } /* end of jmp */

  for(imp = -1; imp <= mpmx+2; imp++){
 	i = imp2i_array[imp];
    if ( (i >= XMIN+2) && ( i <= XMAX )) {

	  for ( k = ZMIN+2; k <= 2; k++){

        for(jmp = -1; jmp <= mpmy+2; jmp++){
          j = jmp2j_array[jmp];
          if ( (j >= YMIN-delta) && ( j <= YMIN+1 )) {
            jcpu = j2jcpu_array[YMIN+2];
            jmp_tmp = j2jmp_array[YMIN+2];
            if ( coords[1] == jcpu ){

		      vp[imp][jmp][k]  = vp[imp][jmp_tmp][k];
		      vs[imp][jmp][k]  = vs[imp][jmp_tmp][k];
		      rho[imp][jmp][k] = rho[imp][jmp_tmp][k];
		      lam[imp][jmp][k] = lam[imp][jmp_tmp][k];
		      mu[imp][jmp][k]  = mu[imp][jmp_tmp][k];

            } /* end of if jcpu */
          } /* end of if j */
        } /* end of jmp */

        for(jmp = -1; jmp <= mpmy+2; jmp++){
 	      j = jmp2j_array[jmp];
          if ( (j >= YMAX+1) && ( j <= YMAX+delta+2 )) {
            jcpu = j2jcpu_array[YMAX];
            jmp_tmp = j2jmp_array[YMAX];
            if ( coords[1] == jcpu ){

		      vp[imp][jmp][k]  = vp[imp][jmp_tmp][k];
		      vs[imp][jmp][k]  = vs[imp][jmp_tmp][k];
		      rho[imp][jmp][k] = rho[imp][jmp_tmp][k];
		      lam[imp][jmp][k] = lam[imp][jmp_tmp][k];
		      mu[imp][jmp][k]  = mu[imp][jmp_tmp][k];

            } /* end of if jcpu */
          } /* end of if j */
        } /* end of jmp */
      } /* end of k */
    } /* end of if i */
  } /* end of imp */

/* 8 arêtes */

  for(imp = -1; imp <= mpmx+2; imp++){
 	i = imp2i_array[imp];
    if ( (i>=XMIN+2) && ( i <= XMAX )) {

      for(jmp = -1; jmp <= mpmy+2; jmp++){
 	    j = jmp2j_array[jmp];
        if ( (j >= YMIN-delta) && ( j <= YMIN+1 )) {
          jcpu = j2jcpu_array[YMIN+2];
          jmp_tmp = j2jmp_array[YMIN+2];
          if ( coords[1] == jcpu ){

	        for ( k = ZMIN-delta; k <= ZMIN+1; k++){

		      vp[imp][jmp][k]  = vp[imp][jmp_tmp][ZMIN+2];
		      vs[imp][jmp][k]  = vs[imp][jmp_tmp][ZMIN+2];
		      rho[imp][jmp][k] = rho[imp][jmp_tmp][ZMIN+2];
		      lam[imp][jmp][k] = lam[imp][jmp_tmp][ZMIN+2];
		      mu[imp][jmp][k]  = mu[imp][jmp_tmp][ZMIN+2];

            } /* end of k */
          } /* end of if jcpu */
        } /* end of if j */
      } /* end of jmp */

      for(jmp = -1; jmp <= mpmy+2; jmp++){
 	    j = jmp2j_array[jmp];
        if ( (j >= YMAX+1) && ( j <= YMAX+delta+2 )) {
          jcpu = j2jcpu_array[YMAX];
          jmp_tmp = j2jmp_array[YMAX];
          if ( coords[1] == jcpu ){

	        for ( k = ZMIN-delta; k <= ZMIN+1; k++){

		      vp[imp][jmp][k]  = vp[imp][jmp_tmp][ZMIN+2];
		      vs[imp][jmp][k]  = vs[imp][jmp_tmp][ZMIN+2];
		      rho[imp][jmp][k] = rho[imp][jmp_tmp][ZMIN+2];
		      lam[imp][jmp][k] = lam[imp][jmp_tmp][ZMIN+2];
		      mu[imp][jmp][k]  = mu[imp][jmp_tmp][ZMIN+2];

            } /* end of k */
          } /* end of if jcpu */
        } /* end of j */
      } /* end of jmp */
    } /* end of i */
  } /* end of imp */

  for(jmp = -1; jmp <= mpmy+2; jmp++){
    j = jmp2j_array[jmp];
    if ( (j >= YMIN+2) && ( j <= YMAX)) {

      for(imp = -1; imp <= mpmx+2; imp++){
        i = imp2i_array[imp];
        if ( (i>=XMIN-delta) && ( i <= XMIN+1 )) {
          icpu = i2icpu_array[XMIN+2];
          imp_tmp = i2imp_array[XMIN+2];
          if ( coords[0] == icpu ){

	        for ( k = ZMIN-delta; k <= ZMIN+1; k++){

	          vp[imp][jmp][k]  = vp[imp_tmp][jmp][ZMIN+2];
		      vs[imp][jmp][k]  = vs[imp_tmp][jmp][ZMIN+2];
		      rho[imp][jmp][k] = rho[imp_tmp][jmp][ZMIN+2];
		      lam[imp][jmp][k] = lam[imp_tmp][jmp][ZMIN+2];
		      mu[imp][jmp][k]  = mu[imp_tmp][jmp][ZMIN+2];

            } /* end of k */
          } /* end of if icpu */
        } /* end of if */
      } /* end of imp */

      for(imp = -1; imp <= mpmx+2; imp++){
        i = imp2i_array[imp];
        if ( (i>=XMAX+1) && ( i <= XMAX+delta+2 )) {
          icpu = i2icpu_array[XMAX];
          imp_tmp = i2imp_array[XMAX];
          if ( coords[0] == icpu ){

	        for ( k = ZMIN-delta; k <= ZMIN+1; k++){

		      vp[imp][jmp][k]  = vp[imp_tmp][jmp][ZMIN+2];
		      vs[imp][jmp][k]  = vs[imp_tmp][jmp][ZMIN+2];
		      rho[imp][jmp][k] = rho[imp_tmp][jmp][ZMIN+2];
		      lam[imp][jmp][k] = lam[imp_tmp][jmp][ZMIN+2];
		      mu[imp][jmp][k]  = mu[imp_tmp][jmp][ZMIN+2];

            } /* end of k */
          } /* end of if icpu */
        } /* end of if i */
      } /* end of imp */
    } /* end of if j */
  } /* end of jmp */

  for ( k = ZMIN+2; k <= 2; k++){

    for(jmp = -1; jmp <= mpmy+2; jmp++){
 	  j = jmp2j_array[jmp];
      if ( (j >= YMIN-delta) && ( j <= YMIN+1 )) {
        jcpu = j2jcpu_array[YMIN+2];
        jmp_tmp = j2jmp_array[YMIN+2];
        if ( coords[1] == jcpu ){

          for(imp = -1; imp <= mpmx+2; imp++){
  	        i = imp2i_array[imp];
            if ( (i>=XMIN-delta) && ( i <= XMIN+1 )) {
              icpu = i2icpu_array[XMIN+2];
              imp_tmp = i2imp_array[XMIN+2];
              if ( coords[0] == icpu ){

	            vp[imp][jmp][k]  = vp[imp_tmp][jmp_tmp][k];
	  	        vs[imp][jmp][k]  = vs[imp_tmp][jmp_tmp][k];
		        rho[imp][jmp][k] = rho[imp_tmp][jmp_tmp][k];
		        lam[imp][jmp][k] = lam[imp_tmp][jmp_tmp][k];
		        mu[imp][jmp][k]  = mu[imp_tmp][jmp_tmp][k];

              } /* end of if icpu */
            } /* end of if i */
          } /* end of imp */

          for(imp = -1; imp <= mpmx+2; imp++){
 	        i = imp2i_array[imp];
	        if ( (i>=XMAX+1) && ( i <= XMAX+delta+2 )) {
              icpu = i2icpu_array[XMAX];
              imp_tmp = i2imp_array[XMAX];
              if ( my_rank == icpu ){

	            vp[imp][jmp][k]  = vp[imp_tmp][jmp_tmp][k];
	       	    vs[imp][jmp][k]  = vs[imp_tmp][jmp_tmp][k];
		        rho[imp][jmp][k] = rho[imp_tmp][jmp_tmp][k];
		        lam[imp][jmp][k] = lam[imp_tmp][jmp_tmp][k];
		        mu[imp][jmp][k]  = mu[imp_tmp][jmp_tmp][k];

              }  /* end of if icpu */
            } /* end of if i */
          } /* end of imp */
        } /* end of if jcpu */
      } /* end of if j */
    } /* end of jmp */

    for(jmp = -1; jmp <= mpmy+2; jmp++){
   	  j = jmp2j_array[jmp];
      if ( (j >= YMAX+1) && ( j <= YMAX+delta+2 )) {
        jcpu = j2jcpu_array[YMAX];
        jmp_tmp = j2jmp_array[YMAX];
        if ( coords[1] == jcpu ){

          for(imp = -1; imp <= mpmx+2; imp++){
  	        i = imp2i_array[imp];
	        if ( (i>=XMIN-delta) && ( i <= XMIN+1 )) {
              icpu = i2icpu_array[XMIN+2];
              imp_tmp = i2imp_array[XMIN+2];
              if ( coords[0] == icpu ){

		        vp[imp][jmp][k]  = vp[imp_tmp][jmp_tmp][k];
		        vs[imp][jmp][k]  = vs[imp_tmp][jmp_tmp][k];
		        rho[imp][jmp][k] = rho[imp_tmp][jmp_tmp][k];
		        lam[imp][jmp][k] = lam[imp_tmp][jmp_tmp][k];
		        mu[imp][jmp][k]  = mu[imp_tmp][jmp_tmp][k];

              } /* end of if icpu */
            } /* end of if i */
          } /* end of imp */

          for(imp = -1; imp <= mpmx+2; imp++){
 	        i = imp2i_array[imp];
            if ( (i>=XMAX+1) && ( i <= XMAX+delta+2 )) {
              icpu = i2icpu_array[XMAX];
              imp_tmp = i2imp_array[XMAX];
              if ( coords[0] == icpu ){

	            vp[imp][jmp][k]  = vp[imp_tmp][jmp_tmp][k];
		        vs[imp][jmp][k]  = vs[imp_tmp][jmp_tmp][k];
		        rho[imp][jmp][k] = rho[imp_tmp][jmp_tmp][k];
		        lam[imp][jmp][k] = lam[imp_tmp][jmp_tmp][k];
	 	        mu[imp][jmp][k]  = mu[imp_tmp][jmp_tmp][k];

              } /* end of if icpu */
            } /* end of if i */
          } /* end of imp */
        } /* end of if jcpu */
      } /* end of if j */
    } /* end of jmp */
  } /* end of k */

/* 4 sommets */

  for(imp = -1; imp <= mpmx+2; imp++){
 	i = imp2i_array[imp];
    if ( (i>=XMIN-delta) && ( i <= XMIN+1 )) {
      icpu = i2icpu_array[XMIN+2];
      imp_tmp = i2imp_array[XMIN+2];
      if ( coords[0] == icpu ){

        for(jmp = -1; jmp <= mpmy+2; jmp++){
   	      j = jmp2j_array[jmp];
	      if ( (j >= YMIN-delta) && ( j <= YMIN+1 )) {
            jcpu = j2jcpu_array[YMIN+2];
            jmp_tmp = j2jmp_array[YMIN+2];
            if ( coords[1] == jcpu ){

	          for ( k = ZMIN-delta; k <= ZMIN+1; k++){

	            vp[imp][jmp][k]  = vp[imp_tmp][jmp_tmp][ZMIN+2];
	 	        vs[imp][jmp][k]  = vs[imp_tmp][jmp_tmp][ZMIN+2];
		        rho[imp][jmp][k] = rho[imp_tmp][jmp_tmp][ZMIN+2];
		        lam[imp][jmp][k] = lam[imp_tmp][jmp_tmp][ZMIN+2];
		        mu[imp][jmp][k]  = mu[imp_tmp][jmp_tmp][ZMIN+2];

              } /* end of k */
            } /* end of if jcpu */
          } /* end of if j */
        } /* end of jmp */

        for(jmp = -1; jmp <= mpmy+2; jmp++){
 	      j = jmp2j_array[jmp];
          if ( (j >= YMAX+1) && ( j <= YMAX+delta+2 )) {
            jcpu = j2jcpu_array[YMAX];
            jmp_tmp = j2jmp_array[YMAX];
            if ( coords[1] == jcpu ){

	          for ( k = ZMIN-delta; k <= ZMIN+1; k++){

		        vp[imp][jmp][k]  = vp[imp_tmp][jmp_tmp][ZMIN+2];
		        vs[imp][jmp][k]  = vs[imp_tmp][jmp_tmp][ZMIN+2];
		        rho[imp][jmp][k] = rho[imp_tmp][jmp_tmp][ZMIN+2];
		        lam[imp][jmp][k] = lam[imp_tmp][jmp_tmp][ZMIN+2];
		        mu[imp][jmp][k]  = mu[imp_tmp][jmp_tmp][ZMIN+2];

              } /* end of k */
            } /* end of if jcpu */
          } /* end of if j */
        } /* end of jmp */
      } /* end of if icpu */
    } /* end of if i */
  } /* end of imp */

  for(imp = -1; imp <= mpmx+2; imp++){
 	i = imp2i_array[imp];
    if ( (i>=XMAX+1) && ( i <= XMAX+delta+2 )) {
      icpu = i2icpu_array[XMAX];
      imp_tmp = i2imp_array[XMAX];
      if ( coords[0] == icpu ){

        for(jmp = -1; jmp <= mpmy+2; jmp++){
    	  j = jmp2j_array[jmp];
    	  if ( (j >= YMIN-delta) && ( j <= YMIN+1 )) {
            jcpu = j2jcpu_array[YMIN+2];
            jmp_tmp = j2jmp_array[YMIN+2];
            if ( coords[1] == jcpu ){

	          for ( k = ZMIN-delta; k <= ZMIN+1; k++){

		        vp[imp][jmp][k]  = vp[imp_tmp][jmp_tmp][ZMIN+2];
		        vs[imp][jmp][k]  = vs[imp_tmp][jmp_tmp][ZMIN+2];
		        rho[imp][jmp][k] = rho[imp_tmp][jmp_tmp][ZMIN+2];
		        lam[imp][jmp][k] = lam[imp_tmp][jmp_tmp][ZMIN+2];
		        mu[imp][jmp][k]  = mu[imp_tmp][jmp_tmp][ZMIN+2];

              } /* end of k */
            } /* end of if jcpu */
          } /* end of if j */
        } /* end of jmp */

        for(jmp = -1; jmp <= mpmy+2; jmp++){
 	      j = jmp2j_array[jmp];
          if ( (j >= YMAX+1) && ( j <= YMAX+delta+2 )) {
            icpu = j2jcpu_array[YMAX];
            jmp_tmp = j2jmp_array[YMAX];
            if ( coords[1] == icpu ){

	          for ( k = ZMIN-delta; k <= ZMIN+1; k++){

		        vp[imp][jmp][k]  = vp[imp_tmp][jmp_tmp][ZMIN+2];
		        vs[imp][jmp][k]  = vs[imp_tmp][jmp_tmp][ZMIN+2];
		        rho[imp][jmp][k] = rho[imp_tmp][jmp_tmp][ZMIN+2];
		        lam[imp][jmp][k] = lam[imp_tmp][jmp_tmp][ZMIN+2];
		        mu[imp][jmp][k]  = mu[imp_tmp][jmp_tmp][ZMIN+2];

              } /* end of k */
            } /* end of if icpu */
          } /* end of if j */
        } /* end of jmp */
      } /* end of if icpu */
    } /* end of if i */
  } /* end of imp */

		/* output of prepared structure model */
    strcpy(flname6, outdir);
    strcat(flname6, char6);
    sprintf(number, "%2.2d", my_rank);
    strcat(flname6, number);
/*
    fp6 = fopen(flname6, "w");
    for(imp = 1; imp <= mpmx; imp++){
      i = imp2i_array[imp];
      if ( (i>=XMIN) && (i<= XMAX)) {
        xdum = (i-1) * ds;

        for(jmp = 1; jmp <= mpmy; jmp++){
          j = jmp2j_array[jmp];

          if ( (j>=YMIN) && (j<= YMAX)) {
            ydum = (j-1) * ds;

            for ( k = ZMIN; k <= 1; k++){
              zdum = (k-1) * ds;

            fprintf(fp6, "%7.2f %7.2f %7.2f %10.3f %10.3f %10.3f %10.3e %10.3e\n",
            xdum, ydum, zdum, vp[imp][jmp][k], vs[imp][jmp][k], rho[imp][jmp][k],
	    lam[imp][jmp][k], mu[imp][jmp][k]);
            }
          }
        }
      }
    }
    fclose(fp6);
*/
                /* Definition of the vectors used in the CPML formulation */

  dumpx = dvector(1,mpmx);
  kappax = dvector(1,mpmx);
  alphax = dvector(1,mpmx);
  dumpx2 = dvector(1,mpmx);
  kappax2 = dvector(1,mpmx);
  alphax2 = dvector(1,mpmx);

  dumpy = dvector(1,mpmy);
  kappay = dvector(1,mpmy);
  alphay = dvector(1,mpmy);
  dumpy2 = dvector(1,mpmy);
  kappay2 = dvector(1,mpmy);
  alphay2 = dvector(1,mpmy);

  dumpz = dvector(ZMIN-delta,1);
  kappaz = dvector(ZMIN-delta,1);
  alphaz = dvector(ZMIN-delta,1);
  dumpz2 = dvector(ZMIN-delta,1);
  kappaz2 = dvector(ZMIN-delta,1);
  alphaz2 = dvector(ZMIN-delta,1);

  for  ( imp = 1 ; imp <= mpmx ; imp++){
    dumpx[imp] = 0.0;
    dumpx2[imp] = 0.0;
    kappax[imp] = 1.0;
    kappax2[imp] = 1.0;
    alphax[imp] = 0.0;
    alphax2[imp] = 0.0;
  }

  for  ( jmp = 1 ; jmp <= mpmy ; jmp++){
    dumpy[jmp] = 0.0;
    dumpy2[jmp] = 0.0;
    kappay[jmp] = 1.0;
    kappay2[jmp] = 1.0;
    alphay[jmp] = 0.0;
    alphay2[jmp] = 0.0;
  }

  for ( k = ZMIN-delta; k <= 1; k++){
    dumpz[k] = 0.0;
    dumpz2[k] = 0.0;
    kappaz[k] = 1.0;
    kappaz2[k] = 1.0;
    alphaz[k] = 0.0;
    alphaz2[k] = 0.0;
  }

                /* For the x axis */

  xoriginleft = XMIN*ds;
  xoriginright = XMAX*ds;


 for(imp = 1; imp <= mpmx; imp++){
 	i = imp2i_array[imp];
    xval = ds * (i-1);

                /* For the left side */

    abscissa_in_PML = xoriginleft - xval;
    if(abscissa_in_PML >= 0.0){
      abscissa_normalized = abscissa_in_PML / (delta * ds);
      dumpx[imp] = dump0 * pow(abscissa_normalized,NPOWER);
      kappax[imp] = 1.0 + (kappa0 - 1.0) * pow(abscissa_normalized,NPOWER);
      alphax[imp] = alpha0 * (1.0 - abscissa_normalized);
    }

    abscissa_in_PML = xoriginleft - (xval + ds/2.0);
    if(abscissa_in_PML >= 0.0){
      abscissa_normalized = abscissa_in_PML / (delta * ds);
      dumpx2[imp] = dump0 * pow(abscissa_normalized,NPOWER);
      kappax2[imp] = 1.0 + (kappa0 - 1.0) * pow(abscissa_normalized,NPOWER);
      alphax2[imp] = alpha0 * (1.0 - abscissa_normalized);
    }

                /* For the right side */

    abscissa_in_PML = xval - xoriginright;
    if(abscissa_in_PML >= 0.0){
      abscissa_normalized = abscissa_in_PML / (delta * ds);
      dumpx[imp] = dump0 * pow(abscissa_normalized,NPOWER);
      kappax[imp] = 1.0 + (kappa0 - 1.0) * pow(abscissa_normalized,NPOWER);
      alphax[imp] = alpha0 * (1.0 - abscissa_normalized);
    }

    abscissa_in_PML = xval + ds/2.0 - xoriginright;
    if(abscissa_in_PML >= 0.0){
      abscissa_normalized = abscissa_in_PML / (delta * ds);
      dumpx2[imp] = dump0 * pow(abscissa_normalized,NPOWER);
      kappax2[imp] = 1.0 + (kappa0 - 1.0) * pow(abscissa_normalized,NPOWER);
      alphax2[imp] = alpha0 * (1.0 - abscissa_normalized);
    }

    if(alphax[imp] < 0.0) alphax[imp] = 0.0;
    if(alphax2[imp] < 0.0) alphax2[imp] = 0.0;

  }

                /* For the y axis */

  yoriginfront = YMIN*ds;
  yoriginback = YMAX*ds;

  for(jmp = 1; jmp <= mpmy; jmp++){
 	j = jmp2j_array[jmp];
    yval = ds * (j-1);

                /* For the front side */

    abscissa_in_PML = yoriginfront - yval;
    if(abscissa_in_PML >= 0.0){
      abscissa_normalized = abscissa_in_PML / (delta * ds);
      dumpy[jmp] = dump0 * pow(abscissa_normalized,NPOWER);
      kappay[jmp] = 1.0 + (kappa0 - 1.0) * pow(abscissa_normalized,NPOWER);
      alphay[jmp] = alpha0 * (1.0 - abscissa_normalized);
    }

    abscissa_in_PML = yoriginfront - (yval + ds/2.0);
    if(abscissa_in_PML >= 0.0){
      abscissa_normalized = abscissa_in_PML / (delta * ds);
      dumpy2[jmp] = dump0 * pow(abscissa_normalized,NPOWER);
      kappay2[jmp] = 1.0 + (kappa0 - 1.0) * pow(abscissa_normalized,NPOWER);
      alphay2[jmp] = alpha0 * (1.0 - abscissa_normalized);
    }

                /* For the back side */

    abscissa_in_PML = yval - yoriginback;
    if(abscissa_in_PML >= 0.0){
      abscissa_normalized = abscissa_in_PML / (delta * ds);
      dumpy[jmp] = dump0 * pow(abscissa_normalized,NPOWER);
      kappay[jmp] = 1.0 + (kappa0 - 1.0) * pow(abscissa_normalized,NPOWER);
      alphay[jmp] = alpha0 * (1.0 - abscissa_normalized);
    }

    abscissa_in_PML = yval + ds/2.0 - yoriginback;
    if(abscissa_in_PML >= 0.0){
      abscissa_normalized = abscissa_in_PML / (delta * ds);
      dumpy2[jmp] = dump0 * pow(abscissa_normalized,NPOWER);
      kappay2[jmp] = 1.0 + (kappa0 - 1.0) * pow(abscissa_normalized,NPOWER);
      alphay2[jmp] = alpha0 * (1.0 - abscissa_normalized);
    }

    if(alphay[jmp] < 0.0) alphay[jmp] = 0.0;
    if(alphay2[jmp] < 0.0) alphay2[jmp] = 0.0;

  }

                /* For the z axis */

  zoriginbottom = ZMIN*ds;

  for ( k = ZMIN-delta; k <= 1; k++){
    zval = ds * (k-1);

                /* For the bottom side */

    abscissa_in_PML = zoriginbottom - zval;
    if(abscissa_in_PML >= 0.0){
      abscissa_normalized = abscissa_in_PML / (delta * ds);
      dumpz[k] = dump0 * pow(abscissa_normalized,NPOWER);
      kappaz[k] = 1.0 + (kappa0 - 1.0) * pow(abscissa_normalized,NPOWER);
      alphaz[k] = alpha0 * (1.0 - abscissa_normalized);
    }

    abscissa_in_PML = zoriginbottom - (zval + ds/2.0);
    if(abscissa_in_PML >= 0.0){
      abscissa_normalized = abscissa_in_PML / (delta * ds);
      dumpz2[k] = dump0 * pow(abscissa_normalized,NPOWER);
      kappaz2[k] = 1.0 + (kappa0 - 1.0) * pow(abscissa_normalized,NPOWER);
      alphaz2[k] = alpha0 * (1.0 - abscissa_normalized);
    }

    if(alphaz[k] < 0.0) alphaz[k] = 0.0;
    if(alphaz2[k] < 0.0) alphaz2[k] = 0.0;

  }

                /* Beginning of the iteration */

  #if(MISS)
    if ( retval=PAPI_reset(EventSet) != PAPI_OK )
      printf ("ERROR stop \n");

    if ( retval=PAPI_start(EventSet) != PAPI_OK )
      printf ("ERROR start \n");
  #endif

  #if(FLOPS)
    if((retval=PAPI_flops(&ireal_time,&iproc_time,&iflpops,&imflops)) < PAPI_OK){
      printf("Could not initialise PAPI_flops \n");
      printf("Your platform may not support floating point operation event.\n");
      printf("retval: %d\n", retval);
      exit(1);
    }
  #endif

  #if (TIMING)
    timing3 = my_second();
  #endif

  #if (TIMING_BARRIER)
    MPI_Barrier(MPI_COMM_WORLD);
    timing3 = my_second();
  #endif

                /* Preparation envoi et reception des messages */
                /* Communications persistantes */

                /* Stress */

  #if (PERSISTANT)

                /* East - West */
                /* Positive direction */

	if ( nest != -1 ) {

      MPI_Send_init(txx0[mpmx-1][-1], nmaxx, MPI_DOUBLE, nest,
       1, comm2d, &sendreq[1]);
      MPI_Send_init(tyy0[mpmx-1][-1], nmaxx, MPI_DOUBLE, nest,
       2, comm2d, &sendreq[2]);
      MPI_Send_init(tzz0[mpmx-1][-1], nmaxx, MPI_DOUBLE, nest,
       3, comm2d, &sendreq[3]);
      MPI_Send_init(txy0[mpmx-1][-1], nmaxx, MPI_DOUBLE, nest,
       4, comm2d, &sendreq[4]);
      MPI_Send_init(tyz0[mpmx-1][-1], nmaxx, MPI_DOUBLE, nest,
       5, comm2d, &sendreq[5]);
      MPI_Send_init(txz0[mpmx-1][-1], nmaxx, MPI_DOUBLE, nest,
       6, comm2d, &sendreq[6]);
	}

	if (nouest != -1 ) {

      MPI_Recv_init(txx0[-1][-1], nmaxx, MPI_DOUBLE, nouest,
       1, comm2d, &recvreq[1]);
      MPI_Recv_init(tyy0[-1][-1], nmaxx, MPI_DOUBLE, nouest,
       2, comm2d, &recvreq[2]);
      MPI_Recv_init(tzz0[-1][-1], nmaxx, MPI_DOUBLE, nouest,
       3, comm2d, &recvreq[3]);
      MPI_Recv_init(txy0[-1][-1], nmaxx, MPI_DOUBLE, nouest,
       4, comm2d, &recvreq[4]);
      MPI_Recv_init(tyz0[-1][-1], nmaxx, MPI_DOUBLE, nouest,
       5, comm2d, &recvreq[5]);
      MPI_Recv_init(txz0[-1][-1], nmaxx, MPI_DOUBLE, nouest,
       6, comm2d, &recvreq[6]);
	}

                /* Negative direction */

	if (nouest != -1 ) {

      MPI_Send_init(txx0[1][-1],nmaxx, MPI_DOUBLE, nouest,
       7, comm2d, &sendreq[7]);
      MPI_Send_init(tyy0[1][-1],nmaxx, MPI_DOUBLE, nouest,
       8, comm2d, &sendreq[8]);
      MPI_Send_init(tzz0[1][-1],nmaxx, MPI_DOUBLE, nouest,
       9, comm2d, &sendreq[9]);
      MPI_Send_init(txy0[1][-1],nmaxx, MPI_DOUBLE, nouest,
       10, comm2d, &sendreq[10]);
      MPI_Send_init(tyz0[1][-1],nmaxx, MPI_DOUBLE, nouest,
       11, comm2d, &sendreq[11]);
      MPI_Send_init(txz0[1][-1],nmaxx, MPI_DOUBLE, nouest,
       12, comm2d, &sendreq[12]);
	}

	if ( nest != -1 ) {

      MPI_Recv_init(txx0[mpmx+1][-1], nmaxx, MPI_DOUBLE, nest,
       7, comm2d, &recvreq[7]);
      MPI_Recv_init(tyy0[mpmx+1][-1], nmaxx, MPI_DOUBLE, nest,
       8, comm2d, &recvreq[8]);
      MPI_Recv_init(tzz0[mpmx+1][-1], nmaxx, MPI_DOUBLE, nest,
       9, comm2d, &recvreq[9]);
      MPI_Recv_init(txy0[mpmx+1][-1], nmaxx, MPI_DOUBLE, nest,
       10, comm2d, &recvreq[10]);
      MPI_Recv_init(tyz0[mpmx+1][-1], nmaxx, MPI_DOUBLE, nest,
       11, comm2d, &recvreq[11]);
      MPI_Recv_init(txz0[mpmx+1][-1], nmaxx, MPI_DOUBLE, nest,
       12, comm2d, &recvreq[12]);
	}

                /* North - South */

    if( coords[1] != (py-1) )
      MPI_Send_init(sxxbuf3, 6*nmaxy, MPI_DOUBLE, nnord,
	   13, MPI_COMM_WORLD, &sendreq3[1]);

    if( coords[1] != 0 )
      MPI_Send_init(sxxbuf4, 6*nmaxy, MPI_DOUBLE, nsud,
	   14, MPI_COMM_WORLD, &sendreq4[1]);

    if( coords[1] != (py-1) )
      MPI_Recv_init(rxxbuf4, 6*nmaxy, MPI_DOUBLE, nnord,
       14, MPI_COMM_WORLD, &recvreq4[1]);

    if( coords[1] != 0 )
      MPI_Recv_init(rxxbuf3, 6*nmaxy, MPI_DOUBLE, nsud,
       13, MPI_COMM_WORLD, &recvreq3[1]);

                /* Velocity */

                /* East - West */

	if ( nest != -1 ) {

      MPI_Send_init(vx0[mpmx-1][-1], nmaxx, MPI_DOUBLE, nest,
       1, comm2d, &sendreq2[1]);
      MPI_Send_init(vy0[mpmx-1][-1], nmaxx, MPI_DOUBLE, nest,
       2, comm2d, &sendreq2[2]);
      MPI_Send_init(vz0[mpmx-1][-1], nmaxx, MPI_DOUBLE, nest,
       3, comm2d, &sendreq2[3]);
	}

	if ( nouest != -1 ) {

      MPI_Recv_init(vx0[-1][-1], nmaxx, MPI_DOUBLE, nouest,
       1, comm2d, &recvreq2[1]);
      MPI_Recv_init(vy0[-1][-1], nmaxx, MPI_DOUBLE, nouest,
       2, comm2d, &recvreq2[2]);
      MPI_Recv_init(vz0[-1][-1], nmaxx, MPI_DOUBLE, nouest,
       3, comm2d, &recvreq2[3]);
	}

	if ( nouest != -1 ) {

      MPI_Send_init(vx0[1][-1],nmaxx, MPI_DOUBLE, nouest,
       4, comm2d, &sendreq2[4]);
      MPI_Send_init(vy0[1][-1],nmaxx, MPI_DOUBLE, nouest,
       5, comm2d, &sendreq2[5]);
      MPI_Send_init(vz0[1][-1],nmaxx, MPI_DOUBLE, nouest,
       6, comm2d, &sendreq2[6]);
	}

	if ( nest != -1 ) {
      MPI_Recv_init(vx0[mpmx+1][-1], nmaxx, MPI_DOUBLE, nest,
       4, comm2d, &recvreq2[4]);
      MPI_Recv_init(vy0[mpmx+1][-1], nmaxx, MPI_DOUBLE, nest,
       5, comm2d, &recvreq2[5]);
      MPI_Recv_init(vz0[mpmx+1][-1], nmaxx, MPI_DOUBLE, nest,
       6, comm2d, &recvreq2[6]);
	}

                /* North - South */

    if( coords[1] != (py-1) )
      MPI_Send_init(sxbuf3, 3*nmaxy, MPI_DOUBLE, nnord,
       21, MPI_COMM_WORLD, &sendreq3[2]);

    if( coords[1] != 0 )
      MPI_Send_init(sxbuf4, 3*nmaxy, MPI_DOUBLE, nsud,
       22, MPI_COMM_WORLD, &sendreq4[2]);

    if( coords[1] != (py-1) )
      MPI_Recv_init(rxbuf4, 3*nmaxy, MPI_DOUBLE, nnord,
       22, MPI_COMM_WORLD, &recvreq4[2]);

    if( coords[1] != 0 )
      MPI_Recv_init(rxbuf3, 3*nmaxy, MPI_DOUBLE, nsud,
       21, MPI_COMM_WORLD, &recvreq3[2]);

  #endif

/*	Allocation output */

	mapping_seis = imatrix (0, IOBS-1, 1, 9);
	seis_output = d3tensor (0, TMAX-1, 0, IOBS-1, 1, 9);

/* mapping cpu X direction (coords[0]) then y direction (coords[1])
   rank = coords[0] + coords[1]*px */

    for ( ir = 0; ir < IOBS; ir++){
      if ( ista[ir] == 1 ){

/* Vx component */

        i = ixobs[ir];
        j = iyobs[ir];
        icpu = i2icpu_array[i];
        jcpu = j2jcpu_array[j];
        mapping_seis[ir][1] = icpu + jcpu*px;

/* Vy component */

        if ( xobswt[ir] >= 0.5 ){
          i = ixobs[ir];
        } else {
          i = ixobs[ir]-1;
        }
        if ( yobswt[ir] >= 0.5 ){
          j = iyobs[ir];
        } else {
          j = iyobs[ir]-1;
        }
        icpu = i2icpu_array[i];
        jcpu = j2jcpu_array[j];
	    mapping_seis[ir][2] = icpu + jcpu*px;

/* Vz component */

        if ( xobswt[ir] >= 0.5 ){
          i = ixobs[ir];
        } else {
          i = ixobs[ir]-1;
        }
        j = iyobs[ir];
        icpu = i2icpu_array[i];
        jcpu = j2jcpu_array[j];
	    mapping_seis[ir][3] = icpu + jcpu*px;

/* Tii component */

        if ( xobswt[ir] >= 0.5 ){
          i = ixobs[ir];
        } else {
          i = ixobs[ir]-1;
        }
        j = iyobs[ir];
        icpu = i2icpu_array[i];
        jcpu = j2jcpu_array[j];
  	    mapping_seis[ir][4] = icpu + jcpu*px;
	    mapping_seis[ir][5] = icpu + jcpu*px;
	    mapping_seis[ir][6] = icpu + jcpu*px;

/* Txy component */

        i = ixobs[ir];
        if ( yobswt[ir] >= 0.5 ){
          j = iyobs[ir];
        } else {
          j = iyobs[ir]-1;
        }
        icpu = i2icpu_array[i];
        jcpu = j2jcpu_array[j];
	    mapping_seis[ir][7] = icpu + jcpu*px;

/* Txz component */

        i = ixobs[ir];
        j = iyobs[ir];
        icpu = i2icpu_array[i];
        jcpu = j2jcpu_array[j];
  	    mapping_seis[ir][8] = icpu + jcpu*px;

/* Tyz component */

        if ( xobswt[ir] >= 0.5 ){
          i = ixobs[ir];
        } else {
          i = ixobs[ir]-1;
        }
        if ( yobswt[ir] >= 0.5 ){
          j = iyobs[ir];
        } else {
          j = iyobs[ir]-1;
        }
        icpu = i2icpu_array[i];
        jcpu = j2jcpu_array[j];
	    mapping_seis[ir][9] = icpu + jcpu*px;

	} /* end of if ista */
  } /* end of ir */

                /* Beginning of time loop */

  for ( l = 1; l <= TMAX; l++ ){

    time = dt * l;

				/* Increment of seismic moment */

    it = (int) (time/dtbiem);

    if ( it < IDUR ){

      for ( is = 0; is < ISRC; is++ ){

	    if ( insrc[is] == 1 ){

          mo = vel[is][it] * dt;

          pxx = radxx(strike[is], dip[is], rake[is]);
          pyy = radyy(strike[is], dip[is], rake[is]);
          pzz = radzz(strike[is], dip[is], rake[is]);
          pxy = radxy(strike[is], dip[is], rake[is]);
          pyz = radyz(strike[is], dip[is], rake[is]);
          pxz = radxz(strike[is], dip[is], rake[is]);
          for ( iw = 0; iw < 8; iw++ ){

  	        weight = 1.0;
            if (  (iw%2) == 0 ){
              i = ixhypo[is];
	          weight = (1.0 - xweight[is]);
            } else {
              i = ixhypo[is] + 1;
	          weight = xweight[is];
            }
            if ( (iw%4) <= 1 ){
              j = iyhypo[is];
	          weight = weight*(1.0 - yweight[is]);
            } else {
              j = iyhypo[is] + 1;
	          weight = weight*yweight[is];
            }
            if ( iw < 4 ){
              k = izhypo[is];
	          weight = weight*(1.0 - zweight[is]);
            } else {
              k = izhypo[is] + 1;
	          weight = weight*zweight[is];
            }

	        icpu = i2icpu_array[i];
	        imp = i2imp_array[i];

            jcpu1 = j2jcpu_array[j-1];
            jcpu2 = j2jcpu_array[j];
            jcpu3 = j2jcpu_array[j+1];

            jmp1 = j2jmp_array[j-1];
            jmp2 = j2jmp_array[j];
            jmp3 = j2jmp_array[j+1];

            if ( coords[0] == icpu ){

              if ( coords[1] == jcpu3 ) fx[imp][jmp3][k]   += 0.5 * mo * pxy * weight;
              if ( coords[1] == jcpu1 ) fx[imp][jmp1][k]   -= 0.5 * mo * pxy * weight;
              if ( coords[1] == jcpu2 ) fx[imp][jmp2][k+1] += 0.5 * mo * pxz * weight;
              if ( coords[1] == jcpu2 ) fx[imp][jmp2][k-1] -= 0.5 * mo * pxz * weight;
              if ( coords[1] == jcpu2 ) fy[imp][jmp2][k]   += 0.5 * mo * pxy * weight;
              if ( coords[1] == jcpu1 ) fy[imp][jmp1][k]   += 0.5 * mo * pxy * weight;
              if ( coords[1] == jcpu2 ) fy[imp][jmp2][k]   += 0.5 * mo * pyy * weight;
              if ( coords[1] == jcpu1 ) fy[imp][jmp1][k]   -= 0.5 * mo * pyy * weight;
              if ( coords[1] == jcpu2 ) fy[imp][jmp2][k+1] += 0.125 * mo * pyz * weight;
              if ( coords[1] == jcpu1 ) fy[imp][jmp1][k+1] += 0.125 * mo * pyz * weight;
              if ( coords[1] == jcpu2 ) fy[imp][jmp2][k-1] -= 0.125 * mo * pyz * weight;
              if ( coords[1] == jcpu1 ) fy[imp][jmp1][k-1] -= 0.125 * mo * pyz * weight;
              if ( coords[1] == jcpu2 ) fz[imp][jmp2][k]   += 0.5 * mo * pxz * weight;
              if ( coords[1] == jcpu2 ) fz[imp][jmp2][k-1] += 0.5 * mo * pxz * weight;
              if ( coords[1] == jcpu3 ) fz[imp][jmp3][k]   += 0.125 * mo * pyz * weight;
              if ( coords[1] == jcpu3 ) fz[imp][jmp3][k-1] += 0.125 * mo * pyz * weight;
              if ( coords[1] == jcpu1 ) fz[imp][jmp1][k]   -= 0.125 * mo * pyz * weight;
              if ( coords[1] == jcpu1 ) fz[imp][jmp1][k-1] -= 0.125 * mo * pyz * weight;
              if ( coords[1] == jcpu2 ) fz[imp][jmp2][k]   += 0.5 * mo * pzz * weight;
              if ( coords[1] == jcpu2 ) fz[imp][jmp2][k-1] -= 0.5 * mo * pzz * weight;
	        }

	        icpu = i2icpu_array[ i-1];
            imp = i2imp_array[i-1];

            if ( coords[0] == icpu ){

              if ( coords[1] == jcpu2 ) fy[imp][jmp2][k]   -= 0.5 * mo * pxy * weight;
              if ( coords[1] == jcpu1 ) fy[imp][jmp1][k]   -= 0.5 * mo * pxy * weight;
              if ( coords[1] == jcpu2 ) fy[imp][jmp2][k]   += 0.5 * mo * pyy * weight;
              if ( coords[1] == jcpu1 ) fy[imp][jmp1][k]   -= 0.5 * mo * pyy * weight;
              if ( coords[1] == jcpu2 ) fy[imp][jmp2][k+1] += 0.125 * mo * pyz * weight;
              if ( coords[1] == jcpu1 ) fy[imp][jmp1][k+1] += 0.125 * mo * pyz * weight;
              if ( coords[1] == jcpu2 ) fy[imp][jmp2][k-1] -= 0.125 * mo * pyz * weight;
              if ( coords[1] == jcpu1 ) fy[imp][jmp1][k-1] -= 0.125 * mo * pyz * weight;
              if ( coords[1] == jcpu2 ) fx[imp][jmp2][k]   -= 0.5 * mo * pxx * weight;
              if ( coords[1] == jcpu3 ) fz[imp][jmp3][k]   += 0.125 * mo * pyz * weight;
              if ( coords[1] == jcpu3 ) fz[imp][jmp3][k-1] += 0.125 * mo * pyz * weight;
              if ( coords[1] == jcpu1 ) fz[imp][jmp1][k]   -= 0.125 * mo * pyz * weight;
              if ( coords[1] == jcpu1 ) fz[imp][jmp1][k-1] -= 0.125 * mo * pyz * weight;
              if ( coords[1] == jcpu2 ) fz[imp][jmp2][k]   += 0.5 * mo * pzz * weight;
              if ( coords[1] == jcpu2 ) fz[imp][jmp2][k-1] -= 0.5 * mo * pzz * weight;
              if ( coords[1] == jcpu2 ) fz[imp][jmp2][k]   -= 0.5 * mo * pxz * weight;
              if ( coords[1] == jcpu2 ) fz[imp][jmp2][k-1] -= 0.5 * mo * pxz * weight;
	        }

	        icpu = i2icpu_array[ i+1];
            imp = i2imp_array[i+1];

            if ( coords[0] == icpu ){

              if ( coords[1] == jcpu2 ) fx[imp][jmp2][k] += 0.5 * mo * pxx * weight;
	        }

          } /* end of iw (weighting) */
  	    } /* end of insrc */
      } /* end of is (each source) */
    } /* end of if */

    #if (TIMING_BARRIER)
      MPI_Barrier(MPI_COMM_WORLD);
      timing1 = my_second();
    #endif

    #if (TIMING)
      timing1 = my_second();
    #endif

				/* Calculation */
                /* First step : t = l + 1/2 for stress */

    npml = 0;

    #if (TAUGLOBAL)
      TAU_PROFILE_START(compute_sig);
    #endif

/* imode : to increase the velocity of the computation, we begin by computing
   the values of stress at the boundaries of the px * py parts of the array
   Afterwise, we can compute the values of the stress in the middle */

    for (imode=1;imode<=5;imode++){

	  if ( imode == 1 ){
	    mpmx_debut = 1;
	    mpmx_fin = 3;
	    mpmy_debut = 1;
	    mpmy_fin = mpmy;
	  }

      if ( imode == 2 ){
	    mpmx_debut = mpmx-2;
	    mpmx_fin = mpmx;
	    mpmy_debut = 1;
	    mpmy_fin = mpmy;
	  }

	  if ( imode == 3 ){
	    mpmy_debut = 1;
	    mpmy_fin = 3;
	    mpmx_debut = 4;
	    mpmx_fin = mpmx-3;
	  }

      if ( imode == 4 ){
	    mpmy_debut = mpmy-2;
	    mpmy_fin = mpmy;
	    mpmx_debut = 4;
	    mpmx_fin = mpmx-3;
	  }

      if ( imode == 5 ){ /* imode = 5 --> middle of each part of the array */
	    mpmx_debut = 4;
	    mpmx_fin = mpmx-3;
	    mpmy_debut = 4;
	    mpmy_fin = mpmy-3;

                /* Communication : first method */

        #if (PERSISTANT)

          if ( nest != -1 )
            MPI_Startall(6,&sendreq[1]);

          if ( nouest != -1 )
            MPI_Startall(6,&recvreq[1]);

          if ( nouest != -1 )
            MPI_Startall(6,&sendreq[7]);

          if ( nest != -1 )
            MPI_Startall(6,&recvreq[7]);

                /* Communication North - South */
                /* Positive direction */

          if( coords[1] != (py-1) ){
            i = 0;
            for ( jmp = mpmy-1; jmp <= mpmy; jmp++ ){
	          for( imp = -1; imp <= mpmx+2; imp++ ){
	            for( k = ZMIN-delta; k <= 1; k++ ){
	              sxxbuf3[i] = txx0[imp][jmp][k];
	              syybuf3[i] = tyy0[imp][jmp][k];
	              szzbuf3[i] = tzz0[imp][jmp][k];
	              sxybuf3[i] = txy0[imp][jmp][k];
	              syzbuf3[i] = tyz0[imp][jmp][k];
	              sxzbuf3[i] = txz0[imp][jmp][k];
	              i = i + 1;
	            }
	          }
            }
          }

				/* Negative direction */

          if( coords[1] != 0 ){
            i = 0;
            for ( jmp = 1; jmp <= 2; jmp++ ){
        	  for( imp = -1; imp <= mpmx+2; imp++ ){
                for( k = ZMIN-delta; k <= 1; k++ ){
                  sxxbuf4[i] = txx0[imp][jmp][k];
                  syybuf4[i] = tyy0[imp][jmp][k];
                  szzbuf4[i] = tzz0[imp][jmp][k];
                  sxybuf4[i] = txy0[imp][jmp][k];
                  syzbuf4[i] = tyz0[imp][jmp][k];
                  sxzbuf4[i] = txz0[imp][jmp][k];
                  i = i + 1;
                }
              }
            }
          }

          if( coords[1] != (py-1) )
            MPI_Start(&sendreq3[1]);

          if( coords[1] != 0 )
            MPI_Start(&recvreq3[1]);

          if( coords[1] != 0 )
            MPI_Start(&sendreq4[1]);

          if( coords[1] != (py-1) )
            MPI_Start(&recvreq4[1]);

        #endif /* end of first method */

                /* Communication : second method */

        #if (NONBLOCKING)

                /* Communication East - West */
                /* Positive direction */

          if( coords[0] != (px-1) ){
            i = 0;
            for ( imp = mpmx-1; imp <= mpmx; imp++ ){
	          for( jmp = -1; jmp <= mpmy+2; jmp++ ){
	            for( k = ZMIN-delta; k <= 1; k++ ){
	              sxxbuf[i] = txx0[imp][jmp][k];
	              syybuf[i] = tyy0[imp][jmp][k];
	              szzbuf[i] = tzz0[imp][jmp][k];
	              sxybuf[i] = txy0[imp][jmp][k];
	              syzbuf[i] = tyz0[imp][jmp][k];
	              sxzbuf[i] = txz0[imp][jmp][k];
	              i = i + 1;
	            }
	          }
            }
          }

          if( coords[0] != (px-1) )
            MPI_Isend(sxxbuf, 6*nmaxx, MPI_DOUBLE, nest,
	  	     3, MPI_COMM_WORLD, &sendreq[1]);

				/* Negative direction */

          if( coords[0] != 0 ){
            i = 0;
            for ( imp = 1; imp <= 2; imp++ ){
	          for( jmp = -1; jmp <= mpmy+2; jmp++ ){
                for( k = ZMIN-delta; k <= 1; k++ ){
                  sxxbuf2[i] = txx0[imp][jmp][k];
                  syybuf2[i] = tyy0[imp][jmp][k];
                  szzbuf2[i] = tzz0[imp][jmp][k];
                  sxybuf2[i] = txy0[imp][jmp][k];
                  syzbuf2[i] = tyz0[imp][jmp][k];
                  sxzbuf2[i] = txz0[imp][jmp][k];
                  i = i + 1;
                }
              }
            }
          }

          if( coords[0] != 0 )
            MPI_Isend(sxxbuf2, 6*nmaxx, MPI_DOUBLE, nouest,
             4, MPI_COMM_WORLD, &sendreq[2]);

/* Receveur non bloquant */

          if( coords[0] != 0 )
            MPI_Irecv(rxxbuf, 6*nmaxx, MPI_DOUBLE, nouest,
             3, MPI_COMM_WORLD,&recvreq[1]);

          if( coords[0] != (px-1) )
            MPI_Irecv(rxxbuf2, 6*nmaxx, MPI_DOUBLE, nest,
             4, MPI_COMM_WORLD, &recvreq[2]);

                /* Communication North - South */
                /* Positive direction */

          if( coords[1] != (py-1) ){
            i = 0;
            for ( jmp = mpmy-1; jmp <= mpmy; jmp++ ){
	          for( imp = -1; imp <= mpmx+2; imp++ ){
	            for( k = ZMIN-delta; k <= 1; k++ ){
	              sxxbuf3[i] = txx0[imp][jmp][k];
	              syybuf3[i] = tyy0[imp][jmp][k];
	              szzbuf3[i] = tzz0[imp][jmp][k];
	              sxybuf3[i] = txy0[imp][jmp][k];
	              syzbuf3[i] = tyz0[imp][jmp][k];
	              sxzbuf3[i] = txz0[imp][jmp][k];
	              i = i + 1;
	            }
	          }
            }
          }

          if( coords[1] != (py-1) )
            MPI_Isend(sxxbuf3, 6*nmaxy, MPI_DOUBLE, nnord,
	    	 13, MPI_COMM_WORLD, &sendreq[3]);

				/* Negative direction */

          if( coords[1] != 0 ){
            i = 0;
            for ( jmp = 1; jmp <= 2; jmp++ ){
	          for( imp = -1; imp <= mpmx+2; imp++ ){
                for( k = ZMIN-delta; k <= 1; k++ ){
                  sxxbuf4[i] = txx0[imp][jmp][k];
                  syybuf4[i] = tyy0[imp][jmp][k];
                  szzbuf4[i] = tzz0[imp][jmp][k];
                  sxybuf4[i] = txy0[imp][jmp][k];
                  syzbuf4[i] = tyz0[imp][jmp][k];
                  sxzbuf4[i] = txz0[imp][jmp][k];
                  i = i + 1;
                }
              }
            }
          }

          if( coords[1] != 0 )
            MPI_Isend(sxxbuf4, 6*nmaxy, MPI_DOUBLE, nsud,
             14, MPI_COMM_WORLD, &sendreq[4]);

/* Receveur non bloquant */

          if( coords[1] != 0 )
            MPI_Irecv(rxxbuf3, 6*nmaxy, MPI_DOUBLE, nsud,
             13, MPI_COMM_WORLD, &recvreq[4]);

          if( coords[1] != (py-1) )
            MPI_Irecv(rxxbuf4, 6*nmaxy, MPI_DOUBLE, nnord,
             14, MPI_COMM_WORLD, &recvreq[3]);

        #endif /* end of second method */

	  } /* end of imode = 5 */

                /* Beginning of stress computation */

      for ( i = mpmx_debut; i <= mpmx_fin; i++ ){
 	    imp = imp2i_array[i];
	    if ( ( imp>=XMIN-delta+1) && (imp<=XMAX+delta+1)){

          for ( j = mpmy_debut; j <= mpmy_fin; j++ ){
 	        jmp = jmp2j_array[j];
	        if ( ( jmp>=YMIN-delta+1) && (jmp<=YMAX+delta+1)){

		      for ( k = ZMIN-delta+1; k <= 1; k++){

                /* 4th order finite-difference */

                if ( NDIM == 4 ){

                /* CPML */

		          if ( imp <= XMIN+2 || imp >= XMAX || jmp <= YMIN+2 || jmp >= YMAX || k <= ZMIN+2 ){
  	                npml += 1;

                /* Calculation of txx, tyy and tzz */

  	                if ( k >= ZMIN-delta+2 && jmp >= YMIN-delta+2 && imp <= XMAX+delta ){

                      mux = 2./(1./mu[i][j][k] + 1./mu[i+1][j][k]);
			          lamx = 2./(1./lam[i][j][k] + 1./lam[i+1][j][k]);
			          vpx = sqrt ( (lamx+2.*mux)/( 0.5*( rho[i][j][k] + rho[i+1][j][k] ) ) );

			          if ( k == 1 ){ /* free surface */

			            b1 = 4. * mux * (lamx + mux) / (lamx + 2.*mux);
			            b2 = 2. * mux * lamx / (lamx + 2.*mux);

			            phixdum = phivxx[npml];
			            phiydum = phivyy[npml];

			            phivxx[npml] = CPML2 (vpx, dumpx2[i], alphax2[i], kappax2[i], phixdum, ds, dt,
                        vx0[i][j][k], vx0[i+1][j][k] );
                        phivyy[npml] = CPML2 (vpx, dumpy[j], alphay[j], kappay[j], phiydum, ds, dt,
                        vy0[i][j-1][k], vy0[i][j][k] );

			            txx0[i][j][k] += b1*dt*phivxx[npml] + b2*dt*phivyy[npml]
			            + b1*dt*(vx0[i+1][j][k]-vx0[i][j][k])/(kappax2[i]*ds)
			            + b2*dt*(vy0[i][j][k] - vy0[i][j-1][k])/(kappay[j]*ds);

			            tyy0[i][j][k] += b1*dt*phivyy[npml] + b2*dt*phivxx[npml]
			            + b1*dt*(vy0[i][j][k]-vy0[i][j-1][k])/(kappay[j]*ds)
			            + b2*dt*(vx0[i+1][j][k]-vx0[i][j][k])/(kappax2[i]*ds);

                        tzz0[i][j][k] = 0;

			          } else if ( k == 0 ){ /* in the first cell, 2nd order finite-difference instead of 4th order finite-difference */

			            phixdum = phivxx[npml];
			            phiydum = phivyy[npml];
			            phizdum = phivzz[npml];

			            phivxx[npml] = CPML2 (vpx, dumpx2[i], alphax2[i], kappax2[i], phixdum, ds, dt,
                        vx0[i][j][k], vx0[i+1][j][k] );
                        phivyy[npml] = CPML2 (vpx, dumpy[j], alphay[j], kappay[j], phiydum, ds, dt,
                        vy0[i][j-1][k], vy0[i][j][k] );
                        phivzz[npml] = CPML2 (vpx, dumpz[k], alphaz[k], kappaz[k], phizdum, ds, dt,
                        vz0[i][j][k-1], vz0[i][j][k] );

                        txx0[i][j][k] += dt*(lamx + 2.*mux)*phivxx[npml] + dt*lamx*( phivyy[npml] + phivzz[npml] )
                        + staggards2 (lamx, mux, kappax2[i], kappay[j], kappaz[k], dt, ds,
                        vx0[i][j][k], vx0[i+1][j][k],
                        vy0[i][j-1][k], vy0[i][j][k],
                        vz0[i][j][k-1], vz0[i][j][k] );

                        tyy0[i][j][k] += dt*lamx*( phivxx[npml] + phivzz[npml] ) + dt*(lamx + 2.*mux)*phivyy[npml]
                        + staggards2 (lamx, mux, kappay[j], kappax2[i], kappaz[k], dt, ds,
                        vy0[i][j-1][k], vy0[i][j][k],
                        vx0[i][j][k], vx0[i+1][j][k],
                        vz0[i][j][k-1], vz0[i][j][k] );

                        tzz0[i][j][k] += dt*lamx*( phivxx[npml] + phivyy[npml] ) + dt*(lamx + 2.*mux)*phivzz[npml]
                        + staggards2 (lamx, mux, kappaz[k], kappax2[i], kappay[j], dt, ds,
                        vz0[i][j][k-1], vz0[i][j][k],
                        vx0[i][j][k], vx0[i+1][j][k],
                        vy0[i][j-1][k], vy0[i][j][k] );

			          } else { /* regular domain */

			            phixdum = phivxx[npml];
			            phiydum = phivyy[npml];
			            phizdum = phivzz[npml];

			            phivxx[npml] = CPML4 (vpx, dumpx2[i], alphax2[i], kappax2[i], phixdum, ds, dt,
                        vx0[i][j][k], vx0[i+1][j][k],
                        vx0[i-1][j][k], vx0[i+2][j][k] );
                        phivyy[npml] = CPML4 (vpx, dumpy[j], alphay[j], kappay[j], phiydum, ds, dt,
                        vy0[i][j-1][k], vy0[i][j][k],
                        vy0[i][j-2][k], vy0[i][j+1][k] );
                        phivzz[npml] = CPML4 (vpx, dumpz[k], alphaz[k], kappaz[k], phizdum, ds, dt,
                        vz0[i][j][k-1], vz0[i][j][k],
                        vz0[i][j][k-2], vz0[i][j][k+1] );

                        txx0[i][j][k] += dt*(lamx + 2.*mux)*phivxx[npml] + dt*lamx*( phivyy[npml] + phivzz[npml] )
                        + staggards4 (lamx, mux, kappax2[i], kappay[j], kappaz[k], dt, ds,
                        vx0[i][j][k], vx0[i+1][j][k],
                        vx0[i-1][j][k], vx0[i+2][j][k],
                        vy0[i][j-1][k], vy0[i][j][k],
                        vy0[i][j-2][k], vy0[i][j+1][k],
                        vz0[i][j][k-1], vz0[i][j][k],
                        vz0[i][j][k-2], vz0[i][j][k+1] );

                        tyy0[i][j][k] += dt*lamx*( phivxx[npml] + phivzz[npml] ) + dt*(lamx + 2.*mux)*phivyy[npml]
                        + staggards4 (lamx, mux, kappay[j], kappax2[i], kappaz[k], dt, ds,
                        vy0[i][j-1][k], vy0[i][j][k],
                        vy0[i][j-2][k], vy0[i][j+1][k],
                        vx0[i][j][k], vx0[i+1][j][k],
                        vx0[i-1][j][k], vx0[i+2][j][k],
                        vz0[i][j][k-1], vz0[i][j][k],
                        vz0[i][j][k-2], vz0[i][j][k+1] );

                        tzz0[i][j][k] += dt*lamx*( phivxx[npml] + phivyy[npml] ) + dt*(lamx + 2.*mux)*phivzz[npml]
                        + staggards4 (lamx, mux, kappaz[k], kappax2[i], kappay[j], dt, ds,
                        vz0[i][j][k-1], vz0[i][j][k],
                        vz0[i][j][k-2], vz0[i][j][k+1],
                        vx0[i][j][k], vx0[i+1][j][k],
                        vx0[i-1][j][k], vx0[i+2][j][k],
                        vy0[i][j-1][k], vy0[i][j][k],
                        vy0[i][j-2][k], vy0[i][j+1][k] );

			          } /* end of if "free surface" */
		            } /* end of calculation of txx, tyy and tzz */

                /* Calculation of txy */

                    if ( jmp <= YMAX+delta && imp >= XMIN-delta+2 ){

                      muy = 2./(1./mu[i][j][k] + 1./mu[i][j+1][k]);
                      lamy = 2./(1./lam[i][j][k] + 1./lam[i][j+1][k]);
                      vpy = sqrt ( (lamy+2.*muy)/( 0.5*( rho[i][j][k] + rho[i][j+1][k] ) ) );

                      if ( k >= 0 ){ /* in the first cell, 2nd order finite-difference instead of 4th order finite-difference */

                        phixdum = phivyx[npml];
                        phiydum = phivxy[npml];

                        phivyx[npml] = CPML2 (vpy, dumpx[i], alphax[i], kappax[i], phixdum, ds, dt,
                        vy0[i-1][j][k], vy0[i][j][k] );
                        phivxy[npml] = CPML2 (vpy, dumpy2[j], alphay2[j], kappay2[j], phiydum, ds, dt,
                        vx0[i][j][k], vx0[i][j+1][k] );

                        txy0[i][j][k] += dt*muy*( phivyx[npml] + phivxy[npml] )
                        + staggardt2 (muy, kappax[i], kappay2[j], dt, ds,
                        vy0[i-1][j][k], vy0[i][j][k],
                        vx0[i][j][k], vx0[i][j+1][k] );

			          } else { /* regular domain */

                        phixdum = phivyx[npml];
                        phiydum = phivxy[npml];

                        phivyx[npml] = CPML4 (vpy, dumpx[i], alphax[i], kappax[i], phixdum, ds, dt,
                        vy0[i-1][j][k], vy0[i][j][k],
                        vy0[i-2][j][k], vy0[i+1][j][k] );
                        phivxy[npml] = CPML4 (vpy, dumpy2[j], alphay2[j], kappay2[j], phiydum, ds, dt,
                        vx0[i][j][k], vx0[i][j+1][k],
                        vx0[i][j-1][k], vx0[i][j+2][k] );

                        txy0[i][j][k] += dt*muy*( phivyx[npml] + phivxy[npml] )
                        + staggardt4 (muy, kappax[i], kappay2[j], dt, ds,
                        vy0[i-1][j][k], vy0[i][j][k],
                        vy0[i-2][j][k], vy0[i+1][j][k],
                        vx0[i][j][k], vx0[i][j+1][k],
                        vx0[i][j-1][k], vx0[i][j+2][k] );

	                  } /* end of if "free surface" */
	                } /* end of calculation of txy */

                /* Calculation of txz */

                    if ( k <= 0 && imp >= XMIN-delta+2 ){

                      muz = 2./(1./mu[i][j][k] + 1./mu[i][j][k+1]);
                      lamz = 2./(1./lam[i][j][k] + 1./lam[i][j][k+1]);
                      vpz = sqrt ( (lamz+2.*muz)/( 0.5*( rho[i][j][k] + rho[i][j][k+1] ) ) );

                      if ( k == 0 ){ /* in the first cell, 2nd order finite-difference instead of 4th order finite-difference */

                        phixdum = phivzx[npml];
                        phizdum = phivxz[npml];

                        phivzx[npml] = CPML2 (vpz, dumpx[i], alphax[i], kappax[i], phixdum, ds, dt,
                        vz0[i-1][j][k], vz0[i][j][k] );
                        phivxz[npml] = CPML2 (vpz, dumpz2[k], alphaz2[k], kappaz2[k], phizdum, ds, dt,
                        vx0[i][j][k], vx0[i][j][k+1] );

                        txz0[i][j][k] += dt*muz*( phivzx[npml] + phivxz[npml] )
                        + staggardt2 (muz, kappax[i], kappaz2[k], dt, ds,
                        vz0[i-1][j][k], vz0[i][j][k],
                        vx0[i][j][k], vx0[i][j][k+1] );

			          } else { /* regular domain */

                        phixdum = phivzx[npml];
                        phizdum = phivxz[npml];

                        phivzx[npml] = CPML4 (vpz, dumpx[i], alphax[i], kappax[i], phixdum, ds, dt,
                        vz0[i-1][j][k], vz0[i][j][k],
                        vz0[i-2][j][k], vz0[i+1][j][k] );
                        phivxz[npml] = CPML4 (vpz, dumpz2[k], alphaz2[k], kappaz2[k], phizdum, ds, dt,
                        vx0[i][j][k], vx0[i][j][k+1],
                        vx0[i][j][k-1], vx0[i][j][k+2] );

                        txz0[i][j][k] += dt*muz*( phivzx[npml] + phivxz[npml] )
                        + staggardt4 (muz, kappax[i], kappaz2[k], dt, ds,
                        vz0[i-1][j][k], vz0[i][j][k],
                        vz0[i-2][j][k], vz0[i+1][j][k],
                        vx0[i][j][k], vx0[i][j][k+1],
                        vx0[i][j][k-1], vx0[i][j][k+2] );

		              } /* end of if "free surface" */
	                } /* end of calculation of txz */

                /* Calculation of tyz */

                    if ( k <= 0 && jmp <= YMAX+delta ){

                      muxyz = 8./(1./mu[i][j][k] + 1./mu[i][j][k+1]
			                    + 1./mu[i][j+1][k] + 1./mu[i][j+1][k+1]
			    		        + 1./mu[i+1][j][k] + 1./mu[i+1][j][k+1]
			  	                + 1./mu[i+1][j+1][k] + 1./mu[i+1][j+1][k+1]);
                      lamxyz = 8./(1./lam[i][j][k] + 1./lam[i][j][k+1]
			                     + 1./lam[i][j+1][k] + 1./lam[i][j+1][k+1]
			    		         + 1./lam[i+1][j][k] + 1./lam[i+1][j][k+1]
			  	                 + 1./lam[i+1][j+1][k] + 1./lam[i+1][j+1][k+1]);
                      vpxyz = sqrt ( (lamxyz+2.*muxyz)/( 0.125*( rho[i][j][k] + rho[i][j][k+1]
			                                                   + rho[i][j+1][k] + rho[i][j+1][k+1]
			  			                                       + rho[i+1][j][k] + rho[i+1][j][k+1]
				                                               + rho[i+1][j+1][k] + rho[i+1][j+1][k+1] ) ) );

                      if ( k == 0 ){ /* in the first cell, 2nd order finite-difference instead of 4th order finite-difference */

                        phiydum = phivzy[npml];
                        phizdum = phivyz[npml];

                        phivzy[npml] = CPML2 (vpxyz, dumpy2[j], alphay2[j], kappay2[j], phiydum, ds, dt,
                        vz0[i][j][k], vz0[i][j+1][k] );
                        phivyz[npml] = CPML2 (vpxyz, dumpz2[k], alphaz2[k], kappaz2[k], phizdum, ds, dt,
                        vy0[i][j][k], vy0[i][j][k+1] );

                        tyz0[i][j][k] += dt*muxyz*( phivzy[npml] + phivyz[npml] )
                        + staggardt2 (muxyz, kappay2[j], kappaz2[k], dt, ds,
                        vz0[i][j][k], vz0[i][j+1][k],
                        vy0[i][j][k], vy0[i][j][k+1] );

			          } else { /* regular domain */

                        phiydum = phivzy[npml];
                        phizdum = phivyz[npml];

                        phivzy[npml] = CPML4 (vpxyz, dumpy2[j], alphay2[j], kappay2[j], phiydum, ds, dt,
                        vz0[i][j][k], vz0[i][j+1][k],
                        vz0[i][j-1][k], vz0[i][j+2][k] );
                        phivyz[npml] = CPML4 (vpxyz, dumpz2[k], alphaz2[k], kappaz2[k], phizdum, ds, dt,
                        vy0[i][j][k], vy0[i][j][k+1],
                        vy0[i][j][k-1], vy0[i][j][k+2] );

                        tyz0[i][j][k] += dt*muxyz*( phivzy[npml] + phivyz[npml] )
                        + staggardt4 (muxyz, kappay2[j], kappaz2[k], dt, ds,
                        vz0[i][j][k], vz0[i][j+1][k],
                        vz0[i][j-1][k], vz0[i][j+2][k],
                        vy0[i][j][k], vy0[i][j][k+1],
                        vy0[i][j][k-1], vy0[i][j][k+2] );

	                  } /* end of if "free surface" */
	                } /* end of calculation of tyz */

                /* txz and tyz are antisymetric */

                    if ( k == 1 ){
				      txz0[i][j][1] = - txz0[i][j][0];
				      tyz0[i][j][1] = - txz0[i][j][0];
			        }

                /* Normal mode */

	              } else {

                    mux = 2./(1./mu[i][j][k] + 1./mu[i+1][j][k]);
			        lamx = 2./(1./lam[i][j][k] + 1./lam[i+1][j][k]);
			        muy = 2./(1./mu[i][j][k] + 1./mu[i][j+1][k]);
			        muz = 2./(1./mu[i][j][k] + 1./mu[i][j][k+1]);
			        muxyz = 8./(1./mu[i][j][k] + 1./mu[i][j][k+1]
			                  + 1./mu[i][j+1][k] + 1./mu[i][j+1][k+1]
			  		          + 1./mu[i+1][j][k] + 1./mu[i+1][j][k+1]
				              + 1./mu[i+1][j+1][k] + 1./mu[i+1][j+1][k+1]);

                    if ( k == 1 ){ /* free surface */

			          b1 = 4. * mux * (lamx + mux) / (lamx + 2.*mux);
			          b2 = 2. * mux * lamx / (lamx + 2.*mux);

			          txx0[i][j][k] += b1*dt*(vx0[i+1][j][k]-vx0[i][j][k])/ds
			          + b2*dt*(vy0[i][j][k] - vy0[i][j-1][k])/ds;

			          tyy0[i][j][k] += b1*dt*(vy0[i][j][k]-vy0[i][j-1][k])/ds
			          + b2*dt*(vx0[i+1][j][k]-vx0[i][j][k])/ds;

			          tzz0[i][j][k] = 0;

			          txy0[i][j][k] += staggardt2 (muy, un, un, dt, ds,
			      	vy0[i-1][j][k], vy0[i][j][k],
                      vx0[i][j][k], vx0[i][j+1][k] );

		            } else if ( k == 0 ){ /* in the first cell, 2nd order finite-difference instead of 4th order finite-difference */

                      txx0[i][j][k] += staggards2 (lamx, mux, un, un, un, dt, ds,
                      vx0[i][j][k], vx0[i+1][j][k],
                      vy0[i][j-1][k], vy0[i][j][k],
                      vz0[i][j][k-1], vz0[i][j][k] );

                      tyy0[i][j][k] += staggards2 (lamx, mux, un, un, un, dt, ds,
                      vy0[i][j-1][k], vy0[i][j][k],
                      vx0[i][j][k], vx0[i+1][j][k],
                      vz0[i][j][k-1], vz0[i][j][k] );

                      tzz0[i][j][k] += staggards2 (lamx, mux, un, un, un, dt, ds,
                      vz0[i][j][k-1], vz0[i][j][k],
                      vx0[i][j][k], vx0[i+1][j][k],
                      vy0[i][j-1][k], vy0[i][j][k] );

                      txy0[i][j][k] += staggardt2 (muy, un, un, dt, ds,
                      vy0[i-1][j][k], vy0[i][j][k],
                      vx0[i][j][k], vx0[i][j+1][k] );

                      txz0[i][j][k] += staggardt2 (muz, un, un, dt, ds,
                      vz0[i-1][j][k], vz0[i][j][k],
                      vx0[i][j][k], vx0[i][j][k+1] );

                      tyz0[i][j][k] += staggardt2 (muxyz, un, un, dt, ds,
                      vz0[i][j][k], vz0[i][j+1][k],
                      vy0[i][j][k], vy0[i][j][k+1] );

		            } else { /* regular domain */

                      txx0[i][j][k] += staggards4 (lamx, mux, un, un, un, dt, ds,
                      vx0[i][j][k], vx0[i+1][j][k],
                      vx0[i-1][j][k], vx0[i+2][j][k],
                      vy0[i][j-1][k], vy0[i][j][k],
                      vy0[i][j-2][k], vy0[i][j+1][k],
                      vz0[i][j][k-1], vz0[i][j][k],
                      vz0[i][j][k-2], vz0[i][j][k+1] );

                      tyy0[i][j][k] += staggards4 (lamx, mux, un, un, un, dt, ds,
                      vy0[i][j-1][k], vy0[i][j][k],
                      vy0[i][j-2][k], vy0[i][j+1][k],
                      vx0[i][j][k], vx0[i+1][j][k],
                      vx0[i-1][j][k], vx0[i+2][j][k],
                      vz0[i][j][k-1], vz0[i][j][k],
                      vz0[i][j][k-2], vz0[i][j][k+1] );

                      tzz0[i][j][k] += staggards4 (lamx, mux, un, un, un, dt, ds,
                      vz0[i][j][k-1], vz0[i][j][k],
                      vz0[i][j][k-2], vz0[i][j][k+1],
                      vx0[i][j][k], vx0[i+1][j][k],
                      vx0[i-1][j][k], vx0[i+2][j][k],
                      vy0[i][j-1][k], vy0[i][j][k],
                      vy0[i][j-2][k], vy0[i][j+1][k]);

                      txy0[i][j][k] += staggardt4 (muy, un, un, dt, ds,
                      vy0[i-1][j][k], vy0[i][j][k],
                      vy0[i-2][j][k], vy0[i+1][j][k],
                      vx0[i][j][k], vx0[i][j+1][k],
                      vx0[i][j-1][k], vx0[i][j+2][k] );

                      txz0[i][j][k] += staggardt4 (muz, un, un, dt, ds,
                      vz0[i-1][j][k], vz0[i][j][k],
                      vz0[i-2][j][k], vz0[i+1][j][k],
                      vx0[i][j][k], vx0[i][j][k+1],
                      vx0[i][j][k-1], vx0[i][j][k+2] );

                      tyz0[i][j][k] += staggardt4 (muxyz, un, un, dt, ds,
                      vz0[i][j][k], vz0[i][j+1][k],
                      vz0[i][j-1][k], vz0[i][j+2][k],
                      vy0[i][j][k], vy0[i][j][k+1],
                      vy0[i][j][k-1], vy0[i][j][k+2] );

			        } /* end of if "free surface" */

                /* txz and tyz are antisymetric */

			        if ( k == 1 ){
			  	      txz0[i][j][1] = - txz0[i][j][0];
			  	      tyz0[i][j][1] = - txz0[i][j][0];
			        }

		          } /* end of normal mode */

			    /* 2nd order finite-difference */

		        } else {

                /* CPML */

		          if ( imp <= XMIN+2 || imp >= XMAX || jmp <= YMIN+2 || jmp >= YMAX || k <= ZMIN+2 ){
  	                npml += 1;

                /* Calculation of txx, tyy and tzz */

  	                if ( k >= ZMIN-delta+2 && jmp >= YMIN-delta+2 && imp <= XMAX+delta ){

                      mux = 2./(1./mu[i][j][k] + 1./mu[i+1][j][k]);
			          lamx = 2./(1./lam[i][j][k] + 1./lam[i+1][j][k]);
			          vpx = sqrt ( (lamx+2.*mux)/( 0.5*( rho[i][j][k] + rho[i+1][j][k] ) ) );

			          if ( k == 1 ){ /* free surface */

			            b1 = 4. * mux * (lamx + mux) / (lamx + 2.*mux);
			            b2 = 2. * mux * lamx / (lamx + 2.*mux);

			            phixdum = phivxx[npml];
			            phiydum = phivyy[npml];

			            phivxx[npml] = CPML2 (vpx, dumpx2[i], alphax2[i], kappax2[i], phixdum, ds, dt,
                        vx0[i][j][k], vx0[i+1][j][k] );
                        phivyy[npml] = CPML2 (vpx, dumpy[j], alphay[j], kappay[j], phiydum, ds, dt,
                        vy0[i][j-1][k], vy0[i][j][k] );

			            txx0[i][j][k] += b1*dt*phivxx[npml] + b2*dt*phivyy[npml]
			            + b1*dt*(vx0[i+1][j][k]-vx0[i][j][k])/(kappax2[i]*ds)
			            + b2*dt*(vy0[i][j][k] - vy0[i][j-1][k])/(kappay[j]*ds);

			            tyy0[i][j][k] += b1*dt*phivyy[npml] + b2*dt*phivxx[npml]
			            + b1*dt*(vy0[i][j][k]-vy0[i][j-1][k])/(kappay[j]*ds)
			            + b2*dt*(vx0[i+1][j][k]-vx0[i][j][k])/(kappax2[i]*ds);

                        tzz0[i][j][k] = 0;

			          } else { /* regular domain */

			            phixdum = phivxx[npml];
			            phiydum = phivyy[npml];
			            phizdum = phivzz[npml];

			            phivxx[npml] = CPML2 (vpx, dumpx2[i], alphax2[i], kappax2[i], phixdum, ds, dt,
                        vx0[i][j][k], vx0[i+1][j][k] );
                        phivyy[npml] = CPML2 (vpx, dumpy[j], alphay[j], kappay[j], phiydum, ds, dt,
                        vy0[i][j-1][k], vy0[i][j][k] );
                        phivzz[npml] = CPML2 (vpx, dumpz[k], alphaz[k], kappaz[k], phizdum, ds, dt,
                        vz0[i][j][k-1], vz0[i][j][k] );

                        txx0[i][j][k] += dt*(lamx + 2.*mux)*phivxx[npml] + dt*lamx*( phivyy[npml] + phivzz[npml] )
                        + staggards2 (lamx, mux, kappax2[i], kappay[j], kappaz[k], dt, ds,
                        vx0[i][j][k], vx0[i+1][j][k],
                        vy0[i][j-1][k], vy0[i][j][k],
                        vz0[i][j][k-1], vz0[i][j][k] );

                        tyy0[i][j][k] += dt*lamx*( phivxx[npml] + phivzz[npml] ) + dt*(lamx + 2.*mux)*phivyy[npml]
                        + staggards2 (lamx, mux, kappay[j], kappax2[i], kappaz[k], dt, ds,
                        vy0[i][j-1][k], vy0[i][j][k],
                        vx0[i][j][k], vx0[i+1][j][k],
                        vz0[i][j][k-1], vz0[i][j][k] );

                        tzz0[i][j][k] += dt*lamx*( phivxx[npml] + phivyy[npml] ) + dt*(lamx + 2.*mux)*phivzz[npml]
                        + staggards2 (lamx, mux, kappaz[k], kappax2[i], kappay[j], dt, ds,
                        vz0[i][j][k-1], vz0[i][j][k],
                        vx0[i][j][k], vx0[i+1][j][k],
                        vy0[i][j-1][k], vy0[i][j][k] );

		              } /* end of if "free surface" */
		            } /* end of calculation of txx, tyy and tzz */

                /* Calculation of txy */

                    if ( jmp <= YMAX+delta && imp >= XMIN-delta+2 ){

                      muy = 2./(1./mu[i][j][k] + 1./mu[i][j+1][k]);
                      lamy = 2./(1./lam[i][j][k] + 1./lam[i][j+1][k]);
                      vpy = sqrt ( (lamy+2.*muy)/( 0.5*( rho[i][j][k] + rho[i][j+1][k] ) ) );

                      phixdum = phivyx[npml];
                      phiydum = phivxy[npml];

                      phivyx[npml] = CPML2 (vpy, dumpx[i], alphax[i], kappax[i], phixdum, ds, dt,
                      vy0[i-1][j][k], vy0[i][j][k] );
                      phivxy[npml] = CPML2 (vpy, dumpy2[j], alphay2[j], kappay2[j], phiydum, ds, dt,
                      vx0[i][j][k], vx0[i][j+1][k] );

                      txy0[i][j][k] += dt*muy*( phivyx[npml] + phivxy[npml] )
                      + staggardt2 (muy, kappax[i], kappay2[j], dt, ds,
                      vy0[i-1][j][k], vy0[i][j][k],
                      vx0[i][j][k], vx0[i][j+1][k] );

	                } /* end of calculation of txy */

                /* Calculation of txz */

                    if ( k <= 0 && imp >= XMIN-delta+2 ){

                      muz = 2./(1./mu[i][j][k] + 1./mu[i][j][k+1]);
                      lamz = 2./(1./lam[i][j][k] + 1./lam[i][j][k+1]);
                      vpz = sqrt ( (lamz+2.*muz)/( 0.5*( rho[i][j][k] + rho[i][j][k+1] ) ) );

                      phixdum = phivzx[npml];
                      phizdum = phivxz[npml];

                      phivzx[npml] = CPML2 (vpz, dumpx[i], alphax[i], kappax[i], phixdum, ds, dt,
                      vz0[i-1][j][k], vz0[i][j][k] );
                      phivxz[npml] = CPML2 (vpz, dumpz2[k], alphaz2[k], kappaz2[k], phizdum, ds, dt,
                      vx0[i][j][k], vx0[i][j][k+1] );

                      txz0[i][j][k] += dt*muz*( phivzx[npml] + phivxz[npml] )
                      + staggardt2 (muz, kappax[i], kappaz2[k], dt, ds,
                      vz0[i-1][j][k], vz0[i][j][k],
                      vx0[i][j][k], vx0[i][j][k+1] );

	                } /* end of calculation of txz */

                /* Calculation of tyz */

                    if ( k <= 0 && jmp <= YMAX+delta ){

                      muxyz = 8./(1./mu[i][j][k] + 1./mu[i][j][k+1]
			                    + 1./mu[i][j+1][k] + 1./mu[i][j+1][k+1]
			  			        + 1./mu[i+1][j][k] + 1./mu[i+1][j][k+1]
				                + 1./mu[i+1][j+1][k] + 1./mu[i+1][j+1][k+1]);
                      lamxyz = 8./(1./lam[i][j][k] + 1./lam[i][j][k+1]
			                     + 1./lam[i][j+1][k] + 1./lam[i][j+1][k+1]
			  			         + 1./lam[i+1][j][k] + 1./lam[i+1][j][k+1]
				                 + 1./lam[i+1][j+1][k] + 1./lam[i+1][j+1][k+1]);
                      vpxyz = sqrt ( (lamxyz+2.*muxyz)/( 0.125*( rho[i][j][k] + rho[i][j][k+1]
			                                                   + rho[i][j+1][k] + rho[i][j+1][k+1]
			  			                                       + rho[i+1][j][k] + rho[i+1][j][k+1]
				                                               + rho[i+1][j+1][k] + rho[i+1][j+1][k+1] ) ) );

                      phiydum = phivzy[npml];
                      phizdum = phivyz[npml];

                      phivzy[npml] = CPML2 (vpxyz, dumpy2[j], alphay2[j], kappay2[j], phiydum, ds, dt,
                      vz0[i][j][k], vz0[i][j+1][k] );
                      phivyz[npml] = CPML2 (vpxyz, dumpz2[k], alphaz2[k], kappaz2[k], phizdum, ds, dt,
                      vy0[i][j][k], vy0[i][j][k+1] );

                      tyz0[i][j][k] += dt*muxyz*( phivzy[npml] + phivyz[npml] )
                      + staggardt2 (muxyz, kappay2[j], kappaz2[k], dt, ds,
                      vz0[i][j][k], vz0[i][j+1][k],
                      vy0[i][j][k], vy0[i][j][k+1] );

	                } /* end of calculation of tyz */

                /* txz and tyz are antisymetric */

                    if ( k == 1 ){
			  	      txz0[i][j][1] = - txz0[i][j][0];
				      tyz0[i][j][1] = - txz0[i][j][0];
			        }

                 /* Normal mode */

	              } else {

                    mux = 2./(1./mu[i][j][k] + 1./mu[i+1][j][k]);
			        lamx = 2./(1./lam[i][j][k] + 1./lam[i+1][j][k]);
			        muy = 2./(1./mu[i][j][k] + 1./mu[i][j+1][k]);
			        muz = 2./(1./mu[i][j][k] + 1./mu[i][j][k+1]);
			        muxyz = 8./(1./mu[i][j][k] + 1./mu[i][j][k+1]
			                  + 1./mu[i][j+1][k] + 1./mu[i][j+1][k+1]
			  		          + 1./mu[i+1][j][k] + 1./mu[i+1][j][k+1]
				              + 1./mu[i+1][j+1][k] + 1./mu[i+1][j+1][k+1]);

                    if ( k == 1 ){ /* free surface */

			          b1 = 4. * mux * (lamx + mux) / (lamx + 2.*mux);
			          b2 = 2. * mux * lamx / (lamx + 2.*mux);

			          txx0[i][j][k] += b1*dt*(vx0[i+1][j][k]-vx0[i][j][k])/ds
			          + b2*dt*(vy0[i][j][k] - vy0[i][j-1][k])/ds;

			          tyy0[i][j][k] += b1*dt*(vy0[i][j][k]-vy0[i][j-1][k])/ds
			          + b2*dt*(vx0[i+1][j][k]-vx0[i][j][k])/ds;

			          tzz0[i][j][k] = 0;

			          txy0[i][j][k] += staggardt2 (muy, un, un, dt, ds,
				      vy0[i-1][j][k], vy0[i][j][k],
                      vx0[i][j][k], vx0[i][j+1][k] );

		            } else { /* regular domain */

                      txx0[i][j][k] += staggards2 (lamx, mux, un, un, un, dt, ds,
                      vx0[i][j][k], vx0[i+1][j][k],
                      vy0[i][j-1][k], vy0[i][j][k],
                      vz0[i][j][k-1], vz0[i][j][k] );

                      tyy0[i][j][k] += staggards2 (lamx, mux, un, un, un, dt, ds,
                      vy0[i][j-1][k], vy0[i][j][k],
                      vx0[i][j][k], vx0[i+1][j][k],
                      vz0[i][j][k-1], vz0[i][j][k] );

                      tzz0[i][j][k] += staggards2 (lamx, mux, un, un, un, dt, ds,
                      vz0[i][j][k-1], vz0[i][j][k],
                      vx0[i][j][k], vx0[i+1][j][k],
                      vy0[i][j-1][k], vy0[i][j][k] );

                      txy0[i][j][k] += staggardt2 (muy, un, un, dt, ds,
                      vy0[i-1][j][k], vy0[i][j][k],
                      vx0[i][j][k], vx0[i][j+1][k] );

                      txz0[i][j][k] += staggardt2 (muz, un, un, dt, ds,
                      vz0[i-1][j][k], vz0[i][j][k],
                      vx0[i][j][k], vx0[i][j][k+1] );

                      tyz0[i][j][k] += staggardt2 (muxyz, un, un, dt, ds,
                      vz0[i][j][k], vz0[i][j+1][k],
                      vy0[i][j][k], vy0[i][j][k+1] );

			        } /* end of if "free surface" */

                /* txz and tyz are antisymetric */

			        if ( k == 1 ){
			    	  txz0[i][j][1] = - txz0[i][j][0];
			  	      tyz0[i][j][1] = - txz0[i][j][0];
			        }

		          } /* end of normal mode */
		        } /* end of 2nd order finite-difference */
              } /* end of k */
            } /* end of if jmp */
          } /* end of j */
	    } /* end of if imp */
	  }	/* end of i */
	} /* end of imode */

    #if (TAUGLOBAL)
      TAU_PROFILE_STOP(compute_sig);
    #endif

    #if (TIMING_BARRIER)
      MPI_Barrier(MPI_COMM_WORLD);
      timing2 = my_second();
      timing_bc1 = timing_bc1 +  (timing2-timing1);
    #endif

    #if (TIMING)
      timing2 = my_second();
      timing_bc1 = timing_bc1 + (timing2-timing1);
    #endif

    #if (TIMING_BARRIER)
      MPI_Barrier(MPI_COMM_WORLD);
      timing1 = my_second();
    #endif

    #if (TIMING)
      timing1 = my_second();
    #endif

    #if (TAU)
      TAU_PROFILE_START(exchange_sig);
    #endif

    #if (TAUGLOBAL)
     TAU_PROFILE_START(exchange_sig);
    #endif

                /* Communication : first method */

    #if (PERSISTANT)

      if ( nest != -1 )
  	   MPI_Waitall(6,&sendreq[1],&sendstatus[1]);

      if ( nouest != -1 )
  	    MPI_Waitall(6,&recvreq[1],&recvstatus[1]);

      if ( nouest != -1 )
	    MPI_Waitall(6,&sendreq[7],&sendstatus[7]);

      if ( nest != -1 )
	    MPI_Waitall(6,&recvreq[7],&recvstatus[7]);

                /* North - South */

      if( coords[1] != (py-1) )
        MPI_Wait(&sendreq3[1],&sendstatus3[1]);

      if( coords[1] != 0 )
        MPI_Wait(&sendreq4[1],&sendstatus4[1]);

      if( coords[1] != (py-1) )
        MPI_Wait(&recvreq4[1],&recvstatus4[1]);

      if( coords[1] != 0 )
        MPI_Wait(&recvreq3[1],&recvstatus3[1]);

      if( coords[1] != 0 ){
        i = 0;
        for( jmp = -1; jmp <= 0; jmp++ ){
          for( imp = -1; imp <= mpmx+2; imp++ ){
            for( k = ZMIN-delta; k <= 1; k++ ){
              txx0[imp][jmp][k] = rxxbuf3[i];
              tyy0[imp][jmp][k] = ryybuf3[i];
              tzz0[imp][jmp][k] = rzzbuf3[i];
              txy0[imp][jmp][k] = rxybuf3[i];
              tyz0[imp][jmp][k] = ryzbuf3[i];
              txz0[imp][jmp][k] = rxzbuf3[i];
              i = i + 1;
            }
          }
        }
      }

      if( coords[1] != (py-1) ){
        i = 0;
        for( jmp = mpmy+1; jmp <= mpmy+2; jmp++ ){
          for( imp = -1; imp <= mpmx+2; imp++ ){
            for( k = ZMIN-delta; k <= 1; k++ ){
              txx0[imp][jmp][k] = rxxbuf4[i];
              tyy0[imp][jmp][k] = ryybuf4[i];
              tzz0[imp][jmp][k] = rzzbuf4[i];
              txy0[imp][jmp][k] = rxybuf4[i];
              tyz0[imp][jmp][k] = ryzbuf4[i];
              txz0[imp][jmp][k] = rxzbuf4[i];
              i = i + 1;
            }
          }
        }
      }

    #endif /* end of first method */

                /* Communication : second method */

    #if (NONBLOCKING)

                /* East - West */

      if( coords[0] != 0 )
        MPI_Wait(&recvreq[1],&recvstatus[1]);

      if( coords[0] != 0 ){
        i = 0;
        for( imp = -1; imp <= 0; imp++ ){
          for( jmp = -1; jmp <= mpmy+2; jmp++ ){
            for( k = ZMIN-delta; k <= 1; k++ ){
              txx0[imp][jmp][k] = rxxbuf[i];
              tyy0[imp][jmp][k] = ryybuf[i];
              tzz0[imp][jmp][k] = rzzbuf[i];
              txy0[imp][jmp][k] = rxybuf[i];
              tyz0[imp][jmp][k] = ryzbuf[i];
              txz0[imp][jmp][k] = rxzbuf[i];
              i = i + 1;
            }
          }
        }
      }

      if( coords[0] != (px-1) )
        MPI_Wait(&recvreq[2],&recvstatus[2]);

      if( coords[0] != (px-1) ){
        i = 0;
        for( imp = mpmx+1; imp <= mpmx+2; imp++ ){
          for( jmp = -1; jmp <= mpmy+2; jmp++ ){
            for( k = ZMIN-delta; k <= 1; k++ ){
              txx0[imp][jmp][k] = rxxbuf2[i];
              tyy0[imp][jmp][k] = ryybuf2[i];
              tzz0[imp][jmp][k] = rzzbuf2[i];
              txy0[imp][jmp][k] = rxybuf2[i];
              tyz0[imp][jmp][k] = ryzbuf2[i];
              txz0[imp][jmp][k] = rxzbuf2[i];
              i = i + 1;
            }
          }
        }
      }

                /* North - South */

      if( coords[1] != 0 )
        MPI_Wait(&sendreq[4],&sendstatus[4]);

      if( coords[1] != 0 )
        MPI_Wait(&recvreq[4],&recvstatus[4]);

      if( coords[1] != 0 ){
        i = 0;
        for( jmp = -1; jmp <= 0; jmp++ ){
          for( imp = -1; imp <= mpmx+2; imp++ ){
            for( k = ZMIN-delta; k <= 1; k++ ){
              txx0[imp][jmp][k] = rxxbuf3[i];
              tyy0[imp][jmp][k] = ryybuf3[i];
              tzz0[imp][jmp][k] = rzzbuf3[i];
              txy0[imp][jmp][k] = rxybuf3[i];
              tyz0[imp][jmp][k] = ryzbuf3[i];
              txz0[imp][jmp][k] = rxzbuf3[i];
              i = i + 1;
            }
          }
        }
      }

      if( coords[1] != (py-1) )
        MPI_Wait(&sendreq[3],&sendstatus[3]);

      if( coords[1] != (py-1) )
        MPI_Wait(&recvreq[3],&recvstatus[3]);

      if( coords[1] != (py-1) ){
        i = 0;
        for( jmp = mpmy+1; jmp <= mpmy+2; jmp++ ){
          for( imp = -1; imp <= mpmx+2; imp++ ){
            for( k = ZMIN-delta; k <= 1; k++ ){
              txx0[imp][jmp][k] = rxxbuf4[i];
              tyy0[imp][jmp][k] = ryybuf4[i];
              tzz0[imp][jmp][k] = rzzbuf4[i];
              txy0[imp][jmp][k] = rxybuf4[i];
              tyz0[imp][jmp][k] = ryzbuf4[i];
              txz0[imp][jmp][k] = rxzbuf4[i];
              i = i + 1;
            }
          }
        }
      }

    #endif /* end of second method */

                /* Communication : third method */

    #if (BLOCKING)

                /* Communication to synchronize */

                /* Communication East - West */
                /* Positive direction */

      if( coords[0] != (px-1) ){
        i = 0;
        for ( imp = mpmx-1; imp <= mpmx; imp++ ){
	      for( jmp = -1; jmp <= mpmy+2; jmp++ ){
	        for( k = ZMIN-delta; k <= 1; k++ ){
	          sxxbuf[i] = txx0[imp][jmp][k];
	          syybuf[i] = tyy0[imp][jmp][k];
	          szzbuf[i] = tzz0[imp][jmp][k];
	          sxybuf[i] = txy0[imp][jmp][k];
	          syzbuf[i] = tyz0[imp][jmp][k];
	          sxzbuf[i] = txz0[imp][jmp][k];
	          i = i + 1;
	        }
	      }
        }
      }

      if( coords[0] != (px-1) )
        MPI_Isend(sxxbuf, 6*nmaxx, MPI_DOUBLE, nest,
	 	 3, MPI_COMM_WORLD, &req);

      if( coords[0] != 0 )
        MPI_Recv(rxxbuf, 6*nmaxx, MPI_DOUBLE, nouest,
	     3, MPI_COMM_WORLD, &status);

      if ( px != 1 ) {
        if( coords[0] != 0 ){
          i = 0;
          for( imp = -1; imp <= 0; imp++ ){
  	        for( jmp = -1; jmp <= mpmy+2; jmp++ ){
	          for( k = ZMIN-delta; k <= 1; k++ ){
		        txx0[imp][jmp][k] = rxxbuf[i];
		        tyy0[imp][jmp][k] = ryybuf[i];
		        tzz0[imp][jmp][k] = rzzbuf[i];
		        txy0[imp][jmp][k] = rxybuf[i];
		        tyz0[imp][jmp][k] = ryzbuf[i];
		        txz0[imp][jmp][k] = rxzbuf[i];
		        i = i + 1;
		      }
	        }
	      }
        } else {
          MPI_Wait (&req, &status) ;
        }
      }

                /* Negative direction */

      if( coords[0] != 0 ){
        i = 0;
        for ( imp = 1; imp <= 2; imp++ ){
	      for( jmp = -1; jmp <= mpmy+2; jmp++ ){
            for( k = ZMIN-delta; k <= 1; k++ ){
              sxxbuf2[i] = txx0[imp][jmp][k];
              syybuf2[i] = tyy0[imp][jmp][k];
              szzbuf2[i] = tzz0[imp][jmp][k];
              sxybuf2[i] = txy0[imp][jmp][k];
              syzbuf2[i] = tyz0[imp][jmp][k];
              sxzbuf2[i] = txz0[imp][jmp][k];
              i = i + 1;
            }
          }
        }
      }

      if( coords[0] != 0 )
        MPI_Isend(sxxbuf2, 6*nmaxx, MPI_DOUBLE, nouest,
         4, MPI_COMM_WORLD, &req);

      if( coords[0] != (px-1) )
        MPI_Recv(rxxbuf2, 6*nmaxx, MPI_DOUBLE, nest,
         4, MPI_COMM_WORLD, &status);

      if ( px != 1 ) {
        if( coords[0] != (px-1) ){
          i = 0;
          for( imp = mpmx+1; imp <= mpmx+2; imp++ ){
            for( jmp = -1; jmp <= mpmy+2; jmp++ ){
	          for( k = ZMIN-delta; k <= 1; k++ ){
	   	        txx0[imp][jmp][k] = rxxbuf2[i];
		        tyy0[imp][jmp][k] = ryybuf2[i];
		        tzz0[imp][jmp][k] = rzzbuf2[i];
		        txy0[imp][jmp][k] = rxybuf2[i];
		        tyz0[imp][jmp][k] = ryzbuf2[i];
		        txz0[imp][jmp][k] = rxzbuf2[i];
		        i = i + 1;
		      }
	        }
	      }
        } else {
          MPI_Wait (&req, &status) ;
        }
      }

                /* Communication North - South */
                /* Positive direction */

      if( coords[1] != (py-1) ){
        i = 0;
        for ( jmp = mpmy-1; jmp <= mpmy; jmp++ ){
	      for( imp = -1; imp <= mpmx+2; imp++ ){
	        for( k = ZMIN-delta; k <= 1; k++ ){
	          sxxbuf3[i] = txx0[imp][jmp][k];
	          syybuf3[i] = tyy0[imp][jmp][k];
	          szzbuf3[i] = tzz0[imp][jmp][k];
	          sxybuf3[i] = txy0[imp][jmp][k];
	          syzbuf3[i] = tyz0[imp][jmp][k];
	          sxzbuf3[i] = txz0[imp][jmp][k];
	          i = i + 1;
	        }
	      }
        }
      }

      if( coords[1] != (py-1) )
        MPI_Isend(sxxbuf3, 6*nmaxy, MPI_DOUBLE, nnord,
	 	 3, MPI_COMM_WORLD, &req);

      if( coords[1] != 0 )
        MPI_Recv(rxxbuf3, 6*nmaxy, MPI_DOUBLE, nsud,
	     3, MPI_COMM_WORLD, &status);

      if (py != 1 ) {
        if( coords[1] != 0 ){
          i = 0;
          for( jmp = -1; jmp <= 0; jmp++ ){
	        for( imp = -1; imp <= mpmx+2; imp++ ){
	          for( k = ZMIN-delta; k <= 1; k++ ){
	  	        txx0[imp][jmp][k] = rxxbuf3[i];
		        tyy0[imp][jmp][k] = ryybuf3[i];
		        tzz0[imp][jmp][k] = rzzbuf3[i];
		        txy0[imp][jmp][k] = rxybuf3[i];
		        tyz0[imp][jmp][k] = ryzbuf3[i];
		        txz0[imp][jmp][k] = rxzbuf3[i];
		        i = i + 1;
		      }
	        }
	      }
        } else {
          MPI_Wait (&req, &status) ;
        }
      }

				/* Negative direction */

      if( coords[1] != 0 ){
        i = 0;
        for ( jmp = 1; jmp <= 2; jmp++ ){
   	      for( imp = -1; imp <= mpmx+2; imp++ ){
            for( k = ZMIN-delta; k <= 1; k++ ){
              sxxbuf4[i] = txx0[imp][jmp][k];
              syybuf4[i] = tyy0[imp][jmp][k];
              szzbuf4[i] = tzz0[imp][jmp][k];
              sxybuf4[i] = txy0[imp][jmp][k];
              syzbuf4[i] = tyz0[imp][jmp][k];
              sxzbuf4[i] = txz0[imp][jmp][k];
              i = i + 1;
            }
          }
        }
      }

      if( coords[1] != 0 )
        MPI_Isend(sxxbuf4, 6*nmaxy, MPI_DOUBLE, nsud,
         4, MPI_COMM_WORLD, &req);

      if( coords[1] != (py-1) )
        MPI_Recv(rxxbuf4, 6*nmaxy, MPI_DOUBLE, nnord,
         4, MPI_COMM_WORLD, &status);

      if (py != 1 ) {
        if( coords[1] != (py-1) ){
          i = 0;
          for( jmp = mpmy+1; jmp <= mpmy+2; jmp++ ){
            for( imp = -1; imp <= mpmx+2; imp++ ){
  	          for( k = ZMIN-delta; k <= 1; k++ ){
		        txx0[imp][jmp][k] = rxxbuf4[i];
		        tyy0[imp][jmp][k] = ryybuf4[i];
		        tzz0[imp][jmp][k] = rzzbuf4[i];
		        txy0[imp][jmp][k] = rxybuf4[i];
		        tyz0[imp][jmp][k] = ryzbuf4[i];
		        txz0[imp][jmp][k] = rxzbuf4[i];
		        i = i + 1;
		      }
	        }
	      }
        } else {
          MPI_Wait (&req, &status) ;
        }
      }

    #endif /* end of third method */

    #if (TAU)
      TAU_PROFILE_STOP(exchange_sig);
    #endif

    #if (TAUGLOBAL)
      TAU_PROFILE_STOP(exchange_sig);
    #endif

    #if (TIMING_BARRIER)
      MPI_Barrier(MPI_COMM_WORLD);
      timing2 = my_second();
      timing_comm1 = timing_comm1 + (timing2-timing1);
    #endif

    #if (TIMING)
      timing2 = my_second();
      timing_comm1 = timing_comm1 + (timing2-timing1);
    #endif

    #if (TIMING_BARRIER)
      MPI_Barrier(MPI_COMM_WORLD);
      timing1 = my_second();
    #endif

    #if (TIMING)
      timing1 = my_second();
    #endif

                /* Second step : t = (l + 1) for velocity */

    npml = 0;

    #if (TAUGLOBAL)
      TAU_PROFILE_START(compute_vit);
    #endif

/* imode : to increase the velocity of the computation, we begin by computing
   the values of stress at the boundaries of the px * py parts of the array
   Afterwise, we can compute the values of the stress in the middle */

    for (imode=1;imode<=5;imode++){

	  if ( imode == 1 ){
	    mpmx_debut = 1;
	    mpmx_fin = 3;
	    mpmy_debut = 1;
	    mpmy_fin = mpmy;
	  }

      if ( imode == 2 ){
	    mpmx_debut = mpmx-2;
	    mpmx_fin = mpmx;
	    mpmy_debut = 1;
	    mpmy_fin = mpmy;
	  }

	  if ( imode == 3 ){
	    mpmy_debut = 1;
	    mpmy_fin = 3;
	    mpmx_debut = 4;
	    mpmx_fin = mpmx-3;
	  }

      if ( imode == 4 ){
	    mpmy_debut = mpmy-2;
	    mpmy_fin = mpmy;
	    mpmx_debut = 4;
	    mpmx_fin = mpmx-3;
	  }

      if ( imode == 5 ){
	    mpmx_debut = 4;
	    mpmx_fin = mpmx-3;
	    mpmy_debut = 4;
	    mpmy_fin = mpmy-3;

                /* Communication : first method */

        #if (PERSISTANT)

          if ( nest != -1 )
	        MPI_Startall(3,&sendreq2[1]);

          if ( nouest != -1 )
	        MPI_Startall(3,&recvreq2[1]);

          if ( nouest != -1 )
  	        MPI_Startall(3,&sendreq2[4]);

          if ( nest != -1 )
	        MPI_Startall(3,&recvreq2[4]);

                /* North - South */

          if(  coords[1] != (py-1) ){
            i = 0;
            for ( jmp = mpmy-1; jmp <= mpmy; jmp++ ){
  	          for( imp = -1; imp <= mpmx+2; imp++ ){
                for( k = ZMIN-delta; k <= 1; k++ ){
                  assert (i < (nmaxy + 1)) ;
                  sxbuf3[i] = vx0[imp][jmp][k];
                  sybuf3[i] = vy0[imp][jmp][k];
                  szbuf3[i] = vz0[imp][jmp][k];
                  i = i + 1;
                }
              }
            }
          }

          if( coords[1] != 0 ){
            i = 0;
            for ( jmp = 1; jmp <= 2; jmp++ ){
	          for( imp = -1; imp <= mpmx+2; imp++ ){
                for( k = ZMIN-delta; k <= 1; k++ ){
                  sxbuf4[i] = vx0[imp][jmp][k];
                  sybuf4[i] = vy0[imp][jmp][k];
                  szbuf4[i] = vz0[imp][jmp][k];
                  i = i + 1;
                }
              }
            }
          }

          if( coords[1] != (py-1) )
            MPI_Start(&sendreq3[2]);

          if( coords[1] != 0 )
            MPI_Start(&recvreq3[2]);

          if( coords[1] != 0 )
            MPI_Start(&sendreq4[2]);

          if( coords[1] != (py-1) )
            MPI_Start(&recvreq4[2]);

        #endif /* end of first method */

                /* Communication : second method */

        #if (NONBLOCKING)

                /* East - West */

          if(  coords[0] != (px-1) ){
            i = 0;
            for ( imp = mpmx-1; imp <= mpmx; imp++ ){
	          for( jmp = -1; jmp <= mpmy+2; jmp++ ){
                for( k = ZMIN-delta; k <= 1; k++ ){
                  assert (i < (nmaxx + 1)) ;
                  sxbuf[i] = vx0[imp][jmp][k];
                  sybuf[i] = vy0[imp][jmp][k];
                  szbuf[i] = vz0[imp][jmp][k];
                  i = i + 1;
                }
              }
            }
          }

          if( coords[0] != (px-1) )
            MPI_Isend(sxbuf, 3*nmaxx, MPI_DOUBLE, nest,
             31, MPI_COMM_WORLD, &sendreq2[1]);

          if( coords[0] != 0 ){
            i = 0;
            for ( imp = 1; imp <= 2; imp++ ){
	          for( jmp = -1; jmp <= mpmy+2; jmp++ ){
                for( k = ZMIN-delta; k <= 1; k++ ){
                  sxbuf2[i] = vx0[imp][jmp][k];
                  sybuf2[i] = vy0[imp][jmp][k];
                  szbuf2[i] = vz0[imp][jmp][k];
                  i = i + 1;
                }
              }
            }
          }

          if( coords[0] != 0 )
            MPI_Isend(sxbuf2, 3*nmaxx, MPI_DOUBLE, nouest,
             32, MPI_COMM_WORLD, &sendreq2[2]);

/* Reception non bloquante */

          if( coords[0] != (px-1) )
            MPI_Irecv(rxbuf2, 3*nmaxx, MPI_DOUBLE, nest,
             32, MPI_COMM_WORLD, &recvreq2[2]);

          if( coords[0] != 0 )
            MPI_Irecv(rxbuf, 3*nmaxx, MPI_DOUBLE, nouest,
             31, MPI_COMM_WORLD, &recvreq2[1]);

                /* North - South */

          if(  coords[1] != (py-1) ){
            i = 0;
            for ( jmp = mpmy-1; jmp <= mpmy; jmp++ ){
	          for( imp = -1; imp <= mpmx+2; imp++ ){
                for( k = ZMIN-delta; k <= 1; k++ ){
                  assert (i < (nmaxy + 1)) ;
                  sxbuf3[i] = vx0[imp][jmp][k];
                  sybuf3[i] = vy0[imp][jmp][k];
                  szbuf3[i] = vz0[imp][jmp][k];
                  i = i + 1;
                }
              }
            }
          }

          if( coords[1] != (py-1) )
            MPI_Isend(sxbuf3, 3*nmaxy, MPI_DOUBLE, nnord,
             21, MPI_COMM_WORLD, &sendreq2[3]);

          if( coords[1] != 0 ){
            i = 0;
            for ( jmp = 1; jmp <= 2; jmp++ ){
	          for( imp = -1; imp <= mpmx+2; imp++ ){
                for( k = ZMIN-delta; k <= 1; k++ ){
                  sxbuf4[i] = vx0[imp][jmp][k];
                  sybuf4[i] = vy0[imp][jmp][k];
                  szbuf4[i] = vz0[imp][jmp][k];
                  i = i + 1;
                }
              }
            }
          }

          if( coords[1] != 0 )
            MPI_Isend(sxbuf4, 3*nmaxy, MPI_DOUBLE, nsud,
             22, MPI_COMM_WORLD, &sendreq2[4]);

/* Reception non bloquante */

          if( coords[1] != (py-1) )
            MPI_Irecv(rxbuf4, 3*nmaxy, MPI_DOUBLE, nnord,
             22, MPI_COMM_WORLD, &recvreq2[4]);

          if( coords[1] != 0 )
            MPI_Irecv(rxbuf3, 3*nmaxy, MPI_DOUBLE, nsud,
             21, MPI_COMM_WORLD, &recvreq2[3]);

        #endif /* end of second method */

	  } /* end of imode = 5 */

                /* Beginning of velocity computation */

      for ( i = mpmx_debut; i <= mpmx_fin; i++ ){
    	imp = imp2i_array[i];
	    if ( ( imp>=XMIN-delta+1) && (imp<=XMAX+delta+1)){

          for ( j = mpmy_debut; j <= mpmy_fin; j++ ){
   	        jmp = jmp2j_array[j];
 	        if ( ( jmp>=YMIN-delta+1) && (jmp<=YMAX+delta+1)){

		      for ( k = ZMIN-delta+1; k <= 1; k++){

                /* 4th order finite-difference */

                if ( NDIM == 4 ){

                /* CPML */

		          if ( imp <= XMIN+2 || imp >= XMAX || jmp <= YMIN+2 || jmp >= YMAX || k <= ZMIN+2 ){
  	                npml += 1;

                /* Calculation of vx */

                    if ( k >= ZMIN-delta+2 && jmp >= YMIN-delta+2 && imp >= XMIN-delta+2 ){

                      if ( k == 1 ){ /* free surface */

                        phixdum = phitxxx[npml];
                        phiydum = phitxyy[npml];
                        phizdum = phitxzz[npml];

                        phitxxx[npml] = CPML2 (vp[i][j][k], dumpx[i], alphax[i], kappax[i], phixdum, ds, dt,
                        txx0[i-1][j][k], txx0[i][j][k] );
                        phitxyy[npml] = CPML2 (vp[i][j][k], dumpy[j], alphay[j], kappay[j], phiydum, ds, dt,
                        txy0[i][j-1][k], txy0[i][j][k] );
                        phitxzz[npml] = CPML2 (vp[i][j][k], dumpz[k], alphaz[k], kappaz[k], phizdum, ds, dt,
                        txz0[i][j][k-1], - txz0[i][j][k-1] );

                        vx0[i][j][k] += (dt/rho[i][j][k])*( phitxxx[npml] + phitxyy[npml] + phitxzz[npml] )
                        + staggardv2 (1./rho[i][j][k], kappax[i], kappay[j], kappaz[k], dt, ds,
                        txx0[i-1][j][k], txx0[i][j][k],
                        txy0[i][j-1][k], txy0[i][j][k],
                        txz0[i][j][k-1], - txz0[i][j][k-1] );

			          } else if ( k == 0 ){ /* in the first cell, 2nd order finite-difference instead of 4th order finite-difference */

                        phixdum = phitxxx[npml];
                        phiydum = phitxyy[npml];
                        phizdum = phitxzz[npml];

                        phitxxx[npml] = CPML2 (vp[i][j][k], dumpx[i], alphax[i], kappax[i], phixdum, ds, dt,
                        txx0[i-1][j][k], txx0[i][j][k] );
                        phitxyy[npml] = CPML2 (vp[i][j][k], dumpy[j], alphay[j], kappay[j], phiydum, ds, dt,
                        txy0[i][j-1][k], txy0[i][j][k] );
                        phitxzz[npml] = CPML2 (vp[i][j][k], dumpz[k], alphaz[k], kappaz[k], phizdum, ds, dt,
                        txz0[i][j][k-1], txz0[i][j][k] );

                        vx0[i][j][k] += (dt/rho[i][j][k])*( phitxxx[npml] + phitxyy[npml] + phitxzz[npml] )
                        + staggardv2 (1./rho[i][j][k], kappax[i], kappay[j], kappaz[k], dt, ds,
                        txx0[i-1][j][k], txx0[i][j][k],
                        txy0[i][j-1][k], txy0[i][j][k],
                        txz0[i][j][k-1], txz0[i][j][k] );

			          } else { /* regular domain */

                        phixdum = phitxxx[npml];
                        phiydum = phitxyy[npml];
                        phizdum = phitxzz[npml];

                        phitxxx[npml] = CPML4 (vp[i][j][k], dumpx[i], alphax[i], kappax[i], phixdum, ds, dt,
                        txx0[i-1][j][k], txx0[i][j][k],
                        txx0[i-2][j][k], txx0[i+1][j][k] );
                        phitxyy[npml] = CPML4 (vp[i][j][k], dumpy[j], alphay[j], kappay[j], phiydum, ds, dt,
                        txy0[i][j-1][k], txy0[i][j][k],
                        txy0[i][j-2][k], txy0[i][j+1][k] );
                        phitxzz[npml] = CPML4 (vp[i][j][k], dumpz[k], alphaz[k], kappaz[k], phizdum, ds, dt,
                        txz0[i][j][k-1], txz0[i][j][k],
                        txz0[i][j][k-2], txz0[i][j][k+1] );

                        vx0[i][j][k] += (dt/rho[i][j][k])*( phitxxx[npml] + phitxyy[npml] + phitxzz[npml] )
                        + staggardv4 (1./rho[i][j][k], kappax[i], kappay[j], kappaz[k], dt, ds,
                        txx0[i-1][j][k], txx0[i][j][k],
                        txx0[i-2][j][k], txx0[i+1][j][k],
                        txy0[i][j-1][k], txy0[i][j][k],
                        txy0[i][j-2][k], txy0[i][j+1][k],
                        txz0[i][j][k-1], txz0[i][j][k],
                        txz0[i][j][k-2], txz0[i][j][k+1] );

                      } /* end of if "free surface" */
                    } /* end of calculation of vx */

                /* Calculation of vy */

                    if ( k >= ZMIN-delta+2 && jmp <= YMAX+delta && imp <= XMAX+delta ){

                      rhoxy = 0.25*(rho[i][j][k] + rho[i][j+1][k]
				                  + rho[i+1][j][k] + rho[i+1][j+1][k]);
                      muxy = 4./(1./mu[i][j][k] + 1./mu[i][j+1][k]
				               + 1./mu[i+1][j][k] + 1./mu[i+1][j+1][k]);
			          lamxy = 4./(1./lam[i][j][k] + 1./lam[i][j+1][k]
				                + 1./lam[i+1][j][k] + 1./lam[i+1][j+1][k]);
			          vpxy = sqrt ( (lamxy+2.*muxy)/( 0.25*( rho[i][j][k] + rho[i][j+1][k]
				                                           + rho[i+1][j][k] + rho[i+1][j+1][k] ) ) );

                      if ( k == 1 ){ /* free surface */

                        phixdum = phitxyx[npml];
                        phiydum = phityyy[npml];
                        phizdum = phityzz[npml];

                        phitxyx[npml] = CPML2 (vpxy, dumpx2[i], alphax2[i], kappax2[i], phixdum, ds, dt,
                        txy0[i][j][k], txy0[i+1][j][k] );
                        phityyy[npml] = CPML2 (vpxy, dumpy2[j], alphay2[j], kappay2[j], phiydum, ds, dt,
                        tyy0[i][j][k], tyy0[i][j+1][k] );
                        phityzz[npml] = CPML2 (vpxy, dumpz[k], alphaz[k], kappaz[k], phizdum, ds, dt,
                        tyz0[i][j][k-1], - tyz0[i][j][k-1] );

                        vy0[i][j][k] += (dt/rhoxy)*( phitxyx[npml] + phityyy[npml] + phityzz[npml] )
                        + staggardv2 (1./rhoxy, kappax2[i], kappay2[j], kappaz[k], dt, ds,
                        txy0[i][j][k], txy0[i+1][j][k],
                        tyy0[i][j][k], tyy0[i][j+1][k],
                        tyz0[i][j][k-1], - tyz0[i][j][k-1] );

			          } else if ( k == 0 ){ /* in the first cell, 2nd order finite-difference instead of 4th order finite-difference */

                        phixdum = phitxyx[npml];
                        phiydum = phityyy[npml];
                        phizdum = phityzz[npml];

                        phitxyx[npml] = CPML2 (vpxy, dumpx2[i], alphax2[i], kappax2[i], phixdum, ds, dt,
                        txy0[i][j][k], txy0[i+1][j][k] );
                        phityyy[npml] = CPML2 (vpxy, dumpy2[j], alphay2[j], kappay2[j], phiydum, ds, dt,
                        tyy0[i][j][k], tyy0[i][j+1][k] );
                        phityzz[npml] = CPML2 (vpxy, dumpz[k], alphaz[k], kappaz[k], phizdum, ds, dt,
                        tyz0[i][j][k-1], tyz0[i][j][k] );

                        vy0[i][j][k] += (dt/rhoxy)*( phitxyx[npml] + phityyy[npml] + phityzz[npml] )
                        + staggardv2 (1./rhoxy, kappax2[i], kappay2[j], kappaz[k], dt, ds,
                        txy0[i][j][k], txy0[i+1][j][k],
                        tyy0[i][j][k], tyy0[i][j+1][k],
                        tyz0[i][j][k-1], tyz0[i][j][k] );

			          } else { /* regular domain */

                        phixdum = phitxyx[npml];
                        phiydum = phityyy[npml];
                        phizdum = phityzz[npml];

                        phitxyx[npml] = CPML4 (vpxy, dumpx2[i], alphax2[i], kappax2[i], phixdum, ds, dt,
                        txy0[i][j][k], txy0[i+1][j][k],
                        txy0[i-1][j][k], txy0[i+2][j][k] );
                        phityyy[npml] = CPML4 (vpxy, dumpy2[j], alphay2[j], kappay2[j], phiydum, ds, dt,
                        tyy0[i][j][k], tyy0[i][j+1][k],
                        tyy0[i][j-1][k], tyy0[i][j+2][k] );
                        phityzz[npml] = CPML4 (vpxy, dumpz[k], alphaz[k], kappaz[k], phizdum, ds, dt,
                        tyz0[i][j][k-1], tyz0[i][j][k],
                        tyz0[i][j][k-2], tyz0[i][j][k+1] );

                        vy0[i][j][k] += (dt/rhoxy)*( phitxyx[npml] + phityyy[npml] + phityzz[npml] )
                        + staggardv4 (1./rhoxy, kappax2[i], kappay2[j], kappaz[k], dt, ds,
                        txy0[i][j][k], txy0[i+1][j][k],
                        txy0[i-1][j][k], txy0[i+2][j][k],
                        tyy0[i][j][k], tyy0[i][j+1][k],
                        tyy0[i][j-1][k], tyy0[i][j+2][k],
                        tyz0[i][j][k-1], tyz0[i][j][k],
                        tyz0[i][j][k-2], tyz0[i][j][k+1] );

	                  } /* end of if "free surface" */
	                } /* end of calculation of vy */

                /* Calculation of vz */

                    if ( jmp >= YMIN-delta+2 && imp <= XMAX+delta ){

                      rhoxz = 0.25*(rho[i][j][k] + rho[i][j][k+1]
				                  + rho[i+1][j][k] + rho[i+1][j][k+1]);
                      muxz = 4./(1./mu[i][j][k] + 1./mu[i][j][k+1]
				               + 1./mu[i+1][j][k] + 1./mu[i+1][j][k+1]);
			          lamxz = 4./(1./lam[i][j][k] + 1./lam[i][j][k+1]
				                + 1./lam[i+1][j][k] + 1./lam[i+1][j][k+1]);
			          vpxz = sqrt ( (lamxz+2.*muxz)/( 0.25*( rho[i][j][k] + rho[i][j][k+1]
				                                           + rho[i+1][j][k] + rho[i+1][j][k+1] ) ) );

                      if ( k == 1 ){ /* free surface */

                        phixdum = phitxzx[npml];
                        phiydum = phityzy[npml];
                        phizdum = phitzzz[npml];

                        phitxzx[npml] = CPML2 (vpxz, dumpx2[i], alphax2[i], kappax2[i], phixdum, ds, dt,
                        - txz0[i][j][k-1], - txz0[i+1][j][k-1] );
                        phityzy[npml] = CPML2 (vpxz, dumpy[j], alphay[j], kappay[j], phiydum, ds, dt,
                        - tyz0[i][j-1][k-1], - tyz0[i][j][k-1] );
                        phitzzz[npml] = CPML2 (vpxz, dumpz2[k], alphaz2[k], kappaz2[k], phizdum, ds, dt,
                        tzz0[i][j][k], - tzz0[i][j][k-1] );

                        vz0[i][j][k] += (dt/rhoxz)*( phitxzx[npml] + phityzy[npml] + phitzzz[npml] )
                        + staggardv2 (1./rhoxz, kappax2[i], kappay[j], kappaz2[k], dt, ds,
			            - txz0[i][j][k-1], - txz0[i+1][j][k-1],
			            - tyz0[i][j-1][k-1], - tyz0[i][j][k-1],
                        tzz0[i][j][k], - tzz0[i][j][k-1] );

			          } else if ( k == 0 ){ /* in the first cell, 2nd order finite-difference instead of 4th order finite-difference */

                        phixdum = phitxzx[npml];
                        phiydum = phityzy[npml];
                        phizdum = phitzzz[npml];

                        phitxzx[npml] = CPML2 (vpxz, dumpx2[i], alphax2[i], kappax2[i], phixdum, ds, dt,
                        txz0[i][j][k], txz0[i+1][j][k] );
                        phityzy[npml] = CPML2 (vpxz, dumpy[j], alphay[j], kappay[j], phiydum, ds, dt,
                        tyz0[i][j-1][k], tyz0[i][j][k] );
                        phitzzz[npml] = CPML2 (vpxz, dumpz2[k], alphaz2[k], kappaz2[k], phizdum, ds, dt,
                        tzz0[i][j][k], tzz0[i][j][k+1] );

                        vz0[i][j][k] += (dt/rhoxz)*( phitxzx[npml] + phityzy[npml] + phitzzz[npml] )
                        + staggardv2 (1./rhoxz, kappax2[i], kappay[j], kappaz2[k], dt, ds,
			            txz0[i][j][k], txz0[i+1][j][k],
			            tyz0[i][j-1][k], tyz0[i][j][k],
                        tzz0[i][j][k], tzz0[i][j][k+1] );

			          } else { /* regular domain */

                        phixdum = phitxzx[npml];
                        phiydum = phityzy[npml];
                        phizdum = phitzzz[npml];

                        phitxzx[npml] = CPML4 (vpxz, dumpx2[i], alphax2[i], kappax2[i], phixdum, ds, dt,
                        txz0[i][j][k], txz0[i+1][j][k],
                        txz0[i-1][j][k], txz0[i+2][j][k] );
                        phityzy[npml] = CPML4 (vpxz, dumpy[j], alphay[j], kappay[j], phiydum, ds, dt,
                        tyz0[i][j-1][k], tyz0[i][j][k],
                        tyz0[i][j-2][k], tyz0[i][j+1][k] );
                        phitzzz[npml] = CPML4 (vpxz, dumpz2[k], alphaz2[k], kappaz2[k], phizdum, ds, dt,
                        tzz0[i][j][k], tzz0[i][j][k+1],
                        tzz0[i][j][k-1], tzz0[i][j][k+2] );

                        vz0[i][j][k] += (dt/rhoxz)*( phitxzx[npml] + phityzy[npml] + phitzzz[npml] )
                        + staggardv4 (1./rhoxz, kappax2[i], kappay[j], kappaz2[k], dt, ds,
			            txz0[i][j][k], txz0[i+1][j][k],
			            txz0[i-1][j][k], txz0[i+2][j][k],
			            tyz0[i][j-1][k], tyz0[i][j][k],
			            tyz0[i][j-2][k], tyz0[i][j+1][k],
                        tzz0[i][j][k], tzz0[i][j][k+1],
                        tzz0[i][j][k-1], tzz0[i][j][k+2] );

	                  } /* end of if "free surface" */
	                } /* end of calculation of vz */

                /* Normal mode */

                  } else {

                    rhoxy = 0.25*(rho[i][j][k] + rho[i][j+1][k]
				                + rho[i+1][j][k] + rho[i+1][j+1][k]);
			        rhoxz = 0.25*(rho[i][j][k] + rho[i][j][k+1]
				                + rho[i+1][j][k] + rho[i+1][j][k+1]);

                    if ( k == 1 ){ /* free surface */

                      vx0[i][j][k] += (1./rho[i][j][k])*fx[i][j][k]*dt/ds
                      + staggardv2 (1./rho[i][j][k], un, un, un, dt, ds,
                      txx0[i-1][j][k], txx0[i][j][k],
                      txy0[i][j-1][k], txy0[i][j][k],
                      txz0[i][j][k-1], - txz0[i][j][k-1] );

                      vy0[i][j][k] += (1./rhoxy)*fy[i][j][k]*dt/ds
                      + staggardv2 (1./rhoxy, un, un, un, dt, ds,
                      txy0[i][j][k], txy0[i+1][j][k],
                      tyy0[i][j][k], tyy0[i][j+1][k],
                      tyz0[i][j][k-1], - tyz0[i][j][k-1] );

                      vz0[i][j][k] += (1./rhoxz)*fz[i][j][k]*dt/ds
                      + staggardv2 (1./rhoxz, un, un, un, dt, ds,
                      - txz0[i][j][k-1], - txz0[i+1][j][k-1],
                      - tyz0[i][j-1][k-1], - tyz0[i][j][k-1],
                      tzz0[i][j][k], - tzz0[i][j][k-1] );

		            } else if ( k == 0 ){ /* in the first cell, 2nd order finite-difference instead of 4th order finite-difference */

                      vx0[i][j][k] += (1./rho[i][j][k])*fx[i][j][k]*dt/ds
                      + staggardv2 (1./rho[i][j][k], un, un, un, dt, ds,
                      txx0[i-1][j][k], txx0[i][j][k],
                      txy0[i][j-1][k], txy0[i][j][k],
                      txz0[i][j][k-1], txz0[i][j][k] );

                      vy0[i][j][k] += (1./rhoxy)*fy[i][j][k]*dt/ds
                      + staggardv2 (1./rhoxy, un, un, un, dt, ds,
                      txy0[i][j][k], txy0[i+1][j][k],
                      tyy0[i][j][k], tyy0[i][j+1][k],
                      tyz0[i][j][k-1], tyz0[i][j][k] );

                      vz0[i][j][k] += (1./rhoxz)*fz[i][j][k]*dt/ds
                      + staggardv2 (1./rhoxz, un, un, un, dt, ds,
                      txz0[i][j][k], txz0[i+1][j][k],
                      tyz0[i][j-1][k], tyz0[i][j][k],
                      tzz0[i][j][k], tzz0[i][j][k+1] );

		            } else { /* regular domain */

                      vx0[i][j][k] += (1./rho[i][j][k])*fx[i][j][k]*dt/ds
                      + staggardv4 (1./rho[i][j][k], un, un, un, dt, ds,
                      txx0[i-1][j][k], txx0[i][j][k],
                      txx0[i-2][j][k], txx0[i+1][j][k],
                      txy0[i][j-1][k], txy0[i][j][k],
                      txy0[i][j-2][k], txy0[i][j+1][k],
                      txz0[i][j][k-1], txz0[i][j][k],
                      txz0[i][j][k-2], txz0[i][j][k+1] );

                      vy0[i][j][k] += (1./rhoxy)*fy[i][j][k]*dt/ds
                      + staggardv4 (1./rhoxy, un, un, un, dt, ds,
                      txy0[i][j][k], txy0[i+1][j][k],
                      txy0[i-1][j][k], txy0[i+2][j][k],
                      tyy0[i][j][k], tyy0[i][j+1][k],
                      tyy0[i][j-1][k], tyy0[i][j+2][k],
                      tyz0[i][j][k-1], tyz0[i][j][k],
                      tyz0[i][j][k-2], tyz0[i][j][k+1] );

                      vz0[i][j][k] += (1./rhoxz)*fz[i][j][k]*dt/ds
                      + staggardv4 (1./rhoxz, un, un, un, dt, ds,
                      txz0[i][j][k], txz0[i+1][j][k],
                      txz0[i-1][j][k], txz0[i+2][j][k],
                      tyz0[i][j-1][k], tyz0[i][j][k],
                      tyz0[i][j-2][k], tyz0[i][j+1][k],
                      tzz0[i][j][k], tzz0[i][j][k+1],
                      tzz0[i][j][k-1], tzz0[i][j][k+2] );

		            } /* end of if "free surface" */
		          } /* end of normal mode */

               /* 2nd order finite-difference */

                } else {

                /* CPML */

		          if ( imp <= XMIN+2 || imp >= XMAX || jmp <= YMIN+2 || jmp >= YMAX || k <= ZMIN+2 ){
  	                npml += 1;

                /* Calculation of vx */

                    if ( k >= ZMIN-delta+2 && jmp >= YMIN-delta+2 && imp >= XMIN-delta+2 ){

                      if ( k == 1 ){ /* free surface */

                        phixdum = phitxxx[npml];
                        phiydum = phitxyy[npml];
                        phizdum = phitxzz[npml];

                        phitxxx[npml] = CPML2 (vp[i][j][k], dumpx[i], alphax[i], kappax[i], phixdum, ds, dt,
                        txx0[i-1][j][k], txx0[i][j][k] );
                        phitxyy[npml] = CPML2 (vp[i][j][k], dumpy[j], alphay[j], kappay[j], phiydum, ds, dt,
                        txy0[i][j-1][k], txy0[i][j][k] );
                        phitxzz[npml] = CPML2 (vp[i][j][k], dumpz[k], alphaz[k], kappaz[k], phizdum, ds, dt,
                        txz0[i][j][k-1], - txz0[i][j][k-1] );

                        vx0[i][j][k] += (dt/rho[i][j][k])*( phitxxx[npml] + phitxyy[npml] + phitxzz[npml] )
                        + staggardv2 (1./rho[i][j][k], kappax[i], kappay[j], kappaz[k], dt, ds,
                        txx0[i-1][j][k], txx0[i][j][k],
                        txy0[i][j-1][k], txy0[i][j][k],
                        txz0[i][j][k-1], - txz0[i][j][k-1] );

			          } else { /* regular domain */

                        phixdum = phitxxx[npml];
                        phiydum = phitxyy[npml];
                        phizdum = phitxzz[npml];

                        phitxxx[npml] = CPML2 (vp[i][j][k], dumpx[i], alphax[i], kappax[i], phixdum, ds, dt,
                        txx0[i-1][j][k], txx0[i][j][k] );
                        phitxyy[npml] = CPML2 (vp[i][j][k], dumpy[j], alphay[j], kappay[j], phiydum, ds, dt,
                        txy0[i][j-1][k], txy0[i][j][k] );
                        phitxzz[npml] = CPML2 (vp[i][j][k], dumpz[k], alphaz[k], kappaz[k], phizdum, ds, dt,
                        txz0[i][j][k-1], txz0[i][j][k] );

                        vx0[i][j][k] += (dt/rho[i][j][k])*( phitxxx[npml] + phitxyy[npml] + phitxzz[npml] )
                        + staggardv2 (1./rho[i][j][k], kappax[i], kappay[j], kappaz[k], dt, ds,
                        txx0[i-1][j][k], txx0[i][j][k],
                        txy0[i][j-1][k], txy0[i][j][k],
                        txz0[i][j][k-1], txz0[i][j][k] );

                      } /* end of if "free surface" */
                    } /* end of calculation of vx */

                /* Calculation of vy */

                    if ( k >= ZMIN-delta+2 && jmp <= YMAX+delta && imp <= XMAX+delta ){

                      rhoxy = 0.25*(rho[i][j][k] + rho[i][j+1][k]
				                  + rho[i+1][j][k] + rho[i+1][j+1][k]);
                      muxy = 4./(1./mu[i][j][k] + 1./mu[i][j+1][k]
				               + 1./mu[i+1][j][k] + 1./mu[i+1][j+1][k]);
			          lamxy = 4./(1./lam[i][j][k] + 1./lam[i][j+1][k]
				                + 1./lam[i+1][j][k] + 1./lam[i+1][j+1][k]);
			          vpxy = sqrt ( (lamxy+2.*muxy)/( 0.25*( rho[i][j][k] + rho[i][j+1][k]
				                                            + rho[i+1][j][k] + rho[i+1][j+1][k] ) ) );

                      if ( k == 1 ){ /* free surface */

                        phixdum = phitxyx[npml];
                        phiydum = phityyy[npml];
                        phizdum = phityzz[npml];

                        phitxyx[npml] = CPML2 (vpxy, dumpx2[i], alphax2[i], kappax2[i], phixdum, ds, dt,
                        txy0[i][j][k], txy0[i+1][j][k] );
                        phityyy[npml] = CPML2 (vpxy, dumpy2[j], alphay2[j], kappay2[j], phiydum, ds, dt,
                        tyy0[i][j][k], tyy0[i][j+1][k] );
                        phityzz[npml] = CPML2 (vpxy, dumpz[k], alphaz[k], kappaz[k], phizdum, ds, dt,
                        tyz0[i][j][k-1], - tyz0[i][j][k-1] );

                        vy0[i][j][k] += (dt/rhoxy)*( phitxyx[npml] + phityyy[npml] + phityzz[npml] )
                        + staggardv2 (1./rhoxy, kappax2[i], kappay2[j], kappaz[k], dt, ds,
                        txy0[i][j][k], txy0[i+1][j][k],
                        tyy0[i][j][k], tyy0[i][j+1][k],
                        tyz0[i][j][k-1], - tyz0[i][j][k-1] );

			          } else { /* regular domain */

                        phixdum = phitxyx[npml];
                        phiydum = phityyy[npml];
                        phizdum = phityzz[npml];

                        phitxyx[npml] = CPML2 (vpxy, dumpx2[i], alphax2[i], kappax2[i], phixdum, ds, dt,
                        txy0[i][j][k], txy0[i+1][j][k] );
                        phityyy[npml] = CPML2 (vpxy, dumpy2[j], alphay2[j], kappay2[j], phiydum, ds, dt,
                        tyy0[i][j][k], tyy0[i][j+1][k] );
                        phityzz[npml] = CPML2 (vpxy, dumpz[k], alphaz[k], kappaz[k], phizdum, ds, dt,
                        tyz0[i][j][k-1], tyz0[i][j][k] );

                        vy0[i][j][k] += (dt/rhoxy)*( phitxyx[npml] + phityyy[npml] + phityzz[npml] )
                        + staggardv2 (1./rhoxy, kappax2[i], kappay2[j], kappaz[k], dt, ds,
                        txy0[i][j][k], txy0[i+1][j][k],
                        tyy0[i][j][k], tyy0[i][j+1][k],
                        tyz0[i][j][k-1], tyz0[i][j][k] );

	                  } /* end of if "free surface" */
	                } /* end of calculation of vy */

                /* Calculation of vz */

                    if ( jmp >= YMIN-delta+2 && imp <= XMAX+delta ){

                      rhoxz = 0.25*(rho[i][j][k] + rho[i][j][k+1]
				                  + rho[i+1][j][k] + rho[i+1][j][k+1]);
                      muxz = 4./(1./mu[i][j][k] + 1./mu[i][j][k+1]
				               + 1./mu[i+1][j][k] + 1./mu[i+1][j][k+1]);
			          lamxz = 4./(1./lam[i][j][k] + 1./lam[i][j][k+1]
				                + 1./lam[i+1][j][k] + 1./lam[i+1][j][k+1]);
			          vpxz = sqrt ( (lamxz+2.*muxz)/( 0.25*( rho[i][j][k] + rho[i][j][k+1]
				                                           + rho[i+1][j][k] + rho[i+1][j][k+1] ) ) );

                      if ( k == 1 ){ /* free surface */

                        phixdum = phitxzx[npml];
                        phiydum = phityzy[npml];
                        phizdum = phitzzz[npml];

                        phitxzx[npml] = CPML2 (vpxz, dumpx2[i], alphax2[i], kappax2[i], phixdum, ds, dt,
                        - txz0[i][j][k-1], - txz0[i+1][j][k-1] );
                        phityzy[npml] = CPML2 (vpxz, dumpy[j], alphay[j], kappay[j], phiydum, ds, dt,
                        - tyz0[i][j-1][k-1], - tyz0[i][j][k-1] );
                        phitzzz[npml] = CPML2 (vpxz, dumpz2[k], alphaz2[k], kappaz2[k], phizdum, ds, dt,
                        tzz0[i][j][k], - tzz0[i][j][k-1] );

                        vz0[i][j][k] += (dt/rhoxz)*( phitxzx[npml] + phityzy[npml] + phitzzz[npml] )
                        + staggardv2 (1./rhoxz, kappax2[i], kappay[j], kappaz2[k], dt, ds,
			            - txz0[i][j][k-1], - txz0[i+1][j][k-1],
			            - tyz0[i][j-1][k-1], - tyz0[i][j][k-1],
                        tzz0[i][j][k], - tzz0[i][j][k-1] );

			          } else { /* regular domain */

                        phixdum = phitxzx[npml];
                        phiydum = phityzy[npml];
                        phizdum = phitzzz[npml];

                        phitxzx[npml] = CPML2 (vpxz, dumpx2[i], alphax2[i], kappax2[i], phixdum, ds, dt,
                        txz0[i][j][k], txz0[i+1][j][k] );
                        phityzy[npml] = CPML2 (vpxz, dumpy[j], alphay[j], kappay[j], phiydum, ds, dt,
                        tyz0[i][j-1][k], tyz0[i][j][k] );
                        phitzzz[npml] = CPML2 (vpxz, dumpz2[k], alphaz2[k], kappaz2[k], phizdum, ds, dt,
                        tzz0[i][j][k], tzz0[i][j][k+1] );

                        vz0[i][j][k] += (dt/rhoxz)*( phitxzx[npml] + phityzy[npml] + phitzzz[npml] )
                        + staggardv2 (1./rhoxz, kappax2[i], kappay[j], kappaz2[k], dt, ds,
			            txz0[i][j][k], txz0[i+1][j][k],
			            tyz0[i][j-1][k], tyz0[i][j][k],
                        tzz0[i][j][k], tzz0[i][j][k+1] );

	                  } /* end of if "free surface" */
	                } /* end of calculation of vz */

                /* Normal mode */

                  } else {

                    rhoxy = 0.25*(rho[i][j][k] + rho[i][j+1][k]
				                + rho[i+1][j][k] + rho[i+1][j+1][k]);
			        rhoxz = 0.25*(rho[i][j][k] + rho[i][j][k+1]
				                + rho[i+1][j][k] + rho[i+1][j][k+1]);

                    if ( k == 1 ){ /* free surface */

                      vx0[i][j][k] += (1./rho[i][j][k])*fx[i][j][k]*dt/ds
                      + staggardv2 (1./rho[i][j][k], un, un, un, dt, ds,
                      txx0[i-1][j][k], txx0[i][j][k],
                      txy0[i][j-1][k], txy0[i][j][k],
                      txz0[i][j][k-1], - txz0[i][j][k-1] );

                      vy0[i][j][k] += (1./rhoxy)*fy[i][j][k]*dt/ds
                      + staggardv2 (1./rhoxy, un, un, un, dt, ds,
                      txy0[i][j][k], txy0[i+1][j][k],
                      tyy0[i][j][k], tyy0[i][j+1][k],
                      tyz0[i][j][k-1], - tyz0[i][j][k-1] );

                      vz0[i][j][k] += (1./rhoxz)*fz[i][j][k]*dt/ds
                      + staggardv2 (1./rhoxz, un, un, un, dt, ds,
                      - txz0[i][j][k-1], - txz0[i+1][j][k-1],
                      - tyz0[i][j-1][k-1], - tyz0[i][j][k-1],
                      tzz0[i][j][k], - tzz0[i][j][k-1] );

		            } else { /* regular domain */

                      vx0[i][j][k] += (1./rho[i][j][k])*fx[i][j][k]*dt/ds
                      + staggardv2 (1./rho[i][j][k], un, un, un, dt, ds,
                      txx0[i-1][j][k], txx0[i][j][k],
                      txy0[i][j-1][k], txy0[i][j][k],
                      txz0[i][j][k-1], txz0[i][j][k] );

                      vy0[i][j][k] += (1./rhoxy)*fy[i][j][k]*dt/ds
                      + staggardv2 (1./rhoxy, un, un, un, dt, ds,
                      txy0[i][j][k], txy0[i+1][j][k],
                      tyy0[i][j][k], tyy0[i][j+1][k],
                      tyz0[i][j][k-1], tyz0[i][j][k] );

                      vz0[i][j][k] += (1./rhoxz)*fz[i][j][k]*dt/ds
                      + staggardv2 (1./rhoxz, un, un, un, dt, ds,
                      txz0[i][j][k], txz0[i+1][j][k],
                      tyz0[i][j-1][k], tyz0[i][j][k],
                      tzz0[i][j][k], tzz0[i][j][k+1] );

		            } /* end of if "free surface" */
		          } /* end of normal mode */
	            } /* end of 2nd order finite-difference */
	          } /* end of k */
            } /* end of if jmp */
          } /* end of j */
	    } /* end of if imp */
	  } /* end of i */

                /* Boundary conditions */

                /* x boundary */

/* On suppose meme processeur avec XMIN-delta et XMIN-delta+1 ( CPML )
   idem pour XMAX + delta */

	  icpu = i2icpu_array[XMIN-delta];
      imp = i2imp_array[XMIN-delta];

      if ( coords[0] == icpu ){
        for(jmp = -1; jmp <= mpmy+2; jmp++){
   	      j = jmp2j_array[jmp];
	      if ( (j >= YMIN-delta) && ( j <= YMAX+delta+2 )) {
            for ( k = ZMIN-delta; k <= 1; k++){
              vx0[imp][jmp][k] = 0.0;
              vy0[imp][jmp][k] = 0.0;
              vz0[imp][jmp][k] = 0.0;
              vx0[imp+1][jmp][k] = 0.0;
              vy0[imp+1][jmp][k] = 0.0;
              vz0[imp+1][jmp][k] = 0.0;
	        }
	      }
	    }
	  }

      icpu = i2icpu_array[XMAX+delta];
      imp = i2imp_array[XMAX+delta];

      if ( coords[0] == icpu ){
 	    for(jmp = -1; jmp <= mpmy+2; jmp++){
 	      j = jmp2j_array[jmp];
	      if ( (j >= YMIN-delta) && ( j <= YMAX+delta+2 )) {
            for ( k = ZMIN-delta; k <= 1; k++){
              vx0[imp+1][jmp][k] = 0.0;
              vy0[imp+1][jmp][k] = 0.0;
              vz0[imp+1][jmp][k] = 0.0;
              vx0[imp+2][jmp][k] = 0.0;
              vy0[imp+2][jmp][k] = 0.0;
              vz0[imp+2][jmp][k] = 0.0;
	        }
          }
	    }
	  }

                /* y boundary */

/* On suppose meme processeur avec YMIN-delta et YMIN-delta+1 ( PML )
   idem pour YMAX + delta */

	  jcpu = j2jcpu_array[YMIN-delta];
      jmp = j2jmp_array[YMIN-delta];

      if ( coords[1] == jcpu ){
        for(imp = -1; imp <= mpmx+2; imp++){
   	      i = imp2i_array[imp];
	      if ( (i >= XMIN-delta) && ( i <= XMAX+delta+2 )) {
            for ( k = ZMIN-delta; k <= 1; k++){
              vx0[imp][jmp][k] = 0.0;
              vy0[imp][jmp][k] = 0.0;
              vz0[imp][jmp][k] = 0.0;
              vx0[imp][jmp+1][k] = 0.0;
              vy0[imp][jmp+1][k] = 0.0;
              vz0[imp][jmp+1][k] = 0.0;
	        }
	      }
	    }
	  }

      jcpu = j2jcpu_array[YMAX+delta];
      jmp = j2jmp_array[YMAX+delta];

      if ( coords[1] == jcpu ){
 	    for(imp = -1; imp <= mpmx+2; imp++){
 	      i = imp2i_array[imp];
	      if ( (i >= XMIN-delta) && ( i <= XMAX+delta+2 )) {
            for ( k = ZMIN-delta; k <= 1; k++){
              vx0[imp][jmp+1][k] = 0.0;
              vy0[imp][jmp+1][k] = 0.0;
              vz0[imp][jmp+1][k] = 0.0;
              vx0[imp][jmp+2][k] = 0.0;
              vy0[imp][jmp+2][k] = 0.0;
              vz0[imp][jmp+2][k] = 0.0;
	        }
          }
	    }
	  }

                /* z boundary */

      for ( imp = -1; imp <= mpmx+2; imp++ ){
   	    i = imp2i_array[imp];
	    for(jmp = -1; jmp <= mpmy+2; jmp++){
 	      j = jmp2j_array[jmp];
	      if ( (i >= XMIN-delta) && ( i <= XMAX+delta+2 ) && (j >= YMIN-delta) && ( j <= YMAX+delta+2 ) ) {
            vx0[imp][jmp][ZMIN-delta] = 0.0;
            vy0[imp][jmp][ZMIN-delta] = 0.0;
            vz0[imp][jmp][ZMIN-delta] = 0.0;
            vx0[imp][jmp][ZMIN-delta+1] = 0.0;
            vy0[imp][jmp][ZMIN-delta+1] = 0.0;
            vz0[imp][jmp][ZMIN-delta+1] = 0.0;
	      }
	    }
	  }

	} /* end of imode */

    #if (TAUGLOBAL)
      TAU_PROFILE_STOP (compute_vit);
    #endif

    #if (TIMING_BARRIER)
      MPI_Barrier (MPI_COMM_WORLD);
      timing2 = my_second ();
      timing_bc2 = timing_bc2 +  (timing2-timing1);
    #endif

    #if (TIMING)
      timing2 = my_second ();
      timing_bc2 = timing_bc2 + (timing2-timing1);
    #endif

    #if (TIMING_BARRIER)
      MPI_Barrier (MPI_COMM_WORLD);
      timing1 = my_second();
    #endif

    #if (TIMING)
      timing1 = my_second();
    #endif

    #if (TAU)
      TAU_PROFILE_START (exchange_vit);
    #endif

    #if (TAUGLOBAL)
     TAU_PROFILE_START (exchange_vit);
    #endif

                /* Communication : first method */

    #if (PERSISTANT)

      if ( nest != -1 )
        MPI_Waitall(3,&sendreq2[1],&sendstatus2[1]);

      if ( nouest != -1 )
        MPI_Waitall(3,&recvreq2[1],&recvstatus2[1]);

      if ( nouest != -1 )
        MPI_Waitall(3,&sendreq2[4],&sendstatus2[4]);

      if ( nest != -1 )
        MPI_Waitall(3,&recvreq2[4],&recvstatus2[4]);

                /* North - South */

      if( coords[1] != (py-1) )
        MPI_Wait(&sendreq3[2],&sendstatus3[2]);

      if( coords[1] != 0 )
        MPI_Wait(&recvreq3[2],&recvstatus3[2]);

      if( coords[1] != 0 )
        MPI_Wait(&sendreq4[2],&sendstatus4[2]);

      if( coords[1] != (py-1) )
        MPI_Wait(&recvreq4[2],&recvstatus4[2]);

      if( coords[1] != 0 ){
        i = 0;
        for( jmp = -1; jmp <= 0; jmp++ ){
	      for( imp = -1; imp <= mpmx+2; imp++ ){
            for( k = ZMIN-delta; k <= 1; k++ ){
              assert (i < (nmaxy + 1)) ;
              vx0[imp][jmp][k] = rxbuf3[i];
              vy0[imp][jmp][k] = rybuf3[i];
              vz0[imp][jmp][k] = rzbuf3[i];
              i = i + 1;
            }
          }
        }
      }

      if( coords[1] != (py-1) ){
        i = 0;
        for( jmp = mpmy+1; jmp <= mpmy+2; jmp++ ){
          for( imp = -1; imp <= mpmx+2; imp++ ){
            for( k = ZMIN-delta; k <= 1; k++ ){
              vx0[imp][jmp][k] = rxbuf4[i];
              vy0[imp][jmp][k] = rybuf4[i];
              vz0[imp][jmp][k] = rzbuf4[i];
              i = i + 1;
            }
          }
        }
      }

    #endif /* end of first method */

                /* Communication : second method */

    #if (NONBLOCKING)

                /* East - West */

      if( coords[0] != 0 )
        MPI_Wait(&recvreq2[1],&recvstatus2[1]);

      if( coords[0] != 0 ){
        i = 0;
        for( imp = -1; imp <= 0; imp++ ){
	      for( jmp = -1; jmp <= mpmy+2; jmp++ ){
            for( k = ZMIN-delta; k <= 1; k++ ){
              assert (i < (nmaxx + 1)) ;
              vx0[imp][jmp][k] = rxbuf[i];
              vy0[imp][jmp][k] = rybuf[i];
              vz0[imp][jmp][k] = rzbuf[i];
              i = i + 1;
            }
          }
        }
      }

      if( coords[0] != (px-1) )
        MPI_Wait(&recvreq2[2],&recvstatus2[2]);

      if( coords[0] != (px-1) ){
        i = 0;
        for( imp = mpmx+1; imp <= mpmx+2; imp++ ){
          for( jmp = -1; jmp <= mpmy+2; jmp++ ){
            for( k = ZMIN-delta; k <= 1; k++ ){
              vx0[imp][jmp][k] = rxbuf2[i];
              vy0[imp][jmp][k] = rybuf2[i];
              vz0[imp][jmp][k] = rzbuf2[i];
              i = i + 1;
            }
          }
        }
      }

                /* North - South */

      if( coords[1] != 0 )
        MPI_Wait(&sendreq2[4],&sendstatus2[4]);

      if( coords[1] != 0 )
        MPI_Wait(&recvreq2[3],&recvstatus2[3]);

      if( coords[1] != 0 ){
        i = 0;
        for( jmp = -1; jmp <= 0; jmp++ ){
	      for( imp = -1; imp <= mpmx+2; imp++ ){
            for( k = ZMIN-delta; k <= 1; k++ ){
              assert (i < (nmaxy + 1)) ;
              vx0[imp][jmp][k] = rxbuf3[i];
              vy0[imp][jmp][k] = rybuf3[i];
              vz0[imp][jmp][k] = rzbuf3[i];
              i = i + 1;
            }
          }
        }
      }

      if( coords[1] != (py-1) )
        MPI_Wait(&sendreq2[3],&sendstatus2[3]);

      if( coords[1] != (py-1) )
        MPI_Wait(&recvreq2[4],&recvstatus2[4]);

      if( coords[1] != (py-1) ){
        i = 0;
        for( jmp = mpmy+1; jmp <= mpmy+2; jmp++ ){
          for( imp = -1; imp <= mpmx+2; imp++ ){
            for( k = ZMIN-delta; k <= 1; k++ ){
              vx0[imp][jmp][k] = rxbuf4[i];
              vy0[imp][jmp][k] = rybuf4[i];
              vz0[imp][jmp][k] = rzbuf4[i];
              i = i + 1;
            }
          }
        }
      }

    #endif /* end of second method */

                /* Communication : third method */

    #if(BLOCKING)

			    /* Communication to synchronize */

			    /* East - West */
				/* Positive direction */

      if( coords[0] != (px-1) ){
        i = 0;
        for ( imp = mpmx-1; imp <= mpmx; imp++ ){
          for( jmp = -1; jmp <= mpmy +2; jmp++ ){
            for( k = ZMIN-delta; k <= 1; k++ ){
	          assert (i < (nmaxx + 1)) ;
              sxbuf[i] = vx0[imp][jmp][k];
              sybuf[i] = vy0[imp][jmp][k];
              szbuf[i] = vz0[imp][jmp][k];
              i = i + 1;
	        }
          }
        }
      }

      if( coords[0] != (px-1) )
        MPI_Isend(sxbuf, 3*nmaxx, MPI_DOUBLE, nest,
         1, MPI_COMM_WORLD, &req);

      if( coords[0] != 0 )
        MPI_Recv(rxbuf, 3*nmaxx, MPI_DOUBLE, nouest,
         1, MPI_COMM_WORLD, &status);

      if (px != 1 ){
        if( coords[0] != 0 ){
          i = 0;
          for( imp = -1; imp <= 0; imp++ ){
            for( jmp = -1; jmp <= mpmy +2; jmp++ ){
	          for( k = ZMIN-delta; k <= 1; k++ ){
		        assert (i < (nmaxx + 1)) ;
		        vx0[imp][jmp][k] = rxbuf[i];
		        vy0[imp][jmp][k] = rybuf[i];
		        vz0[imp][jmp][k] = rzbuf[i];
		        i = i + 1;
		      }
	        }
	      }
        } else {
	      MPI_Wait (&req, &status) ;
        }
      }

				/* Negative direction */

      if( coords[0] != 0 ){
        i = 0;
        for ( imp = 1; imp <= 2; imp++ ){
          for( jmp = -1; jmp <= mpmy +2; jmp++ ){
            for( k = ZMIN-delta; k <= 1; k++ ){
              sxbuf2[i] = vx0[imp][jmp][k];
              sybuf2[i] = vy0[imp][jmp][k];
              szbuf2[i] = vz0[imp][jmp][k];
              i = i + 1;
            }
          }
        }
      }

      if( coords[0] != 0 )
        MPI_Isend(sxbuf2, 3*nmaxx, MPI_DOUBLE, nouest,
         2, MPI_COMM_WORLD, &req);

      if( coords[0] != (px-1) )
        MPI_Recv(rxbuf2, 3*nmaxx, MPI_DOUBLE, nest,
         2, MPI_COMM_WORLD, &status);

      if ( px!=1 ) {
        if( coords[0] != (px-1) ){
	      i = 0;
	      for( imp = mpmx+1; imp <= mpmx+2; imp++ ){
            for( jmp = -1; jmp <= mpmy +2; jmp++ ){
	   	      for( k = ZMIN-delta; k <= 1; k++ ){
		        vx0[imp][jmp][k] = rxbuf2[i];
		        vy0[imp][jmp][k] = rybuf2[i];
		        vz0[imp][jmp][k] = rzbuf2[i];
		        i = i + 1;
		      }
	        }
	      }
        } else {
	      MPI_Wait (&req, &status) ;
        }
      }

                /* North - South */
                /* Positive direction */

      if( coords[1] != (py-1) ){
        i = 0;
        for ( jmp = mpmy-1; jmp <= mpmy; jmp++ ){
          for( imp = -1; imp <= mpmx +2; imp++ ){
            for( k = ZMIN-delta; k <= 1; k++ ){
	          assert (i < (nmaxy + 1)) ;
              sxbuf3[i] = vx0[imp][jmp][k];
              sybuf3[i] = vy0[imp][jmp][k];
              szbuf3[i] = vz0[imp][jmp][k];
              i = i + 1;
	        }
          }
        }
      }

      if( coords[1] != (py-1) )
        MPI_Isend(sxbuf3, 3*nmaxy, MPI_DOUBLE, nnord,
         1, MPI_COMM_WORLD, &req);

      if( coords[1] != 0 )
        MPI_Recv(rxbuf3, 3*nmaxy, MPI_DOUBLE, nsud,
         1, MPI_COMM_WORLD, &status);

      if ( py!=1 ){
        if( coords[1] != 0 ){
          i = 0;
          for( jmp = -1; jmp <= 0; jmp++ ){
            for( imp = -1; imp <= mpmx +2; imp++ ){
	          for( k = ZMIN-delta; k <= 1; k++ ){
	   	        assert (i < (nmaxy + 1)) ;
		        vx0[imp][jmp][k] = rxbuf3[i];
		        vy0[imp][jmp][k] = rybuf3[i];
		        vz0[imp][jmp][k] = rzbuf3[i];
		        i = i + 1;
		      }
	        }
	      }
        } else {
	      MPI_Wait (&req, &status) ;
        }
      }

				/* Negative direction */

      if( coords[1] != 0 ){
        i = 0;
        for ( jmp = 1; jmp <= 2; jmp++ ){
          for( imp = -1; imp <= mpmx +2; imp++ ){
            for( k = ZMIN-delta; k <= 1; k++ ){
              sxbuf4[i] = vx0[imp][jmp][k];
              sybuf4[i] = vy0[imp][jmp][k];
              szbuf4[i] = vz0[imp][jmp][k];
              i = i + 1;
            }
          }
        }
      }

      if( coords[1] != 0 )
        MPI_Isend(sxbuf4, 3*nmaxy, MPI_DOUBLE, nsud,
         2, MPI_COMM_WORLD, &req);

      if( coords[1] != (py-1) )
        MPI_Recv(rxbuf4, 3*nmaxy, MPI_DOUBLE, nnord,
         2, MPI_COMM_WORLD, &status);

      if (py != 1) {
        if( coords[1] != (py-1) ){
	      i = 0;
	      for( jmp = mpmy+1; jmp <= mpmy+2; jmp++ ){
            for( imp = -1; imp <= mpmx +2; imp++ ){
	   	      for( k = ZMIN-delta; k <= 1; k++ ){
		        vx0[imp][jmp][k] = rxbuf4[i];
		        vy0[imp][jmp][k] = rybuf4[i];
		        vz0[imp][jmp][k] = rzbuf4[i];
		        i = i + 1;
		      }
	        }
	      }
        } else {
	      MPI_Wait (&req, &status) ;
        }
      }

    #endif /* end of third method */

    #if (TAU)
      TAU_PROFILE_STOP (exchange_vit);
    #endif

    #if (TAUGLOBAL)
      TAU_PROFILE_STOP (exchange_vit);
    #endif

    #if (TIMING_BARRIER)
      MPI_Barrier (MPI_COMM_WORLD);
      timing2 = my_second ();
      timing_comm2 = timing_comm2 + (timing2-timing1);
    #endif

    #if (TIMING)
      timing2 = my_second ();
      timing_comm2 = timing_comm2 + (timing2-timing1);
    #endif

    #if (TIMING_BARRIER)
      MPI_Barrier (MPI_COMM_WORLD);
      timing1 = my_second ();
    #endif

    #if (TIMING)
      timing1 = my_second ();
    #endif

				/* Calculation of the seismograms */

    for ( ir = 0; ir < IOBS; ir++ ){
      if( ista[ir] == 1 ){

/* imp et jmp peuvent etre decales de +1 mais le recouvrement de +2 / -2 assure la cohérence */

/* Vx component */

        i = ixobs[ir];
        w1 = xobswt[ir];
        j = iyobs[ir];
        w2 = yobswt[ir];
        k = izobs[ir];
        w3 = zobswt[ir];

        icpu = i2icpu_array[i];
        jcpu = j2jcpu_array[j];

	    if ( coords[0] == icpu ) {
          if ( coords[1] == jcpu ) {

            imp = i2imp_array[i];
            jmp = j2jmp_array[j];

            seisx[ir][l-1] = (1.-w3)*(
                   (1.-w2)*( (1.-w1)*vx0[imp][jmp][k]     + w1*vx0[imp+1][jmp][k] )
                      + w2*( (1.-w1)*vx0[imp][jmp+1][k]   + w1*vx0[imp+1][jmp+1][k] ) )
            + w3*( (1.-w2)*( (1.-w1)*vx0[imp][jmp][k+1]   + w1*vx0[imp+1][jmp][k+1] )
                      + w2*( (1.-w1)*vx0[imp][jmp+1][k+1] + w1*vx0[imp+1][jmp+1][k+1] ) );

	        MPI_Isend (&seisx[ir][l-1], 1, MPI_DOUBLE, 0, 81, comm2d, &sendreq[1]);
	        MPI_Wait (&sendreq[1], &status);

          }
        }

/* Vy component */

        if(xobswt[ir] >= 0.5){
          w1 = xobswt[ir] - 0.5;
          i = ixobs[ir];
        } else {
          w1 = xobswt[ir] + 0.5;
          i = ixobs[ir]-1;
        }
        if(yobswt[ir] >= 0.5){
          w2 = yobswt[ir] - 0.5;
          j = iyobs[ir];
        } else {
          w2 = yobswt[ir] + 0.5;
          j = iyobs[ir]-1;
        }
        k = izobs[ir];
        w3 = zobswt[ir];

        icpu = i2icpu_array[i];
        jcpu = j2jcpu_array[j];

	    if ( coords[0] == icpu ) {
          if ( coords[1] == jcpu ) {

            imp = i2imp_array[i];
            jmp = j2jmp_array[j];

            seisy[ir][l-1] = (1.-w3)*(
                   (1.-w1)*( (1.-w2)*vy0[imp][jmp][k]     + w2*vy0[imp][jmp+1][k])
                      + w1*( (1.-w2)*vy0[imp+1][jmp][k]   + w2*vy0[imp+1][jmp+1][k] ) )
            + w3*( (1.-w1)*( (1.-w2)*vy0[imp][jmp][k+1]   + w2*vy0[imp][jmp+1][k+1] )
                      + w1*( (1.-w2)*vy0[imp+1][jmp][k+1] + w2*vy0[imp+1][jmp+1][k+1] ) );

	        MPI_Isend (&seisy[ir][l-1], 1, MPI_DOUBLE, 0, 82, comm2d, &sendreq[2]);
	        MPI_Wait (&sendreq[2], &status);

          }
        }

/* Vz component */

        if(xobswt[ir] >= 0.5){
          w1 = xobswt[ir] - 0.5;
          i = ixobs[ir];
        } else {
          w1 = xobswt[ir] + 0.5;
          i = ixobs[ir]-1;
        }
        j = iyobs[ir];
        w2 = yobswt[ir];
        if(zobswt[ir] >= 0.5){
          w3 = zobswt[ir] - 0.5;
          k = izobs[ir];
        } else {
          w3 = zobswt[ir] + 0.5;
          k = izobs[ir]-1;
        }
        if( izobs[ir] == 1 ){
          w3 = 0.0;
          k = 0;
        }
        icpu = i2icpu_array[i];
        jcpu = j2jcpu_array[j];

	    if ( coords[0] == icpu ) {
          if ( coords[1] == jcpu ) {

            imp = i2imp_array[i];
            jmp = j2jmp_array[j];

            seisz[ir][l-1] = (1.-w3)*(
                   (1.-w1)*( (1.-w2)*vz0[imp][jmp][k]     + w2*vz0[imp][jmp+1][k] )
                      + w1*( (1.-w2)*vz0[imp+1][jmp][k]   + w2*vz0[imp+1][jmp+1][k] ) )
            + w3*( (1.-w1)*( (1.-w2)*vz0[imp][jmp][k+1]   + w2*vz0[imp][jmp+1][k+1] )
                      + w1*( (1.-w2)*vz0[imp+1][jmp][k+1] + w2*vz0[imp+1][jmp+1][k+1] ) );

	        MPI_Isend (&seisz[ir][l-1], 1, MPI_DOUBLE, 0, 83, comm2d, &sendreq[3]);
	        MPI_Wait (&sendreq[3], &status);

          }
        }

/* Tii component */

        if(xobswt[ir] >= 0.5){
          w1 = xobswt[ir] - 0.5;
          i = ixobs[ir];
        } else {
          w1 = xobswt[ir] + 0.5;
          i = ixobs[ir]-1;
        }
        j = iyobs[ir];
        w2 = yobswt[ir];
        k = izobs[ir];
        w3 = zobswt[ir];

        icpu = i2icpu_array[i];
        jcpu = j2jcpu_array[j];

	    if ( coords[0] == icpu ) {
          if ( coords[1] == jcpu ) {

            imp = i2imp_array[i];
            jmp = j2jmp_array[j];

            seisxx[ir][l-1] = (1.-w3)*(
                   (1.-w1)*( (1.-w2)*txx0[imp][jmp][k]     + w2*txx0[imp][jmp+1][k])
                      + w1*( (1.-w2)*txx0[imp+1][jmp][k]   + w2*txx0[imp+1][jmp+1][k] ) )
            + w3*( (1.-w1)*( (1.-w2)*txx0[imp][jmp][k+1]   + w2*txx0[imp][jmp+1][k+1] )
                      + w1*( (1.-w2)*txx0[imp+1][jmp][k+1] + w2*txx0[imp+1][jmp+1][k+1] ) );

            seisyy[ir][l-1] = (1.-w3)*(
                   (1.-w1)*( (1.-w2)*tyy0[imp][jmp][k]     + w2*tyy0[imp][jmp+1][k] )
                      + w1*( (1.-w2)*tyy0[imp+1][jmp][k]   + w2*tyy0[imp+1][jmp+1][k] ) )
            + w3*( (1.-w1)*( (1.-w2)*tyy0[imp][jmp][k+1]   + w2*tyy0[imp][jmp+1][k+1] )
                      + w1*( (1.-w2)*tyy0[imp+1][jmp][k+1] + w2*tyy0[imp+1][jmp+1][k+1] ) );

            seiszz[ir][l-1] = (1.-w3)*(
                   (1.-w1)*( (1.-w2)*tzz0[imp][jmp][k]     + w2*tzz0[imp][jmp+1][k] )
                      + w1*( (1.-w2)*tzz0[imp+1][jmp][k]   + w2*tzz0[imp+1][jmp+1][k] ) )
            + w3*( (1.-w1)*( (1.-w2)*tzz0[imp][jmp][k+1]   + w2*tzz0[imp][jmp+1][k+1] )
                      + w1*( (1.-w2)*tzz0[imp+1][jmp][k+1] + w2*tzz0[imp+1][jmp+1][k+1] ) );

	        MPI_Isend (&seisxx[ir][l-1], 1, MPI_DOUBLE, 0, 84, comm2d, &sendreq[4]);
	        MPI_Wait (&sendreq[4], &status);
	        MPI_Isend (&seisyy[ir][l-1], 1, MPI_DOUBLE, 0, 85, comm2d, &sendreq[5]);
	        MPI_Wait (&sendreq[5], &status);
	        MPI_Isend (&seiszz[ir][l-1], 1, MPI_DOUBLE, 0, 86, comm2d, &sendreq[6]);
	        MPI_Wait (&sendreq[6], &status);

          }
        }

/* Txy component */

        i = ixobs[ir];
        w1 = xobswt[ir];
        if(yobswt[ir] >= 0.5){
          w2 = yobswt[ir] - 0.5;
          j = iyobs[ir];
        } else {
          w2 = yobswt[ir] + 0.5;
          j = iyobs[ir]-1;
        }
        k = izobs[ir];
        w3 = zobswt[ir];

        icpu = i2icpu_array[i];
        jcpu = j2jcpu_array[j];

	    if ( coords[0] == icpu ) {
          if ( coords[1] == jcpu ) {

            imp = i2imp_array[i];
            jmp = j2jmp_array[j];

            seisxy[ir][l-1] = (1.-w3)*(
                   (1.-w1)*( (1.-w2)*txy0[imp][jmp][k]     + w2*txy0[imp][jmp+1][k] )
                      + w1*( (1.-w2)*txy0[imp+1][jmp][k]   + w2*txy0[imp+1][jmp+1][k] ) )
            + w3*( (1.-w1)*( (1.-w2)*txy0[imp][jmp][k+1]   + w2*txy0[imp][jmp+1][k+1] )
                      + w1*( (1.-w2)*txy0[imp+1][jmp][k+1] + w2*txy0[imp+1][jmp+1][k+1] ) );

	        MPI_Isend (&seisxy[ir][l-1], 1, MPI_DOUBLE, 0, 87, comm2d, &sendreq[7]);
	        MPI_Wait (&sendreq[7], &status);

          }
        }

/* Txz component */

        i = ixobs[ir];
        w1 = xobswt[ir];
        j = iyobs[ir];
        w2 = yobswt[ir];
        if(zobswt[ir] >= 0.5){
          w3 = zobswt[ir] - 0.5;
          k = izobs[ir];
        } else {
          w3 = zobswt[ir] + 0.5;
          k = izobs[ir]-1;
        }
        if( izobs[ir] == 1 ){
          w3 = 0.0;
          k = 0;
        }

        icpu = i2icpu_array[i];
        jcpu = j2jcpu_array[j];

	    if ( coords[0] == icpu ) {
          if ( coords[1] == jcpu ) {

            imp = i2imp_array[i];
            jmp = j2jmp_array[j];

            seisxz[ir][l-1] = (1.-w3)*(
                   (1.-w1)*( (1.-w2)*txz0[imp][jmp][k]     + w2*txz0[imp][jmp+1][k] )
                      + w1*( (1.-w2)*txz0[imp+1][jmp][k]   + w2*txz0[imp+1][jmp+1][k] ) )
            + w3*( (1.-w1)*( (1.-w2)*txz0[imp][jmp][k+1]   + w2*txz0[imp][jmp+1][k+1] )
                      + w1*( (1.-w2)*txz0[imp+1][jmp][k+1] + w2*txz0[imp+1][jmp+1][k+1] ) );

	        MPI_Isend (&seisxz[ir][l-1], 1, MPI_DOUBLE, 0, 88, comm2d, &sendreq[8]);
	        MPI_Wait (&sendreq[8], &status);

          }
        }

/* Tyz component */

        if(xobswt[ir] >= 0.5){
          w1 = xobswt[ir] - 0.5;
          i = ixobs[ir];
        } else {
          w1 = xobswt[ir] + 0.5;
          i = ixobs[ir]-1;
        }
        if(yobswt[ir] >= 0.5){
          w2 = yobswt[ir] - 0.5;
          j = iyobs[ir];
        } else {
          w2 = yobswt[ir] + 0.5;
          j = iyobs[ir]-1;
        }
        if(zobswt[ir] >= 0.5){
          w3 = zobswt[ir] - 0.5;
          k = izobs[ir];
        } else {
          w3 = zobswt[ir] + 0.5;
          k = izobs[ir]-1;
        }
        if( izobs[ir] == 1 ){
          w3 = 0.0;
          k = 0;
        }

        icpu = i2icpu_array[i];
        jcpu = j2jcpu_array[j];

	    if ( coords[0] == icpu ) {
          if ( coords[1] == jcpu ) {

            imp = i2imp_array[i];
            jmp = j2jmp_array[j];

            seisyz[ir][l-1] = (1.-w3)*(
                   (1.-w1)*( (1.-w2)*tyz0[imp][jmp][k]     + w2*tyz0[imp][jmp+1][k] )
                      + w1*( (1.-w2)*tyz0[imp+1][jmp][k]   + w2*tyz0[imp+1][jmp+1][k] ) )
            + w3*( (1.-w1)*( (1.-w2)*tyz0[imp][jmp][k+1]   + w2*tyz0[imp][jmp+1][k+1] )
                      + w1*( (1.-w2)*tyz0[imp+1][jmp][k+1] + w2*tyz0[imp+1][jmp+1][k+1] ) );

	        MPI_Isend (&seisyz[ir][l-1], 1, MPI_DOUBLE, 0, 89, comm2d, &sendreq[9]);
	        MPI_Wait (&sendreq[9], &status) ;

          }
        }

	    if ( my_rank == 0 ){
	      for ( i = 1; i < 10; i++){
	        seis_output[l-1][ir][i] = 0.0;
	       }
    	   for ( i = 1; i < 10; i++){
	         j = 80 +i;
	         MPI_Recv (&seis_output[l-1][ir][i], 1, MPI_DOUBLE, mapping_seis[ir][i], j, comm2d, &status);
	       }
	     }

      }	/* end of if ista */
    } /* end of ir */

                /* Output : files to create snapshots */

	strcpy(flname1, outdir);
	strcpy(flname2, outdir);
	strcpy(flname3, outdir);

	strcat(flname1, char1);
	strcat(flname2, char2);
	strcat(flname3, char3);

	sprintf(number, "%4.4d", l);
	strcat(flname1, number);
	strcat(flname2, number);
	strcat(flname3, number);

	sprintf(number, "%2.2d", 0);
	strcat(flname1, number);
	strcat(flname2, number);
	strcat(flname3, number);

/* flname construit avec
   outdir + surface xy/xz/yz + numero dt  + 0
   surfacexyXXXX00 */

/* Sortie des plans
   fp1 --> k = 1
   fp2 --> j= j0
   fp3 --> i= i0 */

    if ( (l % SURFACE_STEP ) == 0 ){

                /* Writing of surfacexy */

      Vxglobal = dmatrix(XMIN-delta,XMAX+delta+2,YMIN-delta,YMAX+delta+2);
      Vyglobal = dmatrix(XMIN-delta,XMAX+delta+2,YMIN-delta,YMAX+delta+2);
      Vzglobal = dmatrix(XMIN-delta,XMAX+delta+2,YMIN-delta,YMAX+delta+2);

      if (my_rank == 0 ) {
        for ( i = 1; i <= mpmx; i++ ){
          for ( j = 1; j <= mpmy; j++ ){
            Vxglobal[XMIN-delta+i-1][YMIN-delta+j-1] = vx0[i][j][1];
            Vyglobal[XMIN-delta+i-1][YMIN-delta+j-1] = vy0[i][j][1];
            Vzglobal[XMIN-delta+i-1][YMIN-delta+j-1] = vz0[i][j][0];
          }
        }
      }

      for (i1= 1; i1 <=4; i1++ ){

        if (my_rank != 0 ) {

	      if (i1 == 1 ){
	        MPI_Isend(coords,2,MPI_INT,0,90,comm2d,&sendreq[4]);
		    MPI_Wait (&sendreq[4], &status) ;
	      } /* end of i1 = 1 */

	      if (i1 == 2 ){
            imp=0;
  	        for ( i = 1; i <= mpmx; i++ ){
              for ( j = 1; j <= mpmy; j++ ){
                assert (imp < test_size) ;
                imp ++;
                Vxtemp1[imp] = vx0[i][j][1];
              }
            }
            MPI_Isend(Vxtemp1,test_size,MPI_DOUBLE,0,80,comm2d,&sendreq[1]);
		    MPI_Wait (&sendreq[1], &status) ;
	      } /* end of i1 = 2 */

          if (i1 == 3 ){
            imp=0;
            for ( i = 1; i <= mpmx; i++ ){
              for ( j = 1; j <= mpmy; j++ ){
                assert (imp < test_size) ;
                imp ++;
                Vxtemp1[imp] = vy0[i][j][1];
              }
            }
            MPI_Isend(Vxtemp1,test_size,MPI_DOUBLE,0,81,comm2d,&sendreq[2]);
		    MPI_Wait (&sendreq[2], &status) ;
	      } /* end of i1 = 3 */

          if (i1 == 4 ){
            imp=0;
            for ( i = 1; i <= mpmx; i++ ){
              for ( j = 1; j <= mpmy; j++ ){
                assert (imp < test_size) ;
                imp ++;
                Vxtemp1[imp] = vz0[i][j][0];
              }
            }
            MPI_Isend(Vxtemp1,test_size,MPI_DOUBLE,0,82,comm2d,&sendreq[3]);
		    MPI_Wait (&sendreq[3], &status) ;
	      } /* end of i1 = 4 */

        } else { /* my_rank = 0 */

          for ( i2 = 1; i2 < np; i2++ ){

	        if ( i1 == 1 ){
              MPI_Recv(proc_coords,2,MPI_INT,i2,90,comm2d,&status);
	          coords_global[0][i2] = proc_coords[0];
	          coords_global[1][i2] = proc_coords[1];
	        } /* end of i1 = 1 */

	        if ( i1 == 2 ){
	          MPI_Recv(Vxtemp1,test_size, MPI_DOUBLE, i2,80, comm2d,&status);
              total_prec_x = 0;
              for (j=0;j<coords_global[0][i2];j++) {
                total_prec_x = total_prec_x + mpmx_tab[j];
              }
              total_prec_y = 0;
              for (j=0;j<coords_global[1][i2];j++) {
                total_prec_y = total_prec_y + mpmy_tab[j];
              }
	          imp = 0;
              for ( i = 1; i <= mpmx_tab[coords_global[0][i2]]; i++ ){
                for ( j = 1; j <= mpmy_tab[coords_global[1][i2]]; j++ ){
                  assert (imp < test_size) ;
		  	      assert ( XMIN-delta-1+i+total_prec_x < XMAX+delta+3 );
			      assert ( YMIN-delta-1+j+total_prec_y < YMAX+delta+3 );
                  imp ++;
                  Vxglobal[XMIN-delta-1+i+total_prec_x][YMIN-delta-1+j+total_prec_y] = Vxtemp1[imp];
                }
              }
	        } /* end of i1 = 2 */

            if ( i1 == 3 ){
           	  MPI_Recv(Vxtemp1,test_size, MPI_DOUBLE, i2,81, comm2d,&status);
              total_prec_x = 0;
              for (j=0;j<coords_global[0][i2];j++) {
                total_prec_x = total_prec_x + mpmx_tab[j];
              }
              total_prec_y = 0;
              for (j=0;j<coords_global[1][i2];j++) {
                total_prec_y = total_prec_y + mpmy_tab[j];
              }
              imp = 0;
              for ( i = 1; i <= mpmx_tab[coords_global[0][i2]]; i++ ){
                for ( j = 1; j <= mpmy_tab[coords_global[1][i2]]; j++ ){
                  assert (imp < test_size) ;
                  assert ( XMIN-delta-1+i+total_prec_x < XMAX+delta+3 );
                  assert ( YMIN-delta-1+j+total_prec_y < YMAX+delta+3 );
                  imp ++;
	              Vyglobal[XMIN-delta-1+i+total_prec_x][YMIN-delta-1+j+total_prec_y] = Vxtemp1[imp];
                }
              }
	        } /* end of i1 = 3 */

            if ( i1 == 4 ){
 	          MPI_Recv(Vxtemp1,test_size, MPI_DOUBLE, i2,82, comm2d,&status);
	          total_prec_x = 0;
              for (j=0;j<coords_global[0][i2];j++) {
                total_prec_x = total_prec_x + mpmx_tab[j];
              }
              total_prec_y = 0;
              for (j=0;j<coords_global[1][i2];j++) {
                total_prec_y = total_prec_y + mpmy_tab[j];
              }
              imp = 0;
              for ( i = 1; i <= mpmx_tab[coords_global[0][i2]]; i++ ){
                for ( j = 1; j <= mpmy_tab[coords_global[1][i2]]; j++ ){
                  assert (imp < test_size) ;
                  assert ( XMIN-delta-1+i+total_prec_x < XMAX+delta+3 );
                  assert ( YMIN-delta-1+j+total_prec_y < YMAX+delta+3 );
                  imp ++;
	              Vzglobal[XMIN-delta-1+i+total_prec_x][YMIN-delta-1+j+total_prec_y] = Vxtemp1[imp];
                }
              }
	        } /* end of i1 = 4 */
	      } /* end of if my_rank */
        } /* end of i2 */
	  } /* end of i1 */

                /* Ecriture */

	  if ( my_rank == 0 ) {
#if  (VTK)

        int ndiv =1;
        double dssurf=ds*ndiv;

        int XMINS=(int) ceil((XMIN-delta)/ndiv);
        int XMAXS=(int) floor((XMAX+delta)/ndiv);
        int dimx= XMAXS-XMINS+1;
        int YMINS=(int) ceil((YMIN-delta)/ndiv);
        int YMAXS=(int) floor((YMAX+delta)/ndiv);
        int dimy= YMAXS-YMINS+1;

        strcpy(flname1, outdir);
        strcat(flname1, char1);
        sprintf(number, "%4.4d", l);
        strcat(flname1, number);
        strcat(flname1, ".vtk");
        fp1 = fopen(flname1, "w");

        /* print VTK header*/
        fprintf(fp1,"# vtk DataFile Version 3.0\n");
        fprintf(fp1,"V\n");
        fprintf(fp1,"BINARY\n");
        fprintf(fp1,"DATASET STRUCTURED_POINTS\n");
        fprintf(fp1,"DIMENSIONS %d %d %d\n",dimx,dimy,1);
        fprintf(fp1,"ORIGIN %f %f %f\n",XMINS*dssurf,YMINS*dssurf,0.);
        fprintf(fp1,"SPACING %f %f %f\n",dssurf,dssurf,dssurf);
        fprintf(fp1,"POINT_DATA %d\n",dimx*dimy*1);
        fprintf(fp1,"SCALARS V float 3\n");
        fprintf(fp1,"LOOKUP_TABLE default\n");

        for ( j = YMIN-delta+1; j <= YMAX+delta+1; j++ ){
          for ( i = XMIN-delta+1; i <= XMAX+delta+1; i++ ){
            if( ((i-1)%ndiv) == 0 && ((j-1)%ndiv) == 0 ){
              write_float(fp1,(float) Vxglobal[i][j]);
              write_float(fp1,(float) Vyglobal[i][j]);
              write_float(fp1,(float) Vzglobal[i][j]);
            }
          }
        }
        fclose(fp1);

#else
	    fp1 = fopen(flname1, "w");
        for ( i = XMIN-delta+1; i <= XMAX+delta+1; i++ ){
          for ( j = YMIN-delta+1; j <= YMAX+delta+1; j++ ){
            if( ( ((int)(ds*(i-1)))%200) == 0 && ( ((int)(ds*(j-1)))%200) == 0 ){
              fprintf(fp1, "%7.2f %7.2f %8.3e %8.3e %8.3e \n",
                  ds*(i-1)/1000., ds*(j-1)/1000.,
		          Vxglobal[i][j], Vyglobal[i][j], Vzglobal[i][j] );
            }
          }
        }

        fclose(fp1);
#endif
	  }

/* Desallocation */

	  free_dmatrix(Vxglobal,XMIN-delta,XMAX+delta+2,YMIN-delta,YMAX+delta+2);
	  free_dmatrix(Vyglobal,XMIN-delta,XMAX+delta+2,YMIN-delta,YMAX+delta+2);
	  free_dmatrix(Vzglobal,XMIN-delta,XMAX+delta+2,YMIN-delta,YMAX+delta+2);
#if  (!VTK)
                /* Writing of surfacexz */

      Vxglobal = dmatrix(XMIN-delta,XMAX+delta+2,ZMIN-delta, 1);
      Vyglobal = dmatrix(XMIN-delta,XMAX+delta+2, ZMIN-delta, 1);
      Vzglobal = dmatrix(XMIN-delta,XMAX+delta+2, ZMIN-delta, 1);

/* Recherche du processeur avec j0 et coords[0] = 0
   on connait ses coordonnées (0,jcpu)
   donc rang  = jcpu*px */

	  jcpu = j2jcpu_array[j0];
	  jmp_tmp =  j2jmp_array[j0];

	  if ( my_rank  == jcpu*px ) {
        for ( i = 1; i <= mpmx; i++ ){
	      for ( k = ZMIN-delta; k <= 1; k++ ){
            Vxglobal[XMIN-delta+i-1][k] = vx0[i][jmp_tmp][k];
            Vyglobal[XMIN-delta+i-1][k] = vy0[i][jmp_tmp][k];
            Vzglobal[XMIN-delta+i-1][k] = vz0[i][jmp_tmp][k];
          }
        }
      }

/* Etape 1 */

/* GTS */

      for (i1= 1; i1 <=4; i1++ ){

        if (( coords[1] == jcpu ) && ( my_rank != px*jcpu ) )  {

	      if (i1 == 1 ){
	        MPI_Isend(coords,2,MPI_INT,jcpu*px,90,comm2d,&sendreq[4]);
		    MPI_Wait (&sendreq[4], &status) ;
	      } /* end of i1 = 1 */

	      if (i1 == 2 ){
            imp=0;
  	        for ( i = 1; i <= mpmx; i++ ){
	          for ( k = ZMIN-delta; k <= 1; k++ ){
                assert (imp < test_size) ;
                imp ++;
                Vxtemp1[imp] = vx0[i][jmp_tmp][k];
              }
            }
            MPI_Isend(Vxtemp1,test_size,MPI_DOUBLE,jcpu*px,80,comm2d,&sendreq[1]);
		    MPI_Wait (&sendreq[1], &status) ;
	      } /* end of i1 = 2 */

          if (i1 == 3 ){
            imp=0;
            for ( i = 1; i <= mpmx; i++ ){
              for ( k = ZMIN-delta; k <= 1; k++ ){
                assert (imp < test_size) ;
                imp ++;
                Vxtemp1[imp] = vy0[i][jmp_tmp][k];
              }
            }
            MPI_Isend(Vxtemp1,test_size,MPI_DOUBLE,jcpu*px,81,comm2d,&sendreq[2]);
		    MPI_Wait (&sendreq[2], &status) ;
	      } /* end of i1 = 3 */

          if (i1 == 4 ){
            imp=0;
            for ( i = 1; i <= mpmx; i++ ){
	          for ( k = ZMIN-delta; k <= 1; k++ ){
                assert (imp < test_size) ;
                imp ++;
                Vxtemp1[imp] = vz0[i][jmp_tmp][k];
              }
            }
            MPI_Isend(Vxtemp1,test_size,MPI_DOUBLE,jcpu*px,82,comm2d,&sendreq[3]);
		    MPI_Wait (&sendreq[3], &status) ;
	      } /* end of i1 = 4 */

        } /* end of if jcpu && my_rank */

        if ( my_rank == jcpu*px ){

          for ( i2 = 1; i2 < px ; i2++ ){

	        if ( i1 == 1 ){
              MPI_Recv(proc_coords,2,MPI_INT,jcpu*px+i2,90,comm2d,&status);
	          coords_global[0][i2] = proc_coords[0];
	          coords_global[1][i2] = proc_coords[1];
	        } /* end of i1 = 1 */

	        if ( i1 == 2 ){
	          MPI_Recv(Vxtemp1,test_size, MPI_DOUBLE, jcpu*px+i2,80, comm2d,&status);
              total_prec_x = 0;
              for (j=0;j<coords_global[0][i2];j++) {
                total_prec_x = total_prec_x + mpmx_tab[j];
              }
 	          imp = 0;
              for ( i = 1; i <= mpmx_tab[coords_global[0][i2]]; i++ ){
	            for ( k = ZMIN-delta; k <= 1; k++ ){
                  assert (imp < test_size) ;
		          assert ( XMIN-delta-1+i+total_prec_x < XMAX+delta+3 );
                  imp ++;
                  Vxglobal[XMIN-delta-1+i+total_prec_x][k] = Vxtemp1[imp];
                }
              }
	        } /* end of i1 = 2 */

            if ( i1 == 3 ){
         	  MPI_Recv(Vxtemp1,test_size, MPI_DOUBLE, jcpu*px+i2,81, comm2d,&status);
              total_prec_x = 0;
              for (j=0;j<coords_global[0][i2];j++) {
                total_prec_x = total_prec_x + mpmx_tab[j];
              }
              imp = 0;
              for ( i = 1; i <= mpmx_tab[coords_global[0][i2]]; i++ ){
	            for ( k = ZMIN-delta; k <= 1; k++ ){
                  assert (imp < test_size) ;
                  assert ( XMIN-delta-1+i+total_prec_x < XMAX+delta+3 );
                  imp ++;
	              Vyglobal[XMIN-delta-1+i+total_prec_x][k] = Vxtemp1[imp];
                }
              }
	        } /* end of i1 = 3 */

            if ( i1 == 4 ){
 	          MPI_Recv(Vxtemp1,test_size, MPI_DOUBLE, jcpu*px+i2,82, comm2d,&status);
	          total_prec_x = 0;
              for (j=0;j<coords_global[0][i2];j++) {
                total_prec_x = total_prec_x + mpmx_tab[j];
              }
              imp = 0;
              for ( i = 1; i <= mpmx_tab[coords_global[0][i2]]; i++ ){
     		    for ( k = ZMIN-delta; k <= 1; k++ ){
                  assert (imp < test_size) ;
                  assert ( XMIN-delta-1+i+total_prec_x < XMAX+delta+3 );
                  imp ++;
	              Vzglobal[XMIN-delta-1+i+total_prec_x][k] = Vxtemp1[imp];
                }
              }
	        } /* end of i1 = 4 */
  	      } /* end of i2 */
        } /* end of if my_rank */
	  } /* end of i1 */

                /* Ecriture */

      if ( my_rank == jcpu*px ) {

	    fp2 = fopen(flname2, "w");
        for ( i = XMIN-delta+1; i <= XMAX+delta+1; i++ ){
          for ( k = ZMIN-delta+1; k <= 1; k++ ){
            if( ( ((int)(ds*(i-1)))%200) == 0 && ( ((int)(ds*(k-1)))%200) == 0 ){
              fprintf(fp2, "%7.2f %7.2f %8.3e %8.3e %8.3e \n",
                  ds*(i-1)/1000., ds*(k-1)/1000.,
	              Vxglobal[i][k], Vyglobal[i][k], Vzglobal[i][k] );
            }
          }
        }

        fclose(fp2);

	  }

/* Desallocation */

	  free_dmatrix(Vxglobal,XMIN-delta,XMAX+delta+2,ZMIN-delta, 1);
	  free_dmatrix(Vyglobal,XMIN-delta,XMAX+delta+2,ZMIN-delta, 1);
	  free_dmatrix(Vzglobal,XMIN-delta,XMAX+delta+2,ZMIN-delta, 1);

                /* Writing of surfaceyz */

      Vxglobal = dmatrix(YMIN-delta,YMAX+delta+2,ZMIN-delta, 1);
      Vyglobal = dmatrix(YMIN-delta,YMAX+delta+2,ZMIN-delta, 1);
      Vzglobal = dmatrix(YMIN-delta,YMAX+delta+2,ZMIN-delta, 1);

/* Recherche du processeur avec i0 et coords[1] = 0
   on connait ses coordonnées (icpu,0)
   donc rang   = icpu */

	  icpu = i2icpu_array[i0];
	  imp_tmp =  i2imp_array[i0];

	  if ( my_rank  == icpu ) {
        for ( j = 1; j <= mpmy; j++ ){
		  for ( k = ZMIN-delta; k <= 1; k++ ){
            Vxglobal[YMIN-delta+j-1][k] = vx0[imp_tmp][j][k];
            Vyglobal[YMIN-delta+j-1][k] = vy0[imp_tmp][j][k];
            Vzglobal[YMIN-delta+j-1][k] = vz0[imp_tmp][j][k];
          }
        }
      }

/* Etape1 */

/* GTS */

      for (i1= 1; i1 <=4; i1++ ){

        if (( coords[0] == icpu ) && ( my_rank != icpu ) )  {

	      if (i1 == 1 ){
	        MPI_Isend(coords,2,MPI_INT,icpu,90,comm2d,&sendreq[4]);
		    MPI_Wait (&sendreq[4], &status) ;
	      } /* end of i1 = 1 */

	      if (i1 == 2 ){
            imp=0;
            for ( j = 1; j <= mpmy; j++ ){
		      for ( k = ZMIN-delta; k <= 1; k++ ){
                assert (imp < test_size) ;
                imp ++;
                Vxtemp1[imp] = vx0[imp_tmp][j][k];
              }
            }
            MPI_Isend(Vxtemp1,test_size,MPI_DOUBLE,icpu,80,comm2d,&sendreq[1]);
		    MPI_Wait (&sendreq[1], &status) ;
	      } /* end of i1 = 2 */

          if (i1 == 3 ){
            imp=0;
            for ( j = 1; j <= mpmy; j++ ){
		      for ( k = ZMIN-delta; k <= 1; k++ ){
                assert (imp < test_size) ;
                imp ++;
                Vxtemp1[imp] = vy0[imp_tmp][j][k];
              }
            }
            MPI_Isend(Vxtemp1,test_size,MPI_DOUBLE,icpu,81,comm2d,&sendreq[2]);
		    MPI_Wait (&sendreq[2], &status) ;
	      } /* end of i1 = 3 */

          if (i1 == 4 ){
            imp=0;
            for ( j = 1; j <= mpmy; j++ ){
		      for ( k = ZMIN-delta; k <= 1; k++ ){
                assert (imp < test_size) ;
                imp ++;
                Vxtemp1[imp] = vz0[imp_tmp][j][k];
              }
            }
            MPI_Isend(Vxtemp1,test_size,MPI_DOUBLE,icpu,82,comm2d,&sendreq[3]);
		    MPI_Wait (&sendreq[3], &status) ;
	      } /* end of i1 = 4 */

        } /* end of if icpu && my_rank */

        if ( my_rank == icpu ){

          for ( i2 = 1; i2 < py; i2++ ){

	        if ( i1 == 1 ){
              MPI_Recv(proc_coords,2,MPI_INT,icpu+px*i2,90,comm2d,&status);
	          coords_global[0][i2] = proc_coords[0];
	          coords_global[1][i2] = proc_coords[1];
	        } /* end of i1 = 1 */

	        if ( i1 == 2 ){
	          MPI_Recv(Vxtemp1,test_size, MPI_DOUBLE, icpu+px*i2,80, comm2d,&status);
              total_prec_y = 0;
              for (j=0;j<coords_global[1][i2];j++) {
                total_prec_y = total_prec_y + mpmy_tab[j];
              }
	          imp = 0;
              for ( j = 1; j <= mpmy_tab[coords_global[1][i2]]; j++ ){
		        for ( k = ZMIN-delta; k <= 1; k++ ){
                  assert (imp < test_size) ;
		          assert ( YMIN-delta-1+j+total_prec_y < YMAX+delta+3 );
                  imp ++;
                  Vxglobal[YMIN-delta-1+j+total_prec_y][k] = Vxtemp1[imp];
                }
              }
	        } /* end of i1 = 2 */

            if ( i1 == 3 ){
         	  MPI_Recv(Vxtemp1,test_size, MPI_DOUBLE, icpu+px*i2,81, comm2d,&status);
              total_prec_y = 0;
              for (j=0;j<coords_global[1][i2];j++) {
                total_prec_y = total_prec_y + mpmy_tab[j];
              }
              imp = 0;
              for ( j = 1; j <= mpmy_tab[coords_global[1][i2]]; j++ ){
		        for ( k = ZMIN-delta; k <= 1; k++ ){
                  assert (imp < test_size) ;
                  assert ( YMIN-delta-1+j+total_prec_y < YMAX+delta+3 );
                  imp ++;
	              Vyglobal[YMIN-delta-1+j+total_prec_y][k] = Vxtemp1[imp];
                }
              }
	        } /* end of i1 = 3 */

            if ( i1 == 4 ){
 	          MPI_Recv(Vxtemp1,test_size, MPI_DOUBLE, icpu+px*i2,82, comm2d,&status);
              total_prec_y = 0;
              for (j=0;j<coords_global[1][i2];j++) {
                total_prec_y = total_prec_y + mpmy_tab[j];
              }
              imp = 0;
              for ( j = 1; j <= mpmy_tab[coords_global[1][i2]]; j++ ){
		        for ( k = ZMIN-delta; k <= 1; k++ ){
                  assert (imp < test_size) ;
                  assert ( YMIN-delta-1+j+total_prec_y < YMAX+delta+3 );
                  imp ++;
	              Vzglobal[YMIN-delta-1+j+total_prec_y][k] = Vxtemp1[imp];
                }
              }
	        } /* end of i1 = 4 */
	      } /* end of i2 */
        } /* end of if my_rank */
	  } /* end of i1 */

                /* Ecriture */

      if ( my_rank == icpu ) {

	    fp3 = fopen(flname3, "w");
        for ( j = YMIN-delta+1; j <= YMAX+delta+1; j++ ){
          for ( k = ZMIN-delta+1; k <= 1; k++ ){
            if( ( ((int)(ds*(j-1)))%200) == 0 && ( ((int)(ds*(k-1)))%200) == 0 ){
              fprintf(fp3, "%7.2f %7.2f %8.3e %8.3e %8.3e \n",
                  ds*(j-1)/1000., ds*(k-1)/1000.,
	              Vxglobal[j][k], Vyglobal[j][k], Vzglobal[j][k] );
            }
          }
        }

        fclose(fp3);

      }

/* Desallocation */

	  free_dmatrix(Vxglobal,YMIN-delta,YMAX+delta+2,ZMIN-delta, 1);
	  free_dmatrix(Vyglobal,YMIN-delta,YMAX+delta+2,ZMIN-delta, 1);
	  free_dmatrix(Vzglobal,YMIN-delta,YMAX+delta+2,ZMIN-delta, 1);
#endif
    } /* end of if ( l % nn == 0 ) */

                /* Output : files which contain the seismograms at the stations */

/* fichier obs//ir+1.dat */

    if ( (l % 100) == 0 || l == TMAX){
      for ( ir = 0; ir < IOBS; ir++ ){

        icpu = i2icpu_array[ixobs[ir]];
        jcpu = j2jcpu_array[iyobs[ir]];

	    if ( my_rank == 0 ){

          strcpy(flname4, outdir);
          strcat(flname4, char4);
          sprintf(number, "%d", ir+1);
          strcat(flname4, number);
          strcat(flname4, ".dat");

          if ( l == TMAX) printf("%d %s\n", ir+1, flname4);
          fp4 = fopen(flname4, "w");
          fprintf(fp4, "%d %f %f %f\n", nobs[ir], xobs[ir], yobs[ir], zobs[ir]);
          fprintf(fp4, "%d %f\n", l, dt);
          for (l1 = 0; l1 < l; l1++)
            fprintf (fp4, "%e %e %e %e %e %e %e %e %e\n",
	            seis_output[l1][ir][1], seis_output[l1][ir][2], seis_output[l1][ir][3],
	            seis_output[l1][ir][4], seis_output[l1][ir][5], seis_output[l1][ir][6],
	            seis_output[l1][ir][7], seis_output[l1][ir][8], seis_output[l1][ir][9]);

            fclose(fp4);

        } /* end of my_rank = 0 */
	  } /* end of ir */
	} /* end of if ( l % nn == 0 ) */

    if ( (l % 100) == 0 && my_rank == 0 ){
	  printf ("\nEnd time %d\n", l);
    }

  } /* end of time loop */
              /* Desallocation */

  free_ivector (ixhypo, 0, ISRC-1);
  free_ivector (iyhypo, 0, ISRC-1);
  free_ivector (izhypo, 0, ISRC-1);
  free_ivector (insrc, 0, ISRC-1);
  free_dvector (xhypo, 0, ISRC-1);
  free_dvector (yhypo, 0, ISRC-1);
  free_dvector (zhypo, 0, ISRC-1);
  free_dvector (strike, 0, ISRC-1);
  free_dvector (dip, 0, ISRC-1);
  free_dvector (rake, 0, ISRC-1);
  free_dvector (slip, 0, ISRC-1);
  free_dvector (xweight, 0, ISRC-1);
  free_dvector (yweight, 0, ISRC-1);
  free_dvector (zweight,0, ISRC-1);

  free_dmatrix (vel, 0, ISRC-1, 0, IDUR-1);

  free_ivector (nobs, 0, IOBS-1);
  free_dvector (xobs, 0, IOBS-1);
  free_dvector (yobs, 0, IOBS-1);
  free_dvector (zobs, 0, IOBS-1);
  free_ivector (ixobs, 0, IOBS-1);
  free_ivector (iyobs, 0, IOBS-1);
  free_ivector (izobs, 0, IOBS-1);
  free_dvector (xobswt, 0, IOBS-1);
  free_dvector (yobswt, 0, IOBS-1);
  free_dvector (zobswt, 0, IOBS-1);
  free_ivector (ista, 0, IOBS-1);
  free_ivector (mpmx_tab, 0, px-1);
  free_ivector (mpmy_tab, 0, py-1);

  free_dmatrix (seisx, 0, IOBS-1, 0, TMAX-1);
  free_dmatrix (seisy, 0, IOBS-1, 0, TMAX-1);
  free_dmatrix (seisz, 0, IOBS-1, 0, TMAX-1);
  free_dmatrix (seisxx, 0, IOBS-1, 0, TMAX-1);
  free_dmatrix (seisyy, 0, IOBS-1, 0, TMAX-1);
  free_dmatrix (seiszz, 0, IOBS-1, 0, TMAX-1);
  free_dmatrix (seisxy, 0, IOBS-1, 0, TMAX-1);
  free_dmatrix (seisxz, 0, IOBS-1, 0, TMAX-1);
  free_dmatrix (seisyz, 0, IOBS-1, 0, TMAX-1);

  free_dvector (Vxtemp1, 1, test_size);

  free_ivector (i2imp_array, XMIN-delta, XMAX+2*delta+2);
  free_ivector (j2jmp_array, YMIN-delta, YMAX+2*delta+2);
  free_ivector (imp2i_array, -1, mpmx+2);
  free_ivector (jmp2j_array, -1, mpmy+2);
  free_ivector (i2icpu_array, XMIN-delta, XMAX+2*delta+2);
  free_ivector (j2jcpu_array, YMIN-delta, YMAX+2*delta+2);

  free_d3tensor (vx0, -1, mpmx+2, -1,mpmy+2, ZMIN-delta, 1);
  free_d3tensor (vy0, -1, mpmx+2, -1,mpmy+2, ZMIN-delta, 1);
  free_d3tensor (vz0, -1, mpmx+2, -1,mpmy+2, ZMIN-delta, 1);

  free_dvector (phivxx, 1, npmlv);
  free_dvector (phivxy, 1, npmlv);
  free_dvector (phivxz, 1, npmlv);
  free_dvector (phivyx, 1, npmlv);
  free_dvector (phivyy, 1, npmlv);
  free_dvector (phivyz, 1, npmlv);
  free_dvector (phivzx, 1, npmlv);
  free_dvector (phivzy, 1, npmlv);
  free_dvector (phivzz, 1, npmlv);

  free_d3tensor (fx, 1, mpmx, 1,mpmy, ZMIN-delta, 1);
  free_d3tensor (fy, 1, mpmx, 1,mpmy, ZMIN-delta, 1);
  free_d3tensor (fz, 1, mpmx, 1,mpmy, ZMIN-delta, 1);

  free_d3tensor (txx0, -1,mpmx+2,-1,mpmy+2,ZMIN-delta,1);
  free_d3tensor (tyy0, -1,mpmx+2,-1,mpmy+2,ZMIN-delta,1);
  free_d3tensor (tzz0, -1,mpmx+2,-1,mpmy+2,ZMIN-delta,1);
  free_d3tensor (txy0, -1,mpmx+2,-1,mpmy+2,ZMIN-delta,1);
  free_d3tensor (txz0, -1,mpmx+2,-1,mpmy+2,ZMIN-delta,1);
  free_d3tensor (tyz0, -1,mpmx+2,-1,mpmy+2,ZMIN-delta,1);

  free_dvector (phitxxx, 1, npmlt);
  free_dvector (phitxyy, 1, npmlt);
  free_dvector (phitxzz, 1, npmlt);
  free_dvector (phitxyx, 1, npmlt);
  free_dvector (phityyy, 1, npmlt);
  free_dvector (phityzz, 1, npmlt);
  free_dvector (phitxzx, 1, npmlt);
  ////////////////////////
  free_dvector (phityzy, 1, npmlt);
  free_dvector (phitzzz, 1, npmlt);

  free_d3tensor (rho, -1,mpmx+2,-1,mpmy+2,ZMIN-delta,2);
  free_d3tensor (mu, -1,mpmx+2,-1,mpmy+2,ZMIN-delta,2);
  free_d3tensor (lam, -1,mpmx+2,-1,mpmy+2,ZMIN-delta,2);
  free_d3tensor (vp, -1,mpmx+2,-1,mpmy+2,ZMIN-delta,2);
  free_d3tensor (vs, -1,mpmx+2,-1,mpmy+2,ZMIN-delta,2);

  free_dvector (dumpx, 1,mpmx);
  free_dvector (kappax, 1,mpmx);
  free_dvector (alphax, 1,mpmx);
  free_dvector (dumpx2, 1,mpmx);
  free_dvector (kappax2, 1,mpmx);
  free_dvector (alphax2, 1,mpmx);

  free_dvector (dumpy, 1,mpmy);
  free_dvector (kappay, 1,mpmy);
  free_dvector (alphay, 1,mpmy);
  free_dvector (dumpy2, 1,mpmy);
  free_dvector (kappay2, 1,mpmy);
  free_dvector (alphay2, 1,mpmy);

  free_dvector (dumpz, ZMIN-delta,1);
  free_dvector (kappaz, ZMIN-delta,1);
  free_dvector (alphaz, ZMIN-delta,1);
  free_dvector (dumpz2, ZMIN-delta,1);
  free_dvector (kappaz2, ZMIN-delta,1);
  free_dvector (alphaz2, ZMIN-delta,1);

  free_imatrix (mapping_seis, 0, IOBS-1, 1, 9);
  free_d3tensor (seis_output, 0, TMAX-1, 0, IOBS-1, 1, 9);



  #if (FLOPS)
    if((retval=PAPI_flops( &real_time, &proc_time, &flpops, &mflops))<PAPI_OK){
      printf("retval: %d\n", retval);
      exit(1);
    }
    printf("Mflops %f \n", mflops);
  #endif

  #if(MISS)
    if ( retval=PAPI_stop(EventSet,values) != PAPI_OK )
    printf("ERROR stop \n");
    perc = (float)100.0*values[1]/values[0];
    printf("Cache Miss %f %d \n", perc,my_rank);
    printf("Cycle %lld %d \n", values[2],my_rank);
    printf("L3 MISS  %lld %d \n", values[1],my_rank);
    printf("L3 acces  %lld %d \n", values[0],my_rank);
  #endif

  timing4 = my_second();
  timing_total = (timing4-timing3);

  MPI_Reduce(&timing_bc1,&timing_bc1_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&timing_bc1,&timing_bc1_min,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&timing_bc1,&timing_sum1,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  MPI_Reduce(&timing_bc2,&timing_sum2,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  MPI_Reduce(&timing_bc2,&timing_bc2_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&timing_bc2,&timing_bc2_min,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&timing_total,&timing4,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

  timing_bc_total = timing_bc1 + timing_bc2;
  timing_comm_total = timing_comm1 + timing_comm2;

/* On prend le proc avec le plus gros ecart au niveau de bc */

  MPI_Reduce(&timing_bc_total,&timing_bc_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&timing_bc_total,&timing_bc_min,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&timing_comm_total,&timing_comm_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&timing_comm_total,&timing_comm_min,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);

  if (my_rank == 0 ) {

	printf("%d %d %d \n",np,px,py);
	printf("Timing total %f \n",timing_total);
    printf("Timing compute max  %f \n",timing_bc_max);
    printf("Timing compute min  %f \n",timing_bc_min);
    printf("Timing comm max - attente + comm reelle  %f \n",timing_comm_max);
    printf("Timing comm min - comm relle  %f \n",timing_comm_min);
    printf("Part communication: %f \n",100*(timing_comm_min)/timing_total);
    printf("Part compute: %f \n",100*(timing_bc_min)/timing_total);
    printf("Ecart BC vs timing total %f \n",100*((timing_bc_max-timing_bc_min)/timing_total));
    printf("Ecart BC %f \n",100*((timing_bc_max-timing_bc_min)/timing_bc_max));
	printf("Ecart Min/Max BC1 %f \n",100*((timing_bc1_max-timing_bc1_min)/timing_bc1_max));
    printf("Ecart Min/Max BC2 %f \n",100*((timing_bc2_max-timing_bc2_min)/timing_bc2_max));
    printf("timing max - BC1  %f \n ",timing_bc1_max);
    printf("timing min - BC1  %f \n ",timing_bc1_min);
    printf("timing max - BC2  %f \n ",timing_bc2_max);
    printf("timing min - BC2  %f \n ",timing_bc2_min);

    fp5 = fopen(BENCH, "w");

    fprintf(fp5,"%d %d %d \n",np,px,py);
    fprintf(fp5,"%f \n",timing_total);
    fprintf(fp5,"%f \n",timing_bc_max);
    fprintf(fp5,"%f \n",timing_bc_min);
    fprintf(fp5,"%f \n",timing_comm_max);
    fprintf(fp5,"%f \n",timing_comm_min);
    fprintf(fp5,"%f \n",100*(timing_comm_min)/timing_total);
    fprintf(fp5,"%f \n",100*(timing_bc_min)/timing_total);
    fprintf(fp5,"%f \n",100*((timing_bc_max-timing_bc_min)/timing_total));
    fprintf(fp5,"%f \n",100*((timing_bc_max-timing_bc_min)/timing_bc_max));
	fprintf(fp5,"%f \n",100*((timing_bc1_max-timing_bc1_min)/timing_bc1_max));
    fprintf(fp5,"%f \n",100*((timing_bc2_max-timing_bc2_min)/timing_bc2_max));

	close(fp5);

/* Fin cout max comm */

  } /* end my_rank */

  MPI_Barrier(MPI_COMM_WORLD);
  printf("------------- FIN -------------- %d \n",my_rank);
  MPI_Barrier(MPI_COMM_WORLD);

  #if (TIMING_BARRIER)
    MPI_Barrier(MPI_COMM_WORLD);
    timing4 = my_second();
    timing_total = (timing4-timing3);
  #endif

  #if (TIMING)
    timing4 = my_second();
    timing_total = (timing4-timing3);
  #endif

  MPI_Finalize();
  return 0;

} /* end of program */

				/* Functions */

int double2int(double x)
{
  if (x >= 0){
	return (int) x;
  } else if ( x == (int) x ){
	return x;
  } else {
	return (int) x - 1;
  }
}

double radxx(double strike, double dip, double rake)
{
  return  cos(rake)*sin(dip)*sin(2.*strike)
          - sin(rake)*sin(2.*dip)*cos(strike)*cos(strike) ;
}
double radyy(double strike, double dip, double rake)
{
  return  - ( cos(rake)*sin(dip)*sin(2.*strike)
          + sin(rake)*sin(2.*dip)*sin(strike)*sin(strike) );
}
double radzz(double strike, double dip, double rake)
{
  return sin(rake)*sin(2.*dip);
}
double radxy(double strike, double dip, double rake)
{
  return cos(rake)*sin(dip)*cos(2.*strike)
         + 0.5*sin(rake)*sin(2.*dip)*sin(2.*strike);
}
double radyz(double strike, double dip, double rake)
{
  return cos(rake)*cos(dip)*cos(strike) + sin(rake)*cos(2.*dip)*sin(strike);
}
double radxz(double strike, double dip, double rake)
{
  return cos(rake)*cos(dip)*sin(strike) - sin(rake)*cos(2.*dip)*cos(strike);
}

double staggardv4 (double b, double kappax, double kappay, double kappaz, double dt, double dx,
	double x1, double x2, double x3, double x4,
	double y1, double y2, double y3, double y4,
	double z1, double z2, double z3, double z4 )
{
  return (9.*b*dt/8.)*( (x2 - x1)/kappax + (y2 - y1)/kappay + (z2 - z1)/kappaz )/dx
         - (b*dt/24.)*( (x4 - x3)/kappax + (y4 - y3)/kappay + (z4 - z3)/kappaz )/dx;
}

double staggardv2 (double b, double kappax, double kappay, double kappaz, double dt, double dx,
	double x1, double x2,
	double y1, double y2,
	double z1, double z2 )
{
  return b*dt*( (x2 - x1)/kappax + (y2 - y1)/kappay + (z2 - z1)/kappaz )/dx;
}

double staggards4 (double lam, double mu, double kappax, double kappay, double kappaz, double dt, double dx,
	double x1, double x2, double x3, double x4,
	double y1, double y2, double y3, double y4,
	double z1, double z2, double z3, double z4 )
{
  return (9.*dt/8.)*( (lam+2.*mu)*(x2 - x1)/kappax + lam*(y2 - y1)/kappay + lam*(z2 - z1)/kappaz )/dx
         - (dt/24.)*( (lam+2.*mu)*(x4 - x3)/kappax + lam*(y4 - y3)/kappay + lam*(z4 - z3)/kappaz )/dx;
}

double staggards2 (double lam, double mu, double kappax, double kappay, double kappaz, double dt, double dx,
	double x1, double x2,
	double y1, double y2,
	double z1, double z2 )
{
  return dt*( (lam+2.*mu)*(x2 - x1)/kappax + lam*(y2 - y1)/kappay + lam*(z2 - z1)/kappaz )/dx;
}

double staggardt4 (double mu, double kappax, double kappay, double dt, double dx,
	double x1, double x2, double x3, double x4,
	double y1, double y2, double y3, double y4 )
{
  return (9.*dt*mu/8.)*( (x2 - x1)/kappax + (y2 - y1)/kappay )/dx
         - (dt*mu/24.)*( (x4 - x3)/kappax + (y4 - y3)/kappay )/dx;
}

double staggardt2 (double mu, double kappax, double kappay, double dt, double dx,
	double x1, double x2,
	double y1, double y2 )
{
  return dt*mu*( (x2 - x1)/kappax + (y2 - y1)/kappay )/dx;
}

double CPML4 (double vp, double dump, double alpha, double kappa, double phidum, double dx, double dt,
    double x1, double x2, double x3, double x4 )
{
  double a, b;

 b = exp ( - ( vp*dump / kappa + alpha ) * dt );
  a = 0.0;
  if ( abs ( vp*dump ) > 0.000001 ) a = vp*dump * ( b - 1.0 ) / ( kappa * ( vp*dump + kappa * alpha ) );

  return b * phidum + a * ( (9./8.)*( x2 - x1 )/dx - (1./24.)*( x4 - x3 )/dx );
}

double CPML2 (double vp, double dump, double alpha, double kappa, double phidum, double dx, double dt,
    double x1, double x2 )
{
  double a, b;

  b = exp ( - ( vp*dump / kappa + alpha ) * dt );
  a = 0.0;
  if ( abs ( vp*dump ) > 0.000001 ) a = vp*dump * ( b - 1.0 ) / ( kappa * ( vp*dump + kappa * alpha ) );

  return b * phidum + a * ( x2 - x1 ) * (1./dx);
}

double my_second()
{
  struct timeval tp;
  struct timezone tzp;
  int i;

  i = gettimeofday(&tp,&tzp);
  return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}
#if  (VTK)
void force_big_endian(unsigned char *bytes)
{
    static int doneTest = 0;
    static int shouldSwap = 0;

    /* declare an int*/
    int tmp1 = 1;

    /* the char pointer points to the tmp1 variable
    // so tmp2+1, +2, +3 points to the bytes of tmp1*/
    unsigned char *tmp2 = (unsigned char *) &tmp1;

    /* look if the endianness
    // of the machine has already been
    // tested*/
    if (!doneTest)
    {
        if (*tmp2 != 0)
            shouldSwap = 1;
        doneTest = 1;
    }

    if (shouldSwap)
    {
        unsigned char tmp;
        tmp = bytes[0];
        bytes[0] = bytes[3];
        bytes[3] = tmp;
        tmp = bytes[1];
        bytes[1] = bytes[2];
        bytes[2] = tmp;
   }
}
void write_float(FILE *fp, float val)
{
    force_big_endian((unsigned char *) &val);
    fwrite(&val, sizeof(float), 1, fp);
}
#endif

