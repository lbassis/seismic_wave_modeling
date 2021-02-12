/* ONDES3D : FDM code for seismic wave propagation */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>
#include <assert.h>
#ifdef LUCAS_INSTRUMENTATION
#include <scorep/SCOREP_User.h>
#endif


#ifdef MPI
#include <mpi.h>
#endif
#ifdef OMP
#include <omp.h>
#endif

#ifdef OUTADIOS
#include <adios.h>
#include <adios_types.h>
#endif




#include "struct.h"
#include "options.h"

#include "nrutil.h"
#include "inlineFunctions.h"

#include "computeVeloAndSource.h"
#include "computeStress.h"
#include "computeIntermediates.h"
#include "IO.h"
#include "alloAndInit.h"
#include "main.h"


#if defined (PROFILE1) || defined (PROFILE2)
#include <tau.h>
#endif

#if defined (MISS) || defined (FLOPS)
#include "papi.h"
#endif


#define HOGE "hoge"

#ifdef TOPO_MPI
static const char TOPOFILE[50] = "topologie.in";
#else
static const char TOPOFILE[50] = QUOTEME(TOPOLOGIE);
#endif

static const char BENCHFILE[50] = "bench.out";
static const char ADIOSFILE[50] = "adios.out";


/* OpenMP */
// Position provisoire - a integrer a la structure PRM
int numThreads_Level_1;


/* Beginning of main */
int main(int argc, char **argv)
{				/* Model parameters */
    struct PARAMETERS PRM = { 0 };
    /* Source : values */
    struct SOURCE SRC = { 0 };

    /* Velocity and stress */
    struct VELOCITY v0 = { 0 };
    struct STRESS t0 = { 0 };
    /* Medium */
    struct MEDIUM MDM = { 0 };

    /* Variables for the PML/CPML */
    struct ABSORBING_BOUNDARY_CONDITION ABC = { 0 };

    /* ANELASTICITIES */
    struct ANELASTICITY ANL = { 0 };

    /* OUTPUTS */
    struct OUTPUTS OUT = { 0 };
    /* File names */
    char flname[50], number[5], flname1[80], flname2[80], flname3[80],
	flname4[80], flname5[80], flname6[80], buf[256], char1[30] =
	"surfacexy", char2[30] = "surfacexz", char3[30] =
	"surfaceyz", char4[30] = "obs";

    /* mapping */
    double DS, DT;		/* parameters PRM */
    int TMAX;
    int XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX;
    int ZMAX0, DELTA;
    int MPMX, MPMY;
    int ME;

    /* Other variables */
    int fStatus = EXIT_FAILURE;	/* status of functions */
    int i, j, k, k2, l, l1, ind, ind1, indx, indy, indz;
    double time;
    FILE *fp1, *fp2, *fp3, *fp5;
    double pgv_tmp;

    /* MPI variables */
#ifdef MPI
    char pname[MPI_MAX_PROCESSOR_NAME];
    MPI_Comm comm2d, rc;

#ifdef OUTADIOS

    FILE *fp_adios;
    MPI_Comm a_comm2d;
    int a_me, a_color, a_key, a_np;
    char adios_filename1[256];
    char adios_filename2[256];
    char adios_filename3[256];


    char recept_group[256];
    char varname[20];
    char varsize[20];
    char transp_method_r[20];
    char transp_method[20];
    char aggr_param_r[30];
    char aggr_param[30];


//    character(len=256)      :: recept_group
//    character(len=10)       :: varname
//    character(len=10)       :: varsize
//    character(len=10)       :: transp_test = "MPI"


//    integer                 :: adios_err
//    integer*8               :: m_adios_group,varid

    int a_receiver_cpt;
    int adios_cpt;
    uint64_t adios_groupsize, adios_totalsize;
    uint64_t machine_size_group, human_size_group;
    int64_t adios_handle1, adios_handle2, adios_handle3;
    int64_t varid1, varid2, varid3;

    int adios_buffer, a_buffer_cpt;
    double **a_buffer_vel;
    double *a_temp;
    int a_buffer_rate, a_cpt;
    int *a_indirect;

#endif

#endif



    /* Variables for the communication between the CPUs */
    struct COMM_DIRECTION NORTH = { 0 }, SOUTH = {
	0
    }, EAST = {
	0
    }, WEST = {
	0
    };
    /* 1 - direction : N->S
     * 2 - direction : S->N
     * 3 - direction : E->W
     * 4 - direction : W->E
     */

    int nnorth, nsouth, neast, nwest;
#ifdef MPI
    MPI_Status status[5];
    MPI_Request req[5];

#if (COMM==1)
    /* WARNING :
     * Originally, optimisations were made
     * by assuming that matrices with 3 indexes :
     * [x][y][z] are contiguous in one direction for some planes.
     * But this depends on "d3tensor" in nrutil.c and of the compiler.
     * So I think, it is kind of machine dependent optimisation but I maybe wrong.
     * That's why, I decided to drop this optimisation for the moment.
     * Please feel free to implement/try it.
     */
    /* Notations :
     * S at the end indicates Send
     * R at the end indicates Receive
     *
     * 1 - direction : N->S
     * 2 - direction : S->N
     * 3 - direction : E->W
     * 4 - direction : W->E
     */
    MPI_Request reqV0R[5];	/* velocity */
    MPI_Request reqT0R[5];	/* stress */
    MPI_Request reqKsiR[5];	/* ksil
				   (kristek & moczo anelasticity method) */

    MPI_Request reqV0S[5];
    MPI_Request reqT0S[5];
    MPI_Request reqKsiS[5];

    MPI_Status statV0S[5];
    MPI_Status statT0S[5];
    MPI_Status statKsiS[5];

    MPI_Status statV0R[5];
    MPI_Status statT0R[5];
    MPI_Status statKsiR[5];
#endif				/* PERSISTANT */

#endif /*MPI*/
    int np, resultlength, imp, jmp, imp_tmp, jmp_tmp, icpu, jcpu, imode;
    int mpmx_begin, mpmx_end;
    int mpmy_begin, mpmy_end;

    /* for others communications (seismograms & surface"ij" ) */


#ifdef MPI
    MPI_Request sendreq[20];
#endif

    int proc_coords[2], coords_global[2][1024];
    int i1, i2;
    int ir;
    double w1, w2, w3, w12, w22, w32;

    /* PAPI */

#if defined (MISS) || (FLOPS)
    float real_time, proc_time, mflops;
    long_long flpops;
    double perc;
    float ireal_time, iproc_time, imflops;
    long_long iflpops;
    int retval;
    int EventSet = PAPI_NULL;
    int EventCode;
    long_long values[3];
#endif

    /* Timing */

    double timing1, timing2, timing3, timing4, timing_comm1, timing_comm2;
    double timing_bc1, timing_bc2, timing_total, timing_sum1, timing_sum2;
    double timing_bc1_max, timing_bc1_min, timing_bc2_min, timing_bc2_max;
    double timing_pml, timing_DS4, timing_dt4;

    double timing_bc_max, timing_bc_min, timing_comm_min, timing_comm_max,
	timing_bc_total, timing_comm_total;




    /* ========================== */
    /* Beginning of the PROGRAM   */
    /* ========================== */

    /* Initialization for MPI */
    PRM.me = 0;
    np = 1;

#ifdef MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    MPI_Comm_dup(MPI_COMM_WORLD, &comm2d);
    MPI_Comm_rank(comm2d, &(ME));
    PRM.me = ME;


#endif



#ifdef OMP

    if (argc < 2) {
	printf("ondes3d numThreads_OMP\n");
	exit(-1);
    }
    numThreads_Level_1 = atoi(argv[1]);
    printf("Number of threads of OMP is %d Threads\n", numThreads_Level_1);
    omp_set_num_threads(numThreads_Level_1);

#endif





    PRM.px = 1;
    PRM.py = 1;
#ifdef MPI
    VerifFunction(ReadTopo(&PRM, TOPOFILE, np), "read topology file", PRM);
#endif

    PRM.coords[1] = PRM.me / PRM.px;
    PRM.coords[0] = PRM.me - PRM.coords[1] * PRM.px;
    printf("%d,%d,%d\n", PRM.me, PRM.coords[0], PRM.coords[1]);
    MPI_Finalize();
    return 0;
    nnorth = PRM.me + PRM.px;
    nsouth = PRM.me - PRM.px;
    neast = PRM.me + 1;
    nwest = PRM.me - 1;
    if (PRM.coords[1] == 0) {
	nsouth = -1;
    }
    if (PRM.coords[1] == PRM.py - 1) {
	nnorth = -1;
    }
    if (PRM.coords[0] == 0) {
	nwest = -1;
    }
    if (PRM.coords[0] == PRM.px - 1) {
	neast = -1;
    }
#ifdef MPI
    MPI_Get_processor_name(pname, &resultlength);
    MPI_Barrier(comm2d);
#endif


#if ( VERBOSE > 2 )
    if (PRM.me == 0) {
	printf("Topology\n");
    }
    printf("me:%i - est:%i - ouest:%i - nord:%i - sud:%i \n", PRM.me,
	   neast, nwest, nnorth, nsouth);
    printf("me:%i - coords(0):%i - coords(1):%i \n", PRM.me, PRM.coords[0],
	   PRM.coords[1]);
#endif



    /* =========================== */
    /* Beginning of Initializations */
    /* =========================== */

    timing_comm1 = 0.;
    timing_comm2 = 0.;
    timing_bc1 = 0.;
    timing_bc2 = 0.;
    timing_total = 0.;
    timing_pml = 0.;
    timing_dt4 = 0.;
    timing_DS4 = 0.;

    /* TAU */

#ifdef PROFILE1
    TAU_PROFILE_TIMER(pml_sig, "pml_sig", "void (int,char **)", TAU_USER);
    TAU_PROFILE_TIMER(free_surface_sig, "free_surface_sig",
		      "void (int,char **)", TAU_USER);
    TAU_PROFILE_TIMER(interior_sig, "interior_sig", "void (int,char **)",
		      TAU_USER);
    TAU_PROFILE_TIMER(exchange_sig, "exchange_sig", "void (int,char **)",
		      TAU_USER);

    TAU_PROFILE_TIMER(pml_vit, "pml_vit", "void (int,char **)", TAU_USER);
    TAU_PROFILE_TIMER(free_surface_vit, "free_surface_vit",
		      "void (int,char **)", TAU_USER);
    TAU_PROFILE_TIMER(interior_vit, "interior_vit", "void (int,char **)",
		      TAU_USER);
    TAU_PROFILE_TIMER(exchange_vit, "exchange_vit", "void (int,char **)",
		      TAU_USER);

    TAU_PROFILE_SET_NODE(PRM.me)
#endif
#ifdef PROFILE2
	TAU_PROFILE_TIMER(compute_sig, "compute_sig", "void (int,char **)",
			  TAU_USER);
    TAU_PROFILE_TIMER(exchange_sig, "exchange_sig", "void (int,char **)",
		      TAU_USER);

    TAU_PROFILE_TIMER(compute_vit, "compute_vit", "void (int,char **)",
		      TAU_USER);
    TAU_PROFILE_TIMER(exchange_vit, "exchange_vit", "void (int,char **)",
		      TAU_USER);

    TAU_PROFILE_SET_NODE(PRM.me)
#endif
#ifdef MISS
	/* PAPI */
	if ((retval =
	     PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
	printf("ERROR Init \n");
    if ((retval = PAPI_create_eventset(&EventSet)) != PAPI_OK)
	printf("ERROR create \n");
    if ((retval = PAPI_add_event(EventSet, PAPI_L3_DCA)) != PAPI_OK)
	printf("ERROR add \n");
    if ((retval = PAPI_add_event(EventSet, PAPI_L3_DCH)) != PAPI_OK)
	printf("ERROR add \n");
    if ((retval = PAPI_add_event(EventSet, PAPI_TOT_CYC)) != PAPI_OK)
	printf("ERROR add \n");

#endif

#ifdef MPI
    MPI_Barrier(comm2d);
#endif



    /* Read files */
    fStatus = ReadPrmFile(&PRM, &MDM, &ABC, &ANL, &OUT, PRMFILE);
    VerifFunction(fStatus, "read parameter file ", PRM);

    fStatus = ReadSrc(&SRC, PRM);
    VerifFunction(fStatus, "read sources file ", PRM);



    /* MPI */
    VerifFunction(InitPartDomain(&PRM, &OUT), "split domain MPI", PRM);


#ifdef MPI
    fStatus = InitializeCOMM(&NORTH, &SOUTH, &EAST, &WEST,	/* outputs */
			     nnorth, nsouth, neast, nwest,	/* inputs */
			     PRM);
    VerifFunction(fStatus, "initialize COMMUNICATIONS ", PRM);
#endif


    /* Read others files ( dependent to MPI partionning ) */
    if (model == GEOLOGICAL)
	VerifFunction(ReadGeoFile(&MDM, PRM), "read geological file", PRM);


    fStatus = ReadStation(&OUT, PRM, MDM);
    VerifFunction(fStatus, "read station file ", PRM);


    /* allocate fields */
    fStatus = AllocateFields(&v0, &t0, &ANL, &ABC, &MDM, &SRC,	/* outputs */
			     PRM	/* inputs */
	);
    VerifFunction(fStatus, "allocate Fields ", PRM);

  /** initialize fields **/
    /* init layers */
    if (model == LAYER) {
	VerifFunction(InitLayerModel(&MDM, &ANL, PRM),
		      "initialize layer model", PRM);
    }

    /* inside the domain */
    if (ANLmethod == DAYandBRADLEY) {
	fStatus = InitializeDayBradley(&MDM, &ANL, PRM);
	VerifFunction(fStatus, "initilize Day and BRADLEY", PRM);
    } else if (ANLmethod == KRISTEKandMOCZO) {
	fStatus = InitializeKManelas(&ANL, &MDM, PRM.dt);
	VerifFunction(fStatus, "initilize KRISTEK and MOCZO anelasticity",
		      PRM);
    }

    /* in the absorbing layers */
    fStatus = InitializeABC(&ABC, &MDM, &ANL, PRM);
    VerifFunction(fStatus, "initilize absorbing boundaries ", PRM);

    if (model == GEOLOGICAL) {
#ifdef OUT_HOGE
	/* Checking the geological model : we write in a binary file */
	VerifFunction(OutGeol(MDM, OUT, PRM, HOGE),
		      "check geological model", PRM);
#endif
	/* Computing the height of the stations */
	VerifFunction(InitializeGeol(&OUT, MDM, PRM),
		      "initialize height station", PRM);
    }



    /* Allocation output */
    VerifFunction(InitializeOutputs(STATION_STEP, &OUT, PRM),
		  " MapSeismograms", PRM);






/////////////:  Begin ADIOS ///////////////////////////////
#ifdef ADIOSOBS
//      MPI Communicator
    a_color = MPI_UNDEFINED;
    for (ir = 0; ir < OUT.iObs; ir++) {
	if (OUT.ista[ir] == 1) {
	    if (PRM.me == OUT.mapping_seis[ir][1]) {
		a_color = 13;
		a_key = PRM.me;
	    }
	}
    }

    MPI_Comm_split(comm2d, a_color, a_key, &a_comm2d);
    a_me = -13;

    if (a_comm2d != MPI_COMM_NULL) {
	MPI_Comm_size(a_comm2d, &a_np);
//              printf("comm2D (size=%d)  ;  a_comm2D(size=%d)\n",np,a_np);

	MPI_Comm_rank(a_comm2d, &(a_me));
//              printf("comm2D (rank=%d)  ;  a_comm2D(rank=%d)\n",PRM.me,a_me);
    }


    strcpy(adios_filename1, "O3D_vx.bp");


//      Init
    if (a_comm2d != MPI_COMM_NULL)
	adios_init_noxml(a_comm2d);
//      adios_set_max_buffer_size(256);




//      Lecture parametres ADIOS
    fp_adios = fopen("adios_option.in", "r");
    fscanf(fp_adios, "%d", &a_buffer_rate);
    fscanf(fp_adios, "%s", transp_method_r);
    fscanf(fp_adios, "%s", aggr_param_r);
    strcpy(transp_method, transp_method_r);
    strcpy(aggr_param, aggr_param_r);

    fclose(fp_adios);
    if (PRM.me == 0)
	printf("ADIOS write rate = %d \n", a_buffer_rate);
    if (PRM.me == 0)
	printf("Adios Transp Method = %s %s \n", transp_method,
	       transp_method_r);
    if (PRM.me == 0)
	printf("Adios MPI_AMR = %s %s \n", aggr_param, aggr_param_r);
    if (PRM.me == 0)
	printf("Total number of receiver  = %d \n", OUT.iObs);



//      Declare group
    if (a_comm2d != MPI_COMM_NULL) {
	adios_declare_group(&adios_handle1, "receiver", "", adios_flag_no);
	adios_select_method(adios_handle1, transp_method, aggr_param, "");
//               adios_select_method (adios_handle1 ,transp_method,"","");
    }



    //      Indirection et nombre de recepteurs
    a_buffer_cpt = 0;
    a_receiver_cpt = 0;

    for (ir = 0; ir < OUT.iObs; ir++) {
	if (OUT.ista[ir] == 1) {
	    if (PRM.me == OUT.mapping_seis[ir][1]) {
		a_receiver_cpt++;
	    }
	}
    }


//      Alloc a_indirect
    a_indirect = ivector(0, a_receiver_cpt - 1);

//      Construction indirection
    a_cpt = 0;
    for (ir = 0; ir < OUT.iObs; ir++) {
	if (OUT.ista[ir] == 1) {
	    if (PRM.me == OUT.mapping_seis[ir][1]) {
		a_indirect[a_cpt] = ir;
		a_cpt++;
	    }
	}
    }







#if defined (ADIOSLOOP) || defined (ADIOSNOLOOP)
    sprintf(varsize, "%d", a_buffer_rate);
    a_buffer_vel = mydmatrix0(0, a_receiver_cpt, 0, a_buffer_rate - 1);
    a_temp = mydvector0(0, a_buffer_rate - 1);

//      Adios Variables
    if (a_comm2d != MPI_COMM_NULL) {
	for (a_cpt = 0; a_cpt < a_receiver_cpt; a_cpt++) {
	    sprintf(varname, "velox%d", a_indirect[a_cpt] + 1);
	    varid1 =
		adios_define_var(adios_handle1, varname, "", adios_double,
				 varsize, "", "");
	}
    }
#endif



#ifdef ADIOSALL
    sprintf(varsize, "%d", a_receiver_cpt * a_buffer_rate);
//      a_buffer_vel=mydmatrix0(0,OUT.iObs,0, a_receiver_cpt*(a_buffer_rate));
    a_temp = mydvector0(0, (a_receiver_cpt) * (a_buffer_rate));

//      Adios Variables
    if (a_comm2d != MPI_COMM_NULL) {
	sprintf(varname, "velox%d", a_me);
	varid1 =
	    adios_define_var(adios_handle1, varname, "", adios_double,
			     varsize, "", "");
    }
#endif


//      Controle
    MPI_Barrier(comm2d);
    printf("RANK = %d - #of Receiver = %d \n", PRM.me, a_receiver_cpt);
    MPI_Barrier(comm2d);





//      Adios OPEN
#ifdef ADIOSNOLOOP
    if ((a_comm2d != MPI_COMM_NULL)) {
	adios_open(&adios_handle1, "receiver", adios_filename1, "w",
		   a_comm2d);
	human_size_group = a_receiver_cpt * 8 * TMAX;
	adios_group_size(adios_handle1, human_size_group,
			 &machine_size_group);
    }
#endif
#endif

/////////////// End ADIOS ///////////////////


#if ( VERBOSE > 0 )
    VerifFunction(PrintInfoMedium(ANL, MDM, SRC, PRM), "InfoMedium", PRM);
#endif



    VerifFunction(EXIT_SUCCESS, "Beginning of the iteration", PRM);

#ifdef MISS
    if (retval = PAPI_reset(EventSet) != PAPI_OK) {
	printf("ERROR stop\n");
    }
    if (retval = PAPI_start(EventSet) != PAPI_OK) {
	printf("ERROR start\n");
    }
#endif

#ifdef FLOPS
    if ((retval =
	 PAPI_flops(&ireal_time, &iproc_time, &iflpops,
		    &imflops)) < PAPI_OK) {
	printf("Could not initialize PAPI_flops\n");
	printf
	    ("Your platform may not support floating point operation event.\n");
	printf("retval: %d\n", retval);
	exit(EXIT_FAILURE);
    }
#endif

#if (TIMER==1)
    timing3 = my_second();
#endif

#if (TIMER==2)
#ifdef MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    timing3 = my_second();
#endif

#ifdef MPI
#if (COMM==1)			/* Prepare Sending & Receiving  messages */
    MPI_Barrier(MPI_COMM_WORLD);
    /* == Stress == */
    /* East - West */
    /*  E->W */
    if (EAST.rank != -1)
	MPI_Send_init(EAST.bufT0S, 6 * EAST.nmax, MPI_DOUBLE,
		      EAST.rank, EAST.channelT0S, MPI_COMM_WORLD,
		      &reqT0S[3]);

    if (WEST.rank != -1)
	MPI_Recv_init(WEST.bufT0R, 6 * WEST.nmax, MPI_DOUBLE,
		      WEST.rank, WEST.channelT0R, MPI_COMM_WORLD,
		      &reqT0R[3]);

    /* W->E */
    if (WEST.rank != -1)
	MPI_Send_init(WEST.bufT0S, 6 * WEST.nmax, MPI_DOUBLE,
		      WEST.rank, WEST.channelT0S, MPI_COMM_WORLD,
		      &reqT0S[4]);

    if (EAST.rank != -1)
	MPI_Recv_init(EAST.bufT0R, 6 * EAST.nmax, MPI_DOUBLE,
		      EAST.rank, EAST.channelT0R, MPI_COMM_WORLD,
		      &reqT0R[4]);

    /* North - South */
    /* N->S */
    if (NORTH.rank != -1)
	MPI_Send_init(NORTH.bufT0S, 6 * NORTH.nmax, MPI_DOUBLE,
		      NORTH.rank, NORTH.channelT0S, MPI_COMM_WORLD,
		      &reqT0S[1]);

    if (SOUTH.rank != -1)
	MPI_Recv_init(SOUTH.bufT0R, 6 * SOUTH.nmax, MPI_DOUBLE,
		      SOUTH.rank, SOUTH.channelT0R, MPI_COMM_WORLD,
		      &reqT0R[1]);
    /* S->N */
    if (SOUTH.rank != -1)
	MPI_Send_init(SOUTH.bufT0S, 6 * SOUTH.nmax, MPI_DOUBLE,
		      SOUTH.rank, SOUTH.channelT0S, MPI_COMM_WORLD,
		      &reqT0S[2]);

    if (NORTH.rank != -1)
	MPI_Recv_init(NORTH.bufT0R, 6 * NORTH.nmax, MPI_DOUBLE,
		      NORTH.rank, NORTH.channelT0R, MPI_COMM_WORLD,
		      &reqT0R[2]);

    /* == Velocity == */
    /* East - West */
    /* E->W */
    if (EAST.rank != -1)
	MPI_Send_init(EAST.bufV0S, 3 * EAST.nmax, MPI_DOUBLE,
		      EAST.rank, EAST.channelV0S, MPI_COMM_WORLD,
		      &reqV0S[3]);

    if (WEST.rank != -1)
	MPI_Recv_init(WEST.bufV0R, 3 * WEST.nmax, MPI_DOUBLE,
		      WEST.rank, WEST.channelV0R, MPI_COMM_WORLD,
		      &reqV0R[3]);

    /* W->E */
    if (WEST.rank != -1)
	MPI_Send_init(WEST.bufV0S, 3 * WEST.nmax, MPI_DOUBLE,
		      WEST.rank, WEST.channelV0S, MPI_COMM_WORLD,
		      &reqV0S[4]);

    if (EAST.rank != -1)
	MPI_Recv_init(EAST.bufV0R, 3 * EAST.nmax, MPI_DOUBLE,
		      EAST.rank, EAST.channelV0R, MPI_COMM_WORLD,
		      &reqV0R[4]);

    /* North - South */
    /* N->S */
    if (NORTH.rank != -1)
	MPI_Send_init(NORTH.bufV0S, 3 * NORTH.nmax, MPI_DOUBLE,
		      NORTH.rank, NORTH.channelV0S, MPI_COMM_WORLD,
		      &reqV0S[1]);

    if (SOUTH.rank != -1)
	MPI_Recv_init(SOUTH.bufV0R, 3 * SOUTH.nmax, MPI_DOUBLE,
		      SOUTH.rank, SOUTH.channelV0R, MPI_COMM_WORLD,
		      &reqV0R[1]);

    /* S->N */
    if (SOUTH.rank != -1)
	MPI_Send_init(SOUTH.bufV0S, 3 * SOUTH.nmax, MPI_DOUBLE,
		      SOUTH.rank, SOUTH.channelV0S, MPI_COMM_WORLD,
		      &reqV0S[2]);

    if (NORTH.rank != -1)
	MPI_Recv_init(NORTH.bufV0R, 3 * NORTH.nmax, MPI_DOUBLE,
		      NORTH.rank, NORTH.channelV0R, MPI_COMM_WORLD,
		      &reqV0R[2]);


    /* == Ksil == */
    if (ANLmethod == KRISTEKandMOCZO) {
	/* East - West */
	/* E->W */
	if (EAST.rank != -1)
	    MPI_Send_init(EAST.bufKsiS, 6 * EAST.nmax, MPI_DOUBLE,
			  EAST.rank, EAST.channelKsiS, MPI_COMM_WORLD,
			  &reqKsiS[3]);

	if (WEST.rank != -1)
	    MPI_Recv_init(WEST.bufKsiR, 6 * WEST.nmax, MPI_DOUBLE,
			  WEST.rank, WEST.channelKsiR, MPI_COMM_WORLD,
			  &reqKsiR[3]);

	/* W->E */
	if (WEST.rank != -1)
	    MPI_Send_init(WEST.bufKsiS, 6 * WEST.nmax, MPI_DOUBLE,
			  WEST.rank, WEST.channelKsiS, MPI_COMM_WORLD,
			  &reqKsiS[4]);

	if (EAST.rank != -1)
	    MPI_Recv_init(EAST.bufKsiR, 6 * EAST.nmax, MPI_DOUBLE,
			  EAST.rank, EAST.channelKsiR, MPI_COMM_WORLD,
			  &reqKsiR[4]);

	/* North - South */
	/* N->S */
	if (NORTH.rank != -1)
	    MPI_Send_init(NORTH.bufKsiS, 6 * NORTH.nmax, MPI_DOUBLE,
			  NORTH.rank, NORTH.channelKsiS, MPI_COMM_WORLD,
			  &reqKsiS[1]);

	if (SOUTH.rank != -1)
	    MPI_Recv_init(SOUTH.bufKsiR, 6 * SOUTH.nmax, MPI_DOUBLE,
			  SOUTH.rank, SOUTH.channelKsiR, MPI_COMM_WORLD,
			  &reqKsiR[1]);

	/* S->N */
	if (SOUTH.rank != -1)
	    MPI_Send_init(SOUTH.bufKsiS, 6 * SOUTH.nmax, MPI_DOUBLE,
			  SOUTH.rank, SOUTH.channelKsiS, MPI_COMM_WORLD,
			  &reqKsiS[2]);

	if (NORTH.rank != -1)
	    MPI_Recv_init(NORTH.bufKsiR, 6 * NORTH.nmax, MPI_DOUBLE,
			  NORTH.rank, NORTH.channelKsiR, MPI_COMM_WORLD,
			  &reqKsiR[2]);

    }				/* end ksil comm */
    MPI_Barrier(MPI_COMM_WORLD);

#endif				/* PERSISTANT */

#endif				/* MPI */
    /* useful mapping of PRM */
    XMIN = PRM.xMin;
    XMAX = PRM.xMax;
    YMIN = PRM.yMin;
    YMAX = PRM.yMax;
    ZMIN = PRM.zMin;
    ZMAX = PRM.zMax;
    ZMAX0 = PRM.zMax0;
    DELTA = PRM.delta;
    DT = PRM.dt;
    DS = PRM.ds;
    MPMX = PRM.mpmx;
    MPMY = PRM.mpmy;
    TMAX = PRM.tMax;




//      Allocation du PGV
#ifdef PGV
    OUT.PGVlocal = dmatrix0(-1, MPMX + 2, -1, MPMY + 2);
    OUT.PGVglobal =
	dmatrix0(XMIN - DELTA, XMAX + DELTA + 2, YMIN - DELTA,
		 YMAX + DELTA + 2);
#endif


////////////////////////////////////////////////////////////////////////
//      Iteration in TIME
/////////////////////////////////////////////////////////////////////////

#ifdef EXPORT_TOPOLOGY
    fprintf(stdout, "EXPORT_TOPOLOGY# %d %d %d %d %d\n", PRM.me,
	    PRM.coords[0], PRM.coords[1], PRM.px, PRM.py);
    fflush(stdout);
    MPI_Finalize();
    exit(0);
#endif

#ifdef LUCAS_INSTRUMENTATION
    SCOREP_USER_REGION_DEFINE(iteration)
#endif
	for (l = 1; l <= PRM.tMax; l++) {
#ifdef LUCAS_INSTRUMENTATION
	SCOREP_USER_REGION_BEGIN(iteration, "iteration",
				 SCOREP_USER_REGION_TYPE_COMMON)
#endif
#ifdef PRINTSTEP
	    if (PRM.me == 0)
	    printf("=== time step : %i === \n", l);
#endif
	time = DT * l;
#if(VERBOSE > 0)
#ifdef MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif

	if (PRM.me == 0) {
	    fprintf(stderr, "=== time step : %i === \n", l);
	}
#endif
	if (source == HISTFILE)
	    fStatus = computeSeisMoment(&SRC, time, PRM);

#if(VERBOSE > 1)
	VerifFunction(fStatus, "increment seismic moment", PRM);
#endif

#if (TIMER==2)
#ifdef MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif

	timing1 = my_second();
#endif

#if (TIMER==1)
	timing1 = my_second();
#endif

	/* Calculation */
	/* === First step : t = l + 1/2 for stress === */

#ifdef PROFILE2
	TAU_PROFILE_START(compute_sig);
#endif

	/* computation of intermediates :
	   Phiv (CPML), t??? (PML), ksi (Day & Bradley), ksil (Kristek and Moczo) */
	/* imode : to increase the velocity of the computation, we begin by computing
	   the values of ksil at the boundaries of the px * py parts of the array
	   Afterwise, we can compute the values of ksil in the middle */
#if (VERBOSE > 1)
#ifdef MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	if (PRM.me == 0) {
	    fprintf(stderr, "/* Compute Intermediates : l */ \n ", l);
	}
#endif


	for (imode = 1; imode <= 5; imode++) {

	    if (imode == 1) {
		mpmx_begin = 1;
		mpmx_end = 3;
		mpmy_begin = 1;
		mpmy_end = MPMY;
	    }

	    if (imode == 2) {
		mpmx_begin = MPMX - 2;
		mpmx_end = MPMX;
		mpmy_begin = 1;
		mpmy_end = MPMY;
	    }

	    if (imode == 3) {
		mpmy_begin = 1;
		mpmy_end = 3;
		mpmx_begin = 4;
		mpmx_end = MPMX - 3;
	    }

	    if (imode == 4) {
		mpmy_begin = MPMY - 2;
		mpmy_end = MPMY;
		mpmx_begin = 4;
		mpmx_end = MPMX - 3;
	    }

	    if (imode == 5) {	/* imode = 5 --> middle of each part of the array */
		mpmx_begin = 4;
		mpmx_end = MPMX - 3;
		mpmy_begin = 4;
		mpmy_end = MPMY - 3;
		/* COMMUNICATIONS */
#ifdef MPI
#if (COMM==1)
		if (ANLmethod == KRISTEKandMOCZO) {
		    /* We are sending computed Ksil for computeStress */
		    /*  0: send, 1: receive */
		    SyncBufKsil(&ANL, 0, &NORTH, PRM);
		    SyncBufKsil(&ANL, 0, &SOUTH, PRM);
		    SyncBufKsil(&ANL, 0, &EAST, PRM);
		    SyncBufKsil(&ANL, 0, &WEST, PRM);
		    /* start send */
		    if (NORTH.rank != -1)
			MPI_Start(&reqKsiS[1]);
		    if (SOUTH.rank != -1)
			MPI_Start(&reqKsiS[2]);
		    if (EAST.rank != -1)
			MPI_Start(&reqKsiS[3]);
		    if (WEST.rank != -1)
			MPI_Start(&reqKsiS[4]);
		    /* start receive */
		    if (SOUTH.rank != -1)
			MPI_Start(&reqKsiR[1]);
		    if (NORTH.rank != -1)
			MPI_Start(&reqKsiR[2]);
		    if (WEST.rank != -1)
			MPI_Start(&reqKsiR[3]);
		    if (EAST.rank != -1)
			MPI_Start(&reqKsiR[4]);

		}		/* end ANLmethod == KRISTEKandMOCZO */
#endif				/* end of first method */
#endif

	    }			/* end of imode = 5 */
#ifndef NOINTERMEDIATES
	    ComputeIntermediates(&ABC, &ANL,
				 /* Parameters */
				 v0, PRM, MDM, mpmx_begin, mpmx_end,	/* local computed domain */
				 mpmy_begin, mpmy_end);
#endif
	}			/* end of imode */
	/* end of computation of intermediates */

	/* COMMUNICATE KSIL (K&M anelasticity method) */
	if (ANLmethod == KRISTEKandMOCZO) {
#ifdef MPI
#if (COMM==1)
	    /* We are receiving computed Ksil for computeStress */
	    /* wait until that is really sent */
	    if (NORTH.rank != -1)
		MPI_Wait(&reqKsiS[1], &statKsiS[1]);
	    if (SOUTH.rank != -1)
		MPI_Wait(&reqKsiS[2], &statKsiS[2]);
	    if (EAST.rank != -1)
		MPI_Wait(&reqKsiS[3], &statKsiS[3]);
	    if (WEST.rank != -1)
		MPI_Wait(&reqKsiS[4], &statKsiS[4]);

	    /* wait until that is really received */
	    if (SOUTH.rank != -1)
		MPI_Wait(&reqKsiR[1], &statKsiR[1]);
	    if (NORTH.rank != -1)
		MPI_Wait(&reqKsiR[2], &statKsiR[2]);
	    if (WEST.rank != -1)
		MPI_Wait(&reqKsiR[3], &statKsiR[3]);
	    if (EAST.rank != -1)
		MPI_Wait(&reqKsiR[4], &statKsiR[4]);

	    /*  0: send, 1: receive */
	    SyncBufKsil(&ANL, 1, &NORTH, PRM);
	    SyncBufKsil(&ANL, 1, &SOUTH, PRM);
	    SyncBufKsil(&ANL, 1, &EAST, PRM);
	    SyncBufKsil(&ANL, 1, &WEST, PRM);
#endif				/* end of first method */
#if (COMM==2)
	    /* Synchronize Sending buffers */
	    /*  0: send, 1: receive */

	    SyncBufKsil(&ANL, 0, &WEST, PRM);
	    SyncBufKsil(&ANL, 0, &EAST, PRM);
	    SyncBufKsil(&ANL, 0, &NORTH, PRM);
	    SyncBufKsil(&ANL, 0, &SOUTH, PRM);

	    /* Communicate  */
	    /* E->W */
	    if (EAST.rank != -1) {
		MPI_Isend(EAST.bufKsiS, 6 * EAST.nmax, MPI_DOUBLE,
			  EAST.rank, EAST.channelKsiS, MPI_COMM_WORLD,
			  &req[3]);
	    }
	    if (WEST.rank != -1) {
		MPI_Recv(WEST.bufKsiR, 6 * WEST.nmax, MPI_DOUBLE,
			 WEST.rank, WEST.channelKsiR, MPI_COMM_WORLD,
			 &status[3]);
	    } else {
		MPI_Wait(&req[3], &status[3]);
	    }


	    /* W->E */
	    if (WEST.rank != -1) {
		MPI_Isend(WEST.bufKsiS, 6 * WEST.nmax, MPI_DOUBLE,
			  WEST.rank, WEST.channelKsiS, MPI_COMM_WORLD,
			  &req[4]);
	    }
	    if (EAST.rank != -1) {
		MPI_Recv(EAST.bufKsiR, 6 * EAST.nmax, MPI_DOUBLE,
			 EAST.rank, EAST.channelKsiR, MPI_COMM_WORLD,
			 &status[4]);
	    } else {
		MPI_Wait(&req[4], &status[4]);
	    }

	    /* N->S */
	    if (NORTH.rank != -1) {
		MPI_Isend(NORTH.bufKsiS, 6 * NORTH.nmax, MPI_DOUBLE,
			  NORTH.rank, NORTH.channelKsiS, MPI_COMM_WORLD,
			  &req[1]);
	    }
	    if (SOUTH.rank != -1) {
		MPI_Recv(SOUTH.bufKsiR, 6 * SOUTH.nmax, MPI_DOUBLE,
			 SOUTH.rank, SOUTH.channelKsiR, MPI_COMM_WORLD,
			 &status[1]);
	    } else {
		MPI_Wait(&req[1], &status[1]);
	    }

	    /* S->N */
	    if (SOUTH.rank != -1) {
		MPI_Isend(SOUTH.bufKsiS, 6 * SOUTH.nmax, MPI_DOUBLE,
			  SOUTH.rank, SOUTH.channelKsiS, MPI_COMM_WORLD,
			  &req[2]);
	    }
	    if (NORTH.rank != -1) {
		MPI_Recv(NORTH.bufKsiR, 6 * NORTH.nmax, MPI_DOUBLE,
			 NORTH.rank, NORTH.channelKsiR, MPI_COMM_WORLD,
			 &status[2]);
	    } else {
		MPI_Wait(&req[2], &status[2]);
	    }

	    /* Synchronize Received buffers */
	    /*  0: send, 1: receive */
	    SyncBufKsil(&ANL, 1, &EAST, PRM);
	    SyncBufKsil(&ANL, 1, &WEST, PRM);
	    SyncBufKsil(&ANL, 1, &SOUTH, PRM);
	    SyncBufKsil(&ANL, 1, &NORTH, PRM);

#endif				/* end of BLOCKING communication method */
#endif				/* MPI */
	}

	/* end ANLmethod == KRISTEKandMOCZO */
	/* imode : to increase the velocity of the computation, we begin by computing
	   the values of stress at the boundaries of the px * py parts of the array
	   Afterwise, we can compute the values of the stress in the middle */
	for (imode = 1; imode <= 5; imode++) {

	    if (imode == 1) {
		mpmx_begin = 1;
		mpmx_end = 3;
		mpmy_begin = 1;
		mpmy_end = MPMY;
	    }

	    if (imode == 2) {
		mpmx_begin = MPMX - 2;
		mpmx_end = MPMX;
		mpmy_begin = 1;
		mpmy_end = MPMY;
	    }

	    if (imode == 3) {
		mpmy_begin = 1;
		mpmy_end = 3;
		mpmx_begin = 4;
		mpmx_end = MPMX - 3;
	    }

	    if (imode == 4) {
		mpmy_begin = MPMY - 2;
		mpmy_end = MPMY;
		mpmx_begin = 4;
		mpmx_end = MPMX - 3;
	    }

	    if (imode == 5) {	/* imode = 5 --> middle of each part of the array */
		mpmx_begin = 4;
		mpmx_end = MPMX - 3;
		mpmy_begin = 4;
		mpmy_end = MPMY - 3;

		/* Communications
		 *   first method & second method
		 * what only differs communications
		 */
#ifdef MPI
#if (COMM==1)

		/* We are sending computed stress for computeVelocity */
		/*  0: send, 1: receive */
		SyncBufStress(&t0, 0, &NORTH, PRM);
		SyncBufStress(&t0, 0, &SOUTH, PRM);
		SyncBufStress(&t0, 0, &EAST, PRM);
		SyncBufStress(&t0, 0, &WEST, PRM);

		/* start send */
		if (NORTH.rank != -1)
		    MPI_Start(&reqT0S[1]);
		if (SOUTH.rank != -1)
		    MPI_Start(&reqT0S[2]);
		if (EAST.rank != -1)
		    MPI_Start(&reqT0S[3]);
		if (WEST.rank != -1)
		    MPI_Start(&reqT0S[4]);


		/* start receive */
		if (SOUTH.rank != -1)
		    MPI_Start(&reqT0R[1]);
		if (NORTH.rank != -1)
		    MPI_Start(&reqT0R[2]);
		if (WEST.rank != -1)
		    MPI_Start(&reqT0R[3]);
		if (EAST.rank != -1)
		    MPI_Start(&reqT0R[4]);
#endif				/* end of first method */
#endif				/* MPI */

	    }
	    /* end of imode = 5 */
	    /* Beginning of stress computation */
#if(VERBOSE > 1)
//        MPI_Barrier (MPI_COMM_WORLD);
	    if (PRM.me == 0) {
		fprintf(stderr,
			" /* Beginning of stress computation : %li */ \n",
			l);
	    }
#endif

#ifndef NOSTRESS

	    ComputeStress(&t0,
			  /* INPUTS */
			  v0, MDM, PRM, ABC, ANL, mpmx_begin, mpmx_end,	/* local computed domain */
			  mpmy_begin, mpmy_end);

#endif

	}			/* end of imode */




#if (VERBOSE > 1)
#ifdef MPI
	MPI_Barrier(MPI_COMM_WORLD);
	if (PRM.me == 0) {
	    perror("## End of Stress ");
	}
#endif				/* MPI */
#endif

#ifdef PROFILE2
	TAU_PROFILE_STOP(compute_sig);
#endif

#if (TIMER==2)
#ifdef MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	timing2 = my_second();
	timing_bc1 = timing_bc1 + (timing2 - timing1);
#endif

#if (TIMER==1)
	timing2 = my_second();
	timing_bc1 = timing_bc1 + (timing2 - timing1);
#endif

#if (TIMER==2)
#ifdef MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	timing1 = my_second();
#endif

#if (TIMER==1)
	timing1 = my_second();
#endif

#ifdef PROFILE1
	TAU_PROFILE_START(exchange_sig);
#endif

#ifdef PROFILE2
	TAU_PROFILE_START(exchange_sig);
#endif


	/* Communication : first method */
#ifdef MPI
#if (COMM==1)

	/* wait until that is really sent */
	if (NORTH.rank != -1)
	    MPI_Wait(&reqT0S[1], &statT0S[1]);
	if (SOUTH.rank != -1)
	    MPI_Wait(&reqT0S[2], &statT0S[2]);
	if (EAST.rank != -1)
	    MPI_Wait(&reqT0S[3], &statT0S[3]);
	if (WEST.rank != -1)
	    MPI_Wait(&reqT0S[4], &statT0S[4]);

	/* wait until that is really received */
	if (SOUTH.rank != -1)
	    MPI_Wait(&reqT0R[1], &statT0R[1]);
	if (NORTH.rank != -1)
	    MPI_Wait(&reqT0R[2], &statT0R[2]);
	if (WEST.rank != -1)
	    MPI_Wait(&reqT0R[3], &statT0R[3]);
	if (EAST.rank != -1)
	    MPI_Wait(&reqT0R[4], &statT0R[4]);

	/*  0: send, 1: receive */
	SyncBufStress(&t0, 1, &NORTH, PRM);
	SyncBufStress(&t0, 1, &SOUTH, PRM);
	SyncBufStress(&t0, 1, &EAST, PRM);
	SyncBufStress(&t0, 1, &WEST, PRM);

#endif				/* end of first method */

#if (COMM==2)
	/* Communication to synchronize */

	/* Synchronize Sending buffers */
	/*  0: send, 1: receive */
	SyncBufStress(&t0, 0, &WEST, PRM);
	SyncBufStress(&t0, 0, &EAST, PRM);
	SyncBufStress(&t0, 0, &NORTH, PRM);
	SyncBufStress(&t0, 0, &SOUTH, PRM);

	/* Communicate  */
	/* E->W */
	if (EAST.rank != -1) {
	    MPI_Isend(EAST.bufT0S, 6 * EAST.nmax, MPI_DOUBLE, EAST.rank,
		      EAST.channelT0S, MPI_COMM_WORLD, &req[3]);
	}
	if (WEST.rank != -1) {
	    MPI_Recv(WEST.bufT0R, 6 * WEST.nmax, MPI_DOUBLE, WEST.rank,
		     WEST.channelT0R, MPI_COMM_WORLD, &status[3]);
	} else {
	    MPI_Wait(&req[3], &status[3]);
	}

	/* W->E */
	if (WEST.rank != -1) {
	    MPI_Isend(WEST.bufT0S, 6 * WEST.nmax, MPI_DOUBLE, WEST.rank,
		      WEST.channelT0S, MPI_COMM_WORLD, &req[4]);
	}
	if (EAST.rank != -1) {
	    MPI_Recv(EAST.bufT0R, 6 * EAST.nmax, MPI_DOUBLE, EAST.rank,
		     EAST.channelT0R, MPI_COMM_WORLD, &status[4]);
	} else {
	    MPI_Wait(&req[4], &status[4]);
	}

	/* N->S */
	if (NORTH.rank != -1) {
	    MPI_Isend(NORTH.bufT0S, 6 * NORTH.nmax, MPI_DOUBLE, NORTH.rank,
		      NORTH.channelT0S, MPI_COMM_WORLD, &req[1]);
	}
	if (SOUTH.rank != -1) {
	    MPI_Recv(SOUTH.bufT0R, 6 * SOUTH.nmax, MPI_DOUBLE, SOUTH.rank,
		     SOUTH.channelT0R, MPI_COMM_WORLD, &status[1]);
	} else {
	    MPI_Wait(&req[1], &status[1]);
	}

	/* S->N */
	if (SOUTH.rank != -1) {
	    MPI_Isend(SOUTH.bufT0S, 6 * SOUTH.nmax, MPI_DOUBLE, SOUTH.rank,
		      SOUTH.channelT0S, MPI_COMM_WORLD, &req[2]);
	}
	if (NORTH.rank != -1) {
	    MPI_Recv(NORTH.bufT0R, 6 * NORTH.nmax, MPI_DOUBLE, NORTH.rank,
		     NORTH.channelT0R, MPI_COMM_WORLD, &status[2]);
	} else {
	    MPI_Wait(&req[2], &status[2]);
	}

	/* Synchronize Received buffers */
	/*  0: send, 1: receive */
	SyncBufStress(&t0, 1, &EAST, PRM);
	SyncBufStress(&t0, 1, &WEST, PRM);
	SyncBufStress(&t0, 1, &SOUTH, PRM);
	SyncBufStress(&t0, 1, &NORTH, PRM);

#endif				/* end of third method */
#endif				/* MPI */


#ifdef PROFILE1
	TAU_PROFILE_STOP(exchange_sig);
#endif

#ifdef PROFILE2
	TAU_PROFILE_STOP(exchange_sig);
#endif

#if (TIMER==2)
#ifdef MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif				/* MPI */
	timing2 = my_second();
	timing_comm1 = timing_comm1 + (timing2 - timing1);
#endif

#if (TIMER==1)
	timing2 = my_second();
	timing_comm1 = timing_comm1 + (timing2 - timing1);
#endif

#if (TIMER==2)
#ifdef MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif				/* MPI */

	timing1 = my_second();
#endif

#if (TIMER==1)
	timing1 = my_second();
#endif


	/* imode : to increase the velocity of the computation, we begin by computing
	   the values of stress at the boundaries of the px * py parts of the array
	   Afterwise, we can compute the values of the stress in the middle */
	for (imode = 1; imode <= 5; imode++) {

	    if (imode == 1) {
		mpmx_begin = 1;
		mpmx_end = 3;
		mpmy_begin = 1;
		mpmy_end = MPMY;
	    }

	    if (imode == 2) {
		mpmx_begin = MPMX - 2;
		mpmx_end = MPMX;
		mpmy_begin = 1;
		mpmy_end = MPMY;
	    }

	    if (imode == 3) {
		mpmy_begin = 1;
		mpmy_end = 3;
		mpmx_begin = 4;
		mpmx_end = MPMX - 3;
	    }

	    if (imode == 4) {
		mpmy_begin = MPMY - 2;
		mpmy_end = MPMY;
		mpmx_begin = 4;
		mpmx_end = MPMX - 3;
	    }

	    if (imode == 5) {	/* imode = 5 --> middle of each part of the array */
		mpmx_begin = 4;
		mpmx_end = MPMX - 3;
		mpmy_begin = 4;
		mpmy_end = MPMY - 3;

		/* Communication : first method */

#ifdef MPI
#if (COMM==1)
		/* We are sending computed velocity */
		/*  0: send, 1: receive */
		SyncBufVelocity(&v0, 0, &NORTH, PRM);
		SyncBufVelocity(&v0, 0, &SOUTH, PRM);
		SyncBufVelocity(&v0, 0, &EAST, PRM);
		SyncBufVelocity(&v0, 0, &WEST, PRM);

		if (NORTH.rank != -1)
		    MPI_Start(&reqV0S[1]);
		if (SOUTH.rank != -1)
		    MPI_Start(&reqV0S[2]);
		if (EAST.rank != -1)
		    MPI_Start(&reqV0S[3]);
		if (WEST.rank != -1)
		    MPI_Start(&reqV0S[4]);

		if (SOUTH.rank != -1)
		    MPI_Start(&reqV0R[1]);
		if (NORTH.rank != -1)
		    MPI_Start(&reqV0R[2]);
		if (WEST.rank != -1)
		    MPI_Start(&reqV0R[3]);
		if (EAST.rank != -1)
		    MPI_Start(&reqV0R[4]);
#endif				/* end of first method */
#endif				/* MPI */

	    }

	    /* end of imode = 5 */
	    /* Beginning of velocity computation */
#if(VERBOSE > 1)
#ifdef MPI
	    MPI_Barrier(MPI_COMM_WORLD);
#endif

	    if (PRM.me == 0) {
		fprintf(stderr,
			" ## begin velocity computation : %i */ \n", l);
	    }
#endif
#ifndef NOVELOCITY
	    computeVelocity(&v0, &ABC,
			    /* INPUTS */
			    t0, MDM, PRM, ANL, SRC, mpmx_begin, mpmx_end,	/* local computed domain */
			    mpmy_begin, mpmy_end, l	/* time index */
		);

#endif

	}			/* end of imode */

	if (source == VELO) {

	    computeSource(&v0, PRM, MDM, SRC, l);

	}
#if(VERBOSE > 1)
#ifdef MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif

	if (PRM.me == 0) {
	    perror(" ## end velocity */ \n");
	}
#endif

#ifdef PROFILE2
	TAU_PROFILE_STOP(compute_vit);
#endif

#if (TIMER==2)
#ifdef MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	timing2 = my_second();
	timing_bc2 = timing_bc2 + (timing2 - timing1);
#endif

#if (TIMER==1)
	timing2 = my_second();
	timing_bc2 = timing_bc2 + (timing2 - timing1);
#endif

#if (TIMER==2)
#ifdef MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	timing1 = my_second();
#endif

#if (TIMER==1)
	timing1 = my_second();
#endif

#ifdef PROFILE1
	TAU_PROFILE_START(exchange_vit);
#endif

#ifdef PROFILE2
	TAU_PROFILE_START(exchange_vit);
#endif

#if(VERBOSE > 1)
#ifdef MPI
	MPI_Barrier(MPI_COMM_WORLD);
	if (PRM.me == 0) {
	    perror("\n /* Beginning velocity communications */ \n");
	}
#endif
#endif

#ifdef MPI
#if (COMM==1)			/* Communication : first method */
	/* wait until that is really sent */
	if (NORTH.rank != -1)
	    MPI_Wait(&reqV0S[1], &statV0S[1]);
	if (SOUTH.rank != -1)
	    MPI_Wait(&reqV0S[2], &statV0S[2]);
	if (EAST.rank != -1)
	    MPI_Wait(&reqV0S[3], &statV0S[3]);
	if (WEST.rank != -1)
	    MPI_Wait(&reqV0S[4], &statV0S[4]);

	/* wait until that is really received */
	if (SOUTH.rank != -1)
	    MPI_Wait(&reqV0R[1], &statV0R[1]);
	if (NORTH.rank != -1)
	    MPI_Wait(&reqV0R[2], &statV0R[2]);
	if (WEST.rank != -1)
	    MPI_Wait(&reqV0R[3], &statV0R[3]);
	if (EAST.rank != -1)
	    MPI_Wait(&reqV0R[4], &statV0R[4]);

	/* synchronise Receiving buffers */
	/*  0: send, 1: receive */
	SyncBufVelocity(&v0, 1, &NORTH, PRM);
	SyncBufVelocity(&v0, 1, &SOUTH, PRM);
	SyncBufVelocity(&v0, 1, &EAST, PRM);
	SyncBufVelocity(&v0, 1, &WEST, PRM);
#endif				/* end of first method */


#if(BLOCKING)			/* Communication : third method */
	/* Synchronize Sending buffers */
	/*  0: send, 1: receive */

	SyncBufVelocity(&v0, 0, &WEST, PRM);
	SyncBufVelocity(&v0, 0, &EAST, PRM);
	SyncBufVelocity(&v0, 0, &NORTH, PRM);
	SyncBufVelocity(&v0, 0, &SOUTH, PRM);

	/* Communicate  */
	/* E->W */
	if (EAST.rank != -1) {
	    MPI_Isend(EAST.bufV0S, 3 * EAST.nmax, MPI_DOUBLE, EAST.rank,
		      EAST.channelV0S, MPI_COMM_WORLD, &req[3]);
	}
	if (WEST.rank != -1) {
	    MPI_Recv(WEST.bufV0R, 3 * WEST.nmax, MPI_DOUBLE, WEST.rank,
		     WEST.channelV0R, MPI_COMM_WORLD, &status[3]);
	} else {
	    MPI_Wait(&req[3], &status[3]);
	}

	/* W->E */
	if (WEST.rank != -1) {
	    MPI_Isend(WEST.bufV0S, 3 * WEST.nmax, MPI_DOUBLE, WEST.rank,
		      WEST.channelV0S, MPI_COMM_WORLD, &req[4]);
	}
	if (EAST.rank != -1) {
	    MPI_Recv(EAST.bufV0R, 3 * EAST.nmax, MPI_DOUBLE, EAST.rank,
		     EAST.channelV0R, MPI_COMM_WORLD, &status[4]);
	} else {
	    MPI_Wait(&req[4], &status[4]);
	}

	/* N->S */
	if (NORTH.rank != -1) {
	    MPI_Isend(NORTH.bufV0S, 3 * NORTH.nmax, MPI_DOUBLE, NORTH.rank,
		      NORTH.channelV0S, MPI_COMM_WORLD, &req[1]);
	}
	if (SOUTH.rank != -1) {
	    MPI_Recv(SOUTH.bufV0R, 3 * SOUTH.nmax, MPI_DOUBLE, SOUTH.rank,
		     SOUTH.channelV0R, MPI_COMM_WORLD, &status[1]);
	} else {
	    MPI_Wait(&req[1], &status[1]);
	}

	/* S->N */
	if (SOUTH.rank != -1) {
	    MPI_Isend(SOUTH.bufV0S, 3 * SOUTH.nmax, MPI_DOUBLE, SOUTH.rank,
		      SOUTH.channelV0S, MPI_COMM_WORLD, &req[2]);
	}
	if (NORTH.rank != -1) {
	    MPI_Recv(NORTH.bufV0R, 3 * NORTH.nmax, MPI_DOUBLE, NORTH.rank,
		     NORTH.channelV0R, MPI_COMM_WORLD, &status[2]);
	} else {
	    MPI_Wait(&req[2], &status[2]);
	}

	/* Synchronize Received buffers */
	/*  0: send, 1: receive */
	SyncBufVelocity(&v0, 1, &EAST, PRM);
	SyncBufVelocity(&v0, 1, &WEST, PRM);
	SyncBufVelocity(&v0, 1, &SOUTH, PRM);
	SyncBufVelocity(&v0, 1, &NORTH, PRM);

#endif				/* end of third method */
#endif				/* MPI */

#ifdef PROFILE1
	TAU_PROFILE_STOP(exchange_vit);
#endif

#ifdef PROFILE2
	TAU_PROFILE_STOP(exchange_vit);
#endif

#if (TIMER==2)
#ifdef MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	timing2 = my_second();
	timing_comm2 = timing_comm2 + (timing2 - timing1);
#endif

#if (TIMER==1)
	timing2 = my_second();
	timing_comm2 = timing_comm2 + (timing2 - timing1);
#endif

#if (TIMER==2)
#ifdef MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	timing1 = my_second();
#endif

#if (TIMER==1)
	timing1 = my_second();
#endif


////////////////////////////////////////////////////////////////////////////////////////////
//      OBS
//
//      STATION_STEP : Correspond a la frequence d ecriture
//      On ecrit tous les STATION_STEP pas de temps mais on ecrit ts les pas de temps
//
///////////////////////////////////////////////////////////////////////////////////////////

//      If ADIOS = True  then STATION_STEP=1 and  OUT->seis_output is indexed on #timestep
	fStatus = ComputeSeismograms(&OUT, v0, t0, PRM, l - 1);

#ifdef ADIOSOBS
#if defined (ADIOSLOOP) || defined (ADIOSNOLOOP)
//      Bufferisation
	if (a_comm2d != MPI_COMM_NULL) {
	    for (a_cpt = 0; a_cpt < a_receiver_cpt; a_cpt++) {
		a_buffer_vel[a_cpt][a_buffer_cpt] = 1000 * l + 0.0001 * a_indirect[a_cpt];	//OUT.seis_output[0][ir][1];     
	    }
	}
#endif


#ifdef ADIOSALL
	if (a_comm2d != MPI_COMM_NULL) {
	    for (a_cpt = 0; a_cpt < a_receiver_cpt; a_cpt++) {
		a_temp[a_buffer_cpt * a_receiver_cpt + a_cpt] =
		    1000 * l + 0.0001 * a_indirect[a_cpt];
	    }
	}
#endif


//      Tracking du nombre d ecritures a bufferiser
	if (l % a_buffer_rate == 0) {
	    a_buffer_cpt = 0;
	} else {
	    a_buffer_cpt++;
	}

//      Manque le cas ou TMAX % a_buffer_rate different de zero et on arrive en fin de simulation
	if (a_comm2d != MPI_COMM_NULL) {


	    if (l % a_buffer_rate == 0) {

#if defined (ADIOSLOOP) || defined (ADIOSALL)

		if ((l == a_buffer_rate)) {
		    adios_open(&adios_handle1, "receiver", adios_filename1,
			       "w", a_comm2d);
		} else {
		    adios_open(&adios_handle1, "receiver", adios_filename1,
			       "a", a_comm2d);
		}
#endif


#ifdef ADIOSALL
		sprintf(varname, "velox%d", a_me);
		adios_write(adios_handle1, varname, a_temp);
#endif

#ifdef ADIOSLOOP
		for (a_cpt = 0; a_cpt < a_receiver_cpt; a_cpt++) {
		    sprintf(varname, "velox%d", a_indirect[a_cpt] + 1);
		    adios_write(adios_handle1, varname,
				a_buffer_vel[a_cpt]);
		}

#endif

#if defined (ADIOSLOOP) || defined (ADIOSALL)
		adios_close(adios_handle1);
#endif


//      if (a_comm2d != MPI_COMM_NULL) printf("Fin Adios Write %d \n",PRM.me);
	    }
	}
#endif




//////////////////////////////////////////////////////////////////////////////////////////
//      Debut Ecriture Stations
//      STATION_STEP : Correspond a la frequence d ecriture
//      On ecrit tous les STATION_STEP pas de temps mais on reecrit ts les pas de temps
//      A modifier avec la versin NetCDF de sortie des observations
////////////////////////////////////////////////////////////////////////////////////////
	if ((l % STATION_STEP == 0) || l == TMAX) {

#if (VERBOSE > 1)
	    VerifFunction(EXIT_SUCCESS,
			  "begin communication of seismogramms", PRM);
#endif
	    int field;
	    for (ir = 0; ir < OUT.iObs; ir++) {
		if (OUT.ista[ir] == 1) {
		    for (field = 1; field <= 9; field++) {	/* vx, vy, vz, tii, txy, txz, tyz */
			if (PRM.me == OUT.mapping_seis[ir][field] &&
			    PRM.me != 0) {
			    for (j = 0; j < STATION_STEP; j++) {	/* each time */
				OUT.seis_buff[j] =
				    OUT.seis_output[j][ir][field];
			    }
#ifdef MPI
			    MPI_Isend(OUT.seis_buff, STATION_STEP,
				      MPI_DOUBLE, 0, 200 + field, comm2d,
				      &sendreq[9]);
			    MPI_Wait(&sendreq[9], &status[1]);
#endif
			}
			/* end sync and send */
			if (PRM.me == 0 &&
			    OUT.mapping_seis[ir][field] != 0) {
#ifdef MPI
			    MPI_Recv(OUT.seis_buff, STATION_STEP,
				     MPI_DOUBLE,
				     OUT.mapping_seis[ir][field],
				     200 + field, comm2d, &status[1]);
#endif
			    for (j = 0; j < STATION_STEP; j++) {	/* each time */
				OUT.seis_output[j][ir][field] =
				    OUT.seis_buff[j];
			    }
			}
			/* end receive and sync */
		    }		/* end for field */

		    if (PRM.me == 0) {

/*
#ifdef OUTSTD
            strcpy(flname4, PRM.dir);
            strcat(flname4, char4);
            sprintf(number, "%d", ir+1);
            strcat(flname4, number);
            strcat(flname4, ".dat");
            fStatus=OutSeismograms(OUT,PRM,ir,l,flname4,STATION_STEP);
            VerifFunction(fStatus, "write seismogramm",PRM);
#endif
*/

#if defined (OUTSTD) || defined (OUTNEMESIS)
			strcpy(flname4, PRM.dir);
			strcat(flname4, char4);
			sprintf(number, "%d", ir + 1);
			strcat(flname4, number);
			strcat(flname4, ".dat");
			fStatus =
			    OutSeismogramsbis(OUT, PRM, ir, l, flname4,
					      STATION_STEP);
			VerifFunction(fStatus, "write seismogramm", PRM);
#endif



#ifdef OUTNETCDF
			strcpy(flname4, PRM.dir);
			strcat(flname4, char4);
			sprintf(number, "%d", ir + 1);
			strcat(flname4, number);
			strcat(flname4, ".nc");
			fStatus =
			    OutObsNetcdf(flname4, l, ir, TMAX, DT, OUT,
					 STATION_STEP);

			VerifFunction(fStatus, "write seismogramm", PRM);
#endif

#ifdef OUTBIN
			strcpy(flname4, PRM.dir);
			strcat(flname4, char4);
			sprintf(number, "%d", ir + 1);
			strcat(flname4, number);
			strcat(flname4, ".bin");
			fStatus =
			    OutSeismogramsbin(OUT, PRM, ir, l, flname4,
					      STATION_STEP);
			VerifFunction(fStatus, "write seismogramm", PRM);
#endif



		    }		/* end of PRM.me = 0 */
		}		/* end of station is inside domain */
	    }			/* end for each station */

	    /* blank seis_output for next values */
	    for (ir = 0; ir < OUT.iObs; ir++) {
		for (i = 0; i < STATION_STEP; i++) {
		    for (j = 1; j <= 9; j++) {	/* each field */
			OUT.seis_output[i][ir][j] = 0.;
		    }
		}
	    }




	}


	/* end output seismograms at l%STATION_STEP */
#if (VERBOSE > 1)
	VerifFunction(fStatus, "compute Seismograms", PRM);
#endif

////////////////////////////////////////////////////////////////////////////////////////
/************************************	Fin ecriture Stations ************************/
///////////////////////////////////////////////////////////////////////////////////////







	/* === Calculation of the snapshots === */
	if (snapType == ODISPL || snapType == OBOTH) {
	    /* xy surface */
	    int K0;
	    const double DT = PRM.dt;
	    if (surface == ABSORBING) {
		K0 = OUT.k0;
	    }

	    for (i = -1; i <= MPMX + 2; i++) {
		for (j = -1; j <= MPMY + 2; j++) {
		    if (surface == ABSORBING) {
			OUT.Uxy[1][i][j] += DT * v0.x[i][j][OUT.k0];
			OUT.Uxy[2][i][j] += DT * v0.y[i][j][OUT.k0];
			OUT.Uxy[3][i][j] += DT * v0.z[i][j][OUT.k0];
		    } else if (surface == FREE) {
			OUT.Uxy[1][i][j] += DT * v0.x[i][j][1];
			OUT.Uxy[2][i][j] += DT * v0.y[i][j][1];
			OUT.Uxy[3][i][j] += DT * v0.z[i][j][0];
		    }
		}
	    }
	    /* xz surface */
	    jcpu = PRM.j2jcpu_array[OUT.j0];
	    jmp_tmp = PRM.j2jmp_array[OUT.j0];
	    if (PRM.coords[1] == jcpu) {
		for (i = -1; i <= MPMX + 2; i++) {
		    for (k = ZMIN - DELTA; k <= ZMAX0; k++) {
			OUT.Uxz[1][i][k] += DT * v0.x[i][jmp_tmp][k];
			OUT.Uxz[2][i][k] += DT * v0.y[i][jmp_tmp][k];
			OUT.Uxz[3][i][k] += DT * v0.z[i][jmp_tmp][k];
		    }
		}
	    }
	    /* yz surface */
	    icpu = PRM.i2icpu_array[OUT.j0];
	    imp_tmp = PRM.i2imp_array[OUT.j0];
	    if (PRM.coords[0] == icpu) {
		for (j = -1; j <= MPMY + 2; j++) {
		    for (k = ZMIN - DELTA; k <= ZMAX0; k++) {
			OUT.Uyz[1][j][k] += DT * v0.x[imp_tmp][j][k];
			OUT.Uyz[2][j][k] += DT * v0.y[imp_tmp][j][k];
			OUT.Uyz[3][j][k] += DT * v0.z[imp_tmp][j][k];
		    }
		}
	    }
	    VerifFunction(EXIT_SUCCESS, "compute displacement snapshots",
			  PRM);
	}

	/* end calculation */
	/////////////////////////////////////////////////////////////////////////
///     Compute PGV
/////////////////////////////////////////////////////////////////////////////

#ifdef PGV
	for (i = 1; i <= MPMX; i++) {
	    for (j = 1; j <= MPMY; j++) {

		if (surface == ABSORBING || model == GEOLOGICAL) {
		    pgv_tmp =
			sqrt((v0.x[i][j][OUT.k0] * v0.x[i][j][OUT.k0] +
			      v0.y[i][j][OUT.k0] * v0.y[i][j][OUT.k0]));
		    if (pgv_tmp > OUT.PGVlocal[i][j])
			OUT.PGVlocal[i][j] = pgv_tmp;

		} else if (surface == FREE && model != GEOLOGICAL) {	/* free surface at z = 0 */
		    pgv_tmp =
			sqrt((v0.x[i][j][1] * v0.x[i][j][1] +
			      v0.y[i][j][1] * v0.y[i][j][1]));
		    if (pgv_tmp > OUT.PGVlocal[i][j])
			OUT.PGVlocal[i][j] = pgv_tmp;
		}

		/* end of if surface */
	    }			/* end of j */
	}			/* end of i */
#endif

/////////////////////////////////////////////////////////////////////////
///     Surface snapshots
///////////////////////////////////////////////////////////////////////////
///     Exchange surface snapshots
///////////////////////////////////////////////////////////////////////////


	if ((l % SURFACE_STEP) == 0) {
	    /* Output : files to create snapshots */
	    /* flname construct with
	       dir + surface xy/xz/yz + vel/disp + numero DT + 0
	       surfacexyXXXX00 */

	    /* Outputs of planes
	       fp1 --> k = 1 ou k0
	       fp2 --> j= j0
	       fp3 --> i= i0 */
	    int snapStep, snapStepMax = 1;
	    if (snapType == OBOTH) {
		snapStepMax = 2;
	    }
	    for (snapStep = 1; snapStep <= snapStepMax; snapStep++) {

		int outvel = 0, outdisp = 0;	/* 0: no, 1: yes */
		if ((snapType == OVELO) ||
		    (snapType == OBOTH && snapStep == 1)) {
		    outvel = 1;
		} else if ((snapType == ODISPL) ||
			   (snapType == OBOTH && snapStep == 2)) {
		    outdisp = 1;
		}

		strcpy(flname1, PRM.dir);
		strcpy(flname2, PRM.dir);
		strcpy(flname3, PRM.dir);
		strcpy(flname5, PRM.dir);


		strcat(flname1, char1);
		strcat(flname2, char2);
		strcat(flname3, char3);
		if (outvel == 1) {
		    strcat(flname1, "vel");
		    strcat(flname2, "vel");
		    strcat(flname3, "vel");
		} else if (outdisp == 1) {
		    strcat(flname1, "disp");
		    strcat(flname2, "disp");
		    strcat(flname3, "disp");
		}

		sprintf(number, "%4.4d", l);
		strcat(flname1, number);
		strcat(flname2, number);
		strcat(flname3, number);

		sprintf(number, "%2.2d", 0);
		strcat(flname1, number);
		strcat(flname2, number);
		strcat(flname3, number);

/**********************************************************************/
// Surfacexy 
/**********************************************************************/


		if (PRM.me == 0) {
		    OUT.Vxglobal =
			dmatrix(XMIN - DELTA, XMAX + DELTA + 2,
				YMIN - DELTA, YMAX + DELTA + 2);
		    OUT.Vyglobal =
			dmatrix(XMIN - DELTA, XMAX + DELTA + 2,
				YMIN - DELTA, YMAX + DELTA + 2);
		    OUT.Vzglobal =
			dmatrix(XMIN - DELTA, XMAX + DELTA + 2,
				YMIN - DELTA, YMAX + DELTA + 2);


		    for (i = 1; i <= MPMX; i++) {
			for (j = 1; j <= MPMY; j++) {

			    if (outvel == 1) {
				if (surface == ABSORBING
				    || model == GEOLOGICAL) {
				    OUT.Vxglobal[XMIN - DELTA + i -
						 1][YMIN - DELTA + j - 1] =
					v0.x[i][j][OUT.k0];
				    OUT.Vyglobal[XMIN - DELTA + i -
						 1][YMIN - DELTA + j - 1] =
					v0.y[i][j][OUT.k0];
				    OUT.Vzglobal[XMIN - DELTA + i -
						 1][YMIN - DELTA + j - 1] =
					v0.z[i][j][OUT.k0 - 1];

				} else if (surface == FREE && model != GEOLOGICAL) {	/* free surface at z = 0 */

				    OUT.Vxglobal[XMIN - DELTA + i -
						 1][YMIN - DELTA + j - 1] =
					v0.x[i][j][1];
				    OUT.Vyglobal[XMIN - DELTA + i -
						 1][YMIN - DELTA + j - 1] =
					v0.y[i][j][1];
				    OUT.Vzglobal[XMIN - DELTA + i -
						 1][YMIN - DELTA + j - 1] =
					v0.z[i][j][0];

				}	/* end of if surface */
			    } else if (outdisp == 1) {
				OUT.Vxglobal[XMIN - DELTA + i - 1][YMIN -
								   DELTA +
								   j - 1] =
				    OUT.Uxy[1][i][j];
				OUT.Vyglobal[XMIN - DELTA + i - 1][YMIN -
								   DELTA +
								   j - 1] =
				    OUT.Uxy[2][i][j];
				OUT.Vzglobal[XMIN - DELTA + i - 1][YMIN -
								   DELTA +
								   j - 1] =
				    OUT.Uxy[3][i][j];
			    }
			    /* end if snapType */
			}	/* end of j */
		    }		/* end of i */




/**********************************************************************************/
//      PGV
/**********************************************************************************/
#ifdef PGV
		    if (l == PRM.tMax) {
			for (i = 1; i <= MPMX; i++) {
			    for (j = 1; j <= MPMY; j++) {

				if (outvel == 1) {
				    if (surface == ABSORBING
					|| model == GEOLOGICAL) {
					OUT.PGVglobal[XMIN - DELTA + i -
						      1][YMIN - DELTA + j -
							 1] =
					    OUT.PGVlocal[i][j];
				    } else if (surface == FREE && model != GEOLOGICAL) {	/* free surface at z = 0 */
					OUT.PGVglobal[XMIN - DELTA + i -
						      1][YMIN - DELTA + j -
							 1] =
					    OUT.PGVlocal[i][j];
				    }	/* end of if surface */
				}
			    }	/* end of j */
			}	/* end of i */
		    }
#endif


		}









		/* end of PRM.me = 0 */
		for (i1 = 1; i1 <= 4; i1++) {

///////////////////////////////////////////////////////////
///////////////////:    PRME = !0
/////////////////////////////////////////////////////////////
		    if (PRM.me != 0) {



			if (i1 == 1) {
#ifdef MPI
			    MPI_Isend(PRM.coords, 2, MPI_INT, 0, 90,
				      comm2d, &sendreq[4]);
			    MPI_Wait(&sendreq[4], &status[1]);
#endif				/* MPI */




///////////////////////////////////////////////////////
//      PGV
///////////////////////////////////////////////////////
#ifdef PGV
			    if (l == PRM.tMax) {
				imp = 0;
				for (i = 1; i <= MPMX; i++) {
				    for (j = 1; j <= MPMY; j++) {
					assert(imp < OUT.test_size);
					imp++;
					if (surface == ABSORBING) {
					    OUT.snapBuff[imp] =
						OUT.PGVlocal[i][j];
					} else if (surface == FREE) {
					    OUT.snapBuff[imp] =
						OUT.PGVlocal[i][j];
					}

				    }	//end of j 
				}	//end of i 


#ifdef MPI
				MPI_Isend(OUT.snapBuff, OUT.test_size,
					  MPI_DOUBLE, 0, 83, comm2d,
					  &sendreq[5]);
				MPI_Wait(&sendreq[5], &status[1]);
#endif				/* MPI */


			    }
#endif
///////////////////////////////////////////////////////////     
			}








			/* end of i1 = 1 */
			if (i1 == 2) {
			    IOsurfhelper1(v0.x, 1, outvel, outdisp, PRM,
					  OUT);

#ifdef MPI
			    MPI_Isend(OUT.snapBuff, OUT.test_size,
				      MPI_DOUBLE, 0, 80, comm2d,
				      &sendreq[1]);
			    MPI_Wait(&sendreq[1], &status[1]);
#endif				/* MPI */

			}

			/* end of i1 = 2 */
			if (i1 == 3) {
			    IOsurfhelper1(v0.y, 2, outvel, outdisp, PRM,
					  OUT);


#ifdef MPI
			    MPI_Isend(OUT.snapBuff, OUT.test_size,
				      MPI_DOUBLE, 0, 81, comm2d,
				      &sendreq[2]);
			    MPI_Wait(&sendreq[2], &status[1]);
#endif				/* MPI */
			}

			/* end of i1 = 3 */
			if (i1 == 4) {
			    IOsurfhelper1(v0.z, 3, outvel, outdisp, PRM,
					  OUT);


#ifdef MPI
			    MPI_Isend(OUT.snapBuff, OUT.test_size,
				      MPI_DOUBLE, 0, 82, comm2d,
				      &sendreq[3]);
			    MPI_Wait(&sendreq[3], &status[1]);
#endif				/* MPI */
			}






			/* end of i1 = 4 */
			///////////////////////////////////////////////////////////
///////////////////:    PRME = 0
/////////////////////////////////////////////////////////////

		    } else {	/* PRM.me = 0 */



			for (i2 = 1; i2 < np; i2++) {




			    if (i1 == 1) {

				coords_global[0][i2] = 0;
				coords_global[1][i2] = 0;
#ifdef MPI
				MPI_Recv(proc_coords, 2, MPI_INT, i2, 90,
					 comm2d, &status[1]);
				coords_global[0][i2] = proc_coords[0];
				coords_global[1][i2] = proc_coords[1];
#endif				/* MPI */








///////////////////////////////////////////////////////
//      PGV
///////////////////////////////////////////////////////
#ifdef PGV
				if (l == PRM.tMax) {


#ifdef MPI
				    MPI_Recv(OUT.snapBuff, OUT.test_size,
					     MPI_DOUBLE, i2, 83, comm2d,
					     &status[1]);
#endif


				    OUT.total_prec_x = 0;
				    for (j = 0; j < coords_global[0][i2];
					 j++) {
					OUT.total_prec_x =
					    OUT.total_prec_x +
					    PRM.mpmx_tab[j];
				    }
				    OUT.total_prec_y = 0;
				    for (j = 0; j < coords_global[1][i2];
					 j++) {
					OUT.total_prec_y =
					    OUT.total_prec_y +
					    PRM.mpmy_tab[j];
				    }
				    imp = 0;
				    for (i = 1;
					 i <=
					 PRM.mpmx_tab[coords_global[0]
						      [i2]]; i++) {
					for (j = 1;
					     j <=
					     PRM.mpmy_tab[coords_global[1]
							  [i2]]; j++) {
					    assert(imp < OUT.test_size);
					    assert(XMIN - DELTA - 1 + i +
						   OUT.total_prec_x <
						   XMAX + DELTA + 3);
					    assert(YMIN - DELTA - 1 + j +
						   OUT.total_prec_y <
						   YMAX + DELTA + 3);
					    imp++;
					    OUT.PGVglobal[XMIN - DELTA -
							  1 + i +
							  OUT.total_prec_x]
						[YMIN - DELTA - 1 + j +
						 OUT.total_prec_y] =
						OUT.snapBuff[imp];
					}
				    }

				}
#endif
///////////////////////////////////////////////////////////     

			    }




			    /* end of i1 = 1 */
			    if (i1 == 2) {

#ifdef MPI
				MPI_Recv(OUT.snapBuff, OUT.test_size,
					 MPI_DOUBLE, i2, 80, comm2d,
					 &status[1]);
#endif

				IOsurfloc2glob1(1, coords_global[0][i2],
						coords_global[1][i2], PRM,
						OUT);


			    }
			    /* end of i1 = 2 */
			    if (i1 == 3) {
#ifdef MPI
				MPI_Recv(OUT.snapBuff, OUT.test_size,
					 MPI_DOUBLE, i2, 81, comm2d,
					 &status[1]);
#endif



				IOsurfloc2glob1(2, coords_global[0][i2],
						coords_global[1][i2], PRM,
						OUT);



			    }
			    /* end of i1 = 3 */
			    if (i1 == 4) {
#ifdef MPI
				MPI_Recv(OUT.snapBuff, OUT.test_size,
					 MPI_DOUBLE, i2, 82, comm2d,
					 &status[1]);
#endif


				IOsurfloc2glob1(3, coords_global[0][i2],
						coords_global[1][i2], PRM,
						OUT);


			    }







			}	/* end of if PRM.me */
		    }		/* end of i2 */
		}		/* end of i1 */




///////////////////////////////////////////////////////////////////////////
///     Write surface snapshots
///////////////////////////////////////////////////////////////////////////
		if (PRM.me == 0) {




#ifdef OUTSTD
		    int ndiv = 1;
		    double dssurf = DS * ndiv;

		    int XMINS = (int) ceil((XMIN - DELTA) / ndiv);
		    int XMAXS = (int) floor((XMAX + DELTA) / ndiv);

//      FD + 2
		    int dimx = XMAXS - XMINS + 1;
		    int YMINS = (int) ceil((YMIN - DELTA) / ndiv);
		    int YMAXS = (int) floor((YMAX + DELTA) / ndiv);

//      FD + 2
		    int dimy = YMAXS - YMINS + 1;

		    strcat(flname1, ".vtk");
		    fp1 = fopen(flname1, "w");

		    /* print VTK header */
		    fprintf(fp1, "# vtk DataFile Version 3.0\n");
		    fprintf(fp1, "V\n");
		    fprintf(fp1, "BINARY\n");
		    fprintf(fp1, "DATASET STRUCTURED_POINTS\n");
		    fprintf(fp1, "DIMENSIONS %d %d %d\n", dimx, dimy, 1);
		    fprintf(fp1, "ORIGIN %f %f %f\n", XMINS * dssurf,
			    YMINS * dssurf, 0.);
		    fprintf(fp1, "SPACING %f %f %f\n", dssurf, dssurf,
			    dssurf);
		    fprintf(fp1, "POINT_DATA %d\n", dimx * dimy * 1);
		    fprintf(fp1, "SCALARS V float 3\n");
		    fprintf(fp1, "LOOKUP_TABLE default\n");

//FD
//      fprintf(fp1,"SCALARS VX double 1 \n");
//        fprintf(fp1,"LOOKUP_TABLE default\n");
//      i=(dimy)*(dimx);
//      fwrite(&OUT.Vxglobal[XMIN-DELTA][YMIN-DELTA],sizeof(double),i,fp1);

//      fprintf(fp1,"SCALARS VY double 1 \n");
//        fprintf(fp1,"LOOKUP_TABLE default\n");
//      i=(dimy)*(dimx);
//      fwrite(&OUT.Vyglobal[XMIN-DELTA][YMIN-DELTA],sizeof(double),i,fp1);

//      fprintf(fp1,"SCALARS VZ double 1 \n");
//        fprintf(fp1,"LOOKUP_TABLE default\n");
//      i=(dimy)*(dimx);
//      fwrite(&OUT.Vzglobal[XMIN-DELTA][YMIN-DELTA],sizeof(double),i,fp1);



//FD
		    for (j = YMIN - DELTA + 1; j <= YMAX + DELTA + 1; j++) {
			for (i = XMIN - DELTA + 1; i <= XMAX + DELTA + 1;
			     i++) {
			    if (((i - 1) % ndiv) == 0
				&& ((j - 1) % ndiv) == 0) {

				write_float(fp1,
					    (float) OUT.Vxglobal[i][j]);
				write_float(fp1,
					    (float) OUT.Vyglobal[i][j]);
				write_float(fp1,
					    (float) OUT.Vzglobal[i][j]);
			    }
			}
		    }


		    fclose(fp1);


#ifdef PGV

		    if (l == PRM.tMax) {
			strcat(flname5, "PGV.vtk");
			fp1 = fopen(flname5, "w");

			/* print VTK header */
			fprintf(fp1, "# vtk DataFile Version 3.0\n");
			fprintf(fp1, "V\n");
			fprintf(fp1, "BINARY\n");
			fprintf(fp1, "DATASET STRUCTURED_POINTS\n");
			fprintf(fp1, "DIMENSIONS %d %d %d\n", dimx, dimy,
				1);
			fprintf(fp1, "ORIGIN %f %f %f\n", XMINS * dssurf,
				YMINS * dssurf, 0.);
			fprintf(fp1, "SPACING %f %f %f\n", dssurf, dssurf,
				dssurf);
			fprintf(fp1, "POINT_DATA %d\n", dimx * dimy * 1);
			fprintf(fp1, "SCALARS V float 1\n");
			fprintf(fp1, "LOOKUP_TABLE default\n");

			for (j = YMIN - DELTA + 1; j <= YMAX + DELTA + 1;
			     j++) {
			    for (i = XMIN - DELTA + 1;
				 i <= XMAX + DELTA + 1; i++) {
				if (((i - 1) % ndiv) == 0
				    && ((j - 1) % ndiv) == 0) {
				    write_float(fp1,
						(float)
						OUT.PGVglobal[i][j]);
				}
			    }
			}
			fclose(fp1);
		    }
#endif
#endif





#ifdef OUTBIN
		    fp1 = fopen(flname1, "w");
		    for (i = XMIN - DELTA + 1; i <= XMAX + DELTA + 1; i++) {
			for (j = YMIN - DELTA + 1; j <= YMAX + DELTA + 1;
			     j++) {
//            if ( (((int)(DS*(i-1))) % 10) == 0 && (((int)(DS*(j-1))) % 10) == 0 ){

			    write_float(fp1, (float) OUT.Vxglobal[i][j]);
			    write_float(fp1, (float) OUT.Vyglobal[i][j]);
			    write_float(fp1, (float) OUT.Vzglobal[i][j]);

//              fprintf(fp1, "%7.2f %7.2f %8.3e %8.3e %8.3e \n",
//                      DS*(i-1)/1000., DS*(j-1)/1000.,
//                      OUT.Vxglobal[i][j], OUT.Vyglobal[i][j], OUT.Vzglobal[i][j] );
//            }
			}
		    }
		    fclose(fp1);


#ifdef PGV

		    if (l == PRM.tMax) {

			strcat(flname5, "PGV.bin");
			fp1 = fopen(flname5, "w");
			for (i = XMIN - DELTA + 1; i <= XMAX + DELTA + 1;
			     i++) {
			    for (j = YMIN - DELTA + 1;
				 j <= YMAX + DELTA + 1; j++) {

//            if ( (((int)(DS*(i-1))) % 10) == 0 && (((int)(DS*(j-1))) % 10) == 0 )

//      {

				write_float(fp1,
					    (float) OUT.PGVglobal[i][j]);
//               fprintf(fp1, "%7.2f %7.2f %8.3e \n",DS*(i-1)/1000., DS*(j-1)/1000.,OUT.PGVglobal[i][j]);


//       }

			    }
			}
			fclose(fp1);
		    }
#endif
#endif




#if defined (OUTNETCDF) || defined (OUTNEMESIS)
		    strcat(flname1, ".nc");
		    fStatus =
			OutSurfNetcdf(flname1, DELTA, XMIN, XMAX, YMIN,
				      YMAX, ZMIN, ZMAX, DS, OUT.Vxglobal,
				      OUT.Vyglobal, OUT.Vzglobal);
#ifdef PGV
		    if (l == PRM.tMax) {
			strcat(flname5, "PGV.nc");
			fStatus =
			    OutSurfNetcdf(flname5, DELTA, XMIN, XMAX, YMIN,
					  YMAX, ZMIN, ZMAX, DS,
					  OUT.PGVglobal, OUT.PGVglobal,
					  OUT.PGVglobal);
		    }
#endif

#endif


		    /* Desallocation */
		    free_dmatrix(OUT.Vxglobal, XMIN - DELTA,
				 XMAX + DELTA + 2, YMIN - DELTA,
				 YMAX + DELTA + 2);
		    free_dmatrix(OUT.Vyglobal, XMIN - DELTA,
				 XMAX + DELTA + 2, YMIN - DELTA,
				 YMAX + DELTA + 2);
		    free_dmatrix(OUT.Vzglobal, XMIN - DELTA,
				 XMAX + DELTA + 2, YMIN - DELTA,
				 YMAX + DELTA + 2);
		}





		/* end of if PRM.me */
				/**********************************************************************/
		// Surfacexz
// Recherche du processeur avec j0 et coords[0] = 0
// //On connait ses coordonnées (0,jcpu)
//           donc rang = jcpu*px */
/***********************************************************************/

#ifndef OUTNEMESIS
		jcpu = PRM.j2jcpu_array[OUT.j0];
		jmp_tmp = PRM.j2jmp_array[OUT.j0];

		if (PRM.me == jcpu * PRM.px) {
		    OUT.Vxglobal =
			dmatrix(XMIN - DELTA, XMAX + DELTA + 2,
				ZMIN - DELTA, ZMAX0);
		    OUT.Vyglobal =
			dmatrix(XMIN - DELTA, XMAX + DELTA + 2,
				ZMIN - DELTA, ZMAX0);
		    OUT.Vzglobal =
			dmatrix(XMIN - DELTA, XMAX + DELTA + 2,
				ZMIN - DELTA, ZMAX0);

		    for (i = 1; i <= MPMX; i++) {
			for (k = ZMIN - DELTA; k <= ZMAX0; k++) {
			    if (outvel == 1) {
				OUT.Vxglobal[XMIN - DELTA + i - 1][k] =
				    v0.x[i][jmp_tmp][k];
				OUT.Vyglobal[XMIN - DELTA + i - 1][k] =
				    v0.y[i][jmp_tmp][k];
				OUT.Vzglobal[XMIN - DELTA + i - 1][k] =
				    v0.z[i][jmp_tmp][k];
			    } else if (outdisp == 1) {
				OUT.Vxglobal[XMIN - DELTA + i - 1][k] =
				    OUT.Uxz[1][i][k];
				OUT.Vyglobal[XMIN - DELTA + i - 1][k] =
				    OUT.Uxz[2][i][k];
				OUT.Vzglobal[XMIN - DELTA + i - 1][k] =
				    OUT.Uxz[3][i][k];
			    }
			}
		    }
		}


		for (i1 = 1; i1 <= 4; i1++) {

		    if ((PRM.coords[1] == jcpu)
			&& (PRM.me != PRM.px * jcpu)) {

			if (i1 == 1) {
#ifdef MPI
			    MPI_Isend(PRM.coords, 2, MPI_INT,
				      jcpu * PRM.px, 90, comm2d,
				      &sendreq[4]);
			    MPI_Wait(&sendreq[4], &status[1]);
#endif

			}
			/* end of i1 = 1 */
			if (i1 == 2) {

			    IOsurfhelper2(v0.x, 1, outvel, outdisp, PRM,
					  OUT);



#ifdef MPI
			    MPI_Isend(OUT.snapBuff, OUT.test_size,
				      MPI_DOUBLE, jcpu * PRM.px, 80,
				      comm2d, &sendreq[1]);
			    MPI_Wait(&sendreq[1], &status[1]);
#endif

			}
			/* end of i1 = 2 */
			if (i1 == 3) {

			    IOsurfhelper2(v0.y, 2, outvel, outdisp, PRM,
					  OUT);

#ifdef MPI
			    MPI_Isend(OUT.snapBuff, OUT.test_size,
				      MPI_DOUBLE, jcpu * PRM.px, 81,
				      comm2d, &sendreq[2]);
			    MPI_Wait(&sendreq[2], &status[1]);
#endif

			}
			/* end of i1 = 3 */
			if (i1 == 4) {

			    IOsurfhelper2(v0.z, 3, outvel, outdisp, PRM,
					  OUT);


#ifdef MPI
			    MPI_Isend(OUT.snapBuff, OUT.test_size,
				      MPI_DOUBLE, jcpu * PRM.px, 82,
				      comm2d, &sendreq[3]);
			    MPI_Wait(&sendreq[3], &status[1]);
#endif

			}
			/* end of i1 = 4 */
		    }
		    /* end of if jcpu && PRM.me */
		    if (PRM.me == jcpu * PRM.px) {

			for (i2 = 1; i2 < PRM.px; i2++) {

			    if (i1 == 1) {

#ifdef MPI
				MPI_Recv(proc_coords, 2, MPI_INT,
					 jcpu * PRM.px + i2, 90, comm2d,
					 &status[1]);
#endif
				coords_global[0][i2] = proc_coords[0];
				coords_global[1][i2] = proc_coords[1];

			    }
			    /* end of i1 = 1 */
			    if (i1 == 2) {

#ifdef MPI
				MPI_Recv(OUT.snapBuff, OUT.test_size,
					 MPI_DOUBLE, jcpu * PRM.px + i2,
					 80, comm2d, &status[1]);
#endif
				IOsurfloc2glob2(1, coords_global[0][i2],
						coords_global[1][i2], PRM,
						OUT);

			    }
			    /* end of i1 = 2 */
			    if (i1 == 3) {
#ifdef MPI
				MPI_Recv(OUT.snapBuff, OUT.test_size,
					 MPI_DOUBLE, jcpu * PRM.px + i2,
					 81, comm2d, &status[1]);
#endif


				IOsurfloc2glob2(2, coords_global[0][i2],
						coords_global[1][i2], PRM,
						OUT);


			    }
			    /* end of i1 = 3 */
			    if (i1 == 4) {
#ifdef MPI
				MPI_Recv(OUT.snapBuff, OUT.test_size,
					 MPI_DOUBLE, jcpu * PRM.px + i2,
					 82, comm2d, &status[1]);
#endif
				IOsurfloc2glob2(3, coords_global[0][i2],
						coords_global[1][i2], PRM,
						OUT);


			    }
			    /* end of i1 = 4 */
			}	/* end of i2 */
		    }		/* end of PRM.me = jcpu*PRM.px */
		}		/* end of i1 */

		if (PRM.me == jcpu * PRM.px) {

#ifdef OUTSTD
		    int ndiv = 1;
		    double dssurf = DS * ndiv;

		    int XMINS = (int) ceil((XMIN - DELTA) / ndiv);
		    int XMAXS = (int) floor((XMAX + DELTA) / ndiv);
		    int dimx = XMAXS - XMINS + 1;
		    int ZMINS = (int) ceil((ZMIN - DELTA) / ndiv);
		    int ZMAXS = (int) floor((1) / ndiv);
		    int dimz = ZMAXS - ZMINS + 1;

		    strcat(flname2, ".vtk");
		    fp2 = fopen(flname2, "w");

		    /* print VTK header */
		    fprintf(fp2, "# vtk DataFile Version 3.0\n");
		    fprintf(fp2, "V\n");
		    fprintf(fp2, "BINARY\n");
		    fprintf(fp2, "DATASET STRUCTURED_POINTS\n");
		    fprintf(fp2, "DIMENSIONS %d %d %d\n", dimx, 1, dimz);
		    fprintf(fp2, "ORIGIN %f %f %f\n", XMINS * dssurf,
			    OUT.j0 * DS, ZMINS * dssurf);
		    fprintf(fp2, "SPACING %f %f %f\n", dssurf, dssurf,
			    dssurf);
		    fprintf(fp2, "POINT_DATA %d\n", dimx * dimz * 1);
		    fprintf(fp2, "SCALARS V float 3\n");
		    fprintf(fp2, "LOOKUP_TABLE default\n");

		    for (k = ZMIN - DELTA + 1; k <= 2; k++) {
			for (i = XMIN - DELTA + 1; i <= XMAX + DELTA + 1;
			     i++) {
			    if (((i - 1) % ndiv) == 0
				&& ((k - 1) % ndiv) == 0) {
				write_float(fp2,
					    (float) OUT.Vxglobal[i][k]);
				write_float(fp2,
					    (float) OUT.Vyglobal[i][k]);
				write_float(fp2,
					    (float) OUT.Vzglobal[i][k]);
			    }
			}
		    }
		    fclose(fp2);
#endif

#ifdef OUTBIN

		    fp2 = fopen(flname2, "w");
		    for (i = XMIN - DELTA + 1; i <= XMAX + DELTA + 1; i++) {
			for (k = ZMIN - DELTA + 1; k <= ZMAX0 - 1; k++) {
			    if ((((int) (DS * (i - 1))) % 10) == 0
				&& (((int) (DS * (k - 1))) % 10) == 0) {
				fprintf(fp2,
					"%7.2f %7.2f %8.3e %8.3e %8.3e \n",
					DS * (i - 1) / 1000.,
					DS * (k - 1) / 1000.,
					OUT.Vxglobal[i][k],
					OUT.Vyglobal[i][k],
					OUT.Vzglobal[i][k]);
			    }
			}
		    }
		    fclose(fp2);
#endif
		    /* Desallocation */
		    free_dmatrix(OUT.Vxglobal, XMIN - DELTA,
				 XMAX + DELTA + 2, ZMIN - DELTA, ZMAX0);
		    free_dmatrix(OUT.Vyglobal, XMIN - DELTA,
				 XMAX + DELTA + 2, ZMIN - DELTA, ZMAX0);
		    free_dmatrix(OUT.Vzglobal, XMIN - DELTA,
				 XMAX + DELTA + 2, ZMIN - DELTA, ZMAX0);
		}




				  /**********************************************************************/
		//      BUG /////////////
// Surfaceyz
// Recherche du processeur avec i0 et coords[1] = 0
//         On connait ses coordonnées (icpu,0)
//         donc rang = icpu */
//***********************************************************************/



		icpu = PRM.i2icpu_array[OUT.i0];
		imp_tmp = PRM.i2imp_array[OUT.i0];

		if (PRM.me == icpu) {

		    OUT.Vxglobal =
			dmatrix(YMIN - DELTA, YMAX + DELTA + 2,
				ZMIN - DELTA, ZMAX0);
		    OUT.Vyglobal =
			dmatrix(YMIN - DELTA, YMAX + DELTA + 2,
				ZMIN - DELTA, ZMAX0);
		    OUT.Vzglobal =
			dmatrix(YMIN - DELTA, YMAX + DELTA + 2,
				ZMIN - DELTA, ZMAX0);

		    for (j = 1; j <= MPMY; j++) {
			for (k = ZMIN - DELTA; k <= ZMAX0; k++) {
			    if (outvel == 1) {
				OUT.Vxglobal[YMIN - DELTA + j - 1][k] =
				    v0.x[imp_tmp][j][k];
				OUT.Vyglobal[YMIN - DELTA + j - 1][k] =
				    v0.y[imp_tmp][j][k];
				OUT.Vzglobal[YMIN - DELTA + j - 1][k] =
				    v0.z[imp_tmp][j][k];
			    } else if (outdisp == 1) {
				OUT.Vxglobal[YMIN - DELTA + j - 1][k] =
				    OUT.Uyz[1][j][k];
				OUT.Vyglobal[YMIN - DELTA + j - 1][k] =
				    OUT.Uyz[2][j][k];
				OUT.Vzglobal[YMIN - DELTA + j - 1][k] =
				    OUT.Uyz[3][j][k];
			    }

			}
		    }
		}

		/* Etape1 */

		for (i1 = 1; i1 <= 4; i1++) {

		    if ((PRM.coords[0] == icpu) && (PRM.me != icpu)) {

			if (i1 == 1) {
#ifdef MPI
			    MPI_Isend(PRM.coords, 2, MPI_INT, icpu, 90,
				      comm2d, &sendreq[4]);
			    MPI_Wait(&sendreq[4], &status[1]);
#endif

			}
			/* end of i1 = 1 */
			if (i1 == 2) {

			    IOsurfhelper2(v0.x, 1, outvel, outdisp, PRM,
					  OUT);


#ifdef MPI
			    MPI_Isend(OUT.snapBuff, OUT.test_size,
				      MPI_DOUBLE, icpu, 80, comm2d,
				      &sendreq[1]);
			    MPI_Wait(&sendreq[1], &status[1]);
#endif

			}
			/* end of i1 = 2 */
			if (i1 == 3) {
			    IOsurfhelper2(v0.y, 2, outvel, outdisp, PRM,
					  OUT);

#ifdef MPI
			    MPI_Isend(OUT.snapBuff, OUT.test_size,
				      MPI_DOUBLE, icpu, 81, comm2d,
				      &sendreq[2]);
			    MPI_Wait(&sendreq[2], &status[1]);
#endif

			}
			/* end of i1 = 3 */
			if (i1 == 4) {


			    IOsurfhelper2(v0.z, 3, outvel, outdisp, PRM,
					  OUT);


#ifdef MPI
			    MPI_Isend(OUT.snapBuff, OUT.test_size,
				      MPI_DOUBLE, icpu, 82, comm2d,
				      &sendreq[3]);
			    MPI_Wait(&sendreq[3], &status[1]);
#endif
			}
			/* end of i1 = 4 */
		    }
		    /* end of if icpu && PRM.me */
		    if (PRM.me == icpu) {

			for (i2 = 1; i2 < PRM.py; i2++) {

			    if (i1 == 1) {


#ifdef MPI
				MPI_Recv(proc_coords, 2, MPI_INT,
					 icpu + PRM.px * i2, 90, comm2d,
					 &status[1]);
#endif
				coords_global[0][i2] = proc_coords[0];
				coords_global[1][i2] = proc_coords[1];


			    }
			    /* end of i1 = 1 */
			    if (i1 == 2) {

#ifdef MPI
				MPI_Recv(OUT.snapBuff, OUT.test_size,
					 MPI_DOUBLE, icpu + PRM.px * i2,
					 80, comm2d, &status[1]);
#endif
				IOsurfloc2glob3(1, coords_global[0][i2],
						coords_global[1][i2], PRM,
						OUT);

			    }
			    /* end of i1 = 2 */
			    if (i1 == 3) {

#ifdef MPI
				MPI_Recv(OUT.snapBuff, OUT.test_size,
					 MPI_DOUBLE, icpu + PRM.px * i2,
					 81, comm2d, &status[1]);
#endif
				IOsurfloc2glob3(2, coords_global[0][i2],
						coords_global[1][i2], PRM,
						OUT);

			    }
			    /* end of i1 = 3 */
			    if (i1 == 4) {

#ifdef MPI
				MPI_Recv(OUT.snapBuff, OUT.test_size,
					 MPI_DOUBLE, icpu + PRM.px * i2,
					 82, comm2d, &status[1]);
#endif
				IOsurfloc2glob3(3, coords_global[0][i2],
						coords_global[1][i2], PRM,
						OUT);


			    }
			    /* end of i1 = 4 */
			}	/* end of i2 */
		    }		/* end of if PRM.me */
		}		/* end of i1 */
		/* Ecriture */

		if (PRM.me == icpu) {
#ifdef OUTSTD
		    int ndiv = 1;
		    double dssurf = DS * ndiv;

		    int YMINS = (int) ceil((YMIN - DELTA) / ndiv);
		    int YMAXS = (int) floor((YMAX + DELTA) / ndiv);
		    int dimy = YMAXS - YMINS + 1;
		    int ZMINS = (int) ceil((ZMIN - DELTA) / ndiv);
		    int ZMAXS = (int) floor((1) / ndiv);
		    int dimz = ZMAXS - ZMINS + 1;

		    strcat(flname3, ".vtk");
		    fp3 = fopen(flname3, "w");

		    /* print VTK header */
		    fprintf(fp3, "# vtk DataFile Version 3.0\n");
		    fprintf(fp3, "V\n");
		    fprintf(fp3, "BINARY\n");
		    fprintf(fp3, "DATASET STRUCTURED_POINTS\n");
		    fprintf(fp3, "DIMENSIONS %d %d %d\n", 1, dimy, dimz);
		    fprintf(fp3, "ORIGIN %f %f %f\n", OUT.i0 * DS,
			    YMINS * dssurf, ZMINS * dssurf);
		    fprintf(fp3, "SPACING %f %f %f\n", dssurf, dssurf,
			    dssurf);
		    fprintf(fp3, "POINT_DATA %d\n", dimy * dimz * 1);
		    fprintf(fp3, "SCALARS V float 3\n");
		    fprintf(fp3, "LOOKUP_TABLE default\n");

		    for (k = ZMIN - DELTA + 1; k <= 2; k++) {
			for (j = YMIN - DELTA + 1; j <= YMAX + DELTA + 1;
			     j++) {
			    if (((j - 1) % ndiv) == 0
				&& ((k - 1) % ndiv) == 0) {
				write_float(fp3,
					    (float) OUT.Vxglobal[j][k]);
				write_float(fp3,
					    (float) OUT.Vyglobal[j][k]);
				write_float(fp3,
					    (float) OUT.Vzglobal[j][k]);
			    }
			}
		    }
		    fclose(fp3);
#endif

#ifdef OUTBIN

		    fp3 = fopen(flname3, "w");
		    for (j = YMIN - DELTA + 1; j <= YMAX + DELTA + 1; j++) {
			for (k = ZMIN - DELTA + 1; k <= ZMAX0 - 1; k++) {
			    if ((((int) (DS * (j - 1))) % 10) == 0
				&& (((int) (DS * (k - 1))) % 10) == 0) {
				fprintf(fp3,
					"%7.2f %7.2f %8.3e %8.3e %8.3e \n",
					DS * (j - 1) / 1000.,
					DS * (k - 1) / 1000.,
					OUT.Vxglobal[j][k],
					OUT.Vyglobal[j][k],
					OUT.Vzglobal[j][k]);
			    }
			}
		    }
		    fclose(fp3);
#endif
		    /* Desallocation */

		    free_dmatrix(OUT.Vxglobal, YMIN - DELTA,
				 YMAX + DELTA + 2, ZMIN - DELTA, ZMAX0);
		    free_dmatrix(OUT.Vyglobal, YMIN - DELTA,
				 YMAX + DELTA + 2, ZMIN - DELTA, ZMAX0);
		    free_dmatrix(OUT.Vzglobal, YMIN - DELTA,
				 YMAX + DELTA + 2, ZMIN - DELTA, ZMAX0);
		}
#endif

	    }			/* end for snapshots steps */
	}



	/* end write surfaceij outputs (l% nn =0) */
#if ( VERBOSE > 0 )
	if ((l % SURFACE_STEP) == 0 && PRM.me == 0) {
	    printf("\nEnd time %d\n", l);
	}
#endif
#ifdef LUCAS_INSTRUMENTATION
	SCOREP_USER_REGION_END(iteration)
#endif
    }				/* end time loop */


//******************************************************************************/
    /* ==================== END OF TIME LOOP ========================= */
//******************************************************************************/







#ifdef MPI
#if (COMM==1)			/* close communications - deallocate requests */
    if (NORTH.rank != -1) {
	MPI_Request_free(&reqT0S[1]);
	MPI_Request_free(&reqT0R[2]);
	MPI_Request_free(&reqV0S[1]);
	MPI_Request_free(&reqV0R[2]);
	if (ANLmethod == KRISTEKandMOCZO) {
	    MPI_Request_free(&reqKsiS[1]);
	    MPI_Request_free(&reqKsiR[2]);
	}
    }
    if (SOUTH.rank != -1) {
	MPI_Request_free(&reqT0R[1]);
	MPI_Request_free(&reqT0S[2]);
	MPI_Request_free(&reqV0R[1]);
	MPI_Request_free(&reqV0S[2]);
	if (ANLmethod == KRISTEKandMOCZO) {
	    MPI_Request_free(&reqKsiR[1]);
	    MPI_Request_free(&reqKsiS[2]);
	}
    }

    if (EAST.rank != -1) {
	MPI_Request_free(&reqT0S[3]);
	MPI_Request_free(&reqT0R[4]);
	MPI_Request_free(&reqV0S[3]);
	MPI_Request_free(&reqV0R[4]);
	if (ANLmethod == KRISTEKandMOCZO) {
	    MPI_Request_free(&reqKsiS[3]);
	    MPI_Request_free(&reqKsiR[4]);
	}
    }
    if (WEST.rank != -1) {
	MPI_Request_free(&reqT0R[3]);
	MPI_Request_free(&reqT0S[4]);
	MPI_Request_free(&reqV0R[3]);
	MPI_Request_free(&reqV0S[4]);
	if (ANLmethod == KRISTEKandMOCZO) {
	    MPI_Request_free(&reqKsiR[3]);
	    MPI_Request_free(&reqKsiS[4]);
	}
    }
#endif
#endif

#ifdef FLOPS
    if ((retval =
	 PAPI_flops(&real_time, &proc_time, &flpops, &mflops)) < PAPI_OK) {
	printf("retval: %d\n", retval);
	exit(EXIT_FAILURE);
    }
    printf("Mflops %f \n", mflops);
#endif

#ifdef MISS
    if (retval = PAPI_stop(EventSet, values) != PAPI_OK)
	printf("ERROR stop \n");
    perc = (float) 100.0 *values[1] / values[0];
    printf("Cache Miss %f %d \n", perc, PRM.me);
    printf("Cycle %lld %d \n", values[2], PRM.me);
    printf("L3 MISS  %lld %d \n", values[1], PRM.me);
    printf("L3 acces  %lld %d \n", values[0], PRM.me);
#endif

    timing4 = my_second();
    timing_total = (timing4 - timing3);

#ifdef MPI
    MPI_Reduce(&timing_bc1, &timing_bc1_max, 1, MPI_DOUBLE, MPI_MAX, 0,
	       MPI_COMM_WORLD);
    MPI_Reduce(&timing_bc1, &timing_bc1_min, 1, MPI_DOUBLE, MPI_MIN, 0,
	       MPI_COMM_WORLD);
    MPI_Reduce(&timing_bc1, &timing_sum1, 1, MPI_DOUBLE, MPI_SUM, 0,
	       MPI_COMM_WORLD);
    MPI_Reduce(&timing_bc2, &timing_sum2, 1, MPI_DOUBLE, MPI_SUM, 0,
	       MPI_COMM_WORLD);
    MPI_Reduce(&timing_bc2, &timing_bc2_max, 1, MPI_DOUBLE, MPI_MAX, 0,
	       MPI_COMM_WORLD);
    MPI_Reduce(&timing_bc2, &timing_bc2_min, 1, MPI_DOUBLE, MPI_MIN, 0,
	       MPI_COMM_WORLD);
    MPI_Reduce(&timing_total, &timing4, 1, MPI_DOUBLE, MPI_MAX, 0,
	       MPI_COMM_WORLD);
#else
    timing_bc1_max = timing_bc1;
    timing_bc1_min = timing_bc1;
    timing_bc2_max = timing_bc2;
    timing_bc2_min = timing_bc2;
    timing4 = timing_total;
#endif



    timing_bc_total = timing_bc1 + timing_bc2;
    timing_comm_total = timing_comm1 + timing_comm2;

    /* On prend le proc avec le plus gros ecart au niveau de bc */
#ifdef MPI
    MPI_Reduce(&timing_bc_total, &timing_bc_max, 1, MPI_DOUBLE, MPI_MAX, 0,
	       MPI_COMM_WORLD);
    MPI_Reduce(&timing_bc_total, &timing_bc_min, 1, MPI_DOUBLE, MPI_MIN, 0,
	       MPI_COMM_WORLD);
    MPI_Reduce(&timing_comm_total, &timing_comm_max, 1, MPI_DOUBLE,
	       MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timing_comm_total, &timing_comm_min, 1, MPI_DOUBLE,
	       MPI_MIN, 0, MPI_COMM_WORLD);
#else
    timing_bc_max = timing_bc_total;
    timing_bc_min = timing_bc_total;
    timing_comm_max = timing_comm_total;
    timing_comm_min = timing_comm_total;
#endif				/*  MPI */



//   printf("Taille domaine x/y - num processeur %d %d %d \n",MPMX,MPMY,PRM.me);

    if (PRM.me == 0) {

	printf("%d %d %d \n", np, PRM.px, PRM.py);
	printf("Timing total %f \n", timing_total);
	printf("Timing compute max  %f \n", timing_bc_max);
	printf("Timing compute min  %f \n", timing_bc_min);
	printf("Timing comm max - attente + comm reelle  %f \n",
	       timing_comm_max);
	printf("Timing comm min - comm relle  %f \n", timing_comm_min);
	printf("Part communication: %f \n",
	       100 * (timing_comm_min) / timing_total);
	printf("Part compute: %f \n",
	       100 * (timing_bc_min) / timing_total);
	printf("Ecart BC vs timing total %f \n",
	       100 * ((timing_bc_max - timing_bc_min) / timing_total));
	printf("Ecart BC %f \n",
	       100 * ((timing_bc_max - timing_bc_min) / timing_bc_max));
	printf("Ecart Min/Max BC1 %f \n",
	       100 * ((timing_bc1_max - timing_bc1_min) / timing_bc1_max));
	printf("Ecart Min/Max BC2 %f \n",
	       100 * ((timing_bc2_max - timing_bc2_min) / timing_bc2_max));
	printf("timing max - BC1  %f \n ", timing_bc1_max);
	printf("timing min - BC1  %f \n ", timing_bc1_min);
	printf("timing max - BC2  %f \n ", timing_bc2_max);
	printf("timing min - BC2  %f \n ", timing_bc2_min);

	fp5 = fopen(BENCHFILE, "w");
	fprintf(fp5, "%d %d %d \n", np, PRM.px, PRM.py);
	fprintf(fp5, "%f \n", timing_total);
	fprintf(fp5, "%f \n", timing_bc_max);
	fprintf(fp5, "%f \n", timing_bc_min);
	fprintf(fp5, "%f \n", timing_comm_max);
	fprintf(fp5, "%f \n", timing_comm_min);
	fprintf(fp5, "%f \n", 100 * (timing_comm_min) / timing_total);
	fprintf(fp5, "%f \n", 100 * (timing_bc_min) / timing_total);
	fprintf(fp5, "%f \n",
		100 * ((timing_bc_max - timing_bc_min) / timing_total));
	fprintf(fp5, "%f \n",
		100 * ((timing_bc_max - timing_bc_min) / timing_bc_max));
	fprintf(fp5, "%f \n",
		100 * ((timing_bc1_max - timing_bc1_min) /
		       timing_bc1_max));
	fprintf(fp5, "%f \n",
		100 * ((timing_bc2_max - timing_bc2_min) /
		       timing_bc2_max));

	fclose(fp5);


#ifdef ADIOS

	fp5 = fopen(ADIOSFILE, "w");
	fprintf(fp5, "%d  \n", np);
#ifdef OMP
	fprintf(fp5, "%d  \n", numThreads_Level_1);
#endif
#ifndef OMP
	fprintf(fp5, "%d  \n", 1);
#endif
	fprintf(fp5, "%s \n", transp_method);
	fprintf(fp5, "%d  \n", a_buffer_rate);
	fprintf(fp5, "%f \n", timing_total);
	fprintf(fp5, "%f \n", timing_bc_max);
	fclose(fp5);
#endif

	/* Fin cout max comm */

    }
    /* end PRM.me */
#ifdef MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    printf("------------- FIN -------------- %d \n", PRM.me);
#ifdef MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif


#if (TIMER==2)
#ifdef MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    timing4 = my_second();
    timing_total = (timing4 - timing3);
#endif

#if (TIMER==1)
    timing4 = my_second();
    timing_total = (timing4 - timing3);
#endif

#ifdef MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif


    fStatus = DeallocateAll(STATION_STEP,
			    &ANL, &ABC, &SRC, &MDM,
			    &t0, &v0, &OUT,
			    &NORTH, &SOUTH, &EAST, &WEST, &PRM);
    VerifFunction(fStatus, "desallocate all", PRM);

#ifdef MPI
#ifdef ADIOS
#ifdef ADIOSNOLOOP
    if (a_comm2d != MPI_COMM_NULL)
	adios_close(adios_handle1);
    if (a_comm2d != MPI_COMM_NULL)
	adios_finalize(a_me);
#endif


#if defined (ADIOSLOOP) || defined (ADIOSNOLOOP)
    if (a_comm2d != MPI_COMM_NULL)
	adios_finalize(a_me);
#endif


#endif


    MPI_Finalize();
#endif
    return (EXIT_SUCCESS);

}				/* end of programm */

/* =============== */
/* SMALL FUNCTIONS */
/* =============== */

int VerifFunction(int exitStatus, const char *msg, struct PARAMETERS PRM)
{
    if (exitStatus != EXIT_SUCCESS) {
	if (PRM.me == 0) {
	    fprintf(stderr, "%-50s [ERROR] with cpu %i \n", msg, PRM.me);
	}
	exit(EXIT_FAILURE);
    } else {
#if (VERBOSE > 0)
	if (PRM.me == 0) {
	    fprintf(stderr, "%-50s [DONE ]\n", msg);
	}
#endif
	return EXIT_SUCCESS;
    }
}


double my_second()
{
    struct timeval tp;
    struct timezone tzp;
    int i;

    i = gettimeofday(&tp, &tzp);
    return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}


/* ========== */
/* COMMUNICATIONS RELATED */
/* ========== */

int SyncBufStress(struct STRESS *t0, int mode,	/*  0: send, 1: receive */
		  struct COMM_DIRECTION *DIR, struct PARAMETERS PRM)
{
    long int i;			/* index for assert Verification */
    int imp, jmp, k;		/* local coordinates */
    assert(mode == 0 || mode == 1);
    if (mode == 0) {		/* send */
	if (DIR->rank == -1) {
	    return CPU_NO_SEND;
	} else {
	    i = 0;
	    for (imp = DIR->iMinS; imp <= DIR->iMaxS; imp++) {
		for (jmp = DIR->jMinS; jmp <= DIR->jMaxS; jmp++) {
		    for (k = PRM.zMin - PRM.delta; k <= PRM.zMax0; k++) {
			assert(i <= DIR->nmax);
			DIR->bufT0S[6 * i] = t0->xx[imp][jmp][k];
			DIR->bufT0S[6 * i + 1] = t0->yy[imp][jmp][k];
			DIR->bufT0S[6 * i + 2] = t0->zz[imp][jmp][k];
			DIR->bufT0S[6 * i + 3] = t0->xy[imp][jmp][k];
			DIR->bufT0S[6 * i + 4] = t0->xz[imp][jmp][k];
			DIR->bufT0S[6 * i + 5] = t0->yz[imp][jmp][k];

			i = i + 1;
		    }
		}
	    }

	}			/* end if CPU */
    } else if (mode == 1) {	/* Receive */
	if (DIR->rank == -1) {
	    return CPU_NO_RECV;
	} else {
	    i = 0;
	    for (imp = DIR->iMinR; imp <= DIR->iMaxR; imp++) {
		for (jmp = DIR->jMinR; jmp <= DIR->jMaxR; jmp++) {
		    for (k = PRM.zMin - PRM.delta; k <= PRM.zMax0; k++) {
			assert(i <= DIR->nmax);
			t0->xx[imp][jmp][k] = DIR->bufT0R[6 * i];
			t0->yy[imp][jmp][k] = DIR->bufT0R[6 * i + 1];
			t0->zz[imp][jmp][k] = DIR->bufT0R[6 * i + 2];
			t0->xy[imp][jmp][k] = DIR->bufT0R[6 * i + 3];
			t0->xz[imp][jmp][k] = DIR->bufT0R[6 * i + 4];
			t0->yz[imp][jmp][k] = DIR->bufT0R[6 * i + 5];
			i = i + 1;
		    }
		}
	    }
	}			/* end if CPU */
    }
    /* end mode */
    return EXIT_SUCCESS;

}				/* end function */

int SyncBufVelocity(struct VELOCITY *v0, int mode,	/*  0: send, 1: receive */
		    struct COMM_DIRECTION *DIR, struct PARAMETERS PRM)
{
    long int i;			/* index for assert Verification */
    int imp, jmp, k;		/* local coordinates */

    assert(mode == 0 || mode == 1);


    if (mode == 0) {		/* send */
	if (DIR->rank == -1) {
	    return CPU_NO_SEND;
	} else {
	    i = 0;
	    for (imp = DIR->iMinS; imp <= DIR->iMaxS; imp++) {
		for (jmp = DIR->jMinS; jmp <= DIR->jMaxS; jmp++) {
		    for (k = PRM.zMin - PRM.delta; k <= PRM.zMax0; k++) {
			assert(i <= DIR->nmax);
			DIR->bufV0S[3 * i] = v0->x[imp][jmp][k];
			DIR->bufV0S[3 * i + 1] = v0->y[imp][jmp][k];
			DIR->bufV0S[3 * i + 2] = v0->z[imp][jmp][k];

			i = i + 1;
		    }
		}
	    }
	}			/* end if CPU */
    } else if (mode == 1) {	/* Receive */
	if (DIR->rank == -1) {
	    return CPU_NO_RECV;
	} else {
	    i = 0;
	    for (imp = DIR->iMinR; imp <= DIR->iMaxR; imp++) {
		for (jmp = DIR->jMinR; jmp <= DIR->jMaxR; jmp++) {
		    for (k = PRM.zMin - PRM.delta; k <= PRM.zMax0; k++) {
			assert(i <= DIR->nmax);
			v0->x[imp][jmp][k] = DIR->bufV0R[3 * i];
			v0->y[imp][jmp][k] = DIR->bufV0R[3 * i + 1];
			v0->z[imp][jmp][k] = DIR->bufV0R[3 * i + 2];

			i = i + 1;
		    }
		}
	    }

	}			/* end if CPU */
    }
    /* end mode */
    return EXIT_SUCCESS;
}				/* end function  */

int SyncBufKsil(struct ANELASTICITY *ANL, int mode,	/*  0: send, 1: receive */
		struct COMM_DIRECTION *DIR, struct PARAMETERS PRM)
{
    long int i;			/* index for assert Verification */
    int imp, jmp, k;		/* local coordinates */

    assert(mode == 0 || mode == 1);

    i = 0;
    if (mode == 0) {		/* send */
	if (DIR->rank == -1) {
	    return CPU_NO_SEND;
	} else {
	    for (imp = DIR->iMinS; imp <= DIR->iMaxS; imp++) {
		for (jmp = DIR->jMinS; jmp <= DIR->jMaxS; jmp++) {
		    for (k = PRM.zMin - PRM.delta; k <= PRM.zMax0; k++) {
			assert(i <= DIR->nmax);
			DIR->bufKsiS[6 * i] = ANL->ksilxx[imp][jmp][k];
			DIR->bufKsiS[6 * i + 1] = ANL->ksilyy[imp][jmp][k];
			DIR->bufKsiS[6 * i + 2] = ANL->ksilzz[imp][jmp][k];
			DIR->bufKsiS[6 * i + 3] = ANL->ksilxy[imp][jmp][k];
			DIR->bufKsiS[6 * i + 4] = ANL->ksilxz[imp][jmp][k];
			DIR->bufKsiS[6 * i + 5] = ANL->ksilyz[imp][jmp][k];

			i = i + 1;
		    }
		}
	    }
	}			/* end if CPU */
    } else if (mode == 1) {	/* Receive */
	if (DIR->rank == -1) {
	    return CPU_NO_RECV;
	} else {
	    for (imp = DIR->iMinR; imp <= DIR->iMaxR; imp++) {
		for (jmp = DIR->jMinR; jmp <= DIR->jMaxR; jmp++) {
		    for (k = PRM.zMin - PRM.delta; k <= PRM.zMax0; k++) {
			assert(i <= DIR->nmax);
			ANL->ksilxx[imp][jmp][k] = DIR->bufKsiR[6 * i];
			ANL->ksilyy[imp][jmp][k] = DIR->bufKsiR[6 * i + 1];
			ANL->ksilzz[imp][jmp][k] = DIR->bufKsiR[6 * i + 2];
			ANL->ksilxy[imp][jmp][k] = DIR->bufKsiR[6 * i + 3];
			ANL->ksilxz[imp][jmp][k] = DIR->bufKsiR[6 * i + 4];
			ANL->ksilyz[imp][jmp][k] = DIR->bufKsiR[6 * i + 5];
			i = i + 1;
		    }
		}
	    }
	}			/* end if CPU */
    }
    /* end mode */
    return EXIT_SUCCESS;
}				/* end function  */


int ComputeSeismograms(struct OUTPUTS *OUT, struct VELOCITY v0,
		       struct STRESS t0, struct PARAMETERS PRM, int l)
{
    int icpu, jcpu;
    int ir;
    int imp, jmp;
    int i0, j0, k0, i2, j2, k2;	/* 0=here, 2=at +ds/2 */
    double wx0, wy0, wz0, wx2, wy2, wz2;
    double weight[3];
    double values[8];

    /* For info :
     * Vx component  : i, j , k
     * Vy component  : i2, j2 , k
     * Vz component  : i2, j
     * Tii component : i2, j
     * Txy component : i, j2
     * Txz component : i, j
     * Tyz component : i2, j2
     */

    for (ir = 0; ir < OUT->iObs; ir++) {
	if (OUT->ista[ir] == 1) {


	    i0 = OUT->ixobs[ir];
	    wx0 = OUT->xobswt[ir];
	    j0 = OUT->iyobs[ir];
	    wy0 = OUT->yobswt[ir];
	    k0 = OUT->izobs[ir];
	    wz0 = OUT->zobswt[ir];

	    /* index and weight for +/-DS/2 component */
	    if (wx0 >= 0.5) {
		i2 = i0;
		wx2 = wx0 - 0.5;
	    } else {
		i2 = i0 - 1;
		wx2 = wx0 + 0.5;
	    }

	    if (wy0 >= 0.5) {
		j2 = j0;
		wy2 = wy0 - 0.5;
	    } else {
		j2 = j0 - 1;
		wy2 = wy0 + 0.5;
	    }

	    if (wz0 >= 0.5) {
		k2 = k0;
		wz2 = wz0 - 0.5;
	    } else {
		k2 = k0 - 1;
		wz2 = wz0 + 0.5;
	    }
	    /* correct index on the last cell on the free surface */
	    if (surface == FREE) {
		if (OUT->izobs[ir] == 1) {
		    wz2 = 0.0;
		    k2 = 0;
		}
	    }

	    /* Vx component */
	    if (PRM.me == OUT->mapping_seis[ir][1]) {
		imp = PRM.i2imp_array[i0];
		jmp = PRM.j2jmp_array[j0];


		weight[0] = wx0;
		weight[1] = wy0;
		weight[2] = wz0;

		values[0] = v0.x[imp][jmp][k0];
		values[1] = v0.x[imp + 1][jmp][k0];
		values[2] = v0.x[imp][jmp + 1][k0];
		values[3] = v0.x[imp + 1][jmp + 1][k0];

		values[4] = v0.x[imp][jmp][k0 + 1];
		values[5] = v0.x[imp + 1][jmp][k0 + 1];
		values[6] = v0.x[imp][jmp + 1][k0 + 1];
		values[7] = v0.x[imp + 1][jmp + 1][k0 + 1];

		OUT->seis_output[l % STATION_STEP][ir][1] =
		    Weight3d(weight, values);


	    }

	    /* Vy component */
	    if (PRM.me == OUT->mapping_seis[ir][2]) {
		imp = PRM.i2imp_array[i2];
		jmp = PRM.j2jmp_array[j2];
		weight[0] = wy2;
		weight[1] = wx2;
		weight[2] = wz0;

		values[0] = v0.y[imp][jmp][k0];
		values[1] = v0.y[imp][jmp + 1][k0];
		values[2] = v0.y[imp + 1][jmp][k0];
		values[3] = v0.y[imp + 1][jmp + 1][k0];

		values[4] = v0.y[imp][jmp][k0 + 1];
		values[5] = v0.y[imp][jmp + 1][k0 + 1];
		values[6] = v0.y[imp + 1][jmp][k0 + 1];
		values[7] = v0.y[imp + 1][jmp + 1][k0 + 1];

		OUT->seis_output[l % STATION_STEP][ir][2] =
		    Weight3d(weight, values);
	    }

	    /* Vz component */
	    if (PRM.me == OUT->mapping_seis[ir][3]) {
		imp = PRM.i2imp_array[i2];
		jmp = PRM.j2jmp_array[j0];
		weight[0] = wy0;
		weight[1] = wx2;
		weight[2] = wz2;

		values[0] = v0.z[imp][jmp][k2];
		values[1] = v0.z[imp][jmp + 1][k2];
		values[2] = v0.z[imp + 1][jmp][k2];
		values[3] = v0.z[imp + 1][jmp + 1][k2];

		values[4] = v0.z[imp][jmp][k2 + 1];
		values[5] = v0.z[imp][jmp + 1][k2 + 1];
		values[6] = v0.z[imp + 1][jmp][k2 + 1];
		values[7] = v0.z[imp + 1][jmp + 1][k2 + 1];

		OUT->seis_output[l % STATION_STEP][ir][3] =
		    Weight3d(weight, values);
	    }

	    /* Tii component i2, j, k */
	    if (PRM.me == OUT->mapping_seis[ir][4]) {
		/*  5 and 6 are the same because tii are at the same cell */
		imp = PRM.i2imp_array[i2];
		jmp = PRM.j2jmp_array[j0];
		weight[0] = wy0;
		weight[1] = wx2;
		weight[2] = wz0;
		/* txx */
		values[0] = t0.xx[imp][jmp][k0];
		values[1] = t0.xx[imp][jmp + 1][k0];
		values[2] = t0.xx[imp + 1][jmp][k0];
		values[3] = t0.xx[imp + 1][jmp + 1][k0];

		values[4] = t0.xx[imp][jmp][k0 + 1];
		values[5] = t0.xx[imp][jmp + 1][k0 + 1];
		values[6] = t0.xx[imp + 1][jmp][k0 + 1];
		values[7] = t0.xx[imp + 1][jmp + 1][k0 + 1];

		OUT->seis_output[l % STATION_STEP][ir][4] =
		    Weight3d(weight, values);
		/* tyy */
		values[0] = t0.yy[imp][jmp][k0];
		values[1] = t0.yy[imp][jmp + 1][k0];
		values[2] = t0.yy[imp + 1][jmp][k0];
		values[3] = t0.yy[imp + 1][jmp + 1][k0];

		values[4] = t0.yy[imp][jmp][k0 + 1];
		values[5] = t0.yy[imp][jmp + 1][k0 + 1];
		values[6] = t0.yy[imp + 1][jmp][k0 + 1];
		values[7] = t0.yy[imp + 1][jmp + 1][k0 + 1];

		OUT->seis_output[l % STATION_STEP][ir][5] =
		    Weight3d(weight, values);
		/* tzz */
		values[0] = t0.zz[imp][jmp][k0];
		values[1] = t0.zz[imp][jmp + 1][k0];
		values[2] = t0.zz[imp + 1][jmp][k0];
		values[3] = t0.zz[imp + 1][jmp + 1][k0];

		values[4] = t0.zz[imp][jmp][k0 + 1];
		values[5] = t0.zz[imp][jmp + 1][k0 + 1];
		values[6] = t0.zz[imp + 1][jmp][k0 + 1];
		values[7] = t0.zz[imp + 1][jmp + 1][k0 + 1];

		OUT->seis_output[l % STATION_STEP][ir][6] =
		    Weight3d(weight, values);
	    }

	    /* Txy component i, j2, k */
	    if (PRM.me == OUT->mapping_seis[ir][7]) {
		imp = PRM.i2imp_array[i0];
		jmp = PRM.j2jmp_array[j2];
		weight[0] = wy2;
		weight[1] = wx0;
		weight[2] = wz0;

		values[0] = t0.xy[imp][jmp][k0];
		values[1] = t0.xy[imp][jmp + 1][k0];
		values[2] = t0.xy[imp + 1][jmp][k0];
		values[3] = t0.xy[imp + 1][jmp + 1][k0];

		values[4] = t0.xy[imp][jmp][k0 + 1];
		values[5] = t0.xy[imp][jmp + 1][k0 + 1];
		values[6] = t0.xy[imp + 1][jmp][k0 + 1];
		values[7] = t0.xy[imp + 1][jmp + 1][k0 + 1];

		OUT->seis_output[l % STATION_STEP][ir][7] =
		    Weight3d(weight, values);
	    }

	    /* Txz component i, j, k2 */

	    if (PRM.me == OUT->mapping_seis[ir][8]) {
		imp = PRM.i2imp_array[i0];
		jmp = PRM.j2jmp_array[j0];
		weight[0] = wy0;
		weight[1] = wx0;
		weight[2] = wz2;

		values[0] = t0.xz[imp][jmp][k2];
		values[1] = t0.xz[imp][jmp + 1][k2];
		values[2] = t0.xz[imp + 1][jmp][k2];
		values[3] = t0.xz[imp + 1][jmp + 1][k2];

		values[4] = t0.xz[imp][jmp][k2 + 1];
		values[5] = t0.xz[imp][jmp + 1][k2 + 1];
		values[6] = t0.xz[imp + 1][jmp][k2 + 1];
		values[7] = t0.xz[imp + 1][jmp + 1][k2 + 1];

		OUT->seis_output[l % STATION_STEP][ir][8] =
		    Weight3d(weight, values);
	    }

	    /* Tyz component i2, j2, k2 */
	    if (PRM.me == OUT->mapping_seis[ir][9]) {
		imp = PRM.i2imp_array[i2];
		jmp = PRM.j2jmp_array[j2];
		weight[0] = wy2;
		weight[1] = wx2;
		weight[2] = wz2;

		values[0] = t0.yz[imp][jmp][k2];
		values[1] = t0.yz[imp][jmp + 1][k2];
		values[2] = t0.yz[imp + 1][jmp][k2];
		values[3] = t0.yz[imp + 1][jmp + 1][k2];

		values[4] = t0.yz[imp][jmp][k2 + 1];
		values[5] = t0.yz[imp][jmp + 1][k2 + 1];
		values[6] = t0.yz[imp + 1][jmp][k2 + 1];
		values[7] = t0.yz[imp + 1][jmp + 1][k2 + 1];

		OUT->seis_output[l % STATION_STEP][ir][9] =
		    Weight3d(weight, values);
	    }


	}			/* end station is inside the domain */
    }				/* end ir */

    return EXIT_SUCCESS;

}				/* end function */

double Weight3d(double w[3],	/* weights */
		double v[8]	/* values */
    )
{
    double result;

    result =
	(1. - w[2]) * ((1. - w[1]) * ((1. - w[0]) * v[0] + w[0] * v[1]) +
		       (w[1]) * ((1. - w[0]) * v[2] + w[0] * v[3])
	) +
	(w[2]) * ((1. - w[1]) * ((1. - w[0]) * v[4] + w[0] * v[5]) +
		  (w[1]) * ((1. - w[0]) * v[6] + w[0] * v[7])
	);

    return result;
}
