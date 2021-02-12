/* USE VIM TO EDIT THIS FILE IF YOU WANT TO USE MY CODE FOLDING PATTERN */
/*              add 'set foldmethod=marker' in ~/.vimrc                 */

// ABOUT VIM FOLDING (type 'zo' for open) {{{
/* Des marqueurs sont places dans le code pour le repliage de code (code folding) sous vim.
 *
 * Pour replier / deplier le code sous vim : 
 * set foldmethod=marker (dans ~/.vimrc)
 *
 * en mode commande (esc) :
 * 	zo : deplie un niveau
 * 	zc : replie un niveau
 * 	zO : deplie recursivement
 * 	zC : replie recursivement
 * 	zM : replie tout
 * 	zR : deplie tout
 * 	za : replie / deplie
 *	zf2j : make a fold with current line and the next 2 lines
 *	:20,100 fold (fo) : create a fold from line 20 through 100
 *	zfa} : if cursor is positionned on a '{', will create fold through the last bracket.
 *	work with () [] and <>. Work also backwards : zfa{
 *
 *	 zf#j  creates a fold from the cursor down  #  lines.
 *	 zf/string creates a fold from the cursor to string .
 *	 zj moves the cursor to the next fold.
 *	 zk moves the cursor to the previous fold.
 *	 zo opens a fold at the cursor.
 *	 zO opens all folds at the cursor.
 *	 zm increases the foldlevel by one.
 *	 zM closes all open folds.
 *	 zr decreases the foldlevel by one.
 *	 zR decreases the foldlevel to zero -- all folds will be open.
 *	 zd deletes the fold at the cursor.
 *	 zE deletes all folds.
 *	 [z move to start of open fold.
 *	 ]z move to end of open fold.
 *
 * en mode visuel (v)
 *	zf : create fold
 *
 * David.
 */
// }}}

// VERSION INFOS {{{
/* created on July 6th 2007  --> v9 */
/* simple precision since NVIDIA 8800GTX card does not support double precision */
/* integration des dernieres corrections de A.D*/
/* free surface at k = 0 */
/* moment force given by name_file.hist and name_file.map */
/* for the CPML, we do as Komatitsch did in hisl program */

// modifs done by David Michea : 
/* ported to NVIDIA CUDA 2.0 on June 12th 2009 */
/* added MPI communications on December 2009 */
/* added constant memory 1D array version for 1D models (july 2010) */
/* added 3D model reading based on Ariane Ducellier code (july 2010) */
/* changed parameter file layout (july 2010) */

// Now this code can run on GPU cluster (1CPU core + 1 NVIDIA GPU / node) with MPI
// It can also be ran sequentially on 1 node
// It accepts 1D & 3D models (3D models need +50% memory than 1D)

// SNAPSHOTS feature have not been parallelized yet and is consequently not present in this version for the moment
// }}}

#include "nr.h"
#include "ondes3D_kernels.h"
#include "ondes3D.h"
// default parameter file. Can be provided as an argument.
#define PRM "./chuetsu3d/chuetsu3d.prm"

int main( int argc, char **argv ) {

	// TEST THE CARD COMPUTE CAPABILITY {{{
	int block_xwidth, block_ywidth;
	struct cudaDeviceProp device_prop;
	print_err(cudaGetDeviceProperties(&device_prop, 0),"cudaGetDeviceProperties");
	if (device_prop.major >= 1 && device_prop.minor >=2) {
		if (!HIGH) {
			printf("your GPU card has a compute capability > 1.2\n>> uncomment line : \n#define HIGH_COMPUTE_CAPABILITY\n in ondes3D_kernels.h and recompile for better performance\n\n");
		}
	}
	// }}}

	// DECLARATIONS {{{
	// NEW ONES {{{
	// material properties arrays
	Material material, d_material;

	// source forces arrays
	Vector_M3D force, d_force;

	// large GPU arrays for stress and velocity
	Vector_M3D d_veloc;
	Tensor_M3D d_stress;

	// local boundaries
	Boundaries bounds;

	// communications infos
	Comm_info comms;

	// CPML indirection array
	int* d_npml_tab;

	// CPML arrays (R+W)
	float *d_phivxx, *d_phivxy, *d_phivxz, *d_phivyx, *d_phivyy, *d_phivyz, *d_phivzx, *d_phivzy, *d_phivzz;
	float *d_phitxxx, *d_phitxyy, *d_phitxzz, *d_phitxyx, *d_phityyy, *d_phityzz, *d_phitxzx, *d_phityzy, *d_phitzzz;

	float *d_dumpx, *d_dumpx2, *d_dumpy, *d_dumpy2, *d_dumpz, *d_dumpz2;
	float *d_alphax, *d_alphax2, *d_alphay, *d_alphay2, *d_alphaz, *d_alphaz2;
	float *d_kappax, *d_kappax2, *d_kappay, *d_kappay2, *d_kappaz, *d_kappaz2;

	// memory allocated on the GPU
	long int memory_used = 0;

	int maxsize;

	clock_t time_wait_k1_start, time_wait_k2_start, time_wait_k1_end, time_wait_k2_end, time_prep_buf_k1_start, time_prep_buf_k2_start, time_prep_buf_k1_end, time_prep_buf_k2_end;
	clock_t time_update_buf_k1_start, time_update_buf_k2_start, time_update_buf_k1_end, time_update_buf_k2_end;
	float time_wait_k1_total, time_wait_k2_total, time_prep_buf_k1_total, time_prep_buf_k2_total, time_update_buf_k1_total, time_update_buf_k2_total;
	
	int dim_model;
	char* parameter_file;

	// for 3D models
	int	imax, jmax, kmax;
	char	srcfile4[50];
	float	theta;
	float	x0;
	float 	y0;
	// }}}

	// OLD ONES {{{
	clock_t time_tl_start, time_tl_end, time_k1_start, time_k1_end, time_k2_start, time_k2_end,
		time_su_start, time_su_end, time_tr_start, time_tr_end, time_se_start, time_se_end;
	float time_tl_total, time_k1_total, time_k2_total, time_su_total, time_tr_total, time_se_total;
	time_tl_total = time_k1_total = time_k2_total = time_su_total = time_tr_total = time_se_total = 0.f;


	int     i, j, k, l, n, is, it, ir, iw, ly, l1;
	float  **vel, **seisx, **seisy, **seisz, **seisxx, **seisyy, **seiszz,
	       **seisxy, **seisxz, **seisyz;
	double *strike, *dip, *rake;
	double pi;
	float	*xhypo, *yhypo, *zhypo, *slip,
		*laydep, *vp0, *vs0, *rho0,	*q0,
		*xweight, *yweight, *zweight,
		*xobs, *yobs, *zobs, *xobswt, *yobswt, *zobswt;

	float	ds, dt, zdum, fd,
		vpdum, vsdum, rhodum, weight,
		pxx, pyy, pzz, pxy, pyz, pxz, dsbiem, dtbiem, time,
		xhypo0, yhypo0, zhypo0, mo,
		dump0, alpha0, kappa0;

	float	*dumpx, *dumpy, *dumpz, *dumpx2, *dumpy2, *dumpz2,
		*alphax, *alphay, *alphaz, *alphax2, *alphay2, *alphaz2,
		*kappax, *kappay, *kappaz, *kappax2, *kappay2, *kappaz2;

	char 	flname[50], number[5], flname4[80], model_file[200], model_type[10], 
		outdir[50], srcfile1[80], srcfile2[80], srcfile3[80], buf[256];

	int 	npmlv;
	int	    *ixhypo, *iyhypo, *izhypo, *insrc, *nobs, *ista, *ixobs, *iyobs, *izobs;
	int	    ISRC, IDUR, IOBS, NLAYER, ixhypo0, iyhypo0, izhypo0,
		    NDIM, XMIN, XMAX, YMIN, YMAX, ZMIN,
		    i0, j0, TMAX ;

	FILE *  fp4;
	FILE *  paramfile;
	FILE *  fp_in1;
	FILE *  fp_in2;
	FILE *	fp_in3;

	float  NPOWER;
	float  xoriginleft, xoriginright, yoriginfront, yoriginback, zoriginbottom;
	float  xval, yval, zval, abscissa_in_PML, abscissa_normalized;

	int few_sources;
	
	char* decal;
	// }}}
	// }}}

// SELECT PARAMETER FILE {{{
			if (argc < 2) {
							parameter_file = PRM;
			} else {
							parameter_file = argv[1];
			}
// }}}	

	// MPI INIT {{{
#ifdef USE_MPI/*{{{*/
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&(comms.nbprocs));
	MPI_Comm_rank(MPI_COMM_WORLD,&(comms.rank));

	comms.nproc_x = NPROCX;
	comms.nproc_y = NPROCY;
	if (comms.nbprocs != (comms.nproc_x * comms.nproc_y)) {
		printf("incoherence entre les parametres MPI : \n\tnbprocs : %d\n\tnproc_x : %d\n\tnproc_y : %d\nexiting ...\n\n",comms.nbprocs, comms.nproc_x, comms.nproc_y);
		_exit(1);
	}
#else/*}}}*/
	comms.nproc_x = NPROCX;
	comms.nproc_y = NPROCY;
	comms.nbprocs = NPROCX*NPROCY;
	comms.rank = 0;
	assert(comms.nbprocs == 1);
#endif

	decal = (char*) malloc((comms.rank*DECALAGE+2)*sizeof(char));
	memset(decal, (int)' ', comms.rank*DECALAGE+1);
	decal[comms.rank*DECALAGE+1] = '\0';
	decal[0] = (char) comms.rank+48;
	if (VERBOSE >= 2) cuda_device_info(decal);
	if (VERBOSE >= 1 && comms.rank == MASTER) printTopo();
	// }}}

	// MASTER READ PARAMETERS FILES AND BROADCAST TO ALL PROCS {{{
	if (VERBOSE >= 3) printf("%sREAD PARAMETERS FILES AND BROADCAST TO ALL PROCS\n",decal);

	// PROC MASTER READS THE MODEL PARAMETERS IN METER, SECOND {{{
	if (VERBOSE >= 3) printf("%sPROC MASTER READS THE MODEL PARAMETERS IN METER, SECOND\n",decal);
	NPOWER = 2.;

	// read parameter file .prm
	if (comms.rank == MASTER) {
		paramfile = fopen( parameter_file, "r");
		if ( paramfile == NULL ){
			fprintf(stderr, "failed opening parameter file %s\n",parameter_file);
			exit(1);
		}
		readIntParam("NDIM", &NDIM, paramfile);
		readIntParam("XMIN", &XMIN, paramfile);
		readIntParam("XMAX", &XMAX, paramfile);
		readIntParam("YMIN", &YMIN, paramfile);
		readIntParam("YMAX", &YMAX, paramfile);
		readIntParam("ZMIN", &ZMIN, paramfile);
		readIntParam("TIME_STEPS", &TMAX, paramfile);
		readIntParam("I0", &i0, paramfile);
		readIntParam("J0", &j0, paramfile);
		readStringParam("OUTPUT_DIR", outdir, paramfile);
		readStringParam("SOURCES", srcfile1, paramfile);
		readStringParam("SOURCE_TIME_FUNCTION", srcfile2, paramfile);
		readStringParam("STATIONS", srcfile3, paramfile);
		readFloatParam("DS", &ds, paramfile);
		readFloatParam("DT", &dt, paramfile);
		readFloatParam("DF", &fd, paramfile);
		readStringParam("MODEL_TYPE", model_type, paramfile);

		if (!strcmp(model_type, "1D")) dim_model = 1;
		else {	if (!strcmp(model_type, "3D")) dim_model = 3;
			else {
				printf("MODEL TYPE NOT RECOGNISED : %s, %d\nexiting\n",model_type, strlen(model_type));
#ifdef USE_MPI
				MPI_Abort(MPI_COMM_WORLD, 1);
#else
				exit(1);
#endif
			}
		}
	}
#ifdef USE_MPI
	MPI_Bcast(&dim_model, 1, MPI_INTEGER, MASTER, MPI_COMM_WORLD);
#endif
	if (dim_model == 1) {
		if (comms.rank == MASTER) {
			readIntParam("NB_LAYERS", &NLAYER, paramfile);
		}
#ifdef USE_MPI
		MPI_Bcast(&NLAYER, 1, MPI_INTEGER, MASTER, MPI_COMM_WORLD);
#endif
		laydep = fvector(0, NLAYER-1);
		vp0 = fvector(0, NLAYER-1);
		vs0 = fvector(0, NLAYER-1);
		rho0 = fvector(0, NLAYER-1);
		q0  = fvector(0, NLAYER-1);

		if (comms.rank == MASTER) {
			for ( ly = 0; ly < NLAYER; ly++){
				readLayerParam("LAYER_",ly+1, &laydep[ly], &vp0[ly], &vs0[ly], &rho0[ly], &q0[ly], paramfile);
			}
			fclose( paramfile );
		}
	} else if(dim_model == 3) {
		if (comms.rank == MASTER) {
			readIntParam("IMAX", &imax, paramfile);
			readIntParam("JMAX", &jmax, paramfile);
			readIntParam("KMAX", &kmax, paramfile);
			readFloatParam("THETA", &theta, paramfile);
			readFloatParam("X0", &x0, paramfile);
			readFloatParam("Y0", &y0, paramfile);
			readStringParam("MODEL_FILE", model_file, paramfile);
			fclose( paramfile );
		}
	}
	if (comms.rank == MASTER && VERBOSE >= 1) {
		printf("\nDimension of FDM order ... %i\n", NDIM );
		printf("\nParameter File ... %s\n", parameter_file);
		printf("Source Model based on ... %s\n", srcfile1 );
		printf("Rupture History from ... %s\n", srcfile2 );
		printf("Station Position at ... %s\n", srcfile3 );
		printf("Output directory ... %s\n", outdir );
		printf("\nspatial grid ds = %f[m]\n", ds);
		printf("time step dt = %f[s]\n", dt);
		printf("\nModel Region (%i:%i, %i:%i, %i:%i)\n",XMIN, XMAX, YMIN, YMAX, ZMIN, 0);
		printf("Absorbing layer thickness: %i\n",delta);
		printf("Model read : %s model\n",dim_model==3?"3D":"1D");
	}

	// BROADCASTING DATA READ TO OTHER PROCS {{{
	if (VERBOSE >= 3) printf("%sBROADCASTING INTS READ TO OTHER PROCS\n",decal);
#ifdef USE_MPI/*{{{*/
	// warning, cutplanes temporary disabled !!!
	{// first ints
		int buffer[7];
		if (comms.rank == MASTER) {
			buffer[0] = NDIM;
			buffer[1] = XMIN;
			buffer[2] = XMAX;
			buffer[3] = YMIN;
			buffer[4] = YMAX;
			buffer[5] = ZMIN;
			buffer[6] = TMAX;
		}
		MPI_Bcast(buffer, 7, MPI_INTEGER, MASTER, MPI_COMM_WORLD);
		if (comms.rank != MASTER) {
			NDIM = buffer[0];
			XMIN = buffer[1];
			XMAX = buffer[2];
			YMIN = buffer[3];
			YMAX = buffer[4];
			ZMIN = buffer[5];
			TMAX = buffer[6];
		}
	}
	if (dim_model == 3) {
		int buffer[3];
		if (comms.rank == MASTER) {
			buffer[0] = imax;
			buffer[1] = jmax;
			buffer[2] = kmax;
		}
		MPI_Bcast(buffer, 3, MPI_INTEGER, MASTER, MPI_COMM_WORLD);
		if (comms.rank != MASTER) {
			imax = buffer[0];
			jmax = buffer[1];
			kmax = buffer[2];
		}
	}
	// then floats
	{
		float* buffer = (float*) malloc ((NLAYER*5+3)*sizeof(float));
		if (comms.rank == MASTER) {
			buffer[0] = ds;
			buffer[1] = dt;
			buffer[2] = fd;
			if (dim_model == 1) {
				for (int i=0; i<NLAYER; i++) {
					buffer[3+i*5] = (float)laydep[i];
					buffer[4+i*5] = (float)vp0[i];
					buffer[5+i*5] = (float)vs0[i];
					buffer[6+i*5] = (float)rho0[i];
					buffer[7+i*5] = (float)q0[i];
				}
			}
		}
		MPI_Bcast(buffer, NLAYER*5+3, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
		if (comms.rank != MASTER) {
			ds = buffer[0];
			dt = buffer[1];
			fd = buffer[2];
			if (dim_model == 1) {
				for (int i=0; i<NLAYER; i++) {
					laydep[i] = buffer[3+i*5];
					vp0[i] = buffer[4+i*5];
					vs0[i] = buffer[5+i*5];
					rho0[i] = buffer[6+i*5];
					q0[i] = buffer[7+i*5];
				}
			}
		}
	}
	if (dim_model == 3) {
		float buffer[3];
		if (comms.rank == MASTER) {
			buffer[0] = theta;
			buffer[1] = x0;
			buffer[2] = y0;
		}
		MPI_Bcast(buffer, 3, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
		if (comms.rank != MASTER) {
			theta = buffer[0];
			x0 = buffer[1];
			y0 = buffer[2];
		}
	}
		

#endif/*}}}*/
	// }}}
	pi = acosf(-1.0);
	dump0 = - (NPOWER + 1) * log(reflect) / (2.0 * delta * ds);
	alpha0 = fd*(float)pi;	/* alpha0 = pi*fd where fd is the dominant frequency of the source */
	kappa0 = 1.0;
	// }}}
	
	// MPI SLICING & ADDRESSING {{{
	if (VERBOSE >= 3) printf("proc %dMPI SLICING & ADDRESSING\n",comms.rank);

	// attention : ZMAX = 0 ??? dans code original, tout est decale de + 1
	MPI_slicing_and_addressing (&comms, &bounds, XMIN, XMAX, YMIN, YMAX, ZMIN, 0);
	// }}}

	// PROC MASTER READS THE SOURCE POSITION {{{
	if (VERBOSE >= 3) printf("%sPROC MASTER READS THE SOURCE POSITION\n",decal);
	if (comms.rank == MASTER) {
		strcpy(flname, srcfile1);
		fp_in1 = fopen( flname, "r");
		if ( fp_in1 == NULL ){
			perror ("failed at fopen 1");
			exit(1);
		}
		fscanf ( fp_in1, "%d", &ISRC );
		fscanf ( fp_in1, "%f %f %f", &xhypo0, &yhypo0, &zhypo0 );

		if (VERBOSE >= 1) printf("\nNUMBER OF SOURCE %d\n", ISRC);
		if (VERBOSE >= 2) printf("Hypocenter ... (%f, %f, %f)\n", xhypo0, yhypo0, zhypo0);

		// ixhypo0 = (int)(xhypo0/ds)+1;
		// iyhypo0 = (int)(yhypo0/ds)+1;
		// izhypo0 = (int)(zhypo0/ds)+1;
		// if( xhypo0 < 0.0 ) ixhypo0 = ixhypo0 - 1;
		// if( yhypo0 < 0.0 ) iyhypo0 = iyhypo0 - 1;
		// if( zhypo0 < 0.0 ) izhypo0 = izhypo0 - 1;
		
		ixhypo0 = my_float2int(xhypo0/ds)+1;
		iyhypo0 = my_float2int(yhypo0/ds)+1;
		izhypo0 = my_float2int(zhypo0/ds);

		if (VERBOSE >= 2) printf(".............. (%i, %i, %i)\n", ixhypo0, iyhypo0, izhypo0);
	}
#ifdef USE_MPI/*{{{*/
	// broadcast
	MPI_Bcast(&ISRC, 1, MPI_INTEGER, MASTER, MPI_COMM_WORLD);
#endif/*}}}*/

	ixhypo = ivector(0, ISRC-1);
	iyhypo = ivector(0, ISRC-1);
	izhypo = ivector(0, ISRC-1);
	insrc  = ivector(0, ISRC-1);
	xhypo   = fvector(0, ISRC-1);
	yhypo   = fvector(0, ISRC-1);
	zhypo   = fvector(0, ISRC-1);
	strike  = dvector(0, ISRC-1);
	dip     = dvector(0, ISRC-1);
	rake    = dvector(0, ISRC-1);
	slip	= fvector(0, ISRC-1);
	xweight = fvector(0, ISRC-1);
	yweight = fvector(0, ISRC-1);
	zweight = fvector(0, ISRC-1);

	if (comms.rank == MASTER) {
		for ( is = 0;  is < ISRC; is++)
			fscanf ( fp_in1, "%d %f %f %f", &n, &xhypo[is], &yhypo[is], &zhypo[is]);
		fclose( fp_in1 );
	}
#ifdef USE_MPI/*{{{*/
	{// broadcast
		float buffer[ISRC*3];
		if (comms.rank == MASTER) {
			for (int i=0; i<ISRC; i++) {
				buffer[0+i*3] = xhypo[i];
				buffer[1+i*3] = yhypo[i];
				buffer[2+i*3] = zhypo[i];
			}
		}
		MPI_Bcast(buffer, ISRC*3, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
		if (comms.rank != MASTER) {
			for (int i=0; i<ISRC; i++) {
				xhypo[i] = buffer[0+i*3];
				yhypo[i] = buffer[1+i*3];
				zhypo[i] = buffer[2+i*3];
			}
		}
	}
#endif/*}}}*/
	int nb_sources_in_mpi_slice = 0;
	int nb_global_sources = 0;
	for ( is = 0; is < ISRC; is++){
		
		ixhypo[is] = my_float2int(xhypo[is]/ds)+1;
		xweight[is] = (xhypo[is]/ds - ixhypo[is]+1);

		iyhypo[is] = my_float2int(yhypo[is]/ds)+1;
		yweight[is] = (yhypo[is]/ds - iyhypo[is]+1);

		izhypo[is] = my_float2int(zhypo[is]/ds);
		zweight[is] = (zhypo[is]/ds - izhypo[is]);
		
		insrc[is] = 1;
		if (comms.rank == MASTER && VERBOSE >= 2) {
			printf("Source %i .... (%f, %f, %f)\n", is+1, xhypo[is], yhypo[is], zhypo[is] );
			printf(".............. (%i, %i, %i)\n", ixhypo[is], iyhypo[is], izhypo[is]);
		}
		if (!( ixhypo[is] > XMAX+2 || ixhypo[is] < XMIN-2 || iyhypo[is] > YMAX+2 || iyhypo[is] < YMIN-2 || izhypo[is] > 2 || izhypo[is] < ZMIN-2 )) {
			nb_global_sources++;
		}
		if ( ixhypo[is] > bounds.xmax+2 || ixhypo[is] < bounds.xmin-2 || iyhypo[is] > bounds.ymax+2 || iyhypo[is] < bounds.ymin-2 || izhypo[is] > bounds.zmax+2 || izhypo[is] < bounds.zmin-2 )
		{
			// pas prise en compte
			insrc[is] = 0;
		} else {
			// dans la tranche
			nb_sources_in_mpi_slice++;
		}
	}
	if (comms.rank == MASTER && nb_global_sources < ISRC) printf("WARNING : %d sources are defined outside the domain\n",ISRC - nb_global_sources);
	few_sources = (nb_sources_in_mpi_slice > SRC_PERF_CRITERIA)?0:1;
	// }}}

	// PROC MASTER READS THE SOURCE TIME FUNCTION {{{
	if (VERBOSE >= 3) printf("%sPROC MASTER READS THE SOURCE TIME FUNCTION\n",decal);
	if (comms.rank == MASTER) {
		strcpy(flname, srcfile2);
		fp_in2 = fopen( flname, "r");
		if ( fp_in2 == NULL ){
			perror ("failed at fopen 2");
			exit(1);
		}
		fgets(buf, 255, fp_in2);
		fgets(buf, 255, fp_in2);

		fscanf ( fp_in2, "%f %f", &dsbiem, &dtbiem );
		fscanf ( fp_in2, "%d", &IDUR );

		if (VERBOSE >= 2) printf("\nSource duration %f sec\n", dtbiem*(IDUR-1));
		if (VERBOSE >= 2) printf("fault segment %f m, %f s\n", dsbiem, dtbiem);
	}
#ifdef USE_MPI/*{{{*/
	MPI_Bcast(&IDUR, 1, MPI_INTEGER, MASTER, MPI_COMM_WORLD);
#endif/*}}}*/

	vel = dmatrix(0, ISRC-1, 0, IDUR-1);
	mo = 0.0;

	if (comms.rank == MASTER) {
		for( is = 0; is < ISRC; is++ ){
			fscanf ( fp_in2, "%d", &n);
			fscanf ( fp_in2, "%lf %lf %lf", &strike[is], &dip[is], &rake[is]);
			strike[is] = strike[is]/180.*pi;
			dip[is] = dip[is]/180.*pi;
			rake[is] = rake[is]/180.*pi;
			slip[is] = 0.;
			for ( it = 0; it < IDUR; it++ ) {
				fscanf ( fp_in2, "%f", &vel[is][it]);
				slip[is] += vel[is][it]*dtbiem;
				vel[is][it] = (dsbiem/ds) * (dsbiem/ds) * vel[is][it] / ds;
			}
			mo += slip[is];
		}
		fclose( fp_in2 );
		mo = dsbiem * dsbiem * mo;
	}
#ifdef USE_MPI/*{{{*/
	{// broadcast
		float buffer[3+ISRC*4+ISRC*IDUR];
		if (comms.rank == MASTER) {
			buffer[0] = dsbiem;
			buffer[1] = dtbiem;
			buffer[2] = mo;
			for (int i=0; i<ISRC; i++) {
				buffer[3+i*4] = strike[i];
				buffer[4+i*4] = dip[i];
				buffer[5+i*4] = rake[i];
				buffer[6+i*4] = slip[i];
			}
			for (int i=0; i<ISRC; i++) {
				for (int j=0; j<IDUR; j++) {
					buffer[7+(ISRC-1)*4+i*IDUR+j] = vel[i][j];
				}
			}
		}
		MPI_Bcast(buffer, 3+ISRC*4+ISRC*IDUR, MPI_FLOAT, MASTER, MPI_COMM_WORLD); 
		if (comms.rank != MASTER) {
			dsbiem = buffer[0];
			dtbiem = buffer[1];
			mo = buffer[2];
			for (int i=0; i<ISRC; i++) {
				strike[i] = buffer[3+i*4];
				dip[i] = buffer[4+i*4];
				rake[i] = buffer[5+i*4];
				slip[i] = buffer[6+i*4];
			}
			for (int i=0; i<ISRC; i++) {
				for (int j=0; j<IDUR; j++) {
					vel[i][j] = buffer[7+(ISRC-1)*4+i*IDUR+j];
				}
			}
		}
	}
#endif/*}}}*/
	// }}}

	// PROC MASTER READS THE STATIONS POSITIONS {{{
	if (VERBOSE >= 3) printf("%sPROC MASTER READS THE STATIONS POSITIONS\n",decal);
	if (comms.rank == MASTER) {
		if (VERBOSE >= 2) printf("\n Stations coordinates :\n");

		fp_in3 = fopen( srcfile3, "r");
		if ( fp_in3 == NULL ){
			perror ("failed at fopen 3");
			exit(1);
		}
		fscanf ( fp_in3, "%d", &IOBS );
	}
#ifdef USE_MPI/*{{{*/
	MPI_Bcast(&IOBS, 1, MPI_INTEGER, MASTER, MPI_COMM_WORLD);
#endif/*}}}*/

	nobs = ivector(0, IOBS-1);
	xobs = fvector(0, IOBS-1);
	yobs = fvector(0, IOBS-1);
	zobs = fvector(0, IOBS-1);
	ixobs = ivector(0, IOBS-1);
	iyobs = ivector(0, IOBS-1);
	izobs = ivector(0, IOBS-1);
	xobswt = fvector(0, IOBS-1);
	yobswt = fvector(0, IOBS-1);
	zobswt = fvector(0, IOBS-1);
	ista = ivector(0, IOBS-1);

	if (comms.rank == MASTER) {
		for ( ir = 0; ir < IOBS; ir++){
			fscanf ( fp_in3, "%d %f %f %f", &nobs[ir], &xobs[ir], &yobs[ir], &zobs[ir] );
		}
		fclose( fp_in3 );
	}
#ifdef USE_MPI/*{{{*/
	{//broadcast
		float buffer[IOBS*3];
		if (comms.rank == MASTER) {
			for (int i=0; i<IOBS; i++) {
				buffer[0+i*3] = xobs[i];
				buffer[1+i*3] = yobs[i];
				buffer[2+i*3] = zobs[i];
			}
		}
		MPI_Bcast(buffer, IOBS*3, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
		if (comms.rank != MASTER) {
			for (int i=0; i<IOBS; i++) {
				xobs[i] = buffer[0+i*3];
				yobs[i] = buffer[1+i*3];
				zobs[i] = buffer[2+i*3];
			}
		}
	}
#endif/*}}}*/
	int nb_local_obs = 0;
	int nb_global_obs = 0;

	for ( ir = 0; ir < IOBS; ir++){
		ista[ir] = 1;
		ixobs[ir] = my_float2int(xobs[ir]/ds)+1;
		xobswt[ir] = (xobs[ir]/ds - ixobs[ir]+1);

		iyobs[ir] = my_float2int(yobs[ir]/ds)+1;
		yobswt[ir] = (yobs[ir]/ds - iyobs[ir]+1);

		izobs[ir] = my_float2int(zobs[ir]/ds);
		zobswt[ir] = (zobs[ir]/ds - izobs[ir]);

		if ( xobs[ir] <= (bounds.xmin-1)*ds || xobs[ir] > bounds.xmax*ds ||
				yobs[ir] <= (bounds.ymin-1)*ds || yobs[ir] > bounds.ymax*ds ||
				zobs[ir] < bounds.zmin*ds || zobs[ir] > bounds.zmax*ds ){
			ista[ir] = 0;
		}
		if (!( xobs[ir] < XMIN*ds || xobs[ir] > XMAX*ds ||
					yobs[ir] < YMIN*ds || yobs[ir] > YMAX*ds ||
					zobs[ir] < ZMIN*ds || zobs[ir] > 0)){
			nb_global_obs++;
		}
		if (ista[ir]!=0) {
			nb_local_obs++;
			if (VERBOSE >= 2) {
				printf("%sstation %d, owned by proc %d : \n",decal, ir+1,comms.rank);
				printf("%s%d %d %f %f %f %d\n",decal, ir+1, ista[ir], xobs[ir], yobs[ir], zobs[ir], izobs[ir]);
				printf("%sobswt : %f %f %f\n",decal, xobswt[ir], yobswt[ir], zobswt[ir]);
				printf("%siobst : %d %d %d\n",decal, ixobs[ir], iyobs[ir], izobs[ir]);
			}
		}
	}
#ifdef USE_MPI/*{{{*/
	int nb_obs;
	int station_mask[IOBS];

	MPI_Reduce ( &nb_local_obs, &nb_obs, 1, MPI_INTEGER, MPI_SUM, MASTER, MPI_COMM_WORLD);
	MPI_Reduce ( ista, station_mask, IOBS, MPI_INTEGER, MPI_BOR, MASTER, MPI_COMM_WORLD);
	if (comms.rank == MASTER) {
		if (IOBS != nb_global_obs) {
			printf("WARNING : %d stations are defined outside the domain\n",IOBS-nb_global_obs);
		}
		if (nb_obs != nb_global_obs) {
			printf("problem with stations repartition : %d != %d\n",nb_obs,nb_global_obs);
			for (ir=0; ir<IOBS; ir++) {
				if (station_mask[ir] == 0) printf("station %d (%f, %f, %f) (%d, %d, %d) non attribuee \n",ir,xobs[ir]/ds, yobs[ir]/ds, zobs[ir]/ds, 
						ixobs[ir], iyobs[ir], izobs[ir]);
			}
			MPI_Abort(MPI_COMM_WORLD, 2);
		}
	}
#endif/*}}}*/

	seisx = dmatrix(0, IOBS-1, 0, TMAX-1);
	seisy = dmatrix(0, IOBS-1, 0, TMAX-1);
	seisz = dmatrix(0, IOBS-1, 0, TMAX-1);
	seisxx = dmatrix(0, IOBS-1, 0, TMAX-1);
	seisyy = dmatrix(0, IOBS-1, 0, TMAX-1);
	seiszz = dmatrix(0, IOBS-1, 0, TMAX-1);
	seisxy = dmatrix(0, IOBS-1, 0, TMAX-1);
	seisxz = dmatrix(0, IOBS-1, 0, TMAX-1);
	seisyz = dmatrix(0, IOBS-1, 0, TMAX-1);
	// }}}
	// }}}

	// ALLOCATION OF CPU ARRAYS {{{
	if (VERBOSE >= 3) printf("%sALLOCATION OF CPU ARRAYS\n",decal);
	allocate_arrays_CPU(&material, &force, &bounds, dim_model);
	// }}}

	// DEFINITION OF THE MATERIAL PROPERTIES {{{
	if (VERBOSE >= 3) printf("%sDEFINITION OF THE MATERIAL PROPERTIES\n",decal);

	if (dim_model == 3) { // 3D MODEL
		float xdum, ydum, zdum;

		// allocation de vp0, vs0 et rho0
		vp0 = (float*) malloc (imax*jmax*kmax*sizeof(float));
		vs0 = (float*) malloc (imax*jmax*kmax*sizeof(float));
		rho0 = (float*) malloc (imax*jmax*kmax*sizeof(float));

		float* xcor = (float*) malloc (imax*sizeof(float));
		float* ycor = (float*) malloc (jmax*sizeof(float));
		float* zcor = (float*) malloc (kmax*sizeof(float));

		theta = theta*pi/180.;

		if (comms.rank == MASTER) {

			// READ MODEL
			FILE* model_fd = fopen (model_file, "r");
			if ( model_fd == NULL ){
				printf("failed opening file %s\n",model_file);
				exit(1);
			}
			for ( k = 0; k < kmax; k++){
				for ( j = 0; j < jmax; j++){
					for ( i = 0; i < imax; i++){
						fscanf (model_fd, "%f %f %f %f %f %f", &xdum, &ydum, &zdum,
								&vp0[k*imax*jmax + j*imax + i], &vs0[k*imax*jmax + j*imax + i], &rho0[k*imax*jmax + j*imax + i]);
						if ( k == 0 && j == 0 ) xcor[i] = xdum;
						if ( i == 0 && k == 0 ) ycor[j] = ydum;
						if ( i == 0 && j == 0 ) zcor[k] = zdum;
					}
				}
			}
			fclose (model_fd);

			if (VERBOSE >=2){
				printf ("Origin of 3D model %f %f\n", x0, y0);
				printf ("Rotation of 3D model %f\n", theta*180./pi);
				for ( i = 0; i < imax; i++) printf ("%lf ", xcor[i]);
				printf("\n");
				for ( j = 0; j < jmax; j++) printf ("%lf ", ycor[j]);
				printf("\n");
				for ( k = 0; k < kmax; k++) printf ("%lf ", zcor[k]);
				printf("\n");
				printf ("\n3D file read\n");
			}
		}
#ifdef USE_MPI
		// BROADCAST MODEL
		if (VERBOSE >=3) printf("BROADCAST 3D MODEL\n");
		MPI_Bcast(xcor, imax, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
		MPI_Bcast(ycor, jmax, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
		MPI_Bcast(zcor, kmax, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

		MPI_Bcast(vp0, imax*jmax*kmax, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
		MPI_Bcast(vs0, imax*jmax*kmax, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
		MPI_Bcast(rho0, imax*jmax*kmax, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
#endif
		// INTERPOLATION ON LOCAL SLICE
		if (VERBOSE >=3) printf("INTERPOLATION ON LOCAL SLICE\n");
		int i1, i2, j1, j2, k1, k2, idum, jdum, kdum;
		float x, y, z;
		float alpha, beta, gamma, vsave, vpave, rhosave;

		for ( kdum = bounds.zmin; kdum <= bounds.zmax; kdum++){
			float zdum = (kdum-1) * ds/1000.;
			for (jdum = bounds.ymin; jdum <= bounds.ymax; jdum++) {
				float ydum = (jdum-1) * ds/1000.;
				for (idum = bounds.xmin; idum <= bounds.xmax; idum++) {
					float xdum = (idum-1) * ds/1000.;

					x = (xdum - x0)*cos(theta) - (ydum - y0)*sin(theta);
					y = (xdum - x0)*sin(theta) + (ydum - y0)*cos(theta);
					z = -zdum;
		
					if ( x < xcor[0] ){
						i1 = 0;
						i2 = i1;
						alpha = 0.;
					} else if ( x > xcor[imax-1] ){
						i1 = imax - 1;
						i2 = i1;
						alpha = 0.;
					} else {
						for ( i = 0; i < imax-1; i++){
							if ( x >= xcor[i] && x < xcor[i+1] ){
								i1 = i;
								i2 = i + 1;
								alpha = (x-xcor[i])/(xcor[i+1]-xcor[i]);
							}
						}
					}
					if ( y < ycor[0] ){
						j1 = 0;
						j2 = j1;
						beta = 0.;
					} else if ( y > ycor[jmax-1] ){
						j1 = jmax - 1;
						j2 = j1;
						beta = 0.;
					} else {
						for ( j = 0; j < jmax-1; j++){
							if ( y >= ycor[j] && y < ycor[j+1] ){
								j1 = j;
								j2 = j + 1;
								beta = (y-ycor[j])/(ycor[j+1]-ycor[j]);
							}
						}
					}
					if ( z < zcor[0] ){
						k1 = 0;
						k2 = k1;
						gamma = 0.;
					} else if ( z > zcor[kmax-1] ){
						k1 = kmax-1;
						k2 = k1;
						gamma = 0.;
					} else {
						for ( k = 0; k < kmax-1; k++){
							if ( z >= zcor[k] && z < zcor[k+1] ){
								k1 = k;
								k2 = k + 1;
								gamma = (z-zcor[k])/(zcor[k+1]-zcor[k]);
							}
						}
					}
					vpave = ( vp0[k1*imax*jmax + j1*imax + i1]*(1.-alpha)*(1.-beta) +
							vp0[k1*imax*jmax + j1*imax + i2]*alpha*(1.-beta) +
							vp0[k1*imax*jmax + j2*imax + i1]*(1.-alpha)*beta +
							vp0[k1*imax*jmax + j2*imax + i2]*alpha*beta)*(1.-gamma) +
						( vp0[k2*imax*jmax + j1*imax + i1]*(1.-alpha)*(1.-beta) +
						  vp0[k2*imax*jmax + j1*imax + i2]*alpha*(1.-beta) +
						  vp0[k2*imax*jmax + j2*imax + i1]*(1.-alpha)*beta +
						  vp0[k2*imax*jmax + j2*imax + i2]*alpha*beta)*gamma ;
					vsave = ( vs0[k1*imax*jmax + j1*imax + i1]*(1.-alpha)*(1.-beta) +
							vs0[k1*imax*jmax + j1*imax + i2]*alpha*(1.-beta) +
							vs0[k1*imax*jmax + j2*imax + i1]*(1.-alpha)*beta +
							vs0[k1*imax*jmax + j2*imax + i2]*alpha*beta)*(1.-gamma) +
						( vs0[k2*imax*jmax + j1*imax + i1]*(1.-alpha)*(1.-beta) +
						  vs0[k2*imax*jmax + j1*imax + i2]*alpha*(1.-beta) +
						  vs0[k2*imax*jmax + j2*imax + i1]*(1.-alpha)*beta +
						  vs0[k2*imax*jmax + j2*imax + i2]*alpha*beta)*gamma ;
					rhosave = ( rho0[k1*imax*jmax + j1*imax + i1]*(1.-alpha)*(1.-beta) +
							rho0[k1*imax*jmax + j1*imax + i2]*alpha*(1.-beta) +
							rho0[k1*imax*jmax + j2*imax + i1]*(1.-alpha)*beta +
							rho0[k1*imax*jmax + j2*imax + i2]*alpha*beta)*(1.-gamma) +
						( rho0[k2*imax*jmax + j1*imax + i1]*(1.-alpha)*(1.-beta) +
						  rho0[k2*imax*jmax + j1*imax + i2]*alpha*(1.-beta) +
						  rho0[k2*imax*jmax + j2*imax + i1]*(1.-alpha)*beta +
						  rho0[k2*imax*jmax + j2*imax + i2]*alpha*beta)*gamma ;

					ACCESS(material,vp,idum,jdum,kdum) = vpave*1000.;
					ACCESS(material,vs,idum,jdum,kdum) = vsave*1000.;
					ACCESS(material,rho,idum,jdum,kdum) = rhosave*1000.;

					ACCESS(material,mu,idum,jdum,kdum)  = ACCESS(material,vs,idum,jdum,kdum)*ACCESS(material,vs,idum,jdum,kdum)*ACCESS(material,rho,idum,jdum,kdum);
					ACCESS(material,lam,idum,jdum,kdum) = ACCESS(material,vp,idum,jdum,kdum)*ACCESS(material,vp,idum,jdum,kdum)*ACCESS(material,rho,idum,jdum,kdum) - 2.0*ACCESS(material,mu,idum,jdum,kdum);
				} // end of idum 
			} // end of if jdum
			if (ixhypo[0]>=bounds.xmin && ixhypo[0]<=bounds.xmax && iyhypo[0]>=bounds.ymin && iyhypo[0]<=bounds.ymax && VERBOSE >= 2) {
				printf("proc %d : param below epicenter :%7.2f, %7.2f, %7.2f, %7.2f, %7.2e, %7.2e\n", comms.rank, zdum, 
				ACCESS(material,rho,ixhypo[0],iyhypo[0],kdum), ACCESS(material,vp,ixhypo[0],iyhypo[0],kdum), ACCESS(material,vs,ixhypo[0],iyhypo[0],kdum), ACCESS(material,mu,ixhypo[0],iyhypo[0],kdum), ACCESS(material,lam,ixhypo[0],iyhypo[0],kdum));
			}
		} // end of kdum
	} else { // 1D MODEL
		// rho, vp, vs are functions of z
		int ind = 0;
		for ( k = bounds.zinf-2; k <= bounds.zsup+2; k++){
			zdum = (k-1) * ds/1000.;
			material.rho[ind] = rho0[NLAYER-1];
			material.rho[ind] = rho0[NLAYER-1]; /* kg/m^3 */
			material.vp[ind] = vp0[NLAYER-1]; /* m/s^3 */
			material.vs[ind] = vs0[NLAYER-1];
			for ( ly = 0; ly < NLAYER-1; ly++){
				if ( zdum <= laydep[ly] && zdum > laydep[ly+1] ){
					material.rho[ind] = rho0[ly];
					material.vp[ind] = vp0[ly];
					material.vs[ind] = vs0[ly];
				}
			}
			if ( zdum > laydep[0] ){ /* shallow part */
				material.vs[ind] = vs0[0];
				material.vp[ind] = vp0[0];
				material.rho[ind] = rho0[0];
				vsdum = hardrock( -zdum ); /* for Alps, Pyrenees */

				vpdum = vsdum*sqrt(3.);
				rhodum = 1741.*powf(vpdum/1000., 0.25);
				if ( vsdum <= material.vs[ind] ) material.vs[ind] = vsdum;
				if ( vpdum <= material.vp[ind] ) material.vp[ind] = vpdum;
				if ( rhodum <= material.rho[ind] ) material.rho[ind] = rhodum;
			}
			ind++;
		}
		if (comms.rank == MASTER) {
			if (VERBOSE >= 2) {
				printf("MATERIAL PROPERTY :\n");
				printf("number of layers : %d\n",NLAYER);
				for (i=0;i<NLAYER;i++){
					printf("layer %d : depth = %f\n\t\t- rho = %f\n",i,laydep[i],rho0[i]);
					printf("\t\t- vp0 = %f\n",vp0[i]);
					printf("\t\t- vs0 = %f\n",vs0[i]);
					printf("\t\t- mu = %f\n",vs0[i]*vs0[i]*rho0[i]);
					printf("\t\t- lambda = %f\n",vp0[i]*vp0[i]*rho0[i]-2.0*(vs0[i]*vs0[i]*rho0[i]));
				}
			}
		}
	}
	// }}}

	// EXTENSION IN THE CPML LAYERS {{{
	if (dim_model == 3) {
		if (VERBOSE >= 3) printf("%sEXTENSION IN THE CPML LAYERS\n",decal);
		/* 5 faces */
		for ( i = bounds.xmin; i <= bounds.xmax; i++){
			for ( j = bounds.ymin; j <= bounds.ymax; j++){
				for ( k = bounds.zinf-2; k < bounds.zmin; k++){
					ACCESS(material,vp,i,j,k) =  ACCESS(material,vp,i,j,bounds.zmin);
					ACCESS(material,vs,i,j,k) =  ACCESS(material,vs,i,j,bounds.zmin);
					ACCESS(material,rho,i,j,k) =  ACCESS(material,rho,i,j,bounds.zmin);
					ACCESS(material,lam,i,j,k) =  ACCESS(material,lam,i,j,bounds.zmin);
					ACCESS(material,mu,i,j,k) =  ACCESS(material,mu,i,j,bounds.zmin);
				}
			}
		}
		for ( j = bounds.ymin; j <= bounds.ymax; j++){
			for ( k = bounds.zmin; k <= bounds.zmax+1; k++){
				for ( i = bounds.xinf-2; i < bounds.xmin; i++){
					ACCESS(material,vp,i,j,k) = ACCESS(material,vp,bounds.xmin,j,k);
					ACCESS(material,vs,i,j,k) = ACCESS(material,vs,bounds.xmin,j,k);
					ACCESS(material,rho,i,j,k) = ACCESS(material,rho,bounds.xmin,j,k);
					ACCESS(material,lam,i,j,k) = ACCESS(material,lam,bounds.xmin,j,k);
					ACCESS(material,mu,i,j,k) = ACCESS(material,mu,bounds.xmin,j,k);
				}
				for ( i = bounds.xmax+1; i <= bounds.xsup+2; i++){
					ACCESS(material,vp,i,j,k) = ACCESS(material,vp,bounds.xmax,j,k);
					ACCESS(material,vs,i,j,k) = ACCESS(material,vs,bounds.xmax,j,k);
					ACCESS(material,rho,i,j,k) = ACCESS(material,rho,bounds.xmax,j,k);
					ACCESS(material,lam,i,j,k) = ACCESS(material,lam,bounds.xmax,j,k);
					ACCESS(material,mu,i,j,k) = ACCESS(material,mu,bounds.xmax,j,k);
				}
			}
		}
		for ( i = bounds.xmin; i <= bounds.xmax; i++){
			for ( k = bounds.zmin; k <= bounds.zmax+1; k++){
				for ( j = bounds.yinf-2; j < bounds.ymin; j++){
					ACCESS(material,vp,i,j,k) = ACCESS(material,vp,i,bounds.ymin,k);
					ACCESS(material,vs,i,j,k) = ACCESS(material,vs,i,bounds.ymin,k);
					ACCESS(material,rho,i,j,k) = ACCESS(material,rho,i,bounds.ymin,k);
					ACCESS(material,lam,i,j,k) = ACCESS(material,lam,i,bounds.ymin,k);
					ACCESS(material,mu,i,j,k) = ACCESS(material,mu,i,bounds.ymin,k);
				}
				for ( j = bounds.ymax+1; j <= bounds.ysup+2; j++){
					ACCESS(material,vp,i,j,k) = ACCESS(material,vp,i,bounds.ymax,k);
					ACCESS(material,vs,i,j,k) = ACCESS(material,vs,i,bounds.ymax,k);
					ACCESS(material,rho,i,j,k) = ACCESS(material,rho,i,bounds.ymax,k);
					ACCESS(material,lam,i,j,k) = ACCESS(material,lam,i,bounds.ymax,k);
					ACCESS(material,mu,i,j,k) = ACCESS(material,mu,i,bounds.ymax,k);
				}
			}
		}
		/* 8 edges */

		for ( i = bounds.xmin; i <= bounds.xmax; i++){
			for ( j = bounds.yinf-2; j < bounds.ymin; j++){
				for ( k = bounds.zinf-2; k < bounds.zmin; k++){
					ACCESS(material,vp,i,j,k) = ACCESS(material,vp,i,bounds.ymin,bounds.zmin);
					ACCESS(material,vs,i,j,k) = ACCESS(material,vs,i,bounds.ymin,bounds.zmin);
					ACCESS(material,rho,i,j,k) = ACCESS(material,rho,i,bounds.ymin,bounds.zmin);
					ACCESS(material,lam,i,j,k) = ACCESS(material,lam,i,bounds.ymin,bounds.zmin);
					ACCESS(material,mu,i,j,k) = ACCESS(material,mu,i,bounds.ymin,bounds.zmin);
				}
			}
			for ( j = bounds.ymax+1; j <= bounds.ysup+2; j++){
				for ( k = bounds.zinf-2; k < bounds.zmin; k++){
					ACCESS(material,vp,i,j,k) = ACCESS(material,vp,i,bounds.ymax,bounds.zmin);
					ACCESS(material,vs,i,j,k) = ACCESS(material,vs,i,bounds.ymax,bounds.zmin);
					ACCESS(material,rho,i,j,k) = ACCESS(material,rho,i,bounds.ymax,bounds.zmin);
					ACCESS(material,lam,i,j,k) = ACCESS(material,lam,i,bounds.ymax,bounds.zmin);
					ACCESS(material,mu,i,j,k) = ACCESS(material,mu,i,bounds.ymax,bounds.zmin);
				}
			}
		}
		for ( j = bounds.ymin; j <= bounds.ymax; j++){
			for ( i = bounds.xinf-2; i < bounds.xmin; i++){
				for ( k = bounds.zinf-2 ; k < bounds.zmin; k++){
					ACCESS(material,vp,i,j,k) = ACCESS(material,vp,bounds.xmin,j,bounds.zmin);
					ACCESS(material,vs,i,j,k) = ACCESS(material,vs,bounds.xmin,j,bounds.zmin);
					ACCESS(material,rho,i,j,k) = ACCESS(material,rho,bounds.xmin,j,bounds.zmin);
					ACCESS(material,lam,i,j,k) = ACCESS(material,lam,bounds.xmin,j,bounds.zmin);
					ACCESS(material,mu,i,j,k) = ACCESS(material,mu,bounds.xmin,j,bounds.zmin);
				}
			}
			for ( i = bounds.xmax+1; i <= bounds.xsup+2; i++){
				for ( k = bounds.zinf-2; k < bounds.zmin; k++){
					ACCESS(material,vp,i,j,k) = ACCESS(material,vp,bounds.xmax,j,bounds.zmin);
					ACCESS(material,vs,i,j,k) = ACCESS(material,vs,bounds.xmax,j,bounds.zmin);
					ACCESS(material,rho,i,j,k) = ACCESS(material,rho,bounds.xmax,j,bounds.zmin);
					ACCESS(material,lam,i,j,k) = ACCESS(material,lam,bounds.xmax,j,bounds.zmin);
					ACCESS(material,mu,i,j,k) = ACCESS(material,mu,bounds.xmax,j,bounds.zmin);
				}
			}
		}
		for ( k = bounds.zmin; k <= bounds.zmax+1; k++){
			for ( j = bounds.yinf-2; j < bounds.ymin; j++){
				for ( i = bounds.xinf-2; i < bounds.xmin; i++){
					ACCESS(material,vp,i,j,k) = ACCESS(material,vp,bounds.xmin,bounds.ymin,k);
					ACCESS(material,vs,i,j,k) = ACCESS(material,vs,bounds.xmin,bounds.ymin,k);
					ACCESS(material,rho,i,j,k) = ACCESS(material,rho,bounds.xmin,bounds.ymin,k);
					ACCESS(material,lam,i,j,k) = ACCESS(material,lam,bounds.xmin,bounds.ymin,k);
					ACCESS(material,mu,i,j,k) = ACCESS(material,mu,bounds.xmin,bounds.ymin,k);
				}
				for ( i = bounds.xmax+1; i <= bounds.xsup+2; i++){
					ACCESS(material,vp,i,j,k) = ACCESS(material,vp,bounds.xmax,bounds.ymin,k);
					ACCESS(material,vs,i,j,k) = ACCESS(material,vs,bounds.xmax,bounds.ymin,k);
					ACCESS(material,rho,i,j,k) = ACCESS(material,rho,bounds.xmax,bounds.ymin,k);
					ACCESS(material,lam,i,j,k) = ACCESS(material,lam,bounds.xmax,bounds.ymin,k);
					ACCESS(material,mu,i,j,k) = ACCESS(material,mu,bounds.xmax,bounds.ymin,k);
				}
			}
			for ( j = bounds.ymax+1; j <= bounds.ysup+2; j++){
				for ( i = bounds.xinf-2; i < bounds.xmin; i++){
					ACCESS(material,vp,i,j,k) = ACCESS(material,vp,bounds.xmin,bounds.ymax,k);
					ACCESS(material,vs,i,j,k) = ACCESS(material,vs,bounds.xmin,bounds.ymax,k);
					ACCESS(material,rho,i,j,k) = ACCESS(material,rho,bounds.xmin,bounds.ymax,k);
					ACCESS(material,lam,i,j,k) = ACCESS(material,lam,bounds.xmin,bounds.ymax,k);
					ACCESS(material,mu,i,j,k) = ACCESS(material,mu,bounds.xmin,bounds.ymax,k);
				}
				for ( i = bounds.xmax+1; i <= bounds.xsup+2; i++){
					ACCESS(material,vp,i,j,k) = ACCESS(material,vp,bounds.xmax,bounds.ymax,k);
					ACCESS(material,vs,i,j,k) = ACCESS(material,vs,bounds.xmax,bounds.ymax,k);
					ACCESS(material,rho,i,j,k) = ACCESS(material,rho,bounds.xmax,bounds.ymax,k);
					ACCESS(material,lam,i,j,k) = ACCESS(material,lam,bounds.xmax,bounds.ymax,k);
					ACCESS(material,mu,i,j,k) = ACCESS(material,mu,bounds.xmax,bounds.ymax,k);
				}
			}
		}

		/* 4 corners */

		for ( i = bounds.xinf-2; i < bounds.xmin; i++){
			for ( j = bounds.yinf-2; j < bounds.ymin; j++){
				for ( k = bounds.zinf-2; k < bounds.zmin; k++){
					ACCESS(material,vp,i,j,k) = ACCESS(material,vp,bounds.xmin,bounds.ymin,bounds.zmin);
					ACCESS(material,vs,i,j,k) = ACCESS(material,vs,bounds.xmin,bounds.ymin,bounds.zmin);
					ACCESS(material,rho,i,j,k) = ACCESS(material,rho,bounds.xmin,bounds.ymin,bounds.zmin);
					ACCESS(material,lam,i,j,k) = ACCESS(material,lam,bounds.xmin,bounds.ymin,bounds.zmin);
					ACCESS(material,mu,i,j,k) = ACCESS(material,mu,bounds.xmin,bounds.ymin,bounds.zmin);
				}
			}
			for ( j = bounds.ymax+1; j <= bounds.ysup+2; j++){
				for ( k = bounds.zinf-2; k < bounds.zmin; k++){
					ACCESS(material,vp,i,j,k) = ACCESS(material,vp,bounds.xmin,bounds.ymax,bounds.zmin);
					ACCESS(material,vs,i,j,k) = ACCESS(material,vs,bounds.xmin,bounds.ymax,bounds.zmin);
					ACCESS(material,rho,i,j,k) = ACCESS(material,rho,bounds.xmin,bounds.ymax,bounds.zmin);
					ACCESS(material,lam,i,j,k) = ACCESS(material,lam,bounds.xmin,bounds.ymax,bounds.zmin);
					ACCESS(material,mu,i,j,k) = ACCESS(material,mu,bounds.xmin,bounds.ymax,bounds.zmin);
				}
			}
		}
		for ( i = bounds.xmax+1; i <= bounds.xsup+2; i++){
			for ( j = bounds.yinf-2; j < bounds.ymin; j++){
				for ( k = bounds.zinf-2; k < bounds.zmin; k++){
					ACCESS(material,vp,i,j,k) = ACCESS(material,vp,bounds.xmax,bounds.ymin,bounds.zmin);
					ACCESS(material,vs,i,j,k) = ACCESS(material,vs,bounds.xmax,bounds.ymin,bounds.zmin);
					ACCESS(material,rho,i,j,k) = ACCESS(material,rho,bounds.xmax,bounds.ymin,bounds.zmin);
					ACCESS(material,lam,i,j,k) = ACCESS(material,lam,bounds.xmax,bounds.ymin,bounds.zmin);
					ACCESS(material,mu,i,j,k) = ACCESS(material,mu,bounds.xmax,bounds.ymin,bounds.zmin);
				}
			}
			for ( j = bounds.ymax+1; j <= bounds.ysup+2; j++){
				for ( k = bounds.zinf-2; k < bounds.zmin; k++){
					ACCESS(material,vp,i,j,k) = ACCESS(material,vp,bounds.xmax,bounds.ymax,bounds.zmin);
					ACCESS(material,vs,i,j,k) = ACCESS(material,vs,bounds.xmax,bounds.ymax,bounds.zmin);
					ACCESS(material,rho,i,j,k) = ACCESS(material,rho,bounds.xmax,bounds.ymax,bounds.zmin);
					ACCESS(material,lam,i,j,k) = ACCESS(material,lam,bounds.xmax,bounds.ymax,bounds.zmin);
					ACCESS(material,mu,i,j,k) = ACCESS(material,mu,bounds.xmax,bounds.ymax,bounds.zmin);
				}
			}
		}
	}
	// }}}

	// DEFINITION OF THE VECTORS USED IN THE CPML FORMULATION {{{
	if (VERBOSE >= 3) printf("%sDEFINITION OF THE VECTORS USED IN THE CPML FORMULATION\n",decal);

	/* Ariane : we define the values of dump, alpha and kappa in the CPML as Komatitsch did */
	dumpx = fvector(bounds.xinf, bounds.xsup);
	kappax = fvector(bounds.xinf, bounds.xsup);
	alphax = fvector(bounds.xinf, bounds.xsup);
	dumpx2 = fvector(bounds.xinf, bounds.xsup);
	kappax2 = fvector(bounds.xinf, bounds.xsup);
	alphax2 = fvector(bounds.xinf, bounds.xsup);

	dumpy = fvector(bounds.yinf, bounds.ysup);
	kappay = fvector(bounds.yinf, bounds.ysup);
	alphay = fvector(bounds.yinf, bounds.ysup);
	dumpy2 = fvector(bounds.yinf, bounds.ysup);
	kappay2 = fvector(bounds.yinf, bounds.ysup);
	alphay2 = fvector(bounds.yinf, bounds.ysup);

	dumpz = fvector(bounds.zinf, bounds.zmax);
	kappaz = fvector(bounds.zinf, bounds.zmax);
	alphaz = fvector(bounds.zinf, bounds.zmax);
	dumpz2 = fvector(bounds.zinf, bounds.zmax);
	kappaz2 = fvector(bounds.zinf, bounds.zmax);
	alphaz2 = fvector(bounds.zinf, bounds.zmax);

	for ( i = bounds.xinf; i <= bounds.xsup; i++){
		dumpx[i] = 0.0;
		dumpx2[i] = 0.0;
		kappax[i] = 1.0;
		kappax2[i] = 1.0;
		alphax[i] = 0.0;
		alphax2[i] = 0.0;
	}

	for ( j = bounds.yinf; j <= bounds.ysup; j++){
		dumpy[j] = 0.0;
		dumpy2[j] = 0.0;
		kappay[j] = 1.0;
		kappay2[j] = 1.0;
		alphay[j] = 0.0;
		alphay2[j] = 0.0;
	}

	for ( k = bounds.zinf; k <=bounds.zmax; k++){
		dumpz[k] = 0.0;
		dumpz2[k] = 0.0;
		kappaz[k] = 1.0;
		kappaz2[k] = 1.0;
		alphaz[k] = 0.0;
		alphaz2[k] = 0.0;
	}

	/* For the x axis */

	xoriginleft = bounds.xmin*ds;
	xoriginright = bounds.xmax*ds;

	for ( i = bounds.xinf; i <= bounds.xsup; i++){

		xval = ds * i;

		/* For the left side */

		abscissa_in_PML = xoriginleft - xval;
		if(abscissa_in_PML >= 0.0){
			abscissa_normalized = abscissa_in_PML / (delta * ds);
			dumpx[i] = dump0 * powf(abscissa_normalized,NPOWER);
			kappax[i] = 1.0 + (kappa0 - 1.0) * powf(abscissa_normalized,NPOWER);
			alphax[i] = alpha0 * (1.0 - abscissa_normalized);
		}

		abscissa_in_PML = xoriginleft - (xval + ds/2.0);
		if(abscissa_in_PML >= 0.0){
			abscissa_normalized = abscissa_in_PML / (delta * ds);
			dumpx2[i] = dump0 * powf(abscissa_normalized,NPOWER);
			kappax2[i] = 1.0 + (kappa0 - 1.0) * powf(abscissa_normalized,NPOWER);
			alphax2[i] = alpha0 * (1.0 - abscissa_normalized);
		}

		/* For the right side */

		abscissa_in_PML = xval - xoriginright;
		if(abscissa_in_PML >= 0.0){
			abscissa_normalized = abscissa_in_PML / (delta * ds);
			dumpx[i] = dump0 * powf(abscissa_normalized,NPOWER);
			kappax[i] = 1.0 + (kappa0 - 1.0) * powf(abscissa_normalized,NPOWER);
			alphax[i] = alpha0 * (1.0 - abscissa_normalized);
		}

		abscissa_in_PML = xval + ds/2.0 - xoriginright;
		if(abscissa_in_PML >= 0.0){
			abscissa_normalized = abscissa_in_PML / (delta * ds);
			dumpx2[i] = dump0 * powf(abscissa_normalized,NPOWER);
			kappax2[i] = 1.0 + (kappa0 - 1.0) * powf(abscissa_normalized,NPOWER);
			alphax2[i] = alpha0 * (1.0 - abscissa_normalized);
		}

		if(alphax[i] < 0.0) alphax[i] = 0.0;
		if(alphax2[i] < 0.0) alphax2[i] = 0.0;

	}

	/* For the y axis */

	yoriginfront = bounds.ymin*ds;
	yoriginback = bounds.ymax*ds;

	for ( j = bounds.yinf; j <= bounds.ysup; j++){

		yval = ds * j;

		/* For the front side */

		abscissa_in_PML = yoriginfront - yval;
		if(abscissa_in_PML >= 0.0){
			abscissa_normalized = abscissa_in_PML / (delta * ds);
			dumpy[j] = dump0 * powf(abscissa_normalized,NPOWER);
			kappay[j] = 1.0 + (kappa0 - 1.0) * powf(abscissa_normalized,NPOWER);
			alphay[j] = alpha0 * (1.0 - abscissa_normalized);
		}

		abscissa_in_PML = yoriginfront - (yval + ds/2.0);
		if(abscissa_in_PML >= 0.0){
			abscissa_normalized = abscissa_in_PML / (delta * ds);
			dumpy2[j] = dump0 * powf(abscissa_normalized,NPOWER);
			kappay2[j] = 1.0 + (kappa0 - 1.0) * powf(abscissa_normalized,NPOWER);
			alphay2[j] = alpha0 * (1.0 - abscissa_normalized);
		}

		/* For the back side */

		abscissa_in_PML = yval - yoriginback;
		if(abscissa_in_PML >= 0.0){
			abscissa_normalized = abscissa_in_PML / (delta * ds);
			dumpy[j] = dump0 * powf(abscissa_normalized,NPOWER);
			kappay[j] = 1.0 + (kappa0 - 1.0) * powf(abscissa_normalized,NPOWER);
			alphay[j] = alpha0 * (1.0 - abscissa_normalized);
		}

		abscissa_in_PML = yval + ds/2.0 - yoriginback;
		if(abscissa_in_PML >= 0.0){
			abscissa_normalized = abscissa_in_PML / (delta * ds);
			dumpy2[j] = dump0 * powf(abscissa_normalized,NPOWER);
			kappay2[j] = 1.0 + (kappa0 - 1.0) * powf(abscissa_normalized,NPOWER);
			alphay2[j] = alpha0 * (1.0 - abscissa_normalized);
		}

		if(alphay[j] < 0.0) alphay[j] = 0.0;
		if(alphay2[j] < 0.0) alphay2[j] = 0.0;

	}

	/* For the z axis */

	zoriginbottom = bounds.zmin*ds;

	for ( k = bounds.zinf; k <= bounds.zmax; k++){

		zval = ds * k;

		/* For the bottom side */

		abscissa_in_PML = zoriginbottom - zval;
		if(abscissa_in_PML >= 0.0){
			abscissa_normalized = abscissa_in_PML / (delta * ds);
			dumpz[k] = dump0 * powf(abscissa_normalized,NPOWER);
			kappaz[k] = 1.0 + (kappa0 - 1.0) * powf(abscissa_normalized,NPOWER);
			alphaz[k] = alpha0 * (1.0 - abscissa_normalized);
		}

		abscissa_in_PML = zoriginbottom - (zval + ds/2.0);
		if(abscissa_in_PML >= 0.0){
			abscissa_normalized = abscissa_in_PML / (delta * ds);
			dumpz2[k] = dump0 * powf(abscissa_normalized,NPOWER);
			kappaz2[k] = 1.0 + (kappa0 - 1.0) * powf(abscissa_normalized,NPOWER);
			alphaz2[k] = alpha0 * (1.0 - abscissa_normalized);
		}

		if(alphaz[k] < 0.0) alphaz[k] = 0.0;
		if(alphaz2[k] < 0.0) alphaz2[k] = 0.0;

	}
	/* End Modif Ariane */
	// }}}

	// INITIALIZATION OF CUDA DATA STRUCTURES {{{
	if (VERBOSE >= 3) printf("%sINITIALIZATION OF CUDA DATA STRUCTURES\n",decal);
	// allocation of GPU arrays
	memory_used += allocate_arrays_GPU(&d_veloc, &d_stress, &d_material, &d_force, &bounds, dim_model);

	copy_material_to_GPU(&d_material, &material, dim_model);

	assert (material.offset_k = 2*d_veloc.pitch*(bounds.size_y+4) + 2*d_veloc.pitch + ALIGN_SEGMENT);

	// create indirection for CPML points
	memory_used += create_CPML_indirection(&d_npml_tab, bounds, &npmlv, XMAX-XMIN + 2*delta + 1, YMAX-YMIN + 2*delta + 1, -ZMIN + delta + 1);

	if (comms.rank == MASTER && VERBOSE >= 2) {
		printf("%d CPML points on a total of %d points : %.2f %\n",npmlv, bounds.size_x*bounds.size_y*bounds.size_z, (float)npmlv/(float)(bounds.size_x*bounds.size_y*bounds.size_z)*100.);
	}

	// this to avoid allocation of size 0 in case of no CPML
	if (npmlv == 0) npmlv++;

	// allocation of CPML data structure
	memory_used += allocate_CPML_arrays(npmlv, &d_phivxx, &d_phivxy, &d_phivxz, &d_phivyx, &d_phivyy, &d_phivyz, &d_phivzx, &d_phivzy, &d_phivzz, 
			&d_phitxxx, &d_phitxyy, &d_phitxzz, &d_phitxyx, &d_phityyy, &d_phityzz, &d_phitxzx, &d_phityzy, &d_phitzzz);

	memory_used += allocate_CPML_vectors(bounds, 
			&d_dumpx, &d_alphax, &d_kappax, &d_dumpx2, &d_alphax2, &d_kappax2,
			&d_dumpy, &d_alphay, &d_kappay, &d_dumpy2, &d_alphay2, &d_kappay2,
			&d_dumpz, &d_alphaz, &d_kappaz, &d_dumpz2, &d_alphaz2, &d_kappaz2,
			&(dumpx[bounds.xinf]), &(alphax[bounds.xinf]), &(kappax[bounds.xinf]), &(dumpx2[bounds.xinf]), &(alphax2[bounds.xinf]), &(kappax2[bounds.xinf]), 
			&(dumpy[bounds.yinf]), &(alphay[bounds.yinf]), &(kappay[bounds.yinf]), &(dumpy2[bounds.yinf]), &(alphay2[bounds.yinf]), &(kappay2[bounds.yinf]), 
			&(dumpz[bounds.zinf]), &(alphaz[bounds.zinf]), &(kappaz[bounds.zinf]), &(dumpz2[bounds.zinf]), &(alphaz2[bounds.zinf]), &(kappaz2[bounds.zinf])); 

	// allocation of MPI buffers
#ifdef USE_MPI/*{{{*/
	memory_used += allocate_MPI_buffers(&comms);
#endif/*}}}*/

	// check for error
	print_err(cudaGetLastError(), "cudaGetLastError after all allocations and texture binding");
	if (comms.rank == MASTER && VERBOSE >= 1) {
		if (VERBOSE >= 2)
		{	// print kernel params	
			printf("CUDA KERNEL PARAMS : \n\t\tGRID SIZE : %d,%d\n\t\tBLOCK SIZE : %d,%d\nDOMAIN SIZE : {%d, %d}\n",(int)ceilf((float)(bounds.size_x)/(float)NPPDX), (int)ceilf((float)(bounds.size_y)/(float)NPPDY), NPPDX, NPPDY, bounds.size_x, bounds.size_y);
		}
		printf("GPU memory usage : %f Mo\n",memory_used/(1024.*1024.));
	}
	// }}}

	// TIMELOOP {{{
	if (VERBOSE >= 3) printf("%sTIMELOOP\n",decal);
	if (comms.rank == MASTER) printf("\nComputing timeloop ...\n");
#ifdef USE_MPI
	// bitvector |...0.0.0.LY.FY.LX.FX| FX = first x ...
	int position = comms.first_x + comms.last_x*2 + comms.first_y*4 + comms.last_y*8;
#else
	int position = 15;
#endif
	for ( l = 1; l <= TMAX; l++ ){

		// INCREMENT TIME STEP {{{
		if (comms.rank == MASTER && VERBOSE >= 1) {
			printf("                               \rTIME step ... %i / %i \r", l,TMAX );
			//printf("TIME step ... %i / %i \n", l,TMAX );
			fflush(stdout);
		}
		time_tl_start = clock();
		time = dt * l;
		it = (int) (time/dtbiem);
		// }}}

		// INCREMENT SEISMIC MOMENT WITH MPI COMMS {{{
		if (VERBOSE >= 3) printf("%sINCREMENT SEISMIC MOMENT\n",decal);
		if ( it < IDUR ){
			time_su_start = clock();
			int imin = INT_MAX;
			int imax = INT_MIN;
			int jmin = INT_MAX;
			int jmax = INT_MIN;
			int kmin = INT_MAX;
			int kmax = INT_MIN;

			for ( is = 0; is < ISRC; is++ ){/*{{{*/
				if ( insrc[is] ){
					mo = vel[is][it] * dt;
					pxx = (float)dradxx((double)strike[is], (double)dip[is], (double)rake[is]);
					pyy = (float)dradyy((double)strike[is], (double)dip[is], (double)rake[is]);
					pzz = (float)dradzz((double)strike[is], (double)dip[is], (double)rake[is]);
					pxy = (float)dradxy((double)strike[is], (double)dip[is], (double)rake[is]);
					pyz = (float)dradyz((double)strike[is], (double)dip[is], (double)rake[is]);
					pxz = (float)dradxz((double)strike[is], (double)dip[is], (double)rake[is]);

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

						// define the volume to be copied on the GPU
						if (i<imin) imin = i;
						if (j<jmin) jmin = j;
						if (k<kmin) kmin = k;

						if (i>imax) imax = i;
						if (j>jmax) jmax = j;
						if (k>kmax) kmax = k;

						if (i < bounds.xinf-4 || i > bounds.xsup+4 || j < bounds.yinf-4 || j > bounds.ysup+4 || k < bounds.zinf-4 || k > bounds.zsup+4) {
							printf("problem with source update : overflow !!!\nexiting\n\n");
#ifdef USE_MPI/**/
							MPI_Abort(MPI_COMM_WORLD, 1);
#else/**/
							exit(1);
#endif
						}

						// update source array on host
						ACCESS(force,x,i+1,j,k) += 0.5 * mo * pxx * weight;
						ACCESS(force,x,i-1,j,k) -= 0.5 * mo * pxx * weight;

						ACCESS(force,x,i,j+1,k) += 0.5 * mo * pxy * weight;
						ACCESS(force,x,i,j-1,k) -= 0.5 * mo * pxy * weight;

						ACCESS(force,x,i,j,k+1) += 0.5 * mo * pxz * weight;
						ACCESS(force,x,i,j,k-1) -= 0.5 * mo * pxz * weight;

						ACCESS(force,y,i,j,k)     += 0.5 * mo * pxy * weight;
						ACCESS(force,y,i,j-1,k)   += 0.5 * mo * pxy * weight;
						ACCESS(force,y,i-1,j,k)   -= 0.5 * mo * pxy * weight;
						ACCESS(force,y,i-1,j-1,k) -= 0.5 * mo * pxy * weight;

						ACCESS(force,y,i,j,k)     += 0.5 * mo * pyy * weight;
						ACCESS(force,y,i-1,j,k)   += 0.5 * mo * pyy * weight;
						ACCESS(force,y,i,j-1,k)   -= 0.5 * mo * pyy * weight;
						ACCESS(force,y,i-1,j-1,k) -= 0.5 * mo * pyy * weight;

						ACCESS(force,y,i,j,k+1)     += 0.125 * mo * pyz * weight;
						ACCESS(force,y,i,j-1,k+1)   += 0.125 * mo * pyz * weight;
						ACCESS(force,y,i-1,j,k+1)   += 0.125 * mo * pyz * weight;
						ACCESS(force,y,i-1,j-1,k+1) += 0.125 * mo * pyz * weight;
						ACCESS(force,y,i,j,k-1)     -= 0.125 * mo * pyz * weight;
						ACCESS(force,y,i,j-1,k-1)   -= 0.125 * mo * pyz * weight;
						ACCESS(force,y,i-1,j,k-1)   -= 0.125 * mo * pyz * weight;
						ACCESS(force,y,i-1,j-1,k-1) -= 0.125 * mo * pyz * weight;

						ACCESS(force,z,i,j,k)     += 0.5 * mo * pxz * weight;
						ACCESS(force,z,i,j,k-1)   += 0.5 * mo * pxz * weight;
						ACCESS(force,z,i-1,j,k)   -= 0.5 * mo * pxz * weight;
						ACCESS(force,z,i-1,j,k-1) -= 0.5 * mo * pxz * weight;

						ACCESS(force,z,i,j+1,k)     += 0.125 * mo * pyz * weight;
						ACCESS(force,z,i,j+1,k-1)   += 0.125 * mo * pyz * weight;
						ACCESS(force,z,i-1,j+1,k)   += 0.125 * mo * pyz * weight;
						ACCESS(force,z,i-1,j+1,k-1) += 0.125 * mo * pyz * weight;
						ACCESS(force,z,i,j-1,k)     -= 0.125 * mo * pyz * weight;
						ACCESS(force,z,i,j-1,k-1)   -= 0.125 * mo * pyz * weight;
						ACCESS(force,z,i-1,j-1,k)   -= 0.125 * mo * pyz * weight;
						ACCESS(force,z,i-1,j-1,k-1) -= 0.125 * mo * pyz * weight;

						ACCESS(force,z,i,j,k)     += 0.5 * mo * pzz * weight;
						ACCESS(force,z,i-1,j,k)   += 0.5 * mo * pzz * weight;
						ACCESS(force,z,i,j,k-1)   -= 0.5 * mo * pzz * weight;
						ACCESS(force,z,i-1,j,k-1) -= 0.5 * mo * pzz * weight;
					} /* end of iw (weighting) */
					// extend the volume to copy to its maximum dimension.

					if (imin > bounds.xinf) imin--;
					if (jmin > bounds.yinf) jmin--;
					if (kmin > bounds.zinf) kmin--;
					if (imax < bounds.xsup) imax++;
					if (jmax < bounds.ysup) jmax++;
					if (kmax < bounds.zsup) kmax++;

					//	printf("volume impacted by source %d : \n\tX : %d - %d\n\tY : %d - %d\n\tZ : %d - %d\n\n",is+1,imin-bounds.xinf,imax-bounds.xinf,jmin-bounds.yinf,jmax-bounds.yinf,kmin-bounds.zinf,kmax-bounds.zinf);
					// copy of the volume updated on the device
					// good for few sources, bad for many.
					if (few_sources) {
						time_tr_start = clock();
						for (k = kmin; k <= kmax; k++) {
							for (j = jmin; j <= jmax; j++) {
								print_err(cudaMemcpy((void*) &(ACCESS(d_force,x,imin,j,k)), &(ACCESS(force,x,imin,j,k)), (imax-imin+1)*sizeof(float), cudaMemcpyHostToDevice),"cudaMemcpy forcex -> d_forcex (part)");
								print_err(cudaMemcpy((void*) &(ACCESS(d_force,y,imin,j,k)), &(ACCESS(force,y,imin,j,k)), (imax-imin+1)*sizeof(float), cudaMemcpyHostToDevice),"cudaMemcpy forcex -> d_forcex (part)");
								print_err(cudaMemcpy((void*) &(ACCESS(d_force,z,imin,j,k)), &(ACCESS(force,z,imin,j,k)), (imax-imin+1)*sizeof(float), cudaMemcpyHostToDevice),"cudaMemcpy forcex -> d_forcex (part)");
							}
						}
						time_tr_end = clock();
					}
				} /* end of insrc */
			} /* end of is (each source) *//*}}}*/
			if (!few_sources) {
				time_tr_start = clock();
				print_err(cudaMemcpy((void*) d_force.x, force.x, force.pitch*force.height*force.depth*sizeof(float), cudaMemcpyHostToDevice),"cudaMemcpy forcex -> d_forcex (all)");
				print_err(cudaMemcpy((void*) d_force.y, force.y, force.pitch*force.height*force.depth*sizeof(float), cudaMemcpyHostToDevice),"cudaMemcpy forcey -> d_forcey (all)");
				print_err(cudaMemcpy((void*) d_force.z, force.z, force.pitch*force.height*force.depth*sizeof(float), cudaMemcpyHostToDevice),"cudaMemcpy forcez -> d_forcez (all)");
				time_tr_end = clock();
			}
			time_su_end = clock();
		} else {
			time_su_start = time_su_end = time_tr_start = time_tr_end = (clock_t)0.f;
		}
		// }}} 

		// COMPUTE STRESS WITH MPI COMMS {{{
		if (VERBOSE >= 3) printf("%sCOMPUTE STRESS\n",decal);
		time_k1_start = clock();
		for (int iloop = 0, compute_external = 1 ; iloop<2; iloop++, compute_external--) {
			if (dim_model == 1) {
				computeStress1D (	d_stress.xx + d_stress.offset_k, 
						d_stress.yy + d_stress.offset_k, 
						d_stress.zz + d_stress.offset_k, 
						d_stress.xy + d_stress.offset_k, 
						d_stress.xz + d_stress.offset_k, 
						d_stress.yz + d_stress.offset_k, 
						d_veloc.x + d_veloc.offset_k, 
						d_veloc.y + d_veloc.offset_k, 
						d_veloc.z + d_veloc.offset_k, 
						d_npml_tab, d_phivxx, d_phivxy, d_phivxz, d_phivyx, d_phivyy, d_phivyz, d_phivzx, d_phivzy, d_phivzz,
						bounds.size_x, bounds.size_y, bounds.size_z, 
						d_veloc.pitch, d_veloc.height, d_veloc.depth, 
						(float)ds, (float)dt, delta, compute_external,
						(int)ceilf((float)(bounds.size_x)/(float)NPPDX), (int)ceilf((float)(bounds.size_y)/(float)NPPDY), 1, 
						NPPDX,NPPDY,1,position);
			} else {
				computeStress3D (	d_stress.xx + d_stress.offset_k, 
						d_stress.yy + d_stress.offset_k, 
						d_stress.zz + d_stress.offset_k, 
						d_stress.xy + d_stress.offset_k, 
						d_stress.xz + d_stress.offset_k, 
						d_stress.yz + d_stress.offset_k, 
						d_veloc.x + d_veloc.offset_k, 
						d_veloc.y + d_veloc.offset_k, 
						d_veloc.z + d_veloc.offset_k, 
						d_npml_tab, d_phivxx, d_phivxy, d_phivxz, d_phivyx, d_phivyy, d_phivyz, d_phivzx, d_phivzy, d_phivzz,
						d_material.mu + d_material.offset_k,
						d_material.lam + d_material.offset_k,
						d_material.vp + d_material.offset_k,
						bounds.size_x, bounds.size_y, bounds.size_z, 
						d_veloc.pitch, d_veloc.height, d_veloc.depth, 
						(float)ds, (float)dt, delta, compute_external,
						(int)ceilf((float)(bounds.size_x)/(float)NPPDX), (int)ceilf((float)(bounds.size_y)/(float)NPPDY), 1, 
						NPPDX,NPPDY,1,position);
			}

			// MPI communications {{{
#ifdef USE_MPI/**/
			if (iloop == 0) {
				maxsize = MAX(bounds.size_x,bounds.size_y);
#ifdef SLOW_BUFFER_TIMING/**/
				cudaThreadSynchronize();
				time_prep_buf_k1_start = clock();
#endif/**/
				getBuffersStress(	comms.d_buff_x_min, comms.d_buff_x_max, comms.d_buff_y_min, comms.d_buff_y_max,
						bounds.size_x, bounds.size_y, bounds.size_z,
						d_stress.pitch, d_stress.height, d_stress.depth, 
						d_stress.xx + d_stress.offset_k,
						d_stress.yy + d_stress.offset_k,
						d_stress.zz + d_stress.offset_k,
						d_stress.xy + d_stress.offset_k,
						d_stress.xz + d_stress.offset_k,
						d_stress.yz + d_stress.offset_k,
						comms.size_buffer_x, comms.size_buffer_y, 
						(int)ceilf((float)maxsize/(float)BLOCKDIMX),bounds.size_z,4,
						BLOCKDIMX,2,1, position);
#ifdef SLOW_BUFFER_TIMING/**/
				cudaThreadSynchronize();
				time_prep_buf_k1_end = clock();
#endif/**/
				// copy buffers device -> host
				print_err(cudaMemcpy((void*)comms.buff_x_min_s, comms.d_buff_x_min, 6*comms.size_buffer_x*sizeof(float),cudaMemcpyDeviceToHost),"cudaMemcpy xmin stress buffers D->H");
				print_err(cudaMemcpy((void*)comms.buff_x_max_s, comms.d_buff_x_max, 6*comms.size_buffer_x*sizeof(float),cudaMemcpyDeviceToHost),"cudaMemcpy xmax stress buffers D->H");
				print_err(cudaMemcpy((void*)comms.buff_y_min_s, comms.d_buff_y_min, 6*comms.size_buffer_y*sizeof(float),cudaMemcpyDeviceToHost),"cudaMemcpy ymin stress buffers D->H");
				print_err(cudaMemcpy((void*)comms.buff_y_max_s, comms.d_buff_y_max, 6*comms.size_buffer_y*sizeof(float),cudaMemcpyDeviceToHost),"cudaMemcpy ymax stress buffers D->H");
				// start non blocking communications
				// SEND
				MPI_Isend( comms.buff_x_min_s, 6*comms.size_buffer_x, MPI_FLOAT, comms.recv_xmin, 0, MPI_COMM_WORLD, &comms.array_req_send[0]);
				MPI_Isend( comms.buff_x_max_s, 6*comms.size_buffer_x, MPI_FLOAT, comms.recv_xmax, 0, MPI_COMM_WORLD, &comms.array_req_send[1]);
				MPI_Isend( comms.buff_y_min_s, 6*comms.size_buffer_y, MPI_FLOAT, comms.recv_ymin, 0, MPI_COMM_WORLD, &comms.array_req_send[2]);
				MPI_Isend( comms.buff_y_max_s, 6*comms.size_buffer_y, MPI_FLOAT, comms.recv_ymax, 0, MPI_COMM_WORLD, &comms.array_req_send[3]);
				// RECV
				MPI_Irecv( comms.buff_x_min_r, 6*comms.size_buffer_x, MPI_FLOAT, comms.send_xmin, 0, MPI_COMM_WORLD, &comms.array_req_recv[0]);
				MPI_Irecv( comms.buff_x_max_r, 6*comms.size_buffer_x, MPI_FLOAT, comms.send_xmax, 0, MPI_COMM_WORLD, &comms.array_req_recv[1]);
				MPI_Irecv( comms.buff_y_min_r, 6*comms.size_buffer_y, MPI_FLOAT, comms.send_ymin, 0, MPI_COMM_WORLD, &comms.array_req_recv[2]);
				MPI_Irecv( comms.buff_y_max_r, 6*comms.size_buffer_y, MPI_FLOAT, comms.send_ymax, 0, MPI_COMM_WORLD, &comms.array_req_recv[3]);
			} else {
				// wait until the end of communications
				time_wait_k1_start = clock();
				MPI_Waitall (4, comms.array_req_recv, comms.array_of_status);
				MPI_Waitall (4, comms.array_req_send, comms.array_of_status);
				time_wait_k1_end = clock();
				// copy buffers host -> device
				print_err(cudaMemcpy((void*)comms.d_buff_x_min, comms.buff_x_min_r, 6*comms.size_buffer_x*sizeof(float),cudaMemcpyHostToDevice),"cudaMemcpy xmin stress buffers H->D");
				print_err(cudaMemcpy((void*)comms.d_buff_x_max, comms.buff_x_max_r, 6*comms.size_buffer_x*sizeof(float),cudaMemcpyHostToDevice),"cudaMemcpy xmax stress buffers H->D");
				print_err(cudaMemcpy((void*)comms.d_buff_y_min, comms.buff_y_min_r, 6*comms.size_buffer_y*sizeof(float),cudaMemcpyHostToDevice),"cudaMemcpy ymin stress buffers H->D");
				print_err(cudaMemcpy((void*)comms.d_buff_y_max, comms.buff_y_max_r, 6*comms.size_buffer_y*sizeof(float),cudaMemcpyHostToDevice),"cudaMemcpy ymax stress buffers H->D");
				// update arrays with buffers (kernel)
				maxsize = MAX(bounds.size_x,bounds.size_y);
#ifdef SLOW_BUFFER_TIMING/**/
				cudaThreadSynchronize();
				time_update_buf_k1_start = clock();
#endif/**/
				updateHaloStress(	comms.d_buff_x_min, comms.d_buff_x_max, comms.d_buff_y_min, comms.d_buff_y_max,
						bounds.size_x, bounds.size_y, bounds.size_z,
						d_stress.pitch, d_stress.height, d_stress.depth, 
						d_stress.xx + d_stress.offset_k,
						d_stress.yy + d_stress.offset_k,
						d_stress.zz + d_stress.offset_k,
						d_stress.xy + d_stress.offset_k,
						d_stress.xz + d_stress.offset_k,
						d_stress.yz + d_stress.offset_k,
						comms.size_buffer_x, comms.size_buffer_y, 
						(int)ceilf((float)maxsize/(float)BLOCKDIMX),bounds.size_z,4, 
						BLOCKDIMX,2,1, position);
#ifdef SLOW_BUFFER_TIMING/**/
				cudaThreadSynchronize();
				time_update_buf_k1_end = clock();
#endif/**/
			}
#endif/**/
			// }}}
		}
		cudaThreadSynchronize();
		time_k1_end = clock();
		// }}}

		// COMPUTE VELOCITY WITH MPI COMMS {{{
		if (VERBOSE >= 3) printf("%sCOMPUTE VELOCITY\n",decal);
		time_k2_start = clock();
		for (int iloop = 0, compute_external=1; iloop<2; iloop++, compute_external--) {
			if (dim_model == 1) {
				computeVeloc1D (	d_stress.xx + d_stress.offset_k, 
						d_stress.yy + d_stress.offset_k, 
						d_stress.zz + d_stress.offset_k, 
						d_stress.xy + d_stress.offset_k, 
						d_stress.xz + d_stress.offset_k, 
						d_stress.yz + d_stress.offset_k, 
						d_veloc.x + d_veloc.offset_k, 
						d_veloc.y + d_veloc.offset_k, 
						d_veloc.z + d_veloc.offset_k, 
						d_force.x + d_veloc.offset_k,
						d_force.y + d_veloc.offset_k,
						d_force.z + d_veloc.offset_k,
						d_npml_tab, d_phitxxx, d_phitxyy, d_phitxzz, d_phitxyx, d_phityyy, d_phityzz, d_phitxzx, d_phityzy, d_phitzzz,
						bounds.size_x, bounds.size_y, bounds.size_z, 
						d_veloc.pitch, bounds.size_y+4, bounds.size_z+4, 
						(float)ds, (float)dt, delta, compute_external, 
						(int)ceilf((float)(bounds.size_x)/(float)NPPDX), (int)ceilf((float)(bounds.size_y)/(float)NPPDY),1, 
						NPPDX_K2,NPPDY_K2,1,position);
			} else {
				computeVeloc3D (	d_stress.xx + d_stress.offset_k, 
						d_stress.yy + d_stress.offset_k, 
						d_stress.zz + d_stress.offset_k, 
						d_stress.xy + d_stress.offset_k, 
						d_stress.xz + d_stress.offset_k, 
						d_stress.yz + d_stress.offset_k, 
						d_veloc.x + d_veloc.offset_k, 
						d_veloc.y + d_veloc.offset_k, 
						d_veloc.z + d_veloc.offset_k, 
						d_force.x + d_veloc.offset_k,
						d_force.y + d_veloc.offset_k,
						d_force.z + d_veloc.offset_k,
						d_npml_tab, d_phitxxx, d_phitxyy, d_phitxzz, d_phitxyx, d_phityyy, d_phityzz, d_phitxzx, d_phityzy, d_phitzzz,
						d_material.vp + material.offset_k,
						d_material.rho + material.offset_k,
						bounds.size_x, bounds.size_y, bounds.size_z, 
						d_veloc.pitch, bounds.size_y+4, bounds.size_z+4, 
						(float)ds, (float)dt, delta, compute_external, 
						(int)ceilf((float)(bounds.size_x)/(float)NPPDX), (int)ceilf((float)(bounds.size_y)/(float)NPPDY),1, 
						NPPDX_K2,NPPDY_K2,1,position);
			}
			// MPI communications {{{
#ifdef USE_MPI/**/
			if (iloop == 0) {
				// after external points have been computed start MPI comms
				// prepare buffers for send
				maxsize = MAX(bounds.size_x,bounds.size_y);
#ifdef SLOW_BUFFER_TIMING/**/
				cudaThreadSynchronize();
				time_prep_buf_k2_start = clock();
#endif/**/
				getBuffersVeloc (	comms.d_buff_x_min, comms.d_buff_x_max, comms.d_buff_y_min, comms.d_buff_y_max,
						bounds.size_x, bounds.size_y, bounds.size_z,
						d_veloc.pitch, d_veloc.height, d_veloc.depth, 
						d_veloc.x + d_veloc.offset_k,
						d_veloc.y + d_veloc.offset_k,
						d_veloc.z + d_veloc.offset_k, 
						comms.size_buffer_x, comms.size_buffer_y,
						(int)ceilf((float)maxsize/(float)BLOCKDIMX),bounds.size_z,4,
						BLOCKDIMX,2,1, position);
#ifdef SLOW_BUFFER_TIMING/**/
				cudaThreadSynchronize();
				time_prep_buf_k2_end = clock();
#endif/**/
				// copy buffers device -> host
				print_err(cudaMemcpy((void*)comms.buff_x_min_s, comms.d_buff_x_min, 3*comms.size_buffer_x*sizeof(float),cudaMemcpyDeviceToHost),"cudaMemcpy xmin veloc buffers D->H");
				print_err(cudaMemcpy((void*)comms.buff_x_max_s, comms.d_buff_x_max, 3*comms.size_buffer_x*sizeof(float),cudaMemcpyDeviceToHost),"cudaMemcpy xmax veloc buffers D->H");
				print_err(cudaMemcpy((void*)comms.buff_y_min_s, comms.d_buff_y_min, 3*comms.size_buffer_y*sizeof(float),cudaMemcpyDeviceToHost),"cudaMemcpy ymin veloc buffers D->H");
				print_err(cudaMemcpy((void*)comms.buff_y_max_s, comms.d_buff_y_max, 3*comms.size_buffer_y*sizeof(float),cudaMemcpyDeviceToHost),"cudaMemcpy ymax veloc buffers D->H");
				// start non blocking communications
				// SEND
				MPI_Isend( comms.buff_x_min_s, 3*comms.size_buffer_x, MPI_FLOAT, comms.recv_xmin, 0, MPI_COMM_WORLD, &comms.array_req_send[0]);
				MPI_Isend( comms.buff_x_max_s, 3*comms.size_buffer_x, MPI_FLOAT, comms.recv_xmax, 0, MPI_COMM_WORLD, &comms.array_req_send[1]);
				MPI_Isend( comms.buff_y_min_s, 3*comms.size_buffer_y, MPI_FLOAT, comms.recv_ymin, 0, MPI_COMM_WORLD, &comms.array_req_send[2]);
				MPI_Isend( comms.buff_y_max_s, 3*comms.size_buffer_y, MPI_FLOAT, comms.recv_ymax, 0, MPI_COMM_WORLD, &comms.array_req_send[3]);
				// RECV
				MPI_Irecv( comms.buff_x_min_r, 3*comms.size_buffer_x, MPI_FLOAT, comms.send_xmin, 0, MPI_COMM_WORLD, &comms.array_req_recv[0]);
				MPI_Irecv( comms.buff_x_max_r, 3*comms.size_buffer_x, MPI_FLOAT, comms.send_xmax, 0, MPI_COMM_WORLD, &comms.array_req_recv[1]);
				MPI_Irecv( comms.buff_y_min_r, 3*comms.size_buffer_y, MPI_FLOAT, comms.send_ymin, 0, MPI_COMM_WORLD, &comms.array_req_recv[2]);
				MPI_Irecv( comms.buff_y_max_r, 3*comms.size_buffer_y, MPI_FLOAT, comms.send_ymax, 0, MPI_COMM_WORLD, &comms.array_req_recv[3]);
			} else {
				// wait until the end of communications
				time_wait_k2_start = clock();
				MPI_Waitall (4, comms.array_req_recv, comms.array_of_status);
				MPI_Waitall (4, comms.array_req_send, comms.array_of_status);
				time_wait_k2_end = clock();
				print_err(cudaGetLastError(), "Compute veloc : internal");
				// copy buffers host -> device
				print_err(cudaMemcpy((void*)comms.d_buff_x_min, comms.buff_x_min_r, 3*comms.size_buffer_x*sizeof(float),cudaMemcpyHostToDevice),"cudaMemcpy xmin veloc buffers H->D");
				print_err(cudaMemcpy((void*)comms.d_buff_x_max, comms.buff_x_max_r, 3*comms.size_buffer_x*sizeof(float),cudaMemcpyHostToDevice),"cudaMemcpy xmax veloc buffers H->D");
				print_err(cudaMemcpy((void*)comms.d_buff_y_min, comms.buff_y_min_r, 3*comms.size_buffer_y*sizeof(float),cudaMemcpyHostToDevice),"cudaMemcpy ymin veloc buffers H->D");
				print_err(cudaMemcpy((void*)comms.d_buff_y_max, comms.buff_y_max_r, 3*comms.size_buffer_y*sizeof(float),cudaMemcpyHostToDevice),"cudaMemcpy ymax veloc buffers H->D");
				// update arrays with buffers (kernel)
				maxsize = MAX(bounds.size_x,bounds.size_y);
#ifdef SLOW_BUFFER_TIMING/**/
				cudaThreadSynchronize();
				time_update_buf_k2_start = clock();
#endif/**/
				updateHaloVeloc (	comms.d_buff_x_min, comms.d_buff_x_max, comms.d_buff_y_min, comms.d_buff_y_max,
						bounds.size_x, bounds.size_y, bounds.size_z,
						d_veloc.pitch, d_veloc.height, d_veloc.depth, 
						d_veloc.x + d_veloc.offset_k,
						d_veloc.y + d_veloc.offset_k,
						d_veloc.z + d_veloc.offset_k, 
						comms.size_buffer_x, comms.size_buffer_y,
						(int)ceilf((float)maxsize/(float)BLOCKDIMX),bounds.size_z,4,
						BLOCKDIMX,2,1, position);
#ifdef SLOW_BUFFER_TIMING/**/
				cudaThreadSynchronize();
				time_update_buf_k2_end = clock();
#endif/**/
			}
#endif/**/
			// }}}
		}
		cudaThreadSynchronize();
		time_k2_end = clock();	
		// }}}

		// CALCULATION OF THE SEISMOGRAMS {{{
#ifdef COMPUTE_SEISMOS
		if (VERBOSE >= 3) printf("%sCALCULATION OF THE SEISMOGRAMS\n",decal);
		time_se_start = clock();
		float w1, w2, w3;
		for ( ir = 0; ir < IOBS; ir++ ){
			if( ista[ir] == 1 ){
				float dummy_x0[2][2][2];
				float dummy_y0[2][2][2];
				float dummy_z0[2][2][2];

				/* Modif Ariane */

				/* Vx component */
				i = ixobs[ir];
				w1 = xobswt[ir];
				j = iyobs[ir];
				w2 = yobswt[ir];
				k = izobs[ir];
				w3 = zobswt[ir];

				// copy the volume needed
				for (int kk = 0; kk < 2; kk++) {
					for (int jj = 0; jj < 2; jj++) {
						char msg[300];
						sprintf(msg,"cudaMemcpy d_Vx seis | rank %d | station %d <%d,%d,%d>  <i:%d, j:%d, k:%d> {%f.2,%f.2,%f.2}",comms.rank,ir,i,j,k,i,j+jj,k+kk,xobs[ir]/ds, yobs[ir]/ds, zobs[ir]/ds);
						//print_err(cudaMemcpy((void*)&(dummy_x0[kk][jj][0]),&(ACCESS(d_veloc,x,i,j+jj,k+kk)),2*sizeof(float),cudaMemcpyDeviceToHost),"cudaMemcpy d_Vx seis");
						print_err(cudaMemcpy((void*)&(dummy_x0[kk][jj][0]),&(ACCESS(d_veloc,x,i,j+jj,k+kk)),2*sizeof(float),cudaMemcpyDeviceToHost),msg);
					}
				}
				seisx[ir][l-1] = (1-w3)*(
						(1-w2)*( (1-w1)*dummy_x0[0][0][0]     + w1*dummy_x0[0][0][1] )
						+ w2*( (1-w1)*dummy_x0[0][1][0]   + w1*dummy_x0[0][1][1] ) )
					+ w3*( (1-w2)*( (1-w1)*dummy_x0[1][0][0]   + w1*dummy_x0[1][0][1] )
							+ w2*( (1-w1)*dummy_x0[1][1][0] + w1*dummy_x0[1][1][1] ) );
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

				// copy the volume needed
				for (int kk = 0; kk < 2; kk++) {
					for (int jj = 0; jj < 2; jj++) {
						print_err(cudaMemcpy((void*)&(dummy_y0[kk][jj][0]),&(ACCESS(d_veloc,y,i,j+jj,k+kk)),2*sizeof(float),cudaMemcpyDeviceToHost),"cudaMemcpy d_Vy seis");
					}
				}
				seisy[ir][l-1] = (1-w3)*(
						(1-w1)*( (1-w2)*dummy_y0[0][0][0]     + w2*dummy_y0[0][1][0])
						+ w1*( (1-w2)*dummy_y0[0][0][1]   + w2*dummy_y0[0][1][1] ) )
					+ w3*( (1-w1)*( (1-w2)*dummy_y0[1][0][0]   + w2*dummy_y0[1][1][0] )
							+ w1*( (1-w2)*dummy_y0[1][0][1] + w2*dummy_y0[1][1][1] ) );

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

				w1 = w2 = w3 = 0.;
				// copy the volume needed
				for (int kk = 0; kk < 2; kk++) {
					for (int jj = 0; jj < 2; jj++) {
						print_err(cudaMemcpy((void*) &(dummy_z0[kk][jj][0]),&(ACCESS(d_veloc,z,i,j+jj,k+kk)), 2*sizeof(float), cudaMemcpyDeviceToHost),"cudaMemcpy d_Vz seis");
					}
				}
				seisz[ir][l-1] = (1-w3)*(
						(1-w1)*( (1-w2)*dummy_z0[0][0][0]     + w2*dummy_z0[0][1][0] )
						+ w1*( (1-w2)*dummy_z0[0][0][1]   + w2*dummy_z0[0][1][1] ) )
					+ w3*( (1-w1)*( (1-w2)*dummy_z0[1][0][0]   + w2*dummy_z0[1][1][0] )
							+ w1*( (1-w2)*dummy_z0[1][0][1] + w2*dummy_z0[1][1][1] ) );

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

				// copy the volume needed
				for (int kk = 0; kk < 2; kk++) {
					for (int jj = 0; jj < 2; jj++) {
						print_err(cudaMemcpy((void*)&(dummy_x0[kk][jj][0]), &(ACCESS(d_stress,xx,i,j+jj,k+kk)),2*sizeof(float),cudaMemcpyDeviceToHost),"cudaMemcpy d_Txx seis");
						print_err(cudaMemcpy((void*)&(dummy_y0[kk][jj][0]), &(ACCESS(d_stress,yy,i,j+jj,k+kk)),2*sizeof(float),cudaMemcpyDeviceToHost),"cudaMemcpy d_Tyy seis");
						print_err(cudaMemcpy((void*)&(dummy_z0[kk][jj][0]), &(ACCESS(d_stress,zz,i,j+jj,k+kk)),2*sizeof(float),cudaMemcpyDeviceToHost),"cudaMemcpy d_Tzz seis");
					}
				}
				seisxx[ir][l-1] = (1-w3)*(
						(1-w1)*( (1-w2)*dummy_x0[0][0][0]     + w2*dummy_x0[0][1][0])
						+ w1*( (1-w2)*dummy_x0[0][0][1]   + w2*dummy_x0[0][1][1] ) )
					+ w3*( (1-w1)*( (1-w2)*dummy_x0[1][0][0]   + w2*dummy_x0[1][1][0] )
							+ w1*( (1-w2)*dummy_x0[1][0][1] + w2*dummy_x0[1][1][1] ) );
				seisyy[ir][l-1] = (1-w3)*(
						(1-w1)*( (1-w2)*dummy_y0[0][0][0]     + w2*dummy_y0[0][1][0] )
						+ w1*( (1-w2)*dummy_y0[0][0][1]   + w2*dummy_y0[0][1][1] ) )
					+ w3*( (1-w1)*( (1-w2)*dummy_y0[1][0][0]   + w2*dummy_y0[1][1][0] )
							+ w1*( (1-w2)*dummy_y0[1][0][1] + w2*dummy_y0[1][1][1] ) );
				seiszz[ir][l-1] = (1-w3)*(
						(1-w1)*( (1-w2)*dummy_z0[0][0][0]     + w2*dummy_z0[0][1][0] )
						+ w1*( (1-w2)*dummy_z0[0][0][1]   + w2*dummy_z0[0][1][1] ) )
					+ w3*( (1-w1)*( (1-w2)*dummy_z0[1][0][0]   + w2*dummy_z0[1][1][0] )
							+ w1*( (1-w2)*dummy_z0[1][0][1] + w2*dummy_z0[1][1][1] ) );

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

				// copy the volume needed
				for (int kk = 0; kk < 2; kk++) {
					for (int jj = 0; jj < 2; jj++) {
						print_err(cudaMemcpy((void*)&(dummy_x0[kk][jj][0]),&(ACCESS(d_stress,xy,i,j+jj,k+kk)),2*sizeof(float),cudaMemcpyDeviceToHost),"cudaMemcpy d_Txy seis");
					}
				}
				seisxy[ir][l-1] = (1-w3)*(
						(1-w1)*( (1-w2)*dummy_x0[0][0][0]     + w2*dummy_x0[0][1][0] )
						+ w1*( (1-w2)*dummy_x0[0][0][1]   + w2*dummy_x0[0][1][1] ) )
					+ w3*( (1-w1)*( (1-w2)*dummy_x0[1][0][0]   + w2*dummy_x0[1][1][0] )
							+ w1*( (1-w2)*dummy_x0[1][0][1] + w2*dummy_x0[1][1][1] ) );

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

				// copy the volume needed
				for (int kk = 0; kk < 2; kk++) {
					for (int jj = 0; jj < 2; jj++) {
						print_err(cudaMemcpy((void*)&(dummy_x0[kk][jj][0]),&(ACCESS(d_stress,xz,i,j+jj,k+kk)),2*sizeof(float),cudaMemcpyDeviceToHost),"cudaMemcpy d_Txz seis");
					}
				}
				seisxz[ir][l-1] = (1-w3)*(
						(1-w1)*( (1-w2)*dummy_x0[0][0][0]     + w2*dummy_x0[0][1][0] )
						+ w1*( (1-w2)*dummy_x0[0][0][1]   + w2*dummy_x0[0][1][1] ) )
					+ w3*( (1-w1)*( (1-w2)*dummy_x0[1][0][0]   + w2*dummy_x0[1][1][0] )
							+ w1*( (1-w2)*dummy_x0[1][0][1] + w2*dummy_x0[1][1][1] ) );

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

				// copy the volume needed
				for (int kk = 0; kk < 2; kk++) {
					for (int jj = 0; jj < 2; jj++) {
						print_err(cudaMemcpy((void*)&(dummy_x0[kk][jj][0]),&(ACCESS(d_stress,yz,i,j+jj,k+kk)),2*sizeof(float),cudaMemcpyDeviceToHost),"cudaMemcpy d_Tyz seis");
					}
				}
				seisyz[ir][l-1] = (1-w3)*(
						(1-w1)*( (1-w2)*dummy_x0[0][0][0]     + w2*dummy_x0[0][1][0] )
						+ w1*( (1-w2)*dummy_x0[0][0][1]   + w2*dummy_x0[0][1][1] ) )
					+ w3*( (1-w1)*( (1-w2)*dummy_x0[1][0][0]   + w2*dummy_x0[1][1][0] )
							+ w1*( (1-w2)*dummy_x0[1][0][1] + w2*dummy_x0[1][1][1] ) );
			}
		}
		/* End Modif Ariane */
		time_se_end = clock();
#endif
		// }}}

		// TIMING {{{
		if (VERBOSE >= 3) printf("%sTIMING\n",decal);
		// tl : timeloop, k1 : kernel_1, k2 : kernel_2, su : source update, tr : transfer (of source data), se : seismograms, sn : snapshots
		time_tl_end = clock();
		time_tl_total += (((float)time_tl_end - (float)time_tl_start)/(float)CLOCKS_PER_SEC);
		time_k1_total += (((float)time_k1_end - (float)time_k1_start)/(float)CLOCKS_PER_SEC);
		time_k2_total += (((float)time_k2_end - (float)time_k2_start)/(float)CLOCKS_PER_SEC);
		time_su_total += (((float)time_su_end - (float)time_su_start)/(float)CLOCKS_PER_SEC);
		time_tr_total += (((float)time_tr_end - (float)time_tr_start)/(float)CLOCKS_PER_SEC);
		time_se_total += (((float)time_se_end - (float)time_se_start)/(float)CLOCKS_PER_SEC);
		time_wait_k1_total += (((float)time_wait_k1_end - (float)time_wait_k1_start)/(float)CLOCKS_PER_SEC);
		time_wait_k2_total += (((float)time_wait_k2_end - (float)time_wait_k2_start)/(float)CLOCKS_PER_SEC);
#ifdef SLOW_BUFFER_TIMING
		time_prep_buf_k1_total += (((float)time_prep_buf_k1_end - (float)time_prep_buf_k1_start)/(float)CLOCKS_PER_SEC);
		time_prep_buf_k2_total += (((float)time_prep_buf_k2_end - (float)time_prep_buf_k2_start)/(float)CLOCKS_PER_SEC);
		time_update_buf_k1_total += (((float)time_update_buf_k1_end - (float)time_update_buf_k1_start)/(float)CLOCKS_PER_SEC);
		time_update_buf_k2_total += (((float)time_update_buf_k2_end - (float)time_update_buf_k2_start)/(float)CLOCKS_PER_SEC);
#endif
		// }}}

		// PROC MASTER RECV SEISMOS FROM OWNERS AND WRITE THEM TO DISK  {{{
		// yet just velocity, no stress
#ifdef COMPUTE_SEISMOS
		if (VERBOSE >= 3) printf("%sPROC MASTER RECV SEISMOS FROM OWNERS AND WRITE THEM TO DISK\n",decal);
		if (l == TMAX){
#ifdef USE_MPI/**/
			// tell proc MASTER who has the seismo
			int* tab_stations;
			if (comms.rank == MASTER) tab_stations = (int*) malloc (comms.nbprocs*IOBS*sizeof(int));

			MPI_Gather( &(ista[0]), IOBS, MPI_INTEGER, tab_stations, IOBS, MPI_INTEGER, MASTER, MPI_COMM_WORLD);
#endif/**/
			// proc MASTER recv seismos from owners
			if (comms.rank == MASTER) {
				printf("\r                                \rdone.\n\nWriting seismograms ...\n");
				for ( ir = 0; ir < IOBS; ir++ ){
#ifdef USE_MPI/**/
					int found = 0;
					for (int iproc = 0; iproc<comms.nbprocs; iproc++) {
						if (tab_stations[iproc*IOBS+ir] == 1) { // le proc iproc possede les sismos
							if (VERBOSE >= 2) printf("proc %d has the station %d\n",iproc,ir+1);
							found = 1;
							if (iproc != MASTER) {
								MPI_Recv( &(seisx[ir][0]), l, MPI_FLOAT, iproc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
								MPI_Recv( &(seisy[ir][0]), l, MPI_FLOAT, iproc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
								MPI_Recv( &(seisz[ir][0]), l, MPI_FLOAT, iproc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
							}
						}
					}
					if (found) {
#endif/**/
						sprintf(flname4, "%sobs%d.dat", outdir,ir+1);
						if ( l == TMAX) printf("%d %s\n", ir+1, flname4);
						fp4 = fopen(flname4, "w");
						fprintf(fp4, "%d %f %f %f\n", nobs[ir], xobs[ir], yobs[ir], zobs[ir]);
						fprintf(fp4, "%d %f\n", l, dt);
						for (l1 = 0; l1 < l; l1++)
							fprintf(fp4, "%e %e %e %e %e %e %e %e %e\n",
									seisx[ir][l1], seisy[ir][l1], seisz[ir][l1],
									seisxx[ir][l1], seisyy[ir][l1], seiszz[ir][l1],
									seisxy[ir][l1], seisxz[ir][l1], seisyz[ir][l1] );
						fclose(fp4);
#ifdef USE_MPI/**/
					} else {
						printf("station %d not found\n",ir+1);
					}
#endif/**/
				}
#ifdef USE_MPI/**/
			} else { // owners send their seismos to MASTER proc
				for ( ir = 0; ir < IOBS; ir++ ){
					if (ista[ir] == 1) {
						MPI_Send(&(seisx[ir][0]), l, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD);
						MPI_Send(&(seisy[ir][0]), l, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD);
						MPI_Send(&(seisz[ir][0]), l, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD);
					}
				}
#endif/**/
			}
#ifdef USE_MPI/**/
			if (comms.rank == MASTER) free(tab_stations);
#endif/**/
		}
#endif
		// }}}
	}
	// }}}

	// PRINT TIME MEASUREMENTS {{{
	printf("\n\n%sTIMING : \n",decal);
	printf("%s\ttimeloop : \t%f\n",decal,time_tl_total);
	printf("\n");

	printf("%s\tkernel 1 : \t%f -> %.2f%\n",decal,time_k1_total, (time_k1_total/time_tl_total)*100.);
	printf("%s\t\tcomm waiting : \t%f -> %.2f%\n",decal,time_wait_k1_total, (time_wait_k1_total/time_tl_total)*100.);
#ifdef SLOW_BUFFER_TIMING
	printf("%s\t\tbuf building : \t%f -> %.2f%\n",decal,time_prep_buf_k1_total, (time_prep_buf_k1_total/time_tl_total)*100.);
	printf("%s\t\tbuf updating : \t%f -> %.2f%\n",decal,time_update_buf_k1_total, (time_update_buf_k1_total/time_tl_total)*100.);
#endif
	printf("\n");

	printf("%s\tkernel 2 : \t%f -> %.2f%\n",decal,time_k2_total, (time_k2_total/time_tl_total)*100.);
	printf("%s\t\tcomm waiting : \t%f -> %.2f%\n",decal,time_wait_k2_total, (time_wait_k2_total/time_tl_total)*100.);
#ifdef SLOW_BUFFER_TIMING
	printf("%s\t\tbuf building : \t%f -> %.2f%\n",decal,time_prep_buf_k2_total, (time_prep_buf_k2_total/time_tl_total)*100.);
	printf("%s\t\tbuf updating : \t%f -> %.2f%\n",decal,time_update_buf_k2_total, (time_update_buf_k2_total/time_tl_total)*100.);
#endif
	printf("\n");

	printf("%s\tsourceupdate : \t%f -> %.2f%\n",decal, time_su_total, (time_su_total/time_tl_total)*100.);
	printf("%s\ttransfer : \t%f -> %.2f%\n",decal,time_tr_total, (time_tr_total/time_tl_total)*100.);
	printf("%s\tseismograms : \t%f -> %.2f%\n",decal, time_se_total, (time_se_total/time_tl_total)*100.);
	printf("\n");
	printf("%sK1 + K2 : %f -> %.2f%\n",decal,time_k1_total+time_k2_total, ((time_k1_total+time_k2_total)/time_tl_total)*100.);
	printf("%sTotal comms waiting : %f -> %.2f%\n",decal,time_wait_k1_total+time_wait_k2_total, ((time_wait_k1_total+time_wait_k2_total)/time_tl_total)*100.);
#ifdef SLOW_BUFFER_TIMING
	printf("%sTotal for buffers : %f -> %.2f%\n",decal,time_prep_buf_k1_total+time_update_buf_k1_total+time_prep_buf_k2_total+time_update_buf_k2_total,((time_prep_buf_k1_total+time_update_buf_k1_total+time_prep_buf_k2_total+time_update_buf_k2_total)/time_tl_total)*100.);
#endif
	printf("\n");
	printf("\n");
	// }}}

// FREE DYN ALLOCATED MEMORY {{{
	// HOST MEMORY {{{
	free_fvector(laydep, 0, NLAYER-1);
	free_fvector(vp0, 0, NLAYER-1);
	free_fvector(vs0, 0, NLAYER-1);
	free_fvector(rho0, 0, NLAYER-1);
	free_fvector(q0, 0, NLAYER-1);

	free_ivector(ixhypo, 0, ISRC-1);
	free_ivector(iyhypo, 0, ISRC-1);
	free_ivector(izhypo, 0, ISRC-1);
	free_ivector(insrc, 0, ISRC-1);
	free_fvector(xhypo, 0, ISRC-1);
	free_fvector(yhypo, 0, ISRC-1);
	free_fvector(zhypo, 0, ISRC-1);
	free_dvector(strike, 0, ISRC-1);
	free_dvector(dip, 0, ISRC-1);
	free_dvector(rake, 0, ISRC-1);
	free_fvector(slip, 0, ISRC-1);
	free_fvector(xweight, 0, ISRC-1);
	free_fvector(yweight, 0, ISRC-1);
	free_fvector(zweight, 0, ISRC-1);

	free_ivector(nobs, 0, IOBS-1);
	free_fvector(xobs, 0, IOBS-1);
	free_fvector(yobs, 0, IOBS-1);
	free_fvector(zobs, 0, IOBS-1);
	free_ivector(ixobs, 0, IOBS-1);
	free_ivector(iyobs, 0, IOBS-1);
	free_ivector(izobs, 0, IOBS-1);
	free_fvector(xobswt, 0, IOBS-1);
	free_fvector(yobswt, 0, IOBS-1);
	free_fvector(zobswt, 0, IOBS-1);
	free_ivector(ista, 0, IOBS-1);

	free_fvector(dumpx, bounds.xinf, bounds.xsup);
	free_fvector(kappax, bounds.xinf, bounds.xsup);
	free_fvector(alphax, bounds.xinf, bounds.xsup);
	free_fvector(dumpx2, bounds.xinf, bounds.xsup);
	free_fvector(kappax2, bounds.xinf, bounds.xsup);
	free_fvector(alphax2, bounds.xinf, bounds.xsup);

	free_fvector(dumpy, bounds.yinf, bounds.ysup);
	free_fvector(kappay, bounds.yinf, bounds.ysup);
	free_fvector(alphay, bounds.yinf, bounds.ysup);
	free_fvector(dumpy2, bounds.yinf, bounds.ysup);
	free_fvector(kappay2, bounds.yinf, bounds.ysup);
	free_fvector(alphay2, bounds.yinf, bounds.ysup);

	free_fvector(dumpz, bounds.zinf, bounds.zmax);
	free_fvector(kappaz, bounds.zinf, bounds.zmax);
	free_fvector(alphaz, bounds.zinf, bounds.zmax);
	free_fvector(dumpz2, bounds.zinf, bounds.zmax);
	free_fvector(kappaz2, bounds.zinf, bounds.zmax);
	free_fvector(alphaz2, bounds.zinf, bounds.zmax);

	free_dmatrix(vel, 0, ISRC-1, 0, IDUR-1);
	free_dmatrix(seisx, 0, IOBS-1, 0, TMAX-1);
	free_dmatrix(seisy, 0, IOBS-1, 0, TMAX-1);
	free_dmatrix(seisz, 0, IOBS-1, 0, TMAX-1);
	free_dmatrix(seisxx, 0, IOBS-1, 0, TMAX-1);
	free_dmatrix(seisyy, 0, IOBS-1, 0, TMAX-1);
	free_dmatrix(seiszz, 0, IOBS-1, 0, TMAX-1);
	free_dmatrix(seisxy, 0, IOBS-1, 0, TMAX-1);
	free_dmatrix(seisxz, 0, IOBS-1, 0, TMAX-1);
	free_dmatrix(seisyz, 0, IOBS-1, 0, TMAX-1);

	free_arrays_CPU(&material, &force, dim_model);
	// }}}

	// DEVICE MEMORY {{{
	free_CPML_data(d_phivxx, d_phivxy, d_phivxz, d_phivyx, d_phivyy, d_phivyz, d_phivzx, d_phivzy, d_phivzz, 
			d_phitxxx, d_phitxyy, d_phitxzz, d_phitxyx, d_phityyy, d_phityzz, d_phitxzx, d_phityzy, d_phitzzz,
			d_dumpx, d_alphax, d_kappax, d_dumpx2, d_alphax2, d_kappax2,
			d_dumpy, d_alphay, d_kappay, d_dumpy2, d_alphay2, d_kappay2,
			d_dumpz, d_alphaz, d_kappaz, d_dumpz2, d_alphaz2, d_kappaz2);

	free_arrays_GPU(&d_veloc, &d_stress, &d_material, &d_force, d_npml_tab, dim_model);

#ifdef USE_MPI/*{{{*/
	free_MPI_buffers(&comms);
#endif/*}}}*/
	// }}}
// }}}

	// END {{{
#ifdef USE_MPI
	MPI_Finalize();
#endif
	return 0;
	// }}}

}

