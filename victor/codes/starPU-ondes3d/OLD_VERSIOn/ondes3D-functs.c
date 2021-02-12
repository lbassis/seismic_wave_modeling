#include "stencil.h"
#include <string.h>

// PARAM FILE READ FUNCTIONS {{{
	void chomp(const char *s)
	{
		char *p;
		while (NULL != s && NULL != (p = strrchr(s, '\n')) && NULL != (p = strrchr(s, '\r'))){
			*p = '\0';
		}
	} 

	void readIntParam(char* entry, int* out, FILE* fd)
	{	
		char cur_line[STRMAX];
		char *name, *equal, *str_value;
		int value;
		char* saveptr;
		
		fseek(fd, 0, 0);
		while (fgets(cur_line, STRMAX, fd) != NULL) {
			// perl-like chomp to avoid \r & \n problems
			chomp(cur_line);
			if (strlen(cur_line) == 0) continue;
			name = strtok_r(cur_line," =\t", &saveptr);
			if (!strcmp("#", name)) continue;
			if (strcmp(name, entry)) continue;
			str_value = strtok_r(NULL," =\t", &saveptr);
			sscanf(str_value,"%d",&value);
			if (VERBOSE == 3) printf("found %s = %d\n",name,value);
			*out = value;
			return;
		}
		printf("ENTRY %s NOT FOUND !!!\nexiting\n",entry);
	STOP
	}

	void readFloatParam(char* entry, float* out, FILE* fd)
	{
		char cur_line[STRMAX];
		char *name, *str_value;
		float value;
		char* saveptr;
		
		fseek(fd, 0, 0);
		while (fgets(cur_line, STRMAX, fd) != NULL) {
			chomp(cur_line);
			if (strlen(cur_line) == 0) continue;
			name = strtok_r(cur_line," =\t", &saveptr);
			if (!strcmp("#", name)) continue;
			if (strcmp(name, entry)) continue;
			str_value = strtok_r(NULL," =\t", &saveptr);
			sscanf(str_value,"%f",&value);
			if (VERBOSE == 3) printf("found %s = %f\n",name,value);
			*out = value;
			return;
		}
		printf("ENTRY %s NOT FOUND !!!\nexiting\n",entry);
	STOP
	}

	void readStringParam(char* entry, char* out, FILE* fd)
	{
		char cur_line[STRMAX];
		char *name, *str_value;
		char* saveptr;
		
		fseek(fd, 0, 0);
		while (fgets(cur_line, STRMAX, fd) != NULL) {
			chomp(cur_line);
			if (strlen(cur_line) == 0) continue;
			name = strtok_r(cur_line," =\t", &saveptr);
			if (!strcmp("#", name)) continue;
			if (strcmp(name, entry)) continue;
			str_value = strtok_r(NULL," =\t", &saveptr);
			if (VERBOSE == 3) printf("found %s = %s\n",name,str_value);
			strcpy(out, str_value);
			return; 
		}
		printf("ENTRY %s NOT FOUND !!!\nexiting\n",entry);
	STOP
	}

	void readLayerParam(char* entry, int num, float* depth, float* vp, float* vs, float* rho, float* q, FILE* fd)
	{
		char cur_line[STRMAX];
		char *name, *str_value;
		char *completed_entry;
		char numero[10];
		char* saveptr;
		
		completed_entry = (char*) malloc (strlen(entry) + 10);
		sprintf(numero, "%d",num);
		strcat(strcpy(completed_entry,entry),numero);
		
		fseek(fd, 0, 0);
		while (fgets(cur_line, STRMAX, fd) != NULL) {
			chomp(cur_line);
			if (strlen(cur_line) == 0) continue;
			name = strtok_r(cur_line," =\t", &saveptr);
			if (!strcmp("#", name)) continue;
			if (strcmp(name, completed_entry)) continue;
			free(completed_entry);

			str_value = strtok_r(NULL," =\t", &saveptr);
			sscanf(str_value,"%f",depth);
			str_value = strtok_r(NULL," =\t", &saveptr);
			sscanf(str_value,"%f",vp);
			str_value = strtok_r(NULL," =\t", &saveptr);
			sscanf(str_value,"%f",vs);
			str_value = strtok_r(NULL," =\t", &saveptr);
			sscanf(str_value,"%f",rho);
			str_value = strtok_r(NULL," =\t", &saveptr);
			sscanf(str_value,"%f",q);

			if (VERBOSE == 3) printf("found %s > depth: %f, vp: %f, vs: %f, rho: %f, q: %f\n", name, *depth, *vp, *vs, *rho, *q);
			return; 
		}
		printf("ENTRY %s NOT FOUND !!!\nexiting\n",completed_entry);
	STOP
	}
// }}}

// SIMULATION INITIALIZATION {{{
	void read_parameter_file(ondes3d_params* par, char* param_file, int rank)
	{
		FILE* paramfile;

		if (rank == MASTER) {
			paramfile = fopen(param_file, "r");
			if ( paramfile == NULL ){
				fprintf(stderr, "failed opening parameter file %s\n",param_file);
				STOP
			}
			readIntParam("NDIM", &par->ndim, paramfile);
			readIntParam("XMIN", &par->xmin, paramfile);
			readIntParam("XMAX", &par->xmax, paramfile);
			readIntParam("YMIN", &par->ymin, paramfile);
			readIntParam("YMAX", &par->ymax, paramfile);
			readIntParam("ZMIN", &par->zmin, paramfile);
			readIntParam("TIME_STEPS", &par->tmax, paramfile);
			// readIntParam("I0", &i0, paramfile);
			// readIntParam("J0", &j0, paramfile);
			readStringParam("OUTPUT_DIR", par->outdir, paramfile);
			readStringParam("SOURCES", par->src_file, paramfile);
			readStringParam("SOURCE_TIME_FUNCTION", par->stf_file, paramfile);
			readStringParam("STATIONS", par->sta_file, paramfile);
			readFloatParam("DS", &par->ds, paramfile);
			readFloatParam("DT", &par->dt, paramfile);
			readFloatParam("DF", &par->fd, paramfile);
			// pour l'instant tableaux 3D, mais modèle 1D en couches, faute de lecteur de modèle 3D

			readIntParam("NB_LAYERS", &par->nlayer, paramfile);
		}
		set_niter(par->tmax-1);
	#ifdef STARPU_USE_MPI
		MPI_Bcast(&par->nlayer, 1, MPI_INTEGER, MASTER, MPI_COMM_WORLD);
	#endif
		par->laydep = (float*)malloc(par->nlayer*sizeof(float));
		par->vp0 = (float*)malloc(par->nlayer*sizeof(float));
		par->vs0 = (float*)malloc(par->nlayer*sizeof(float));
		par->rho0 = (float*)malloc(par->nlayer*sizeof(float));
		par->q0  = (float*)malloc(par->nlayer*sizeof(float));
		if (rank == MASTER) {
			for (int ly = 0; ly < par->nlayer; ly++){
				readLayerParam("LAYER_",ly+1, &par->laydep[ly], &par->vp0[ly], &par->vs0[ly], &par->rho0[ly], &par->q0[ly], paramfile);
			}
			fclose( paramfile );
			if (VERBOSE > 2) {
				printf("\nDimension of FDM order ... %i\n", par->ndim );
				printf("\nParameter File ... %s\n", param_file);
				printf("Source Model based on ... %s\n", par->src_file );
				printf("Rupture History from ... %s\n", par->stf_file );
				printf("Station Position at ... %s\n", par->sta_file );
				printf("Output directory ... %s\n", par->outdir );
				printf("\nspatial grid ds = %f[m]\n", par->ds);
				printf("time step dt = %f[s]\n", par->dt);
				printf("\nModel Region (%i:%i, %i:%i, %i:%i)\n",par->xmin, par->xmax, par->ymin, par->ymax, par->zmin, 0);
				printf("Absorbing layer thickness: %i\n",DELTA);
			}
		}
	#ifdef STARPU_USE_MPI
		// broadcasting data read to other procs
		{// first ints
			int buffer[7];
			if (rank == MASTER) {
				buffer[0] = par->ndim;
				buffer[1] = par->xmin;
				buffer[2] = par->xmax;
				buffer[3] = par->ymin;
				buffer[4] = par->ymax;
				buffer[5] = par->zmin;
				buffer[6] = par->tmax;
			}
			MPI_Bcast(buffer, 7, MPI_INTEGER, MASTER, MPI_COMM_WORLD);
			if (rank != MASTER) {
				par->ndim = buffer[0];
				par->xmin = buffer[1];
				par->xmax = buffer[2];
				par->ymin = buffer[3];
				par->ymax = buffer[4];
				par->zmin = buffer[5];
				par->tmax = buffer[6];
			}
		}
		{// then floats
			float* buffer = (float*) malloc ((par->nlayer*5+3)*sizeof(float));
			if (rank == MASTER) {
				buffer[0] = par->ds;
				buffer[1] = par->dt;
				buffer[2] = par->fd;
				for (int i=0; i<par->nlayer; i++) {
					buffer[3+i*5] = par->laydep[i];
					buffer[4+i*5] = par->vp0[i];
					buffer[5+i*5] = par->vs0[i];
					buffer[6+i*5] = par->rho0[i];
					buffer[7+i*5] = par->q0[i];
				}
			}
			MPI_Bcast(buffer, par->nlayer*5+3, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
			if (rank != MASTER) {
				par->ds = buffer[0];
				par->dt = buffer[1];
				par->fd = buffer[2];
				for (int i=0; i<par->nlayer; i++) {
					par->laydep[i] = buffer[3+i*5];
					par->vp0[i] = buffer[4+i*5];
					par->vs0[i] = buffer[5+i*5];
					par->rho0[i] = buffer[6+i*5];
					par->q0[i] = buffer[7+i*5];
				}
			}
			free(buffer);
		}
	#endif
		float pi = acosf(-1.0);
		par->dump0 = - (NPOWER + 1) * log(reflect) / (2.0 * DELTA * par->ds);
		par->alpha0 = par->fd*(float)pi;	/* alpha0 = pi*fd where fd is the dominant frequency of the source */
		par->kappa0 = 1.0;

		set_dump0(par->dump0);
		set_kappa0(par->kappa0);
		set_alpha0(par->alpha0);

		return;
	}

	void read_sources_positions(ondes3d_params* par, int rank){
		// DGN : sources rangées dans structure globale -> rearranger par bloc ??
		
		if (rank == MASTER && VERBOSE >= 3) printf("PROC MASTER READS THE SOURCE POSITION\n");
		FILE* filein;
		int dummy;

	   if (rank == MASTER) {
	      filein = fopen( par->src_file, "r");
	      if ( filein == NULL ){
	         perror ("failed opening sources positions file");
	         STOP
	      }
	      fscanf ( filein, "%d", &par->nb_src );
	      fscanf ( filein, "%f %f %f", &par->xhypo0, &par->yhypo0, &par->zhypo0 );

	      if (VERBOSE >= 2) printf("\nNUMBER OF SOURCE %d\n", par->nb_src);
	      if (VERBOSE >= 2) printf("Hypocenter ... (%f, %f, %f)\n", par->xhypo0, par->yhypo0, par->zhypo0);
	      
	      par->ixhypo0 = my_float2int(par->xhypo0/par->ds)+1;
	      par->iyhypo0 = my_float2int(par->yhypo0/par->ds)+1;
	      par->izhypo0 = my_float2int(par->zhypo0/par->ds);

	      if (VERBOSE >= 2) printf(".............. (%i, %i, %i)\n", par->ixhypo0, par->iyhypo0, par->izhypo0);
	   }
	#ifdef STARPU_USE_MPI
	   MPI_Bcast(&par->nb_src, 1, MPI_INTEGER, MASTER, MPI_COMM_WORLD);
	#endif
	   
	   par->xhypo   = (float*)malloc(par->nb_src*sizeof(float));
	   par->yhypo   = (float*)malloc(par->nb_src*sizeof(float));
	   par->zhypo   = (float*)malloc(par->nb_src*sizeof(float));
	   par->slip    = (float*)malloc(par->nb_src*sizeof(float));
	   par->xweight = (float*)malloc(par->nb_src*sizeof(float));
	   par->yweight = (float*)malloc(par->nb_src*sizeof(float));
	   par->zweight = (float*)malloc(par->nb_src*sizeof(float));

	   par->ixhypo = (int*)malloc(par->nb_src*sizeof(int));
	   par->iyhypo = (int*)malloc(par->nb_src*sizeof(int));
	   par->izhypo = (int*)malloc(par->nb_src*sizeof(int));

		par->strike = (double*)malloc(par->nb_src*sizeof(double));
		par->dip    = (double*)malloc(par->nb_src*sizeof(double));
		par->rake   = (double*)malloc(par->nb_src*sizeof(double));

	   if (rank == MASTER) {
	      for ( int is = 0;  is < par->nb_src; is++)
	         fscanf ( filein, "%d %f %f %f", &dummy, &par->xhypo[is], &par->yhypo[is], &par->zhypo[is]);
	      fclose( filein );
	   }
	#ifdef STARPU_USE_MPI
	   {// broadcast
	      float buffer[par->nb_src*3];
	      if (rank == MASTER) {
	         for (int i=0; i<par->nb_src; i++) {
	            buffer[0+i*3] = par->xhypo[i];
	            buffer[1+i*3] = par->yhypo[i];
	            buffer[2+i*3] = par->zhypo[i];
	         }
	      }
	      MPI_Bcast(buffer, par->nb_src*3, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
	      if (rank != MASTER) {
	         for (int i=0; i<par->nb_src; i++) {
	            par->xhypo[i] = buffer[0+i*3];
	            par->yhypo[i] = buffer[1+i*3];
	            par->zhypo[i] = buffer[2+i*3];
	         }
	      }
	   }
	#endif
	   
	   for ( int is = 0; is < par->nb_src; is++) {
	     	
	     	// DGN -> TODO : peu-être décalé des trucs ici ...

	      // par->ixhypo[is] = my_float2int(par->xhypo[is]/par->ds)+1;
	      // par->xweight[is] = (par->xhypo[is]/par->ds - par->ixhypo[is]+1);

	      // par->iyhypo[is] = my_float2int(par->yhypo[is]/par->ds)+1;
	      // par->yweight[is] = (par->yhypo[is]/par->ds - par->iyhypo[is]+1);

	      // par->izhypo[is] = my_float2int(par->zhypo[is]/par->ds);
	      // par->zweight[is] = (par->zhypo[is]/par->ds - par->izhypo[is]);
	      
	      // coords globales commençant en zero
	      par->ixhypo[is] = my_float2int(par->xhypo[is]/par->ds)+1 - par->xmin + DELTA;
	      /*(- par->xmin + DELTA) pour avoir les coordonnées par rapport au zéro du tableau : cpml comprise, halo pas compris*/
	      par->xweight[is] = (par->xhypo[is]/par->ds - (my_float2int(par->xhypo[is]/par->ds)+1)+1);

	      // par->iyhypo[is] = my_float2int((par->yhypo[is] - par->ymin)/par->ds)+1 - par->ymin + DELTA;
	      par->iyhypo[is] = my_float2int((par->yhypo[is])/par->ds)+1 - par->ymin + DELTA;
	      par->yweight[is] = (par->yhypo[is]/par->ds - (my_float2int((par->yhypo[is] - par->ymin)/par->ds)+1)+1);

			float sizez = -par->zmin + 1 + DELTA;
	      par->izhypo[is] = sizez + my_float2int(par->zhypo[is]/par->ds);
	      par->zweight[is] = (par->zhypo[is]/par->ds - (my_float2int(par->zhypo[is]/par->ds)));
	      
	      // if (rank == MASTER && VERBOSE >= 2) {
	      //    printf("Source %i .... (%f, %f, %f)\n", is+1, par->xhypo[is], par->yhypo[is], par->zhypo[is] );
	      //    printf(".............. (%i, %i, %i)\n", par->ixhypo[is], par->iyhypo[is], par->izhypo[is]);
	      // }
	   }
	   
	   unsigned bx, by, nbx, nby;
	   nbx = get_nbx();
	   nby = get_nby();
	   for (bx = 0; bx < nbx; bx++)
	   {	for (by = 0; by < nby; by++)
	      	{
		      	struct block_description * block = get_block_description(bx, by);
		      	unsigned node = block->mpi_node;

				if (node == rank) {
			      	block->nb_src = 0;
		     		block->insrc = (unsigned*)malloc(par->nb_src*sizeof(unsigned));

					// bounds
					int xmin_loc, xmax_loc, ymin_loc, ymax_loc;
					xmin_loc = block->xshift;
					ymin_loc = block->yshift;
					xmax_loc = xmin_loc + block->sizex;
					ymax_loc = ymin_loc + block->sizey;

					// pas de sources dans les CPMLS
					if (xmin_loc < DELTA) xmin_loc = DELTA;
					if (ymin_loc < DELTA) ymin_loc = DELTA;

					int sizex = par->xmax- par->xmin + 1 + 2*DELTA;
					int sizey = par->ymax- par->ymin + 1 + 2*DELTA;

					if (xmax_loc >= (sizex-DELTA)) xmax_loc = sizex-DELTA-1;
					if (ymax_loc >= (sizey-DELTA)) ymax_loc = sizey-DELTA-1;

					DEBUG("block %d,%d loc : X:%d,%d  Y:%d,%d\n", bx,by,xmin_loc, xmax_loc, ymin_loc, ymax_loc, 2);

			      	for ( int is = 0; is < par->nb_src; is++) {
						// DGN verifiier pour zmin ...
						if (!( par->ixhypo[is] > xmax_loc+2 || par->ixhypo[is] < xmin_loc-2 || par->iyhypo[is] > ymax_loc+2 || par->iyhypo[is] < ymin_loc-2 || par->izhypo[is] < DELTA || par->izhypo[is] >  block->sizez )) {
							// source is in this block
							block->nb_src++;
							block->insrc[is] = 1;
							DEBUG("\tSource %i .... (%f, %f, %f)\n", is+1, par->xhypo[is], par->yhypo[is], par->zhypo[is] );
	         				DEBUG("\t.............. (%i, %i, %i)\n", par->ixhypo[is], par->iyhypo[is], par->izhypo[is]);
						} else {
							block->insrc[is] = 0;
						}
			      	}
   			      	DEBUG("block {%d, %d} has %d sources\n",bx,by,block->nb_src);
		      	}
	    	}
		}
	}

	void read_source_time_function(ondes3d_params* par, int rank)
	{

		if (rank == MASTER && VERBOSE >= 3) printf("PROC MASTER READS THE SOURCE TIME FUNCTION\n");
		
		FILE* filein;
		int dummy;
		float mo;
		char buf[256];

		if (rank == MASTER) {
			filein = fopen( par->stf_file, "r");
			if ( filein == NULL ){
				perror ("failed opening source time function file\n");
				STOP
			}
			fgets(buf, 255, filein);
			fgets(buf, 255, filein);

			fscanf ( filein, "%f %f", &par->dsbiem, &par->dtbiem );
			fscanf ( filein, "%d", &par->idur );

			if (VERBOSE >= 2) printf("\nSource duration %f sec\n", par->dtbiem*(par->idur-1));
			if (VERBOSE >= 2) printf("fault segment %f m, %f s\n", par->dsbiem, par->dtbiem);
		}
	#ifdef STARPU_USE_MPI
		MPI_Bcast(&par->idur, 1, MPI_INTEGER, MASTER, MPI_COMM_WORLD);
	#endif
		
		par->vel = (double*)malloc(par->nb_src*par->idur*sizeof(double));
		mo = 0.0;

		if (rank == MASTER) {
			for( int is = 0; is < par->nb_src; is++ ){
				fscanf ( filein, "%d", &dummy);
				fscanf ( filein, "%lf %lf %lf", &par->strike[is], &par->dip[is], &par->rake[is]);
				par->strike[is] = par->strike[is]/180.*PI;
				par->dip[is] = par->dip[is]/180.*PI;
				par->rake[is] = par->rake[is]/180.*PI;
				par->slip[is] = 0.;
				for ( int it = 0; it < par->idur; it++ ) {
					int index = is*par->idur+it;
					fscanf ( filein, "%f", &par->vel[index]);
					par->slip[is] += par->vel[index]*par->dtbiem;
					par->vel[index] = (par->dsbiem/par->ds) * (par->dsbiem/par->ds) * par->vel[index] / par->ds;
				}
				mo += par->slip[is];
			}
			fclose(filein);
			mo *= par->dsbiem * par->dsbiem;
		}
		
	#ifdef STARPU_USE_MPI
		{// broadcast
			float buffer[3+par->nb_src*4+par->nb_src*par->idur];
			if (rank == MASTER) {
				buffer[0] = par->dsbiem;
				buffer[1] = par->dtbiem;
				buffer[2] = mo;
				for (int i=0; i<par->nb_src; i++) {
					buffer[3+i*4] = par->strike[i];
					buffer[4+i*4] = par->dip[i];
					buffer[5+i*4] = par->rake[i];
					buffer[6+i*4] = par->slip[i];
				}
				for (int i=0; i<par->nb_src; i++) {
					for (int j=0; j<par->idur; j++) {
						buffer[7+(par->nb_src-1)*4+i*par->idur+j] = par->vel[i*par->idur+j];
					}
				}
			}
			MPI_Bcast(buffer, 3+par->nb_src*4+par->nb_src*par->idur, MPI_FLOAT, MASTER, MPI_COMM_WORLD); 
			if (rank != MASTER) {
				par->dsbiem = buffer[0];
				par->dtbiem = buffer[1];
				mo = buffer[2];
				for (int i=0; i<par->nb_src; i++) {
					par->strike[i] = buffer[3+i*4];
					par->dip[i] = buffer[4+i*4];
					par->rake[i] = buffer[5+i*4];
					par->slip[i] = buffer[6+i*4];
				}
				for (int i=0; i<par->nb_src; i++) {
					for (int j=0; j<par->idur; j++) {
						par->vel[i*par->idur+j] = buffer[7+(par->nb_src-1)*4+i*par->idur+j];
					}
				}
			}
		}
	#endif
		
	}

	void read_stations_positions(ondes3d_params* par, int rank)
	{
		if (VERBOSE >= 3 && (rank == MASTER)) printf("PROC MASTER READS THE STATIONS POSITIONS\n");

		FILE* filein;
		
		if (rank == MASTER) {
			if (VERBOSE >= 2) printf("\n Stations coordinates :\n");

			filein = fopen( par->sta_file, "r");
			if ( filein == NULL ){
				perror ("failed at fopen 3");
				exit(1);
			}
			fscanf ( filein, "%d", &par->iobs );
		}
	#ifdef STARPU_USE_MPI
		MPI_Bcast(&par->iobs, 1, MPI_INTEGER, MASTER, MPI_COMM_WORLD);
	#endif
	 
		par->nobs = (int*)malloc(par->iobs*sizeof(int));
		par->xobs = (float*)malloc(par->iobs*sizeof(float));
		par->yobs = (float*)malloc(par->iobs*sizeof(float));
		par->zobs = (float*)malloc(par->iobs*sizeof(float));
		par->ixobs = (int*)malloc(par->iobs*sizeof(int));
		par->iyobs = (int*)malloc(par->iobs*sizeof(int));
		par->izobs = (int*)malloc(par->iobs*sizeof(int));
		par->xobswt = (float*)malloc(par->iobs*sizeof(float));
		par->yobswt = (float*)malloc(par->iobs*sizeof(float));
		par->zobswt = (float*)malloc(par->iobs*sizeof(float));
		
		if (rank == MASTER) {
			for (unsigned ir = 0; ir < par->iobs; ir++){
				fscanf ( filein, "%d %f %f %f", &par->nobs[ir], &par->xobs[ir], &par->yobs[ir], &par->zobs[ir] );
			}
			fclose( filein );
		}
	#ifdef STARPU_USE_MPI
		{//broadcast
			float buffer[par->iobs*3];
			if (rank == MASTER) {
				for (int i=0; i<par->iobs; i++) {
					buffer[0+i*3] = par->xobs[i];
					buffer[1+i*3] = par->yobs[i];
					buffer[2+i*3] = par->zobs[i];
				}
			}
			MPI_Bcast(buffer, par->iobs*3, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
			if (rank != MASTER) {
				for (int i=0; i<par->iobs; i++) {
					par->xobs[i] = buffer[0+i*3];
					par->yobs[i] = buffer[1+i*3];
					par->zobs[i] = buffer[2+i*3];
				}
			}
		}
	#endif
		int nb_local_obs = 0;
		int nb_global_obs = 0;
		

		for (int ir = 0; ir < par->iobs; ir++){
			par->ixobs[ir] = my_float2int(par->xobs[ir]/par->ds)+1 - par->xmin + DELTA;
			par->xobswt[ir] = (par->xobs[ir]/par->ds - (my_float2int(par->xobs[ir]/par->ds)+1)+1);

			par->iyobs[ir] = my_float2int(par->yobs[ir]/par->ds)+1 - par->ymin + DELTA;
			par->yobswt[ir] = (par->yobs[ir]/par->ds - (my_float2int(par->yobs[ir]/par->ds)+1)+1);

			par->izobs[ir] = get_block_description(0,0)->sizez + my_float2int(par->zobs[ir]/par->ds) -1;
			par->zobswt[ir] = (par->zobs[ir]/par->ds - (my_float2int(par->zobs[ir]/par->ds)));

			// printf("station %d\n\tobswt : %f %f %f\n", ir, par->xobswt[ir], par->yobswt[ir], par->zobswt[ir]);
			// printf("\tiobst : %d %d %d\n", par->ixobs[ir], par->iyobs[ir], par->izobs[ir]);
		}
		
		unsigned bx, by, nbx, nby;
	   nbx = get_nbx();
	   nby = get_nby();
	   for (bx = 0; bx < nbx; bx++)
	   {	for (by = 0; by < nby; by++)
	    	{
		      	struct block_description * block = get_block_description(bx, by);
		      	unsigned node = block->mpi_node;

				if (node == rank) {
		      		block->nb_sta = 0;
		     		block->ista = (unsigned*)malloc(par->iobs*sizeof(unsigned));

					// bounds
					int xmin_loc, xmax_loc, ymin_loc, ymax_loc;
					xmin_loc = block->xshift;
					ymin_loc = block->yshift;
					xmax_loc = xmin_loc + block->sizex;
					ymax_loc = ymin_loc + block->sizey;

					// pas de stations dans les CPMLS
					if (xmin_loc < DELTA) xmin_loc = DELTA;
					if (ymin_loc < DELTA) ymin_loc = DELTA;

					int sizex = par->xmax- par->xmin + 1 + 2*DELTA;
					int sizey = par->ymax- par->ymin + 1 + 2*DELTA;

					if (xmax_loc >= (sizex-DELTA)) xmax_loc = sizex-DELTA-1;
					if (ymax_loc >= (sizey-DELTA)) ymax_loc = sizey-DELTA-1;

					INFO("PROC %d, BLOCK %d/%d\n",rank,bx,by);

			      	for (int ir = 0; ir < par->iobs; ir++){

						if (!( par->ixobs[ir] > xmax_loc-2 || par->ixobs[ir] < xmin_loc || par->iyobs[ir] > ymax_loc-2 || par->iyobs[ir] < ymin_loc || par->izobs[ir] < DELTA || par->izobs[ir] > block->sizez )) {
							// source ir in thir block
							block->nb_sta++;
							nb_local_obs++;
							block->ista[ir] = 1;

							if (VERBOSE >= 3) {
								printf("\tstation %d\n", ir+1);
								printf("\t\tobswt : %f %f %f\n", par->xobswt[ir], par->yobswt[ir], par->zobswt[ir]);
								printf("\t\tiobst : %d %d %d\n", par->ixobs[ir], par->iyobs[ir], par->izobs[ir]);
							}
						} else {
							block->ista[ir] = 0;
						}
				    }
					// allocate_dble_vector_on_node(&block->seismo_handle[0], &block->seisx , par->iobs*par->tmax);
					// allocate_dble_vector_on_node(&block->seismo_handle[1], &block->seisy , par->iobs*par->tmax);
					// allocate_dble_vector_on_node(&block->seismo_handle[2], &block->seisz , par->iobs*par->tmax);
					// allocate_dble_vector_on_node(&block->seismo_handle[3], &block->seisxx, par->iobs*par->tmax);
					// allocate_dble_vector_on_node(&block->seismo_handle[4], &block->seisyy, par->iobs*par->tmax);
					// allocate_dble_vector_on_node(&block->seismo_handle[5], &block->seiszz, par->iobs*par->tmax);
					// allocate_dble_vector_on_node(&block->seismo_handle[6], &block->seisxy, par->iobs*par->tmax);
					// allocate_dble_vector_on_node(&block->seismo_handle[7], &block->seisxz, par->iobs*par->tmax);
					// allocate_dble_vector_on_node(&block->seismo_handle[8], &block->seisyz, par->iobs*par->tmax);
					if (block->nb_sta*par->tmax) {
						allocate_dble_vector_on_node(&block->seismo_handle[0], &block->seisx , block->nb_sta*par->tmax);
						allocate_dble_vector_on_node(&block->seismo_handle[1], &block->seisy , block->nb_sta*par->tmax);
						allocate_dble_vector_on_node(&block->seismo_handle[2], &block->seisz , block->nb_sta*par->tmax);
						allocate_dble_vector_on_node(&block->seismo_handle[3], &block->seisxx, block->nb_sta*par->tmax);
						allocate_dble_vector_on_node(&block->seismo_handle[4], &block->seisyy, block->nb_sta*par->tmax);
						allocate_dble_vector_on_node(&block->seismo_handle[5], &block->seiszz, block->nb_sta*par->tmax);
						allocate_dble_vector_on_node(&block->seismo_handle[6], &block->seisxy, block->nb_sta*par->tmax);
						allocate_dble_vector_on_node(&block->seismo_handle[7], &block->seisxz, block->nb_sta*par->tmax);
						allocate_dble_vector_on_node(&block->seismo_handle[8], &block->seisyz, block->nb_sta*par->tmax);
					} else {/*allocate with size=1 to avoid null pointers*/
						allocate_dble_vector_on_node(&block->seismo_handle[0], &block->seisx , 1);
						allocate_dble_vector_on_node(&block->seismo_handle[1], &block->seisy , 1);
						allocate_dble_vector_on_node(&block->seismo_handle[2], &block->seisz , 1);
						allocate_dble_vector_on_node(&block->seismo_handle[3], &block->seisxx, 1);
						allocate_dble_vector_on_node(&block->seismo_handle[4], &block->seisyy, 1);
						allocate_dble_vector_on_node(&block->seismo_handle[5], &block->seiszz, 1);
						allocate_dble_vector_on_node(&block->seismo_handle[6], &block->seisxy, 1);
						allocate_dble_vector_on_node(&block->seismo_handle[7], &block->seisxz, 1);
						allocate_dble_vector_on_node(&block->seismo_handle[8], &block->seisyz, 1);
					}
					INFO("block {%d, %d} has %d stations\n",bx,by,block->nb_sta);
			    }	
			}
		}
	   
	#ifdef STARPU_USE_MPI
		int iobs;
		// int* station_mask = (int*)malloc;(par->iobs*sizeof(int));
		MPI_Reduce ( &nb_local_obs, &nb_global_obs, 1, MPI_INTEGER, MPI_SUM, MASTER, MPI_COMM_WORLD);
		if (rank == MASTER && par->iobs != nb_global_obs) {
				printf("WARNING : %d stations on %d are defined outside the domain\n",par->iobs-nb_global_obs, par->iobs);
		}
		// free(station_mask);
	#endif
		
	}

	void set_material_properties(ondes3d_params* par, int rank) 
	{

		if (VERBOSE >= 3) printf("DEFINITION OF THE MATERIAL PROPERTIES\n");
		
		unsigned bx, by, nbx, nby;
	   nbx = get_nbx();
	   nby = get_nby();
	   for (bx = 0; bx < nbx; bx++)
	   {  for (by = 0; by < nby; by++)
	      {
	      	struct block_description * block = get_block_description(bx, by);
	      	unsigned node = block->mpi_node;

	      	if (node == rank) {
	      		// DGN ATTENTION : on définit les CPMLs aussi
	      		for (int k=0; k<block->pitchz; k++) {
	      			// z coord réelle
	      			int depth = k + par->zmin - DELTA - K;
	      			float zdum = (depth-1)*par->ds/1000.;
	      			bool layer_not_found = true;

						if ( zdum > par->laydep[0] ){ /* shallow part */
							for (int j=0; j<block->pitchy; j++) {
				   			for (int i=0; i<block->pitchx; i++) {
				   				if (indomain(block, i, j, k, true)) {
										float vsdum = hardrock( -zdum ); /* for Alps, Pyrenees */
										float vpdum = vsdum*sqrt(3.);
										float rhodum = 1741.*powf(vpdum/1000., 0.25);

					      			block->rho[RAW_ACCESS(block, i,j,k)] = MIN(par->rho0[0], vsdum);
					      			block->vp[RAW_ACCESS(block, i,j,k)] = MIN(par->vp0[0], vpdum);
					      			block->vs[RAW_ACCESS(block, i,j,k)] = MIN(par->vs0[0], rhodum);
					      		}
		   					}
		   				}
						} else {
							int ly;
							for (ly = 0; ly < par->nlayer-1; ly++)
								if ( zdum <= par->laydep[ly] && zdum > par->laydep[ly+1] ) {
									layer_not_found = false;
									break;
								}

							if (layer_not_found) {
								perror("set_material_properties() : layer not found\n");
								STOP
							} else {
								for (int j=0; j<block->pitchy; j++) {
					   			for (int i=0; i<block->pitchx; i++) {
					   				if (indomain(block, i, j, k, true)) {
						      			block->rho[RAW_ACCESS(block, i,j,k)] = par->rho0[ly];
						      			block->vp[RAW_ACCESS(block, i,j,k)] = par->vp0[ly];
						      			block->vs[RAW_ACCESS(block, i,j,k)] = par->vs0[ly];
						      		}
			   					}
			   				}
							}
						}
	   			}
	      	}
	      }
	   }
	   
		if (rank == MASTER) {
			if (VERBOSE >= 2) {
				printf("MATERIAL PROPERTY :\n");
				printf("number of layers : %d\n",par->nlayer);
				for (int i=0;i<par->nlayer;i++){
					printf("layer %d : depth = %f\n",i,par->laydep[i]);
					printf("\t\t- rho = %f\n",par->rho0[i]);
					printf("\t\t- vp0 = %f\n",par->vp0[i]);
					printf("\t\t- vs0 = %f\n",par->vs0[i]);
					printf("\t\t- mu = %f\n",par->vs0[i]*par->vs0[i]*par->rho0[i]);
					printf("\t\t- lambda = %f\n",par->vp0[i]*par->vp0[i]*par->rho0[i]-2.0*(par->vs0[i]*par->vs0[i]*par->rho0[i]));
				}
			}
		}
	}

	void set_cpmls(ondes3d_params* par, int rank)
	{

		if (VERBOSE >= 3 && rank == MASTER) printf("EXTENSION OF MATERIAL PROPERTIES IN THE CPML LAYERS\n& DEFINITION OF THE VECTORS USED IN THE CPML FORMULATION\n");

		unsigned bx, by, nbx, nby;
	   nbx = get_nbx();
	   nby = get_nby();

		bool bxmin, bxmax, bymin, bymax;
		int ixs_min, ixe_min, ixs_max, ixe_max;
		int iys_min, iye_min, iys_max, iye_max;
		
	   for (bx = 0; bx < nbx; bx++)
	   {  for (by = 0; by < nby; by++)
	      {
	      	struct block_description * block = get_block_description(bx, by);
	      	unsigned node = block->mpi_node;

	      	if (node == rank) {
	      		/* 5 faces */
	      		for (int k=-K; k<DELTA; k++) { /*zmin -> every block*/
						for (int j=0; j<block->sizey; j++) {
						   for (int i=0; i<block->sizex; i++) {
								block->vp[ACCESS(block,i,j,k)] = block->vp[ACCESS(block,i,j,DELTA)];
								block->vs[ACCESS(block,i,j,k)] = block->vs[ACCESS(block,i,j,DELTA)];
								block->rho[ACCESS(block,i,j,k)] = block->rho[ACCESS(block,i,j,DELTA)];
								block->lam[ACCESS(block,i,j,k)] = block->lam[ACCESS(block,i,j,DELTA)];
								block->mu[ACCESS(block,i,j,k)] = block->mu[ACCESS(block,i,j,DELTA)];
							}
						}
					}
					if (has_cpml(block, &bxmin, &bxmax, &bymin, &bymax, &ixs_min, &ixe_min, &iys_min, &iye_min, &ixs_max, &ixe_max, &iys_max, &iye_max)) {
						if (bxmin) { /*xmin face + bottom edge*/
							for (int k=0; k<block->sizez; k++) {
								for (int j=0; j<block->sizey; j++) {
								   for (int i=ixs_min-K; i<ixe_min; i++) {
										block->vp[ACCESS(block,i,j,k)] = block->vp[ACCESS(block,ixe_min,j,k)];
										block->vs[ACCESS(block,i,j,k)] = block->vs[ACCESS(block,ixe_min,j,k)];
										block->rho[ACCESS(block,i,j,k)] = block->rho[ACCESS(block,ixe_min,j,k)];
										block->lam[ACCESS(block,i,j,k)] = block->lam[ACCESS(block,ixe_min,j,k)];
										block->mu[ACCESS(block,i,j,k)] = block->mu[ACCESS(block,ixe_min,j,k)];
									}
								}
							}
							for (int k=-K; k<DELTA; k++) {
								for (int j=0; j<block->sizey; j++) {
								   for (int i=ixs_min-K; i<ixe_min; i++) {
										block->vp[ACCESS(block,i,j,k)] = block->vp[ACCESS(block,ixe_min,j,DELTA)];
										block->vs[ACCESS(block,i,j,k)] = block->vs[ACCESS(block,ixe_min,j,DELTA)];
										block->rho[ACCESS(block,i,j,k)] = block->rho[ACCESS(block,ixe_min,j,DELTA)];
										block->lam[ACCESS(block,i,j,k)] = block->lam[ACCESS(block,ixe_min,j,DELTA)];
										block->mu[ACCESS(block,i,j,k)] = block->mu[ACCESS(block,ixe_min,j,DELTA)];
									}
								}
							}
						}
						if (bymin) { /*ymin face + bottom edge*/
							for (int k=0; k<block->sizez; k++) {
								for (int j=iys_min-K; j<iye_min; j++) {
								   for (int i=0; i<block->sizex; i++) {
										block->vp[ACCESS(block,i,j,k)] = block->vp[ACCESS(block,i,iye_min,k)];
										block->vs[ACCESS(block,i,j,k)] = block->vs[ACCESS(block,i,iye_min,k)];
										block->rho[ACCESS(block,i,j,k)] = block->rho[ACCESS(block,i,iye_min,k)];
										block->lam[ACCESS(block,i,j,k)] = block->lam[ACCESS(block,i,iye_min,k)];
										block->mu[ACCESS(block,i,j,k)] = block->mu[ACCESS(block,i,iye_min,k)];
									}
								}
							}
							for (int k=-K; k<DELTA; k++) {
								for (int j=iys_min-K; j<iye_min; j++) {
								   for (int i=0; i<block->sizex; i++) {
										block->vp[ACCESS(block,i,j,k)] = block->vp[ACCESS(block,i,iye_min,DELTA)];
										block->vs[ACCESS(block,i,j,k)] = block->vs[ACCESS(block,i,iye_min,DELTA)];
										block->rho[ACCESS(block,i,j,k)] = block->rho[ACCESS(block,i,iye_min,DELTA)];
										block->lam[ACCESS(block,i,j,k)] = block->lam[ACCESS(block,i,iye_min,DELTA)];
										block->mu[ACCESS(block,i,j,k)] = block->mu[ACCESS(block,i,iye_min,DELTA)];
									}
								}
							}
						}
						if (bxmax) { /*xmax face + bottom edge*/
							for (int k=0; k<block->sizez; k++) {
								for (int j=0; j<block->sizey; j++) {
								   for (int i=ixs_max; i<ixe_max+K; i++) {
										block->vp[ACCESS(block,i,j,k)] = block->vp[ACCESS(block,ixs_max-1,j,k)];
										block->vs[ACCESS(block,i,j,k)] = block->vs[ACCESS(block,ixs_max-1,j,k)];
										block->rho[ACCESS(block,i,j,k)] = block->rho[ACCESS(block,ixs_max-1,j,k)];
										block->lam[ACCESS(block,i,j,k)] = block->lam[ACCESS(block,ixs_max-1,j,k)];
										block->mu[ACCESS(block,i,j,k)] = block->mu[ACCESS(block,ixs_max-1,j,k)];
									}
								}
							}
							for (int k=-K; k<DELTA; k++) {
								for (int j=0; j<block->sizey; j++) {
								   for (int i=ixs_max; i<ixe_max+K; i++) {
										block->vp[ACCESS(block,i,j,k)] = block->vp[ACCESS(block,ixs_max-1,j,DELTA)];
										block->vs[ACCESS(block,i,j,k)] = block->vs[ACCESS(block,ixs_max-1,j,DELTA)];
										block->rho[ACCESS(block,i,j,k)] = block->rho[ACCESS(block,ixs_max-1,j,DELTA)];
										block->lam[ACCESS(block,i,j,k)] = block->lam[ACCESS(block,ixs_max-1,j,DELTA)];
										block->mu[ACCESS(block,i,j,k)] = block->mu[ACCESS(block,ixs_max-1,j,DELTA)];
									}
								}
							}
						}
						if (bymax) { /*ymax face + bottom edge*/
							for (int k=0; k<block->sizez; k++) {
								for (int j=iys_max; j<iye_max+K; j++) {
								   for (int i=0; i<block->sizex; i++) {
										block->vp[ACCESS(block,i,j,k)] = block->vp[ACCESS(block,i,iys_max-1,k)];
										block->vs[ACCESS(block,i,j,k)] = block->vs[ACCESS(block,i,iys_max-1,k)];
										block->rho[ACCESS(block,i,j,k)] = block->rho[ACCESS(block,i,iys_max-1,k)];
										block->lam[ACCESS(block,i,j,k)] = block->lam[ACCESS(block,i,iys_max-1,k)];
										block->mu[ACCESS(block,i,j,k)] = block->mu[ACCESS(block,i,iys_max-1,k)];
									}
								}
							}
							for (int k=-K; k<DELTA; k++) {
								for (int j=iys_max; j<iye_max+K; j++) {
								   for (int i=0; i<block->sizex; i++) {
										block->vp[ACCESS(block,i,j,k)] = block->vp[ACCESS(block,i,iys_max-1,DELTA)];
										block->vs[ACCESS(block,i,j,k)] = block->vs[ACCESS(block,i,iys_max-1,DELTA)];
										block->rho[ACCESS(block,i,j,k)] = block->rho[ACCESS(block,i,iys_max-1,DELTA)];
										block->lam[ACCESS(block,i,j,k)] = block->lam[ACCESS(block,i,iys_max-1,DELTA)];
										block->mu[ACCESS(block,i,j,k)] = block->mu[ACCESS(block,i,iys_max-1,DELTA)];
									}
								}
							}
						}
						/*4 vertical edges + 4 bottom corners*/
						if (bxmin && bymin) {
							for (int k=0; k<block->sizez; k++) {
								for (int j=iys_min-K; j<ixe_min; j++) {
								   for (int i=ixs_min-K; i<ixe_min; i++) {
										block->vp[ACCESS(block,i,j,k)] = block->vp[ACCESS(block,ixe_min,iye_min,k)];
										block->vs[ACCESS(block,i,j,k)] = block->vs[ACCESS(block,ixe_min,iye_min,k)];
										block->rho[ACCESS(block,i,j,k)] = block->rho[ACCESS(block,ixe_min,iye_min,k)];
										block->lam[ACCESS(block,i,j,k)] = block->lam[ACCESS(block,ixe_min,iye_min,k)];
										block->mu[ACCESS(block,i,j,k)] = block->mu[ACCESS(block,ixe_min,iye_min,k)];
									}
								}
							}
							for (int k=-K; k<DELTA; k++) {
								for (int j=iys_min-K; j<ixe_min; j++) {
								   for (int i=ixs_min-K; i<ixe_min; i++) {
										block->vp[ACCESS(block,i,j,k)] = block->vp[ACCESS(block,ixe_min,iye_min,DELTA)];
										block->vs[ACCESS(block,i,j,k)] = block->vs[ACCESS(block,ixe_min,iye_min,DELTA)];
										block->rho[ACCESS(block,i,j,k)] = block->rho[ACCESS(block,ixe_min,iye_min,DELTA)];
										block->lam[ACCESS(block,i,j,k)] = block->lam[ACCESS(block,ixe_min,iye_min,DELTA)];
										block->mu[ACCESS(block,i,j,k)] = block->mu[ACCESS(block,ixe_min,iye_min,DELTA)];
									}
								}
							}
						}
						if (bxmin && bymax) {
							for (int k=0; k<block->sizez; k++) {
								for (int j=iys_max; j<iye_max+K; j++) {
								   for (int i=ixs_min-K; i<ixe_min; i++) {
										block->vp[ACCESS(block,i,j,k)] = block->vp[ACCESS(block,ixe_min,iys_max-1,k)];
										block->vs[ACCESS(block,i,j,k)] = block->vs[ACCESS(block,ixe_min,iys_max-1,k)];
										block->rho[ACCESS(block,i,j,k)] = block->rho[ACCESS(block,ixe_min,iys_max-1,k)];
										block->lam[ACCESS(block,i,j,k)] = block->lam[ACCESS(block,ixe_min,iys_max-1,k)];
										block->mu[ACCESS(block,i,j,k)] = block->mu[ACCESS(block,ixe_min,iys_max-1,k)];
									}
								}
							}
							for (int k=-K; k<DELTA; k++) {
								for (int j=iys_max; j<iye_max+K; j++) {
								   for (int i=ixs_min-K; i<ixe_min; i++) {
										block->vp[ACCESS(block,i,j,k)] = block->vp[ACCESS(block,ixe_min,iys_max-1,DELTA)];
										block->vs[ACCESS(block,i,j,k)] = block->vs[ACCESS(block,ixe_min,iys_max-1,DELTA)];
										block->rho[ACCESS(block,i,j,k)] = block->rho[ACCESS(block,ixe_min,iys_max-1,DELTA)];
										block->lam[ACCESS(block,i,j,k)] = block->lam[ACCESS(block,ixe_min,iys_max-1,DELTA)];
										block->mu[ACCESS(block,i,j,k)] = block->mu[ACCESS(block,ixe_min,iys_max-1,DELTA)];
									}
								}
							}
						}
						if (bxmax && bymin) {
							for (int k=0; k<block->sizez; k++) {
								for (int j=iys_min-K; j<ixe_min; j++) {
								   for (int i=ixs_max; i<ixe_max+K; i++) {
										block->vp[ACCESS(block,i,j,k)] = block->vp[ACCESS(block,ixs_max-1,iye_min,k)];
										block->vs[ACCESS(block,i,j,k)] = block->vs[ACCESS(block,ixs_max-1,iye_min,k)];
										block->rho[ACCESS(block,i,j,k)] = block->rho[ACCESS(block,ixs_max-1,iye_min,k)];
										block->lam[ACCESS(block,i,j,k)] = block->lam[ACCESS(block,ixs_max-1,iye_min,k)];
										block->mu[ACCESS(block,i,j,k)] = block->mu[ACCESS(block,ixs_max-1,iye_min,k)];
									}
								}
							}
							for (int k=-K; k<DELTA; k++) {
								for (int j=iys_min-K; j<ixe_min; j++) {
								   for (int i=ixs_max; i<ixe_max+K; i++) {
										block->vp[ACCESS(block,i,j,k)] = block->vp[ACCESS(block,ixs_max-1,iye_min,DELTA)];
										block->vs[ACCESS(block,i,j,k)] = block->vs[ACCESS(block,ixs_max-1,iye_min,DELTA)];
										block->rho[ACCESS(block,i,j,k)] = block->rho[ACCESS(block,ixs_max-1,iye_min,DELTA)];
										block->lam[ACCESS(block,i,j,k)] = block->lam[ACCESS(block,ixs_max-1,iye_min,DELTA)];
										block->mu[ACCESS(block,i,j,k)] = block->mu[ACCESS(block,ixs_max-1,iye_min,DELTA)];
									}
								}
							}
						}
						if (bxmax && bymax) {
							for (int k=0; k<block->sizez; k++) {
								for (int j=iys_max; j<iye_max+K; j++) {
								   for (int i=ixs_max; i<ixe_max+K; i++) {
										block->vp[ACCESS(block,i,j,k)] = block->vp[ACCESS(block,ixs_max-1,iys_max-1,k)];
										block->vs[ACCESS(block,i,j,k)] = block->vs[ACCESS(block,ixs_max-1,iys_max-1,k)];
										block->rho[ACCESS(block,i,j,k)] = block->rho[ACCESS(block,ixs_max-1,iys_max-1,k)];
										block->lam[ACCESS(block,i,j,k)] = block->lam[ACCESS(block,ixs_max-1,iys_max-1,k)];
										block->mu[ACCESS(block,i,j,k)] = block->mu[ACCESS(block,ixs_max-1,iys_max-1,k)];
									}
								}
							}
							for (int k=-K; k<DELTA; k++) {
								for (int j=iys_max; j<iye_max+K; j++) {
								   for (int i=ixs_max; i<ixe_max+K; i++) {
										block->vp[ACCESS(block,i,j,k)] = block->vp[ACCESS(block,ixs_max-1,iys_max-1,DELTA)];
										block->vs[ACCESS(block,i,j,k)] = block->vs[ACCESS(block,ixs_max-1,iys_max-1,DELTA)];
										block->rho[ACCESS(block,i,j,k)] = block->rho[ACCESS(block,ixs_max-1,iys_max-1,DELTA)];
										block->lam[ACCESS(block,i,j,k)] = block->lam[ACCESS(block,ixs_max-1,iys_max-1,DELTA)];
										block->mu[ACCESS(block,i,j,k)] = block->mu[ACCESS(block,ixs_max-1,iys_max-1,DELTA)];
									}
								}
							}
						}
					}/*endif has_cpml*/

					/*vu qu'ils sont alloués inconditionnellement pour tous les blocs, on les défini*/
					// for (int i = 0; i < block->sizex; i++){
					// 	block->dumpx[i] = 0.0;
					// 	block->dumpx2[i] = 0.0;
					// 	block->kappax[i] = 1.0;
					// 	block->kappax2[i] = 1.0;
					// 	block->alphax[i] = 0.0;
					// 	block->alphax2[i] = 0.0;
					// }
					// for (int j = 0; j < block->sizey; j++){
					// 	block->dumpy[j] = 0.0;
					// 	block->dumpy2[j] = 0.0;
					// 	block->kappay[j] = 1.0;
					// 	block->kappay2[j] = 1.0;
					// 	block->alphay[j] = 0.0;
					// 	block->alphay2[j] = 0.0;
					// }
					// for (int k = 0; k < block->sizez; k++){
					// 	block->dumpz[k] = 0.0;
					// 	block->dumpz2[k] = 0.0;
					// 	block->kappaz[k] = 1.0;
					// 	block->kappaz2[k] = 1.0;
					// 	block->alphaz[k] = 0.0;
					// 	block->alphaz2[k] = 0.0;
					// }
					// float abscissa_normalized;
					// if (has_cpml(block, &bxmin, &bxmax, &bymin, &bymax, &ixs_min, &ixe_min, &iys_min, &iye_min, &ixs_max, &ixe_max, &iys_max, &iye_max)) {
					// 	if (bxmin) {
					// 		for(int i=0; i< block->sizex; i++) {/*DGN : attention si la taille du bloc > DELTA, ça ne marche plus ...*/
								
					// 			abscissa_normalized = (ixe_min - i)/(float)DELTA;
					// 			if (abscissa_normalized >= 0.0) {
					// 				block->dumpx[i] = par->dump0 * powf(abscissa_normalized,NPOWER);
					// 				block->kappax[i] = 1.0 + (par->kappa0 - 1.0) * powf(abscissa_normalized,NPOWER);
					// 				block->alphax[i] = par->alpha0 * (1.0 - abscissa_normalized);
					// 			}
								
					// 			abscissa_normalized = (ixe_min - i - 0.5)/(float)DELTA;
					// 			if (abscissa_normalized >= 0.0) {
					// 				block->dumpx2[i] = par->dump0 * powf(abscissa_normalized,NPOWER);
					// 				block->kappax2[i] = 1.0 + (par->kappa0 - 1.0) * powf(abscissa_normalized,NPOWER);
					// 				block->alphax2[i] = par->alpha0 * (1.0 - abscissa_normalized);
					// 			}

					// 			if(alphax[i] < 0.0) alphax[i] = 0.0;
					// 			if(alphax2[i] < 0.0) alphax2[i] = 0.0;	
					// 		}
					// 	}
					// 	if(bxmax) {
					// 		for(int i=0; i< block->sizex; i++) {/*DGN : attention si la taille du bloc > DELTA, ça ne marche plus ...*/
								
					// 			abscissa_normalized = (i - (ixs_max-1)/(float)DELTA;
					// 			if (abscissa_normalized >= 0.0) {
					// 				block->dumpx[i] = par->dump0 * powf(abscissa_normalized,NPOWER);
					// 				block->kappax[i] = 1.0 + (par->kappa0 - 1.0) * powf(abscissa_normalized,NPOWER);
					// 				block->alphax[i] = par->alpha0 * (1.0 - abscissa_normalized);
					// 			}
								
					// 			abscissa_normalized = (i - (ixs_max-1) + 0.5)/(float)DELTA;
					// 			if (abscissa_normalized >= 0.0) {
					// 				block->dumpx2[i] = par->dump0 * powf(abscissa_normalized,NPOWER);
					// 				block->kappax2[i] = 1.0 + (par->kappa0 - 1.0) * powf(abscissa_normalized,NPOWER);
					// 				block->alphax2[i] = par->alpha0 * (1.0 - abscissa_normalized);
					// 			}

					// 			if(alphax[i] < 0.0) alphax[i] = 0.0;
					// 			if(alphax2[i] < 0.0) alphax2[i] = 0.0;	
					// 		}
					// 	}
					// 	if (bymin) {
					// 		for(int i=0; i< block->sizey; i++) {

					// 			abscissa_normalized = (iye_min - i)/(float)DELTA;
					// 			if (abscissa_normalized >= 0.0) {
					// 				block->dumpy[i] = par->dump0 * powf(abscissa_normalized,NPOWER);
					// 				block->kappay[i] = 1.0 + (par->kappa0 - 1.0) * powf(abscissa_normalized,NPOWER);
					// 				block->alphay[i] = par->alpha0 * (1.0 - abscissa_normalized);
					// 			}
								
					// 			abscissa_normalized = (iye_min - i - 0.5)/(float)DELTA;
					// 			if (abscissa_normalized >= 0.0) {
					// 				block->dumpy2[i] = par->dump0 * powf(abscissa_normalized,NPOWER);
					// 				block->kappay2[i] = 1.0 + (par->kappa0 - 1.0) * powf(abscissa_normalized,NPOWER);
					// 				block->alphay2[i] = par->alpha0 * (1.0 - abscissa_normalized);
					// 			}

					// 			if(alphay[i] < 0.0) alphay[i] = 0.0;
					// 			if(alphay2[i] < 0.0) alphay2[i] = 0.0;	
					// 		}
					// 	}
					// 	if(bymax) {
					// 		for(int i=0; i< block->sizey; i++) {
								
					// 			abscissa_normalized = (i - (iys_max-1)/(float)DELTA;
					// 			if (abscissa_normalized >= 0.0) {
					// 				block->dumpy[i] = par->dump0 * powf(abscissa_normalized,NPOWER);
					// 				block->kappay[i] = 1.0 + (par->kappa0 - 1.0) * powf(abscissa_normalized,NPOWER);
					// 				block->alphay[i] = par->alpha0 * (1.0 - abscissa_normalized);
					// 			}
								
					// 			abscissa_normalized = (i - (iys_max-1) + 0.5)/(float)DELTA;
					// 			if (abscissa_normalized >= 0.0) {
					// 				block->dumpy2[i] = par->dump0 * powf(abscissa_normalized,NPOWER);
					// 				block->kappay2[i] = 1.0 + (par->kappa0 - 1.0) * powf(abscissa_normalized,NPOWER);
					// 				block->alphay2[i] = par->alpha0 * (1.0 - abscissa_normalized);
					// 			}

					// 			if(alphay[i] < 0.0) alphay[i] = 0.0;
					// 			if(alphay2[i] < 0.0) alphay2[i] = 0.0;	
					// 		}
					// 	}
					// }
					// for(int i=0; i< block->sizez; i++) {
						
					// 	abscissa_normalized = (DELTA - i)/(float)DELTA;
					// 	if (abscissa_normalized >= 0.0) {
					// 		block->dumpz[i] = par->dump0 * powf(abscissa_normalized,NPOWER);
					// 		block->kappaz[i] = 1.0 + (par->kappa0 - 1.0) * powf(abscissa_normalized,NPOWER);
					// 		block->alphaz[i] = par->alpha0 * (1.0 - abscissa_normalized);
					// 	}
						
					// 	abscissa_normalized = (DELTA - i - 0.5)/(float)DELTA;
					// 	if (abscissa_normalized >= 0.0) {
					// 		block->dumpz2[i] = par->dump0 * powf(abscissa_normalized,NPOWER);
					// 		block->kappaz2[i] = 1.0 + (par->kappa0 - 1.0) * powf(abscissa_normalized,NPOWER);
					// 		block->alphaz2[i] = par->alpha0 * (1.0 - abscissa_normalized);
					// 	}

					// 	if(alphaz[i] < 0.0) alphaz[i] = 0.0;
					// 	if(alphaz2[i] < 0.0) alphaz2[i] = 0.0;	
					// }
	      	}/*endif node==rank*/
	      }
	   }
	   
	   return;
	}
// }}}

// CPML LOCATION FUNCTS {{{
	// test if a coordinate in a block is in domain (extended or not) or in padding
	bool indomain(struct block_description* block, unsigned i, unsigned j, unsigned k, bool extended) {
		/*domain extended : size+2*K on each direction*/
		if (aligned()) {
			if (extended) {
				if (i<(ALIGN-K) || i>block->sizex+ALIGN+K) return false;
				if (j<(ALIGN-K) || j>block->sizey+ALIGN+K) return false;
				if (k<(ALIGN-K) || k>block->sizez+ALIGN+K) return false;
				return true;
			} else {
				if (i<ALIGN || i>block->sizex+ALIGN) return false;
				if (j<ALIGN || j>block->sizey+ALIGN) return false;
				if (k<ALIGN || k>block->sizez+ALIGN) return false;
				return true;
			}
		} else {
			if (extended) return true;
			else {
				if (i<K || i>block->sizex+K) return false;
				if (j<K || j>block->sizey+K) return false;
				if (k<K || k>block->sizez+K) return false;
				return true;
			}
		}
	}

	bool has_cpml(struct block_description* block, bool* bxmin, bool* bxmax, bool* bymin, bool* bymax, int* ixs_min, int* ixe_min, int* iys_min, int* iye_min, int* ixs_max, int* ixe_max, int* iys_max, int* iye_max)
	{
		if (block->nb_cpml > block->sizex*block->sizey*DELTA) return false;

		// DGN : doit-on étendre les propriétés CPMLs dans les ghosts du bloc vers l'intérieur du domaine : attention aux blocs petits (size<DELTA : surement mal géré !!!)

		if (block->xshift > DELTA) *bxmin = false;
		else {
			*bxmin = true;
			*ixs_min=0;
			if (block->bx == 0) {
				*ixe_min=MIN(DELTA, block->sizex);
			} else {
				*ixe_min=MIN(DELTA-block->xshift, block->sizex);
			}
		}

		if (block->yshift > DELTA) *bymin = false;
		else {
			*bymin = true;
			*iys_min=0;
			if (block->by == 0) {
				*iye_min=MIN(DELTA, block->sizey);
			} else {
				*iye_min=MIN(DELTA-block->yshift, block->sizey);
			}
		}

		ondes3d_params* par = get_params();
		int restex, restey;
		restex = par->sizex - block->xshift - block->sizex;
		restey = par->sizey - block->yshift - block->sizey;

		if (restex > DELTA) *bxmax = false;
		else {
			*bxmax = true;
			*ixe_max=block->sizex;
			if (block->bx == (get_nbx()-1)) {
				*ixs_max = MAX(0, block->sizex-DELTA);
			} else {
				*ixs_max = MAX(0, block->sizex-DELTA+restex);
			}
		}

		if (restey > DELTA) *bymax = false;
		else {
			*bymax = true;
			*iye_max=block->sizey;
			if (block->by == (get_nby()-1)) {
				*iys_max = MAX(0, block->sizey-DELTA);
			} else {
				*iys_max = MAX(0, block->sizey-DELTA+restey);
			}
		}
		return true;
	}
// }}}	

// CPU COMPUTATION KERNELS {{{
	void cpu_compute_veloc (	float* txx0, float* tyy0, float* tzz0, float* txy0, float* txz0, float* tyz0,
										float* vx0, float* vy0, float* vz0,
										float* fx, float* fy, float* fz, 
										int* npml_tab, float* phitxxx, float* phitxyy, float* phitxzz, float *phitxyx, float *phityyy, float *phityzz, float *phitxzx, float *phityzy, float *phitzzz,
										float* vp, float* rho,
										int sizex, int sizey, int sizez,
										int pitch_x, int pitch_y, int pitch_z, 
										float ds, float dt, int delta, int position,
										int ixe_min, int ixs_max, int iye_min, int iys_max, float dump0, float kappa0, float alpha0)
	{
		/* Boundary conditions included*/
		float phixdum, phiydum, phizdum, rhoxy, vpxy, rhoxz, vpxz;
		float abscissa_normalized;

		for ( int k = 0; k < sizez; k++){
			abscissa_normalized = (DELTA - k)/(float)DELTA;
		   int distance_zmin = k;
         int distance_zmax = sizez - 1 - k;

			float dumpz_  = (abscissa_normalized>=0)?dump0*powf(abscissa_normalized,NPOWER):0.0;
			float kappaz_ = (abscissa_normalized>=0)?1.0 + (kappa0 - 1.0) * powf(abscissa_normalized,NPOWER):1.0;
			float alphaz_ = (abscissa_normalized>=0)?alpha0 * (1.0 - abscissa_normalized):0.0;
			float dumpz2_ = (abscissa_normalized>=0)?dump0*powf(abscissa_normalized,NPOWER):0.0;
			float kappaz2_= (abscissa_normalized>=0)?1.0 + (kappa0 - 1.0) * powf(abscissa_normalized,NPOWER):1.0;
			float alphaz2_= (abscissa_normalized>=0)?alpha0 * (1.0 - abscissa_normalized):0.0;


			for ( int j = 0; j < sizey; j++ ){
				abscissa_normalized = MAX((iye_min - j)/(float)DELTA, (j - (iys_max-1)/(float)DELTA));
            int distance_ymin = (position & MASK_FIRST_Y)?j:DUMMY_VALUE;
            int distance_ymax = (position & MASK_LAST_Y)?(sizey - j - 1):DUMMY_VALUE;
				
				float dumpy_  = (abscissa_normalized>=0)?dump0*powf(abscissa_normalized,NPOWER):0.0;
				float kappay_ = (abscissa_normalized>=0)?1.0 + (kappa0 - 1.0) * powf(abscissa_normalized,NPOWER):1.0;
				float alphay_ = (abscissa_normalized>=0)?alpha0 * (1.0 - abscissa_normalized):0.0;
				float dumpy2_ = (abscissa_normalized>=0)?dump0*powf(abscissa_normalized,NPOWER):0.0;
				float kappay2_= (abscissa_normalized>=0)?1.0 + (kappa0 - 1.0) * powf(abscissa_normalized,NPOWER):1.0;
				float alphay2_= (abscissa_normalized>=0)?alpha0 * (1.0 - abscissa_normalized):0.0;

				for ( int i = 0; i < sizex; i++ ){
					int offset = k*pitch_x*pitch_y + j*pitch_x + i;
					int npml = npml_tab[k*(sizex)*(sizey) + j*(sizex) + i];

				   int distance_xmin = (position & MASK_FIRST_X)?i:DUMMY_VALUE;
	            int distance_xmax = (position & MASK_LAST_X)?(sizex - i - 1):DUMMY_VALUE;

					if (npml >= 0) {/* CPML */
	            	abscissa_normalized = MAX((ixe_min - i)/(float)DELTA, (i - (ixs_max-1)/(float)DELTA));

						float dumpx_  = (abscissa_normalized>=0)?dump0*powf(abscissa_normalized,NPOWER):0.0;
						float kappax_ = (abscissa_normalized>=0)?1.0 + (kappa0 - 1.0) * powf(abscissa_normalized,NPOWER):1.0;
						float alphax_ = (abscissa_normalized>=0)?alpha0 * (1.0 - abscissa_normalized):0.0;
						float dumpx2_ = (abscissa_normalized>=0)?dump0*powf(abscissa_normalized,NPOWER):0.0;
						float kappax2_= (abscissa_normalized>=0)?1.0 + (kappa0 - 1.0) * powf(abscissa_normalized,NPOWER):1.0;
						float alphax2_= (abscissa_normalized>=0)?alpha0 * (1.0 - abscissa_normalized):0.0;

						/* Calculation of vx */
						if (distance_zmin >= 1 && distance_ymin >= 1 && distance_xmin >= 1) {
							if ( distance_zmax == 0 ){ /* free surface */
								phixdum = phitxxx[npml];
								phiydum = phitxyy[npml];
								phizdum = phitxzz[npml];
								phitxxx[npml] = CPML2 (vp[offset], dumpx_, alphax_, kappax_, phixdum, ds, dt,
										txx0[k*pitch_x*pitch_y + j*pitch_x + i-1], txx0[k*pitch_x*pitch_y + j*pitch_x + i] );
								phitxyy[npml] = CPML2 (vp[offset], dumpy_, alphay_, kappay_, phiydum, ds, dt,
										txy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], txy0[k*pitch_x*pitch_y + j*pitch_x + i] );
								phitxzz[npml] = CPML2 (vp[offset], dumpz_, alphaz_, kappaz_, phizdum, ds, dt,
										txz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], - txz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i] );

								if (distance_xmax == 0 || distance_ymax ==0) { // boundary condition
	                        vx0[offset] = 0.0;
	                     } else {
									vx0[offset] += (dt/rho[offset])*( phitxxx[npml] + phitxyy[npml] + phitxzz[npml] )
										+ staggardv2 (1./rho[offset], kappax_, kappay_, kappaz_, dt, ds,
												txx0[k*pitch_x*pitch_y + j*pitch_x + i-1], txx0[k*pitch_x*pitch_y + j*pitch_x + i],
												txy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], txy0[k*pitch_x*pitch_y + j*pitch_x + i],
												txz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], - txz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i] );
								}
							} else if  ( distance_zmax == 1 ){ /* in the first cell, 2nd order finite-difference instead of 4th order finite-difference */
								phixdum = phitxxx[npml];
								phiydum = phitxyy[npml];
								phizdum = phitxzz[npml];
								phitxxx[npml] = CPML2 (vp[offset], dumpx_, alphax_, kappax_, phixdum, ds, dt,
										txx0[k*pitch_x*pitch_y + j*pitch_x + i-1], txx0[k*pitch_x*pitch_y + j*pitch_x + i] );
								phitxyy[npml] = CPML2 (vp[offset], dumpy_, alphay_, kappay_, phiydum, ds, dt,
										txy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], txy0[k*pitch_x*pitch_y + j*pitch_x + i] );
								phitxzz[npml] = CPML2 (vp[offset], dumpz_, alphaz_, kappaz_, phizdum, ds, dt,
										txz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], txz0[k*pitch_x*pitch_y + j*pitch_x + i] );
								
								if (distance_xmax == 0 || distance_ymax ==0) { // boundary condition
	                        vx0[offset] = 0.0f;
	                     } else {
									vx0[offset] += (dt/rho[offset])*( phitxxx[npml] + phitxyy[npml] + phitxzz[npml] )
										+ staggardv2 (1./rho[offset], kappax_, kappay_, kappaz_, dt, ds,
												txx0[k*pitch_x*pitch_y + j*pitch_x + i-1], txx0[k*pitch_x*pitch_y + j*pitch_x + i],
												txy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], txy0[k*pitch_x*pitch_y + j*pitch_x + i],
												txz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], txz0[k*pitch_x*pitch_y + j*pitch_x + i] );
								}
							} else { /* regular domain */
								phixdum = phitxxx[npml];
								phiydum = phitxyy[npml];
								phizdum = phitxzz[npml];
								phitxxx[npml] = CPML4 (vp[offset], dumpx_, alphax_, kappax_, phixdum, ds, dt,
										txx0[k*pitch_x*pitch_y + j*pitch_x + i-1], txx0[k*pitch_x*pitch_y + j*pitch_x + i],
										txx0[k*pitch_x*pitch_y + j*pitch_x + i-2], txx0[k*pitch_x*pitch_y + j*pitch_x + i+1] );
								phitxyy[npml] = CPML4 (vp[offset], dumpy_, alphay_, kappay_, phiydum, ds, dt,
										txy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], txy0[k*pitch_x*pitch_y + j*pitch_x + i],
										txy0[k*pitch_x*pitch_y + (j-2)*pitch_x + i], txy0[k*pitch_x*pitch_y + (j+1)*pitch_x + i] );
								phitxzz[npml] = CPML4 (vp[offset], dumpz_, alphaz_, kappaz_, phizdum, ds, dt,
										txz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], txz0[k*pitch_x*pitch_y + j*pitch_x + i],
										txz0[(k-2)*pitch_x*pitch_y + j*pitch_x + i], txz0[(k+1)*pitch_x*pitch_y + j*pitch_x + i] );

								if (distance_xmax == 0 || distance_ymax ==0) { // boundary condition
	                        vx0[offset] = 0.0;
	                     } else {
									vx0[offset] += (dt/rho[offset])*( phitxxx[npml] + phitxyy[npml] + phitxzz[npml] )
										+ staggardv4 (1./rho[offset], kappax_, kappay_, kappaz_, dt, ds,
												txx0[k*pitch_x*pitch_y + j*pitch_x + i-1], txx0[k*pitch_x*pitch_y + j*pitch_x + i],
												txx0[k*pitch_x*pitch_y + j*pitch_x + i-2], txx0[k*pitch_x*pitch_y + j*pitch_x + i+1],
												txy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], txy0[k*pitch_x*pitch_y + j*pitch_x + i],
												txy0[k*pitch_x*pitch_y + (j-2)*pitch_x + i], txy0[k*pitch_x*pitch_y + (j+1)*pitch_x + i],
												txz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], txz0[k*pitch_x*pitch_y + j*pitch_x + i],
												txz0[(k-2)*pitch_x*pitch_y + j*pitch_x + i], txz0[(k+1)*pitch_x*pitch_y + j*pitch_x + i] );
								}
							} /* end of if "free surface" */
						} /* end of calculation of vx */
						/* Calculation of vy */
						if ( distance_zmin >= 1 && distance_ymax >= 1 && distance_xmax >= 1 ){
							rhoxy = 0.25*(rho[offset] + rho[k*pitch_x*pitch_y + (j+1)*pitch_x + i]
									+ rho[k*pitch_x*pitch_y + j*pitch_x + i+1] + rho[k*pitch_x*pitch_y + (j+1)*pitch_x + i+1]);
							vpxy = 0.25f*(vp[offset] + vp[k*pitch_x*pitch_y + (j+1)*pitch_x + i] + vp[k*pitch_x*pitch_y + j*pitch_x + i+1] + vp[k*pitch_x*pitch_y + (j+1)*pitch_x + i+1]);

							if ( distance_zmax == 0 ){ /* free surface */
								phixdum = phitxyx[npml];
								phiydum = phityyy[npml];
								phizdum = phityzz[npml];
								phitxyx[npml] = CPML2 (vpxy, dumpx2_, alphax2_, kappax2_, phixdum, ds, dt,
										txy0[k*pitch_x*pitch_y + j*pitch_x + i], txy0[k*pitch_x*pitch_y + j*pitch_x + i+1] );
								phityyy[npml] = CPML2 (vpxy, dumpy2_, alphay2_, kappay2_, phiydum, ds, dt,
										tyy0[k*pitch_x*pitch_y + j*pitch_x + i], tyy0[k*pitch_x*pitch_y + (j+1)*pitch_x + i] );
								phityzz[npml] = CPML2 (vpxy, dumpz_, alphaz_, kappaz_, phizdum, ds, dt,
										tyz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], - tyz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i] );
								
								if (distance_xmin == 0 || distance_ymin == 0) { // boundary condition
	                        vy0[offset] = 0.0;
	                     } else {
									vy0[offset] += (dt/rhoxy)*( phitxyx[npml] + phityyy[npml] + phityzz[npml] )
										+ staggardv2 (1./rhoxy, kappax2_, kappay2_, kappaz_, dt, ds,
												txy0[k*pitch_x*pitch_y + j*pitch_x + i], txy0[k*pitch_x*pitch_y + j*pitch_x + i+1],
												tyy0[k*pitch_x*pitch_y + j*pitch_x + i], tyy0[k*pitch_x*pitch_y + (j+1)*pitch_x + i],
												tyz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], - tyz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i] );
								}
							} else if( distance_zmax == 1 ){ /* in the first cell, 2nd order finite-difference instead of 4th order finite-difference */
								phixdum = phitxyx[npml];
								phiydum = phityyy[npml];
								phizdum = phityzz[npml];
								phitxyx[npml] = CPML2 (vpxy, dumpx2_, alphax2_, kappax2_, phixdum, ds, dt,
										txy0[k*pitch_x*pitch_y + j*pitch_x + i], txy0[k*pitch_x*pitch_y + j*pitch_x + i+1] );
								phityyy[npml] = CPML2 (vpxy, dumpy2_, alphay2_, kappay2_, phiydum, ds, dt,
										tyy0[k*pitch_x*pitch_y + j*pitch_x + i], tyy0[k*pitch_x*pitch_y + (j+1)*pitch_x + i] );
								phityzz[npml] = CPML2 (vpxy, dumpz_, alphaz_, kappaz_, phizdum, ds, dt,
										tyz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], tyz0[k*pitch_x*pitch_y + j*pitch_x + i] );

								if (distance_xmin == 0 || distance_ymin == 0) { // boundary condition
	                        vy0[offset] = 0.0;
	                     } else {
									vy0[offset] += (dt/rhoxy)*( phitxyx[npml] + phityyy[npml] + phityzz[npml] )
										+ staggardv2 (1./rhoxy, kappax2_, kappay2_, kappaz_, dt, ds,
												txy0[k*pitch_x*pitch_y + j*pitch_x + i], txy0[k*pitch_x*pitch_y + j*pitch_x + i+1],
												tyy0[k*pitch_x*pitch_y + j*pitch_x + i], tyy0[k*pitch_x*pitch_y + (j+1)*pitch_x + i],
												tyz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], tyz0[k*pitch_x*pitch_y + j*pitch_x + i] );
								}
							} else { /* regular domain */
								phixdum = phitxyx[npml];
								phiydum = phityyy[npml];
								phizdum = phityzz[npml];
								phitxyx[npml] = CPML4 (vpxy, dumpx2_, alphax2_, kappax2_, phixdum, ds, dt,
										txy0[k*pitch_x*pitch_y + j*pitch_x + i], txy0[k*pitch_x*pitch_y + j*pitch_x + i+1],
										txy0[k*pitch_x*pitch_y + j*pitch_x + i-1], txy0[k*pitch_x*pitch_y + j*pitch_x + i+2] );
								phityyy[npml] = CPML4 (vpxy, dumpy2_, alphay2_, kappay2_, phiydum, ds, dt,
										tyy0[k*pitch_x*pitch_y + j*pitch_x + i], tyy0[k*pitch_x*pitch_y + (j+1)*pitch_x + i],
										tyy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], tyy0[k*pitch_x*pitch_y + (j+2)*pitch_x + i] );
								phityzz[npml] = CPML4 (vpxy, dumpz_, alphaz_, kappaz_, phizdum, ds, dt,
										tyz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], tyz0[k*pitch_x*pitch_y + j*pitch_x + i],
										tyz0[(k-2)*pitch_x*pitch_y + j*pitch_x + i], tyz0[(k+1)*pitch_x*pitch_y + j*pitch_x + i] );

								if (distance_xmin == 0 || distance_ymin == 0) { // boundary condition
	                        vy0[offset] = 0.0;
	                     } else {
									vy0[offset] += (dt/rhoxy)*( phitxyx[npml] + phityyy[npml] + phityzz[npml] )
										+ staggardv4 (1./rhoxy, kappax2_, kappay2_, kappaz_, dt, ds,
												txy0[k*pitch_x*pitch_y + j*pitch_x + i], txy0[k*pitch_x*pitch_y + j*pitch_x + i+1],
												txy0[k*pitch_x*pitch_y + j*pitch_x + i-1], txy0[k*pitch_x*pitch_y + j*pitch_x + i+2],
												tyy0[k*pitch_x*pitch_y + j*pitch_x + i], tyy0[k*pitch_x*pitch_y + (j+1)*pitch_x + i],
												tyy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], tyy0[k*pitch_x*pitch_y + (j+2)*pitch_x + i],
												tyz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], tyz0[k*pitch_x*pitch_y + j*pitch_x + i],
												tyz0[(k-2)*pitch_x*pitch_y + j*pitch_x + i], tyz0[(k+1)*pitch_x*pitch_y + j*pitch_x + i] );
								}
							} /* end of if "free surface" */
						} /* end of calculation of vy */
						/* Calculation of vz */
						if ( distance_ymin >= 1 && distance_xmax >= 1 ){
							rhoxz = 0.25*(rho[offset] + rho[(k+1)*pitch_x*pitch_y + j*pitch_x + i]
									+ rho[k*pitch_x*pitch_y + j*pitch_x + i+1] + rho[(k+1)*pitch_x*pitch_y + j*pitch_x + i+1]);
							vpxz = 0.25f*(vp[offset] + vp[(k+1)*pitch_x*pitch_y + j*pitch_x + i] + vp[k*pitch_x*pitch_y + j*pitch_x + i+1] + vp[(k+1)*pitch_x*pitch_y + j*pitch_x + i+1]);
							if ( distance_zmax == 0 ){ /* free surface */
								phixdum = phitxzx[npml];
								phiydum = phityzy[npml];
								phizdum = phitzzz[npml];
								phitxzx[npml] = CPML2 (vpxz, dumpx2_, alphax2_, kappax2_, phixdum, ds, dt,
										- txz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], - txz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i+1] );
								phityzy[npml] = CPML2 (vpxz, dumpy_, alphay_, kappay_, phiydum, ds, dt,
										- tyz0[(k-1)*pitch_x*pitch_y + (j-1)*pitch_x + i], - tyz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i] );
								phitzzz[npml] = CPML2 (vpxz, dumpz2_, alphaz2_, kappaz2_, phizdum, ds, dt,
										tzz0[k*pitch_x*pitch_y + j*pitch_x + i], - tzz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i] );

								if (distance_xmin == 0 || distance_ymax ==0) { // boundary condition
	                        vz0[offset] = 0.0;
	                     } else {
									vz0[offset] += (dt/rhoxz)*( phitxzx[npml] + phityzy[npml] + phitzzz[npml] )
										+ staggardv2 (1./rhoxz, kappax2_, kappay_, kappaz2_, dt, ds,
												- txz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], - txz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i+1],
												- tyz0[(k-1)*pitch_x*pitch_y + (j-1)*pitch_x + i], - tyz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i],
												tzz0[k*pitch_x*pitch_y + j*pitch_x + i], - tzz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i] );
								}
							} else if ( distance_zmax == 1 ){ /* in the first cell, 2nd order finite-difference instead of 4th order finite-difference */
								phixdum = phitxzx[npml];
								phiydum = phityzy[npml];
								phizdum = phitzzz[npml];
								phitxzx[npml] = CPML2 (vpxz, dumpx2_, alphax2_, kappax2_, phixdum, ds, dt,
										txz0[k*pitch_x*pitch_y + j*pitch_x + i], txz0[k*pitch_x*pitch_y + j*pitch_x + i+1] );
								phityzy[npml] = CPML2 (vpxz, dumpy_, alphay_, kappay_, phiydum, ds, dt,
										tyz0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], tyz0[k*pitch_x*pitch_y + j*pitch_x + i] );
								phitzzz[npml] = CPML2 (vpxz, dumpz2_, alphaz2_, kappaz2_, phizdum, ds, dt,
										tzz0[k*pitch_x*pitch_y + j*pitch_x + i], tzz0[(k+1)*pitch_x*pitch_y + j*pitch_x + i] );

								if (distance_xmin == 0 || distance_ymax ==0) { // boundary condition
	                        vz0[offset] = 0.0;
	                     } else {
									vz0[offset] += (dt/rhoxz)*( phitxzx[npml] + phityzy[npml] + phitzzz[npml] )
										+ staggardv2 (1./rhoxz, kappax2_, kappay_, kappaz2_, dt, ds,
												txz0[k*pitch_x*pitch_y + j*pitch_x + i], txz0[k*pitch_x*pitch_y + j*pitch_x + i+1],
												tyz0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], tyz0[k*pitch_x*pitch_y + j*pitch_x + i],
												tzz0[k*pitch_x*pitch_y + j*pitch_x + i], tzz0[(k+1)*pitch_x*pitch_y + j*pitch_x + i] );
								}
							} else { /* regular domain */
								phixdum = phitxzx[npml];
								phiydum = phityzy[npml];
								phizdum = phitzzz[npml];
								phitxzx[npml] = CPML4 (vpxz, dumpx2_, alphax2_, kappax2_, phixdum, ds, dt,
										txz0[k*pitch_x*pitch_y + j*pitch_x + i], txz0[k*pitch_x*pitch_y + j*pitch_x + i+1],
										txz0[k*pitch_x*pitch_y + j*pitch_x + i-1], txz0[k*pitch_x*pitch_y + j*pitch_x + i+2] );
								phityzy[npml] = CPML4 (vpxz, dumpy_, alphay_, kappay_, phiydum, ds, dt,
										tyz0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], tyz0[k*pitch_x*pitch_y + j*pitch_x + i],
										tyz0[k*pitch_x*pitch_y + (j-2)*pitch_x + i], tyz0[k*pitch_x*pitch_y + (j+1)*pitch_x + i] );
								phitzzz[npml] = CPML4 (vpxz, dumpz2_, alphaz2_, kappaz2_, phizdum, ds, dt,
										tzz0[k*pitch_x*pitch_y + j*pitch_x + i], tzz0[(k+1)*pitch_x*pitch_y + j*pitch_x + i],
										tzz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], tzz0[(k+2)*pitch_x*pitch_y + j*pitch_x + i] );

								if (distance_xmin == 0 || distance_ymax ==0) { // boundary condition
	                        vz0[offset] = 0.0;
	                     } else {
									vz0[offset] += (dt/rhoxz)*( phitxzx[npml] + phityzy[npml] + phitzzz[npml] )
										+ staggardv4 (1./rhoxz, kappax2_, kappay_, kappaz2_, dt, ds,
												txz0[k*pitch_x*pitch_y + j*pitch_x + i], txz0[k*pitch_x*pitch_y + j*pitch_x + i+1],
												txz0[k*pitch_x*pitch_y + j*pitch_x + i-1], txz0[k*pitch_x*pitch_y + j*pitch_x + i+2],
												tyz0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], tyz0[k*pitch_x*pitch_y + j*pitch_x + i],
												tyz0[k*pitch_x*pitch_y + (j-2)*pitch_x + i], tyz0[k*pitch_x*pitch_y + (j+1)*pitch_x + i],
												tzz0[k*pitch_x*pitch_y + j*pitch_x + i], tzz0[(k+1)*pitch_x*pitch_y + j*pitch_x + i],
												tzz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], tzz0[(k+2)*pitch_x*pitch_y + j*pitch_x + i] );
								}
							} /* end of if "free surface" */
						} /* end of calculation of vz */
						/* Normal mode */
					} else {
						rhoxy = 0.25*(rho[offset] + rho[k*pitch_x*pitch_y + (j+1)*pitch_x + i]
								+ rho[k*pitch_x*pitch_y + j*pitch_x + i+1] + rho[k*pitch_x*pitch_y + (j+1)*pitch_x + i+1]);
						rhoxz = 0.25*(rho[offset] + rho[(k+1)*pitch_x*pitch_y + j*pitch_x + i]
								+ rho[k*pitch_x*pitch_y + j*pitch_x + i+1] + rho[(k+1)*pitch_x*pitch_y + j*pitch_x + i+1]);
						if ( distance_zmax == 0 ){ /* free surface */
							vx0[offset] += (1./rho[offset])*fx[offset]*dt/ds
								+ staggardv2 (1./rho[offset], 1.0, 1.0, 1.0, dt, ds,
										txx0[k*pitch_x*pitch_y + j*pitch_x + i-1], txx0[k*pitch_x*pitch_y + j*pitch_x + i],
										txy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], txy0[k*pitch_x*pitch_y + j*pitch_x + i],
										txz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], - txz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i] );
							vy0[offset] += (1./rhoxy)*fy[offset]*dt/ds
								+ staggardv2 (1./rhoxy, 1.0, 1.0, 1.0, dt, ds,
										txy0[k*pitch_x*pitch_y + j*pitch_x + i], txy0[k*pitch_x*pitch_y + j*pitch_x + i+1],
										tyy0[k*pitch_x*pitch_y + j*pitch_x + i], tyy0[k*pitch_x*pitch_y + (j+1)*pitch_x + i],
										tyz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], - tyz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i] );
							vz0[offset] += (1./rhoxz)*fz[offset]*dt/ds
								+ staggardv2 (1./rhoxz, 1.0, 1.0, 1.0, dt, ds,
										- txz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], - txz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i+1],
										- tyz0[(k-1)*pitch_x*pitch_y + (j-1)*pitch_x + i], - tyz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i],
										tzz0[k*pitch_x*pitch_y + j*pitch_x + i], - tzz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i] );
						} else if ( distance_zmax == 1 ){ /* in the first cell, 2nd order finite-difference instead of 4th order finite-difference */
							vx0[offset] += (1./rho[offset])*fx[offset]*dt/ds
								+ staggardv2 (1./rho[offset], 1.0, 1.0, 1.0, dt, ds,
										txx0[k*pitch_x*pitch_y + j*pitch_x + i-1], txx0[k*pitch_x*pitch_y + j*pitch_x + i],
										txy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], txy0[k*pitch_x*pitch_y + j*pitch_x + i],
										txz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], txz0[k*pitch_x*pitch_y + j*pitch_x + i] );
							vy0[offset] += (1./rhoxy)*fy[offset]*dt/ds
								+ staggardv2 (1./rhoxy, 1.0, 1.0, 1.0, dt, ds,
										txy0[k*pitch_x*pitch_y + j*pitch_x + i], txy0[k*pitch_x*pitch_y + j*pitch_x + i+1],
										tyy0[k*pitch_x*pitch_y + j*pitch_x + i], tyy0[k*pitch_x*pitch_y + (j+1)*pitch_x + i],
										tyz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], tyz0[k*pitch_x*pitch_y + j*pitch_x + i] );
							vz0[offset] += (1./rhoxz)*fz[offset]*dt/ds
								+ staggardv2 (1./rhoxz, 1.0, 1.0, 1.0, dt, ds,
										txz0[k*pitch_x*pitch_y + j*pitch_x + i], txz0[k*pitch_x*pitch_y + j*pitch_x + i+1],
										tyz0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], tyz0[k*pitch_x*pitch_y + j*pitch_x + i],
										tzz0[k*pitch_x*pitch_y + j*pitch_x + i], tzz0[(k+1)*pitch_x*pitch_y + j*pitch_x + i] );
						} else { /* regular domain */
							vx0[offset] += (1./rho[offset])*fx[offset]*dt/ds
								+ staggardv4 (1./rho[offset], 1.0, 1.0, 1.0, dt, ds,
										txx0[k*pitch_x*pitch_y + j*pitch_x + i-1], txx0[k*pitch_x*pitch_y + j*pitch_x + i],
										txx0[k*pitch_x*pitch_y + j*pitch_x + i-2], txx0[k*pitch_x*pitch_y + j*pitch_x + i+1],
										txy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], txy0[k*pitch_x*pitch_y + j*pitch_x + i],
										txy0[k*pitch_x*pitch_y + (j-2)*pitch_x + i], txy0[k*pitch_x*pitch_y + (j+1)*pitch_x + i],
										txz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], txz0[k*pitch_x*pitch_y + j*pitch_x + i],
										txz0[(k-2)*pitch_x*pitch_y + j*pitch_x + i], txz0[(k+1)*pitch_x*pitch_y + j*pitch_x + i] );
							vy0[offset] += (1./rhoxy)*fy[offset]*dt/ds
								+ staggardv4 (1./rhoxy, 1.0, 1.0, 1.0, dt, ds,
										txy0[k*pitch_x*pitch_y + j*pitch_x + i], txy0[k*pitch_x*pitch_y + j*pitch_x + i+1],
										txy0[k*pitch_x*pitch_y + j*pitch_x + i-1], txy0[k*pitch_x*pitch_y + j*pitch_x + i+2],
										tyy0[k*pitch_x*pitch_y + j*pitch_x + i], tyy0[k*pitch_x*pitch_y + (j+1)*pitch_x + i],
										tyy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], tyy0[k*pitch_x*pitch_y + (j+2)*pitch_x + i],
										tyz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], tyz0[k*pitch_x*pitch_y + j*pitch_x + i],
										tyz0[(k-2)*pitch_x*pitch_y + j*pitch_x + i], tyz0[(k+1)*pitch_x*pitch_y + j*pitch_x + i] );
							vz0[offset] += (1./rhoxz)*fz[offset]*dt/ds
								+ staggardv4 (1./rhoxz, 1.0, 1.0, 1.0, dt, ds,
										txz0[k*pitch_x*pitch_y + j*pitch_x + i], txz0[k*pitch_x*pitch_y + j*pitch_x + i+1],
										txz0[k*pitch_x*pitch_y + j*pitch_x + i-1], txz0[k*pitch_x*pitch_y + j*pitch_x + i+2],
										tyz0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], tyz0[k*pitch_x*pitch_y + j*pitch_x + i],
										tyz0[k*pitch_x*pitch_y + (j-2)*pitch_x + i], tyz0[k*pitch_x*pitch_y + (j+1)*pitch_x + i],
										tzz0[k*pitch_x*pitch_y + j*pitch_x + i], tzz0[(k+1)*pitch_x*pitch_y + j*pitch_x + i],
										tzz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], tzz0[(k+2)*pitch_x*pitch_y + j*pitch_x + i] );
						} /* end of if "free surface" */
					} /* endif CPML */
				} /* for i */
			} /* for j */
		} /* for k */
	}

	void cpu_compute_stress (	float* txx0, float* tyy0, float* tzz0, float* txy0, float* txz0, float* tyz0,
										float* vx0, float* vy0, float* vz0,
										int* npml_tab, float* phivxx, float* phivxy, float* phivxz, float* phivyx, float* phivyy, float* phivyz, float* phivzx, float* phivzy, float* phivzz, 
										float* mu, float* lam, float* vp, 
										int sizex, int sizey, int sizez,
										int pitch_x, int pitch_y, int pitch_z, 
										float ds, float dt, int delta, int position,
										int ixe_min, int ixs_max, int iye_min, int iys_max, float dump0, float kappa0, float alpha0, int iter)
	{
		float mux, lamx, vpx, muy, vpy, muz, lamz, vpz, muxyz, lamxyz, vpxyz, b1, b2, phixdum, phiydum, phizdum;
		float abscissa_normalized;

		for ( int k = 0; k < sizez; k++){
			abscissa_normalized = (DELTA - k)/(float)DELTA;
		   int distance_zmin = k;
         int distance_zmax = sizez - 1 - k;

			float dumpz_  = (abscissa_normalized>=0)?dump0*powf(abscissa_normalized,NPOWER):0.0;
			float kappaz_ = (abscissa_normalized>=0)?1.0 + (kappa0 - 1.0) * powf(abscissa_normalized,NPOWER):1.0;
			float alphaz_ = (abscissa_normalized>=0)?alpha0 * (1.0 - abscissa_normalized):0.0;
			float dumpz2_ = (abscissa_normalized>=0)?dump0*powf(abscissa_normalized,NPOWER):0.0;
			float kappaz2_= (abscissa_normalized>=0)?1.0 + (kappa0 - 1.0) * powf(abscissa_normalized,NPOWER):1.0;
			float alphaz2_= (abscissa_normalized>=0)?alpha0 * (1.0 - abscissa_normalized):0.0;


			for ( int j = 0; j < sizey; j++ ){
				abscissa_normalized = MAX((iye_min - j)/(float)DELTA, (j - (iys_max-1)/(float)DELTA));
            int distance_ymin = (position & MASK_FIRST_Y)?j:DUMMY_VALUE;
            int distance_ymax = (position & MASK_LAST_Y)?(sizey - j - 1):DUMMY_VALUE;
	
				float dumpy_  = (abscissa_normalized>=0)?dump0*powf(abscissa_normalized,NPOWER):0.0;
				float kappay_ = (abscissa_normalized>=0)?1.0 + (kappa0 - 1.0) * powf(abscissa_normalized,NPOWER):1.0;
				float alphay_ = (abscissa_normalized>=0)?alpha0 * (1.0 - abscissa_normalized):0.0;
				float dumpy2_ = (abscissa_normalized>=0)?dump0*powf(abscissa_normalized,NPOWER):0.0;
				float kappay2_= (abscissa_normalized>=0)?1.0 + (kappa0 - 1.0) * powf(abscissa_normalized,NPOWER):1.0;
				float alphay2_= (abscissa_normalized>=0)?alpha0 * (1.0 - abscissa_normalized):0.0;

				for ( int i = 0; i < sizex; i++ ){
					int offset = k*pitch_x*pitch_y + j*pitch_x + i;
					int npml = npml_tab[k*(sizex)*(sizey) + j*(sizex) + i];

				   int distance_xmin = (position & MASK_FIRST_X)?i:DUMMY_VALUE;
	            int distance_xmax = (position & MASK_LAST_X)?(sizex - i - 1):DUMMY_VALUE;

					if (npml >= 0) {/* CPML */
// if (!k) DGN_DBG
	            	abscissa_normalized = MAX((ixe_min - i)/(float)DELTA, (i - (ixs_max-1)/(float)DELTA));

						float dumpx_  = (abscissa_normalized>=0)?dump0*powf(abscissa_normalized,NPOWER):0.0;
						float kappax_ = (abscissa_normalized>=0)?1.0 + (kappa0 - 1.0) * powf(abscissa_normalized,NPOWER):1.0;
						float alphax_ = (abscissa_normalized>=0)?alpha0 * (1.0 - abscissa_normalized):0.0;
						float dumpx2_ = (abscissa_normalized>=0)?dump0*powf(abscissa_normalized,NPOWER):0.0;
						float kappax2_= (abscissa_normalized>=0)?1.0 + (kappa0 - 1.0) * powf(abscissa_normalized,NPOWER):1.0;
						float alphax2_= (abscissa_normalized>=0)?alpha0 * (1.0 - abscissa_normalized):0.0;

						/* Calculation of txx, tyy and tzz */
						if (distance_ymin >= 1 && distance_xmax >= 1 ){
							mux = 2./(1./mu[offset] + 1./mu[k*pitch_x*pitch_y + j*pitch_x + i+1]);
							lamx = 2./(1./lam[offset] + 1./lam[k*pitch_x*pitch_y + j*pitch_x + i+1]);
							vpx = 2./(1./vp[offset] + 1./vp[k*pitch_x*pitch_y + j*pitch_x + i+1]);

							if ( distance_zmax == 0 ){ /* free surface */
								b1 = 4. * mux * (lamx + mux) / (lamx + 2.*mux);
								b2 = 2. * mux * lamx / (lamx + 2.*mux);
								phixdum = phivxx[npml];
								phiydum = phivyy[npml];
								phivxx[npml] = CPML2 (vpx, dumpx2_, alphax2_, kappax2_, phixdum, ds, dt,
										vx0[offset], vx0[k*pitch_x*pitch_y + j*pitch_x + i+1] );
								phivyy[npml] = CPML2 (vpx, dumpy_, alphay_, kappay_, phiydum, ds, dt,
										vy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], vy0[offset] );
								txx0[offset] += b1*dt*phivxx[npml] + b2*dt*phivyy[npml]
									+ b1*dt*(vx0[k*pitch_x*pitch_y + j*pitch_x + i+1]-vx0[offset])/(kappax2_*ds)
									+ b2*dt*(vy0[offset] - vy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i])/(kappay_*ds);
								tyy0[offset] += b1*dt*phivyy[npml] + b2*dt*phivxx[npml]
									+ b1*dt*(vy0[offset]-vy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i])/(kappay_*ds)
									+ b2*dt*(vx0[k*pitch_x*pitch_y + j*pitch_x + i+1]-vx0[offset])/(kappax2_*ds);
								tzz0[offset] = 0;
							} else if  ( distance_zmax == 1 ){ /* in the first cell, 2nd order finite-difference instead of 4th order finite-difference */
								phixdum = phivxx[npml];
								phiydum = phivyy[npml];
								phizdum = phivzz[npml];
								phivxx[npml] = CPML2 (vpx, dumpx2_, alphax2_, kappax2_, phixdum, ds, dt,
										vx0[offset], vx0[k*pitch_x*pitch_y + j*pitch_x + i+1] );
								phivyy[npml] = CPML2 (vpx, dumpy_, alphay_, kappay_, phiydum, ds, dt,
										vy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], vy0[offset] );
								phivzz[npml] = CPML2 (vpx, dumpz_, alphaz_, kappaz_, phizdum, ds, dt,
										vz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], vz0[offset] );
								txx0[offset] += dt*(lamx + 2.*mux)*phivxx[npml] + dt*lamx*( phivyy[npml] + phivzz[npml] )
									+ staggards2 (lamx, mux, kappax2_, kappay_, kappaz_, dt, ds,
											vx0[offset], vx0[k*pitch_x*pitch_y + j*pitch_x + i+1],
											vy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], vy0[offset],
											vz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], vz0[offset] );
								tyy0[offset] += dt*lamx*( phivxx[npml] + phivzz[npml] ) + dt*(lamx + 2.*mux)*phivyy[npml]
									+ staggards2 (lamx, mux, kappay_, kappax2_, kappaz_, dt, ds,
											vy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], vy0[offset],
											vx0[offset], vx0[k*pitch_x*pitch_y + j*pitch_x + i+1],
											vz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], vz0[offset] );
								tzz0[offset] += dt*lamx*( phivxx[npml] + phivyy[npml] ) + dt*(lamx + 2.*mux)*phivzz[npml]
									+ staggards2 (lamx, mux, kappaz_, kappax2_, kappay_, dt, ds,
											vz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], vz0[offset],
											vx0[offset], vx0[k*pitch_x*pitch_y + j*pitch_x + i+1],
											vy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], vy0[offset] );
							} else { /* regular domain */
								phixdum = phivxx[npml];
								phiydum = phivyy[npml];
								phizdum = phivzz[npml];
								phivxx[npml] = CPML4 (vpx, dumpx2_, alphax2_, kappax2_, phixdum, ds, dt,
										vx0[offset], vx0[k*pitch_x*pitch_y + j*pitch_x + i+1],
										vx0[k*pitch_x*pitch_y + j*pitch_x + i-1], vx0[k*pitch_x*pitch_y + j*pitch_x + i+2] );
								phivyy[npml] = CPML4 (vpx, dumpy_, alphay_, kappay_, phiydum, ds, dt,
										vy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], vy0[offset],
										vy0[k*pitch_x*pitch_y + (j-2)*pitch_x + i], vy0[k*pitch_x*pitch_y + (j+1)*pitch_x + i] );
								phivzz[npml] = CPML4 (vpx, dumpz_, alphaz_, kappaz_, phizdum, ds, dt,
										vz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], vz0[offset],
										vz0[(k-2)*pitch_x*pitch_y + j*pitch_x + i], vz0[(k+1)*pitch_x*pitch_y + j*pitch_x + i] );
								txx0[offset] += dt*(lamx + 2.*mux)*phivxx[npml] + dt*lamx*( phivyy[npml] + phivzz[npml] )
									+ staggards4 (lamx, mux, kappax2_, kappay_, kappaz_, dt, ds,
											vx0[offset], vx0[k*pitch_x*pitch_y + j*pitch_x + i+1],
											vx0[k*pitch_x*pitch_y + j*pitch_x + i-1], vx0[k*pitch_x*pitch_y + j*pitch_x + i+2],
											vy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], vy0[offset],
											vy0[k*pitch_x*pitch_y + (j-2)*pitch_x + i], vy0[k*pitch_x*pitch_y + (j+1)*pitch_x + i],
											vz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], vz0[offset],
											vz0[(k-2)*pitch_x*pitch_y + j*pitch_x + i], vz0[(k+1)*pitch_x*pitch_y + j*pitch_x + i] );
								tyy0[offset] += dt*lamx*( phivxx[npml] + phivzz[npml] ) + dt*(lamx + 2.*mux)*phivyy[npml]
									+ staggards4 (lamx, mux, kappay_, kappax2_, kappaz_, dt, ds,
											vy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], vy0[offset],
											vy0[k*pitch_x*pitch_y + (j-2)*pitch_x + i], vy0[k*pitch_x*pitch_y + (j+1)*pitch_x + i],
											vx0[offset], vx0[k*pitch_x*pitch_y + j*pitch_x + i+1],
											vx0[k*pitch_x*pitch_y + j*pitch_x + i-1], vx0[k*pitch_x*pitch_y + j*pitch_x + i+2],
											vz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], vz0[offset],
											vz0[(k-2)*pitch_x*pitch_y + j*pitch_x + i], vz0[(k+1)*pitch_x*pitch_y + j*pitch_x + i] );
								tzz0[offset] += dt*lamx*( phivxx[npml] + phivyy[npml] ) + dt*(lamx + 2.*mux)*phivzz[npml]
									+ staggards4 (lamx, mux, kappaz_, kappax2_, kappay_, dt, ds,
											vz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], vz0[offset],
											vz0[(k-2)*pitch_x*pitch_y + j*pitch_x + i], vz0[(k+1)*pitch_x*pitch_y + j*pitch_x + i],
											vx0[offset], vx0[k*pitch_x*pitch_y + j*pitch_x + i+1],
											vx0[k*pitch_x*pitch_y + j*pitch_x + i-1], vx0[k*pitch_x*pitch_y + j*pitch_x + i+2],
											vy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], vy0[offset],
											vy0[k*pitch_x*pitch_y + (j-2)*pitch_x + i], vy0[k*pitch_x*pitch_y + (j+1)*pitch_x + i] );
							} /* end of if "free surface" */
						} /* end of calculation of txx, tyy and tzz */
						/* Calculation of txy */
						if ( distance_ymax >= 1 && distance_xmin >= 1 ){
							muy = 2./(1./mu[offset] + 1./mu[k*pitch_x*pitch_y + (j+1)*pitch_x + i]);
							vpy = 2./(1./vp[offset] + 1./vp[k*pitch_x*pitch_y + (j+1)*pitch_x + i]);
							if ( distance_zmax <= 1 ){ /* in the first cell, 2nd order finite-difference instead of 4th order finite-difference */
								phixdum = phivyx[npml];
								phiydum = phivxy[npml];
								phivyx[npml] = CPML2 (vpy, dumpx_, alphax_, kappax_, phixdum, ds, dt,
										vy0[k*pitch_x*pitch_y + j*pitch_x + i-1], vy0[offset] );
								phivxy[npml] = CPML2 (vpy, dumpy2_, alphay2_, kappay2_, phiydum, ds, dt,
										vx0[offset], vx0[k*pitch_x*pitch_y + (j+1)*pitch_x + i] );
								txy0[offset] += dt*muy*( phivyx[npml] + phivxy[npml] )
									+ staggardt2 (muy, kappax_, kappay2_, dt, ds,
											vy0[k*pitch_x*pitch_y + j*pitch_x + i-1], vy0[offset],
											vx0[offset], vx0[k*pitch_x*pitch_y + (j+1)*pitch_x + i] );
							} else { /* regular domain */
								phixdum = phivyx[npml];
								phiydum = phivxy[npml];
								phivyx[npml] = CPML4 (vpy, dumpx_, alphax_, kappax_, phixdum, ds, dt,
										vy0[k*pitch_x*pitch_y + j*pitch_x + i-1], vy0[offset],
										vy0[k*pitch_x*pitch_y + j*pitch_x + i-2], vy0[k*pitch_x*pitch_y + j*pitch_x + i+1] );
								phivxy[npml] = CPML4 (vpy, dumpy2_, alphay2_, kappay2_, phiydum, ds, dt,
										vx0[offset], vx0[k*pitch_x*pitch_y + (j+1)*pitch_x + i],
										vx0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], vx0[k*pitch_x*pitch_y + (j+2)*pitch_x + i] );
								txy0[offset] += dt*muy*( phivyx[npml] + phivxy[npml] )
									+ staggardt4 (muy, kappax_, kappay2_, dt, ds,
											vy0[k*pitch_x*pitch_y + j*pitch_x + i-1], vy0[offset],
											vy0[k*pitch_x*pitch_y + j*pitch_x + i-2], vy0[k*pitch_x*pitch_y + j*pitch_x + i+1],
											vx0[offset], vx0[k*pitch_x*pitch_y + (j+1)*pitch_x + i],
											vx0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], vx0[k*pitch_x*pitch_y + (j+2)*pitch_x + i] );
							} /* end of if "free surface" */
						} /* end of calculation of txy */
						/* Calculation of txz */
						if ( distance_zmax >= 1 && distance_xmin >= 1 ){
							muz = 2./(1./mu[offset] + 1./mu[(k+1)*pitch_x*pitch_y + j*pitch_x + i]);
							vpz = 2./(1./vp[offset] + 1./vp[(k+1)*pitch_x*pitch_y + j*pitch_x + i]);

							if ( k == 0 ){ /* in the first cell, 2nd order finite-difference instead of 4th order finite-difference */
								phixdum = phivzx[npml];
								phizdum = phivxz[npml];
								phivzx[npml] = CPML2 (vpz, dumpx_, alphax_, kappax_, phixdum, ds, dt,
										vz0[k*pitch_x*pitch_y + j*pitch_x + i-1], vz0[offset] );
								phivxz[npml] = CPML2 (vpz, dumpz2_, alphaz2_, kappaz2_, phizdum, ds, dt,
										vx0[offset], vx0[(k+1)*pitch_x*pitch_y + j*pitch_x + i] );
								txz0[offset] += dt*muz*( phivzx[npml] + phivxz[npml] )
									+ staggardt2 (muz, kappax_, kappaz2_, dt, ds,
											vz0[k*pitch_x*pitch_y + j*pitch_x + i-1], vz0[offset],
											vx0[offset], vx0[(k+1)*pitch_x*pitch_y + j*pitch_x + i] );
							} else { /* regular domain */
								phixdum = phivzx[npml];
								phizdum = phivxz[npml];
								phivzx[npml] = CPML4 (vpz, dumpx_, alphax_, kappax_, phixdum, ds, dt,
										vz0[k*pitch_x*pitch_y + j*pitch_x + i-1], vz0[offset],
										vz0[k*pitch_x*pitch_y + j*pitch_x + i-2], vz0[k*pitch_x*pitch_y + j*pitch_x + i+1] );
								phivxz[npml] = CPML4 (vpz, dumpz2_, alphaz2_, kappaz2_, phizdum, ds, dt,
										vx0[offset], vx0[(k+1)*pitch_x*pitch_y + j*pitch_x + i],
										vx0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], vx0[(k+2)*pitch_x*pitch_y + j*pitch_x + i] );
								txz0[offset] += dt*muz*( phivzx[npml] + phivxz[npml] )
									+ staggardt4 (muz, kappax_, kappaz2_, dt, ds,
											vz0[k*pitch_x*pitch_y + j*pitch_x + i-1], vz0[offset],
											vz0[k*pitch_x*pitch_y + j*pitch_x + i-2], vz0[k*pitch_x*pitch_y + j*pitch_x + i+1],
											vx0[offset], vx0[(k+1)*pitch_x*pitch_y + j*pitch_x + i],
											vx0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], vx0[(k+2)*pitch_x*pitch_y + j*pitch_x + i] );
							} /* end of if "free surface" */
						} /* end of calculation of txz */
						/* Calculation of tyz */
						if ( distance_zmax >= 1 && distance_ymax >= 1 ){
							muxyz = 8./(1./mu[offset] + 1./mu[(k+1)*pitch_x*pitch_y + j*pitch_x + i]
									+ 1./mu[k*pitch_x*pitch_y + (j+1)*pitch_x + i] + 1./mu[(k+1)*pitch_x*pitch_y + (j+1)*pitch_x + i]
									+ 1./mu[k*pitch_x*pitch_y + j*pitch_x + i+1] + 1./mu[(k+1)*pitch_x*pitch_y + j*pitch_x + i+1]
									+ 1./mu[k*pitch_x*pitch_y + (j+1)*pitch_x + i+1] + 1./mu[(k+1)*pitch_x*pitch_y + (j+1)*pitch_x + i+1]);
							vpxyz = 8./(1./vp[offset] + 1./vp[(k+1)*pitch_x*pitch_y + j*pitch_x + i]
									+ 1./vp[k*pitch_x*pitch_y + (j+1)*pitch_x + i] + 1./vp[(k+1)*pitch_x*pitch_y + (j+1)*pitch_x + i]
									+ 1./vp[k*pitch_x*pitch_y + j*pitch_x + i+1] + 1./vp[(k+1)*pitch_x*pitch_y + j*pitch_x + i+1]
									+ 1./vp[k*pitch_x*pitch_y + (j+1)*pitch_x + i+1] + 1./vp[(k+1)*pitch_x*pitch_y + (j+1)*pitch_x + i+1]);
							if ( distance_zmax == 1 ){ /* in the first cell, 2nd order finite-difference instead of 4th order finite-difference */
								phiydum = phivzy[npml];
								phizdum = phivyz[npml];
								phivzy[npml] = CPML2 (vpxyz, dumpy2_, alphay2_, kappay2_, phiydum, ds, dt,
										vz0[offset], vz0[k*pitch_x*pitch_y + (j+1)*pitch_x + i] );
								phivyz[npml] = CPML2 (vpxyz, dumpz2_, alphaz2_, kappaz2_, phizdum, ds, dt,
										vy0[offset], vy0[(k+1)*pitch_x*pitch_y + j*pitch_x + i] );
								tyz0[offset] += dt*muxyz*( phivzy[npml] + phivyz[npml] )
									+ staggardt2 (muxyz, kappay2_, kappaz2_, dt, ds,
											vz0[offset], vz0[k*pitch_x*pitch_y + (j+1)*pitch_x + i],
											vy0[offset], vy0[(k+1)*pitch_x*pitch_y + j*pitch_x + i] );
							} else { /* regular domain */
								phiydum = phivzy[npml];
								phizdum = phivyz[npml];
								phivzy[npml] = CPML4 (vpxyz, dumpy2_, alphay2_, kappay2_, phiydum, ds, dt,
										vz0[offset], vz0[k*pitch_x*pitch_y + (j+1)*pitch_x + i],
										vz0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], vz0[k*pitch_x*pitch_y + (j+2)*pitch_x + i] );
								phivyz[npml] = CPML4 (vpxyz, dumpz2_, alphaz2_, kappaz2_, phizdum, ds, dt,
										vy0[offset], vy0[(k+1)*pitch_x*pitch_y + j*pitch_x + i],
										vy0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], vy0[(k+2)*pitch_x*pitch_y + j*pitch_x + i] );
								tyz0[offset] += dt*muxyz*( phivzy[npml] + phivyz[npml] )
									+ staggardt4 (muxyz, kappay2_, kappaz2_, dt, ds,
											vz0[offset], vz0[k*pitch_x*pitch_y + (j+1)*pitch_x + i],
											vz0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], vz0[k*pitch_x*pitch_y + (j+2)*pitch_x + i],
											vy0[offset], vy0[(k+1)*pitch_x*pitch_y + j*pitch_x + i],
											vy0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], vy0[(k+2)*pitch_x*pitch_y + j*pitch_x + i] );
							} /* end of if "free surface" */
						} /* end of calculation of tyz */
						/* txz and tyz are antisymetric */
						if ( distance_zmax == 0 ){
							txz0[pitch_x*pitch_y + j*pitch_x + i] = - txz0[j*pitch_x + i];
							tyz0[pitch_x*pitch_y + j*pitch_x + i] = - txz0[j*pitch_x + i];
						}
						/* Normal mode */
					} else {
// if (!k) DGN_DBG
						mux = 2./(1./mu[offset] + 1./mu[k*pitch_x*pitch_y + j*pitch_x + i+1]);
						lamx = 2./(1./lam[offset] + 1./lam[k*pitch_x*pitch_y + j*pitch_x + i+1]);
						muy = 2./(1./mu[offset] + 1./mu[k*pitch_x*pitch_y + (j+1)*pitch_x + i]);
						muz = 2./(1./mu[offset] + 1./mu[(k+1)*pitch_x*pitch_y + j*pitch_x + i]);
						muxyz = 8./(1./mu[offset] + 1./mu[(k+1)*pitch_x*pitch_y + j*pitch_x + i]
								+ 1./mu[k*pitch_x*pitch_y + (j+1)*pitch_x + i] + 1./mu[(k+1)*pitch_x*pitch_y + (j+1)*pitch_x + i]
								+ 1./mu[k*pitch_x*pitch_y + j*pitch_x + i+1] + 1./mu[(k+1)*pitch_x*pitch_y + j*pitch_x + i+1]
								+ 1./mu[k*pitch_x*pitch_y + (j+1)*pitch_x + i+1] + 1./mu[(k+1)*pitch_x*pitch_y + (j+1)*pitch_x + i+1]);
						if ( distance_zmax == 0 ){ /* free surface */
							b1 = 4. * mux * (lamx + mux) / (lamx + 2.*mux);
							b2 = 2. * mux * lamx / (lamx + 2.*mux);
							txx0[offset] += b1*dt*(vx0[k*pitch_x*pitch_y + j*pitch_x + i+1]-vx0[offset])/ds
								+ b2*dt*(vy0[offset] - vy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i])/ds;
							tyy0[offset] += b1*dt*(vy0[offset]-vy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i])/ds
								+ b2*dt*(vx0[k*pitch_x*pitch_y + j*pitch_x + i+1]-vx0[offset])/ds;
							tzz0[offset] = 0;
							txy0[offset] += staggardt2 (muy, un, un, dt, ds,
									vy0[k*pitch_x*pitch_y + j*pitch_x + i-1], vy0[offset],
									vx0[offset], vx0[k*pitch_x*pitch_y + (j+1)*pitch_x + i] );
						} else if ( distance_zmax == 1 ){ /* in the first cell, 2nd order finite-difference instead of 4th order finite-difference */
							txx0[offset] += staggards2 (lamx, mux, un, un, un, dt, ds,
									vx0[offset], vx0[k*pitch_x*pitch_y + j*pitch_x + i+1],
									vy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], vy0[offset],
									vz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], vz0[offset] );
							tyy0[offset] += staggards2 (lamx, mux, un, un, un, dt, ds,
									vy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], vy0[offset],
									vx0[offset], vx0[k*pitch_x*pitch_y + j*pitch_x + i+1],
									vz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], vz0[offset] );
							tzz0[offset] += staggards2 (lamx, mux, un, un, un, dt, ds,
									vz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], vz0[offset],
									vx0[offset], vx0[k*pitch_x*pitch_y + j*pitch_x + i+1],
									vy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], vy0[offset] );
							txy0[offset] += staggardt2 (muy, un, un, dt, ds,
									vy0[k*pitch_x*pitch_y + j*pitch_x + i-1], vy0[offset],
									vx0[offset], vx0[k*pitch_x*pitch_y + (j+1)*pitch_x + i] );
							txz0[offset] += staggardt2 (muz, un, un, dt, ds,
									vz0[k*pitch_x*pitch_y + j*pitch_x + i-1], vz0[offset],
									vx0[offset], vx0[(k+1)*pitch_x*pitch_y + j*pitch_x + i] );
							tyz0[offset] += staggardt2 (muxyz, un, un, dt, ds,
									vz0[offset], vz0[k*pitch_x*pitch_y + (j+1)*pitch_x + i],
									vy0[offset], vy0[(k+1)*pitch_x*pitch_y + j*pitch_x + i] );
						} else { /* regular domain */
							txx0[offset] += staggards4 (lamx, mux, un, un, un, dt, ds,
									vx0[offset], vx0[k*pitch_x*pitch_y + j*pitch_x + i+1],
									vx0[k*pitch_x*pitch_y + j*pitch_x + i-1], vx0[k*pitch_x*pitch_y + j*pitch_x + i+2],
									vy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], vy0[offset],
									vy0[k*pitch_x*pitch_y + (j-2)*pitch_x + i], vy0[k*pitch_x*pitch_y + (j+1)*pitch_x + i],
									vz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], vz0[offset],
									vz0[(k-2)*pitch_x*pitch_y + j*pitch_x + i], vz0[(k+1)*pitch_x*pitch_y + j*pitch_x + i] );
							tyy0[offset] += staggards4 (lamx, mux, un, un, un, dt, ds,
									vy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], vy0[offset],
									vy0[k*pitch_x*pitch_y + (j-2)*pitch_x + i], vy0[k*pitch_x*pitch_y + (j+1)*pitch_x + i],
									vx0[offset], vx0[k*pitch_x*pitch_y + j*pitch_x + i+1],
									vx0[k*pitch_x*pitch_y + j*pitch_x + i-1], vx0[k*pitch_x*pitch_y + j*pitch_x + i+2],
									vz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], vz0[offset],
									vz0[(k-2)*pitch_x*pitch_y + j*pitch_x + i], vz0[(k+1)*pitch_x*pitch_y + j*pitch_x + i] );
							tzz0[offset] += staggards4 (lamx, mux, un, un, un, dt, ds,
									vz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], vz0[offset],
									vz0[(k-2)*pitch_x*pitch_y + j*pitch_x + i], vz0[(k+1)*pitch_x*pitch_y + j*pitch_x + i],
									vx0[offset], vx0[k*pitch_x*pitch_y + j*pitch_x + i+1],
									vx0[k*pitch_x*pitch_y + j*pitch_x + i-1], vx0[k*pitch_x*pitch_y + j*pitch_x + i+2],
									vy0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], vy0[offset],
									vy0[k*pitch_x*pitch_y + (j-2)*pitch_x + i], vy0[k*pitch_x*pitch_y + (j+1)*pitch_x + i]);
							txy0[offset] += staggardt4 (muy, un, un, dt, ds,
									vy0[k*pitch_x*pitch_y + j*pitch_x + i-1], vy0[offset],
									vy0[k*pitch_x*pitch_y + j*pitch_x + i-2], vy0[k*pitch_x*pitch_y + j*pitch_x + i+1],
									vx0[offset], vx0[k*pitch_x*pitch_y + (j+1)*pitch_x + i],
									vx0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], vx0[k*pitch_x*pitch_y + (j+2)*pitch_x + i] );
							txz0[offset] += staggardt4 (muz, un, un, dt, ds,
									vz0[k*pitch_x*pitch_y + j*pitch_x + i-1], vz0[offset],
									vz0[k*pitch_x*pitch_y + j*pitch_x + i-2], vz0[k*pitch_x*pitch_y + j*pitch_x + i+1],
									vx0[offset], vx0[(k+1)*pitch_x*pitch_y + j*pitch_x + i],
									vx0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], vx0[(k+2)*pitch_x*pitch_y + j*pitch_x + i] );
							tyz0[offset] += staggardt4 (muxyz, un, un, dt, ds,
									vz0[offset], vz0[k*pitch_x*pitch_y + (j+1)*pitch_x + i],
									vz0[k*pitch_x*pitch_y + (j-1)*pitch_x + i], vz0[k*pitch_x*pitch_y + (j+2)*pitch_x + i],
									vy0[offset], vy0[(k+1)*pitch_x*pitch_y + j*pitch_x + i],
									vy0[(k-1)*pitch_x*pitch_y + j*pitch_x + i], vy0[(k+2)*pitch_x*pitch_y + j*pitch_x + i] );
						} /* end of if "free surface" */
						/* txz and tyz are antisymetric */
						if ( distance_zmax == 0 ){
							txz0[pitch_x*pitch_y + j*pitch_x + i] = - txz0[j*pitch_x + i];
							tyz0[pitch_x*pitch_y + j*pitch_x + i] = - txz0[j*pitch_x + i];
						}
					} /* endif CPML */
				} /* for i */
			} /* for j */
		} /* for k */
	}
// }}}

// CPML & STAGGERED GRID FUNCTIONS {{{
	float staggardv4 (float b, float kappax, float kappay, float kappaz, float dt, float dx,
		float x1, float x2, float x3, float x4,
		float y1, float y2, float y3, float y4,
		float z1, float z2, float z3, float z4 )
	{
	  return (9.*b*dt/8.)*( (x2 - x1)/kappax + (y2 - y1)/kappay + (z2 - z1)/kappaz )/dx
	         - (b*dt/24.)*( (x4 - x3)/kappax + (y4 - y3)/kappay + (z4 - z3)/kappaz )/dx;
	}

	float staggardv2 (float b, float kappax, float kappay, float kappaz, float dt, float dx,
		float x1, float x2,
		float y1, float y2,
		float z1, float z2 )
	{
	  return b*dt*( (x2 - x1)/kappax + (y2 - y1)/kappay + (z2 - z1)/kappaz )/dx;
	}

	float staggards4 (float lam, float mu, float kappax, float kappay, float kappaz, float dt, float dx,
		float x1, float x2, float x3, float x4,
		float y1, float y2, float y3, float y4,
		float z1, float z2, float z3, float z4 )
	{
	  return (9.*dt/8.)*( (lam+2.*mu)*(x2 - x1)/kappax + lam*(y2 - y1)/kappay + lam*(z2 - z1)/kappaz )/dx
	         - (dt/24.)*( (lam+2.*mu)*(x4 - x3)/kappax + lam*(y4 - y3)/kappay + lam*(z4 - z3)/kappaz )/dx;
	}

	float staggards2 (float lam, float mu, float kappax, float kappay, float kappaz, float dt, float dx,
		float x1, float x2,
		float y1, float y2,
		float z1, float z2 )
	{
	  return dt*( (lam+2.*mu)*(x2 - x1)/kappax + lam*(y2 - y1)/kappay + lam*(z2 - z1)/kappaz )/dx;
	}

	float staggardt4 (float mu, float kappax, float kappay, float dt, float dx,
		float x1, float x2, float x3, float x4,
		float y1, float y2, float y3, float y4 )
	{
	  return (9.*dt*mu/8.)*( (x2 - x1)/kappax + (y2 - y1)/kappay )/dx
	         - (dt*mu/24.)*( (x4 - x3)/kappax + (y4 - y3)/kappay )/dx;
	}

	float staggardt2 (float mu, float kappax, float kappay, float dt, float dx,
		float x1, float x2,
		float y1, float y2 )
	{
	  return dt*mu*( (x2 - x1)/kappax + (y2 - y1)/kappay )/dx;
	}

	float CPML4 (float vp, float dump, float alpha, float kappa, float phidum, float dx, float dt,
	    float x1, float x2, float x3, float x4 )
	{
	  float a, b;

	 b = exp ( - ( vp*dump / kappa + alpha ) * dt );
	  a = 0.0;
	  if ( abs ( vp*dump ) > 0.000001 ) a = vp*dump * ( b - 1.0 ) / ( kappa * ( vp*dump + kappa * alpha ) );

	  return b * phidum + a * ( (9./8.)*( x2 - x1 )/dx - (1./24.)*( x4 - x3 )/dx );
	}

	float CPML2 (float vp, float dump, float alpha, float kappa, float phidum, float dx, float dt,
	    float x1, float x2 )
	{
	  float a, b;

	  b = exp ( - ( vp*dump / kappa + alpha ) * dt );
	  a = 0.0;
	  if ( abs ( vp*dump ) > 0.000001 ) a = vp*dump * ( b - 1.0 ) / ( kappa * ( vp*dump + kappa * alpha ) );

	  return b * phidum + a * ( x2 - x1 ) * (1./dx);
	}
// }}}


// DGN nr.c {{{
	int my_float2int(float x)
	{
		if (x >= 0){
			return (int) x;
		} else if (x == (int) x) {
			return (int) x;
		} else {
			return (int) x - 1;
		}
	}

	float hardrock(float z){
		if( z <= 0.75 ){
			return 3260.;
		} else if ( z <= 2.70 ){
			return 3324.*powf(z, 0.067);
		} else if ( z <= 8.0 ){
			return 3447.*powf(z, 0.0209);
		} else {
			return 0;
		}
	}

	double dradxx(double strike, double dip, double rake)
	{
		return  cos(rake)*sin(dip)*sin(2.*strike)
			- sin(rake)*sin(2.*dip)*cos(strike)*cos(strike) ;
	}

	double dradyy(double strike, double dip, double rake)
	{
		return  - ( cos(rake)*sin(dip)*sin(2.*strike)
				+ sin(rake)*sin(2.*dip)*sin(strike)*sin(strike) );
	}

	double dradzz(double strike, double dip, double rake)
	{
		return sin(rake)*sin(2.*dip);
	}

	double dradxy(double strike, double dip, double rake)
	{
		return cos(rake)*sin(dip)*cos(2.*strike)+ 0.5*sin(rake)*sin(2.*dip)*sin(2.*strike);
	}

	double dradyz(double strike, double dip, double rake)
	{
		return cos(rake)*cos(dip)*cos(strike) + sin(rake)*cos(2.*dip)*sin(strike);
	}

	double dradxz(double strike, double dip, double rake)
	{
		return cos(rake)*cos(dip)*sin(strike) - sin(rake)*cos(2.*dip)*cos(strike);
	}
// }}}
