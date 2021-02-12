/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2011, 2012  Centre National de la Recherche Scientifique
 * Copyright (C) 2010-2011  Université de Bordeaux 1
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <sys/time.h>
#include "stencil.h"

/* Main application */

/* default parameter values */
static unsigned  bind_tasks = 0;

static unsigned niter = 32;
static unsigned ticks = 1000;

/* Problem size */
static unsigned sizex = 0;
static unsigned sizey = 0;
static unsigned sizez = 0;

static unsigned nbx = 8;
static unsigned nby = 8;

ondes3d_params params;

/*pour un decoupage en blocs de taille fixe*/
bool fixed_block_size = false;
bool alignment = false;

unsigned bsizex = 16*4;
unsigned bsizey = 8*4;

char parameter_file[STRMAX];
// DGN

/* StarPU top variables */
struct starpu_top_data* starpu_top_init_loop;
struct starpu_top_data* starpu_top_achieved_loop;

/*
 *	Initialization
 */

ondes3d_params* get_params(void)
{
	return &params;
}

unsigned get_bind_tasks(void)
{
	return bind_tasks;
}

unsigned get_niter(void)
{
	return niter;
}

void set_niter(unsigned n)
{
	niter = n;
	return;
}

unsigned get_ticks(void)
{
	return ticks;
}

unsigned get_nbx()
{
	return nbx;
}

unsigned get_nby()
{
	return nby;
}

unsigned get_sizex()
{
	return sizex;
}

unsigned get_sizey()
{
	return sizey;
}

unsigned get_sizez()
{
	return sizez;
}

bool aligned()
{
	return alignment;
}

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-b") == 0)
		{
			bind_tasks = 1;
		}

		// DGN
		if (strcmp(argv[i], "-nbx") == 0)
		{
			nbx = atoi(argv[++i]);
		}

		// DGN
		if (strcmp(argv[i], "-nby") == 0)
		{
			nby = atoi(argv[++i]);
		}

		// DGN
		if (strcmp(argv[i], "-align") == 0)
		{
			alignment = true;
		}

		// DGN
		if (strcmp(argv[i], "-param") == 0)
		{
			i++;
			strncpy(parameter_file, argv[i], MAX(strlen(argv[i]), STRMAX-1)) ;
		}

		// DGN
		if (strcmp(argv[i], "-fixed") == 0)
		{
			fixed_block_size = true;
		}

		// DGN
		if (strcmp(argv[i], "-bsizex") == 0)
		{
			bsizex = atoi(argv[++i]);
		}
		
		// DGN
		if (strcmp(argv[i], "-bsizey") == 0)
		{
			bsizey = atoi(argv[++i]);
		}

		// DGN
		if (strcmp(argv[i], "-ticks") == 0)
		{
			ticks = atoi(argv[++i]);
		}

		if (strcmp(argv[i], "-h") == 0)
		{
			 fprintf(stderr, "Usage : %s [options...]\n", argv[0]);
			 fprintf(stderr, "\n");
			 fprintf(stderr, "Options:\n");
			 fprintf(stderr, "-b				   Bind tasks on CPUs/GPUs\n");
			 fprintf(stderr, "-nbx <n>		   Number of blocks on X axis (%d by default)\n", nbx);
			 fprintf(stderr, "-nby <n>		   Number of blocks on Y axis (%d by default)\n", nby);
			 fprintf(stderr, "-align		   Use padding to align data for coalesced GPU reads\n");
			 fprintf(stderr, "-param <path>	Parameter file (%s by default)\n", PRM);
			 fprintf(stderr, "-fixed		   Split domain with fixed size blocks\n");
			 fprintf(stderr, "\t-bsizex <n>	Size of blocks on X axis (%d by default)\n", bsizex);
			 fprintf(stderr, "\t-bsizey <n>	Size of blocks on X axis (%d by default)\n", bsizey);
			 fprintf(stderr, "-ticks <t>		How often to put ticks in the output (ms, %d by default)\n", ticks);
			 exit(0);
		}
	}
}

// DGN
static void init_problem(int argc, char **argv, int rank, int world_size)
{
	parse_args(argc, argv);

	if (getenv("STARPU_TOP"))
	{
		starpu_top_init_loop = starpu_top_add_data_integer("Task creation iter", 0, niter, 1);
		starpu_top_achieved_loop = starpu_top_add_data_integer("Task achieved iter", 0, niter, 1);
		starpu_top_init_and_wait("stencil_top example");
	}

	// read param file
	
	read_parameter_file(&params, parameter_file, rank);

	// decoupage du domaine en blocs
	sizex = params.xmax-params.xmin + 1 + 2*DELTA;
	sizey = params.ymax-params.ymin + 1 + 2*DELTA;
	// DGN zmax = 0 ... sûr ??
	sizez = 0 - params.zmin + 1 + DELTA;

	params.sizex = sizex;
	params.sizey = sizey;
	params.sizez = sizez;

	if (fixed_block_size) {
		nbx = (sizex + bsizex -1) / bsizex;
		nby = (sizey + bsizey -1) / bsizey;
	} else {
		bsizex = (sizex + nbx -1) / nbx;
		bsizey = (sizey + nby -1) / nby;
	}
	

	if (rank == 0) printf("sizex : %d\nsizey : %d \nsizez : %d\nnbx : %d\nnby : %d\nbsizex : %d\nbsizey : %d\n", sizex, sizey, sizez, nbx, nby, bsizex, bsizey);

	// DGN temporaire : ça m'évite plein de complications, à revoir qd plus de temps !!!
	STARPU_ASSERT(bsizex >= DELTA);
	STARPU_ASSERT(bsizey >= DELTA);

	// allocation des blocs
	create_blocks_array(sizex, sizey, sizez, nbx, nby, bsizex, bsizey, &params);
	
	/* Select the MPI process which should compute the different blocks */
	assign_blocks_to_mpi_nodes(world_size);
	

	assign_blocks_to_workers(rank);
	

	/* Allocate the different memory blocks, if used by the MPI process */
	allocate_memory_on_node(rank);
	

	// create indirection for CPML points
	create_cpml_indirection(rank);
	

	// lecture sources
	read_sources_positions(&params, rank);
	

	// source time function
	read_source_time_function(&params, rank);
	

	// lecture stations
	// DGN TODO : passer en local les indices des stations (comme pour les sources)
	read_stations_positions(&params, rank);
	

	// definition material properties
	set_material_properties(&params, rank);
	

	// extension dans les CPMLs
	set_cpmls(&params, rank);
	
	display_memory_consumption(rank);

	who_runs_what_len = 2*niter;
	who_runs_what = (int *) calloc(nbx*nby * who_runs_what_len, sizeof(*who_runs_what));
	who_runs_what_index = (int *) calloc(nbx*nby, sizeof(*who_runs_what_index));
	last_tick = (struct timeval *) calloc(nbx*nby, sizeof(*last_tick));
	
}

/*
 *	Main body
 */

struct timeval start;
struct timeval end;
double timing; 

void f(unsigned task_per_worker[STARPU_NMAXWORKERS])
{
	unsigned total = 0;
	int worker;

	for (worker = 0; worker < STARPU_NMAXWORKERS; worker++)
		total += task_per_worker[worker];
	for (worker = 0; worker < STARPU_NMAXWORKERS; worker++)
	{
		if (task_per_worker[worker])
		{
			char name[32];
			starpu_worker_get_name(worker, name, sizeof(name));
			fprintf(stdout,"\t%s -> %d (%2.2f%%)\n", name, task_per_worker[worker], (100.0*task_per_worker[worker])/total);
		}
	}
}

void ff(unsigned task_per_worker[STARPU_NMAXWORKERS], float time_per_worker[STARPU_NMAXWORKERS])
{
	unsigned total_task = 0;
	int worker;
	float total_time;

	for (worker = 0; worker < STARPU_NMAXWORKERS; worker++)
		total_task += task_per_worker[worker];
	for (worker = 0; worker < STARPU_NMAXWORKERS; worker++)
		total_time += time_per_worker[worker];
	for (worker = 0; worker < STARPU_NMAXWORKERS; worker++)
	{
		if (task_per_worker[worker])
		{
			char name[32];
			starpu_worker_get_name(worker, name, sizeof(name));
			fprintf(stdout,"\t%s -> \t%d (%2.2f%%)\n\t\t\ttotal time : %.2f,\tby task : %.2f, (%2.2f%%)\n", name, task_per_worker[worker], (100.0*task_per_worker[worker])/total_task, time_per_worker[worker], time_per_worker[worker]/task_per_worker[worker], 100.*(time_per_worker[worker]/total_time));
		}
	}
}

unsigned global_workerid(unsigned local_workerid)
{
#ifdef STARPU_USE_MPI
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	unsigned workers_per_node = starpu_worker_get_count();

	return (local_workerid + rank*workers_per_node);
#else
	return local_workerid;
#endif
}

// DGN
int main(int argc, char **argv)
{
	int rank;
	int world_size;
	int ret;

#ifdef STARPU_USE_MPI
	int thread_support;
	if (MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &thread_support))
	{
		fprintf(stdout, "MPI_Init_thread failed\n");
	}
	if (thread_support == MPI_THREAD_FUNNELED)
		fprintf(stdout,"Warning: MPI only has funneled thread support, not serialized, hoping this will work\n");
	if (thread_support < MPI_THREAD_FUNNELED)
		fprintf(stdout,"Warning: MPI does not have thread support!\n");
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#else
	rank = 0;
	world_size = 1;
#endif
	if (rank == 0)
	{
		fprintf(stdout, "Running on %d nodes\n", world_size);
		fflush(stdout);
	}

	// DGN
	STARPU_ASSERT(world_size == (NPROCY*NPROCX));

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

#ifdef STARPU_USE_MPI
	starpu_mpi_initialize();
#endif

#ifdef STARPU_USE_OPENCL
        opencl_life_init();
        opencl_shadow_init();
#endif /*STARPU_USE_OPENCL*/

   strncpy(parameter_file, PRM, strlen(PRM));
   DGN_DBG
	init_problem(argc, argv, rank, world_size);
	DGN_DBG
	create_tasks(rank);
	DGN_DBG

#ifdef STARPU_USE_MPI
	int barrier_ret = MPI_Barrier(MPI_COMM_WORLD);
	STARPU_ASSERT(barrier_ret == MPI_SUCCESS);
#endif
	if (rank == 0)
		fprintf(stdout, "GO !\n");

	gettimeofday(&start, NULL);

	starpu_tag_notify_from_apps(TAG_INIT_TASK);

	wait_end_tasks(rank);
	
	if (rank == 0)
		fprintf(stdout, "DONE ...\n");

	gettimeofday(&end, NULL);

#ifdef STARPU_USE_MPI
	barrier_ret = MPI_Barrier(MPI_COMM_WORLD);
	STARPU_ASSERT(barrier_ret == MPI_SUCCESS);
#endif

#if 0
	check(rank);
#endif

	/*display_debug(nbz, niter, rank);*/

#ifdef STARPU_USE_MPI
	starpu_mpi_shutdown();
#endif

	/* timing in us */
	timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

	double min_timing = timing;
	double max_timing = timing;
	double sum_timing = timing;

#ifdef STARPU_USE_MPI
	int reduce_ret;

	reduce_ret = MPI_Reduce(&timing, &min_timing, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	STARPU_ASSERT(reduce_ret == MPI_SUCCESS);

	reduce_ret = MPI_Reduce(&timing, &max_timing, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	STARPU_ASSERT(reduce_ret == MPI_SUCCESS);

	reduce_ret = MPI_Reduce(&timing, &sum_timing, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	STARPU_ASSERT(reduce_ret == MPI_SUCCESS);

	/* XXX we should do a gather instead, here we assume that non initialized values are still 0 */
	int *who_runs_what_tmp = malloc(nbx * nby * who_runs_what_len * sizeof(*who_runs_what));
	reduce_ret = MPI_Reduce(who_runs_what, who_runs_what_tmp, nbx * nby * who_runs_what_len, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	STARPU_ASSERT(reduce_ret == MPI_SUCCESS);

	memcpy(who_runs_what, who_runs_what_tmp, nbx * nby * who_runs_what_len * sizeof(*who_runs_what));

	/* XXX we should do a gather instead, here we assume that non initialized values are still 0 */
	int *who_runs_what_index_tmp = malloc(nbx * nby * sizeof(*who_runs_what_index));
	reduce_ret = MPI_Reduce(who_runs_what_index, who_runs_what_index_tmp, nbx * nby, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	STARPU_ASSERT(reduce_ret == MPI_SUCCESS);

	memcpy(who_runs_what_index, who_runs_what_index_tmp, nbx * nby * sizeof(*who_runs_what_index));
#endif

	if (rank == 0)
	{
#if 1 
		fprintf(stdout, "compute veloc:\n");
		ff(veloc_update_per_worker, time_spent_veloc_update);
		fprintf(stdout, "XP:\n");
		ff(veloc_xp_per_worker, time_spent_veloc_xp);
		fprintf(stdout, "XM:\n");
		ff(veloc_xm_per_worker, time_spent_veloc_xm);
		fprintf(stdout, "YP:\n");
		ff(veloc_yp_per_worker, time_spent_veloc_yp);
		fprintf(stdout, "YM:\n");
		ff(veloc_ym_per_worker, time_spent_veloc_ym);

		fprintf(stdout, "compute stress:\n");
		ff(stress_update_per_worker, time_spent_stress_update);
		fprintf(stdout, "XP:\n");
		ff(stress_xp_per_worker, time_spent_stress_xp);
		fprintf(stdout, "XM:\n");
		ff(stress_xm_per_worker, time_spent_stress_xm);
		fprintf(stdout, "YP:\n");
		ff(stress_yp_per_worker, time_spent_stress_yp);
		fprintf(stdout, "YM:\n");
		ff(stress_ym_per_worker, time_spent_stress_ym);
#endif
#if 0
		unsigned nb_blocks_per_process = (nbx*nby + world_size - 1) / world_size;

		unsigned bx, by, iter;
		unsigned last;
		for (iter = 0; iter < who_runs_what_len; iter++)
		{
			last = 1;
			for (bx = 0; bx < nbx; bx++)
			{	for (by = 0; by < nby; by++)
				{
					if ((IND(bx,by) % nb_blocks_per_process) == 0)
						fprintf(stdout, "| ");

					if (who_runs_what_index[IND(bx,by)] <= iter)
						fprintf(stdout,"_ ");
					else
					{
						last = 0;
						if (who_runs_what[IND(bx,by) + iter * nbx*nby] == -1)
							fprintf(stdout,"* ");
						else
							fprintf(stdout, "%d ", who_runs_what[IND(bx,by) + iter * nbx*nby]);
					}
				}
			}
			fprintf(stdout, "\n");

			if (last)
				break;
		}
#endif

		fflush(stdout);

		fprintf(stdout, "Computation took: %f ms on %d MPI processes\n", max_timing/1000, world_size);
		fprintf(stdout, "\tMIN : %f ms\n", min_timing/1000);
		fprintf(stdout, "\tMAX : %f ms\n", max_timing/1000);
		fprintf(stdout, "\tAVG : %f ms\n", sum_timing/(world_size*1000));
	}

	starpu_shutdown();

#ifdef STARPU_USE_MPI
	MPI_Finalize();
#endif

	return 0;
}
