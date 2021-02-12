/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2012  Université de Bordeaux 1
 * Copyright (C) 2012  Centre National de la Recherche Scientifique
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

#include "stencil.h"
#include <sys/time.h>

#ifndef timersub
#define	timersub(x, y, res) \
	do \
	{						   \
		(res)->tv_sec = (x)->tv_sec - (y)->tv_sec; \
		(res)->tv_usec = (x)->tv_usec - (y)->tv_usec; \
		if ((res)->tv_usec < 0) \
		{			 \
			(res)->tv_sec--; \
			(res)->tv_usec += 1000000; \
		} \
	} while (0)
#endif
#ifndef timeradd
#define	timeradd(x, y, res) \
	do \
	{						   \
		(res)->tv_sec = (x)->tv_sec + (y)->tv_sec; \
		(res)->tv_usec = (x)->tv_usec + (y)->tv_usec; \
		if ((res)->tv_usec >= 1000000) \
		{			       \
			(res)->tv_sec++; \
			(res)->tv_usec -= 1000000; \
		} \
	} while (0)
#endif

/* Computation Kernels */

/*
 * There are three codeletets:
 *
 * - cl_update, which takes a block and the boundaries of its neighbours, loads
 *   the boundaries into the block and perform some update loops:
 *
 *     comp. buffer      save. buffers        comp. buffer        save. buffers        comp. buffer
 *   |     ...     |                                                                                      
 *   |             | +------------------+ +------------------+                                            
 *   |     #N+1    | | #N+1 bottom copy====>#N+1 bottom copy |                                            
 *   +-------------+ +------------------+ +------------------+                                            
 *   | #N top copy | |   #N top copy    | |                  |                                            
 *   +-------------+ +------------------+ |                  |                                            
 *                                        | #N               |                                            
 *                                                 ...                                                    
 *                                        |                  | +----------------+ +----------------------+
 *                                        |                  | | #N bottom copy | | block #N bottom copy |
 * ^                                      +------------------+ +----------------+ +----------------------+
 * |                                      | #N-1 top copy   <====#N-1 top copy  | |  block #N-1          |
 * |                                      +------------------+ +----------------+ |                      |
 * Z                                                                                     ...
 *
 * - save_cl_top, which take a block and its top boundary, and saves the top of
 *   the block into the boundary (to be given as bottom of the neighbour above
 *   this block).
 *
 *     comp. buffer      save. buffers        comp. buffer        save. buffers        comp. buffer
 *   |     ...     |                                                                                      
 *   |             | +------------------+ +------------------+                                            
 *   |     #N+1    | | #N+1 bottom copy | | #N+1 bottom copy |                                            
 *   +-------------+ +------------------+ +------------------+                                            
 *   | #N top copy | |   #N top copy   <====                 |                                            
 *   +-------------+ +------------------+ |..................|                                            
 *                                        | #N               |                                            
 *                                                 ...                                                    
 *                                        |                  | +----------------+ +----------------------+
 *                                        |                  | | #N bottom copy | | block #N bottom copy |
 * ^                                      +------------------+ +----------------+ +----------------------+
 * |                                      | #N-1 top copy    | | #N-1 top copy  | |  block #N-1          |
 * |                                      +------------------+ +----------------+ |                      |
 * Z                                                                                     ...
 *
 * - save_cl_bottom, same for the bottom
 *     comp. buffer      save. buffers        comp. buffer        save. buffers        comp. buffer
 *   |     ...     |                                                                                      
 *   |             | +------------------+ +------------------+                                            
 *   |     #N+1    | | #N+1 bottom copy | | #N+1 bottom copy |                                            
 *   +-------------+ +------------------+ +------------------+                                            
 *   | #N top copy | |   #N top copy    | |                  |                                            
 *   +-------------+ +------------------+ |                  |                                            
 *                                        | #N               |                                            
 *                                                 ...                                                    
 *                                        |..................| +----------------+ +----------------------+
 *                                        |                 ====>#N bottom copy | | block #N bottom copy |
 * ^                                      +------------------+ +----------------+ +----------------------+
 * |                                      | #N-1 top copy    | | #N-1 top copy  | |  block #N-1          |
 * |                                      +------------------+ +----------------+ |                      |
 * Z                                                                                     ...
 *
 * The idea is that the computation buffers thus don't have to move, only their
 * boundaries are copied to buffers that do move (be it CPU/GPU, GPU/GPU or via
 * MPI)
 *
 * For each of the buffers above, there are two (0/1) buffers to make new/old switch costless.
 * 
 * 
 * 
 * Edit David : 12 codelets
 * cl_update_source
 * 
 * cl_compute_veloc -> done (cpu + cuda)
 * cl_compute_stress -> done (cpu + cuda)
 * 
 * cl_record_seismo
 * 
 * save_veloc_xp_cl -> done (cpu + cuda)
 * save_veloc_xm_cl -> done (cpu + cuda)
 * save_veloc_yp_cl -> done (cpu + cuda)
 * save_veloc_ym_cl -> done (cpu + cuda)
 * 
 * save_stress_xp_cl -> done (cpu + cuda)
 * save_stress_xm_cl -> done (cpu + cuda)
 * save_stress_yp_cl -> done (cpu + cuda)
 * save_stress_ym_cl -> done (cpu + cuda)
 */



/* Record which GPU ran which block, for nice pictures */
int who_runs_what_len;
int *who_runs_what;
int *who_runs_what_index;
struct timeval *last_tick;

/* Achieved iterations */
static int achieved_iter;

static float dump0, kappa0, alpha0;

float time_spent_veloc_xp[STARPU_NMAXWORKERS];
float time_spent_veloc_xm[STARPU_NMAXWORKERS];
float time_spent_veloc_yp[STARPU_NMAXWORKERS];
float time_spent_veloc_ym[STARPU_NMAXWORKERS];
float time_spent_stress_xp[STARPU_NMAXWORKERS];
float time_spent_stress_xm[STARPU_NMAXWORKERS];
float time_spent_stress_yp[STARPU_NMAXWORKERS];
float time_spent_stress_ym[STARPU_NMAXWORKERS];

float time_spent_veloc_update[STARPU_NMAXWORKERS];
float time_spent_stress_update[STARPU_NMAXWORKERS];

// Divers {{{
	void set_dump0(float val) {
		dump0 = val;
	}

	void set_kappa0(float val) {
		kappa0 = val;
	}

	void set_alpha0(float val) {
		alpha0 = val;
	}


	/* Record how many updates each worker performed */
	unsigned veloc_update_per_worker[STARPU_NMAXWORKERS];
	unsigned stress_update_per_worker[STARPU_NMAXWORKERS];

	// -> ????
	static void record_who_runs_what(struct block_description *block)
	{
		struct timeval tv, tv2, diff, delta = {.tv_sec = 0, .tv_usec = get_ticks() * 1000};
		int workerid = starpu_worker_get_id();

		gettimeofday(&tv, NULL);
		timersub(&tv, &start, &tv2);
		timersub(&tv2, &last_tick[IND(block->bx, block->by)], &diff);
		while (timercmp(&diff, &delta, >=))
		{
			timeradd(&last_tick[IND(block->bx, block->by)], &delta, &last_tick[IND(block->bx, block->by)]);
			timersub(&tv2, &last_tick[IND(block->bx, block->by)], &diff);
			// DGN TODO !!!!!
			if (who_runs_what_index[IND(block->bx, block->by)] < who_runs_what_len)
				who_runs_what[IND(block->bx, block->by) + (who_runs_what_index[IND(block->bx, block->by)]++) * get_nbx()*get_nby()] = -1;
		}

		if (who_runs_what_index[IND(block->bx, block->by)] < who_runs_what_len)
			who_runs_what[IND(block->bx, block->by) + (who_runs_what_index[IND(block->bx, block->by)]++) * get_nbx()*get_nby()] = global_workerid(workerid);
	}

	static void check_load(struct starpu_block_interface *block, struct starpu_block_interface *boundary, unsigned dir)
	{
		/* Sanity checks */
	  //INFO( "CHECK : dir : %i, block : %ix%ix%i, boundary : %ix%ix%i\n", dir, block->nx, block->ny, block->nz, boundary->nx, boundary->ny, boundary->nz);
		switch(dir)
		{
			case XP :	
			case XM :	
								STARPU_ASSERT(boundary->nx == K);
								STARPU_ASSERT(block->ny == boundary->ny);
								STARPU_ASSERT(block->nz == boundary->nz);
								break;
			case YP :	
			case YM :	STARPU_ASSERT(boundary->ny == K);
								STARPU_ASSERT(block->nx == boundary->nx);
								STARPU_ASSERT(block->nz == boundary->nz);
								break;

			default : STARPU_ASSERT(0);
		}
	}
// }}}

// load_subblock_from_buffer_cpu {{{
	/*
	 * Load a neighbour's boundary into block, CPU version
	 */
	static void load_subblock_from_buffer_cpu(void *_block, void *_boundary, unsigned direction, struct block_description* descr)
	{
		struct starpu_block_interface *block = (struct starpu_block_interface *)_block;
		struct starpu_block_interface *boundary = (struct starpu_block_interface *)_boundary;
		//check_load(block, boundary, direction);

		float *block_data = (float *)block->ptr;
		float *boundary_data = (float *)boundary->ptr;
		unsigned offset, nb_block_per_slice;
		unsigned offset_b = 0;

		switch(direction)
		{	
			case XP :	offset = aligned()? block->ldy-descr->padding-K:block->ldy-K;
							break;
			case XM :	offset = aligned()?(ALIGN-K):0;
							break;
			case YP :	offset = block->ldz - K*block->ldy;
							break;
			case YM :	offset = 0;
							break;
			default : STARPU_ASSERT(0);
		}

		if (descr->bx==0 && descr->by==1 && direction==2) DGN_DBG
		/* We do a bunch of contiguous memory transfers */
		switch(direction)
		{
			case XP :	
			case XM :	nb_block_per_slice = K;
						for (int iz=0; iz<block->nz; iz++) {
							for (int iy=0; iy<block->ny; iy++) {
								memcpy(&block_data[offset], &boundary_data[offset_b], nb_block_per_slice*block->elemsize);
								offset += block->ldy;
								offset_b += nb_block_per_slice;
							}
						}
						break;
			case YP :	
			case YM :	nb_block_per_slice = K*block->ldy;
						for (int iz=0; iz<block->nz; iz++) {
							memcpy(&block_data[offset], &boundary_data[offset_b], nb_block_per_slice*block->elemsize);
							offset += block->ldz;
							offset_b += nb_block_per_slice;
						}
						break;


			default : STARPU_ASSERT(0);
		}
	}
// }}}

#ifdef STARPU_USE_CUDA
// load_subblock_from_buffer_cuda {{{
	/*
	 * Load a neighbour's boundary into block, CUDA version
	 */
	static void load_subblock_from_buffer_cuda(void *_block, void *_boundary, unsigned direction, struct block_description* descr)
	{

		struct starpu_block_interface *block = (struct starpu_block_interface *)_block;
		struct starpu_block_interface *boundary = (struct starpu_block_interface *)_boundary;
		float *block_data = (float *)block->ptr;
		float *boundary_data = (float *)boundary->ptr;

#ifdef USE_KERNEL_FOR_COPY
		cudaError_t cures;
		// trop d'indirections ... a condenser !
    copyBlockBoundary( block_data, boundary_data,
                    block->nx, block->ny,  block->nz,
                    aligned(), descr->padding, direction, FROM_BUF);
		if ((cures = cudaStreamSynchronize(starpu_cuda_get_local_stream())) != cudaSuccess)
			STARPU_CUDA_REPORT_ERROR(cures);
# else
		unsigned offset, nb_block_per_slice;
		unsigned offset_b = 0;

		switch(direction)
		{	
			case XP :	offset = aligned()? block->ldy-descr->padding-K:block->ldy-K;
							break;
			case XM :	offset = aligned()?(ALIGN-K):0;
							break;
			case YP :	offset = block->ldz - K*block->ldy;
							break;
			case YM :	offset = 0;
							break;
			default : STARPU_ASSERT(0);
		}

		if (descr->bx==0 && descr->by==1 && direction==2) DGN_DBG
		/* We do a bunch of contiguous memory transfers */
		switch(direction)
		{
			case XP :	
			case XM :	nb_block_per_slice = K;
						for (int iz=0; iz<block->nz; iz++) {
							for (int iy=0; iy<block->ny; iy++) {
								cudaMemcpyAsync(&block_data[offset], &boundary_data[offset_b], nb_block_per_slice*block->elemsize, cudaMemcpyDeviceToDevice, starpu_cuda_get_local_stream());
								offset += block->ldy;
								offset_b += nb_block_per_slice;
							}
						}
						break;
			case YP :	
			case YM :	nb_block_per_slice = K*block->ldy;
						for (int iz=0; iz<block->nz; iz++) {
							cudaMemcpyAsync(&block_data[offset], &boundary_data[offset_b], nb_block_per_slice*block->elemsize, cudaMemcpyDeviceToDevice, starpu_cuda_get_local_stream());
							offset += block->ldz;
							offset_b += nb_block_per_slice;
						}
						break;


			default : STARPU_ASSERT(0);
		}
#endif
	}
// }}}

// compute_veloc_func_cuda {{{
	/*
	 * cl_compute_veloc (CUDA version)
	 */
	static void compute_veloc_func_cuda(void *descr[], void *arg)
	{
					DGN_DBG
		struct block_description *block = (struct block_description *) arg;
		int workerid = starpu_worker_get_id();
		block->iter++;
		// if (block->bx == 0 && block->by == 0)
		// 	fprintf(stderr,"!!! DO compute_veloc_cuda block{%d, %d} CUDA%d !!!\n", block->bx, block->by, workerid);
		// else
		INFO( "!!! DO compute_veloc_cuda it=%d, block{%d, %d} CUDA%d !!!\n", block->iter, block->bx, block->by, workerid);
	#ifdef STARPU_USE_MPI
		int rank = 0;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		DEBUG( "!!!                 RANK %d                    !!!\n", rank);
	#endif
		

		// unsigned block_size_z = get_block_size(block->bz);
		unsigned i;
		veloc_update_per_worker[workerid]++;


		record_who_runs_what(block);

#if 0
		/*already done in record_seismo*/
		/*
		 *	Load neighbours' boundaries :
		 */
		 /*XP*/
		if (block->bx < get_nbx()-1) {
			load_subblock_from_buffer_cuda(descr[3], descr[9], XP, block);
			load_subblock_from_buffer_cuda(descr[4], descr[10], XP, block);
			load_subblock_from_buffer_cuda(descr[5], descr[11], XP, block);
			load_subblock_from_buffer_cuda(descr[6], descr[12], XP, block);
			load_subblock_from_buffer_cuda(descr[7], descr[13], XP, block);
			load_subblock_from_buffer_cuda(descr[8], descr[14], XP, block);
		}

		 /*XM*/
		if (block->bx > 0) {
			load_subblock_from_buffer_cuda(descr[3], descr[15], XM, block);
			load_subblock_from_buffer_cuda(descr[4], descr[16], XM, block);
			load_subblock_from_buffer_cuda(descr[5], descr[17], XM, block);
			load_subblock_from_buffer_cuda(descr[6], descr[18], XM, block);
			load_subblock_from_buffer_cuda(descr[7], descr[19], XM, block);
			load_subblock_from_buffer_cuda(descr[8], descr[20], XM, block);
		}

		 /*YP*/
		if (block->by < get_nby()-1) {
			load_subblock_from_buffer_cuda(descr[3], descr[21], YP, block);
			load_subblock_from_buffer_cuda(descr[4], descr[22], YP, block);
			load_subblock_from_buffer_cuda(descr[5], descr[23], YP, block);
			load_subblock_from_buffer_cuda(descr[6], descr[24], YP, block);
			load_subblock_from_buffer_cuda(descr[7], descr[25], YP, block);
			load_subblock_from_buffer_cuda(descr[8], descr[26], YP, block);
		}
		
		/*YM*/
		if (block->by > 0) {
			load_subblock_from_buffer_cuda(descr[3], descr[27], YM, block);
			load_subblock_from_buffer_cuda(descr[4], descr[28], YM, block);
			load_subblock_from_buffer_cuda(descr[5], descr[29], YM, block);
			load_subblock_from_buffer_cuda(descr[6], descr[30], YM, block);
			load_subblock_from_buffer_cuda(descr[7], descr[31], YM, block);
			load_subblock_from_buffer_cuda(descr[8], descr[32], YM, block);
		}
#endif

		// get Data pointers
		struct starpu_block_interface *b_vx = (struct starpu_block_interface *)descr[0];
		struct starpu_block_interface *b_vy = (struct starpu_block_interface *)descr[1];
		struct starpu_block_interface *b_vz = (struct starpu_block_interface *)descr[2];

		struct starpu_block_interface *b_txx = (struct starpu_block_interface *)descr[3];
		struct starpu_block_interface *b_tyy = (struct starpu_block_interface *)descr[4];
		struct starpu_block_interface *b_tzz = (struct starpu_block_interface *)descr[5];
		struct starpu_block_interface *b_txy = (struct starpu_block_interface *)descr[6];
		struct starpu_block_interface *b_txz = (struct starpu_block_interface *)descr[7];
		struct starpu_block_interface *b_tyz = (struct starpu_block_interface *)descr[8];

		struct starpu_block_interface *b_fx = (struct starpu_block_interface *)descr[33];
		struct starpu_block_interface *b_fy = (struct starpu_block_interface *)descr[34];
		struct starpu_block_interface *b_fz = (struct starpu_block_interface *)descr[35];

		struct starpu_block_interface *b_npml = (struct starpu_block_interface *)descr[38];

		struct starpu_block_interface *b_phitxxx = (struct starpu_block_interface *)descr[39];
		struct starpu_block_interface *b_phitxyy = (struct starpu_block_interface *)descr[40];
		struct starpu_block_interface *b_phitxzz = (struct starpu_block_interface *)descr[41];
		struct starpu_block_interface *b_phitxyx = (struct starpu_block_interface *)descr[42];
		struct starpu_block_interface *b_phityyy = (struct starpu_block_interface *)descr[43];
		struct starpu_block_interface *b_phityzz = (struct starpu_block_interface *)descr[44];
		struct starpu_block_interface *b_phitxzx = (struct starpu_block_interface *)descr[45];
		struct starpu_block_interface *b_phityzy = (struct starpu_block_interface *)descr[46];
		struct starpu_block_interface *b_phitzzz = (struct starpu_block_interface *)descr[47];

		struct starpu_block_interface *b_vp = (struct starpu_block_interface *)descr[37];
		struct starpu_block_interface *b_rho = (struct starpu_block_interface *)descr[36];

		float *d_vx = (float *)b_vx->ptr;
		float *d_vy = (float *)b_vy->ptr;
		float *d_vz = (float *)b_vz->ptr;

		float *d_txx = (float *)b_txx->ptr;
		float *d_tyy = (float *)b_tyy->ptr;
		float *d_tzz = (float *)b_tzz->ptr;
		float *d_txy = (float *)b_txy->ptr;
		float *d_txz = (float *)b_txz->ptr;
		float *d_tyz = (float *)b_tyz->ptr;

		float *d_fx = (float *)b_fx->ptr;
		float *d_fy = (float *)b_fy->ptr;
		float *d_fz = (float *)b_fz->ptr;

		float *d_phitxxx = (float *)b_phitxxx->ptr;
		float *d_phitxyy = (float *)b_phitxyy->ptr;
		float *d_phitxzz = (float *)b_phitxzz->ptr;
		float *d_phitxyx = (float *)b_phitxyx->ptr;
		float *d_phityyy = (float *)b_phityyy->ptr;
		float *d_phityzz = (float *)b_phityzz->ptr;
		float *d_phitxzx = (float *)b_phitxzx->ptr;
		float *d_phityzy = (float *)b_phityzy->ptr;
		float *d_phitzzz = (float *)b_phitzzz->ptr;

		float *d_vp = (float *)b_vp->ptr;
		float *d_rho = (float *)b_rho->ptr;

		int *d_npml = (int *)b_npml->ptr;

		// for CPML computations
		bool bxmin, bxmax, bymin, bymax;
		int ixs_min, ixe_min, ixs_max, ixe_max;
		int iys_min, iye_min, iys_max, iye_max;
		ixe_min = -1;
		ixs_max = block->sizex+1;
		iye_min = -1;
		iys_max = block->sizey+1;

		// bien reverifier cette fonction !!!!
		has_cpml(block, &bxmin, &bxmax, &bymin, &bymax, &ixs_min, &ixe_min, &iys_min, &iye_min, &ixs_max, &ixe_max, &iys_max, &iye_max);

		// set in constant memory -> sinon passage en parametre -> plus lourdingue et consommateur de registres
		setConstCPMLIndices(ixe_min, ixs_max, iye_min, iys_max);
		setConstCPMLValue(dump0, kappa0, alpha0);

		int position = 0;
		if (block->bx == 0) 					position += 1;
		if (block->bx == (get_nbx()-1))	position += 2;
		if (block->by == 0)					position += 4;
		if (block->by == (get_nby()-1))	position += 8;

		int grid_x = (int)ceilf((float)block->sizex/(float)NPPDX);
		int grid_y = (int)ceilf((float)block->sizey/(float)NPPDY);
		int grid_z = 1;

		int block_x = NPPDX;
		int block_y = NPPDY;
		int block_z = 1;


		DGN_DBG
		clock_t start = clock();
		// call kernel wrapper
		DEBUG(">>> compute veloc cuda kernel\tit=%d\t{%d, %d}\tCPU %d\n", block->iter, block->bx, block->by, workerid);
		cuda_compute_veloc_host(	&d_txx[block->offset], &d_tyy[block->offset], &d_tzz[block->offset], &d_txy[block->offset], &d_txz[block->offset], &d_tyz[block->offset], 
											&d_vx[block->offset], &d_vy[block->offset], &d_vz[block->offset], 
											&d_fx[block->offset], &d_fy[block->offset], &d_fz[block->offset], 
											d_npml, d_phitxxx, d_phitxyy, d_phitxzz, d_phitxyx, d_phityyy, d_phityzz, d_phitxzx, d_phityzy, d_phitzzz,
											&d_vp[block->offset], &d_rho[block->offset], 
											block->sizex, block->sizey, block->sizez, 
											block->pitchx, block->pitchy, block->pitchz, 
											block->params->ds, block->params->dt, DELTA, position, 
											grid_x, grid_y, grid_z, block_x, block_y, block_z);

		cudaError_t cures;
		if ((cures = cudaStreamSynchronize(starpu_cuda_get_local_stream())) != cudaSuccess)
			STARPU_CUDA_REPORT_ERROR(cures);
		else
			time_spent_veloc_update[starpu_worker_get_id()] += (clock()-start)/(float)(CLOCKS_PER_SEC/1000.);
		DEBUG("<<< compute veloc cuda kernel\tit=%d\t{%d, %d}\tCPU %d\n", block->iter, block->bx, block->by, workerid);
		DGN_DBG
	}
// }}}

// compute_stress_func_cuda {{{
	/*
	 * cl_compute_stress (CUDA version)
	 */
	static void compute_stress_func_cuda(void *descr[], void *arg)
	{
		struct block_description *block = (struct block_description *) arg;
		int workerid = starpu_worker_get_id();
		
	// 	if (block->bx == 0 && block->by == 0)
	// fprintf(stderr,"!!! DO compute_stress_cuda block{%d, %d} CUDA%d !!!\n", block->bx, block->by, workerid);
	// 	else
		INFO( "!!! DO compute_stress_cuda it=%d, block{%d, %d} CUDA%d !!!\n", block->iter, block->bx, block->by, workerid);
	#ifdef STARPU_USE_MPI
		int rank = 0;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		DEBUG( "!!!                 RANK %d                    !!!\n", rank);
	#endif
		

		// unsigned block_size_z = get_block_size(block->bz);
		unsigned i;
		stress_update_per_worker[workerid]++;

		record_who_runs_what(block);

		/*
		 *	Load neighbours' boundaries :
		 */
		 /*XP*/
		if (block->bx < get_nbx()-1) {
			load_subblock_from_buffer_cuda(descr[0], descr[9], XP, block);
			load_subblock_from_buffer_cuda(descr[1], descr[10], XP, block);
			load_subblock_from_buffer_cuda(descr[2], descr[11], XP, block);
		}

		 /*XM*/
		if (block->bx > 0) {
			load_subblock_from_buffer_cuda(descr[0], descr[12], XM, block);
			load_subblock_from_buffer_cuda(descr[1], descr[13], XM, block);
			load_subblock_from_buffer_cuda(descr[2], descr[14], XM, block);
		}


		 /*YP*/
		if (block->by < get_nby()-1) {
			load_subblock_from_buffer_cuda(descr[0], descr[15], YP, block);
			load_subblock_from_buffer_cuda(descr[1], descr[16], YP, block);
			load_subblock_from_buffer_cuda(descr[2], descr[17], YP, block);
		}

		
		/*YM*/
		if (block->by > 0) {
			load_subblock_from_buffer_cuda(descr[0], descr[18], YM, block);
			load_subblock_from_buffer_cuda(descr[1], descr[19], YM, block);
			load_subblock_from_buffer_cuda(descr[2], descr[20], YM, block);
		}


		/*
		 *	Stencils ... do the actual work here :) TODO
		 */
		// get Data pointers
		struct starpu_block_interface *b_vx = (struct starpu_block_interface *)descr[0];
		struct starpu_block_interface *b_vy = (struct starpu_block_interface *)descr[1];
		struct starpu_block_interface *b_vz = (struct starpu_block_interface *)descr[2];

		struct starpu_block_interface *b_txx = (struct starpu_block_interface *)descr[3];
		struct starpu_block_interface *b_tyy = (struct starpu_block_interface *)descr[4];
		struct starpu_block_interface *b_tzz = (struct starpu_block_interface *)descr[5];
		struct starpu_block_interface *b_txy = (struct starpu_block_interface *)descr[6];
		struct starpu_block_interface *b_txz = (struct starpu_block_interface *)descr[7];
		struct starpu_block_interface *b_tyz = (struct starpu_block_interface *)descr[8];


		struct starpu_block_interface *b_npml = (struct starpu_block_interface *)descr[24];

		struct starpu_block_interface *b_phivxx = (struct starpu_block_interface *)descr[25];
		struct starpu_block_interface *b_phivxy = (struct starpu_block_interface *)descr[26];
		struct starpu_block_interface *b_phivxz = (struct starpu_block_interface *)descr[27];
		struct starpu_block_interface *b_phivyx = (struct starpu_block_interface *)descr[28];
		struct starpu_block_interface *b_phivyy = (struct starpu_block_interface *)descr[29];
		struct starpu_block_interface *b_phivyz = (struct starpu_block_interface *)descr[30];
		struct starpu_block_interface *b_phivzx = (struct starpu_block_interface *)descr[31];
		struct starpu_block_interface *b_phivzy = (struct starpu_block_interface *)descr[32];
		struct starpu_block_interface *b_phivzz = (struct starpu_block_interface *)descr[33];

		struct starpu_block_interface *b_mu = (struct starpu_block_interface *)descr[21];
		struct starpu_block_interface *b_lam = (struct starpu_block_interface *)descr[22];
		struct starpu_block_interface *b_vp = (struct starpu_block_interface *)descr[23];

		float *d_vx = (float *)b_vx->ptr;
		float *d_vy = (float *)b_vy->ptr;
		float *d_vz = (float *)b_vz->ptr;

		float *d_txx = (float *)b_txx->ptr;
		float *d_tyy = (float *)b_tyy->ptr;
		float *d_tzz = (float *)b_tzz->ptr;
		float *d_txy = (float *)b_txy->ptr;
		float *d_txz = (float *)b_txz->ptr;
		float *d_tyz = (float *)b_tyz->ptr;

		float *d_mu = (float *)b_mu->ptr;
		float *d_lam = (float *)b_lam->ptr;
		float *d_vp = (float *)b_vp->ptr;

		float *d_phivxx = (float *)b_phivxx->ptr;
		float *d_phivxy = (float *)b_phivxy->ptr;
		float *d_phivxz = (float *)b_phivxz->ptr;
		float *d_phivyx = (float *)b_phivyx->ptr;
		float *d_phivyy = (float *)b_phivyy->ptr;
		float *d_phivyz = (float *)b_phivyz->ptr;
		float *d_phivzx = (float *)b_phivzx->ptr;
		float *d_phivzy = (float *)b_phivzy->ptr;
		float *d_phivzz = (float *)b_phivzz->ptr;

		int *d_npml = (int *)b_npml->ptr;

		int position = 0;
		if (block->bx == 0) 					position += 1;
		if (block->bx == (get_nbx()-1))	position += 2;
		if (block->by == 0)					position += 4;
		if (block->by == (get_nby()-1))	position += 8;

		// for CPML computations
		bool bxmin, bxmax, bymin, bymax;
		int ixs_min, ixe_min, ixs_max, ixe_max;
		int iys_min, iye_min, iys_max, iye_max;
		ixe_min = -1;
		ixs_max = block->sizex+1;
		iye_min = -1;
		iys_max = block->sizey+1;

		// bien reverifier cette fonction !!!!
		has_cpml(block, &bxmin, &bxmax, &bymin, &bymax, &ixs_min, &ixe_min, &iys_min, &iye_min, &ixs_max, &ixe_max, &iys_max, &iye_max);

		// set in constant memory -> sinon passage en parametre -> plus lourdingue et consommateur de registres
		setConstCPMLIndices(ixe_min, ixs_max, iye_min, iys_max);
		setConstCPMLValue(dump0, kappa0, alpha0);

		// call kernel wrapper
		clock_t start =clock();
		DEBUG(">>> compute stress cuda kernel\tit=%d\t{%d, %d}\tCPU %d\n", block->iter, block->bx, block->by, workerid);
		cuda_compute_stress_host(	&d_txx[block->offset], &d_tyy[block->offset], &d_tzz[block->offset], &d_txy[block->offset], &d_txz[block->offset], &d_tyz[block->offset], 
											&d_vx[block->offset], &d_vy[block->offset], &d_vz[block->offset], 
											d_npml, d_phivxx, d_phivxy, d_phivxz, d_phivyx, d_phivyy, d_phivyz, d_phivzx, d_phivzy, d_phivzz,
											&d_mu[block->offset], &d_lam[block->offset], &d_vp[block->offset], 
											block->sizex, block->sizey, block->sizez, 
											block->pitchx, block->pitchy, block->pitchz, 
											block->params->ds, block->params->dt, DELTA, position, 
											(int)ceilf((float)block->sizex/(float)NPPDX), (int)ceilf((float)block->sizey/(float)NPPDY), 1, NPPDX, NPPDY, 1
										);
		DEBUG("<<< compute stress cuda kernel\tit=%d\t{%d, %d}\tCPU %d\n", block->iter, block->bx, block->by, workerid);

		cudaError_t cures;
		if ((cures = cudaStreamSynchronize(starpu_cuda_get_local_stream())) != cudaSuccess)
			STARPU_CUDA_REPORT_ERROR(cures);
		else
			time_spent_stress_update[workerid] += (clock()-start)/(float)(CLOCKS_PER_SEC/1000.);

		if (block->bx == 0 && block->by == 0)
			starpu_top_update_data_integer(starpu_top_achieved_loop, ++achieved_iter);
	}
// }}}
#endif /* STARPU_USE_CUDA */

// compute_veloc_func_cpu {{{
	/*
	 * cl_compute_veloc (CPU version)
	 */
	static void compute_veloc_func_cpu(void *descr[], void *arg)
	{
		struct block_description *block = (struct block_description *) arg;
		int workerid = starpu_worker_get_id();
		block->iter++;	
	// 	if (block->bx == 0 && block->by == 0)
	// fprintf(stderr,"!!! DO compute_veloc_cpu block{%d, %d} CPU%d !!!\n", block->bx, block->by, workerid);
	// 	else
		INFO("!!! DO compute_veloc_cpu it=%d, block{%d, %d} CPU%d !!!\n", block->iter, block->bx, block->by, workerid);
	#ifdef STARPU_USE_MPI
		int rank = 0;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		DEBUG( "!!!                 RANK %d                    !!!\n", rank);
	#endif
		

		unsigned i;
		veloc_update_per_worker[workerid]++;

		record_who_runs_what(block);


		/*
		 *	Load neighbours' boundaries :
		 */
#if 0
		 // DGN : virer ça !!! deja fait dans record_seismo
		 /*XP*/
		DGN_DBG
		if (block->bx < get_nbx()-1) {
			// DEBUG("load_subblock_from_buffer_cpu XP\n");
			load_subblock_from_buffer_cpu(descr[3], descr[9], XP, block);
			load_subblock_from_buffer_cpu(descr[4], descr[10], XP, block);
			load_subblock_from_buffer_cpu(descr[5], descr[11], XP, block);
			load_subblock_from_buffer_cpu(descr[6], descr[12], XP, block);
			load_subblock_from_buffer_cpu(descr[7], descr[13], XP, block);
			load_subblock_from_buffer_cpu(descr[8], descr[14], XP, block);
		}

		 /*XM*/
		DGN_DBG
		if (block->bx > 0) {
			// DEBUG("load_subblock_from_buffer_cpu XM\n");
			load_subblock_from_buffer_cpu(descr[3], descr[15], XM, block);
			load_subblock_from_buffer_cpu(descr[4], descr[16], XM, block);
			load_subblock_from_buffer_cpu(descr[5], descr[17], XM, block);
			load_subblock_from_buffer_cpu(descr[6], descr[18], XM, block);
			load_subblock_from_buffer_cpu(descr[7], descr[19], XM, block);
			load_subblock_from_buffer_cpu(descr[8], descr[20], XM, block);
		}

		 /*YP*/
		DGN_DBG
		if (block->by < get_nby()-1) {
			// DEBUG("load_subblock_from_buffer_cpu YP\n");
			load_subblock_from_buffer_cpu(descr[3], descr[21], YP, block);
			load_subblock_from_buffer_cpu(descr[4], descr[22], YP, block);
			load_subblock_from_buffer_cpu(descr[5], descr[23], YP, block);
			load_subblock_from_buffer_cpu(descr[6], descr[24], YP, block);
			load_subblock_from_buffer_cpu(descr[7], descr[25], YP, block);
			load_subblock_from_buffer_cpu(descr[8], descr[26], YP, block);
		}
		
		/*YM*/
		DGN_DBG
		if (block->by > 0) {
			// DEBUG("load_subblock_from_buffer_cpu YM\n");
			load_subblock_from_buffer_cpu(descr[3], descr[27], YM, block);
			load_subblock_from_buffer_cpu(descr[4], descr[28], YM, block);
			load_subblock_from_buffer_cpu(descr[5], descr[29], YM, block);
			load_subblock_from_buffer_cpu(descr[6], descr[30], YM, block);
			load_subblock_from_buffer_cpu(descr[7], descr[31], YM, block);
			load_subblock_from_buffer_cpu(descr[8], descr[32], YM, block);
		}
		DGN_DBG
#endif

		// get Data pointers
		struct starpu_block_interface *b_vx = (struct starpu_block_interface *)descr[0];
		struct starpu_block_interface *b_vy = (struct starpu_block_interface *)descr[1];
		struct starpu_block_interface *b_vz = (struct starpu_block_interface *)descr[2];

		struct starpu_block_interface *b_txx = (struct starpu_block_interface *)descr[3];
		struct starpu_block_interface *b_tyy = (struct starpu_block_interface *)descr[4];
		struct starpu_block_interface *b_tzz = (struct starpu_block_interface *)descr[5];
		struct starpu_block_interface *b_txy = (struct starpu_block_interface *)descr[6];
		struct starpu_block_interface *b_txz = (struct starpu_block_interface *)descr[7];
		struct starpu_block_interface *b_tyz = (struct starpu_block_interface *)descr[8];

		struct starpu_block_interface *b_fx = (struct starpu_block_interface *)descr[33];
		struct starpu_block_interface *b_fy = (struct starpu_block_interface *)descr[34];
		struct starpu_block_interface *b_fz = (struct starpu_block_interface *)descr[35];

		struct starpu_block_interface *b_npml = (struct starpu_block_interface *)descr[38];

		struct starpu_block_interface *b_phitxxx = (struct starpu_block_interface *)descr[39];
		struct starpu_block_interface *b_phitxyy = (struct starpu_block_interface *)descr[40];
		struct starpu_block_interface *b_phitxzz = (struct starpu_block_interface *)descr[41];
		struct starpu_block_interface *b_phitxyx = (struct starpu_block_interface *)descr[42];
		struct starpu_block_interface *b_phityyy = (struct starpu_block_interface *)descr[43];
		struct starpu_block_interface *b_phityzz = (struct starpu_block_interface *)descr[44];
		struct starpu_block_interface *b_phitxzx = (struct starpu_block_interface *)descr[45];
		struct starpu_block_interface *b_phityzy = (struct starpu_block_interface *)descr[46];
		struct starpu_block_interface *b_phitzzz = (struct starpu_block_interface *)descr[47];

		struct starpu_block_interface *b_vp = (struct starpu_block_interface *)descr[37];
		struct starpu_block_interface *b_rho = (struct starpu_block_interface *)descr[36];

		float *d_vx = (float *)b_vx->ptr;
		float *d_vy = (float *)b_vy->ptr;
		float *d_vz = (float *)b_vz->ptr;

		float *d_txx = (float *)b_txx->ptr;
		float *d_tyy = (float *)b_tyy->ptr;
		float *d_tzz = (float *)b_tzz->ptr;
		float *d_txy = (float *)b_txy->ptr;
		float *d_txz = (float *)b_txz->ptr;
		float *d_tyz = (float *)b_tyz->ptr;

		float *d_fx = (float *)b_fx->ptr;
		float *d_fy = (float *)b_fy->ptr;
		float *d_fz = (float *)b_fz->ptr;

		float *d_phitxxx = (float *)b_phitxxx->ptr;
		float *d_phitxyy = (float *)b_phitxyy->ptr;
		float *d_phitxzz = (float *)b_phitxzz->ptr;
		float *d_phitxyx = (float *)b_phitxyx->ptr;
		float *d_phityyy = (float *)b_phityyy->ptr;
		float *d_phityzz = (float *)b_phityzz->ptr;
		float *d_phitxzx = (float *)b_phitxzx->ptr;
		float *d_phityzy = (float *)b_phityzy->ptr;
		float *d_phitzzz = (float *)b_phitzzz->ptr;

		float *d_vp = (float *)b_vp->ptr;
		float *d_rho = (float *)b_rho->ptr;

		int *d_npml = (int *)b_npml->ptr;

		// for CPML computations
		bool bxmin, bxmax, bymin, bymax;
		int ixs_min, ixe_min, ixs_max, ixe_max;
		int iys_min, iye_min, iys_max, iye_max;
		ixe_min = -1;
		ixs_max = block->sizex+1;
		iye_min = -1;
		iys_max = block->sizey+1;

		// bien reverifier cette fonction !!!!
		has_cpml(block, &bxmin, &bxmax, &bymin, &bymax, &ixs_min, &ixe_min, &iys_min, &iye_min, &ixs_max, &ixe_max, &iys_max, &iye_max);

		// set in constant memory -> sinon passage en parametre -> plus lourdingue et consommateur de registres
		setConstCPMLIndices(ixe_min, ixs_max, iye_min, iys_max);
		setConstCPMLValue(dump0, kappa0, alpha0);

		int position = 0;
		if (block->bx == 0) 					position += 1;
		if (block->bx == (get_nbx()-1))	position += 2;
		if (block->by == 0)					position += 4;
		if (block->by == (get_nby()-1))	position += 8;

		clock_t start =clock();

		DEBUG(">>> compute veloc cpu kernel\tit=%d\t{%d, %d}\tCPU %d\n", block->iter, block->bx, block->by, workerid);
		cpu_compute_veloc	(	&d_txx[block->offset], &d_tyy[block->offset], &d_tzz[block->offset], &d_txy[block->offset], &d_txz[block->offset], &d_tyz[block->offset], 
									&d_vx[block->offset], &d_vy[block->offset], &d_vz[block->offset], 
									&d_fx[block->offset], &d_fy[block->offset], &d_fz[block->offset], 
									d_npml, d_phitxxx, d_phitxyy, d_phitxzz, d_phitxyx, d_phityyy, d_phityzz, d_phitxzx, d_phityzy, d_phitzzz,
									&d_vp[block->offset], &d_rho[block->offset], 
									block->sizex, block->sizey, block->sizez, 
									block->pitchx, block->pitchy, block->pitchz, 
									block->params->ds, block->params->dt, DELTA, position, 
									ixe_min, ixs_max, iye_min, iys_max, dump0, kappa0, alpha0
								);
		float duration = (clock()-start)/(float)(CLOCKS_PER_SEC/1000.);
		time_spent_veloc_update[starpu_worker_get_id()] += duration;
		DEBUG("<<< compute veloc cpu kernel\tit=%d\t{%d, %d}\tCPU %d\n", block->iter, block->bx, block->by, workerid);
	}
// }}}

// compute_stress_func_cpu {{{
	/*
	 * cl_compute_stress (CPU version)
	 */
	static void compute_stress_func_cpu(void *descr[], void *arg)
	{
		DGN_DBG
		struct block_description *block = (struct block_description *) arg;
		int workerid = starpu_worker_get_id();
		
		// if (block->bx == 0 && block->by == 0)
		// 	fprintf(stderr,"!!! DO compute_stress_cpu block{%d, %d} CPU%d !!!\n", block->bx, block->by, workerid);
		// else
		INFO("!!! DO compute_stress_cpu it=%d, block{%d, %d} CPU%d !!!\n", block->iter, block->bx, block->by, workerid);
	#ifdef STARPU_USE_MPI
		int rank = 0;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		DEBUG( "!!!                 RANK %d                    !!!\n", rank);
	#endif
		

		unsigned i;
		stress_update_per_worker[workerid]++;

		record_who_runs_what(block);


		/*
		 *	Load neighbours' boundaries :
		 */
		 /*XP*/
		DGN_DBG
		if (block->bx < get_nbx()-1) {
			load_subblock_from_buffer_cpu(descr[0], descr[9], XP, block);
			load_subblock_from_buffer_cpu(descr[1], descr[10], XP, block);
			load_subblock_from_buffer_cpu(descr[2], descr[11], XP, block);
		}

		 /*XM*/
		DGN_DBG
		if (block->bx > 0) {
			load_subblock_from_buffer_cpu(descr[0], descr[12], XM, block);
			load_subblock_from_buffer_cpu(descr[1], descr[13], XM, block);
			load_subblock_from_buffer_cpu(descr[2], descr[14], XM, block);
		}


		 /*YP*/
		DGN_DBG
		if (block->by < get_nby()-1) {
			load_subblock_from_buffer_cpu(descr[0], descr[15], YP, block);
			load_subblock_from_buffer_cpu(descr[1], descr[16], YP, block);
			load_subblock_from_buffer_cpu(descr[2], descr[17], YP, block);
		}

		
		/*YM*/
		DGN_DBG
		if (block->by > 0) {
			load_subblock_from_buffer_cpu(descr[0], descr[18], YM, block);
			load_subblock_from_buffer_cpu(descr[1], descr[19], YM, block);
			load_subblock_from_buffer_cpu(descr[2], descr[20], YM, block);
		}
		DGN_DBG

		// get Data pointers
		struct starpu_block_interface *b_vx = (struct starpu_block_interface *)descr[0];
		struct starpu_block_interface *b_vy = (struct starpu_block_interface *)descr[1];
		struct starpu_block_interface *b_vz = (struct starpu_block_interface *)descr[2];

		struct starpu_block_interface *b_txx = (struct starpu_block_interface *)descr[3];
		struct starpu_block_interface *b_tyy = (struct starpu_block_interface *)descr[4];
		struct starpu_block_interface *b_tzz = (struct starpu_block_interface *)descr[5];
		struct starpu_block_interface *b_txy = (struct starpu_block_interface *)descr[6];
		struct starpu_block_interface *b_txz = (struct starpu_block_interface *)descr[7];
		struct starpu_block_interface *b_tyz = (struct starpu_block_interface *)descr[8];


		struct starpu_block_interface *b_npml = (struct starpu_block_interface *)descr[24];

		struct starpu_block_interface *b_phivxx = (struct starpu_block_interface *)descr[25];
		struct starpu_block_interface *b_phivxy = (struct starpu_block_interface *)descr[26];
		struct starpu_block_interface *b_phivxz = (struct starpu_block_interface *)descr[27];
		struct starpu_block_interface *b_phivyx = (struct starpu_block_interface *)descr[28];
		struct starpu_block_interface *b_phivyy = (struct starpu_block_interface *)descr[29];
		struct starpu_block_interface *b_phivyz = (struct starpu_block_interface *)descr[30];
		struct starpu_block_interface *b_phivzx = (struct starpu_block_interface *)descr[31];
		struct starpu_block_interface *b_phivzy = (struct starpu_block_interface *)descr[32];
		struct starpu_block_interface *b_phivzz = (struct starpu_block_interface *)descr[33];

		struct starpu_block_interface *b_mu = (struct starpu_block_interface *)descr[21];
		struct starpu_block_interface *b_lam = (struct starpu_block_interface *)descr[22];
		struct starpu_block_interface *b_vp = (struct starpu_block_interface *)descr[23];

		float *d_vx = (float *)b_vx->ptr;
		float *d_vy = (float *)b_vy->ptr;
		float *d_vz = (float *)b_vz->ptr;

		float *d_txx = (float *)b_txx->ptr;
		float *d_tyy = (float *)b_tyy->ptr;
		float *d_tzz = (float *)b_tzz->ptr;
		float *d_txy = (float *)b_txy->ptr;
		float *d_txz = (float *)b_txz->ptr;
		float *d_tyz = (float *)b_tyz->ptr;

		float *d_mu = (float *)b_mu->ptr;
		float *d_lam = (float *)b_lam->ptr;
		float *d_vp = (float *)b_vp->ptr;

		float *d_phivxx = (float *)b_phivxx->ptr;
		float *d_phivxy = (float *)b_phivxy->ptr;
		float *d_phivxz = (float *)b_phivxz->ptr;
		float *d_phivyx = (float *)b_phivyx->ptr;
		float *d_phivyy = (float *)b_phivyy->ptr;
		float *d_phivyz = (float *)b_phivyz->ptr;
		float *d_phivzx = (float *)b_phivzx->ptr;
		float *d_phivzy = (float *)b_phivzy->ptr;
		float *d_phivzz = (float *)b_phivzz->ptr;

		int *d_npml = (int *)b_npml->ptr;

		int position = 0;
		if (block->bx == 0) 					position += 1;
		if (block->bx == (get_nbx()-1))	position += 2;
		if (block->by == 0)					position += 4;
		if (block->by == (get_nby()-1))	position += 8;

		// for CPML computations
		bool bxmin, bxmax, bymin, bymax;
		int ixs_min, ixe_min, ixs_max, ixe_max;
		int iys_min, iye_min, iys_max, iye_max;
		ixe_min = -1;
		ixs_max = block->sizex+1;
		iye_min = -1;
		iys_max = block->sizey+1;

		// bien reverifier cette fonction !!!!
		has_cpml(block, &bxmin, &bxmax, &bymin, &bymax, &ixs_min, &ixe_min, &iys_min, &iye_min, &ixs_max, &ixe_max, &iys_max, &iye_max);

		// set in constant memory -> sinon passage en parametre -> plus lourdingue et consommateur de registres
		setConstCPMLIndices(ixe_min, ixs_max, iye_min, iys_max);
		setConstCPMLValue(dump0, kappa0, alpha0);

		// call kernel wrapper
		clock_t start =clock();

		DEBUG(">>> compute stress cpu kernel\tit=%d\t{%d, %d}\tCPU %d\n", block->iter, block->bx, block->by, workerid);
		cpu_compute_stress(	&d_txx[block->offset], &d_tyy[block->offset], &d_tzz[block->offset], &d_txy[block->offset], &d_txz[block->offset], &d_tyz[block->offset], 
									&d_vx[block->offset], &d_vy[block->offset], &d_vz[block->offset], 
									d_npml, d_phivxx, d_phivxy, d_phivxz, d_phivyx, d_phivyy, d_phivyz, d_phivzx, d_phivzy, d_phivzz,
									&d_mu[block->offset], &d_lam[block->offset], &d_vp[block->offset], 
									block->sizex, block->sizey, block->sizez, 
									block->pitchx, block->pitchy, block->pitchz, 
									block->params->ds, block->params->dt, DELTA, position, 
									ixe_min, ixs_max, iye_min, iys_max, dump0, kappa0, alpha0, block->iter
								);
		DEBUG("<<< compute stress cpu kernel\tit=%d\t{%d, %d}\tCPU %d\n", block->iter, block->bx, block->by, workerid);
		float duration = (clock()-start)/(float)(CLOCKS_PER_SEC/1000.);
		time_spent_stress_update[starpu_worker_get_id()] += duration;
		if (block->bx == 0 && block->by == 0)
			starpu_top_update_data_integer(starpu_top_achieved_loop, ++achieved_iter);
	}
// }}}

// Performance models and codelets structures {{{
	static struct starpu_perfmodel cl_compute_veloc_model =
	{
		.type = STARPU_HISTORY_BASED,
		.symbol = "cl_compute_veloc" 
	};

	static struct starpu_perfmodel cl_compute_stress_model =
	{
		.type = STARPU_HISTORY_BASED,
		.symbol = "cl_compute_stress" 
	};

	struct starpu_codelet cl_compute_veloc =
	{
		.where = 0 |
	#ifdef STARPU_USE_CUDA
			STARPU_CUDA |
	#endif
		 	STARPU_CPU,
		 .cpu_funcs = {compute_veloc_func_cpu, NULL},
	#ifdef STARPU_USE_CUDA
		.cuda_funcs = {compute_veloc_func_cuda, NULL},
	#endif
		.model = &cl_compute_veloc_model,
		.nbuffers = 48,
		.modes = {	STARPU_RW, STARPU_RW, STARPU_RW, /*veloc*/
					STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, /*stress*/
					STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, /*stress boundaries*/
					STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, 
					STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, 
					STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, 
					STARPU_R, STARPU_R, STARPU_R, /*source force*/
					STARPU_R, STARPU_R,  /*rho, vp*/
					STARPU_R, /*cpml indirection*/
					STARPU_RW, STARPU_RW, STARPU_RW, STARPU_RW, STARPU_RW, STARPU_RW, STARPU_RW, STARPU_RW, STARPU_RW, /*phit*/
			     }
	};

	struct starpu_codelet cl_compute_stress =
	{
		.where = 0 |
	#ifdef STARPU_USE_CUDA
			STARPU_CUDA |
	#endif
		 	STARPU_CPU,
		 .cpu_funcs = {compute_stress_func_cpu, NULL},
	#ifdef STARPU_USE_CUDA
		.cuda_funcs = {compute_stress_func_cuda, NULL},
	#endif

		.model = &cl_compute_stress_model,
		.nbuffers = 34,
		.modes = {	STARPU_R, STARPU_R, STARPU_R, /*veloc*/
					STARPU_RW, STARPU_RW, STARPU_RW, STARPU_RW, STARPU_RW, STARPU_RW, /*stress*/
					STARPU_R, STARPU_R, STARPU_R, /*veloc boundaries*/
					STARPU_R, STARPU_R, STARPU_R, 
					STARPU_R, STARPU_R, STARPU_R, 
					STARPU_R, STARPU_R, STARPU_R, 
					STARPU_R, STARPU_R, STARPU_R, /*mu, lam, vp*/
					STARPU_R, /*cpml indirection*/
					STARPU_RW, STARPU_RW, STARPU_RW, STARPU_RW, STARPU_RW, STARPU_RW, STARPU_RW, STARPU_RW, STARPU_RW, /*phiv*/
			     }
	};
// }}}

// load_subblock_into_buffer_cpu {{{
	/*
	 * Save the block internal boundaries to give them to our neighbours.
	 */
	static void load_subblock_into_buffer_cpu(void *_block, void *_boundary, unsigned direction, struct block_description* descr)
	{
		struct starpu_block_interface *block = (struct starpu_block_interface *)_block;
		struct starpu_block_interface *boundary = (struct starpu_block_interface *)_boundary;
		DGN_DBUG(descr->bx, descr->by, direction);
		check_load(block, boundary, direction);
		float *block_data = (float *)block->ptr;
		float *boundary_data = (float *)boundary->ptr;
		unsigned offset, nb_block_per_slice;
		unsigned offset_b = 0;

		switch(direction)
		{	
			case XP :	offset = aligned()? block->ldy-descr->padding-2*K:block->ldy-2*K;
							break;
			case XM :	offset = aligned()?ALIGN:K;
							break;
			case YP :	offset = block->ldz - 2*K*block->ldy;
							break;
			case YM :	offset = K*block->ldy;
							break;
			default : STARPU_ASSERT(0);
		}

		/* We do a bunch of contiguous memory transfers */
		switch(direction)
		{
			case YP :	
			case YM :	nb_block_per_slice = K*block->ldy;
						for (int iz=0; iz<block->nz; iz++) {
							memcpy(&boundary_data[offset_b], &block_data[offset], nb_block_per_slice*block->elemsize);
							offset += block->ldz;
							offset_b += nb_block_per_slice;
						}
						break;
			case XP :	
			case XM :	nb_block_per_slice = K;
						for (int iz=0; iz<block->nz; iz++) {
							for (int iy=0; iy<block->ny; iy++) {
								memcpy(&boundary_data[offset_b], &block_data[offset], nb_block_per_slice*block->elemsize);
								offset += block->ldy;
								offset_b += nb_block_per_slice;
							}
						}
						break;

			default : STARPU_ASSERT(0);
		}
	}
// }}}

#ifdef STARPU_USE_CUDA
// load_subblock_into_buffer_cuda {{{
	static void load_subblock_into_buffer_cuda(void *_block, void *_boundary, unsigned direction, struct block_description* descr)
	{

		struct starpu_block_interface *block = (struct starpu_block_interface *)_block;
		struct starpu_block_interface *boundary = (struct starpu_block_interface *)_boundary;
		float *block_data = (float *)block->ptr;
		float *boundary_data = (float *)boundary->ptr;
#ifdef USE_KERNEL_FOR_COPY
		cudaError_t cures;
		// trop d'indirections ... a condenser !
    copyBlockBoundary( block_data, boundary_data,
                    block->nx, block->ny,  block->nz,
                    aligned(), descr->padding, direction, TO_BUF);
		if ((cures = cudaStreamSynchronize(starpu_cuda_get_local_stream())) != cudaSuccess)
			STARPU_CUDA_REPORT_ERROR(cures);
#else
		unsigned offset, nb_block_per_slice;
		unsigned offset_b = 0;

		switch(direction)
		{	
			case XP :	offset = aligned()? block->ldy-descr->padding-2*K:block->ldy-2*K;
							break;
			case XM :	offset = aligned()?ALIGN:K;
							break;
			case YP :	offset = block->ldz - 2*K*block->ldy;
							break;
			case YM :	offset = K*block->ldy;
							break;
			default : STARPU_ASSERT(0);
		}

		/* We do a bunch of contiguous memory transfers */
		switch(direction)
		{
			case YP :	
			case YM :	nb_block_per_slice = K*block->ldy;
						for (int iz=0; iz<block->nz; iz++) {
							cudaMemcpyAsync(&boundary_data[offset_b], &block_data[offset], nb_block_per_slice*block->elemsize, cudaMemcpyDeviceToDevice, starpu_cuda_get_local_stream());
							offset += block->ldz;
							offset_b += nb_block_per_slice;
						}
						break;
			case XP :	
			case XM :	nb_block_per_slice = K;
						for (int iz=0; iz<block->nz; iz++) {
							for (int iy=0; iy<block->ny; iy++) {
								cudaMemcpyAsync(&boundary_data[offset_b], &block_data[offset], nb_block_per_slice*block->elemsize, cudaMemcpyDeviceToDevice, starpu_cuda_get_local_stream());
								offset += block->ldy;
								offset_b += nb_block_per_slice;
							}
						}
						break;

			default : STARPU_ASSERT(0);
		}
#endif
	}
// }}}
#endif

// SAVE FUNCTS CPU {{{
	// Save veloc {{{
		/* Record how many top/bottom saves each worker performed */
		unsigned veloc_xp_per_worker[STARPU_NMAXWORKERS];
		unsigned veloc_xm_per_worker[STARPU_NMAXWORKERS];
		unsigned veloc_yp_per_worker[STARPU_NMAXWORKERS];
		unsigned veloc_ym_per_worker[STARPU_NMAXWORKERS];

		/* xp save, CPU version */
		static void save_veloc_xp_cpu(void *descr[] __attribute__((unused)), void *arg)
		{
			struct block_description *block = (struct block_description *) arg;
			int workerid = starpu_worker_get_id();
			veloc_xp_per_worker[workerid]++;

			// DEBUG( "DO SAVE veloc xp block {%d, %d} CPU %d\n", block->bx, block->by, workerid);
			clock_t start =clock();

			load_subblock_into_buffer_cpu(descr[0], descr[3], XP, block);
			load_subblock_into_buffer_cpu(descr[1], descr[4], XP, block);
			load_subblock_into_buffer_cpu(descr[2], descr[5], XP, block);
			// DEBUG( "DO SAVE veloc xp block {%d, %d} CPU %d\tDONE\n", block->bx, block->by, workerid);
			float duration = (clock()-start)/(float)(CLOCKS_PER_SEC/1000.);
			time_spent_veloc_xp[workerid] += duration;
		}

		/* xm save, CPU version */
		static void save_veloc_xm_cpu(void *descr[] __attribute__((unused)), void *arg)
		{
			struct block_description *block = (struct block_description *) arg;
			int workerid = starpu_worker_get_id();
			veloc_xm_per_worker[workerid]++;

			// DEBUG( "DO SAVE veloc xm block {%d, %d} CPU %d\n", block->bx, block->by, workerid);
			clock_t start =clock();

			load_subblock_into_buffer_cpu(descr[0], descr[3], XM, block);
			load_subblock_into_buffer_cpu(descr[1], descr[4], XM, block);
			load_subblock_into_buffer_cpu(descr[2], descr[5], XM, block);
			float duration = (clock()-start)/(float)(CLOCKS_PER_SEC/1000.);
			time_spent_veloc_xm[workerid] += duration;
			// DEBUG( "DO SAVE veloc xm block {%d, %d} CPU %d\tDONE\n", block->bx, block->by, workerid);
		}

		/* yp save, CPU version */
		static void save_veloc_yp_cpu(void *descr[] __attribute__((unused)), void *arg)
		{
			struct block_description *block = (struct block_description *) arg;
			int workerid = starpu_worker_get_id();
			veloc_yp_per_worker[workerid]++;

			// DEBUG( "DO SAVE veloc yp block {%d, %d} CPU %d\n", block->bx, block->by, workerid);
			clock_t start =clock();

			load_subblock_into_buffer_cpu(descr[0], descr[3], YP, block);
			load_subblock_into_buffer_cpu(descr[1], descr[4], YP, block);
			load_subblock_into_buffer_cpu(descr[2], descr[5], YP, block);
			float duration = (clock()-start)/(float)(CLOCKS_PER_SEC/1000.);
			time_spent_veloc_yp[workerid] += duration;
			// DEBUG( "DO SAVE veloc yp block {%d, %d} CPU %d\tDONE\n", block->bx, block->by, workerid);
		}

		/* ym save, CPU version */
		static void save_veloc_ym_cpu(void *descr[] __attribute__((unused)), void *arg)
		{
			struct block_description *block = (struct block_description *) arg;
			int workerid = starpu_worker_get_id();
			veloc_ym_per_worker[workerid]++;

			// DEBUG( "DO SAVE veloc ym block {%d, %d} CPU %d\n", block->bx, block->by, workerid);
			clock_t start =clock();

			load_subblock_into_buffer_cpu(descr[0], descr[3], YM, block);
			load_subblock_into_buffer_cpu(descr[1], descr[4], YM, block);
			load_subblock_into_buffer_cpu(descr[2], descr[5], YM, block);
			// DEBUG( "DO SAVE veloc ym block {%d, %d} CPU %d\tDONE\n", block->bx, block->by, workerid);
			float duration = (clock()-start)/(float)(CLOCKS_PER_SEC/1000.);
			time_spent_veloc_ym[workerid] += duration;
		}
	// }}}

	// Save stress {{{
		/* Record how many top/bottom saves each worker performed */
		unsigned stress_xp_per_worker[STARPU_NMAXWORKERS];
		unsigned stress_xm_per_worker[STARPU_NMAXWORKERS];
		unsigned stress_yp_per_worker[STARPU_NMAXWORKERS];
		unsigned stress_ym_per_worker[STARPU_NMAXWORKERS];

		/* xp save, CPU version */
		static void save_stress_xp_cpu(void *descr[] __attribute__((unused)), void *arg)
		{
			struct block_description *block = (struct block_description *) arg;
			int workerid = starpu_worker_get_id();
			stress_xp_per_worker[workerid]++;

			// DEBUG( "DO SAVE stress xp block {%d, %d} CPU %d\n", block->bx, block->by, workerid);
			clock_t start =clock();

			load_subblock_into_buffer_cpu(descr[0], descr[6], XP, block);
			load_subblock_into_buffer_cpu(descr[1], descr[7], XP, block);
			load_subblock_into_buffer_cpu(descr[2], descr[8], XP, block);
			load_subblock_into_buffer_cpu(descr[3], descr[9], XP, block);
			load_subblock_into_buffer_cpu(descr[4], descr[10], XP, block);
			load_subblock_into_buffer_cpu(descr[5], descr[11], XP, block);
			float duration = (clock()-start)/(float)(CLOCKS_PER_SEC/1000.);
			time_spent_stress_xp[workerid] += duration;
			// DEBUG( "DO SAVE stress xp block {%d, %d} CPU %d\tDONE\n", block->bx, block->by, workerid);
		}

		/* xm save, CPU version */
		static void save_stress_xm_cpu(void *descr[] __attribute__((unused)), void *arg)
		{
			struct block_description *block = (struct block_description *) arg;
			int workerid = starpu_worker_get_id();
			stress_xm_per_worker[workerid]++;

			// DEBUG( "DO SAVE stress xm block {%d, %d} CPU %d\n", block->bx, block->by, workerid);
			clock_t start =clock();

			load_subblock_into_buffer_cpu(descr[0], descr[6], XM, block);
			load_subblock_into_buffer_cpu(descr[1], descr[7], XM, block);
			load_subblock_into_buffer_cpu(descr[2], descr[8], XM, block);
			load_subblock_into_buffer_cpu(descr[3], descr[9], XM, block);
			load_subblock_into_buffer_cpu(descr[4], descr[10], XM, block);
			load_subblock_into_buffer_cpu(descr[5], descr[11], XM, block);
			float duration = (clock()-start)/(float)(CLOCKS_PER_SEC/1000.);
			time_spent_stress_xm[workerid] += duration;
			// DEBUG( "DO SAVE stress xm block {%d, %d} CPU %d\tDONE\n", block->bx, block->by, workerid);
		}

		/* yp save, CPU version */
		static void save_stress_yp_cpu(void *descr[] __attribute__((unused)), void *arg)
		{
			struct block_description *block = (struct block_description *) arg;
			int workerid = starpu_worker_get_id();
			stress_yp_per_worker[workerid]++;

			// DEBUG( "DO SAVE stress yp block {%d, %d} CPU %d\n", block->bx, block->by, workerid);
			clock_t start =clock();

			load_subblock_into_buffer_cpu(descr[0], descr[6], YP, block);
			load_subblock_into_buffer_cpu(descr[1], descr[7], YP, block);
			load_subblock_into_buffer_cpu(descr[2], descr[8], YP, block);
			load_subblock_into_buffer_cpu(descr[3], descr[9], YP, block);
			load_subblock_into_buffer_cpu(descr[4], descr[10], YP, block);
			load_subblock_into_buffer_cpu(descr[5], descr[11], YP, block);
			float duration = (clock()-start)/(float)(CLOCKS_PER_SEC/1000.);
			time_spent_stress_yp[workerid] += duration;
			// DEBUG( "DO SAVE stress yp block {%d, %d} CPU %d\tDONE\n", block->bx, block->by, workerid);
		}

		/* ym save, CPU version */
		static void save_stress_ym_cpu(void *descr[] __attribute__((unused)), void *arg)
		{
			struct block_description *block = (struct block_description *) arg;
			int workerid = starpu_worker_get_id();
			stress_ym_per_worker[workerid]++;

			// DEBUG( "DO SAVE stress ym block {%d, %d} CPU %d\n", block->bx, block->by, workerid);
			clock_t start =clock();

			load_subblock_into_buffer_cpu(descr[0], descr[6], YM, block);
			load_subblock_into_buffer_cpu(descr[1], descr[7], YM, block);
			load_subblock_into_buffer_cpu(descr[2], descr[8], YM, block);
			load_subblock_into_buffer_cpu(descr[3], descr[9], YM, block);
			load_subblock_into_buffer_cpu(descr[4], descr[10], YM, block);
			load_subblock_into_buffer_cpu(descr[5], descr[11], YM, block);
			float duration = (clock()-start)/(float)(CLOCKS_PER_SEC/1000.);
			time_spent_stress_ym[workerid] += duration;
			// DEBUG( "DO SAVE stress ym block {%d, %d} CPU %d\tDONE\n", block->bx, block->by, workerid);
		}
	// }}}
// }}}

#ifdef STARPU_USE_CUDA
// SAVE FUNCTS CUDA {{{
	// Save veloc {{{
		/* xp save, CUDA version */
		static void save_veloc_xp_cuda(void *descr[] __attribute__((unused)), void *arg)
		{
			struct block_description *block = (struct block_description *) arg;
			int workerid = starpu_worker_get_id();
			veloc_xp_per_worker[workerid]++;

			// DEBUG( "DO SAVE veloc xp block {%d, %d} CUDA %d\n", block->bx, block->by, workerid);
			clock_t start =clock();

			load_subblock_into_buffer_cuda(descr[0], descr[3], XP, block);
			load_subblock_into_buffer_cuda(descr[1], descr[4], XP, block);
			load_subblock_into_buffer_cuda(descr[2], descr[5], XP, block);
			cudaStreamSynchronize(starpu_cuda_get_local_stream());
			float duration = (clock()-start)/(float)(CLOCKS_PER_SEC/1000.);
			time_spent_veloc_xp[workerid] += duration;
			// DEBUG( "DO SAVE veloc xp block {%d, %d} CUDA %d\tDONE\n", block->bx, block->by, workerid);
		}

		/* xm save, CUDA version */
		static void save_veloc_xm_cuda(void *descr[] __attribute__((unused)), void *arg)
		{
			struct block_description *block = (struct block_description *) arg;
			int workerid = starpu_worker_get_id();
			veloc_xm_per_worker[workerid]++;

			// DEBUG( "DO SAVE veloc xm block {%d, %d} CUDA %d\n", block->bx, block->by, workerid);
			clock_t start =clock();
			load_subblock_into_buffer_cuda(descr[0], descr[3], XM, block);
			load_subblock_into_buffer_cuda(descr[1], descr[4], XM, block);
			load_subblock_into_buffer_cuda(descr[2], descr[5], XM, block);
			cudaStreamSynchronize(starpu_cuda_get_local_stream());
			float duration = (clock()-start)/(float)(CLOCKS_PER_SEC/1000.);
			time_spent_veloc_xm[workerid] += duration;
			// DEBUG( "DO SAVE veloc xm block {%d, %d} CUDA %d\tDONE\n", block->bx, block->by, workerid);
		}

		/* yp save, CUDA version */
		static void save_veloc_yp_cuda(void *descr[] __attribute__((unused)), void *arg)
		{
			struct block_description *block = (struct block_description *) arg;
			int workerid = starpu_worker_get_id();
			veloc_yp_per_worker[workerid]++;

			// DEBUG( "DO SAVE veloc yp block {%d, %d} CUDA %d\n", block->bx, block->by, workerid);
			clock_t start =clock();
			load_subblock_into_buffer_cuda(descr[0], descr[3], YP, block);
			load_subblock_into_buffer_cuda(descr[1], descr[4], YP, block);
			load_subblock_into_buffer_cuda(descr[2], descr[5], YP, block);
			cudaStreamSynchronize(starpu_cuda_get_local_stream());
			// DEBUG( "DO SAVE veloc yp block {%d, %d} CUDA %d\tDONE\n", block->bx, block->by, workerid);
			float duration = (clock()-start)/(float)(CLOCKS_PER_SEC/1000.);
			time_spent_veloc_yp[workerid] += duration;
		}

		/* ym save, CUDA version */
		static void save_veloc_ym_cuda(void *descr[] __attribute__((unused)), void *arg)
		{
			struct block_description *block = (struct block_description *) arg;
			int workerid = starpu_worker_get_id();
			veloc_ym_per_worker[workerid]++;
			clock_t start =clock();
			// DEBUG( "DO SAVE veloc ym block {%d, %d} CUDA %d\n", block->bx, block->by, workerid);

			load_subblock_into_buffer_cuda(descr[0], descr[3], YM, block);
			load_subblock_into_buffer_cuda(descr[1], descr[4], YM, block);
			load_subblock_into_buffer_cuda(descr[2], descr[5], YM, block);
			cudaStreamSynchronize(starpu_cuda_get_local_stream());
			// DEBUG( "DO SAVE veloc ym block {%d, %d} CUDA %d\tDONE\n", block->bx, block->by, workerid);
			float duration = (clock()-start)/(float)(CLOCKS_PER_SEC/1000.);
			time_spent_veloc_ym[workerid] += duration;
		}
	// }}}

	// Save stress {{{
		/* xp save, CUDA version */
		static void save_stress_xp_cuda(void *descr[] __attribute__((unused)), void *arg)
		{
			struct block_description *block = (struct block_description *) arg;
			int workerid = starpu_worker_get_id();
			stress_xp_per_worker[workerid]++;

			// DEBUG( "DO SAVE stress xp block {%d, %d} CUDA %d\n", block->bx, block->by, workerid);
			clock_t start =clock();

			load_subblock_into_buffer_cuda(descr[0], descr[6], XP, block);
			load_subblock_into_buffer_cuda(descr[1], descr[7], XP, block);
			load_subblock_into_buffer_cuda(descr[2], descr[8], XP, block);
			load_subblock_into_buffer_cuda(descr[3], descr[9], XP, block);
			load_subblock_into_buffer_cuda(descr[4], descr[10], XP, block);
			load_subblock_into_buffer_cuda(descr[5], descr[11], XP, block);
			cudaStreamSynchronize(starpu_cuda_get_local_stream());
			float duration = (clock()-start)/(float)(CLOCKS_PER_SEC/1000.);
			time_spent_stress_xp[workerid] += duration;
			// DEBUG( "DO SAVE stress xp block {%d, %d} CUDA %d\tDONE\n", block->bx, block->by, workerid);
		}

		/* xm save, CUDA version */
		static void save_stress_xm_cuda(void *descr[] __attribute__((unused)), void *arg)
		{
			struct block_description *block = (struct block_description *) arg;
			int workerid = starpu_worker_get_id();
			stress_xm_per_worker[workerid]++;

			// DEBUG( "DO SAVE stress xm block {%d, %d} CUDA %d\n", block->bx, block->by, workerid);
			clock_t start =clock();

			load_subblock_into_buffer_cuda(descr[0], descr[6], XM, block);
			load_subblock_into_buffer_cuda(descr[1], descr[7], XM, block);
			load_subblock_into_buffer_cuda(descr[2], descr[8], XM, block);
			load_subblock_into_buffer_cuda(descr[3], descr[9], XM, block);
			load_subblock_into_buffer_cuda(descr[4], descr[10], XM, block);
			load_subblock_into_buffer_cuda(descr[5], descr[11], XM, block);
			cudaStreamSynchronize(starpu_cuda_get_local_stream());
			float duration = (clock()-start)/(float)(CLOCKS_PER_SEC/1000.);
			time_spent_stress_xm[workerid] += duration;
			// DEBUG( "DO SAVE stress xm block {%d, %d} CUDA %d\tDONE\n", block->bx, block->by, workerid);
		}

		/* yp save, CUDA version */
		static void save_stress_yp_cuda(void *descr[] __attribute__((unused)), void *arg)
		{
			struct block_description *block = (struct block_description *) arg;
			int workerid = starpu_worker_get_id();
			stress_yp_per_worker[workerid]++;

			// DEBUG( "DO SAVE stress yp block {%d, %d} CUDA %d\n", block->bx, block->by, workerid);
			clock_t start =clock();

			load_subblock_into_buffer_cuda(descr[0], descr[6], YP, block);
			load_subblock_into_buffer_cuda(descr[1], descr[7], YP, block);
			load_subblock_into_buffer_cuda(descr[2], descr[8], YP, block);
			load_subblock_into_buffer_cuda(descr[3], descr[9], YP, block);
			load_subblock_into_buffer_cuda(descr[4], descr[10], YP, block);
			load_subblock_into_buffer_cuda(descr[5], descr[11], YP, block);
			cudaStreamSynchronize(starpu_cuda_get_local_stream());
			float duration = (clock()-start)/(float)(CLOCKS_PER_SEC/1000.);
			time_spent_stress_yp[workerid] += duration;
			// DEBUG( "DO SAVE stress yp block {%d, %d} CUDA %d\tDONE\n", block->bx, block->by, workerid);
		}

		/* ym save, CUDA version */
		static void save_stress_ym_cuda(void *descr[] __attribute__((unused)), void *arg)
		{
			struct block_description *block = (struct block_description *) arg;
			int workerid = starpu_worker_get_id();
			stress_ym_per_worker[workerid]++;

			// DEBUG( "DO SAVE stress ym block {%d, %d} CUDA %d\n", block->bx, block->by, workerid);
			clock_t start =clock();

			load_subblock_into_buffer_cuda(descr[0], descr[6], YM, block);
			load_subblock_into_buffer_cuda(descr[1], descr[7], YM, block);
			load_subblock_into_buffer_cuda(descr[2], descr[8], YM, block);
			load_subblock_into_buffer_cuda(descr[3], descr[9], YM, block);
			load_subblock_into_buffer_cuda(descr[4], descr[10], YM, block);
			load_subblock_into_buffer_cuda(descr[5], descr[11], YM, block);
			cudaStreamSynchronize(starpu_cuda_get_local_stream());
			// DEBUG( "DO SAVE stress ym block {%d, %d} CUDA %d\tDONE\n", block->bx, block->by, workerid);
			float duration = (clock()-start)/(float)(CLOCKS_PER_SEC/1000.);
			time_spent_stress_ym[workerid] += duration;
		}
	// }}}
// }}}
#endif /*STARPU_USE_CUDA*/


// Performance models and codelets for save {{{
	static struct starpu_perfmodel save_veloc_cl_xp_model =
	{
		.type = STARPU_HISTORY_BASED,
		.symbol = "save_veloc_xp_cl" 
	};

	static struct starpu_perfmodel save_veloc_cl_xm_model =
	{
		.type = STARPU_HISTORY_BASED,
		.symbol = "save_veloc_xm_cl" 
	};

	static struct starpu_perfmodel save_veloc_cl_yp_model =
	{
		.type = STARPU_HISTORY_BASED,
		.symbol = "save_veloc_yp_cl" 
	};

	static struct starpu_perfmodel save_veloc_cl_ym_model =
	{
		.type = STARPU_HISTORY_BASED,
		.symbol = "save_veloc_ym_cl" 
	};

	static struct starpu_perfmodel save_stress_cl_xp_model =
	{
		.type = STARPU_HISTORY_BASED,
		.symbol = "save_stress_xp_cl" 
	};

	static struct starpu_perfmodel save_stress_cl_xm_model =
	{
		.type = STARPU_HISTORY_BASED,
		.symbol = "save_stress_xm_cl" 
	};

	static struct starpu_perfmodel save_stress_cl_yp_model =
	{
		.type = STARPU_HISTORY_BASED,
		.symbol = "save_stress_yp_cl" 
	};

	static struct starpu_perfmodel save_stress_cl_ym_model =
	{
		.type = STARPU_HISTORY_BASED,
		.symbol = "save_stress_ym_cl" 
	};


	struct starpu_codelet save_veloc_xp_cl =
	{
		.where = 0 |
	#ifdef STARPU_USE_CUDA
			STARPU_CUDA|
	#endif
			STARPU_CPU,
		.cpu_funcs = {save_veloc_xp_cpu, NULL},
	#ifdef STARPU_USE_CUDA
		.cuda_funcs = {save_veloc_xp_cuda, NULL},
	#endif
		.model = &save_veloc_cl_xp_model,
		.nbuffers = 6,
		.modes = {STARPU_R, STARPU_R, STARPU_R, STARPU_W, STARPU_W, STARPU_W}
	};

	struct starpu_codelet save_veloc_xm_cl =
	{
		.where = 0 |
	#ifdef STARPU_USE_CUDA
			STARPU_CUDA|
	#endif
			STARPU_CPU,
		.cpu_funcs = {save_veloc_xm_cpu, NULL},
	#ifdef STARPU_USE_CUDA
		.cuda_funcs = {save_veloc_xm_cuda, NULL},
	#endif
		.model = &save_veloc_cl_xm_model,
		.nbuffers = 6,
		.modes = {STARPU_R, STARPU_R, STARPU_R, STARPU_W, STARPU_W, STARPU_W}
	};

	struct starpu_codelet save_veloc_yp_cl =
	{
		.where = 0 |
	#ifdef STARPU_USE_CUDA
			STARPU_CUDA|
	#endif
			STARPU_CPU,
		.cpu_funcs = {save_veloc_yp_cpu, NULL},
	#ifdef STARPU_USE_CUDA
		.cuda_funcs = {save_veloc_yp_cuda, NULL},
	#endif
		.model = &save_veloc_cl_yp_model,
		.nbuffers = 6,
		.modes = {STARPU_R, STARPU_R, STARPU_R, STARPU_W, STARPU_W, STARPU_W}
	};

	struct starpu_codelet save_veloc_ym_cl =
	{
		.where = 0 |
	#ifdef STARPU_USE_CUDA
			STARPU_CUDA|
	#endif
			STARPU_CPU,
		.cpu_funcs = {save_veloc_ym_cpu, NULL},
	#ifdef STARPU_USE_CUDA
		.cuda_funcs = {save_veloc_ym_cuda, NULL},
	#endif
		.model = &save_veloc_cl_ym_model,
		.nbuffers = 6,
		.modes = {STARPU_R, STARPU_R, STARPU_R, STARPU_W, STARPU_W, STARPU_W}
	};


	struct starpu_codelet save_stress_xp_cl =
	{
		.where = 0 |
	#ifdef STARPU_USE_CUDA
			STARPU_CUDA|
	#endif
			STARPU_CPU,
		.cpu_funcs = {save_stress_xp_cpu, NULL},
	#ifdef STARPU_USE_CUDA
		.cuda_funcs = {save_stress_xp_cuda, NULL},
	#endif
		.model = &save_stress_cl_xp_model,
		.nbuffers = 12,
		.modes = {	STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, 
					STARPU_W, STARPU_W, STARPU_W, STARPU_W, STARPU_W, STARPU_W  }
	};

	struct starpu_codelet save_stress_xm_cl =
	{
		.where = 0 |
	#ifdef STARPU_USE_CUDA
			STARPU_CUDA|
	#endif
			STARPU_CPU,
		.cpu_funcs = {save_stress_xm_cpu, NULL},
	#ifdef STARPU_USE_CUDA
		.cuda_funcs = {save_stress_xm_cuda, NULL},
	#endif
		.model = &save_stress_cl_xm_model,
		.nbuffers = 12,
		.modes = {	STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, 
					STARPU_W, STARPU_W, STARPU_W, STARPU_W, STARPU_W, STARPU_W  }
	};

	struct starpu_codelet save_stress_yp_cl =
	{
		.where = 0 |
	#ifdef STARPU_USE_CUDA
			STARPU_CUDA|
	#endif
			STARPU_CPU,
		.cpu_funcs = {save_stress_yp_cpu, NULL},
	#ifdef STARPU_USE_CUDA
		.cuda_funcs = {save_stress_yp_cuda, NULL},
	#endif
		.model = &save_stress_cl_yp_model,
		.nbuffers = 12,
		.modes = {	STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, 
					STARPU_W, STARPU_W, STARPU_W, STARPU_W, STARPU_W, STARPU_W  }
	};

	struct starpu_codelet save_stress_ym_cl =
	{
		.where = 0 |
	#ifdef STARPU_USE_CUDA
			STARPU_CUDA|
	#endif
			STARPU_CPU,
		.cpu_funcs = {save_stress_ym_cpu, NULL},
	#ifdef STARPU_USE_CUDA
		.cuda_funcs = {save_stress_ym_cuda, NULL},
	#endif
		.model = &save_stress_cl_ym_model,
		.nbuffers = 12,
		.modes = {	STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, 
					STARPU_W, STARPU_W, STARPU_W, STARPU_W, STARPU_W, STARPU_W  }
	};
// }}}

// Sources update {{{
	static void update_sources_cpu(void *descr[], void *arg_)
	{
					DGN_DBG
		block_iter_arg_t* arg = (block_iter_arg_t*) arg_;
		struct block_description* block = arg->block;
		unsigned iter = arg->iter;

		// first ietration (increment is in compute_veloc because all chunks do'nt have sources)
		if (iter<0) iter=0;


		int workerid = starpu_worker_get_id();
		
	// 	if (block->bx == 0 && block->by == 0)
	// fprintf(stderr,"!!! DO update_sources_cpu block{%d, %d} CPU%d !!!\n", block->bx, block->by, workerid);
	// 	else
		INFO("!!! DO update_sources_cpu it=%d, block{%d, %d} CPU%d !!!\n", block->iter, block->bx, block->by, workerid);
	#ifdef STARPU_USE_MPI
		int rank = 0;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		DEBUG( "!!!                 RANK %d                    !!!\n", rank);
	#endif
		

	/*		record_who_runs_what(block);*/

		// get Data pointers
		struct starpu_block_interface *b_fx = (struct starpu_block_interface *)descr[0];
		struct starpu_block_interface *b_fy = (struct starpu_block_interface *)descr[1];
		struct starpu_block_interface *b_fz = (struct starpu_block_interface *)descr[2];

		float *fx = (float *)b_fx->ptr;
		float *fy = (float *)b_fy->ptr;
		float *fz = (float *)b_fz->ptr;

		ondes3d_params* par = block->params;

      // INCREMENT SEISMIC MOMENT WITH MPI COMMS {{{
      if ( iter < par->idur ){
         int imin = INT_MAX;
         int imax = INT_MIN;
         int jmin = INT_MAX;
         int jmax = INT_MIN;
         int kmin = INT_MAX;
         int kmax = INT_MIN;

         float mo, weight, pxx, pyy, pzz, pxy, pyz, pxz;
         int i,j,k;

         for (int is = 0; is < par->nb_src; is++ ){/*{{{*/
            if ( block->insrc[is] ){

               mo = par->vel[is*par->idur+iter] * par->dt;
               pxx = (float)dradxx((double)par->strike[is], (double)par->dip[is], (double)par->rake[is]);
               pyy = (float)dradyy((double)par->strike[is], (double)par->dip[is], (double)par->rake[is]);
               pzz = (float)dradzz((double)par->strike[is], (double)par->dip[is], (double)par->rake[is]);
               pxy = (float)dradxy((double)par->strike[is], (double)par->dip[is], (double)par->rake[is]);
               pyz = (float)dradyz((double)par->strike[is], (double)par->dip[is], (double)par->rake[is]);
               pxz = (float)dradxz((double)par->strike[is], (double)par->dip[is], (double)par->rake[is]);

               // ix, iy hypo : sûrement des décalages à faire !!! global->local
               for ( unsigned iw = 0; iw < 8; iw++ ){
                  weight = 1.0;
                  if (  (iw%2) == 0 ){
                     i = par->ixhypo[is] - block->xshift;
                     weight = (1.0 - par->xweight[is]);
                  } else {
                     i = par->ixhypo[is] + 1 - block->xshift;
                     weight = par->xweight[is];
                  }
                  if ( (iw%4) <= 1 ){
                     j = par->iyhypo[is] - block->yshift;
                     weight = weight*(1.0 - par->yweight[is]);
                  } else {
                     j = par->iyhypo[is] + 1  - block->yshift;
                     weight = weight*par->yweight[is];
                  }
                  if ( iw < 4 ){
                     k = par->izhypo[is];
                     weight = weight*(1.0 - par->zweight[is]);
                  } else {
                     k = par->izhypo[is] + 1;
                     weight = weight*par->zweight[is];
                  }

                  // DGN Debug : a virer
                  if (i<-2 || j<-2 || k<-2) {
							fprintf(stderr, "Error : bad index for source %d : %d,%d,%d\n",is,i,j,k); fflush(stderr);
							STARPU_ASSERT(0);
                  }

                  // update source array
                  fx[ACCESS(block,i+1,j,k)] += 0.5 * mo * pxx * weight;
                  fx[ACCESS(block,i-1,j,k)] -= 0.5 * mo * pxx * weight;

                  fx[ACCESS(block,i,j+1,k)] += 0.5 * mo * pxy * weight;
                  fx[ACCESS(block,i,j-1,k)] -= 0.5 * mo * pxy * weight;

                  fx[ACCESS(block,i,j,k+1)] += 0.5 * mo * pxz * weight;
                  fx[ACCESS(block,i,j,k-1)] -= 0.5 * mo * pxz * weight;

                  fy[ACCESS(block,i,j,k)]     += 0.5 * mo * pxy * weight;
                  fy[ACCESS(block,i,j-1,k)]   += 0.5 * mo * pxy * weight;
                  fy[ACCESS(block,i-1,j,k)]   -= 0.5 * mo * pxy * weight;
                  fy[ACCESS(block,i-1,j-1,k)] -= 0.5 * mo * pxy * weight;

                  fy[ACCESS(block,i,j,k)]     += 0.5 * mo * pyy * weight;
                  fy[ACCESS(block,i-1,j,k)]   += 0.5 * mo * pyy * weight;
                  fy[ACCESS(block,i,j-1,k)]   -= 0.5 * mo * pyy * weight;
                  fy[ACCESS(block,i-1,j-1,k)] -= 0.5 * mo * pyy * weight;

                  fy[ACCESS(block,i,j,k+1)]     += 0.125 * mo * pyz * weight;
                  fy[ACCESS(block,i,j-1,k+1)]   += 0.125 * mo * pyz * weight;
                  fy[ACCESS(block,i-1,j,k+1)]   += 0.125 * mo * pyz * weight;
                  fy[ACCESS(block,i-1,j-1,k+1)] += 0.125 * mo * pyz * weight;
                  fy[ACCESS(block,i,j,k-1)]     -= 0.125 * mo * pyz * weight;
                  fy[ACCESS(block,i,j-1,k-1)]   -= 0.125 * mo * pyz * weight;
                  fy[ACCESS(block,i-1,j,k-1)]   -= 0.125 * mo * pyz * weight;
                  fy[ACCESS(block,i-1,j-1,k-1)] -= 0.125 * mo * pyz * weight;

                  fz[ACCESS(block,i,j,k)]     += 0.5 * mo * pxz * weight;
                  fz[ACCESS(block,i,j,k-1)]   += 0.5 * mo * pxz * weight;
                  fz[ACCESS(block,i-1,j,k)]   -= 0.5 * mo * pxz * weight;
                  fz[ACCESS(block,i-1,j,k-1)] -= 0.5 * mo * pxz * weight;

                  fz[ACCESS(block,i,j+1,k)]     += 0.125 * mo * pyz * weight;
                  fz[ACCESS(block,i,j+1,k-1)]   += 0.125 * mo * pyz * weight;
                  fz[ACCESS(block,i-1,j+1,k)]   += 0.125 * mo * pyz * weight;
                  fz[ACCESS(block,i-1,j+1,k-1)] += 0.125 * mo * pyz * weight;
                  fz[ACCESS(block,i,j-1,k)]     -= 0.125 * mo * pyz * weight;
                  fz[ACCESS(block,i,j-1,k-1)]   -= 0.125 * mo * pyz * weight;
                  fz[ACCESS(block,i-1,j-1,k)]   -= 0.125 * mo * pyz * weight;
                  fz[ACCESS(block,i-1,j-1,k-1)] -= 0.125 * mo * pyz * weight;

                  fz[ACCESS(block,i,j,k)]     += 0.5 * mo * pzz * weight;
                  fz[ACCESS(block,i-1,j,k)]   += 0.5 * mo * pzz * weight;
                  fz[ACCESS(block,i,j,k-1)]   -= 0.5 * mo * pzz * weight;
                  fz[ACCESS(block,i-1,j,k-1)] -= 0.5 * mo * pzz * weight;
               } /* end of iw (weighting) */
            } /* end of insrc */
         } /* end of is (each source) *//*}}}*/
      }
      free(arg);
					DGN_DBG
	}

	static struct starpu_perfmodel cl_update_source_model =
	{
		.type = STARPU_HISTORY_BASED,
		.symbol = "cl_update_source" 
	};

	struct starpu_codelet cl_update_source =
	{
		.where = 0 |
	/*#ifdef STARPU_USE_CUDA
			STARPU_CUDA|
	#endif*/
			STARPU_CPU,
		.cpu_funcs = {update_sources_cpu, NULL},
	/*#ifdef STARPU_USE_CUDA
		.cuda_funcs = {update_sources_cuda, NULL},
	#endif
	*/	.model = &cl_update_source_model,
		.nbuffers = 3,
		.modes = {STARPU_RW, STARPU_RW, STARPU_RW}
	};
// }}}

int test_access(int i, int j, int k,struct block_description* block)
{
	unsigned maxsize = block->pitchx * block->pitchy * block->pitchz;

	if (ACCESS(block,i,j,k) > maxsize || ACCESS(block,i,j,k)<0 || ACCESS(block,i+1,j+1,k+1) > maxsize || ACCESS(block,i+1,j+1,k+1)<0) {
		INFO("ijk : %d,%d,%d\nsizex: %d\nsizey: %d\nsizez: %d\n", i,j,k,block->sizex, block->sizey, block->sizez);


		return 0;	
	}

	return 1;
}


// Seismo record {{{
	void record_seismo_cpu(void *descr[], void *arg_)
	{

		block_iter_arg_t* arg = (block_iter_arg_t*) arg_;
		struct block_description* block = arg->block;
		unsigned iter = arg->iter;
		free(arg);

		int workerid = starpu_worker_get_id();
		// 
	// 	if (block->bx == 0 && block->by == 0)
	// fprintf(stderr,"!!! DO record_seismo_cpu block{%d, %d} CPU%d !!!\n", block->bx, block->by, workerid);
	// 	else

		// DGN ici : acces à block

		INFO("!!! DO record_seismo_cpu it=%d, block{%d, %d} CPU%d !!!\n", block->iter, block->bx, block->by, workerid);
		// INFO("!!! DO record_seismo_cpu\n");
	#ifdef STARPU_USE_MPI
		int rank = 0;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		DEBUG( "!!!                 RANK %d                    !!!\n", rank);
	#endif
		

		if (iter < get_niter()) {

			/*XP*/
			if (block->bx < get_nbx()-1) {
				// DEBUG("load_subblock_from_buffer_cpu XP\n");
				load_subblock_from_buffer_cpu(descr[3], descr[9], XP, block);
				load_subblock_from_buffer_cpu(descr[4], descr[10], XP, block);
				load_subblock_from_buffer_cpu(descr[5], descr[11], XP, block);
				load_subblock_from_buffer_cpu(descr[6], descr[12], XP, block);
				load_subblock_from_buffer_cpu(descr[7], descr[13], XP, block);
				load_subblock_from_buffer_cpu(descr[8], descr[14], XP, block);
			}

			 /*XM*/
			if (block->bx > 0) {
				// DEBUG("load_subblock_from_buffer_cpu XM\n");
				load_subblock_from_buffer_cpu(descr[3], descr[15], XM, block);
				load_subblock_from_buffer_cpu(descr[4], descr[16], XM, block);
				load_subblock_from_buffer_cpu(descr[5], descr[17], XM, block);
				load_subblock_from_buffer_cpu(descr[6], descr[18], XM, block);
				load_subblock_from_buffer_cpu(descr[7], descr[19], XM, block);
				load_subblock_from_buffer_cpu(descr[8], descr[20], XM, block);
			}

			 /*YP*/
			if (block->by < get_nby()-1) {
				// DEBUG("load_subblock_from_buffer_cpu YP\n");
				load_subblock_from_buffer_cpu(descr[3], descr[21], YP, block);
				load_subblock_from_buffer_cpu(descr[4], descr[22], YP, block);
				load_subblock_from_buffer_cpu(descr[5], descr[23], YP, block);
				load_subblock_from_buffer_cpu(descr[6], descr[24], YP, block);
				load_subblock_from_buffer_cpu(descr[7], descr[25], YP, block);
				load_subblock_from_buffer_cpu(descr[8], descr[26], YP, block);
			}
			
			/*YM*/
			if (block->by > 0) {
				// DEBUG("load_subblock_from_buffer_cpu YM\n");
				load_subblock_from_buffer_cpu(descr[3], descr[27], YM, block);
				load_subblock_from_buffer_cpu(descr[4], descr[28], YM, block);
				load_subblock_from_buffer_cpu(descr[5], descr[29], YM, block);
				load_subblock_from_buffer_cpu(descr[6], descr[30], YM, block);
				load_subblock_from_buffer_cpu(descr[7], descr[31], YM, block);
				load_subblock_from_buffer_cpu(descr[8], descr[32], YM, block);
			}

			// get Data pointers
			struct starpu_block_interface *b_vx = (struct starpu_block_interface *)descr[0];
			struct starpu_block_interface *b_vy = (struct starpu_block_interface *)descr[1];
			struct starpu_block_interface *b_vz = (struct starpu_block_interface *)descr[2];

			struct starpu_block_interface *b_txx = (struct starpu_block_interface *)descr[3];
			struct starpu_block_interface *b_tyy = (struct starpu_block_interface *)descr[4];
			struct starpu_block_interface *b_tzz = (struct starpu_block_interface *)descr[5];
			struct starpu_block_interface *b_txy = (struct starpu_block_interface *)descr[6];
			struct starpu_block_interface *b_txz = (struct starpu_block_interface *)descr[7];
			struct starpu_block_interface *b_tyz = (struct starpu_block_interface *)descr[8];

			struct starpu_block_interface *b_seisx = (struct starpu_block_interface *)descr[33];
			struct starpu_block_interface *b_seisy = (struct starpu_block_interface *)descr[34];
			struct starpu_block_interface *b_seisz = (struct starpu_block_interface *)descr[35];

			struct starpu_block_interface *b_seisxx = (struct starpu_block_interface *)descr[36];
			struct starpu_block_interface *b_seisyy = (struct starpu_block_interface *)descr[37];
			struct starpu_block_interface *b_seiszz = (struct starpu_block_interface *)descr[38];
			struct starpu_block_interface *b_seisxy = (struct starpu_block_interface *)descr[39];
			struct starpu_block_interface *b_seisxz = (struct starpu_block_interface *)descr[40];
			struct starpu_block_interface *b_seisyz = (struct starpu_block_interface *)descr[41];


			float *vx = (float *)b_vx->ptr;
			float *vy = (float *)b_vy->ptr;
			float *vz = (float *)b_vz->ptr;

			float *txx = (float *)b_txx->ptr;
			float *tyy = (float *)b_tyy->ptr;
			float *tzz = (float *)b_tzz->ptr;
			float *txy = (float *)b_txy->ptr;
			float *txz = (float *)b_txz->ptr;
			float *tyz = (float *)b_tyz->ptr;

			float *seisx = (float *)b_seisx->ptr;
			float *seisy = (float *)b_seisy->ptr;
			float *seisz = (float *)b_seisz->ptr;

			float *seisxx = (float *)b_seisxx->ptr;
			float *seisyy = (float *)b_seisyy->ptr;
			float *seiszz = (float *)b_seiszz->ptr;
			float *seisxy = (float *)b_seisxy->ptr;
			float *seisxz = (float *)b_seisxz->ptr;
			float *seisyz = (float *)b_seisyz->ptr;

			ondes3d_params* par = block->params;


			// CALCULATION OF THE SEISMOGRAMS
			unsigned loc_sta = -1;
			float w1, w2, w3;
			int i,j,k;

			for ( int ir = 0; ir < par->iobs; ir++ ){
				if(block->ista[ir]){

					loc_sta++;
					// DGN : ATTENTION, FAUX !!!
					// DGN : convertir ça en indices locaux !!!
					/* Vx component */
					i = par->ixobs[ir]  - block->xshift;
					w1 = par->xobswt[ir];
					j = par->iyobs[ir]  - block->yshift;
					w2 = par->yobswt[ir];
					k = par->izobs[ir];
					w3 = par->zobswt[ir];

					// seisx[ir][l-1] = (1-w3)*(
					STARPU_ASSERT(loc_sta<block->nb_sta);


					unsigned index = iter*block->nb_sta+loc_sta;

					// INFO("i=%d, j=%d, k=%d\n",i,j,k);			
					STARPU_ASSERT(test_access(i,j,k,block));
					seisx[index] =	(1-w3)*(
																	(1-w2)*( (1-w1)*vx[ACCESS(block,i,j,k+0)]     + w1*vx[ACCESS(block,i,j,k+1)] )
																	+ w2*( (1-w1)*vx[ACCESS(block,i,j+1,k+0)]   + w1*vx[ACCESS(block,i,j+1,k+1)] ) )
																	+ w3*( (1-w2)*( (1-w1)*vx[ACCESS(block,i+1,j,k+0)]   + w1*vx[ACCESS(block,i+1,j,k+1)] )
																	+ w2*( (1-w1)*vx[ACCESS(block,i+1,j+1,k+0)] + w1*vx[ACCESS(block,i+1,j+1,k+1)] ) );
					/* Vy component */
					if(par->xobswt[ir] >= 0.5){
						w1 = par->xobswt[ir] - 0.5;
						i = par->ixobs[ir] - block->xshift;
					} else {
						w1 = par->xobswt[ir] + 0.5;
						i = par->ixobs[ir]-1 - block->xshift;
					}
					if(par->yobswt[ir] >= 0.5){
						w2 = par->yobswt[ir] - 0.5;
						j = par->iyobs[ir]  - block->yshift;
					} else {
						w2 = par->yobswt[ir] + 0.5;
						j = par->iyobs[ir]-1 - block->yshift;
					}
					k = par->izobs[ir];
					w3 = par->zobswt[ir];
		
					// STARPU_ASSERT(i>=0 && i<block->sizex-1);
					// STARPU_ASSERT(j>=0 && j<block->sizey-1);
					// STARPU_ASSERT(k>=0 && k<block->sizez-1);

					seisy[index] =	(1-w3)*(
											(1-w1)*( (1-w2)*vy[ACCESS(block,i,j,k)]     + w2*vy[ACCESS(block,i,j+1,k)])
											+ w1*( (1-w2)*vy[ACCESS(block,i,j,k+1)]   + w2*vy[ACCESS(block,i,j+1,k+1)] ) )
											+ w3*( (1-w1)*( (1-w2)*vy[ACCESS(block,i+1,j,k)]   + w2*vy[ACCESS(block,i+1,j+1,k)] )
											+ w1*( (1-w2)*vy[ACCESS(block,i+1,j,k+1)] + w2*vy[ACCESS(block,i+1,j+1,k+1)] ) );

					/* Vz component */
					if(par->xobswt[ir] >= 0.5){
						w1 = par->xobswt[ir] - 0.5;
						i = par->ixobs[ir]  - block->xshift;
					} else {
						w1 = par->xobswt[ir] + 0.5;
						i = par->ixobs[ir]-1 - block->xshift;
					}
					j = par->iyobs[ir] - block->yshift;
					w2 = par->yobswt[ir];
					if(par->zobswt[ir] >= 0.5){
						w3 = par->zobswt[ir] - 0.5;
						k = par->izobs[ir];
					} else {
						w3 = par->zobswt[ir] + 0.5;
						k = par->izobs[ir]-1;
					}
					if( par->izobs[ir] == 1 ){
						w3 = 0.0;
						k = 0;
					}
					// STARPU_ASSERT(i>=0 && i<block->sizex-1);
					// STARPU_ASSERT(j>=0 && j<block->sizey-1);
					// STARPU_ASSERT(k>=0 && k<block->sizez-1);

					w1 = w2 = w3 = 0.;

					seisz[index] =	(1-w3)*(
											(1-w1)*( (1-w2)*vz[ACCESS(block,i,j,k)]     + w2*vz[ACCESS(block,i,j+1,k)] )
											+ w1*( (1-w2)*vz[ACCESS(block,i,j,k+1)]   + w2*vz[ACCESS(block,i,j+1,k+1)] ) )
											+ w3*( (1-w1)*( (1-w2)*vz[ACCESS(block,i+1,j,k)]   + w2*vz[ACCESS(block,i+1,j+1,k)] )
											+ w1*( (1-w2)*vz[ACCESS(block,i+1,j,k+1)] + w2*vz[ACCESS(block,i+1,j+1,k+1)] ) );

	#if 0
					/* Tii component */

					if(par->xobswt[ir] >= 0.5){
						w1 = par->xobswt[ir] - 0.5;
						i = par->ixobs[ir] - block->xshift;
					} else {
						w1 = par->xobswt[ir] + 0.5;
						i = par->ixobs[ir]-1 - block->xshift;
					}
					j = par->iyobs[ir] - block->yshift;
					w2 = par->yobswt[ir];
					k = par->izobs[ir];
					w3 = par->zobswt[ir];
					STARPU_ASSERT(i>=0 && i<block->sizex);
					STARPU_ASSERT(j>=0 && j<block->sizey);
					STARPU_ASSERT(k>=0 && k<block->sizez);

					seisxx[index] = (1-w3)*(
							(1-w1)*( (1-w2)*txx[ACCESS(block,i,j,k)]     + w2*txx[ACCESS(block,i,j+1,k)])
							+ w1*( (1-w2)*txx[ACCESS(block,i,j,k+1)]   + w2*txx[ACCESS(block,i,j+1,k+1)] ) )
						+ w3*( (1-w1)*( (1-w2)*txx[ACCESS(block,i+1,j,k)]   + w2*txx[ACCESS(block,i+1,j+1,k)] )
								+ w1*( (1-w2)*txx[ACCESS(block,i+1,j,k+1)] + w2*txx[ACCESS(block,i+1,j+1,k+1)] ) );
					seisyy[index] = (1-w3)*(
							(1-w1)*( (1-w2)*tyy[ACCESS(block,i,j,k)]     + w2*tyy[ACCESS(block,i,j+1,k)] )
							+ w1*( (1-w2)*tyy[ACCESS(block,i,j,k+1)]   + w2*tyy[ACCESS(block,i,j+1,k+1)] ) )
						+ w3*( (1-w1)*( (1-w2)*tyy[ACCESS(block,i+1,j,k)]   + w2*tyy[ACCESS(block,i+1,j+1,k)] )
								+ w1*( (1-w2)*tyy[ACCESS(block,i+1,j,k+1)] + w2*tyy[ACCESS(block,i+1,j+1,k+1)] ) );
					seiszz[index] = (1-w3)*(
							(1-w1)*( (1-w2)*tzz[ACCESS(block,i,j,k)]     + w2*tzz[ACCESS(block,i,j+1,k)] )
							+ w1*( (1-w2)*tzz[ACCESS(block,i,j,k+1)]   + w2*tzz[ACCESS(block,i,j+1,k+1)] ) )
						+ w3*( (1-w1)*( (1-w2)*tzz[ACCESS(block,i+1,j,k)]   + w2*tzz[ACCESS(block,i+1,j+1,k)] )
								+ w1*( (1-w2)*tzz[ACCESS(block,i+1,j,k+1)] + w2*tzz[ACCESS(block,i+1,j+1,k+1)] ) );

					/* Txy component */
					i = par->ixobs[ir] - block->xshift;
					w1 = par->xobswt[ir];
					if(par->yobswt[ir] >= 0.5){
						w2 = par->yobswt[ir] - 0.5;
						j = par->iyobs[ir] - block->yshift;
					} else {
						w2 = par->yobswt[ir] + 0.5;
						j = par->iyobs[ir]-1 - block->yshift;
					}
					k = par->izobs[ir];
					w3 = par->zobswt[ir];
					STARPU_ASSERT(i>=0 && i<block->sizex);
					STARPU_ASSERT(j>=0 && j<block->sizey);
					STARPU_ASSERT(k>=0 && k<block->sizez);

					seisxy[index] = (1-w3)*(
							(1-w1)*( (1-w2)*txy[ACCESS(block,i,j,k)]     + w2*txy[ACCESS(block,i,j+1,k)] )
							+ w1*( (1-w2)*txy[ACCESS(block,i,j,k+1)]   + w2*txy[ACCESS(block,i,j+1,k+1)] ) )
						+ w3*( (1-w1)*( (1-w2)*txy[ACCESS(block,i+1,j,k)]   + w2*txy[ACCESS(block,i+1,j+1,k)] )
								+ w1*( (1-w2)*txy[ACCESS(block,i+1,j,k+1)] + w2*txy[ACCESS(block,i+1,j+1,k+1)] ) );

					/* Txz component */
					i = par->ixobs[ir] - block->xshift;
					w1 = par->xobswt[ir];
					j = par->iyobs[ir] - block->yshift;
					w2 = par->yobswt[ir];
					if(par->zobswt[ir] >= 0.5){
						w3 = par->zobswt[ir] - 0.5;
						k = par->izobs[ir];
					} else {
						w3 = par->zobswt[ir] + 0.5;
						k = par->izobs[ir]-1;
					}
					if( par->izobs[ir] == 1 ){
						w3 = 0.0;
						k = 0;
					}
					STARPU_ASSERT(i>=0 && i<block->sizex);
					STARPU_ASSERT(j>=0 && j<block->sizey);
					STARPU_ASSERT(k>=0 && k<block->sizez);

					seisxz[index] = (1-w3)*(
							(1-w1)*( (1-w2)*txz[ACCESS(block,i,j,k)]     + w2*txz[ACCESS(block,i,j+1,k)] )
							+ w1*( (1-w2)*txz[ACCESS(block,i,j,k+1)]   + w2*txz[ACCESS(block,i,j+1,k+1)] ) )
						+ w3*( (1-w1)*( (1-w2)*txz[ACCESS(block,i+1,j,k)]   + w2*txz[ACCESS(block,i+1,j+1,k)] )
								+ w1*( (1-w2)*txz[ACCESS(block,i+1,j,k+1)] + w2*txz[ACCESS(block,i+1,j+1,k+1)] ) );

					/* Tyz component */
					if(par->xobswt[ir] >= 0.5){
						w1 = par->xobswt[ir] - 0.5;
						i = par->ixobs[ir] - block->xshift;
					} else {
						w1 = par->xobswt[ir] + 0.5;
						i = par->ixobs[ir]-1 - block->xshift;
					}
					if(par->yobswt[ir] >= 0.5){
						w2 = par->yobswt[ir] - 0.5;
						j = par->iyobs[ir] - block->yshift;
					} else {
						w2 = par->yobswt[ir] + 0.5;
						j = par->iyobs[ir]-1 - block->yshift;
					}
					if(par->zobswt[ir] >= 0.5){
						w3 = par->zobswt[ir] - 0.5;
						k = par->izobs[ir];
					} else {
						w3 = par->zobswt[ir] + 0.5;
						k = par->izobs[ir]-1;
					}
					if( par->izobs[ir] == 1 ){
						w3 = 0.0;
						k = 0;
					}
					STARPU_ASSERT(i>=0 && i<block->sizex);
					STARPU_ASSERT(j>=0 && j<block->sizey);
					STARPU_ASSERT(k>=0 && k<block->sizez);

					seisyz[index] = (1-w3)*(
							(1-w1)*( (1-w2)*tyz[ACCESS(block,i,j,k)]     + w2*tyz[ACCESS(block,i,j+1,k)] )
							+ w1*( (1-w2)*tyz[ACCESS(block,i,j,k+1)]   + w2*tyz[ACCESS(block,i,j+1,k+1)] ) )
						+ w3*( (1-w1)*( (1-w2)*tyz[ACCESS(block,i+1,j,k)]   + w2*tyz[ACCESS(block,i+1,j+1,k)] )
								+ w1*( (1-w2)*tyz[ACCESS(block,i+1,j,k+1)] + w2*tyz[ACCESS(block,i+1,j+1,k+1)] ) );
	#endif
				}
			}
		}
	}

	static struct starpu_perfmodel cl_record_seismo_model =
	{
		.type = STARPU_HISTORY_BASED,
		.symbol = "cl_record_seismo" 
	};

	struct starpu_codelet cl_record_seismo =
	{
		.where = 0 |
	/*#ifdef STARPU_USE_CUDA
			STARPU_CUDA|
	#endif*/
			STARPU_CPU,
		.cpu_funcs = {record_seismo_cpu, NULL},
	/*#ifdef STARPU_USE_CUDA
		.cuda_funcs = {record_seismo_cuda, NULL},
	#endif
	*/	.model = &cl_record_seismo_model,
		.nbuffers = 42,
		.modes = {	STARPU_R, STARPU_R, STARPU_R, 
						STARPU_RW, STARPU_RW, STARPU_RW, STARPU_RW, STARPU_RW, STARPU_RW, 
						/*DGN il faudra copier les buffers stress dans le bloc avant d'écrire les sismos. enlever cette opération de compute_veloc !*/
						STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, 
						STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, 
						STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, 
						STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, 

						STARPU_W, STARPU_W, STARPU_W, 
						STARPU_W, STARPU_W, STARPU_W, STARPU_W, STARPU_W, STARPU_W
		}
	};
// }}}
