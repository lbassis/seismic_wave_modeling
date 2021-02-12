/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010  Université de Bordeaux 1
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

#define BIND_LAST 1

/*
 * Schedule tasks for updates and saves
 */

/*
 * NB: iter = 0: initialization phase, TAG_U(z, 0) = TAG_INIT
 *
 * dir is -1 or +1.
 */

/*
 *	SAVE
 */

/* R(z) = R(z+d) = local, just call the save kernel */
// DGN
static void create_task_save_local(unsigned iter, unsigned x, unsigned y, int dir, unsigned local_rank, data_type type)
{
	struct starpu_task *save_task = starpu_task_create();
	struct block_description *descr = get_block_description(x, y);
	struct block_description *neighbour = descr->boundary_blocks[dir];

	switch (type)
	{
		case VELOC : 	switch(dir)
							{
								case XP :	save_task->cl = &save_veloc_xp_cl;
												break;
								case XM :	save_task->cl = &save_veloc_xm_cl;
												break;
								case YP :	save_task->cl = &save_veloc_yp_cl;
												break;
								case YM :	save_task->cl = &save_veloc_ym_cl;
												break;
								default : STARPU_ASSERT(0);
							}
							/* Saving our border... */
							save_task->handles[0] = descr->velocity_handle[0];
							save_task->handles[1] = descr->velocity_handle[1];
							save_task->handles[2] = descr->velocity_handle[2];

							/* ... to the neighbour's copy */
							if (neighbour) { /*no more circularity => no neighbour on domain's border*/
								save_task->handles[3] = neighbour->v_boundaries_handle[anti[dir]][0];
								save_task->handles[4] = neighbour->v_boundaries_handle[anti[dir]][1];
								save_task->handles[5] = neighbour->v_boundaries_handle[anti[dir]][2];
							} else {
								STARPU_ASSERT(0);
							}
							break;

		case STRESS : 	switch(dir)
							{
								case XP :	save_task->cl = &save_stress_xp_cl;
												break;
								case XM :	save_task->cl = &save_stress_xm_cl;
												break;
								case YP :	save_task->cl = &save_stress_yp_cl;
												break;
								case YM :	save_task->cl = &save_stress_ym_cl;
												break;
								default : STARPU_ASSERT(0);
							}
							/* Saving our border... */
							save_task->handles[0] = descr->stress_handle[0];
							save_task->handles[1] = descr->stress_handle[1];
							save_task->handles[2] = descr->stress_handle[2];
							save_task->handles[3] = descr->stress_handle[3];
							save_task->handles[4] = descr->stress_handle[4];
							save_task->handles[5] = descr->stress_handle[5];

							/* ... to the neighbour's copy */
							if (neighbour) { /*no more circularity => no neighbour on domain's border*/
								save_task->handles[6] = neighbour->t_boundaries_handle[anti[dir]][0];
								save_task->handles[7] = neighbour->t_boundaries_handle[anti[dir]][1];
								save_task->handles[8] = neighbour->t_boundaries_handle[anti[dir]][2];
								save_task->handles[9] = neighbour->t_boundaries_handle[anti[dir]][3];
								save_task->handles[10] = neighbour->t_boundaries_handle[anti[dir]][4];
								save_task->handles[11] = neighbour->t_boundaries_handle[anti[dir]][5];
							} else {
								STARPU_ASSERT(0);
							}
							break;

		default : STARPU_ASSERT(0);
	}
	save_task->cl_arg = descr;

	/* Bind */
	if (iter <= BIND_LAST)
		save_task->execute_on_a_specific_worker = get_bind_tasks();
	save_task->workerid = descr->preferred_worker;

	int ret = starpu_task_submit(save_task);
	if (ret)
	{
		fprintf(stderr, "Could not submit task save: %d\n", ret);
		STARPU_ASSERT(0);
	}
}

/* R(z) = local & R(z+d) != local */
/* We need to send our save over MPI */

static void send_done(void *arg)
{
	uintptr_t z = (uintptr_t) arg;
	DEBUG("DO SEND %d\n", (int)z);
}

#ifdef STARPU_USE_MPI
/* Post MPI send */
// DGN
static void create_task_save_mpi_send(unsigned iter, unsigned x, unsigned y, int dir, unsigned local_rank, data_type type)
{
	struct block_description *descr = get_block_description(x, y);
	STARPU_ASSERT(descr->mpi_node == local_rank);

	struct block_description *neighbour = descr->boundary_blocks[dir];
	// DGN -> s'il n'y avait pas de voisin, on ne devrait pas être là
	STARPU_ASSERT(neighbour);
	int dest = neighbour->mpi_node;
	STARPU_ASSERT(neighbour->mpi_node != local_rank);

	switch (type)
	{	/* Send neighbour's border copy to the neighbour */
		case VELOC : 	{
								starpu_data_handle_t handle0 = neighbour->v_boundaries_handle[anti[dir]][0];
								starpu_data_handle_t handle1 = neighbour->v_boundaries_handle[anti[dir]][1];
								starpu_data_handle_t handle2 = neighbour->v_boundaries_handle[anti[dir]][2];

								starpu_mpi_isend_detached(handle0, dest,MPI_TAG0(x, y, iter, dir), MPI_COMM_WORLD, send_done, (void*)(uintptr_t)IND(x,y));
								starpu_mpi_isend_detached(handle1, dest,MPI_TAG1(x, y, iter, dir), MPI_COMM_WORLD, send_done, (void*)(uintptr_t)IND(x,y));
								starpu_mpi_isend_detached(handle2, dest,MPI_TAG2(x, y, iter, dir), MPI_COMM_WORLD, send_done, (void*)(uintptr_t)IND(x,y));
								break;
							}	

		case STRESS : 	{
								starpu_data_handle_t handle0 = neighbour->t_boundaries_handle[anti[dir]][0];
								starpu_data_handle_t handle1 = neighbour->t_boundaries_handle[anti[dir]][1];
								starpu_data_handle_t handle2 = neighbour->t_boundaries_handle[anti[dir]][2];
								starpu_data_handle_t handle3 = neighbour->t_boundaries_handle[anti[dir]][3];
								starpu_data_handle_t handle4 = neighbour->t_boundaries_handle[anti[dir]][4];
								starpu_data_handle_t handle5 = neighbour->t_boundaries_handle[anti[dir]][5];

								starpu_mpi_isend_detached(handle0, dest,MPI_TAG0(x, y, iter, dir), MPI_COMM_WORLD, send_done, (void*)(uintptr_t)IND(x,y));
								starpu_mpi_isend_detached(handle1, dest,MPI_TAG1(x, y, iter, dir), MPI_COMM_WORLD, send_done, (void*)(uintptr_t)IND(x,y));
								starpu_mpi_isend_detached(handle2, dest,MPI_TAG2(x, y, iter, dir), MPI_COMM_WORLD, send_done, (void*)(uintptr_t)IND(x,y));
								starpu_mpi_isend_detached(handle3, dest,MPI_TAG3(x, y, iter, dir), MPI_COMM_WORLD, send_done, (void*)(uintptr_t)IND(x,y));
								starpu_mpi_isend_detached(handle4, dest,MPI_TAG4(x, y, iter, dir), MPI_COMM_WORLD, send_done, (void*)(uintptr_t)IND(x,y));
								starpu_mpi_isend_detached(handle5, dest,MPI_TAG5(x, y, iter, dir), MPI_COMM_WORLD, send_done, (void*)(uintptr_t)IND(x,y));
								break;
							}

		default : STARPU_ASSERT(0);
	}
}

/* R(z) != local & R(z+d) = local */
/* We need to receive over MPI */

static void recv_done(void *arg)
{
	uintptr_t z = (uintptr_t) arg;
	DEBUG("DO RECV %d\n", (int)z);
}

/* Post MPI recv */
// DGN
static void create_task_save_mpi_recv(unsigned iter, unsigned x, unsigned y, int dir, unsigned local_rank, data_type type)
{
	struct block_description *descr = get_block_description(x, y);
	STARPU_ASSERT(descr->mpi_node != local_rank);

	struct block_description *neighbour = descr->boundary_blocks[dir];
	// DGN -> s'il n'y avait pas de voisin, on ne devrait pas être là
	STARPU_ASSERT(neighbour);
	int source = descr->mpi_node;
	STARPU_ASSERT(neighbour->mpi_node == local_rank);

	switch (type)
	{	/* Receive our neighbour's border in our neighbour copy */
		case VELOC : 	{
								starpu_data_handle_t handle0 = neighbour->v_boundaries_handle[anti[dir]][0];
								starpu_data_handle_t handle1 = neighbour->v_boundaries_handle[anti[dir]][1];
								starpu_data_handle_t handle2 = neighbour->v_boundaries_handle[anti[dir]][2];

								starpu_mpi_irecv_detached(handle0, source,MPI_TAG0(x, y, iter, dir), MPI_COMM_WORLD, send_done, (void*)(uintptr_t)IND(x,y));
								starpu_mpi_irecv_detached(handle1, source,MPI_TAG1(x, y, iter, dir), MPI_COMM_WORLD, send_done, (void*)(uintptr_t)IND(x,y));
								starpu_mpi_irecv_detached(handle2, source,MPI_TAG2(x, y, iter, dir), MPI_COMM_WORLD, send_done, (void*)(uintptr_t)IND(x,y));
								break;
							}

		case STRESS : 	{
								starpu_data_handle_t handle0 = neighbour->t_boundaries_handle[anti[dir]][0];
								starpu_data_handle_t handle1 = neighbour->t_boundaries_handle[anti[dir]][1];
								starpu_data_handle_t handle2 = neighbour->t_boundaries_handle[anti[dir]][2];
								starpu_data_handle_t handle3 = neighbour->t_boundaries_handle[anti[dir]][3];
								starpu_data_handle_t handle4 = neighbour->t_boundaries_handle[anti[dir]][4];
								starpu_data_handle_t handle5 = neighbour->t_boundaries_handle[anti[dir]][5];

								starpu_mpi_irecv_detached(handle0, source,MPI_TAG0(x, y, iter, dir), MPI_COMM_WORLD, send_done, (void*)(uintptr_t)IND(x,y));
								starpu_mpi_irecv_detached(handle1, source,MPI_TAG1(x, y, iter, dir), MPI_COMM_WORLD, send_done, (void*)(uintptr_t)IND(x,y));
								starpu_mpi_irecv_detached(handle2, source,MPI_TAG2(x, y, iter, dir), MPI_COMM_WORLD, send_done, (void*)(uintptr_t)IND(x,y));
								starpu_mpi_irecv_detached(handle3, source,MPI_TAG3(x, y, iter, dir), MPI_COMM_WORLD, send_done, (void*)(uintptr_t)IND(x,y));
								starpu_mpi_irecv_detached(handle4, source,MPI_TAG4(x, y, iter, dir), MPI_COMM_WORLD, send_done, (void*)(uintptr_t)IND(x,y));
								starpu_mpi_irecv_detached(handle5, source,MPI_TAG5(x, y, iter, dir), MPI_COMM_WORLD, send_done, (void*)(uintptr_t)IND(x,y));
								break;
							}

		default : STARPU_ASSERT(0);
	}
}
#endif /* STARPU_USE_MPI */

/*
 * Schedule saving boundaries of blocks to communication buffers
 */
// DGN
void create_task_save(unsigned iter, unsigned x, unsigned y, int dir, unsigned local_rank, data_type type)
{
	// DGN : ATTENTION aux bords du domaine !!!!
	unsigned node = get_block_mpi_node(x, y);
	int node_d;
	int time_to_stop = 0;

	int xd = x, yd = y;
	switch(dir)
	{
		case XP :	xd++;
						break;
		case XM :	xd--;
						break;
		case YP :	yd++;
						break;
		case YM :	yd--;
						break;
	}

	// no boundary saving if on domain's border
	if (!(xd < 0 || yd < 0 || xd >= get_nbx() || yd >= get_nby())) 
	{	
		node_d = get_block_mpi_node(xd, yd);

#ifdef STARPU_USE_MPI
		if (node == local_rank)
		{
			/* Save data from update */
			create_task_save_local(iter, x, y, dir, local_rank, type);
			if (node_d != local_rank)
			{ /* We have to send the data */
				create_task_save_mpi_send(iter, x, y, dir, local_rank, type);
			}
		}
		else
		{	/* node_d != local_rank, this MPI node doesn't have the saved data */
			if (node_d == local_rank)
			{
				create_task_save_mpi_recv(iter, x, y, dir, local_rank, type);
			}
			else
			{ 
				STARPU_ASSERT(0);
			}
		}
#else /* !STARPU_USE_MPI */
		STARPU_ASSERT((node == local_rank) && (node_d == local_rank));
		create_task_save_local(iter, x, y, dir, local_rank, type);
#endif /* STARPU_USE_MPI */
	}
}

void create_task_update_sources(unsigned iter, unsigned x, unsigned y, unsigned local_rank)
{

	struct starpu_task *task = starpu_task_create();

	struct block_description *descr = get_block_description(x,y);

	// data handles
	task->handles[0] = descr->force_handle[0];
	task->handles[1] = descr->force_handle[1];
	task->handles[2] = descr->force_handle[2];

	// DGN penser à libérer ...
	block_iter_arg_t* arg = (block_iter_arg_t*)malloc(sizeof(block_iter_arg_t));
	arg->block = descr;
	arg->iter = iter;


	task->cl = &cl_update_source;
	task->cl_arg = arg;

	// DGN verifier ça ...
	if (iter <= BIND_LAST)
		task->execute_on_a_specific_worker = get_bind_tasks();

	task->workerid = descr->preferred_worker;

	int ret = starpu_task_submit(task);
	if (ret)
	{
		fprintf(stderr, "Could not submit task update_source block: {%d, %d}, iter: %d, retval: %d\n", x,y,iter,ret);
		STARPU_ASSERT(0);
	}
}

void create_task_compute_veloc(unsigned iter, unsigned x, unsigned y, unsigned local_rank)
{

	struct starpu_task *task = starpu_task_create();

	struct block_description *descr = get_block_description(x,y);

	// data handles
	task->handles[0] = descr->velocity_handle[0];
	task->handles[1] = descr->velocity_handle[1];
	task->handles[2] = descr->velocity_handle[2];

	task->handles[3] = descr->stress_handle[0];
	task->handles[4] = descr->stress_handle[1];
	task->handles[5] = descr->stress_handle[2];
	task->handles[6] = descr->stress_handle[3];
	task->handles[7] = descr->stress_handle[4];
	task->handles[8] = descr->stress_handle[5];

	// boundaries handles
	// stress
	task->handles[9] = descr->t_boundaries_handle[XP][0];
	task->handles[10] = descr->t_boundaries_handle[XP][1];
	task->handles[11] = descr->t_boundaries_handle[XP][2];
	task->handles[12] = descr->t_boundaries_handle[XP][3];
	task->handles[13] = descr->t_boundaries_handle[XP][4];
	task->handles[14] = descr->t_boundaries_handle[XP][5];

	task->handles[15] = descr->t_boundaries_handle[XM][0];
	task->handles[16] = descr->t_boundaries_handle[XM][1];
	task->handles[17] = descr->t_boundaries_handle[XM][2];
	task->handles[18] = descr->t_boundaries_handle[XM][3];
	task->handles[19] = descr->t_boundaries_handle[XM][4];
	task->handles[20] = descr->t_boundaries_handle[XM][5];

	task->handles[21] = descr->t_boundaries_handle[YP][0];
	task->handles[22] = descr->t_boundaries_handle[YP][1];
	task->handles[23] = descr->t_boundaries_handle[YP][2];
	task->handles[24] = descr->t_boundaries_handle[YP][3];
	task->handles[25] = descr->t_boundaries_handle[YP][4];
	task->handles[26] = descr->t_boundaries_handle[YP][5];

	task->handles[27] = descr->t_boundaries_handle[YM][0];
	task->handles[28] = descr->t_boundaries_handle[YM][1];
	task->handles[29] = descr->t_boundaries_handle[YM][2];
	task->handles[30] = descr->t_boundaries_handle[YM][3];
	task->handles[31] = descr->t_boundaries_handle[YM][4];
	task->handles[32] = descr->t_boundaries_handle[YM][5];

	task->handles[33] = descr->force_handle[0];
	task->handles[34] = descr->force_handle[1];
	task->handles[35] = descr->force_handle[2];

	task->handles[36] = descr->rho_handle;
	task->handles[37] = descr->vp_handle;

	task->handles[38] = descr->npml_tab_handle;

	task->handles[39] = descr->phit_handle[0];
	task->handles[40] = descr->phit_handle[1];
	task->handles[41] = descr->phit_handle[2];
	task->handles[42] = descr->phit_handle[3];
	task->handles[43] = descr->phit_handle[4];
	task->handles[44] = descr->phit_handle[5];
	task->handles[45] = descr->phit_handle[6];
	task->handles[46] = descr->phit_handle[7];
	task->handles[47] = descr->phit_handle[8];

	task->cl = &cl_compute_veloc;
	task->cl_arg = descr;

	if (iter <= BIND_LAST)
		task->execute_on_a_specific_worker = get_bind_tasks();
	
	task->workerid = descr->preferred_worker;
	int ret = starpu_task_submit(task);
	if (ret)
	{
		fprintf(stderr, "Could not submit task update block: %d\n", ret);
		STARPU_ASSERT(0);
	}
	INFO("created task compute veloc, for it %d & block {%d, %d}\n", iter, x, y);
}

void create_task_compute_stress(unsigned iter, unsigned x, unsigned y, unsigned local_rank)
{

	struct starpu_task *task = starpu_task_create();

	struct block_description *descr = get_block_description(x,y);

	// data handles
	task->handles[0] = descr->velocity_handle[0];
	task->handles[1] = descr->velocity_handle[1];
	task->handles[2] = descr->velocity_handle[2];

	task->handles[3] = descr->stress_handle[0];
	task->handles[4] = descr->stress_handle[1];
	task->handles[5] = descr->stress_handle[2];
	task->handles[6] = descr->stress_handle[3];
	task->handles[7] = descr->stress_handle[4];
	task->handles[8] = descr->stress_handle[5];

	// boundaries handles
	// velocity
	task->handles[9] = descr->v_boundaries_handle[XP][0];
	task->handles[10] = descr->v_boundaries_handle[XP][1];
	task->handles[11] = descr->v_boundaries_handle[XP][2];

	task->handles[12] = descr->v_boundaries_handle[XM][0];
	task->handles[13] = descr->v_boundaries_handle[XM][1];
	task->handles[14] = descr->v_boundaries_handle[XM][2];

	task->handles[15] = descr->v_boundaries_handle[YP][0];
	task->handles[16] = descr->v_boundaries_handle[YP][1];
	task->handles[17] = descr->v_boundaries_handle[YP][2];

	task->handles[18] = descr->v_boundaries_handle[YM][0];
	task->handles[19] = descr->v_boundaries_handle[YM][1];
	task->handles[20] = descr->v_boundaries_handle[YM][2];

	task->handles[21] = descr->mu_handle;
	task->handles[22] = descr->lam_handle;
	task->handles[23] = descr->vp_handle;

	task->handles[24] = descr->npml_tab_handle;

	task->handles[25] = descr->phiv_handle[0];
	task->handles[26] = descr->phiv_handle[1];
	task->handles[27] = descr->phiv_handle[2];
	task->handles[28] = descr->phiv_handle[3];
	task->handles[29] = descr->phiv_handle[4];
	task->handles[30] = descr->phiv_handle[5];
	task->handles[31] = descr->phiv_handle[6];
	task->handles[32] = descr->phiv_handle[7];
	task->handles[33] = descr->phiv_handle[8];

	task->cl = &cl_compute_stress;
	task->cl_arg = descr;

	if (iter <= BIND_LAST)
		task->execute_on_a_specific_worker = get_bind_tasks();
	
	task->workerid = descr->preferred_worker;

	int ret = starpu_task_submit(task);
	if (ret)
	{
		fprintf(stderr, "Could not submit task update block: %d\n", ret);
		STARPU_ASSERT(0);
	}
	INFO("created task compute stress, for it %d & block {%d, %d}\n", iter, x, y);
}

void create_task_record_seismo(unsigned iter, unsigned x, unsigned y, unsigned local_rank)
{

	struct starpu_task *task = starpu_task_create();

	struct block_description *descr = get_block_description(x,y);

	// data handles
	task->handles[0] = descr->velocity_handle[0];
	task->handles[1] = descr->velocity_handle[1];
	task->handles[2] = descr->velocity_handle[2];

	task->handles[3] = descr->stress_handle[0];
	task->handles[4] = descr->stress_handle[1];
	task->handles[5] = descr->stress_handle[2];
	task->handles[6] = descr->stress_handle[3];
	task->handles[7] = descr->stress_handle[4];
	task->handles[8] = descr->stress_handle[5];

	// boundaries handles
	// stress
	task->handles[9] = descr->t_boundaries_handle[XP][0];
	task->handles[10] = descr->t_boundaries_handle[XP][1];
	task->handles[11] = descr->t_boundaries_handle[XP][2];
	task->handles[12] = descr->t_boundaries_handle[XP][3];
	task->handles[13] = descr->t_boundaries_handle[XP][4];
	task->handles[14] = descr->t_boundaries_handle[XP][5];

	task->handles[15] = descr->t_boundaries_handle[XM][0];
	task->handles[16] = descr->t_boundaries_handle[XM][1];
	task->handles[17] = descr->t_boundaries_handle[XM][2];
	task->handles[18] = descr->t_boundaries_handle[XM][3];
	task->handles[19] = descr->t_boundaries_handle[XM][4];
	task->handles[20] = descr->t_boundaries_handle[XM][5];

	task->handles[21] = descr->t_boundaries_handle[YP][0];
	task->handles[22] = descr->t_boundaries_handle[YP][1];
	task->handles[23] = descr->t_boundaries_handle[YP][2];
	task->handles[24] = descr->t_boundaries_handle[YP][3];
	task->handles[25] = descr->t_boundaries_handle[YP][4];
	task->handles[26] = descr->t_boundaries_handle[YP][5];

	task->handles[27] = descr->t_boundaries_handle[YM][0];
	task->handles[28] = descr->t_boundaries_handle[YM][1];
	task->handles[29] = descr->t_boundaries_handle[YM][2];
	task->handles[30] = descr->t_boundaries_handle[YM][3];
	task->handles[31] = descr->t_boundaries_handle[YM][4];
	task->handles[32] = descr->t_boundaries_handle[YM][5];

	task->handles[33] = descr->seismo_handle[0];
	task->handles[34] = descr->seismo_handle[1];
	task->handles[35] = descr->seismo_handle[2];

	task->handles[36] = descr->seismo_handle[3];
	task->handles[37] = descr->seismo_handle[4];
	task->handles[38] = descr->seismo_handle[5];
	task->handles[39] = descr->seismo_handle[6];
	task->handles[40] = descr->seismo_handle[7];
	task->handles[41] = descr->seismo_handle[8];


	// DGN to free
	block_iter_arg_t* descr_it = (block_iter_arg_t*)malloc(sizeof(block_iter_arg_t));
	descr_it->block = descr; 
	descr_it->iter = iter; 

	task->cl = &cl_record_seismo;
	task->cl_arg = descr_it;

	/* We are going to synchronize with the last tasks */
	if (iter == get_niter())
	{
		task->detach = 0;
		task->use_tag = 1;
		task->tag_id = TAG_FINISH(x,y);
	}

	if (iter <= BIND_LAST)
		task->execute_on_a_specific_worker = get_bind_tasks();
	
	task->workerid = descr->preferred_worker;

	int ret = starpu_task_submit(task);
	if (ret)
	{
		fprintf(stderr, "Could not submit task update block: %d\n", ret);
		STARPU_ASSERT(0);
	}
	INFO("created task record seismos, for it %d & block {%d, %d}\n", iter, x, y);
}

/* Dummy empty codelet taking one buffer */
static void null_func(void *descr[] __attribute__((unused)), void *arg __attribute__((unused))) { }

static struct starpu_codelet null =
{
	.modes = { STARPU_W, STARPU_W, STARPU_W, STARPU_W, STARPU_W, STARPU_W, STARPU_W, STARPU_W, STARPU_W },
	.where = STARPU_CPU|STARPU_CUDA|STARPU_OPENCL,
	.cpu_funcs = {null_func, NULL},
	.cuda_funcs = {null_func, NULL},
	.opencl_funcs = {null_func, NULL},
	.nbuffers = 9
};

void create_start_task(int x, int y, int dir)
{
	/* Dumb task depending on the init task and simulating writing the
	   neighbour buffers, to avoid communications and computation running
	   before we start measuring time */
	struct starpu_task *wait_init = starpu_task_create();

	if (!wait_init) {
		fprintf(stderr, "Could not create task\n");
		STARPU_ASSERT(0);
	}

	struct block_description *descr = get_block_description(x, y);
	starpu_tag_t tag_init = TAG_INIT_TASK;
	wait_init->cl = &null;
	wait_init->use_tag = 1;
	wait_init->tag_id = TAG_START(x, y, dir);

	wait_init->handles[0] = descr->v_boundaries_handle[dir][0];
	wait_init->handles[1] = descr->v_boundaries_handle[dir][1];
	wait_init->handles[2] = descr->v_boundaries_handle[dir][2];

	wait_init->handles[3] = descr->t_boundaries_handle[dir][0];
	wait_init->handles[4] = descr->t_boundaries_handle[dir][1];
	wait_init->handles[5] = descr->t_boundaries_handle[dir][2];
	wait_init->handles[6] = descr->t_boundaries_handle[dir][3];
	wait_init->handles[7] = descr->t_boundaries_handle[dir][4];
	wait_init->handles[8] = descr->t_boundaries_handle[dir][5];

	starpu_tag_declare_deps_array(wait_init->tag_id, 1, &tag_init);
	int ret = starpu_task_submit(wait_init);
	if (ret)
	{
		fprintf(stderr, "Could not submit task initial wait: %d\n", ret);
		STARPU_ASSERT(0);
	}
}

void create_end_task(int x, int y, int dir)
{
	/* Dumb task depending on the init task and simulating writing the
	   neighbour buffers, to avoid communications and computation running
	   before we start measuring time */
	struct starpu_task *wait_init = starpu_task_create();

	if (!wait_init) {
		fprintf(stderr, "Could not create task\n");
		STARPU_ASSERT(0);
	}

	struct block_description *descr = get_block_description(x, y);
	starpu_tag_t tag_init = TAG_INIT_TASK;
	wait_init->cl = &null;
	wait_init->use_tag = 1;
	wait_init->tag_id = TAG_START(x, y, dir);

	wait_init->handles[0] = descr->v_boundaries_handle[dir][0];
	wait_init->handles[1] = descr->v_boundaries_handle[dir][1];
	wait_init->handles[2] = descr->v_boundaries_handle[dir][2];

	wait_init->handles[3] = descr->t_boundaries_handle[dir][0];
	wait_init->handles[4] = descr->t_boundaries_handle[dir][1];
	wait_init->handles[5] = descr->t_boundaries_handle[dir][2];
	wait_init->handles[6] = descr->t_boundaries_handle[dir][3];
	wait_init->handles[7] = descr->t_boundaries_handle[dir][4];
	wait_init->handles[8] = descr->t_boundaries_handle[dir][5];

	starpu_tag_declare_deps_array(wait_init->tag_id, 1, &tag_init);
	int ret = starpu_task_submit(wait_init);
	if (ret)
	{
		fprintf(stderr, "Could not submit task initial wait: %d\n", ret);
		STARPU_ASSERT(0);
	}
}

/*
 * Create all the tasks
 */
// DGN
void create_tasks(int rank)
{
	unsigned iter;
	int niter = get_niter();
	int nbx = get_nbx();
	int nby = get_nby();
   unsigned bx, by;
   
   // DGN
   
   for (bx = 0; bx < nbx; bx++)
   {  for (by = 0; by < nby; by++)
      {
			if ((get_block_mpi_node(bx, by) == rank) || (get_block_mpi_node(bx+1, by) == rank))
				create_start_task(bx, by, XP);
			if ((get_block_mpi_node(bx, by) == rank) || (get_block_mpi_node(bx-1, by) == rank))
				create_start_task(bx, by, XM);

			if ((get_block_mpi_node(bx, by) == rank) || (get_block_mpi_node(bx, by+1) == rank))
				create_start_task(bx, by, YP);
			if ((get_block_mpi_node(bx, by) == rank) || (get_block_mpi_node(bx, by-1) == rank))
				create_start_task(bx, by, YM);
		}
	}

	
	for (iter = 0; iter <= niter; iter++)
	{  for (bx = 0; bx < nbx; bx++)
	   {  for (by = 0; by < nby; by++)
	      {
	      	// DGN
	      	struct block_description * block = get_block_description(bx, by);
	      	// I) creer une tâche d'update par bloc possédant une source
	      	if (block->mpi_node == rank && block->nb_src > 0) {
	      		create_task_update_sources(iter, bx, by, rank);
		      }
		      
	      	// II) Kernel_1 : V = f(T)
		      if (block->mpi_node == rank) {
		      	create_task_compute_veloc(iter, bx, by, rank);
		      }
		      
				// III) save boundaries(V)
	      	if ((block->mpi_node == rank) || (get_block_mpi_node(bx+1, by) == rank)) 
					create_task_save(iter, bx, by, XP, rank, VELOC);
		
			if ((block->mpi_node == rank) || (get_block_mpi_node(bx-1, by) == rank))
				create_task_save(iter, bx, by, XM, rank, VELOC);
			
			if ((block->mpi_node == rank) || (get_block_mpi_node(bx, by+1) == rank))
				create_task_save(iter, bx, by, YP, rank, VELOC);
	
			if ((block->mpi_node == rank) || (get_block_mpi_node(bx, by-1) == rank))
				create_task_save(iter, bx, by, YM, rank, VELOC);

				// IV) Kernel_2 : T = f(V)
		    if (block->mpi_node == rank)
		      	create_task_compute_stress(iter, bx, by, rank);


				// V) save boundaries(T)
	      	if ((block->mpi_node == rank) || (get_block_mpi_node(bx+1, by) == rank)) 
				create_task_save(iter, bx, by, XP, rank, STRESS);
		
			if ((block->mpi_node == rank) || (get_block_mpi_node(bx-1, by) == rank))
				create_task_save(iter, bx, by, XM, rank, STRESS);
			
			if ((block->mpi_node == rank) || (get_block_mpi_node(bx, by+1) == rank))
				create_task_save(iter, bx, by, YP, rank, STRESS);
	
			if ((block->mpi_node == rank) || (get_block_mpi_node(bx, by-1) == rank))
				create_task_save(iter, bx, by, YM, rank, STRESS);

// if (iter==niter && block->nb_sta == 0)
// if (iter==niter)

			// VI) creer une tâche d'ecriture des sismos par bloc possédant une station
	      	if (block->mpi_node == rank && ((iter < niter && block->nb_sta > 0) || (iter == niter)))
	      		create_task_record_seismo(iter, bx, by, rank);
		      // DGN : comment synchroniser sur la dernière tâche pour les blocs sans sismos ??
		      // DGN TODO : créer une tâche fictive pour la derniere iteration pour la synchro
		      // synchroniser create_task_save_stress (cf vieux code update)
			}
		}
	}
	
}

/*
 * Wait for termination
 */
void wait_end_tasks(int rank)
{
	unsigned bx, by;
	int nby = get_nby();
	int nbx = get_nbx();

	for (bx = 0; bx < nbx; bx++)
	{	for (by = 0; by < nby; by++)
		{
			if (get_block_mpi_node(bx, by) == rank)
			{
				/* Wait for the task producing block "bx, by" */
				starpu_tag_wait(TAG_FINISH(bx, by));

				/* Get the result back to memory */
				struct block_description *block = get_block_description(bx, by);

				// DGN a finir
/*				starpu_data_acquire(block->layers_handle[0], STARPU_R);
				starpu_data_acquire(block->layers_handle[1], STARPU_R);
*/			}
		}
	}
}
