/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010  Université de Bordeaux 1
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
#include <math.h>

/* Manage block and tags allocation */

static struct block_description *blocks;
static unsigned *block_sizes_x;
static unsigned *block_sizes_y;
unsigned anti[4] = {1, 0, 3, 2};

/*
 *	Tags for various codelet completion
 */

/*
 * common tag format:
 */
static starpu_tag_t tag_common(int z, int dir, int type)
{
	return (((((starpu_tag_t)type) << 4) | dir) << 32)|(starpu_tag_t)z;
}

/* Completion of last update tasks */
// DGN
starpu_tag_t TAG_FINISH(int x, int y)
{
	// z = (z + nbz)%nbz;
	starpu_tag_t tag = tag_common(IND(x,y), 0, 1);
	return tag;
}

/* Completion of the save codelet for MPI send/recv */
// DGN
starpu_tag_t TAG_START(int x, int y, int dir)
{
	/*z = (z + nbz)%nbz;*/

	starpu_tag_t tag = tag_common(IND(x,y), dir, 2);
	return tag;
}

/*
 * common MPI tag format:
 * iter is actually not needed for coherency, but it makes debugging easier

 DGN : OK
 */
static int mpi_tag_common(int z, int iter, int dir, int buffer)
{
	return (((((iter << 12)|z)<<4) | dir)<<4)|buffer;
}

/*
buffer -> 4 bits (ok : 3)
dir -> 4 bits (ok : 2)
z -> 12 bits (4096 blocs possibles)
iter -> 32 - 12 - 4 - 4 = 12 bits (4096 iter possibles)
*/

// DGN
int MPI_TAG0(int x, int y, int iter, int dir)
{
	int tag = mpi_tag_common(IND(x,y), iter, dir, 0);

	return tag;
}

// DGN
int MPI_TAG1(int x, int y, int iter, int dir)
{
	int tag = mpi_tag_common(IND(x,y), iter, dir, 1);

	return tag;
}

int MPI_TAG2(int x, int y, int iter, int dir)
{
	int tag = mpi_tag_common(IND(x,y), iter, dir, 2);

	return tag;
}

int MPI_TAG3(int x, int y, int iter, int dir)
{
	int tag = mpi_tag_common(IND(x,y), iter, dir, 3);

	return tag;
}

// DGN
int MPI_TAG4(int x, int y, int iter, int dir)
{
	int tag = mpi_tag_common(IND(x,y), iter, dir, 4);

	return tag;
}

int MPI_TAG5(int x, int y, int iter, int dir)
{
	int tag = mpi_tag_common(IND(x,y), iter, dir, 5);

	return tag;
}

/*
 *	Block descriptors
 */

/* Compute the size of the different blocks */
// DGN
static void compute_block_sizes(unsigned bsizex, unsigned bsizey)
{
	block_sizes_x = (unsigned *) malloc(get_nbx()*sizeof(unsigned));
	block_sizes_y = (unsigned *) malloc(get_nby()*sizeof(unsigned));
	STARPU_ASSERT(block_sizes_x);
	STARPU_ASSERT(block_sizes_y);

	/* Perhaps the last chunk is smaller */
	unsigned remaining = get_sizex();

	unsigned b;
	for (b = 0; b < get_nbx(); b++)
	{
		block_sizes_x[b] = MIN(bsizex, remaining);
		remaining -= block_sizes_x[b];
		INFO("Size: Xdir : block %d/%d ->\t%d\n", b+1, get_nbx(), block_sizes_x[b]);
	}
	STARPU_ASSERT(remaining == 0);
	
	remaining = get_sizey();

	for (b = 0; b < get_nby(); b++)
	{
		block_sizes_y[b] = MIN(bsizey, remaining);
		remaining -= block_sizes_y[b];
		INFO("Size: Ydir : block %d/%d ->\t%d\n", b+1, get_nby(), block_sizes_y[b]);
	}
	STARPU_ASSERT(remaining == 0);
}

// DGN
unsigned get_block_size_x(int bx)
{
	return block_sizes_x[bx];
}

// DGN
unsigned get_block_size_y(int by)
{
	return block_sizes_y[by];
}

// DGN
struct block_description *get_block_description(int x, int y)
{
	// DGN -> circulaire : ne nous intéresse pas pour Ondes3D!!!
	// x = (x + nbx)%nbx;
	// y = (y + nby)%nby;
	// DGN
	if (x<0 || x>=get_nbx() || y<0 || y>=get_nby()) return NULL;

	STARPU_ASSERT(&blocks[IND(x,y)]);

	return &blocks[IND(x,y)];
}

// DGN
int get_block_mpi_node(int x, int y)
{
	// DGN -> si en dehors du domaine -1
	if (x<0 || y<0 || x>=get_nbx() || y>= get_nby()) return -1;

	return blocks[IND(x,y)].mpi_node;
}

// DGN
void create_blocks_array(unsigned _sizex, unsigned _sizey, unsigned _sizez,  unsigned _nbx, unsigned _nby, unsigned bsizex, unsigned bsizey, ondes3d_params* params)
{
	/* Store the parameters */
	// DGN
	unsigned nbx = _nbx;
	unsigned nby = _nby;
	unsigned sizex = _sizex;
	unsigned sizey = _sizey;
	unsigned sizez = _sizez;

	/* Create a grid of block descriptors */
	// DGN
	blocks = (struct block_description *) calloc(nbx*nby, sizeof(struct block_description));
	STARPU_ASSERT(blocks);

	/* What is the size of the different blocks ? */
	compute_block_sizes(bsizex, bsizey);

	// DGN
	unsigned bx;
	unsigned by;
	for (bx = 0; bx < nbx; bx++)
	{	for (by = 0; by < nby; by++)
		{
			struct block_description * block = get_block_description(bx, by);

			/* Which block is it ? */
			block->bx = bx;
			block->by = by;

			block->sizex = get_block_size_x(bx);
			block->sizey = get_block_size_y(by);
			block->sizez = sizez;

			block->xshift = bx*get_block_size_x(0);
			block->yshift = by*get_block_size_y(0);

			if (aligned()) {
				int right_padding = (((ALIGN + block->sizex + K)+ALIGN-1)/ALIGN)*ALIGN - (ALIGN + block->sizex + K);
				block->pitchx = ALIGN + block->sizex + K + right_padding;
				block->padding = right_padding;
			} else {
				block->pitchx = block->sizex + 2*K;
				block->padding = 0;
			}
			block->pitchy = block->sizey + 2*K;
			block->pitchz = block->sizez + 2*K;

			block->params = params;
			block->iter = -1;

			// DGN CORRECTION
			// offset du point (0,0,0) dans le bloc
			if (aligned()) {			
							block->offset = K*block->pitchx*block->pitchy + K*block->pitchx + ALIGN;
			} else {
							block->offset = K*block->pitchx*block->pitchy + K*block->pitchx + K;
			}

			block->nb_cpml = compute_nb_cpml(sizex, sizey, sizez, bx, by, block->sizex, block->sizey, get_block_size_x(0), get_block_size_y(0), DELTA);
			INFO("block {%d, %d} has %d CPML points on %d\n", bx, by, block->nb_cpml, sizex*sizey*sizez);
			// DGN ATTENTION : le pointeur vers le bloc peut etre null !!!
			/* For simplicity, we store which are the neighbours blocks */
			block->boundary_blocks[XM] = get_block_description(bx-1, by);
			block->boundary_blocks[XP] = get_block_description(bx+1, by);
			block->boundary_blocks[YM] = get_block_description(bx, by-1);
			block->boundary_blocks[YP] = get_block_description(bx, by+1);
		}
	}
}

// DGN
unsigned compute_nb_cpml(unsigned _sizex, unsigned _sizey, unsigned _sizez, unsigned bx, unsigned by, unsigned thisbsizex, unsigned thisbsizey, unsigned bsizex, unsigned bsizey, unsigned delta)
{
	// calcule les dimensions du volume sans cpmls, retranche du volume du bloc
	unsigned sx = thisbsizex;
	unsigned sy = thisbsizey;
	unsigned sz = _sizez;
	unsigned volume = sx*sy*sz;

	sz -= delta; // cpmls du fond
	if (bx*bsizex < delta) { // xmin
		sx -= delta - bx*bsizex;
	}
	if ((_sizex-(bx*bsizex+thisbsizex)) < delta) { // xmax
		sx -= delta - (_sizex-(bx*bsizex+thisbsizex));
	}
	if (by*bsizey < delta) {// ymin
		sy -= delta - by*bsizey;
	}
	if ((_sizey-(by*bsizey+thisbsizey)) < delta) { // ymax
		sy -= delta - (_sizey-(by*bsizey+thisbsizey));
	}
	return volume-sx*sy*sz;
}

/*
 *	Initialization of the blocks
 */

// DGN
void assign_blocks_to_workers(int rank)
{
	unsigned bx, by;

	/* NB: perhaps we could count a GPU as multiple workers */

	/* how many workers are there ? */
	/*unsigned nworkers = starpu_worker_get_count();*/

	/* how many blocks are on that MPI node ? */
	unsigned nblocks = 0;
	for (bx = 0; bx < get_nbx(); bx++)
	{	for (by = 0; by < get_nby(); by++)
		{
			struct block_description *block = get_block_description(bx, by);

			if (block->mpi_node == rank)
				nblocks++;
		}
	}

	/* how many blocks per worker ? */
	/*unsigned nblocks_per_worker = (nblocks + nworkers - 1)/nworkers;*/

	/* we now attribute up to nblocks_per_worker blocks per workers */
	unsigned attributed = 0;
	for (bx = 0; bx < get_nbx(); bx++)
	{	for (by = 0; by < get_nby(); by++)
		{
			struct block_description *block = get_block_description(bx, by);

			if (block->mpi_node == rank)
			{
				unsigned workerid;
				/* Manage initial block distribution between CPU and GPU */
// DGN TODO : reflechir à la distribution en fonction de la granularité
			#if 1
				#if 1
				/* GPUs then CPUs */
				if (attributed < 3*18)
					workerid = attributed / 18;
				else
					workerid = 3+ (attributed - 3*18) / 2;
				#else
				/* GPUs interleaved with CPUs */
				if ((attributed % 20) <= 1)
					workerid = 3 + attributed / 20;
				else if (attributed < 60)
					workerid = attributed / 20;
				else
					workerid = (attributed - 60)/2 + 6;
				#endif
			#else
				/* Only GPUS */
				workerid = (attributed / 21) % 3;
			#endif
				/*= attributed/nblocks_per_worker;*/

				block->preferred_worker = workerid;

				attributed++;
			}
		}
	}
}



// DGN
void assign_blocks_to_mpi_nodes(int world_size)
{
	unsigned nxblocks_per_process = (get_nbx() + NPROCX - 1) / NPROCX;
	unsigned nyblocks_per_process = (get_nby() + NPROCY - 1) / NPROCY;

	unsigned bx, by;
	for (bx = 0; bx < get_nbx(); bx++)
	{	for (by = 0; by < get_nby(); by++)
		{
			struct block_description *block = get_block_description(bx, by);

			 int mpix = bx / nxblocks_per_process;
			 int mpiy = by / nyblocks_per_process;

			 block->mpi_node = mpiy*NPROCX+mpix;
		}
	}
}

static size_t allocated = 0;

static void allocate_block_on_node(starpu_data_handle_t *handleptr, float **ptr, unsigned nx, unsigned ny, unsigned nz)
{
	int ret;
	size_t block_size = nx*ny*nz*sizeof(float);

	/* Allocate memory */
#ifdef STARPU_MALLOC
	*ptr = malloc(block_size);
	STARPU_ASSERT(*ptr);
#else
	ret = starpu_malloc((void **)ptr, block_size);
	STARPU_ASSERT(ret == 0);
#endif
	allocated += block_size;

	/* Fill the blocks with 0 */
	memset(*ptr, 0, block_size);

	/* Register it to StarPU */
	starpu_block_data_register(handleptr, 0, (uintptr_t)*ptr, nx, nx*ny, nx, ny, nz, sizeof(float));
}

static void allocate_block_on_node_spec(starpu_data_handle_t *handleptr, float **ptr, unsigned nx, unsigned ny, unsigned nz)
{
	int ret;
	size_t block_size = nx*ny*nz*sizeof(float);

	/* Allocate memory */
#ifdef STARPU_MALLOC
	*ptr = malloc(block_size);
	STARPU_ASSERT(*ptr);
#else
	ret = starpu_malloc((void **)ptr, block_size);
	STARPU_ASSERT(ret == 0);
#endif
	allocated += block_size;

	/* Fill the blocks with dummy value */
	*ptr[0] = 25.1273;

	/* Register it to StarPU */
	starpu_block_data_register(handleptr, 0, (uintptr_t)*ptr, nx, nx*ny, nx, ny, nz, sizeof(float));
}

static void allocate_float_vector_on_node(starpu_data_handle_t *handleptr, float **ptr, unsigned nx)
{
	int ret;
	size_t vector_size = nx*sizeof(float);

	/* Allocate memory */
#ifdef STARPU_MALLOC
	*ptr = malloc(vector_size);
	STARPU_ASSERT(*ptr);
#else
	ret = starpu_malloc((void **)ptr, vector_size);
	STARPU_ASSERT(ret == 0);
#endif

	allocated += vector_size;

	/* Fill the vectors with 0 */
	memset(*ptr, 0, vector_size);

	/* Register it to StarPU */
	starpu_vector_data_register(handleptr, 0, (uintptr_t)*ptr, nx, sizeof(float));
}

static void allocate_int_vector_on_node(starpu_data_handle_t *handleptr, int **ptr, unsigned nx)
{
	int ret;
	size_t vector_size = nx*sizeof(int);

	/* Allocate memory */
#ifdef STARPU_MALLOC
	*ptr = malloc(vector_size);
	STARPU_ASSERT(*ptr);
#else
	ret = starpu_malloc((void **)ptr, vector_size);
	STARPU_ASSERT(ret == 0);
#endif

	allocated += vector_size;

	/* Fill the vectors with 0 */
	memset(*ptr, 0, vector_size);

	/* Register it to StarPU */
	starpu_vector_data_register(handleptr, 0, (uintptr_t)*ptr, nx, sizeof(int));
}

void allocate_dble_vector_on_node(starpu_data_handle_t *handleptr, double **ptr, unsigned nx)
{
	double ret;
	size_t vector_size = nx*sizeof(double);
	// INFO("size=%d\n",vector_size);

	/* Allocate memory */
#ifdef STARPU_MALLOC
	*ptr = malloc(vector_size);
	STARPU_ASSERT(*ptr);
#else
	ret = starpu_malloc((void **)ptr, vector_size);
	STARPU_ASSERT(ret == 0);
#endif

	allocated += vector_size;

	/* Fill the vectors with 0 */
	memset(*ptr, 0, vector_size);

	/* Register it to StarPU */
	starpu_vector_data_register(handleptr, 0, (uintptr_t)*ptr, nx, sizeof(double));
}

void create_cpml_indirection(int rank)
{
	unsigned bsizex = get_block_size_x(0);
	unsigned bsizey = get_block_size_y(0);

   unsigned bx, by;
   unsigned xshift, yshift;
   unsigned x,y,z;

   for (bx = 0; bx < get_nbx(); bx++)
   {	xshift = bx*bsizex;
   	for (by = 0; by < get_nby(); by++)
      {	yshift = by*bsizey;
      	int ncpml = 0;

      	struct block_description *block = get_block_description(bx, by);
			unsigned node = block->mpi_node;

			if (node == rank) {
				for(int iz=0; iz<block->sizez; iz++) {			
					z = iz;
					for(int iy=0; iy<block->sizey; iy++) {			
						y = iy+yshift;
						for(int ix=0; ix<block->sizex; ix++) {			
							// global coordinates
							x = ix+xshift;
							if (x < DELTA || x >= (get_sizex()-DELTA) || y < DELTA || y >= (get_sizey()-DELTA) || z < DELTA) {
								block->npml_tab[ix + iy*block->sizex + iz*block->sizex*block->sizey] = ncpml++;
							} else {
								block->npml_tab[ix + iy*block->sizex + iz*block->sizex*block->sizey] = -1;
							}
						}
					}
				}
				STARPU_ASSERT(ncpml == block->nb_cpml);
				if (DELTA == 0) STARPU_ASSERT(ncpml == 0);
			}
      }
   }
}

void display_memory_consumption(int rank)
{
	fprintf(stdout, "%lu MB of memory were allocated on node %d\n", allocated/(1024*1024), rank);
}

// DGN
void allocate_memory_on_node(int rank)
{
	unsigned bx, by;
	for (bx = 0; bx < get_nbx(); bx++)
	{	for (by = 0; by < get_nby(); by++)
		{
			struct block_description *block = get_block_description(bx, by);

			unsigned node = block->mpi_node;

			unsigned size_bx = block_sizes_x[bx];
			unsigned size_by = block_sizes_y[by];

			/* Main blocks */
			if (node == rank)
			{
				// velocity vector (with boundaries)
				allocate_block_on_node(&block->velocity_handle[0], &block->vx, block->pitchx, block->pitchy, block->pitchz);
				allocate_block_on_node(&block->velocity_handle[1], &block->vy, block->pitchx, block->pitchy, block->pitchz);
				allocate_block_on_node(&block->velocity_handle[2], &block->vz, block->pitchx, block->pitchy, block->pitchz);

				// stress tensor (with boundaries)
				allocate_block_on_node(&block->stress_handle[0], &block->txx, block->pitchx, block->pitchy, block->pitchz);
				allocate_block_on_node(&block->stress_handle[1], &block->tyy, block->pitchx, block->pitchy, block->pitchz);
				allocate_block_on_node(&block->stress_handle[2], &block->tzz, block->pitchx, block->pitchy, block->pitchz);
				allocate_block_on_node(&block->stress_handle[3], &block->txy, block->pitchx, block->pitchy, block->pitchz);
				allocate_block_on_node(&block->stress_handle[4], &block->txz, block->pitchx, block->pitchy, block->pitchz);
				allocate_block_on_node(&block->stress_handle[5], &block->tyz, block->pitchx, block->pitchy, block->pitchz);

				// source momentum vector (no boundary)
				allocate_block_on_node(&block->force_handle[0], &block->fx, block->pitchx, block->pitchy, block->pitchz);
				allocate_block_on_node(&block->force_handle[1], &block->fy, block->pitchx, block->pitchy, block->pitchz);
				allocate_block_on_node(&block->force_handle[2], &block->fz, block->pitchx, block->pitchy, block->pitchz);

				// material properties (no boundary)
				allocate_block_on_node(&block->mu_handle, &block->mu, block->pitchx, block->pitchy, block->pitchz);
				allocate_block_on_node(&block->rho_handle, &block->rho, block->pitchx, block->pitchy, block->pitchz);
				allocate_block_on_node(&block->lam_handle, &block->lam, block->pitchx, block->pitchy, block->pitchz);
				allocate_block_on_node(&block->vp_handle, &block->vp, block->pitchx, block->pitchy, block->pitchz);
				allocate_block_on_node(&block->vs_handle, &block->vs, block->pitchx, block->pitchy, block->pitchz);

				// DGN DEBUG
				// DGN_DBG
				// allocate_dble_vector_on_node(&block->seismo_handle[0], &block->seisx , 1024);
				// DGN_DBG
				// allocate_dble_vector_on_node(&block->seismo_handle[1], &block->seisy , 1024);
				// DGN_DBG
				// allocate_dble_vector_on_node(&block->seismo_handle[2], &block->seisz , 1024);
				// allocate_dble_vector_on_node(&block->seismo_handle[3], &block->seisxx, 1024);
				// allocate_dble_vector_on_node(&block->seismo_handle[4], &block->seisyy, 1024);
				// allocate_dble_vector_on_node(&block->seismo_handle[5], &block->seiszz, 1024);
				// allocate_dble_vector_on_node(&block->seismo_handle[6], &block->seisxy, 1024);
				// allocate_dble_vector_on_node(&block->seismo_handle[7], &block->seisxz, 1024);
				// allocate_dble_vector_on_node(&block->seismo_handle[8], &block->seisyz, 1024);
				// DGN_DBG

				// CPMLS
				// arrays
				allocate_float_vector_on_node(&block->phiv_handle[0], &block->phivxx, block->nb_cpml>0?block->nb_cpml:1);
				allocate_float_vector_on_node(&block->phiv_handle[1], &block->phivxy, block->nb_cpml>0?block->nb_cpml:1);
				allocate_float_vector_on_node(&block->phiv_handle[2], &block->phivxz, block->nb_cpml>0?block->nb_cpml:1);
				allocate_float_vector_on_node(&block->phiv_handle[3], &block->phivyx, block->nb_cpml>0?block->nb_cpml:1);
				allocate_float_vector_on_node(&block->phiv_handle[4], &block->phivyy, block->nb_cpml>0?block->nb_cpml:1);
				allocate_float_vector_on_node(&block->phiv_handle[5], &block->phivyz, block->nb_cpml>0?block->nb_cpml:1);
				allocate_float_vector_on_node(&block->phiv_handle[6], &block->phivzx, block->nb_cpml>0?block->nb_cpml:1);
				allocate_float_vector_on_node(&block->phiv_handle[7], &block->phivzy, block->nb_cpml>0?block->nb_cpml:1);
				allocate_float_vector_on_node(&block->phiv_handle[8], &block->phivzz, block->nb_cpml>0?block->nb_cpml:1);

				allocate_float_vector_on_node(&block->phit_handle[0], &block->phitxxx, block->nb_cpml>0?block->nb_cpml:1);
				allocate_float_vector_on_node(&block->phit_handle[1], &block->phitxyy, block->nb_cpml>0?block->nb_cpml:1);
				allocate_float_vector_on_node(&block->phit_handle[2], &block->phitxzz, block->nb_cpml>0?block->nb_cpml:1);
				allocate_float_vector_on_node(&block->phit_handle[3], &block->phitxyx, block->nb_cpml>0?block->nb_cpml:1);
				allocate_float_vector_on_node(&block->phit_handle[4], &block->phityyy, block->nb_cpml>0?block->nb_cpml:1);
				allocate_float_vector_on_node(&block->phit_handle[5], &block->phityzz, block->nb_cpml>0?block->nb_cpml:1);
				allocate_float_vector_on_node(&block->phit_handle[6], &block->phitxzx, block->nb_cpml>0?block->nb_cpml:1);
				allocate_float_vector_on_node(&block->phit_handle[7], &block->phityzy, block->nb_cpml>0?block->nb_cpml:1);
				allocate_float_vector_on_node(&block->phit_handle[8], &block->phitzzz, block->nb_cpml>0?block->nb_cpml:1);

				// indirection xyz -> num cplm
				allocate_int_vector_on_node(&block->npml_tab_handle, &block->npml_tab, block->sizex*block->sizey*block->sizez);
			}

			/* Boundary blocks : X+ */ 
			// DGN : ATTENTION : boundaries non allouées pour blocs aux bords du domaine !!!
			if (block->boundary_blocks[XP]) {
				unsigned xp_node = block->boundary_blocks[XP]->mpi_node;
				if ((node == rank) || (xp_node == rank))
				{
					// DGN : attention à l'ordre d'alignement des blocs : on respecte toujours l'ordre x puis y , puis z
					allocate_block_on_node(&block->v_boundaries_handle[XP][0], &block->v_boundaries[XP][0], K, block->pitchy, block->pitchz);
					allocate_block_on_node(&block->v_boundaries_handle[XP][1], &block->v_boundaries[XP][1], K, block->pitchy, block->pitchz);
					allocate_block_on_node(&block->v_boundaries_handle[XP][2], &block->v_boundaries[XP][2], K, block->pitchy, block->pitchz);
					
					allocate_block_on_node(&block->t_boundaries_handle[XP][0], &block->t_boundaries[XP][0], K, block->pitchy, block->pitchz);
					allocate_block_on_node(&block->t_boundaries_handle[XP][1], &block->t_boundaries[XP][1], K, block->pitchy, block->pitchz);
					allocate_block_on_node(&block->t_boundaries_handle[XP][2], &block->t_boundaries[XP][2], K, block->pitchy, block->pitchz);
					allocate_block_on_node(&block->t_boundaries_handle[XP][3], &block->t_boundaries[XP][3], K, block->pitchy, block->pitchz);
					allocate_block_on_node(&block->t_boundaries_handle[XP][4], &block->t_boundaries[XP][4], K, block->pitchy, block->pitchz);
					allocate_block_on_node(&block->t_boundaries_handle[XP][5], &block->t_boundaries[XP][5], K, block->pitchy, block->pitchz);
				}
			} else { /*on domain border, allocate with size=1 to avoid null pointers*/
				if (node == rank) {
					allocate_block_on_node_spec(&block->v_boundaries_handle[XP][0], &block->v_boundaries[XP][0], 1, 1, 1);
					allocate_block_on_node_spec(&block->v_boundaries_handle[XP][1], &block->v_boundaries[XP][1], 1, 1, 1);
					allocate_block_on_node_spec(&block->v_boundaries_handle[XP][2], &block->v_boundaries[XP][2], 1, 1, 1);
					
					allocate_block_on_node_spec(&block->t_boundaries_handle[XP][0], &block->t_boundaries[XP][0], 1, 1, 1);
					allocate_block_on_node_spec(&block->t_boundaries_handle[XP][1], &block->t_boundaries[XP][1], 1, 1, 1);
					allocate_block_on_node_spec(&block->t_boundaries_handle[XP][2], &block->t_boundaries[XP][2], 1, 1, 1);
					allocate_block_on_node_spec(&block->t_boundaries_handle[XP][3], &block->t_boundaries[XP][3], 1, 1, 1);
					allocate_block_on_node_spec(&block->t_boundaries_handle[XP][4], &block->t_boundaries[XP][4], 1, 1, 1);
					allocate_block_on_node_spec(&block->t_boundaries_handle[XP][5], &block->t_boundaries[XP][5], 1, 1, 1);
				}
			}

			/* Boundary blocks : X- */
			if (block->boundary_blocks[XM]) {
				unsigned xm_node = block->boundary_blocks[XM]->mpi_node;
				if ((node == rank) || (xm_node == rank))
				{
					allocate_block_on_node(&block->v_boundaries_handle[XM][0], &block->v_boundaries[XM][0], K, block->pitchy, block->pitchz);
					allocate_block_on_node(&block->v_boundaries_handle[XM][1], &block->v_boundaries[XM][1], K, block->pitchy, block->pitchz);
					allocate_block_on_node(&block->v_boundaries_handle[XM][2], &block->v_boundaries[XM][2], K, block->pitchy, block->pitchz);
					
					allocate_block_on_node(&block->t_boundaries_handle[XM][0], &block->t_boundaries[XM][0], K, block->pitchy, block->pitchz);
					allocate_block_on_node(&block->t_boundaries_handle[XM][1], &block->t_boundaries[XM][1], K, block->pitchy, block->pitchz);
					allocate_block_on_node(&block->t_boundaries_handle[XM][2], &block->t_boundaries[XM][2], K, block->pitchy, block->pitchz);
					allocate_block_on_node(&block->t_boundaries_handle[XM][3], &block->t_boundaries[XM][3], K, block->pitchy, block->pitchz);
					allocate_block_on_node(&block->t_boundaries_handle[XM][4], &block->t_boundaries[XM][4], K, block->pitchy, block->pitchz);
					allocate_block_on_node(&block->t_boundaries_handle[XM][5], &block->t_boundaries[XM][5], K, block->pitchy, block->pitchz);
				} 
			} else {
				if (node == rank) {/*on domain border, allocate with size=1 to avoid null pointers*/
					allocate_block_on_node_spec(&block->v_boundaries_handle[XM][0], &block->v_boundaries[XM][0], 1, 1, 1);
					allocate_block_on_node_spec(&block->v_boundaries_handle[XM][1], &block->v_boundaries[XM][1], 1, 1, 1);
					allocate_block_on_node_spec(&block->v_boundaries_handle[XM][2], &block->v_boundaries[XM][2], 1, 1, 1);
					
					allocate_block_on_node_spec(&block->t_boundaries_handle[XM][0], &block->t_boundaries[XM][0], 1, 1, 1);
					allocate_block_on_node_spec(&block->t_boundaries_handle[XM][1], &block->t_boundaries[XM][1], 1, 1, 1);
					allocate_block_on_node_spec(&block->t_boundaries_handle[XM][2], &block->t_boundaries[XM][2], 1, 1, 1);
					allocate_block_on_node_spec(&block->t_boundaries_handle[XM][3], &block->t_boundaries[XM][3], 1, 1, 1);
					allocate_block_on_node_spec(&block->t_boundaries_handle[XM][4], &block->t_boundaries[XM][4], 1, 1, 1);
					allocate_block_on_node_spec(&block->t_boundaries_handle[XM][5], &block->t_boundaries[XM][5], 1, 1, 1);
				}
			}


			/* Boundary blocks : Y+ */
			if (block->boundary_blocks[YP]) {
				unsigned yp_node = block->boundary_blocks[YP]->mpi_node;
				if ((node == rank) || (yp_node == rank))
				{
					// DGN : attention à l'ordre d'alignement des blocs : on respecte toujours l'ordre x puis y , puis z
					allocate_block_on_node(&block->v_boundaries_handle[YP][0], &block->v_boundaries[YP][0], block->pitchx, K, block->pitchz);
					allocate_block_on_node(&block->v_boundaries_handle[YP][1], &block->v_boundaries[YP][1], block->pitchx, K, block->pitchz);
					allocate_block_on_node(&block->v_boundaries_handle[YP][2], &block->v_boundaries[YP][2], block->pitchx, K, block->pitchz);
					
					allocate_block_on_node(&block->t_boundaries_handle[YP][0], &block->t_boundaries[YP][0], block->pitchx, K, block->pitchz);
					allocate_block_on_node(&block->t_boundaries_handle[YP][1], &block->t_boundaries[YP][1], block->pitchx, K, block->pitchz);
					allocate_block_on_node(&block->t_boundaries_handle[YP][2], &block->t_boundaries[YP][2], block->pitchx, K, block->pitchz);
					allocate_block_on_node(&block->t_boundaries_handle[YP][3], &block->t_boundaries[YP][3], block->pitchx, K, block->pitchz);
					allocate_block_on_node(&block->t_boundaries_handle[YP][4], &block->t_boundaries[YP][4], block->pitchx, K, block->pitchz);
					allocate_block_on_node(&block->t_boundaries_handle[YP][5], &block->t_boundaries[YP][5], block->pitchx, K, block->pitchz);
				} 
			} else {
				if (node == rank) {/*on domain border, allocate with size=1 to avoid null pointers*/
					allocate_block_on_node(&block->v_boundaries_handle[YP][0], &block->v_boundaries[YP][0], 1, 1, 1);
					allocate_block_on_node(&block->v_boundaries_handle[YP][1], &block->v_boundaries[YP][1], 1, 1, 1);
					allocate_block_on_node(&block->v_boundaries_handle[YP][2], &block->v_boundaries[YP][2], 1, 1, 1);
					
					allocate_block_on_node(&block->t_boundaries_handle[YP][0], &block->t_boundaries[YP][0], 1, 1, 1);
					allocate_block_on_node(&block->t_boundaries_handle[YP][1], &block->t_boundaries[YP][1], 1, 1, 1);
					allocate_block_on_node(&block->t_boundaries_handle[YP][2], &block->t_boundaries[YP][2], 1, 1, 1);
					allocate_block_on_node(&block->t_boundaries_handle[YP][3], &block->t_boundaries[YP][3], 1, 1, 1);
					allocate_block_on_node(&block->t_boundaries_handle[YP][4], &block->t_boundaries[YP][4], 1, 1, 1);
					allocate_block_on_node(&block->t_boundaries_handle[YP][5], &block->t_boundaries[YP][5], 1, 1, 1);
				}
			}

			/* Boundary blocks : Y- */
			if (block->boundary_blocks[YM]) {
				unsigned ym_node = block->boundary_blocks[YM]->mpi_node;
				if ((node == rank) || (ym_node == rank))
				{
					allocate_block_on_node(&block->v_boundaries_handle[YM][0], &block->v_boundaries[YM][0], block->pitchx, K, block->pitchz);
					allocate_block_on_node(&block->v_boundaries_handle[YM][1], &block->v_boundaries[YM][1], block->pitchx, K, block->pitchz);
					allocate_block_on_node(&block->v_boundaries_handle[YM][2], &block->v_boundaries[YM][2], block->pitchx, K, block->pitchz);
					
					allocate_block_on_node(&block->t_boundaries_handle[YM][0], &block->t_boundaries[YM][0], block->pitchx, K, block->pitchz);
					allocate_block_on_node(&block->t_boundaries_handle[YM][1], &block->t_boundaries[YM][1], block->pitchx, K, block->pitchz);
					allocate_block_on_node(&block->t_boundaries_handle[YM][2], &block->t_boundaries[YM][2], block->pitchx, K, block->pitchz);
					allocate_block_on_node(&block->t_boundaries_handle[YM][3], &block->t_boundaries[YM][3], block->pitchx, K, block->pitchz);
					allocate_block_on_node(&block->t_boundaries_handle[YM][4], &block->t_boundaries[YM][4], block->pitchx, K, block->pitchz);
					allocate_block_on_node(&block->t_boundaries_handle[YM][5], &block->t_boundaries[YM][5], block->pitchx, K, block->pitchz);
				} 
			} else {
				if (node == rank) {/*on domain border, allocate with size=1 to avoid null pointers*/
					allocate_block_on_node(&block->v_boundaries_handle[YM][0], &block->v_boundaries[YM][0], 1, 1, 1);
					allocate_block_on_node(&block->v_boundaries_handle[YM][1], &block->v_boundaries[YM][1], 1, 1, 1);
					allocate_block_on_node(&block->v_boundaries_handle[YM][2], &block->v_boundaries[YM][2], 1, 1, 1);
					
					allocate_block_on_node(&block->t_boundaries_handle[YM][0], &block->t_boundaries[YM][0], 1, 1, 1);
					allocate_block_on_node(&block->t_boundaries_handle[YM][1], &block->t_boundaries[YM][1], 1, 1, 1);
					allocate_block_on_node(&block->t_boundaries_handle[YM][2], &block->t_boundaries[YM][2], 1, 1, 1);
					allocate_block_on_node(&block->t_boundaries_handle[YM][3], &block->t_boundaries[YM][3], 1, 1, 1);
					allocate_block_on_node(&block->t_boundaries_handle[YM][4], &block->t_boundaries[YM][4], 1, 1, 1);
					allocate_block_on_node(&block->t_boundaries_handle[YM][5], &block->t_boundaries[YM][5], 1, 1, 1);
				}
			}
		}
	}
}

/* check how many cells are alive */
// DGN
void check(int rank)
{
	unsigned bx, by;
	for (bx = 0; bx < get_nbx(); bx++)
	{	for (by = 0; by < get_nby(); by++)
		{
			struct block_description *block = get_block_description(bx, by);

			unsigned node = block->mpi_node;

			/* Main blocks */
			if (node == rank)
			{
				unsigned size_bx = block_sizes_x[bx];
				unsigned size_by = block_sizes_y[by];
#ifdef LIFE
				unsigned x, y, z;
				unsigned sum = 0;
				for (x = 0; x < size_bx; x++)
					for (y = 0; y < size_by; y++)
						for (z = 0; z < sizez; z++)
							sum += block->layers[0][(K+x)+(K+y)*(get_sizex() + 2*K)+(K+z)*(get_sizex()+2*K)*(get_sizey()+2*K)];
				printf("block %d got %d/%d alive\n", IND(bx,by), sum, size_bx*size_by*sizez);
#endif
			}
		}
	}
}
