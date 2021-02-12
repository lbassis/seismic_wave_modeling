/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
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

#ifndef __STENCIL_H__
#define __STENCIL_H__

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <time.h>

#include <starpu.h>
#include <starpu_top.h>
#ifdef STARPU_USE_CUDA
#include <starpu_cuda.h>
#endif

#ifndef ACTUALLY_USE_MPI
#undef STARPU_USE_MPI
#else
#include <mpi.h>
#endif


#ifdef STARPU_USE_MPI
#include <starpu_mpi.h>
#endif

#include "ondes3D-common.h"
#include <stdlib.h>
// DGN DEBUG
// #undef STARPU_USE_CUDA

// use Cuda kernel instead of cudaMemCopy() for blocks/boundaries copies
//#define USE_KERNEL_FOR_COPY


#define STRMAX 200
#define PRM "./essai.prm"
#define MASTER 0

// historical
#define reflect 0.001
#define f0 1
#define un 1.0

#define bool int
#define true 1
#define false 0

#define VERBOSE 1

#define PI ((double)(acosf(-1.0)))

#define RAW_ACCESS(block,i,j,k) ((k)*block->pitchx*block->pitchy + (j)*block->pitchx + (i))

#define ACCESS(block,i,j,k) ((k)*block->pitchx*block->pitchy + (j)*block->pitchx + (i) + block->offset)

#define STARPU_MALLOC


// DGN
#define NDIRS 4
#define VDIMS 3
#define TDIMS 6
#define CDIMS 9

extern struct starpu_top_data* starpu_top_init_loop;
extern struct starpu_top_data* starpu_top_achieved_loop;


/* Split only on x & y axis */
// DGN
#ifdef STARPU_USE_MPI
#define STOP MPI_Abort(MPI_COMM_WORLD, 1); 
#define NPROCX 1
#define NPROCY 1
#else
#define STOP exit(1);
#define NPROCX 1
#define NPROCY 1
#endif 

#define NODBUG
#ifndef NODBUG
#define DGN_DBG printf("@@@@@@@@@@@@@@@ file %s, function %s, line %d @@@@@@@@@@@@@@@\n", __FILE__, __FUNCTION__, __LINE__);fflush(stdout);
#define DGN_DBUG(a,b,c) printf("@@@@@@@@@@@@@@@ file %s, function %s, line %d, a=%d, b=%d, c=%d @@@@@@@@@@@@@@@\n", __FILE__, __FUNCTION__, __LINE__, a, b, c);
#else 
#define DGN_DBG ;
#define DGN_DBUG(a,b,c) ;
#endif

#if 0
# define DEBUG(fmt, ...) fprintf(stdout,fmt,##__VA_ARGS__)
#else
# define DEBUG(fmt, ...) (void) 0
#endif

#if 1
# define INFO(fmt, ...) fprintf(stdout,fmt,##__VA_ARGS__)
#else
# define INFO(fmt, ...) (void) 0
#endif

#define ENABLE_VERY_SLOW_ERROR_CHECKING

typedef enum
{  
	VELOC = 0,
	STRESS = 1
} data_type;

// give corresponding neighbour buffer for direction: anti[XP] = XM
extern unsigned anti[4];

/*simulation parameters*/
typedef struct {
   float ds, dt, fd;
   char src_file[STRMAX]; /*sources*/
   char stf_file[STRMAX]; /*sources time functions*/
   char sta_file[STRMAX];  /*stations*/
   char outdir[STRMAX];   /*output directory*/
   int tmax;       /*time steps*/
   int xmin, xmax, ymin, ymax, zmin; /*domain*/
   unsigned sizex, sizey, sizez;
   int ndim;

   // modele 1D
   int nlayer;
   float* laydep;
   float* vp0;
   float* vs0;
   float* rho0;
   float* q0;

   float dump0;
   float alpha0;
   float kappa0;

   // sources
   int nb_src;

   float xhypo0, yhypo0, zhypo0;
   int ixhypo0, iyhypo0, izhypo0;

   int *ixhypo, *iyhypo, *izhypo;
   int *insrc;

   float *xhypo, *yhypo, *zhypo;
   double* strike;
   double* dip;
   double* rake;
   float* slip;
   float *xweight, *yweight, *zweight;

   // sources time function
   float dsbiem;
   float dtbiem;
   int idur;
   double* vel; /*vel = dmatrix(0, ISRC-1, 0, idur-1);*/

   // stations
   int iobs;
   
   int* nobs;
   float* xobs;
   float* yobs;
   float* zobs;
   int* ixobs;
   int* iyobs;
   int* izobs;

   float* xobswt;
   float* yobswt;
   float* zobswt;

} ondes3d_params;

/* Description of a domain block */
struct block_description
{
	/* Which MPI node should process that block ? */
	unsigned mpi_node;
	
	unsigned preferred_worker;

	// coord du bloc
	unsigned bx, by;
	int xshift, yshift;

	// nb sources in the block
	unsigned nb_src;
	unsigned nb_sta;
   unsigned* insrc;
   // peut etre merdé là ...
   int* ista;

   int iter;

	// dimensions du bloc
	unsigned sizex, sizey, sizez;
	unsigned pitchx, pitchy, pitchz;
	unsigned offset;
   unsigned padding;

	/* includes neighbours' border to make computation easier */
	/*velocity*/
	float* vx;
	float* vy;
	float* vz;
	starpu_data_handle_t velocity_handle[VDIMS];

	/*stress*/
	float* txx;
	float* tyy;
	float* tzz;
	float* txy;
	float* txz;
	float* tyz;
	starpu_data_handle_t stress_handle[TDIMS];

	/* This is the "save" buffer, i.e. a copy of our neighbour's border.
	 * This one is used for CPU/GPU or MPI communication (rather than the
	 * whole domain block) */
	float* v_boundaries[NDIRS][VDIMS];
	starpu_data_handle_t v_boundaries_handle[NDIRS][VDIMS];

	float* t_boundaries[NDIRS][TDIMS];
	starpu_data_handle_t t_boundaries_handle[NDIRS][TDIMS];

	/*materials*/
	float* mu;
	starpu_data_handle_t mu_handle;
	float* rho;
	starpu_data_handle_t rho_handle;
	float* lam;
	starpu_data_handle_t lam_handle;
	float* vp;
	starpu_data_handle_t vp_handle;
	float* vs;
	starpu_data_handle_t vs_handle;

	/*seismic source momentum*/
	float* fx;
	float* fy;
	float* fz;
	starpu_data_handle_t force_handle[VDIMS];

   /*cpmls arrays [RW]*/
	unsigned nb_cpml;

	float* phivxx;
	float* phivxy;
	float* phivxz;
	float* phivyx;
	float* phivyy;
	float* phivyz;
	float* phivzx;
	float* phivzy;
	float* phivzz;
	starpu_data_handle_t phiv_handle[CDIMS];

	float* phitxxx;
	float* phitxyy;
	float* phitxzz;
	float* phitxyx;
	float* phityyy;
	float* phityzz;
	float* phitxzx;
	float* phityzy;
	float* phitzzz;
	starpu_data_handle_t phit_handle[CDIMS];

	// indirection xyz -> num cplm
	int* npml_tab;
	starpu_data_handle_t npml_tab_handle;

	/* Shortcut pointer to the neighbours */
	struct block_description *boundary_blocks[NDIRS];

	/*sismos*/ // NON ALOUES !!!
	double* seisx;
	double* seisy;
	double* seisz;
	double* seisxx;
	double* seisyy;
	double* seiszz;
	double* seisxy;
	double* seisxz;
	double* seisyz;

	starpu_data_handle_t seismo_handle[9];

	/* Shortcut pointer to the simulation parameters */
	ondes3d_params* params;
};

typedef struct {
	struct block_description* block;
	unsigned iter;
} block_iter_arg_t;

#define TAG_INIT_TASK			((starpu_tag_t)1)

starpu_tag_t TAG_FINISH(int x, int y);
starpu_tag_t TAG_START(int x, int y, int dir);
int MPI_TAG0(int x, int y, int iter, int dir);
int MPI_TAG1(int x, int y, int iter, int dir);
int MPI_TAG2(int x, int y, int iter, int dir);
int MPI_TAG3(int x, int y, int iter, int dir);
int MPI_TAG4(int x, int y, int iter, int dir);
int MPI_TAG5(int x, int y, int iter, int dir);

#define IND(x,y)  ((y)*get_nbx()+(x))

// DGN
void create_blocks_array(unsigned sizex, unsigned sizey, unsigned sizez, unsigned nbx, unsigned nby, unsigned bsizex, unsigned bsizey, ondes3d_params* params);
unsigned compute_nb_cpml(unsigned _sizex, unsigned _sizey, unsigned _sizez, unsigned bx, unsigned by, unsigned thisbsizex, unsigned thisbsizey, unsigned bsizex, unsigned bsizey, unsigned delta);
// DGN
struct block_description *get_block_description(int x, int y);
unsigned get_nbx(void);
unsigned get_nby(void);
unsigned get_sizex(void);
unsigned get_sizey(void);
unsigned get_sizez(void);


void allocate_dble_vector_on_node(starpu_data_handle_t *handleptr, double **ptr, unsigned nx);

void assign_blocks_to_mpi_nodes(int world_size);
void allocate_memory_on_node(int rank);

// DGN
void create_cpml_indirection(int rank);
void assign_blocks_to_workers(int rank);
void create_tasks(int rank);
void wait_end_tasks(int rank);
void check(int rank);

void display_memory_consumption(int rank);

// DGN
int get_block_mpi_node(int x, int y);
// DGN
unsigned get_block_size(int x, int y);
unsigned get_bind_tasks(void);

unsigned get_niter(void);
void set_niter(unsigned n);
unsigned get_ticks(void);
bool aligned(void);

unsigned global_workerid(unsigned local_workerid);


// DGN
void create_task_update_source(unsigned iter, unsigned x, unsigned y, unsigned local_rank);
void create_task_record_seismo(unsigned iter, unsigned x, unsigned y, unsigned local_rank);
void create_task_save(unsigned iter, unsigned x, unsigned y, int dir, unsigned local_rank, data_type type);
void create_task_compute_veloc(unsigned iter, unsigned x, unsigned y, unsigned local_rank);
void create_task_compute_stress(unsigned iter, unsigned x, unsigned y, unsigned local_rank);

// Read params
void chomp(const char *s);
void readIntParam(char* entry, int* out, FILE* fd);
void readFloatParam(char* entry, float* out, FILE* fd);
void readStringParam(char* entry, char* out, FILE* fd);
void readLayerParam(char* entry, int num, float* depth, float* vp, float* vs, float* rho, float* q, FILE* fd);

bool indomain(struct block_description* block, unsigned i, unsigned j, unsigned k, bool extended);
bool has_cpml(struct block_description* block, bool* bxmin, bool* bxmax, bool* bymin, bool* bymax, int* ixs_min, int* ixe_min, int* iys_min, int* iye_min, int* ixs_max, int* ixe_max, int* iys_max, int* iye_max);

void read_parameter_file(ondes3d_params* par, char* param_file, int rank);
void read_sources_positions(ondes3d_params* par, int rank);
void read_source_time_function(ondes3d_params* par, int rank);
void read_stations_positions(ondes3d_params* par, int rank);
void set_cpmls(ondes3d_params* par, int rank);
void set_material_properties(ondes3d_params* par, int rank);

ondes3d_params* get_params(void);


extern int starpu_mpi_initialize(void);
extern int starpu_mpi_shutdown(void);

/* kernels DGN tout ça a revoir et corriger !!!*/
void set_dump0(float val);
void set_kappa0(float val);
void set_alpha0(float val);

extern struct starpu_codelet cl_compute_veloc;
extern struct starpu_codelet cl_compute_stress;
extern struct starpu_codelet cl_update_source;
extern struct starpu_codelet cl_record_seismo;

extern struct starpu_codelet save_veloc_xp_cl;
extern struct starpu_codelet save_veloc_xm_cl;
extern struct starpu_codelet save_veloc_yp_cl;
extern struct starpu_codelet save_veloc_ym_cl;
extern struct starpu_codelet save_stress_xp_cl;
extern struct starpu_codelet save_stress_xm_cl;
extern struct starpu_codelet save_stress_yp_cl;
extern struct starpu_codelet save_stress_ym_cl;

extern unsigned veloc_xp_per_worker[STARPU_NMAXWORKERS];
extern unsigned veloc_xm_per_worker[STARPU_NMAXWORKERS];
extern unsigned veloc_yp_per_worker[STARPU_NMAXWORKERS];
extern unsigned veloc_ym_per_worker[STARPU_NMAXWORKERS];
extern unsigned stress_xp_per_worker[STARPU_NMAXWORKERS];
extern unsigned stress_xm_per_worker[STARPU_NMAXWORKERS];
extern unsigned stress_yp_per_worker[STARPU_NMAXWORKERS];
extern unsigned stress_ym_per_worker[STARPU_NMAXWORKERS];

extern unsigned veloc_update_per_worker[STARPU_NMAXWORKERS];
extern unsigned stress_update_per_worker[STARPU_NMAXWORKERS];

extern float time_spent_veloc_xp[STARPU_NMAXWORKERS];
extern float time_spent_veloc_xm[STARPU_NMAXWORKERS];
extern float time_spent_veloc_yp[STARPU_NMAXWORKERS];
extern float time_spent_veloc_ym[STARPU_NMAXWORKERS];
extern float time_spent_stress_xp[STARPU_NMAXWORKERS];
extern float time_spent_stress_xm[STARPU_NMAXWORKERS];
extern float time_spent_stress_yp[STARPU_NMAXWORKERS];
extern float time_spent_stress_ym[STARPU_NMAXWORKERS];

extern float time_spent_veloc_update[STARPU_NMAXWORKERS];
extern float time_spent_stress_update[STARPU_NMAXWORKERS];

extern struct timeval start;
extern int who_runs_what_len;
extern int *who_runs_what;
extern int *who_runs_what_index;
extern struct timeval *last_tick;

// cuda kernels
#include "ondes3D-kernels.h"

void cpu_compute_veloc (	float* d_txx, float* d_tyy, float* d_tzz, float* d_txy, float* d_txz, float* d_tyz,
									float* d_vx, float* d_vy, float* d_vz,
									float* d_fx, float* d_fy, float* d_fz, 
									int* d_npml_tab, float* d_phitxxx, float* d_phitxyy, float* d_phitxzz, float *d_phitxyx, float *d_phityyy, float *d_phityzz, float *d_phitxzx, float *d_phityzy, float *d_phitzzz,
									float* d_vp, float* d_rho,
									int sizex, int sizey, int sizez,
									int pitch_x, int pitch_y, int pitch_z, 
									float ds, float dt, int delta, int position,
									int ixe_min_, int ixs_max_, int iye_min_, int iys_max_, float dump0, float kappa0, float alpha0
								);

void cpu_compute_stress (  float* txx0, float* tyy0, float* tzz0, float* txy0, float* txz0, float* tyz0,
                           float* vx0, float* vy0, float* vz0,
                           int* npml_tab, float* phivxx, float* phivxy, float* phivxz, float* phivyx, float* phivyy, float* phivyz, float* phivzx, float* phivzy, float* phivzz, 
                           float* mu, float* lam, float* vp, 
                           int sizex, int sizey, int sizez,
                           int pitch_x, int pitch_y, int pitch_z, 
                           float ds, float dt, int delta, int position,
                           int ixe_min, int ixs_max, int iye_min, int iys_max, float dump0, float kappa0, float alpha0, int iter
                        );

float staggardv4 (float b, float kappax, float kappay, float kappaz, float dt, float dx,
	float x1, float x2, float x3, float x4,
	float y1, float y2, float y3, float y4,
	float z1, float z2, float z3, float z4 );

float staggardv2 (float b, float kappax, float kappay, float kappaz, float dt, float dx,
	float x1, float x2,
	float y1, float y2,
	float z1, float z2 );

float staggards4 (float lam, float mu, float kappax, float kappay, float kappaz, float dt, float dx,
	float x1, float x2, float x3, float x4,
	float y1, float y2, float y3, float y4,
	float z1, float z2, float z3, float z4 );

float staggards2 (float lam, float mu, float kappax, float kappay, float kappaz, float dt, float dx,
	float x1, float x2,
	float y1, float y2,
	float z1, float z2 );

float staggardt4 (float mu, float kappax, float kappay, float dt, float dx,
	float x1, float x2, float x3, float x4,
	float y1, float y2, float y3, float y4 );

float staggardt2 (float mu, float kappax, float kappay, float dt, float dx,
	float x1, float x2,
	float y1, float y2 );

float CPML4 (float vp, float dump, float alpha, float kappa, float phidum, float dx, float dt,
    float x1, float x2, float x3, float x4 );

float CPML2 (float vp, float dump, float alpha, float kappa, float phidum, float dx, float dt,
    float x1, float x2 );

// DGN nr.h
int my_float2int(float x);
float hardrock(float z);
double dradxx(double strike, double dip, double rake);
double dradyy(double strike, double dip, double rake);
double dradzz(double strike, double dip, double rake);
double dradxy(double strike, double dip, double rake);
double dradyz(double strike, double dip, double rake);
double dradxz(double strike, double dip, double rake);
#endif /* __STENCIL_H__ */
