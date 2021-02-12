// INCLUDES {{{
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <limits.h>

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef USE_MPI/*{{{*/
#include "mpi.h"
#include "topo_mpi.h"
#endif/*}}}*/
// }}}


// DEFINES {{{
// SWITCHES
#define COMPUTE_SEISMOS

// HISTORICAL
#ifdef NOCPML
#define delta 0
#else
#define delta 10
#endif
#define reflect 0.001
#define f0 1
#define un 1.0

// MPI
#define MASTER 0
#ifndef USE_MPI/*{{{*/
#define NPROCX 1
#define NPROCY 1
#endif/*}}}*/

// CUDA & PERF
// for alignment align_segment = (128/sizeof(float)), ie 32
// this can be reduced to 16 to save memory with a little loss of performance (~1%). Not other value than 16 or 32.
#define ALIGN_SEGMENT 16
#define BLOCKDIMX 64
#define SRC_PERF_CRITERIA 10

// MISC
#define MAX_LINE_SIZE 200

// DEBUG
#define DECALAGE 10
// 0 : muet, 1 : peu dissert, 2 : dissert, 3 : debug
#define VERBOSE 1
#define DEBUG_OUT if (comms.rank == MASTER) fprintf(stderr,"OK line %d\n",__LINE__);
// }}}

// MACROS {{{
// facilitate access to arrays with alignment & padding
#define ACCESS(name,tab,i,j,k) name.tab[((k)-name.offset_z)*name.pitch*name.height+((j)-name.offset_y)*name.pitch+((i)-name.offset_x)+name.offset] 
#define MIN(a,b) ((a<b)?a:b)
#define MAX(a,b) ((a>b)?a:b)
// }}}

// STRUCTURES {{{
typedef struct {
	float* x;
	float* y;
	float* z;
	int width,height,depth,pitch,offset, offset_k;	
	int offset_x, offset_y, offset_z;
} Vector_M3D;

typedef struct {
	float* xx;
	float* yy;
	float* zz;
	float* xy;
	float* xz;
	float* yz;
	int width,height,depth,pitch,offset, offset_k;	
	int offset_x, offset_y, offset_z;
} Tensor_M3D;

typedef struct {
	float* mu;
	float* rho;
	float* lam;
	float* vp;
	float* vs;
	int width,height,depth,pitch,offset, offset_k;	
	int offset_x, offset_y, offset_z;
} Material;

typedef struct {
	// MPI rank
	int rank,nbprocs,nproc_x,nproc_y;

	// receivers
	int recv_xmin, recv_xmax, recv_ymin, recv_ymax;

	// senders
	int send_xmin, send_xmax, send_ymin, send_ymax;

	// buffers
	// host : send
	float* buff_x_min_s;
	float* buff_y_min_s;
	float* buff_x_max_s;
	float* buff_y_max_s;
	// host : recv
	float* buff_x_min_r;
	float* buff_y_min_r;
	float* buff_x_max_r;
	float* buff_y_max_r;

	// device : send+recv
	float* d_buff_x_min;
	float* d_buff_y_min;
	float* d_buff_x_max;
	float* d_buff_y_max;

	// sizes
	int size_buffer_x;
	int size_buffer_y;

#ifdef USE_MPI
	// MPI Requests
	MPI_Request array_req_send[4];
	MPI_Request array_req_recv[4];

	MPI_Status array_of_status[4];
#endif
	// position
	int first_x, last_x, first_y, last_y;
} Comm_info;

typedef struct {
	int xmin,xmax,ymin,ymax,zmin,zmax;
	int xinf,xsup,yinf,ysup,zinf,zsup;
	int size_x, size_y, size_z;
} Boundaries;
// }}}

// FUNCTIONS {{{
// Read params
void chomp(const char *s);
void readIntParam(char* entry, int* out, FILE* fd);
void readFloatParam(char* entry, float* out, FILE* fd);
void readStringParam(char* entry, char* out, FILE* fd);
void readLayerParam(char* entry, int num, float* depth, float* vp, float* vs, float* rho, float* q, FILE* fd);

// Print cuda errors
void print_err(cudaError_t err, char* err_str);

// CPU & GPU arrays
void allocate_arrays_CPU(Material *M, Vector_M3D *F, Boundaries *B, int dim_model);
long int allocate_arrays_GPU(Vector_M3D *V, Tensor_M3D *T, Material *M, Vector_M3D *F, Boundaries *B, int dim_model);
void copy_material_to_GPU(Material *d_M, Material *M, int dim_model);
void free_arrays_CPU(Material *M, Vector_M3D *F, int dim_model);
void free_arrays_GPU(Vector_M3D *V, Tensor_M3D *T, Material *M, Vector_M3D *F, int *d_npml_tab, int dim_model);

// CPMLs
int create_CPML_indirection(int **p_d_npml_tab, Boundaries B, int* npmlv, int sizex, int sizey, int sizez);
long int allocate_CPML_arrays(int npmlv, float **p_d_phivxx, float **p_d_phivxy, float **p_d_phivxz, float **p_d_phivyx, float **p_d_phivyy, float **p_d_phivyz, float **p_d_phivzx, float **p_d_phivzy, float **p_d_phivzz, 
		float **p_d_phitxxx, float **p_d_phitxyy, float **p_d_phitxzz, float **p_d_phitxyx, float **p_d_phityyy, float **p_d_phityzz, float **p_d_phitxzx, float **p_d_phityzy, float **p_d_phitzzz);
long int allocate_CPML_vectors(Boundaries B,	float** p_d_dumpx, float** p_d_alphax, float** p_d_kappax, float** p_d_dumpx2, float** p_d_alphax2, float** p_d_kappax2,
		float** p_d_dumpy, float** p_d_alphay, float** p_d_kappay, float** p_d_dumpy2, float** p_d_alphay2, float** p_d_kappay2,
		float** p_d_dumpz, float** p_d_alphaz, float** p_d_kappaz, float** p_d_dumpz2, float** p_d_alphaz2, float** p_d_kappaz2,
		float* dumpx, float* alphax, float* kappax, float* dumpx2, float* alphax2, float* kappax2,
		float* dumpy, float* alphay, float* kappay, float* dumpy2, float* alphay2, float* kappay2,
		float* dumpz, float* alphaz, float* kappaz, float* dumpz2, float* alphaz2, float* kappaz2);
void free_CPML_data(float *d_phivxx, float *d_phivxy, float *d_phivxz, float *d_phivyx, float *d_phivyy, float *d_phivyz, float *d_phivzx, float *d_phivzy, float *d_phivzz, 
		float *d_phitxxx, float *d_phitxyy, float *d_phitxzz, float *d_phitxyx, float *d_phityyy, float *d_phityzz, float *d_phitxzx, float *d_phityzy, float *d_phitzzz,
		float* d_dumpx, float* d_alphax, float* d_kappax, float* d_dumpx2, float* d_alphax2, float* d_kappax2,
		float* d_dumpy, float* d_alphay, float* d_kappay, float* d_dumpy2, float* d_alphay2, float* d_kappay2,
		float* d_dumpz, float* d_alphaz, float* d_kappaz, float* d_dumpz2, float* d_alphaz2, float* d_kappaz2);

// MPI
void printTopo(void);
void MPI_slicing_and_addressing (	Comm_info *C, Boundaries *B, int XMIN, int XMAX, int YMIN, int YMAX, int ZMIN, int ZMAX);
long int allocate_MPI_buffers(Comm_info *C);
void free_MPI_buffers(Comm_info *C);
// }}}
