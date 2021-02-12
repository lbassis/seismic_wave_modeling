// THIS HEADER CONTAINS STUBS OF KERNEL WRAPPERS

#ifndef ONDES3D_KERNELS_H
#define ONDES3D_KERNELS_H

#include "ondes3D-common.h"

#define NPPDX 16
#define NPPDY 8
#define NPPDX_K2 16
#define NPPDY_K2 8

#define VOFF(I,J) (int)(((int)threadIdx.y+2+(J))*(NPPDX+4) + (int)threadIdx.x+2+(I))
#define TOFF(I,J) (int)(((int)threadIdx.y+2+(J))*(NPPDX+4) + (int)threadIdx.x+2+(I))
#define MUOFF(I,J) (int)(((int)threadIdx.y+(J))*(NPPDX+1) + (int)threadIdx.x+(I))
#define RHOFF(I,J) (int)(((int)threadIdx.y+(J))*(NPPDX+1) + (int)threadIdx.x+(I))
#define VPOFF(I,J) (int)(((int)threadIdx.y+(J))*(NPPDX+1) + (int)threadIdx.x+(I))
#define LAMOFF(I,J) (int)(((int)threadIdx.y+(J))*(NPPDX+1) + (int)threadIdx.x+(I))

#ifdef DEVICE_SIDE_INCLUDE
extern "C"
{
#endif
            

   void cuda_compute_stress_host (  float* d_txx0, float* d_tyy0, float* d_tzz0, float* d_txy0, float* d_txz0, float* d_tyz0,
            float* d_vx0, float* d_vy0, float* d_vz0,
            int* d_npml_tab, float* d_phivxx, float* d_phivxy, float* d_phivxz, float* d_phivyx, float* d_phivyy, float* d_phivyz, float* d_phivzx, float* d_phivzy, float* d_phivzz, 
            float* d_mu, float* d_lam, float* d_vp, 
            int sizex, int sizey, int sizez,
            int pitch_x, int pitch_y, int pitch_z, 
            float ds, float dt, int delta, int position,
            int grid_x, int grid_y, int grid_z, int block_x, int block_y, int block_z);


   void cuda_compute_veloc_host (   float* d_txx0, float* d_tyy0, float* d_tzz0, float* d_txy0, float* d_txz0, float* d_tyz0,
            float* d_vx0, float* d_vy0, float* d_vz0,
            float* d_fx, float* d_fy, float* d_fz, 
            int* d_npml_tab, float* d_phitxxx, float* d_phitxyy, float* d_phitxzz, float *d_phitxyx, float *d_phityyy, float *d_phityzz, float *d_phitxzx, float *d_phityzy, float *d_phitzzz,
            float* d_vp, float* d_rho,
            int sizex, int sizey, int sizez,
            int pitch_x, int pitch_y, int pitch_z, 
            float ds, float dt, int delta, int position,
            int grid_x, int grid_y, int grid_z, int block_x, int block_y, int block_z);

   void setConstCPMLValue(float d0, float k0, float a0);

   void setConstCPMLIndices(int ixe_min_, int ixs_max_, int iye_min_, int iys_max_);

   void copyBlockBoundary(	float* block, float* boundary,
						unsigned size_x, unsigned size_y, unsigned size_z, 
						bool aligned, unsigned padding, unsigned direction, unsigned sens);

#ifdef DEVICE_SIDE_INCLUDE
}
#endif
#endif
