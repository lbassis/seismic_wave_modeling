// THIS HEADER CONTAINS STUBS OF KERNEL WRAPPERS

#ifndef ONDES3D_KERNELS_H
#define ONDES3D_KERNELS_H

// comment on card with compute capability < 1.2
// #define HIGH_COMPUTE_CAPABILITY
#define NPPDX 16
#define NPPDX_K2 16
#ifdef HIGH_COMPUTE_CAPABILITY
#define NPPDY 12
#define NPPDY_K2 12
#define HIGH 1
#else
#define NPPDY 8
#define NPPDY_K2 8
#define HIGH 0
#endif

// 8ko contant mem
// (int)(8*1024/4(size of float)/3(arrays)) = 682
#define CONSTANT_MAX_SIZE 682

// for debug
//#define ENABLE_VERY_SLOW_ERROR_CHECKING
#define VOFF(I,J) (int)(((int)threadIdx.y+2+(J))*(NPPDX+4) + (int)threadIdx.x+2+(I))
#define TOFF(I,J) (int)(((int)threadIdx.y+2+(J))*(NPPDX+4) + (int)threadIdx.x+2+(I))
#define MUOFF(I,J) (int)(((int)threadIdx.y+(J))*(NPPDX+1) + (int)threadIdx.x+(I))
#define RHOFF(I,J) (int)(((int)threadIdx.y+(J))*(NPPDX+1) + (int)threadIdx.x+(I))
#define VPOFF(I,J) (int)(((int)threadIdx.y+(J))*(NPPDX+1) + (int)threadIdx.x+(I))
#define LAMOFF(I,J) (int)(((int)threadIdx.y+(J))*(NPPDX+1) + (int)threadIdx.x+(I))

// uncomment to remove CPML
//#define NOCPML

#ifdef DEVICE_SIDE_INCLUDE
extern "C"
{
#endif
	void bindTexturesCpmlVector(	float* d_dumpx, float* d_alphax, float* d_kappax, float* d_dumpx2, float* d_alphax2, float* d_kappax2,
					float* d_dumpy, float* d_alphay, float* d_kappay, float* d_dumpy2, float* d_alphay2, float* d_kappay2,
					float* d_dumpz, float* d_alphaz, float* d_kappaz, float* d_dumpz2, float* d_alphaz2, float* d_kappaz2,
					int size_x, int size_y, int size_z);


	void computeStress1D (	float* d_txx0, float* d_tyy0, float* d_tzz0, float* d_txy0, float* d_txz0, float* d_tyz0,
				float* d_vx0, float* d_vy0, float* d_vz0,
				int* d_npml_tab, float* d_phivxx, float* d_phivxy, float* d_phivxz, float* d_phivyx, float* d_phivyy, float* d_phivyz, float* d_phivzx, float* d_phivzy, float* d_phivzz, 
				int sizex, int sizey, int sizez,
				int pitch_x, int pitch_y, int pitch_z, 
				float ds, float dt, int delta, int compute_external,
				int grid_x, int grid_y, int grid_z, int block_x, int block_y, int block_z, int position);


	void computeVeloc1D (	float* d_txx0, float* d_tyy0, float* d_tzz0, float* d_txy0, float* d_txz0, float* d_tyz0,
				float* d_vx0, float* d_vy0, float* d_vz0,
				float* d_fx, float* d_fy, float* d_fz, 
				int* d_npml_tab, float* d_phitxxx, float* d_phitxyy, float* d_phitxzz, float *d_phitxyx, float *d_phityyy, float *d_phityzz, float *d_phitxzx, float *d_phityzy, float *d_phitzzz,
				int sizex, int sizey, int sizez,
				int pitch_x, int pitch_y, int pitch_z, 
				float ds, float dt, int delta, int compute_external,
				int grid_x, int grid_y, int grid_z, int block_x, int block_y, int block_z, int position);
				

	void computeStress3D (	float* d_txx0, float* d_tyy0, float* d_tzz0, float* d_txy0, float* d_txz0, float* d_tyz0,
				float* d_vx0, float* d_vy0, float* d_vz0,
				int* d_npml_tab, float* d_phivxx, float* d_phivxy, float* d_phivxz, float* d_phivyx, float* d_phivyy, float* d_phivyz, float* d_phivzx, float* d_phivzy, float* d_phivzz, 
				float* d_mu, float* d_lam, float* d_vp, 
				int sizex, int sizey, int sizez,
				int pitch_x, int pitch_y, int pitch_z, 
				float ds, float dt, int delta, int compute_external,
				int grid_x, int grid_y, int grid_z, int block_x, int block_y, int block_z, int position);


	void computeVeloc3D (	float* d_txx0, float* d_tyy0, float* d_tzz0, float* d_txy0, float* d_txz0, float* d_tyz0,
				float* d_vx0, float* d_vy0, float* d_vz0,
				float* d_fx, float* d_fy, float* d_fz, 
				int* d_npml_tab, float* d_phitxxx, float* d_phitxyy, float* d_phitxzz, float *d_phitxyx, float *d_phityyy, float *d_phityzz, float *d_phitxzx, float *d_phityzy, float *d_phitzzz,
				float* d_vp, float* d_rho,
				int sizex, int sizey, int sizez,
				int pitch_x, int pitch_y, int pitch_z, 
				float ds, float dt, int delta, int compute_external,
				int grid_x, int grid_y, int grid_z, int block_x, int block_y, int block_z, int position);


	void getBuffersStress(	float* d_buff_x_min, float* d_buff_x_max, float* d_buff_y_min, float* d_buff_y_max, 
				int size_x, int size_y, int size_z, int pitch_x, int pitch_y, int pitch_z, 
				float* d_Txx, float* d_Tyy, float* d_Tzz, float* d_Txy, float* d_Txz, float* d_Tyz, int size_buf_x, int size_buf_y,
				int grid_x, int grid_y, int grid_z, int block_x, int block_y, int block_z, int position);


	void getBuffersVeloc(	float* d_buff_x_min, float* d_buff_x_max, float* d_buff_y_min, float* d_buff_y_max, 
				int size_x, int size_y, int size_z, int pitch_x, int pitch_y, int pitch_z, 
				float* d_Vx, float* d_Vy, float* d_Vz, int size_buf_x, int size_buf_y,
				int grid_x, int grid_y, int grid_z, int block_x, int block_y, int block_z, int position);


	void updateHaloStress(	float* d_buff_x_min, float* d_buff_x_max, float* d_buff_y_min, float* d_buff_y_max, 
				int size_x, int size_y, int size_z, int pitch_x, int pitch_y, int pitch_z, 
				float* d_Txx, float* d_Tyy, float* d_Tzz, float* d_Txy, float* d_Txz, float* d_Tyz, int size_buf_x, int size_buf_y,
				int grid_x, int grid_y, int grid_z, int block_x, int block_y, int block_z, int position);


	void updateHaloVeloc(	float* d_buff_x_min, float* d_buff_x_max, float* d_buff_y_min, float* d_buff_y_max, 
				int size_x, int size_y, int size_z, int pitch_x, int pitch_y, int pitch_z, 
				float* d_Vx, float* d_Vy, float* d_Vz, int size_buf_x, int size_buf_y,
				int grid_x, int grid_y, int grid_z, int block_x, int block_y, int block_z, int position);

	void cuda_device_info(char* decal);
	
	void setConstRho(float* array, int size);

	void setConstVp(float* array, int size);

	void setConstVs(float* array, int size);
#ifdef DEVICE_SIDE_INCLUDE
}
#endif
#endif
