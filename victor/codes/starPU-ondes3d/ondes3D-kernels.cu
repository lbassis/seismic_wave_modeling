#define DEVICE_SIDE_INCLUDE

#include <stdio.h>

#include <cuda_runtime.h>
#include <starpu.h>
#include "ondes3D-kernels.h"

#define ENABLE_VERY_SLOW_ERROR_CHECKING
#define OFFSET_PITCH(_nx,_ny,_i,_j,_k) ((_i)+_nx*((_j)+_ny*(_k)))

void printCudaErr(cudaError_t err, const char* where)
{   if (err != cudaSuccess) {
        printf("\nError <%s> detected %s\n",cudaGetErrorString(err),where);fflush(stdout);
    }
}

// constant mem {{{
   __constant__ float dump0;
   __constant__ float kappa0;
   __constant__ float alpha0;

   __constant__ int ixe_min;
   __constant__ int ixs_max;
   __constant__ int iye_min;
   __constant__ int iys_max;

   // setters
   void setConstCPMLValue(float d0, float k0, float a0)
   {
      printCudaErr(cudaMemcpyToSymbolAsync(dump0, &d0, sizeof(float)),__FUNCTION__);
      printCudaErr(cudaMemcpyToSymbolAsync(kappa0, &k0, sizeof(float)),__FUNCTION__);
      printCudaErr(cudaMemcpyToSymbolAsync(alpha0, &a0, sizeof(float)),__FUNCTION__);
   }

   void setConstCPMLIndices(int ixe_min_, int ixs_max_, int iye_min_, int iys_max_)
   {
      printCudaErr(cudaMemcpyToSymbol(ixe_min, &ixe_min_, sizeof(int)),__FUNCTION__);
      printCudaErr(cudaMemcpyToSymbol(ixs_max, &ixs_max_, sizeof(int)),__FUNCTION__);
      printCudaErr(cudaMemcpyToSymbol(iye_min, &iye_min_, sizeof(int)),__FUNCTION__);
      printCudaErr(cudaMemcpyToSymbol(iys_max, &iys_max_, sizeof(int)),__FUNCTION__);
   }
// }}}

// DEVICE FUNCTIONS {{{
   // previously CPMLS arrays in textures {{{
      /*tex1Dfetch(tex_dumpx, i)*/
      inline __device__ float dumpx(int i)
      {
            float abscissa_normalized = (DELTA>0)?MAX((ixe_min - i)/(float)DELTA, (i - (ixs_max-1)/(float)DELTA)):-1.0f;

            return (abscissa_normalized>=0)?dump0*powf(abscissa_normalized,NPOWER):0.0f;
      }
      inline __device__ float dumpy(int i)
      {
            float abscissa_normalized = (DELTA>0)?MAX((iye_min - i)/(float)DELTA, (i - (iys_max-1)/(float)DELTA)):-1.0f;

            return (abscissa_normalized>=0)?dump0*powf(abscissa_normalized,NPOWER):0.0f;
      }
      inline __device__ float dumpz(int i)
      {
            float abscissa_normalized = (DELTA>0)?(DELTA - i)/(float)DELTA:-1.0f;

            return (abscissa_normalized>=0)?dump0*powf(abscissa_normalized,NPOWER):0.0f;
      }

      /*tex1Dfetch(tex_kappax, i)*/
      inline __device__ float kappax(int i)
      {
            float abscissa_normalized = (DELTA>0)?MAX((ixe_min - i)/(float)DELTA, (i - (ixs_max-1)/(float)DELTA)):-1.0f;

            return (abscissa_normalized>=0)?1.f + (kappa0 - 1.f) * powf(abscissa_normalized,NPOWER):1.f;
      }
      inline __device__ float kappay(int i)
      {
            float abscissa_normalized = (DELTA>0)?MAX((iye_min - i)/(float)DELTA, (i - (iys_max-1)/(float)DELTA)):-1.0f;

            return (abscissa_normalized>=0)?1.f + (kappa0 - 1.f) * powf(abscissa_normalized,NPOWER):1.f;
      }
      inline __device__ float kappaz(int i)
      {
            float abscissa_normalized = (DELTA>0)?(DELTA - i)/(float)DELTA:-1.0f;

            return (abscissa_normalized>=0)?1.f + (kappa0 - 1.f) * powf(abscissa_normalized,NPOWER):1.f;
      }

      /*tex1Dfetch(tex_alphax, i)*/
      inline __device__ float alphax(int i)
      {
            float abscissa_normalized = (DELTA>0)?MAX((ixe_min - i)/(float)DELTA, (i - (ixs_max-1)/(float)DELTA)):-1.0f;

            return (abscissa_normalized>=0)?alpha0 * (1.f - abscissa_normalized):0.0f;
      }
      inline __device__ float alphay(int i)
      {
            float abscissa_normalized = (DELTA>0)?MAX((iye_min - i)/(float)DELTA, (i - (iys_max-1)/(float)DELTA)):-1.0f;

            return (abscissa_normalized>=0)?alpha0 * (1.f - abscissa_normalized):0.0f;
      }
      inline __device__ float alphaz(int i)
      {
            float abscissa_normalized = (DELTA>0)?(DELTA - i)/(float)DELTA:-1.0f;

            return (abscissa_normalized>=0)?alpha0 * (1.f - abscissa_normalized):0.0f;
      }

      /*tex1Dfetch(tex_dumpx2, i)*/
      inline __device__ float dumpx2(int i)
      {
            float abscissa_normalized = (DELTA>0)?MAX((ixe_min - i - 0.5)/(float)DELTA, (i - (ixs_max-1) + 0.5)/(float)DELTA):-1.0f;

            return (abscissa_normalized>=0)?dump0*powf(abscissa_normalized,NPOWER):0.0f;
      }
      inline __device__ float dumpy2(int i)
      {
            float abscissa_normalized = (DELTA>0)?MAX((iye_min - i - 0.5)/(float)DELTA, (i - (iys_max-1) + 0.5)/(float)DELTA):-1.0f;

            return (abscissa_normalized>=0)?dump0*powf(abscissa_normalized,NPOWER):0.0f;
      }
      inline __device__ float dumpz2(int i)
      {
            float abscissa_normalized = (DELTA>0)?(DELTA - i - 0.5)/(float)DELTA:-1.0f;

            return (abscissa_normalized>=0)?dump0*powf(abscissa_normalized,NPOWER):0.0f;
      }

      /*tex1Dfetch(tex_kappax2, i)*/
      inline __device__ float kappax2(int i)
      {
            float abscissa_normalized = (DELTA>0)?MAX((ixe_min - i - 0.5)/(float)DELTA, (i - (ixs_max-1) + 0.5)/(float)DELTA):-1.0f;

            return (abscissa_normalized>=0)?1.f + (kappa0 - 1.f) * powf(abscissa_normalized,NPOWER):1.f;
      }
      inline __device__ float kappay2(int i)
      {
            float abscissa_normalized = (DELTA>0)?MAX((iye_min - i - 0.5)/(float)DELTA, (i - (iys_max-1) + 0.5)/(float)DELTA):-1.0f;

            return (abscissa_normalized>=0)?1.f + (kappa0 - 1.f) * powf(abscissa_normalized,NPOWER):1.f;
      }
      inline __device__ float kappaz2(int i)
      {
            float abscissa_normalized = (DELTA>0)?(DELTA - i - 0.5)/(float)DELTA:-1.0f;

            return (abscissa_normalized>=0)?1.f + (kappa0 - 1.f) * powf(abscissa_normalized,NPOWER):1.f;
      }

      /*tex1Dfetch(tex_alphax2, i)*/
      inline __device__ float alphax2(int i)
      {
            float abscissa_normalized = (DELTA>0)?MAX((ixe_min - i - 0.5)/(float)DELTA, (i - (ixs_max-1) + 0.5)/(float)DELTA):-1.0f;

            return (abscissa_normalized>=0)?alpha0 * (1.f - abscissa_normalized):0.0f;
      }
      inline __device__ float alphay2(int i)
      {
            float abscissa_normalized = (DELTA>0)?MAX((iye_min - i - 0.5)/(float)DELTA, (i - (iys_max-1) + 0.5)/(float)DELTA):-1.0f;

            return (abscissa_normalized>=0)?alpha0 * (1.f - abscissa_normalized):0.0f;
      }
      inline __device__ float alphaz2(int i)
      {
            float abscissa_normalized = (DELTA>0)?(DELTA - i - 0.5)/(float)DELTA:-1.0f;

            return (abscissa_normalized>=0)?alpha0 * (1.f - abscissa_normalized):0.0f;
      }
   // }}}

   // historical functs {{{
      __device__ float CPML2 (float vp, float dump, float alpha, float kappa, float phidum, float dx, float dt, float x1, float x2 )
      {
        float a, b;

        b = expf ( - ( vp*dump / kappa + alpha ) * dt );
        a = 0.0f;

        if ((vp*dump) > 0.000001f || (vp*dump) < -0.000001f ) 
         a = vp*dump * ( b - 1.f) / ( kappa * ( vp*dump + kappa * alpha ) );

        return b * phidum + a * ( x2 - x1 ) * (1.f/dx);
      }

      __device__ float CPML4 (float vp, float dump, float alpha, float kappa, float phidum, float dx, float dt, float x1, float x2, float x3, float x4 )
      {
        float a, b;

        b = expf ( - ( vp*dump / kappa + alpha ) * dt );
        a = 0.0f;
        
        if ((vp*dump) > 0.000001f || (vp*dump) < -0.000001f ) 
         a = vp*dump * ( b - 1.f ) / ( kappa * ( vp*dump + kappa * alpha ) );
         
        return b * phidum + a * ( (9.f/8.f)*( x2 - x1 )/dx - (1.f/24.f)*( x4 - x3 )/dx );
      }

      __device__ float staggards2 (float lam, float mu, float kappax, float kappay, float kappaz, float dt, float dx, float x1, float x2, float y1, float y2, float z1, float z2 )
      {
        return dt*( (lam+2.f*mu)*(x2 - x1)/kappax + lam*(y2 - y1)/kappay + lam*(z2 - z1)/kappaz )/dx;
      }

      __device__ float staggardt2 (float mu, float kappax, float kappay, float dt, float dx, float x1, float x2, float y1, float y2 )
      {
        return dt*mu*( (x2 - x1)/kappax + (y2 - y1)/kappay )/dx;
      }

      __device__ float staggards4 (float lam, float mu, float kappax, float kappay, float kappaz, float dt, float dx,
         float x1, float x2, float x3, float x4,
         float y1, float y2, float y3, float y4,
         float z1, float z2, float z3, float z4 )
      {
        return (9.f*dt/8.f)*( (lam+2.f*mu)*(x2 - x1)/kappax + lam*(y2 - y1)/kappay + lam*(z2 - z1)/kappaz )/dx
              - (dt/24.f)*( (lam+2.f*mu)*(x4 - x3)/kappax + lam*(y4 - y3)/kappay + lam*(z4 - z3)/kappaz )/dx;
      }

      __device__ float staggardt4 (float mu, float kappax, float kappay, float dt, float dx,
         float x1, float x2, float x3, float x4,
         float y1, float y2, float y3, float y4 )
      {
        return (9.f*dt*mu/8.f)*( (x2 - x1)/kappax + (y2 - y1)/kappay )/dx
              - (dt*mu/24.f)*( (x4 - x3)/kappax + (y4 - y3)/kappay )/dx;
      }

      __device__ float staggardv4 (float b, float kappax, float kappay, float kappaz, float dt, float dx,
         float x1, float x2, float x3, float x4,
         float y1, float y2, float y3, float y4,
         float z1, float z2, float z3, float z4 )
      {
        return (9.f*b*dt/8.f)*( (x2 - x1)/kappax + (y2 - y1)/kappay + (z2 - z1)/kappaz )/dx
              - (b*dt/24.f)*( (x4 - x3)/kappax + (y4 - y3)/kappay + (z4 - z3)/kappaz )/dx;
      }

      __device__ float staggardv2 (float b, float kappax, float kappay, float kappaz, float dt, float dx,
         float x1, float x2,
         float y1, float y2,
         float z1, float z2 )
      {
        return b*dt*( (x2 - x1)/kappax + (y2 - y1)/kappay + (z2 - z1)/kappaz )/dx;
      }
   // }}}
// }}}

// KERNELS {{{
   // COMPUTE STRESS {{{
      // IMPLEMENTATION {{{
#ifdef OPTI_ARCH_SM_20
__launch_bounds__(NPPDX*NPPDY, 6)
#endif
         __global__ void compute_stress_3d ( float* d_txx0, float* d_tyy0, float* d_tzz0, float* d_txy0, float* d_txz0, float* d_tyz0,
                              float* d_vx0, float* d_vy0, float* d_vz0,
                              int* d_npml_tab, float* d_phivxx, float* d_phivxy, float* d_phivxz, float* d_phivyx, float* d_phivyy, float* d_phivyz, float* d_phivzx, float* d_phivzy, float* d_phivzz, 
                              float* d_mu, float* d_lam, float* d_vp, 
                              int sizex, int sizey, int sizez,
                              int pitch_x, int pitch_y, int pitch_z, 
                              float ds, float dt, int delta, int position)
         {
            __shared__ float s_mu[(NPPDX+1)*(NPPDY+1)][2];
            __shared__ float s_vp[(NPPDX+1)*(NPPDY+1)][2];
            __shared__ float s_lam[(NPPDX+1)*NPPDY]; // on peut s'en passer au prix d'une lecture supplementaire en mem glob (a voir qd je connaitrai la limitation : registres ou shmem)
            __shared__ float s_vx0[(NPPDX+4)*(NPPDY+4)];
            __shared__ float s_vy0[(NPPDX+4)*(NPPDY+4)];
            __shared__ float s_vz0[(NPPDX+4)*(NPPDY+4)];

            float vx0_m1, vx0_p1, vx0_p2; // vx0 pour k-1, vx0 pour k-2, vx0 pour k+1, vx0 pour k+2
            float vy0_m1, vy0_p1, vy0_p2;
            float vz0_m1, vz0_m2, vz0_p1, vz0_p2;
            
            int i = blockIdx.x*blockDim.x + threadIdx.x;
            int j = blockIdx.y*blockDim.y + threadIdx.y;
            int k = 0;
            int distance_xmin = i;
            int distance_xmax = sizex - i - 1;
            int distance_ymin = j;
            int distance_ymax = sizey - j - 1;
            int distance_zmax = sizez - k - 1;
            int offset, tx;
            bool last_x, last_y;
            
            //on ne calcule pas les points qui sont en dehors du domaine. Les threads correspondants ne font rien.   
            bool active = (distance_xmax >=0 && distance_ymax >=0)?true:false;
            last_x = last_y = false;
            if (distance_xmax == 0 || (active && threadIdx.x == (NPPDX-1))) {
               last_x = true;
            }
            if (distance_ymax == 0 || (active && threadIdx.y == (NPPDY-1))) {
               last_y = true;
            }

            // ici distance pour le modele global (on ne s'interesse qu'aux bords, donc si on n'est pas pres du bord, une valeur quelconque >2 suffit)
            distance_xmin = (position & MASK_FIRST_X)?i:DUMMY_VALUE;
            distance_xmax = (position & MASK_LAST_X)?(sizex - i - 1):DUMMY_VALUE;
            distance_ymin = (position & MASK_FIRST_Y)?j:DUMMY_VALUE;
            distance_ymax = (position & MASK_LAST_Y)?(sizey - j - 1):DUMMY_VALUE;
         // pour k = 0 -------------------------------------------------------------------------------------------->>>
            // chargement initial des valeurs dans les registres et en memoire partagee
            if (active) {/*{{{*/
               // le tableau est initialisé à zéro et ces éléments ne sont jamais mis à jour
               vx0_m1 = 0.f;
               vy0_m1 = 0.f;
               vz0_m1 = vz0_m2 = 0.f;

               offset = pitch_x*pitch_y + j*pitch_x + i;
               vx0_p1 = d_vx0[offset];
               vy0_p1 = d_vy0[offset];
               vz0_p1 = d_vz0[offset];

               offset += pitch_x*pitch_y;
               vx0_p2 = d_vx0[offset];
               vy0_p2 = d_vy0[offset];
               vz0_p2 = d_vz0[offset];

               // chaque thread charge une donnée du domaine pour k=0
               tx = threadIdx.y*(NPPDX+1) + threadIdx.x;
               offset = j*pitch_x + i;
               s_mu[tx][0] = d_mu[offset];
               s_vp[tx][0] = d_vp[offset];
               s_lam[tx] = d_lam[offset];
               // maintenant, chaque thread charge une donnée pour k=1 pour mu et vp
               offset = pitch_x*(pitch_y) + j*pitch_x + i;
               s_mu[tx][1] = d_mu[offset];
               s_vp[tx][1] = d_vp[offset];
               // maintenant, on charge les données pour i+1(hors du block)
               // la dernière rangée charge les données du halo
               if (last_x) {
                  tx = threadIdx.y*(NPPDX+1) +threadIdx.x+1;
                  offset = j*pitch_x + i+1;
                  s_lam[tx] = d_lam[offset];
                  s_mu[tx][0] = d_mu[offset];
                  s_vp[tx][0] = d_vp[offset];
                  // i+1, k+1
                  offset = pitch_x*(pitch_y) + j*pitch_x + i+1;
                  s_mu[tx][1] = d_mu[offset];
                  s_vp[tx][1] = d_vp[offset];
               }
               // maintenant, on charge les données pour j+1(hors du block) : uniquement vp et mu
               if (last_y) {
                  tx = (threadIdx.y+1)*(NPPDX+1) + threadIdx.x;
                  offset = (j+1)*pitch_x + i;
                  s_mu[tx][0] = d_mu[offset];
                  s_vp[tx][0] = d_vp[offset];
                  offset = pitch_x*(pitch_y) + (j+1)*pitch_x + i;
                  s_mu[tx][1] = d_mu[offset];
                  s_vp[tx][1] = d_vp[offset];
               }
               // maintenant, on charge la donnée pour j+1 & i+1(hors du block) : uniquement vp et mu
               if (last_y && last_x) {
                  tx = (threadIdx.y+1)*(NPPDX+1) + threadIdx.x+1;
                  offset = (j+1)*pitch_x + i+1;
                  s_mu[tx][0] = d_mu[offset];
                  s_vp[tx][0] = d_vp[offset];
                  offset = pitch_x*(pitch_y) + (j+1)*pitch_x + i+1;
                  s_mu[tx][1] = d_mu[offset];
                  s_vp[tx][1] = d_vp[offset];
               }
               // vx0, vy0, vz0
               // chaque thread charge sa valeur en shmem
               tx = (threadIdx.y+2)*(NPPDX+4) + threadIdx.x+2;
               offset = j*pitch_x + i;
               s_vx0[tx] = d_vx0[offset];
               s_vy0[tx] = d_vy0[offset];
               s_vz0[tx] = d_vz0[offset];
               // on charge les deux rangées i-1 et i-2
               if (threadIdx.x == 0) {
                  // i-2
                  tx = (threadIdx.y+2)*(NPPDX+4) + 0;
                  offset = j*pitch_x + i-2;
                  s_vx0[tx] = d_vx0[offset];
                  s_vy0[tx] = d_vy0[offset];
                  s_vz0[tx] = d_vz0[offset];
                  // i-1
                  tx = (threadIdx.y+2)*(NPPDX+4) + 1;
                  offset = j*pitch_x + i-1;
                  s_vx0[tx] = d_vx0[offset];
                  s_vy0[tx] = d_vy0[offset];
                  s_vz0[tx] = d_vz0[offset];
               }
               // on charge les deux rangées i+1 et i+2
               if (last_x) {
                  // i+1
                  tx = (threadIdx.y+2)*(NPPDX+4) + threadIdx.x + 3;
                  offset = j*pitch_x + i + 1;
                  s_vx0[tx] = d_vx0[offset];
                  s_vy0[tx] = d_vy0[offset];
                  s_vz0[tx] = d_vz0[offset];
                  // i+2
                  tx = (threadIdx.y+2)*(NPPDX+4) + threadIdx.x + 4;
                  offset = j*pitch_x + i + 2;
                  s_vx0[tx] = d_vx0[offset];
                  s_vy0[tx] = d_vy0[offset];
                  s_vz0[tx] = d_vz0[offset];
               }
               // on charge les rangées j-1 et j-2
               if (threadIdx.y == 0) {
                  // j-2
                  tx = threadIdx.x+2;
                  offset = (j-2)*pitch_x + i;
                  s_vx0[tx] = d_vx0[offset];
                  s_vy0[tx] = d_vy0[offset];
                  s_vz0[tx] = d_vz0[offset];
                  // j-1
                  tx = (NPPDX+4) + threadIdx.x+2;
                  offset = (j-1)*pitch_x + i;
                  s_vx0[tx] = d_vx0[offset];
                  s_vy0[tx] = d_vy0[offset];
                  s_vz0[tx] = d_vz0[offset];
               }
               // on charge les rangées j+1 et j+2
               if (last_y) {
                  // j+2
                  tx = (threadIdx.y + 4)*(NPPDX+4) + threadIdx.x+2;
                  offset = (j+2)*pitch_x + i;
                  s_vx0[tx] = d_vx0[offset];
                  s_vy0[tx] = d_vy0[offset];
                  s_vz0[tx] = d_vz0[offset];
                  // j+1
                  tx = (threadIdx.y + 3)*(NPPDX+4) + threadIdx.x+2;
                  offset = (j+1)*pitch_x + i;
                  s_vx0[tx] = d_vx0[offset];
                  s_vy0[tx] = d_vy0[offset];
                  s_vz0[tx] = d_vz0[offset];
               }
            }/*}}}*/
            __syncthreads();
            // calcul
         #ifndef NOCPML
            if (active) {/*{{{*/
               // acces coalescent à d_npml_tab
               int npml = d_npml_tab[k*(sizex)*(sizey) + j*(sizex) + i];
               offset = k*pitch_x*pitch_y + j*pitch_x + i;
               /* Calculation of txx, tyy and tzz */
               /* Calculation of txy */
               if ( distance_ymax >= 1 && distance_xmin >= 1 ){
                  float muy = 2.f/(1.f/s_mu[MUOFF(0,0)][0]  + 1.f/s_mu[MUOFF(0,1)][0]);
                  float vpy = 2.f/(1.f/s_vp[VPOFF(0,0)][0] +  1.f/s_vp[VPOFF(0,1)][0]);

                  float phixdum =   d_phivyx[npml];
                  float phiydum = d_phivxy[npml];
                  phixdum = CPML4 (vpy, dumpx(i), alphax(i), kappax(i), phixdum, ds, dt,
                  s_vy0[VOFF(-1,0)], s_vy0[VOFF(0,0)],
                  s_vy0[VOFF(-2,0)], s_vy0[VOFF(1,0)]);
                  phiydum = CPML4 (vpy, dumpy2(j), alphay2(j), kappay2(j), phiydum, ds, dt,
                  s_vx0[VOFF(0,0)], s_vx0[VOFF(0,1)],
                  s_vx0[VOFF(0,-1)], s_vx0[VOFF(0,2)]);

                  d_txy0[offset] += dt*muy*( phixdum + phiydum )
                  + staggardt4 (muy, kappax(i), kappay2(j), dt, ds,
                  s_vy0[VOFF(-1,0)], s_vy0[VOFF(0,0)],
                  s_vy0[VOFF(-2,0)], s_vy0[VOFF(1,0)],
                  s_vx0[VOFF(0,0)], s_vx0[VOFF(0,1)],
                  s_vx0[VOFF(0,-1)], s_vx0[VOFF(0,2)]);

                  d_phivyx[npml] = phixdum;
                  d_phivxy[npml] = phiydum;
               }
               /* Calculation of txz */
               if (distance_xmin >= 1 ){

                  float muz = 2.f/(1.f/s_mu[MUOFF(0,0)][0]  + 1.f/s_mu[MUOFF(0,0)][1]);
                  float vpz = 2.f/(1.f/s_vp[VPOFF(0,0)][0]  + 1.f/s_vp[VPOFF(0,0)][1]);
                  float phixdum =   d_phivzx[npml];
                  float phizdum = d_phivxz[npml];

                  phixdum = CPML4 (vpz, dumpx(i), alphax(i), kappax(i), phixdum, ds, dt,
                  s_vz0[VOFF(-1,0)], s_vz0[VOFF(0,0)],
                  s_vz0[VOFF(-2,0)], s_vz0[VOFF(1,0)]);
                  phizdum = CPML4 (vpz, dumpz2(k), alphaz2(k), kappaz2(k), phizdum, ds, dt,
                  s_vx0[VOFF(0,0)], vx0_p1,
                  vx0_m1, vx0_p2);

                  d_txz0[offset] += dt*muz*( phixdum + phizdum )
                  + staggardt4 (muz, kappax(i), kappaz2(k), dt, ds,
                  s_vz0[VOFF(-1,0)], s_vz0[VOFF(0,0)],
                  s_vz0[VOFF(-2,0)], s_vz0[VOFF(1,0)],
                  s_vx0[VOFF(0,0)], vx0_p1,
                  vx0_m1, vx0_p2);

                  d_phivzx[npml] = phixdum;
                  d_phivxz[npml] = phizdum;
               }
               /* Calculation of tyz */
               if (distance_ymax >= 1){
                  // (distance_xmax==0)?mu(i,j,k):mu(i+1,j,k);
                  float muxyz = 8.f/(1.f/s_mu[MUOFF(0,0)][0] + 1.f/s_mu[MUOFF(0,0)][1]
                            + 1.f/s_mu[MUOFF(0,1)][0] + 1.f/s_mu[MUOFF(0,1)][1]
                            + 1.f/s_mu[MUOFF(1,0)][0] + 1.f/s_mu[MUOFF(1,0)][1]
                            + 1.f/s_mu[MUOFF(1,1)][0] + 1.f/s_mu[MUOFF(1,1)][1]);
                  float vpxyz = 8.f/(1.f/s_vp[VPOFF(0,0)][0] + 1.f/s_vp[VPOFF(0,0)][1]
                            + 1.f/s_vp[VPOFF(0,1)][0] + 1.f/s_vp[VPOFF(0,1)][1]
                            + 1.f/s_vp[VPOFF(1,0)][0] + 1.f/s_vp[VPOFF(1,0)][1]
                            + 1.f/s_vp[VPOFF(1,1)][0] + 1.f/s_vp[VPOFF(1,1)][1]);
                  float phiydum = d_phivzy[npml];
                  float phizdum = d_phivyz[npml];

                  phiydum = CPML4 (vpxyz, dumpy2(j), alphay2(j), kappay2(j), phiydum, ds, dt,
                  s_vz0[VOFF(0,0)], s_vz0[VOFF(0,1)],
                  s_vz0[VOFF(0,-1)], s_vz0[VOFF(0,2)] );
                  phizdum = CPML4 (vpxyz, dumpz2(k), alphaz2(k), kappaz2(k), phizdum, ds, dt,
                  s_vy0[VOFF(0,0)], vy0_p1,
                  vy0_m1, vy0_p2 );

                  d_tyz0[offset] += dt*muxyz*( phiydum + phizdum )
                  + staggardt4 (muxyz, kappay2(j), kappaz2(k), dt, ds,
                  s_vz0[VOFF(0,0)], s_vz0[VOFF(0,1)],
                  s_vz0[VOFF(0,-1)], s_vz0[VOFF(0,2)],
                  s_vy0[VOFF(0,0)], vy0_p1,
                  vy0_m1, vy0_p2 );
                  
                  d_phivzy[npml] = phiydum;
                  d_phivyz[npml] = phizdum;
               }
            }/*}}}*/
            __syncthreads();
         // pour k= 1 a k = delta - 1 (CPML only) ----------------------------------------------------------------->>>
            for (k = 1; k < delta; k++) {/*{{{*/
               // decalage des donnees
               if (active) {/*{{{*/
                  // chaque thread charge une donnée du domaine
                  tx = threadIdx.y*(NPPDX+1) + threadIdx.x;
                  // on a déjà lu les données pour k à l'itération précédente pour mu et vp
                  s_mu[tx][0] = s_mu[tx][1];
                  s_vp[tx][0] = s_vp[tx][1];
                  offset = k*pitch_x*(pitch_y) + j*pitch_x + i;
                  s_lam[tx] = d_lam[offset];
                  // maintenant, chaque thread charge une donnée pour k+1 pour mu et vp
                  offset = (k+1)*pitch_x*(pitch_y) + j*pitch_x + i;
                  s_mu[tx][1] = d_mu[offset];
                  s_vp[tx][1] = d_vp[offset];
                  // maintenant, on charge les données pour i+1(hors du block)
                  if (last_x) {
                     // i+1
                     tx = threadIdx.y*(NPPDX+1) +threadIdx.x+1;
                     s_mu[tx][0] = s_mu[tx][1];
                     s_vp[tx][0] = s_vp[tx][1];
                     offset = k*pitch_x*(pitch_y)+j*pitch_x + i+1;
                     s_lam[tx] = d_lam[offset];
                     // i+1, k+1
                     offset = (k+1)*pitch_x*(pitch_y)+j*pitch_x + i+1;
                     s_mu[tx][1] = d_mu[offset];
                     s_vp[tx][1] = d_vp[offset];
                  }
                  // maintenant, on charge les données pour j+1(hors du block) : uniquement vp et mu
                  if (last_y) {
                     tx = (threadIdx.y+1)*(NPPDX+1) +threadIdx.x;
                     s_mu[tx][0] = s_mu[tx][1];
                     s_vp[tx][0] = s_vp[tx][1];
                     offset = (k+1)*pitch_x*(pitch_y) + (j+1)*pitch_x + i;
                     s_mu[tx][1] = d_mu[offset];
                     s_vp[tx][1] = d_vp[offset];
                  }
                  // maintenant, on charge la donnée pour j+1 & i+1(hors du block) : uniquement vp et mu
                  if (last_y && last_x) {
                     tx = (threadIdx.y+1)*(NPPDX+1) + threadIdx.x+1;
                     s_mu[tx][0] = s_mu[tx][1];
                     s_vp[tx][0] = s_vp[tx][1];
                     offset = (k+1)*pitch_x*(pitch_y) + (j+1)*pitch_x + i+1;
                     s_mu[tx][1] = d_mu[offset];
                     s_vp[tx][1] = d_vp[offset];
                  }
                  // décalage des données selon l'axe Z
                  tx = (threadIdx.y+2)*(NPPDX+4) + threadIdx.x+2;
                  vx0_m1 = s_vx0[tx];
                  s_vx0[tx] = vx0_p1;
                  vx0_p1 = vx0_p2;

                  vy0_m1 = s_vy0[tx];
                  s_vy0[tx] = vy0_p1;
                  vy0_p1 = vy0_p2;

                  vz0_m2 = vz0_m1;
                  vz0_m1 = s_vz0[tx];
                  s_vz0[tx] = vz0_p1;
                  vz0_p1 = vz0_p2;

                  // remarque : le tableau dépasse de un point ds chaque direction du domaine calculé, donc si on est au bord du domaine,
                  // i+1 est défini et la valeur à ce point est nulle (car hors du domaine)
                  // i+2 correspond à la valeur pour j+1 et i=-1, cad 0 car hors du domaine => ça reste cohérent
                  // idem pour i - 1 et i - 2 (i-2 toujours défini à cause du padding d'alignement)
                  // on charge les deux rangées i-1 et i-2
                  if (threadIdx.x == 0) {
                     // i-2
                     tx = (threadIdx.y+2)*(NPPDX+4) + 0;
                     offset = k*pitch_x*pitch_y + j*pitch_x + i-2;
                     s_vx0[tx] = d_vx0[offset];
                     s_vy0[tx] = d_vy0[offset];
                     s_vz0[tx] = d_vz0[offset];
                     // i-1
                     tx = (threadIdx.y+2)*(NPPDX+4) + 1;
                     offset = k*pitch_x*pitch_y + j*pitch_x + i-1;
                     s_vx0[tx] = d_vx0[offset];
                     s_vy0[tx] = d_vy0[offset];
                     s_vz0[tx] = d_vz0[offset];
                  }
                  // on charge les deux rangées i+1 et i+2
                  if (last_x) {
                     // i+1
                     tx = (threadIdx.y+2)*(NPPDX+4) + threadIdx.x + 3;
                     offset = k*pitch_x*pitch_y + j*pitch_x + i + 1;
                     s_vx0[tx] = d_vx0[offset];
                     s_vy0[tx] = d_vy0[offset];
                     s_vz0[tx] = d_vz0[offset];
                     // i+2
                     tx = (threadIdx.y+2)*(NPPDX+4) + threadIdx.x + 4;
                     offset = k*pitch_x*pitch_y + j*pitch_x + i + 2;
                     s_vx0[tx] = d_vx0[offset];
                     s_vy0[tx] = d_vy0[offset];
                     s_vz0[tx] = d_vz0[offset];
                  }
                  // on charge les rangées j-1 et j-2
                  if (threadIdx.y == 0) {
                     // j-2
                     tx = threadIdx.x+2;
                     offset = k*pitch_x*pitch_y + (j-2)*pitch_x + i;
                     s_vx0[tx] = d_vx0[offset];
                     s_vy0[tx] = d_vy0[offset];
                     s_vz0[tx] = d_vz0[offset];
                     // j-1
                     tx = (NPPDX+4) + threadIdx.x+2;
                     offset = k*pitch_x*pitch_y + (j-1)*pitch_x + i;
                     s_vx0[tx] = d_vx0[offset];
                     s_vy0[tx] = d_vy0[offset];
                     s_vz0[tx] = d_vz0[offset];
                  }
                  // on charge les rangées j+1 et j+2
                  if (last_y) {
                     // j+2
                     tx = (threadIdx.y + 4)*(NPPDX+4) + threadIdx.x+2;
                     offset = k*pitch_x*pitch_y + (j+2)*pitch_x + i;
                     s_vx0[tx] = d_vx0[offset];
                     s_vy0[tx] = d_vy0[offset];
                     s_vz0[tx] = d_vz0[offset];
                     // j+1
                     tx = (threadIdx.y + 3)*(NPPDX+4) + threadIdx.x+2;
                     offset = k*pitch_x*pitch_y + (j+1)*pitch_x + i;
                     s_vx0[tx] = d_vx0[offset];
                     s_vy0[tx] = d_vy0[offset];
                     s_vz0[tx] = d_vz0[offset];
                  }
                  // on charge les points pour k+2 (acces en mem globale)
                  if (distance_zmax < 2) {
                     vx0_p2 = 0.f;
                     vy0_p2 = 0.f;
                     vz0_p2 = 0.f;
                  } else {
                     vx0_p2 = d_vx0[(k+2)*pitch_x*pitch_y + j*pitch_x + i];
                     vy0_p2 = d_vy0[(k+2)*pitch_x*pitch_y + j*pitch_x + i];
                     vz0_p2 = d_vz0[(k+2)*pitch_x*pitch_y + j*pitch_x + i];
                  }
               }/*}}}*/
               __syncthreads();

               offset = k*pitch_x*pitch_y + j*pitch_x + i;
               // calcul
               if (active) {/*{{{*/
                  // acces coalescent à d_npml_tab
                  int npml = d_npml_tab[k*(sizex)*(sizey) + j*(sizex) + i];
                  offset = k*pitch_x*pitch_y + j*pitch_x + i;
                  /* Calculation of txx, tyy and tzz */
                  if (distance_ymin >= 1 && distance_xmax >= 1 ){
                     float mux = 2.f/(1.f/s_mu[MUOFF(0,0)][0]  + 1.f/s_mu[MUOFF(1,0)][0]);
                     float vpx = 2.f/(1.f/s_vp[VPOFF(0,0)][0]  + 1.f/s_vp[VPOFF(1,0)][0]);
                     float lamx = 2.f/(1.f/s_lam[LAMOFF(0,0)]  + 1.f/s_lam[LAMOFF(1,0)]);
                     
                     float phixdum =   d_phivxx[npml];
                     float phiydum = d_phivyy[npml];
                     float phizdum = d_phivzz[npml];

                     phixdum = CPML4 (vpx, dumpx2(i), alphax2(i), kappax2(i), phixdum, ds, dt,
                     s_vx0[VOFF(0,0)], s_vx0[VOFF(1,0)],
                     s_vx0[VOFF(-1,0)], s_vx0[VOFF(2,0)]);
                     phiydum = CPML4 (vpx, dumpy(j), alphay(j), kappay(j), phiydum, ds, dt,
                     s_vy0[VOFF(0,-1)], s_vy0[VOFF(0,0)],
                     s_vy0[VOFF(0,-2)], s_vy0[VOFF(0,1)]);
                     phizdum = CPML4 (vpx, dumpz(k), alphaz(k), kappaz(k), phizdum, ds, dt,
                     vz0_m1, s_vz0[VOFF(0,0)],
                     vz0_m2, vz0_p1);

                     d_txx0[offset] += dt*(lamx + 2*mux)*phixdum + dt*lamx*( phiydum + phizdum )
                     + staggards4 (lamx, mux, kappax2(i), kappay(j), kappaz(k), dt, ds,
                     s_vx0[VOFF(0,0)], s_vx0[VOFF(1,0)],
                     s_vx0[VOFF(-1,0)], s_vx0[VOFF(2,0)],
                     s_vy0[VOFF(0,-1)], s_vy0[VOFF(0,0)],
                     s_vy0[VOFF(0,-2)], s_vy0[VOFF(0,1)],
                     vz0_m1, s_vz0[VOFF(0,0)],
                     vz0_m2, vz0_p1);

                     d_tyy0[offset] += dt*lamx*( phixdum + phizdum ) + dt*(lamx + 2*mux)*phiydum
                     + staggards4 (lamx, mux, kappay(j), kappax2(i), kappaz(k), dt, ds,
                     s_vy0[VOFF(0,-1)], s_vy0[VOFF(0,0)],
                     s_vy0[VOFF(0,-2)], s_vy0[VOFF(0,1)],
                     s_vx0[VOFF(0,0)], s_vx0[VOFF(1,0)],
                     s_vx0[VOFF(-1,0)], s_vx0[VOFF(2,0)],
                     vz0_m1, s_vz0[VOFF(0,0)],
                     vz0_m2, vz0_p1);

                     d_tzz0[offset] += dt*lamx*( phixdum + phiydum ) + dt*(lamx + 2*mux)*phizdum
                     + staggards4 (lamx, mux, kappaz(k), kappax2(i), kappay(j), dt, ds,
                     vz0_m1, s_vz0[VOFF(0,0)],
                     vz0_m2, vz0_p1,
                     s_vx0[VOFF(0,0)], s_vx0[VOFF(1,0)],
                     s_vx0[VOFF(-1,0)], s_vx0[VOFF(2,0)],
                     s_vy0[VOFF(0,-1)], s_vy0[VOFF(0,0)],
                     s_vy0[VOFF(0,-2)], s_vy0[VOFF(0,1)]);

                     d_phivxx[npml] = phixdum;
                     d_phivyy[npml] = phiydum;
                     d_phivzz[npml] = phizdum;
                  } // if (distance_ymin >= 1 && distance_xmax <= 1 )
                  /* Calculation of txy */
                  if ( distance_ymax >= 1 && distance_xmin >= 1 ){
                     float muy = 2.f/(1.f/s_mu[MUOFF(0,0)][0]  + 1.f/s_mu[MUOFF(0,1)][0]);
                     float vpy = 2.f/(1.f/s_vp[VPOFF(0,0)][0]  + 1.f/s_vp[VPOFF(0,1)][0]);

                     float phixdum =   d_phivyx[npml];
                     float phiydum = d_phivxy[npml];

                     phixdum = CPML4 (vpy, dumpx(i), alphax(i), kappax(i), phixdum, ds, dt,
                     s_vy0[VOFF(-1,0)], s_vy0[VOFF(0,0)],
                     s_vy0[VOFF(-2,0)], s_vy0[VOFF(1,0)]);
                     phiydum = CPML4 (vpy, dumpy2(j), alphay2(j), kappay2(j), phiydum, ds, dt,
                     s_vx0[VOFF(0,0)], s_vx0[VOFF(0,1)],
                     s_vx0[VOFF(0,-1)], s_vx0[VOFF(0,2)]);

                     d_txy0[offset] += dt*muy*( phixdum + phiydum )
                     + staggardt4 (muy, kappax(i), kappay2(j), dt, ds,
                     s_vy0[VOFF(-1,0)], s_vy0[VOFF(0,0)],
                     s_vy0[VOFF(-2,0)], s_vy0[VOFF(1,0)],
                     s_vx0[VOFF(0,0)], s_vx0[VOFF(0,1)],
                     s_vx0[VOFF(0,-1)], s_vx0[VOFF(0,2)]);
                     
                     d_phivyx[npml] = phixdum;
                     d_phivxy[npml] = phiydum;
                  }
                  /* Calculation of txz */
                  if (distance_xmin >= 1 ){

                     float muz = 2.f/(1.f/s_mu[MUOFF(0,0)][0]  + 1.f/s_mu[MUOFF(0,0)][1]);
                     float vpz = 2.f/(1.f/s_vp[VPOFF(0,0)][0]  + 1.f/s_vp[VPOFF(0,0)][1]);

                     float phixdum =   d_phivzx[npml];
                     float phizdum = d_phivxz[npml];

                     phixdum = CPML4 (vpz, dumpx(i), alphax(i), kappax(i), phixdum, ds, dt,
                     s_vz0[VOFF(-1,0)], s_vz0[VOFF(0,0)],
                     s_vz0[VOFF(-2,0)], s_vz0[VOFF(1,0)]);
                     phizdum = CPML4 (vpz, dumpz2(k), alphaz2(k), kappaz2(k), phizdum, ds, dt,
                     s_vx0[VOFF(0,0)], vx0_p1,
                     vx0_m1, vx0_p2);

                     d_txz0[offset] += dt*muz*( phixdum + phizdum )
                     + staggardt4 (muz, kappax(i), kappaz2(k), dt, ds,
                     s_vz0[VOFF(-1,0)], s_vz0[VOFF(0,0)],
                     s_vz0[VOFF(-2,0)], s_vz0[VOFF(1,0)],
                     s_vx0[VOFF(0,0)], vx0_p1,
                     vx0_m1, vx0_p2);

                     d_phivzx[npml] = phixdum;
                     d_phivxz[npml] = phizdum;
                  }
                  /* Calculation of tyz */
                  if (distance_ymax >= 1){
                     // (distance_xmax==0)?mu(i,j,k):mu(i+1,j,k);
                     float muxyz = 8.f/(1.f/s_mu[MUOFF(0,0)][0] + 1.f/s_mu[MUOFF(0,0)][1]
                               + s_mu[MUOFF(0,1)][0] + s_mu[MUOFF(0,1)][1]
                               + s_mu[MUOFF(1,0)][0] + s_mu[MUOFF(1,0)][1]
                               + s_mu[MUOFF(1,1)][0] + s_mu[MUOFF(1,1)][1]);
                     float vpxyz = 8.f/(1.f/s_vp[VPOFF(0,0)][0] + 1.f/s_vp[VPOFF(0,0)][1]
                               + 1.f/s_vp[VPOFF(0,1)][0] + 1.f/s_vp[VPOFF(0,1)][1]
                               + 1.f/s_vp[VPOFF(1,0)][0] + 1.f/s_vp[VPOFF(1,0)][1]
                               + 1.f/s_vp[VPOFF(1,1)][0] + 1.f/s_vp[VPOFF(1,1)][1]);
                     float phiydum = d_phivzy[npml];
                     float phizdum = d_phivyz[npml];

                     phiydum = CPML4 (vpxyz, dumpy2(j), alphay2(j), kappay2(j), phiydum, ds, dt,
                     s_vz0[VOFF(0,0)], s_vz0[VOFF(0,1)],
                     s_vz0[VOFF(0,-1)], s_vz0[VOFF(0,2)] );
                     phizdum = CPML4 (vpxyz, dumpz2(k), alphaz2(k), kappaz2(k), phizdum, ds, dt,
                     s_vy0[VOFF(0,0)], vy0_p1,
                     vy0_m1, vy0_p2 );

                     d_tyz0[offset] += dt*muxyz*( phiydum + phizdum )
                     + staggardt4 (muxyz, kappay2(j), kappaz2(k), dt, ds,
                     s_vz0[VOFF(0,0)], s_vz0[VOFF(0,1)],
                     s_vz0[VOFF(0,-1)], s_vz0[VOFF(0,2)],
                     s_vy0[VOFF(0,0)], vy0_p1,
                     vy0_m1, vy0_p2 );
                     
                     d_phivzy[npml] = phiydum;
                     d_phivyz[npml] = phizdum;
                  }
               }/*}}}*/
               __syncthreads();
            }/*}}}*/
#if 0
#endif

         // pour k= delta a k = sizez - 3 (CPML + ordre 4) ---------------------------------------------------->>>
            for (k = delta ; k < sizez - 2; k++) {/*{{{*/
               // decalage des donnees
               if (active && k>0) {/*{{{*/
                  // chaque thread charge une donnée du domaine
                  tx = threadIdx.y*(NPPDX+1) + threadIdx.x;
                  // on a déjà lu les données pour k à l'itération précédente pour mu et vp
                  s_mu[tx][0] = s_mu[tx][1];
                  s_vp[tx][0] = s_vp[tx][1];
                  offset = k*pitch_x*(pitch_y) + j*pitch_x + i;
                  s_lam[tx] = d_lam[offset];
                  // maintenant, chaque thread charge une donnée pour k+1 pour mu et vp
                  offset = (k+1)*pitch_x*(pitch_y) + j*pitch_x + i;
                  s_mu[tx][1] = d_mu[offset];
                  s_vp[tx][1] = d_vp[offset];
                  // maintenant, on charge les données pour i+1(hors du block)
                  if (last_x) {
                     // i+1
                     tx = threadIdx.y*(NPPDX+1) +threadIdx.x+1;
                     s_mu[tx][0] = s_mu[tx][1];
                     s_vp[tx][0] = s_vp[tx][1];
                     offset = k*pitch_x*(pitch_y)+j*pitch_x + i+1;
                     s_lam[tx] = d_lam[offset];
                     // i+1, k+1
                     offset = (k+1)*pitch_x*(pitch_y)+j*pitch_x + i+1;
                     s_mu[tx][1] = d_mu[offset];
                     s_vp[tx][1] = d_vp[offset];
                  }
                  // maintenant, on charge les données pour j+1(hors du block) : uniquement vp et mu
                  if (last_y) {
                     tx = (threadIdx.y+1)*(NPPDX+1) +threadIdx.x;
                     s_mu[tx][0] = s_mu[tx][1];
                     s_vp[tx][0] = s_vp[tx][1];
                     offset = (k+1)*pitch_x*(pitch_y) + (j+1)*pitch_x + i;
                     s_mu[tx][1] = d_mu[offset];
                     s_vp[tx][1] = d_vp[offset];
                  }
                  // maintenant, on charge la donnée pour j+1 & i+1(hors du block) : uniquement vp et mu
                  if (last_y && last_x) {
                     tx = (threadIdx.y+1)*(NPPDX+1) + threadIdx.x+1;
                     s_mu[tx][0] = s_mu[tx][1];
                     s_vp[tx][0] = s_vp[tx][1];
                     offset = (k+1)*pitch_x*(pitch_y) + (j+1)*pitch_x + i+1;
                     s_mu[tx][1] = d_mu[offset];
                     s_vp[tx][1] = d_vp[offset];
                  }
                  // décalage des données selon l'axe Z
                  tx = (threadIdx.y+2)*(NPPDX+4) + threadIdx.x+2;
                  vx0_m1 = s_vx0[tx];
                  s_vx0[tx] = vx0_p1;
                  vx0_p1 = vx0_p2;

                  vy0_m1 = s_vy0[tx];
                  s_vy0[tx] = vy0_p1;
                  vy0_p1 = vy0_p2;

                  vz0_m2 = vz0_m1;
                  vz0_m1 = s_vz0[tx];
                  s_vz0[tx] = vz0_p1;
                  vz0_p1 = vz0_p2;

                  // remarque : le tableau dépasse de un point ds chaque direction du domaine calculé, donc si on est au bord du domaine,
                  // i+1 est défini et la valeur à ce point est nulle (car hors du domaine)
                  // i+2 correspond à la valeur pour j+1 et i=-1, cad 0 car hors du domaine => ça reste cohérent
                  // idem pour i - 1 et i - 2 (i-2 toujours défini à cause du padding d'alignement)
                  // on charge les deux rangées i-1 et i-2
                  if (threadIdx.x == 0) {
                     // i-2
                     tx = (threadIdx.y+2)*(NPPDX+4) + 0;
                     offset = k*pitch_x*pitch_y + j*pitch_x + i-2;
                     s_vx0[tx] = d_vx0[offset];
                     s_vy0[tx] = d_vy0[offset];
                     s_vz0[tx] = d_vz0[offset];
                     // i-1
                     tx = (threadIdx.y+2)*(NPPDX+4) + 1;
                     offset = k*pitch_x*pitch_y + j*pitch_x + i-1;
                     s_vx0[tx] = d_vx0[offset];
                     s_vy0[tx] = d_vy0[offset];
                     s_vz0[tx] = d_vz0[offset];
                  }
                  // on charge les deux rangées i+1 et i+2
                  if (last_x) {
                     // i+1
                     tx = (threadIdx.y+2)*(NPPDX+4) + threadIdx.x + 3;
                     offset = k*pitch_x*pitch_y + j*pitch_x + i + 1;
                     s_vx0[tx] = d_vx0[offset];
                     s_vy0[tx] = d_vy0[offset];
                     s_vz0[tx] = d_vz0[offset];
                     // i+2
                     tx = (threadIdx.y+2)*(NPPDX+4) + threadIdx.x + 4;
                     offset = k*pitch_x*pitch_y + j*pitch_x + i + 2;
                     s_vx0[tx] = d_vx0[offset];
                     s_vy0[tx] = d_vy0[offset];
                     s_vz0[tx] = d_vz0[offset];
                  }
                  // on charge les rangées j-1 et j-2
                  if (threadIdx.y == 0) {
                     // j-2
                     tx = threadIdx.x+2;
                     offset = k*pitch_x*pitch_y + (j-2)*pitch_x + i;
                     s_vx0[tx] = d_vx0[offset];
                     s_vy0[tx] = d_vy0[offset];
                     s_vz0[tx] = d_vz0[offset];
                     // j-1
                     tx = (NPPDX+4) + threadIdx.x+2;
                     offset = k*pitch_x*pitch_y + (j-1)*pitch_x + i;
                     s_vx0[tx] = d_vx0[offset];
                     s_vy0[tx] = d_vy0[offset];
                     s_vz0[tx] = d_vz0[offset];
                  }
                  // on charge les rangées j+1 et j+2
                  if (last_y) {
                     // j+2
                     tx = (threadIdx.y + 4)*(NPPDX+4) + threadIdx.x+2;
                     offset = k*pitch_x*pitch_y + (j+2)*pitch_x + i;
                     s_vx0[tx] = d_vx0[offset];
                     s_vy0[tx] = d_vy0[offset];
                     s_vz0[tx] = d_vz0[offset];
                     // j+1
                     tx = (threadIdx.y + 3)*(NPPDX+4) + threadIdx.x+2;
                     offset = k*pitch_x*pitch_y + (j+1)*pitch_x + i;
                     s_vx0[tx] = d_vx0[offset];
                     s_vy0[tx] = d_vy0[offset];
                     s_vz0[tx] = d_vz0[offset];
                  }
                  // on charge les points pour k+2 (acces en mem globale)
                  if (distance_zmax < 2) {
                     vx0_p2 = 0.f;
                     vy0_p2 = 0.f;
                     vz0_p2 = 0.f;
                  } else {
                     vx0_p2 = d_vx0[(k+2)*pitch_x*pitch_y + j*pitch_x + i];
                     vy0_p2 = d_vy0[(k+2)*pitch_x*pitch_y + j*pitch_x + i];
                     vz0_p2 = d_vz0[(k+2)*pitch_x*pitch_y + j*pitch_x + i];
                  }
               }/*}}}*/
               __syncthreads();
               
               offset = k*pitch_x*pitch_y + j*pitch_x + i;
               // calcul
               if (active) {/*{{{*/
                  // plus couteux que le test sur les bords -> a revoir un jour
                  int npml = d_npml_tab[k*(sizex)*(sizey) + j*(sizex) + i];
                  if ( npml >= 0){/*{{{*/
                     // acces coalescent à d_npml_tab
                     /* Calculation of txx, tyy and tzz */
                     if (distance_ymin >= 1 && distance_xmax >= 1 ){
                        float mux = 2.f/(1.f/s_mu[MUOFF(0,0)][0]  + 1.f/s_mu[MUOFF(1,0)][0]);
                        float vpx = 2.f/(1.f/s_vp[VPOFF(0,0)][0]  + 1.f/s_vp[VPOFF(1,0)][0]);
                        float lamx = 2.f/(1.f/s_lam[LAMOFF(0,0)]  + 1.f/s_lam[LAMOFF(1,0)]);
                        float phixdum =   d_phivxx[npml];
                        float phiydum = d_phivyy[npml];
                        float phizdum = d_phivzz[npml];

                        phixdum = CPML4 (vpx, dumpx2(i), alphax2(i), kappax2(i), phixdum, ds, dt,
                        s_vx0[VOFF(0,0)], s_vx0[VOFF(1,0)],
                        s_vx0[VOFF(-1,0)], s_vx0[VOFF(2,0)]);
                        phiydum = CPML4 (vpx, dumpy(j), alphay(j), kappay(j), phiydum, ds, dt,
                        s_vy0[VOFF(0,-1)], s_vy0[VOFF(0,0)],
                        s_vy0[VOFF(0,-2)], s_vy0[VOFF(0,1)]);
                        phizdum = CPML4 (vpx, dumpz(k), alphaz(k), kappaz(k), phizdum, ds, dt,
                        vz0_m1, s_vz0[VOFF(0,0)],
                        vz0_m2, vz0_p1);

                        d_txx0[offset] += dt*(lamx + 2.f*mux)*phixdum + dt*lamx*( phiydum + phizdum )
                        + staggards4 (lamx, mux, kappax2(i), kappay(j), kappaz(k), dt, ds,
                        s_vx0[VOFF(0,0)], s_vx0[VOFF(1,0)],
                        s_vx0[VOFF(-1,0)], s_vx0[VOFF(2,0)],
                        s_vy0[VOFF(0,-1)], s_vy0[VOFF(0,0)],
                        s_vy0[VOFF(0,-2)], s_vy0[VOFF(0,1)],
                        vz0_m1, s_vz0[VOFF(0,0)],
                        vz0_m2, vz0_p1);

                        d_tyy0[offset] += dt*lamx*( phixdum + phizdum ) + dt*(lamx + 2*mux)*phiydum
                        + staggards4 (lamx, mux, kappay(j), kappax2(i), kappaz(k), dt, ds,
                        s_vy0[VOFF(0,-1)], s_vy0[VOFF(0,0)],
                        s_vy0[VOFF(0,-2)], s_vy0[VOFF(0,1)],
                        s_vx0[VOFF(0,0)], s_vx0[VOFF(1,0)],
                        s_vx0[VOFF(-1,0)], s_vx0[VOFF(2,0)],
                        vz0_m1, s_vz0[VOFF(0,0)],
                        vz0_m2, vz0_p1);

                        d_tzz0[offset] += dt*lamx*( phixdum + phiydum ) + dt*(lamx + 2*mux)*phizdum
                        + staggards4 (lamx, mux, kappaz(k), kappax2(i), kappay(j), dt, ds,
                        vz0_m1, s_vz0[VOFF(0,0)],
                        vz0_m2, vz0_p1,
                        s_vx0[VOFF(0,0)], s_vx0[VOFF(1,0)],
                        s_vx0[VOFF(-1,0)], s_vx0[VOFF(2,0)],
                        s_vy0[VOFF(0,-1)], s_vy0[VOFF(0,0)],
                        s_vy0[VOFF(0,-2)], s_vy0[VOFF(0,1)]);

                        d_phivxx[npml] = phixdum;
                        d_phivyy[npml] = phiydum;
                        d_phivzz[npml] = phizdum;

                     } // if ( distance_zmin >= 1 && distance_ymin >= 1 && distance_xmax <= 1 )
                     /* Calculation of txy */
                     if ( distance_ymax >= 1 && distance_xmin >= 1 ){
                        float muy = 2.f/(1.f/s_mu[MUOFF(0,0)][0]  + 1.f/s_mu[MUOFF(0,1)][0]);
                        float vpy = 2.f/(1.f/s_vp[VPOFF(0,0)][0]  + 1.f/s_vp[VPOFF(0,1)][0]);

                        float phixdum =   d_phivyx[npml];
                        float phiydum = d_phivxy[npml];

                        phixdum = CPML4 (vpy, dumpx(i), alphax(i), kappax(i), phixdum, ds, dt,
                        s_vy0[VOFF(-1,0)], s_vy0[VOFF(0,0)],
                        s_vy0[VOFF(-2,0)], s_vy0[VOFF(1,0)]);
                        phiydum = CPML4 (vpy, dumpy2(j), alphay2(j), kappay2(j), phiydum, ds, dt,
                        s_vx0[VOFF(0,0)], s_vx0[VOFF(0,1)],
                        s_vx0[VOFF(0,-1)], s_vx0[VOFF(0,2)]);

                        d_txy0[offset] += dt*muy*( phixdum + phiydum )
                        + staggardt4 (muy, kappax(i), kappay2(j), dt, ds,
                        s_vy0[VOFF(-1,0)], s_vy0[VOFF(0,0)],
                        s_vy0[VOFF(-2,0)], s_vy0[VOFF(1,0)],
                        s_vx0[VOFF(0,0)], s_vx0[VOFF(0,1)],
                        s_vx0[VOFF(0,-1)], s_vx0[VOFF(0,2)]);
                        
                        d_phivyx[npml] = phixdum;
                        d_phivxy[npml] = phiydum;
                     }
                     /* Calculation of txz */
                     if (distance_xmin >= 1 ){

                        float muz = 2.f/(1.f/s_mu[MUOFF(0,0)][0]  + 1.f/s_mu[MUOFF(0,0)][1]);
                        float vpz = 2.f/(1.f/s_vp[VPOFF(0,0)][0]  + 1.f/s_vp[VPOFF(0,0)][1]);

                        float phixdum =   d_phivzx[npml];
                        float phizdum = d_phivxz[npml];

                        phixdum = CPML4 (vpz, dumpx(i), alphax(i), kappax(i), phixdum, ds, dt,
                        s_vz0[VOFF(-1,0)], s_vz0[VOFF(0,0)],
                        s_vz0[VOFF(-2,0)], s_vz0[VOFF(1,0)]);
                        phizdum = CPML4 (vpz, dumpz2(k), alphaz2(k), kappaz2(k), phizdum, ds, dt,
                        s_vx0[VOFF(0,0)], vx0_p1,
                        vx0_m1, vx0_p2);

                        d_txz0[offset] += dt*muz*( phixdum + phizdum )
                        + staggardt4 (muz, kappax(i), kappaz2(k), dt, ds,
                        s_vz0[VOFF(-1,0)], s_vz0[VOFF(0,0)],
                        s_vz0[VOFF(-2,0)], s_vz0[VOFF(1,0)],
                        s_vx0[VOFF(0,0)], vx0_p1,
                        vx0_m1, vx0_p2);

                        d_phivzx[npml] = phixdum;
                        d_phivxz[npml] = phizdum;
                     }
                     /* Calculation of tyz */
                     if (distance_ymax >= 1){
                        // (distance_xmax==0)?mu(i,j,k):mu(i+1,j,k);
                        float muxyz = 8.f/(1.f/s_mu[MUOFF(0,0)][0] + 1.f/s_mu[MUOFF(0,0)][1]
                                  + 1.f/s_mu[MUOFF(0,1)][0] + 1.f/s_mu[MUOFF(0,1)][1]
                                  + 1.f/s_mu[MUOFF(1,0)][0] + 1.f/s_mu[MUOFF(1,0)][1]
                                  + 1.f/s_mu[MUOFF(1,1)][0] + 1.f/s_mu[MUOFF(1,1)][1]);
                        float vpxyz = 8.f/(1.f/s_vp[VPOFF(0,0)][0] + 1.f/s_vp[VPOFF(0,0)][1]
                                  + 1.f/s_vp[VPOFF(0,1)][0] + 1.f/s_vp[VPOFF(0,1)][1]
                                  + 1.f/s_vp[VPOFF(1,0)][0] + 1.f/s_vp[VPOFF(1,0)][1]
                                  + 1.f/s_vp[VPOFF(1,1)][0] + 1.f/s_vp[VPOFF(1,1)][1]);
                        float phiydum = d_phivzy[npml];
                        float phizdum = d_phivyz[npml];

                        phiydum = CPML4 (vpxyz, dumpy2(j), alphay2(j), kappay2(j), phiydum, ds, dt,
                        s_vz0[VOFF(0,0)], s_vz0[VOFF(0,1)],
                        s_vz0[VOFF(0,-1)], s_vz0[VOFF(0,2)] );
                        phizdum = CPML4 (vpxyz, dumpz2(k), alphaz2(k), kappaz2(k), phizdum, ds, dt,
                        s_vy0[VOFF(0,0)], vy0_p1,
                        vy0_m1, vy0_p2 );

                        d_tyz0[offset] += dt*muxyz*( phiydum + phizdum )
                        + staggardt4 (muxyz, kappay2(j), kappaz2(k), dt, ds,
                        s_vz0[VOFF(0,0)], s_vz0[VOFF(0,1)],
                        s_vz0[VOFF(0,-1)], s_vz0[VOFF(0,2)],
                        s_vy0[VOFF(0,0)], vy0_p1,
                        vy0_m1, vy0_p2 );

                        
                        d_phivzy[npml] = phiydum;
                        d_phivyz[npml] = phizdum;
                     }
                  } else {/*}}}*/
                     float mux = 2.f/(1.f/s_mu[MUOFF(0,0)][0]  + 1.f/s_mu[MUOFF(1,0)][0]);
                     float lamx = 2.f/(1.f/s_lam[LAMOFF(0,0)]  + 1.f/s_lam[LAMOFF(1,0)]);
                     float muy = 2.f/(1.f/s_mu[MUOFF(0,0)][0]  + 1.f/s_mu[MUOFF(0,1)][0]);
                     float muz = 2.f/(1.f/s_mu[MUOFF(0,0)][0]  + 1.f/s_mu[MUOFF(0,0)][1]);
                     float muxyz = 8.f/(1.f/s_mu[MUOFF(0,0)][0] + 1.f/s_mu[MUOFF(0,0)][1]
                              + 1.f/s_mu[MUOFF(0,1)][0] + 1.f/s_mu[MUOFF(0,1)][1]
                              + 1.f/s_mu[MUOFF(1,0)][0] + 1.f/s_mu[MUOFF(1,0)][1]
                              + 1.f/s_mu[MUOFF(1,1)][0] + 1.f/s_mu[MUOFF(1,1)][1]);

                     d_txx0[offset] += (9.f*dt/8.f)*( (lamx+2.f*mux)*(s_vx0[VOFF(1,0)] - s_vx0[VOFF(0,0)]) + lamx*(s_vy0[VOFF(0,0)] - s_vy0[VOFF(0,-1)]) + lamx*(s_vz0[VOFF(0,0)] - vz0_m1) )/ds - (dt/24.f)*( (lamx+2.f*mux)*(s_vx0[VOFF(2,0)] - s_vx0[VOFF(-1,0)]) + lamx*(s_vy0[VOFF(0,1)] - s_vy0[VOFF(0,-2)]) + lamx*(vz0_p1 - vz0_m2) )/ds;
                     d_tyy0[offset] += (9.f*dt/8.f)*( (lamx+2.f*mux)*(s_vy0[VOFF(0,0)] - s_vy0[VOFF(0,-1)]) + lamx*(s_vx0[VOFF(1,0)] - s_vx0[VOFF(0,0)]) + lamx*(s_vz0[VOFF(0,0)] - vz0_m1) )/ds - (dt/24.f)*( (lamx+2.f*mux)*(s_vy0[VOFF(0,1)] - s_vy0[VOFF(0,-2)]) + lamx*(s_vx0[VOFF(2,0)] - s_vx0[VOFF(-1,0)]) + lamx*(vz0_p1 - vz0_m2))/ds;
                     d_tzz0[offset] += (9.f*dt/8.f)*( (lamx+2.f*mux)*(s_vz0[VOFF(0,0)] - vz0_m1) + lamx*(s_vx0[VOFF(1,0)] - s_vx0[VOFF(0,0)]) + lamx*(s_vy0[VOFF(0,0)] - s_vy0[VOFF(0,-1)]) )/ds - (dt/24.f)*( (lamx+2.f*mux)*(vz0_p1 - vz0_m2) + lamx*(s_vx0[VOFF(2,0)] - s_vx0[VOFF(-1,0)]) + lamx*(s_vy0[VOFF(0,1)] - s_vy0[VOFF(0,-2)]))/ds;
                     
                     d_txy0[offset] += (9.f*dt*muy/8.f)*((s_vy0[VOFF(0,0)]-s_vy0[VOFF(-1,0)]) + (s_vx0[VOFF(0,1)] - s_vx0[VOFF(0,0)]))/ds - (dt*muy/24.f)*((s_vy0[VOFF(1,0)] - s_vy0[VOFF(-2,0)]) + (s_vx0[VOFF(0,2)]  - s_vx0[VOFF(0,-1)]))/ds;
                     d_txz0[offset] += (9.f*dt*muz/8.f)*((s_vz0[VOFF(0,0)]-s_vz0[VOFF(-1,0)]) + (vx0_p1 - s_vx0[VOFF(0,0)]))/ds - (dt*muz/24.f)*((s_vz0[VOFF(1,0)] - s_vz0[VOFF(-2,0)]) + (vx0_p2  - vx0_m1))/ds;
                     d_tyz0[offset] += (9.f*dt*muxyz/8.f)*((s_vz0[VOFF(0,1)]-s_vz0[VOFF(0,0)]) + (vy0_p1 - s_vy0[VOFF(0,0)]))/ds - (dt*muxyz/24.f)*((s_vz0[VOFF(0,2)] - s_vz0[VOFF(0,-1)]) + (vy0_p2  - vy0_m1))/ds;
                  }
               }/*}}}*/
               // synchro avant de glisser la fenêtre
               __syncthreads();
            }/*}}}*/

         // pour k = sizez - 2 (distance_zmax == 1, CPML + ordre 2) ----------------------------------------------->>>
            k = sizez - 2;
            // decalage des donnees
            if (active) {/*{{{*/
               // chaque thread charge une donnée du domaine
               tx = threadIdx.y*(NPPDX+1) + threadIdx.x;
               // on a déjà lu les données pour k à l'itération précédente pour mu et vp
               s_mu[tx][0] = s_mu[tx][1];
               s_vp[tx][0] = s_vp[tx][1];
               offset = k*pitch_x*(pitch_y) + j*pitch_x + i;
               s_lam[tx] = d_lam[offset];
               // maintenant, chaque thread charge une donnée pour k+1 pour mu et vp
               offset = (k+1)*pitch_x*(pitch_y) + j*pitch_x + i;
               s_mu[tx][1] = d_mu[offset];
               s_vp[tx][1] = d_vp[offset];
               // maintenant, on charge les données pour i+1(hors du block)
               if (last_x) {
                  // i+1
                  tx = threadIdx.y*(NPPDX+1) +threadIdx.x+1;
                  s_mu[tx][0] = s_mu[tx][1];
                  s_vp[tx][0] = s_vp[tx][1];
                  offset = k*pitch_x*(pitch_y)+j*pitch_x + i+1;
                  s_lam[tx] = d_lam[offset];
                  // i+1, k+1
                  offset = (k+1)*pitch_x*(pitch_y)+j*pitch_x + i+1;
                  s_mu[tx][1] = d_mu[offset];
                  s_vp[tx][1] = d_vp[offset];
               }
               // maintenant, on charge les données pour j+1(hors du block) : uniquement vp et mu
               if (last_y) {
                  tx = (threadIdx.y+1)*(NPPDX+1) +threadIdx.x;
                  s_mu[tx][0] = s_mu[tx][1];
                  s_vp[tx][0] = s_vp[tx][1];
                  offset = (k+1)*pitch_x*(pitch_y) + (j+1)*pitch_x + i;
                  s_mu[tx][1] = d_mu[offset];
                  s_vp[tx][1] = d_vp[offset];
               }
               // maintenant, on charge la donnée pour j+1 & i+1(hors du block) : uniquement vp et mu
               if (last_y && last_x) {
                  tx = (threadIdx.y+1)*(NPPDX+1) + threadIdx.x+1;
                  s_mu[tx][0] = s_mu[tx][1];
                  s_vp[tx][0] = s_vp[tx][1];
                  offset = (k+1)*pitch_x*(pitch_y) + (j+1)*pitch_x + i+1;
                  s_mu[tx][1] = d_mu[offset];
                  s_vp[tx][1] = d_vp[offset];
               }
               // décalage des données selon l'axe Z
               tx = (threadIdx.y+2)*(NPPDX+4) + threadIdx.x+2;
               vx0_m1 = s_vx0[tx];
               s_vx0[tx] = vx0_p1;
               vx0_p1 = vx0_p2;

               vy0_m1 = s_vy0[tx];
               s_vy0[tx] = vy0_p1;
               vy0_p1 = vy0_p2;

               vz0_m2 = vz0_m1;
               vz0_m1 = s_vz0[tx];
               s_vz0[tx] = vz0_p1;
               vz0_p1 = vz0_p2;

               // remarque : le tableau dépasse de un point ds chaque direction du domaine calculé, donc si on est au bord du domaine,
               // i+1 est défini et la valeur à ce point est nulle (car hors du domaine)
               // i+2 correspond à la valeur pour j+1 et i=-1, cad 0 car hors du domaine => ça reste cohérent
               // idem pour i - 1 et i - 2 (i-2 toujours défini à cause du padding d'alignement)
               // on charge les deux rangées i-1 et i-2
               if (threadIdx.x == 0) {
                  // i-2
                  tx = (threadIdx.y+2)*(NPPDX+4) + 0;
                  offset = k*pitch_x*pitch_y + j*pitch_x + i-2;
                  s_vx0[tx] = d_vx0[offset];
                  s_vy0[tx] = d_vy0[offset];
                  s_vz0[tx] = d_vz0[offset];
                  // i-1
                  tx = (threadIdx.y+2)*(NPPDX+4) + 1;
                  offset = k*pitch_x*pitch_y + j*pitch_x + i-1;
                  s_vx0[tx] = d_vx0[offset];
                  s_vy0[tx] = d_vy0[offset];
                  s_vz0[tx] = d_vz0[offset];
               }
               // on charge les deux rangées i+1 et i+2
               if (last_x) {
                  // i+1
                  tx = (threadIdx.y+2)*(NPPDX+4) + threadIdx.x + 3;
                  offset = k*pitch_x*pitch_y + j*pitch_x + i + 1;
                  s_vx0[tx] = d_vx0[offset];
                  s_vy0[tx] = d_vy0[offset];
                  s_vz0[tx] = d_vz0[offset];
                  // i+2
                  tx = (threadIdx.y+2)*(NPPDX+4) + threadIdx.x + 4;
                  offset = k*pitch_x*pitch_y + j*pitch_x + i + 2;
                  s_vx0[tx] = d_vx0[offset];
                  s_vy0[tx] = d_vy0[offset];
                  s_vz0[tx] = d_vz0[offset];
               }
               // on charge les rangées j-1 et j-2
               if (threadIdx.y == 0) {
                  // j-2
                  tx = threadIdx.x+2;
                  offset = k*pitch_x*pitch_y + (j-2)*pitch_x + i;
                  s_vx0[tx] = d_vx0[offset];
                  s_vy0[tx] = d_vy0[offset];
                  s_vz0[tx] = d_vz0[offset];
                  // j-1
                  tx = (NPPDX+4) + threadIdx.x+2;
                  offset = k*pitch_x*pitch_y + (j-1)*pitch_x + i;
                  s_vx0[tx] = d_vx0[offset];
                  s_vy0[tx] = d_vy0[offset];
                  s_vz0[tx] = d_vz0[offset];
               }
               // on charge les rangées j+1 et j+2
               if (last_y) {
                  // j+2
                  tx = (threadIdx.y + 4)*(NPPDX+4) + threadIdx.x+2;
                  offset = k*pitch_x*pitch_y + (j+2)*pitch_x + i;
                  s_vx0[tx] = d_vx0[offset];
                  s_vy0[tx] = d_vy0[offset];
                  s_vz0[tx] = d_vz0[offset];
                  // j+1
                  tx = (threadIdx.y + 3)*(NPPDX+4) + threadIdx.x+2;
                  offset = k*pitch_x*pitch_y + (j+1)*pitch_x + i;
                  s_vx0[tx] = d_vx0[offset];
                  s_vy0[tx] = d_vy0[offset];
                  s_vz0[tx] = d_vz0[offset];
               }
               // on charge les points pour k+2 (acces en mem globale)
               if (distance_zmax < 2) {
                  vx0_p2 = 0.f;
                  vy0_p2 = 0.f;
                  vz0_p2 = 0.f;
               } else {
                  vx0_p2 = d_vx0[(k+2)*pitch_x*pitch_y + j*pitch_x + i];
                  vy0_p2 = d_vy0[(k+2)*pitch_x*pitch_y + j*pitch_x + i];
                  vz0_p2 = d_vz0[(k+2)*pitch_x*pitch_y + j*pitch_x + i];
               }
            }/*}}}*/
            __syncthreads();
            offset = k*pitch_x*pitch_y + j*pitch_x + i;
            // calcul
            if (active) {/*{{{*/
               int npml = d_npml_tab[k*(sizex)*(sizey) + j*(sizex) + i];
               if ( npml >= 0){/*{{{*/
                  /* Calculation of txx, tyy and tzz */
                  if (distance_ymin >= 1 && distance_xmax >= 1 ){
                     float mux = 2.f/(1.f/s_mu[MUOFF(0,0)][0]  + 1.f/s_mu[MUOFF(1,0)][0]);
                     float vpx = 2.f/(1.f/s_vp[VPOFF(0,0)][0]  + 1.f/s_vp[VPOFF(1,0)][0]);
                     float lamx = 2.f/(1.f/s_lam[LAMOFF(0,0)]  + 1.f/s_lam[LAMOFF(1,0)]);
                     
                     float phixdum =   d_phivxx[npml];
                     float phiydum = d_phivyy[npml];
                     float phizdum = d_phivzz[npml];

                     phixdum = CPML2 (vpx, dumpx2(i), alphax2(i), kappax2(i), phixdum, ds, dt, s_vx0[VOFF(0,0)], s_vx0[VOFF(1,0)]);
                     phiydum = CPML2 (vpx, dumpy(j), alphay(j), kappay(j), phiydum, ds, dt, s_vy0[VOFF(0,-1)], s_vy0[VOFF(0,0)]);
                     phizdum = CPML2 (vpx, dumpz(k), alphaz(k), kappaz(k), phizdum, ds, dt, vz0_m1, s_vz0[VOFF(0,0)]);

                     d_txx0[offset] += dt*(lamx + 2.f*mux)*phixdum + dt*lamx*( phiydum + phizdum )
                     + staggards2 (lamx, mux, kappax2(i), kappay(j), kappaz(k), dt, ds,
                     s_vx0[VOFF(0,0)], s_vx0[VOFF(1,0)],
                     s_vy0[VOFF(0,-1)], s_vy0[VOFF(0,0)], 
                     vz0_m1, s_vz0[VOFF(0,0)]);
                     
                     d_tyy0[offset] += dt*lamx*( phixdum + phizdum ) + dt*(lamx + 2*mux)*phiydum
                     + staggards2 (lamx, mux, kappay(j), kappax2(i), kappaz(k), dt, ds,
                     s_vy0[VOFF(0,-1)], s_vy0[VOFF(0,0)],
                     s_vx0[VOFF(0,0)], s_vx0[VOFF(1,0)],
                     vz0_m1, s_vz0[VOFF(0,0)]);

                     d_tzz0[offset] += dt*lamx*( phixdum + phiydum ) + dt*(lamx + 2*mux)*phizdum
                     + staggards2 (lamx, mux, kappaz(k), kappax2(i), kappay(j), dt, ds,
                     vz0_m1, s_vz0[VOFF(0,0)],
                     s_vx0[VOFF(0,0)], s_vx0[VOFF(1,0)],
                     s_vy0[VOFF(0,-1)], s_vy0[VOFF(0,0)]);
                     
                     d_phivxx[npml] = phixdum;
                     d_phivyy[npml] = phiydum;
                     d_phivzz[npml] = phizdum;
                  } // if ( distance_zmin >= 1 && distance_ymin >= 1 && distance_xmax <= 1 )
                  /* Calculation of txy */
                  if ( distance_ymax >= 1 && distance_xmin >= 1 ){
                     float muy = 2.f/(1.f/s_mu[MUOFF(0,0)][0]  + 1.f/s_mu[MUOFF(0,1)][0]);
                     float vpy = 2.f/(1.f/s_vp[VPOFF(0,0)][0]  + 1.f/s_vp[VPOFF(0,1)][0]);

                     float phixdum =   d_phivyx[npml];
                     float phiydum = d_phivxy[npml];

                     phixdum = CPML2 (vpy, dumpx(i), alphax(i), kappax(i), phixdum, ds, dt,
                     s_vy0[VOFF(-1,0)], s_vy0[VOFF(0,0)]);

                     phiydum = CPML2 (vpy, dumpy2(j), alphay2(j), kappay2(j), phiydum, ds, dt,
                     s_vx0[VOFF(0,0)], s_vx0[VOFF(0,1)]);
                     
                     d_txy0[offset] += dt*muy*( phixdum + phiydum )
                     + staggardt2 (muy, kappax(i), kappay2(j), dt, ds,
                     s_vy0[VOFF(-1,0)], s_vy0[VOFF(0,0)],
                     s_vx0[VOFF(0,0)], s_vx0[VOFF(0,1)]);

                     d_phivyx[npml] = phixdum;
                     d_phivxy[npml] = phiydum;
                  }
                  /* Calculation of txz */
                  if (distance_xmin >= 1 ){

                     float muz = 2.f/(1.f/s_mu[MUOFF(0,0)][0]  + 1.f/s_mu[MUOFF(0,0)][1]);
                     float vpz = 2.f/(1.f/s_vp[VPOFF(0,0)][0]  + 1.f/s_vp[VPOFF(0,0)][1]);

                     float phixdum =   d_phivzx[npml];
                     float phizdum = d_phivxz[npml];

                     phixdum = CPML2 (vpz, dumpx(i), alphax(i), kappax(i), phixdum, ds, dt,
                     s_vz0[VOFF(-1,0)], s_vz0[VOFF(0,0)]);
                     phizdum = CPML2 (vpz, dumpz2(k), alphaz2(k), kappaz2(k), phizdum, ds, dt,
                     s_vx0[VOFF(0,0)], vx0_p1 );

                     d_txz0[offset] += dt*muz*( phixdum + phizdum )
                     + staggardt2 (muz, kappax(i), kappaz2(k), dt, ds,
                     s_vz0[VOFF(-1,0)], s_vz0[VOFF(0,0)],
                     s_vx0[VOFF(0,0)], vx0_p1);

                     d_phivzx[npml] = phixdum;
                     d_phivxz[npml] = phizdum;
                  }
                  /* Calculation of tyz */
                  if (distance_ymax >= 1){
                     // (distance_xmax==0)?mu(i,j,k):mu(i+1,j,k);
                     float muxyz = 8.f/(1.f/s_mu[MUOFF(0,0)][0] + 1.f/s_mu[MUOFF(0,0)][1]
                               + 1.f/s_mu[MUOFF(0,1)][0] + 1.f/s_mu[MUOFF(0,1)][1]
                               + 1.f/s_mu[MUOFF(1,0)][0] + 1.f/s_mu[MUOFF(1,0)][1]
                               + 1.f/s_mu[MUOFF(1,1)][0] + 1.f/s_mu[MUOFF(1,1)][1]);
                     float vpxyz = 8.f/(1.f/s_vp[VPOFF(0,0)][0] + 1.f/s_vp[VPOFF(0,0)][1]
                               + 1.f/s_vp[VPOFF(0,1)][0] + 1.f/s_vp[VPOFF(0,1)][1]
                               + 1.f/s_vp[VPOFF(1,0)][0] + 1.f/s_vp[VPOFF(1,0)][1]
                               + 1.f/s_vp[VPOFF(1,1)][0] + 1.f/s_vp[VPOFF(1,1)][1]);
                     float phiydum = d_phivzy[npml];
                     float phizdum = d_phivyz[npml];

                     phiydum = CPML2 (vpxyz, dumpy2(j), alphay2(j), kappay2(j), phiydum, ds, dt,
                     s_vz0[VOFF(0,0)], s_vz0[VOFF(0,1)] );
                     phizdum = CPML2 (vpxyz, dumpz2(k), alphaz2(k), kappaz2(k), phizdum, ds, dt,
                     s_vy0[VOFF(0,0)], vy0_p1 );

                     d_tyz0[offset] += dt*muxyz*( phiydum + phizdum )
                     + staggardt2 (muxyz, kappay2(j), kappaz2(k), dt, ds,
                     s_vz0[VOFF(0,0)], s_vz0[VOFF(0,1)],
                     s_vy0[VOFF(0,0)], vy0_p1 );
                     
                     d_phivzy[npml] = phiydum;
                     d_phivyz[npml] = phizdum;
                  }
               } else {/*}}}*/
                  float mux = 2.f/(1.f/s_mu[MUOFF(0,0)][0]  + 1.f/s_mu[MUOFF(1,0)][0]);
                  float lamx = 2.f/(1.f/s_lam[LAMOFF(0,0)]  + 1.f/s_lam[LAMOFF(1,0)]);
                  float muy = 2.f/(1.f/s_mu[MUOFF(0,0)][0]  + 1.f/s_mu[MUOFF(0,1)][0]);
                  float muz = 2.f/(1.f/s_mu[MUOFF(0,0)][0]  + 1.f/s_mu[MUOFF(0,0)][1]);
                  float muxyz = 8.f/(1.f/s_mu[MUOFF(0,0)][0] + 1.f/s_mu[MUOFF(0,0)][1]
                           + 1.f/s_mu[MUOFF(0,1)][0] + 1.f/s_mu[MUOFF(0,1)][1]
                           + 1.f/s_mu[MUOFF(1,0)][0] + 1.f/s_mu[MUOFF(1,0)][1]
                           + 1.f/s_mu[MUOFF(1,1)][0] + 1.f/s_mu[MUOFF(1,1)][1]);

                  d_txx0[offset] += staggards2 (lamx, mux, 1.f, 1.f, 1.f, dt, ds,
                  s_vx0[VOFF(0,0)], s_vx0[VOFF(1,0)],
                  s_vy0[VOFF(0,-1)], s_vy0[VOFF(0,0)],
                  vz0_m1, s_vz0[VOFF(0,0)] );

                  d_tyy0[offset] += staggards2 (lamx, mux, 1.f, 1.f, 1.f, dt, ds,
                  s_vy0[VOFF(0,-1)], s_vy0[VOFF(0,0)],
                  s_vx0[VOFF(0,0)], s_vx0[VOFF(1,0)],
                  vz0_m1, s_vz0[VOFF(0,0)] );

                  d_tzz0[offset] += staggards2 (lamx, mux, 1.f, 1.f, 1.f, dt, ds,
                  vz0_m1, s_vz0[VOFF(0,0)],
                  s_vx0[VOFF(0,0)], s_vx0[VOFF(1,0)],
                  s_vy0[VOFF(0,-1)], s_vy0[VOFF(0,0)] );

                  d_txy0[offset] += staggardt2 (muy, 1.f, 1.f, dt, ds,
                  s_vy0[VOFF(-1,0)], s_vy0[VOFF(0,0)],
                  s_vx0[VOFF(0,0)], s_vx0[VOFF(0,1)] );

                  d_txz0[offset] += staggardt2 (muz, 1.f, 1.f, dt, ds,
                  s_vz0[VOFF(-1,0)], s_vz0[VOFF(0,0)],
                  s_vx0[VOFF(0,0)], vx0_p1 );

                  d_tyz0[offset] += staggardt2 (muxyz, 1.f, 1.f, dt, ds,
                  s_vz0[VOFF(0,0)], s_vz0[VOFF(0,1)],
                  s_vy0[VOFF(0,0)], vy0_p1 );
               }
            }/*}}}*/
            __syncthreads();

         // pour k = sizez-1 (surface libre) ---------------------------------------------------------------------->>>
            k++;
            // decalage des donnees
            if (active) {/*{{{*/
               // chaque thread charge une donnée du domaine
               tx = threadIdx.y*(NPPDX+1) + threadIdx.x;
               // on a déjà lu les données pour k à l'itération précédente pour mu et vp
               s_mu[tx][0] = s_mu[tx][1];
               s_vp[tx][0] = s_vp[tx][1];
               offset = k*pitch_x*(pitch_y) + j*pitch_x + i;
               s_lam[tx] = d_lam[offset];
               // maintenant, chaque thread charge une donnée pour k+1 pour mu et vp
               offset = (k+1)*pitch_x*(pitch_y) + j*pitch_x + i;
               s_mu[tx][1] = d_mu[offset];
               s_vp[tx][1] = d_vp[offset];
               // maintenant, on charge les données pour i+1(hors du block)
               if (last_x) {
                  // i+1
                  tx = threadIdx.y*(NPPDX+1) +threadIdx.x+1;
                  s_mu[tx][0] = s_mu[tx][1];
                  s_vp[tx][0] = s_vp[tx][1];
                  offset = k*pitch_x*(pitch_y)+j*pitch_x + i+1;
                  s_lam[tx] = d_lam[offset];
                  // i+1, k+1
                  offset = (k+1)*pitch_x*(pitch_y)+j*pitch_x + i+1;
                  s_mu[tx][1] = d_mu[offset];
                  s_vp[tx][1] = d_vp[offset];
               }
               // maintenant, on charge les données pour j+1(hors du block) : uniquement vp et mu
               if (last_y) {
                  tx = (threadIdx.y+1)*(NPPDX+1) +threadIdx.x;
                  s_mu[tx][0] = s_mu[tx][1];
                  s_vp[tx][0] = s_vp[tx][1];
                  offset = (k+1)*pitch_x*(pitch_y) + (j+1)*pitch_x + i;
                  s_mu[tx][1] = d_mu[offset];
                  s_vp[tx][1] = d_vp[offset];
               }
               // maintenant, on charge la donnée pour j+1 & i+1(hors du block) : uniquement vp et mu
               if (last_y && last_x) {
                  tx = (threadIdx.y+1)*(NPPDX+1) + threadIdx.x+1;
                  s_mu[tx][0] = s_mu[tx][1];
                  s_vp[tx][0] = s_vp[tx][1];
                  offset = (k+1)*pitch_x*(pitch_y) + (j+1)*pitch_x + i+1;
                  s_mu[tx][1] = d_mu[offset];
                  s_vp[tx][1] = d_vp[offset];
               }
               // décalage des données selon l'axe Z
               tx = (threadIdx.y+2)*(NPPDX+4) + threadIdx.x+2;
               vx0_m1 = s_vx0[tx];
               s_vx0[tx] = vx0_p1;
               vx0_p1 = vx0_p2;

               vy0_m1 = s_vy0[tx];
               s_vy0[tx] = vy0_p1;
               vy0_p1 = vy0_p2;

               vz0_m2 = vz0_m1;
               vz0_m1 = s_vz0[tx];
               s_vz0[tx] = vz0_p1;
               vz0_p1 = vz0_p2;

               // remarque : le tableau dépasse de un point ds chaque direction du domaine calculé, donc si on est au bord du domaine,
               // i+1 est défini et la valeur à ce point est nulle (car hors du domaine)
               // i+2 correspond à la valeur pour j+1 et i=-1, cad 0 car hors du domaine => ça reste cohérent
               // idem pour i - 1 et i - 2 (i-2 toujours défini à cause du padding d'alignement)
               // on charge les deux rangées i-1 et i-2
               if (threadIdx.x == 0) {
                  // i-2
                  tx = (threadIdx.y+2)*(NPPDX+4) + 0;
                  offset = k*pitch_x*pitch_y + j*pitch_x + i-2;
                  s_vx0[tx] = d_vx0[offset];
                  s_vy0[tx] = d_vy0[offset];
                  s_vz0[tx] = d_vz0[offset];
                  // i-1
                  tx = (threadIdx.y+2)*(NPPDX+4) + 1;
                  offset = k*pitch_x*pitch_y + j*pitch_x + i-1;
                  s_vx0[tx] = d_vx0[offset];
                  s_vy0[tx] = d_vy0[offset];
                  s_vz0[tx] = d_vz0[offset];
               }
               // on charge les deux rangées i+1 et i+2
               if (last_x) {
                  // i+1
                  tx = (threadIdx.y+2)*(NPPDX+4) + threadIdx.x + 3;
                  offset = k*pitch_x*pitch_y + j*pitch_x + i + 1;
                  s_vx0[tx] = d_vx0[offset];
                  s_vy0[tx] = d_vy0[offset];
                  s_vz0[tx] = d_vz0[offset];
                  // i+2
                  tx = (threadIdx.y+2)*(NPPDX+4) + threadIdx.x + 4;
                  offset = k*pitch_x*pitch_y + j*pitch_x + i + 2;
                  s_vx0[tx] = d_vx0[offset];
                  s_vy0[tx] = d_vy0[offset];
                  s_vz0[tx] = d_vz0[offset];
               }
               // on charge les rangées j-1 et j-2
               if (threadIdx.y == 0) {
                  // j-2
                  tx = threadIdx.x+2;
                  offset = k*pitch_x*pitch_y + (j-2)*pitch_x + i;
                  s_vx0[tx] = d_vx0[offset];
                  s_vy0[tx] = d_vy0[offset];
                  s_vz0[tx] = d_vz0[offset];
                  // j-1
                  tx = (NPPDX+4) + threadIdx.x+2;
                  offset = k*pitch_x*pitch_y + (j-1)*pitch_x + i;
                  s_vx0[tx] = d_vx0[offset];
                  s_vy0[tx] = d_vy0[offset];
                  s_vz0[tx] = d_vz0[offset];
               }
               // on charge les rangées j+1 et j+2
               if (last_y) {
                  // j+2
                  tx = (threadIdx.y + 4)*(NPPDX+4) + threadIdx.x+2;
                  offset = k*pitch_x*pitch_y + (j+2)*pitch_x + i;
                  s_vx0[tx] = d_vx0[offset];
                  s_vy0[tx] = d_vy0[offset];
                  s_vz0[tx] = d_vz0[offset];
                  // j+1
                  tx = (threadIdx.y + 3)*(NPPDX+4) + threadIdx.x+2;
                  offset = k*pitch_x*pitch_y + (j+1)*pitch_x + i;
                  s_vx0[tx] = d_vx0[offset];
                  s_vy0[tx] = d_vy0[offset];
                  s_vz0[tx] = d_vz0[offset];
               }
               // on charge les points pour k+2 (acces en mem globale)
               if (distance_zmax < 2) {
                  vx0_p2 = 0.f;
                  vy0_p2 = 0.f;
                  vz0_p2 = 0.f;
               } else {
                  vx0_p2 = d_vx0[(k+2)*pitch_x*pitch_y + j*pitch_x + i];
                  vy0_p2 = d_vy0[(k+2)*pitch_x*pitch_y + j*pitch_x + i];
                  vz0_p2 = d_vz0[(k+2)*pitch_x*pitch_y + j*pitch_x + i];
               }
            }/*}}}*/
            __syncthreads();
            offset = k*pitch_x*pitch_y + j*pitch_x + i;
            // calcul
            if (active) {/*{{{*/
               int npml = d_npml_tab[k*(sizex)*(sizey) + j*(sizex) + i];
               if ( npml >= 0){/*{{{*/
                  /* Calculation of txx, tyy and tzz */
                  if (distance_ymin >= 1 && distance_xmax >= 1 ){
                     float mux = 2.f/(1.f/s_mu[MUOFF(0,0)][0]  + 1.f/s_mu[MUOFF(1,0)][0]);
                     float vpx = 2.f/(1.f/s_vp[VPOFF(0,0)][0]  + 1.f/s_vp[VPOFF(1,0)][0]);
                     float lamx = 2.f/(1.f/s_lam[LAMOFF(0,0)]  + 1.f/s_lam[LAMOFF(1,0)]);
                     
                     float b1 = 4.f * mux * (lamx + mux) / (lamx + 2.f*mux);
                     float b2 = 2.f * mux * lamx / (lamx + 2.f*mux);
                     float phixdum =   d_phivxx[npml];
                     float phiydum = d_phivyy[npml];

                     phixdum = CPML2 (vpx, dumpx2(i), alphax2(i), kappax2(i), phixdum, ds, dt, s_vx0[VOFF(0,0)], s_vx0[VOFF(1,0)]);
                     phiydum = CPML2 (vpx, dumpy(j), alphay(j), kappay(j), phiydum, ds, dt, s_vy0[VOFF(0,-1)], s_vy0[VOFF(0,0)]);

                     d_txx0[offset] += b1*dt*phixdum + b2*dt*phiydum
                     + b1*dt*(s_vx0[VOFF(1,0)] - s_vx0[VOFF(0,0)])/(kappax2(i)*ds)
                     + b2*dt*(s_vy0[VOFF(0,0)] - s_vy0[VOFF(0,-1)])/(kappay(j)*ds);

                     d_tyy0[offset] += b1*dt*phiydum + b2*dt*phixdum
                     + b1*dt*(s_vy0[VOFF(0,0)] - s_vy0[VOFF(0,-1)])/(kappay(j)*ds)
                     + b2*dt*(s_vx0[VOFF(1,0)] - s_vx0[VOFF(0,0)])/(kappax2(i)*ds);
                  
                     d_tzz0[offset] = 0.f;
                     
                     d_phivxx[npml] = phixdum;
                     d_phivyy[npml] = phiydum;
                  } // if ( distance_zmin >= 1 && distance_ymin >= 1 && distance_xmax <= 1 )
                  /* Calculation of txy */
                  if ( distance_ymax >= 1 && distance_xmin >= 1 ){
                     float muy = 2.f/(1.f/s_mu[MUOFF(0,0)][0]  + 1.f/s_mu[MUOFF(0,1)][0]);
                     float vpy = 2.f/(1.f/s_vp[VPOFF(0,0)][0]  + 1.f/s_vp[VPOFF(0,1)][0]);

                     float phixdum =   d_phivyx[npml];
                     float phiydum = d_phivxy[npml];

                     phixdum = CPML2 (vpy, dumpx(i), alphax(i), kappax(i), phixdum, ds, dt,
                     s_vy0[VOFF(-1,0)], s_vy0[VOFF(0,0)]);

                     phiydum = CPML2 (vpy, dumpy2(j), alphay2(j), kappay2(j), phiydum, ds, dt,
                     s_vx0[VOFF(0,0)], s_vx0[VOFF(0,1)]);
                     
                     d_txy0[offset] += dt*muy*( phixdum + phiydum )
                     + staggardt2 (muy, kappax(i), kappay2(j), dt, ds,
                     s_vy0[VOFF(-1,0)], s_vy0[VOFF(0,0)],
                     s_vx0[VOFF(0,0)], s_vx0[VOFF(0,1)]);

                     d_phivyx[npml] = phixdum;
                     d_phivxy[npml] = phiydum;
                  }
                  d_txz0[offset] = - d_txz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i]; // calculé à l'itération précédente
                  d_tyz0[offset] = - d_txz0[(k-1)*pitch_x*pitch_y + j*pitch_x + i]; // calculé à l'itération précédente
               } else {/*}}}*/
                  float mux = 2.f/(1.f/s_mu[MUOFF(0,0)][0]  + 1.f/s_mu[MUOFF(1,0)][0]);
                  float lamx = 2.f/(1.f/s_lam[LAMOFF(0,0)]  + 1.f/s_lam[LAMOFF(1,0)]);
                  float muy = 2.f/(1.f/s_mu[MUOFF(0,0)][0]  + 1.f/s_mu[MUOFF(0,1)][0]);
                  float b1 = 4.f * mux * (lamx + mux) / (lamx + 2.f*mux);
                  float b2 = 2.f * mux * lamx / (lamx + 2.f*mux);

                  d_txx0[offset] += b1*dt*(s_vx0[VOFF(1,0)]-s_vx0[VOFF(0,0)])/ds + b2*dt*(s_vy0[VOFF(0,0)] - s_vy0[VOFF(0,-1)])/ds;
                  d_tyy0[offset] += b1*dt*(s_vy0[VOFF(0,0)]-s_vy0[VOFF(0,-1)])/ds   + b2*dt*(s_vx0[VOFF(1,0)]-s_vx0[VOFF(0,0)])/ds;
                  d_tzz0[offset] = 0.f;

                  d_txy0[offset] += staggardt2 (muy, 1.f, 1.f, dt, ds, s_vy0[VOFF(-1,0)], s_vy0[VOFF(0,0)], s_vx0[VOFF(0,0)], s_vx0[VOFF(0,1)]);
               }
            }/*}}}*/
#endif
            return;
         }
      // }}}
      // WRAPPER {{{
         void cuda_compute_stress_host (  float* d_txx0, float* d_tyy0, float* d_tzz0, float* d_txy0, float* d_txz0, float* d_tyz0,
                  float* d_vx0, float* d_vy0, float* d_vz0,
                  int* d_npml_tab, float* d_phivxx, float* d_phivxy, float* d_phivxz, float* d_phivyx, float* d_phivyy, float* d_phivyz, float* d_phivzx, float* d_phivzy, float* d_phivzz, 
                  float* d_mu, float* d_lam, float* d_vp, 
                  int sizex, int sizey, int sizez,
                  int pitch_x, int pitch_y, int pitch_z, 
                  float ds, float dt, int delta, int position,
                  int grid_x, int grid_y, int grid_z, int block_x, int block_y, int block_z) {

         #ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
            cudaThreadSynchronize();
            printCudaErr(cudaGetLastError(), "before computeStress kernel");
         #endif
            dim3 grid_dim(grid_x,grid_y,grid_z);
            dim3 block_dim(block_x, block_y, block_z);
            compute_stress_3d <<< grid_dim, block_dim, 0,starpu_cuda_get_local_stream() >>> ( d_txx0, d_tyy0, d_tzz0, d_txy0, d_txz0, d_tyz0,
                                                               d_vx0, d_vy0, d_vz0,
                                                               d_npml_tab, d_phivxx, d_phivxy, d_phivxz, d_phivyx, d_phivyy, d_phivyz, d_phivzx, d_phivzy, d_phivzz, 
                                                               d_mu, d_lam, d_vp, 
                                                               sizex, sizey, sizez,
                                                               pitch_x, pitch_y, pitch_z, 
                                                               ds, dt, delta, position);
         #ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
            cudaThreadSynchronize();
            printCudaErr(cudaGetLastError(), "after computeStress kernel");
         #endif
         }
      // }}}
   // }}}

   // COMPUTE VELOCITY {{{
      // IMPLEMENTATION {{{
#ifdef OPTI_ARCH_SM_20
__launch_bounds__(NPPDX*NPPDY, 5)
#endif
         __global__ void compute_veloc_3d (  float* d_txx0, float* d_tyy0, float* d_tzz0, float* d_txy0, float* d_txz0, float* d_tyz0,
                              float* d_vx0, float* d_vy0, float* d_vz0,
                              float* d_fx, float* d_fy, float* d_fz, 
                              int* d_npml_tab, float* d_phitxxx, float* d_phitxyy, float* d_phitxzz, float *d_phitxyx, float *d_phityyy, float *d_phityzz, float *d_phitxzx, float *d_phityzy, float *d_phitzzz,
                              float* d_vp, float* d_rho,
                              int sizex, int sizey, int sizez,
                              int pitch_x, int pitch_y, int pitch_z, 
                              float ds, float dt, int delta, int position)
         {

            __shared__ float s_rho[(NPPDX_K2+1)*(NPPDY_K2+1)][2];
 //           __shared__ float s_vp[(NPPDX_K2+1)*(NPPDY_K2+1)][2];

            __shared__ float s_txx0[(NPPDX_K2+4)*(NPPDY_K2+4)];
            __shared__ float s_tyy0[(NPPDX_K2+4)*(NPPDY_K2+4)];
            __shared__ float s_tzz0[(NPPDX_K2+4)*(NPPDY_K2+4)];
            __shared__ float s_txy0[(NPPDX_K2+4)*(NPPDY_K2+4)];
            __shared__ float s_txz0[(NPPDX_K2+4)*(NPPDY_K2+4)];
            __shared__ float s_tyz0[(NPPDX_K2+4)*(NPPDY_K2+4)];
            
            // m1 pour k-1, m2 pour k-2, p1 pour k+1, p2 pour k+2
            float tzz0_m1, tzz0_p1, tzz0_p2;
            float txz0_m1, txz0_m2, txz0_p1;// txz0_ip1_km1;
            float tyz0_m1, tyz0_m2, tyz0_p1;// tyz0_jm1_km1;
            
            int i = blockIdx.x*blockDim.x + threadIdx.x;
            int j = blockIdx.y*blockDim.y + threadIdx.y;
// {{{ CHANGED CODE
/*            int distance_xmin = i;
            int distance_xmax = sizex - i - 1;
            int distance_ymin = j;
            int distance_ymax = sizey - j - 1;
            int offset, offset_source, tx;
            bool last_x, last_y;

            //on ne calcule pas les points qui sont en dehors du domaine. Les threads correspondants ne font rien.   
            bool active = (distance_xmax >=0 && distance_ymax >=0)?true:false;

            last_x = last_y = false;
            if (distance_xmax == 0 || (active && threadIdx.x == (NPPDX-1))) {
               last_x = true;
            }
            if (distance_ymax == 0 || (active && threadIdx.y == (NPPDY-1))) {
               last_y = true;
            }

            // ici distance pour le modele global (on ne s'interesse qu'aux bords, donc si on n'est pas pres du bord, une valeur quelconque >2 suffit)
            distance_xmin = (position & MASK_FIRST_X)?i:DUMMY_VALUE;
            distance_xmax = (position & MASK_LAST_X)?(sizex - i - 1):DUMMY_VALUE;
            distance_ymin = (position & MASK_FIRST_Y)?j:DUMMY_VALUE;
            distance_ymax = (position & MASK_LAST_Y)?(sizey - j - 1):DUMMY_VALUE;*/
///}}}

/// {{{ NEW CODE
            int offset, offset_source, tx;
            for (int k = 0; k < sizez; k++) {
 
               __syncthreads();
               tx = threadIdx.y*(NPPDX_K2+1) + threadIdx.x;
	       
               offset = j*pitch_x + i;
	               s_rho[tx][0] = d_rho[offset];
        	       offset = pitch_x*(pitch_y) + j*pitch_x + i;
	               s_rho[tx][1] = d_rho[offset];
        	       
		       offset = pitch_x*(pitch_y) + j*pitch_x + i;
        	       offset_source = k*pitch_x*pitch_y + j*pitch_x + i;
	               s_txx0[tx] = d_txx0[offset];
	               s_tyy0[tx] = d_tyy0[offset];
	               s_tzz0[tx] = d_tzz0[offset];
        	       s_txy0[tx] = d_txy0[offset];
	               s_txz0[tx] = d_txz0[offset];
        	       s_tyz0[tx] = d_tyz0[offset];
 
	               tzz0_m1 = 0.5f;
	               txz0_m1 = txz0_m2 = 0.5f;
        	       tyz0_m1 = tyz0_m2 = 0.5f;
 //             	 txz0_ip1_km1 = 0.f;
 //	              tyz0_jm1_km1 = 0.f;

	               offset = pitch_x*pitch_y + j*pitch_x + i;
        	       tzz0_p1 = d_tzz0[offset];
	               txz0_p1 = d_txz0[offset];
        	       tyz0_p1 = d_tyz0[offset];

	               offset = 2*pitch_x*pitch_y + j*pitch_x + i;
        	       tzz0_p2 = d_tzz0[offset];

float rhoxy, rhoxz;
			rhoxy = 0.25f*(s_rho[RHOFF(0,0)][0] + s_rho[RHOFF(0,1)][0]
                                 + s_rho[RHOFF(1,0)][0] + s_rho[RHOFF(1,1)][0]);
 			rhoxz = 0.25f*(s_rho[RHOFF(0,0)][0] + s_rho[RHOFF(0,0)][1]
                                 + s_rho[RHOFF(1,0)][0] + s_rho[RHOFF(1,0)][1]);
                                               
	               __syncthreads();
	       	 if  (s_rho[RHOFF(0,0)][0] !=0) {
        	       d_vx0[offset] += (1.f/s_rho[RHOFF(0,0)][0])*d_fx[offset_source]*dt/ds
	                         + staggardv4 (1.f/s_rho[RHOFF(0,0)][0], 1.f, 1.f, 1.f, dt, ds,
        	                s_txx0[TOFF(-1,0)], s_txx0[TOFF(0,0)],
                	        s_txx0[TOFF(-2,0)], s_txx0[TOFF(1,0)],
                        	s_txy0[TOFF(0,-1)], s_txy0[TOFF(0,0)],
	                        s_txy0[TOFF(0,-2)], s_txy0[TOFF(0,1)],
        	                txz0_m1, s_txz0[TOFF(0,0)],
                	        txz0_m2, txz0_p1 );

	               d_vy0[offset] += (1.f/s_rho[RHOFF(0,0)][0])*d_fy[offset_source]*dt/ds
        	                + staggardv4 (1.f/rhoxy, 1.f, 1.f, 1.f, dt, ds,
                	        s_txy0[TOFF(0,0)], s_txy0[TOFF(1,0)],
                        	s_txy0[TOFF(-1,0)], s_txy0[TOFF(2,0)],
	                        s_tyy0[TOFF(0,0)], s_tyy0[TOFF(0,1)],
        	                s_tyy0[TOFF(0,-1)], s_tyy0[TOFF(0,2)],
                	        tyz0_m1, s_tyz0[TOFF(0,0)],
                        	tyz0_m2, tyz0_p1 );

	               d_vz0[offset] += (1.f/s_rho[RHOFF(0,0)][0])*d_fz[offset_source]*dt/ds
        	                + staggardv4 (1.f/rhoxz, 1.f, 1.f, 1.f, dt, ds,
                	        s_txz0[TOFF(0,0)], s_txz0[TOFF(1,0)],
                        	s_txz0[TOFF(-1,0)], s_txz0[TOFF(2,0)],
	                        s_tyz0[TOFF(0,-1)], s_tyz0[TOFF(0,0)],
        	                s_tyz0[TOFF(0,-2)], s_tyz0[TOFF(0,1)],
                	        s_tzz0[TOFF(0,0)], tzz0_p1,
	                        tzz0_m1, tzz0_p2 );
        	       __syncthreads();
	    }
	 }
/// }}}

///{{{ CHANGED CODE
/*
            // chargement initial des valeurs dans les registres et en memoire partagee
            if (active) {///{{{
               // le tableau est initialisé à zéro et ces éléments ne sont jamais mis à jour
               tzz0_m1 = 0.f;
               txz0_m1 = txz0_m2 = 0.f;
               tyz0_m1 = tyz0_m2 = 0.f;
               txz0_ip1_km1 = 0.f;
               tyz0_jm1_km1 = 0.f;

               offset = pitch_x*pitch_y + j*pitch_x + i;
               tzz0_p1 = d_tzz0[offset];
               txz0_p1 = d_txz0[offset];
               tyz0_p1 = d_tyz0[offset];

               offset = 2*pitch_x*pitch_y + j*pitch_x + i;
               tzz0_p2 = d_tzz0[offset];

               // chaque thread charge une donnée du domaine pour k=0
               tx = threadIdx.y*(NPPDX_K2+1) + threadIdx.x;
               offset = j*pitch_x + i;
               s_rho[tx][0] = d_rho[offset];
               s_vp[tx][0] = d_vp[offset];
               // maintenant, chaque thread charge une donnée pour k=1 pour rho et vp
               offset = pitch_x*(pitch_y) + j*pitch_x + i;
               s_rho[tx][1] = d_rho[offset];
               s_vp[tx][1] = d_vp[offset];
               // maintenant, on charge les données pour i+1(hors du block)
               if (last_x) {
                  tx = threadIdx.y*(NPPDX_K2+1) +threadIdx.x+1;
                  offset = j*pitch_x + i+1;
                  s_rho[tx][0] = d_rho[offset];
                  s_vp[tx][0] = d_vp[offset];
                  // i+1, k+1
                  offset = pitch_x*(pitch_y) + j*pitch_x + i+1;
                  s_rho[tx][1] = d_rho[offset];
                  s_vp[tx][1] = d_vp[offset];
               }
               // maintenant, on charge les données pour j+1(hors du block) : uniquement vp et rho
               if (last_y) {
                  tx = (threadIdx.y+1)*(NPPDX_K2+1) + threadIdx.x;
                  offset = (j+1)*pitch_x + i;
                  s_rho[tx][0] = d_rho[offset];
                  s_vp[tx][0] = d_vp[offset];
                  offset = pitch_x*(pitch_y) + (j+1)*pitch_x + i;
                  s_rho[tx][1] = d_rho[offset];
                  s_vp[tx][1] = d_vp[offset];
               }
               // maintenant, on charge la donnée pour j+1 & i+1(hors du block) : uniquement vp et rho
               if (last_y && last_x) {
                  tx = (threadIdx.y+1)*(NPPDX_K2+1) + threadIdx.x+1;
                  offset = (j+1)*pitch_x + i+1;
                  s_rho[tx][0] = d_rho[offset];
                  s_vp[tx][0] = d_vp[offset];
                  offset = pitch_x*(pitch_y) + (j+1)*pitch_x + i+1;
                  s_rho[tx][1] = d_rho[offset];
                  s_vp[tx][1] = d_vp[offset];
               }
               // vx0, vy0, vz0
               // chaque thread charge sa valeur en shmem
               tx = (threadIdx.y+2)*(NPPDX_K2+4) + threadIdx.x+2;
               offset = j*pitch_x + i;
               s_txx0[tx] = d_txx0[offset];
               s_tyy0[tx] = d_tyy0[offset];
               s_tzz0[tx] = d_tzz0[offset];
               s_txy0[tx] = d_txy0[offset];
               s_txz0[tx] = d_txz0[offset];
               s_tyz0[tx] = d_tyz0[offset];
               // on charge les deux rangées i-1 et i-2
               if (threadIdx.x == 0) {
                  // i-2
                  tx = (threadIdx.y+2)*(NPPDX_K2+4) + 0;
                  offset = j*pitch_x + i-2;
                  s_txx0[tx] = d_txx0[offset];
                  s_tyy0[tx] = d_tyy0[offset];
                  s_tzz0[tx] = d_tzz0[offset];
                  s_txy0[tx] = d_txy0[offset];
                  s_txz0[tx] = d_txz0[offset];
                  s_tyz0[tx] = d_tyz0[offset];
                  // i-1
                  tx = (threadIdx.y+2)*(NPPDX_K2+4) + 1;
                  offset = j*pitch_x + i-1;
                  s_txx0[tx] = d_txx0[offset];
                  s_tyy0[tx] = d_tyy0[offset];
                  s_tzz0[tx] = d_tzz0[offset];
                  s_txy0[tx] = d_txy0[offset];
                  s_txz0[tx] = d_txz0[offset];
                  s_tyz0[tx] = d_tyz0[offset];
               }
               // on charge les deux rangées i+1 et i+2
               if (last_x) {
                  // i+1
                  tx = (threadIdx.y+2)*(NPPDX_K2+4) + threadIdx.x + 3;
                  offset = j*pitch_x + i + 1;
                  s_txx0[tx] = d_txx0[offset];
                  s_tyy0[tx] = d_tyy0[offset];
                  s_tzz0[tx] = d_tzz0[offset];
                  s_txy0[tx] = d_txy0[offset];
                  s_txz0[tx] = d_txz0[offset];
                  s_tyz0[tx] = d_tyz0[offset];
                  // i+2
                  tx = (threadIdx.y+2)*(NPPDX_K2+4) + threadIdx.x + 4;
                  offset = j*pitch_x + i + 2;
                  s_txx0[tx] = d_txx0[offset];
                  s_tyy0[tx] = d_tyy0[offset];
                  s_tzz0[tx] = d_tzz0[offset];
                  s_txy0[tx] = d_txy0[offset];
                  s_txz0[tx] = d_txz0[offset];
                  s_tyz0[tx] = d_tyz0[offset];
               }
               // on charge les rangées j-1 et j-2
               if (threadIdx.y == 0) {
                  // j-2
                  tx = threadIdx.x+2;
                  offset = (j-2)*pitch_x + i;
                  s_txx0[tx] = d_txx0[offset];
                  s_tyy0[tx] = d_tyy0[offset];
                  s_tzz0[tx] = d_tzz0[offset];
                  s_txy0[tx] = d_txy0[offset];
                  s_txz0[tx] = d_txz0[offset];
                  s_tyz0[tx] = d_tyz0[offset];
                  // j-1
                  tx = (NPPDX_K2+4) + threadIdx.x+2;
                  offset = (j-1)*pitch_x + i;
                  s_txx0[tx] = d_txx0[offset];
                  s_tyy0[tx] = d_tyy0[offset];
                  s_tzz0[tx] = d_tzz0[offset];
                  s_txy0[tx] = d_txy0[offset];
                  s_txz0[tx] = d_txz0[offset];
                  s_tyz0[tx] = d_tyz0[offset];
               }
               // on charge les rangées j+1 et j+2
               if (last_y) {
                  // j+2
                  tx = (threadIdx.y + 4)*(NPPDX_K2+4) + threadIdx.x+2;
                  offset = (j+2)*pitch_x + i;
                  s_txx0[tx] = d_txx0[offset];
                  s_tyy0[tx] = d_tyy0[offset];
                  s_tzz0[tx] = d_tzz0[offset];
                  s_txy0[tx] = d_txy0[offset];
                  s_txz0[tx] = d_txz0[offset];
                  s_tyz0[tx] = d_tyz0[offset];
                  // j+1
                  tx = (threadIdx.y + 3)*(NPPDX_K2+4) + threadIdx.x+2;
                  offset = (j+1)*pitch_x + i;
                  s_txx0[tx] = d_txx0[offset];
                  s_tyy0[tx] = d_tyy0[offset];
                  s_tzz0[tx] = d_tzz0[offset];
                  s_txy0[tx] = d_txy0[offset];
                  s_txz0[tx] = d_txz0[offset];
                  s_tyz0[tx] = d_tyz0[offset];
               }
            }///}}}
            int npml=-2;
            //float rhoxy, rhoxz;
            // boucle sur z
            for (int k = 0; k < sizez; k++) {///{{{
               int distance_zmin = k;
               int distance_zmax = sizez - k - 1;
               
               // apres decalage de la fenetre, on decalle les valeurs selon l'axe des Z.
               if (active) {///{{{
                  if (k>0) {
                     // chaque thread charge une donnée du domaine
                     tx = threadIdx.y*(NPPDX_K2+1) + threadIdx.x;
                     // on a déjà lu les données pour k à l'itération précédente pour rho et vp
                     s_rho[tx][0] = s_rho[tx][1];
                     s_vp[tx][0] = s_vp[tx][1];
                     // maintenant, chaque thread charge une donnée pour k+1 pour rho et vp
                     offset = (k+1)*pitch_x*(pitch_y) + j*pitch_x + i;
                     s_rho[tx][1] = d_rho[offset];
                     s_vp[tx][1] = d_vp[offset];
                     // maintenant, on charge les données pour i+1(hors du block)
                     if (last_x) {
                        // i+1
                        tx = threadIdx.y*(NPPDX_K2+1) +threadIdx.x+1;
                        s_rho[tx][0] = s_rho[tx][1];
                        s_vp[tx][0] = s_vp[tx][1];
                        // i+1, k+1
                        offset = (k+1)*pitch_x*(pitch_y)+j*pitch_x + i+1;
                        s_rho[tx][1] = d_rho[offset];
                        s_vp[tx][1] = d_vp[offset];
                     }
                     // maintenant, on charge les données pour j+1(hors du block) : uniquement vp et rho
                     if (last_y) {
                        tx = (threadIdx.y+1)*(NPPDX_K2+1) +threadIdx.x;
                        s_rho[tx][0] = s_rho[tx][1];
                        s_vp[tx][0] = s_vp[tx][1];
                        offset = (k+1)*pitch_x*(pitch_y) + (j+1)*pitch_x + i;
                        s_rho[tx][1] = d_rho[offset];
                        s_vp[tx][1] = d_vp[offset];
                     }
                     // maintenant, on charge la donnée pour j+1 & i+1(hors du block) : uniquement vp et rho
                     if (last_y && last_x) {
                        tx = (threadIdx.y+1)*(NPPDX_K2+1) + threadIdx.x+1;
                        s_rho[tx][0] = s_rho[tx][1];
                        s_vp[tx][0] = s_vp[tx][1];
                        offset = (k+1)*pitch_x*(pitch_y) + (j+1)*pitch_x + i+1;
                        s_rho[tx][1] = d_rho[offset];
                        s_vp[tx][1] = d_vp[offset];
                     }
                     // décalage des données selon l'axe Z
                     txz0_ip1_km1 = s_txz0[(threadIdx.y+2)*(NPPDX_K2+4) + threadIdx.x+2+1];
                     tyz0_jm1_km1 = s_tyz0[(threadIdx.y+2-1)*(NPPDX_K2+4) + threadIdx.x+2];
                  }
               }///}}}
               // synchro avant d'ecraser s_txz0 et s_tyz0
               __syncthreads();
               if (active) {///{{{
                  if (k>0) {///{{{
                     tx = (threadIdx.y+2)*(NPPDX_K2+4) + threadIdx.x+2;
                     tzz0_m1 = s_tzz0[tx];
                     s_tzz0[tx] = tzz0_p1;
                     tzz0_p1 = tzz0_p2;
                     
                     txz0_m2 = txz0_m1;
                     txz0_m1 = s_txz0[tx];
                     s_txz0[tx] = txz0_p1;
                     
                     tyz0_m2 = tyz0_m1;
                     tyz0_m1 = s_tyz0[tx];
                     s_tyz0[tx] = tyz0_p1;

                     // remarque : le tableau dépasse de un point ds chaque direction du domaine calculé, donc si on est au bord du domaine,
                     // i+1 est défini et la valeur à ce point est nulle (car hors du domaine)
                     // i+2 correspond à la valeur pour j+1 et i=-1, cad 0 car hors du domaine => ça reste cohérent
                     // idem pour i - 1 et i - 2 (i-2 toujours défini à cause du padding d'alignement)
                     tx = (threadIdx.y+2)*(NPPDX_K2+4) + threadIdx.x+2;
                     offset = k*pitch_x*pitch_y + j*pitch_x + i;
                     s_txx0[tx] = d_txx0[offset];
                     s_tyy0[tx] = d_tyy0[offset];
                     // ! ici !!! : txy a la place de tyz
                     s_txy0[tx] = d_txy0[offset];
                     // on charge les deux rangées i-1 et i-2
                     if (threadIdx.x == 0) {
                        // i-2
                        tx = (threadIdx.y+2)*(NPPDX_K2+4) + 0;
                        offset = k*pitch_x*pitch_y + j*pitch_x + i-2;
                        s_txx0[tx] = d_txx0[offset];
                        s_tyy0[tx] = d_tyy0[offset];
                        s_tzz0[tx] = d_tzz0[offset];
                        s_txy0[tx] = d_txy0[offset];
                        s_txz0[tx] = d_txz0[offset];
                        s_tyz0[tx] = d_tyz0[offset];
                        // i-1
                        tx = (threadIdx.y+2)*(NPPDX_K2+4) + 1;
                        offset = k*pitch_x*pitch_y + j*pitch_x + i-1;
                        s_txx0[tx] = d_txx0[offset];
                        s_tyy0[tx] = d_tyy0[offset];
                        s_tzz0[tx] = d_tzz0[offset];
                        s_txy0[tx] = d_txy0[offset];
                        s_txz0[tx] = d_txz0[offset];
                        s_tyz0[tx] = d_tyz0[offset];
                     }
                     // on charge les deux rangées i+1 et i+2
                     if (last_x) {
                        // i+1
                        tx = (threadIdx.y+2)*(NPPDX_K2+4) + threadIdx.x + 3;
                        offset = k*pitch_x*pitch_y + j*pitch_x + i + 1;
                        s_txx0[tx] = d_txx0[offset];
                        s_tyy0[tx] = d_tyy0[offset];
                        s_tzz0[tx] = d_tzz0[offset];
                        s_txy0[tx] = d_txy0[offset];
                        s_txz0[tx] = d_txz0[offset];
                        s_tyz0[tx] = d_tyz0[offset];
                        // i+2
                        tx = (threadIdx.y+2)*(NPPDX_K2+4) + threadIdx.x + 4;
                        offset = k*pitch_x*pitch_y + j*pitch_x + i + 2;
                        s_txx0[tx] = d_txx0[offset];
                        s_tyy0[tx] = d_tyy0[offset];
                        s_tzz0[tx] = d_tzz0[offset];
                        s_txy0[tx] = d_txy0[offset];
                        s_txz0[tx] = d_txz0[offset];
                        s_tyz0[tx] = d_tyz0[offset];
                     }
                     // on charge les rangées j-1 et j-2
                     if (threadIdx.y == 0) {
                        // j-2
                        tx = threadIdx.x+2;
                        offset = k*pitch_x*pitch_y + (j-2)*pitch_x + i;
                        s_txx0[tx] = d_txx0[offset];
                        s_tyy0[tx] = d_tyy0[offset];
                        s_tzz0[tx] = d_tzz0[offset];
                        s_txy0[tx] = d_txy0[offset];
                        s_txz0[tx] = d_txz0[offset];
                        s_tyz0[tx] = d_tyz0[offset];
                        // j-1
                        tx = (NPPDX_K2+4) + threadIdx.x+2;
                        offset = k*pitch_x*pitch_y + (j-1)*pitch_x + i;
                        s_txx0[tx] = d_txx0[offset];
                        s_tyy0[tx] = d_tyy0[offset];
                        s_tzz0[tx] = d_tzz0[offset];
                        s_txy0[tx] = d_txy0[offset];
                        s_txz0[tx] = d_txz0[offset];
                        s_tyz0[tx] = d_tyz0[offset];
                     }
                     // on charge les rangées j+1 et j+2
                     if (last_y) {
                        // j+2
                        tx = (threadIdx.y + 4)*(NPPDX_K2+4) + threadIdx.x+2;
                        offset = k*pitch_x*pitch_y + (j+2)*pitch_x + i;
                        s_txx0[tx] = d_txx0[offset];
                        s_tyy0[tx] = d_tyy0[offset];
                        s_tzz0[tx] = d_tzz0[offset];
                        s_txy0[tx] = d_txy0[offset];
                        s_txz0[tx] = d_txz0[offset];
                        s_tyz0[tx] = d_tyz0[offset];
                        // j+1
                        tx = (threadIdx.y + 3)*(NPPDX_K2+4) + threadIdx.x+2;
                        offset = k*pitch_x*pitch_y + (j+1)*pitch_x + i;
                        s_txx0[tx] = d_txx0[offset];
                        s_tyy0[tx] = d_tyy0[offset];
                        s_tzz0[tx] = d_tzz0[offset];
                        s_txy0[tx] = d_txy0[offset];
                        s_txz0[tx] = d_txz0[offset];
                        s_tyz0[tx] = d_tyz0[offset];
                     }
                     // on charge les points pour k+2 (acces en mem globale)
                     if (distance_zmax < 2) {
                        tzz0_p2 = 0.f;
                     } else {
                        tzz0_p2 = d_tzz0[(k+2)*pitch_x*pitch_y + j*pitch_x + i];
                     }
                     if (distance_zmax < 1) {
                        txz0_p1 = 0.f;
                        tyz0_p1 = 0.f;
                     } else {
                        txz0_p1 = d_txz0[(k+1)*pitch_x*pitch_y + j*pitch_x + i];
                        tyz0_p1 = d_tyz0[(k+1)*pitch_x*pitch_y + j*pitch_x + i];
                     }
                  }///}}}
               }///}}}
               
               // synchro : on attends que tous les threads aient fini d'ecrire dans la memoire partagee
               __syncthreads();
               
               offset = k*pitch_x*pitch_y + j*pitch_x + i;
               offset_source = k*pitch_x*pitch_y + j*pitch_x + i;
            
               if (active) {///{{{
                  npml = d_npml_tab[k*(sizex)*(sizey) + j*(sizex) + i];
                  if (npml >= 0) {///{{{
                     //CPML
                     // ICI !!!!!!!!

             //        if (distance_zmin >= 1 && distance_ymin >= 1 && distance_xmin >= 1) {/// VX {{{
                        // Calculation of vx 
	//		   float phixdum = d_phitxxx[npml];
          //                 float phiydum = d_phitxyy[npml];
            //               float phizdum = d_phitxzz[npml];

                           phixdum = CPML4 (s_vp[VPOFF(0,0)][0], dumpx(i), alphax(i), kappax(i), phixdum, ds, dt,
                           s_txx0[TOFF(-1,0)], s_txx0[TOFF(0,0)],
                           s_txx0[TOFF(-2,0)], s_txx0[TOFF(1,0)] );
                           phiydum = CPML4 (s_vp[VPOFF(0,0)][0], dumpy(j), alphay(j), kappay(j), phiydum, ds, dt,
                           s_txy0[TOFF(0,-1)], s_txy0[TOFF(0,0)],
                           s_txy0[TOFF(0,-2)], s_txy0[TOFF(0,1)] );
                           phizdum = CPML4 (s_vp[VPOFF(0,0)][0], dumpz(k), alphaz(k), kappaz(k), phizdum, ds, dt,
                           txz0_m1, s_txz0[TOFF(0,0)],
                           txz0_m2, txz0_p1 );

                         
				d_vx0[offset] += (dt/s_rho[RHOFF(0,0)][0])
					+ staggardv4 (1.f/s_rho[RHOFF(0,0)][0], kappax(i), kappay(j), kappaz(k), dt, ds,
							s_txx0[TOFF(-1,0)], s_txx0[TOFF(0,0)],
							s_txx0[TOFF(-2,0)], s_txx0[TOFF(1,0)],
							s_txy0[TOFF(0,-1)], s_txy0[TOFF(0,0)],
							s_txy0[TOFF(0,-2)], s_txy0[TOFF(0,1)],
							txz0_m1, s_txz0[TOFF(0,0)],
							txz0_m2, txz0_p1 );



                        if ( distance_zmax == 0 ){ // free surface
                           float phixdum = d_phitxxx[npml];
                           float phiydum = d_phitxyy[npml];
                           float phizdum = d_phitxzz[npml];

                           phixdum = CPML2 (s_vp[VPOFF(0,0)][0], dumpx(i), alphax(i), kappax(i), phixdum, ds, dt, s_txx0[TOFF(-1,0)], s_txx0[TOFF(0,0)] );
                           phiydum = CPML2 (s_vp[VPOFF(0,0)][0], dumpy(j), alphay(j), kappay(j), phiydum, ds, dt, s_txy0[TOFF(0,-1)], s_txy0[TOFF(0,0)] );
                           phizdum = CPML2 (s_vp[VPOFF(0,0)][0], dumpz(k), alphaz(k), kappaz(k), phizdum, ds, dt, txz0_m1, - txz0_m1 );

                           if (distance_xmax == 0 || distance_ymax ==0) { // boundary condition
                              d_vx0[offset] = 0.0f;
                           } else {
                              d_vx0[offset] += (dt/s_rho[RHOFF(0,0)][0])*( phixdum + phiydum + phizdum )
                              + staggardv2 (1.f/s_rho[RHOFF(0,0)][0], kappax(i), kappay(j), kappaz(k), dt, ds,
                              s_txx0[TOFF(-1,0)], s_txx0[TOFF(0,0)],
                              s_txy0[TOFF(0,-1)], s_txy0[TOFF(0,0)],
                              txz0_m1, - txz0_m1 );
                           }
                           
                           d_phitxxx[npml] = phixdum;
                           d_phitxyy[npml] = phiydum;
                           d_phitxzz[npml] = phizdum;
                        } else if ( distance_zmax == 1 ){ // in the first cell, 2nd order finite-difference instead of 4th order finite-difference
                           float phixdum = d_phitxxx[npml];
                           float phiydum = d_phitxyy[npml];
                           float phizdum = d_phitxzz[npml];

                           phixdum = CPML2 (s_vp[VPOFF(0,0)][0], dumpx(i), alphax(i), kappax(i), phixdum, ds, dt,
                           s_txx0[TOFF(-1,0)], s_txx0[TOFF(0,0)] );
                           phiydum = CPML2 (s_vp[VPOFF(0,0)][0], dumpy(j), alphay(j), kappay(j), phiydum, ds, dt,
                           s_txy0[TOFF(0,-1)], s_txy0[TOFF(0,0)] );
                           phizdum = CPML2 (s_vp[VPOFF(0,0)][0], dumpz(k), alphaz(k), kappaz(k), phizdum, ds, dt,
                           txz0_m1, s_txz0[TOFF(0,0)] );
                           
                           if (distance_xmax == 0 || distance_ymax ==0) { // boundary condition
                              d_vx0[offset] = 0.0f;
                           } else {
                              d_vx0[offset] += (dt/s_rho[RHOFF(0,0)][0])*( phixdum + phiydum + phizdum )
                              + staggardv2 (1.f/s_rho[RHOFF(0,0)][0], kappax(i), kappay(j), kappaz(k), dt, ds,
                              s_txx0[TOFF(-1,0)], s_txx0[TOFF(0,0)],
                              s_txy0[TOFF(0,-1)], s_txy0[TOFF(0,0)],
                              txz0_m1, s_txz0[TOFF(0,0)] );
                           }
                           
                           d_phitxxx[npml] = phixdum;
                           d_phitxyy[npml] = phiydum;
                           d_phitxzz[npml] = phizdum;
                        } else { // regular domain
                           // ICI !!!!!!!!
                           float phixdum = d_phitxxx[npml];
                           float phiydum = d_phitxyy[npml];
                           float phizdum = d_phitxzz[npml];

                           phixdum = CPML4 (s_vp[VPOFF(0,0)][0], dumpx(i), alphax(i), kappax(i), phixdum, ds, dt,
                           s_txx0[TOFF(-1,0)], s_txx0[TOFF(0,0)],
                           s_txx0[TOFF(-2,0)], s_txx0[TOFF(1,0)] );
                           phiydum = CPML4 (s_vp[VPOFF(0,0)][0], dumpy(j), alphay(j), kappay(j), phiydum, ds, dt,
                           s_txy0[TOFF(0,-1)], s_txy0[TOFF(0,0)],
                           s_txy0[TOFF(0,-2)], s_txy0[TOFF(0,1)] );
                           phizdum = CPML4 (s_vp[VPOFF(0,0)][0], dumpz(k), alphaz(k), kappaz(k), phizdum, ds, dt,
                           txz0_m1, s_txz0[TOFF(0,0)],
                           txz0_m2, txz0_p1 );

                           if (distance_xmax == 0 || distance_ymax ==0) { // boundary condition
                              d_vx0[offset] = 0.0f;
                           } else {
                              d_vx0[offset] += (dt/s_rho[RHOFF(0,0)][0])*( phixdum + phiydum + phizdum )
                              + staggardv4 (1.f/s_rho[RHOFF(0,0)][0], kappax(i), kappay(j), kappaz(k), dt, ds,
                              s_txx0[TOFF(-1,0)], s_txx0[TOFF(0,0)],
                              s_txx0[TOFF(-2,0)], s_txx0[TOFF(1,0)],
                              s_txy0[TOFF(0,-1)], s_txy0[TOFF(0,0)],
                              s_txy0[TOFF(0,-2)], s_txy0[TOFF(0,1)],
                              txz0_m1, s_txz0[TOFF(0,0)],
                              txz0_m2, txz0_p1 );
                           }
                              
                           d_phitxxx[npml] = phixdum;
                           d_phitxyy[npml] = phiydum;
                           d_phitxzz[npml] = phizdum;
                        } // end of if "free surface"






               //      }///}}}


                     // Calculation of vy
                     if ( distance_zmin >= 1 && distance_ymax >= 1 && distance_xmax >= 1 ){///VY{{{
                        rhoxy = 0.25f*(s_rho[RHOFF(0,0)][0] + s_rho[RHOFF(0,1)][0]
                                 + s_rho[RHOFF(1,0)][0] + s_rho[RHOFF(1,1)][0]);
                        float vpxy = 0.25f*(s_vp[VPOFF(0,0)][0] + s_vp[VPOFF(0,1)][0]
                                 + s_vp[VPOFF(1,0)][0] + s_vp[VPOFF(1,1)][0]);
                        if ( distance_zmax == 0 ){ // free surface
                           float phixdum = d_phitxyx[npml];
                           float phiydum = d_phityyy[npml];
                           float phizdum = d_phityzz[npml];

                           phixdum = CPML2 (vpxy, dumpx2(i), alphax2(i), kappax2(i), phixdum, ds, dt,
                           s_txy0[TOFF(0,0)], s_txy0[TOFF(1,0)] );
                           phiydum = CPML2 (vpxy, dumpy2(j), alphay2(j), kappay2(j), phiydum, ds, dt,
                           s_tyy0[TOFF(0,0)], s_tyy0[TOFF(0,1)] );
                           phizdum = CPML2 (vpxy, dumpz(k), alphaz(k), kappaz(k), phizdum, ds, dt,
                           tyz0_m1, - tyz0_m1 );
                           
                           if (distance_xmin == 0 || distance_ymin == 0) { // boundary condition
                              d_vy0[offset] = 0.0f;
                           } else {
                              d_vy0[offset] += (dt/rhoxy)*( phixdum + phiydum + phizdum )
                              + staggardv2 (1.f/rhoxy, kappax2(i), kappay2(j), kappaz(k), dt, ds,
                              s_txy0[TOFF(0,0)], s_txy0[TOFF(1,0)],
                              s_tyy0[TOFF(0,0)], s_tyy0[TOFF(0,1)],
                              tyz0_m1, - tyz0_m1 );
                           }
                           
                           d_phitxyx[npml] = phixdum;
                           d_phityyy[npml] = phiydum;
                           d_phityzz[npml] = phizdum;
                        } else if ( distance_zmax == 1 ){ // in the first cell, 2nd order finite-difference instead of 4th order finite-difference
                           float phixdum = d_phitxyx[npml];
                           float phiydum = d_phityyy[npml];
                           float phizdum = d_phityzz[npml];

                           phixdum = CPML2 (vpxy, dumpx2(i), alphax2(i), kappax2(i), phixdum, ds, dt,
                           s_txy0[TOFF(0,0)], s_txy0[TOFF(1,0)] );
                           phiydum = CPML2 (vpxy, dumpy2(j), alphay2(j), kappay2(j), phiydum, ds, dt,
                           s_tyy0[TOFF(0,0)], s_tyy0[TOFF(0,1)] );
                           phizdum = CPML2 (vpxy, dumpz(k), alphaz(k), kappaz(k), phizdum, ds, dt,
                           tyz0_m1, s_tyz0[TOFF(0,0)] );

                           if (distance_xmin == 0 || distance_ymin == 0) { // boundary condition
                              d_vy0[offset] = 0.0f;
                           } else {
                              d_vy0[offset] += (dt/rhoxy)*( phixdum + phiydum + phizdum )
                              + staggardv2 (1.f/rhoxy, kappax2(i), kappay2(j), kappaz(k), dt, ds,
                              s_txy0[TOFF(0,0)], s_txy0[TOFF(1,0)],
                              s_tyy0[TOFF(0,0)], s_tyy0[TOFF(0,1)],
                              tyz0_m1, s_tyz0[TOFF(0,0)] );
                           }

                           d_phitxyx[npml] = phixdum;
                           d_phityyy[npml] = phiydum;
                           d_phityzz[npml] = phizdum;
                        } else { // regular domain
                           float phixdum = d_phitxyx[npml];
                           float phiydum = d_phityyy[npml];
                           float phizdum = d_phityzz[npml];
                           
                           phixdum = CPML4 (vpxy, dumpx2(i), alphax2(i), kappax2(i), phixdum, ds, dt,
                           s_txy0[TOFF(0,0)], s_txy0[TOFF(1,0)],
                           s_txy0[TOFF(-1,0)], s_txy0[TOFF(2,0)] );
                           phiydum = CPML4 (vpxy, dumpy2(j), alphay2(j), kappay2(j), phiydum, ds, dt,
                           s_tyy0[TOFF(0,0)], s_tyy0[TOFF(0,1)],
                           s_tyy0[TOFF(0,-1)], s_tyy0[TOFF(0,2)] );
                           phizdum = CPML4 (vpxy, dumpz(k), alphaz(k), kappaz(k), phizdum, ds, dt,
                           tyz0_m1, s_tyz0[TOFF(0,0)],
                           tyz0_m2, tyz0_p1 );

                           if (distance_xmin == 0 || distance_ymin == 0) { // boundary condition
                              d_vy0[offset] = 0.0f;
                           } else {
                              d_vy0[offset] += (dt/rhoxy)*( phixdum + phiydum + phizdum )
                              + staggardv4 (1.f/rhoxy, kappax2(i), kappay2(j), kappaz(k), dt, ds,
                              s_txy0[TOFF(0,0)], s_txy0[TOFF(1,0)],
                              s_txy0[TOFF(-1,0)], s_txy0[TOFF(2,0)],
                              s_tyy0[TOFF(0,0)], s_tyy0[TOFF(0,1)],
                              s_tyy0[TOFF(0,-1)], s_tyy0[TOFF(0,2)],
                              tyz0_m1, s_tyz0[TOFF(0,0)],
                              tyz0_m2, tyz0_p1 );
                           }
                           
                           d_phitxyx[npml] = phixdum;
                           d_phityyy[npml] = phiydum;
                           d_phityzz[npml] = phizdum;
                        } // end of if "free surface"
                     }///}}}
                     // Calculation of vz
                     if ( distance_ymin >= 1 && distance_xmax >= 1 ){///VZ{{{
                        rhoxz = 0.25f*(s_rho[RHOFF(0,0)][0] + s_rho[RHOFF(0,0)][1]
                                 + s_rho[RHOFF(1,0)][0] + s_rho[RHOFF(1,0)][1]);
                        float vpxz = 0.25f*(s_vp[VPOFF(0,0)][0] + s_vp[VPOFF(0,0)][1]
                                 + s_vp[VPOFF(1,0)][0] + s_vp[VPOFF(1,0)][1]);
                        if ( distance_zmax == 0 ){ // free surface
                           float phixdum = d_phitxzx[npml];
                           float phiydum = d_phityzy[npml];
                           float phizdum = d_phitzzz[npml];

                           phixdum = CPML2 (vpxz, dumpx2(i), alphax2(i), kappax2(i), phixdum, ds, dt,
                           - txz0_m1, - txz0_ip1_km1 );
                           phiydum = CPML2 (vpxz, dumpy(j), alphay(j), kappay(j), phiydum, ds, dt,
                           - tyz0_jm1_km1, - tyz0_m1 );
                           phizdum = CPML2 (vpxz, dumpz2(k), alphaz2(k), kappaz2(k), phizdum, ds, dt,
                           s_tzz0[TOFF(0,0)], - tzz0_m1 );

                           if (distance_xmin == 0 || distance_ymax ==0) { // boundary condition
                              d_vz0[offset] = 0.0f;
                           } else {
                              d_vz0[offset] += (dt/rhoxz)*( phixdum + phiydum + phizdum )
                              + staggardv2 (1.f/rhoxz, kappax2(i), kappay(j), kappaz2(k), dt, ds,
                              - txz0_m1, - txz0_ip1_km1,
                              - tyz0_jm1_km1, - tyz0_m1,
                              s_tzz0[TOFF(0,0)], - tzz0_m1 );
                           }
                                          
                           d_phitxzx[npml] = phixdum;
                           d_phityzy[npml] = phiydum;
                           d_phitzzz[npml] = phizdum;
                        } else if ( distance_zmax == 1 ){ // in the first cell, 2nd order finite-difference instead of 4th order finite-difference
                           float phixdum = d_phitxzx[npml];
                           float phiydum = d_phityzy[npml];
                           float phizdum = d_phitzzz[npml];

                           phixdum = CPML2 (vpxz, dumpx2(i), alphax2(i), kappax2(i), phixdum, ds, dt,
                           s_txz0[TOFF(0,0)], s_txz0[TOFF(1,0)] );
                           phiydum = CPML2 (vpxz, dumpy(j), alphay(j), kappay(j), phiydum, ds, dt,
                           s_tyz0[TOFF(0,-1)], s_tyz0[TOFF(0,0)] );
                           phizdum = CPML2 (vpxz, dumpz2(k), alphaz2(k), kappaz2(k), phizdum, ds, dt,
                           s_tzz0[TOFF(0,0)], tzz0_p1 );

                           
                           if (distance_xmin == 0 || distance_ymax ==0) { // boundary condition
                              d_vz0[offset] = 0.0f;
                           } else {
                              d_vz0[offset] += (dt/rhoxz)*( phixdum + phiydum + phizdum )
                              + staggardv2 (1.f/rhoxz, kappax2(i), kappay(j), kappaz2(k), dt, ds,
                              s_txz0[TOFF(0,0)], s_txz0[TOFF(1,0)],
                              s_tyz0[TOFF(0,-1)], s_tyz0[TOFF(0,0)],
                              s_tzz0[TOFF(0,0)], tzz0_p1 );
                           }
                           
                           d_phitxzx[npml] = phixdum;
                           d_phityzy[npml] = phiydum;
                           d_phitzzz[npml] = phizdum;
                        } else { // regular domain
                           float phixdum = d_phitxzx[npml];
                           float phiydum = d_phityzy[npml];
                           float phizdum = d_phitzzz[npml];

                           phixdum = CPML4 (vpxz, dumpx2(i), alphax2(i), kappax2(i), phixdum, ds, dt,
                           s_txz0[TOFF(0,0)], s_txz0[TOFF(1,0)],
                           s_txz0[TOFF(-1,0)], s_txz0[TOFF(2,0)] );
                           phiydum = CPML4 (vpxz, dumpy(j), alphay(j), kappay(j), phiydum, ds, dt,
                           s_tyz0[TOFF(0,-1)], s_tyz0[TOFF(0,0)],
                           s_tyz0[TOFF(0,-2)], s_tyz0[TOFF(0,1)] );
                           phizdum = CPML4 (vpxz, dumpz2(k), alphaz2(k), kappaz2(k), phizdum, ds, dt,
                           s_tzz0[TOFF(0,0)], tzz0_p1,
                           tzz0_m1, tzz0_p2 );

                           
                           if (distance_xmin == 0 || distance_ymax ==0 || distance_zmin == 0) { // boundary condition
                              d_vz0[offset] = 0.0f;
                           } else {
                              d_vz0[offset] += (dt/rhoxz)*( phixdum + phiydum + phizdum )
                              + staggardv4 (1.f/rhoxz, kappax2(i), kappay(j), kappaz2(k), dt, ds,
                              s_txz0[TOFF(0,0)], s_txz0[TOFF(1,0)],
                              s_txz0[TOFF(-1,0)], s_txz0[TOFF(2,0)],
                              s_tyz0[TOFF(0,-1)], s_tyz0[TOFF(0,0)],
                              s_tyz0[TOFF(0,-2)], s_tyz0[TOFF(0,1)],
                              s_tzz0[TOFF(0,0)], tzz0_p1,
                              tzz0_m1, tzz0_p2 );
                           }
                           
                           d_phitxzx[npml] = phixdum;
                           d_phityzy[npml] = phiydum;
                           d_phitzzz[npml] = phizdum;
                        } / end of if "free surface"
                     }///}}}
                  /// Normal mode }}}
                  } else {///{{{
                     rhoxy = 0.25f*(s_rho[RHOFF(0,0)][0] + s_rho[RHOFF(0,1)][0]
                             + s_rho[RHOFF(1,0)][0] + s_rho[RHOFF(1,1)][0]);
                     rhoxz = 0.25f*(s_rho[RHOFF(0,0)][0] + s_rho[RHOFF(0,0)][1]
                             + s_rho[RHOFF(1,0)][0] + s_rho[RHOFF(1,0)][1]);
                     if (distance_xmin == 0 || distance_xmax == 0 || distance_ymin == 0 || distance_ymax == 0 || distance_zmin == 0) {
                                                d_vx0[offset] = 0.f;
                                                d_vy0[offset] = 0.f;
                                                d_vz0[offset] = 0.f;
                     } else if ( distance_zmax == 0 ){ // free surface 
                        d_vx0[offset] += (1.f/s_rho[RHOFF(0,0)][0])*d_fx[offset_source]*dt/ds
                        + staggardv2 (1.f/s_rho[RHOFF(0,0)][0], 1.f, 1.f, 1.f, dt, ds,
                        s_txx0[TOFF(-1,0)], s_txx0[TOFF(0,0)],
                        s_txy0[TOFF(0,-1)], s_txy0[TOFF(0,0)],
                        txz0_m1, - txz0_m1 );

                        d_vy0[offset] += (1.f/rhoxy)*d_fy[offset_source]*dt/ds
                        + staggardv2 (1.f/rhoxy, 1.f, 1.f, 1.f, dt, ds,
                        s_txy0[TOFF(0,0)], s_txy0[TOFF(1,0)],
                        s_tyy0[TOFF(0,0)], s_tyy0[TOFF(0,1)],
                        tyz0_m1, - tyz0_m1 );

                        d_vz0[offset] += (1.f/rhoxz)*d_fz[offset_source]*dt/ds
                        + staggardv2 (1.f/rhoxz, 1.f, 1.f, 1.f, dt, ds,
                        - txz0_m1, - txz0_ip1_km1,
                        - tyz0_jm1_km1, - tyz0_m1,
                        s_tzz0[TOFF(0,0)], - tzz0_m1 );
                     } else if ( distance_zmax == 1 ){ // in the first cell, 2nd order finite-difference instead of 4th order finite-difference
                        d_vx0[offset] += (1.f/s_rho[RHOFF(0,0)][0])*d_fx[offset_source]*dt/ds
                        + staggardv2 (1.f/s_rho[RHOFF(0,0)][0], 1.f, 1.f, 1.f, dt, ds,
                        s_txx0[TOFF(-1,0)], s_txx0[TOFF(0,0)],
                        s_txy0[TOFF(0,-1)], s_txy0[TOFF(0,0)],
                        txz0_m1, s_txz0[TOFF(0,0)] );

                        d_vy0[offset] += (1.f/rhoxy)*d_fy[offset_source]*dt/ds
                        + staggardv2 (1.f/rhoxy, 1.f, 1.f, 1.f, dt, ds,
                        s_txy0[TOFF(0,0)], s_txy0[TOFF(1,0)],
                        s_tyy0[TOFF(0,0)], s_tyy0[TOFF(0,1)],
                        tyz0_m1, s_tyz0[TOFF(0,0)] );

                        d_vz0[offset] += (1.f/rhoxz)*d_fz[offset_source]*dt/ds
                        + staggardv2 (1.f/rhoxz, 1.f, 1.f, 1.f, dt, ds,
                        s_txz0[TOFF(0,0)], s_txz0[TOFF(1,0)],
                        s_tyz0[TOFF(0,-1)], s_tyz0[TOFF(0,0)],
                        s_tzz0[TOFF(0,0)], tzz0_p1 );
                     } else { // regular domain

                        d_vx0[offset] += (1.f/s_rho[RHOFF(0,0)][0])*d_fx[offset_source]*dt/ds
                        + staggardv4 (1.f/s_rho[RHOFF(0,0)][0], 1.f, 1.f, 1.f, dt, ds,
                        s_txx0[TOFF(-1,0)], s_txx0[TOFF(0,0)],
                        s_txx0[TOFF(-2,0)], s_txx0[TOFF(1,0)],
                        s_txy0[TOFF(0,-1)], s_txy0[TOFF(0,0)],
                        s_txy0[TOFF(0,-2)], s_txy0[TOFF(0,1)],
                        txz0_m1, s_txz0[TOFF(0,0)],
                        txz0_m2, txz0_p1 );

                        d_vy0[offset] += (1.f/rhoxy)*d_fy[offset_source]*dt/ds
                        + staggardv4 (1.f/rhoxy, 1.f, 1.f, 1.f, dt, ds,
                        s_txy0[TOFF(0,0)], s_txy0[TOFF(1,0)],
                        s_txy0[TOFF(-1,0)], s_txy0[TOFF(2,0)],
                        s_tyy0[TOFF(0,0)], s_tyy0[TOFF(0,1)],
                        s_tyy0[TOFF(0,-1)], s_tyy0[TOFF(0,2)],
                        tyz0_m1, s_tyz0[TOFF(0,0)],
                        tyz0_m2, tyz0_p1 );

                        d_vz0[offset] += (1.f/rhoxz)*d_fz[offset_source]*dt/ds
                        + staggardv4 (1.f/rhoxz, 1.f, 1.f, 1.f, dt, ds,
                        s_txz0[TOFF(0,0)], s_txz0[TOFF(1,0)],
                        s_txz0[TOFF(-1,0)], s_txz0[TOFF(2,0)],
                        s_tyz0[TOFF(0,-1)], s_tyz0[TOFF(0,0)],
                        s_tyz0[TOFF(0,-2)], s_tyz0[TOFF(0,1)],
                        s_tzz0[TOFF(0,0)], tzz0_p1,
                        tzz0_m1, tzz0_p2 );
                     } // end of if "free surface"
                  }///}}}  end of normal mode
               } // end of active///}}}
               __syncthreads();
            }///}}}
*/
/// }}}
         
	}
      // }}}
      // WRAPPER {{{
         void cuda_compute_veloc_host (   float* d_txx0, float* d_tyy0, float* d_tzz0, float* d_txy0, float* d_txz0, float* d_tyz0,
                  float* d_vx0, float* d_vy0, float* d_vz0,
                  float* d_fx, float* d_fy, float* d_fz, 
                  int* d_npml_tab, float* d_phitxxx, float* d_phitxyy, float* d_phitxzz, float *d_phitxyx, float *d_phityyy, float *d_phityzz, float *d_phitxzx, float *d_phityzy, float *d_phitzzz,
                  float* d_vp, float* d_rho,
                  int sizex, int sizey, int sizez,
                  int pitch_x, int pitch_y, int pitch_z, 
                  float ds, float dt, int delta, int position,
                  int grid_x, int grid_y, int grid_z, int block_x, int block_y, int block_z){
//		  FILE* f_vx) {
/*	float* h_t1, *h_t2, *h_t3;
//	float h_ds, h_dt;
        h_t1 = (float*)malloc(pitch_x*pitch_y*sizez*sizeof(float));
        h_t2 = (float*)malloc(pitch_x*pitch_y*sizez*sizeof(float));
        h_t3 = (float*)malloc(pitch_x*pitch_y*sizez*sizeof(float));
         for ( int k = 0; k < sizez; k++)
            for ( int j = 0; j < sizey; j++ )
                for ( int i = 0; i < sizex; i++ ){
                    int offset = OFFSET_PITCH(pitch_x, pitch_y, i, j, k);
                    h_t1[offset]=(float)rand()/(float)RAND_MAX*1000;
		    h_t2[offset]=(float)rand()/(float)RAND_MAX*1000;
                    h_t3[offset]=(float)rand()/(float)RAND_MAX*1000;
                }
      //  cudaMemcpy(d_fx, h_t1, pitch_x*pitch_y*sizez*sizeof(float),cudaMemcpyHostToDevice);
       // cudaMemcpy(d_fy, h_t2, pitch_x*pitch_y*sizez*sizeof(float), cudaMemcpyHostToDevice);
       // cudaMemcpy(d_fz, h_t3, pitch_x*pitch_y*sizez*sizeof(float), cudaMemcpyHostToDevice);*/
 
         #ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
            cudaThreadSynchronize();
            printCudaErr(cudaGetLastError(), "before computeVeloc kernel");
         #endif
            dim3 grid_dim(grid_x,grid_y,grid_z);
            dim3 block_dim(block_x, block_y, block_z);
            compute_veloc_3d <<< grid_dim, block_dim, 0,starpu_cuda_get_local_stream() >>> (  d_txx0, d_tyy0, d_tzz0, d_txy0, d_txz0, d_tyz0,
                              d_vx0, d_vy0, d_vz0,
                              d_fx, d_fy, d_fz, 
                              d_npml_tab, d_phitxxx, d_phitxyy, d_phitxzz, d_phitxyx, d_phityyy, d_phityzz, d_phitxzx, d_phityzy, d_phitzzz,
                              d_vp, d_rho,
                              sizex, sizey, sizez,
                              pitch_x, pitch_y, pitch_z, 
                              ds, dt, delta, position);

         #ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
            cudaThreadSynchronize();
            printCudaErr(cudaGetLastError(), "after computeVeloc kernel");
        #endif


    //   cudaMemcpy(h_t1, d_vx0, pitch_x*pitch_y*sizez*sizeof(float), cudaMemcpyDeviceToHost);
  //     cudaMemcpy(h_t2, d_vy0, pitch_x*pitch_y*sizez*sizeof(float), cudaMemcpyDeviceToHost);
//       cudaMemcpy(h_t3, d_vz0, pitch_x*pitch_y*sizez*sizeof(float), cudaMemcpyDeviceToHost);
 //       cudaMemcpy(h_dt, dt, sizeof(float), cudaMemcpyDeviceToHost);
   //     cudaMemcpy(h_ds, ds, sizeof(float), cudaMemcpyDeviceToHost);
/*        for ( int k = 0; k < sizez; k++)
            for ( int j = 0; j < sizey; j++ )
                for ( int i = 0; i < sizex; i++ ){
                    int offset = OFFSET_PITCH(pitch_x, pitch_y, i, j, k);
                    fprintf(stdout, "%d\t %d\t %d\t\t %10.12f\t %10.12f\t %10.12f\n", i, j, k, h_t1[offset], h_t2[offset], h_t3[offset]);
                }
        free(h_t1);
        free(h_t2);
        free(h_t3);*/
	/*	for ( int k = 0; k < sizez; k++)
			for ( int j = 0; j < sizey; j++ )
				for ( int i = 0; i < sizex; i++ ){
					int offset = OFFSET_PITCH(pitch_x, pitch_y, i, j, k);
					fprintf(f_vx, "%d\t %d\t %d\t\t %10.12f\t %10.12f\t %10.12f\n", i, j, k, d_vx0[offset], d_vy0[offset], d_vz0[offset]);
					fprintf(stdout, "%d\t %d\t %d\t\t %10.12f\t %10.12f\t\n", i, j, k, d_rho[offset], d_vp[offset]);
				}*/
         }
      // }}}
   // }}}

   // COPY BLOCK <-> BOUNDARY {{{
      // IMPLEMENTATION {{{
        __global__ void copy_block_boundary(float* block, float* boundary,
                          int size_x, int size_y, int size_z,
                          bool aligned, int padding, int direction, int sens) {

          bool active;
          int offset_block, offset_bound = 0;
          // coord in X or Y direction
          int xy_shift = blockIdx.x*blockDim.x+threadIdx.x;
          // coord in stencil (K) direction
          int K_shift = threadIdx.y;
          // coord in z direction
          int z_shift = blockIdx.y;

          switch(sens) {
            case FROM_BUF : switch(direction)
                      {
                        case XP : offset_block = aligned? size_x-padding-K:size_x-K;
                              offset_block += xy_shift*size_x + z_shift*size_x*size_y + K_shift;
                              offset_bound += xy_shift*K + z_shift*K*size_y + K_shift;
                              active = (xy_shift < size_y)?true:false;
                                  break;
                        case XM : offset_block = aligned?(ALIGN-K):0;
                              offset_block += xy_shift*size_x + z_shift*size_x*size_y + K_shift;
                              offset_bound += xy_shift*K + z_shift*K*size_y + K_shift;
                              active = (xy_shift < size_y)?true:false;
                                  break;
                        case YP : offset_block = size_x*size_y - K*size_x;
                              offset_block += xy_shift + z_shift*size_x*size_y + K_shift*size_x;
                              offset_bound += xy_shift + z_shift*size_x*K + K_shift*size_x;
                              active = (xy_shift < size_x)?true:false;
                                  break;
                        case YM : offset_block = 0;
                              offset_block += xy_shift + z_shift*size_x*size_y + K_shift*size_x;
                              offset_bound += xy_shift + z_shift*size_x*K + K_shift*size_x;
                              active = (xy_shift < size_x)?true:false;
                                  break;
                      }
                      break;
              case TO_BUF : switch(direction)
                      {
                        case XP : offset_block = aligned? size_x-padding-2*K:size_x-2*K;
                              offset_block += xy_shift*size_x + z_shift*size_x*size_y + K_shift;
                              offset_bound += xy_shift*K + z_shift*K*size_y + K_shift;
                              active = (xy_shift < size_y)?true:false;
                                  break;
                        case XM : offset_block = aligned?ALIGN:K;
                              offset_block += xy_shift*size_x + z_shift*size_x*size_y + K_shift;
                              offset_bound += xy_shift*K + z_shift*K*size_y + K_shift;
                              active = (xy_shift < size_y)?true:false;
                                  break;
                        case YP : offset_block = size_x*size_y - 2*K*size_x;
                              offset_block += xy_shift + z_shift*size_x*size_y + K_shift*size_x;
                              offset_bound += xy_shift + z_shift*size_x*K + K_shift*size_x;
                              active = (xy_shift < size_x)?true:false;
                                  break;
                        case YM : offset_block = K*size_x;
                              offset_block += xy_shift + z_shift*size_x*size_y + K_shift*size_x;
                              offset_bound += xy_shift + z_shift*size_x*K + K_shift*size_x;
                              active = (xy_shift < size_x)?true:false;
                                  break;
                      }
                      break;
          }
          if (active) block[offset_block] = boundary[offset_bound];
          return;
        }
      // }}}

      // WRAPPER {{{
        void copyBlockBoundary( float* block, float* boundary,
                    unsigned size_x, unsigned size_y, unsigned size_z,
                    bool aligned, unsigned padding, unsigned direction, unsigned sens){

        #ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
          cudaThreadSynchronize();
          printCudaErr(cudaGetLastError(), "before copy_block_boundary kernel");
        #endif
	    
					unsigned grid_x;
					switch(direction)
					{
						case XP :
						case XM : grid_x = int(ceilf((float)size_y/BLOCKSIZE));
										  break;
						case YP :
						case YM : grid_x = int(ceilf((float)size_x/BLOCKSIZE));
										  break;
					}

          dim3 grid_dim(grid_x,size_z);
          dim3 block_dim(BLOCKSIZE, K);
          copy_block_boundary <<< grid_dim, block_dim,0,starpu_cuda_get_local_stream() >>> ( block, boundary,
                                    size_x, size_y, size_z,
                                    aligned, padding, direction, sens);

        #ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
          cudaThreadSynchronize();
          printCudaErr(cudaGetLastError(), "after copy_block_boundary kernel");
        #endif
        }
      // }}}
   // }}}
// }}}
                                                                                           
// MISC {{{
   void cuda_device_info(char* decal)
   {
      int  ndev;
      cudaGetDeviceCount(&ndev);
      cudaThreadSynchronize();
      printf("%sThere are %d GPUs.\n",decal,ndev);
    
      for(int i=0;i<ndev;i++) {
        cudaDeviceProp pdev;
        cudaGetDeviceProperties(&pdev,i);
        cudaThreadSynchronize();
        printf("%sName         : %s\n",decal,pdev.name);
        printf("%sCapability   : %d %d\n",decal,pdev.major,pdev.minor);
        printf("%sMemory Global: %d Mb\n",decal,(pdev.totalGlobalMem+1024*1024)/1024/1024);
        printf("%sMemory Const : %d Kb\n",decal,pdev.totalConstMem/1024);
        printf("%sMemory Shared: %d Kb\n",decal,pdev.sharedMemPerBlock/1024);
        printf("%sClock        : %.3f GHz\n",decal,pdev.clockRate/1000000.f);
        printf("%sProcessors   : %d\n",decal,pdev.multiProcessorCount);
        printf("%sCores        : %d\n",decal,8*pdev.multiProcessorCount);
        printf("%sWarp         : %d\n",decal,pdev.warpSize);
        printf("%sMax Thr/Blk  : %d\n",decal,pdev.maxThreadsPerBlock);
        printf("%sMax Blk Size : %d %d %d\n",decal,pdev.maxThreadsDim[0],pdev.maxThreadsDim[1],pdev.maxThreadsDim[2]);
        printf("%sMax Grid Size: %d %d %d\n",decal,pdev.maxGridSize[0],pdev.maxGridSize[1],pdev.maxGridSize[2]);
      }
   }
// }}}

