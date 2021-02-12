#include <stdio.h>
#include "common.h"
#include "util.h"
#include "cycle.h"
#include "dim3d.h"
#define un 1.0


void StencilProbe(
	float* vx0,
        float* vy0,
        float* vz0,
        float* txx0,
        float* tyy0,
        float* tzz0,
        float* txy0,
        float* txz0,
        float* tyz0,
        float* fx,
        float* fy,
        float* fz,
        int nx, int ny, int nz,
        int tx, int ty, int tz, int timesteps)
{

  int i, j, k, t;
  float spt;
	int id;





     /* find conversion factor from ticks to seconds */

#pragma omp parallel  default(shared) private(t)
{
  for (t = 0; t < timesteps; t+=2) {


#pragma omp  for private(i,j,k) collapse(3)
	for (k = 2; k < nz - 2; k++) {
	for (j = 2; j < ny - 2; j++) {
	for (i = 2; i < nx - 2; i++) {

	 vx0[Index3D (nx, ny, i, j, k)] += (1./rho)*fx[Index3D (nx, ny, i, j, k)]*dt/ds +
	(9*(1./rho)*dt/8)*((txx0[Index3D (nx, ny, i, j, k)] - txx0[Index3D (nx, ny, i, j, k-1)])
	+ ( txy0[Index3D (nx, ny, i, j, k)] - txy0[Index3D (nx, ny, i, j-1, k)])
	+ ( txz0[Index3D (nx, ny, i, j, k)] -  txz0[Index3D (nx, ny, i-1, j, k)] ))
	 - ((1./rho)*dt/24)*( (txx0[Index3D (nx, ny, i, j, k+1)] - txx0[Index3D (nx, ny, i, j, k-2)])
	+ ( txy0[Index3D (nx, ny, i, j+1, k)] - txy0[Index3D (nx, ny, i, j-2, k)] )
	+ ( txz0[Index3D (nx, ny, i+1, j, k)] -  txz0[Index3D (nx, ny, i-2, j, k)] ));


         vy0[Index3D (nx, ny, i, j, k)] += (1./rho)*fy[Index3D (nx, ny, i, j, k)]*dt/ds +
        (9*(1./rho)*dt/8)*((txy0[Index3D (nx, ny, i, j, k+1)] - txy0[Index3D (nx, ny, i, j, k)] )
        + (tyy0[Index3D (nx, ny, i, j+1, k)] - tyy0[Index3D (nx, ny, i, j, k)] 	)
        + (tyz0[Index3D (nx, ny, i, j, k)] -  tyz0[Index3D (nx, ny, i-1, j, k)] ))
         - ((1./rho)*dt/24)*((txy0[Index3D (nx, ny, i, j, k+2)] - txy0[Index3D (nx, ny, i, j, k-1)])
        + (tyy0[Index3D (nx, ny, i, j+2, k)] - tyy0[Index3D (nx, ny, i, j-1, k)] 	 )
        + (tyz0[Index3D (nx, ny, i+1, j, k)] -  tyz0[Index3D (nx, ny, i-2, j, k)] ));



         vz0[Index3D (nx, ny, i, j, k)] += (1./rho)*fz[Index3D (nx, ny, i, j, k)]*dt/ds +
        (9*(1./rho)*dt/8)*((txz0[Index3D (nx, ny, i, j, k+1)] -  txz0[Index3D (nx, ny, i, j, k)]     )
        + (tyz0[Index3D (nx, ny, i, j, k)] - tyz0[Index3D (nx, ny, i, j-1, k)] )
        + ( tzz0[Index3D (nx, ny, i+1, j, k)] -  tzz0[Index3D (nx, ny, i, j, k)] ))
         - ((1./rho)*dt/24)*((txz0[Index3D (nx, ny, i, j, k+2)] -  txz0[Index3D (nx, ny, i, j, k-1)] )
        + ( tyz0[Index3D (nx, ny, i, j+1, k)]-tyz0[Index3D (nx, ny, i, j-2, k)]  )
        + (tzz0[Index3D (nx, ny, i+2, j, k)]  - tzz0[Index3D (nx, ny, i-1, j, k)] ));




	}
	}
	}



#pragma omp  for private(i,j,k) collapse(3)
        for (k = 2; k < nz - 2; k++) {
        for (j = 2; j < ny - 2; j++) {
        for (i = 2; i < nx - 2; i++) {


		txx0[Index3D (nx, ny, i, j, k)] +=
		(9*dt/8)*((lamx+2*mux)*(vx0[Index3D (nx, ny, i, j, k+1)] - vx0[Index3D (nx, ny, i, j, k)])
		+  lamx*(vy0[Index3D (nx, ny, i, j, k)] - vy0[Index3D (nx, ny, i, j-1, k)])
		+  lamx*(vz0[Index3D (nx, ny, i, j, k)] - vz0[Index3D (nx, ny, i-1, j, k)]))/ds 
		- (dt/24)*( (lamx+2*mux)*(vx0[Index3D (nx, ny, i, j, k+2)] - vx0[Index3D (nx, ny, i, j, k-1)])
		+  lamx*(vy0[Index3D (nx, ny, i, j+1, k)] - vy0[Index3D (nx, ny, i, j-2, k)]) 
		+  lamx*(vz0[Index3D (nx, ny, i+1, j, k)] - vz0[Index3D (nx, ny, i-2, j, k)]))/ds;

                tyy0[Index3D (nx, ny, i, j, k)] +=
                (9*dt/8)*( (lamx+2*mux)*(vy0[Index3D (nx, ny, i, j, k)] - vy0[Index3D (nx, ny, i, j-1, k)])
                +  lamx*(vx0[Index3D (nx, ny, i, j, k+1)] - vx0[Index3D (nx, ny, i, j, k)]         ) 
                +  lamx*(vz0[Index3D (nx, ny, i, j, k)] - vz0[Index3D (nx, ny, i-1, j, k)] ))/ds
                - (dt/24)*( (lamx+2*mux)*(vy0[Index3D (nx, ny, i, j+1, k)] - vy0[Index3D (nx, ny, i, j-2, k)])
                +  lamx*(vx0[Index3D (nx, ny, i, j, k+2)] - vx0[Index3D (nx, ny, i, j, k-1)])
                +  lamx*(vz0[Index3D (nx, ny, i+1, j, k)] -vz0[Index3D (nx, ny, i-2, j, k)] ))/ds;

		tzz0[Index3D (nx, ny, i, j, k)] +=
                (9*dt/8)*( (lamx+2*mux)*(vz0[Index3D (nx, ny, i, j, k)] - vz0[Index3D (nx, ny, i-1, j, k)]  )
                +  lamx*(vx0[Index3D (nx, ny, i, j, k+1)] -  vx0[Index3D (nx, ny, i, j, k)] )
                +  lamx*(vy0[Index3D (nx, ny, i, j, k)] - vy0[Index3D (nx, ny, i, j-1, k)] ))/ds
                - (dt/24)*( (lamx+2*mux)*(vz0[Index3D (nx, ny, i+1, j, k)] - vz0[Index3D (nx, ny, i-2, j, k)] )
                +  lamx*(vx0[Index3D (nx, ny, i, j, k+2)] - vx0[Index3D (nx, ny, i, j, k-1)] )
                +  lamx*(vy0[Index3D (nx, ny, i, j+1, k)]) - vy0[Index3D (nx, ny, i, j-2, k)] )/ds;

                txy0[Index3D (nx, ny, i, j, k)] += 
		   (9*dt*muy/8)*(( vy0[Index3D (nx, ny, i, j, k)] - vy0[Index3D (nx, ny, i, j, k-1)]  ) 
                   + (vx0[Index3D (nx, ny, i, j+1, k)] - vx0[Index3D (nx, ny, i, j, k)] ))/ds
      		   - (dt*muy/24)*((vy0[Index3D (nx, ny, i, j, k+1)] - vy0[Index3D (nx, ny, i, j, k-2)] ) 
                   +  (vx0[Index3D (nx, ny, i, j+2, k)] - vx0[Index3D (nx, ny, i, j-1, k)]))/ds;

                txz0[Index3D (nx, ny, i, j, k)] +=
                   (9*dt*muz/8)*((vz0[Index3D (nx, ny, i, j, k)] - vz0[Index3D (nx, ny, i, j, k-1)]   )
                   + (vx0[Index3D (nx, ny, i+1, j, k)] - vx0[Index3D (nx, ny, i, j, k)] ))/ds
                   - (dt*muz/24)*((vz0[Index3D (nx, ny, i, j, k)] - vz0[Index3D (nx, ny, i, j, k-2)]  )
                   +  (vx0[Index3D (nx, ny, i+2, j, k)] - vx0[Index3D (nx, ny, i-1, j, k)] ))/ds;

                tyz0[Index3D (nx, ny, i, j, k)] +=
                   (9*dt*muxyz/8)*((vz0[Index3D (nx, ny, i, j+1, k)] - vz0[Index3D (nx, ny, i, j, k)]   )
                   + (vy0[Index3D (nx, ny, i+1, j, k)] - vy0[Index3D (nx, ny, i, j, k)] ))/ds
                   - (dt*muxyz/24)*((vz0[Index3D (nx, ny, i, j+1, k)] - vz0[Index3D (nx, ny, i, j, k)]  )
                   +  (vy0[Index3D (nx, ny, i+2, j, k)] - vy0[Index3D (nx, ny, i-1, j, k)] ))/ds;


        }
	}
        }




}
}//dt
}//end
