#include <starpu.h>
#include <math.h>

#include "../include/inlineFunctions.h"
#include "../include/new_nrutil.h"
#include "../include/new_compute_seismoment.h"

void seis_moment_task(void *buffers[], void *cl_arg) {

  // unpack structures
  double *fx = (double *)STARPU_BLOCK_GET_PTR(buffers[0]);
  double *fy = (double *)STARPU_BLOCK_GET_PTR(buffers[1]);
  double *fz = (double *)STARPU_BLOCK_GET_PTR(buffers[2]);
  double *vel = (double *)STARPU_MATRIX_GET_PTR(buffers[3]);
  double *strike = (double *)STARPU_VECTOR_GET_PTR(buffers[4]);
  double *dip = (double *)STARPU_VECTOR_GET_PTR(buffers[5]);
  double *rake = (double *)STARPU_VECTOR_GET_PTR(buffers[6]);
  double *xweight = (double *)STARPU_VECTOR_GET_PTR(buffers[7]);
  double *yweight = (double *)STARPU_VECTOR_GET_PTR(buffers[8]);
  double *zweight = (double *)STARPU_VECTOR_GET_PTR(buffers[9]);
  int *insrc = (int *)STARPU_VECTOR_GET_PTR(buffers[10]);
  int *ixhypo = (int *)STARPU_VECTOR_GET_PTR(buffers[11]);
  int *iyhypo = (int *)STARPU_VECTOR_GET_PTR(buffers[12]);
  int *izhypo = (int *)STARPU_VECTOR_GET_PTR(buffers[13]);
  int *i2imp_array = (int *)STARPU_VECTOR_GET_PTR(buffers[14]);
  int *j2jmp_array = (int *)STARPU_VECTOR_GET_PTR(buffers[15]);

  int iDur, iSrc, dtbiem;
  double time, ds, dt;
  struct PARAMETERS prm;
  starpu_codelet_unpack_args(cl_arg, &time, &prm, &dtbiem, &iDur, &iSrc);

  const int XMIN = prm.xMin;
  const int XMAX = prm.xMax;
  const int YMIN = prm.yMin;
  const int YMAX = prm.yMax;
  const int ZMIN = prm.zMin;
  const int ZMAX = prm.zMax;
  const int ZMAX0 = prm.zMax0;
  const int DELTA = prm.delta;

  const int MPMX = prm.mpmx;
  const int MPMY = prm.mpmy;

  ds = prm.ds;
  dt = prm.dt;

  // computeseismoment
  int it, is, iw;
  int i, j, k;
  double pxx, pyy, pzz, pxy, pyz, pxz;
  double mo, weight;
  int imp, jmp;
  int jcpu1, jcpu2, jcpu3;
  int jmp1, jmp2, jmp3;

  it = ceil(time / dtbiem);
  if (it < iDur) {
    for (is = 0; is < iSrc; is++) {
      if (insrc[is] == 1) {
	mo = vel[is*iSrc+it] * dt;
	pxx = radxx(strike[is], dip[is], rake[is]);
	pyy = radyy(strike[is], dip[is], rake[is]);
	pzz = radzz(strike[is], dip[is], rake[is]);
	pxy = radxy(strike[is], dip[is], rake[is]);
	pyz = radyz(strike[is], dip[is], rake[is]);
	pxz = radxz(strike[is], dip[is], rake[is]);

	for (iw = 0; iw < 8; iw++) {
	  weight = 1.0;
	  if ((iw % 2) == 0) {
	    i = ixhypo[is];
	    weight = (1.0 - xweight[is]);
	  } else {
	    i = ixhypo[is] + 1;
	    weight = xweight[is];
	  }
	  if ((iw % 4) <= 1) {
	    j = iyhypo[is];
	    weight = weight * (1.0 - yweight[is]);
	  } else {
	    j = iyhypo[is] + 1;
	    weight = weight * yweight[is];
	  }
	  if (iw < 4) {
	    k = izhypo[is];
	    weight = weight * (1.0 - zweight[is]);
	  } else {
	    k = izhypo[is] + 1;
	    weight = weight * zweight[is];
	  }

	  
	  imp = ivector_access(prm.i2imp_array, XMIN - DELTA, XMAX + 2 * DELTA + 2, i);
	  jmp1 = ivector_access(prm.j2jmp_array, YMIN - DELTA, YMAX + 2 * DELTA + 2, j-1);
	  jmp2 = ivector_access(prm.j2jmp_array, YMIN - DELTA, YMAX + 2 * DELTA + 2, j);
	  jmp3 = ivector_access(prm.j2jmp_array, YMIN - DELTA, YMAX + 2 * DELTA + 2, j+1);

	  i3access(fx, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp3, k) += 0.5 * mo * pxy * weight;
	  i3access(fx, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp1, k) -= 0.5 * mo * pxy * weight;
	  i3access(fx, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp2, k + 1) += 0.5 * mo * pxz * weight;
	  i3access(fx, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp2, k - 1) -= 0.5 * mo * pxz * weight;
	  i3access(fy, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp2, k) += 0.5 * mo * pxy * weight;
	  i3access(fy, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp1, k) += 0.5 * mo * pxy * weight;
	  i3access(fy, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp2, k) += 0.5 * mo * pyy * weight;
	  i3access(fy, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp1, k) -= 0.5 * mo * pyy * weight;
	  i3access(fy, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp2, k + 1) += 0.125 * mo * pyz * weight;
	  i3access(fy, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp1, k + 1) += 0.125 * mo * pyz * weight;
	  i3access(fy, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp2, k - 1) -= 0.125 * mo * pyz * weight;
	  i3access(fy, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp1, k - 1) -= 0.125 * mo * pyz * weight;
	  i3access(fz, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp2, k) += 0.5 * mo * pxz * weight;
	  i3access(fz, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp2, k - 1) += 0.5 * mo * pxz * weight;
	  i3access(fz, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp3, k) += 0.125 * mo * pyz * weight;
	  i3access(fz, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp3, k - 1) += 0.125 * mo * pyz * weight;
	  i3access(fz, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp1, k) -= 0.125 * mo * pyz * weight;
	  i3access(fz, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp1, k - 1) -= 0.125 * mo * pyz * weight;
	  i3access(fz, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp2, k) += 0.5 * mo * pzz * weight;
	  i3access(fz, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp2, k - 1) -= 0.5 * mo * pzz * weight;

	  //imp = i2imp_array[i - 1];
	  imp = ivector_access(prm.i2imp_array, XMIN - DELTA, XMAX + 2 * DELTA + 2, i-1);

	  i3access(fy, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp2, k) -= 0.5 * mo * pxy * weight;
	  i3access(fy, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp1, k) -= 0.5 * mo * pxy * weight;
	  i3access(fy, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp2, k) += 0.5 * mo * pyy * weight;
	  i3access(fy, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp1, k) -= 0.5 * mo * pyy * weight;
	  i3access(fy, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp2, k + 1) += 0.125 * mo * pyz * weight;
	  i3access(fy, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp1, k + 1) += 0.125 * mo * pyz * weight;
	  i3access(fy, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp2, k - 1) -= 0.125 * mo * pyz * weight;
	  i3access(fy, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp1, k - 1) -= 0.125 * mo * pyz * weight;
	  i3access(fx, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp2, k) -= 0.5 * mo * pxx * weight;
	  i3access(fz, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp3, k) += 0.125 * mo * pyz * weight;
	  i3access(fz, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp3, k - 1) += 0.125 * mo * pyz * weight;
	  i3access(fz, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp1, k) -= 0.125 * mo * pyz * weight;
	  i3access(fz, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp1, k - 1) -= 0.125 * mo * pyz * weight;
	  i3access(fz, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp2, k) += 0.5 * mo * pzz * weight;
	  i3access(fz, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp2, k - 1) -= 0.5 * mo * pzz * weight;
	  i3access(fz, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp2, k) -= 0.5 * mo * pxz * weight;
	  i3access(fz, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp2, k - 1) -= 0.5 * mo * pxz * weight;

	  //imp = i2imp_array[i + 1];
	  imp = ivector_access(prm.i2imp_array, XMIN - DELTA, XMAX + 2 * DELTA + 2, i+1);

	  i3access(fx, 1, MPMX, 1, MPMY, ZMIN - DELTA, ZMAX0, imp, jmp2, k) += 0.5 * mo * pxx * weight;	  
	}		/* end of iw (weighting) */
      }		        /* end of if SRC.insrc */
    }			/* end of is (each source) */
  }			/* end of if it */
}
