#include <starpu.h>
#include "include/new_compute_seismoment.h"

struct starpu_codelet seis_moment_cl = {
					.cpu_funcs = {seis_moment_task},
					.nbuffers = 16,
					.modes = {STARPU_R, STARPU_R, STARPU_R, STARPU_R,
						  STARPU_R, STARPU_R, STARPU_R, STARPU_R,
						  STARPU_R, STARPU_R, STARPU_R, STARPU_R,
						  STARPU_R  STARPU_W, STARPU_W, STARPU_W},
};

void seis_moment_task(void *buffers[], void *cl_arg) {

  // unpack structures
  double *vel = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);
  double *strike = (float *)STARPU_VECTOR_GET_PTR(buffers[1]);
  double *dip = (float *)STARPU_VECTOR_GET_PTR(buffers[2]);
  double *rake = (float *)STARPU_VECTOR_GET_PTR(buffers[3]);
  double *xweight = (float *)STARPU_VECTOR_GET_PTR(buffers[4]);
  double *yweight = (float *)STARPU_VECTOR_GET_PTR(buffers[5]);
  double *zweight = (float *)STARPU_VECTOR_GET_PTR(buffers[6]);
  double *insrc = (float *)STARPU_VECTOR_GET_PTR(buffers[7]);
  double *ixhypo = (float *)STARPU_VECTOR_GET_PTR(buffers[8]);
  double *iyhypo = (float *)STARPU_VECTOR_GET_PTR(buffers[9]);
  double *izhypo = (float *)STARPU_VECTOR_GET_PTR(buffers[10]);
  double *i2imp_array = (float *)STARPU_VECTOR_GET_PTR(buffers[11]);
  double *j2jmp_array = (float *)STARPU_VECTOR_GET_PTR(buffers[12]);
  double *fx = (float *)STARPU_VECTOR_GET_PTR(buffers[13]);
  double *fy = (float *)STARPU_VECTOR_GET_PTR(buffers[14]);
  double *fz = (float *)STARPU_VECTOR_GET_PTR(buffers[15]);

  int iDur, iSrc, dtbiem;
  double ds, dt;
  starpu_codelet_unpack_args(cl_arg, &ds, &dt, &iDur, &iSrc);

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
	mo = vel[is][it] * dt;
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

	  imp = i2imp_array[i];
	  jmp1 = j2jmp_array[j - 1];
	  jmp2 = j2jmp_array[j];
	  jmp3 = j2jmp_array[j + 1];

	  fx[imp][jmp3][k] += 0.5 * mo * pxy * weight;
	  fx[imp][jmp1][k] -= 0.5 * mo * pxy * weight;
	  fx[imp][jmp2][k + 1] += 0.5 * mo * pxz * weight;
	  fx[imp][jmp2][k - 1] -= 0.5 * mo * pxz * weight;
	  fy[imp][jmp2][k] += 0.5 * mo * pxy * weight;
	  fy[imp][jmp1][k] += 0.5 * mo * pxy * weight;
	  fy[imp][jmp2][k] += 0.5 * mo * pyy * weight;
	  fy[imp][jmp1][k] -= 0.5 * mo * pyy * weight;
	  fy[imp][jmp2][k + 1] += 0.125 * mo * pyz * weight;
	  fy[imp][jmp1][k + 1] += 0.125 * mo * pyz * weight;
	  fy[imp][jmp2][k - 1] -= 0.125 * mo * pyz * weight;
	  fy[imp][jmp1][k - 1] -= 0.125 * mo * pyz * weight;
	  fz[imp][jmp2][k] += 0.5 * mo * pxz * weight;
	  fz[imp][jmp2][k - 1] += 0.5 * mo * pxz * weight;
	  fz[imp][jmp3][k] += 0.125 * mo * pyz * weight;
	  fz[imp][jmp3][k - 1] += 0.125 * mo * pyz * weight;
	  fz[imp][jmp1][k] -= 0.125 * mo * pyz * weight;
	  fz[imp][jmp1][k - 1] -= 0.125 * mo * pyz * weight;
	  fz[imp][jmp2][k] += 0.5 * mo * pzz * weight;
	  fz[imp][jmp2][k - 1] -= 0.5 * mo * pzz * weight;

	  imp = i2imp_array[i - 1];

	  fy[imp][jmp2][k] -= 0.5 * mo * pxy * weight;
	  fy[imp][jmp1][k] -= 0.5 * mo * pxy * weight;
	  fy[imp][jmp2][k] += 0.5 * mo * pyy * weight;
	  fy[imp][jmp1][k] -= 0.5 * mo * pyy * weight;
	  fy[imp][jmp2][k + 1] += 0.125 * mo * pyz * weight;
	  fy[imp][jmp1][k + 1] += 0.125 * mo * pyz * weight;
	  fy[imp][jmp2][k - 1] -= 0.125 * mo * pyz * weight;
	  fy[imp][jmp1][k - 1] -= 0.125 * mo * pyz * weight;
	  fx[imp][jmp2][k] -= 0.5 * mo * pxx * weight;
	  fz[imp][jmp3][k] += 0.125 * mo * pyz * weight;
	  fz[imp][jmp3][k - 1] += 0.125 * mo * pyz * weight;
	  fz[imp][jmp1][k] -= 0.125 * mo * pyz * weight;
	  fz[imp][jmp1][k - 1] -= 0.125 * mo * pyz * weight;
	  fz[imp][jmp2][k] += 0.5 * mo * pzz * weight;
	  fz[imp][jmp2][k - 1] -= 0.5 * mo * pzz * weight;
	  fz[imp][jmp2][k] -= 0.5 * mo * pxz * weight;
	  fz[imp][jmp2][k - 1] -= 0.5 * mo * pxz * weight;

	  imp = i2imp_array[i + 1];

	  fx[imp][jmp2][k] += 0.5 * mo * pxx * weight;	  
	}		/* end of iw (weighting) */
      }		        /* end of if SRC.insrc */
    }			/* end of is (each source) */
  }			/* end of if it */
}
