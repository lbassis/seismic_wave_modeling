#include <starpu.h>

#include "../include/struct.h"
#include "../include/inlineFunctions.h"
#include "../include/new_nrutil.h"
#include "../include/new_compute_intermediates.h"

void compute_intermediates_task(void *buffers[], void *cl_arg) {

  //unpack structures
  double *phivxx = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
  double *phivyy = (double *)STARPU_VECTOR_GET_PTR(buffers[1]);
  double *phivzz = (double *)STARPU_VECTOR_GET_PTR(buffers[2]);
  double *phivyx = (double *)STARPU_VECTOR_GET_PTR(buffers[3]);
  double *phivxy = (double *)STARPU_VECTOR_GET_PTR(buffers[4]);
  double *phivzx = (double *)STARPU_VECTOR_GET_PTR(buffers[5]);
  double *phivxz = (double *)STARPU_VECTOR_GET_PTR(buffers[6]);
  double *phivzy = (double *)STARPU_VECTOR_GET_PTR(buffers[7]);
  double *phivyz = (double *)STARPU_VECTOR_GET_PTR(buffers[8]);

  int *k2ly0 = (int *)STARPU_VECTOR_GET_PTR(buffers[9]);
  int *k2ly2 = (int *)STARPU_VECTOR_GET_PTR(buffers[10]);

  double *mu0 = (double *)STARPU_VECTOR_GET_PTR(buffers[11]);
  double *mu2 = (double *)STARPU_VECTOR_GET_PTR(buffers[12]);
  double *kap0 = (double *)STARPU_VECTOR_GET_PTR(buffers[13]);
  double *kap2 = (double *)STARPU_VECTOR_GET_PTR(buffers[14]);
  double *rho0 = (double *)STARPU_VECTOR_GET_PTR(buffers[15]);
  double *rho2 = (double *)STARPU_VECTOR_GET_PTR(buffers[16]);

  double *dumpx = (double *)STARPU_VECTOR_GET_PTR(buffers[17]);
  double *dumpx2 = (double *)STARPU_VECTOR_GET_PTR(buffers[18]);
  double *dumpy = (double *)STARPU_VECTOR_GET_PTR(buffers[19]);
  double *dumpy2 = (double *)STARPU_VECTOR_GET_PTR(buffers[20]);
  double *dumpz = (double *)STARPU_VECTOR_GET_PTR(buffers[21]);
  double *dumpz2 = (double *)STARPU_VECTOR_GET_PTR(buffers[22]);

  double *alphax = (double *)STARPU_VECTOR_GET_PTR(buffers[23]);
  double *alphax2 = (double *)STARPU_VECTOR_GET_PTR(buffers[24]);
  double *alphay = (double *)STARPU_VECTOR_GET_PTR(buffers[25]);
  double *alphay2 = (double *)STARPU_VECTOR_GET_PTR(buffers[26]);
  double *alphaz = (double *)STARPU_VECTOR_GET_PTR(buffers[27]);
  double *alphaz2 = (double *)STARPU_VECTOR_GET_PTR(buffers[28]);

  double *kappax = (double *)STARPU_VECTOR_GET_PTR(buffers[29]);
  double *kappax2 = (double *)STARPU_VECTOR_GET_PTR(buffers[30]);
  double *kappay = (double *)STARPU_VECTOR_GET_PTR(buffers[31]);
  double *kappay2 = (double *)STARPU_VECTOR_GET_PTR(buffers[32]);
  double *kappaz = (double *)STARPU_VECTOR_GET_PTR(buffers[33]);
  double *kappaz2 = (double *)STARPU_VECTOR_GET_PTR(buffers[34]);

  int *ipml = (int *)STARPU_VECTOR_GET_PTR(buffers[35]);

  double *v0_x = (double *)STARPU_BLOCK_GET_PTR(buffers[36]);
  double *v0_y = (double *)STARPU_BLOCK_GET_PTR(buffers[37]);
  double *v0_z = (double *)STARPU_BLOCK_GET_PTR(buffers[38]);

  long int first_npml;
  int i_block, j_block;
  int i, j, k, imp, jmp;
  double ds, dt;
  struct PARAMETERS prm;
  starpu_codelet_unpack_args(cl_arg, &i_block, &j_block, &first_npml, &prm);
    
  //computeintermediates
  
  /* approximations of a value in the corner of the cube */
  double rhox, rhoy, rhoz, rhoxyz;	/* rho at +0 and +ds/2 */
  double kapxyz, kapxy, kapxz, kapx, kapy, kapz, muxy, muxz, mux, muy, muz, muxyz;
  double vpx, vpy, vpz, vpxyz;	/* approximations of vp on the corner of the cube */

  enum typePlace place;	/* What type of cell  */
  long int npml;		/* index in Absorbing Layer */

  double phixdum, phiydum, phizdum;	/* intermediates */

  double dzV0z;		/* derivative approximation of Vz in z direction  */
  double a, b;		/* intermediates for FREEABS */

  int med1, med2;

  int ly0, ly2;

  ds = prm.ds;
  dt = prm.dt;


  /* loop */
  for (i = 0; i < prm.block_size; i++) {
    for (j = 0; j < prm.block_size; j++) {

      // A FAZER: CONFERIR SE NAO EH UMA DAS 4 BORDAS GERAIS
      if (i == 0 || i == prm.block_size-1 || j == 0 || j == prm.block_size-1)
	continue;

      jmp = ivector_access(prm.jmp2j_array, -1, prm.mpmy + 2, j);
      imp = ivector_access(prm.imp2i_array, -1, prm.mpmx + 2, i);

      for (k = prm.zMin - prm.delta; k <= prm.zMax0; k++) {

	/* INITIALISATIONS */
	place = WhereAmI(imp, jmp, k, prm);

	/* jump "Not computed area" */
	if ((place == OUTSIDE) || (place == LIMIT)) {
	  continue;
	}
	/* find the right npml number */
	if ((place == ABSORBINGLAYER) || (place == FREEABS)) {
	  npml = first_npml+k-(prm.zMin - prm.delta);
	  printf("npml = %ld\n", npml);
	}
	/* medium */
	/* Warning : k2ly0 & k2ly2
	   give 0 if k in FREEABS or if depth(k) > laydep[0] in general */

	ly0 = ivector_access(k2ly0, prm.zMin-prm.delta, prm.zMax0, k);
	ly2 = ivector_access(k2ly2, prm.zMin-prm.delta, prm.zMax0, k);;

	mux = mu0[ly0];
	kapx = kap0[ly0];

	/*=====================================================*\
	  ELASTIC PART : PML and CPML

	  (nothing to do for regular domain, or only FreeSurface domain)
	  structure :
	  + PML/CPML
	  -- common initialisations
	  -- ABSORBING LAYER 
	  -- FREEABS

	  \*=====================================================*/

	/* ABSORBING Layer, FreeAbs common initialisations */
	/* ---------------------- */
	if (place == ABSORBINGLAYER || place == FREEABS) {
	  /* Initialize corner coefficients */
	  printf("absorbing layer ou freeabs\n");
	  muy = mu0[ly0];
	  kapy = kap0[ly0];
	  rhox = rho0[ly0];
	  rhoy = rho0[ly0];

	  rhoz = rho2[ly2];
	  rhoxyz = rho2[ly2];
	  muz = mu2[ly2];
	  muxyz = mu2[ly2];
	  kapz = kap2[ly2];
	  kapxyz = kap2[ly2];

	  vpx = RhoMuKap2Vp(rhox, mux, kapx);
	  vpy = vpx;

	  vpz = RhoMuKap2Vp(rhoz, muz, kapz);
	  vpxyz = vpz;
	}
	/* end initialize corners coeff in PML/CPML */

	/* ABSORBING LAYER      */
	/* -------------------- */
	if (place == ABSORBINGLAYER) {
	  /* Z Coefficients */

	  /* Compute  PHIV */
	  /* ------------- */
	  /* txx, tyy, tzz */

	  // nao sei o tamanho do vetor aqui, mas isso nao muda em nada entao pus 1000
	  phixdum = ivector_access(phivxx, 1, 1000, npml);
	  phiydum = ivector_access(phivyy, 1, 1000, npml);
	  phizdum = ivector_access(phivzz, 1, 1000, npml);


	  ivector_access(phivxx, 1, 1000, npml) =
	    CPML4(vpx, ivector_access(dumpx2, 1, prm.mpmx, i), ivector_access(alphax2, 1, prm.mpmx, i),
		  ivector_access(kappax2, 1, prm.mpmx, i), phixdum, ds, dt,
		  v0_x[i][j][k], v0_x[i + 1][j][k],
		  v0_x[i - 1][j][k],
		  v0_x[i + 2][j][k]);
	  ivector_access(phivyy, 1, 1000, npml) =
	    CPML4(vpx, ivector_access(dumpy, 1, prm.mpmy, j), ivector_access(alphay, 1, prm.mpmy, j),
		  ivector_access(kappay, 1, prm.mpmy, j), phiydum, ds, dt,
		  v0_y[i][j - 1][k], v0_y[i][j][k],
		  v0_y[i][j - 2][k],
		  v0_y[i][j + 1][k]);
	  ivector_access(phivzz, 1, 1000, npml) =
	    CPML4(vpx, ivector_access(dumpz, prm.zMin - prm.delta, 1000, k), ivector_access(alphaz, prm.zMin - prm.delta, 1000, k),
		  ivector_access(dumpz, prm.zMin - prm.delta, 1000, k), phizdum, ds, dt,
		  v0_z[i][j][k - 1], v0_z[i][j][k],
		  v0_z[i][j][k - 2],
		  v0_z[i][j][k + 1]);
	  /* txy */
	  phixdum = ivector_access(phivyx, 1, 1000, npml);
	  phiydum = ivector_access(phivxy, 1, 1000, npml);

	  ivector_access(phivyx, 1, 1000, npml) =
	    CPML4(vpy, ivector_access(dumpx, 1, prm.mpmx, i), ivector_access(alphax, 1, prm.mpmx, i),
		  ivector_access(kappax, 1, prm.mpmx, i), phixdum, ds, dt,
		  v0_y[i - 1][j][k], v0_y[i][j][k],
		  v0_y[i - 2][j][k],
		  v0_y[i + 1][j][k]);
	  ivector_access(phivxy, 1, 1000, npml) =
	    CPML4(vpy, ivector_access(dumpy2, 1, prm.mpmy, j), ivector_access(alphay2, 1, prm.mpmy, j),
		  ivector_access(kappay2, 1, prm.mpmy, j), phiydum, ds, dt,
		  v0_x[i][j][k], v0_x[i][j + 1][k],
		  v0_x[i][j - 1][k],
		  v0_x[i][j + 2][k]);
	  /* txz */
	  phixdum = ivector_access(phivzx, 1, 1000, npml);
	  phizdum = ivector_access(phivxz, 1, 1000, npml);

	  ivector_access(phivzx, 1, 1000, npml) =
	    CPML4(vpz, ivector_access(dumpx, 1, prm.mpmx, i), ivector_access(alphax, 1, prm.mpmx, i),
		  ivector_access(kappax, 1, prm.mpmx, i), phixdum, ds, dt,
		  v0_z[i - 1][j][k], v0_z[i][j][k],
		  v0_z[i - 2][j][k],
		  v0_z[i + 1][j][k]);
	  ivector_access(phivxz, 1, 1000, npml) =
	    CPML4(vpz, ivector_access(dumpz2, prm.zMin - prm.delta, prm.zMax0, k), ivector_access(alphaz2, prm.zMin - prm.delta, prm.zMax0, k),
		  ivector_access(kappaz2, prm.zMin - prm.delta, prm.zMax0, k), phizdum, ds, dt,
		  v0_x[i][j][k], v0_x[i][j][k + 1],
		  v0_x[i][j][k - 1],
		  v0_x[i][j][k + 2]);
	  /* tyz */
	  phiydum = ivector_access(phivzy, 1, 1000, npml);
	  phizdum = ivector_access(phivyz, 1, 1000, npml);

	  ivector_access(phivzy, 1, 1000, npml) =
	    CPML4(vpxyz, ivector_access(dumpy2, 1, prm.mpmy, j),
		  ivector_access(alphay2, 1, prm.mpmy, j), ivector_access(kappay2, 1, prm.mpmy, j),
		  phiydum, ds, dt, v0_z[i][j][k],
		  v0_z[i][j + 1][k], v0_z[i][j - 1][k],
		  v0_z[i][j + 2][k]);
	  ivector_access(phivyz, 1, 1000, npml) =
	    CPML4(vpxyz, ivector_access(dumpz2, prm.zMin - prm.delta, prm.zMax0, k),
		  ivector_access(alphaz2, prm.zMin - prm.delta, prm.zMax0, k), ivector_access(kappaz2, prm.zMin - prm.delta, prm.zMax0, k),
		  phizdum, ds, dt, v0_y[i][j][k],
		  v0_y[i][j][k + 1], v0_y[i][j][k - 1],
		  v0_y[i][j][k + 2]);
	}		/* End compute Absorbing Layers */

	/* FREEABS      */
	/* ------------ */
	/* We only need to compute t0xx(k=1),t0yy(k=1),t0xy(k=1) 
	 * So each coefficient will be computed only for those part 
	 *
	 * Expressions are simply obtain from approximation of dz(v0.z) in an elastic medium :
	 * dt(t0.zz)|z=0 = 0 = lam *(dx(v0.x) + dy(v0.y) + dz(v0.z)) + 2*mu * dz(v0.z);
	 *
	 */
	if (place == FREEABS) {
	  /* approximate dz(v0.z) (4th order) */
	  dzV0z =
	    (kapx - 2. / 3. * mux) / (kapx +
				      4. / 3. * mux) *
	    (Diff4
	     (ds, v0_x[i][j][k], v0_x[i + 1][j][k],
	      v0_x[i - 1][j][k],
	      v0_x[i + 2][j][k]) + Diff4(ds,
					 v0_y[i][j - 1][k],
					 v0_y[i][j][k],
					 v0_y[i][j - 2][k],
					 v0_y[i][j + 1][k])
	     );

	  /* txx, tyy */
	  phixdum = ivector_access(phivxx, 1, 1000, npml);
	  phiydum = ivector_access(phivyy, 1, 1000, npml);
	  phizdum = ivector_access(phivzz, 1, 1000, npml);
	  /* (copy&paste) */
	  ivector_access(phivxx, 1, 1000, npml) =
	    CPML4(vpx, ivector_access(dumpx2, 1, prm.mpmx, i), ivector_access(alphax2, 1, prm.mpmx, i),
		  ivector_access(kappax2, 1, prm.mpmx, i), phixdum, ds, dt,
		  v0_x[i][j][k], v0_x[i + 1][j][k],
		  v0_x[i - 1][j][k],
		  v0_x[i + 2][j][k]);
	  ivector_access(phivyy, 1, 1000, npml) =
	    CPML4(vpx, ivector_access(dumpy, 1, prm.mpmy, j), ivector_access(alphay, 1, prm.mpmy, j),
		  ivector_access(kappay, 1, prm.mpmy, j), phiydum, ds, dt,
		  v0_y[i][j - 1][k], v0_y[i][j][k],
		  v0_y[i][j - 2][k],
		  v0_y[i][j + 1][k]);
	  /* special */
	  b = exp(-
		  (vpx * ivector_access(dumpz, prm.zMin - prm.delta, 1000, k) / ivector_access(dumpz, prm.zMin - prm.delta, 1000, k) +
		   ivector_access(alphaz, prm.zMin - prm.delta, 1000, k)) * dt);
	  a = 0.0;
	  if (abs(vpx * ivector_access(dumpz, prm.zMin - prm.delta, 1000, k)) > 0.000001) {
	    a = vpx * ivector_access(dumpz, prm.zMin - prm.delta, 1000, k) * (b -
				  1.0) /
	      (ivector_access(dumpz, prm.zMin - prm.delta, 1000, k) *
	       (vpx * ivector_access(dumpz, prm.zMin - prm.delta, 1000, k) +
		ivector_access(dumpz, prm.zMin - prm.delta, 1000, k) * ivector_access(alphaz, prm.zMin - prm.delta, 1000, k)));
	  }
	  ivector_access(phivzz, 1, 1000, npml) = b * phizdum + a * (dzV0z);

	  /* txy ( copy&paste) */
	  phixdum = ivector_access(phivyx, 1, 1000, npml);
	  phiydum = ivector_access(phivxy, 1, 1000, npml);

	  ivector_access(phivyx, 1, 1000, npml) =
	    CPML4(vpy, ivector_access(dumpx, 1, prm.mpmx, i), ivector_access(alphax, 1, prm.mpmx, i),
		  ivector_access(kappax, 1, prm.mpmx, i), phixdum, ds, dt,
		  v0_y[i - 1][j][k], v0_y[i][j][k],
		  v0_y[i - 2][j][k],
		  v0_y[i + 1][j][k]);
	  ivector_access(phivxy, 1, 1000, npml) =
	    CPML4(vpy, ivector_access(dumpy2, 1, prm.mpmy, j), ivector_access(alphay2, 1, prm.mpmy, j),
		  ivector_access(kappay2, 1, prm.mpmy, j), phiydum, ds, dt,
		  v0_x[i][j][k], v0_x[i][j + 1][k],
		  v0_x[i][j - 1][k],
		  v0_x[i][j + 2][k]);
	}
      }
    }
  }
}
