#include <starpu.h>

#include "../include/struct.h"
#include "../include/inlineFunctions.h"
#include "../include/new_nrutil.h"
#include "../include/new_compute_intermediates.h"

void compute_intermediates_task(void *buffers[], void *cl_arg) {

  //unpack structures

  int *k2ly0 = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);
  int *k2ly2 = (int *)STARPU_VECTOR_GET_PTR(buffers[2]);

  double *mu0 = (double *)STARPU_VECTOR_GET_PTR(buffers[3]);
  double *mu2 = (double *)STARPU_VECTOR_GET_PTR(buffers[4]);
  double *kap0 = (double *)STARPU_VECTOR_GET_PTR(buffers[5]);
  double *kap2 = (double *)STARPU_VECTOR_GET_PTR(buffers[6]);
  double *rho0 = (double *)STARPU_VECTOR_GET_PTR(buffers[7]);
  double *rho2 = (double *)STARPU_VECTOR_GET_PTR(buffers[8]);

  double *dumpx = (double *)STARPU_VECTOR_GET_PTR(buffers[9]);
  double *dumpx2 = (double *)STARPU_VECTOR_GET_PTR(buffers[10]);
  double *dumpy = (double *)STARPU_VECTOR_GET_PTR(buffers[11]);
  double *dumpy2 = (double *)STARPU_VECTOR_GET_PTR(buffers[12]);
  double *dumpz = (double *)STARPU_VECTOR_GET_PTR(buffers[13]);
  double *dumpz2 = (double *)STARPU_VECTOR_GET_PTR(buffers[14]);

  double *alphax = (double *)STARPU_VECTOR_GET_PTR(buffers[15]);
  double *alphax2 = (double *)STARPU_VECTOR_GET_PTR(buffers[16]);
  double *alphay = (double *)STARPU_VECTOR_GET_PTR(buffers[17]);
  double *alphay2 = (double *)STARPU_VECTOR_GET_PTR(buffers[18]);
  double *alphaz = (double *)STARPU_VECTOR_GET_PTR(buffers[19]);
  double *alphaz2 = (double *)STARPU_VECTOR_GET_PTR(buffers[20]);

  double *kappax = (double *)STARPU_VECTOR_GET_PTR(buffers[21]);
  double *kappax2 = (double *)STARPU_VECTOR_GET_PTR(buffers[22]);
  double *kappay = (double *)STARPU_VECTOR_GET_PTR(buffers[23]);
  double *kappay2 = (double *)STARPU_VECTOR_GET_PTR(buffers[24]);
  double *kappaz = (double *)STARPU_VECTOR_GET_PTR(buffers[25]);
  double *kappaz2 = (double *)STARPU_VECTOR_GET_PTR(buffers[26]);

  int *ipml = (int *)STARPU_VECTOR_GET_PTR(buffers[27]);

  double *v0_x = (double *)STARPU_BLOCK_GET_PTR(buffers[28]);
  double *v0_y = (double *)STARPU_BLOCK_GET_PTR(buffers[29]);
  double *v0_z = (double *)STARPU_BLOCK_GET_PTR(buffers[30]);

  long int first_npml;
  int i_block, j_block;
  int i, j, k, imp, jmp;
  int inner_i, inner_j;
  double ds, dt;
  struct PARAMETERS prm;

  //printf("deu %f\n", CPML4(6000.000000, 0.029142, 0.785398, 1.000000, 0.000000, 100.000000, 0.008000, 0.000000, 0.000000, 0.000000, 0.000000));
  
  int phiv_size, phiv_offset;
  starpu_codelet_unpack_args(cl_arg, &i_block, &j_block, &first_npml, &prm, &phiv_size, &phiv_offset);

  int block_size = prm.block_size;
  double *phiv_base_ptr = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
  struct phiv_s phiv;
  phiv.base_ptr = phiv_base_ptr;

  phiv.size = phiv_size;//9 * prm.block_size * prm.block_size * prm.depth;
  phiv.offset = phiv_offset;//prm.block_size * prm.block_size * prm.depth;
  COMPUTE_ADDRESS_PHIV_S(phiv);

  int n_blocks_x = ceil((prm.mpmx+4)/prm.block_size);
  int n_blocks_y = ceil((prm.mpmy+4)/prm.block_size);

  int nx = (j_block < n_blocks_x-1) ? prm.block_size : (prm.mpmx+4)%prm.block_size;
  int ny = (i_block < n_blocks_y-1) ? prm.block_size : (prm.mpmy+4)%prm.block_size; 

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

  if (i_block == 2 && j_block == 1) {
    printf("agora dentro da task\n");
    for (inner_i = 0; inner_i < phiv_size; inner_i++) {
      printf("phiv[%d] = %f\n", inner_i, phiv.base_ptr[inner_i]);
    }
    printf("o phiv[26] ta na posicao %p\n", &(phiv.base_ptr[26]));
  }

  for (inner_i = 0; inner_i < ny; inner_i++) {
    for (inner_j = 0; inner_j < nx; inner_j++) {

      i = block_size*i_block+inner_i;
      j = block_size*j_block+inner_j;

      if (i_block == 2 && j_block == 1) {
	printf("agora dentro da task\n");
	for (int _inner_i = 0; _inner_i < phiv_size; _inner_i++) {
	  printf("phiv[%d] = %f\n", _inner_i, phiv.base_ptr[_inner_i]);
	}
      }


      if (i == 0 || i > prm.mpmx || j == 0 || j > prm.mpmx) {
	continue;
      }

      jmp = ivector_access(prm.jmp2j_array, -1, prm.mpmy + 2, j); //??????
      imp = ivector_access(prm.imp2i_array, -1, prm.mpmx + 2, i);

      for (k = prm.zMin - prm.delta; k <= prm.zMax0; k++) {

	/* INITIALISATIONS */
	place = WhereAmI(imp, jmp, k, prm);

	//printf("i = %d, j = %d, k = %d, imp = %d, jmp = %d, place = %d\n", i, j, k, imp, jmp, place);
	/* jump "Not computed area" */
	if ((place == OUTSIDE) || (place == LIMIT)) {
	  continue;
	}
	/* find the right npml number */
	if ((place == ABSORBINGLAYER) || (place == FREEABS)) {
	  npml = i3access(ipml, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin-prm.delta, prm.zMax0, i, j, k);
	  printf("npml = -1!!!! i = %d, j = %d, k = %d imp = %d jmp = %d place = %d\n", i, j, k, imp, jmp, place);
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
	  phixdum = ivector_access(phiv.xx, 1, 1000, npml);
	  phiydum = ivector_access(phiv.yy, 1, 1000, npml);
	  phizdum = ivector_access(phiv.zz, 1, 1000, npml);

	  ivector_access(phiv.xx, 1, 1000, npml) =
	    CPML4(vpx, ivector_access(dumpx2, 1, prm.mpmx, i), ivector_access(alphax2, 1, prm.mpmx, i),
		  ivector_access(kappax2, 1, prm.mpmx, i), phixdum, ds, dt,
		  i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k), i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 1, j, k),
		  i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i - 1, j, k),
		  i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 2, j, k));
	  ivector_access(phiv.yy, 1, 1000, npml) =
	    CPML4(vpx, ivector_access(dumpy, 1, prm.mpmy, j), ivector_access(alphay, 1, prm.mpmy, j),
		  ivector_access(kappay, 1, prm.mpmy, j), phiydum, ds, dt,
		  i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 1, k), i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
		  i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 2, k),
		  i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 1, k));
	  ivector_access(phiv.zz, 1, 1000, npml) =
	    CPML4(vpx, ivector_access(dumpz, prm.zMin - prm.delta, 1000, k), ivector_access(alphaz, prm.zMin - prm.delta, 1000, k),
		  ivector_access(dumpz, prm.zMin - prm.delta, 1000, k), phizdum, ds, dt,
		  i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k - 1), i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
		  i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k - 2),
		  i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k + 1));
	  /* txy */
	  //phixdum = ivector_access(phiv.yx, 1, 1000, npml);
	  //phiydum = ivector_access(phiv.xy, 1, 1000, npml);
	  phixdum = phiv.yx[npml-1];
	  phiydum = phiv.xy[npml-1];
	  
	  if (phixdum != 0 || phiydum != 0) {
	    printf("i = %d, j = %d\n", i, j);
	    printf("leu %f\n", ivector_access(phiv.xy, 1, 1000, npml));
	  //  printf("phixydum = %f\nphiv.xy[%ld] = %f\nphiv.base_ptr[%ld] = %f\n", phiydum, npml, phiv.xy[npml-1], phiv.offset*3+npml-1, phiv.base_ptr[phiv.offset*3+npml-1]);
	  }	  
	  ivector_access(phiv.yx, 1, 1000, npml) =
	    CPML4(vpy, ivector_access(dumpx, 1, prm.mpmx, i), ivector_access(alphax, 1, prm.mpmx, i),
		  ivector_access(kappax, 1, prm.mpmx, i), phixdum, ds, dt,
		  i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i - 1, j, k), i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
		  i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i - 2, j, k),
		  i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 1, j, k));

	  ivector_access(phiv.xy, 1, 1000, npml) =
	    CPML4(vpy, ivector_access(dumpy2, 1, prm.mpmy, j), ivector_access(alphay2, 1, prm.mpmy, j),
		  ivector_access(kappay2, 1, prm.mpmy, j), phiydum, ds, dt,
		  i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k), i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 1, k),
		  i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 1, k),
		  i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 2, k));
	  if (ivector_access(phiv.xy, 1, 1000, npml) != 0.) {
	    printf("deu ruim aqui: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", vpy, ivector_access(dumpy2, 1, prm.mpmy, j), ivector_access(alphay2, 1, prm.mpmy, j),
		   ivector_access(kappay2, 1, prm.mpmy, j), phiydum, ds, dt,
		   i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k), i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 1, k),
		   i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 1, k),
		   i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 2, k));
	  }
	  
	  /* txz */
	  phixdum = ivector_access(phiv.zx, 1, 1000, npml);
	  phizdum = ivector_access(phiv.xz, 1, 1000, npml);

	  
	  
	  ivector_access(phiv.zx, 1, 1000, npml) =
	    CPML4(vpz, ivector_access(dumpx, 1, prm.mpmx, i), ivector_access(alphax, 1, prm.mpmx, i),
		  ivector_access(kappax, 1, prm.mpmx, i), phixdum, ds, dt,
		  i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i - 1, j, k), i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
		  i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i - 2, j, k),
		  i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 1, j, k));
	  ivector_access(phiv.xz, 1, 1000, npml) =
	    CPML4(vpz, ivector_access(dumpz2, prm.zMin - prm.delta, prm.zMax0, k), ivector_access(alphaz2, prm.zMin - prm.delta, prm.zMax0, k),
		  ivector_access(kappaz2, prm.zMin - prm.delta, prm.zMax0, k), phizdum, ds, dt,
		  i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k), i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k + 1),
		  i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k - 1),
		  i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k + 2));
	  /* tyz */
	  phiydum = ivector_access(phiv.zy, 1, 1000, npml);
	  phizdum = ivector_access(phiv.yz, 1, 1000, npml);

	  ivector_access(phiv.zy, 1, 1000, npml) =
	    CPML4(vpxyz, ivector_access(dumpy2, 1, prm.mpmy, j),
		  ivector_access(alphay2, 1, prm.mpmy, j), ivector_access(kappay2, 1, prm.mpmy, j),
		  phiydum, ds, dt, i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
		  i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 1, k), i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 1, k),
		  i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 2, k));
	  ivector_access(phiv.yz, 1, 1000, npml) =
	    CPML4(vpxyz, ivector_access(dumpz2, prm.zMin - prm.delta, prm.zMax0, k),
		  ivector_access(alphaz2, prm.zMin - prm.delta, prm.zMax0, k), ivector_access(kappaz2, prm.zMin - prm.delta, prm.zMax0, k),
		  phizdum, ds, dt, i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
		  i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k + 1), i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k - 1),
		  i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k + 2));
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
	     (ds, i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k), i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 1, j, k),
	      i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i - 1, j, k),
	      i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 2, j, k)) + Diff4(ds,
					 i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 1, k),
					 i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
					 i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 2, k),
					 i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 1, k))
	     );

	  /* txx, tyy */
	  phixdum = ivector_access(phiv.xx, 1, 1000, npml);
	  phiydum = ivector_access(phiv.yy, 1, 1000, npml);
	  phizdum = ivector_access(phiv.zz, 1, 1000, npml);
	  /* (copy&paste) */
	  ivector_access(phiv.xx, 1, 1000, npml) =
	    CPML4(vpx, ivector_access(dumpx2, 1, prm.mpmx, i), ivector_access(alphax2, 1, prm.mpmx, i),
		  ivector_access(kappax2, 1, prm.mpmx, i), phixdum, ds, dt,
		  i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k), i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 1, j, k),
		  i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i - 1, j, k),
		  i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 2, j, k));
	  ivector_access(phiv.yy, 1, 1000, npml) =
	    CPML4(vpx, ivector_access(dumpy, 1, prm.mpmy, j), ivector_access(alphay, 1, prm.mpmy, j),
		  ivector_access(kappay, 1, prm.mpmy, j), phiydum, ds, dt,
		  i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 1, k), i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
		  i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 2, k),
		  i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 1, k));
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
	  ivector_access(phiv.zz, 1, 1000, npml) = b * phizdum + a * (dzV0z);

	  /* txy ( copy&paste) */
	  phixdum = ivector_access(phiv.yx, 1, 1000, npml);
	  phiydum = ivector_access(phiv.xy, 1, 1000, npml);

	  ivector_access(phiv.yx, 1, 1000, npml) =
	    CPML4(vpy, ivector_access(dumpx, 1, prm.mpmx, i), ivector_access(alphax, 1, prm.mpmx, i),
		  ivector_access(kappax, 1, prm.mpmx, i), phixdum, ds, dt,
		  i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i - 1, j, k), i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
		  i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i - 2, j, k),
		  i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 1, j, k));
	  ivector_access(phiv.xy, 1, 1000, npml) =
	    CPML4(vpy, ivector_access(dumpy2, 1, prm.mpmy, j), ivector_access(alphay2, 1, prm.mpmy, j),
		  ivector_access(kappay2, 1, prm.mpmy, j), phiydum, ds, dt,
		  i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k), i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 1, k),
		  i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 1, k),
		  i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 2, k));
	}
      }
    }
  }
}
