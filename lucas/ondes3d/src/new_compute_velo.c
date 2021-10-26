#include <starpu.h>

#include "../include/struct.h"
#include "../include/inlineFunctions.h"
#include "../include/new_nrutil.h"
#include "../include/new_compute_velo.h"

void compute_velo_task(void *buffers[], void *cl_arg) {

  //unpack structures
  double *v0_x = (double *)STARPU_BLOCK_GET_PTR(buffers[0]);
  double *v0_y = (double *)STARPU_BLOCK_GET_PTR(buffers[1]);
  double *v0_z = (double *)STARPU_BLOCK_GET_PTR(buffers[2]);

  double *phit_base_ptr = (double *)STARPU_VECTOR_GET_PTR(buffers[3]);

  int *k2ly0 = (int *)STARPU_VECTOR_GET_PTR(buffers[4]);
  int *k2ly2 = (int *)STARPU_VECTOR_GET_PTR(buffers[5]);
  int *rho0 = (int *)STARPU_VECTOR_GET_PTR(buffers[6]);
  int *rho2 = (int *)STARPU_VECTOR_GET_PTR(buffers[7]);
  double *mu0 = (double *)STARPU_VECTOR_GET_PTR(buffers[8]);
  double *mu2 = (double *)STARPU_VECTOR_GET_PTR(buffers[9]);
  double *kap0 = (double *)STARPU_VECTOR_GET_PTR(buffers[10]);
  double *kap2 = (double *)STARPU_VECTOR_GET_PTR(buffers[11]);

  double *dumpx = (double *)STARPU_VECTOR_GET_PTR(buffers[12]);
  double *dumpx2 = (double *)STARPU_VECTOR_GET_PTR(buffers[13]);
  double *dumpy = (double *)STARPU_VECTOR_GET_PTR(buffers[14]);
  double *dumpy2 = (double *)STARPU_VECTOR_GET_PTR(buffers[15]);
  double *dumpz = (double *)STARPU_VECTOR_GET_PTR(buffers[16]);
  double *dumpz2 = (double *)STARPU_VECTOR_GET_PTR(buffers[17]);

  double *alphax = (double *)STARPU_VECTOR_GET_PTR(buffers[18]);
  double *alphax2 = (double *)STARPU_VECTOR_GET_PTR(buffers[19]);
  double *alphay = (double *)STARPU_VECTOR_GET_PTR(buffers[20]);
  double *alphay2 = (double *)STARPU_VECTOR_GET_PTR(buffers[21]);
  double *alphaz = (double *)STARPU_VECTOR_GET_PTR(buffers[22]);
  double *alphaz2 = (double *)STARPU_VECTOR_GET_PTR(buffers[23]);

  double *kappax = (double *)STARPU_VECTOR_GET_PTR(buffers[24]);
  double *kappax2 = (double *)STARPU_VECTOR_GET_PTR(buffers[25]);
  double *kappay = (double *)STARPU_VECTOR_GET_PTR(buffers[26]);
  double *kappay2 = (double *)STARPU_VECTOR_GET_PTR(buffers[27]);
  double *kappaz = (double *)STARPU_VECTOR_GET_PTR(buffers[28]);
  double *kappaz2 = (double *)STARPU_VECTOR_GET_PTR(buffers[29]);

  int *ipml = (int *)STARPU_VECTOR_GET_PTR(buffers[30]);

  double *t0_xx = (double *)STARPU_BLOCK_GET_PTR(buffers[31]);
  double *t0_yy = (double *)STARPU_BLOCK_GET_PTR(buffers[32]);
  double *t0_zz = (double *)STARPU_BLOCK_GET_PTR(buffers[33]);
  double *t0_xy = (double *)STARPU_BLOCK_GET_PTR(buffers[34]);
  double *t0_xz = (double *)STARPU_BLOCK_GET_PTR(buffers[35]);
  double *t0_yz = (double *)STARPU_BLOCK_GET_PTR(buffers[36]);

  double *fx = (double *)STARPU_BLOCK_GET_PTR(buffers[37]);
  double *fy = (double *)STARPU_BLOCK_GET_PTR(buffers[38]);
  double *fz = (double *)STARPU_BLOCK_GET_PTR(buffers[39]);

  long int first_npml;
  int i, j, k, imp, jmp;
  int inner_i, inner_j;
  double ds, dt;
  struct PARAMETERS prm;
  starpu_codelet_unpack_args(cl_arg, &i, &j, &first_npml, &prm);

  int block_size = prm.block_size;
  int i_block = i%prm.block_size;
  int j_block = j%prm.block_size;


  struct phit_s phit;
  phit.base_ptr = phit_base_ptr;

  phit.size = 9 * prm.block_size * prm.block_size * prm.depth;
  phit.offset = prm.block_size * prm.block_size * prm.depth;
  COMPUTE_ADDRESS_PHIT_S(phit);

  //compute velocity
  /* approximations of a value in the corner of the cube */
  int ly0, ly2;		/* layer xy (+0) or z (+ds/2) */
  double kapxy, kapxz, muxy, muxz,	/* rigidity and mu */
    kapx, kapy, mux, muy;
  double rhoxy, rhoxz;	/* density  */

  /*  */
  double bx, by, bz;		/* inverses of rho */
  double vp, vpxy, vpxz;	/* approximations of vp on the corner of the cube */

  /*  */
  enum typePlace place;	/* What type of cell  */
  long int npml;		/* index in Absorbing Layer */
  /* intermediates */
  double xdum, ydum, zdum;
  double phixdum, phiydum, phizdum;

  /* source == VELO */
  double rho;			/* density at [imp][jmp][k] */

  /* mapping */
  ds = prm.ds;
  dt = prm.dt;

  /* loop */
  for (inner_i = 0; inner_i < prm.block_size; inner_i++) {
    for (inner_j = 0; inner_j < prm.block_size; inner_j++) {

      i = block_size*i_block+inner_i;
      j = block_size*j_block+inner_j;

      if (i == 0 || i >= prm.mpmx || j == 0 || j >= prm.mpmx) {
	continue;
      }

      jmp = ivector_access(prm.jmp2j_array, -1, prm.mpmy + 2, j);
      imp = ivector_access(prm.imp2i_array, -1, prm.mpmx + 2, i);

      for (k = prm.zMin - prm.delta; k <= prm.zMax0; k++) {

	/* INITIALISATIONS */
	place = WhereAmI(imp, jmp, k, prm);

	if (place == OUTSIDE) {
	  continue;
	} else if (place == LIMIT) {
	  i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) = 0.;
	  i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) = 0.;
	  i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) = 0.;
	  continue;
	} else if (place == ABSORBINGLAYER || place == FREEABS) {
	  //npml = first_npml+k-(prm.zMin - prm.delta);
    npml = i3access(ipml, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin-prm.delta, prm.zMax0, i, j, k);
	}

	/*=====================================================*\
	  ELASTIC PART :

	  (nothing to do for regular domain, or only FreeSurface domain)
	  structure :
	  + PML/CPML
	  -- common initialisations
	  -- REGULAR & ABSORBING LAYER common part
	  -- FREESURFACE & FREEABS common part

	  -- ABSORBING LAYER & FREEABS special part
	  \*=====================================================*/
	/*******************************************/
	/* COMMON INITIALISATIONS                  */
	/*******************************************/
	ly0 = ivector_access(k2ly0, prm.zMin-prm.delta, prm.zMax0, k);
	ly2 = ivector_access(k2ly2, prm.zMin-prm.delta, prm.zMax0, k);;

	bx = 1.0 / rho0[ly0];
	by = 1.0 / rho0[ly0];

	rhoxz = rho2[ly2];
	bz = 1.0 / rhoxz;


	/*******************************************/
	/* REGULAR & ABSORBING LAYER               */
	/*******************************************/
	if (place == REGULAR || place == ABSORBINGLAYER) {
	  /* Computation of Vx,Vy and Vz */
	  i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) += staggardv4(bx,
															     ivector_access(kappax, 1, prm.mpmx, i),
															     ivector_access(kappay, 1, prm.mpmy, j),
															     ivector_access(kappaz, prm.zMin - prm.delta, prm.zMax0, k), dt,
															     ds,
															     i3access(t0_xx, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i - 1, j, k),
															     i3access(t0_xx, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
															     i3access(t0_xx, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i - 2, j, k),
															     i3access(t0_xx, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 1, j, k),
															     i3access(t0_xy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 1, k),
															     i3access(t0_xy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
															     i3access(t0_xy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 2, k),
															     i3access(t0_xy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 1, k),
															     i3access(t0_xz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k - 1),
															     i3access(t0_xz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
															     i3access(t0_xz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k - 2),
															     i3access(t0_xz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k + 1),
															     place, ABCmethod);

	  i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) += staggardv4(by,
															     ivector_access(kappax2, 1, prm.mpmx, i),
															     ivector_access(kappay2, 1, prm.mpmy, j),
															     ivector_access(kappaz, prm.zMin - prm.delta, prm.zMax0, k), dt,
															     ds, i3access(t0_xy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
															     i3access(t0_xy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 1, j, k),
															     i3access(t0_xy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i - 1, j, k),
															     i3access(t0_xy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 2, j, k),
															     i3access(t0_yy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
															     i3access(t0_yy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 1, k),
															     i3access(t0_yy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 1, k),
															     i3access(t0_yy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 2, k),
															     i3access(t0_yz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k - 1),
															     i3access(t0_yz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
															     i3access(t0_yz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k - 2),
															     i3access(t0_yz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k + 1),
															     place, ABCmethod);

	  i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) += staggardv4(bz,
															     ivector_access(kappax2, 1, prm.mpmx, i),
															     ivector_access(kappay, 1, prm.mpmy, j),
															     ivector_access(kappaz2, prm.zMin - prm.delta, prm.zMax0, k), dt,
															     ds, i3access(t0_xz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
															     i3access(t0_xz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 1, j, k),
															     i3access(t0_xz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i - 1, j, k),
															     i3access(t0_xz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 2, j, k),
															     i3access(t0_yz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 1, k),
															     i3access(t0_yz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
															     i3access(t0_yz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 2, k),
															     i3access(t0_yz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 1, k),
															     i3access(t0_zz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
															     i3access(t0_zz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k + 1),
															     i3access(t0_zz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k - 1),
															     i3access(t0_zz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k + 2),
															     place, ABCmethod);
	}		/* end REGULAR and ABSORBINGLAYER */

	/* ********************************** */
	/* FREE SURFACE & FREEABS COMMON PART */
	/* ********************************** */
	if (place == FREESURFACE || place == FREEABS) {

	  /* Description :
	   * # NB : no free surface for geological model so this part is only use with model == lAYER
	   * # What we need/compute :
	   * for k=1, full v0
	   * for k=2, only v0->x, v0->y (for t0(k=0) computation)
	   *
	   * # Details For k=1 :
	   *  For K=1
	   *  v0->x, v0->y are computed like usual.
	   *  v0->z need special treatment
	   */

	  /* k=1 */
	  /*-----*/
	  if (k == 1) {
	    	/* v0->x, v0->y (copied & pasted) */
	    i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) += staggardv4(bx,
		     ivector_access(kappax, 1, prm.mpmx, i),
		     ivector_access(kappay, 1, prm.mpmy, j),
		     ivector_access(kappaz, prm.zMin - prm.delta, prm.zMax0, k),
		     dt, ds,
		     i3access(t0_xx, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i - 1, j, k),
		     i3access(t0_xx, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
		     i3access(t0_xx, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i - 2, j, k),
		     i3access(t0_xx, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 1, j, k),
		     i3access(t0_xy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 1, k),
		     i3access(t0_xy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
		     i3access(t0_xy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 2, k),
		     i3access(t0_xy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 1, k),
		     i3access(t0_xz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k - 1),
		     i3access(t0_xz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
		     i3access(t0_xz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k - 2),
		     i3access(t0_xz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k + 1),
		     place, ABCmethod);

	    i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) += staggardv4(by,
		     ivector_access(kappax2, 1, prm.mpmx, i),
		     ivector_access(kappay2, 1, prm.mpmy, j),
		     ivector_access(kappaz, prm.zMin - prm.delta, prm.zMax0, k),
		     dt, ds,
		     i3access(t0_xy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
		     i3access(t0_xy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 1, j, k),
		     i3access(t0_xy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i - 1, j, k),
		     i3access(t0_xy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 2, j, k),
		     i3access(t0_yy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
		     i3access(t0_yy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 1, k),
		     i3access(t0_yy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 1, k),
		     i3access(t0_yy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 2, k),
		     i3access(t0_yz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k - 1),
		     i3access(t0_yz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
		     i3access(t0_yz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k - 2),
		     i3access(t0_yz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k + 1),
		     place, ABCmethod);
	  }	/* end k = 1 */

	  /* k=2 */
	  /*-----*/
	  if (k == 2) {
	  }
	}		/* end FREE SURFACE and FREEABS COMMON PART */


	/* ***************************** */
	/* ABSORBING LAYER & FREEABS part */
	/* ***************************** */
	/* NB : we also compute phit here
	 * For FREEABS,
	 * we only need to atenuate for k=1; the rest is already computed with symetries.
	 * what only differs is phitzzz = 0. since dz(tzz)==0 at Free Surface
	 */
	if ((place == ABSORBINGLAYER) ||
	    (place == FREEABS && k == prm.zMax0 - 1)) {
	  /* initialize */

	  vp = RhoMuKap2Vp(rho0[ly0], mu0[ly0], kap0[ly0]);	/* vx */

	  muxy = mu0[ly0];	/* vy */
	  kapxy = kap0[ly0];
	  rhoxy = rho0[ly0];

	  muxz = mu2[ly2];	/* vz */
	  kapxz = kap2[ly2];

	  vpxy = RhoMuKap2Vp(rhoxy, muxy, kapxy);	/* vy */
	  vpxz = RhoMuKap2Vp(rhoxz, muxz, kapxz);	/* vz */

	  /* Calculation of vx */
	  phixdum = ivector_access(phit.xxx, 1, 1000, npml);
	  phiydum = ivector_access(phit.xyy, 1, 1000, npml);
	  phizdum = ivector_access(phit.xzz, 1, 1000, npml);

	  ivector_access(phit.xxx, 1, 1000, npml) =
	    CPML4(vp, dumpx[i], alphax[i],
		  ivector_access(kappax, 1, prm.mpmx, i), phixdum, ds, dt,
		  i3access(t0_xx, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i - 1, j, k), i3access(t0_xx, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
		  i3access(t0_xx, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i - 2, j, k),
		  i3access(t0_xx, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 1, j, k));
	  ivector_access(phit.xyy, 1, 1000, npml) =
	    CPML4(vp, dumpy[j], alphay[j],
		  ivector_access(kappay, 1, prm.mpmy, j), phiydum, ds, dt,
		  i3access(t0_xy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 1, k), i3access(t0_xy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
		  i3access(t0_xy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 2, k),
		  i3access(t0_xy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 1, k));
	  ivector_access(phit.xzz, 1, 1000, npml) =
	    CPML4(vp, dumpz[k], alphaz[k],
		  ivector_access(kappaz, prm.zMin - prm.delta, prm.zMax0, k), phizdum, ds, dt,
		  i3access(t0_xz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k - 1), i3access(t0_xz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
		  i3access(t0_xz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k - 2),
		  i3access(t0_xz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k + 1));
	  i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) +=
	    bx * dt * (ivector_access(phit.xxx, 1, 1000, npml) +
		       ivector_access(phit.xyy, 1, 1000, npml) +
		       ivector_access(phit.xzz, 1, 1000, npml));

	  /* Calculation of vy */
	  phixdum = ivector_access(phit.xyx, 1, 1000, npml);
	  phiydum = ivector_access(phit.yyy, 1, 1000, npml);
	  phizdum = ivector_access(phit.yzz, 1, 1000, npml);

	  ivector_access(phit.xyx, 1, 1000, npml) =
	    CPML4(vpxy, dumpx2[i],
		  alphax2[i], ivector_access(kappax2, 1, prm.mpmx, i),
		  phixdum, ds, dt, i3access(t0_xy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
		  i3access(t0_xy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 1, j, k),
		  i3access(t0_xy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i - 1, j, k),
		  i3access(t0_xy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 2, j, k));
	  ivector_access(phit.yyy, 1, 1000, npml) =
	    CPML4(vpxy, dumpy2[j],
		  alphay2[j], ivector_access(kappay2, 1, prm.mpmy, j),
		  phiydum, ds, dt, i3access(t0_yy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
		  i3access(t0_yy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 1, k),
		  i3access(t0_yy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 1, k),
		  i3access(t0_yy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 2, k));
	  ivector_access(phit.yzz, 1, 1000, npml) =
	    CPML4(vpxy, dumpz[k], alphaz[k],
		  ivector_access(kappaz, prm.zMin - prm.delta, prm.zMax0, k), phizdum, ds, dt,
		  i3access(t0_yz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k - 1), i3access(t0_yz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
		  i3access(t0_yz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k - 2),
		  i3access(t0_yz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k + 1));

	  i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) +=
	    by * dt * (ivector_access(phit.xyx, 1, 1000, npml) +
		       ivector_access(phit.yyy, 1, 1000, npml) +
		       ivector_access(phit.yzz, 1, 1000, npml));

	  /* Calculation of vz */
	  phixdum = ivector_access(phit.xzx, 1, 1000, npml);
	  phiydum = ivector_access(phit.yzy, 1, 1000, npml);
	  phizdum = ivector_access(phit.zzz, 1, 1000, npml);

	  ivector_access(phit.xzx, 1, 1000, npml) =
	    CPML4(vpxz, dumpx2[i],
		  alphax2[i], ivector_access(kappax2, 1, prm.mpmx, i),
		  phixdum, ds, dt, i3access(t0_xz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
		  i3access(t0_xz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 1, j, k),
		  i3access(t0_xz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i - 1, j, k),
		  i3access(t0_xz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 2, j, k));
	  ivector_access(phit.yzy, 1, 1000, npml) =
	    CPML4(vpxz, dumpy[j], alphay[j],
		  ivector_access(kappay, 1, prm.mpmy, j), phiydum, ds, dt,
		  i3access(t0_yz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 1, k), i3access(t0_yz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
		  i3access(t0_yz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 2, k),
		  i3access(t0_yz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 1, k));
	  if (place == ABSORBINGLAYER) {
	    ivector_access(phit.zzz, 1, 1000, npml) =
	      CPML4(vpxz, dumpz2[k],
		    alphaz2[k], ivector_access(kappaz2, prm.zMin - prm.delta, prm.zMax0, k),
		    phizdum, ds, dt, i3access(t0_zz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
		    i3access(t0_zz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k + 1),
		    i3access(t0_zz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k - 1),
		    i3access(t0_zz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k + 2));
	  } else if (place == FREEABS && k == prm.zMax0 - 1) {	/* phitzzz = 0. since dz(tzz)==0 at Free Surface */
	    ivector_access(phit.zzz, 1, 1000, npml) = 0.;
	  }

	  i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) +=
	    bz * dt * (ivector_access(phit.xzx, 1, 1000, npml) +
		       ivector_access(phit.yzy, 1, 1000, npml) +
		       ivector_access(phit.zzz, 1, 1000, npml));
	  /* end of Calculation of Vz */
	}		/*         end of ( place == ABSORBINGLAYER  ) */

	/*=========================================*
	 * Add Source PART                histfile *
	 *=========================================*/
	/* Ajout des Sources.hist */
	if (source == HISTFILE && k != 1 && k != 2) {
	  i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) += bx * i3access(fx, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k) * dt / ds;
	  i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) += by * i3access(fy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k) * dt / ds;
	  i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) += bz * i3access(fz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k) * dt / ds;
	}		/* end of if source */
      }		/* end for k */
    }
  }
}

void compute_velo_k1(void *buffers[], void *cl_arg) {

  //unpack structures
  double *v0_x = (double *)STARPU_BLOCK_GET_PTR(buffers[0]);
  double *v0_y = (double *)STARPU_BLOCK_GET_PTR(buffers[1]);
  double *v0_z = (double *)STARPU_BLOCK_GET_PTR(buffers[2]);

  int *k2ly0 = (int *)STARPU_VECTOR_GET_PTR(buffers[3]);
  int *k2ly2 = (int *)STARPU_VECTOR_GET_PTR(buffers[4]);
  int *rho0 = (int *)STARPU_VECTOR_GET_PTR(buffers[5]);
  int *rho2 = (int *)STARPU_VECTOR_GET_PTR(buffers[6]);
  double *mu0 = (double *)STARPU_VECTOR_GET_PTR(buffers[7]);
  double *kap0 = (double *)STARPU_VECTOR_GET_PTR(buffers[8]);

  double *fx = (double *)STARPU_BLOCK_GET_PTR(buffers[9]);
  double *fy = (double *)STARPU_BLOCK_GET_PTR(buffers[10]);
  double *fz = (double *)STARPU_BLOCK_GET_PTR(buffers[11]);

  double *next_i = (double *)STARPU_BLOCK_GET_PTR(buffers[12]);
  double *prev_j = (double *)STARPU_BLOCK_GET_PTR(buffers[13]);

  long int first_npml;
  int i, j, k, imp, jmp;
  int inner_i, inner_j;
  double ds, dt;
  struct PARAMETERS prm;
  starpu_codelet_unpack_args(cl_arg, &i, &j, &first_npml, &prm);

  int block_size = prm.block_size;
  int i_block = i%prm.block_size;
  int j_block = j%prm.block_size;

  //compute velocity
  /* approximations of a value in the corner of the cube */
  int ly0, ly2;		/* layer xy (+0) or z (+ds/2) */
  double kapx, kapy, mux, muy;
  double rhoxy, rhoxz;	/* density  */

  /*  */
  double bx, by, bz;		/* inverses of rho */

  /*  */
  enum typePlace place;	/* What type of cell  */
  long int npml;		/* index in Absorbing Layer */
  /* intermediates */

  /* source == VELO */
  double rho;			/* density at [imp][jmp][k] */

  double i_plus_one, j_less_one;
  /* mapping */
  ds = prm.ds;
  dt = prm.dt;

  /* loop */
  for (inner_i = 0; inner_i < prm.block_size; inner_i++) {
    for (inner_j = 0; inner_j < prm.block_size; inner_j++) {

      i = block_size*i_block+inner_i;
      j = block_size*j_block+inner_j;

      if (i == 0 || i >= prm.mpmx || j == 0 || j >= prm.mpmx) {
	continue;
      }

      jmp = ivector_access(prm.jmp2j_array, -1, prm.mpmy + 2, j);
      imp = ivector_access(prm.imp2i_array, -1, prm.mpmx + 2, i);

      k = 1;
      /* INITIALISATIONS */
      place = WhereAmI(imp, jmp, k, prm);

      npml = first_npml+k-(prm.zMin - prm.delta);

      /*=====================================================*\
	ELASTIC PART :

	(nothing to do for regular domain, or only FreeSurface domain)
	structure :
	+ PML/CPML
	-- common initialisations
	-- REGULAR & ABSORBING LAYER common part
	-- FREESURFACE & FREEABS common part

	-- ABSORBING LAYER & FREEABS special part
	\*=====================================================*/
      /*******************************************/
      /* COMMON INITIALISATIONS                  */
      /*******************************************/
      ly0 = ivector_access(k2ly0, prm.zMin-prm.delta, prm.zMax0, k);
      ly2 = ivector_access(k2ly2, prm.zMin-prm.delta, prm.zMax0, k);;

      bx = 1.0 / rho0[ly0];
      by = 1.0 / rho0[ly0];

      rhoxz = rho2[ly2];
      bz = 1.0 / rhoxz;

      /* ********************************** */
      /* FREE SURFACE & FREEABS COMMON PART */
      /* ********************************** */
      if (place == FREESURFACE || place == FREEABS) {

	kapx = kap0[ly0];
	mux = mu0[ly0];

	kapy = kap0[ly0];
	muy = mu0[ly0];

	if (inner_i == block_size-1) {
	  i_plus_one = i3access(next_i, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, 0, inner_j, k);
	}
	else {
	  i_plus_one = i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i+1, inner_j, k);
	}

	if (inner_j == 0) {
	  j_less_one = i3access(prev_j, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, block_size-1, k);
	}
	else {
	  j_less_one = i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j-1, k);
	}

	i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) =
	  i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k-1)
	  - (kapx - 2. / 3. * mux) / (kapx + 4. / 3. * mux) *
	  (i_plus_one - i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k))
	  - (kapy - 2. / 3. * mux) / (kapy + 4. / 3. * muy) *
	  (i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) - j_less_one);

      	/* end k = 1 */
      }		/* end FREE SURFACE and FREEABS COMMON PART */

      /*=========================================*
       * Add Source PART                histfile *
       *=========================================*/
      /* Ajout des Sources.hist */
      if (source == HISTFILE) {
	i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) += bx * i3access(fx, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k) * dt / ds;
	i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) += by * i3access(fy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k) * dt / ds;
	i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) += bz * i3access(fz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k) * dt / ds;
      }		/* end of if source */
    }
  }
}


void compute_velo_k2(void *buffers[], void *cl_arg) {

  //unpack structures
  double *v0_x = (double *)STARPU_BLOCK_GET_PTR(buffers[0]);
  double *v0_y = (double *)STARPU_BLOCK_GET_PTR(buffers[1]);
  double *v0_z = (double *)STARPU_BLOCK_GET_PTR(buffers[2]);

  int *k2ly0 = (int *)STARPU_VECTOR_GET_PTR(buffers[3]);
  int *k2ly2 = (int *)STARPU_VECTOR_GET_PTR(buffers[4]);
  int *rho0 = (int *)STARPU_VECTOR_GET_PTR(buffers[5]);
  int *rho2 = (int *)STARPU_VECTOR_GET_PTR(buffers[6]);

  double *fx = (double *)STARPU_BLOCK_GET_PTR(buffers[7]);
  double *fy = (double *)STARPU_BLOCK_GET_PTR(buffers[8]);
  double *fz = (double *)STARPU_BLOCK_GET_PTR(buffers[9]);

  double *prev_i = (double *)STARPU_BLOCK_GET_PTR(buffers[10]);
  double *next_j = (double *)STARPU_BLOCK_GET_PTR(buffers[11]);

  long int first_npml;
  int i, j, k, imp, jmp;
  int inner_i, inner_j;
  double ds, dt;
  struct PARAMETERS prm;
  starpu_codelet_unpack_args(cl_arg, &i, &j, &first_npml, &prm);

  int block_size = prm.block_size;
  int i_block = i%prm.block_size;
  int j_block = j%prm.block_size;

  //compute velocity
  /*  */
  enum typePlace place;	/* What type of cell  */
  int ly0, ly2;		/* layer xy (+0) or z (+ds/2) */
  double bx, by, bz;		/* inverses of rho */
  double rhoxy, rhoxz;	/* density  */
  double i_less_one1, i_less_one2, j_plus_one1, j_plus_one2;
  /* mapping */
  ds = prm.ds;
  dt = prm.dt;

  /* loop */
  for (inner_i = 0; inner_i < prm.block_size; inner_i++) {
    for (inner_j = 0; inner_j < prm.block_size; inner_j++) {

      i = block_size*i_block+inner_i;
      j = block_size*j_block+inner_j;

      if (i == 0 || i >= prm.mpmx || j == 0 || j >= prm.mpmx) {
	continue;
      }

      jmp = ivector_access(prm.jmp2j_array, -1, prm.mpmy + 2, j);
      imp = ivector_access(prm.imp2i_array, -1, prm.mpmx + 2, i);

      k = 2;
      /* INITIALISATIONS */
      place = WhereAmI(imp, jmp, k, prm);

      ly0 = ivector_access(k2ly0, prm.zMin-prm.delta, prm.zMax0, k);
      ly2 = ivector_access(k2ly2, prm.zMin-prm.delta, prm.zMax0, k);;

      bx = 1.0 / rho0[ly0];
      by = 1.0 / rho0[ly0];

      rhoxz = rho2[ly2];
      bz = 1.0 / rhoxz;

      if (place == FREESURFACE || place == FREEABS) {

	// this is disgusting but i'm so tired

	if (inner_i == 0) { // both need to be taken from its neighboor
	  i_less_one1 = i3access(prev_i, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, block_size-1, inner_j, k-1);
	  i_less_one2 = i3access(prev_i, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, block_size-1, inner_j, k-2);
	}
	else { // at least i-1 is local
	  i_less_one1 = i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i-1, inner_j, k-1);
	  i_less_one2 = i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i-1, inner_j, k-2);
	}

	if (inner_j == block_size-1) { // both need to be taken from its neighboor
	  j_plus_one1 = i3access(next_j, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, 0, k-1);
	  j_plus_one2 = i3access(next_j, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, 0, k-2);
	}
	else { // at least j+1 is local
	  j_plus_one1 = i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j+1, k-1);
	  j_plus_one2 = i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j+1, k-2);
	}


	i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) = i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k - 1)
	  - (i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k - 1) -
	     i_less_one1)
	  - (i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k - 1) -
	     i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k - 2) + i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k - 2) -
	     i_less_one2);

	i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) = i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k - 1)
	  - (j_plus_one1 -
	     i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k - 1))
	  - (i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k - 1) -
	     i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k - 2) + j_plus_one2
	     - i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k - 2));

      }		/* end FREE SURFACE and FREEABS COMMON PART */

      /*=========================================*
       * Add Source PART                histfile *
       *=========================================*/
      /* Ajout des Sources.hist */
      if (source == HISTFILE) {
	i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) += bx * i3access(fx, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k) * dt / ds;
	i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) += by * i3access(fy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k) * dt / ds;
	i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) += bz * i3access(fz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k) * dt / ds;
      }		/* end of if source */
    }
  }
}
