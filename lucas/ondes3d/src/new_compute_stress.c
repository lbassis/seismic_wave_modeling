#include <starpu.h>

#include "../include/struct.h"
#include "../include/inlineFunctions.h"
#include "../include/new_nrutil.h"
#include "../include/new_compute_stress.h"

void compute_stress_task(void *buffers[], void *cl_arg) {

  //unpack structures
  double *t0_xx = (double *)STARPU_BLOCK_GET_PTR(buffers[0]);
  double *t0_yy = (double *)STARPU_BLOCK_GET_PTR(buffers[1]);
  double *t0_zz = (double *)STARPU_BLOCK_GET_PTR(buffers[2]);
  double *t0_xy = (double *)STARPU_BLOCK_GET_PTR(buffers[3]);
  double *t0_xz = (double *)STARPU_BLOCK_GET_PTR(buffers[4]);
  double *t0_yz = (double *)STARPU_BLOCK_GET_PTR(buffers[5]);

  int *k2ly0 = (int *)STARPU_VECTOR_GET_PTR(buffers[6]);
  int *k2ly2 = (int *)STARPU_VECTOR_GET_PTR(buffers[7]);
  double *mu0 = (double *)STARPU_VECTOR_GET_PTR(buffers[8]);
  double *mu2 = (double *)STARPU_VECTOR_GET_PTR(buffers[9]);
  double *kap0 = (double *)STARPU_VECTOR_GET_PTR(buffers[10]);

  double *kappax = (double *)STARPU_VECTOR_GET_PTR(buffers[11]);
  double *kappax2 = (double *)STARPU_VECTOR_GET_PTR(buffers[12]);
  double *kappay = (double *)STARPU_VECTOR_GET_PTR(buffers[13]);
  double *kappay2 = (double *)STARPU_VECTOR_GET_PTR(buffers[14]);
  double *kappaz = (double *)STARPU_VECTOR_GET_PTR(buffers[15]);
  double *kappaz2 = (double *)STARPU_VECTOR_GET_PTR(buffers[16]);

  double *phivxx = (double *)STARPU_VECTOR_GET_PTR(buffers[17]);
  double *phivyy = (double *)STARPU_VECTOR_GET_PTR(buffers[18]);
  double *phivzz = (double *)STARPU_VECTOR_GET_PTR(buffers[19]);
  double *phivyx = (double *)STARPU_VECTOR_GET_PTR(buffers[20]);
  double *phivxy = (double *)STARPU_VECTOR_GET_PTR(buffers[21]);
  double *phivzx = (double *)STARPU_VECTOR_GET_PTR(buffers[22]);
  double *phivxz = (double *)STARPU_VECTOR_GET_PTR(buffers[23]);
  double *phivzy = (double *)STARPU_VECTOR_GET_PTR(buffers[24]);
  double *phivyz = (double *)STARPU_VECTOR_GET_PTR(buffers[25]);

  int *ipml = (int *)STARPU_VECTOR_GET_PTR(buffers[26]);

  double *v0_x = (double *)STARPU_BLOCK_GET_PTR(buffers[27]);
  double *v0_y = (double *)STARPU_BLOCK_GET_PTR(buffers[28]);
  double *v0_z = (double *)STARPU_BLOCK_GET_PTR(buffers[29]);

  long int first_npml;
  int i_block, j_block, nb_blocks_dim;
  int i, j, k, imp, jmp;
  int inner_i, inner_j;
  double ds, dt;
  struct PARAMETERS prm;
  starpu_codelet_unpack_args(cl_arg, &i_block, &j_block, &nb_blocks_dim, &first_npml, &prm);

  int block_size = prm.block_size;
  //computestress
  /* approximations of a value in the corner of the cube */
  double kapxyz, kapxy, kapxz, kapx, kapy, kapz, muxy, muxz, mux, muy, muz, muxyz;
  double b1, b2;
  int ly0, ly2;		/* layer index */
  
  /*  */
  enum typePlace place;	/* What type of cell  */
  long int npml;

  ds = prm.ds;
  dt = prm.dt;

  /* loop */
  for (inner_i = 0; inner_i < prm.block_size; inner_i++) {
    for (inner_j = 0; inner_j < prm.block_size; inner_j++) {

      i = block_size*i_block+inner_i;
      j = block_size*j_block+inner_j;
      
      if (i == 0 || i == nb_blocks_dim-1 || j == 0 || j == nb_blocks_dim-1) {
	continue;
      }

      jmp = ivector_access(prm.jmp2j_array, -1, prm.mpmy + 2, j);
      imp = ivector_access(prm.imp2i_array, -1, prm.mpmx + 2, i);

      for (k = prm.zMin - prm.delta; k <= prm.zMax0; k++) {
	printf("eita\n");
	/* INITIALISATIONS */
	place = WhereAmI(imp, jmp, k, prm);

	/* jump "Not computed area" */
	if ((place == OUTSIDE) || (place == LIMIT)) {
	  continue;
	}
	/* find the right npml number */
	if ((place == ABSORBINGLAYER) || (place == FREEABS)) {
	  npml = ipml[k];
	}
	/* medium */

	/* Warning : k2ly0 & k2ly2
	   give 0 if k in FREEABS or if depth(k) > laydep[0] in general */
	ly0 = ivector_access(k2ly0, prm.zMin-prm.delta, prm.zMax0, k);
	ly2 = ivector_access(k2ly2, prm.zMin-prm.delta, prm.zMax0, k);;

	kapx = kap0[ly0];
	mux = mu0[ly0];


	muy = mu0[ly0];

	muz = mu2[ly2];
	muxyz = mu2[ly2];


	/*=====================================================*\
	  ELASTIC PART : 

	  structure :
	  -- common elastic initialisation
	  -- REGULAR & ABSORBING LAYER common part
	  -- ABSORBING LAYER special part
	  -- FREESURFACE & FREEABS common part
	  -- FREEABS special part

	  \*=====================================================*/
	if ((place == REGULAR) || (place == ABSORBINGLAYER)) {
	  /*********************************************/
	  /* REGULAR & ABSORBING LAYER initialisations */
	  /*********************************************/

	  /* NB : 
	   * We Modify the derivative in CPML using kappa and alpha.
	   * In regular domain or when PML is used, we do not modify derivatives, 
	   * that is to say :
	   * kapCPML = 1.
	   * aCPML = 0.,
	   * 
	   * So kapCPML and aCMPL are ingnored in Regular domain and PML formulation,
	   * there is no need to make a special case for them
	   */
	  /* Source : An improved PML for the wave equation
	     - Dimitri Komatitsch & Roland Martin, [Geophsysics 2007] */

	  /* Computation of Stress T  */
	  i3access(t0_xx, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) += staggards4(kapx, mux,
				       ivector_access(kappax2, 1, prm.mpmx, i),
				       ivector_access(kappay, 1, prm.mpmy, j),
				       ivector_access(kappaz, prm.zMin - prm.delta, prm.zMax0, k), dt,
				       ds, i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
				       i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 1, j, k),
				       i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i - 1, j, k),
				       i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 2, j, k),
				       i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 1, k),
				       i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
				       i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 2, k),
				       i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 1, k),
				       i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k - 1),
				       i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
				       i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k - 2),
				       i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k + 1),
				       place, ABCmethod);

	  i3access(t0_yy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) += staggards4(kapx, mux,
				       ivector_access(kappay, 1, prm.mpmy, j),
				       ivector_access(kappax2, 1, prm.mpmx, i),
				       ivector_access(kappaz, prm.zMin - prm.delta, prm.zMax0, k), dt,
				       ds,
				       i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 1, k),
				       i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
				       i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 2, k),
				       i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 1, k),
				       i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
				       i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 1, j, k),
				       i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i - 1, j, k),
				       i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 2, j, k),
				       i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k - 1),
				       i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
				       i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k - 2),
				       i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k + 1),
				       place, ABCmethod);

	  i3access(t0_zz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) += staggards4(kapx, mux,
				       ivector_access(kappaz, prm.zMin - prm.delta, prm.zMax0, k),
				       ivector_access(kappax2, 1, prm.mpmx, i),
				       ivector_access(kappay, 1, prm.mpmy, j), dt,
				       ds,
				       i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k - 1),
				       i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
				       i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k - 2),
				       i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k + 1),
				       i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
				       i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 1, j, k),
				       i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i - 1, j, k),
				       i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 2, j, k),
				       i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 1, k),
				       i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
				       i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 2, k),
				       i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 1, k),
				       place, ABCmethod);

	  i3access(t0_xy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) += staggardt4(muy,
				       ivector_access(kappax, 1, prm.mpmx, i),
				       ivector_access(kappay2, 1, prm.mpmy, j), dt,
				       ds,
				       i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i - 1, j, k),
				       i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
				       i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i - 2, j, k),
				       i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 1, j, k),
				       i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
				       i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 1, k),
				       i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 1, k),
				       i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 2, k),
				       place, ABCmethod);

	  i3access(t0_xz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) += staggardt4(muz,
				       ivector_access(kappax, 1, prm.mpmx, i),
				       ivector_access(kappaz2, prm.zMin - prm.delta, prm.zMax0, k), dt,
				       ds,
				       i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i - 1, j, k),
				       i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
				       i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i - 2, j, k),
				       i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 1, j, k),
				       i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
				       i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k + 1),
				       i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k - 1),
				       i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k + 2),
				       place, ABCmethod);

	  i3access(t0_yz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) += staggardt4(muxyz,
				       ivector_access(kappay2, 1, prm.mpmy, j),
				       ivector_access(kappaz2, prm.zMin - prm.delta, prm.zMax0, k), dt,
				       ds, i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
				       i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 1, k),
				       i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 1, k),
				       i3access(v0_z, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 2, k),
				       i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
				       i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k + 1),
				       i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k - 1),
				       i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k + 2),
				       place, ABCmethod);
	}

	/* ****************************** */
	/*  ABSORBING LAYER special part  */
	/* ****************************** */
	if (place == ABSORBINGLAYER) {
	  b1 = kapx + 4. * mux / 3.;
	  b2 = kapx - 2. * mux / 3.;

	  i3access(t0_xx, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) +=
	    dt * b2 * (phivyy[npml] +
		       phivzz[npml]) +
	    dt * b1 * phivxx[npml];
	  i3access(t0_yy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) +=
	    dt * b2 * (phivxx[npml] +
		       phivzz[npml]) +
	    dt * b1 * phivyy[npml];
	  i3access(t0_zz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) +=
	    dt * b2 * (phivxx[npml] +
		       phivyy[npml]) +
	    dt * b1 * phivzz[npml];

	  i3access(t0_xy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) +=
	    dt * muy * (phivyx[npml] +
			phivxy[npml]);
	  i3access(t0_xz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) +=
	    dt * muz * (phivzx[npml] +
			phivxz[npml]);
	  i3access(t0_yz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) +=
	    dt * muxyz * (phivzy[npml] +
			  phivyz[npml]);
	}
	/* ************************************** */
	/*  FREESURFACE and FREEABS common part   */
	/* ************************************** */
	/* # Description :
	 * -- k=0 
	 * we keep 4th order approximation of t0 for k=0 (just before the free surface),
	 * so we need values of the 2 above cells : k=1 (fully), k=2(partially).
	 * Partially, since only t0->xz(k=0) and t0->yz(k=0) depend on v0.x(k=2) and v0.y(k=2) 
	 * Details of those v0.x,v0.y can be found in the Function ComputeVelocity
	 * -- k=1 (freeSurface)
	 * 
	 * # From local index to z coordinates
	 * for k=1, t0->xx,t0->yy,t0->zz, t0->xy -> z=0; 
	 *          t0->xz,t0->yz            -> z=DS/2.
	 * for k=2, t0->xx,t0->yy,t0->zz, t0->xy -> z=DS 
	 *          t0->xz,t0->yz            -> z=3*DS/2. 
	 *          we only need to compute t0->zz, t0->xz and t0->yz 
	 *          (for v0.z(k=0), v0.x(k=1), v0.y(k=1) respectively)
	 * # Equations
	 * Cf."Simulating Seismic Wave propagation in 3D elastic Media Using staggered-Grid Finite Difference"
	 *  [ Robert W.Graves, p. 1099 ]
	 *   Bulletin of the Seismological Society of America Vol 4. August 1996
	 */
	if (place == FREESURFACE || place == FREEABS) {
	  if (k == 1) {
	    /* imposed values */
	    i3access(t0_xz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, 1) = -i3access(t0_xz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, 0);
	    i3access(t0_yz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, 1) = -i3access(t0_yz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, 0);
	    i3access(t0_zz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, 1) = 0.;

	    /* for t0->xx, t0->yy, we consider elastic formulation 
	     * giving dt(tzz)|z=0 = 0 => vz,t0->xx and t0->yy formulation
	     * Cf. eqn 98 to 100 p.13, Technical Note : Formulation of Finite Difference Method [ Hideo Aochi ]
	     */
	    b1 = 4. * mux * (kapx + mux / 3.) / (kapx +
						 4. / 3. *
						 mux);
	    b2 = 2. * mux * (kapx -
			     2. / 3. * mux) / (kapx +
					       4. / 3. *
					       mux);

	    i3access(t0_xx, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) +=
	      b1 * dt * ((9. / 8.) *
			 (i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i+1, j, k) -
			  i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k))
			 -
			 (1. / 24.) *
			 (i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i+2, j, k) -
			  i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i -
			       1, j, k))) / (ds *
					 
					     ivector_access(kappax2, 1, prm.mpmx, i))
	      +
	      b2 * dt * ((9. / 8.) *
			 (i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k) -
			  i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 1, k))
			 -
			 (1. / 24.) *
			 (i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j+1, k) -
			  i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j -
				  2, k))) / (ds *
					 
					     ivector_access(kappay, 1, prm.mpmy, j));

	    i3access(t0_yy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) +=
	      b1 * dt * ((9. / 8.) *
			 (i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k) -
			  i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 1, k))
			 -
			 (1. / 24.) *
			 (i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j+1, k) -
			  i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j -
				  2, k))) / (ds *
					 
					     ivector_access(kappay, 1, prm.mpmy, j))
	      +
	      b2 * dt * ((9. / 8.) *
			 (i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i+1, j, k) -
			  i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k))
			 -
			 (1. / 24.) *
			 (i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i+2, j, k) -
			  i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i -
			       1, j, k))) / (ds *
					 
					     ivector_access(kappax2, 1, prm.mpmx, i));

	    /* t0->xy computed like usual */
	    i3access(t0_xy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) +=
	      staggardt4(muy, un, un, dt, ds,
			 i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i - 1, j, k),
			 i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
			 i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i - 2, j, k),
			 i3access(v0_y, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i + 1, j, k),
			 i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j, k),
			 i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 1, k),
			 i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j - 1, k),
			 i3access(v0_x, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, i, j + 2, k), place,
			 ABCmethod);
	  }	/* end if k=1 */
	  if (k == 2) {
	    /* imposed values */
	    /* (other values are not needed) */
	    i3access(t0_xz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, 2) = -i3access(t0_xz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, -1);
	    i3access(t0_yz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, 2) = -i3access(t0_yz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, -1);
	    i3access(t0_zz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, 2) = -i3access(t0_zz, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, 0);
	  }
	}		/* end FreeSurface and FreeAbs common part */

	/*********************** */
	/* FREEABS special part  */
	/*********************** */
	if (place == FREEABS) {
	  /* Nothing to do for imposed values 
	   *  (since k=0 and k=-1  already contains PML/CPML) 
	   * what's left is : for k=1, t0->xx, t0->yy and t0->xy
	   */
	  if (k == 1) {
	    if (ABCmethod == CPML) {
	      //CTUL 07.12
	      //b1 = kapx + 4.*mux/3.;
	      //b2 = kapx - 2.*mux/3.;
	      b1 = 4. * mux * (kapx + mux / 3.) / (kapx +
						   4. /
						   3. *
						   mux);
	      b2 = 2. * mux * (kapx -
			       2. / 3. * mux) / (kapx +
						 4. /
						 3. *
						 mux);


	      i3access(t0_xx, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) +=
		dt * b2 * (phivyy[npml] +
			   phivzz[npml]) +
		dt * b1 * phivxx[npml];
	      i3access(t0_yy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) +=
		dt * b2 * (phivxx[npml] +
			   phivzz[npml]) +
		dt * b1 * phivyy[npml];
	      i3access(t0_xy, -1, prm.mpmx+2, -1, prm.mpmy+2, prm.zMin - prm.delta, prm.zMax0, inner_i, inner_j, k) +=
		dt * muy * (phivyx[npml] +
			    phivxy[npml]);
	    }

	  }	/* end if k == 1 */
	}		/* end FREEABS special part */
      }		/* end for k */
    }
  }
}
