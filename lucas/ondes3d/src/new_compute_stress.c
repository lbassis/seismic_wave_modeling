#include <starpu.h>
#include "include/new_compute_stress.h"

struct starpu_codelet stress_cl = {
				   .cpu_funcs = {compute_stress_task},
				   .nbuffers = 30,
				   .modes = {STARPU_W, STARPU_W, STARPU_RW, STARPU_W, STARPU_RW, STARPU_RW,
					     STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R,
					     STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R,
					     STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R,
					     STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R}
};


void compute_stress_task(void *buffers[], void *cl_arg) {

  //unpack structures
  double ***t0_xx = (double ***)STARPU_BLOCK_GET_PTR(buffers[0]);
  double ***t0_yy = (double ***)STARPU_BLOCK_GET_PTR(buffers[1]);
  double ***t0_zz = (double ***)STARPU_BLOCK_GET_PTR(buffers[2]);
  double ***t0_xy = (double ***)STARPU_BLOCK_GET_PTR(buffers[3]);
  double ***t0_xz = (double ***)STARPU_BLOCK_GET_PTR(buffers[4]);
  double ***t0_yz = (double ***)STARPU_BLOCK_GET_PTR(buffers[5]);

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

  double *phivxx = (float *)STARPU_VECTOR_GET_PTR(buffers[17]);
  double *phivyy = (float *)STARPU_VECTOR_GET_PTR(buffers[18]);
  double *phivzz = (float *)STARPU_VECTOR_GET_PTR(buffers[19]);
  double *phivyx = (float *)STARPU_VECTOR_GET_PTR(buffers[20]);
  double *phivxy = (float *)STARPU_VECTOR_GET_PTR(buffers[21]);
  double *phivzx = (float *)STARPU_VECTOR_GET_PTR(buffers[22]);
  double *phivxz = (float *)STARPU_VECTOR_GET_PTR(buffers[23]);
  double *phivzy = (float *)STARPU_VECTOR_GET_PTR(buffers[24]);
  double *phivyz = (float *)STARPU_VECTOR_GET_PTR(buffers[25]);

  int *ipml = (int *)STARPU_VECTOR_GET_PTR(buffers[26]);

  double ***v0_x = (double ***)STARPU_BLOCK_GET_PTR(buffers[27]);
  double ***v0_y = (double ***)STARPU_BLOCK_GET_PTR(buffers[28]);
  double ***v0_z = (double ***)STARPU_BLOCK_GET_PTR(buffers[29]);

  long int first_npml;
  int i, j, k, imp, jmp;
  double ds, dt;
  struct PARAMETERS prm;
  starpu_codelet_unpack_args(cl_arg, &i, &j, &first_npml, &prm);
  
  int i_block = i%prm.block_size;
  int j_block = j%prm.block_size;
  
  //computestress
  /* approximations of a value in the corner of the cube */
  double kapxyz,		/* rigidity and mu */
    kapxy, kapxz, kapx, kapy, kapz, muxy, muxz, mux, muy, muz, muxyz;

  int ly0, ly2;		/* layer index */

  /*  */
  enum typePlace place;	/* What type of cell  */
  long int npml;

  jmp = prm.jmp2j_array[j];
  imp = prm.imp2i_array[i];

  ds = prm.ds;
  dt = prm.dt;

  jmp = prm.jmp2j_array[j];
  imp = prm.imp2i_array[i];
  for (k = prm.zMin - prm.delta; k <= prm.zMax0; k++) {

    /* INITIALISATIONS */
    place = WhereAmI(imp, jmp, k, PRM);

    /* jump "Not computed area" */
    if ((place == OUTSIDE) || (place == LIMIT)) {
      continue;
    }
    /* find the right npml number */
    if ((place == ABSORBINGLAYER) || (place == FREEABS)) {
      npml = ipml[i][j][k];
    }
    /* medium */

    /* Warning : k2ly0 & k2ly2
       give 0 if k in FREEABS or if depth(k) > laydep[0] in general */
    ly0 = k2ly0[k];
    ly2 = k2ly2[k];

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
      t0_xx[i][j][k] += staggards4(kapx, mux,
				    kappax2[i],
				    kappay[j],
				    kappaz[k], DT,
				    DS, v0.x[i][j][k],
				    v0.x[i + 1][j][k],
				    v0.x[i - 1][j][k],
				    v0.x[i + 2][j][k],
				    v0.y[i][j - 1][k],
				    v0.y[i][j][k],
				    v0.y[i][j - 2][k],
				    v0.y[i][j + 1][k],
				    v0.z[i][j][k - 1],
				    v0.z[i][j][k],
				    v0.z[i][j][k - 2],
				    v0.z[i][j][k + 1],
				    place, ABCmethod);

      t0_yy[i][j][k] += staggards4(kapx, mux,
				    kappay[j],
				    kappax2[i],
				    kappaz[k], DT,
				    DS,
				    v0.y[i][j - 1][k],
				    v0.y[i][j][k],
				    v0.y[i][j - 2][k],
				    v0.y[i][j + 1][k],
				    v0.x[i][j][k],
				    v0.x[i + 1][j][k],
				    v0.x[i - 1][j][k],
				    v0.x[i + 2][j][k],
				    v0.z[i][j][k - 1],
				    v0.z[i][j][k],
				    v0.z[i][j][k - 2],
				    v0.z[i][j][k + 1],
				    place, ABCmethod);

      t0_zz[i][j][k] += staggards4(kapx, mux,
				    kappaz[k],
				    kappax2[i],
				    kappay[j], DT,
				    DS,
				    v0.z[i][j][k - 1],
				    v0.z[i][j][k],
				    v0.z[i][j][k - 2],
				    v0.z[i][j][k + 1],
				    v0.x[i][j][k],
				    v0.x[i + 1][j][k],
				    v0.x[i - 1][j][k],
				    v0.x[i + 2][j][k],
				    v0.y[i][j - 1][k],
				    v0.y[i][j][k],
				    v0.y[i][j - 2][k],
				    v0.y[i][j + 1][k],
				    place, ABCmethod);

      t0_xy[i][j][k] += staggardt4(muy,
				    kappax[i],
				    kappay2[j], DT,
				    DS,
				    v0.y[i - 1][j][k],
				    v0.y[i][j][k],
				    v0.y[i - 2][j][k],
				    v0.y[i + 1][j][k],
				    v0.x[i][j][k],
				    v0.x[i][j + 1][k],
				    v0.x[i][j - 1][k],
				    v0.x[i][j + 2][k],
				    place, ABCmethod);

      t0_xz[i][j][k] += staggardt4(muz,
				    kappax[i],
				    kappaz2[k], DT,
				    DS,
				    v0.z[i - 1][j][k],
				    v0.z[i][j][k],
				    v0.z[i - 2][j][k],
				    v0.z[i + 1][j][k],
				    v0.x[i][j][k],
				    v0.x[i][j][k + 1],
				    v0.x[i][j][k - 1],
				    v0.x[i][j][k + 2],
				    place, ABCmethod);

      t0_yz[i][j][k] += staggardt4(muxyz,
				    kappay2[j],
				    kappaz2[k], DT,
				    DS, v0.z[i][j][k],
				    v0.z[i][j + 1][k],
				    v0.z[i][j - 1][k],
				    v0.z[i][j + 2][k],
				    v0.y[i][j][k],
				    v0.y[i][j][k + 1],
				    v0.y[i][j][k - 1],
				    v0.y[i][j][k + 2],
				    place, ABCmethod);
    }

    /* ****************************** */
    /*  ABSORBING LAYER special part  */
    /* ****************************** */
    if (place == ABSORBINGLAYER) {
      b1 = kapx + 4. * mux / 3.;
      b2 = kapx - 2. * mux / 3.;

      t0_xx[i][j][k] +=
	DT * b2 * (phivyy[npml] +
		   phivzz[npml]) +
	DT * b1 * phivxx[npml];
      t0_yy[i][j][k] +=
	DT * b2 * (phivxx[npml] +
		   phivzz[npml]) +
	DT * b1 * phivyy[npml];
      t0_zz[i][j][k] +=
	DT * b2 * (phivxx[npml] +
		   phivyy[npml]) +
	DT * b1 * phivzz[npml];

      t0_xy[i][j][k] +=
	DT * muy * (phivyx[npml] +
		    phivxy[npml]);
      t0_xz[i][j][k] +=
	DT * muz * (phivzx[npml] +
		    phivxz[npml]);
      t0_yz[i][j][k] +=
	DT * muxyz * (phivzy[npml] +
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
	t0_xz[i][j][1] = -t0_xz[i][j][0];
	t0_yz[i][j][1] = -t0_yz[i][j][0];
	t0_zz[i][j][1] = 0.;

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

	t0_xx[i][j][k] +=
	  b1 * DT * ((9. / 8.) *
		     (v0.x[i + 1][j][k] -
		      v0.x[i][j][k])
		     -
		     (1. / 24.) *
		     (v0.x[i + 2][j][k] -
		      v0.x[i -
			   1][j][k])) / (DS *
					 
					 kappax2[i])
	  +
	  b2 * DT * ((9. / 8.) *
		     (v0.y[i][j][k] -
		      v0.y[i][j - 1][k])
		     -
		     (1. / 24.) *
		     (v0.y[i][j + 1][k] -
		      v0.y[i][j -
			      2][k])) / (DS *
					 
					 kappay[j]);

	t0_yy[i][j][k] +=
	  b1 * DT * ((9. / 8.) *
		     (v0.y[i][j][k] -
		      v0.y[i][j - 1][k])
		     -
		     (1. / 24.) *
		     (v0.y[i][j + 1][k] -
		      v0.y[i][j -
			      2][k])) / (DS *
					 
					 kappay[j])
	  +
	  b2 * DT * ((9. / 8.) *
		     (v0.x[i + 1][j][k] -
		      v0.x[i][j][k])
		     -
		     (1. / 24.) *
		     (v0.x[i + 2][j][k] -
		      v0.x[i -
			   1][j][k])) / (DS *
					 
					 kappax2[i]);

	/* t0->xy computed like usual */
	t0_xy[i][j][k] +=
	  staggardt4(muy, un, un, DT, DS,
		     v0.y[i - 1][j][k],
		     v0.y[i][j][k],
		     v0.y[i - 2][j][k],
		     v0.y[i + 1][j][k],
		     v0.x[i][j][k],
		     v0.x[i][j + 1][k],
		     v0.x[i][j - 1][k],
		     v0.x[i][j + 2][k], place,
		     ABCmethod);
      }	/* end if k=1 */
      if (k == 2) {
	/* imposed values */
	/* (other values are not needed) */
	t0_xz[i][j][2] = -t0_xz[i][j][-1];
	t0_yz[i][j][2] = -t0_yz[i][j][-1];
	t0_zz[i][j][2] = -t0_zz[i][j][0];
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


	  t0_xx[i][j][k] +=
	    DT * b2 * (phivyy[npml] +
		       phivzz[npml]) +
	    DT * b1 * phivxx[npml];
	  t0_yy[i][j][k] +=
	    DT * b2 * (phivxx[npml] +
		       phivzz[npml]) +
	    DT * b1 * phivyy[npml];
	  t0_xy[i][j][k] +=
	    DT * muy * (phivyx[npml] +
			phivxy[npml]);
	}

      }	/* end if k == 1 */
    }		/* end FREEABS special part */
  }		/* end for k */
}
