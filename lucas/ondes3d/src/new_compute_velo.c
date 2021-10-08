struct starpu_codelet velo_cl = {
				 .cpu_funcs = {compute_velo_task},
				 .nbuffers = 42,
				 .modes = {STARPU_RW, STARPU_RW, STARPU_RW, STARPU_W, STARPU_W, STARPU_W,
					   STARPU_W, STARPU_W, STARPU_W, STARPU_W, STARPU_W, STARPU_W,
					   STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R,
					   STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R,
					   STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R,
					   STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R,
					   STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R,}
};


void compute_velo_task(void *buffers[], void *cl_arg) {

  //unpack structures
  double ***v0_x = (double ***)STARPU_BLOCK_GET_PTR(buffers[0]);
  double ***v0_y = (double ***)STARPU_BLOCK_GET_PTR(buffers[1]);
  double ***v0_z = (double ***)STARPU_BLOCK_GET_PTR(buffers[2]);

  double *phitxxx = (float *)STARPU_VECTOR_GET_PTR(buffers[3]);
  double *phitxyy = (float *)STARPU_VECTOR_GET_PTR(buffers[4]);
  double *phitxzz = (float *)STARPU_VECTOR_GET_PTR(buffers[5]);
  double *phitxyx = (float *)STARPU_VECTOR_GET_PTR(buffers[6]);
  double *phityyy = (float *)STARPU_VECTOR_GET_PTR(buffers[7]);
  double *phityzz = (float *)STARPU_VECTOR_GET_PTR(buffers[8]);
  double *phitxzx = (float *)STARPU_VECTOR_GET_PTR(buffers[9]);
  double *phityzy = (float *)STARPU_VECTOR_GET_PTR(buffers[10]);
  double *phitzzz = (float *)STARPU_VECTOR_GET_PTR(buffers[11]);

  int *k2ly0 = (int *)STARPU_VECTOR_GET_PTR(buffers[12]);
  int *k2ly2 = (int *)STARPU_VECTOR_GET_PTR(buffers[13]);
  int *rho0 = (int *)STARPU_VECTOR_GET_PTR(buffers[14]);
  int *rho2 = (int *)STARPU_VECTOR_GET_PTR(buffers[15]);
  double *mu0 = (double *)STARPU_VECTOR_GET_PTR(buffers[16]);
  double *mu2 = (double *)STARPU_VECTOR_GET_PTR(buffers[17]);
  double *kap0 = (double *)STARPU_VECTOR_GET_PTR(buffers[18]);
  double *kap2 = (double *)STARPU_VECTOR_GET_PTR(buffers[19]);

  double *dumpx = (double *)STARPU_VECTOR_GET_PTR(buffers[20]);
  double *dumpx2 = (double *)STARPU_VECTOR_GET_PTR(buffers[21]);
  double *dumpy = (double *)STARPU_VECTOR_GET_PTR(buffers[22]);
  double *dumpy2 = (double *)STARPU_VECTOR_GET_PTR(buffers[23]);
  double *dumpz = (double *)STARPU_VECTOR_GET_PTR(buffers[24]);
  double *dumpz2 = (double *)STARPU_VECTOR_GET_PTR(buffers[25]);

  double *alphax = (double *)STARPU_VECTOR_GET_PTR(buffers[26]);
  double *alphax2 = (double *)STARPU_VECTOR_GET_PTR(buffers[27]);
  double *alphay = (double *)STARPU_VECTOR_GET_PTR(buffers[28]);
  double *alphay2 = (double *)STARPU_VECTOR_GET_PTR(buffers[29]);
  double *alphaz = (double *)STARPU_VECTOR_GET_PTR(buffers[30]);
  double *alphaz2 = (double *)STARPU_VECTOR_GET_PTR(buffers[31]);

  double *kappax = (double *)STARPU_VECTOR_GET_PTR(buffers[32]);
  double *kappax2 = (double *)STARPU_VECTOR_GET_PTR(buffers[33]);
  double *kappay = (double *)STARPU_VECTOR_GET_PTR(buffers[34]);
  double *kappay2 = (double *)STARPU_VECTOR_GET_PTR(buffers[35]);
  double *kappaz = (double *)STARPU_VECTOR_GET_PTR(buffers[36]);
  double *kappaz2 = (double *)STARPU_VECTOR_GET_PTR(buffers[37]);

  int *ipml = (int *)STARPU_VECTOR_GET_PTR(buffers[38]);

  double ***fx = (double ***)STARPU_BLOCK_GET_PTR(buffers[39]);
  double ***fy = (double ***)STARPU_BLOCK_GET_PTR(buffers[40]);
  double ***fz = (double ***)STARPU_BLOCK_GET_PTR(buffers[41]);

  long int first_npml;
  int i, j, k, imp, jmp;
  double ds, dt;
  struct PARAMETERS prm;
  starpu_codelet_unpack_args(cl_arg, &i, &j, &imp, &jmp, &first_npml, &prm);
  
  int i_block = i%prm.block_size;
  int j_block = j%prm.block_size;

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
  int i, j, k;		/* local position of the cell */
  int imp, jmp;		/* global position of the cell (NB : kmp=k) */
  /* intermediates */
  double xdum, ydum, zdum;
  double phixdum, phiydum, phizdum;

  /* source == VELO */
  double rho;			/* density at [imp][jmp][k] */

  /* For mapping */
  double DS, DT;		/* PARAMETERS */

  /* mapping */
  DS = prm.ds;
  DT = prm.dt;

  jmp = prm.jmp2j_array[j];
  imp = prm.imp2i_array[i];
  for (k = prm.zMin - prm.delta; k <= prm.zMax0; k++) {

    /* INITIALISATIONS */
    place = WhereAmI(imp, jmp, k, prm);

    if (place == OUTSIDE) {
      continue;
    } else if (place == LIMIT) {
      v0_x[i][j][k] = 0.;
      v0_y[i][j][k] = 0.;
      v0_z[i][j][k] = 0.;
      continue;
    } else if (place == ABSORBINGLAYER || place == FREEABS) {
      npml = ipml[i][j][k];
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
    ly0 = k2ly0[k];
    ly2 = k2ly2[k];

    bx = 1.0 / rho0[ly0];
    by = 1.0 / rho0[ly0];

    rhoxz = rho2[ly2];
    bz = 1.0 / rhoxz;


    /*******************************************/
    /* REGULAR & ABSORBING LAYER               */
    /*******************************************/
    if (place == REGULAR || place == ABSORBINGLAYER) {
      /* Computation of Vx,Vy and Vz */
      v0_x[i][j][k] += staggardv4(bx,
				   kappax[i],
				   kappay[j],
				   kappaz[k], DT,
				   DS,
				   t0.xx[i - 1][j][k],
				   t0.xx[i][j][k],
				   t0.xx[i - 2][j][k],
				   t0.xx[i + 1][j][k],
				   t0.xy[i][j - 1][k],
				   t0.xy[i][j][k],
				   t0.xy[i][j - 2][k],
				   t0.xy[i][j + 1][k],
				   t0.xz[i][j][k - 1],
				   t0.xz[i][j][k],
				   t0.xz[i][j][k - 2],
				   t0.xz[i][j][k + 1],
				   place, ABCmethod);

      v0_y[i][j][k] += staggardv4(by,
				   kappax2[i],
				   kappay2[j],
				   kappaz[k], DT,
				   DS, t0.xy[i][j][k],
				   t0.xy[i + 1][j][k],
				   t0.xy[i - 1][j][k],
				   t0.xy[i + 2][j][k],
				   t0.yy[i][j][k],
				   t0.yy[i][j + 1][k],
				   t0.yy[i][j - 1][k],
				   t0.yy[i][j + 2][k],
				   t0.yz[i][j][k - 1],
				   t0.yz[i][j][k],
				   t0.yz[i][j][k - 2],
				   t0.yz[i][j][k + 1],
				   place, ABCmethod);

      v0_z[i][j][k] += staggardv4(bz,
				   kappax2[i],
				   kappay[j],
				   kappaz2[k], DT,
				   DS, t0.xz[i][j][k],
				   t0.xz[i + 1][j][k],
				   t0.xz[i - 1][j][k],
				   t0.xz[i + 2][j][k],
				   t0.yz[i][j - 1][k],
				   t0.yz[i][j][k],
				   t0.yz[i][j - 2][k],
				   t0.yz[i][j + 1][k],
				   t0.zz[i][j][k],
				   t0.zz[i][j][k + 1],
				   t0.zz[i][j][k - 1],
				   t0.zz[i][j][k + 2],
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
	v0_x[i][j][k] += staggardv4(bx,
				     kappax[i],
				     kappay[j],
				     kappaz[k],
				     DT, DS,
				     t0.xx[i -
					   1][j][k],
				     t0.xx[i][j][k],
				     t0.xx[i -
					   2][j][k],
				     t0.xx[i +
					   1][j][k],
				     t0.xy[i][j -
					      1][k],
				     t0.xy[i][j][k],
				     t0.xy[i][j -
					      2][k],
				     t0.xy[i][j +
					      1][k],
				     t0.xz[i][j][k -
						 1],
				     t0.xz[i][j][k],
				     t0.xz[i][j][k -
						 2],
				     t0.xz[i][j][k +
						 1],
				     place, ABCmethod);

	v0_y[i][j][k] += staggardv4(by,
				     kappax2[i],
				     kappay2[j],
				     kappaz[k],
				     DT, DS,
				     t0.xy[i][j][k],
				     t0.xy[i +
					   1][j][k],
				     t0.xy[i -
					   1][j][k],
				     t0.xy[i +
					   2][j][k],
				     t0.yy[i][j][k],
				     t0.yy[i][j +
					      1][k],
				     t0.yy[i][j -
					      1][k],
				     t0.yy[i][j +
					      2][k],
				     t0.yz[i][j][k -
						 1],
				     t0.yz[i][j][k],
				     t0.yz[i][j][k -
						 2],
				     t0.yz[i][j][k +
						 1],
				     place, ABCmethod);
	/* v0->z */
	/* expression is obtained considering :
	 *  0- Elastic Formulation approximation (no anelasticity part taken into account)
	 *  1- a 2nd order of approximation for vz
	 *  2- we are searching vz(k=1) as dz(vz)|z=0 consistent with the followings expressions :
	 *
	 *  (dt(tzz)|z=0) = 0 = ( (kappa-2/3 mu)*( dx(vx) + dy(vy) ) + (kappa + 4/3 mu) * dz(vz) )|z=0
	 *
	 *  and 2nd Order of deviration
	 *
	 *  dz(vz)|z=0 = (vz(z=+3/2*DS) - vz(DS/2))/dz
	 *
	 *  so vz(k=1) can be determined with vy, vx, and vz(k=0).
	 */
	kapx = kap0[ly0];
	mux = mu0[ly0];

	kapy = kap0[ly0];
	muy = mu0[ly0];

	v0_z[i][j][k] = v0_z[i][j][k - 1]
	  - (kapx - 2. / 3. * mux) / (kapx +
				      4. / 3. *
				      mux) *
	  (v0_x[i + 1][j][k] - v0_x[i][j][k])
	  - (kapy - 2. / 3. * mux) / (kapy +
				      4. / 3. *
				      muy) *
	  (v0_y[i][j][k] - v0_y[i][j - 1][k]);

      }	/* end k = 1 */

      /* k=2 */
      /*-----*/
      /*
       * 2nd order approximation.
       * Details :
       * Cf."Simulating Seismic Wave propagation in 3D elastic Media Using staggered-Grid Finite Difference"
       *  [ Robert W.Graves, p. 1099 ]
       *   Bulletin of the Seismological Society of America Vol 4. August 1996
       */
      if (k == 2) {
	v0_x[i][j][k] = v0_x[i][j][k - 1]
	  - (v0_z[i][j][k - 1] -
	     v0_z[i - 1][j][k - 1])
	  - (v0_x[i][j][k - 1] -
	     v0_x[i][j][k - 2] + v0_z[i][j][k -
					      2] -
	     v0_z[i - 1][j][k - 2]);

	v0_y[i][j][k] = v0_y[i][j][k - 1]
	  - (v0_z[i][j + 1][k - 1] -
	     v0_z[i][j][k - 1])
	  - (v0_y[i][j][k - 1] -
	     v0_y[i][j][k - 2] + v0_z[i][j + 1][k -
						  2]
	     - v0_z[i][j][k - 2]);

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
      phixdum = phitxxx[npml];
      phiydum = phitxyy[npml];
      phizdum = phitxzz[npml];

      phitxxx[npml] =
	CPML4(vp, dumpx[i], alphax[i],
	      kappax[i], phixdum, DS, DT,
	      t0.xx[i - 1][j][k], t0.xx[i][j][k],
	      t0.xx[i - 2][j][k],
	      t0.xx[i + 1][j][k]);
      phitxyy[npml] =
	CPML4(vp, dumpy[j], alphay[j],
	      kappay[j], phiydum, DS, DT,
	      t0.xy[i][j - 1][k], t0.xy[i][j][k],
	      t0.xy[i][j - 2][k],
	      t0.xy[i][j + 1][k]);
      phitxzz[npml] =
	CPML4(vp, dumpz[k], alphaz[k],
	      kappaz[k], phizdum, DS, DT,
	      t0.xz[i][j][k - 1], t0.xz[i][j][k],
	      t0.xz[i][j][k - 2],
	      t0.xz[i][j][k + 1]);
      v0_x[i][j][k] +=
	bx * DT * (phitxxx[npml] +
		   phitxyy[npml] +
		   phitxzz[npml]);

      /* Calculation of vy */
      phixdum = phitxyx[npml];
      phiydum = phityyy[npml];
      phizdum = phityzz[npml];

      phitxyx[npml] =
	CPML4(vpxy, dumpx2[i],
	      alphax2[i], kappax2[i],
	      phixdum, DS, DT, t0.xy[i][j][k],
	      t0.xy[i + 1][j][k],
	      t0.xy[i - 1][j][k],
	      t0.xy[i + 2][j][k]);
      phityyy[npml] =
	CPML4(vpxy, dumpy2[j],
	      alphay2[j], kappay2[j],
	      phiydum, DS, DT, t0.yy[i][j][k],
	      t0.yy[i][j + 1][k],
	      t0.yy[i][j - 1][k],
	      t0.yy[i][j + 2][k]);
      phityzz[npml] =
	CPML4(vpxy, dumpz[k], alphaz[k],
	      kappaz[k], phizdum, DS, DT,
	      t0.yz[i][j][k - 1], t0.yz[i][j][k],
	      t0.yz[i][j][k - 2],
	      t0.yz[i][j][k + 1]);

      v0_y[i][j][k] +=
	by * DT * (phitxyx[npml] +
		   phityyy[npml] +
		   phityzz[npml]);

      /* Calculation of vz */
      phixdum = phitxzx[npml];
      phiydum = phityzy[npml];
      phizdum = phitzzz[npml];

      phitxzx[npml] =
	CPML4(vpxz, dumpx2[i],
	      alphax2[i], kappax2[i],
	      phixdum, DS, DT, t0.xz[i][j][k],
	      t0.xz[i + 1][j][k],
	      t0.xz[i - 1][j][k],
	      t0.xz[i + 2][j][k]);
      phityzy[npml] =
	CPML4(vpxz, dumpy[j], alphay[j],
	      kappay[j], phiydum, DS, DT,
	      t0.yz[i][j - 1][k], t0.yz[i][j][k],
	      t0.yz[i][j - 2][k],
	      t0.yz[i][j + 1][k]);
      if (place == ABSORBINGLAYER) {
	phitzzz[npml] =
	  CPML4(vpxz, dumpz2[k],
		alphaz2[k], kappaz2[k],
		phizdum, DS, DT, t0.zz[i][j][k],
		t0.zz[i][j][k + 1],
		t0.zz[i][j][k - 1],
		t0.zz[i][j][k + 2]);
      } else if (place == FREEABS && k == prm.zMax0 - 1) {	/* phitzzz = 0. since dz(tzz)==0 at Free Surface */
	phitzzz[npml] = 0.;
      }

      v0_z[i][j][k] +=
	bz * DT * (phitxzx[npml] +
		   phityzy[npml] +
		   phitzzz[npml]);
      /* end of Calculation of Vz */
    }		/*         end of ( place == ABSORBINGLAYER  ) */

    /*=========================================*
     * Add Source PART                histfile *
     *=========================================*/
    /* Ajout des Sources.hist */
    if (source == HISTFILE) {
      v0_x[i][j][k] += bx * fx[i][j][k] * DT / DS;
      v0_y[i][j][k] += by * fy[i][j][k] * DT / DS;
      v0_z[i][j][k] += bz * fz[i][j][k] * DT / DS;
    }		/* end of if source */
  }		/* end for k */
}
