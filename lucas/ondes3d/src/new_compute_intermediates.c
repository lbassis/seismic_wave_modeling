struct starpu_codelet intermediates_cl = {
					  .cpu_funcs = {compute_intermediate_task},
					  .nbuffers = 39,
					  .modes = {STARPU_W, STARPU_W, STARPU_W, STARPU_W, STARPU_W, STARPU_W, STARPU_W, STARPU_W, STARPU_W,
						    STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R,
						    STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R,
						    STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R,
						    STARPU_R, STARPU_R, STARPU_R},
};


void compute_intermediate_task(void *buffers[], void *cl_arg) {

  //unpack structures
  double *phivxx = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);
  double *phivyy = (float *)STARPU_VECTOR_GET_PTR(buffers[1]);
  double *phivzz = (float *)STARPU_VECTOR_GET_PTR(buffers[2]);
  double *phivyx = (float *)STARPU_VECTOR_GET_PTR(buffers[3]);
  double *phivxy = (float *)STARPU_VECTOR_GET_PTR(buffers[4]);
  double *phivzx = (float *)STARPU_VECTOR_GET_PTR(buffers[5]);
  double *phivxz = (float *)STARPU_VECTOR_GET_PTR(buffers[6]);
  double *phivzy = (float *)STARPU_VECTOR_GET_PTR(buffers[7]);
  double *phivyz = (float *)STARPU_VECTOR_GET_PTR(buffers[8]);

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

  double ***v0_x = (double ***)STARPU_BLOCK_GET_PTR(buffers[36]);
  double ***v0_y = (double ***)STARPU_BLOCK_GET_PTR(buffers[37]);
  double ***v0_z = (double ***)STARPU_BLOCK_GET_PTR(buffers[38]);

  int first_npml;
  int i, j, k, imp, jmp;
  double ds, dt;
  struct PARAMETERS prm;
  starpu_codelet_unpack_args(cl_arg, &i, &j, &imp, &jmp, &first_npml, &prm);
  
  int i_block = i%prm.block_size;
  int j_block = j%prm.block_size;
  
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

  ds = prm.ds;
  dt = prm.dt;
  
  /* loop */
  for (k = prm.zMin - prm.delta; k <= prm.zMax0; k++) {

    /* INITIALISATIONS */
    place = WhereAmI(imp, jmp, k, prm);

    /* jump "Not computed area" */
    if ((place == OUTSIDE) || (place == LIMIT)) {
      continue;
    }
    /* find the right npml number */
    if ((place == ABSORBINGLAYER) || (place == FREEABS)) {
      npml = ipml[k] - first_npml;
    }
    /* medium */
    /* Warning : k2ly0 & k2ly2
       give 0 if k in FREEABS or if depth(k) > laydep[0] in general */
    ly0 = k2ly0[k];
    ly2 = k2ly2[k];

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
      phixdum = phivxx[npml];
      phiydum = phivyy[npml];
      phizdum = phivzz[npml];

      phivxx[npml] =
	CPML4(vpx, dumpx2[i], alphax2[i],
	      kappax2[i], phixdum, DS, DT,
	      v0.x[i][j][k], v0.x[i + 1][j][k],
	      v0.x[i - 1][j][k],
	      v0.x[i + 2][j][k]);
      phivyy[npml] =
	CPML4(vpx, dumpy[j], alphay[j],
	      kappay[j], phiydum, DS, DT,
	      v0.y[i][j - 1][k], v0.y[i][j][k],
	      v0.y[i][j - 2][k],
	      v0.y[i][j + 1][k]);
      phivzz[npml] =
	CPML4(vpx, dumpz[k], alphaz[k],
	      kappaz[k], phizdum, DS, DT,
	      v0.z[i][j][k - 1], v0.z[i][j][k],
	      v0.z[i][j][k - 2],
	      v0.z[i][j][k + 1]);
      /* txy */
      phixdum = phivyx[npml];
      phiydum = phivxy[npml];

      phivyx[npml] =
	CPML4(vpy, dumpx[i], alphax[i],
	      kappax[i], phixdum, DS, DT,
	      v0.y[i - 1][j][k], v0.y[i][j][k],
	      v0.y[i - 2][j][k],
	      v0.y[i + 1][j][k]);
      phivxy[npml] =
	CPML4(vpy, dumpy2[j], alphay2[j],
	      kappay2[j], phiydum, DS, DT,
	      v0.x[i][j][k], v0.x[i][j + 1][k],
	      v0.x[i][j - 1][k],
	      v0.x[i][j + 2][k]);
      /* txz */
      phixdum = phivzx[npml];
      phizdum = phivxz[npml];

      phivzx[npml] =
	CPML4(vpz, dumpx[i], alphax[i],
	      kappax[i], phixdum, DS, DT,
	      v0.z[i - 1][j][k], v0.z[i][j][k],
	      v0.z[i - 2][j][k],
	      v0.z[i + 1][j][k]);
      phivxz[npml] =
	CPML4(vpz, dumpz2[k], alphaz2[k],
	      kappaz2[k], phizdum, DS, DT,
	      v0.x[i][j][k], v0.x[i][j][k + 1],
	      v0.x[i][j][k - 1],
	      v0.x[i][j][k + 2]);
      /* tyz */
      phiydum = phivzy[npml];
      phizdum = phivyz[npml];

      phivzy[npml] =
	CPML4(vpxyz, dumpy2[j],
	      alphay2[j], kappay2[j],
	      phiydum, DS, DT, v0.z[i][j][k],
	      v0.z[i][j + 1][k], v0.z[i][j - 1][k],
	      v0.z[i][j + 2][k]);
      phivyz[npml] =
	CPML4(vpxyz, dumpz2[k],
	      alphaz2[k], kappaz2[k],
	      phizdum, DS, DT, v0.y[i][j][k],
	      v0.y[i][j][k + 1], v0.y[i][j][k - 1],
	      v0.y[i][j][k + 2]);
    }		/* End compute Absorbing Layers */

    /* FREEABS      */
    /* ------------ */
    /* We only need to compute t0xx(k=1),t0yy(k=1),t0xy(k=1) 
     * So each coefficient will be computed only for those part 
     *
     * Expressions are simply obtain from approximation of dz(v0.z) in an elastic medium :
     * DT(t0.zz)|z=0 = 0 = lam *(dx(v0.x) + dy(v0.y) + dz(v0.z)) + 2*mu * dz(v0.z);
     *
     */
    if (place == FREEABS) {
      /* approximate dz(v0.z) (4th order) */
      dzV0z =
	(kapx - 2. / 3. * mux) / (kapx +
				  4. / 3. * mux) *
	(Diff4
	 (DS, v0.x[i][j][k], v0.x[i + 1][j][k],
	  v0.x[i - 1][j][k],
	  v0.x[i + 2][j][k]) + Diff4(DS,
				     v0.y[i][j - 1][k],
				     v0.y[i][j][k],
				     v0.y[i][j - 2][k],
				     v0.y[i][j + 1][k])
	 );

      /* txx, tyy */
      phixdum = phivxx[npml];
      phiydum = phivyy[npml];
      phizdum = phivzz[npml];
      /* (copy&paste) */
      phivxx[npml] =
	CPML4(vpx, dumpx2[i], alphax2[i],
	      kappax2[i], phixdum, DS, DT,
	      v0.x[i][j][k], v0.x[i + 1][j][k],
	      v0.x[i - 1][j][k],
	      v0.x[i + 2][j][k]);
      phivyy[npml] =
	CPML4(vpx, dumpy[j], alphay[j],
	      kappay[j], phiydum, DS, DT,
	      v0.y[i][j - 1][k], v0.y[i][j][k],
	      v0.y[i][j - 2][k],
	      v0.y[i][j + 1][k]);
      /* special */
      b = exp(-
	      (vpx * dumpz[k] / kappaz[k] +
	       alphaz[k]) * DT);
      a = 0.0;
      if (abs(vpx * dumpz[k]) > 0.000001) {
	a = vpx * dumpz[k] * (b -
				   1.0) /
	  (kappaz[k] *
	   (vpx * dumpz[k] +
	    kappaz[k] * alphaz[k]));
      }
      phivzz[npml] = b * phizdum + a * (dzV0z);

      /* txy ( copy&paste) */
      phixdum = phivyx[npml];
      phiydum = phivxy[npml];

      phivyx[npml] =
	CPML4(vpy, dumpx[i], alphax[i],
	      kappax[i], phixdum, DS, DT,
	      v0.y[i - 1][j][k], v0.y[i][j][k],
	      v0.y[i - 2][j][k],
	      v0.y[i + 1][j][k]);
      phivxy[npml] =
	CPML4(vpy, dumpy2[j], alphay2[j],
	      kappay2[j], phiydum, DS, DT,
	      v0.x[i][j][k], v0.x[i][j + 1][k],
	      v0.x[i][j - 1][k],
	      v0.x[i][j + 2][k]);
    }
  }
}
