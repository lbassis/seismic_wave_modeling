#include <starpu.h>

#include "../include/struct.h"
#include "../include/new_nrutil.h"
#include "../include/new_compute_seismoment.h"
#include "../include/new_compute_intermediates.h"
#include "../include/new_compute_stress.h"
#include "../include/new_compute_velo.h"
#include "../include/inlineFunctions.h"

enum starpu_data_access_mode modes_seis[16] =
{
		STARPU_W, STARPU_W, STARPU_W, STARPU_R,
			STARPU_R, STARPU_R, STARPU_R, STARPU_R,
			STARPU_R, STARPU_R, STARPU_R, STARPU_R,
			STARPU_R, STARPU_R, STARPU_R, STARPU_R
};

struct starpu_codelet seis_moment_cl = {
					.cpu_funcs = {seis_moment_task},
					.nbuffers = 16,
					.name = "seis_moment",
					.dyn_modes = modes_seis,
};

enum starpu_data_access_mode modes_intermediates[31] =
{
	STARPU_W,
			STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R,
			STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R,
			STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R,
			STARPU_R, STARPU_R, STARPU_R
};

struct starpu_codelet intermediates_cl = {
					  .cpu_funcs = {compute_intermediates_task},
					  .nbuffers = 31,
						.name = "intermediates",
					  .dyn_modes = modes_intermediates
};

enum starpu_data_access_mode modes_stress[22] =
{
 STARPU_W, STARPU_W, STARPU_RW, STARPU_W, STARPU_RW, STARPU_RW,
 STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R,
 STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R,
 STARPU_R, STARPU_R, STARPU_R, STARPU_R};

struct starpu_codelet stress_cl = {
				   .cpu_funcs = {compute_stress_task},
				   .nbuffers = 22,
					 .name = "stress",
				   .dyn_modes = modes_stress,
};

enum starpu_data_access_mode modes_velo[40] =
{
 STARPU_RW, STARPU_RW, STARPU_RW, STARPU_W,
 STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R,
 STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R,
 STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R,
 STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R,
 STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R,
 STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R,
};

struct starpu_codelet velo_cl = {
				 .cpu_funcs = {compute_velo_task},
				 .nbuffers = 40,
				 .name = "velo",
				 .dyn_modes = modes_velo
};


void main_loop(struct SOURCE *SRC, struct ABSORBING_BOUNDARY_CONDITION *ABC,
	       struct MEDIUM *MDM, struct STRESS *t0, struct VELOCITY *v0, struct PARAMETERS *PRM) {

  int i, j, k, index, neighboors, desc_index, current_neighboor;
  int i_block, j_block;
  double time;

  // starpu structures common to every iteration
  int nrows = PRM->mpmx + 4;
  int ncols = PRM->mpmy + 4;
  int depth = PRM->zMax0 - (PRM->zMin - PRM->delta) + 1;

  int n_blocks_x = ncols/PRM->block_size;
  int n_blocks_y = nrows/PRM->block_size;
	PRM->n_blocks_x = n_blocks_x;
	PRM->n_blocks_y = n_blocks_y;
	PRM->depth = depth;
  printf("nblocksx = %d, nblocksy = %d\n", n_blocks_x, n_blocks_y);
  int mpmz = PRM->zMax0 - (PRM->zMin - PRM->delta);

  struct starpu_data_filter x_filter = {.filter_func = starpu_block_filter_block, .nchildren = n_blocks_x};
  struct starpu_data_filter y_filter = {.filter_func = starpu_block_filter_vertical_block,.nchildren = n_blocks_y};

  // computeSeisMoment handles:

  starpu_data_handle_t src_vel_handle, src_strike_handle, src_dip_handle, src_rake_handle, src_xweight_handle, src_yweight_handle, src_zweight_handle,
    src_insrc_handle, src_ixhypo_handle, src_iyhypo_handle, src_izhypo_handle, prm_i2imp_handle, prm_j2jmp_handle, src_fx_handle, src_fy_handle, src_fz_handle;

  starpu_matrix_data_register(&src_vel_handle, STARPU_MAIN_RAM, (uintptr_t)SRC->vel, SRC->iDur, SRC->iDur, SRC->iSrc, sizeof(SRC->vel[0]));
  starpu_vector_data_register(&src_strike_handle, STARPU_MAIN_RAM, (uintptr_t)SRC->strike, SRC->iSrc, sizeof(SRC->strike[0]));
  starpu_vector_data_register(&src_dip_handle, STARPU_MAIN_RAM, (uintptr_t)SRC->dip, SRC->iSrc, sizeof(SRC->dip[0]));
  starpu_vector_data_register(&src_rake_handle, STARPU_MAIN_RAM, (uintptr_t)SRC->rake, SRC->iSrc, sizeof(SRC->rake[0]));
  starpu_vector_data_register(&src_xweight_handle, STARPU_MAIN_RAM, (uintptr_t)SRC->xweight, SRC->iSrc, sizeof(SRC->xweight[0]));
  starpu_vector_data_register(&src_yweight_handle, STARPU_MAIN_RAM, (uintptr_t)SRC->yweight, SRC->iSrc, sizeof(SRC->yweight[0]));
  starpu_vector_data_register(&src_zweight_handle, STARPU_MAIN_RAM, (uintptr_t)SRC->zweight, SRC->iSrc, sizeof(SRC->zweight[0]));
  starpu_vector_data_register(&src_insrc_handle, STARPU_MAIN_RAM, (uintptr_t)SRC->insrc, SRC->iSrc, sizeof(SRC->insrc[0]));
  starpu_vector_data_register(&src_ixhypo_handle, STARPU_MAIN_RAM, (uintptr_t)SRC->ixhypo, SRC->iSrc, sizeof(SRC->ixhypo[0]));
  starpu_vector_data_register(&src_iyhypo_handle, STARPU_MAIN_RAM, (uintptr_t)SRC->iyhypo, SRC->iSrc, sizeof(SRC->iyhypo[0]));
  starpu_vector_data_register(&src_izhypo_handle, STARPU_MAIN_RAM, (uintptr_t)SRC->izhypo, SRC->iSrc, sizeof(SRC->izhypo[0]));
  starpu_vector_data_register(&prm_i2imp_handle, STARPU_MAIN_RAM, (uintptr_t)PRM->i2imp_array, ((PRM->xMax+2*PRM->delta+2) - (PRM->xMin - PRM->delta) + 1) , sizeof(PRM->i2imp_array[0]));
  starpu_vector_data_register(&prm_j2jmp_handle, STARPU_MAIN_RAM, (uintptr_t)PRM->j2jmp_array, ((PRM->yMax+2*PRM->delta+2) - (PRM->yMin - PRM->delta) + 1) , sizeof(PRM->j2jmp_array[0]));

  starpu_block_data_register(&src_fx_handle, STARPU_MAIN_RAM, (uintptr_t)SRC->fx, ncols, depth, nrows, ncols, depth, sizeof(SRC->fx[0]));
  starpu_block_data_register(&src_fy_handle, STARPU_MAIN_RAM, (uintptr_t)SRC->fy, ncols, depth, nrows, ncols, depth, sizeof(SRC->fy[0]));
  starpu_block_data_register(&src_fz_handle, STARPU_MAIN_RAM, (uintptr_t)SRC->fz, ncols, depth, nrows, ncols, depth, sizeof(SRC->fz[0]));


  // computeIntermediate handles:
  starpu_data_handle_t ipml_handle;
  starpu_data_handle_t k2ly0_handle, k2ly2_handle, mu0_handle, mu2_handle, kap0_handle, kap2_handle, rho0_handle, rho2_handle;
  starpu_data_handle_t dumpx_handle, dumpx2_handle, dumpy_handle, dumpy2_handle, dumpz_handle, dumpz2_handle;
  starpu_data_handle_t alphax_handle, alphax2_handle, alphay_handle, alphay2_handle, alphaz_handle, alphaz2_handle;
  starpu_data_handle_t kappax_handle, kappax2_handle, kappay_handle, kappay2_handle, kappaz_handle, kappaz2_handle;
  starpu_data_handle_t phiv_handle;
  starpu_data_handle_t v0_x_handle, v0_y_handle, v0_z_handle;

  starpu_vector_data_register(&k2ly0_handle,  STARPU_MAIN_RAM, (uintptr_t)MDM->k2ly0, PRM->zMax0 - (PRM->zMin - PRM->delta) + 1, sizeof(MDM->k2ly0[0]));
  starpu_vector_data_register(&k2ly2_handle,  STARPU_MAIN_RAM, (uintptr_t)MDM->k2ly2, PRM->zMax0 - (PRM->zMin - PRM->delta) + 1, sizeof(MDM->k2ly2[0]));
  starpu_vector_data_register(&mu0_handle,  STARPU_MAIN_RAM, (uintptr_t)MDM->mu0, MDM->nLayer, sizeof(MDM->mu0[0]));
  starpu_vector_data_register(&mu2_handle,  STARPU_MAIN_RAM, (uintptr_t)MDM->mu2, MDM->nLayer, sizeof(MDM->mu2[0]));
  starpu_vector_data_register(&kap0_handle,  STARPU_MAIN_RAM, (uintptr_t)MDM->kap0, MDM->nLayer, sizeof(MDM->kap0[0]));
  starpu_vector_data_register(&kap2_handle,  STARPU_MAIN_RAM, (uintptr_t)MDM->kap2, MDM->nLayer, sizeof(MDM->kap2[0]));
  starpu_vector_data_register(&rho0_handle,  STARPU_MAIN_RAM, (uintptr_t)MDM->rho0, MDM->nLayer, sizeof(MDM->rho0[0]));
  starpu_vector_data_register(&rho2_handle,  STARPU_MAIN_RAM, (uintptr_t)MDM->rho2, MDM->nLayer, sizeof(MDM->rho2[0]));

  starpu_vector_data_register(&dumpx_handle,  STARPU_MAIN_RAM, (uintptr_t)ABC->dumpx,  PRM->mpmx, sizeof(ABC->dumpx[0]));
  starpu_vector_data_register(&dumpx2_handle, STARPU_MAIN_RAM, (uintptr_t)ABC->dumpx2, PRM->mpmx, sizeof(ABC->dumpx[0]));
  starpu_vector_data_register(&dumpy_handle,  STARPU_MAIN_RAM, (uintptr_t)ABC->dumpy,  PRM->mpmy, sizeof(ABC->dumpx[0]));
  starpu_vector_data_register(&dumpy2_handle, STARPU_MAIN_RAM, (uintptr_t)ABC->dumpy2, PRM->mpmy, sizeof(ABC->dumpx[0]));
  starpu_vector_data_register(&dumpz_handle,  STARPU_MAIN_RAM, (uintptr_t)ABC->dumpz,  mpmz, sizeof(ABC->dumpx[0]));
  starpu_vector_data_register(&dumpz2_handle, STARPU_MAIN_RAM, (uintptr_t)ABC->dumpz2, mpmz, sizeof(ABC->dumpx[0]));

  starpu_vector_data_register(&alphax_handle,  STARPU_MAIN_RAM, (uintptr_t)ABC->alphax,  PRM->mpmx, sizeof(ABC->alphax[0]));
  starpu_vector_data_register(&alphax2_handle, STARPU_MAIN_RAM, (uintptr_t)ABC->alphax2, PRM->mpmx, sizeof(ABC->alphax[0]));
  starpu_vector_data_register(&alphay_handle,  STARPU_MAIN_RAM, (uintptr_t)ABC->alphay,  PRM->mpmy, sizeof(ABC->alphax[0]));
  starpu_vector_data_register(&alphay2_handle, STARPU_MAIN_RAM, (uintptr_t)ABC->alphay2, PRM->mpmy, sizeof(ABC->alphax[0]));
  starpu_vector_data_register(&alphaz_handle,  STARPU_MAIN_RAM, (uintptr_t)ABC->alphaz,  mpmz, sizeof(ABC->alphax[0]));
  starpu_vector_data_register(&alphaz2_handle, STARPU_MAIN_RAM, (uintptr_t)ABC->alphaz2, mpmz, sizeof(ABC->alphax[0]));

  starpu_vector_data_register(&kappax_handle,  STARPU_MAIN_RAM, (uintptr_t)ABC->kappax,  PRM->mpmx, sizeof(ABC->kappax[0]));
  starpu_vector_data_register(&kappax2_handle, STARPU_MAIN_RAM, (uintptr_t)ABC->kappax2, PRM->mpmx, sizeof(ABC->kappax[0]));
  starpu_vector_data_register(&kappay_handle,  STARPU_MAIN_RAM, (uintptr_t)ABC->kappay,  PRM->mpmy, sizeof(ABC->kappax[0]));
  starpu_vector_data_register(&kappay2_handle, STARPU_MAIN_RAM, (uintptr_t)ABC->kappay2, PRM->mpmy, sizeof(ABC->kappax[0]));
  starpu_vector_data_register(&kappaz_handle,  STARPU_MAIN_RAM, (uintptr_t)ABC->kappaz,  mpmz, sizeof(ABC->kappax[0]));
  starpu_vector_data_register(&kappaz2_handle, STARPU_MAIN_RAM, (uintptr_t)ABC->kappaz2, mpmz, sizeof(ABC->kappax[0]));

  // register handle to be partitioned
  starpu_block_data_register(&v0_x_handle, STARPU_MAIN_RAM, (uintptr_t)v0->x, ncols, depth, nrows, ncols, depth, sizeof(v0->x[0]));
  starpu_block_data_register(&v0_y_handle, STARPU_MAIN_RAM, (uintptr_t)v0->y, ncols, depth, nrows, ncols, depth, sizeof(v0->y[0]));
  starpu_block_data_register(&v0_z_handle, STARPU_MAIN_RAM, (uintptr_t)v0->z, ncols, depth, nrows, ncols, depth, sizeof(v0->z[0]));

  //starpu_data_map_filters(v0_x_handle, 2, &x_filter, &y_filter);
  //starpu_data_map_filters(v0_y_handle, 2, &x_filter, &y_filter);
  //starpu_data_map_filters(v0_z_handle, 2, &x_filter, &y_filter);

  // compute stress handles:
  // t0->xx, t0->yy, t0->zz, t0->xy, t0->xz, t0->yz -> o t0 sempre mexe soh em i,j,k, soh o v0 que pode precisar dos vizinhos
  starpu_data_handle_t t0_xx_handle, t0_yy_handle, t0_zz_handle, t0_xy_handle, t0_xz_handle, t0_yz_handle;

  starpu_block_data_register(&t0_xx_handle, STARPU_MAIN_RAM, (uintptr_t)t0->xx, ncols, depth, nrows, ncols, depth, sizeof(t0->xx[0]));
  starpu_block_data_register(&t0_yy_handle, STARPU_MAIN_RAM, (uintptr_t)t0->yy, ncols, depth, nrows, ncols, depth, sizeof(t0->yy[0]));
  starpu_block_data_register(&t0_zz_handle, STARPU_MAIN_RAM, (uintptr_t)t0->zz, ncols, depth, nrows, ncols, depth, sizeof(t0->zz[0]));
  starpu_block_data_register(&t0_xy_handle, STARPU_MAIN_RAM, (uintptr_t)t0->xy, ncols, depth, nrows, ncols, depth, sizeof(t0->xy[0]));
  starpu_block_data_register(&t0_xz_handle, STARPU_MAIN_RAM, (uintptr_t)t0->xz, ncols, depth, nrows, ncols, depth, sizeof(t0->xz[0]));
  starpu_block_data_register(&t0_yz_handle, STARPU_MAIN_RAM, (uintptr_t)t0->yz, ncols, depth, nrows, ncols, depth, sizeof(t0->yz[0]));

  starpu_data_map_filters(t0_xx_handle, 2, &x_filter, &y_filter);
  starpu_data_map_filters(t0_yy_handle, 2, &x_filter, &y_filter);
  starpu_data_map_filters(t0_zz_handle, 2, &x_filter, &y_filter);
  starpu_data_map_filters(t0_xy_handle, 2, &x_filter, &y_filter);
  starpu_data_map_filters(t0_xz_handle, 2, &x_filter, &y_filter);
  starpu_data_map_filters(t0_yz_handle, 2, &x_filter, &y_filter);

  // compute velo handles:
  starpu_data_handle_t phit_handle;

  /* loops */
  int it;
  for (it = 0; it < 1/*PRM->tMax*/; it++) {
    // seismoment
    time = PRM->dt * it;
    starpu_insert_task(&seis_moment_cl,
		       STARPU_W, src_fx_handle, STARPU_W, src_fy_handle, STARPU_W, src_fz_handle,
		       STARPU_R, src_vel_handle, STARPU_R, src_strike_handle, STARPU_R, src_dip_handle, STARPU_R, src_rake_handle,
		       STARPU_R, src_xweight_handle, STARPU_R, src_yweight_handle, STARPU_R, src_zweight_handle,
		       STARPU_R, src_insrc_handle, STARPU_R, src_ixhypo_handle, STARPU_R, src_iyhypo_handle, STARPU_R, src_izhypo_handle,
		       STARPU_R, prm_i2imp_handle, STARPU_R, prm_j2jmp_handle,
		       STARPU_VALUE, &time, sizeof(time),
		       STARPU_VALUE, PRM, sizeof(*PRM),
		       STARPU_VALUE, &(SRC->dtbiem), sizeof(SRC->dtbiem),
		       STARPU_VALUE, &(SRC->iDur), sizeof(SRC->iDur),
		       STARPU_VALUE, &(SRC->iSrc), sizeof(SRC->iSrc),
		       0);


    //loop compute intermediates
    ////________________________________________
    /// ATENCAO
    ///
    /// GERALMENTE COMECA COM I = 1, MAS NOS BLOCOS NAO DEVE SER ASSIM
    ///
    //__________________________________________
		starpu_task_wait_for_all();
    for (i_block = 0; i_block < n_blocks_y; i_block++) {
      for (j_block = 0; j_block < n_blocks_x; j_block++) {

    	i = i_block*PRM->block_size;
    	j = j_block*PRM->block_size;


	starpu_vector_data_register(&ipml_handle, STARPU_MAIN_RAM, (uintptr_t)ABC->ipml, (PRM->mpmx+3)*(PRM->mpmy+3)*(PRM->zMax0 - (PRM->zMin - PRM->delta)), sizeof(ABC->ipml[0]));

    	long int first_npml = i3access(ABC->ipml, -1, PRM->block_size + 2, -1, PRM->block_size + 2, PRM->zMin - PRM->delta, PRM->zMax0, i, j, PRM->zMin - PRM->delta);

	starpu_vector_data_register(&phiv_handle, STARPU_MAIN_RAM, (uintptr_t)ABC->phiv[i_block * PRM->n_blocks_x + j_block].base_ptr,
	                             ABC->phiv[i_block * PRM->n_blocks_x + j_block].size, sizeof(double));


    	starpu_task_insert(&intermediates_cl,
    			   STARPU_W, phiv_handle,
    			   STARPU_R, k2ly0_handle, STARPU_R, k2ly2_handle, STARPU_R, mu0_handle, STARPU_R, mu2_handle,
    			   STARPU_R, kap0_handle, STARPU_R, kap2_handle, STARPU_R, rho0_handle, STARPU_R, rho2_handle,
    			   STARPU_R, dumpx_handle, STARPU_R, dumpx2_handle, STARPU_R, dumpy_handle, STARPU_R, dumpy2_handle, STARPU_R, dumpz_handle, STARPU_R, dumpz2_handle,
    			   STARPU_R, alphax_handle, STARPU_R, alphax2_handle, STARPU_R, alphay_handle, STARPU_R, alphay2_handle, STARPU_R, alphaz_handle, STARPU_R, alphaz2_handle,
    			   STARPU_R, kappax_handle, STARPU_R, kappax2_handle, STARPU_R, kappay_handle, STARPU_R, kappay2_handle, STARPU_R, kappaz_handle, STARPU_R, kappaz2_handle,
    			   STARPU_R, ipml_handle, STARPU_R, v0_x_handle, STARPU_R, v0_y_handle, STARPU_R, v0_z_handle,
    			   STARPU_VALUE, &i_block, sizeof(i_block),
    			   STARPU_VALUE, &j_block, sizeof(j_block),
    			   STARPU_VALUE, &first_npml, sizeof(first_npml),
    			   STARPU_VALUE, PRM, sizeof(*PRM),
    			   0);
      }
    }
		starpu_task_wait_for_all();
		//Lets Dump phiv.xx
		/*FILE* f = fopen("/tmp/dump_ondes_sp", "w");
		int imp, jmp, k, i, j, place, *ipml;
		for (int i_block = 0; i_block < PRM->n_blocks_y; i_block++) {
			for (int j_block = 0; j_block < PRM->n_blocks_x; j_block++) {
				for(imp = -1 + PRM->block_size * i_block; imp < PRM->block_size * (i_block+1); imp++ ){
					i = ivector_access(PRM->imp2i_array, -1, PRM->mpmx + 2, imp);
					for(jmp = -1 + PRM->block_size * j_block; jmp < PRM->block_size * (j_block+1); jmp++ ){
						j = ivector_access(PRM->jmp2j_array, -1, PRM->mpmy + 2, jmp);
						for (k = PRM->zMin - PRM->delta; k <= PRM->zMax0; k++) {
							place = WhereAmI(i, j, k, *PRM);
							double ret = -2;
							if ((place == OUTSIDE) || (place == LIMIT)) {
							  continue;
							}
							if (place == ABSORBINGLAYER || place == FREEABS) {
								ipml = &i3access(ABC->ipml, -1, PRM->mpmx+2, -1, PRM->mpmy+2, PRM->zMin-PRM->delta, PRM->zMax0, imp, jmp, k);
								ret = ivector_access(ABC->phiv[i_block * PRM->n_blocks_x + j_block].xy, 1, 1000, *ipml);
							}
							fprintf(f, "%d,%d,%d,%lf\n", imp, jmp, k, ret);
						}
					}
				}
			}
		}
		fclose(f);*/


    //// loop compute stress
		printf("deu %d blocos\n", n_blocks_x);
    for (i_block = 0; i_block < n_blocks_y; i_block++) {
      for (j_block = 0; j_block < n_blocks_x; j_block++) {

    	i = i_block*PRM->block_size;
    	j = j_block*PRM->block_size;
    
    	starpu_vector_data_register(&ipml_handle, STARPU_MAIN_RAM, (uintptr_t)ABC->ipml, (PRM->mpmx+3)*(PRM->mpmy+3)*(PRM->zMax0 - (PRM->zMin - PRM->delta)), sizeof(ABC->ipml[0]));
    
    	long int first_npml = i3access(ABC->ipml, -1, PRM->block_size + 2, -1, PRM->block_size + 2, PRM->zMin - PRM->delta, PRM->zMax0, i, j, PRM->zMin - PRM->delta);

	starpu_vector_data_register(&phiv_handle, STARPU_MAIN_RAM, (uintptr_t)ABC->phiv[i_block * PRM->n_blocks_x + j_block].base_ptr,
	                             ABC->phiv[i_block * PRM->n_blocks_x + j_block].size, sizeof(double));

    	starpu_data_handle_t block_xx, block_yy, block_zz, block_xy, block_xz, block_yz;
    	block_xx = starpu_data_get_sub_data(t0_xx_handle, 2, i_block, j_block);
    	block_yy = starpu_data_get_sub_data(t0_yy_handle, 2, i_block, j_block);
    	block_zz = starpu_data_get_sub_data(t0_zz_handle, 2, i_block, j_block);
    	block_xy = starpu_data_get_sub_data(t0_xy_handle, 2, i_block, j_block);
    	block_xz = starpu_data_get_sub_data(t0_xz_handle, 2, i_block, j_block);
    	block_yz = starpu_data_get_sub_data(t0_yz_handle, 2, i_block, j_block);
    
    	starpu_task_insert(&stress_cl,
    			   STARPU_W, block_xx, STARPU_W, block_yy, STARPU_RW, block_zz, STARPU_W, block_xy, STARPU_RW, block_xz, STARPU_RW, block_yz,
    			   STARPU_R, k2ly0_handle, STARPU_R, k2ly2_handle, STARPU_R, mu0_handle, STARPU_R, mu2_handle, STARPU_R, kap0_handle,
    			   STARPU_R, kappax_handle, STARPU_R, kappax2_handle, STARPU_R, kappay_handle, STARPU_R, kappay2_handle,
			   STARPU_R, kappaz_handle, STARPU_R, kappaz2_handle, STARPU_R, phiv_handle,
    			   STARPU_R, ipml_handle, STARPU_R, v0_x_handle, STARPU_R, v0_y_handle, STARPU_R, v0_z_handle,
    			   STARPU_VALUE, &i, sizeof(i),
    			   STARPU_VALUE, &j, sizeof(j),
    			   STARPU_VALUE, &first_npml, sizeof(first_npml),
    			   STARPU_VALUE, PRM, sizeof(*PRM),
    			   0);
      }
    }

    // loop compute velo
    //////________________________________________
    ///// ATENCAO
    /////
    ///// GERALMENTE COMECA COM I = 1, MAS NOS BLOCOS NAO DEVE SER ASSIM
    /////
    ////__________________________________________
    for (i_block = 1; i_block <= n_blocks_y; i_block++) {
      for (j_block = 1; j_block <= n_blocks_x; j_block++) {
    
    	i = i_block*PRM->block_size;
    	j = j_block*PRM->block_size;
    
    	starpu_vector_data_register(&ipml_handle, STARPU_MAIN_RAM, (uintptr_t)ABC->ipml, (PRM->mpmx+3)*(PRM->mpmy+3)*(PRM->zMax0 - (PRM->zMin - PRM->delta)), sizeof(ABC->ipml[0]));
    
    	long int first_npml = i3access(ABC->ipml, -1, PRM->block_size + 2, -1, PRM->block_size + 2, PRM->zMin - PRM->delta, PRM->zMax0, i, j, PRM->zMin - PRM->delta);

	starpu_vector_data_register(&phit_handle, STARPU_MAIN_RAM, (uintptr_t)ABC->phit[i_block * PRM->n_blocks_x + j_block].base_ptr,
	                             ABC->phit[i_block * PRM->n_blocks_x + j_block].size, sizeof(double));
    
    	starpu_data_handle_t block_v0_x, block_v0_y, block_v0_z;
    	block_v0_x = starpu_data_get_sub_data(v0_x_handle, 2, i, j);
    	block_v0_y = starpu_data_get_sub_data(v0_y_handle, 2, i, j);
    	block_v0_z = starpu_data_get_sub_data(v0_z_handle, 2, i, j);
    
    	starpu_task_insert(&velo_cl,
    			   STARPU_RW, block_v0_x, STARPU_RW, block_v0_y, STARPU_RW, block_v0_z, STARPU_W, phit_handle,
    			   STARPU_R, k2ly0_handle, STARPU_R, k2ly2_handle, STARPU_R, rho0_handle, STARPU_R, rho2_handle,
    			   STARPU_R, mu0_handle, STARPU_R, mu2_handle, STARPU_R, kap0_handle, STARPU_R, kap2_handle,
    			   STARPU_R, dumpx_handle, STARPU_R, dumpx2_handle, STARPU_R, dumpy_handle, STARPU_R, dumpy2_handle, STARPU_R, dumpz_handle, STARPU_R, dumpz2_handle,
    			   STARPU_R, alphax_handle, STARPU_R, alphax2_handle, STARPU_R, alphay_handle, STARPU_R, alphay2_handle, STARPU_R, alphaz_handle, STARPU_R, alphaz2_handle,
    			   STARPU_R, kappax_handle, STARPU_R, kappax2_handle, STARPU_R, kappay_handle, STARPU_R, kappay2_handle, STARPU_R, kappaz_handle, STARPU_R, kappaz2_handle,
    			   STARPU_R, ipml_handle, STARPU_R, t0_xx_handle, STARPU_R, t0_yy_handle, STARPU_R, t0_zz_handle,
    			   STARPU_R, t0_xy_handle, STARPU_R, t0_xz_handle, STARPU_R, t0_yz_handle, STARPU_R, src_fx_handle, STARPU_R, src_fy_handle, STARPU_R, src_fz_handle,
    			   STARPU_VALUE, &i, sizeof(i),
    			   STARPU_VALUE, &j, sizeof(j),
    			   STARPU_VALUE, &first_npml, sizeof(first_npml),
    			   STARPU_VALUE, PRM, sizeof(*PRM),
    			   0);
      }
    }
    starpu_task_wait_for_all();
  }
}
