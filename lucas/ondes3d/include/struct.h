#ifndef STRUCT_H_
#define STRUCT_H_

/* constants ========================================================================*/
static const double un = 1.;
static const double NPOWER = 2.;
static const double f0 = 1.;
static const int DELTA_VAL = 10;
static const double REFLECT = 0.001;


/* STRUCTURES  ========================================================================*/
/* Description : if "typedef struct{ } name;" we need the use of volatile attribute
 * WARNING about "typedef struct{ } name;"
 * typedef just make an alias;
 * so by saying typedef struct{ } name : you create a global variable "name" each time you load struct.h
 * name el; : el is the global variable "name"
 * name *ptr; : ptr is a pointer on this global variable (in fact it should not be really defined );
 *
 * so for each file, you have a global variable "name" which should be volatile
 * (because others functions in others file can modify this variable "name"
 *
 * better approach is to use : struct name{ }; where you only define a structure and not a name
 *
 * Infos available on :
 * http://testincprogrammingandembeddedsystem.com/volatile.html
 * [french] http://cpp.developpez.com/cours/cpp/?page=page_5 (3.1.6 and 3.2)
 */

/*** T0 ***/
struct STRESS {
    double *xx;
    double *yy;
    double *zz;
    double *xy;
    double *xz;
    double *yz;
};

/*** V0 ***/
struct VELOCITY {
    double *x;
    double *y;
    double *z;
};

/*** SRC ***/
enum typeSource { VELO, HISTFILE };
struct SOURCE {
    int iSrc;			/* number of sources */
    int *insrc;			/* say if source is inside the domain */

    int ixhypo0;
    int iyhypo0;
    int izhypo0;		/* indexes of sources' hypocenter */
    double xhypo0;
    double yhypo0;
    double zhypo0;		/* real coordinates hypocenter */

    int *ixhypo;
    int *iyhypo;
    int *izhypo;		/* coordinates indexes of each source */


    /* For HISTFILE */
    double mo;
    double mw;
    double dsbiem;		/* ds of sources */
    double dtbiem;		/* time step of sources */
    int iDur;			/* number of time step for source duration */

    double *strike;
    double *dip;
    double *rake;		/* source orientation */
    double *slip;
    double *xweight;
    double *yweight;
    double *zweight;
    double *vel;
    double *fx;
    double *fy;
    double *fz;

    /*   /\* to test if it speed up computation ? *\/
       double  *pxx;  double  *pyy;   double  *pzz;
       double   *pxy;  double  *pyz;   double  *pxz;  */
};

/*** PARAMETERS PRM ****/

struct PARAMETERS {
    double dt;			/* time step */
    double ds;			/* spacial step */

    int nDim;			/* spatial approximation order */
    int tMax;			/* number of time step */

    /* global of domain */
    int xMin;
    int xMax;
    int yMin;
    int yMax;
    int zMin;
    int zMax;
    int delta;			/* number of absorbing cells in the absorbing layer */
    int zMax0;			/* real maximum of z (absorbing layer or FreeSurface included ) */

    double x0, y0, z0;		/* [GEOLOGICAL] coordinates of the south-east lower corner */

    /* local domain */
    int me;			/* index of the CPU */
    int coords[2];		/* position of the CPU */
    int px;
    int py;			/* Limit of CPU's coordinates in x and y directions */
    int mpmx;
    int mpmy;			/* local computed domain limit */
    long int nmaxx;
    long int nmaxy;		/* number of fields in each direction  */

    /* local <-> global */
    int *imp2i_array;
    int *jmp2j_array;		/* local -> global */
    int *i2imp_array;
    int *j2jmp_array;		/* global -> local */

    int *i2icpu_array;
    int *j2jcpu_array;		/* find the right cpu */
    int *mpmx_tab;
    int *mpmy_tab;		/* only useful for outputs? */

    /* others */
    double pi;			/* necessary? */
    char dir[50];
    char fsrcMap[50];
    char fsrcHist[50];
    char fstatMap[50];
    char fgeo[50];

    /* lucas */
    int block_size;
    int n_blocks_y;
    int n_blocks_x;
    int depth;
};

/*** MEDIUM MDM ***/
enum typeSurface { ABSORBING, FREE };
enum typeModel { LAYER, GEOLOGICAL };
enum typeInterface { USUAL, KMINTERFACE };

struct MEDIUM {
  /** to find the index medium **/
    /* model == GEOLOGICAL */
    int *imed;		/* medium ( model == GEOLOGICAL) */
    int numVoid;		/* number of medium for the air */
    int numSea;			/* number of medium for the sea */
    char **name_mat;

    /* model == LAYER */
    double *laydep;		/* Layer depth */
    int nLayer;
    int *k2ly0;			/* k to layer index ( floor of cell ) */

    double *laydep2;		/* Layer depth (z+ds/2)  */
    int nLayer2;
    int *k2ly2;			/* k to layer index ( ceiling of cell +ds/2 in z ) */
    double *rho2, *mu2, *kap2;	/* contains values on the interface between 2 layers */

    /* fields */
    double *rho0, *mu0, *kap0;	/* contains value in the layer */


};

struct phiv_s{
  double* base_ptr;
  int size;
  int offset;
  double *xx;
  double *yy;
  double *zz;		/* for CPML; contributions to txx;tyy; tzz */
  double *xy;
  double *yx;		/* contributions to txy */
  double *xz;
  double *zx;		/* contributions to txz */
  double *yz;
  double *zy;		/* contributions to tyz */
};

#define COMPUTE_ADDRESS_PHIV_S(phiv) phiv.xx = phiv.base_ptr;\
                                     phiv.yy = phiv.base_ptr + phiv.offset;\
                                     phiv.zz = phiv.base_ptr + 2*phiv.offset;\
                                     phiv.xy = phiv.base_ptr + 3*phiv.offset;\
                                     phiv.yx = phiv.base_ptr + 4*phiv.offset;\
                                     phiv.xz = phiv.base_ptr + 5*phiv.offset;\
                                     phiv.zx = phiv.base_ptr + 6*phiv.offset;\
                                     phiv.yz = phiv.base_ptr + 7*phiv.offset;\
                                     phiv.zy = phiv.base_ptr + 8*phiv.offset;

struct phit_s{
  double* base_ptr;
  int size;
  int offset;

  double *xxx;
  double *xyy;
  double *xzz;		/* for CPML; contributions to vx */
  double *xyx;
  double *yyy;
  double *yzz;		/* contributions to vy */
  double *xzx;
  double *yzy;
  double *zzz;		/* contributions to vz */
};

#define COMPUTE_ADDRESS_PHIT_S(phit) phiv.xxx = phit.base_ptr;\
                                     phiv.xyy = phit.base_ptr + phit.offset;\
                                     phiv.xzz = phit.base_ptr + 2*phit.offset;\
                                     phiv.xyx = phit.base_ptr + 3*phit.offset;\
                                     phiv.yyy = phit.base_ptr + 4*phit.offset;\
                                     phiv.yzz = phit.base_ptr + 5*phit.offset;\
                                     phiv.xzx = phit.base_ptr + 6*phit.offset;\
                                     phiv.yzy = phit.base_ptr + 7*phit.offset;\
                                     phiv.zzz = phit.base_ptr + 8*phit.offset;



/*** Absorbing Boundary Condition ***/
enum typePML { PML, CPML };
struct ABSORBING_BOUNDARY_CONDITION {
    double dump0;		/* parameters */
    double nPower;
    double reflect;
    int npmlv;			/* numbers of cell in ABC */
    int npmlt;
    int *ipml;		/* index in the PML */

    double *dumpx;
    double *dumpy;
    double *dumpz;
    double *dumpx2;
    double *dumpy2;
    double *dumpz2;

    /* CPML */
    double fd;
    double alpha0;
    double kappa0;
    struct phiv_s *phiv; //List of phiv blocks
    struct phit_s *phit; //List of phit blocks
  
    double *kappax;
    double *kappay;
    double *kappaz;		/* modifications of the derivative */
    double *kappax2;
    double *kappay2;
    double *kappaz2;
    double *alphax;
    double *alphay;
    double *alphaz;
    double *alphax2;
    double *alphay2;
    double *alphaz2;

    /* PML */
    double *txxx;
    double *txxy;
    double *txxz;
    double *tyyx;
    double *tyyy;
    double *tyyz;
    double *tzzx;
    double *tzzy;
    double *tzzz;
    double *txyx;
    double *txyy;
    double *txzx;
    double *txzz;
    double *tyzy;
    double *tyzz;

    double *vxx;
    double *vxy;
    double *vxz;
    double *vyx;
    double *vyy;
    double *vyz;
    double *vzx;
    double *vzy;
    double *vzz;
};


/*** ANELASTICITY  ANL ***/
enum typeAnelas { ELASTIC, DAYandBRADLEY, ANOTHER, KRISTEKandMOCZO };
struct ANELASTICITY {

    /* anelastic coefficients */
    double *Qp0;
    double *Qs0;		/* anelastic coeffs */
    double *Qp2;
    double *Qs2;		/* beetween layers */

    /* ANOTHER */
    double *q0;
    double *q2;
    double *amp;		/*  */
    double *amp2;		/* amp beetween layers */

    /* DAY and BRADLEY */
    double ***ksixx;
    double ***ksiyy;
    double ***ksizz;
    double ***ksixy;
    double ***ksixz;
    double ***ksiyz;

    double ***fxx;
    double ***fyy;
    double ***fzz;
    double ***fxy;
    double ***fxz;
    double ***fyz;
    double w0;
    double tm;
    double tM;

    /* KRISTEK and MOCZO */
    double ***ksilxx;
    double ***ksilyy;
    double ***ksilzz;
    double ***ksilxy;
    double ***ksilxz;
    double ***ksilyz;

    double **ylmu;		/* first index 1:4 second 0:NLAYER-1 */
    double **ylmu2;
    double **ylkap;
    double **ylkap2;

    double *wl;			/* index 1:4 */
    double wmin;
    double wmax;
};


/*** COMMUNICATION_DIRECTION : NORTH SOUTH EAST WEST ***/
/* enum typeComm {PERSISTENT, BLOCKING}; */
struct COMM_DIRECTION {
    long int nmax;		/* number of fields to exchange */
    int rank;			/* cpu index of the neighbor */

    /* Send */
    int iMinS;
    int iMaxS;
    int jMinS;
    int jMaxS;
    double *bufV0S;		/* Velocity */
    int channelV0S;
    double *bufT0S;		/* Stress */
    int channelT0S;
    double *bufKsiS;		/* Kristek and Moczo */
    int channelKsiS;

    /* Receive */
    int iMinR;
    int iMaxR;
    int jMinR;
    int jMaxR;
    double *bufV0R;		/* Velocity */
    int channelV0R;
    double *bufT0R;		/* Stress */
    int channelT0R;
    double *bufKsiR;		/* Kristek and Moczo */
    int channelKsiR;

};


/*** OUTPUTS  OUT ***/
enum typeSnapshot { OVELO, ODISPL, OBOTH };
struct OUTPUTS {
    /* seismogramms */
    int iObs;			/* number of stations */
    int *ixobs;
    int *iyobs;
    int *izobs;			/* indexes of the stations */
    double *xobs;
    double *yobs;
    double *zobs;		/* coordinates of the stations */
    int *nobs;
    int *ista;			/* to verify that the station is inside the domain */

    double *xobswt;
    double *yobswt;
    double *zobswt;		/* weight on the stations */

    int *mapping_seis;		/* store the CPU index containing the station.
				 * first index = index of station,
				 * second index = 1-3 : v0, 4-9 : t0
				 */
    double *seis_output;	/* contains the seismogramms (output) */
    double *seis_buff;		/* buffer to communicate seismograms */

    /* velocity planes */
    int i0, j0, k0;		/* coordinates of the output plane */
    double *snapBuff;
    double **Vxglobal;
    double **Vyglobal;
    double **Vzglobal;
    double **PGVlocal;		// PGV
    double **PGVglobal;		//PGV
    long int test_size;
    double *Uxy, *Uyz, *Uxz;	/* displacement, for snapType==SNAPDISPL */
    /* partition domain related */
    long int total_prec_x;
    long int total_prec_y;

};

/* other enums ========================================================================*/

enum typePlace { REGULAR, ABSORBINGLAYER, FREESURFACE, FREEABS, LIMIT,
	OUTSIDE };
/* NB : FREEABS = FREESURFACE && ABSORBINGLAYER */

enum typeSea { NOSEA, SEA };	/* only used in Geological models */

enum typeDiff { ORDER2, ORDER4, CURV2 };



#endif				/* STRUCT_H_ */
