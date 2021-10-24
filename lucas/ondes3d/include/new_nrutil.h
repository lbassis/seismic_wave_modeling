#define NR_END 0
#define FREE_ARG char*

#define i3access(p, nrl, nrh, ncl, nch, ndl, ndh, i, j, k)  p[(i-(nrl))*((nch)-(ncl)+1)*((ndh)-(ndl)+1)+(j-(ncl))*((ndh)-(ndl)+1)+(k-(ndl))]

#define imatrix_access(m, nrl, nrh, ncl, nch, i, j) m[(i-(nrl))*((nch)-(ncl)+1)+(j-(ncl))]

#define ivector_access(p, nl, nh, i)  p[(i) - (nl) + NR_END]
#define ivector_access(p, nl, nh, i)  p[(i) - (nl) + NR_END]
#define ivector_address(p, nl, nh, i)  p+i-nl+NR_END

static float sqrarg;
#define SQR(a) ((sqrarg=(a)) == 0.0 ? 0.0 : sqrarg*sqrarg)

static double dsqrarg;
#define DSQR(a) ((dsqrarg=(a)) == 0.0 ? 0.0 : dsqrarg*dsqrarg)

static double dmaxarg1, dmaxarg2;
#define DMAX(a,b) (dmaxarg1=(a),dmaxarg2=(b),(dmaxarg1) > (dmaxarg2) ?\
        (dmaxarg1) : (dmaxarg2))

static double dminarg1, dminarg2;
#define DMIN(a,b) (dminarg1=(a),dminarg2=(b),(dminarg1) < (dminarg2) ?\
        (dminarg1) : (dminarg2))

static float maxarg1, maxarg2;
#define FMAX(a,b) (maxarg1=(a),maxarg2=(b),(maxarg1) > (maxarg2) ?\
        (maxarg1) : (maxarg2))

static float minarg1, minarg2;
#define FMIN(a,b) (minarg1=(a),minarg2=(b),(minarg1) < (minarg2) ?\
        (minarg1) : (minarg2))

static long lmaxarg1, lmaxarg2;
#define LMAX(a,b) (lmaxarg1=(a),lmaxarg2=(b),(lmaxarg1) > (lmaxarg2) ?\
        (lmaxarg1) : (lmaxarg2))

static long lminarg1, lminarg2;
#define LMIN(a,b) (lminarg1=(a),lminarg2=(b),(lminarg1) < (lminarg2) ?\
        (lminarg1) : (lminarg2))

static int imaxarg1, imaxarg2;
#define IMAX(a,b) (imaxarg1=(a),imaxarg2=(b),(imaxarg1) > (imaxarg2) ?\
        (imaxarg1) : (imaxarg2))

static int iminarg1, iminarg2;
#define IMIN(a,b) (iminarg1=(a),iminarg2=(b),(iminarg1) < (iminarg2) ?\
        (iminarg1) : (iminarg2))

#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

#define MATRIX_POSITION(m,i,j,nrl,ncl,nch) m[(i-nrl)*(nch-ncl)+(j-ncl)]
#define TENSOR_POSITION(t,i,j,k,nrl,ncl,nch,ndl,ndh) t[(i-nrl)*(nch-ncl)*(ndh-ndl)+(j-ncl)*(ndh-ndl)+(k-ndl)]

double *myd3tensor0(long nrl, long nrh, long ncl, long nch, long ndl, long ndh);
double *mydmatrix0(long nrl, long nrh, long ncl, long nch);
double *mydvector0(long nl, long nh);
double *mydvectorRealloc(double *v, long nl, long nh);
void nrerror(char error_text[]);
float *vector(long nl, long nh);
int *ivector(long nl, long nh);
unsigned char *cvector(long nl, long nh);
unsigned long *lvector(long nl, long nh);
double *dvector(const long nl, const long nh);
double *dvector0(long nl, long nh);
float *matrix(long nrl, long nrh, long ncl, long nch);
double *dmatrix(long nrl, long nrh, long ncl, long nch);
double *dmatrix0(long nrl, long nrh, long ncl, long nch);
int *imatrix(long nrl, long nrh, long ncl, long nch);
float *f3tensor(long nrl, long nrh, long ncl, long nch, long ndl, long ndh);
double *d3tensor(long nrl, long nrh, long ncl, long nch, long ndl, long ndh);
int *i3tensor(long nrl, long nrh, long ncl, long nch, long ndl, long ndh);
double *d3tensor0(long nrl, long nrh, long ncl, long nch, long ndl, long ndh);
