#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <assert.h>
#include <errno.h>

#include "../include/new_nrutil.h"
#include "../include/struct.h"

#define MALLOC_CHECK_ 1

double *myd3tensor0(long nrl, long nrh, long ncl, long nch, long ndl,
		      long ndh)
{
    double *t;
    long i, j, k;
    t = d3tensor(nrl, nrh, ncl, nch, ndl, ndh);
    /* we manually initiate to zero (calloc don't work on double)  */
    for (i = 0; i < nrh-nrl; i++) {
	for (j = 0; j < nch-ncl; j++) {
	    for (k = 0; k <= ndh-ndl; k++) {
		t[i*(nch-ncl)*(ndh-ndl)+j*(ndh-ndl)+k] = 0.;
	    }
	}
    }
    return t;
}

double *mydmatrix0(long nrl, long nrh, long ncl, long nch)
{
    long int i, j;
    double *mat;
    mat = dmatrix(nrl, nrh, ncl, nch);
    /* we manually initiate to zero since calloc do not work for doubles */
    for (i = 0; i < nrh-nrl; i++) {
	for (j = 0; j < nch-ncl; j++) {
	  mat[i*(nrh-nrl)+j] = 0.;
	}
    }
    return mat;
}

double *mydvector0(long nl, long nh)
{
    long int i;
    double *v;
    v = dvector(nl, nh);
    /* we manually initiate to zero since calloc do not work for doubles */
    for (i = 0; i < nh-nl; i++) {
	v[i] = 0.;
    }
    return v;
}

double *mydvectorRealloc(double *v, long nl, long nh)
{
    double *new;
    new = v + nl - NR_END;
    new = realloc(new, (size_t) ((nh - nl + 1 + NR_END) * sizeof(double)));
    if (new == NULL) {
	fprintf(stderr, "error in mydvectorRealloc");
    }
    return new;
}

void nrerror(char error_text[])
/* Numerical Recipes standard error handler */
{
    fprintf(stderr, "Numerical Recipes run-time error...\n");
    fprintf(stderr, "%s\n", error_text);
    fprintf(stderr, "...now exiting to system...\n");
    exit(1);
}

float *vector(long nl, long nh)
/* allocate a float vector with subscript range v[nl..nh] */
{
    float *v;

    v = malloc((size_t) ((nh - nl + 1 + NR_END) * sizeof(float)));
    if (!v)
	nrerror("allocation failure in vector()");
    return v;
}

int *ivector(long nl, long nh)
/* allocate an int vector with subscript range v[nl..nh] */
{
    int *v;

    v = malloc((size_t) ((nh - nl + 1 + NR_END) * sizeof(int)));
    if (!v)
	nrerror("allocation failure in ivector()");
    return v;
}

unsigned char *cvector(long nl, long nh)
/* allocate an unsigned char vector with subscript range v[nl..nh] */
{
    unsigned char *v;

    v = malloc((size_t) ((nh - nl + 1 + NR_END) * sizeof(unsigned char)));
    if (!v)
	nrerror("allocation failure in cvector()");
    return v;
}

unsigned long *lvector(long nl, long nh)
/* allocate an unsigned long vector with subscript range v[nl..nh] */
{
    unsigned long *v;

    v = malloc((size_t) ((nh - nl + 1 + NR_END) * sizeof(long)));
    if (!v)
	nrerror("allocation failure in lvector()");
    return v;
}

double *dvector(const long nl, const long nh)
/* allocate a double vector with subscript range v[nl..nh] */
{
    double *v;

    v = malloc((size_t) ((nh - nl + 1 + NR_END) * sizeof(double)));
    if (!v)
	nrerror("allocation failure in dvector()");
    return v;
}

double *dvector0(long nl, long nh)
/* allocate a double vector with subscript range v[nl..nh] and 
initiate to 0 */
{
    double *v;
    long i;
    v = calloc((nh - nl + 1 + NR_END), (size_t) (sizeof(double)));
    if (!v)
	nrerror("allocation failure in dvector()");
    return v;
}


float *matrix(long nrl, long nrh, long ncl, long nch)
/* allocate a float matrix with subscript range m[nrl..nrh][ncl..nch] */
{
    long i, nrow = nrh - nrl + 1, ncol = nch - ncl + 1;
    float *m;

    /* allocate pointers to rows */
    m = (float *) malloc((size_t) ((nrow*ncol) * sizeof(float)));
    if (!m)
	nrerror("allocation failure 1 in matrix()");

    /* return pointer to array of pointers to rows */
    return m;
}

double *dmatrix(long nrl, long nrh, long ncl, long nch)
/* allocate a double matrix with subscript range m[nrl..nrh][ncl..nch] */
{
    long i, nrow = nrh - nrl + 1, ncol = nch - ncl + 1;
    double *m;

    /* allocate pointers to rows */
    /* m=(double **) malloc((size_t)((nrow+NR_END)*sizeof(double*))); */
    m = malloc((size_t) ((nrow*ncol) * sizeof(double)));

    if (!m)
	nrerror("allocation failure 1 in matrix()");

    /* return pointer to array of pointers to rows */
    return m;
}

double *dmatrix0(long nrl, long nrh, long ncl, long nch)
/* allocate a double matrix with subscript range m[nrl..nrh][ncl..nch] 
and initiate 0*/
{
  long i, j, nrow = nrh - nrl + 1, ncol = nch - ncl + 1;
    double *m;

    /* allocate pointers to rows */
    m = (double *) calloc((nrow*ncol), (size_t) (sizeof(double)));
    if (!m)
	nrerror("allocation failure 1 in matrix()");

    for (i = 0; i < nrow; i++) {
      for (j = 0; j < ncol; j++) {
	m[(i)*(nch-ncl)+j] = 0.;
      }
    }
    
    /* return pointer to array of pointers to rows */
    return m;
}


int *imatrix(long nrl, long nrh, long ncl, long nch)
/* allocate a int matrix with subscript range m[nrl..nrh][ncl..nch] */
{
    long i, nrow = nrh - nrl + 1, ncol = nch - ncl + 1;
    int *m;

    /* allocate pointers to rows */
    m = (int *) malloc((size_t) ((nrow*ncol) * sizeof(int)));
    if (!m)
	nrerror("allocation failure 1 in matrix()");

    /* return pointer to array of pointers to rows */
    return m;
}

float *f3tensor(long nrl, long nrh, long ncl, long nch, long ndl,
		  long ndh)
/* allocate a float 3tensor with range t[nrl..nrh][ncl..nch][ndl..ndh] */
{
    long i, j, nrow = nrh - nrl + 1, ncol = nch - ncl + 1, ndep =
	ndh - ndl + 1;
    float *t;

    /* allocate pointers to pointers to rows */
    t = (float *) malloc((size_t) ((nrow*ncol*ndep) * sizeof(float)));
    if (!t)
	nrerror("allocation failure 1 in f3tensor()");

    /* return pointer to array of pointers to rows */
    return t;
}

double *d3tensor(long nrl, long nrh, long ncl, long nch, long ndl,
		   long ndh)
/* allocate a double 3tensor with range t[nrl..nrh][ncl..nch][ndl..ndh] */
{
    long i, j, nrow = nrh - nrl + 1, ncol = nch - ncl + 1, ndep =
	ndh - ndl + 1;
    double *t;

    /* allocate pointers to pointers to rows */
/*         t=(double ***) malloc((size_t)((nrow+NR_END)*sizeof(double**))); */
    t = malloc((size_t) ((nrow*ncol*ndep) * sizeof(double)));
    if (!t)
	nrerror("allocation failure 1 in d3tensor()");

    /* return pointer to array of pointers to rows */
    return t;
}

int *i3tensor(long nrl, long nrh, long ncl, long nch, long ndl, long ndh)
/* allocate a double 3tensor with range t[nrl..nrh][ncl..nch][ndl..ndh] */
{
    long i, j, nrow = nrh - nrl + 1, ncol = nch - ncl + 1, ndep =
	ndh - ndl + 1;
    int *t;

    /* allocate pointers to pointers to rows */
    t = (int *) malloc((size_t) ((nrow*ncol*ndep) * sizeof(int)));
    if (!t)
	nrerror("allocation failure 1 in i3tensor()");

    /* return pointer to array of pointers to rows */
    return t;
}

double *d3tensor0(long nrl, long nrh, long ncl, long nch, long ndl,
		    long ndh)
/* allocate a double 3tensor with range t[nrl..nrh][ncl..nch][ndl..ndh] 
and initiate 0*/
{
    long i, j, k, nrow = nrh - nrl + 1, ncol = nch - ncl + 1, ndep =
	ndh - ndl + 1;
    double *t;

    /* allocate pointers to pointers to rows */
    t = (double *) calloc((nrow*ncol*ndep), (size_t) (sizeof(double)));
    if (!t)
	nrerror("allocation failure 1 in f3tensor()");

    for (i = 0; i < nrow; i++) {
      for (j = 0; j < ncol; j++) {
	for (k = 0; k < ndep; k++) {
	  t[i*(nch-ncl)*(ndh-ndl)+j*(ndh-ndl)+k] = 0.;
	}
      }
    }
    /* return pointer to array of pointers to rows */
    return t;
}
