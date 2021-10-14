#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <assert.h>
#include <errno.h>

#include "../include/nrutil.h"
#include "../include/struct.h"

#define MALLOC_CHECK_ 1

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
