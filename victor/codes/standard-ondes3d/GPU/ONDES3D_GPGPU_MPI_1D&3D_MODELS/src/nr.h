#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define NR_END 1
#define FREE_ARG char*

void nrerror(char error_text[]);

float *vector(long nl, long nh);

int *ivector(long nl, long nh);

unsigned char *cvector(long nl, long nh);

unsigned long *lvector(long nl, long nh);

float *fvector(long nl, long nh);

double *dvector(long nl, long nh);

float *fvector0(long nl, long nh);

float **matrix(long nrl, long nrh, long ncl, long nch);

float **dmatrix(long nrl, long nrh, long ncl, long nch);

float **dmatrix0(long nrl, long nrh, long ncl, long nch);

int **imatrix(long nrl, long nrh, long ncl, long nch);

float **submatrix(float **a, long oldrl, long oldrh, long oldcl, long oldch, long newrl, long newcl);

float **convert_matrix(float *a, long nrl, long nrh, long ncl, long nch);

float ***f3tensor(long nrl, long nrh, long ncl, long nch, long ndl, long ndh);

float ***d3tensor(long nrl, long nrh, long ncl, long nch, long ndl, long ndh);

int ***i3tensor(long nrl, long nrh, long ncl, long nch, long ndl, long ndh);

float ***d3tensor0(long nrl, long nrh, long ncl, long nch, long ndl, long ndh);

void free_vector(float *v, long nl, long nh);

void free_ivector(int *v, long nl, long nh);

void free_cvector(unsigned char *v, long nl, long nh);

void free_lvector(unsigned long *v, long nl, long nh);

void free_fvector(float *v, long nl, long nh);

void free_dvector(double *v, long nl, long nh);

void free_fvector0(float *v, long nl, long nh);

void free_matrix(float **m, long nrl, long nrh, long ncl, long nch);

void free_dmatrix(float **m, long nrl, long nrh, long ncl, long nch);

void free_dmatrix0(float **m, long nrl, long nrh, long ncl, long nch);

void free_imatrix(int **m, long nrl, long nrh, long ncl, long nch);

void free_submatrix(float **b, long nrl, long nrh, long ncl, long nch);

void free_convert_matrix(float **b, long nrl, long nrh, long ncl, long nch);

void free_f3tensor(float ***t, long nrl, long nrh, long ncl, long nch, long ndl, long ndh);

void free_d3tensor(float ***t, long nrl, long nrh, long ncl, long nch, long ndl, long ndh);

void free_d3tensor0(float ***t, long nrl, long nrh, long ncl, long nch, long ndl, long ndh);

float ***d3tensor_fix(long nrl, long nrh, long ncl, long nch, long ndl, long ndh, float ***t);

int my_float2int(float x);

float hardrock(float z);

float softrock(float z);

float my_second();

float radxx(float strike, float dip, float rake);

float radyy(float strike, float dip, float rake);

float radzz(float strike, float dip, float rake);

float radxy(float strike, float dip, float rake);

float radyz(float strike, float dip, float rake);

float radxz(float strike, float dip, float rake);

double dradxx(double strike, double dip, double rake);

double dradyy(double strike, double dip, double rake);

double dradzz(double strike, double dip, double rake);

double dradxy(double strike, double dip, double rake);

double dradyz(double strike, double dip, double rake);

double dradxz(double strike, double dip, double rake);

