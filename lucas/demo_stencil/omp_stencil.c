#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define gflops(x, y, steps) (x*y/1e9)*11*steps
#define perf(x, y, steps, us) (x*y/us/1e3)*11*steps

static int STENCIL_SIZE_X;
static int STENCIL_SIZE_Y;

/** number of buffers for N-buffering; should be at least 2 */
#define STENCIL_NBUFFERS 2

/** conduction coeff used in computation */
static const double alpha = 0.02;

/** threshold for convergence */
static const double epsilon = 0.001;//0.0001;

/** max number of steps */
static const int stencil_max_steps = 10000;

static double ***values;

/** latest computed buffer */
static int current_buffer = 0;

/** init global variables */
void global_init(int x, int y) {

  int i,j;
  STENCIL_SIZE_X = x;
  STENCIL_SIZE_Y = y;

  values = malloc(STENCIL_NBUFFERS*sizeof(double**));
  for (i = 0; i< STENCIL_NBUFFERS; i++) {
    values[i] = malloc(STENCIL_SIZE_X*sizeof(double *));
    for (j = 0; j < STENCIL_SIZE_X; j++) {
      values[i][j] = (double *)malloc(STENCIL_SIZE_Y*sizeof(double));
    }
  }
}

void global_free() {

  int i, j;
  for (i = 0; i < STENCIL_NBUFFERS; i++) {
    for (j = 0; j < STENCIL_SIZE_X; j++) {
      free(values[i][j]);
    }
    free(values[i]);
  }
  free(values);
}

/** init stencil values to 0, borders to non-zero */
static void stencil_init(void)
{
  int b, x, y;
  for(b = 0; b < STENCIL_NBUFFERS; b++)
    {
      for(x = 0; x < STENCIL_SIZE_X; x++)
	{
	  for(y = 0; y < STENCIL_SIZE_Y; y++)
	    {
	      values[b][x][y] = 0.0;
	    }
	}
      for(x = 0; x < STENCIL_SIZE_X; x++)
	{
	  values[b][x][0] = x;
	  values[b][x][STENCIL_SIZE_Y - 1] = STENCIL_SIZE_X - x;
	}
      for(y = 0; y < STENCIL_SIZE_Y; y++)
	{
	  values[b][0][y] = y;
	  values[b][STENCIL_SIZE_X - 1][y] = STENCIL_SIZE_Y - y;
	}
    }
}

/** display a (part of) the stencil values */
static void stencil_display(int b, int x0, int x1, int y0, int y1)
{
  int x, y;
  for(y = y0; y <= y1; y++)
    {
      for(x = x0; x <= x1; x++)
	{
	  printf("%8.5g ", values[b][x][y]);
	}
      printf("\n");
    }
}

/** compute the next stencil step */
static void stencil_step(void)
{
  int prev_buffer = current_buffer;
  int next_buffer = (current_buffer + 1) % STENCIL_NBUFFERS;
  int x, y;

#pragma omp parallel for collapse(2) default(none) private(x, y) shared(STENCIL_SIZE_X, STENCIL_SIZE_Y, values, prev_buffer, next_buffer)
  for(x = 1; x < STENCIL_SIZE_X - 1; x++)
    {
      for(y = 1; y < STENCIL_SIZE_Y - 1; y++)
	{
	  values[next_buffer][x][y] =
	    alpha * values[prev_buffer][x - 1][y] +
	    alpha * values[prev_buffer][x + 1][y] +
	    alpha * values[prev_buffer][x][y - 1] +
	    alpha * values[prev_buffer][x][y + 1] +
	    (1.0 - 4.0 * alpha) * values[prev_buffer][x][y];
	}
    }
  current_buffer = next_buffer;
}

/** return 1 if computation has converged */
static int stencil_test_convergence(void)
{
  int prev_buffer = (current_buffer - 1 + STENCIL_NBUFFERS) % STENCIL_NBUFFERS;
  int x, y;
  for(x = 1; x < STENCIL_SIZE_X - 1; x++)
    {
      for(y = 1; y < STENCIL_SIZE_Y - 1; y++)
	{
	  if(fabs(values[prev_buffer][x][y] - values[current_buffer][x][y]) > epsilon)
	    return 0;
	}
    }
  return 1;
}

int main(int argc, char**argv)
{

  
  global_init(atoi(argv[1]), atoi(argv[2]));
  stencil_init();
  //stencil_display(current_buffer, 0, STENCIL_SIZE_X - 1, 0, STENCIL_SIZE_Y - 1);

  struct timespec t1, t2;
  clock_gettime(CLOCK_MONOTONIC, &t1);
  int s;
  for(s = 0; s < stencil_max_steps; s++)
    {
      stencil_step();
    }
  clock_gettime(CLOCK_MONOTONIC, &t2);
  const double t_usec = (t2.tv_sec - t1.tv_sec) * 1000000.0 + (t2.tv_nsec - t1.tv_nsec) / 1000.0;
  //printf("# time = %g usecs.\n", t_usec);
  //stencil_display(current_buffer, 0, STENCIL_SIZE_X - 1, 0, STENCIL_SIZE_Y - 1);

  /* printf("%.4f gflops in %.4f us\n", gflops(STENCIL_SIZE_X, STENCIL_SIZE_Y, s), t_usec);  */
  printf("%d;%d;%.4f gflops/s\n", STENCIL_SIZE_X, s, perf(STENCIL_SIZE_X, STENCIL_SIZE_Y, s, t_usec));

  global_free();

  return 0;
}

