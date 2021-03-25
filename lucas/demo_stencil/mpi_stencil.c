#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define DEBUG 0

#define gflops(x, y, steps) (x*y/1e9)*11*steps
#define perf(x, y, steps, us) (x*y/us/1e3)*11*steps

static int STENCIL_SIZE_X;
static int STENCIL_SIZE_Y;

/** number of buffers for N-buffering; should be at least 2 */
#define STENCIL_NBUFFERS 2

#define SCATTER_B1 479
#define SCATTER_B2 974

/** conduction coeff used in computation */
static const double alpha = 0.02;

/** threshold for convergence */
static const double epsilon = 0.001;//0.0001;

/** max number of steps */
static const int stencil_max_steps = 10000;

static double ***values;

/** latest computed buffer */
static int current_buffer = 0;

void global_free(int cols) {

  int i, j;
  for (i = 0; i < STENCIL_NBUFFERS; i++) {
    for (j = 0; j < cols; j++) {
      free(values[i][j]);
    }
    free(values[i]);
  }
  free(values);
}

/** init stencil values to 0, borders to non-zero */
static void stencil_init_dist(int prows, int pcols)
{
  int b, x, y, grank;
  MPI_Comm_rank(MPI_COMM_WORLD, &grank);

  int is_first_col = grank%pcols == 0;
  int is_first_row = grank < pcols;
  int is_last_col = (grank+1)%pcols == 0;
  int is_last_row = grank >= (prows-1)*pcols; 

  /* Storage info */
  int nrows = (is_last_row)? STENCIL_SIZE_Y/prows+STENCIL_SIZE_Y%prows : STENCIL_SIZE_Y/prows;
  int ncols = (is_last_col)? STENCIL_SIZE_X/pcols+STENCIL_SIZE_X%pcols : STENCIL_SIZE_X/pcols;

  nrows += 2;
  ncols += 2;

  values = malloc(STENCIL_NBUFFERS*sizeof(double**));
  for (x = 0; x< STENCIL_NBUFFERS; x++) {
    values[x] = malloc(ncols*sizeof(double *));
    for (y = 0; y < ncols; y++) {
      values[x][y] = (double *)malloc(nrows*sizeof(double));
    }
  }

  int my_row = grank/prows;
  int my_col = grank%prows;

  for(b = 0; b < STENCIL_NBUFFERS; b++) {
    for(x = 0; x < ncols; x++)
      {
	for(y = 0; y < nrows; y++) {
	  values[b][x][y] = 0.0;
	}
      }
    if (is_first_row) { // values = col number
      for(x = 1; x < nrows-1; x++) {
	values[b][1][x] = my_col*(ncols-2)+(x-1);
      }
    }
    if (is_last_col) { // values = rows - row number
      for(y = 1; y < ncols-1; y++) {
	values[b][y][nrows-2] = STENCIL_SIZE_Y - (my_row*(nrows-2)+(y-1));
      }
    }
    if (is_last_row) { // values = cols - col number
      for(x = 1; x < nrows-1; x++) {
	values[b][nrows-2][x] = STENCIL_SIZE_X - (my_col*(ncols-2)+(x-1));
      }
    }
    if (is_first_col) { // values = row number
      for(y = 1; y < ncols-1; y++) {
	values[b][y][1] = my_row*(nrows-2)+(y-1);
      }
    }
  }
}

/** display a (part of) the stencil values */
/* static void stencil_display(int b, int x0, int x1, int y0, int y1) */
/* { */
/*   int x, y; */
/*   for(y = y0; y <= y1; y++) */
/*     { */
/*       for(x = x0; x <= x1; x++) */
/* 	{ */
/* 	  printf("%8.5g ", values[b][x][y]); */
/* 	} */
/*       printf("\n"); */
/*     } */
/* } */

void send_ghosts(int direction, int size, int prev_buffer, int rank,
		 int local_rows, int local_cols, int pcols) {
 
  int i, target, tag;
  double *send_ghost = malloc(sizeof(double)*size);

  switch (direction) {
  case 0: // northern neighboor: copy first row
    target = rank - pcols;
    memcpy(send_ghost, values[prev_buffer][1]+1, size*sizeof(double));
    break;
  case 1: // eastern neighboor: copy last col
    target = rank+1;
    for (i = 0; i < size; i++) {
      send_ghost[i] = values[prev_buffer][i+1][local_cols-2];
    }
    break;
  case 2: // southern neighboor: copy last row
    target = rank + pcols;
    memcpy(send_ghost, values[prev_buffer][local_rows-2]+1, size*sizeof(double));
    break;
  case 3: // western neighboor: copy first col
    target = rank-1;
    for (i = 0; i < size; i++) {
      send_ghost[i] = values[prev_buffer][i+1][1];
    }
    break;
  default:
    target = -1;
    break;
  }
  tag = rank*100 + target;
  MPI_Send(send_ghost, size, MPI_DOUBLE, target, tag, MPI_COMM_WORLD);
  free(send_ghost);
}

void recv_ghosts(int direction, int size, int prev_buffer, int rank,
		 int local_rows, int local_cols, int pcols) {
  int i, source, tag;
  double *recv_ghost = malloc(sizeof(double)*size);
  MPI_Status status;
  switch (direction) {
  case 0: // northern neighboor: copy to first row
    source = rank - pcols;
    tag = source*100 + rank;
    MPI_Recv(recv_ghost, size, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
    memcpy(values[prev_buffer][0]+1, recv_ghost, size*sizeof(double));
    break;
  case 1: // eastern neighboor: copy to last col
    source = rank+1;
    tag = source*100 + rank;
    MPI_Recv(recv_ghost, size, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
    for (i = 0; i < size; i++) {
      values[prev_buffer][i+1][local_cols-1] = recv_ghost[i];
    }
    break;
  case 2: // southern neighboor: copy to last row
    source = rank + pcols;
    tag = source*100 + rank;
    MPI_Recv(recv_ghost, size, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
    memcpy(values[prev_buffer][local_rows-1]+1, recv_ghost, size*sizeof(double));
    break;
  case 3: // western neighboor: copy first col
    source = rank-1;
    tag = source*100 + rank;
    MPI_Recv(recv_ghost, size, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
    for (i = 0; i < size; i++) {
      values[prev_buffer][i+1][0] = recv_ghost[i];
    }
    break;
  default:
    break;
  }
}

/** compute the next stencil step */
static void stencil_step(int sizes[], int local_rows, int local_cols, int prows, int pcols)
{
  int prev_buffer = current_buffer;
  int next_buffer = (current_buffer + 1) % STENCIL_NBUFFERS;
  int x, y, i, rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* Ghosts management */
  for (i = 0; i < 4; i++) {
    /* If there is a neighboor */
    if (sizes[i] > 0) {
      send_ghosts(i, sizes[i], prev_buffer, rank, local_rows, local_cols, pcols);
      recv_ghosts(i, sizes[i], prev_buffer, rank, local_rows, local_cols, pcols);      
    }
  }

  int is_first_col = rank%pcols == 0;
  int is_first_row = rank < pcols;
  int is_last_col = (rank+1)%pcols == 0;
  int is_last_row = rank >= (prows-1)*pcols;

  int begin_x = (is_first_row)? 2 : 1;
  int end_x = (is_last_row)? local_rows-2 : local_rows-1;
  int begin_y = (is_first_col)? 2 : 1;
  int end_y = (is_last_col)? local_rows-2 : local_rows-1;

  for(x = begin_x; x < end_x; x++)
    {
      for(y = begin_y; y < end_y; y++)
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
static int stencil_test_convergence(int rank, int local_rows, int local_cols,
				    int prows, int pcols)
{
  int prev_buffer = (current_buffer - 1 + STENCIL_NBUFFERS) % STENCIL_NBUFFERS;
  int x, y;

  int is_first_col = rank%pcols == 0;
  int is_first_row = rank < pcols;
  int is_last_col = (rank+1)%pcols == 0;
  int is_last_row = rank >= (prows-1)*pcols;

  int begin_x = (is_first_row)? 2 : 1;
  int end_x = (is_last_row)? local_rows-2 : local_rows-1;
  int begin_y = (is_first_col)? 2 : 1;
  int end_y = (is_last_col)? local_rows-2 : local_rows-1;
  
  for(x = begin_x; x < end_x; x++)
    {
      for(y = begin_y; y < end_y; y++)
	{
	  if(fabs(values[prev_buffer][x][y] - values[current_buffer][x][y]) > epsilon)
	    return 0;
	}
    }
  return 1;
}

int main(int argc, char**argv)
{

  /* Constants */
  STENCIL_SIZE_X = atoi(argv[1]);
  STENCIL_SIZE_Y = atoi(argv[2]);;

  /* Distribution info */
  int np, prows, pcols, grank;
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &grank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  prows = pcols = sqrt(np);

  /* Topological info */
  int is_first_col = grank%pcols == 0;
  int is_first_row = grank < pcols;
  int is_last_col = (grank+1)%pcols == 0;
  int is_last_row = grank >= (prows-1)*pcols;

  /* Storage info */
  int local_rows = (is_last_row)? STENCIL_SIZE_Y/prows+STENCIL_SIZE_Y%prows : STENCIL_SIZE_Y/prows;
  int local_cols = (is_last_col)? STENCIL_SIZE_X/pcols+STENCIL_SIZE_X%pcols : STENCIL_SIZE_X/pcols;

  /* Add ghost's rows/cols */
  local_rows += 2;
  local_cols += 2;

  /* Matrix initialization */
  stencil_init_dist(prows, pcols);

  /* MPI Datatypes creation */
  MPI_Datatype regular_row, regular_col, big_row, big_col;
  MPI_Type_contiguous(STENCIL_SIZE_Y/prows, MPI_DOUBLE, &regular_row);
  MPI_Type_contiguous(STENCIL_SIZE_Y/prows+STENCIL_SIZE_Y%prows, MPI_DOUBLE, &big_row);
  MPI_Type_contiguous(STENCIL_SIZE_X/pcols, MPI_DOUBLE, &regular_col);
  MPI_Type_contiguous(STENCIL_SIZE_X/pcols+STENCIL_SIZE_X%pcols, MPI_DOUBLE, &big_col);
  MPI_Type_commit(&regular_row);
  MPI_Type_commit(&big_row);
  MPI_Type_commit(&regular_col);
  MPI_Type_commit(&big_col);

  /* Ghost sizes - clockwise starting from the north */
  int ghost_sizes[4];
  ghost_sizes[0] = (is_first_row)? 0 : local_cols-2;
  ghost_sizes[1] = (is_last_col)? 0 : local_rows-2;
  ghost_sizes[2] = (is_last_row)? 0 : local_cols-2;
  ghost_sizes[3] = (is_first_col)? 0 : local_rows-2;    

  struct timespec t1, t2;
  clock_gettime(CLOCK_MONOTONIC, &t1);
  int s, local_converged, global_converged = 0;
  for(s = 0; s < stencil_max_steps; s++)
    {
      stencil_step(ghost_sizes, local_rows, local_cols, prows, pcols);
    }
  clock_gettime(CLOCK_MONOTONIC, &t2);
  const double t_usec = (t2.tv_sec - t1.tv_sec) * 1000000.0 + (t2.tv_nsec - t1.tv_nsec) / 1000.0;

  if (grank == 0) {
    printf("mpi;%d;%d;%d;%.4f gflops/s\n", STENCIL_SIZE_X, STENCIL_SIZE_Y, s, perf(STENCIL_SIZE_X, STENCIL_SIZE_Y, s, t_usec));
  }
  MPI_Finalize();
  global_free(local_cols);

  return 0;
}

