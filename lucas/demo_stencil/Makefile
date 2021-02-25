CC      = mpicc
CFLAGS += -Wall -g -O4
LDLIBS += -lm -lrt -fopenmp

all: stencil omp_stencil mpi_stencil hybrid_stencil

clean:
	-rm stencil omp_stencil mpi_stencil hybrid_stencil

