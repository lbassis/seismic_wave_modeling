STARPU_VERSION=1.3
CPPFLAGS += $(shell pkg-config --cflags starpu-$(STARPU_VERSION))
LDLIBS += $(shell pkg-config --libs starpu-$(STARPU_VERSION))
LDLIBS += -Wl,-rpath -Wl,$(shell pkg-config --variable=libdir starpu-$(STARPU_VERSION))

stencil.o: stencil.c
	gcc $(CPPFLAGS) $< -c -o $@ $(LDLIBS)

stencil: stencil.o
	gcc stencil.o $(LDLIBS) -o demo		

clean:
	rm -f *.o  *~

#CC      = mpicc
#CFLAGS += -Wall -g -O4
#LDLIBS += -lm -lrt -fopenmp
#
#all: stencil omp_stencil mpi_stencil hybrid_stencil
#
#clean:
#	-rm stencil omp_stencil mpi_stencil hybrid_stencil

