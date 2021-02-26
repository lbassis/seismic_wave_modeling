STARPU_VERSION=1.3
CPPFLAGS += $(shell pkg-config --cflags starpu-$(STARPU_VERSION))
CPPFLAGS += -g
LDLIBS += $(shell pkg-config --libs starpu-$(STARPU_VERSION))
LDLIBS += -Wl,-rpath -Wl,$(shell pkg-config --variable=libdir starpu-$(STARPU_VERSION))

starpu_stencil.o: starpu_stencil.c
	gcc $(CPPFLAGS) $< -c -o $@ $(LDLIBS)

all: starpu_stencil.o
	gcc $< $(LDLIBS) -o demo		

clean:
	rm -f *.o  *~ demo

#CC      = mpicc
#CFLAGS += -Wall -g -O4
#LDLIBS += -lm -lrt -fopenmp
#
#all: stencil omp_stencil mpi_stencil hybrid_stencil
#
#clean:
#	-rm stencil omp_stencil mpi_stencil hybrid_stencil

