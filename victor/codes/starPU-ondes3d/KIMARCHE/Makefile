
#CFLAGS   = -O2 -g -std=c99
CFLAGS   = -O3 -std=gnu99
NVCFLAGS = -O3 -arch sm_10 -DOPTI_ARCH_SM_20  
LDFLAGS  = -L/opt/cluster/plafrim-dev/hwloc/latest/hwloc-1.7.2/src/.libs


ifeq ($(USE_MPI),yes)
EXE	 = ondes3d-starpu-mpi
CC       = mpicc
CFLAGS  += $(shell pkg-config --cflags starpumpi-1.0) -DACTUALLY_USE_MPI
LDFLAGS += $(shell pkg-config --libs starpumpi-1.0)
#LDFLAGS += -lm -lstarpumpi-1.0 -lstarpu-1.0 -lcudart -lcublas -lstdc++ -lhwloc
else
EXE	 = ondes3d-starpu
#CC       = icc
CC       = gcc
CFLAGS  += $(shell pkg-config --cflags starpu-1.0)
LDFLAGS += $(shell pkg-config --libs starpu-1.0)
#LDFLAGS += -lm -lstarpu-1.0 -lcudart -lcublas -lstdc++ -lhwloc
endif

OBJ      = ondes3D-functs.o stencil-kernels.o stencil-blocks.o stencil-tasks.o stencil.o ondes3D-kernels.o

all: $(OBJ)
	$(CC) -o $(EXE) $^ $(CFLAGS) $(LDFLAGS)

%.o: %.cu
	nvcc  -o $@ -c $< $(NVCFLAGS)

%.o: %.c
	$(CC) -o $@ -c $< $(CFLAGS) 

clean:
	rm -f $(EXE) *.o

mpi:
	@export USE_MPI=yes && $(MAKE)

nompi: 
	@export USE_MPI=no && $(MAKE)
