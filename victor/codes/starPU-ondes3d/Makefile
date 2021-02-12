
#CFLAGS   = -O2 -g -std=c99
CFLAGS   = -O3 -std=gnu99
#NVCFLAGS = -O3 -arch sm_10 -DOPTI_ARCH_SM_20  
NVCFLAGS = -O3 -arch sm_35
BUILD_DIR = /home/vmartinez/salmao/build
LDFLAGS  = -L$(BUILD_DIR)/hwloc-1.10.0/lib -L/usr/local/cuda-6.5/lib64 -L$(BUILD_DIR)/openmpi-1.8.4


ifeq ($(USE_MPI),yes)
EXE	 = ondes3d-starpu-mpi
CC       = mpicc
CFLAGS  += $(shell pkg-config --cflags starpumpi-1.2) -DACTUALLY_USE_MPI
LDFLAGS += $(shell pkg-config --libs starpumpi-1.2)
#LDFLAGS += -lm -lstarpumpi-1.0 -lstarpu-1.0 -lcudart -lcublas -lstdc++ -lhwloc
else
EXE	 = ondes3d-starpu
CC       = gcc
#CC       = gcc
CFLAGS  += $(shell pkg-config --cflags starpu-1.2)
LDFLAGS += $(shell pkg-config --libs starpu-1.2)
#LDFLAGS += -lm -lstarpu-1.0 -lcudart -lcublas -lstdc++ -lhwloc
endif

OBJ      = ondes3D-functs.o stencil-kernels.o stencil-blocks.o stencil-tasks.o stencil.o ondes3D-kernels.o utils.o

all: $(OBJ)
	$(CC) -o $(EXE) $^ $(CFLAGS) $(LDFLAGS) -lrt -lm -lcudart

%.o: %.cu
	nvcc  -o $@ -c $< $(NVCFLAGS) -I/home/vmartinez/salmao/build/starpu-1.2.0/include/starpu/1.2/

%.o: %.c
	$(CC) -o $@ -c $< $(CFLAGS) 

clean:
	rm -f $(EXE) *.o

mpi:
	@export USE_MPI=yes && $(MAKE)

nompi: 
	@export USE_MPI=no && $(MAKE)
