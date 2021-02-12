BIN_DIR= ../bin

##############
# MPI settings
##############
ifeq ($(USE_MPI),yes)
	MESSAGE="*** COMPILED WITH MPI ***"
	EXECUTABLE=$(BIN_DIR)/ondes3D_cuda_mpi_exe
	MPI_INC_DIR= -I/home/admin/DAVID/ONDES3D/MPI_VERSIONS/my_mpi_install/include
	MPI_LIB_DIR= -L/home/admin/DAVID/ONDES3D/MPI_VERSIONS/my_mpi_install/lib
	#MPI_LIB_NAME=
	MPI_LIB_NAME= -lmpi
else
	MESSAGE="*** COMPILED WITHOUT MPI ***"
	EXECUTABLE=$(BIN_DIR)/ondes3D_cuda_exe
	MPI_INC_DIR=
	MPI_LIB_DIR=
	MPI_LIB_NAME=
endif
EXE_GEN=ondes3D_cuda*_exe

##################
# CUDA directories
##################

CUDA_INC_DIR= -I/usr/local/cuda/include
CUDA_LIB_DIR= -L/usr/local/cuda/lib -L/home/admin/NVIDIA_CUDA_SDK/common/lib/linux
CUDA_LIB_NAME= -lcudart -lcuda -lcudpp64

CC_CUDA=nvcc

#CFLAGS_CUDA= --host-compilation C -arch sm_10 -O3 --ptxas-options=-v # --maxrregcount 16
#CFLAGS_CUDA= --host-compilation C -arch sm_10 -O3 --ptxas-options=-v
#CFLAGS_CUDA= --host-compilation C -arch sm_10 -O3 -G
#CFLAGS_CUDA= --host-compilation C -arch sm_10 -O3
ifeq ($(EMULATION),yes)
	CFLAGS_CUDA= --host-compilation C -arch sm_10 -O3 -deviceemu -DEMU_DGN
	MESSAGE="*** COMPILED WITHOUT MPI IN EMULATION MODE ***"
else
	CFLAGS_CUDA= --host-compilation C -arch sm_10 -O3 
endif

# add settings defined above
CFLAGS_CUDA := $(CFLAGS_CUDA) $(CUDA_INC_DIR)

######################
# HOST compliler flags
######################
CFLAGS_HOST= -fno-trapping-math -fno-signaling-nans -std=gnu99 -O3
#CFLAGS_HOST= -fno-trapping-math -fno-signaling-nans -std=gnu99 -O3 -g

ifeq ($(USE_MPI),yes)
	CC_HOST= /home/admin/DAVID/ONDES3D/MPI_VERSIONS/my_mpi_install/bin/mpicc
	CFLAGS_HOST := $(CFLAGS_HOST) -DUSE_MPI
else
	CC_HOST= gcc
endif

# add settings defined above
CFLAGS_HOST := $(CFLAGS_HOST) $(MPI_INC_DIR) $(CUDA_INC_DIR)


# linker
ifeq ($(USE_MPI),yes)
	LD_ALL= /home/admin/DAVID/ONDES3D/MPI_VERSIONS/my_mpi_install/bin/mpicc
else
	LD_ALL= gcc
endif
ifeq ($(EMULATION),yes)
	LDFLAGS= $(CFLAGS_HOST) $(MPI_LIB_DIR) $(MPI_LIB_NAME) $(CUDA_LIB_DIR) $(CUDA_LIB_NAME) -lm
else
	LDFLAGS= $(CFLAGS_HOST) $(MPI_LIB_DIR) $(MPI_LIB_NAME) $(CUDA_LIB_DIR) $(CUDA_LIB_NAME)
endif

###########################################
# CUDA source files (kernels plus wrappers)
###########################################

# this a file list!
SRC_CUDA=ondes3D_kernels.cu

############################
# HOST source files (C code)
############################

# this a file list!
SRC_HOST=ondes3D_main.c ondes3D_funct.c nr.c 

SRC_HEADER=ondes3D.h ondes3D_kernels.h topo_mpi.h nr.h

##############################
# end of configuration section
##############################

# create list of object files
OBJ_CUDA=$(SRC_CUDA:%.cu=%.co)
LINKINFO_CUDA=$(SRC_CUDA:%.cu=%.linkinfo)
OBJ_HOST=$(SRC_HOST:%.c=%.o) 

# implicit build rules for all files
%.co: %.cu 
	@$(CC_CUDA) $(CFLAGS_CUDA) -c -o $@ $<

%.o: %.c
	@$(CC_HOST) $(CFLAGS_HOST) -c -o $@ $<

# finally, the targets
all : comp 
	@echo $(MESSAGE) && \rm -f $(OBJ_CUDA) $(OBJ_HOST) $(LINKINFO_CUDA)
	
comp : clean $(OBJ_CUDA) $(OBJ_HOST) $(SRC_HEADER)
	@$(LD_ALL) $(OBJ_CUDA) $(OBJ_HOST) -o $(EXECUTABLE)	$(LDFLAGS)

clean:
	@\rm -f $(OBJ_CUDA) $(OBJ_HOST) $(LINKINFO_CUDA) $(EXE_GEN)

mpi :
	@export USE_MPI=yes && $(MAKE)

nompi : 
	@export USE_MPI=no && $(MAKE)

emu : 
	@export EMULATION=yes && export USE_MPI=no && $(MAKE)
.PHONY : 
