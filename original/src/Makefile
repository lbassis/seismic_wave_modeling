CC		=	mpicc

#List of Path to search sources files
VPATH		= .:../src
#Ansi conformity
#TESTFLAGS	+= -Xs

#######################################################
## DEBUG PARAMETERS
TESTFLAGS       += -DVERBOSE=0

# lecture
#TESTFLAGS	+= -DDEBUG_READ
# allocation  
#TESTFLAGS	+= -DDEBUG_ALLO
# no velocity computation
#TESTFLAGS      += -DNOVELOCITY
# no stress computation
#TESTFLAGS      += -DNOSTRESS
# no intermediates computation
#TESTFLAGS      += -DNOINTERMEDIATES
# no absorbing condition computation
#TESTFLAGS       += -DNOABS
# no anelasticity computation
#TESTFLAGS      += -DNOANELASTICITY
###################################################

####################################################
#EXECUTION FLAGS
TESTFLAGS	+= -DMPI
MPI_FLAGS	=	
TESTFLAGS	+= #-DOMP
OMP_FLAGS	= #-fopenmp
# COMM=1 : persistant / COMM=2 : blocking
TESTFLAGS	+= -DCOMM=1
# with MPI topologie file (topologie.in)
#TESTFLAGS	+= -DTOPO_MPI
#####################################################
#ADIOS FLAGS
TESTFLAGS       += -DPRINTSTEP # -DOUTSTD  -DPGV -DPRINTSTEP 


#NETCDF_INC=-I/share/apps/netcdf-4.1/gcc/include 
#NETCDF_LIB= -L//share/apps/netcdf-4.1/gcc/lib -lnetcdf

#ADIOS_DIR=/home/dupros/Lib/adios-install
#ADIOS_INC=-I/home/dupros/Lib/adios-install/include -I/home/dupros/Lib/hdf5-install/include
#ADIOS_LIBW=-L/home/dupros/Lib/adios-install/lib -ladios -L/home/dupros/Lib/hdf5-install/lib -lm -lmxml -lhdf5_hl -lhdf5 -lz -libverbs -lz


#ADIOS_DIR=/home/dupros/2016/adios-install
#ADIOS_INC=-I/home/dupros/2016/adios-install/include
#ADIOS_LIBW=-L/home/dupros/2016/adios-install/lib -ladios -lm -lmxml -lz -libverbs -lz  -L/home/dupros/2016/hdf5-install/lib  -lhdf5_hl -lhdf5 -lz -libverbs -lz


#ADIOS_DIR=/home/dupros/Lib/adios-install-trunk
#ADIOS_INC=-I/home/dupros/Lib/adios-install-trunk/include
#ADIOS_LIBW=-L/home/dupros/Lib/adios-install-trunk/lib -ladios -lm -lmxml -lz -libverbs -lz  

#ADIOS_DIR=/home/dupros/2016/LIB5/adios-install
#ADIOS_INC=-I/home/dupros/2016/LIB5/adios-install/include
#ADIOS_LIBW=-L/home/dupros/2016/LIB5/adios-install/lib -ladios -lm -lmxml -lz -libverbs -lz  -L/usr/lib64//lib64 -llustreapi




####################################################
#PROFILING FLAG

# TIMER=1 : standard / TIMER = 2 : timer vith MPI barriers
TESTFLAGS       += -DTIMER=1
#TAU detailed profiling
#TESTFLAGS	+= -DPROFILE1
#TAU global profiling
#TESTFLAGS       += -DPROFILE2
#FLOPS based on top of PAPI library
#TESTFLAGS       += -DFLOPS
#PAPI counters (cache misses) using PAPI library
#TESTFLAGS       += -DMISS

#######################################################
# OUTPUT
#Write geological model
#TESTFLAGS	+= -DOUT_HOGE

########################################################



#########################################################"
## OPTIMISATION PARAMETERS
OPTI 		+=  -O3 
# -g -traceback
###########################################################


#MODEL parameters; default values are in options.h
CFLAGS		=   $(TESTFLAGS)  $(OPTI)  $(MODEL) 
PREFIX =../ondes3d
OBJS = main.o nrutil.o  computeVeloAndSource.o computeStress.o computeIntermediates.o alloAndInit.o IO.o alloAndInit_LayerModel.o memory_dump.o
HEADERS=struct.h inlineFunctions.h options.h


all: $(PREFIX)
$(PREFIX): $(OBJS)
	$(CC)  $(MPI_FLAGS) $(OMP_FLAGS)  -o $@ $^ -lm $(ADIOS_LIBW)  $(NETCDF_LIB)

%.o: %.c
	$(CC) $(MPI_FLAGS) $(OMP_FLAGS) $(CFLAGS) $(ADIOS_INC) $(NETCDF_INC) -c $<

clean:
	rm -f *.o  *~ ../ondes3d
