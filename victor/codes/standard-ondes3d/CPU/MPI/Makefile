CC		=	mpicc 
#List of Path to search sources files
VPATH		= .:../src
#Ansi conformity
#TESTFLAGS	+= -Xs

## DEBOGING PARAMETERS
# lecture
#TESTFLAGS	+= -DDEBUG_READ

# allocation  
#TESTFLAGS	+= -DDEBUG_ALLO

# ondes3d.c 
#TESTFLAGS	+= -DVERBOSE=5
#TESTFLAGS	+= -DNOINTERMEDIATES
#TESTFLAGS	+= -DNOSTRESS
#TESTFLAGS	+= -DNOVELOCITY

# Geological
#TESTFLAGS	+= -DOUT_HOGE

# compute without PML/CPML absorption
#TESTFLAGS	+= -DNOABSORB
# compute without anelasticity part
#TESTFLAGS	+= -DNOANELASTICITY

## OPTIMISATION PARAMETERS
# optimisation level
OPTI 		+=  -O2 # -g  -\#  # -Minfo=all -Mlarge_arrays # -Minline=name:ComputeStress,ComputeVelocity,ComputeIntermediates #-O0 -g  
# remove asserts
#OPTI		+= -DNDEBUG
#COMM		= -DPERSISTANT -UBLOCKING 
COMM		= -DPERSISTANT -UBLOCKING 

#MODEL parameters; default values are in options.h
CFLAGS		=  -DVTK $(TESTFLAGS)  $(OPTI) $(COMM) $(MODEL) -UPAPI -UMISS -UFLOPS -DTIMING -UTIMING_BARRIER -DDECOUP2 -UTAU -UTAUGLOBAL 
MPI_FLAGS =
PREFIX =ondes3d$(POST)
OBJS = main.o nrutil.o  computeVeloAndSource.o computeStress.o computeIntermediates.o alloAndInit.o IO.o alloAndInit_LayerModel.o
HEADERS=struct.h inlineFunctions.h options.h

all: $(PREFIX)
$(PREFIX): $(OBJS)
	$(CC) -lm $(MPI_FLAGS) -o $@ $^ 
%.o: %.c
	$(CC) $(MPI_FLAGS) $(CFLAGS) -c $<

clean:
	rm -f *.o  *~
