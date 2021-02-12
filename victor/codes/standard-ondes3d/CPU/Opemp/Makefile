CC		=	gcc 
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
TESTFLAGS	+= -DNOMPI
#TESTFLAGS       += -DOMP=1
TESTFLAGS       += -DNOPML


# Geological
#TESTFLAGS	+= -DOUT_HOGE

# compute without anelasticity part
#TESTFLAGS	+= -DNOANELASTICITY

## OPTIMISATION PARAMETERS
# optimisation level
OPTI 		+=  -O2 -Wno-format -Wno-unused-result # -g  -\#  # -Minfo=all -Mlarge_arrays # -Minline=name:ComputeStress,ComputeVelocity,ComputeIntermediates #-O0 -g  
# remove asserts
#OPTI		+= -DNDEBUG
COMM		= -DPERSISTANT -UBLOCKING 

#MODEL parameters; default values are in options.h
CFLAGS		=  -DVTK $(TESTFLAGS)  $(OPTI) $(COMM) $(MODEL) -UPAPI -UMISS -UFLOPS -DTIMING -UTIMING_BARRIER -DDECOUP2 -UTAU -UTAUGLOBAL 
MPI_FLAGS =
PREFIX =ondes3d$(POST)
OBJS = main.o nrutil.o  computeVeloAndSource.o computeStress.o computeIntermediates.o alloAndInit.o IO.o alloAndInit_LayerModel.o
HEADERS=struct.h inlineFunctions.h options.h

OMP_FLAGS = -fopenmp


all: $(PREFIX)
$(PREFIX): $(OBJS)
	$(CC)  $(MPI_FLAGS) $(OMP_FLAGS) -o $@ $^ -lm
%.o: %.c
	$(CC) $(MPI_FLAGS) $(OMP_FLAGS) $(CFLAGS) -c $<

clean:
	rm -f *.o  *~
