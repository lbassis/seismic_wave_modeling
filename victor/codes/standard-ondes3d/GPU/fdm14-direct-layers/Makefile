CC	= mpicc 
CFLAGS  =  $(TESTFLAGS) -O2 -Bstatic -DVTK -UPAPI -UMISS -UFLOPS -DTIMING -UTIMING_BARRIER -UDECOUP1 -DDECOUP2 -UTAU -UTAUGLOBAL -UPERSISTANT -UBLOCKING -DNONBLOCKING

MPI_FLAGS = 
PREFIX    = fdm14-mpi-layers


all: EXE-ONDES-mpi4

EXE-ONDES-mpi4: $(PREFIX).c nrutil.c
	$(CC) $(MPI_FLAGS) $(CFLAGS) -c $(PREFIX).c
	$(CC) $(CFLAGS) -c nrutil.c
	$(CC) $(MPI_FLAGS) -o $@ $(PREFIX).o nrutil.o -lm 

clean:
	rm -f *.o EXE-ONDES-mpi4
