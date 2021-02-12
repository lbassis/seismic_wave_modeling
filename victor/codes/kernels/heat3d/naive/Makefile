CC = gcc
PAPI_DIR = /share/apps/papi-5.2/
PAPI_INC_DIR = -I$(PAPI_DIR)/include 
PAPI_LIB_DIR = -L$(PAPI_DIR)/lib
SCOREP_DIR = $(HOME)/build/scorep-1.4.2
SCOREP_INC_DIR = -I$(PAPI_DIR)/include 
SCOREP_LIB_DIR = -L$(PAPI_DIR)/lib
LIB_DIR = $(PAPI_LIB_DIR) $(SCOREP_LIB_DIR)
INC_DIR = $(PAPI_INC_DIR) $(SCOREP_INC_DIR)
CFLAGS =  -O3 -fopenmp -funroll-all-loops -ftree-vectorize -UPINIT -UPROUT -DBLOCKING1 -UOUTPUTINIT -UOUTPUTFINAL -DPAPI
# -fopenmp -funroll-all-loops -ftree-vectorize -DPAPI -O3 -fopenmp  -UPINIT -UOUTPUTFINAL -UOUTPUTINIT

#CC = icc  -O3 -openmp -DMALLOC_STD -ansi-alias -fno-alias -fimf-precision=low -fast-transcendentals -no-prec-sqrt -no-prec-div -fp-model fast=2 -vec-report0  -free-zero -UPROUT -DOUTPUTINIT -DOUTPUTFINAL
#CC = icc  -O3 -vec-report6 -openmp -DMALLOC_STD -ansi-alias -fno-alias  -UPROUT -UOUTPUTINIT -UOUTPUTFINAL -DPINIT
naive-probe:	main.c util.c timer.c probe_iso3dfd_naive.c  
	$(TRACES) $(CC) $(INC_DIR) $(LIB_DIR) $(COPTFLAGS) $(TIMER) $(CFLAGS) -O3 main.c util.c timer.c probe_iso3dfd_naive.c $(CLDFLAGS) -lm -lpapi -o probe

clean:
	rm -f *.o probe	probe_naive
