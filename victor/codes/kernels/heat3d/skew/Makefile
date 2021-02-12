CC = gcc
PAPI_DIR = /share/apps/papi-5.2/
PAPI_INC_DIR = -I$(PAPI_DIR)/include
PAPI_LIB_DIR = -L$(PAPI_DIR)/lib
SCOREP_DIR = $(HOME)/build/scorep-1.4.2
SCOREP_INC_DIR = -I$(PAPI_DIR)/include
SCOREP_LIB_DIR = -L$(PAPI_DIR)/lib
LIB_DIR = $(PAPI_LIB_DIR) $(SCOREP_LIB_DIR)
INC_DIR = $(PAPI_INC_DIR) $(SCOREP_INC_DIR)

CFLAGS = -O3 -UUNROLL -fopenmp -funroll-all-loops  -UPINIT -UPROUTALL -DMALLOC_STD -UOUTPUTINIT -UOUTPUTFINAL -DPAPI
SRC_HOST = main.c probe_iso3dfd_naive.c timer.c util.c
SRC_HEADER = common.h cycle.h dim3d.h run-bench.sh run.h timer.h util.h
OBJ_HOST = $(SRC_HOST:%.c=%.o)

#CC = icc -UPINIT  -O3 -ansi-alias -fno-alias -vec-report5 -UUNROLL -openmp -DMALLOC_STD    -UOUTPUTINIT -UOUTPUTFINAL  -UPROUT
#CLDFLAGS = $(PAPI)

timeskew-probe:	main.c util.c timer.c run.h probe_iso3dfd_skew.c compute_subdomain.c compute_stencil.c cycle.h 
	$(TRACES) $(CC) $(INC_DIR) $(LIB_DIR) $(CFLAGS)  $(COPTFLAGS) $(TIMER) main.c util.c probe_iso3dfd_skew.c timer.c  compute_subdomain.c compute_stencil.c -lm -lpapi $(CLDFLAGS) -o probe

clean:
	rm -f *.o probe	
