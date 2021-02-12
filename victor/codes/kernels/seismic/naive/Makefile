#CC=pgcc -Mvect -Munroll -fast  -mp -tp=bulldozer -Mvect=prefetch -Mipa=fast,inline
CC = icc -O3 -openmp -DPINIT  -ansi-alias -fno-alias  -UPROUT
#CC = gcc -O3 -fopenmp -DPINIT -UPROUT

naive-probe:	main.c util.c timer.c cycle.h run.h probe_seismic_naive.c  cycle.h
	$(CC) $(CFLAGS) $(COPTFLAGS) $(TIMER)  main.c util.c timer.c probe_seismic_naive.c   $(CLDFLAGS) -o probe

clean:
	rm -f *.o probe	probe_naive
