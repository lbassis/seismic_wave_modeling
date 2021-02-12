#!/bin/bash

#####################################################################
#  This script performs several runs and must be configured according
#  to your needs. To get energy consumption set GET_ENERGY=1 and run
#  this script with "sudo". You can also set the number of runs for
#  each experiment (see variable REPETITIONS).
#####################################################################

GET_ENERGY=0
REPETITIONS=5
LIKWID_POWER=/home/bastosca/tools/likwid-3.0/likwid-powermeter

X=$1
Y=$2
Z=$3
BLOCK_X=16
BLOCK_Y=16
BLOCK_Z=$1
TIMESTEPS=$4

OUTPUT=results/srv186_${X}_${Y}_${Z}_block_${BLOCK_X}_${BLOCK_Y}_${BLOCK_Z}_${TIMESTEPS}.csv
rm -f $OUPUT

mkdir -p results/
make clean && make
#export KMP_AFFINITY=granularity=core,compact
export OMP_SCHEDULE=static
rm -rf results/tmp.csv

if [ $GET_ENERGY == 1 ]; then
    echo "Size X;Size Y;Size Z;Timesteps;Threads;Time computation (us);Sockets;Time global (s);Energy (AVG);Power (AVG);Energy (Socket 1); Power (Socket 1);Energy (Socket 2); Power (Socket 2);Energy (Socket 3); Power (Socket 3);Energy (Socket 4); Power (Socket 4);Energy (Socket 5); Power (Socket 5);Energy (Socket 6); Power (Socket 6);Energy (Socket 7); Power (Socket 7);Energy (Socket 8); Power (Socket 8);Energy (Socket 9); Power (Socket 9);Energy (Socket 10); Power (Socket 10);Energy (Socket 11); Power (Socket 11);Energy (Socket 12); Power (Socket 12);Energy (Socket 13); Power (Socket 13);Energy (Socket 14); Power (Socket 14);Energy (Socket 15); Power (Socket 15);Energy (Socket 16); Power (Socket 16);Energy (Socket 17); Power (Socket 17);Energy (Socket 18); Power (Socket 18);Energy (Socket 19); Power (Socket 19);Energy (Socket 20); Power (Socket 20);Energy (Socket 21); Power (Socket 21);Energy (Socket 22); Power (Socket 22);Energy (Socket 23); Power (Socket 23);Energy (Socket 24); Power (Socket 24)" >> results/tmp.csv
else
    echo "Size X;Size Y;Size Z;Timesteps;Threads;Time computation (us)" >> results/tmp.csv
fi

# =============== SINGLE NODE ===============
#sequential
let "MAX_THREAD_ID = $THREADS - 1"

export OMP_NUM_THREADS=1
#export GOMP_CPU_AFFINITY="0"

echo "Running with 1 thread: affinity=0"

if [ $GET_ENERGY == 1 ]; then
    $LIKWID_POWER -c 1 ./stencil $X $Y $Z $BLOCK_X $BLOCK_Y $BLOCK_Z $TIMESTEPS >> results/tmp.csv
else
    ./probe $X $Y $Z $BLOCK_X $BLOCK_Y $BLOCK_Z $TIMESTEPS >> results/tmp.csv
    echo "" >> results/tmp.csv
fi

#parallel
for THREADS in {2..2}; do

    let "MAX_THREAD_ID = $THREADS - 1"
    
    export OMP_NUM_THREADS=$THREADS
#   export GOMP_CPU_AFFINITY="0-$MAX_THREAD_ID"
    
    for REP in `seq 1 $REPETITIONS`; do
	echo "Running with $THREADS threads: affinity=0-$MAX_THREAD_ID, $REP/$REPETITIONS..."
	
	if [ $GET_ENERGY == 1 ]; then
	    $LIKWID_POWER -c 1 ./stencil $X $Y $Z $BLOCK_X $BLOCK_Y $BLOCK_Z $TIMESTEPS >> results/tmp.csv
	else
	    ./probe $X $Y $Z $BLOCK_X $BLOCK_Y $BLOCK_Z $TIMESTEPS >> results/tmp.csv
	    echo "" >> results/tmp.csv
	fi
    done
done

##########################################################################################################
cat results/tmp.csv >> $OUTPUT
rm -rf results/tmp.csv
