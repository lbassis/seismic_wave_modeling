#!/usr/bin/gnuplot -persist

set term postscript eps enhanced color
set output "starpu_cl_compute_veloc.jane.eps"
set title "Model for codelet cl-compute-veloc.jane"
set xlabel "Total data size"
set ylabel "GFlops"

set key top left
set logscale x
set logscale y

set xrange [1:10**9]

plot	"starpu_cl_compute_veloc.jane_avg.data" using 1:2:3 with errorlines title "Average cpu0-impl0 (Comb0)",\
	"starpu_cl_compute_veloc.jane_avg.data" using 1:4:5 with errorlines title "Average cuda0-impl0 (Comb1)"
