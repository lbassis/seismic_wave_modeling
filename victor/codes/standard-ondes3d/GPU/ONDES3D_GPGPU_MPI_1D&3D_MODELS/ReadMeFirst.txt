ONDES3D CUDA+MPI (single precision, 4th order FD), models 1D & 3D
corrected version from 30/04/2010


TO COMPILE :
	- adapt Makefile for your machine (libs location)
	- with MPI : make mpi
	- without MPI : make nompi

TO RUN :
	./launch.pl

PARAMETER FILES
	in ./DATA/
	you can pass .prm file location as an argument

CHANGE NUMBER OF MPI PROCS : 
	modify NPROCX et NPROCY in topo_mpi.h then recompile.
	total number of procs = NPROCX * NPROCY

LIMITATIONS
	- no snapshots yet
	- not fully validated with 3D models
	- tuned for old Nvidia 8800 GTX cards. have to be retuned for new architectures.




Credits :
Original code : Hideo Aochi (h.aochi@brgm.fr)
CPMLs : Ariane Ducellier (a.ducellier@brgm.fr)
several OpenMP + MPI versions : Fabrice Dupros (f.dupros@brgm.fr)

this version (CUDA + MPI) : David Michea (d.michea@brgm.fr)
Thanks to Thomas Ulrich (t.ulrich@brgm.fr) for validation & corrections.