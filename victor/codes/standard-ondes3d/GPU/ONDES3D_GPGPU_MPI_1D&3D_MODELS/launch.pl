#!/usr/bin/perl	
# script we used on our 6 nodes cluster, each one equiped with one 8800 GTX card.
# There was some problem with myrinet, so we use ethernet to communicate with openMPI (--mca btl_tcp_if_include eth0).
# therefore you should adapt the command line for your machine



# hard limitation for our GPU cluster, you should change this ...
my $nb_GPU_in_cluster = 6;

my $mpi_exec="./bin/ondes3D_cuda_mpi_exe";
my $seq_exec="./bin/ondes3D_cuda_exe";

my $paramfile = "";
if (scalar(@ARGV) == 1) {
		if (-f $ARGV[0]) {
				print "launching run with parameter file : $ARGV[0]\n";
				$paramfile = $ARGV[0];
		} else {
			  print "parameter file $ARGV[0] doesn't exist\naborting.\n";
				exit;
		}
} else {
		my $PRM = `grep '#define PRM' src/ondes3D_main.c`;
		$PRM =~ s/^#define PRM "(.*)"$/$1/;
		chomp($PRM);
		print "launching run with default parameter file : $PRM\n";
}

my $mpi_on;
if (-f "$mpi_exec") {
	print "cuda + mpi version\n";
	$mpi_on	 = 1;
} elsif (-f "$seq_exec") {
	print "sequential cuda version\n";
	$mpi_on	 = 0;
} else {
	print "unknown version, maybe a compilation problem ??\n";
}

if ($mpi_on) {
	open(CODE, "./src/topo_mpi.h");
	my $nprocx, $nprocy;
	while (my $line = <CODE>) {
		chomp $line;
		if ($line =~m/define NPROCX (.*)$/) {
			$nprocx = $1;
		}
		if ($line =~m/define NPROCY (.*)$/) {
			$nprocy = $1;
		}
	}
	my $nbproc = $nprocy*$nprocx;
	if ($nbproc > $nb_GPU_in_cluster) {
		print "You try to launch on $nbproc GPUs, but there are only $nb_GPU_in_cluster GPUs in the cluster\n";
	} else {
		print "parallel run sur $nbproc GPUs\n";
		system("mpirun --mca btl_tcp_if_include eth0 -np $nbproc -hostfile ./DATA/hostfile -rankfile ./DATA/rankfile $mpi_exec $paramfile") == 0
			or begin 
			{
				print "If the run does not start, check that the $nbproc first machines of file ./DATA/hostfile are on :\n";
				system("head -n $nbproc ./DATA/hostfile");
				print "\n\n";
				exit(1);
			}
			end;
	}
} else {
	print "sequential run on 1 GPU\n";
	system("$seq_exec $paramfile");
}

