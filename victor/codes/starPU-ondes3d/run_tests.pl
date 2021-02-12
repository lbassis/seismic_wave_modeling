#!/usr/bin/perl

my $nbtries = 6;
my @sizes = (160, 80, 64, 32);
my $resdir = "results_big_allpu_blocksize_varying";


foreach my $size(@sizes) {
	print "block size : $size\n";	
	for(my $i=0;$i<$nbtries; $i++) {
		print "\tEssai ".($i+1)."/$nbtries\n";
		my $align = "-align";
		my $cmde = "optirun ./ondes3d-starpu -fixed -bsizex $size -bsizey $size $align -param ./DATA/aquila_no_cpml.prm";
		my $fileout = "./$resdir/big_bl".$size."_run".$i."_align.txt";
		my $svg = "./$resdir/big_bl".$size."_run".$i."_align.svg";

		system("set | grep STARPU > $fileout");
		system("$cmde >> $fileout");
		system("mv schedule_time_line.svg $svg");
	}
}

sub setenv {
	my ($name, $val) = @_;
	$ENV{"$name"} = "$val";
}
