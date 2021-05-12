#ifndef IO_H_
#define IO_H_
#include "struct.h"
#include "options.h"


static const int STRL = 256;	/* default lentgh of strings */

int IOsurfloc2glob3(int flag, int coord0, int coord1,
		    struct PARAMETERS PRM, struct OUTPUTS OUT);
int IOsurfloc2glob2(int flag, int coord0, int coord1,
		    struct PARAMETERS PRM, struct OUTPUTS OUT);
int IOsurfloc2glob1(int flag, int coord0, int coord1,
		    struct PARAMETERS PRM, struct OUTPUTS OUT);
int IOsurfhelper1(double ***v, int flag, int outvel, int outdisp,
		  struct PARAMETERS PRM, struct OUTPUTS OUT);
int IOsurfhelper2(double ***v, int flag, int outvel, int outdisp,
		  struct PARAMETERS PRM, struct OUTPUTS OUT);
int IOsurfhelper3(double ***v, int flag, int outvel, int outdisp,
		  struct PARAMETERS PRM, struct OUTPUTS OUT);

int OutObsNetcdf(char *flname, int l, int ir, int TMAX, double dt,
		 struct OUTPUTS OUT, int station_step);
int OutSurfNetcdf(char *flname, int d, int XMIN, int XMAX, int YMIN,
		  int YMAX, int ZMIN, int ZMAX, double ds, double **vx0,
		  double **vy0, double **vz0);
int OutSeismograms(struct OUTPUTS OUT, struct PARAMETERS PRM, int ir,
		   int l, const char *flname, int station_step);
int OutSeismogramsbis(struct OUTPUTS OUT, struct PARAMETERS PRM, int ir,
		      int l, const char *flname, int station_step);

int OutSeismogramsbin(struct OUTPUTS OUT, struct PARAMETERS PRM, int ir,
		      int l, const char *flname, int station_step);

int OutGeol(struct MEDIUM MDM, struct OUTPUTS OUT, struct PARAMETERS PRM,
	    const char *flname);
int ERR(int retval);
int PrintInfoMedium(struct ANELASTICITY ANL, struct MEDIUM MDM,
		    struct SOURCE SRC, struct PARAMETERS PRM);

/*  */
int ReadTopo(struct PARAMETERS *PRM, const char *topologieFile,
	     const int np);

int ReadPrmFile(struct PARAMETERS *PRM,
		struct MEDIUM *MDM,
		struct ABSORBING_BOUNDARY_CONDITION *ABC,
		struct ANELASTICITY *ANL,
		struct OUTPUTS *OUT, const char *prmFile);


int ReadSrc(struct SOURCE *SRC, struct PARAMETERS PRM);

int ReadStation(struct OUTPUTS *OUT, struct PARAMETERS PRM,
		struct MEDIUM MDM);


int ReadGeoFile(struct MEDIUM *MDM, struct PARAMETERS PRM);
  /* OTHER FUNCTIONS (TO read PRMFILE ) */
  /* =============== */
char *FindField(const char *cs, const char *ct);
int FindInt(const char *strIn, const char *srch);
double FindDble(const char *strIn, const char *srch);
char *File2str(const char *flname);



void force_big_endian(unsigned char *bytes);
void write_float(FILE * fp, float val);



#endif				/* IO_H_ */
