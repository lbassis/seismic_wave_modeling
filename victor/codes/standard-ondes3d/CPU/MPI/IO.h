#ifndef IO_H_
#define IO_H_
#include "struct.h"
#include "options.h"


static const int STRL=256; 	/* default lentgh of strings */
int OutSeismograms(struct OUTPUTS OUT, struct PARAMETERS PRM,
				   int ir, int l, const char* flname,int station_step );
int OutGeol(struct MEDIUM MDM,struct  OUTPUTS OUT,struct  PARAMETERS PRM, const char* flname );

int PrintInfoMedium( struct ANELASTICITY ANL, struct  MEDIUM MDM, 
                struct SOURCE SRC, struct PARAMETERS PRM );

/*  */
int ReadTopo(struct PARAMETERS *PRM, const char* topologieFile, const int np );

int ReadPrmFile (struct  PARAMETERS *PRM,
				 struct MEDIUM *MDM,
				 struct ABSORBING_BOUNDARY_CONDITION *ABC, 
				 struct ANELASTICITY *ANL, 
				 struct OUTPUTS *OUT,
				 const char* prmFile );


int ReadSrc( struct SOURCE *SRC, struct  PARAMETERS PRM  );
 
int ReadStation( struct OUTPUTS *OUT, struct  PARAMETERS PRM);


int ReadGeoFile( struct MEDIUM *MDM, struct PARAMETERS PRM );
  /* OTHER FUNCTIONS (TO read PRMFILE ) */
  /* =============== */
char* FindField(const char *cs, const char *ct);
int FindInt(const char* strIn, const char* srch );
double FindDble(const char* strIn, const char* srch );
char* File2str(const char* flname);


#endif	/* IO_H_ */
 
