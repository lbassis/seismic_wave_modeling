#ifndef MAIN_H_
#define MAIN_H_


int VerifFunction( int exitStatus, const char* errorMsg, struct PARAMETERS PRM );
int double2int(const double);
double my_second();

/* ===COMMUNICATIONS RELATED=== */
static const int CPU_NO_SEND = 2 ;
static const int CPU_NO_RECV = 3 ;

int SyncBufStress(struct STRESS *t0,
				  int mode, /*  0: send, 1: receive */
				  struct COMM_DIRECTION *DIR,
				  struct PARAMETERS PRM
				  );

int SyncBufVelocity(struct VELOCITY *v0,
		     int mode, /*  0: send, 1: receive */
		     struct COMM_DIRECTION *DIR,
		     struct PARAMETERS PRM
		     );

int SyncBufKsil( struct ANELASTICITY *ANL,
				 int mode, /*  0: send, 1: receive */
				 struct COMM_DIRECTION *DIR,
				 struct PARAMETERS PRM
				 );

/* ===Seismograms related ===*/
int ComputeSeismograms( struct OUTPUTS *OUT, 
						struct VELOCITY v0, struct STRESS t0, struct PARAMETERS PRM,int l );
double Weight3d( double w[3],	/* weights */
		 double v[8] 	/* values */
		 );
#if (VTK)
void write_float(FILE *fp, float val);
#endif
#endif	/* MAIN_H_ */
