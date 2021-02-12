/* Definition of Input Function */

int double2int(double);

double radxx(double, double, double);
double radyy(double, double, double);
double radzz(double, double, double);
double radxy(double, double, double);
double radyz(double, double, double);
double radxz(double, double, double);

double staggardv4 (double, double, double, double, double, double,
	double, double, double, double,
	double, double, double, double,
	double, double, double, double );

double staggardv2 (double, double, double, double, double, double,
	double, double,
	double, double,
	double, double );

double staggards4 (double, double, double, double, double, double, double,
	double, double, double, double,
	double, double, double, double,
	double, double, double, double );

double staggards2 (double, double, double, double, double, double, double,
	double, double,
	double, double,
	double, double );

double staggardt4 (double, double, double, double, double,
	double, double, double, double,
	double, double, double, double );

double staggardt2 (double, double, double, double, double,
	double, double,
	double, double );

double CPML4 (double, double, double, double, double, double, double,
    double, double, double, double );

double CPML2 (double, double, double, double, double, double, double,
    double, double );

double my_second();

#if  (VTK)
void force_big_endian(unsigned char *bytes);
void write_float(FILE *fp, float val);
#endif

