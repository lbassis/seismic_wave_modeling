#include "ondes3D_kernels.h"
#include "ondes3D.h"

void printTopo(void) {/*{{{*/
#ifdef USE_MPI
	int nb_char_number = (int) log10(NPROCX*NPROCY-1)+1;
	int a,p;
	int num=0;

	a = (int)floor((8-nb_char_number)/2.);
	p = (int)ceil((8-nb_char_number)/2.);
	
	printf("MPI Topology : \n\n");
	for (int j = NPROCY-1; j>= 0; j--) {
		num = NPROCX*j;
		for (int i = 0; i< NPROCX; i++) {
			printf("+--------");
		}
		printf("+\n");
		for (int i = 0; i< NPROCX; i++) {
			printf("|        ");
		}
		printf("|\n");
		for (int i = 0; i< NPROCX; i++) {
			printf("|");
			for (int n = 0; n< a; n++) printf(" ");
			printf("%d",num);
			for (int n = 0; n< p; n++) printf(" ");
			num++;
		}
		printf("|\n");
		for (int i = 0; i< NPROCX; i++) {
			printf("|        ");
		}
		printf("|\n");
	}
	for (int i = 0; i< NPROCX; i++) {
		printf("+--------");
	}
	printf("+\n\n^\n| Y\n|\n|\n+----->\n   X\n");
#endif
	return;
}/*}}}*/

// PARAM FILE READ FUNCTIONS {{{
void chomp(const char *s)
{
	char *p;
	while (NULL != s && NULL != (p = strrchr(s, '\n')) && NULL != (p = strrchr(s, '\r'))){
		*p = '\0';
	}
} 

void readIntParam(char* entry, int* out, FILE* fd)
{	
	char cur_line[MAX_LINE_SIZE];
	char *name, *equal, *str_value;
	int value;
	char* saveptr;
	
	fseek(fd, 0, 0);
	while (fgets(cur_line, MAX_LINE_SIZE, fd) != NULL) {
		// perl-like chomp to avoid \r & \n problems
		chomp(cur_line);
		if (strlen(cur_line) == 0) continue;
		name = strtok_r(cur_line," =\t", &saveptr);
		if (!strcmp("#", name)) continue;
		if (strcmp(name, entry)) continue;
		str_value = strtok_r(NULL," =\t", &saveptr);
		sscanf(str_value,"%d",&value);
		if (VERBOSE == 3) printf("found %s = %d\n",name,value);
		*out = value;
		return;
	}
	printf("ENTRY %s NOT FOUND !!!\nexiting\n",entry);
#ifdef USE_MPI/*{{{*/
	MPI_Abort(MPI_COMM_WORLD, 1);
#else
	exit(1);
#endif/*}}}*/
}

void readFloatParam(char* entry, float* out, FILE* fd)
{
	char cur_line[MAX_LINE_SIZE];
	char *name, *str_value;
	float value;
	char* saveptr;
	
	fseek(fd, 0, 0);
	while (fgets(cur_line, MAX_LINE_SIZE, fd) != NULL) {
		chomp(cur_line);
		if (strlen(cur_line) == 0) continue;
		name = strtok_r(cur_line," =\t", &saveptr);
		if (!strcmp("#", name)) continue;
		if (strcmp(name, entry)) continue;
		str_value = strtok_r(NULL," =\t", &saveptr);
		sscanf(str_value,"%f",&value);
		if (VERBOSE == 3) printf("found %s = %f\n",name,value);
		*out = value;
		return;
	}
	printf("ENTRY %s NOT FOUND !!!\nexiting\n",entry);
#ifdef USE_MPI/*{{{*/
	MPI_Abort(MPI_COMM_WORLD, 1);
#else
	exit(1);
#endif/*}}}*/
}

void readStringParam(char* entry, char* out, FILE* fd)
{
	char cur_line[MAX_LINE_SIZE];
	char *name, *str_value;
	char* saveptr;
	
	fseek(fd, 0, 0);
	while (fgets(cur_line, MAX_LINE_SIZE, fd) != NULL) {
		chomp(cur_line);
		if (strlen(cur_line) == 0) continue;
		name = strtok_r(cur_line," =\t", &saveptr);
		if (!strcmp("#", name)) continue;
		if (strcmp(name, entry)) continue;
		str_value = strtok_r(NULL," =\t", &saveptr);
		if (VERBOSE == 3) printf("found %s = %s\n",name,str_value);
		strcpy(out, str_value);
		return; 
	}
	printf("ENTRY %s NOT FOUND !!!\nexiting\n",entry);
#ifdef USE_MPI/*{{{*/
	MPI_Abort(MPI_COMM_WORLD, 1);
#else
	exit(1);
#endif/*}}}*/
}

void readLayerParam(char* entry, int num, float* depth, float* vp, float* vs, float* rho, float* q, FILE* fd)
{
	char cur_line[MAX_LINE_SIZE];
	char *name, *str_value;
	char *completed_entry;
	char numero[10];
	char* saveptr;
	
	completed_entry = (char*) malloc (strlen(entry) + 10);
	sprintf(numero, "%d",num);
	strcat(strcpy(completed_entry,entry),numero);
	
	fseek(fd, 0, 0);
	while (fgets(cur_line, MAX_LINE_SIZE, fd) != NULL) {
		chomp(cur_line);
		if (strlen(cur_line) == 0) continue;
		name = strtok_r(cur_line," =\t", &saveptr);
		if (!strcmp("#", name)) continue;
		if (strcmp(name, completed_entry)) continue;
		free(completed_entry);

		str_value = strtok_r(NULL," =\t", &saveptr);
		sscanf(str_value,"%f",depth);
		str_value = strtok_r(NULL," =\t", &saveptr);
		sscanf(str_value,"%f",vp);
		str_value = strtok_r(NULL," =\t", &saveptr);
		sscanf(str_value,"%f",vs);
		str_value = strtok_r(NULL," =\t", &saveptr);
		sscanf(str_value,"%f",rho);
		str_value = strtok_r(NULL," =\t", &saveptr);
		sscanf(str_value,"%f",q);

		if (VERBOSE == 3) printf("found %s > depth: %f, vp: %f, vs: %f, rho: %f, q: %f\n", name, *depth, *vp, *vs, *rho, *q);
		return; 
	}
	printf("ENTRY %s NOT FOUND !!!\nexiting\n",completed_entry);
#ifdef USE_MPI/*{{{*/
	MPI_Abort(MPI_COMM_WORLD, 1);
#else
	exit(1);
#endif/*}}}*/
}
// }}}

// print_err() {{{
void print_err(cudaError_t err, char* err_str)
{   if (err) {
		printf("\n!!! ERROR <%s> thrown for operation : %s\n",cudaGetErrorString(err),err_str);fflush(stdout);
#ifdef USE_MPI
		MPI_Abort(MPI_COMM_WORLD, 2);
#else
		exit(1);
#endif
	     }
}
// }}}

// allocate_arrays_CPU() {{{
void allocate_arrays_CPU(Material *M, Vector_M3D *F, Boundaries *B, int dim_model) {

	// definition des tailles de tableaux
	F->width = M->width = (B->xsup - B->xinf + 1) + 4;
	F->height = M->height = (B->ysup - B->yinf + 1) + 4;
	F->depth = M->depth = (B->zsup - B->zinf + 1) + 4;
	// offset to pass from geographic index to array index
	M->offset_x = F->offset_x = B->xinf-2;
	M->offset_y = F->offset_y = B->yinf-2;
	M->offset_z = F->offset_z = B->zinf-2;

	// memes tailles sur host & device -> moins de sources d'erreurs et copies + rapides
	int pitch = ALIGN_SEGMENT - 2 + M->width;
	pitch += (pitch%ALIGN_SEGMENT)?ALIGN_SEGMENT - (pitch%ALIGN_SEGMENT):0;
	F->pitch = M->pitch = pitch;
	F->offset = M->offset =  ALIGN_SEGMENT - 2;
	
	// allocation des tableaux
	if (dim_model == 1) {
		if (M->depth > CONSTANT_MAX_SIZE) {
			printf("model too deep to run on this 1D version : model depth limited to %d points\nexiting\n",CONSTANT_MAX_SIZE-4);
#ifdef USE_MPI/*{{{*/
			MPI_Abort(MPI_COMM_WORLD, 2);
#else
			exit(2);
#endif/*}}}*/
		}
		M->rho = (float*) malloc (CONSTANT_MAX_SIZE*sizeof(float));
		M->vp = (float*) malloc (CONSTANT_MAX_SIZE*sizeof(float));
		M->vs = (float*) malloc (CONSTANT_MAX_SIZE*sizeof(float));
	} else {
		M->rho = (float*) malloc (M->pitch*M->height*M->depth*sizeof(float));
		M->mu = (float*) malloc (M->pitch*M->height*M->depth*sizeof(float));
		M->lam = (float*) malloc (M->pitch*M->height*M->depth*sizeof(float));
		M->vp = (float*) malloc (M->pitch*M->height*M->depth*sizeof(float));
		M->vs = (float*) malloc (M->pitch*M->height*M->depth*sizeof(float));
	}
#ifndef HOSTALLOC
	F->x = (float*) malloc (F->pitch*F->height*F->depth*sizeof(float));
	F->y = (float*) malloc (F->pitch*F->height*F->depth*sizeof(float));
	F->z = (float*) malloc (F->pitch*F->height*F->depth*sizeof(float));
#else
	print_err(cudaMallocHost((void**)&(F->x),F->pitch*F->height*F->depth*sizeof(float)), "cudaMallocHost F->x");
	print_err(cudaMallocHost((void**)&(F->y),F->pitch*F->height*F->depth*sizeof(float)), "cudaMallocHost F->y");
	print_err(cudaMallocHost((void**)&(F->z),F->pitch*F->height*F->depth*sizeof(float)), "cudaMallocHost F->z");
#endif
	memset(F->x, 0, F->pitch*F->height*F->depth*sizeof(float));
	memset(F->y, 0, F->pitch*F->height*F->depth*sizeof(float));
	memset(F->z, 0, F->pitch*F->height*F->depth*sizeof(float));

	return;
}
// }}}

// allocate_arrays_GPU() {{{
long int allocate_arrays_GPU(Vector_M3D *V, Tensor_M3D *T, Material *M, Vector_M3D *F, Boundaries *B, int dim_model) {

	F->width = M->width = V->width = T->width = (B->xsup - B->xinf + 1) + 4;
	F->height = M->height = V->height = T->height = (B->ysup - B->yinf + 1) + 4;
	F->depth = M->depth = V->depth = T->depth = (B->zsup - B->zinf + 1) + 4;

	int pitch = ALIGN_SEGMENT - 2 + M->width;
	pitch += (pitch%ALIGN_SEGMENT)?ALIGN_SEGMENT - (pitch%ALIGN_SEGMENT):0;
	F->pitch = M->pitch = V->pitch = T->pitch = pitch;

	// a verifier !!!
	V->offset_k = T->offset_k = M->offset_k = F->offset_k = 2*M->pitch*M->height + 2*M->pitch + ALIGN_SEGMENT;
	M->offset = V->offset = T->offset = F->offset = ALIGN_SEGMENT - 2;

	T->offset_x = V->offset_x = F->offset_x = B->xinf-2;
	T->offset_y = V->offset_y = F->offset_y = B->yinf-2;
	T->offset_z = V->offset_z = F->offset_z = B->zinf-2;

	print_err(cudaMalloc((void**) &(V->x), V->pitch*V->height*V->depth*sizeof(float)),"cudaMalloc V->x");
	print_err(cudaMalloc((void**) &(V->y), V->pitch*V->height*V->depth*sizeof(float)),"cudaMalloc V->y");
	print_err(cudaMalloc((void**) &(V->z), V->pitch*V->height*V->depth*sizeof(float)),"cudaMalloc V->z");

	print_err(cudaMalloc((void**) &(F->x), F->pitch*F->height*F->depth*sizeof(float)),"cudaMalloc F->x");
	print_err(cudaMalloc((void**) &(F->y), F->pitch*F->height*F->depth*sizeof(float)),"cudaMalloc F->y");
	print_err(cudaMalloc((void**) &(F->z), F->pitch*F->height*F->depth*sizeof(float)),"cudaMalloc F->z");

	print_err(cudaMalloc((void**) &(T->xx), T->pitch*T->height*T->depth*sizeof(float)),"cudaMalloc T->xx");
	print_err(cudaMalloc((void**) &(T->yy), T->pitch*T->height*T->depth*sizeof(float)),"cudaMalloc T->yy");
	print_err(cudaMalloc((void**) &(T->zz), T->pitch*T->height*T->depth*sizeof(float)),"cudaMalloc T->zz");
	print_err(cudaMalloc((void**) &(T->xy), T->pitch*T->height*T->depth*sizeof(float)),"cudaMalloc T->xy");
	print_err(cudaMalloc((void**) &(T->xz), T->pitch*T->height*T->depth*sizeof(float)),"cudaMalloc T->xz");
	print_err(cudaMalloc((void**) &(T->yz), T->pitch*T->height*T->depth*sizeof(float)),"cudaMalloc T->yz");

	if (dim_model == 3) {
		print_err(cudaMalloc((void**) &(M->rho), M->pitch*M->height*M->depth*sizeof(float)),"cudaMalloc M->rho");
		print_err(cudaMalloc((void**) &(M->mu), M->pitch*M->height*M->depth*sizeof(float)),"cudaMalloc M->mu");
		print_err(cudaMalloc((void**) &(M->lam), M->pitch*M->height*M->depth*sizeof(float)),"cudaMalloc M->lam");
		print_err(cudaMalloc((void**) &(M->vp), M->pitch*M->height*M->depth*sizeof(float)),"cudaMalloc M->vp");
		print_err(cudaMalloc((void**) &(M->vs), M->pitch*M->height*M->depth*sizeof(float)),"cudaMalloc M->vs");
	}
	
	//set them to 0->
	print_err(cudaMemset(V->x,0,V->pitch*V->height*V->depth*sizeof(float)),"cudaMemset V->x");
	print_err(cudaMemset(V->y,0,V->pitch*V->height*V->depth*sizeof(float)),"cudaMemset V->y");
	print_err(cudaMemset(V->z,0,V->pitch*V->height*V->depth*sizeof(float)),"cudaMemset V->z");

	print_err(cudaMemset(F->x,0,F->pitch*F->height*F->depth*sizeof(float)),"cudaMemset F->x");
	print_err(cudaMemset(F->y,0,F->pitch*F->height*F->depth*sizeof(float)),"cudaMemset F->y");
	print_err(cudaMemset(F->z,0,F->pitch*F->height*F->depth*sizeof(float)),"cudaMemset F->z");

	print_err(cudaMemset(T->xx,0,T->pitch*T->height*T->depth*sizeof(float)),"cudaMemset T->xx");
	print_err(cudaMemset(T->yy,0,T->pitch*T->height*T->depth*sizeof(float)),"cudaMemset T->yy");
	print_err(cudaMemset(T->zz,0,T->pitch*T->height*T->depth*sizeof(float)),"cudaMemset T->zz");
	print_err(cudaMemset(T->xy,0,T->pitch*T->height*T->depth*sizeof(float)),"cudaMemset T->xy");
	print_err(cudaMemset(T->xz,0,T->pitch*T->height*T->depth*sizeof(float)),"cudaMemset T->xz");
	print_err(cudaMemset(T->yz,0,T->pitch*T->height*T->depth*sizeof(float)),"cudaMemset T->yz");

	long int mem_used =  V->pitch*V->height*V->depth*sizeof(float) * (dim_model==3?17:12);
	return mem_used;
}
// }}}

// MPI_slicing_and_addressing() {{{
void MPI_slicing_and_addressing (	Comm_info *C, Boundaries *B,
		int XMIN, int XMAX, int YMIN, int YMAX, int ZMIN, int ZMAX) {
	// xmin = borne inf domaine calculable
	// xinf = borne inf domaine + CPML
	// largeur tableau = (x_sup - x_inf + 1) + 4 (ordre 4)
	// largeur domaine = (x_max - x_min + 1)

	int error_code=0;

	// adresse en x et y
	int iproc_y = (int)floorf((float)C->rank/(float)C->nproc_x);
	int iproc_x = C->rank - iproc_y*C->nproc_x;

	// comms
#ifdef USE_MPI
	C->send_xmin = C->recv_xmin = (iproc_x>0)?((iproc_x-1)+iproc_y*C->nproc_x):MPI_PROC_NULL;
	C->send_xmax = C->recv_xmax = (iproc_x<(C->nproc_x-1))?((iproc_x+1)+iproc_y*C->nproc_x):MPI_PROC_NULL;
	C->send_ymin = C->recv_ymin = (iproc_y>0)?((iproc_y-1)*C->nproc_x + iproc_x):MPI_PROC_NULL;
	C->send_ymax = C->recv_ymax = (iproc_y<(C->nproc_y-1))?((iproc_y+1)*C->nproc_x + iproc_x):MPI_PROC_NULL;
#endif

	// calcul des dimensions globales du domaine (sans halo)
	int size_x = XMAX-XMIN + 2*delta + 1;
	int size_y = YMAX-YMIN + 2*delta + 1;

	// dimensions locales
	// en X
	int size_for_all_x = (int)ceilf((float)size_x/(float)C->nproc_x);
	int size_for_the_last_x = size_x - (C->nproc_x-1)*size_for_all_x;
	B->size_x = (iproc_x == (C->nproc_x-1))?size_for_the_last_x:size_for_all_x;

	// en Y
	int size_for_all_y = (int)ceilf((float)size_y/(float)C->nproc_y);
	int size_for_the_last_y = size_y - (C->nproc_y-1)*size_for_all_y;
	B->size_y = (iproc_y == (C->nproc_y-1))?size_for_the_last_y:size_for_all_y;

	// en Z
	B->size_z = ZMAX-ZMIN + delta + 1;

	C->size_buffer_y = B->size_x*B->size_z*2;
	C->size_buffer_x = B->size_y*B->size_z*2;

	// calcul des bornes du domaine avec et sans CPML
	C->first_x = C->last_x = C->first_y = C->last_y = 0;
	// X
	if (C->nproc_x>1) {
		if (iproc_x == 0) { // first slice among x
			C->first_x = 1;
			B->xmin = XMIN;
			B->xmax = XMIN - delta + B->size_x - 1;
			B->xinf = XMIN - delta;
			B->xsup = B->xmax;
		} else if (iproc_x == (C->nproc_x-1)) {// last slice among x
			C->last_x = 1;
			B->xmin = XMIN - delta + (iproc_x)*size_for_all_x;
			B->xmax = B->xmin + B->size_x - delta - 1;
			B->xinf = B->xmin;
			B->xsup = B->xmax + delta;
			if (B->xmax != XMAX) {
				printf("calcul du local xmax faux pour proc %d : \n\tlocal_xmax = %d\n\tXMAX = %d\n\n", C->rank, B->xmax, XMAX);
				error_code = 3;
			}
		} else {
			B->xmin = XMIN - delta + iproc_x*size_for_all_x;
			B->xmax = B->xmin + B->size_x - 1;
			B->xinf = B->xmin;
			B->xsup = B->xmax;
		}
	} else {
		C->first_x = C->last_x = 1;
		B->xmin = XMIN;
		B->xmax = XMAX;
		B->xinf = XMIN-delta;
		B->xsup = XMAX+delta;
	}
	// Y
	if (C->nproc_y>1) {
		if (iproc_y == 0) { // first slice among y
			C->first_y = 1;
			B->ymin = YMIN;
			B->ymax = YMIN - delta + B->size_y - 1;
			B->yinf = YMIN-delta;
			B->ysup = B->ymax;
		} else if (iproc_y == (C->nproc_y-1)) { // last slice among y
			C->last_y = 1;
			B->ymin = YMIN - delta + (iproc_y)*size_for_all_y;
			B->ymax = B->ymin + B->size_y - delta - 1;
			B->yinf = B->ymin;
			B->ysup = B->ymax + delta;
			if (B->ymax != YMAX) {
				printf("calcul de local_ymax faux pour proc %d : \n\tlocal_ymax = %d\n\tYMAX = %d\n\n",C->rank,B->ymax, YMAX);
				error_code = 4;
			}
		} else {
			B->ymin = YMIN - delta + (iproc_y)*size_for_all_y;
			B->ymax = B->ymin + B->size_y - 1;
			B->yinf = B->ymin;
			B->ysup = B->ymax;
		}
	} else {
		C->first_y = C->last_y = 1;
		B->ymin = YMIN;
		B->ymax = YMAX;
		B->yinf = YMIN-delta;
		B->ysup = YMAX+delta;
	}
	// Z
	B->zmax = B->zsup = ZMAX;
	B->zmin = ZMIN;
	B->zinf = ZMIN - delta;

	assert(B->size_x == (B->xsup-B->xinf+1));
	assert(B->size_y == (B->ysup-B->yinf+1));
	assert(B->size_z == (B->zsup-B->zinf+1));

	if (VERBOSE >= 2) {
		printf("BORNES : \n");
		printf("xsup : %d\n",B->xsup);
		printf("xinf : %d\n",B->xinf);
		printf("ysup : %d\n",B->ysup);
		printf("yinf : %d\n",B->yinf);
		printf("xmax : %d\n",B->xmax);
		printf("xmin : %d\n",B->xmin);
		printf("ymax : %d\n",B->ymax);
		printf("ymin : %d\n",B->ymin);
	}
	if (error_code) {
#ifdef USE_MPI
		MPI_Abort(MPI_COMM_WORLD,error_code);
#else
		exit(error_code);
#endif
	}

	return;
}
// }}}

// copy_material_to_GPU() {{{
void copy_material_to_GPU(Material *d_M, Material *M, int dim_model) {

	if (dim_model == 3) {
		print_err(cudaMemcpy((void*) d_M->mu, M->mu, M->pitch*M->height*M->depth*sizeof(float), cudaMemcpyHostToDevice),"cudaMemcpy M->mu -> d_M->mu");
		print_err(cudaMemcpy((void*) d_M->rho, M->rho, M->pitch*M->height*M->depth*sizeof(float), cudaMemcpyHostToDevice),"cudaMemcpy M->rho -> d_M->rho");
		print_err(cudaMemcpy((void*) d_M->vp, M->vp, M->pitch*M->height*M->depth*sizeof(float), cudaMemcpyHostToDevice),"cudaMemcpy M->vp -> d_M->vp");
		print_err(cudaMemcpy((void*) d_M->vs, M->vs, M->pitch*M->height*M->depth*sizeof(float), cudaMemcpyHostToDevice),"cudaMemcpy M->vs -> d_M->vs");
		print_err(cudaMemcpy((void*) d_M->lam, M->lam, M->pitch*M->height*M->depth*sizeof(float), cudaMemcpyHostToDevice),"cudaMemcpy M->lam -> d_M->lam");
	} else {
		setConstRho(M->rho, CONSTANT_MAX_SIZE);
		setConstVp(M->vp, CONSTANT_MAX_SIZE);
		setConstVs(M->vs, CONSTANT_MAX_SIZE);	
	}
	return;
}
// }}}

// create_CPML_indirection(){{{
int create_CPML_indirection(int **p_d_npml_tab, Boundaries B, int* npmlv, int sizex, int sizey, int sizez){
	int i,j,k;
	int* npml_tab = (int*)malloc(B.size_x*B.size_y*B.size_z*sizeof(int));
	*npmlv = 0;
	for(k=0;k<B.size_z;k++) {
		for(j=0;j<B.size_y;j++) {
			for(i=0;i<B.size_x;i++) {
				if (i<(B.xmin-B.xinf) || i>=(B.size_x-(B.xsup-B.xmax)) || j<(B.ymin-B.yinf) || j>=(B.size_y-(B.ysup-B.ymax)) || k<(B.zmin-B.zinf)) {// in CPML
					npml_tab[k*B.size_y*B.size_x + j*B.size_x + i] = (*npmlv)++;
				} else { // not in CPML
					npml_tab[k*B.size_y*B.size_x + j*B.size_x + i] = -1;
				}
			}
		}
	}
#ifndef USE_MPI
	assert(*npmlv == B.size_x*B.size_y*delta + 2*(B.size_z-delta)*B.size_x*delta + 2*(B.size_z-delta)*(B.size_y-2*delta)*delta);
#else
	int sum, myrank;
	MPI_Reduce ( npmlv, &sum, 1, MPI_INTEGER, MPI_SUM, MASTER, MPI_COMM_WORLD);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
	if (myrank == MASTER) {
		assert(sum == sizex*sizey*delta + 2*(sizez-delta)*sizex*delta + 2*(sizez-delta)*(sizey-2*delta)*delta);
	}
#endif
	print_err(cudaMalloc((void**) p_d_npml_tab, B.size_x*B.size_y*B.size_z*sizeof(int)),"cudaMalloc d_npml_tab");
	print_err(cudaMemcpy((void*) (*p_d_npml_tab), npml_tab, B.size_x*B.size_y*B.size_z*sizeof(int), cudaMemcpyHostToDevice),"cudaMemcpy npml_tab -> d_npml_tab");
	free(npml_tab);

	return (B.size_x*B.size_y*B.size_z*sizeof(int));
}
// }}}

// allocate_CPML_arrays() {{{
long int allocate_CPML_arrays(int npmlv, float **p_d_phivxx, float **p_d_phivxy, float **p_d_phivxz, float **p_d_phivyx, float **p_d_phivyy, float **p_d_phivyz, float **p_d_phivzx, float **p_d_phivzy, float **p_d_phivzz, 
		float **p_d_phitxxx, float **p_d_phitxyy, float **p_d_phitxzz, float **p_d_phitxyx, float **p_d_phityyy, float **p_d_phityzz, float **p_d_phitxzx, float **p_d_phityzy, float **p_d_phitzzz) {
	// CORRECTED
	long int memory_used = 18*npmlv*sizeof(float);

	print_err(cudaMalloc((void**) p_d_phivxx, npmlv*sizeof(float)),"cudaMalloc d_phivxx");
	print_err(cudaMalloc((void**) p_d_phivxy, npmlv*sizeof(float)),"cudaMalloc d_phivxy");
	print_err(cudaMalloc((void**) p_d_phivxz, npmlv*sizeof(float)),"cudaMalloc d_phivxz");
	print_err(cudaMalloc((void**) p_d_phivyx, npmlv*sizeof(float)),"cudaMalloc d_phivyx");
	print_err(cudaMalloc((void**) p_d_phivyy, npmlv*sizeof(float)),"cudaMalloc d_phivyy");
	print_err(cudaMalloc((void**) p_d_phivyz, npmlv*sizeof(float)),"cudaMalloc d_phivyz");
	print_err(cudaMalloc((void**) p_d_phivzx, npmlv*sizeof(float)),"cudaMalloc d_phivzx");
	print_err(cudaMalloc((void**) p_d_phivzy, npmlv*sizeof(float)),"cudaMalloc d_phivzy");
	print_err(cudaMalloc((void**) p_d_phivzz, npmlv*sizeof(float)),"cudaMalloc d_phivzz");

	print_err(cudaMalloc((void**) p_d_phitxxx, npmlv*sizeof(float)),"cudaMalloc d_phitxxx");
	print_err(cudaMalloc((void**) p_d_phitxyy, npmlv*sizeof(float)),"cudaMalloc d_phitxyy");
	print_err(cudaMalloc((void**) p_d_phitxzz, npmlv*sizeof(float)),"cudaMalloc d_phitxzz");
	print_err(cudaMalloc((void**) p_d_phitxyx, npmlv*sizeof(float)),"cudaMalloc d_phitxyx");
	print_err(cudaMalloc((void**) p_d_phityyy, npmlv*sizeof(float)),"cudaMalloc d_phityyy");
	print_err(cudaMalloc((void**) p_d_phityzz, npmlv*sizeof(float)),"cudaMalloc d_phityzz");
	print_err(cudaMalloc((void**) p_d_phitxzx, npmlv*sizeof(float)),"cudaMalloc d_phitxzx");
	print_err(cudaMalloc((void**) p_d_phityzy, npmlv*sizeof(float)),"cudaMalloc d_phityzy");
	print_err(cudaMalloc((void**) p_d_phitzzz, npmlv*sizeof(float)),"cudaMalloc d_phitzzz");

	print_err(cudaMemset(*(p_d_phivxx), 0, npmlv*sizeof(float)),"cudaMemset *(p_d_phivxx");
	print_err(cudaMemset(*(p_d_phivxy), 0, npmlv*sizeof(float)),"cudaMemset *(p_d_phivxy");
	print_err(cudaMemset(*(p_d_phivxz), 0, npmlv*sizeof(float)),"cudaMemset *(p_d_phivxz");
	print_err(cudaMemset(*(p_d_phivyx), 0, npmlv*sizeof(float)),"cudaMemset *(p_d_phivyx");
	print_err(cudaMemset(*(p_d_phivyy), 0, npmlv*sizeof(float)),"cudaMemset *(p_d_phivyy");
	print_err(cudaMemset(*(p_d_phivyz), 0, npmlv*sizeof(float)),"cudaMemset *(p_d_phivyz");
	print_err(cudaMemset(*(p_d_phivzx), 0, npmlv*sizeof(float)),"cudaMemset *(p_d_phivzx");
	print_err(cudaMemset(*(p_d_phivzy), 0, npmlv*sizeof(float)),"cudaMemset *(p_d_phivzy");
	print_err(cudaMemset(*(p_d_phivzz), 0, npmlv*sizeof(float)),"cudaMemset *(p_d_phivzz");

	print_err(cudaMemset(*(p_d_phitxxx), 0, npmlv*sizeof(float)),"cudaMemset *(p_d_phitxxx");
	print_err(cudaMemset(*(p_d_phitxyy), 0, npmlv*sizeof(float)),"cudaMemset *(p_d_phitxyy");
	print_err(cudaMemset(*(p_d_phitxzz), 0, npmlv*sizeof(float)),"cudaMemset *(p_d_phitxzz");
	print_err(cudaMemset(*(p_d_phitxyx), 0, npmlv*sizeof(float)),"cudaMemset *(p_d_phitxyx");
	print_err(cudaMemset(*(p_d_phityyy), 0, npmlv*sizeof(float)),"cudaMemset *(p_d_phityyy");
	print_err(cudaMemset(*(p_d_phityzz), 0, npmlv*sizeof(float)),"cudaMemset *(p_d_phityzz");
	print_err(cudaMemset(*(p_d_phitxzx), 0, npmlv*sizeof(float)),"cudaMemset *(p_d_phitxzx");
	print_err(cudaMemset(*(p_d_phityzy), 0, npmlv*sizeof(float)),"cudaMemset *(p_d_phityzy");
	print_err(cudaMemset(*(p_d_phitzzz), 0, npmlv*sizeof(float)),"cudaMemset *(p_d_phitzzz");

	return memory_used;
}
// }}}

// allocate_CPML_vectors() {{{
long int allocate_CPML_vectors(Boundaries B,	float** p_d_dumpx, float** p_d_alphax, float** p_d_kappax, float** p_d_dumpx2, float** p_d_alphax2, float** p_d_kappax2,
		float** p_d_dumpy, float** p_d_alphay, float** p_d_kappay, float** p_d_dumpy2, float** p_d_alphay2, float** p_d_kappay2,
		float** p_d_dumpz, float** p_d_alphaz, float** p_d_kappaz, float** p_d_dumpz2, float** p_d_alphaz2, float** p_d_kappaz2,
		float* dumpx, float* alphax, float* kappax, float* dumpx2, float* alphax2, float* kappax2,
		float* dumpy, float* alphay, float* kappay, float* dumpy2, float* alphay2, float* kappay2,
		float* dumpz, float* alphaz, float* kappaz, float* dumpz2, float* alphaz2, float* kappaz2) {

	// CORRECTED
	long int mem_used = 6*(B.size_x+B.size_y+B.size_z)*sizeof(float);

	// declaration, allocation and copies of dump*, alpha* & kappa* arrays
	print_err(cudaMalloc((void**) p_d_dumpx, B.size_x*sizeof(float)),"cudaMalloc d_dumpx");
	print_err(cudaMalloc((void**) p_d_alphax, B.size_x*sizeof(float)),"cudaMalloc d_alphax");
	print_err(cudaMalloc((void**) p_d_kappax, B.size_x*sizeof(float)),"cudaMalloc d_kappax");
	print_err(cudaMalloc((void**) p_d_dumpx2, B.size_x*sizeof(float)),"cudaMalloc d_dumpx2");
	print_err(cudaMalloc((void**) p_d_alphax2, B.size_x*sizeof(float)),"cudaMalloc d_alphax2");
	print_err(cudaMalloc((void**) p_d_kappax2, B.size_x*sizeof(float)),"cudaMalloc d_kappax2");

	print_err(cudaMalloc((void**) p_d_dumpy, B.size_y*sizeof(float)),"cudaMalloc d_dumpy");
	print_err(cudaMalloc((void**) p_d_alphay, B.size_y*sizeof(float)),"cudaMalloc d_alphay");
	print_err(cudaMalloc((void**) p_d_kappay, B.size_y*sizeof(float)),"cudaMalloc d_kappay");
	print_err(cudaMalloc((void**) p_d_dumpy2, B.size_y*sizeof(float)),"cudaMalloc d_dumpy2");
	print_err(cudaMalloc((void**) p_d_alphay2, B.size_y*sizeof(float)),"cudaMalloc d_alphay2");
	print_err(cudaMalloc((void**) p_d_kappay2, B.size_y*sizeof(float)),"cudaMalloc d_kappay2");

	print_err(cudaMalloc((void**) p_d_dumpz, B.size_z*sizeof(float)),"cudaMalloc d_dumpz");
	print_err(cudaMalloc((void**) p_d_alphaz, B.size_z*sizeof(float)),"cudaMalloc d_alphaz");
	print_err(cudaMalloc((void**) p_d_kappaz, B.size_z*sizeof(float)),"cudaMalloc d_kappaz");
	print_err(cudaMalloc((void**) p_d_dumpz2, B.size_z*sizeof(float)),"cudaMalloc d_dumpz2");
	print_err(cudaMalloc((void**) p_d_alphaz2, B.size_z*sizeof(float)),"cudaMalloc d_alphaz2");
	print_err(cudaMalloc((void**) p_d_kappaz2, B.size_z*sizeof(float)),"cudaMalloc d_kappaz2");

	print_err(cudaMemcpy((void*) (*p_d_dumpx), dumpx, B.size_x*sizeof(float), cudaMemcpyHostToDevice),"cudaMemcpy dumpx -> (*p_d_dumpx");
	print_err(cudaMemcpy((void*) (*p_d_alphax), alphax, B.size_x*sizeof(float), cudaMemcpyHostToDevice),"cudaMemcpy alphax -> (*p_d_alphax");
	print_err(cudaMemcpy((void*) (*p_d_kappax), kappax, B.size_x*sizeof(float), cudaMemcpyHostToDevice),"cudaMemcpy kappax -> (*p_d_kappax");
	print_err(cudaMemcpy((void*) (*p_d_dumpx2), dumpx2, B.size_x*sizeof(float), cudaMemcpyHostToDevice),"cudaMemcpy dumpx2 -> (*p_d_dumpx2");
	print_err(cudaMemcpy((void*) (*p_d_alphax2), alphax2, B.size_x*sizeof(float), cudaMemcpyHostToDevice),"cudaMemcpy alphax2 -> (*p_d_alphax2");
	print_err(cudaMemcpy((void*) (*p_d_kappax2), kappax2, B.size_x*sizeof(float), cudaMemcpyHostToDevice),"cudaMemcpy kappax2 -> (*p_d_kappax2");

	print_err(cudaMemcpy((void*) (*p_d_dumpy), dumpy, B.size_y*sizeof(float), cudaMemcpyHostToDevice),"cudaMemcpy dumpy -> (*p_d_dumpy");
	print_err(cudaMemcpy((void*) (*p_d_alphay), alphay, B.size_y*sizeof(float), cudaMemcpyHostToDevice),"cudaMemcpy alphay -> (*p_d_alphay");
	print_err(cudaMemcpy((void*) (*p_d_kappay), kappay, B.size_y*sizeof(float), cudaMemcpyHostToDevice),"cudaMemcpy kappay -> (*p_d_kappay");
	print_err(cudaMemcpy((void*) (*p_d_dumpy2), dumpy2, B.size_y*sizeof(float), cudaMemcpyHostToDevice),"cudaMemcpy dumpy2 -> (*p_d_dumpy2");
	print_err(cudaMemcpy((void*) (*p_d_alphay2), alphay2, B.size_y*sizeof(float), cudaMemcpyHostToDevice),"cudaMemcpy alphay2 -> (*p_d_alphay2");
	print_err(cudaMemcpy((void*) (*p_d_kappay2), kappay2, B.size_y*sizeof(float), cudaMemcpyHostToDevice),"cudaMemcpy kappay2 -> (*p_d_kappay2");

	print_err(cudaMemcpy((void*) (*p_d_dumpz), dumpz, B.size_z*sizeof(float), cudaMemcpyHostToDevice),"cudaMemcpy dumpz -> (*p_d_dumpz");
	print_err(cudaMemcpy((void*) (*p_d_alphaz), alphaz, B.size_z*sizeof(float), cudaMemcpyHostToDevice),"cudaMemcpy alphaz -> (*p_d_alphaz");
	print_err(cudaMemcpy((void*) (*p_d_kappaz), kappaz, B.size_z*sizeof(float), cudaMemcpyHostToDevice),"cudaMemcpy kappaz -> (*p_d_kappaz");
	print_err(cudaMemcpy((void*) (*p_d_dumpz2), dumpz2, B.size_z*sizeof(float), cudaMemcpyHostToDevice),"cudaMemcpy dumpz2 -> (*p_d_dumpz2");
	print_err(cudaMemcpy((void*) (*p_d_alphaz2), alphaz2, B.size_z*sizeof(float), cudaMemcpyHostToDevice),"cudaMemcpy alphaz2 -> (*p_d_alphaz2");
	print_err(cudaMemcpy((void*) (*p_d_kappaz2), kappaz2, B.size_z*sizeof(float), cudaMemcpyHostToDevice),"cudaMemcpy kappaz2 -> (*p_d_kappaz2");

	// texture binding
	bindTexturesCpmlVector( (*p_d_dumpx), (*p_d_alphax), (*p_d_kappax), (*p_d_dumpx2), (*p_d_alphax2), (*p_d_kappax2),
			(*p_d_dumpy), (*p_d_alphay), (*p_d_kappay), (*p_d_dumpy2), (*p_d_alphay2), (*p_d_kappay2),
			(*p_d_dumpz), (*p_d_alphaz), (*p_d_kappaz), (*p_d_dumpz2), (*p_d_alphaz2), (*p_d_kappaz2),
			B.size_x, B.size_y, B.size_z);
	return mem_used;
}
// }}}

// allocate_MPI_buffers() {{{
#ifdef USE_MPI/*{{{*/
long int allocate_MPI_buffers(Comm_info *C) {
	// for the host : should use cudaMallocHost
	C->buff_x_min_s = (float*)malloc(6*C->size_buffer_x*sizeof(float));
	C->buff_x_max_s = (float*)malloc(6*C->size_buffer_x*sizeof(float));
	C->buff_y_min_s = (float*)malloc(6*C->size_buffer_y*sizeof(float));
	C->buff_y_max_s = (float*)malloc(6*C->size_buffer_y*sizeof(float));

	C->buff_x_min_r = (float*)malloc(6*C->size_buffer_x*sizeof(float));
	C->buff_x_max_r = (float*)malloc(6*C->size_buffer_x*sizeof(float));
	C->buff_y_min_r = (float*)malloc(6*C->size_buffer_y*sizeof(float));
	C->buff_y_max_r = (float*)malloc(6*C->size_buffer_y*sizeof(float));

	// for the device
	print_err(cudaMalloc((void**) &(C->d_buff_x_min), 6*C->size_buffer_x*sizeof(float)),"cudaMalloc d_buff_x_min");
	print_err(cudaMalloc((void**) &(C->d_buff_x_max), 6*C->size_buffer_x*sizeof(float)),"cudaMalloc d_buff_x_max");
	print_err(cudaMalloc((void**) &(C->d_buff_y_min), 6*C->size_buffer_y*sizeof(float)),"cudaMalloc d_buff_y_min");
	print_err(cudaMalloc((void**) &(C->d_buff_y_max), 6*C->size_buffer_y*sizeof(float)),"cudaMalloc d_buff_y_max");

	return (long int)((12*C->size_buffer_x + 12*C->size_buffer_y)*sizeof(float));
}
#endif/*}}}*/
// }}}

// free_arrays_CPU() {{{
void free_arrays_CPU(Material *M, Vector_M3D *F, int dim_model) {

	free(M->rho);
	free(M->vp);
	free(M->vs);
	if (dim_model == 3) {
		free(M->mu);
		free(M->lam);
	} 

#ifndef HOSTALLOC
	free(F->x);
	free(F->y);
	free(F->z);
#else
	cudaFreeHost(F->x);
	cudaFreeHost(F->y);
	cudaFreeHost(F->z);
#endif
	return;
}
// }}}

// free_arrays_GPU() {{{
void free_arrays_GPU(Vector_M3D *V, Tensor_M3D *T, Material *M, Vector_M3D *F, int *d_npml_tab, int dim_model) {

	cudaFree(V->x);
	cudaFree(V->y);
	cudaFree(V->z);

	cudaFree(F->x);
	cudaFree(F->y);
	cudaFree(F->z);

	cudaFree(T->xx);
	cudaFree(T->yy);
	cudaFree(T->zz);
	cudaFree(T->xy);
	cudaFree(T->xz);
	cudaFree(T->yz);

	if (dim_model == 3) {
		cudaFree(M->rho);
		cudaFree(M->mu);
		cudaFree(M->lam);
		cudaFree(M->vp);
		cudaFree(M->vs);
	}
	cudaFree(d_npml_tab);
	return;
}
// }}}

// free_CPML_data{{{
void free_CPML_data(float *d_phivxx, float *d_phivxy, float *d_phivxz, float *d_phivyx, float *d_phivyy, float *d_phivyz, float *d_phivzx, float *d_phivzy, float *d_phivzz, 
		float *d_phitxxx, float *d_phitxyy, float *d_phitxzz, float *d_phitxyx, float *d_phityyy, float *d_phityzz, float *d_phitxzx, float *d_phityzy, float *d_phitzzz,
		float* d_dumpx, float* d_alphax, float* d_kappax, float* d_dumpx2, float* d_alphax2, float* d_kappax2,
		float* d_dumpy, float* d_alphay, float* d_kappay, float* d_dumpy2, float* d_alphay2, float* d_kappay2,
		float* d_dumpz, float* d_alphaz, float* d_kappaz, float* d_dumpz2, float* d_alphaz2, float* d_kappaz2) {

	cudaFree(d_phivxx);
	cudaFree(d_phivxy);
	cudaFree(d_phivxz);
	cudaFree(d_phivyx);
	cudaFree(d_phivyy);
	cudaFree(d_phivyz);
	cudaFree(d_phivzx);
	cudaFree(d_phivzy);
	cudaFree(d_phivzz);

	cudaFree(d_phitxxx);
	cudaFree(d_phitxyy);
	cudaFree(d_phitxzz);
	cudaFree(d_phitxyx);
	cudaFree(d_phityyy);
	cudaFree(d_phityzz);
	cudaFree(d_phitxzx);
	cudaFree(d_phityzy);
	cudaFree(d_phitzzz);

	cudaFree(d_dumpx);
	cudaFree(d_alphax);
	cudaFree(d_kappax);
	cudaFree(d_dumpx2);
	cudaFree(d_alphax2);
	cudaFree(d_kappax2);

	cudaFree(d_dumpy);
	cudaFree(d_alphay);
	cudaFree(d_kappay);
	cudaFree(d_dumpy2);
	cudaFree(d_alphay2);
	cudaFree(d_kappay2);

	cudaFree(d_dumpz);
	cudaFree(d_alphaz);
	cudaFree(d_kappaz);
	cudaFree(d_dumpz2);
	cudaFree(d_alphaz2);
	cudaFree(d_kappaz2);
	return;
}
// }}}

// free_MPI_buffers() {{{
void free_MPI_buffers(Comm_info *C) {
	// for the host
	free(C->buff_x_min_s);
	free(C->buff_x_max_s);
	free(C->buff_y_min_s);
	free(C->buff_y_max_s);

	free(C->buff_x_min_r);
	free(C->buff_x_max_r);
	free(C->buff_y_min_r);
	free(C->buff_y_max_r);

	// for the device
	cudaFree(C->d_buff_x_min);
	cudaFree(C->d_buff_x_max);
	cudaFree(C->d_buff_y_min);
	cudaFree(C->d_buff_y_max);
	return;
}
// }}}
