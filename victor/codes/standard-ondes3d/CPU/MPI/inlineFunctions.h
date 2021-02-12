#ifndef INLINE_FUNCTIONS_H_
#define INLINE_FUNCTIONS_H_
#include <math.h>
#include <stdio.h>

#include "struct.h"
#include "assert.h"

#include "options.h" 


static const double PRECISION_INVERSE=1.e-6;


/*============= convert Functions=========================== */

static inline double RhoMuKap2Vp(double rho, double mu, double kap){
  return sqrt( (kap + 4./3.* mu)/rho ); 
}

static inline double RhoMu2Vs(double rho, double mu){
  return sqrt( mu/rho ); 
}

static inline void RhoVpVs2MuKap( double rho, double vp, double vs, 
                           double *mu, double *kap ){
  *mu = rho * vs * vs;
  *kap = rho * vp*vp  - (4./3.)* (*mu) ;
}


/*============= Place Functions=========================== */
static inline enum typePlace WhereAmI( 		
							   int i, int j, int k, 
							   struct PARAMETERS PRM )
{

  /* mapping PRM */
  const int DELTA=PRM.delta;

  const int XMIN=PRM.xMin;
  const int XMAX=PRM.xMax;
  const int YMIN=PRM.yMin;
  const int YMAX=PRM.yMax;
  const int ZMIN=PRM.zMin;
  const int ZMAX=PRM.zMax;
  const int ZMAX0=PRM.zMax0;


  /* Descriptions :
   *  - domain not centered :  [xMin-DELTA:xMax+DELTA+2] x [yMin-DELTA:yMax+DELTA+2] x [zMin-DELTA:zMax+DELTA+2]
   *  i0=i-1,j0=j-1,k0=k-1  (0 indicates real domain) 
   *  - the 2 last cells of the domain is considered in the border
   *  - CPML :  We do not compute the 2 last cells of the CPML (because of 4th Order approximation
   *  - Free surface :  We compute +1 cell on z axis if Free Surface

   *  Changing algorithm :
   *  xMin-DELTA-1 <= i0 <= xMax+DELTA+1
   *  (last cell of domain is not seen in this algorithm)
   *   xMin-DELTA+1 <= i0 <= xMax+DELTA-1	: CPML we remove the 2 cells
   *   xMin-DELTA+2 <= i  <= xMax+DELTA   : Domain is not centered 
   */
  /* I am OutSide */
  if (i < XMIN-DELTA || i > XMAX+DELTA+2 || 
      j < YMIN-DELTA || j > YMAX+DELTA+2 ||
      k < ZMIN-DELTA || k > ZMAX0 ){ return OUTSIDE; }
  else { 			/* not outside */

    if ( model == GEOLOGICAL ){
      if((i < XMIN-DELTA+2 || i > XMAX+DELTA) ||
         (j < YMIN-DELTA+2 || j > YMAX+DELTA) ||
         (k < ZMIN-DELTA+2 || k > ZMAX   )  )
        {return LIMIT;}
      else{
        if( i <= XMIN+1 || i >= XMAX+1 || 
            j <= YMIN+1 || j >= YMAX+1 ||
            k <= ZMIN+1                )
          {return ABSORBINGLAYER;}
        else{ return REGULAR; }
      }	/* no limit */

    }else if ( model == LAYER ){
    if (surface == ABSORBING  ){	/* absorbing layer above case */
        
      if((i < XMIN-DELTA+2 || i > XMAX+DELTA) ||
           (j < YMIN-DELTA+2 || j > YMAX+DELTA) ||
           (k < ZMIN-DELTA+2 || k > ZMAX0-2   )  )
          {return LIMIT;}
        else{
          if( i <= XMIN+1 || i >= XMAX+1 || 
              j <= YMIN+1 || j >= YMAX+1 ||
              k <= ZMIN+1 || k >= ZMAX+1 )
            {return ABSORBINGLAYER;}
          else{ return REGULAR; }
        }	/* no limit */
      } /* no absorbing surface */
  
    else if ( surface == FREE ){     /* free surface case */
      if( (i < XMIN-DELTA+2 || i > XMAX+DELTA) ||
		  (j < YMIN-DELTA+2 || j > YMAX+DELTA) ||
		  (k < ZMIN-DELTA+2                    )) 
		{return LIMIT; }
      else{
		if ( (k == ZMAX0-1) || (k == ZMAX0) ){ /* Free Surface and Absorbing Layer */
		  if ( i <= XMIN+1 || i >= XMAX+1 || 
			   j <= YMIN+1 || j >= YMAX+1 )
			{return FREEABS;} 
		  else { return FREESURFACE; }
		} /* end free surface & absorbing layer+Freesurface=FreeAbs */
		else{
		  if(   i <= XMIN+1 || i >= XMAX+1 || 
				j <= YMIN+1 || j >= YMAX+1 ||
				k <= ZMIN+1                )
			{ return ABSORBINGLAYER;}
		  else { return REGULAR; }
	
		}/* no freesurface or FreeAbs*/
      }	/* no limit */
	} /* no freesurface */
    } /* end if model */
  } /* no outside */
}/* end Function WhereAmI */

 


/* =====IndexL */
/* GOAL : find the right index in the Kristek & Moczo anelasticity Method
   Status : Used
*/
static inline int IndexL(int i, int j, int k){
  return (abs(j)%2)  * ( 1 + abs(k-1)%2 + 2*( abs(i-1)%2 ) )
    + (1-(abs(j)%2)) * ( 1 + abs(k)%2   + 2*( abs(i)%2   ) ) ;
}

/*============= Averaging Functions=========================== */
/* Description :
 * When averaging, we are strict with the values except when the model is GEOLOGICAL (there, 0. => reflections at FreeSurface )
 */
static inline double averageInverseDouble8( double value1, double value2,  double value3,  double value4,/* Status : Used */
									 double value5, double value6,  double value7,  double value8,
									 enum typeModel model
									 ) 
{
  assert ( (model == LAYER || model == GEOLOGICAL) );
  if (model == LAYER){
    assert ( value1 > PRECISION_INVERSE
			 && value2 > PRECISION_INVERSE
			 && value3 > PRECISION_INVERSE
			 && value4 > PRECISION_INVERSE
			 && value5 > PRECISION_INVERSE
			 && value6 > PRECISION_INVERSE
			 && value7 > PRECISION_INVERSE
			 && value8 > PRECISION_INVERSE);
  }else if ( model == GEOLOGICAL ){
    if ( value1 <= PRECISION_INVERSE
		 || value2 <= PRECISION_INVERSE
		 || value3 <= PRECISION_INVERSE
		 || value4 <= PRECISION_INVERSE
		 || value5 <= PRECISION_INVERSE
		 || value6 <= PRECISION_INVERSE
		 || value7 <= PRECISION_INVERSE
		 || value8 <= PRECISION_INVERSE){
      return 0.;
    }
  }
  return (8./( 1./value1 + 1./value2 +1./value3 + 1./value4 +  1./value5 + 1./value6 +1./value7 + 1./value8 ));

}

static inline double averageInverseDouble4( double value1, double value2,  double value3,  double value4,
									 enum typeModel model ) /* Status : Used */
{
  assert ( (model == LAYER || model == GEOLOGICAL) );
  if (model == LAYER){
    assert ( value1 > PRECISION_INVERSE
			 && value2 > PRECISION_INVERSE
			 && value3 > PRECISION_INVERSE
			 && value4 > PRECISION_INVERSE);
  }else if ( model == GEOLOGICAL ){
    if ( value1 <= PRECISION_INVERSE
		 || value2 <= PRECISION_INVERSE
		 || value3 <= PRECISION_INVERSE
		 || value4 <= PRECISION_INVERSE){
      return 0.;
    }
  }
  return (4./( 1./value1 + 1./value2 +1./value3 + 1./value4 )) ;
}

static inline double averageInverseDouble2(double value1, double value2,
									enum typeModel model ) /* Status : Used */
{
  assert ( (model == LAYER || model == GEOLOGICAL) );
  if (model == LAYER){
    assert ( value1 > PRECISION_INVERSE
			 && value2 > PRECISION_INVERSE);
  }else if ( model == GEOLOGICAL ){
    if ( value1 <= PRECISION_INVERSE
		 || value2 <= PRECISION_INVERSE){
      return 0.;
    }
  }
  return 2./( 1./(value1)  + 1./(value2)  ) ;
}

static inline double CornerXYZ_GeolInverse(double *field,
									int i, int j, int k,
									struct MEDIUM MDM  ) 
{ 
  int idum, jdum , kdum ;
  double res=0.;
  const int NUM_VOID=0;
  int med;
  for ( idum = 0; idum <= 1; idum++ ){
	for ( jdum = 0; jdum <= 1; jdum++ ){
	  for ( kdum = 0; kdum <= 1; kdum++ ){

        med = MDM.imed[ i + idum][ j + jdum][ k + kdum]; 
		if ( field[med] > PRECISION_INVERSE  ){ 
		  res += 1./field[med]; 
		} else{
		  return 0.;
		}
	  }
	}
  }
  res = 8./res; 
  return res;
}

static inline double CornerXYZ_Geol(double *field,
							 int i, int j, int k,
							 struct MEDIUM MDM  ) 
{ 
  int idum, jdum , kdum ;
  double res=0.;
  int med;
  for ( idum = 0; idum <= 1; idum++ ){
	for ( jdum = 0; jdum <= 1; jdum++ ){
	  for ( kdum = 0; kdum <= 1; kdum++ ){

        med = MDM.imed[ i + idum][ j + jdum][ k + kdum];

        res += field[med];
  
	  }
	}
  }
  res = res/8.; 
  return res;
}



/*============= increment Functions=========================== */
/* Description :
 * CPML => modify derivatives througth kappax/y/z and alphax/y/z
 * those functions will ignore kappax and alphax inside normal Domain
 */
static inline double staggardv4 ( double b,
						   double kappax,  double kappay,  double kappaz,
						   double dt,  double dx,
						   double x1,  double x2,  double x3,  double x4,
						   double y1,  double y2,  double y3,  double y4,
						   double z1,  double z2,  double z3,  double z4,
						   enum typePlace place ,  enum typePML method)
{
  if ( (method == CPML )&&(place == ABSORBINGLAYER || place == FREEABS) ) {
    return (9.*b*dt/8.)*( (x2 - x1)/kappax + (y2 - y1)/kappay + (z2 - z1)/kappaz )/dx
      - (b*dt/24.)*( (x4 - x3)/kappax + (y4 - y3)/kappay + (z4 - z3)/kappaz )/dx;
  } else {			/* usual */
    return (9.*b*dt/8.)*( x2 - x1 + y2 - y1 + z2 - z1 )/dx
      - (b*dt/24.)*( x4 - x3 + y4 - y3 + z4 - z3 )/dx;
  }
}

static inline double staggards4 ( double kap,  double mu,
						   double kappax,  double kappay,  double kappaz,
						   double dt,  double dx,
						   double x1,  double x2,  double x3,  double x4,
						   double y1,  double y2,  double y3,  double y4,
						   double z1,  double z2,  double z3,  double z4,
						   enum typePlace place,  enum typePML method )
{
  if ( (place == ABSORBINGLAYER || place == FREEABS) ) {
    if (method == CPML){
      return (9.*dt/8.)*( (kap+4.*mu/3.)*(x2 - x1)/kappax + (kap-2.*mu/3.)*(y2 - y1)/kappay + (kap-2.*mu/3.)*(z2 - z1)/kappaz )/dx
		- (dt/24.)*( (kap+4.*mu/3.)*(x4 - x3)/kappax + (kap-2.*mu/3.)*(y4 - y3)/kappay + (kap-2.*mu/3.)*(z4 - z3)/kappaz )/dx;
    }else if ( method == PML ){
      return 0.;
    }
  }else {			/* usual */
    return (9.*dt/8.)*( (kap+4.*mu/3.)*(x2 - x1) + (kap-2.*mu/3.)*(y2 - y1 + z2 - z1)) /dx
      - (dt/24.)*( (kap+4.*mu/3.)*(x4 - x3) + (kap-2.*mu/3.)*(y4 - y3  +  z4 - z3) )/dx;
  }
}

static inline double staggardt4 ( double mu, 
						   double kappax,  double kappay,  double dt,  double dx,
						   double x1,  double x2,  double x3,  double x4,
						   double y1,  double y2,  double y3,  double y4,
						   enum typePlace place,  enum typePML method)
{ 
  if ( (place == ABSORBINGLAYER || place == FREEABS) ) {
    if ( method == CPML){
      return (9.*dt*mu/8.)*( (x2 - x1)/kappax + (y2 - y1)/kappay )/dx
		- (dt*mu/24.)*( (x4 - x3)/kappax + (y4 - y3)/kappay )/dx;
    }else if ( method == PML ){
      return 0.;
    }
  }else {			/* usual */
    return (9.*dt*mu/8.)*( x2 - x1 + y2 - y1 )/dx
      - (dt*mu/24.)*( x4 - x3  + y4 - y3 )/dx;
  }
}

/* [source==HISTFILE] increment seismic moment */
static inline double radxx( double strike,  double dip,  double rake)
{
  return  cos(rake)*sin(dip)*sin(2.*strike)
    - sin(rake)*sin(2.*dip)*cos(strike)*cos(strike) ;
}
static inline double radyy( double strike,  double dip,  double rake)
{
  return  - ( cos(rake)*sin(dip)*sin(2.*strike)
			  + sin(rake)*sin(2.*dip)*sin(strike)*sin(strike) );
}
static inline double radzz( double strike,  double dip,  double rake)
{
  return sin(rake)*sin(2.*dip);
}
static inline double radxy( double strike,  double dip,  double rake)
{
  return cos(rake)*sin(dip)*cos(2.*strike)
    + 0.5*sin(rake)*sin(2.*dip)*sin(2.*strike);
}
static inline double radyz( double strike,  double dip,  double rake)
{
  return cos(rake)*cos(dip)*cos(strike) + sin(rake)*cos(2.*dip)*sin(strike);
}
static inline double radxz( double strike,  double dip,  double rake)
{
  return cos(rake)*cos(dip)*sin(strike) - sin(rake)*cos(2.*dip)*cos(strike);
}

/*============= PML/CPML Functions=========================== */
static inline double PMLdump4( double b,  double dt,  double dx,  double dump,
						double v,  double x1,  double x2,  double x3,  double x4 )
{
  return ( (2. - dt*dump)*v + b*( (9./8.)*(x2 - x1) - (1./24.)*(x4 - x3) )*(2.*dt/dx) )/(2. + dt*dump);
}

static inline double CPML4 ( double vp, double dump, double alpha,  double kappa,  double phidum,  double dx,  double dt,
					  double x1,  double x2,  double x3,  double x4 )
{
  double a, b;
  
  b = exp ( - ( vp*dump / kappa + alpha ) * dt );
  a = 0.0;
  if ( abs ( vp*dump ) > 0.000001 ) a = vp*dump * ( b - 1.0 ) / ( kappa * ( vp*dump + kappa * alpha ) );
  
  return b * phidum + a * ( (9./8.)*( x2 - x1 )/dx - (1./24.)*( x4 - x3 )/dx );
}


/*============= ComputeIntermediates Functions=========================== */
/* DAY and BRADLEY : ABSORBING LAYER related */

/* basic function */
static inline double FKSIs4 ( double mu,  double kap,  double dx,
					   double As,  double Ap,
					   double x1,  double x2,  double x3,  double x4,
					   double y1,  double y2,  double y3,  double y4,
					   double z1,  double z2,  double z3,  double z4 )
{
  return 2.*mu*As*( (9./8.)*(x2-x1) - (1./24.)*(x4-x3) )/dx +
    ( ( kap+(4./3.)*mu)*Ap - 2.*mu*As ) * ( (9./8.)*(x2-x1) - (1./24.)*(x4-x3) +
											(9./8.)*(y2-y1) - (1./24.)*(y4-y3) + 
											(9./8.)*(z2-z1) - (1./24.)*(z4-z3) )/dx;
}

static inline double FKSIt4 ( double mu,  double dx,  double As,
					   double x1,  double x2,  double x3,  double x4,
					   double y1,  double y2,  double y3,  double y4 )
{
  return 2.*mu*As*0.5*( (9./8.)*(x2-x1) - (1./24.)*(x4-x3) + (9./8.)*(y2-y1) - (1./24.)*(y4-y3) )/dx;
}

static inline double FKSIs4CPML ( double mu,  double kap,  double dx,
						   double dt,  double As,  double Ap,  double vp,
						   double phixdum,  double phiydum,  double phizdum,
						   double dumpx,  double alphax,  double kappax,
						   double dumpy,  double alphay,  double kappay,
						   double dumpz,  double alphaz,  double kappaz,
						   double x1,  double x2,  double x3,  double x4,
						   double y1,  double y2,  double y3,  double y4,
						   double z1,  double z2,  double z3,  double z4 )
{
  double a_x, a_y, a_z, b_x, b_y, b_z;
  double dxt, dyt, dzt;

  b_x = exp ( - ( vp*dumpx / kappax + alphax ) * dt );
  a_x = 0.0;
  if ( abs ( vp*dumpx ) > 0.000001 ) a_x = vp*dumpx * ( b_x - 1.0 ) / ( kappax * ( vp*dumpx + kappax * alphax ) );

  b_y = exp ( - ( vp*dumpy / kappay + alphay ) * dt );
  a_y = 0.0;
  if ( abs ( vp*dumpy ) > 0.000001 ) a_y = vp*dumpy * ( b_y - 1.0 ) / ( kappay * ( vp*dumpy + kappay * alphay ) );

  b_z = exp ( - ( vp*dumpz / kappaz + alphaz ) * dt );
  a_z = 0.0;
  if ( abs ( vp*dumpz ) > 0.000001 ) a_z = vp*dumpz * ( b_z - 1.0 ) / ( kappaz * ( vp*dumpz + kappaz * alphaz ) );

  dxt = b_x * phixdum + ( 1./kappax + a_x ) * ( (9./8.)*( x2 - x1 )/dx - (1./24.)*( x4 - x3 )/dx );
  dyt = b_y * phiydum + ( 1./kappay + a_y ) * ( (9./8.)*( y2 - y1 )/dx - (1./24.)*( y4 - y3 )/dx );
  dzt = b_z * phizdum + ( 1./kappaz + a_z ) * ( (9./8.)*( z2 - z1 )/dx - (1./24.)*( z4 - z3 )/dx );

  return 2.*mu*As*dxt + ( ( kap+(4./3.)*mu)*Ap - 2.*mu*As ) * ( dxt + dyt + dzt );
}


static inline double FKSIt4CPML ( double mu,  double dx,  double dt,  double As,  double vp,
						   double phixdum,  double phiydum,
						   double dumpx,  double alphax,  double kappax,
						   double dumpy,  double alphay,  double kappay,
						   double x1,  double x2,  double x3,  double x4,
						   double y1,  double y2,  double y3,  double y4 )
{

  double a_x, a_y, b_x, b_y;
  double dxt, dyt;

  b_x = exp ( - ( vp*dumpx / kappax + alphax ) * dt );
  a_x = 0.0;
  if ( abs ( vp*dumpx ) > 0.000001 ) a_x = vp*dumpx * ( b_x - 1.0 ) / ( kappax * ( vp*dumpx + kappax * alphax ) );

  b_y = exp ( - ( vp*dumpy / kappay + alphay ) * dt );
  a_y = 0.0;
  if ( abs ( vp*dumpy ) > 0.000001 ) a_y = vp*dumpy * ( b_y - 1.0 ) / ( kappay * ( vp*dumpy + kappay * alphay ) );

  dxt = b_x * phixdum + ( 1./kappax + a_x ) * ( (9./8.)*( x2 - x1 )/dx - (1./24.)*( x4 - x3 )/dx );
  dyt = b_y * phiydum + ( 1./kappay + a_y ) * ( (9./8.)*( y2 - y1 )/dx - (1./24.)*( y4 - y3 )/dx );

  return 2*mu*As*0.5*( dxt + dyt );
}


/* KRISTEK AND MOCZO */
static inline double Diff4( double dx, double x1, double x2, double x3, double x4 ){
  return 	( 9./8.* ( x2 - x1 ) - (x4 - x3)/24. )/dx ;
}

static inline double SumYlKsil( double *** ksil, double *yMuMe,
                                double *yMuNeighP, double 	*yMuNeighN,
                                int i, int j, int k )
{
  double result;
  result = yMuMe[0]* ksil[i][j][k] ;
  
  result += 0.5* yMuNeighP[0]* ksil[i+1][j][k] ;
  result += 0.5* yMuNeighN[0]* ksil[i-1][j][k] ;

  result += 0.5* yMuNeighP[1]* ksil[i][j+1][k] ;
  result += 0.5* yMuNeighN[1]* ksil[i][j-1][k] ;

  result += 0.5* yMuNeighP[2]* ksil[i][j][k+1] ;
  result += 0.5* yMuNeighN[2]* ksil[i][j][k-1] ;

  return result;
}

/* ============================================================= */
/* TEST FUNCTIONS */
#ifdef TEST

/* ==== */
void test_WhereAmI(void){

  struct PARAMETERS PRM;
  PRM.zMax=10;

  PRM.xMin=0;
  PRM.yMin=0;
  PRM.zMin=0;
  PRM.xMax=10;
  PRM.yMax=10;
  PRM.delta=2; 

  enum typePlace place; 
  int i,j,k;

  printf(",: Regular, # :AbsorbingLayer,- : FreeSurface,| : Limit, . : OutSide : :FREEABS\n" );
  for (j=PRM.yMin-PRM.delta;j<=PRM.yMax+PRM.delta+2;j++)
    {
      printf("\ntest of xz plane at y=%i\n",j);
      printf("Absorbing \t\t\tFree \n",j);
      for (k=PRM.zMax+PRM.delta+3;k>=PRM.zMin-PRM.delta-1;k--)
		{
		  PRM.zMax0 = PRM.zMax + PRM.delta + 2; /* Absorbing */
		  for (i=PRM.xMin-PRM.delta-1;i<=PRM.xMax+PRM.delta+3;i++) 
			{
			  place=WhereAmI(i, j,  k, PRM, ABSORBING);
			  switch(place){
			  case (REGULAR) :
				printf(", ");
				break;
			  case ABSORBINGLAYER :
				printf("# ");
				break;
			  case FREESURFACE :
				printf("- ");
				break;
			  case FREEABS :
				printf(": ");
				break;
			  case OUTSIDE :
				printf(". ");
				break;
			  case LIMIT :
				printf("| ");
				break;
		
			  default :
				printf("");
			  }	/* end switch */
	      
			}
		  printf("\t\t\t");
		  PRM.zMax0 = PRM.zMax + 2; /* Free*/
		  for (i=PRM.xMin-PRM.delta-1;i<=PRM.xMax+PRM.delta+3;i++) 
			{
			  place=WhereAmI(i, j,  k, PRM, FREE);
			  switch(place){
			  case (REGULAR) :
				printf(", ");
				break;
			  case ABSORBINGLAYER :
				printf("# ");
				break;
			  case FREESURFACE :
				printf("- ");
				break;
			  case FREEABS :
				printf(": ");
				break;
			  case OUTSIDE :
				printf(". ");
				break;
			  case LIMIT :
				printf("| ");
			  default :
				printf("");
			  }	/* end switch */
			} /* end Free */
		  printf("\n");
		}
      printf("\n");
    }
} /* end function test_WhereAmI */

#endif	/* TEST */

#endif/*  INLINE_FUNCTIONS_H_ */
