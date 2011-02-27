#include "copyright.h"

/*==============================================================================
 * FILE: rad1d.c
 *
 * PURPOSE:  Problem generator for a 1D test of radiative transfer routine
 *
 * Initial conditions available:
 *  iang = 1 - use Gauss-Legendre quadrature
 *  iang = 2 - use Eddington approximation
 *
 *============================================================================*/

#include <math.h>

#include <stdlib.h>
#include <string.h>
#include "defs.h"
#include "athena.h"
#include "globals.h"
#include "prototypes.h"

static Real eps0;
/*==============================================================================
 * PRIVATE FUNCTION PROTOTYPES:
 * gauleg()           - gauss-legendre quadrature from NR
 *============================================================================*/
static Real const_B(const GridS *pG, const int ifr, const int i, const int j, 
		    const int k);
static Real const_eps(const GridS *pG, const int ifr, const int i, const int j, 
		      const int k);
static Real const_opacity(const GridS *pG, const int ifr, const int i, const int j, 
			  const int k);

void problem(DomainS *pDomain)
{
  RadGridS *pRG = (pDomain->RadGrid);
  GridS *pG = (pDomain->Grid);
  int is = pG->is, ie = pG->ie;
  int js = pG->js, je = pG->je;
  int ks = pG->ks, ke = pG->ke;
  int nf=pRG->nf, nang=pRG->nang;
  int ifr, i, m;
  Real x1, xtop, xbtm;  
  Real den = 1.0;
  Real *tau = NULL;

/* Read problem parameters. */

  eps0 = par_getd("problem","eps");

/* Setup density structure */ 
/* tau is used to initialize density grid */
  if ((tau = (Real *)calloc_1d_array(pG->Nx[0]+nghost+2,sizeof(Real))) == NULL) {
    ath_error("[problem]: Error allocating memory");
  }

  xtop = pDomain->RootMaxX[0];
  xbtm = pDomain->RootMinX[0];

  for(i=pG->is; i<=pG->ie+2; i++) {
    x1 = pG->MinX[0] + (Real)(i-is)*pG->dx1;
    tau[i] = pow(10.0,-3.0 + 10.0 * ((x1-xbtm)/(xtop-xbtm)));
  }

  for (i=is-1; i<=ie+1; i++) {
    pG->U[ks][js][i].d  = (tau[i+1] - tau[i]) / pG->dx1;
  }
/* Free up memory */
  free_1d_array(tau);

/* Initialize mean intensity */
  for(ifr=0; ifr<nf; ifr++)
    for(i=pRG->is; i<=pRG->ie+1; i++)
      pRG->R[0][js][i][ifr].J = 1;

/* Initialize boundary emission */

  for(ifr=0; ifr<nf; ifr++) 
    for(m=0; m<=nang; m++) {
/* lower boundary is tau=0, no irradiation */
      pRG->l1imu[ifr][pRG->ks][pRG->js][0][m] = 0.0; 
/* upper boundary is large tau, eps=1 */
      pRG->r1imu[ifr][pRG->ks][pRG->js][1][m] = 1.0;
    }

/* enroll radiation specification functions */
get_thermal_source = const_B;
get_thermal_fraction = const_eps;
get_total_opacity = const_opacity;

  return;
}

/*==============================================================================
 * PUBLIC PROBLEM USER FUNCTIONS:
 * problem_write_restart() - writes problem-specific user data to restart files
 * problem_read_restart()  - reads problem-specific user data from restart files
 * get_usr_expr()          - sets pointer to expression for special output data
 * get_usr_out_fun()       - returns a user defined output function pointer
 * get_usr_par_prop()      - returns a user defined particle selection function
 * Userwork_in_loop        - problem specific work IN     main loop
 * Userwork_after_loop     - problem specific work AFTER  main loop
 *----------------------------------------------------------------------------*/

void problem_write_restart(MeshS *pM, FILE *fp)
{
  return;
}


void problem_read_restart(MeshS *pM, FILE *fp)
{
  return;
}

ConsFun_t get_usr_expr(const char *expr)
{
  return NULL;
}

VOutFun_t get_usr_out_fun(const char *name){
  return NULL;
}

void Userwork_in_loop(MeshS *pM)
{
  return;
}

void Userwork_after_loop(MeshS *pM)
{
  return;
}

static Real const_B(const GridS *pG, const int ifr, const int i, const int j, 
		    const int k)
{
  return 1.0;
}

static Real const_eps(const GridS *pG, const int ifr, const int i, const int j, 
		      const int k)
{

  return eps0;
  
}

static Real const_opacity(const GridS *pG, const int ifr, const int i, const int j, 
			  const int k)
{

  return pG->U[k][j][i].d;
  
}

