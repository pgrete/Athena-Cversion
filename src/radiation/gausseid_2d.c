#include "../copyright.h"
/*==============================================================================
 * FILE: gausseid_2d.c
 *
 * PURPOSE: Solves a single iteration of the formal solution of radiative
 *          transfer on a 2D grid using the Gauss-Seidel method.  The basic algorithm
 *          is described in Trujillo Bueno and Fabiani Benedicho, ApJ, 455, 646.
 *          Uses linear interpolation to compuet intensities at edges.
 *
 * CONTAINS PUBLIC FUNCTIONS: 
 *   formal_solution_2d.c()
 *   formal_solution_2d_destruct()
 *   formal_solution_2d_init()
 *============================================================================*/

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "../defs.h"
#include "../athena.h"
#include "../globals.h"
#include "../prototypes.h"


#ifdef RADIATION_TRANSFER
#ifdef GAUSSEID

/* Working arrays used in formal solution */
static Real ****psiint = NULL;
static Real ***lamstr = NULL;
static Real *****imuo = NULL;
static Real dSrmx;

/*==============================================================================
 * PRIVATE FUNCTION PROTOTYPES:
 *   update_cell() - update variables in a single grid cell
 *   update_sfunc() - updates source function after compute of mean intensity
 *   sweep_2d_forward() - sweep grid from lower left to upper right
 *   sweep_2d_backward() - sweep grid from upper right to lower left
 *============================================================================*/

static void update_cell(RadGridS *pRG, Real *****imuo, int ifr, int k, int j, int i, int l);
static void update_sfunc(RadS *R, Real *dS, Real lamstr);
static void sweep_2d_forward(RadGridS *pRG, int ifr);
static void sweep_2d_backward(RadGridS *pRG, int ifr);

/*=========================== PUBLIC FUNCTIONS ===============================*/

/*----------------------------------------------------------------------------*/
/*! \fn void formal_solution_2d(RadGridS *pRG, Real *dSrmax, int ifr)
 *  \brief formal solution for single freq. in 2D using Gauss-Seidel method
 *  with linear interpolation of intensities */
void formal_solution_2d(RadGridS *pRG, Real *dSrmax, int ifr)
{
  int i, j, l, m;
  int is = pRG->is, ie = pRG->ie; 
  int js = pRG->js, je = pRG->je; 
  int ks = pRG->ks; 
  int nf = pRG->nf;
 

#ifdef QUADRATIC_INTENSITY
  ath_error("[gausseid_2d.c]: quadratic intensity not supported with Gauss-Seidel.\n");
#endif

/* Initialize dSrmx */
  dSrmx = 0.0;

/* initialize mean intensities at all depths to zero */
  for(j=js; j<=je; j++)
    for(i=is; i<=ie; i++) {
      pRG->R[ifr][ks][j][i].J = 0.0;
      pRG->R[ifr][ks][j][i].H[0] = 0.0;
      pRG->R[ifr][ks][j][i].H[1] = 0.0;
      pRG->R[ifr][ks][j][i].K[0] = 0.0;
      pRG->R[ifr][ks][j][i].K[1] = 0.0;
      pRG->R[ifr][ks][j][i].K[2] = 0.0;
      lamstr[ifr][j][i] = 0.0;
      for(l=0; l<4; l++)
	psiint[ifr][j][i][l] = 0.0;      
    }

/* Compute formal solution and for all rays in each gridzone and 
 * update boundary emission*/
  sweep_2d_forward(pRG,ifr);

  sweep_2d_backward(pRG,ifr);

/* Return maximum relative change to test convergence*/
  (*dSrmax) = dSrmx;

  return;
}

/*----------------------------------------------------------------------------*/
/*! \fn void formal_solution_2d_destruct(void)
 *  \brief free temporary working arrays */
void formal_solution_2d_destruct(void)
{

  if (psiint != NULL) free_4d_array(psiint);
  if (lamstr != NULL) free_3d_array(lamstr);
  if (imuo   != NULL) free_5d_array(imuo);

  return;
}

/*----------------------------------------------------------------------------*/
/*! \fn void formal_solution_2d_init(DomainS *pD)
 *  \brief Allocate memory for working arrays */
void formal_solution_2d_init(DomainS *pD)
{
  
  RadGridS *pRG, *pROG;
  int nx1, nx2, nmx;
  Real dx, dy;
  int nf, nang;

  if (radt_mode == 0) { /* integration only */
    pRG = pD->RadGrid;
    nx1 = pRG->Nx[0]; nx2 = pRG->Nx[1];
    dx = pRG->dx1; dy = pRG->dx2;
    nf = pRG->nf;
    nang = pRG->nang;
  } else if (radt_mode == 1) { /* output only */
    pRG = pD->RadOutGrid;
    nx1 = pRG->Nx[0]; nx2 = pRG->Nx[1];
    dx = pRG->dx1; dy = pRG->dx2;
    nf = pRG->nf;
    nang = pRG->nang;
  } else if (radt_mode == 2) { /* integration and output */
    pRG = pD->RadGrid;
    pROG = pD->RadOutGrid;
    nx1 = pRG->Nx[0]; nx2 = pRG->Nx[1];
    dx = pRG->dx1; dy = pRG->dx2;
    nf = MAX(pRG->nf,pROG->nf);
    nang = MAX(pRG->nang,pROG->nang);
  }

  nmx = MAX(nx1,nx2);

  if ((lamstr = (Real ***)calloc_3d_array(nf,nx2+2,nx1+2,sizeof(Real))) == NULL) 
    goto on_error;

  if ((psiint = (Real ****)calloc_4d_array(nf,nx2+2,nx1+2,4,sizeof(Real))) == NULL) 
    goto on_error;

  if ((imuo = (Real *****)calloc_5d_array(nf,nmx+2,4,nang,2,sizeof(Real))) == NULL)
    goto on_error;

  return;

  on_error:
  formal_solution_2d_destruct();
  ath_error("[formal_solution__2d_init]: Error allocating memory\n");
  return;

}

/*=========================== PRIVATE FUNCTIONS ==============================*/

/*----------------------------------------------------------------------------*/
/*! \fn static void sweep_2d_forward(RadGridS *pRG, int ifr)
 *  \brief Perform Gauss-Seidel sweep from lower left to upper right */
static void sweep_2d_forward(RadGridS *pRG, int ifr)
{
  int i, j, l, m;
  int is = pRG->is, ie = pRG->ie;
  int js = pRG->js, je = pRG->je;
  int ks = pRG->ks;   
  int nf = pRG->nf, nang = pRG->nang;

/* Account for ix2 boundary intensities */
  for(i=is-1; i<=ie+1; i++) {
    for(l=0; l<=1; l++)  {
      for(m=0; m<nang; m++) {
	imuo[ifr][i][l][m][0] = pRG->Ghstl2i[ifr][ks][i][l][m];
      }}}

  /* sweep forward in x2 */
  for(j=js; j<=je; j++) {

    /* Account for ix1 boundary intensities */
    for(m=0; m<nang; m++) {
      /* ix1/ox1 boundary conditions*/
      imuo[ifr][is-1][0][m][1] = imuo[ifr][is-1][0][m][0];
      imuo[ifr][ie+1][1][m][1] = imuo[ifr][ie+1][1][m][0];
      imuo[ifr][is-1][0][m][0] = pRG->Ghstl1i[ifr][ks][j][0][m];
      imuo[ifr][ie+1][1][m][0] = pRG->Ghstr1i[ifr][ks][j][1][m];
    }

    /* Sweep forward in x1 */
#ifdef SHEARING_BOX
    update_cell(pRG,imuo,ifr,ks,j,is,0);
    /* Update intensity at the ix1 boundary */
    for(m=0; m<nang; m++)  {
      pRG->l1imu[ifr][ks][j][0][m] = imuo[ifr][is][0][m][0];
    }
    for(i=is+1; i<=ie; i++) 
      update_cell(pRG,imuo,ifr,ks,j,i,0);
#else
    for(i=is; i<=ie; i++) 
      update_cell(pRG,imuo,ifr,ks,j,i,0);
#endif

    /* Update intensity at the ox1 boundary */
    for(m=0; m<nang; m++)  {
	pRG->r1imu[ifr][ks][j][0][m] = imuo[ifr][ie][0][m][0];
    }
    /* Sweep backward in x1 */
#ifdef SHEARING_BOX
    update_cell(pRG,imuo,ifr,ks,j,ie,1);
    /* Update intensity at the ox1 boundary */
    for(m=0; m<nang; m++)  {
      pRG->r1imu[ifr][ks][j][1][m] = imuo[ifr][ie][1][m][0];
    }
    for(i=ie-1; i>=is; i--) 
      update_cell(pRG,imuo,ifr,ks,j,i,1);
#else
    for(i=ie; i>=is; i--) 
      update_cell(pRG,imuo,ifr,ks,j,i,1);
#endif
    /* Update intensity at the ix1 boundary */
    for(m=0; m<nang; m++)  {
      pRG->l1imu[ifr][ks][j][1][m] = imuo[ifr][is][1][m][0];
      }
  }
  /* Update intensity at the ox2 boundary */
  for(i=is; i<=ie; i++) { 
    for(l=0; l<=1; l++) { 
      for(m=0; m<nang; m++) { 
	pRG->r2imu[ifr][ks][i][l][m] = imuo[ifr][i][l][m][0];
      }}}

  return;
}

/*----------------------------------------------------------------------------*/
/*! \fn static void sweep_2d_backward(RadGridS *pRG, int ifr)
 *  \brief Perform Gauss-Seidel sweep from upper right to lower left*/
static void sweep_2d_backward(RadGridS *pRG, int ifr)
{
  int i, j, l, m;
  int is = pRG->is, ie = pRG->ie;
  int js = pRG->js, je = pRG->je;
  int ks = pRG->ks;   
  int nf = pRG->nf, nang = pRG->nang;

/* Account for ox2 boundary intensities */
  for(i=is-1; i<=ie+1; i++) {
    for(l=2; l<=3; l++)  {
      for(m=0; m<nang; m++) {
	imuo[ifr][i][l][m][0] = pRG->Ghstr2i[ifr][ks][i][l][m];
      }}}
  
  /* sweep backward in x2 */
  for(j=je; j>=js; j--) {

    /* Account for ix1 boundary intensities */
    for(m=0; m<nang; m++) {
      /* ix1/ox1 boundary conditions*/
      imuo[ifr][is-1][2][m][1] = imuo[ifr][is-1][2][m][0];
      imuo[ifr][ie+1][3][m][1] = imuo[ifr][ie+1][3][m][0];
      imuo[ifr][is-1][2][m][0] = pRG->Ghstl1i[ifr][ks][j][2][m];
      imuo[ifr][ie+1][3][m][0] = pRG->Ghstr1i[ifr][ks][j][3][m];
    }

    /* Sweep forward in x1 */
#ifdef SHEARING_BOX
    update_cell(pRG,imuo,ifr,ks,j,is,2);
    /* Update intensity at the ix1 boundary */
    for(m=0; m<nang; m++)  {
      pRG->l1imu[ifr][ks][j][2][m] = imuo[ifr][is][2][m][0];
    }
    for(i=is+1; i<=ie; i++) 
      update_cell(pRG,imuo,ifr,ks,j,i,2);
#else
    for(i=is; i<=ie; i++) 
      update_cell(pRG,imuo,ifr,ks,j,i,2);
#endif

    /* Update intensity at the ox1 boundary */
    for(m=0; m<nang; m++)  {
      pRG->r1imu[ifr][ks][j][2][m] = imuo[ifr][ie][2][m][0];
    }

    /* Sweep backward in x1 */
#ifdef SHEARING_BOX
    update_cell(pRG,imuo,ifr,ks,j,ie,3);
    /* Update intensity at the ox1 boundary */
    for(m=0; m<nang; m++)  {
      pRG->r1imu[ifr][ks][j][3][m] = imuo[ifr][ie][3][m][0];
    }
    for(i=ie-1; i>=is; i--) 
      update_cell(pRG,imuo,ifr,ks,j,i,3);
#else
    for(i=ie; i>=is; i--) 
      update_cell(pRG,imuo,ifr,ks,j,i,3);
#endif
    /* Update intensity at the ix1 boundary */
    for(m=0; m<nang; m++)  {
      pRG->l1imu[ifr][ks][j][3][m] = imuo[ifr][is][3][m][0];
    }
  }

  /* Update intensity at the ix2 boundary */
  for(i=is; i<=ie; i++) { 
    for(l=2; l<=3; l++) { 
      for(m=0; m<nang; m++) { 
	pRG->l2imu[ifr][ks][i][l][m] = imuo[ifr][i][l][m][0];
      }}}

  return;
}

/*----------------------------------------------------------------------------*/
/*! \fn static void update_cell(RadGridS *pRG, Real *****imuo, int ifr, int k, 
 *                              int j, int i, int l)
 *  \brief Update radiation variables in a single cell */
static void update_cell(RadGridS *pRG, Real *****imuo, int ifr, int k, int j, int i, int l)
{

  int im, ip, jm, jp;
  int m, nf = pRG->nf, nang = pRG->nang;
  int is = pRG->is, js = pRG->js, ie = pRG->ie;
  Real imu, imu0, wimu;
  Real S0, S2;
  Real am, am1, bm, bm1;
  Real w0, w1, w2;
  Real dx = pRG->dx1, dy = pRG->dx2;
  Real chi0, chi1, chi2, dtaum, dtaup;
  Real edtau, a0, a1, a2;
  Real dS;

/* initialize stencil base on quadrant*/  
  if(l == 0) {
    jp = j + 1;  jm = j - 1;
    ip = i + 1;  im = i - 1;
  } else if (l == 1) {
    jp = j + 1;  jm = j - 1;
    ip = i - 1;  im = i + 1;
  } else if (l == 2) {
    jp = j - 1;  jm = j + 1;
    ip = i + 1;  im = i - 1;
  } else {
    jp = j - 1;  jm = j + 1;
    ip = i - 1;  im = i + 1;
  }  


  for(m=0; m<nang; m++) {
    chi1 = pRG->R[ifr][k][j][i].chi;
/* --------- Interpolate intensity and source functions at endpoints --------- 
 * --------- of characteristics                                      --------- */
    am = fabs( dy * pRG->mu[0][m][0] / (dx * pRG->mu[0][m][1]) );
    if (am <= 1.0) {
      am1 = 1.0 - am;
      /* Use linear interpolation for source functions */
      S0 = am  * pRG->R[ifr][k][jm][im].S +
           am1 * pRG->R[ifr][k][jm][i ].S;
      S2 = am  * pRG->R[ifr][k][jp][ip].S +
	   am1 * pRG->R[ifr][k][jp][i ].S;
      /* Use linear interpolation for intensity */
      imu0 = am  * imuo[ifr][im][l][m][1] + am1 * imuo[ifr][i][l][m][0];
    } else {
      bm = 1.0 / am;
      bm1 = 1.0 - bm;

      /* Use linear interpolation for source functions */
      S0 = bm  * pRG->R[ifr][k][jm][im].S +
           bm1 * pRG->R[ifr][k][j ][im].S;
      S2 = bm  * pRG->R[ifr][k][jp][ip].S +
           bm1 * pRG->R[ifr][k][j ][ip].S;
      /* Use linear interpolation for intensity */
      imu0 = bm  * imuo[ifr][im][l][m][1] + bm1 * imuo[ifr][im][l][m][0];
    }
/* ---------  compute intensity at grid center and add to mean intensity ------- */

    if (am <= 1.0) {
      chi0 = am  * pRG->R[ifr][k][jm][im].chi + 
	     am1 * pRG->R[ifr][k][jm][i ].chi;
      chi2 = am  * pRG->R[ifr][k][jp][ip].chi + 
	     am1 * pRG->R[ifr][k][jp][i ].chi;
      interp_quad_chi(chi0,chi1,chi2,&dtaum);
      interp_quad_chi(chi2,chi1,chi0,&dtaup);
      dtaum *= dy / pRG->mu[0][m][1];
      dtaup *= dy / pRG->mu[0][m][1];
    } else {
      chi0 = bm  * pRG->R[ifr][k][jm][im].chi + 
	     bm1 * pRG->R[ifr][k][j ][im].chi;
      chi2 = bm  * pRG->R[ifr][k][jp][ip].chi +
	     bm1 * pRG->R[ifr][k][j ][ip].chi;
      interp_quad_chi(chi0,chi1,chi2,&dtaum);
      interp_quad_chi(chi2,chi1,chi0,&dtaup);
      dtaum *= dx / pRG->mu[0][m][0];
      dtaup *= dx / pRG->mu[0][m][0];
    }
    interp_quad_source_slope_lim(dtaum, dtaup, &edtau, &a0, &a1, &a2,
			         S0, pRG->R[ifr][k][j][i].S, S2);
    imu = a0 * S0 + a1 * pRG->R[ifr][k][j][i].S + a2 * S2 + edtau * imu0;
    lamstr[ifr][j][i] += pRG->wmu[m] * a1;
/* Save weights for Gauss-Seidel update */
    if (l == 0) {
      if (am <= 1.0) {
	psiint[ifr][j][i][1] += am  * pRG->wmu[m] * a2; 
	psiint[ifr][j][i][2] += am1 * pRG->wmu[m] * a2; 
      } else {
	psiint[ifr][j][i][0] += bm1 * pRG->wmu[m] * a2; 
	psiint[ifr][j][i][1] += bm  * pRG->wmu[m] * a2; 
      }
    } else if (l == 1) {
      if (am <= 1.0) {
	psiint[ifr][j][i][2] += am1 * pRG->wmu[m] * a2; 
	psiint[ifr][j][i][3] += am  * pRG->wmu[m] * a2; 
      } else {
	psiint[ifr][j][i][3] += bm  * pRG->wmu[m] * a2; 
	psiint[ifr][j][i][0] += bm1 * pRG->wmu[m] * a0; 
      }
    } else if (l == 2) {
      if (am > 1.0)
	psiint[ifr][j][i][0] += bm1 * pRG->wmu[m] * a2; 
    }
/* Add to radiation moments and save for next iteration */
    wimu = pRG->wmu[m] * imu;
    pRG->R[ifr][k][j][i].J += wimu;
    pRG->R[ifr][k][j][i].H[0] += pRG->mu[l][m][0] * wimu;
    pRG->R[ifr][k][j][i].H[1] += pRG->mu[l][m][1] * wimu;
    pRG->R[ifr][k][j][i].K[0] += pRG->mu[l][m][0] * pRG->mu[l][m][0] * wimu;
    pRG->R[ifr][k][j][i].K[1] += pRG->mu[l][m][0] * pRG->mu[l][m][1] * wimu;
    pRG->R[ifr][k][j][i].K[2] += pRG->mu[l][m][1] * pRG->mu[l][m][1] * wimu;
/* Update intensity workspace */
    imuo[ifr][i][l][m][1] = imuo[ifr][i][l][m][0];
    imuo[ifr][i][l][m][0] = imu;
  }
  if (l == 3) {
/* Update source function when all angles have ben computed */
    update_sfunc(&(pRG->R[ifr][k][j][i]), &dS, lamstr[ifr][j][i]);

/* Correct J w/ updated S from "new" neighbors, but not in ghostzones */
    if(i != is) {
      pRG->R[ifr][k][j][i-1].J += dS * psiint[ifr][j][i-1][0];
      if (j != js) 
	pRG->R[ifr][k][j-1][i-1].J += dS * psiint[ifr][j-1][i-1][1];
    }
    if (j != js) {
      pRG->R[ifr][k][j-1][i].J += dS * psiint[ifr][j-1][i][2];
      if(i != ie) 
	pRG->R[ifr][k][j-1][i+1].J += dS * psiint[ifr][j-1][i+1][3];
    }
  }
  return;
}

/*----------------------------------------------------------------------------*/
/*! \fn static void update_sfunc(RadS *R, Real *dS, Real lamstr)
 *  \brief Gauss-Siedel update of source function with new mean intensity */
static void update_sfunc(RadS *R, Real *dS, Real lamstr)
{
  Real Snew, dSr;
  
  Snew = (1.0 - R->eps) * R->J + R->eps * R->B + R->Snt;
  (*dS) = (Snew - R->S) / (1.0 - (1.0 - R->eps) * lamstr);
  if (R->S > 0.0) dSr = fabs((*dS) / R->S);
  R->S += (*dS);
  if (dSr > dSrmx) dSrmx = dSr; 
  return;
}

#endif /* GAUSSEID */
#endif /* RADIATION_TRANSFER */