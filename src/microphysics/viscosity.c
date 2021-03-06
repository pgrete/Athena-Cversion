#include "../copyright.h"
/*============================================================================*/
/*! \file viscosity.c
 *  \brief Adds explicit viscosity terms to the momentum and energy equations
 *
 * PURPOSE: Adds explicit viscosity terms to the momentum and energy equations,
 *  -   dM/dt = Div(T)    
 *  -   dE/dt = Div(v.T)
 *   where 
 *  - T = nu_iso Grad(V) + T_Brag = TOTAL viscous stress tensor
 *
 *   Note T contains contributions from both isotropic (Navier-Stokes) and
 *   anisotropic (Braginskii) viscosity.  These contributions are computed in
 *   calls to ViscStress_* functions.
 *
 * CONTAINS PUBLIC FUNCTIONS:
 *- viscosity() - updates momentum and energy equations with viscous terms
 *- viscosity_init() - allocates memory needed
 *- viscosity_destruct() - frees memory used */
/*============================================================================*/

#include <math.h>
#include <float.h>
#include "../defs.h"
#include "../athena.h"
#include "../globals.h"
#include "prototypes.h"
#include "../prototypes.h"

#ifdef VISCOSITY

/*! \struct ViscFluxS
 *  \brief Structure to contain 4-components of the viscous fluxes */
typedef struct ViscFlux_t{
  Real Mx;
  Real My;
  Real Mz;
#ifndef BAROTROPIC
  Real E;
#endif
}ViscFluxS;

static ViscFluxS ***x1Flux=NULL, ***x2Flux=NULL, ***x3Flux=NULL;
static Real3Vect ***Vel=NULL;
static Real ***divv=NULL;

/*==============================================================================
 * PRIVATE FUNCTION PROTOTYPES:
 *   ViscStress_iso   - computes   isotropic viscous fluxes
 *   ViscStress_aniso - computes anisotropic viscous fluxes
 *============================================================================*/

void ViscStress_iso(DomainS *pD);
void ViscStress_aniso(DomainS *pD);

static Real limiter2(const Real A, const Real B);
static Real limiter4(const Real A, const Real B, const Real C, const Real D);
static Real vanleer (const Real A, const Real B);
static Real minmod  (const Real A, const Real B);

/*=========================== PUBLIC FUNCTIONS ===============================*/
/*----------------------------------------------------------------------------*/
/*! \fn void viscosity(DomainS *pD)
 *  \brief Adds explicit viscosity terms to the momentum and energy equations
 */

void viscosity(DomainS *pD)
{
  GridS *pG = (pD->Grid);
  int i, is = pG->is, ie = pG->ie;
  int j, jl, ju, js = pG->js, je = pG->je;
  int k, kl, ku, ks = pG->ks, ke = pG->ke;
#ifdef STS
  Real my_dt = STS_dt;
#else
  Real my_dt = pG->dt;
#endif
  Real x1,x2,x3,dtodx1=my_dt/pG->dx1, dtodx2=0.0, dtodx3=0.0;
  
  if (pG->Nx[1] > 1){
    jl = js - 2;
    ju = je + 2;
    dtodx2 = my_dt/pG->dx2;
  } else { 
    jl = js;
    ju = je;
  } 
  if (pG->Nx[2] > 1){
    kl = ks - 2;
    ku = ke + 2;
    dtodx3 = my_dt/pG->dx3;
  } else { 
    kl = ks;
    ku = ke;
  }

/* Zero viscous fluxes; compute vel and div(v) at cell centers. */

  for (k=kl; k<=ku; k++) {
  for (j=jl; j<=ju; j++) {
    for (i=is-2; i<=ie+2; i++) {

      x1Flux[k][j][i].Mx = 0.0;
      x1Flux[k][j][i].My = 0.0;
      x1Flux[k][j][i].Mz = 0.0;
#ifndef BAROTROPIC
      x1Flux[k][j][i].E = 0.0;
#endif

      x2Flux[k][j][i].Mx = 0.0;
      x2Flux[k][j][i].My = 0.0;
      x2Flux[k][j][i].Mz = 0.0;
#ifndef BAROTROPIC
      x2Flux[k][j][i].E = 0.0;
#endif
 
      x3Flux[k][j][i].Mx = 0.0;
      x3Flux[k][j][i].My = 0.0;
      x3Flux[k][j][i].Mz = 0.0;
#ifndef BAROTROPIC
      x3Flux[k][j][i].E = 0.0;
#endif

      Vel[k][j][i].x1 = pG->U[k][j][i].M1/pG->U[k][j][i].d;
      Vel[k][j][i].x2 = pG->U[k][j][i].M2/pG->U[k][j][i].d;
#ifdef FARGO
      cc_pos(pG,i,j,k,&x1,&x2,&x3);
      Vel[k][j][i].x2 -= qshear*Omega_0*x1;
#endif
      Vel[k][j][i].x3 = pG->U[k][j][i].M3/pG->U[k][j][i].d;
    }
  }}
  
  for (k=kl; k<=ku; k++) {
  for (j=jl; j<=ju; j++) {
    for (i=is-1; i<=ie+1; i++) {
      divv[k][j][i] = (Vel[k][j][i+1].x1 - Vel[k][j][i-1].x1)/(2.0*pG->dx1);
    }
  }}

  if (pG->Nx[1] > 1) {
    for (k=kl; k<=ku; k++) {
    for (j=js-1; j<=je+1; j++) {
      for (i=is-1; i<=ie+1; i++) {
        divv[k][j][i] += (Vel[k][j+1][i].x2 - Vel[k][j-1][i].x2)/(2.0*pG->dx2);
      }
    }}
  }

  if (pG->Nx[2] > 1) {
    for (k=ks-1; k<=ke+1; k++) {
    for (j=js-1; j<=je+1; j++) {
      for (i=is-1; i<=ie+1; i++) {
        divv[k][j][i] += (Vel[k+1][j][i].x3 - Vel[k-1][j][i].x3)/(2.0*pG->dx3);
      }
    }}
  }

/* Compute isotropic and anisotropic viscous fluxes.  Fluxes, V and div(V)
 * are global variables in this file. */

  if (nu_iso > 0.0)   ViscStress_iso(pD);
  if (nu_aniso > 0.0) ViscStress_aniso(pD);

/* Update momentum and energy using x1-fluxes (dM/dt = Div(T)) */

  for (k=ks; k<=ke; k++) {
  for (j=js; j<=je; j++) { 
    for (i=is; i<=ie; i++) { 
      pG->U[k][j][i].M1 += dtodx1*(x1Flux[k][j][i+1].Mx - x1Flux[k][j][i].Mx);
      pG->U[k][j][i].M2 += dtodx1*(x1Flux[k][j][i+1].My - x1Flux[k][j][i].My);
      pG->U[k][j][i].M3 += dtodx1*(x1Flux[k][j][i+1].Mz - x1Flux[k][j][i].Mz);
#ifndef BAROTROPIC
      pG->U[k][j][i].E  += dtodx1*(x1Flux[k][j][i+1].E  - x1Flux[k][j][i].E );
#endif /* BAROTROPIC */
    }
  }}

/* Update momentum and energy using x2-fluxes (dM/dt = Div(T)) */

  if (pG->Nx[1] > 1){
    for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {
        pG->U[k][j][i].M1 += dtodx2*(x2Flux[k][j+1][i].Mx - x2Flux[k][j][i].Mx);
        pG->U[k][j][i].M2 += dtodx2*(x2Flux[k][j+1][i].My - x2Flux[k][j][i].My);
        pG->U[k][j][i].M3 += dtodx2*(x2Flux[k][j+1][i].Mz - x2Flux[k][j][i].Mz);
#ifndef BAROTROPIC
        pG->U[k][j][i].E  += dtodx2*(x2Flux[k][j+1][i].E  - x2Flux[k][j][i].E );
#endif /* BAROTROPIC */
      }
    }}
  }

/* Update momentum and energy using x3-fluxes (dM/dt = Div(T)) */

  if (pG->Nx[2] > 1){
    for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {
        pG->U[k][j][i].M1 += dtodx3*(x3Flux[k+1][j][i].Mx - x3Flux[k][j][i].Mx);
        pG->U[k][j][i].M2 += dtodx3*(x3Flux[k+1][j][i].My - x3Flux[k][j][i].My);
        pG->U[k][j][i].M3 += dtodx3*(x3Flux[k+1][j][i].Mz - x3Flux[k][j][i].Mz);
#ifndef BAROTROPIC
        pG->U[k][j][i].E  += dtodx3*(x3Flux[k+1][j][i].E  - x3Flux[k][j][i].E );
#endif /* BAROTROPIC */
      }
    }}
  }

  return;
}

/*----------------------------------------------------------------------------*/
/*! \fn void ViscStress_iso(DomainS *pD)
 *  \brief Calculate viscous stresses with isotropic (NS) viscosity
 */

void ViscStress_iso(DomainS *pD)
{
  GridS *pG = (pD->Grid);
  ViscFluxS VStress;
  int i, is = pG->is, ie = pG->ie;
  int j, js = pG->js, je = pG->je;
  int k, ks = pG->ks, ke = pG->ke;
  Real nud;

/* Add viscous fluxes in 1-direction */

  for (k=ks; k<=ke; k++) {
  for (j=js; j<=je; j++) {
    for (i=is; i<=ie+1; i++) {
      VStress.Mx = 2.0*(Vel[k][j][i].x1 - Vel[k][j][i-1].x1)/pG->dx1
         - ONE_3RD*(divv[k][j][i] + divv[k][j][i-1]);

      VStress.My = (Vel[k][j][i].x2 - Vel[k][j][i-1].x2)/pG->dx1;
      if (pG->Nx[1] > 1) {
        VStress.My +=(0.25/pG->dx2)*((Vel[k][j+1][i].x1 + Vel[k][j+1][i-1].x1)
                                   - (Vel[k][j-1][i].x1 + Vel[k][j-1][i-1].x1));
      }

      VStress.Mz = (Vel[k][j][i].x3 - Vel[k][j][i-1].x3)/pG->dx1;
      if (pG->Nx[2] > 1) {
        VStress.Mz +=(0.25/pG->dx3)*((Vel[k+1][j][i].x1 + Vel[k+1][j][i-1].x1)
                                   - (Vel[k-1][j][i].x1 + Vel[k-1][j][i-1].x1));
      }

      nud = nu_iso*0.5*(pG->U[k][j][i].d + pG->U[k][j][i-1].d);
      x1Flux[k][j][i].Mx += nud*VStress.Mx;
      x1Flux[k][j][i].My += nud*VStress.My;
      x1Flux[k][j][i].Mz += nud*VStress.Mz;

#ifndef BAROTROPIC
      x1Flux[k][j][i].E  += 
         0.5*nud*((Vel[k][j][i-1].x1 + Vel[k][j][i].x1)*VStress.Mx +
                  (Vel[k][j][i-1].x2 + Vel[k][j][i].x2)*VStress.My +
                  (Vel[k][j][i-1].x3 + Vel[k][j][i].x3)*VStress.Mz);
#endif /* BAROTROPIC */
    }
  }}

/* Add viscous fluxes in 2-direction */

  if (pG->Nx[1] > 1) {
    for (k=ks; k<=ke; k++) {
    for (j=js; j<=je+1; j++) {
      for (i=is; i<=ie; i++) {
        VStress.Mx = (Vel[k][j][i].x1 - Vel[k][j-1][i].x1)/pG->dx2
          + ((Vel[k][j][i+1].x2 + Vel[k][j-1][i+1].x2) - 
             (Vel[k][j][i-1].x2 + Vel[k][j-1][i-1].x2))/(4.0*pG->dx1);

        VStress.My = 2.0*(Vel[k][j][i].x2 - Vel[k][j-1][i].x2)/pG->dx2
           - ONE_3RD*(divv[k][j][i] + divv[k][j-1][i]);

        VStress.Mz = (Vel[k][j][i].x3 - Vel[k][j-1][i].x3)/pG->dx2;
        if (pG->Nx[2] > 1) {
          VStress.Mz +=
            ((Vel[k+1][j][i].x2 + Vel[k+1][j-1][i].x2) -
             (Vel[k-1][j][i].x2 + Vel[k-1][j-1][i].x2))/(4.0*pG->dx3);
        }

        nud = nu_iso*0.5*(pG->U[k][j][i].d + pG->U[k][j-1][i].d);
        x2Flux[k][j][i].Mx += nud*VStress.Mx;
        x2Flux[k][j][i].My += nud*VStress.My;
        x2Flux[k][j][i].Mz += nud*VStress.Mz;

#ifndef BAROTROPIC
        x2Flux[k][j][i].E  +=
           0.5*nud*((Vel[k][j-1][i].x1 + Vel[k][j][i].x1)*VStress.Mx +
                    (Vel[k][j-1][i].x2 + Vel[k][j][i].x2)*VStress.My +
                    (Vel[k][j-1][i].x3 + Vel[k][j][i].x3)*VStress.Mz);
#endif /* BAROTROPIC */
      }
    }}
  }

/* Add viscous fluxes in 3-direction */

  if (pG->Nx[2] > 1) {
    for (k=ks; k<=ke+1; k++) {
    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {
        VStress.Mx = (Vel[k][j][i].x1 - Vel[k-1][j][i].x1)/pG->dx3
          + ((Vel[k][j][i+1].x3 + Vel[k-1][j][i+1].x3) -
             (Vel[k][j][i-1].x3 + Vel[k-1][j][i-1].x3))/(4.0*pG->dx1);

        VStress.My = (Vel[k][j][i].x2 - Vel[k-1][j][i].x2)/pG->dx3
          + ((Vel[k][j+1][i].x3 + Vel[k-1][j+1][i].x3) -
             (Vel[k][j-1][i].x3 + Vel[k-1][j-1][i].x3))/(4.0*pG->dx2);

        VStress.Mz = 2.0*(Vel[k][j][i].x3 - Vel[k-1][j][i].x3)/pG->dx3
           - ONE_3RD*(divv[k][j][i] + divv[k-1][j][i]);

        nud = nu_iso*0.5*(pG->U[k][j][i].d + pG->U[k-1][j][i].d);
        x3Flux[k][j][i].Mx += nud*VStress.Mx;
        x3Flux[k][j][i].My += nud*VStress.My;
        x3Flux[k][j][i].Mz += nud*VStress.Mz;

#ifndef BAROTROPIC
        x3Flux[k][j][i].E  +=
           0.5*nud*((Vel[k-1][j][i].x1 + Vel[k][j][i].x1)*VStress.Mx +
                    (Vel[k-1][j][i].x2 + Vel[k][j][i].x2)*VStress.My +
                    (Vel[k-1][j][i].x3 + Vel[k][j][i].x3)*VStress.Mz);
#endif /* BAROTROPIC */
      }
    }}
  }

  return;
}

/*----------------------------------------------------------------------------*/
/*! \fn void ViscStress_aniso(DomainS *pD) 
 *  \brief Calculate viscous stresses with anisotropic (Braginskii)
 *  viscosity */

void ViscStress_aniso(DomainS *pD)
{
  GridS *pG = (pD->Grid);
  ViscFluxS VStress;
  int i, is = pG->is, ie = pG->ie;
  int j, js = pG->js, je = pG->je;
  int k, ks = pG->ks, ke = pG->ke;
  Real Bx,By,Bz,B02,dVc,dVl,dVr,lim_slope,BBdV,divV,qa,nud;
  Real dVxdx,dVydx,dVzdx,dVxdy,dVydy,dVzdy,dVxdz,dVydz,dVzdz;

  if (pD->Nx[1] == 1) return;  /* problem must be at least 2D */
#ifdef MHD

/* Compute viscous fluxes in 1-direction, centered at x1-Faces --------------- */

  for (k=ks; k<=ke; k++) {
  for (j=js; j<=je; j++) {
    for (i=is; i<=ie+1; i++) {

      /* Monotonized Velocity gradient dVx/dy */
      dVxdy = limiter4(Vel[k][j+1][i  ].x1 - Vel[k][j  ][i  ].x1,
                       Vel[k][j  ][i  ].x1 - Vel[k][j-1][i  ].x1,
                       Vel[k][j+1][i-1].x1 - Vel[k][j  ][i-1].x1,
                       Vel[k][j  ][i-1].x1 - Vel[k][j-1][i-1].x1);
      dVxdy /= pG->dx2;
      
      /* Monotonized Velocity gradient dVy/dy */
      dVydy = limiter4(Vel[k][j+1][i  ].x2 - Vel[k][j  ][i  ].x2,
                       Vel[k][j  ][i  ].x2 - Vel[k][j-1][i  ].x2,
                       Vel[k][j+1][i-1].x2 - Vel[k][j  ][i-1].x2,
                       Vel[k][j  ][i-1].x2 - Vel[k][j-1][i-1].x2);
      dVydy /= pG->dx2;
      
      /* Monotonized Velocity gradient dVz/dy */
      dVzdy = limiter4(Vel[k][j+1][i  ].x3 - Vel[k][j  ][i  ].x3,
                       Vel[k][j  ][i  ].x3 - Vel[k][j-1][i  ].x3,
                       Vel[k][j+1][i-1].x3 - Vel[k][j  ][i-1].x3,
                       Vel[k][j  ][i-1].x3 - Vel[k][j-1][i-1].x3);
      dVzdy /= pG->dx2;
      
      /* Monotonized Velocity gradient dVx/dz, 3D problem ONLY */
      if (pD->Nx[2] > 1) {
        dVxdz = limiter4(Vel[k+1][j][i  ].x1 - Vel[k  ][j][i  ].x1,
                         Vel[k  ][j][i  ].x1 - Vel[k-1][j][i  ].x1,
                         Vel[k+1][j][i-1].x1 - Vel[k  ][j][i-1].x1,
                         Vel[k  ][j][i-1].x1 - Vel[k-1][j][i-1].x1);
        dVxdz /= pG->dx3;
      }
      
      /* Monotonized Velocity gradient dVy/dz */
      if (pD->Nx[2] > 1) {
        dVydz = limiter4(Vel[k+1][j][i  ].x2 - Vel[k  ][j][i  ].x2,
                         Vel[k  ][j][i  ].x2 - Vel[k-1][j][i  ].x2,
                         Vel[k+1][j][i-1].x2 - Vel[k  ][j][i-1].x2,
                         Vel[k  ][j][i-1].x2 - Vel[k-1][j][i-1].x2);
        dVydz /= pG->dx3;
      }
      
      /* Monotonized Velocity gradient dVz/dz */
      if (pD->Nx[2] > 1) {
        dVzdz = limiter4(Vel[k+1][j][i  ].x3 - Vel[k  ][j][i  ].x3,
                         Vel[k  ][j][i  ].x3 - Vel[k-1][j][i  ].x3,
                         Vel[k+1][j][i-1].x3 - Vel[k  ][j][i-1].x3,
                         Vel[k  ][j][i-1].x3 - Vel[k-1][j][i-1].x3);
        dVzdz /= pG->dx3;
      }
      
/* Compute field components at x1-interface */

      Bx = pG->B1i[k][j][i];
      By = 0.5*(pG->U[k][j][i].B2c + pG->U[k][j][i-1].B2c);
      Bz = 0.5*(pG->U[k][j][i].B3c + pG->U[k][j][i-1].B3c);
      B02 = Bx*Bx + By*By + Bz*Bz;
      B02 = MAX(B02,TINY_NUMBER);  /* limit in case B=0 */

/* compute BBdV and div(V) */

      if (pD->Nx[2] == 1) {
        BBdV =
          Bx*(Bx*(Vel[k][j][i].x1-Vel[k][j][i-1].x1)/pG->dx1 + By*dVxdy) +
          By*(Bx*(Vel[k][j][i].x2-Vel[k][j][i-1].x2)/pG->dx1 + By*dVydy) +
          Bz*(Bx*(Vel[k][j][i].x3-Vel[k][j][i-1].x3)/pG->dx1 + By*dVzdy);
        BBdV /= B02;

        divV = (Vel[k][j][i].x1-Vel[k][j][i-1].x1)/pG->dx1 + dVydy;

      } else {
        BBdV =
          Bx*(Bx*(Vel[k][j][i].x1-Vel[k][j][i-1].x1)/pG->dx1+By*dVxdy+Bz*dVxdz)+
          By*(Bx*(Vel[k][j][i].x2-Vel[k][j][i-1].x2)/pG->dx1+By*dVydy+Bz*dVydz)+
          Bz*(Bx*(Vel[k][j][i].x3-Vel[k][j][i-1].x3)/pG->dx1+By*dVzdy+Bz*dVzdz);
        BBdV /= B02;

        divV = (Vel[k][j][i].x1-Vel[k][j][i-1].x1)/pG->dx1 + dVydy + dVzdz;
      }

/* Add fluxes */

      nud = nu_aniso*0.5*(pG->U[k][j][i].d + pG->U[k][j][i-1].d);
      qa = nud*(BBdV - ONE_3RD*divV);

      VStress.Mx = qa*(3.0*Bx*Bx/B02 - 1.0);
      VStress.My = qa*(3.0*By*Bx/B02);
      VStress.Mz = qa*(3.0*Bz*Bx/B02);

      x1Flux[k][j][i].Mx += VStress.Mx;
      x1Flux[k][j][i].My += VStress.My;
      x1Flux[k][j][i].Mz += VStress.Mz;

#ifndef BAROTROPIC
      x1Flux[k][j][i].E =
         0.5*(Vel[k][j][i-1].x1 + Vel[k][j][i].x1)*VStress.Mx +
         0.5*(Vel[k][j][i-1].x2 + Vel[k][j][i].x2)*VStress.My +
         0.5*(Vel[k][j][i-1].x3 + Vel[k][j][i].x3)*VStress.Mz;
#endif /* BAROTROPIC */
    }
  }}

/* Compute viscous fluxes in 2-direction, centered at X2-Faces ---------------*/

  for (k=ks; k<=ke; k++) {
  for (j=js; j<=je+1; j++) {
    for (i=is; i<=ie; i++) {

      /* Monotonized Velocity gradient dVx/dx */
      dVxdx = limiter4(Vel[k][j  ][i+1].x1 - Vel[k][j  ][i  ].x1,
                       Vel[k][j  ][i  ].x1 - Vel[k][j  ][i-1].x1,
                       Vel[k][j-1][i+1].x1 - Vel[k][j-1][i  ].x1,
                       Vel[k][j-1][i  ].x1 - Vel[k][j-1][i-1].x1);
      dVxdx /= pG->dx1;
      
      /* Monotonized Velocity gradient dVy/dx */
      dVydx = limiter4(Vel[k][j  ][i+1].x2 - Vel[k][j  ][i  ].x2,
                       Vel[k][j  ][i  ].x2 - Vel[k][j  ][i-1].x2,
                       Vel[k][j-1][i+1].x2 - Vel[k][j-1][i  ].x2,
                       Vel[k][j-1][i  ].x2 - Vel[k][j-1][i-1].x2);
      dVydx /= pG->dx1;
      
      /* Monotonized Velocity gradient dVz/dx */
      dVzdx = limiter4(Vel[k][j  ][i+1].x3 - Vel[k][j  ][i  ].x3,
                       Vel[k][j  ][i  ].x3 - Vel[k][j  ][i-1].x3,
                       Vel[k][j-1][i+1].x3 - Vel[k][j-1][i  ].x3,
                       Vel[k][j-1][i  ].x3 - Vel[k][j-1][i-1].x3);
      dVzdx /= pG->dx1;
      
      /* Monotonized Velocity gradient dVx/dz */
      if (pD->Nx[2] > 1) {
        dVxdz = limiter4(Vel[k+1][j  ][i].x1 - Vel[k  ][j  ][i].x1,
                         Vel[k  ][j  ][i].x1 - Vel[k-1][j  ][i].x1,
                         Vel[k+1][j-1][i].x1 - Vel[k  ][j-1][i].x1,
                         Vel[k  ][j-1][i].x1 - Vel[k-1][j-1][i].x1);
        dVxdz /= pG->dx3;
      }
      
      /* Monotonized Velocity gradient dVy/dz */
      if (pD->Nx[2] > 1) {
        dVydz =limiter4(Vel[k+1][j  ][i].x2 - Vel[k  ][j  ][i].x2,
                        Vel[k  ][j  ][i].x2 - Vel[k-1][j  ][i].x2,
                        Vel[k+1][j-1][i].x2 - Vel[k  ][j-1][i].x2,
                        Vel[k  ][j-1][i].x2 - Vel[k-1][j-1][i].x2);
        dVydz /= pG->dx3;
      }
      
      /* Monotonized Velocity gradient dVz/dz */
      if (pD->Nx[2] > 1) {
        dVzdz =limiter4(Vel[k+1][j  ][i].x3 - Vel[k  ][j  ][i].x3,
                        Vel[k  ][j  ][i].x3 - Vel[k-1][j  ][i].x3,
                        Vel[k+1][j-1][i].x3 - Vel[k  ][j-1][i].x3,
                        Vel[k  ][j-1][i].x3 - Vel[k-1][j-1][i].x3);
        dVzdz /= pG->dx3;
      }
      
/* Compute field components at x2-interface */

      Bx = 0.5*(pG->U[k][j][i].B1c + pG->U[k][j-1][i].B1c);
      By = pG->B2i[k][j][i];
      Bz = 0.5*(pG->U[k][j][i].B3c + pG->U[k][j-1][i].B3c);
      B02 = Bx*Bx + By*By + Bz*Bz;
      B02 = MAX(B02,TINY_NUMBER); /* limit in case B=0 */

/* compute BBdV and div(V) */

      if (pD->Nx[2] == 1) {
        BBdV =
          Bx*(Bx*dVxdx + By*(Vel[ks][j][i].x1-Vel[ks][j-1][i].x1)/pG->dx2) +
          By*(Bx*dVydx + By*(Vel[ks][j][i].x2-Vel[ks][j-1][i].x2)/pG->dx2) +
          Bz*(Bx*dVzdx + By*(Vel[ks][j][i].x3-Vel[ks][j-1][i].x3)/pG->dx2);
        BBdV /= B02;

        divV = dVxdx + (Vel[ks][j][i].x2-Vel[ks][j-1][i].x2)/pG->dx2;

      } else {
        BBdV =
          Bx*(Bx*dVxdx+By*(Vel[k][j][i].x1-Vel[k][j-1][i].x1)/pG->dx2+Bz*dVxdz)+
          By*(Bx*dVydx+By*(Vel[k][j][i].x2-Vel[k][j-1][i].x2)/pG->dx2+Bz*dVydz)+
          Bz*(Bx*dVzdx+By*(Vel[k][j][i].x3-Vel[k][j-1][i].x3)/pG->dx2+Bz*dVzdz);
        BBdV /= B02;

        divV = dVxdx + (Vel[k][j][i].x2-Vel[k][j-1][i].x2)/pG->dx2 + dVzdz;
      }

/* Add fluxes */

      nud = nu_aniso*0.5*(pG->U[k][j][i].d + pG->U[k][j-1][i].d);
      qa = nud*(BBdV - ONE_3RD*divV);

      VStress.Mx = qa*(3.0*Bx*By/B02);
      VStress.My = qa*(3.0*By*By/B02 - 1.0);
      VStress.Mz = qa*(3.0*Bz*By/B02);

      x2Flux[k][j][i].Mx += VStress.Mx;
      x2Flux[k][j][i].My += VStress.My;
      x2Flux[k][j][i].Mz += VStress.Mz;

#ifndef BAROTROPIC
        x2Flux[k][j][i].E +=
           0.5*(Vel[k][j-1][i].x1 + Vel[k][j][i].x1)*VStress.Mx +
           0.5*(Vel[k][j-1][i].x2 + Vel[k][j][i].x2)*VStress.My +
           0.5*(Vel[k][j-1][i].x3 + Vel[k][j][i].x3)*VStress.Mz;
#endif /* BAROTROPIC */
    }
  }}

/* Compute viscous fluxes in 3-direction, centered at x3-Faces ---------------*/

  if (pD->Nx[2] > 1) {
    for (k=ks; k<=ke+1; k++) {
    for (j=js; j<=je; j++) { 
      for (i=is; i<=ie; i++) {

        /* Monotonized Velocity gradient dVx/dx */
        dVxdx = limiter4(Vel[k  ][j][i+1].x1 - Vel[k  ][j][i  ].x1,
                         Vel[k  ][j][i  ].x1 - Vel[k  ][j][i-1].x1,
                         Vel[k-1][j][i+1].x1 - Vel[k-1][j][i  ].x1,
                         Vel[k-1][j][i  ].x1 - Vel[k-1][j][i-1].x1);
        dVxdx /= pG->dx1;
        
        /* Monotonized Velocity gradient dVy/dx */
        dVydx = limiter4(Vel[k  ][j][i+1].x2 - Vel[k  ][j][i  ].x2,
                         Vel[k  ][j][i  ].x2 - Vel[k  ][j][i-1].x2,
                         Vel[k-1][j][i+1].x2 - Vel[k-1][j][i  ].x2,
                         Vel[k-1][j][i  ].x2 - Vel[k-1][j][i-1].x2);
        dVydx /= pG->dx1;
        
        /* Monotonized Velocity gradient dVz/dx */
        dVzdx = limiter4(Vel[k  ][j][i+1].x3 - Vel[k  ][j][i  ].x3,
                         Vel[k  ][j][i  ].x3 - Vel[k  ][j][i-1].x3,
                         Vel[k-1][j][i+1].x3 - Vel[k-1][j][i  ].x3,
                         Vel[k-1][j][i  ].x3 - Vel[k-1][j][i-1].x3);
        dVzdx /= pG->dx1;
        
        /* Monotonized Velocity gradient dVx/dy */
        dVxdy = limiter4(Vel[k  ][j+1][i].x1 - Vel[k  ][j  ][i].x1,
                         Vel[k  ][j  ][i].x1 - Vel[k  ][j-1][i].x1,
                         Vel[k-1][j+1][i].x1 - Vel[k-1][j  ][i].x1,
                         Vel[k-1][j  ][i].x1 - Vel[k-1][j-1][i].x1);
        dVxdy /= pG->dx2;
        
        /* Monotonized Velocity gradient dVy/dy */
        dVydy = limiter4(Vel[k  ][j+1][i].x2 - Vel[k  ][j  ][i].x2,
                         Vel[k  ][j  ][i].x2 - Vel[k  ][j-1][i].x2,
                         Vel[k-1][j+1][i].x2 - Vel[k-1][j  ][i].x2,
                         Vel[k-1][j  ][i].x2 - Vel[k-1][j-1][i].x2);
        dVydy /= pG->dx2;
        
        /* Monotonized Velocity gradient dVz/dy */
        dVzdy = limiter4(Vel[k  ][j+1][i].x3 - Vel[k  ][j  ][i].x3,
                         Vel[k  ][j  ][i].x3 - Vel[k  ][j-1][i].x3,
                         Vel[k-1][j+1][i].x3 - Vel[k-1][j  ][i].x3,
                         Vel[k-1][j  ][i].x3 - Vel[k-1][j-1][i].x3);
        dVzdy /= pG->dx2;
        
/* Compute field components at x3-interface */

        Bx = 0.5*(pG->U[k][j][i].B1c + pG->U[k-1][j][i].B1c);
        By = 0.5*(pG->U[k][j][i].B2c + pG->U[k-1][j][i].B2c);
        Bz = pG->B3i[k][j][i];
        B02 = Bx*Bx + By*By + Bz*Bz;
        B02 = MAX(B02,TINY_NUMBER); /* limit in case B=0 */

/* compute BBdV and div(V) */

        BBdV =
          Bx*(Bx*dVxdx+By*dVxdy+Bz*(Vel[k][j][i].x1-Vel[k-1][j][i].x1)/pG->dx3)+
          By*(Bx*dVydx+By*dVydy+Bz*(Vel[k][j][i].x2-Vel[k-1][j][i].x2)/pG->dx3)+
          Bz*(Bx*dVzdx+By*dVzdy+Bz*(Vel[k][j][i].x3-Vel[k-1][j][i].x3)/pG->dx3);
        BBdV /= B02;

        divV = dVxdx + dVydy + (Vel[k][j][i].x3-Vel[k-1][j][i].x3)/pG->dx3;

/* Add fluxes */

        nud = nu_aniso*0.5*(pG->U[k][j][i].d + pG->U[k-1][j][i].d);
        qa = nud*(BBdV - ONE_3RD*divV);

        VStress.Mx = qa*(3.0*Bx*Bz/B02);
        VStress.My = qa*(3.0*By*Bz/B02);
        VStress.Mz = qa*(3.0*Bz*Bz/B02 - 1.0);

        x3Flux[k][j][i].Mx += VStress.Mx;
        x3Flux[k][j][i].My += VStress.My;
        x3Flux[k][j][i].Mz += VStress.Mz;

#ifndef BAROTROPIC
        x3Flux[k][j][i].E  +=
           0.5*(Vel[k-1][j][i].x1 + Vel[k][j][i].x1)*VStress.Mx +
           0.5*(Vel[k-1][j][i].x2 + Vel[k][j][i].x2)*VStress.My +
           0.5*(Vel[k-1][j][i].x3 + Vel[k][j][i].x3)*VStress.Mz;
#endif /* BAROTROPIC */
      }
    }}
  }
#endif /* MHD */

  return;
}

/*----------------------------------------------------------------------------*/
/* limiter2 and limiter4: call slope limiters to preserve monotonicity                                       
 */

static Real limiter2(const Real A, const Real B)
{
  /* van Leer slope limiter */
  return vanleer(A,B);
  
  /* monotonized central (MC) limiter */
  /* return minmod(2.0*minmod(A,B),0.5*(A+B)); */
}

static Real limiter4(const Real A, const Real B, const Real C, const Real D)
{
  return limiter2(limiter2(A,B),limiter2(C,D));
}

/*----------------------------------------------------------------------------*/
/* vanleer: van Leer slope limiter                                                                           
 */

static Real vanleer(const Real A, const Real B)
{
  if (A*B > 0) {
    return 2.0*A*B/(A+B);
  } else {
    return 0.0;
  }
}

/*----------------------------------------------------------------------------*/
/* minmod: minmod slope limiter                                                                              
 */

static Real minmod(const Real A, const Real B)
{
  if (A*B > 0) {
    if (A > 0) {
      return MIN(A,B);
    } else {
      return MAX(A,B);
    }
  } else {
    return 0.0;
  }
}

/*----------------------------------------------------------------------------*/
/*! \fn void viscosity_init(MeshS *pM)
 *  \brief Allocate temporary arrays
 */

void viscosity_init(MeshS *pM)
{
  int nl,nd,size1=1,size2=1,size3=1,Nx1,Nx2,Nx3;

/* Cycle over all Grids on this processor to find maximum Nx1, Nx2, Nx3 */
  for (nl=0; nl<(pM->NLevels); nl++){
    for (nd=0; nd<(pM->DomainsPerLevel[nl]); nd++){
      if (pM->Domain[nl][nd].Grid != NULL) {
        if (pM->Domain[nl][nd].Grid->Nx[0] > size1){
          size1 = pM->Domain[nl][nd].Grid->Nx[0];
        }
        if (pM->Domain[nl][nd].Grid->Nx[1] > size2){
          size2 = pM->Domain[nl][nd].Grid->Nx[1];
        }
        if (pM->Domain[nl][nd].Grid->Nx[2] > size3){
          size3 = pM->Domain[nl][nd].Grid->Nx[2];
        }
      }
    }
  }

  Nx1 = size1 + 2*nghost;

  if (pM->Nx[1] > 1){
    Nx2 = size2 + 2*nghost;
  } else {
    Nx2 = size2;
  }

  if (pM->Nx[2] > 1){
    Nx3 = size3 + 2*nghost;
  } else {
    Nx3 = size3;
  }

  if ((x1Flux = (ViscFluxS***)calloc_3d_array(Nx3,Nx2,Nx1, sizeof(ViscFluxS)))
    == NULL) goto on_error;
  if ((x2Flux = (ViscFluxS***)calloc_3d_array(Nx3,Nx2,Nx1, sizeof(ViscFluxS)))
    == NULL) goto on_error;
  if ((x3Flux = (ViscFluxS***)calloc_3d_array(Nx3,Nx2,Nx1, sizeof(ViscFluxS)))
    == NULL) goto on_error;
  if ((Vel = (Real3Vect***)calloc_3d_array(Nx3,Nx2,Nx1, sizeof(Real3Vect)))
    == NULL) goto on_error;
  if ((divv = (Real***)calloc_3d_array(Nx3,Nx2,Nx1, sizeof(Real))) == NULL)
    goto on_error;
  return;

  on_error:
  viscosity_destruct();
  ath_error("[viscosity_init]: malloc returned a NULL pointer\n");
  return;
}

/*----------------------------------------------------------------------------*/
/*! \fn void viscosity_destruct(void) 
 *  \brief Free temporary arrays
 */      

void viscosity_destruct(void)
{   
  if (x1Flux != NULL) free_3d_array(x1Flux);
  if (x2Flux != NULL) free_3d_array(x2Flux);
  if (x3Flux != NULL) free_3d_array(x3Flux);
  if (Vel != NULL) free_3d_array(Vel);
  if (divv != NULL) free_3d_array(divv);
  return;
}
#endif /* VISCOSITY */
