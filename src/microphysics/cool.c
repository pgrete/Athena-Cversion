#include "../copyright.h"
/*============================================================================*/
/*! \file cool.c
 *  \brief Implements various optically thin cooling functions.  
 *
 *  These can be
 *  enrolled by setting CoolingFunc=NAME in the problem generator, where NAME
 *  is one of the functions in this file.
 *
 *  Each cooling function returns the cooling rate per volume.  The total 
 *  (or equivalently the internal) energy then evolves as
 *   -   dE/dt = de/dt = - CoolingFunc
 *
 *  Some of these cooling functions return the cooling rate per volume in
 *  cgs units [ergs/cm^{3}/s].  Thus, to use these functions, the entire
 *  calculation must be in cgs, or else the cooling rate has to scaled
 *  appropriately in the calling function. 
 *
 *  To add a new cooling function, implement it below and add the name to 
 *  src/microphysics/prototypes.h.  Note the argument list must be (d,P,dt).
 *
 * CONTAINS PUBLIC FUNCTIONS:
 * - KoyInut() - Koyama & Inutsuka cooling function */
/*============================================================================*/

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "../defs.h"
#include "../athena.h"
#include "../globals.h"
#include "prototypes.h"
#include "../prototypes.h"

/* These constants are in cgs */
static const Real mbar = (1.37)*(1.6733e-24);
static const Real kb = 1.380658e-16;
static const Real HeatRate = 2.0e-26;
static const Real Tmin = 10;

/* Minimum pressure in parameter cooling function */
static Real CoolMinPres;
/* Power index of density in parameter cooling function */
static Real CoolIdxRho;
/* Power index of temperature/internal energy in parameter cooling function */
static Real CoolIdxT;
/* Prefactor in parameter cooling function */
static Real CoolPrefac;

/*=========================== PUBLIC FUNCTIONS ===============================*/
/*----------------------------------------------------------------------------*/
/*! \fn Real KoyInut(const Real dens, const Real Press, const Real dt)
 *  \brief Analytic fit to cooling in the diffuse ISM given by eq. (4) in
 *   Koyama & Inutsuka, ApJ 564, L97 (2002);  Returns rate in cgs.
 */

#ifndef BAROTROPIC
Real KoyInut(const Real dens, const Real Press, const Real dt)
{
  Real n,coolrate=0.0;
  Real T,coolratepp,MaxdT,dT;
	Real Teq, logn, lognT;

/* Compute number density and Temperature */
  n = dens/mbar;
	logn = log10(n);
  T = MAX((Press/(n*kb)),Tmin);

/* Compute the minimun Temperature*/
    Teq = Tmin;

/* KI cooling rate per particle */
  coolratepp = HeatRate*
   (n*(1.0e7*exp(-1.184e5/(T+1000.)) + 0.014*sqrt(T)*exp(-92.0/T)) - 1.0);

/* Expected dT by KI cooling rate */
  dT = coolratepp*dt*Gamma_1/kb;

  if ((T-dT) <= 185.0){
    lognT = 3.9247499 - 1.8479378*logn + 1.5335032*logn*logn
     -0.47665872*pow(logn,3) + 0.076789136*pow(logn,4)-0.0049052587*pow(logn,5);
    Teq = pow(10.0,lognT) / n;
  }

/* Compute maximum change in T allowed to keep T positive, and limit cooling
 * rate to this value */
  MaxdT = kb*(T-Teq)/(dt*Gamma_1);
  coolrate = MIN(coolratepp,MaxdT);
  return n*coolrate;
}

void InitCooling() {
  CoolIdxRho = par_getd("problem","CoolIdxRho");
  CoolIdxT = par_getd("problem","CoolIdxT");
  CoolPrefac = par_getd("problem","CoolPrefac");
  CoolMinPres = par_getd("problem","CoolMinPres");
}

Real ParamCool(const Real dens, const Real Press, const Real dt)
{
  if (Press < CoolMinPres) {
    return(0.);
  }

  Real eInt = Press / Gamma_1 / dens;
  Real dEdt = CoolPrefac * pow(dens,CoolIdxRho) * pow(eInt,CoolIdxT);

/* Expected pressure after update */

  Real newPress = Press - dt*Gamma_1*dEdt;

  if (newPress < CoolMinPres) {
    printf("Limiting cooling to %.3g (from %.3g).\n",
      (Press - CoolMinPres)/(dt*Gamma_1),dEdt);
    dEdt = (Press - CoolMinPres)/(dt*Gamma_1);
  }

  return dEdt;
}
#endif /* BAROTROPIC */
