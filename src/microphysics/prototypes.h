#ifndef MICROPHYS_PROTOTYPES_H
#define MICROPHYS_PROTOTYPES_H 
#include "../copyright.h"
/*============================================================================*/
/*! \file prototypes.h
 *  \brief Prototypes for all public functions in the /src/microphysics dir */
/*============================================================================*/
#include <stdio.h>
#include <stdarg.h>
#include "../athena.h"
#include "../defs.h"

#include "../config.h"

/* conduction.c */
#ifdef THERMAL_CONDUCTION
void conduction(DomainS *pD);
void conduction_init(MeshS *pM);
void conduction_destruct(void);
#endif

/* cool.c */
Real KoyInut(const Real dens, const Real Press, const Real dt);
void InitCooling();
Real ParamCool(const Real dens, const Real Press, const Real dt);

/* get_eta.c */
#ifdef RESISTIVITY
void get_eta(GridS *pG);
void eta_standard(GridS *pG, int i, int j, int k,
                  Real *eta_O, Real *eta_H, Real *eta_A);
void convert_diffusion(Real sigma_O, Real sigma_H, Real sigma_P,
                       Real *eta_O,  Real *eta_H,  Real *eta_A );
#endif

/* integrate_diffusion.c */
void integrate_diff(MeshS *pM);
void integrate_diff_init(MeshS *pM);
void integrate_diff_destruct(void);

/* new_dt_diff.c */
Real new_dt_diff(MeshS *pM);

/* resistivity.c */
#ifdef RESISTIVITY
void resistivity(DomainS *pD);
void resistivity_init(MeshS *pM);
void resistivity_destruct();
#endif

/* viscosity.c */
#ifdef VISCOSITY
void viscosity(DomainS *pD);
void viscosity_init(MeshS *pM);
void viscosity_destruct(void);
#endif

/* integrate_cooling.c */
void integrate_cooling(MeshS *pM);
void integrate_cooling_init(MeshS *pM);
void integrate_cooling_destruct(void);

#ifdef OPERATOR_SPLIT_COOLING
/* cool_solver.c */
void cooling_solver(GridS *pG);
void cooling_solver_init(MeshS *pM);
void cooling_solver_destruct(void);
#endif


#endif /* MICROPHYS_PROTOTYPES_H */
