#include "copyright.h"
/*============================================================================*/
/*! \file SF.c
 *  \brief Problem generator for stochastically driven turbulence.
 *
 * PURPOSE: Problem generator for driven turbulence. Only works
 *   in 3D with periodic BC. 
 * PARAMETERS:
 *  - beta
 *      plasma beta for initial uniform magnetic field (x-direction)
 *  - SolenoidalWeight
 *      Weight of power in solenoidal modes of the acceleration field.
 *      0. means purely compressive forcing with curl(A) = 0
 *      1. means purely solenoidal forcing with Div(A) = 0
 *      Can bet set to any value betweeen 0 and 1
 *  - CharacteristicWavenumber
 *      Peak of the parabolic forcing spectrum in unis of 2 pi / L, i.e.,
 *       a value of 2 corresponds to a large scale characteristic length of
 *       half the box size
 *   - ForcingAmpl
 *      Amplitude (RMS value) that the acceleration field is normalized to on
 *      every timestep before the velocities are updated.
 *   - ForcingTime
 *      Autocorrelation time of the forcing (in units of code_time) on which the
 *      acceleration field evolves.
 *      For a delta in time forcing (not recommended) set below the smallest 
 *      timestep in the simulation.
 *  - rseed
 *      Seed value for random number generator. For fresh simulations this number
 *      must be negative in order to properly initialize a different RNG on each
 *      process.
 *
 *  HISTORY:
 *  - This version written by P. Grete, 2017/18
 *  - Problem generator based on turb.c originally written 
 *    by H. Stone and N. Lemaster
 *
 *  REFERENCE: tbd
 *                                  			      */
/*============================================================================*/

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "defs.h"
#include "athena.h"
#include "prototypes.h"
#include "globals.h"
#include "p3dfft.h"

#ifdef MPI_PARALLEL
#include "mpi.h"
#ifdef DOUBLE_PREC
#define MPI_RL MPI_DOUBLE
#else /* DOUBLE_PREC */
#define MPI_RL MPI_FLOAT
#endif /* DOUBLE_PREC */
#endif /* MPI_PARALLEL */

#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NDIV (1+IMM1/NTAB)
#define RNMX (1.0-DBL_EPSILON)
#define NTAB 32
  
static long int idum2=123456789;
static long int iy=0;
static long int iv[NTAB];

/*! \fn double ran2(long int *idum){
 *  \brief The routine ran2() is extracted from the Numerical Recipes in C 
 *
 * The routine ran2() is extracted from the Numerical Recipes in C
 * (version 2) code.  I've modified it to use doubles instead of
 * floats. -- T. A. Gardiner -- Aug. 12, 2003 
 *
 * Long period (> 2 x 10^{18}) random number generator of L'Ecuyer
 * with Bays-Durham shuffle and added safeguards.  Returns a uniform
 * random deviate between 0.0 and 1.0 (exclusive of the endpoint
 * values).  Call with idum = a negative integer to initialize;
 * thereafter, do not alter idum between successive deviates in a
 * sequence.  RNMX should appriximate the largest floating point value
 * that is less than 1. */

double ran2(long int *idum){
  int j;
  long int k;
  double temp;

  if (*idum <= 0) { /* Initialize */
    if (-(*idum) < 1) *idum=1; /* Be sure to prevent idum = 0 */
    else *idum = -(*idum);
    idum2=(*idum);
    for (j=NTAB+7;j>=0;j--) { /* Load the shuffle table (after 8 warm-ups) */
      k=(*idum)/IQ1;
      *idum=IA1*(*idum-k*IQ1)-k*IR1;
      if (*idum < 0) *idum += IM1;
      if (j < NTAB) iv[j] = *idum;
    }
    iy=iv[0];
  }
  k=(*idum)/IQ1;                 /* Start here when not initializing */
  *idum=IA1*(*idum-k*IQ1)-k*IR1; /* Compute idum=(IA1*idum) % IM1 without */
  if (*idum < 0) *idum += IM1;   /* overflows by Schrage's method */
  k=idum2/IQ2;
  idum2=IA2*(idum2-k*IQ2)-k*IR2; /* Compute idum2=(IA2*idum) % IM2 likewise */
  if (idum2 < 0) idum2 += IM2;
  j=(int)(iy/NDIV);              /* Will be in the range 0...NTAB-1 */
  iy=iv[j]-idum2;                /* Here idum is shuffled, idum and idum2 */
  iv[j] = *idum;                 /* are combined to generate output */
  if (iy < 1) iy += IMM1;
  if ((temp=AM*iy) > RNMX) return RNMX; /* No endpoint values */
  else return temp;
}

#undef IM1
#undef IM2
#undef AM
#undef IMM1
#undef IA1
#undef IA2
#undef IQ1
#undef IQ2
#undef IR1
#undef IR2
#undef NDIV
#undef RNMX

typedef double Complex[2];

/* Between calls to generate(), these have unshifted, unnormalized
 * velocity perturbations. */
static Complex *fv1=NULL, *fv2=NULL, *fv3=NULL;
/* Given that the inverse FFT overwrite the inputs, we need a persistent
 * storage for the spectral acceleration field so that it can be evolved.
 * */
static Complex *fv1Persistent=NULL, *fv2Persistent=NULL, *fv3Persistent=NULL;
/* Acceleration field in real space */
static Real *dv1=NULL, *dv2=NULL, *dv3=NULL;

/* Cutoff wavenumbers, G&O spect peak, power law spect exponent, 2 pi/L */
static Real klow,khigh,kpeak;
/* Amplitude, autocorrelation time and weight of solenoidal component of the
 * acceleration field */
static Real ForcingAmpl, ForcingTime, SolWeight;
/* Number of cells in local grid, number of cells in global grid */
static int nx1,nx2,nx3,gnx1,gnx2,gnx3;
/* Global dimensions for P3DFFT. Are set during initialization according
 * to striding.*/
static int P3Dng[3];
/* Dimension for the local complex P3D grid.
 *P3DCst - start indices (fortran style starting with 1)
 *P3DCse - end indices (fortran style starting with 1)
 *P3DCsz - size
 * */
static int P3DCst[3],P3DCse[3],P3DCsz[3];
/* Starting and ending indices for global grid */
static int gis,gie,gjs,gje,gks,gke;
/* local coordinates */
Real x1,x2,x3;
/* Seed for random number generator */
long int rseed = -1;
static  double globalAmplNorm[1];
#ifdef MHD
/* beta = isothermal pressure / magnetic pressure
 * B0 = sqrt(2.0*Iso_csound2*rhobar/beta) is init magnetic field strength */
static Real beta,B0;
/* magnetic field configuration
 * 0 - uniform in one direction
 * 1 - no net flux, with uniform in opposite directions
 */
static int BFieldConfig;
#endif /* MHD */
/* Initial density (will be average density throughout simulation) */
static const Real rhobar = 1.0;

/* Initial pressure. */
static Real p0;

/* Functions appear in this file in the same order that they appear in the
 * prototypes below */

/* Function prototypes for generating velocity perturbations */
static void inject(Complex *ampl);
static void project(Complex *ampl1,Complex *ampl2,Complex *ampl3);
static inline void transform();
static inline void generate(Real dt);
static void perturb(GridS *pGrid, Real dt);

/* Function prototypes for initializing and interfacing with Athena */
static void initialize(GridS *pGrid, DomainS *pD);
/* void problem(Grid *pGrid, Domain *pD); */
/* void Userwork_in_loop(Grid *pGrid, Domain *pD); */
/* void Userwork_after_loop(Grid *pGrid, Domain *pD); */
/* void problem_write_restart(Grid *pG, Domain *pD, FILE *fp); */
/* void problem_read_restart(Grid *pG, Domain *pD, FILE *fp); */
/* Gasfun_t get_usr_expr(const char *expr); */

/* Function prototypes for analysis and outputs */
static Real hst_dEk(const GridS *pG, const int i, const int j, const int k);
static Real hst_dEb(const GridS *pG, const int i, const int j, const int k);
static Real hst_MeanMach(const GridS *pG, const int i, const int j, const int k);
static Real hst_MeanAlfvenicMach(const GridS *pG, const int i, const int j, const int k);
static Real hst_MeanPressure(const GridS *pG, const int i, const int j, const int k);
static Real hst_MeanPlasmaBeta(const GridS *pG, const int i, const int j, const int k);

/* Function prototypes for Numerical Recipes functions */
static double ran2(long int *idum);



/* ========================================================================== */

/*! \fn static void inject(Real *ampl)
 *  \brief computes component of velocity with specific power
 *  spectrum in Fourier space
 *
 *  Note that the fourier amplitudes are stored in an array with no
 *  ghost zones
 */
static void inject(Complex *ampl)
{
  int i,j,k,ind,mpierr;

  /* set power spectrum
   *  parabolic form from Schmidt et al. A&A, 2009, 494, 127-145
   */
  Real tmp;

  //klow = 0.;
  //khigh = 2. * kpeak; 

  int x,y,z,kx,ky,kz;
  Real q3;

  double v_sqr, v1, v2;
  double norm;
  
  for(x=0;x < P3DCsz[2];x++) {
    kx = x + (P3DCst[2]-1);
    for(y=0;y < P3DCsz[1];y++) {
      ky = y +P3DCst[1]-1;
      if(ky > P3Dng[1]/2)
        ky = ky - P3Dng[1];
      for(z=0;z < P3DCsz[0];z++) {
        kz = z +P3DCst[0]-1;
        if(kz > P3Dng[0]/2)
          kz = kz - P3Dng[0];

        q3 = sqrt((Real)(kx *kx +ky *ky + kz*kz));
        
        ind = z + P3DCsz[0]*(y + P3DCsz[1]*x);
        if ((q3 > klow) && (q3 < khigh)) {
          /* parabolic form */
          tmp = pow(q3/kpeak,2.)*(2.-pow(q3/kpeak,2.));
          if (tmp < 0.)
            tmp = 0.;
          ampl[ind][0] = tmp;
          ampl[ind][1] = tmp;
        
          /* Apply Gaussian deviations (from Numerical Recipes and W. Schmidt) */
          do {        
            v1 = 2.0* ran2(&rseed) - 1.0;
            v2 = 2.0* ran2(&rseed) - 1.0;
            v_sqr = v1*v1+v2*v2;
          } while (v_sqr >= 1.0 || v_sqr == 0.0);

          norm = sqrt(-2.0*log(v_sqr)/v_sqr);

          ampl[ind][0] *= norm * v1;
          ampl[ind][1] *= norm * v2;
          
        } else {
          /* introduce cut-offs at klow and khigh */
          ampl[ind][0] = 0.0;
          ampl[ind][1] = 0.0;
        }

      }
    }
  }

  return;
}

/* ========================================================================== */

/*! \fn static void project()
 *  \brief Projects the acceleration field to match the desired power
 *  in solenoidal versus compressive modes.
 */
static void project(Complex *ampl1,Complex *ampl2,Complex *ampl3)
{
  int i,j,k,m,ind;
  Real kap[3], kapn[3], mag;
  Complex dot;
  
  int x,y,z,kx,ky,kz;
  
  for(x=0;x < P3DCsz[2];x++) {
    kx = x + (P3DCst[2]-1);
    for(y=0;y < P3DCsz[1];y++) {
      ky = y +P3DCst[1]-1;
      if(ky > P3Dng[1]/2)
        ky = ky - P3Dng[1];
      for(z=0;z < P3DCsz[0];z++) {
        kz = z +P3DCst[0]-1;
        if(kz > P3Dng[0]/2)
          kz = kz - P3Dng[0];
        ind = z + P3DCsz[0]*(y + P3DCsz[1]*x);
          
        kap[0] = (Real)kx;
        kap[1] = (Real)ky;
        kap[2] = (Real)kz;
          
          /* make kapn a unit vector */
          mag = sqrt(SQR(kap[0]) + SQR(kap[1]) + SQR(kap[2]));
          if (mag == 0.)
              continue;

          for (m=0; m<3; m++) kapn[m] = kap[m] / mag;

          /* find ampl_0 dot kapn */
          dot[0] = ampl1[ind][0]*kapn[0]+ampl2[ind][0]*kapn[1]+ampl3[ind][0]*kapn[2];
          dot[1] = ampl1[ind][1]*kapn[0]+ampl2[ind][1]*kapn[1]+ampl3[ind][1]*kapn[2];

          /* ampl = ampl_0 - (ampl_0 dot kapn) * kapn */
          ampl1[ind][0] = SolWeight*ampl1[ind][0] + (1. - 2.*SolWeight) * dot[0]*kapn[0];
          ampl2[ind][0] = SolWeight*ampl2[ind][0] + (1. - 2.*SolWeight) * dot[0]*kapn[1];
          ampl3[ind][0] = SolWeight*ampl3[ind][0] + (1. - 2.*SolWeight) * dot[0]*kapn[2];

          ampl1[ind][1] = SolWeight*ampl1[ind][1] + (1. - 2.*SolWeight) * dot[1]*kapn[0];
          ampl2[ind][1] = SolWeight*ampl2[ind][1] + (1. - 2.*SolWeight) * dot[1]*kapn[1];
          ampl3[ind][1] = SolWeight*ampl3[ind][1] + (1. - 2.*SolWeight) * dot[1]*kapn[2];
        
      }
    }
  }

  return;
}

/* ========================================================================== */

/*! \fn static inline void transform()
 *  \brief Generate velocities from fourier transform
 */
static inline void transform()
{
  /* Transform velocities from k space to physical space */
  unsigned char op_f[]="fft", op_b[]="tff";
  Cp3dfft_btran_c2r((Real *)fv1,dv1,op_b);
  Cp3dfft_btran_c2r((Real *)fv2,dv2,op_b);
  Cp3dfft_btran_c2r((Real *)fv3,dv3,op_b);

  return;
}

/* ========================================================================== */

/*! \fn static inline void generate()
 *  \brief Evolve the acceleration field (persistent) in spectral and copy
 *  it to the the variable (injection) which is transform backwards.
 */
static void evolve(Complex *persistent, Complex *injection, Real dt)
{
  int i,j,k;
  double q3;
 
  int x,y,z,kx,ky,kz;
  
  /*Drift and diffusion coefficient of the OU process */
  Real driftCoeff = exp(-dt/ForcingTime);
  Real diffCoeff = sqrt(1. - driftCoeff*driftCoeff);
  
  int ind; 
  for(x=0;x < P3DCsz[2];x++) {
    for(y=0;y < P3DCsz[1];y++) {
      for(z=0;z < P3DCsz[0];z++) {
        ind = z + P3DCsz[0]*(y + P3DCsz[1]*x);
        persistent[ind][0] = 
          driftCoeff * persistent[ind][0] +
          diffCoeff * injection[ind][0];          
        persistent[ind][1] = 
          driftCoeff * persistent[ind][1] +
          diffCoeff * injection[ind][1];

        /* Now overwriting injection with actual field
         * as the injection array is used in the inverse FFT (thus overwritten)
         * and reconstructed on every timestep anyway
         * */

        injection[ind][0] = persistent[ind][0];
        injection[ind][1] = persistent[ind][1];
      }
    }
  }
}

/* ========================================================================== */

/*! \fn static inline void generate()
 *  \brief Generate the velocity perturbations
 */
static inline void generate(Real dt)
{
  /* Generate new perturbations following appropriate power spectrum 
   * fv_i are sigma(k) * N(0,1)_i now
   * */
  inject(fv1);
  inject(fv2);
  inject(fv3);

  /* Project, i.e. distribute power to compressive and solenoidal components
   * fv_i contains sigma(k) P_ij(k,SolenoidalWeight) N(0,1)_j 
   * */
  project(fv1,fv2,fv3);


  /* Evolve spectrum smoothly */

  evolve(fv1Persistent,fv1,dt);
  evolve(fv2Persistent,fv2,dt);
  evolve(fv3Persistent,fv3,dt);

  /* Transform perturbations to real space, but don't normalize until
   * just before we apply them in perturb() */
  transform();
  
  return;
}

/* ========================================================================== */

/*! \fn static void perturb(Grid *pGrid, Real dt)
 *  \brief  Normalizes acceleration field to given value and then sets velocities
 */
static void perturb(GridS *pGrid, Real dt)
{
  int i, is=pGrid->is, ie = pGrid->ie;
  int j, js=pGrid->js, je = pGrid->je;
  int k, ks=pGrid->ks, ke = pGrid->ke;
  int ind, mpierr;
  Real qa, dvol;
  
  int x,y,z,kx,ky,kz;
  int mytid;
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);

  double amplNorm[1];
  amplNorm[0] = 0.;


  /* Now we calculate the normalization. Thanks for Parseval we can just do it
   * on the real space field and thus don't need to be concerned with normalizing
   * the spectral profile and projection operator.
   */
  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {
        ind = i-is + nx1 * ((j-js) + nx2 * (k-ks));
        amplNorm[0] += SQR(dv1[ind]) + SQR(dv2[ind]) + SQR(dv3[ind]);
      }
    }
  }

#ifdef MPI_PARALLEL
  mpierr = MPI_Allreduce(amplNorm, globalAmplNorm, 1, MPI_RL, MPI_SUM, MPI_COMM_WORLD);
  if (mpierr) ath_error("[normalize]: MPI_Allreduce error = %d\n", mpierr);
#else
  globalAmplNorm[0] = amplNorm[0];
#endif /* MPI_PARALLEL */ 

  double gsize = ((long int) (gnx1)) *((long int) (gnx2)) * ((long int) (gnx3));
  
  globalAmplNorm[0] = ForcingAmpl / sqrt(globalAmplNorm[0]/gsize);

  /* Set the velocities in real space */
  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {
        ind = i-is + nx1 * ((j-js) + nx2 * (k-ks));
        
        qa = dt * pGrid->U[k][j][i].d * globalAmplNorm[0];
        
        pGrid->U[k][j][i].M1 += qa*dv1[ind];
        pGrid->U[k][j][i].M2 += qa*dv2[ind];
        pGrid->U[k][j][i].M3 += qa*dv3[ind];
#ifndef ISOTHERMAL
        pGrid->U[k][j][i].E += (
          pGrid->U[k][j][i].M1 * dt * globalAmplNorm[0] * dv1[ind] +
          pGrid->U[k][j][i].M2 * dt * globalAmplNorm[0] * dv2[ind] +
          pGrid->U[k][j][i].M3 * dt * globalAmplNorm[0] * dv3[ind] + (
          dv1[ind] * dv1[ind] + dv2[ind] * dv2[ind] + dv3[ind] * dv3[ind]
          ) * qa * qa /(2.* pGrid->U[k][j][i].d));
#endif /* ISOTHERMAL */
      }
    }
  }
  return;
}

/* ========================================================================== */
/*! \fn static void initialize(Grid *pGrid, Domain *pD)
 *  \brief  Allocate memory and initialize FFT plans */
static void initialize(GridS *pGrid, DomainS *pD)
{
  int i, is=pGrid->is, ie = pGrid->ie;
  int j, js=pGrid->js, je = pGrid->je;
  int k, ks=pGrid->ks, ke = pGrid->ke;
  int ixs,jxs,kxs;
  int nbuf, mpierr, nx1gh, nx2gh, nx3gh;
  float kwv, kpara, kperp;

  /* -----------------------------------------------------------
 * Variables within this block are stored globally, and used
 * within preprocessor macros.  
 */

  /* Get local grid size */
  nx1 = (ie-is+1);
  nx2 = (je-js+1);
  nx3 = (ke-ks+1);

  /* Get global grid size */
  gnx1 = pD->Nx[0];
  gnx2 = pD->Nx[1];
  gnx3 = pD->Nx[2];
  
  if ((gnx1 > 10480) || (gnx2 > 10480) || (gnx3 > 10480))
      ath_error("[problem]: Test/verify integer indices are big enough!\n");

  /* Get extents of local FFT grid in global coordinates */
  gis=is+pGrid->Disp[0]-nghost;  gie=ie+pGrid->Disp[0];
  gjs=js+pGrid->Disp[1]-nghost;  gje=je+pGrid->Disp[1];
  gks=ks+pGrid->Disp[2]-nghost;  gke=ke+pGrid->Disp[2];

/* ----------------------------------------------------------- */

  /* Get size of arrays with ghost cells */
  nx1gh = nx1 + 2*nghost;
  nx2gh = nx2 + 2*nghost;
  nx3gh = nx3 + 2*nghost;

#ifndef ISOTHERMAL

  p0 = par_getd_def("problem","p0",-1.0);

  /* This sets c_s = 1 throughout the box. */
  if (p0 == -1.0)
    p0 = 1./Gamma;
#endif /* ISOTHERMAL */

#ifdef VISCOSITY
  nu_iso = par_getd_def("problem","nu_iso",0.0);
  nu_aniso = par_getd_def("problem","nu_aniso",0.0);
#endif

#ifdef THERMAL_CONDUCTION
  kappa_iso = par_getd_def("problem","kappa_iso",0.0);
  kappa_aniso = par_getd_def("problem","kappa_aniso",0.0);
#endif
  
  /* Get input parameters */
#ifdef MHD
  /* magnetic field strength */
  beta = par_getd_def("problem","beta",-1.0);
  B0 = par_getd_def("problem","B0",0.0);

  if ((beta == -1.0) && (B0 == 0.0)) 
      ath_error("Please initialize beta or B0 for an MHD problem!\n");
  
  BFieldConfig = par_geti_def("problem","BFieldConfig",0);

  if (B0 == 0.0) {
#ifdef ISOTHERMAL
  /* beta = isothermal pressure/magnetic pressure */
    B0 = sqrt(2.0*Iso_csound2*rhobar/beta);
#else
    B0 = sqrt(2.0 * p0/beta);
#endif /* ISOTHERMAL */
  }
#endif /* MHD */
  
  /* determines weight of solenoidal relative to dilatational components */
  SolWeight = par_getd("problem","SolenoidalWeight");

  /* Forcing amplitude, i.e. the rms value of the acceleration field.
   * In incompressible turb this would just be A = U/T with
   * U charc./integral velocity and T integral time in the targeted
   * stationary regime.
   * However, in compressible and/or _M_HD turbulence this does not hold any 
   * more and should be determined by testing.
   * */
  ForcingAmpl = par_getd("problem","ForcingAmpl");

  /* Forcing time is the autocorrelation time of the acceleration field in
   * units of code time.
   * We are not using units of T here as T is not easily determined a priori
   * in compressible and/or _M_HD turbulence.
   * */
  ForcingTime = par_getd("problem","ForcingTime");

  /*
   * char. wavenumber where the forcing peaks 
   * */
  kpeak = par_getd("problem","CharacteristicWavenumber");
  klow =  par_getd("problem","klow");
  khigh = par_getd("problem","khigh");

#ifndef ISOTHERMAL
  if (par_geti_def("problem","Cooling",0) == 1) {
    CoolingFunc = ParamCool;
    InitCooling();
  }
#endif


  /* if this is a fresh seed from initial conditions (identified by being < 0) */
  if (rseed < 0) {
    rseed = (long)par_getd("problem","rseed");

    ixs = pGrid->Disp[0];
    jxs = pGrid->Disp[1];
    kxs = pGrid->Disp[2];
    /* make it unique seed for each MPI process */
    rseed -= (ixs + pD->Nx[0]*(jxs + pD->Nx[1]*kxs));
  } 

  /* Initialize the FFT plan */
  int nproc, proc_id;
  MPI_Comm_size(MPI_COMM_WORLD,&nproc);
  MPI_Comm_rank(MPI_COMM_WORLD,&proc_id);


  int fstart[3],fsize[3],fend[3];
  int memsize[3];
  int dims[2];
  int conf;

  
  dims[0] = dims[1]=0;
  MPI_Dims_create(nproc,2,dims);
  if(dims[0] > dims[1]) {
    dims[0] = dims[1];
    dims[1] = nproc/dims[0];
  }
  
  /* Test whether the domain decomposition among MPI processes
   * matches the expectation of P3DFFT.
   * */
  if (proc_id == 0) {
    if (par_geti("domain1","NGrid_x1") != 1)
      ath_error("Please use full pencil along x-axis, i.e. NGrid_x1 = 1.\n");

    if (par_geti("domain1","NGrid_x2") != dims[0])
      printf("WARNING: input domain decomp in y (%d) differs from P3DFFT recomm (%d)\n",
        par_geti("domain1","NGrid_x2"),dims[0]);

    if (par_geti("domain1","NGrid_x3") != dims[1])
      printf("WARNING: input domain decomp in z (%d) differs from P3DFFT recomm (%d)\n",
        par_geti("domain1","NGrid_x3"),dims[1]);
  }

  /* Initialize P3DFFT */ 
  Cp3dfft_setup(dims,gnx1,gnx2,gnx3,MPI_Comm_c2f(MPI_COMM_WORLD),gnx1,gnx2,gnx3,1,memsize);
  
  /* Get dimensions for input array - complex numbers, Z-pencil shape.
     Stride-1 dimension could be X or Z, depending on how the library
     was compiled (stride1 option).
     Note : returns Fortran-style dimensions, i.e. starting with 1. */ 
  conf = 2;
  Cp3dfft_get_dims(P3DCst,P3DCse,P3DCsz,conf);
  
  /* Get dimensions for input array - real numbers, X-pencil shape.
     Note that we are following the Fortran ordering, i.e.
     the dimension  with stride-1 is X. */
  conf = 1;
  Cp3dfft_get_dims(fstart,fend,fsize,conf);

  /* Set global FFT dimension for stride 1 */
  P3Dng[0] = gnx3;
  P3Dng[1] = gnx2;
  P3Dng[2] = gnx1;

  /* Make sure Athena and P3D grids are aligned.*/
  for (i = 0; i < 3; i++) {
      if (pGrid->Disp[i] != fstart[i]-1)
        ath_error("Mismatch between Athena and P3DFFT grid alignment in \
 dim %d. Start %d versus %d\n",i,pGrid->Disp[i],fstart[i]-1);
  }

  /* This should never occur if the previus loop passed. Just making sure. */
  if ((nx1 != fsize[0]) || (nx2 != fsize[1]) || (nx3 != fsize[2]))
      ath_error("Mismatch between Athena and P3DFFT local grid sizes.\n");

  /* Allocate memory for FFTs */
  fv1 = (Complex *) malloc(sizeof(Complex) * P3DCsz[0]*P3DCsz[1]*P3DCsz[2]);;
  fv2 = (Complex *) malloc(sizeof(Complex) * P3DCsz[0]*P3DCsz[1]*P3DCsz[2]);;
  fv3 = (Complex *) malloc(sizeof(Complex) * P3DCsz[0]*P3DCsz[1]*P3DCsz[2]);;

  fv1Persistent = (Complex *) malloc(sizeof(Complex) * P3DCsz[0]*P3DCsz[1]*P3DCsz[2]);;
  fv2Persistent = (Complex *) malloc(sizeof(Complex) * P3DCsz[0]*P3DCsz[1]*P3DCsz[2]);;
  fv3Persistent = (Complex *) malloc(sizeof(Complex) * P3DCsz[0]*P3DCsz[1]*P3DCsz[2]);;

  dv1 = (Real *) malloc(sizeof(Real) * fsize[0]*fsize[1]*fsize[2]);
  dv2 = (Real *) malloc(sizeof(Real) * fsize[0]*fsize[1]*fsize[2]);
  dv3 = (Real *) malloc(sizeof(Real) * fsize[0]*fsize[1]*fsize[2]);

  /* Calc initial spectrum 
   * */
  inject(fv1Persistent);
  inject(fv2Persistent);
  inject(fv3Persistent);

  project(fv1Persistent,fv2Persistent,fv3Persistent);

 
  /* Enroll outputs */
  dump_history_enroll(hst_dEk,"<dE_K>");
  dump_history_enroll(hst_dEb,"<dE_B>");
  dump_history_enroll(hst_MeanMach,"<SonicMach>");
  dump_history_enroll(hst_MeanAlfvenicMach,"<AlfvenicMach>");
  dump_history_enroll(hst_MeanPressure,"<Pressure>");
  dump_history_enroll(hst_MeanPlasmaBeta,"<PlasmaBeta>");

  return;
}

/* ========================================================================== */

/*
 *  Function problem
 *
 *  Set up initial conditions, allocate memory, and initialize FFT plans
 */

void problem(DomainS *pDomain)
{
  GridS *pGrid = (pDomain->Grid);
  int i, is=pGrid->is, ie = pGrid->ie;
  int j, js=pGrid->js, je = pGrid->je;
  int k, ks=pGrid->ks, ke = pGrid->ke;

  initialize(pGrid, pDomain);

  /* Initialize uniform density and momenta */
  for (k=ks-nghost; k<=ke+nghost; k++) {
    for (j=js-nghost; j<=je+nghost; j++) {
      for (i=is-nghost; i<=ie+nghost; i++) {
        pGrid->U[k][j][i].d = rhobar;
        pGrid->U[k][j][i].M1 = 0.0;
        pGrid->U[k][j][i].M2 = 0.0;
        pGrid->U[k][j][i].M3 = 0.0;
#ifndef ISOTHERMAL
        pGrid->U[k][j][i].E = p0/Gamma_1;
#endif
      }
    }
  }

#ifdef MHD
  Real localB0;
  Real x2center = 0.5*(pDomain->RootMaxX[1] - pDomain->RootMinX[1]);
  
  /* Initialize uniform magnetic field */
  for (k=ks-nghost; k<=ke+nghost; k++) {
    for (j=js-nghost; j<=je+nghost; j++) {
      for (i=is-nghost; i<=ie+nghost; i++) {
        cc_pos(pGrid,i,j,k,&x1,&x2,&x3);

        // uniform in one direction
        if (BFieldConfig == 0) {
          localB0 = B0;
        
        // uniform no net flux
        } else if (BFieldConfig == 1) {
          if (x2 < x2center)
            localB0 = B0;
          else
            localB0 = -B0;

        } else {
          ath_error("Error: unknown magnetic field configuration!");
        }

        pGrid->U[k][j][i].B1c  = localB0;
        pGrid->U[k][j][i].B2c  = 0.0;
        pGrid->U[k][j][i].B3c  = 0.0;
        pGrid->B1i[k][j][i] = localB0;
        pGrid->B2i[k][j][i] = 0.0;
        pGrid->B3i[k][j][i] = 0.0;
#ifndef ISOTHERMAL
        pGrid->U[k][j][i].E += 0.5 * localB0 * localB0;
#endif
      }
    }
  }
#endif /* MHD */

  /* Set the initial perturbations.  
   * We'll start of with 1/100 of the intended (later/stationary) input 
   *
   * */
  generate(0.01*sqrt(1./(kpeak * ForcingAmpl)));
  perturb(pGrid, 0.01*sqrt(1./(kpeak * ForcingAmpl)));

  return;
}

static Real getDVOne(const GridS *pG, const int i, const int j, const int k)
{
  return (globalAmplNorm[0] * dv1[i + nx1 * (j + nx2*k)]); 
}

static Real getDVTwo(const GridS *pG, const int i, const int j, const int k)
{
  return (globalAmplNorm[0] * dv2[i + nx1 * (j + nx2*k)]); 
}

static Real getDVThree(const GridS *pG, const int i, const int j, const int k)
{
  return (globalAmplNorm[0] * dv3[i + nx1 * (j + nx2*k)]); 
}

ConsFun_t get_usr_expr(const char *expr)
{
  if(strcmp(expr,"DV1")==0) return getDVOne;
  if(strcmp(expr,"DV2")==0) return getDVTwo;
  if(strcmp(expr,"DV3")==0) return getDVThree;
  return NULL;
}

VOutFun_t get_usr_out_fun(const char *name)
{
  return NULL;
}


/* ========================================================================== */

/*
 *  Function Userwork_in_loop
 *
 *  Drive velocity field 
 */

void Userwork_in_loop(MeshS *pM)
{
  GridS *pGrid;
  int nl,nd;

  for (nl=0; nl<(pM->NLevels); nl++){
    for (nd=0; nd<(pM->DomainsPerLevel[nl]); nd++){
      if (pM->Domain[nl][nd].Grid != NULL){

        pGrid = pM->Domain[nl][nd].Grid;

        if (isnan(pGrid->dt)) ath_error("Time step is NaN!");

        /* Drive on every time step */
        generate(pGrid->dt);
        perturb(pGrid, pGrid->dt);
        
      }
    }
  }
  return;
}

/* ========================================================================== */

void Userwork_after_loop(MeshS *pM)
{
  /* Don't free memory here if doing any analysis because final
   * output hasn't been written yet!! */
  return;
}

void problem_write_restart(MeshS *pM, FILE *fp)
{
  /* write the previous random seed to output file as
   * the generate() function is called on every restart
   * and we thus produce the the forcing field that was
   * present at the time of the dump
   */
  fwrite(&rseed, sizeof(long int),1,fp);  
  fwrite(&idum2, sizeof(long int),1,fp);  
  fwrite(&iy, sizeof(long int),1,fp);  
  fwrite(&iv, sizeof(long int),NTAB,fp);  
  fwrite(fv1Persistent, sizeof(Complex),P3DCsz[2]*P3DCsz[1]*P3DCsz[0],fp);
  fwrite(fv2Persistent, sizeof(Complex),P3DCsz[2]*P3DCsz[1]*P3DCsz[0],fp);
  fwrite(fv3Persistent, sizeof(Complex),P3DCsz[2]*P3DCsz[1]*P3DCsz[0],fp);
  return;
}

void problem_read_restart(MeshS *pM, FILE *fp)
{  
  GridS *pGrid;
  DomainS *pDomain;
  int nl, nd;
  double dirty;
  long int origrseed;
  int ixs,jxs,kxs;
  
  for (nl=0; nl<(pM->NLevels); nl++){
    for (nd=0; nd<(pM->DomainsPerLevel[nl]); nd++){
      if (pM->Domain[nl][nd].Grid != NULL){

         pGrid = pM->Domain[nl][nd].Grid;
         pDomain = &(pM->Domain[nl][nd]);  //Allocate memory and initialize everything
         initialize(pGrid, pDomain);
         
         /* get current random seed for individual process */
         fread(&rseed, sizeof(long int),1,fp);
         fread(&idum2, sizeof(long int),1,fp);
         fread(&iy, sizeof(long int),1,fp);
         fread(&iv, sizeof(long int),NTAB,fp);
         
         fread(fv1Persistent, sizeof(Complex),P3DCsz[2]*P3DCsz[1]*P3DCsz[0],fp);
         fread(fv2Persistent, sizeof(Complex),P3DCsz[2]*P3DCsz[1]*P3DCsz[0],fp);
         fread(fv3Persistent, sizeof(Complex),P3DCsz[2]*P3DCsz[1]*P3DCsz[0],fp);

      }
    }
  }
  return;
}


/* ========================================================================== */

/*
 *  Function hst_*
 *
 *  Dumps to history file
 */

/*! \fn static Real hst_dEk(const Grid *pG, const int i,const int j,const int k)
 *  \brief Dump kinetic energy in perturbations */
static Real hst_dEk(const GridS *pG, const int i, const int j, const int k)
{ /* The kinetic energy in perturbations is 0.5*d*V^2 */
  return 0.5*(pG->U[k][j][i].M1*pG->U[k][j][i].M1 +
	      pG->U[k][j][i].M2*pG->U[k][j][i].M2 +
	      pG->U[k][j][i].M3*pG->U[k][j][i].M3)/pG->U[k][j][i].d;
}

static Real hst_MeanPlasmaBeta(const GridS *pG, const int i, const int j, const int k)
{ /* plasma beta p_th/p_B */
#ifdef MHD

  Real B2 = ( 
    pG->U[k][j][i].B1c*pG->U[k][j][i].B1c + 
    pG->U[k][j][i].B2c*pG->U[k][j][i].B2c + 
    pG->U[k][j][i].B3c*pG->U[k][j][i].B3c); 

#ifdef ISOTHERMAL
  Real Pres =  Iso_csound2 * pG->U[k][j][i].d;
#else

  Real M2 = (
    pG->U[k][j][i].M1*pG->U[k][j][i].M1 +
    pG->U[k][j][i].M2*pG->U[k][j][i].M2 +
    pG->U[k][j][i].M3*pG->U[k][j][i].M3);

  Real eInt = pG->U[k][j][i].E - 0.5 * M2 / pG->U[k][j][i].d - 0.5 * B2;
  eInt = MAX(eInt,TINY_NUMBER);
  Real Pres =  Gamma_1 * eInt;

#endif /* ISOTHERMAL */

  return Pres/(0.5 * B2);

#else
  return 0.0;
#endif /* MHD */
}

static Real hst_MeanPressure(const GridS *pG, const int i, const int j, const int k)
{ /* pressure*/
#ifdef ISOTHERMAL
  return Iso_csound2 * pG->U[k][j][i].d;
#else

  Real M2 = (
    pG->U[k][j][i].M1*pG->U[k][j][i].M1 +
    pG->U[k][j][i].M2*pG->U[k][j][i].M2 +
    pG->U[k][j][i].M3*pG->U[k][j][i].M3);

  Real eInt = pG->U[k][j][i].E - 0.5 * M2 / pG->U[k][j][i].d;
#ifdef MHD
  eInt -= 0.5 * (
    pG->U[k][j][i].B1c*pG->U[k][j][i].B1c + 
    pG->U[k][j][i].B2c*pG->U[k][j][i].B2c + 
    pG->U[k][j][i].B3c*pG->U[k][j][i].B3c); 
#endif

  eInt = MAX(eInt,TINY_NUMBER);
  return Gamma_1 * eInt;

#endif /* ISOTHERMAL */
}

static Real hst_MeanMach(const GridS *pG, const int i, const int j, const int k)
{ /* Sonic Mach number*/
#ifdef ISOTHERMAL
  return sqrt((pG->U[k][j][i].M1*pG->U[k][j][i].M1 +
	      pG->U[k][j][i].M2*pG->U[k][j][i].M2 +
	      pG->U[k][j][i].M3*pG->U[k][j][i].M3)/Iso_csound2)/pG->U[k][j][i].d;
#else

  Real M2 = (
    pG->U[k][j][i].M1*pG->U[k][j][i].M1 +
    pG->U[k][j][i].M2*pG->U[k][j][i].M2 +
    pG->U[k][j][i].M3*pG->U[k][j][i].M3);

  Real eInt = pG->U[k][j][i].E - 0.5 * M2 / pG->U[k][j][i].d;
#ifdef MHD
  eInt -= 0.5 * (
    pG->U[k][j][i].B1c*pG->U[k][j][i].B1c + 
    pG->U[k][j][i].B2c*pG->U[k][j][i].B2c + 
    pG->U[k][j][i].B3c*pG->U[k][j][i].B3c); 
#endif

  eInt = MAX(eInt,TINY_NUMBER);
  Real cs = sqrt(Gamma * Gamma_1 * eInt / pG->U[k][j][i].d);

  return sqrt(M2)/pG->U[k][j][i].d/cs;
#endif /* ISOTHERMAL */
}

static Real hst_MeanAlfvenicMach(const GridS *pG, const int i, const int j, const int k)
{ /* Alfvenic Mach number */
#ifdef MHD
  return sqrt((pG->U[k][j][i].M1*pG->U[k][j][i].M1 +
	      pG->U[k][j][i].M2*pG->U[k][j][i].M2 +
	      pG->U[k][j][i].M3*pG->U[k][j][i].M3)/pG->U[k][j][i].d/(
        pG->U[k][j][i].B1c*pG->U[k][j][i].B1c +
        pG->U[k][j][i].B2c*pG->U[k][j][i].B2c +
        pG->U[k][j][i].B3c*pG->U[k][j][i].B3c));
#else /* MHD */
  return 0.0;
#endif /* MHD */
}

/*! \fn static Real hst_dEb(const Grid *pG, const int i,const int j,const int k)
 *  \brief Dump magnetic energy in perturbations */
static Real hst_dEb(const GridS *pG, const int i, const int j, const int k)
{ /* The magnetic energy in perturbations is 0.5*B^2 - 0.5*B0^2 */
#ifdef MHD
  return 0.5*((pG->U[k][j][i].B1c*pG->U[k][j][i].B1c +
	       pG->U[k][j][i].B2c*pG->U[k][j][i].B2c +
	       pG->U[k][j][i].B3c*pG->U[k][j][i].B3c)-B0*B0);
#else /* MHD */
  return 0.0;
#endif /* MHD */
}

/* ========================================================================== */

#undef NTAB
