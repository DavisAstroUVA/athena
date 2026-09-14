//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mc_disksims.cpp
//  \brief Problem generator for monte carlo disk simulation
//
//========================================================================================

#include <iostream> // temporary for testing
#include <string.h>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../monte_carlo/montecarlo.hpp"
#include "../monte_carlo/photon.hpp"
#include "../monte_carlo/photonpusher.hpp"

#if !MONTE_CARLO_ENABLED
#error "This problem requires monte carlo"
#endif

/* This stores the domain information of each vtk file */
typedef struct Domain_s{
  char *fname;
  char *comment;
  FILE *fp;
  int Nx, Ny, Nz;    /* Grid dimensions */
  double ox, oy, oz; /* Origin of this particular domain */
  double dx, dy, dz; /* grid cell size */
  float *X, *Y, *Z; /* X,Y,Z coordinates*/
}VTK_Domain;


static VTK_Domain *domain_1d;
static int grid_version; /* fixed or flexible grids */
static int nrmax, nrmin;
static void strip_trail_white(char *pc);
static inline void Swap4Bytes2(void *vdat);
static void init_domain_1d(void);
int IsBigEndian2(void);

static bool tnorm;
static Real logemin, logemax;

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief monte carlo test problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  Real rideal = 8.314e7;
  Real gamma = peos->GetGamma();

  int nx1 = pin->GetInteger("mesh","nx1");
  nrmax = pin->GetOrAddInteger("problem","nrmax",nx1);
  nrmin = pin->GetOrAddInteger("problem","nrmin",0);

  // Read in vtk file
  FILE *fp; /* A temporary copy to make the code cleaner */
  int i,j, ndat, cell_dat;
  char line[256];
  int nx,ny,nz,nread;
  float fcoord;
  int big_end = IsBigEndian2(); // =1 on big endian machine

  printf("Is it big Endian? %d\n", big_end);
  VTK_Domain domain;
  int grid_version;
  if((fp = fopen("grid.vtk","r")) == NULL)
    return;

  /* get header */
  fgets(line,256,fp);
  strip_trail_white(line);
  //if(strcmp(line,"# vtk DataFile Version 3.0") != 0 /* mymhd  */ &&
  //   strcmp(line,"# vtk DataFile Version 2.0") != 0 /* athena */ )

  /* get comment field */
  fgets(line,256,fp);
  strip_trail_white(line);
  printf("Comment Field: \"%s\"\n",line);


  /* get BINARY or ASCII */
  fgets(line,256,fp);
  strip_trail_white(line);
  //if(strcmp(line,"BINARY") != 0)
  //  join_error("Unsupported file type: %s",line);

  /* get DATASET STRUCTURED_POINTS */
  fgets(line,256,fp);
  strip_trail_white(line);
  if(strcmp(line,"DATASET STRUCTURED_POINTS") == 0)
    grid_version=1;
  else if
    (strcmp(line,"DATASET RECTILINEAR_GRID") == 0) grid_version=2;

  /* I'm assuming from this point on that the header is in good shape */

  /* Dimensions */
  fscanf(fp,"DIMENSIONS %d %d %d\n",
         &(domain.Nx), &(domain.Ny), &(domain.Nz));
  printf("DIMENSIONS %d %d %d\n",
         domain.Nx, domain.Ny, domain.Nz);

  /* We want to store the number of grid cells, not the number of grid
     cell corners */
  if(domain.Nx > 1) domain.Nx--;
  if(domain.Ny > 1) domain.Ny--;
  if(domain.Nz > 1) domain.Nz--;

  if(grid_version == 1){ // fixed linear grids in athena 4.2

    /* Origin */
    fscanf(fp,"ORIGIN %le %le %le\n",
           &(domain.ox), &(domain.oy), &(domain.oz));
    printf("ORIGIN %e %e %e\n",
           domain.ox, domain.oy, domain.oz);

    /* Spacing, dx, dy, dz */
    fscanf(fp,"SPACING %le %le %le\n",&(domain.dx), &(domain.dy), &(domain.dz));
    printf("SPACING %e %e %e\n",domain.dx, domain.dy, domain.dz);
  } else if(grid_version == 2) { // flexible grids in athena++
    /* X_COORDINATES NX float */
    fscanf(fp,"X_COORDINATES %d float\n", &nx);
    printf("X_COORDINATES %d float\n", nx);
    domain.X = (float*)malloc(nx*sizeof(float));
    for(j=0; j<nx; ++j){
      fread(&fcoord, sizeof(float), 1, fp);
      if(!big_end) Swap4Bytes2(&fcoord);
      domain.X[j] = fcoord;
    }

    /* Y_COORDINATES NY float */
    fscanf(fp,"\nY_COORDINATES %d float\n", &ny);
    printf("Y_COORDINATES %d float\n", ny);
    domain.Y = (float*)malloc(ny*sizeof(float));
    for(j=0; j<ny; ++j){
      fread(&fcoord, sizeof(float), 1, fp);
      if(!big_end) Swap4Bytes2(&fcoord);
      domain.Y[j] = fcoord;
    }

    /* Z_COORDINATES NZ float */
    fscanf(fp,"\nZ_COORDINATES %d float\n", &nz);
    printf("Z_COORDINATES %d float\n", nz);
    domain.Z = (float*)malloc(nz*sizeof(float));
    for(j=0; j<nz; ++j){
      fread(&fcoord, sizeof(float), 1, fp);
      if(!big_end) Swap4Bytes2(&fcoord);
      domain.Z[j] = fcoord;
    }

    domain.ox = domain.X[0];
    domain.oy = domain.Y[0];
    domain.oz = domain.Z[0];
    if(nx>1) domain.dx = 1.0; // just to make it nonzero
    else     domain.dx = 0.0;
    if(ny>1) domain.dy = 1.0;
    else     domain.dy = 0.0;
    if(nz>1) domain.dz = 1.0;
    else     domain.dz = 0.0;
  }

  /* Cell Data = Nx*Ny*Nz */
  fscanf(fp,"\nCELL_DATA %d\n",&cell_dat);
  printf("CELL_DATA %d\n",cell_dat);
  ndat = (domain.Nx)*(domain.Ny)*(domain.Nz);
  char ltype[128],type[128], variable[128], format[128];
  fscanf(fp,"%s %s %s\n",type, variable, format);
  if(strcmp(type, "SCALARS") == 0){
    /* Read in the LOOKUP_TABLE (only default supported for now) */
    fscanf(fp,"%s %s\n", ltype, format);
    if(strcmp(ltype, "LOOKUP_TABLE") != 0 ||
       strcmp(format, "default") != 0 ){
      fprintf(stderr,"Expected \"LOOKUP_TABLE default\"\n");
      return;
    }
  }
  float fdat;
  Real c = 2.99792458e10;
  //Real c = 0.;
  printf("\n%s %s %s\n",type,variable,format);
  if(strcmp(type, "SCALARS") == 0){
    for(int k=0; k<domain.Nz; k++){
      for(int j=0; j<domain.Ny; j++){
        for(int i=0; i<domain.Nx; i++){
          fread(&fdat, sizeof(float), 1, fp);
          if(!big_end) Swap4Bytes2(&fdat);
          phydro->u(IDN,k+ks,j+js,i+is) = static_cast<Real>(fdat);
        }}}
  }
  fscanf(fp,"%s %s %s\n",type, variable, format);
  if(strcmp(type, "SCALARS") == 0){
    /* Read in the LOOKUP_TABLE (only default supported for now) */
    fscanf(fp,"%s %s\n", ltype, format);
    if(strcmp(ltype, "LOOKUP_TABLE") != 0 ||
       strcmp(format, "default") != 0 ){
      fprintf(stderr,"Expected \"LOOKUP_TABLE default\"\n");
      return;
    }
  }
  printf("\n%s %s %s\n",type,variable,format);
  if(strcmp(type, "SCALARS") == 0){
    for(int k=0; k<domain.Nz; k++){
      for(int j=0; j<domain.Ny; j++){
        for(int i=0; i<domain.Nx; i++){
          fread(&fdat, sizeof(float), 1, fp);
          if(!big_end) Swap4Bytes2(&fdat);
          phydro->u(IM1,k+ks,j+js,i+is) = phydro->u(IDN,k+ks,j+js,i+is) *c *
            static_cast<Real>(fdat);
        }
      }
    }
  }
  fscanf(fp,"%s %s %s\n",type, variable, format);
  if(strcmp(type, "SCALARS") == 0){
    /* Read in the LOOKUP_TABLE (only default supported for now) */
    fscanf(fp,"%s %s\n", ltype, format);
    if(strcmp(ltype, "LOOKUP_TABLE") != 0 ||
       strcmp(format, "default") != 0 ){
      fprintf(stderr,"Expected \"LOOKUP_TABLE default\"\n");
      return;
    }
  }
  printf("\n%s %s %s\n",type,variable,format);
  if(strcmp(type, "SCALARS") == 0){
    for(int k=0; k<domain.Nz; k++){
      for(int j=0; j<domain.Ny; j++){
        for(int i=0; i<domain.Nx; i++){
          fread(&fdat, sizeof(float), 1, fp);
          if(!big_end) Swap4Bytes2(&fdat);
          phydro->u(IM2,k+ks,j+js,i+is) = phydro->u(IDN,k+is,j+js,i+is) * c
            * static_cast<Real>(fdat);
        }
      }
    }
  }
  fscanf(fp,"%s %s %s\n",type, variable, format);
  if(strcmp(type, "SCALARS") == 0){
    /* Read in the LOOKUP_TABLE (only default supported for now) */
    fscanf(fp,"%s %s\n", ltype, format);
    if(strcmp(ltype, "LOOKUP_TABLE") != 0 ||
       strcmp(format, "default") != 0 ){
      fprintf(stderr,"Expected \"LOOKUP_TABLE default\"\n");
      return;
    }
  }
  printf("\n%s %s %s\n",type,variable,format);
  if(strcmp(type, "SCALARS") == 0){
    for(int k=0; k<domain.Nz; k++){
      for(int j=0; j<domain.Ny; j++){
        for(int i=0; i<domain.Nx; i++){
          fread(&fdat, sizeof(float), 1, fp);
          if(!big_end) Swap4Bytes2(&fdat);
          phydro->u(IM3,k+ks,j+js,i+is) = phydro->u(IDN,k+ks,j+js,i+is) * c
            * static_cast<Real>(fdat);
        }
      }
    }
  }
    fscanf(fp,"%s %s %s\n",type, variable, format);
  if(strcmp(type, "SCALARS") == 0){
    /* Read in the LOOKUP_TABLE (only default supported for now) */
    fscanf(fp,"%s %s\n", ltype, format);
    if(strcmp(ltype, "LOOKUP_TABLE") != 0 ||
       strcmp(format, "default") != 0 ){
      fprintf(stderr,"Expected \"LOOKUP_TABLE default\"\n");
      return;
    }
  }
  printf("\n%s %s %s\n",type,variable,format);
  if(strcmp(type, "SCALARS") == 0){
    for(int k=0; k<domain.Nz; k++){
      for(int j=0; j<domain.Ny; j++){
        for(int i=0; i<domain.Nx; i++){
          fread(&fdat, sizeof(float), 1, fp);
        }}}
  }
  fscanf(fp,"%s %s %s\n",type, variable, format);
  if(strcmp(type, "SCALARS") == 0){
    /* Read in the LOOKUP_TABLE (only default supported for now) */
    fscanf(fp,"%s %s\n", ltype, format);
    if(strcmp(ltype, "LOOKUP_TABLE") != 0 ||
       strcmp(format, "default") != 0 ){
      fprintf(stderr,"Expected \"LOOKUP_TABLE default\"\n");
      return;
    }
  }
  printf("\n%s %s %s\n",type,variable,format);
  if(strcmp(type, "SCALARS") == 0){
    for(int k=0; k<domain.Nz; k++){
      for(int j=0; j<domain.Ny; j++){
        for(int i=0; i<domain.Nx; i++){
          fread(&fdat, sizeof(float), 1, fp);
          if(!big_end) Swap4Bytes2(&fdat);
          Real tgas = static_cast<Real>(fdat)/phydro->u(IDN,k+ks,j+js,i+is);
          //printf("%d %d %d %g %g\n",k,j,i,tgas,phydro->u(IDN,k+ks,j+js,i+is));
          if (tgas < 1.0e4)
            tgas = 1.0e4;
          Real press = tgas * phydro->u(IDN,k+ks,j+js,i+is);
          phydro->u(IEN,k+ks,j+js,i+is) = 8.314e7/(gamma-1.)*press;
          //if ((j==10)&&(k==10))
          //  std::cout << phydro->u(IEN,k+ks,j+js,i+is) << " " << fdat << std::endl;
        }}}
  }
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        phydro->u(IEN,k,j,i) += 0.5*SQR(phydro->u(IM1,k,j,i))/phydro->u(IDN,k,j,i);
        phydro->u(IEN,k,j,i) += 0.5*SQR(phydro->u(IM2,k,j,i))/phydro->u(IDN,k,j,i);
        phydro->u(IEN,k,j,i) += 0.5*SQR(phydro->u(IM3,k,j,i))/phydro->u(IDN,k,j,i);
      }}}


}

//========================================================================================
//! \fn void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin)
//! \brief Analogous to problem generator but used in support of InitializePhoton
//========================================================================================

void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin) {

  int nx1 = pin->GetInteger("mesh","nx1");
  nrmax = pin->GetOrAddInteger("problem","nrmax",nx1);
  nrmin = pin->GetOrAddInteger("problem","nrmin",0);
  tnorm = pin->GetOrAddBoolean("problem","tnorm",false);
  if (tnorm) {
    // interpret as xmin/xmax with x=E/(kb*T)
    Real kb = 1.380649e-16;
    logemin = log(kb*pin->GetReal("problem", "emin"));
    logemax = log(kb*pin->GetReal("problem", "emax"));
  } else {
    // Set the energy boundaries for free-free emission
    Real everg = 1.6021772e-12;
    logemin = log(everg*pin->GetReal("problem", "emin"));
    logemax = log(everg*pin->GetReal("problem", "emax"));
  }

}

//========================================================================================
//! \fn void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype)
//! \brief Initializes Photon packets before integration
//========================================================================================

void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype) {

  // Set initial cells and emission weights for all photon samples
  SetEmissionCellWeight(pphot,ips,ipe);

  for (int ip=ips; ip<=ipe; ip++) {

    // Set status flag
    pphot->statp[ip] = EVOLVING;
    pphot->dtp[ip] = pphot->pmy_mcb->pmy_mc->tmax;

    // Obtain initial position within zone
    GetZonePosition(pphot,pran,pcoord,ip);


    // Obtain intitial energy, polarization, direction and weight
    // Utilize free-free emission function in emission.cpp
    if(tnorm) {
      Real logtg = log(tgas(pphot->i3p[ip],pphot->i2p[ip],pphot->i1p[ip]));
      PhotonEmitFreeFree(this,pphot,logemin+logtg,logemax+logtg,ip);
    } else{
      PhotonEmitFreeFree(this,pphot,logemin,logemax,ip);
    }

    // initialize scattering number
    pphot->nscp[ip] = 0;

    // Initialize the absorption and scattering extinction coefficients
    // to the values appropriate in the emitted zone
    pphot->acp[ip] = AbsorptionOpacity(this,pphot,ip);
    pphot->scp[ip] = ScatteringOpacity(this,pphot,ip);

  } // loop over ip

}


/* ========================================================================== */


int IsBigEndian2(void)
{
  short int n = 1;
  char *ep = (char *)&n;
  return (*ep == 0); // Returns 1 on a big endian machine
  }

static inline void Swap4Bytes2(void *vdat) {
  char tmp, *dat = (char *) vdat;
  tmp = dat[0];  dat[0] = dat[3];  dat[3] = tmp;
  tmp = dat[1];  dat[1] = dat[2];  dat[2] = tmp;
}

/* Input character pointer is to the start of a NUL terminated string */
static void strip_trail_white(char *pc){
  char *cp = pc;

  /* iterate down to the NUL terminator */
  while(*cp != '\0') cp++;

  while(cp > pc){
    cp--;
    if(isspace(*cp)) *cp = '\0';
    else break;
  }

  return;
}
