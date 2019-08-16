
/***********************************************************************
/
/  INITIALIZE HYDROSTATIC GALACTIC DISKS
/
/  written by: Kai Rodenbeck (based on code by Greg Bryan)
/  date:       2015
/  modified1:  Wolfram Schmidt, 2018
/
/  PURPOSE:
/    Set up a number of spheres with disk galaxies and magnetic fields
/
/  RETURNS: SUCCESS or FAIL
/
************************************************************************/

#include "preincludes.h"
#include <stdlib.h>
#include "macros_and_parameters.h"
#include "typedefs.h"
#include "global_data.h"
#include "phys_constants.h"
#include "Fluxes.h"
#include "GridList.h"
#include "ExternalBoundary.h"
#include "Grid.h"
#include "CosmologyParameters.h"
#include "HydrostaticDisk.h"


#define NTHETA 1000
#define NR 1000

/********************* PROTOTYPES *********************/

int GetUnits(float *DensityUnits, float *LengthUnits,
	     float *TemperatureUnits, float *TimeUnits,
	     float *VelocityUnits, float Time);
int CosmologyComputeExpansionFactor(float time, float *a, float *dadt);

float gasdev();

double BESSI0(double y);
double BESSI1(double y);
double BESSK0(double y);
double BESSK1(double y);

/* Set various units. */

// Used to compute Bonner-Ebert density profile
//double BE(double r);
//double q(double r);
//double Ang(double a1, double a2, double R, double r);

// Returns random velocity from Maxwellian distribution
double Maxwellian(double c_tilda, double vel_unit, double mu, double gamma);
//double ERF(double x);

//int ComputeRadialVelocity(float density, double mass, float r_init, 
//			  double VelocityUnits, double LengthUnits,
//			  float dm_velrot_corr, double radius_vr[], double Vr[], 
//			  double exterior_rho[], int Npts);


/*******************************************************/

static int CollapseTestParticleCount = 0;

static float CosmologySimulationOmegaBaryonNow       = 0.0463;
static float CosmologySimulationInitialFractionHII   = 1.2e-5;
static float CosmologySimulationInitialFractionHeII  = 1.0e-14;
static float CosmologySimulationInitialFractionHeIII = 1.0e-17;
static float CosmologySimulationInitialFractionHM    = 2.0e-9;
static float CosmologySimulationInitialFractionH2I   = 2.0e-20;
static float CosmologySimulationInitialFractionH2II  = 3.0e-14;

int grid::MHDGalaxyDiskInitializeGrid(  int NumberOfSpheres,
					float SphereRadius[MAX_SPHERES][MAX_DIMENSION],
					float SphereAngularMomentum[MAX_SPHERES][MAX_DIMENSION],
					float SphereCoreRadius[MAX_SPHERES][MAX_DIMENSION],
					float SphereDensity[MAX_SPHERES],
					float SphereTemperature[MAX_SPHERES],
					float SphereMetallicity[MAX_SPHERES],
					float SpherePosition[MAX_SPHERES][MAX_DIMENSION],
					float SphereVelocity[MAX_SPHERES][MAX_DIMENSION],
					float SphereFracKeplerianRot[MAX_SPHERES],
					float SphereTurbulence[MAX_SPHERES],
					float SphereDispersion[MAX_SPHERES],
					float SphereCutOff[MAX_SPHERES],
					float SphereAng1[MAX_SPHERES],
					float SphereAng2[MAX_SPHERES],
					int   SphereNumShells[MAX_SPHERES],
					int   SphereType[MAX_SPHERES],
					int   SphereConstantPressure[MAX_SPHERES],
					int   SphereSmoothSurface[MAX_SPHERES],
					float SphereSmoothRadius[MAX_SPHERES],
					float SphereMagnFactor[MAX_SPHERES],
					int   SphereMagnEquipart[MAX_SPHERES],
					float HaloMass[MAX_SPHERES],
					float HaloCoreRadius[MAX_SPHERES],
					float HaloRadius[MAX_SPHERES],
					int   SphereUseParticles,
					float ParticleMeanDensity,
					float UniformVelocity[MAX_DIMENSION],
					int   SphereUseColour,
					int   SphereUseMetals,
					float InitialTemperature,
					float InitialDensity,
					float InitialMagnField,
					int level)
{
	/* declarations */
	
	int dim, i, j, k, m, field, sphere, size;
	int DeNum, HINum, HIINum, HeINum, HeIINum, HeIIINum, HMNum, H2INum, H2IINum,
		DINum, DIINum, HDINum, MetalNum;
	float xdist,ydist,zdist;
	
	/* create fields */
	
	NumberOfBaryonFields = 0;
	FieldType[NumberOfBaryonFields++] = Density;
	FieldType[NumberOfBaryonFields++] = TotalEnergy;
	if (DualEnergyFormalism)
		FieldType[NumberOfBaryonFields++] = InternalEnergy;
	int ivel = NumberOfBaryonFields;
	FieldType[NumberOfBaryonFields++] = Velocity1;
	if (GridRank > 1) 
		FieldType[NumberOfBaryonFields++] = Velocity2;
	if (GridRank > 2)
		FieldType[NumberOfBaryonFields++] = Velocity3;
	int imag = NumberOfBaryonFields;
	if (HydroMethod == MHD_RK)
	{
		FieldType[NumberOfBaryonFields++] = Bfield1;
		if (GridRank > 1)
			FieldType[NumberOfBaryonFields++] = Bfield2;
		if (GridRank > 2)
			FieldType[NumberOfBaryonFields++] = Bfield3;
		FieldType[NumberOfBaryonFields++] = PhiField;
	}
	if (MultiSpecies)
	{
		FieldType[DeNum    = NumberOfBaryonFields++] = ElectronDensity;
		FieldType[HINum    = NumberOfBaryonFields++] = HIDensity;
		FieldType[HIINum   = NumberOfBaryonFields++] = HIIDensity;
		FieldType[HeINum   = NumberOfBaryonFields++] = HeIDensity;
		FieldType[HeIINum  = NumberOfBaryonFields++] = HeIIDensity;
		FieldType[HeIIINum = NumberOfBaryonFields++] = HeIIIDensity;
		if (MultiSpecies > 1)
		{
			FieldType[HMNum    = NumberOfBaryonFields++] = HMDensity;
			FieldType[H2INum   = NumberOfBaryonFields++] = H2IDensity;
			FieldType[H2IINum  = NumberOfBaryonFields++] = H2IIDensity;
		}
		if (MultiSpecies > 2)
		{
			FieldType[DINum   = NumberOfBaryonFields++] = DIDensity;
			FieldType[DIINum  = NumberOfBaryonFields++] = DIIDensity;
			FieldType[HDINum  = NumberOfBaryonFields++] = HDIDensity;
		}
	}
	if (SphereUseMetals)
		FieldType[MetalNum = NumberOfBaryonFields++] = SNColour;
	
	int ColourNum = NumberOfBaryonFields;
	if (SphereUseColour)
		FieldType[NumberOfBaryonFields++] = Metallicity; /* fake it with metals */
	
	/* Return if this doesn't concern us. */
	
	if (ProcessorNumber != MyProcessorNumber)
	{
		NumberOfParticles = (SphereUseParticles > 0) ? 1 : 0;
		for (int dim = 0; dim < GridRank; dim++)
			NumberOfParticles *= (GridEndIndex[dim] - GridStartIndex[dim] + 1);
		return SUCCESS;
	}
	
	float	DensityUnits,
			LengthUnits,
			TemperatureUnits,
			TimeUnits,
			VelocityUnits,
			CriticalDensity = 1,
			BoxLength = 1,
			mu = 0.6;
	
	double MyGrav=GravConst;
	
	float a, dadt, ExpansionFactor = 1;
	GetUnits(	&DensityUnits,
				&LengthUnits,
				&TemperatureUnits,
				&TimeUnits,
				&VelocityUnits,
				Time);
	if (ComovingCoordinates)
	{
		CosmologyComputeExpansionFactor(Time, &a, &dadt);
		ExpansionFactor = a/(1.0+InitialRedshift);
		CriticalDensity = 2.78e11*pow(HubbleConstantNow, 2); // in Msolar/Mpc^3
		BoxLength = ComovingBoxSize*ExpansionFactor/HubbleConstantNow;  // in Mpc
	}
	else
	{
		CriticalDensity = 2.78e11*pow(0.74,2); // in Msolar/Mpc^3 for h=0.74
		BoxLength = LengthUnits / 3.086e24;
		HubbleConstantNow = 1.0;
		OmegaMatterNow = 1.0;
		MyGrav=4*pi*GravConst*DensityUnits*TimeUnits*TimeUnits;
	}
	
	double MagnConversionFactor=sqrt(4.0*3.14159/(VelocityUnits*VelocityUnits)/DensityUnits);
	
	/* Loop over the set-up twice, once to count the particles, the second time to initialize them. */
	
	int SetupLoopCount, npart = 0;
	for (SetupLoopCount = 0; SetupLoopCount < 1+min(SphereUseParticles, 1);SetupLoopCount++)
	{
		/* Set densities */
		
		float BaryonMeanDensity, ParticleCount = 0;
		switch (SphereUseParticles)
		{
			case 1:
				BaryonMeanDensity = CosmologySimulationOmegaBaryonNow / OmegaMatterNow;
				break;
			case 2:
				BaryonMeanDensity = 1 - CosmologySimulationOmegaBaryonNow / OmegaMatterNow;
				break;
			default:
				BaryonMeanDensity = 1.0;
	} // ENDSWITCH SphereUseParticles
	
	
	BaryonMeanDensity = 1.0;
	
	if (ParticleMeanDensity == FLOAT_UNDEFINED)
		ParticleMeanDensity = 1.0 - BaryonMeanDensity;
	else
		BaryonMeanDensity = 1.0 - ParticleMeanDensity;
	
	/* Set particles. */
	
	if (SphereUseParticles > 0 && SetupLoopCount > 0)
	{
		/* If particles already exist (coarse particles), then delete. */
		if (NumberOfParticles > 0)
			this->DeleteParticles();
		
		/* Use count from previous loop to set particle number. */
		NumberOfParticles = npart;
		npart = 0;
		
		/* Allocate space. */
		this->AllocateNewParticles(NumberOfParticles);
		
		/* Particle values will be set below. */
	} // end: particle initialization
	
	/* Set up the baryon field. */
	
	/* compute size of fields */
	
	size = 1;
	for (int dim = 0; dim < GridRank; dim++)
		size *= GridDimension[dim];
	
	/* allocate fields */
	
	if (SetupLoopCount == 0)
		for (field = 0; field < NumberOfBaryonFields; field++)
			if (BaryonField[field] == NULL)
				BaryonField[field] = new float[size];
	
	/* Loop over the mesh. */
	
	float density, dens1, old_density, Velocity[MAX_DIMENSION], MagnField[MAX_DIMENSION],
		DiskVelocity[MAX_DIMENSION], temperature, temp1, sigma, sigma1, 
		colour, weight, a, DMVelocity[MAX_DIMENSION], metallicity, 
		outer_radius;
	float r, rcyl, x, y = 0, z = 0;
	int n = 0, ibin;
	
	double SphereTransformMatrix[MAX_SPHERES][MAX_DIMENSION][MAX_DIMENSION];
	
	double SphereRotationalPeriod[MAX_SPHERES];
	double VelocityKep = 0;
	double RotVelocity[MAX_DIMENSION];
	double RealSphereDensity[MAX_SPHERES];
	float  DMRotVelocityCorrection = 1, GasRotVelocityCorrection = 1;
	double VelocitySound[MAX_SPHERES];
	double h_0[MAX_SPHERES];
	double SphereMass, SphereCoreMass, SphereCoreDens;
	double alpha, beta, theta;
	double SphereCritMass;
	double Scale_Factor[MAX_SPHERES];
	double term1, term2;
	double radius_vr[NR], vr[NR], exterior_rho[NR], radial_velocity;
	float sin_deltaDisk[MAX_SPHERES];
	double SchwarzschildRadius, CavityRadius, InnerDensity, InnerTemperature,
		ThickenTransitionRadius, BHMass, ScaleHeight, InnerScaleHeight;
	double MidplaneDensity, MidplaneTemperature;
	
	double Halo_disp[MAX_SPHERES];
	double Halo_rho[MAX_SPHERES];
	
	hydrostatic_disk Galaxy[MAX_SPHERES];
	
	double DM_rho=0.0;
	double DM_vel[MAX_DIMENSION];
	for(int dim=0;dim<GridRank;dim++)
		DM_vel[dim]=0.0;
	double DM_sigma=1.0;
	
	/* Pre-compute cloud properties before looping over mesh */
	for (sphere = 0; sphere < NumberOfSpheres; sphere++)
	{
		
		Halo_rho[sphere]=(HaloMass[sphere]*SolarMass/(2.0*pi*pow(HaloCoreRadius[sphere]*LengthUnits,3.0)))/DensityUnits;
		Halo_disp[sphere]=sqrt(GravConst*HaloMass[sphere]*SolarMass/(12.0*HaloCoreRadius[sphere]*LengthUnits))/VelocityUnits;
		
		if(MAX_DIMENSION==3 && !(SphereAngularMomentum[sphere][2]==0.0 && SphereAngularMomentum[sphere][1]==0))
		{
			double Ang_Vel=sqrt(SphereAngularMomentum[sphere][0]*SphereAngularMomentum[sphere][0]
							+	SphereAngularMomentum[sphere][1]*SphereAngularMomentum[sphere][1]
							+	SphereAngularMomentum[sphere][2]*SphereAngularMomentum[sphere][2]);
			double z_sl[3],x_sl[3];
			
			z_sl[0]=SphereAngularMomentum[sphere][0]/Ang_Vel;
			z_sl[1]=SphereAngularMomentum[sphere][1]/Ang_Vel;
			z_sl[2]=SphereAngularMomentum[sphere][2]/Ang_Vel;
			
			for(int it=0;it<MAX_DIMENSION;it++)
				SphereAngularMomentum[sphere][it]/=Ang_Vel;
			
			
			x_sl[0]=1.0-z_sl[0];
			x_sl[1]=-z_sl[0];
			x_sl[2]=-z_sl[1];
			
			double temp_norm=sqrt(x_sl[0]*x_sl[0]+x_sl[1]*x_sl[1]+x_sl[2]*x_sl[2]);
			x_sl[0]/=temp_norm;
			x_sl[1]/=temp_norm;
			x_sl[2]/=temp_norm;
			
			SphereTransformMatrix[sphere][0][0]=x_sl[0];
			SphereTransformMatrix[sphere][2][0]=z_sl[0];
			SphereTransformMatrix[sphere][1][0]=z_sl[1]*x_sl[2]-z_sl[2]*x_sl[1];
			
			SphereTransformMatrix[sphere][0][1]=x_sl[1];
			SphereTransformMatrix[sphere][2][1]=z_sl[1];
			SphereTransformMatrix[sphere][1][1]=z_sl[2]*x_sl[0]-z_sl[0]*x_sl[2];
			
			SphereTransformMatrix[sphere][0][2]=x_sl[2];
			SphereTransformMatrix[sphere][2][2]=z_sl[2];
			SphereTransformMatrix[sphere][1][2]=z_sl[0]*x_sl[1]-z_sl[1]*x_sl[0];
		}
		
		Scale_Factor[sphere] = SphereCutOff[sphere] / SphereRadius[sphere][0];
		sin_deltaDisk[sphere] = sin(pi * SphereCutOff[sphere] / 180.0);
		
		// Calculate speed of sound for this sphere
		VelocitySound[sphere] = sqrt((SphereTemperature[sphere] * Gamma * kboltz) /
									(mu * mh)) / VelocityUnits;
		
		if((MyProcessorNumber == ROOT_PROCESSOR))
			printf("\nVelocitySound (cm s^-1): %"GSYM"\n", VelocitySound[sphere] * VelocityUnits);
		
		if(SphereType[sphere]==3)
		{
			RealSphereDensity[sphere]=MyGrav/DensityUnits/pow(TimeUnits,2)/8.0/pow(2.0*pi*VelocitySound[sphere]*VelocityUnits,2)
								*pow(SphereDensity[sphere]*SolarMass
																/(SphereCoreRadius[sphere][0]*LengthUnits)
																/(SphereCoreRadius[sphere][1]*LengthUnits),2)
																/DensityUnits;
			h_0[sphere]=4.0*pow(VelocitySound[sphere]*VelocityUnits,2)	/(MyGrav/DensityUnits/pow(TimeUnits,2))*2.0*pi
																	/(
																		SphereDensity[sphere]*SolarMass
																		/(SphereCoreRadius[sphere][0]*LengthUnits)
																		/(SphereCoreRadius[sphere][1]*LengthUnits)
																	)/LengthUnits;
			//h_0[sphere]=VelocitySound[sphere]/sqrt(pi*MyGrav/2.0*RealSphereDensity[sphere]);
			if((MyProcessorNumber == ROOT_PROCESSOR))
				printf( "Natural scale height (kpc) =  %.2e\n",h_0[sphere]*LengthUnits/Mpc_cm*1.0e3);
		}
		else
		{
			RealSphereDensity[sphere]=SphereDensity[sphere];
		}
		
		
		if(SphereType[sphere]==3)
		{
			VelocityKep=sqrt(
					(MyGrav/DensityUnits/pow(TimeUnits,2))*(SphereDensity[sphere]*SolarMass)
					/(2.0*pi*(sqrt(SphereCoreRadius[sphere][0]*SphereCoreRadius[sphere][1])*LengthUnits))
					)/VelocityUnits;
			// VelocityKep=sqrt(0.5*MyGrav*RealSphereDensity[sphere]*SphereCoreRadius[sphere][1]*SphereCoreRadius[sphere][2])
			// /((double)TimeUnits/LengthUnits*VelocityUnits);
			
			if((MyProcessorNumber == ROOT_PROCESSOR))
				printf(	"\nGalaxy Mass (M_sun) = %.2e\n",	SphereDensity[sphere]);
		}
		else
		{
			VelocityKep = sqrt(MyGrav*RealSphereDensity[sphere]
						* SphereCoreRadius[sphere][1] * SphereCoreRadius[sphere][2] )
						/ ((double)TimeUnits/LengthUnits * VelocityUnits);
		}
		
		if((MyProcessorNumber == ROOT_PROCESSOR))
			switch(SphereConstantPressure[sphere])
			{
				case -1:
					printf("\nUsing adiabatic temperature profile.\n");
					break;
				case 0:
					printf("\nUsing constant temperature.\n");
					break;
				case 1:
					printf("\nUsing consant pressure temperature profile.\n");
					break;
			}
		
		if (SphereFracKeplerianRot[sphere] > 0)
		{
			SphereRotationalPeriod[sphere] = 1.0 / (SphereFracKeplerianRot[sphere]*VelocityKep);
			
			if(MyProcessorNumber == ROOT_PROCESSOR)
				printf("SphereFracKeplerianRot[%"ISYM"] = %"GSYM"\n",sphere,SphereFracKeplerianRot[sphere]);
		}
		else
		{
			SphereRotationalPeriod[sphere] = 0.0;
		}
		if(MyProcessorNumber == ROOT_PROCESSOR)
			printf("\nKepler Velocity (code units): %lf\n",VelocityKep);

		if((MyProcessorNumber == ROOT_PROCESSOR) && SphereFracKeplerianRot[sphere]>0)
		{
			printf("\nRotation Speed (vel_units) / (cm/s):%"GSYM" / %"GSYM"\n",1.0/SphereRotationalPeriod[sphere],
															     1.0/(SphereRotationalPeriod[sphere]/VelocityUnits));
			printf("\nRoatation Time (t_code) :%"GSYM"\n",2.0*pi*SphereCoreRadius[sphere][0]*SphereRotationalPeriod[sphere]);
		}
		
		if(MyProcessorNumber == ROOT_PROCESSOR)
			printf("\nt_ff = %"GSYM" (t_code)\n", sqrt(3.0*pi/32.0 / (MyGrav *RealSphereDensity[sphere]*(1.0-exp(-0.5)))));
		
		
		// Calculate speed of sound for this sphere
		VelocitySound[sphere] =	sqrt((SphereTemperature[sphere] * Gamma * kboltz) /
							(mu * mh)) / VelocityUnits;
		if(MyProcessorNumber == ROOT_PROCESSOR)
			printf("\nVelocitySound (cm s^-1): %"GSYM"\n", VelocitySound[sphere] * VelocityUnits);
		
		double sig=SphereDensity[sphere]*SolarMass/(2.0*pi*SphereCoreRadius[sphere][0]*LengthUnits*SphereCoreRadius[sphere][1]*LengthUnits);
		
		int N_x = 500;
		int N_z = 500;
		
		double del_x=SphereCoreRadius[sphere][0]*SphereRadius[sphere][0]*LengthUnits/(N_x-1.0);
		double del_z=SphereCoreRadius[sphere][2]*SphereRadius[sphere][2]*LengthUnits/(N_z-1.0);
			
		Galaxy[sphere] = hydrostatic_disk(N_x, N_z, del_x, del_z, 
		                                SphereCoreRadius[sphere][0]*LengthUnits, SphereCoreRadius[sphere][2]*LengthUnits,
                                                sig, VelocitySound[sphere]*VelocityUnits);
		 
		Galaxy[sphere].set_magn_fract(SphereMagnFactor[sphere]);
		Galaxy[sphere].set_halo_mass(HaloMass[sphere]*SolarMass);
		Galaxy[sphere].set_halo_scale(HaloCoreRadius[sphere]*LengthUnits);
		if(SphereMagnEquipart[sphere]==1)
			Galaxy[sphere].set_magn_equipart();
		
		double err=1.0;	
		int counter=0;
		
		while( err > 1.0e-6 && counter < 100 )
		{
			err=Galaxy[sphere].integrate();
			counter++;
			if(MyProcessorNumber == ROOT_PROCESSOR)
				printf("\nGrid for Sphere %i: %ith Iteration. Error=%e\n", sphere, counter, err);
		}
		
		Galaxy[sphere].rotation_curve();
				
		if(MyProcessorNumber == ROOT_PROCESSOR)
			printf("\nFinished setting up Grid for Sphere %i.\n",sphere);
		
	}// ENDFOR sphere
	
	for (k = 0; k < GridDimension[2]; k++)
		for (j = 0; j < GridDimension[1]; j++)
			for (i = 0; i < GridDimension[0]; i++, n++)
	{
		
		density=0.0;//InitialDensity;
		DM_rho=0.0;
		for(int dim=0;dim<GridRank;dim++)
			DM_vel[dim]=0.0;
		DM_sigma=1.0;
		
	 	/* Compute position */
		
		x = CellLeftEdge[0][i] + 0.5*CellWidth[0][i];
		if (GridRank > 1)
			y = CellLeftEdge[1][j] + 0.5*CellWidth[1][j];
		if (GridRank > 2)
			z = CellLeftEdge[2][k] + 0.5*CellWidth[2][k];
		
		/* Loop over spheres. */
		
		temperature = temp1 = InitialTemperature;
		sigma = sigma1 = 0;
		metallicity = tiny_number;
		for (int dim = 0; dim < MAX_DIMENSION; dim++)
		{
			Velocity[dim] = 0.0;
			MagnField[dim] = 0.0;
			DMVelocity[dim] = 0.0;
		}
		for (sphere = 0; sphere < NumberOfSpheres; sphere++)
		{
			/* Find distance from center. */
			
			float xpos, ypos, zpos;
			xpos = x-SpherePosition[sphere][0];
			ypos = y-SpherePosition[sphere][1];
			zpos = z-SpherePosition[sphere][2];
			
			double xpos_rs=xpos;
			double ypos_rs=ypos;
			double zpos_rs=zpos;
			
			if(MAX_DIMENSION==3 && !(SphereAngularMomentum[sphere][2]==0.0 && SphereAngularMomentum[sphere][1]==0))
			{
				
				double x_temp=	xpos*SphereTransformMatrix[sphere][0][0]+
								ypos*SphereTransformMatrix[sphere][0][1]+
	 							zpos*SphereTransformMatrix[sphere][0][2];
				
				double y_temp=  xpos*SphereTransformMatrix[sphere][1][0]+
								ypos*SphereTransformMatrix[sphere][1][1]+
								zpos*SphereTransformMatrix[sphere][1][2];
				
				double z_temp=  xpos*SphereTransformMatrix[sphere][2][0]+
								ypos*SphereTransformMatrix[sphere][2][1]+
								zpos*SphereTransformMatrix[sphere][2][2];
				
			    xpos = x_temp;
				ypos = y_temp;
				zpos = z_temp;
				
			}
			double xposn=xpos/SphereCoreRadius[sphere][0];
			double yposn=ypos/SphereCoreRadius[sphere][1];
			double zposn=zpos/SphereCoreRadius[sphere][2];
			
			r = sqrt(xpos*xpos + ypos*ypos + zpos*zpos);
			rcyl = sqrt(xpos*xpos +ypos*ypos);
			r = fmax(r, 0.01*CellWidth[0][0]);
			//rcyl = max(rcyl, 0.1*CellWidth[0][0]);
			
			double rn = sqrt(xposn*xposn + yposn*yposn + zposn*zposn);
			double rcyln = sqrt(xposn*xposn +yposn*yposn);
			rn = fmax(rn, 0.01*CellWidth[0][0]);
			//rcyln = max(rcyln, 0.1*CellWidth[0][0]);
			
			double rh = r/HaloCoreRadius[sphere];
			if(rn<HaloRadius[sphere] && SphereUseParticles==1)
			{
				DM_rho=Halo_rho[sphere]/(rh*pow(1.0+rh,3.0));
				DM_vel[0]=SphereVelocity[sphere][0];
				DM_vel[1]=SphereVelocity[sphere][1];
				DM_vel[2]=SphereVelocity[sphere][2];
				DM_sigma=Halo_disp[sphere]*(
				12.0*rh*(rh+1.0)*(rh+1.0)*(rh+1.0)*log((rh+1.0)/rh)
				 - rh/(rh+1.0)*(25.0+52.0*rh+42.0*rh*rh+12.0*rh*rh*rh)
				 );
			}
			
			/* Compute Cartesian coordinates for rotational properties */
			
			//outer_radius = (SphereSmoothSurface[sphere] == TRUE) ? 
			//SphereSmoothRadius[sphere]*SphereRadius[sphere][0] : SphereRadius[sphere][0];
			
			outer_radius =	 pow(xposn/SphereRadius[sphere][0],2)
							+pow(yposn/SphereRadius[sphere][1],2)
							+pow(zposn/SphereRadius[sphere][2],2);
			
			outer_radius = fmax(r/(SphereCoreRadius[sphere][0]*SphereRadius[sphere][0]),fabs(zpos)/(SphereCoreRadius[sphere][2]*SphereRadius[sphere][2]));
			
			bool inside = ( ( r<SphereCoreRadius[sphere][0]*SphereRadius[sphere][0]) && (fabs(zpos)<SphereCoreRadius[sphere][2]*SphereRadius[sphere][2]) );
			
			if (inside) {
			
			/* Compute spherical coordinate theta */
			
			if (xpos != 0)
			{
				if (xpos > 0 && ypos >= 0)
					theta = atan(ypos/xpos);
				else
					if (xpos < 0 && ypos >= 0)
						theta = pi + atan(ypos/xpos);
					else
						if (xpos < 0 && ypos < 0)
							theta = pi + atan(ypos/xpos);
						else
							if (xpos > 0 && ypos < 0)
								theta = 2*pi + atan(ypos/xpos);
			}
			else
			{
				if (xpos == 0 && ypos > 0)
					theta = 3.14159 / 2.0;
				else
					if (xpos == 0 && ypos < 0)
						theta = (3*3.14159) / 2.0;
					else
						theta = 0.0;
			}
			
			double sc_h=0;
			
			bool appl_grid=true;
			
			switch(SphereType[sphere])
			{
				case 3:
					{
					//sc_h=pow(VelocitySound[sphere],2)
					//			/(pi*MyGrav/2.0*SphereDensity[sphere]*SphereCoreRadius[sphere][2])
					//			*exp(rcyln);
					sc_h=h_0[sphere]*exp(rcyln);
					//density=RealSphereDensity[sphere]/exp(2.0*rcyln)/pow(cosh(zpos/sc_h),2);
					
					
					double den=Galaxy[sphere].intpl_rho(rcyl*LengthUnits, fabs(zpos)*LengthUnits)/DensityUnits;
					
					density=den;
					
					if(den < 0.0)
					{
						appl_grid=false;
						density=0.0;
					}
					//density=SphereDensity[sphere]*exp(-rcyln)*exp(-abs(zposn));
					break;
					}	
				case 4:
					{
					density=RealSphereDensity[sphere]*exp( -0.5*pow(rn,2) ); 
					break;
					}	
			}
			
			
			if (SphereRotationalPeriod[sphere] > 0)
			{
				
				
				RotVelocity[0]=SphereAngularMomentum[sphere][1]*zpos_rs-SphereAngularMomentum[sphere][2]*ypos_rs;
				RotVelocity[1]=SphereAngularMomentum[sphere][2]*xpos_rs-SphereAngularMomentum[sphere][0]*zpos_rs;
				RotVelocity[2]=SphereAngularMomentum[sphere][0]*ypos_rs-SphereAngularMomentum[sphere][1]*xpos_rs;
				
				//double r_2=sqrt(xpos_rs*xpos_rs+ypos_rs*ypos_rs+zpos_rs*zpos_rs);
				double r_2=sqrt(RotVelocity[0]*RotVelocity[0]+RotVelocity[1]*RotVelocity[1]+RotVelocity[2]*RotVelocity[2]);
				
				if(r_2>0.1*CellWidth[0][0])
				{
					RotVelocity[0]/=r_2;
					RotVelocity[1]/=r_2;
					RotVelocity[2]/=r_2;
				}
				else
				{
					RotVelocity[0]=0.0;
					RotVelocity[1]=0.0;
					RotVelocity[2]=0.0;
				}
			}
			else
			{
				RotVelocity[0] = RotVelocity[1] = RotVelocity[2] = 0;
			}
			
			double r_test_n=rcyl/SphereCoreRadius[sphere][0];
			double r_f= sqrt((1.0-exp(-0.5*rn*rn))/(r/SphereCoreRadius[sphere][0]));//SphereFracKeplerianRot[sphere]*min(1.0,1.0/rn)*density;
			
			double s_p=0.0;
			
			if(SphereType[sphere]==3)
			{
				r_f=1.0/SphereRotationalPeriod[sphere]*0.5*r_test_n*sqrt(BESSI0(0.5*r_test_n)*BESSK0(0.5*r_test_n)-BESSI1(0.5*r_test_n)*BESSK1(0.5*r_test_n));
				//s_p=-2.0*r_test_n*exp(-2.0*r_test_n)*pow(VelocitySound[sphere]*SphereFracKeplerianRot[sphere],2);
				//s_p=-2.0*r_test_n*pow(VelocitySound[sphere]*SphereFracKeplerianRot[sphere],2);
				
				if(appl_grid)
					s_p = Galaxy[sphere].intpl_v_sqr(rcyl*LengthUnits, fabs(zpos)*LengthUnits)*pow(SphereFracKeplerianRot[sphere],2) /
					      (VelocityUnits*VelocityUnits);
			}
			
			double vel_h_sq=0.0;
			
			if(SphereUseParticles == 1)
			{
				vel_h_sq = Galaxy[sphere].halo_vel_sq(rcyl*LengthUnits, 0.0)*pow(SphereFracKeplerianRot[sphere],2.0) / (VelocityUnits*VelocityUnits);
			}
			double f_sm = 1.0 - fmin(1.0, fmax(0.0, (outer_radius-SphereSmoothRadius[sphere])/(1.0-SphereSmoothRadius[sphere])));
			
			Velocity[0] = sqrt(fmax(0.0,r_f*r_f+s_p+vel_h_sq))*RotVelocity[0] + SphereVelocity[sphere][0];
			Velocity[1] = sqrt(fmax(0.0,r_f*r_f+s_p+vel_h_sq))*RotVelocity[1] + SphereVelocity[sphere][1];
			Velocity[2] = sqrt(fmax(0.0,r_f*r_f+s_p+vel_h_sq))*RotVelocity[2] + SphereVelocity[sphere][2];
			
			
			Velocity[0] += SphereTurbulence[sphere] * 
				Maxwellian(VelocitySound[sphere], VelocityUnits, mu, Gamma);
			Velocity[1] += SphereTurbulence[sphere] * 
				Maxwellian(VelocitySound[sphere], VelocityUnits, mu, Gamma);
			Velocity[2] += SphereTurbulence[sphere] * 
				Maxwellian(VelocitySound[sphere], VelocityUnits, mu, Gamma);
			m = 0;
			
			if (HydroMethod == MHD_RK)
			{
				for(int dim=0;dim<MAX_DIMENSION;dim++)
				{
					if(SphereMagnEquipart[sphere]==1)
						MagnField[dim]=sqrt(2.0*SphereMagnFactor[sphere]*density)*VelocitySound[sphere]*RotVelocity[dim];
					else
						MagnField[dim]=MagnConversionFactor*Galaxy[sphere].B(rcyl*LengthUnits, fabs(zpos)*LengthUnits)*RotVelocity[dim];
					//MagnField[dim]=SphereMagnFactor[sphere]*RotVelocity[dim]*rcyln/exp(2.0*rcyln)/pow(cosh(zpos/sc_h),2);
					//MagnField[dim]+=0.001*Maxwellian(VelocitySound[sphere], VelocityUnits, mu, Gamma);
				}
				m = 0;
			}
			
			
			for(int dim=0;dim<MAX_DIMENSION;dim++)
			{
				Velocity[dim] *= f_sm;
				if(HydroMethod == MHD_RK)
					MagnField[dim] *= f_sm;
			}
			density*=f_sm;
			
			switch(SphereConstantPressure[sphere])
			{
				case -1:
					temperature = SphereTemperature[sphere] * pow(fmax(density,InitialDensity)/RealSphereDensity[sphere],Gamma-1.0);
					break;
				case 0:
					temperature = SphereTemperature[sphere];
					break;
				case 1:
					temperature = SphereTemperature[sphere] / (fmax(density,InitialDensity)/RealSphereDensity[sphere]);
					break;
	   		}
			//temperature = SphereTemperature[sphere] * pow(density/SphereDensity[sphere],Gamma-1.0);
			//temperature = SphereTemperature[sphere] / (density/SphereDensity[sphere]);
			
			metallicity += SphereMetallicity[sphere];
			//density=max(InitialDensity,SphereDensity[sphere]*exp(-0.5*pow(rn,2)));
		} // end: if (r < SphereRadius)
	} // end: loop over spheres
	
	if(density<InitialDensity)
	{
		if(HydroMethod==MHD_RK)
		{
			for(int dim=0;dim<MAX_DIMENSION;dim++)
				MagnField[dim]*=density/InitialDensity;
		}
		for(int dim=0;dim<MAX_DIMENSION;dim++)
			Velocity[dim]*=density/InitialDensity;
		temperature=( (density*temperature)+(InitialDensity-density)*InitialTemperature )/(InitialDensity);
		density=InitialDensity;
	}
	
	//temperature = InitialTemperature;
	
	
	for(int dim=0;dim<MAX_DIMENSION;dim++)
	{
		MagnField[dim]+=MagnConversionFactor*InitialMagnField*gasdev();//1.0e-30;
		//MagnField[dim]=0.0;
	}
	
	//double c_sound = 	sqrt((temperature * Gamma * kboltz) / 
	//					(mu * mh)) / VelocityUnits;
	
	//Velocity[0] += Maxwellian(c_sound, VelocityUnits, mu, Gamma);
	//Velocity[1] += Maxwellian(c_sound, VelocityUnits, mu, Gamma);
	//Velocity[2] += Maxwellian(c_sound, VelocityUnits, mu, Gamma);
	
	
	/* Set density. */
	
	BaryonField[0][n] = density*BaryonMeanDensity;
	
	/* If doing multi-species (HI, etc.), set these. */
	
	if (MultiSpecies > 0)
	{
		BaryonField[HIINum][n] = CosmologySimulationInitialFractionHII *
		CoolData.HydrogenFractionByMass * BaryonField[0][n] *
			sqrt(OmegaMatterNow)/
			(OmegaMatterNow*BaryonMeanDensity*HubbleConstantNow);
		//(CosmologySimulationOmegaBaryonNow*HubbleConstantNow);
		
		BaryonField[HeIINum][n] = CosmologySimulationInitialFractionHeII*
			BaryonField[0][n] * 4.0 * (1.0-CoolData.HydrogenFractionByMass);
		BaryonField[HeIIINum][n] = CosmologySimulationInitialFractionHeIII*
			BaryonField[0][n] * 4.0 * (1.0-CoolData.HydrogenFractionByMass);
		BaryonField[HeINum][n] = (1.0 - CoolData.HydrogenFractionByMass)*BaryonField[0][n] -
			BaryonField[HeIINum][n] - BaryonField[HeIIINum][n];
		
		if (MultiSpecies > 1)
		{
			BaryonField[HMNum][n] = CosmologySimulationInitialFractionHM*
			BaryonField[HIINum][n]* pow(temperature,float(0.88));
			BaryonField[H2IINum][n] = CosmologySimulationInitialFractionH2II*
				2.0*BaryonField[HIINum][n]* pow(temperature,float(1.8));
			BaryonField[H2INum][n] = CosmologySimulationInitialFractionH2I*
				BaryonField[0][n]*CoolData.HydrogenFractionByMass*pow(301.0,5.1)*
				pow(OmegaMatterNow, float(1.5))/
				(OmegaMatterNow*BaryonMeanDensity)/
				//CosmologySimulationOmegaBaryonNow/
			HubbleConstantNow*2.0;
		}
		
		BaryonField[HINum][n] = CoolData.HydrogenFractionByMass*BaryonField[0][n]
			- BaryonField[HIINum][n];
		if (MultiSpecies > 1)
			BaryonField[HINum][n] -= BaryonField[HMNum][n]
				+ BaryonField[H2IINum][n]
				+ BaryonField[H2INum][n];
		
		BaryonField[DeNum][n] = BaryonField[HIINum][n] + 
			0.25*BaryonField[HeIINum][n] + 0.5*BaryonField[HeIIINum][n];
		if (MultiSpecies > 1)
			BaryonField[DeNum][n] += 0.5*BaryonField[H2IINum][n] - 
		BaryonField[HMNum][n];
		
		/* Set Deuterium species (assumed to be negligible). */
		
		if (MultiSpecies > 2)
		{
			BaryonField[DINum][n] = CoolData.DeuteriumToHydrogenRatio * BaryonField[HINum][n];
			BaryonField[DIINum][n] = CoolData.DeuteriumToHydrogenRatio * BaryonField[HIINum][n];
			BaryonField[HDINum][n] = CoolData.DeuteriumToHydrogenRatio * BaryonField[H2INum][n];
		}
	}
	
	/* If there are metals, set it. */
	
	if (SphereUseMetals)
		BaryonField[MetalNum][n] = metallicity * CoolData.SolarMetalFractionByMass * BaryonField[0][n];
	
	
	/* Set Velocities. */
	
	for (int dim = 0; dim < GridRank; dim++)
	{
		BaryonField[ivel+dim][n] = Velocity[dim] + UniformVelocity[dim];
		if (HydroMethod == MHD_RK)
			BaryonField[imag+dim][n] = MagnField[dim];
	}
	if(HydroMethod == MHD_RK)
		BaryonField[imag+GridRank][n] = 0.0;//1.0e-30;
	
	/* Set energy (thermal and then total if necessary). */
	
	BaryonField[1][n] = temperature/TemperatureUnits/ ((Gamma-1.0)*mu);
	
	if (DualEnergyFormalism)
		BaryonField[2][n] = BaryonField[1][n];
	
	if (HydroMethod != Zeus_Hydro)
		for (int dim = 0; dim < GridRank; dim++)
		{
			BaryonField[1][n] += 0.5*pow(BaryonField[ivel+dim][n], 2);
			if(HydroMethod == MHD_RK)
				BaryonField[1][n] += 0.5*pow(BaryonField[imag+dim][n], 2)/BaryonField[0][n];
		}
		
	/* Set particles if being used (generate a number of particle proportional to density). */
	
	if (SphereUseParticles)
		if (i >= GridStartIndex[0] && i <= GridEndIndex[0] &&
			j >= GridStartIndex[1] && j <= GridEndIndex[1] &&
			k >= GridStartIndex[2] && k <= GridEndIndex[2]  )
		{
			//ParticleCount += density/pow(float(RefineBy), GridRank*level);
			//while (ParticleCount > 1)
			if(DM_rho>0.0)
			{
				if (SetupLoopCount > 0)
				{
					ParticleMass[npart] = DM_rho;
					//ParticleMass[npart] = ParticleMeanDensity * pow(float(RefineBy), GridRank*level);
					ParticleNumber[npart] = CollapseTestParticleCount++;
					ParticleType[npart] = PARTICLE_TYPE_DARK_MATTER;
					
					/* Set random position within cell. */
					
					ParticlePosition[0][npart] = x +
						CellWidth[0][0]*(float(rand())/float(RAND_MAX) - 0.5);
					ParticlePosition[1][npart] = y +
						CellWidth[1][0]*(float(rand())/float(RAND_MAX) - 0.5);
					ParticlePosition[2][npart] = z +
						CellWidth[2][0]*(float(rand())/float(RAND_MAX) - 0.5);
					
					/* Set bulk velocity. */
					
					for (sphere = 0; sphere<NumberOfSpheres;sphere++)
					{
						xdist = ParticlePosition[0][npart]-SpherePosition[sphere][0];
						ydist = ParticlePosition[1][npart]-SpherePosition[sphere][1];
						zdist = ParticlePosition[2][npart]-SpherePosition[sphere][2];
						for (int dim = 0; dim < GridRank; dim++)
							if (sqrt(xdist*xdist + ydist*ydist + zdist*zdist) < SphereRadius[sphere][0])
							{ //particle is inside sphere
								ParticleVelocity[dim][npart] = DMVelocity[dim]+UniformVelocity[dim]+SphereVelocity[sphere][dim] ;
							}
							else
							{ //particle is outside sphere
								ParticleVelocity[dim][npart] = DM_vel[dim]+UniformVelocity[dim];
							}
						}
						/* Add random velocity; */
						
						if (DM_sigma != 0)
							for (int dim = 0; dim < GridRank; dim++)
								ParticleVelocity[dim][npart] += gasdev()*DM_sigma/VelocityUnits;
						
					}
					npart++;
					ParticleCount -= 1.0;
				}
			} // end: if statement
			
		} // end loop over grid
	
	for(int sphere=0;sphere<NumberOfSpheres;sphere++)
	{
		Galaxy[sphere].free_data();
	}
	
	} // end loop SetupLoopCount
	
	if (SphereUseParticles)
		printf("CollapseTestInitialize: NumberOfParticles = %"ISYM"\n", NumberOfParticles);
	
	return SUCCESS;
}
