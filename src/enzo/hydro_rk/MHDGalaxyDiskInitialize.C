/***********************************************************************
/
/  INITIALIZE A COLLAPSE TEST
/
/  written by: Greg Bryan
/  date:       May, 1998
/  modified1:  Simon Selg
/  date1:      11/2019
/
/  PURPOSE:
/    Set up a number of spherical objects
/
/  RETURNS: SUCCESS or FAIL
/
************************************************************************/

// This routine intializes a new simulation based on the parameter file.
//

#include <stdlib.h>
#include "preincludes.h"
#include "macros_and_parameters.h"
#include "typedefs.h"
#include "global_data.h"
#include "Fluxes.h"
#include "GridList.h"
#include "ExternalBoundary.h"
#include "Grid.h"
#include "Hierarchy.h"
#include "LevelHierarchy.h"
#include "TopGridData.h"

void WriteListOfFloats(FILE *fptr, int N, float floats[]);
void WriteListOfFloats(FILE *fptr, int N, FLOAT floats[]);
void AddLevel(LevelHierarchyEntry *Array[], HierarchyEntry *Grid, int level);
int RebuildHierarchy(TopGridData *MetaData,
		     LevelHierarchyEntry *LevelArray[], int level);

// ============================================================================
// S. Selg (11/2019): Adjustments in order to use PRGIO. Mind that there is a
// discrimination between the states of ``SetBaryonFiels``.
// ----------------------------------------------------------------------------

int MHDGalaxyDiskInitialize(FILE *fptr, FILE *Outfptr, 
			  HierarchyEntry &TopGrid, TopGridData &MetaData,
			  int SetBaryonFields
		)
{
  const char *DensName = "Density";
  const char *TEName   = "TotalEnergy";
  const char *GEName   = "GasEnergy";
  const char *Vel1Name = "x-velocity";
  const char *Vel2Name = "y-velocity";
  const char *Vel3Name = "z-velocity";
  
  // S. Selg (08/2019): Gravitational Potential (for output)
  const char *GPotName          = "Grav_Potential";  
  const char *Bfield1Name	= "Bx";
  const char *Bfield2Name	= "By";
  const char *Bfield3Name	= "Bz";
  const char *PhiName		= "Phi";
	
  const char *ColourName = "colour";
  const char *ElectronName = "Electron_Density";
  const char *HIName    = "HI_Density";
  const char *HIIName   = "HII_Density";
  const char *HeIName   = "HeI_Density";
  const char *HeIIName  = "HeII_Density";
  const char *HeIIIName = "HeIII_Density";
  const char *HMName    = "HM_Density";
  const char *H2IName   = "H2I_Density";
  const char *H2IIName  = "H2II_Density";
  const char *DIName    = "DI_Density";
  const char *DIIName   = "DII_Density";
  const char *HDIName   = "HDI_Density";
  const char *MetalName = "Metal_Density";

  /* declarations */

  char  line[MAX_LINE_LENGTH];
  int   dim, ret, level, sphere, i;

  /* set default parameters */
  int MHDGalaxyDiskNumberOfSpheres = 1;
  int MHDGalaxyDiskRefineAtStart   = TRUE;
  int MHDGalaxyDiskUseParticles    = FALSE;
  int MHDGalaxyDiskUseColour       = FALSE;
  int MHDGalaxyDiskUseMetals       = FALSE;
  float MHDGalaxyDiskParticleMeanDensity = FLOAT_UNDEFINED;
  float MHDGalaxyDiskInitialTemperature = 1000;
  float MHDGalaxyDiskInitialDensity     = 1.0;
  float MHDGalaxyDiskInitialMagnField   = 0.0;

  int MHDGalaxyDiskNumShells[MAX_SPHERES],
      MHDGalaxyDiskInitialLevel[MAX_SPHERES],
      MHDGalaxyDiskType[MAX_SPHERES],
      MHDGalaxyDiskConstantPressure[MAX_SPHERES],
      MHDGalaxyDiskMagnEquipart[MAX_SPHERES],
      MHDGalaxyDiskSmoothSurface[MAX_SPHERES],
      MHDGalaxyDiskPressureGradientType[MAX_SPHERES]; // S.C.S (08/2019)

  float MHDGalaxyDiskDensity[MAX_SPHERES],
        MHDGalaxyDiskTemperature[MAX_SPHERES],
        MHDGalaxyDiskFracKeplerianRot[MAX_SPHERES],
        MHDGalaxyDiskTurbulence[MAX_SPHERES],
        MHDGalaxyDiskDispersion[MAX_SPHERES],
        MHDGalaxyDiskCutOff[MAX_SPHERES],
        MHDGalaxyDiskAng1[MAX_SPHERES],
        MHDGalaxyDiskAng2[MAX_SPHERES],
        MHDGalaxyDiskMetallicity[MAX_SPHERES],
        MHDGalaxyDiskSmoothRadius[MAX_SPHERES],
        MHDGalaxyDiskMagnFactor[MAX_SPHERES],
        MHDGalaxyDiskHaloRadius[MAX_SPHERES],
        MHDGalaxyDiskHaloCoreRadius[MAX_SPHERES],
        MHDGalaxyDiskHaloMass[MAX_SPHERES];

  float MHDGalaxyDiskUniformVelocity[MAX_DIMENSION];

  float MHDGalaxyDiskPosition[MAX_SPHERES][MAX_DIMENSION], 
        MHDGalaxyDiskRadius[MAX_SPHERES][MAX_DIMENSION],
        MHDGalaxyDiskCoreRadius[MAX_SPHERES][MAX_DIMENSION],
        MHDGalaxyDiskVelocity[MAX_SPHERES][MAX_DIMENSION],
        MHDGalaxyDiskAngularMomentum[MAX_SPHERES][MAX_DIMENSION];

  for (sphere = 0; sphere < MAX_SPHERES; sphere++) {
    MHDGalaxyDiskPressureGradientType[sphere]   = 2; // S.C.S (08/2019)
    MHDGalaxyDiskDensity[sphere]		= 1.0;
    MHDGalaxyDiskTemperature[sphere]	= 1.0;
    MHDGalaxyDiskFracKeplerianRot[sphere]	= 0.0;
    MHDGalaxyDiskTurbulence[sphere]	= 0.0;
    MHDGalaxyDiskDispersion[sphere]	= 0.0;
    MHDGalaxyDiskCutOff[sphere]		= 6.5;
    MHDGalaxyDiskAng1[sphere]			= 0;
    MHDGalaxyDiskAng2[sphere]			= 0;
    MHDGalaxyDiskNumShells[sphere]		= 1;
    MHDGalaxyDiskSmoothRadius[sphere]	= 1.2;
    MHDGalaxyDiskMetallicity[sphere]	= 0;
    MHDGalaxyDiskInitialLevel[sphere]	= 0;
    MHDGalaxyDiskMagnFactor[sphere]	= 0.0;
    MHDGalaxyDiskHaloCoreRadius[sphere]		= 0.2;
    MHDGalaxyDiskHaloRadius[sphere]			= 10.0;
    MHDGalaxyDiskHaloMass[sphere]			= 1.0e11;

    for (dim = 0; dim < MAX_DIMENSION; dim++) {
      MHDGalaxyDiskPosition[sphere][dim] = 0.5*(DomainLeftEdge[dim] +
						     DomainRightEdge[dim]);
      MHDGalaxyDiskVelocity[sphere][dim] = 0;
    MHDGalaxyDiskRadius[sphere][dim]     = 5.0;
    MHDGalaxyDiskCoreRadius[sphere][dim] = 0.2;
	MHDGalaxyDiskAngularMomentum[sphere][dim]=0.0;
    }
    MHDGalaxyDiskAngularMomentum[sphere][0]=1.0;
    MHDGalaxyDiskType[sphere]       = 0;
    MHDGalaxyDiskConstantPressure[sphere] = FALSE;
    MHDGalaxyDiskMagnEquipart[sphere] = FALSE;
    MHDGalaxyDiskSmoothSurface[sphere] = FALSE;
  }
  for (dim = 0; dim < MAX_DIMENSION; dim++)
    MHDGalaxyDiskUniformVelocity[dim] = 0;

  /* read input from file */

  while (fgets(line, MAX_LINE_LENGTH, fptr) != NULL) {

    ret = 0;

    /* read parameters */

    ret += sscanf(line, "MHDGalaxyDiskNumberOfSpheres = %"ISYM,
		  &MHDGalaxyDiskNumberOfSpheres);
    ret += sscanf(line, "MHDGalaxyDiskRefineAtStart = %"ISYM, 
		  &MHDGalaxyDiskRefineAtStart);
    ret += sscanf(line, "MHDGalaxyDiskUseParticles = %"ISYM, 
		  &MHDGalaxyDiskUseParticles);
    ret += sscanf(line, "MHDGalaxyDiskParticleMeanDensity = %"FSYM,
		  &MHDGalaxyDiskParticleMeanDensity);
    ret += sscanf(line, "MHDGalaxyDiskUseColour = %"ISYM, 
		  &MHDGalaxyDiskUseColour);
    ret += sscanf(line, "MHDGalaxyDiskUseMetals = %"ISYM, 
		  &MHDGalaxyDiskUseMetals);
    ret += sscanf(line, "MHDGalaxyDiskInitialTemperature = %"FSYM, 
		  &MHDGalaxyDiskInitialTemperature);
    ret += sscanf(line, "MHDGalaxyDiskInitialDensity = %"FSYM,
		  &MHDGalaxyDiskInitialDensity);
    ret += sscanf(line, "MHDGalaxyDiskInitialMagnField = %"FSYM,
		  &MHDGalaxyDiskInitialMagnField);
    ret += sscanf(line, "MHDGalaxyDiskUniformVelocity = %"FSYM" %"FSYM" %"FSYM, 
		  MHDGalaxyDiskUniformVelocity, MHDGalaxyDiskUniformVelocity+1,
		  MHDGalaxyDiskUniformVelocity+2);
    if (sscanf(line, "MHDGalaxyDiskPressureGradientType[%"ISYM"]", &sphere) > 0)
	    ret += sscanf(line, "MHDGalaxyDiskPressureGradientType[%"ISYM"] = %"ISYM, &sphere,
			    &MHDGalaxyDiskPressureGradientType[sphere]);
    if (sscanf(line, "MHDGalaxyDiskType[%"ISYM"]", &sphere) > 0)
      ret += sscanf(line, "MHDGalaxyDiskType[%"ISYM"] = %"ISYM, &sphere,
		    &MHDGalaxyDiskType[sphere]);
    if (sscanf(line, "MHDGalaxyDiskDensity[%"ISYM"]", &sphere) > 0)
      ret += sscanf(line, "MHDGalaxyDiskDensity[%"ISYM"] = %"FSYM, &sphere,
		    &MHDGalaxyDiskDensity[sphere]);
    if (sscanf(line, "MHDGalaxyDiskTemperature[%"ISYM"]", &sphere) > 0)
      ret += sscanf(line, "MHDGalaxyDiskTemperature[%"ISYM"] = %"FSYM, &sphere,
		    &MHDGalaxyDiskTemperature[sphere]);
    if (sscanf(line, "MHDGalaxyDiskMetallicity[%"ISYM"]", &sphere) > 0)
      ret += sscanf(line, "MHDGalaxyDiskMetallicity[%"ISYM"] = %"FSYM, &sphere,
		    &MHDGalaxyDiskMetallicity[sphere]);
    if (sscanf(line, "MHDGalaxyDiskPosition[%"ISYM"]", &sphere) > 0)
      ret += sscanf(line, "MHDGalaxyDiskPosition[%"ISYM"] = %"PSYM" %"PSYM" %"PSYM, 
		    &sphere, &MHDGalaxyDiskPosition[sphere][0],
		    &MHDGalaxyDiskPosition[sphere][1],
		    &MHDGalaxyDiskPosition[sphere][2]);
	if (sscanf(line, "MHDGalaxyDiskCoreRadius[%"ISYM"]", &sphere) > 0)
      ret += sscanf(line, "MHDGalaxyDiskCoreRadius[%"ISYM"] = %"PSYM" %"PSYM" %"PSYM, 
                  &sphere, &MHDGalaxyDiskCoreRadius[sphere][0],
                  &MHDGalaxyDiskCoreRadius[sphere][1], 
                  &MHDGalaxyDiskCoreRadius[sphere][2]);
    if (sscanf(line, "MHDGalaxyDiskRadius[%"ISYM"]", &sphere) > 0)
      ret += sscanf(line, "MHDGalaxyDiskRadius[%"ISYM"] = %"PSYM" %"PSYM" %"PSYM,
                    &sphere, &MHDGalaxyDiskRadius[sphere][0],
                    &MHDGalaxyDiskRadius[sphere][1],
                    &MHDGalaxyDiskRadius[sphere][2]);
    if (sscanf(line, "MHDGalaxyDiskAngularMomentum[%"ISYM"]", &sphere) > 0)
      ret += sscanf(line, "MHDGalaxyDiskAngularMomentum[%"ISYM"] = %"PSYM" %"PSYM" %"PSYM,
                    &sphere, &MHDGalaxyDiskAngularMomentum[sphere][0],
                    &MHDGalaxyDiskAngularMomentum[sphere][1],
                    &MHDGalaxyDiskAngularMomentum[sphere][2]);
    if (sscanf(line, "MHDGalaxyDiskVelocity[%"ISYM"]", &sphere) > 0)
      ret += sscanf(line, "MHDGalaxyDiskVelocity[%"ISYM"] = %"FSYM" %"FSYM" %"FSYM, 
		    &sphere, &MHDGalaxyDiskVelocity[sphere][0],
		    &MHDGalaxyDiskVelocity[sphere][1],
		    &MHDGalaxyDiskVelocity[sphere][2]);
	if (sscanf(line, "MHDGalaxyDiskHaloMass[%"ISYM"]", &sphere) > 0)
	      ret += sscanf(line, "MHDGalaxyDiskHaloMass[%"ISYM"] = %"FSYM, &sphere,
	                          &MHDGalaxyDiskHaloMass[sphere]);
	if (sscanf(line, "MHDGalaxyDiskHaloCoreRadius[%"ISYM"]", &sphere) > 0)
	      ret += sscanf(line, "MHDGalaxyDiskHaloCoreRadius[%"ISYM"] = %"FSYM, &sphere,
	                          &MHDGalaxyDiskHaloCoreRadius[sphere]);
	if (sscanf(line, "MHDGalaxyDiskHaloRadius[%"ISYM"]", &sphere) > 0)
	      ret += sscanf(line, "MHDGalaxyDiskHaloRadius[%"ISYM"] = %"FSYM, &sphere,
	                          &MHDGalaxyDiskHaloRadius[sphere]);
    if (sscanf(line, "MHDGalaxyDiskFracKeplerianRot[%"ISYM"]", &sphere) > 0)
      ret += sscanf(line, "MHDGalaxyDiskFracKeplerianRot[%"ISYM"] = %"FSYM, &sphere,
                    &MHDGalaxyDiskFracKeplerianRot[sphere]);
    if (sscanf(line, "MHDGalaxyDiskTurbulence[%"ISYM"]", &sphere) > 0)
      ret += sscanf(line, "MHDGalaxyDiskTurbulence[%"ISYM"] = %"FSYM, &sphere,
                    &MHDGalaxyDiskTurbulence[sphere]);
    if (sscanf(line, "MHDGalaxyDiskDispersion[%"ISYM"]", &sphere) > 0)
      ret += sscanf(line, "MHDGalaxyDiskDispersion[%"ISYM"] = %"FSYM, &sphere,
                    &MHDGalaxyDiskDispersion[sphere]);
    if (sscanf(line, "MHDGalaxyDiskCutOff[%"ISYM"]", &sphere) > 0)
      ret += sscanf(line, "MHDGalaxyDiskCutOff[%"ISYM"] = %"FSYM, &sphere,
                    &MHDGalaxyDiskCutOff[sphere]);
    if (sscanf(line, "MHDGalaxyDiskAng1[%"ISYM"]", &sphere) > 0)
      ret += sscanf(line, "MHDGalaxyDiskAng1[%"ISYM"] = %"FSYM, &sphere,
                    &MHDGalaxyDiskAng1[sphere]);
    if (sscanf(line, "MHDGalaxyDiskAng2[%"ISYM"]", &sphere) > 0)
      ret += sscanf(line, "MHDGalaxyDiskAng2[%"ISYM"] = %"FSYM, &sphere,
                    &MHDGalaxyDiskAng2[sphere]);
    if (sscanf(line, "MHDGalaxyDiskNumShells[%"ISYM"]", &sphere) > 0)
      ret += sscanf(line, "MHDGalaxyDiskNumShells[%"ISYM"] = %"ISYM, &sphere,
                    &MHDGalaxyDiskNumShells[sphere]);
    if (sscanf(line, "MHDGalaxyDiskInitialLevel[%"ISYM"]", &sphere) > 0)
      ret += sscanf(line, "MHDGalaxyDiskInitialLevel[%"ISYM"] = %"ISYM, &sphere,
                    &MHDGalaxyDiskInitialLevel[sphere]);
    if (sscanf(line, "MHDGalaxyDiskConstantPressure[%"ISYM"]", &sphere) > 0)
      ret += sscanf(line, "MHDGalaxyDiskConstantPressure[%"ISYM"] = %"ISYM, &sphere,
		    &MHDGalaxyDiskConstantPressure[sphere]);
	if (sscanf(line, "MHDGalaxyDiskMagnEquipart[%"ISYM"]", &sphere) > 0)
	      ret += sscanf(line, "MHDGalaxyDiskMagnEquipart[%"ISYM"] = %"ISYM, &sphere,
	                  &MHDGalaxyDiskMagnEquipart[sphere]);
    if (sscanf(line, "MHDGalaxyDiskSmoothSurface[%"ISYM"]", &sphere) > 0)
      ret += sscanf(line, "MHDGalaxyDiskSmoothSurface[%"ISYM"] = %"ISYM, &sphere,
		    &MHDGalaxyDiskSmoothSurface[sphere]);
    if (sscanf(line, "MHDGalaxyDiskSmoothRadius[%"ISYM"]", &sphere) > 0)
      ret += sscanf(line, "MHDGalaxyDiskSmoothRadius[%"FSYM"] = %"FSYM, &sphere,
		    &MHDGalaxyDiskSmoothRadius[sphere]);
	if (sscanf(line, "MHDGalaxyDiskMagnFactor[%"ISYM"]", &sphere) > 0)
	      ret += sscanf(line, "MHDGalaxyDiskMagnFactor[%"FSYM"] = %"FSYM, &sphere,
	                  &MHDGalaxyDiskMagnFactor[sphere]);


    /* if the line is suspicious, issue a warning */

    if (ret == 0 && strstr(line, "=") && strstr(line, "MHDGalaxyDisk") 
	&& line[0] != '#' && MyProcessorNumber == ROOT_PROCESSOR)
      fprintf(stderr, "warning: the following parameter line was not interpreted:\n%s\n", line);

  } // end input from parameter file

  /* set up grid */

  /* =========================================================================
   * S. Selg (11/2019): Implementation of Parallel Root Grid IO (PRGIO)
   * =========================================================================
   */

  HierarchyEntry *CurrentGrid;
  CurrentGrid = &TopGrid;
  while (CurrentGrid != NULL) {
  	if (CurrentGrid->GridData->MHDGalaxyDiskInitializeGrid(
					MHDGalaxyDiskNumberOfSpheres,
					MHDGalaxyDiskRadius,
					MHDGalaxyDiskAngularMomentum,
					MHDGalaxyDiskCoreRadius,
					MHDGalaxyDiskDensity,
					MHDGalaxyDiskTemperature,
					MHDGalaxyDiskMetallicity,
					MHDGalaxyDiskPosition,
					MHDGalaxyDiskVelocity,
					MHDGalaxyDiskFracKeplerianRot,
					MHDGalaxyDiskTurbulence,
					MHDGalaxyDiskDispersion,
					MHDGalaxyDiskCutOff,
					MHDGalaxyDiskAng1,
					MHDGalaxyDiskAng2,
					MHDGalaxyDiskNumShells,
					MHDGalaxyDiskType,
					MHDGalaxyDiskConstantPressure,
					MHDGalaxyDiskSmoothSurface,
					MHDGalaxyDiskSmoothRadius,
					MHDGalaxyDiskMagnFactor,
					MHDGalaxyDiskMagnEquipart,
					MHDGalaxyDiskHaloMass,
					MHDGalaxyDiskHaloCoreRadius,
					MHDGalaxyDiskHaloRadius,
					MHDGalaxyDiskUseParticles,
					MHDGalaxyDiskParticleMeanDensity,
					MHDGalaxyDiskUniformVelocity,
					MHDGalaxyDiskUseColour,
					MHDGalaxyDiskUseMetals,
					MHDGalaxyDiskInitialTemperature,
					MHDGalaxyDiskInitialDensity,
					MHDGalaxyDiskInitialMagnField,
		        		MHDGalaxyDiskPressureGradientType,	
					0,
					SetBaryonFields,
					1) == FAIL) {
    			ENZO_FAIL("Error in MHDGalaxyDiskInitializeGrid.");
  			}
	CurrentGrid = CurrentGrid->NextGridThisLevel;
  }

  /* S. Selg (11/2019): This will be done after first initialization! (if you are
   * using prgio. */

  if (SetBaryonFields) {

  /* Convert minimum initial overdensity for refinement to mass
     (unless MinimumMass itself was actually set). */
  for (int count = 0; count < MAX_FLAGGING_METHODS; count++)
    if (MinimumMassForRefinement[count] == FLOAT_UNDEFINED) {
      MinimumMassForRefinement[count] = MinimumOverDensityForRefinement[count];
      for (dim = 0; dim < MetaData.TopGridRank; dim++)
        MinimumMassForRefinement[count] *=(DomainRightEdge[dim]-DomainLeftEdge[dim])/
	     float(MetaData.TopGridDims[dim]);
    }

  // (S. Selg, 10/2019) The following lines use the code from ClusterInitialize and are to 
  // implement a refinement at start 
  // If requested, refine the grids to the desired level
  if (MHDGalaxyDiskRefineAtStart) {
	  // Declare, initialize and fill out the LevelArray
	  LevelHierarchyEntry *LevelArray[MAX_DEPTH_OF_HIERARCHY];
	  for (level = 0; level < MAX_DEPTH_OF_HIERARCHY; level++)
		  LevelArray[level] = NULL;
	  AddLevel(LevelArray, &TopGrid, 0);
	  // Add levels to the maximum depth until no new levels are created,
	  // and re-initialize the level after it's created. 
	  for (level = 0; level < MaximumRefinementLevel; level++) 
	  {
		  if (RebuildHierarchy(&MetaData, LevelArray, level) == FAIL)
		  {
			  fprintf(stderr, "Error in RebuildHierarchy.\n");
			  return FAIL;
		  }
		  if (LevelArray[level+1] == NULL)
			  break;
		  LevelHierarchyEntry *Temp = LevelArray[level+1];
		  while (Temp != NULL)
		  {
			  if (Temp->GridData->MHDGalaxyDiskInitializeGrid(
				MHDGalaxyDiskNumberOfSpheres,
				MHDGalaxyDiskRadius,
				MHDGalaxyDiskAngularMomentum,
				MHDGalaxyDiskCoreRadius,
				MHDGalaxyDiskDensity,
				MHDGalaxyDiskTemperature,
				MHDGalaxyDiskMetallicity,
				MHDGalaxyDiskPosition,
				MHDGalaxyDiskVelocity,
				MHDGalaxyDiskFracKeplerianRot,
				MHDGalaxyDiskTurbulence,
				MHDGalaxyDiskDispersion,
				MHDGalaxyDiskCutOff,
				MHDGalaxyDiskAng1,
				MHDGalaxyDiskAng2,
				MHDGalaxyDiskNumShells,
				MHDGalaxyDiskType,
				MHDGalaxyDiskConstantPressure,
				MHDGalaxyDiskSmoothSurface,
				MHDGalaxyDiskSmoothRadius,
				MHDGalaxyDiskMagnFactor,
				MHDGalaxyDiskMagnEquipart,
				MHDGalaxyDiskHaloMass,
				MHDGalaxyDiskHaloCoreRadius,
				MHDGalaxyDiskHaloRadius,
				MHDGalaxyDiskUseParticles,
				MHDGalaxyDiskParticleMeanDensity,
				MHDGalaxyDiskUniformVelocity,
				MHDGalaxyDiskUseColour,
				MHDGalaxyDiskUseMetals,
				MHDGalaxyDiskInitialTemperature,
				MHDGalaxyDiskInitialDensity,
				MHDGalaxyDiskInitialMagnField,
				MHDGalaxyDiskPressureGradientType,
				level,   // S. Selg (11/2019, used to be level+1)
				SetBaryonFields,
				0) == FAIL) 
				{
					fprintf(stderr, "Error in MHDGalaxyDiskInitializeGrid.\n");
					return FAIL;
				}
			  Temp = Temp->NextGridThisLevel;
		  }
	  } // end: loop over levels

	  // Loop back from the bottom, restoring the consistency among levels.
	  for (level = MaximumRefinementLevel; level > 0; level --)
	  {
		  LevelHierarchyEntry *Temp = LevelArray[level];
		  while (Temp != NULL)
		  {
	//		  if (Temp->GridData->ProjectSolutionToParentGrid(*Temp->GridHierarchyEntry->ParentGrid->GridData) == FAIL) 
			  if (Temp->GridData->ProjectSolutionToParentGrid(
				*LevelArray[level-1]->GridData) == FAIL)		  
			  {
				  fprintf(stderr, "Error in grid->ProjectSolutionToParentGrid.\n");
				  return FAIL;
			  }
			  Temp = Temp->NextGridThisLevel;
		  }
	  }
  } //end: if (MHDGalaxyDiskRefineAtStart)



  /* If requested and there are no manual settings of the refinement
     of spheres, refine the grid to the desired level. */

/*  int MaxInitialLevel = 0;
  for (sphere = 0; sphere < MHDGalaxyDiskNumberOfSpheres; sphere++)
    MaxInitialLevel = max(MaxInitialLevel, MHDGalaxyDiskInitialLevel[sphere]);

  if (MHDGalaxyDiskRefineAtStart) {

    * If the user specified an initial refinement level for a sphere,
       then manually create the hierarchy first. 

    if (MaxInitialLevel > 0) {

      int lev, max_level;
      float dx;
      HierarchyEntry **Subgrid;
      int NumberOfSubgridDims[MAX_DIMENSION];
      FLOAT ThisLeftEdge[MAX_DIMENSION], ThisRightEdge[MAX_DIMENSION];

      for (sphere = 0; sphere < MHDGalaxyDiskNumberOfSpheres; sphere++) {
	
	max_level = MHDGalaxyDiskInitialLevel[sphere];
	if (max_level > 0) {

	  Subgrid = new HierarchyEntry*[max_level];
	  for (lev = 0; lev < max_level; lev++)
	    Subgrid[lev] = new HierarchyEntry;

	  for (lev = 0; lev < max_level; lev++) {
	    
	    for (dim = 0; dim < MetaData.TopGridRank; dim++) {
	      dx = 1.0 / float(MetaData.TopGridDims[dim]) / POW(RefineBy, lev);
	      ThisLeftEdge[dim] = MHDGalaxyDiskPosition[sphere][dim] -
		0.5 * MHDGalaxyDiskRadius[sphere][dim] - 2*dx;  // plus some buffer
	      ThisLeftEdge[dim] = nint(ThisLeftEdge[dim] / dx) * dx;
	      ThisRightEdge[dim] = MHDGalaxyDiskPosition[sphere][dim] +
		0.5 * MHDGalaxyDiskRadius[sphere][dim] + 2*dx;
	      ThisRightEdge[dim] = nint(ThisRightEdge[dim] / dx) * dx;
	      NumberOfSubgridDims[dim] = 
		nint((ThisRightEdge[dim] - ThisLeftEdge[dim]) / 
		     (DomainRightEdge[dim] - DomainLeftEdge[dim]) / dx);		
	    } // ENDFOR dims

	    if (debug)
	      printf("MHDGalaxyDisk:: Level[%"ISYM"]: NumberOfSubgridZones[0] = %"ISYM"\n",
		     lev+1, NumberOfSubgridDims[0]);
	    
	    if (NumberOfSubgridDims[0] > 0) {

	      // Insert into AMR hierarchy
	      if (lev == 0) {
		Subgrid[lev]->NextGridThisLevel = TopGrid.NextGridNextLevel;
		TopGrid.NextGridNextLevel = Subgrid[lev];
		Subgrid[lev]->ParentGrid = &TopGrid;
	      } else {
		Subgrid[lev]->NextGridThisLevel = NULL;
		Subgrid[lev]->ParentGrid = Subgrid[lev-1];
	      }
	      if (lev == max_level-1)
		Subgrid[lev]->NextGridNextLevel = NULL;
	      else
		Subgrid[lev]->NextGridNextLevel = Subgrid[lev+1];

	      // Create grid
	      for (dim = 0; dim < MetaData.TopGridRank; dim++)
		NumberOfSubgridDims[dim] += 2*NumberOfGhostZones;
	      Subgrid[lev]->GridData = new grid;
	      Subgrid[lev]->GridData->InheritProperties(TopGrid.GridData);
	      Subgrid[lev]->GridData->PrepareGrid(MetaData.TopGridRank, 
						  NumberOfSubgridDims,
						  ThisLeftEdge,
						  ThisRightEdge, 0);


	      if (Subgrid[lev]->GridData->MHDGalaxyDiskInitializeGrid(
				MHDGalaxyDiskNumberOfSpheres,
				MHDGalaxyDiskRadius,
				MHDGalaxyDiskAngularMomentum,
				MHDGalaxyDiskCoreRadius,
				MHDGalaxyDiskDensity,
				MHDGalaxyDiskTemperature,
				MHDGalaxyDiskMetallicity,
				MHDGalaxyDiskPosition,
				MHDGalaxyDiskVelocity,
				MHDGalaxyDiskFracKeplerianRot,
				MHDGalaxyDiskTurbulence,
				MHDGalaxyDiskDispersion,
				MHDGalaxyDiskCutOff,
				MHDGalaxyDiskAng1,
				MHDGalaxyDiskAng2,
				MHDGalaxyDiskNumShells,
				MHDGalaxyDiskType,
				MHDGalaxyDiskConstantPressure,
				MHDGalaxyDiskSmoothSurface,
				MHDGalaxyDiskSmoothRadius,
				MHDGalaxyDiskMagnFactor,
				MHDGalaxyDiskMagnEquipart,
				MHDGalaxyDiskHaloMass,
				MHDGalaxyDiskHaloCoreRadius,
				MHDGalaxyDiskHaloRadius,
				MHDGalaxyDiskUseParticles,
				MHDGalaxyDiskParticleMeanDensity,
				MHDGalaxyDiskUniformVelocity,
				MHDGalaxyDiskUseColour,
				MHDGalaxyDiskUseMetals,
				MHDGalaxyDiskInitialTemperature,
				MHDGalaxyDiskInitialDensity,
				MHDGalaxyDiskInitialMagnField,
				MHDGalaxyDiskPressureGradientType,
				lev-1) == FAIL) {
		ENZO_FAIL("Error in MHDGalaxyDiskInitializeGrid.");
	      }
	      
	    } // ENDIF zones exist
	  } // ENDFOR levels
	} // ENDIF max_level > 0
      } // ENDFOR spheres
    } // ENDIF MaxInitialLevel > 0

     Declare, initialize and fill out the LevelArray. 

    LevelHierarchyEntry *LevelArray[MAX_DEPTH_OF_HIERARCHY];
    for (level = 0; level < MAX_DEPTH_OF_HIERARCHY; level++)
      LevelArray[level] = NULL;
    AddLevel(LevelArray, &TopGrid, 0);

     Add levels to the maximum depth or until no new levels are created,
       and re-initialize the level after it is created. 

    if (MaxInitialLevel == 0) {
      for (level = 0; level < MaximumRefinementLevel; level++) {
	if (RebuildHierarchy(&MetaData, LevelArray, level) == FAIL) {
	  ENZO_FAIL("Error in RebuildHierarchy.");
	}
	if (LevelArray[level+1] == NULL)
	  break;
	LevelHierarchyEntry *Temp = LevelArray[level+1];
	while (Temp != NULL) {
		if (Temp->GridData->MHDGalaxyDiskInitializeGrid(
			MHDGalaxyDiskNumberOfSpheres,
			MHDGalaxyDiskRadius,
			MHDGalaxyDiskAngularMomentum,
			MHDGalaxyDiskCoreRadius,
			MHDGalaxyDiskDensity,
			MHDGalaxyDiskTemperature,
			MHDGalaxyDiskMetallicity,
			MHDGalaxyDiskPosition,
			MHDGalaxyDiskVelocity,
			MHDGalaxyDiskFracKeplerianRot,
			MHDGalaxyDiskTurbulence,
			MHDGalaxyDiskDispersion,
			MHDGalaxyDiskCutOff,
			MHDGalaxyDiskAng1,
			MHDGalaxyDiskAng2,
			MHDGalaxyDiskNumShells,
			MHDGalaxyDiskType,
			MHDGalaxyDiskConstantPressure,
			MHDGalaxyDiskSmoothSurface,
			MHDGalaxyDiskSmoothRadius,
			MHDGalaxyDiskMagnFactor,
			MHDGalaxyDiskMagnEquipart,
			MHDGalaxyDiskHaloMass,
			MHDGalaxyDiskHaloCoreRadius,
			MHDGalaxyDiskHaloRadius,
			MHDGalaxyDiskUseParticles,
			MHDGalaxyDiskParticleMeanDensity,
			MHDGalaxyDiskUniformVelocity,
			MHDGalaxyDiskUseColour,
			MHDGalaxyDiskUseMetals,
			MHDGalaxyDiskInitialTemperature,
			MHDGalaxyDiskInitialDensity,
			MHDGalaxyDiskInitialMagnField,
			MHDGalaxyDiskPressureGradientType,
			level+1) == FAIL) {
	    ENZO_FAIL("Error in MHDGalaxyDiskInitializeGrid.");
	  }
	  Temp = Temp->NextGridThisLevel;
	}
      } // end: loop over levels
    } // ENDELSE manually set refinement levels

     Loop back from the bottom, restoring the consistency among levels. 

    for (level = MaximumRefinementLevel; level > 0; level--) {
      LevelHierarchyEntry *Temp = LevelArray[level];
      while (Temp != NULL) {
	if (Temp->GridData->ProjectSolutionToParentGrid(
			      *LevelArray[level-1]->GridData) == FAIL) {
	  ENZO_FAIL("Error in grid->ProjectSolutionToParentGrid.");
	}
	Temp = Temp->NextGridThisLevel;
      }
    }

  } // end: if (MHDGalaxyDiskRefineAtStart) */

	/* set up field names and units */
	
	int count = 0;
	DataLabel[count++] = (char*) DensName;
	DataLabel[count++] = (char*) TEName;
	
	if (DualEnergyFormalism)
		DataLabel[count++] = (char*) GEName;
	
	DataLabel[count++] = (char*) Vel1Name;
	DataLabel[count++] = (char*) Vel2Name;
	DataLabel[count++] = (char*) Vel3Name;
	
	if(HydroMethod == MHD_RK)
	{
		DataLabel[count++] = (char*) Bfield1Name;
		DataLabel[count++] = (char*) Bfield2Name;
		DataLabel[count++] = (char*) Bfield3Name;
		DataLabel[count++] = (char*) PhiName;
	}
	if (MultiSpecies)
	{
		DataLabel[count++] = (char*) ElectronName;
		DataLabel[count++] = (char*) HIName;
		DataLabel[count++] = (char*) HIIName;
		DataLabel[count++] = (char*) HeIName;
		DataLabel[count++] = (char*) HeIIName;
		DataLabel[count++] = (char*) HeIIIName;
		if (MultiSpecies > 1)
		{
			DataLabel[count++] = (char*) HMName;
			DataLabel[count++] = (char*) H2IName;
			DataLabel[count++] = (char*) H2IIName;
		}
		if (MultiSpecies > 2)
		{
			DataLabel[count++] = (char*) DIName;
			DataLabel[count++] = (char*) DIIName;
			DataLabel[count++] = (char*) HDIName;
		}
	}  // if Multispecies
  if (MHDGalaxyDiskUseColour)
    DataLabel[count++] = (char*) ColourName;
  if (MHDGalaxyDiskUseMetals)
    DataLabel[count++] = (char*) MetalName;

  // S. Selg (08/2019): toggle output of gravitational potential
  if (WritePotential)
	  DataLabel[count++] = (char*) GPotName;

  for (i = 0; i < count; i++)
    DataUnits[i] = NULL;

  /* Write parameters to parameter output file */

  if (MyProcessorNumber == ROOT_PROCESSOR) {
    fprintf(Outfptr, "MHDGalaxyDiskNumberOfSpheres    = %"ISYM"\n",
	    MHDGalaxyDiskNumberOfSpheres);
    fprintf(Outfptr, "MHDGalaxyDiskRefineAtStart      = %"ISYM"\n",
	    MHDGalaxyDiskRefineAtStart);
    fprintf(Outfptr, "MHDGalaxyDiskUseParticles       = %"ISYM"\n",
	    MHDGalaxyDiskUseParticles);
    fprintf(Outfptr, "MHDGalaxyDiskUseColour          = %"ISYM"\n",
	    MHDGalaxyDiskUseColour);
    fprintf(Outfptr, "MHDGalaxyDiskUseMetals          = %"ISYM"\n",
	    MHDGalaxyDiskUseMetals);
    fprintf(Outfptr, "MHDGalaxyDiskInitialTemperature = %"FSYM"\n",
	    MHDGalaxyDiskInitialTemperature);
    fprintf(Outfptr, "MHDGalaxyDiskInitialDensity     = %"FSYM"\n",
	    MHDGalaxyDiskInitialDensity);
    fprintf(Outfptr, "MHDGalaxyDiskUniformVelocity    = %"FSYM" %"FSYM" %"FSYM"\n",
	    MHDGalaxyDiskUniformVelocity[0], MHDGalaxyDiskUniformVelocity[1],
	    MHDGalaxyDiskUniformVelocity[2]);
    for (sphere = 0; sphere < MHDGalaxyDiskNumberOfSpheres; sphere++) {
      fprintf(Outfptr, "MHDGalaxyDiskType[%"ISYM"] = %"ISYM"\n", sphere,
	      MHDGalaxyDiskType[sphere]);
      fprintf(Outfptr, "MHDGalaxyDiskRadius[%"ISYM"] = ", sphere);
	WriteListOfFloats(Outfptr, MetaData.TopGridRank,
                        MHDGalaxyDiskRadius[sphere]);
      fprintf(Outfptr, "MHDGalaxyDiskAngularMomentum[%"ISYM"] = ", sphere);
      WriteListOfFloats(Outfptr, MetaData.TopGridRank,
                        MHDGalaxyDiskAngularMomentum[sphere]);
      fprintf(Outfptr, "MHDGalaxyDiskCoreRadius[%"ISYM"] = %"GOUTSYM"\n", sphere,
	      MHDGalaxyDiskCoreRadius[sphere]);
      fprintf(Outfptr, "MHDGalaxyDiskDensity[%"ISYM"] = %"FSYM"\n", sphere,
	      MHDGalaxyDiskDensity[sphere]);
      fprintf(Outfptr, "MHDGalaxyDiskTemperature[%"ISYM"] = %"FSYM"\n", sphere,
	      MHDGalaxyDiskTemperature[sphere]);
      fprintf(Outfptr, "MHDGalaxyDiskMetallicity[%"ISYM"] = %"FSYM"\n", sphere,
	      MHDGalaxyDiskMetallicity[sphere]);
      fprintf(Outfptr, "MHDGalaxyDiskPosition[%"ISYM"] = ", sphere);
      WriteListOfFloats(Outfptr, MetaData.TopGridRank,
			MHDGalaxyDiskPosition[sphere]);
      fprintf(Outfptr, "MHDGalaxyDiskVelocity[%"ISYM"] = ", sphere);
      WriteListOfFloats(Outfptr, MetaData.TopGridRank,
			MHDGalaxyDiskVelocity[sphere]);
      fprintf(Outfptr, "MHDGalaxyDiskFracKeplerianRot[%"ISYM"] = %"GOUTSYM"\n", sphere,
              MHDGalaxyDiskFracKeplerianRot[sphere]);
      fprintf(Outfptr, "MHDGalaxyDiskTurbulence[%"ISYM"] = %"GOUTSYM"\n", sphere,
              MHDGalaxyDiskTurbulence[sphere]);
      fprintf(Outfptr, "MHDGalaxyDiskCutOff[%"ISYM"] = %"GOUTSYM"\n", sphere,
              MHDGalaxyDiskCutOff[sphere]);
      fprintf(Outfptr, "MHDGalaxyDiskAng1[%"ISYM"] = %"GOUTSYM"\n", sphere,
              MHDGalaxyDiskAng1[sphere]);
      fprintf(Outfptr, "MHDGalaxyDiskAng2[%"ISYM"] = %"GOUTSYM"\n", sphere,
              MHDGalaxyDiskAng2[sphere]);
      fprintf(Outfptr, "MHDGalaxyDiskNumShells[%"ISYM"] = %"ISYM"\n", sphere,
              MHDGalaxyDiskNumShells[sphere]);
      fprintf(Outfptr, "MHDGalaxyDiskConstantPressure[%"ISYM"] = %"ISYM"\n", sphere,
	      MHDGalaxyDiskConstantPressure[sphere]);
      fprintf(Outfptr, "MHDGalaxyDiskSmoothSurface[%"ISYM"] = %"ISYM"\n", sphere,
	      MHDGalaxyDiskSmoothSurface[sphere]);
      fprintf(Outfptr, "MHDGalaxyDiskSmoothRadius[%"ISYM"] = %"GOUTSYM"\n", sphere,
	      MHDGalaxyDiskSmoothRadius[sphere]);
    }
  }
  } // endif SetBaryonFields
  return SUCCESS;

}
