/***********************************************************************
/
/  HYDROSTATIC DISK CLASS
/
/  written by: Kai Rodenbeck
/  date:       2015
/  modified1:  Wolfram Schmidt, January 2018
/
/  PURPOSE: Member functions to compute disk structure
/
************************************************************************/

#include "preincludes.h"
#include "macros_and_parameters.h"
#include "typedefs.h"
#include "global_data.h"
#include "phys_constants.h"
#include "HydrostaticDisk.h"

// auxiliary function to iteratively fill zones from low to high with step calculated from center values
void hydrostatic_disk::fill_zone(zone *ptr)
{
	zone *low = ptr;
        zone *center = ptr; // Euler forward
	zone *high = ptr+1;

	double h_p = halo_pot(high->r, high->z) - halo_pot(high->r, 0.0);

	high->dphi = low->dphi + dz*4.0*pi*GravConst*center->rho;
	high->phi  = low->phi  + dz*center->dphi;

	if(equipart)
	{
		high->rho     = fmax(rho_min, low->rho_0 * exp(-(high->phi+h_p)/((1.0+magn_fract)*pow(high->c_s, 2.0))));
		high->phi_int = low->phi_int + dz*2.0*exp(-(high->phi+h_p)/((1.0+magn_fract)*pow(high->c_s, 2.0)));
		high->rho_0   = low->rho_0;
	}
	else
	{
		high->ln_rho  = low->ln_rho  + dz*(-(center->dphi + pd_Emag_z(center->r, center->z)/center->rho)/pow(center->c_s, 2.0));
		high->e_mag   = low->e_mag   + dz*pd_Emag_z(center->r, center->z)/center->rho;
		high->phi_int = low->phi_int + dz*2.0*exp(-(center->phi+h_p + center->e_mag)/pow(center->c_s, 2.0));
		
		high->rho_0 = low->rho_0;
		high->rho   = fmax(exp(high->ln_rho)*high->rho_0, rho_min);
	}
}

// iterative function to integrate potential
// returns relative deviation of old and new potential
double hydrostatic_disk::integrate()
{
	double sum_old, sum_new;

	sum_old = 0.0;

	for(int i=1; i<N_r; i++)
	{
		for(int j=0; j<N_z; j++)
			sum_old += data[i][j].phi;
	}

	for(int i=0; i<N_r; i++)
	{
		zone *zp = data[i];

		for(int j=1; j<N_z; j++)
			fill_zone(zp++);
			
		data[i][0].rho   = fmax(sigma*exp(-data[i][0].r/r_sc)/data[i][N_z-1].phi_int, rho_min);
		data[i][0].rho_0 = data[i][0].rho;
	}

	sum_new = 0.0;

	for(int i=1; i<N_r; i++)
	{
		for(int j=0; j<N_z; j++)
			sum_new += data[i][j].phi;
	}

	return fabs(sum_new-sum_old)/sum_old;
}

// function to compute rotation curve from gas density and potential
void hydrostatic_disk::rotation_curve()
{
	for(int i=1; i<N_r; i++)
	{
		for(int j=0; j<N_z; j++)
			if(equipart)
				data[i][j].v_sqr = 
					data[i][j].r/data[i][j].rho*(1.0+magn_fract)*pow(data[i][j].c_s, 2.0)*(data[i][j].rho - data[i-1][j].rho)/dr +
					(data[i][j].phi - data[i-1][j].phi)/dr;
					
			else
				data[i][j].v_sqr = 
					data[i][j].r/data[i][j].rho*(pd_Emag_z(data[i][j].r, data[i][j].z) + 
					                             pow(data[i][j].c_s, 2.0)*(data[i][j].rho - data[i-1][j].rho)/dr) +
					(data[i][j].phi - data[i-1][j].phi)/dr;
	}
}

