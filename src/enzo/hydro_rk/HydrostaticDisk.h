/***********************************************************************
/
/  HYDROSTATIC DISK CLASS
/
/  written by: Kai Rodenbeck
/  date:       2015
/  modified1:  Wolfram Schmidt, January 2018
/  modified2:  Simon Selg, August 2019
/
/  PURPOSE: Iterative computation of axisymmetric disk galaxy 
/     in hydrostatic equilibrium
/
************************************************************************/

struct zone {
	double r, z;                      // radial and vertical coordinates
	double phi, dphi, phi_int, e_mag; // potential and magnetic field
	double rho, ln_rho, rho_0;        // density
	double v_sqr, c_s;                // velocity and speed of sound (temperature)
};


class hydrostatic_disk
{
private:
	bool equipart;
	int pressureGradientType; // S.Selg; see Wang+2010, Eq. 30
	int N_r, N_z;

	static const double eight_pi = 8.0*pi;

	double dr, dz;
	double r_sc,z_sc;
	double sigma;
	double magn_fract;
	double rho_min;
	double halo_mass, halo_scale;

	zone **data;

public:

	// default constructor
	hydrostatic_disk() 
	{
		equipart = false;	
		pressureGradientType = 0;	
		N_r=0;
		N_z=0;
		dr=0.0;
		dz=0.0;
		r_sc=0.0;
		z_sc=0.0;
		sigma=0.0;
		magn_fract=0.0;
		rho_min=0.0;
	};

	// parameterized constructor
	hydrostatic_disk(int N1, int N2, double delta1, double delta2, double scale1, double scale2, double vel, double sound_speed)
	{
		equipart = false;
		pressureGradientType = 0;
		N_r=N1;
		N_z=N2;
		dr=delta1;
		dz=delta2;
		r_sc=scale1;
		z_sc=scale2;
		sigma=vel;
		magn_fract=0.0;
		rho_min=1.0e-35;

		data = new zone*[N_r];

		for(int i=0;i<N_r;i++)
		{
			data[i] = new zone[N_z];

			for(int j=0;j<N_z;j++)
			{
				data[i][j].r = i*dr; data[i][j].z = j*dz;
				data[i][j].dphi = data[i][j].phi = data[i][j].phi_int = data[i][j].e_mag = 0.0;
				data[i][j].rho = data[i][j].ln_rho = data[i][j].rho_0 = data[i][j].v_sqr = 0.0;
				data[i][j].c_s = sound_speed;
			}
			
			data[i][0].rho = sigma*exp(-data[i][0].r/r_sc)/z_sc;
			data[i][0].rho_0 = data[i][0].rho;
		}
		
		if(MyProcessorNumber == ROOT_PROCESSOR)
			printf("\nGrid Settings:\nN_r=%i\tN_z=%i\nr_sc=%e\t%e\n%e\t%e\nSigma=%e\trho_o(0)=%e\tc_s = %e\n",
			       N_r, N_z, r_sc, dr, z_sc, dz, sigma, data[0][0].rho, sound_speed);		
	}

/*
        // to be fixed: explicit destructor causes crash 
	~hydrostatic_disk()
	{
		for(int i=0; i<N_r; i++)
		{
			if (data[i] != NULL) delete[] data[i];
		}
		if (data != NULL) delete[] data;
	}
*/

	// deallocate data
	void free_data()
	{
		for(int i=0; i<N_r; i++)
		{
			delete[] data[i];
		}
		delete[] data;
	}

	/*
	 methods to iterate disk structure
	 */

	double integrate();

	void rotation_curve();

	/*
	 methods to access parameters
	 */

 	void set_magn_equipart()
	{
		equipart = true;
	}

	void set_pressureGradientType(int val)
	{
		pressureGradientType = val;
	}

 	void set_magn_fract(double val)
	{
		magn_fract = val;
	}

 	void set_halo_mass(double val)
	{
		halo_mass = val;
	}

 	void set_halo_scale(double val)
	{
		halo_scale = val;
	}

 	double get_halo_mass()
	{
		return halo_mass;
	}

 	double get_halo_scale()
	{
		return halo_scale;
	}

        /*
         interpolation of zone values to given coordinates
         */

	double intpl_rho(double r, double z)
	{
		int i=r/dr;
		int j=z/dz;
		
		double f_r=r/dr-i;
		double f_z=z/dz-j;
		
		// outside domain
		if (i<0 || i>N_r-2 || j<0 || j>N_z-2)
			return -1;
                // interpolate value inside domain
		else
			return ((1.0-f_r)*(1.0-f_z)*data[i][j].rho + 
			        (1.0-f_r)*f_z*data[i][j+1].rho +
			        f_r*(1.0-f_z)*data[i+1][j].rho + 
				f_r*f_z*data[i+1][j+1].rho);
	}

	double intpl_v_sqr(double r, double z)
	{
		int i=r/dr;
		int j=z/dz;
		
		double f_r=r/dr-i;
		double f_z=z/dz-j;
		
		// outside domain
		if (i<0 || i>N_r-2 || j<0 || j>N_z-2)
			return -1;
                // interpolate value inside domain
		else
			return ((1.0-f_r)*(1.0-f_z)*data[i][j].v_sqr +
			        (1.0-f_r)*f_z*data[i][j+1].v_sqr +
			        f_r*(1.0-f_z)*data[i+1][j].v_sqr +
			        f_r*f_z*data[i+1][j+1].v_sqr);
	}

        /*
         computation of halo potential and velocity for given coordinates
         */

	double halo_pot(double r, double z)
	{
		// Hernquist (1990), Eq. 2
		return -GravConst * halo_mass / (sqrt(r*r+z*z) + halo_scale);
	}

	double halo_vel_sq(double r, double z)
	{
		// Hernquist (1990), Eq. 16
		return GravConst * halo_mass *sqrt(r*r+z*z) /
			pow(sqrt(r*r+z*z) + halo_scale, 2.0);
	}

        /*
         computation of variables and derivatives related to magnetic field for given coordinates
         */

	double B(double r, double z)
	{
		return magn_fract*r/r_sc*exp(-r/r_sc)*exp(-z/z_sc);
	}

	double pd_B_r(double r, double z)
	{
		return magn_fract*(1.0-r/r_sc)*exp(-r/r_sc)*exp(-z/z_sc)/kpc_cm;
	}

	double pd_B_z(double r, double z)
	{
		return magn_fract*r/r_sc*exp(-r/r_sc)*(-1.0/z_sc)*exp(-z/z_sc)/kpc_cm;
	}

	double Emag(double r, double z)
	{
		return pow(B(r, z), 2.0)/eight_pi;
	}

	double pd_Emag_r(double r, double z)
	{
		return 2.0*B(r, z)*pd_B_r(r, z)/eight_pi;
	}

	double pd_Emag_z(double r, double z)
	{
		return 2.0*B(r, z)*pd_B_z(r, z)/eight_pi;
	}

private:

	/*
	 auxiliary methods
	 */

	void fill_zone(zone *ptr);
};

