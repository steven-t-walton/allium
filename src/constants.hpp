#pragma once 

#include <numbers>
#include <cmath>

namespace constants {

// values taken from NIST (https://physics.nist.gov/cuu/Constants/) 
// and converted to CGS units 

namespace internal {
	constexpr double Joule_to_erg = 1e7; // 1e7 ergs / Joule 
	constexpr double Joule_to_eV = 1.0/1.602176634e-19; // value eV / Joule 
	constexpr double erg_to_eV = Joule_to_eV / Joule_to_erg; // value eV per erg 
}

constexpr double pi = std::numbers::pi; 
constexpr double SpeedOfLight = 2.99792458e10; // cm/s 
constexpr double Planck = 6.62607015e-27; // erg s  
constexpr double Boltzmann = 1.380649e-16; // erg/K 
// black body thermal emission is StefanBoltzmann * T^4 (includes factor of speed of light)
// erg/cm^2/s/eV^4 
constexpr double StefanBoltzmann = 8.0*pow(pi, 5)/(15.0*pow(SpeedOfLight, 2)*pow(Planck, 3))/pow(internal::erg_to_eV,4); 
// the "a" in black body emission ac T^4 
constexpr double RadiationConstant = StefanBoltzmann / SpeedOfLight; 

} // end namespace constants 