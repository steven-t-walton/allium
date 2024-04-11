#pragma once 

#include <numbers>

namespace constants {

// values taken from NIST (https://physics.nist.gov/cuu/Constants/) 
// and converted to CGS units 
constexpr double pi = std::numbers::pi; 
constexpr double SpeedOfLight = 2.99792458e10; // cm/s 
constexpr double Boltzmann = 1.380649e-16; // erg/K 
constexpr double Planck = 6.62607015e-27; // erg s  
// black body thermal emission is Stefan * T^4 
// constexpr double Stefan = 2.0*pow(pi, 5)*pow(Boltzmann, 4)/(15.0*pow(SpeedOfLight, 2)*pow(Planck, 3)); // erg/cm^2/s/K^4 
constexpr double Stefan = 137.0 * SpeedOfLight; 

// constexpr double SpeedOfLight = 1.0; 
// constexpr double Boltzmann = 1.380649e-16; // erg/K 
// constexpr double Planck = 6.62607015e-27; // erg s  
// // black body thermal emission is Stefan * T^4 
// // constexpr double Stefan = 2.0*pow(pi, 5)*pow(Boltzmann, 4)/(15.0*pow(SpeedOfLight, 2)*pow(Planck, 3)); // erg/cm^2/s/K^4 
// constexpr double Stefan = 1.0; 

} // end namespace constants 