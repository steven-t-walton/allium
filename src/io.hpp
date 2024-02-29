#pragma once 

#include "mfem.hpp"
#include "sol/sol.hpp"
#include "yaml-cpp/yaml.h"
#include "transport_op.hpp"
#include "block_smm_op.hpp"

namespace io 
{

std::string FormatTimeString(double time); 

// iterate over table and print to yaml map 
// tables in Lua do not have an order => this function 
// kind of annoyingly prints values in random order :( 
void PrintSolTable(YAML::Emitter &out, sol::table &table); 

// given a solver sol::table, construct an MFEM iterative solver object 
// with the specified options 
mfem::IterativeSolver *CreateIterativeSolver(sol::table &table, MPI_Comm comm); 

// set AMG options like max levels, num sweeps, etc via a lua table 
void SetAMGOptions(sol::table &table, mfem::HypreBoomerAMG &amg); 

struct SundialsUserCallbackData {
	YAML::Emitter *out; 
	const MomentMethodFixedPointOperator *G; 
	const mfem::IterativeSolver * const inner_solver; 
	mfem::Array<int> inner_it; 
	SundialsUserCallbackData(YAML::Emitter &out, const MomentMethodFixedPointOperator &G, 
		const mfem::IterativeSolver * const isolver) 
		: out(&out), G(&G), inner_solver(isolver)
	{

	}
};

void SundialsCallbackFunction(const char *module, const char *function, char *msg, void *user_data); 
void SundialsErrorFunction(int error_code, const char *module, const char *function, char *msg, void *user_data); 

DiffusionBoundaryConditionType GetDiffusionBCType(sol::table &table, std::string key, std::string defaulted); 

template<typename T> 
T GetAndValidateOption(sol::table &table, std::string key, std::initializer_list<T> options, T def) 
{
	sol::optional<T> avail = table[key]; 
	if (avail) {
		const T res = avail.value(); 
		bool valid = false; 
		for (const auto &opt : options) {
			if (res == opt) valid = true; 
		}
		if (!valid) {
			std::stringstream ss; 
			ss << "\"" << res << "\" not a valid input for key \"" << key << "\". Valid options are:";
			for (const auto &opt : options) {
				ss << " \"" << opt << "\""; 
			}
			MFEM_ABORT(ss.str()); 
		}
		return res; 	
	}
	else {
		return def; 
	}
}

} // end namespace io 

// convenience function to print sol::table's into YAML maps 
// calls parse::PrintSolTable 
YAML::Emitter &operator<<(YAML::Emitter &out, sol::table &table); 