#pragma once 

#include "mfem.hpp"
#include "sol/sol.hpp"
#include "yaml-cpp/yaml.h"
#include "smm_op.hpp"

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

}

// convenience function to print sol::table's into YAML maps 
// calls parse::PrintSolTable 
YAML::Emitter &operator<<(YAML::Emitter &out, sol::table &table); 