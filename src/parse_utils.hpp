#pragma once 

#include "mfem.hpp"
#include "sol/sol.hpp"
#include "yaml-cpp/yaml.h"

namespace parse 
{

// iterate over table and print to yaml map 
// tables in Lua do not have an order => this function 
// kind of annoyingly prints values in random order :( 
void PrintSolTable(YAML::Emitter &out, sol::table &table); 

// given a solver sol::table, construct an MFEM iterative solver object 
// with the specified options 
mfem::IterativeSolver *CreateIterativeSolver(sol::table &table, MPI_Comm comm); 

// set AMG options like max levels, num sweeps, etc via a lua table 
void SetAMGOptions(sol::table &table, mfem::HypreBoomerAMG &amg); 

}

// convenience function to print sol::table's into YAML maps 
// calls parse::PrintSolTable 
YAML::Emitter &operator<<(YAML::Emitter &out, sol::table &table); 