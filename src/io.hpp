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
mfem::IterativeSolver *CreateIterativeSolver(sol::table &table, std::optional<MPI_Comm> comm); 

mfem::Mesh CreateMesh(sol::table &table, YAML::Emitter &out, bool root=true); 
void SetMeshAttributes(mfem::Mesh &mesh, std::function<std::string(double,double,double)> f,
	const std::unordered_map<std::string,int> &map, bool root=true);
void SetMeshBdrAttributes(mfem::Mesh &mesh, std::function<std::string(double,double,double)> f, 
	const std::unordered_map<std::string,int> &map, bool root=true);

// set AMG options like max levels, num sweeps, etc via a lua table 
void SetAMGOptions(sol::table &table, mfem::HypreBoomerAMG &amg, bool root=true); 
#ifdef MFEM_USE_SUPERLU
void SetSuperLUOptions(sol::table &table, mfem::SuperLUSolver &slu, bool root=true); 
#endif

struct SundialsUserCallbackData {
	YAML::Emitter *out; 
	const MomentMethodFixedPointOperator *G; 
	const mfem::IterativeSolver * const inner_solver; 
	mfem::Array<int> inner_it; 
	mfem::Array<double> sweep_time, moment_time; 
	SundialsUserCallbackData(YAML::Emitter &out, const MomentMethodFixedPointOperator &G, 
		const mfem::IterativeSolver * const isolver) 
		: out(&out), G(&G), inner_solver(isolver)
	{

	}
};

bool ParseKINSOLMessage(char *msg, int &it, double &norm);
void SundialsCallbackFunction(const char *module, const char *function, char *msg, void *user_data); 
void SundialsErrorFunction(int error_code, const char *module, const char *function, char *msg, void *user_data); 

DiffusionBoundaryConditionType GetDiffusionBCType(std::string type); 

template<typename T>
void PrintOptionsAbort(std::string key, T res, std::initializer_list<T> options)
{
	std::stringstream ss; 
	ss << "\"" << res << "\" not a valid input for key \"" << key << "\". Valid options are:";
	for (const auto &opt : options) {
		ss << " \"" << opt << "\""; 
	}
	MFEM_ABORT(ss.str()); 
}

template<typename T>
void ValidateOption(std::string key, T res, std::initializer_list<T> options, bool root=true)
{
	bool valid = false; 
	for (const auto &opt : options) {
		if (res == opt) valid = true; 
	}
	if (!valid and root) {
		PrintOptionsAbort(key, res, options); 
	}	
}

template<>
void ValidateOption<const char*>(std::string key, const char *res, std::initializer_list<const char*> options, bool root);

template<typename T> 
T GetAndValidateOption(sol::table &table, std::string key, std::initializer_list<T> options, T def, bool root=true) 
{
	sol::optional<T> avail = table[key]; 
	if (avail) {
		const T res = avail.value(); 
		ValidateOption<T>(key, res, options, root); 
		return res; 	
	}
	else {
		return def; 
	}
}

std::string ResolveRelativePath(std::string path); 

} // end namespace io 

// convenience function to print sol::table's into YAML maps 
// calls parse::PrintSolTable 
YAML::Emitter &operator<<(YAML::Emitter &out, sol::table &table); 