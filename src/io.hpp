#pragma once 

#include "mfem.hpp"
#include "sol/sol.hpp"
#include "yaml-cpp/yaml.h"
#include "transport_op.hpp"

namespace io 
{

// 6D specification of phase-space functions from Lua 
using LuaPhaseFunction = std::function<double(double,double,double,double,double,double)>; 

// convert seconds to days-hours:minutes:seconds format 
std::string FormatTimeString(double time); 
// print git commit, branch, tag information 
void PrintGitString(YAML::Emitter &out); 
std::string FormatScientific(double val, int precision=3);

// iterate over table and print to yaml map 
// tables in Lua do not have an order => this function 
// kind of annoyingly prints values in random order :( 
void PrintSolTable(YAML::Emitter &out, sol::table &table); 
template<typename T>
void PrintMap(YAML::Emitter &out, const T &map)
{
	out << YAML::BeginMap;
	for (const auto &it : map) {
		out << YAML::Key << it.first << YAML::Value << it.second;
	}
	out << YAML::EndMap;
}
template<typename T>
void PrintTimingMap(YAML::Emitter &out, const T &map)
{
	out << YAML::BeginMap;
	for (const auto &it : map) {
		out << YAML::Key << it.first << YAML::Value << FormatTimeString(it.second);
	}
	out << YAML::EndMap;
}
template<typename T>
void PrintArray(YAML::Emitter &out, const mfem::Array<T> &x)
{
	out << YAML::BeginSeq; 
	for (const auto &it : x) {
		out << it;
	}
	out << YAML::EndSeq;
}

mfem::DataCollection *CreateDataCollection(std::string type, std::string output_root, 
	mfem::Mesh &mesh, bool root);

// given a solver sol::table, construct an MFEM solver object 
// with the specified options 
mfem::Solver *CreateSolver(sol::table &table, std::optional<MPI_Comm> comm);
mfem::IterativeSolver *CreateIterativeSolver(sol::table &table, std::optional<MPI_Comm> comm);
// set common iterative solver settings such as 
// tolerances, max iterations, print level, iterative mode, etc 
void SetIterativeSolverOptions(sol::table &table, mfem::IterativeSolver &solver); 

// --- helper functions for creating meshes --- 
// creates a mesh from lua input either from a mesh file 
// or from specification of cartesian domain 
mfem::Mesh CreateMesh(sol::table &table, YAML::Emitter &out, bool root=true); 
// set element attribute according to a provided material map function 
void SetMeshAttributes(mfem::Mesh &mesh, std::function<std::string(double,double,double)> f,
	const std::unordered_map<std::string,int> &map, bool root=true);
// set bdr element attribute according to a provide boundary condition map 
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

// DiffusionBoundaryConditionType GetDiffusionBCType(std::string type);  

// --- functions to validate input --- 
// print an error message if input is not valid 
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

// check if res is in options 
// key is an identifier in the abort message 
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

// combine getting option from lua table and checking for valid input 
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

// given a relative path, convert it to absolute 
std::string ResolveRelativePath(std::string path); 

} // end namespace io 

// convenience function to print sol::table's into YAML maps 
// calls io::PrintSolTable 
YAML::Emitter &operator<<(YAML::Emitter &out, sol::table &table); 

// print std::map-type objects to YAML map
// calls io::PrintMap
template<typename T>
YAML::Emitter &operator<<(YAML::Emitter &out, const T &map)
{
	io::PrintMap(out, map);
	return out;
}

template<typename T>
YAML::Emitter &operator<<(YAML::Emitter &out, const mfem::Array<T> &x)
{
	io::PrintArray(out, x);
	return out;
}