#include "parse_utils.hpp"
#include <algorithm>

namespace parse 
{

void PrintSolTable(YAML::Emitter &out, sol::table &table) 
{
	out << YAML::BeginMap; 
	for (const auto &it : table) {
		out << YAML::Key << it.first.as<std::string>() << YAML::Value; 
		const auto &val = it.second; 
		if (val.get_type() == sol::type::number) { out << val.as<double>(); }
		else if (val.get_type() == sol::type::boolean) { out << val.as<bool>(); }
		else if (val.get_type() == sol::type::string) { out << val.as<std::string>(); }
		else if (val.get_type() == sol::type::function) { out << "function"; }
		else if (val.get_type() == sol::type::lua_nil) { out << "not specified"; }
		else if (val.get_type() == sol::type::table) {
			sol::table table_nested = val.as<sol::table>(); 
			PrintSolTable(out, table_nested); 
		}
		else { MFEM_ABORT("lua type not understood"); }
	}
	out << YAML::EndMap; 
}

mfem::IterativeSolver *CreateIterativeSolver(sol::table &table, MPI_Comm comm) 
{
	mfem::IterativeSolver *s = nullptr; 
	std::string type = table["type"]; 
	std::transform(type.begin(), type.end(), type.begin(), ::tolower); 
	if (type == "cg" or type == "conjugate gradient") {
		s = new mfem::CGSolver(comm); 
	} 

	else if (type == "gmres") {
		auto *gmres = new mfem::GMRESSolver(comm); 
		sol::optional<int> kdim_avail = table["kdim"]; 
		if (kdim_avail) { gmres->SetKDim(kdim_avail.value()); }
		s = gmres; 
	}

	else if (type == "fgmres") {
		auto *fgmres = new mfem::FGMRESSolver(comm); 
		sol::optional<int> kdim_avail = table["kdim"]; 
		if (kdim_avail) { fgmres->SetKDim(kdim_avail.value()); }
		// mfem's default 
		else { table["kdim"] = 50; }
		s = fgmres; 		
	}

	else if (type == "sli") {
		s = new mfem::SLISolver(comm); 
		// if reltol not set SLISolver defaults to 
		// fixed number of iterations 
		s->SetRelTol(1e-15); 
	}

	else if (type == "bicg" or type == "bicgstab") {
		s = new mfem::BiCGSTABSolver(comm); 
	}

	else if (type == "direct" or type == "superlu") {
		return nullptr; 
	}

	else {
		MFEM_ABORT("solver type " << type << " not supported"); 
	}

	// load generic iterative solver options 
	int print_level = table["print_level"].get_or(0); 
	s->SetPrintLevel(print_level); 
	sol::optional<double> abstol = table["abstol"]; 
	sol::optional<double> reltol = table["reltol"]; 
	if (!abstol and !reltol) {
		MFEM_ABORT("must specify one of \"abstol\" or \"reltol\""); 
	}
	if (abstol) { s->SetAbsTol(abstol.value()); }
	else { table["abstol"] = 0.0; }
	if (reltol) { s->SetRelTol(reltol.value()); }
	else { table["reltol"] = 0.0; }
	int maxit = table["max_iter"].get_or(50);
	s->SetMaxIter(maxit);  
	table["max_iter"] = maxit; 
	bool iterative_mode = table["iterative_mode"].get_or(false);
	s->iterative_mode = iterative_mode; 
	table["iterative_mode"] = iterative_mode; 
	return s; 
}

void SetAMGOptions(sol::table &table, mfem::HypreBoomerAMG &amg) 
{
	sol::optional<int> pre_sweeps = table["pre_sweeps"]; 
	sol::optional<int> post_sweeps = table["post_sweeps"]; 
	sol::optional<int> max_levels = table["max_levels"]; 
	sol::optional<int> relax_type = table["relax_type"]; 
	sol::optional<int> cycle_type = table["cycle_type"]; 
	sol::optional<int> agg_coarsen = table["aggressive_coarsening"]; 
	sol::optional<int> interpolation = table["interpolation"]; 
	sol::optional<int> coarsening = table["coarsening"]; 
	sol::optional<int> strength_thresh = table["strength_threshold"]; 
	if (pre_sweeps or post_sweeps) {
		int pre = 1, post = 1; 
		if (pre_sweeps) pre = pre_sweeps.value(); 
		if (post_sweeps) post = post_sweeps.value(); 
		amg.SetCycleNumSweeps(pre, post); 
	}
	if (max_levels) {
		amg.SetMaxLevels(max_levels.value()); 
	}
	if (relax_type) {
		amg.SetRelaxType(relax_type.value()); 
	}
	if (cycle_type) {
		amg.SetCycleType(cycle_type.value()); 
	}
	if (agg_coarsen) {
		amg.SetAggressiveCoarsening(agg_coarsen.value()); 
	}
	if (interpolation) {
		amg.SetInterpolation(interpolation.value()); 
	}
	if (coarsening) {
		amg.SetCoarsening(coarsening.value()); 
	}
	if (strength_thresh) {
		amg.SetStrengthThresh(strength_thresh.value()); 
	}
}

}

YAML::Emitter &operator<<(YAML::Emitter &out, sol::table &table) 
{ 
	parse::PrintSolTable(out, table); 
	return out; 
}