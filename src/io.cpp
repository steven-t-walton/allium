#include "io.hpp"
#include "linalg.hpp"
#include <algorithm>
#include <regex>

namespace io 
{

std::string FormatTimeString(double time) {
	std::stringstream ss; 
	if (time < 60) {
		ss << std::fixed << std::setprecision(3) << time; 
		return ss.str(); 
	}
	double remainder = std::fmod(time, 3600*24); 
	int days = time / (3600*24); 
	int hours = remainder / 3600; 
	remainder = std::fmod(time, 3600); 
	int minutes = remainder / 60; 
	auto seconds = std::fmod(remainder, 60); 
	if (days > 0) {
		ss << days << "-"; 
	}
	if (hours > 0 or days > 0) {
		ss << std::setfill('0') << std::setw(2) << hours << ":"; 
	} 
	if (minutes > 0 or hours > 0 or days > 0) {
		ss << std::setfill('0') << std::setw(2) << minutes << ":";
	}
	ss << std::setfill('0') << std::fixed << std::setprecision(3) << std::setw(6) << seconds; 		
	return ss.str(); 
}

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
		else { table["kdim"] = 50; }
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
		s = new SLISolver(comm); 
	}

	else if (type == "bicg" or type == "bicgstab") {
		s = new mfem::BiCGSTABSolver(comm); 
	}

	else if (type == "direct" or type == "superlu") {
		return nullptr; 
	}

	else if (type == "fixed point" or type == "fp") {
		s = new FixedPointIterationSolver(comm);
	}

	else if (type == "kinsol") {
	#ifdef MFEM_USE_SUNDIALS 
		auto *kn = new mfem::KINSolver(comm, KIN_FP, false); 
		int kdim = table["kdim"].get_or(1); 
		kn->SetMAA(kdim); 
		table["kdim"] = kdim; 
		s = kn; 
		kn->SetPrintLevel(1); // default to 1 to call callback function 
	#else 
		MFEM_ABORT("MFEM not built with sundials"); 
	#endif
	}

	else {
		MFEM_ABORT("solver type " << type << " not supported"); 
	}

	// load generic iterative solver options 
	int print_level = table["print_level"].get_or(0); 
	s->SetPrintLevel( (dynamic_cast<mfem::KINSolver*>(s) ? 1 : print_level)); 
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

void SetAMGOptions(sol::table &table, mfem::HypreBoomerAMG &amg, bool root) 
{
	for (const auto &it : table) {
		std::string key = it.first.as<std::string>();
		ValidateOption<std::string>("BoomerAMG", key, 
			{"max_iter", "relax_sweeps", "max_levels", "relax_type", "cycle_type", 
			 "aggressive_coarsening", "interpolation", "coarsening", "strength_threshold"},
			root
		); 
	}
	sol::optional<int> max_iter = table["max_iter"]; 
	sol::optional<int> sweeps = table["relax_sweeps"]; 
	sol::optional<int> max_levels = table["max_levels"]; 
	sol::optional<int> relax_type = table["relax_type"]; 
	sol::optional<int> cycle_type = table["cycle_type"]; 
	sol::optional<int> agg_coarsen = table["aggressive_coarsening"]; 
	sol::optional<int> interpolation = table["interpolation"]; 
	sol::optional<int> coarsening = table["coarsening"]; 
	sol::optional<double> strength_thresh = table["strength_threshold"]; 
	if (max_iter) {
		amg.SetMaxIter(max_iter.value()); 
	}
	if (sweeps) {
		HYPRE_BoomerAMGSetNumSweeps(amg, sweeps.value());	
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

#ifdef MFEM_USE_SUPERLU
void SetSuperLUOptions(sol::table &table, mfem::SuperLUSolver &slu, bool root)
{
	for (const auto &it : table) {
		auto key = it.first.as<std::string>(); 
		ValidateOption<std::string>("superlu", key, 
			{"symmetric_pattern", "iterative_refine", "equilibriate", 
			 "column_permutation", "ParSymbFact", "print_statistics"}, 
			root); 
	}
	sol::optional<bool> sym = table["symmetric_pattern"]; 
	sol::optional<std::string> itref = table["iterative_refine"]; 
	sol::optional<bool> equil = table["equilibriate"]; 
	sol::optional<std::string> colperm = table["column_permutation"]; 
	sol::optional<bool> parsymbfact = table["ParSymbFact"]; 
	bool print = table["print_statistics"].get_or(false); 

	if (sym) 
		slu.SetSymmetricPattern(sym.value()); 
	if (itref) {
		std::string val = itref.value(); 
		ValidateOption<std::string>("superlu::iterative_refine", val, 
			{"none", "single", "double"}, root); 
		if (val == "none") {
			slu.SetIterativeRefine(mfem::superlu::NOREFINE); 
		} else if (val == "single") {
			slu.SetIterativeRefine(mfem::superlu::SLU_SINGLE); 
		} else if (val == "double") {
			slu.SetIterativeRefine(mfem::superlu::SLU_DOUBLE); 
		}
	} 
	// default to no iterative refine 
	else {
		slu.SetIterativeRefine(mfem::superlu::NOREFINE); 
	}
	if (equil) {
		slu.SetEquilibriate(equil.value()); 
	}
	if (colperm) {
		std::string val = colperm.value(); 
		ValidateOption<std::string>("superlu::colperm", val, 
			{"natural", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD", "METIS_AT_PLUS_A", "PARMETIS"}, root); 
		mfem::superlu::ColPerm r; 
		if (val == "natural") {
			r = mfem::superlu::NATURAL; 
		} else if (val == "MMD_ATA") {
			r = mfem::superlu::MMD_ATA; 
		} else if (val == "MMD_AT_PLUS_A") {
			r = mfem::superlu::MMD_AT_PLUS_A; 
		} else if (val == "COLAMD") {
			r = mfem::superlu::COLAMD; 
		} else if (val == "METIS_AT_PLUS_A") {
			r = mfem::superlu::METIS_AT_PLUS_A; 
		} else if (val == "PARMETIS") {
			r = mfem::superlu::PARMETIS; 
		}
		slu.SetColumnPermutation(r); 
	}
	if (parsymbfact) {
		slu.SetParSymbFact(parsymbfact.value()); 
	}
	slu.SetPrintStatistics(print); 
}
#endif

void SundialsCallbackFunction(const char *module, const char *function, char *msg, void *user_data) 
{
	auto *data = static_cast<SundialsUserCallbackData*>(user_data); 
	auto &out = *data->out; 
	MFEM_ASSERT(data, "sundials user data not set properly"); 
	std::regex nni_reg("nni =\\s+([0-9]+)\\s"); 
	std::cmatch nni_match; 
	if (std::regex_search(msg, nni_match, nni_reg)) {
		auto &G = *data->G; 
		// fnorm =      0.0005730787351824196
		std::regex norm_reg("fnorm =\\s+(\\S+)"); 
		std::cmatch norm_match; 
		out << YAML::BeginMap; 
		out << YAML::Key << "it" << YAML::Value << nni_match[1].str(); 
		if (std::regex_search(msg, norm_match, norm_reg)) {
			double norm = std::stod(norm_match[1].str());
			std::stringstream ss; 
			ss << std::setprecision(3) << std::scientific << norm;  
			out << YAML::Key << "norm" << YAML::Value << ss.str(); 
		}
		if (data->inner_solver) {
			out << YAML::Key << "inner solver" << YAML::Value << YAML::BeginMap; 
				out << YAML::Key << "it" << YAML::Value << data->inner_solver->GetNumIterations(); 
				std::stringstream ss; 
				ss << std::scientific << std::setprecision(3) << std::scientific << data->inner_solver->GetFinalNorm(); 
				out << YAML::Key << "norm" << YAML::Value << ss.str(); 
			out << YAML::EndMap; 		
			data->inner_it.Append(data->inner_solver->GetNumIterations());		
		}
		const auto total_time = G.TotalTimer().RealTime(); 
		const auto sweep_time = G.SweepTimer().RealTime(); 
		const auto moment_time = G.MomentTimer().RealTime(); 
		data->sweep_time.Append(sweep_time); 
		data->moment_time.Append(moment_time); 
		out << YAML::Key << "timings" << YAML::BeginMap; 
		out << YAML::Key << "total" << YAML::Value << FormatTimeString(total_time); 
		out << YAML::Key << "sweep" << YAML::Value << FormatTimeString(sweep_time); 
		out << YAML::Key << "moment" << YAML::Value << FormatTimeString(moment_time); 
		out << YAML::EndMap; 
		out << YAML::EndMap; 
		out << YAML::Newline; 		
	}
	else {
		std::regex start_reg("scsteptol"); 
		std::cmatch start_match; 
		if (std::regex_search(msg, start_match, start_reg)) {
			out << YAML::Key << "transport iterations" << YAML::Value << YAML::BeginSeq; 
		}

		std::regex end_reg("Return"); 
		std::cmatch end_match; 
		if (std::regex_search(msg, end_match, end_reg)) {
			out << YAML::EndSeq; 
		}
	}
}

void SundialsErrorFunction(int error_code, const char *module, const char *function, char *msg, void *user_data)
{
	
} 

template<>
void ValidateOption<const char*>(std::string key, const char *res, std::initializer_list<const char*> options, bool root)
{
	bool valid = false; 
	for (const auto &opt : options) {
		if (std::string_view(opt) == std::string_view(res)) valid = true; 
	}
	if (!valid and root) {
		PrintOptionsAbort(key, res, options); 
	}
}

DiffusionBoundaryConditionType GetDiffusionBCType(std::string type) 
{
	DiffusionBoundaryConditionType bc; 
	if (type == "half range") 
		bc = DiffusionBoundaryConditionType::HALF_RANGE; 	
	else if (type == "full range") 
		bc = DiffusionBoundaryConditionType::FULL_RANGE; 
	else if (type == "half range reflect") 
		bc = DiffusionBoundaryConditionType::HALF_RANGE_REFLECT; 
	return bc; 
}

std::string ResolveRelativePath(std::string path) 
{
	char output_name_resolve[PATH_MAX];
	realpath(path.c_str(), output_name_resolve);  	
	return std::string(output_name_resolve); 
}

}

YAML::Emitter &operator<<(YAML::Emitter &out, sol::table &table) 
{ 
	io::PrintSolTable(out, table); 
	return out; 
}