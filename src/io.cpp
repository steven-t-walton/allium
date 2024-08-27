#include "io.hpp"
#include "linalg.hpp"
#include <algorithm>
#include <regex>
#include <filesystem>

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

void PrintGitString(YAML::Emitter &out)
{
#ifdef ALLIUM_GIT_COMMIT 
	out << YAML::Key << "git" << YAML::Value << YAML::BeginMap; 
		out << YAML::Key << "branch" << YAML::Value << ALLIUM_GIT_BRANCH; 
		out << YAML::Key << "commit" << YAML::Value << ALLIUM_GIT_COMMIT; 
	#ifdef ALLIUM_GIT_TAG 
		out << YAML::Key << "tag" << YAML::Value << ALLIUM_GIT_TAG; 
	#endif
	#ifdef MFEM_GIT_STRING
		out << YAML::Key << "mfem" << YAML::Value << MFEM_GIT_STRING; 
	#endif
	out << YAML::EndMap; 
#endif
}

std::string FormatScientific(double val, int precision)
{
	std::stringstream ss; 
	ss << std::scientific << std::setprecision(precision) << val; 
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

void ProcessGlobalLogs(YAML::Emitter &out)
{
	EventLog.Synchronize();
	TimingLog.Synchronize();
	ValueLog.Synchronize();

	if (EventLog.size() or TimingLog.size() or ValueLog.size()) {
		out << YAML::Key << "logs" << YAML::Value << YAML::BeginMap; 
			if (EventLog.size())
				out << YAML::Key << "event" << EventLog;
			if (TimingLog.size()) {
				out << YAML::Key << "timing" << YAML::Value;
				PrintTimingMap(out, TimingLog);
			}
			if (ValueLog.size())
				out << YAML::Key << "value" << ValueLog;
		out << YAML::EndMap;
	}

	EventLog.clear();
	TimingLog.clear();
	ValueLog.clear();
}

mfem::DataCollection *CreateDataCollection(std::string type, std::string output_root, 
	mfem::Mesh &mesh, bool root)
{
	ValidateOption<std::string>("data collection type", type, {"paraview", "visit", "glvis"}, root);
	// clear trailing '/'
	if (output_root.back() == '/')
		output_root.pop_back();		
	if (type == "paraview") {
		return new mfem::ParaViewDataCollection(output_root, &mesh);
	} else if (type == "visit") {
		auto path = std::filesystem::path(output_root);
		if (!path.has_filename()) MFEM_ABORT("visit path must have a filename component");
		path.append(path.filename().string());
		return new mfem::VisItDataCollection(path.string(), &mesh); 
	} else if (type == "glvis") {
		auto path = std::filesystem::path(output_root);
		if (!path.has_filename()) MFEM_ABORT("visit path must have a filename component");
		path.append(path.filename().string());
		return new mfem::DataCollection(path.string(), &mesh); 
	}
	return nullptr;
}

mfem::Solver *CreateSolver(sol::table &table, std::optional<MPI_Comm> comm)
{
	int rank; 
	if (comm) MPI_Comm_rank(*comm, &rank);
	else rank = 0;
	const bool root = rank == 0;

	std::string type = table["type"];
	std::transform(type.begin(), type.end(), type.begin(), ::tolower);

	io::ValidateOption<std::string>("solver type", type, 
		{"cg", "conjugate gradient", "gmres", "fgmres", 
		"sli", "bicg", "bicgstab", "direct", "superlu", "fixed point", "fp", "kinsol", "newton"}, root); 

	mfem::Solver *s;

	if (type == "direct" or type == "superlu") {
	#ifdef MFEM_USE_SUPERLU
		if (!comm) MFEM_ABORT("MPI communicator required for SuperLU constructor");
		auto *slu = new SuperLUSolver(*comm);
		SetSuperLUOptions(table, slu->GetSolver(), root);
		s = slu;
	#else 
		if (root) MFEM_ABORT("MFEM not built with superlu");
	#endif
	}

	else {
		s = CreateIterativeSolver(table, comm);
	}
	return s;
}

mfem::IterativeSolver *CreateIterativeSolver(sol::table &table, std::optional<MPI_Comm> comm)
{
	int rank; 
	if (comm) MPI_Comm_rank(*comm, &rank);
	else rank = 0;
	const bool root = rank == 0;

	mfem::IterativeSolver *s;
	std::string type = table["type"];
	std::transform(type.begin(), type.end(), type.begin(), ::tolower);
	io::ValidateOption<std::string>("solver type", type, 
		{"cg", "conjugate gradient", "gmres", "fgmres", 
		"sli", "bicg", "bicgstab", "fixed point", "fp", "kinsol", "newton"}, root); 

	if (type == "cg" or type == "conjugate gradient") {
		if (comm)
			s = new mfem::CGSolver(*comm); 
		else
			s = new mfem::CGSolver; 
	} 

	else if (type == "gmres") {
		mfem::GMRESSolver *gmres; 
		if (comm) gmres = new mfem::GMRESSolver(*comm); 
		else gmres = new mfem::GMRESSolver; 
		sol::optional<int> kdim_avail = table["kdim"]; 
		if (kdim_avail) { gmres->SetKDim(kdim_avail.value()); }
		else { table["kdim"] = 50; }
		s = gmres; 
	}

	else if (type == "fgmres") {
		mfem::FGMRESSolver *fgmres; 
		if (comm) fgmres = new mfem::FGMRESSolver(*comm); 
		else fgmres = new mfem::FGMRESSolver; 
		sol::optional<int> kdim_avail = table["kdim"]; 
		if (kdim_avail) { fgmres->SetKDim(kdim_avail.value()); }
		// mfem's default 
		else { table["kdim"] = 50; }
		s = fgmres; 		
	}

	else if (type == "sli") {
		if (comm) s = new SLISolver(*comm); 
		else s = new SLISolver; 
	}

	else if (type == "bicg" or type == "bicgstab") {
		if (comm) s = new mfem::BiCGSTABSolver(*comm); 
		else s = new mfem::BiCGSTABSolver; 
	}

	else if (type == "fixed point" or type == "fp") {
		if (comm) s = new FixedPointIterationSolver(*comm); 
		else if (root) MFEM_ABORT("serial not implemented"); 
	}

	else if (type == "kinsol") {
	#ifdef MFEM_USE_SUNDIALS 
		const std::string mode = table["strategy"].get_or(std::string("fp")); 
		io::ValidateOption<std::string>("kinsol strategy", mode, 
			{"fp", "picard", "none", "linesearch"}, root); 
		int strategy; 
		if (mode == "fp") {
			strategy = KIN_FP; 
		} else if (mode == "picard") {
			strategy = KIN_PICARD; 
		} else if (mode == "none") {
			strategy = KIN_NONE; 
		} else if (mode == "linesearch") {
			strategy = KIN_LINESEARCH; 
		}
		table["strategy"] = mode; 
		mfem::KINSolver *kn; 
		if (comm) kn = new mfem::KINSolver(*comm, strategy, true); 
		else kn = new mfem::KINSolver(strategy, true); 
		int kdim = table["kdim"].get_or(0); 
		kn->SetMAA(kdim); 
		table["kdim"] = kdim; 
		s = kn; 
	#else 
		if (root) MFEM_ABORT("MFEM not built with sundials"); 
	#endif
	}

	else if (type == "newton") {
		if (comm) s = new mfem::NewtonSolver(*comm);
		else s = new mfem::NewtonSolver; 
	}
	SetIterativeSolverOptions(table, *s);
	return s;
}

void SetIterativeSolverOptions(sol::table &table, mfem::IterativeSolver &solver)
{
	// load generic iterative solver options 
	int print_level = table["print_level"].get_or(0); 
	solver.SetPrintLevel( (dynamic_cast<mfem::KINSolver*>(&solver) ? 1 : print_level)); 
	sol::optional<double> abstol = table["abstol"]; 
	sol::optional<double> reltol = table["reltol"]; 
	if (abstol) { solver.SetAbsTol(abstol.value()); }
	else { table["abstol"] = 0.0; }
	if (reltol) { solver.SetRelTol(reltol.value()); }
	else { table["reltol"] = 0.0; }
	int maxit = table["max_iter"].get_or(50);
	solver.SetMaxIter(maxit);  
	table["max_iter"] = maxit; 
	bool iterative_mode = table["iterative_mode"].get_or(false);
	solver.iterative_mode = iterative_mode; 
	table["iterative_mode"] = iterative_mode; 
}

mfem::Mesh CreateMesh(sol::table &table, YAML::Emitter &out, bool root) 
{
	mfem::Mesh mesh; 
	sol::optional<std::string> fname = table["file"]; 
	out << YAML::Key << "mesh" << YAML::Value << YAML::BeginMap; 
	// load from a mesh file 
	if (fname) {
		mesh = mfem::Mesh::LoadFromFile(fname.value(), 1, 1);
		out << YAML::Key << "file name" << YAML::Value << io::ResolveRelativePath(fname.value()); 
	} 

	// create a cartesian mesh from extents and elements/axis 
	else {
		sol::table ne = table["num_elements"]; 
		sol::table extents = table["extents"]; 
		assert(ne.size() == extents.size()); 
		int num_dim = ne.size(); 
		io::ValidateOption("mesh::num_dim", num_dim, {1,2,3}, root); 
		std::string eltype_str; 
		mfem::Element::Type eltype; 
		bool sfc_ordering = table["sfc_ordering"].get_or(false); 
		if (num_dim==1) {
			eltype_str = io::GetAndValidateOption(table, "element_type", {"segment"}, "segment", root); 
			mesh = mfem::Mesh::MakeCartesian1D(ne[1], extents[1]);
		} else if (num_dim==2) {
			eltype_str = io::GetAndValidateOption(table, "element_type", 
				{"quadrilateral", "triangle"}, "quadrilateral", root); 
			if (eltype_str == "quadrilateral") {
				eltype = mfem::Element::QUADRILATERAL; 
			} 
			else if (eltype_str == "triangle") {
				eltype = mfem::Element::TRIANGLE; 
			} 
			mesh = mfem::Mesh::MakeCartesian2D(ne[1], ne[2], eltype, true, extents[1], extents[2], sfc_ordering); 
		} else if (num_dim==3) {
			eltype_str = io::GetAndValidateOption(table, "element_type", 
				{"hexahedron", "tetrahedron"}, "hexahedron", root); 
			if (eltype_str == "hexahedron") {
				eltype = mfem::Element::HEXAHEDRON; 
			} 
			else if (eltype_str == "tetrahedron") {
				eltype = mfem::Element::TETRAHEDRON; 
			}
			mesh = mfem::Mesh::MakeCartesian3D(ne[1], ne[2], ne[3], eltype, extents[1], extents[2], extents[3], sfc_ordering); 
		}

		out << YAML::Key << "extents" << YAML::Value << YAML::Flow << YAML::BeginSeq; 
		for (auto i=1; i<=extents.size(); i++) {
			out << (double)extents[i]; 
		} 
		out << YAML::EndSeq; 

		out << YAML::Key << "elements/axis" << YAML::Value << YAML::Flow << YAML::BeginSeq; 
		for (auto i=1; i<=ne.size(); i++) {
			// out << (int)ne[i] * pow(2,tot_ref); 
			out << (int)ne[i]; 
		} 
		out << YAML::EndSeq; 		
		out << YAML::Key << "element type" << YAML::Value << eltype_str; 
		out << YAML::Key << "space filling ordering" << YAML::Value << sfc_ordering; 
	}
	return mesh; 
}

void SetMeshAttributes(mfem::Mesh &mesh, std::function<std::string(double,double,double)> f,
	const std::unordered_map<std::string,int> &map, bool root)
{
	const auto dim = mesh.Dimension(); 
	for (int e=0; e<mesh.GetNE(); e++) {
		double c[3]; 
		mfem::Vector cvec(c, dim); 
		mesh.GetElementCenter(e, cvec);
		std::string attr_name = f(c[0], c[1], c[2]); 
		if (!map.contains(attr_name) and root) {
			MFEM_ABORT("material named \"" << attr_name << "\" not defined"); 
		}
		int attr = map.at(attr_name); 
		mesh.SetAttribute(e, attr); 
	}		
}

void SetMeshBdrAttributes(mfem::Mesh &mesh, std::function<std::string(double,double,double)> f,
	const std::unordered_map<std::string,int> &map, bool root)
{
	const auto dim = mesh.Dimension(); 
	for (int e=0; e<mesh.GetNBE(); e++) {
		const mfem::Element &el = *mesh.GetBdrElement(e); 
		int geom = mesh.GetBdrElementGeometry(e);
		mfem::ElementTransformation &trans = *mesh.GetBdrElementTransformation(e); 
		double c[3]; 
		mfem::Vector cvec(c, dim);  
		trans.Transform(mfem::Geometries.GetCenter(geom), cvec); 
		std::string attr_name = f(c[0], c[1], c[2]); 
		if (!map.contains(attr_name) and root) {
			MFEM_ABORT("boundary condition named \"" << attr_name << "\" not defined"); 
		}
		mesh.SetBdrAttribute(e, map.at(attr_name)); 
	}
}

void PrintMeshCharacteristics(YAML::Emitter &out, mfem::ParMesh &mesh, int sr, int pr)
{
	double hmin, hmax, kmin, kmax; 
	mesh.GetCharacteristics(hmin, hmax, kmin, kmax); 
	const auto global_ne = mesh.ReduceInt(mesh.GetNE()); 
	const auto nranks = mesh.GetNRanks();
	const auto dim = mesh.Dimension();
	out << YAML::Key << "dim" << YAML::Value << dim; 
	out << YAML::Key << "elements" << YAML::Value << global_ne; 
	out << YAML::Key << "mesh size" << YAML::Value << YAML::BeginMap; 
		out << YAML::Key << "min" << YAML::Value << hmin; 
		out << YAML::Key << "max" << YAML::Value << hmax; 
	out << YAML::EndMap; 
	out << YAML::Key << "mesh conditioning" << YAML::Value << YAML::BeginMap; 
		out << YAML::Key << "min" << YAML::Value << kmin; 
		out << YAML::Key << "max" << YAML::Value << kmax; 
	out << YAML::EndMap; 
	out << YAML::Key << "MPI ranks" << YAML::Value << nranks; 
	out << YAML::Key << "refinements" << YAML::Value << YAML::BeginMap; 
		out << YAML::Key << "serial" << YAML::Value << sr; 
		out << YAML::Key << "parallel" << YAML::Value << pr; 
	out << YAML::EndMap; 
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
	int print_level = table["print_level"].get_or(0); 
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
	amg.SetPrintLevel(print_level); 
}

#ifdef MFEM_USE_SUPERLU
void SetSuperLUOptions(sol::table &table, mfem::SuperLUSolver &slu, bool root)
{
	for (const auto &it : table) {
		auto key = it.first.as<std::string>(); 
		ValidateOption<std::string>("superlu", key, 
			{"type", "symmetric_pattern", "iterative_refine", "equilibriate", 
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

bool ParseKINSOLMessage(char *msg, int &it, double &norm)
{
	std::regex nni_reg("nni =\\s+([0-9]+)\\s"); 
	std::cmatch nni_match; 
	if (std::regex_search(msg, nni_match, nni_reg)) {
		it = std::stoi(nni_match[1]); 
		// fnorm =      0.0005730787351824196
		std::regex norm_reg("fnorm =\\s+(\\S+)"); 
		std::cmatch norm_match; 
		if (std::regex_search(msg, norm_match, norm_reg)) {
			norm = std::stod(norm_match[1].str());
		}
		return true; 
	}
	else {
		return false; 
	}
}

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

// DiffusionBoundaryConditionType GetDiffusionBCType(std::string type) 
// {
// 	DiffusionBoundaryConditionType bc; 
// 	if (type == "half range") 
// 		bc = DiffusionBoundaryConditionType::HALF_RANGE; 	
// 	else if (type == "full range") 
// 		bc = DiffusionBoundaryConditionType::FULL_RANGE; 
// 	else if (type == "half range reflect") 
// 		bc = DiffusionBoundaryConditionType::HALF_RANGE_REFLECT; 
// 	return bc; 
// }

std::string ResolveRelativePath(std::string path) 
{
	std::filesystem::path obj(path);
	auto resolved = std::filesystem::absolute(path);
	return resolved.string();
}

} // end namespace io 

YAML::Emitter &operator<<(YAML::Emitter &out, sol::table &table) 
{ 
	io::PrintSolTable(out, table); 
	return out; 
}