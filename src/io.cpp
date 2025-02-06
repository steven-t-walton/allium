#include "io.hpp"
#include "linalg.hpp"
#include "fixed_point.hpp"
#include <algorithm>
#include <regex>
#include <filesystem>
#include <omp.h>

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

LogMap<double,SUM,MAX> TimingLogPersistent;
LogMap<unsigned long int,SUM> EventLogPersistent;
void ProcessGlobalLogs(YAML::Emitter &out, int verbose)
{
	EventLog.Synchronize();
	TimingLog.Synchronize();

	const bool print_events = EventLog.size() and (verbose & 1);
	const bool print_timing = TimingLog.size() and (verbose & 2);

	if (print_events) {
		out << YAML::Key << "event log" << YAML::Value << EventLog;
	}

	if (print_timing) {
		out << YAML::Key << "timing log" << YAML::Value; 
		io::PrintTimingMap(out, TimingLog);
	}

	if (TimingLog.size()) {
		for (const auto &it : TimingLog) {
			TimingLogPersistent.Log(it.first, it.second);
		}
	}
	if (EventLog.size()) {
		for (const auto &it : EventLog) {
			EventLogPersistent.Log(it.first, it.second);
		}
	}

	EventLog.clear();
	TimingLog.clear();
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

IterativeSolverOptions GetIterativeSolverOptions(sol::table &table)
{
	IterativeSolverOptions opts;
	opts.print_level = table["print_level"].get_or(0);
	sol::optional<double> abstol = table["abstol"]; 
	sol::optional<double> reltol = table["reltol"]; 
	if (abstol) { opts.abstol = abstol.value(); }
	else { table["abstol"] = opts.abstol; }
	if (reltol) { opts.reltol = reltol.value(); }
	else { table["reltol"] = opts.reltol; }
	opts.max_iter = table["max_iter"].get_or(opts.max_iter);
	table["max_iter"] = opts.max_iter; 
	opts.iterative_mode = table["iterative_mode"].get_or(opts.iterative_mode);
	table["iterative_mode"] = opts.iterative_mode; 	
	return opts;
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

int *GeneratePartitioning(sol::table &table, YAML::Emitter &out, 
	mfem::Mesh &ser_mesh, int nparts, mfem::Coefficient &total, bool root)
{
	const auto dim = ser_mesh.Dimension();
	int *partitioning = nullptr;
	const std::string type = io::GetAndValidateOption(table, "type", {"metis", "cartesian"}, root);
	out << YAML::Key << "partitioning" << YAML::Value << YAML::BeginMap;
	out << YAML::Key << "type" << YAML::Value << type; 
	if (type == "metis") {
		const bool edge_weight = table["weight_edges"].get_or(false);
		const bool node_weight = table["weight_nodes"].get_or(false);
		const int scheme = (node_weight << 1) | edge_weight; 
		partitioning = utils::GenerateMetisPartitioning(ser_mesh, nparts, total, scheme);
		out << YAML::Key << "weight edges" << YAML::Value << edge_weight;
		out << YAML::Key << "weight nodes" << YAML::Value << node_weight; 
		out << YAML::Key << "scheme" << YAML::Value << scheme;
	} else if (type == "cartesian") {
		sol::table n = table["partitions"]; 
		if (n.size() != dim) { MFEM_ABORT("partition size incorrect for mesh dimension"); }
		int nxyz[3];
		int nranks = 1;
		out << YAML::Key << "partitions" << YAML::Value << YAML::Flow << YAML::BeginSeq;
		for (int d=0; d<dim; d++) {
			nxyz[d] = n[d+1];
			nranks *= nxyz[d];
			out << nxyz[d];
		}
		out << YAML::EndSeq;
		if (nranks != nparts) MFEM_ABORT("supplied partitions do not match ranks available");
		partitioning = ser_mesh.CartesianPartitioning(nxyz);
	}
	out << YAML::EndMap;
	return partitioning;
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
	out << YAML::Key << "refinements" << YAML::Value << YAML::BeginMap; 
		out << YAML::Key << "serial" << YAML::Value << sr; 
		out << YAML::Key << "parallel" << YAML::Value << pr; 
	out << YAML::EndMap; 
}

AngularQuadrature *CreateAngularQuadrature(sol::table &table, YAML::Emitter &out, int dim, bool root)
{
	std::string type = table["type"]; 
	ValidateOption<std::string>("sn quadrature type", type, 
		{"legendre", "level symmetric", "abu shumays", "product"}, root);
	AngularQuadrature *quad;
	const int order = table["order"];
	if (dim==1) type = "legendre";

	out << YAML::Key << "sn quadrature" << YAML::Value << YAML::BeginMap; 
	out << YAML::Key << "type" << YAML::Value << type; 
	out << YAML::Key << "order" << YAML::Key << order; 

	if (type == "legendre") {
		quad = new LegendreQuadrature(order, dim); 
	}

	else if (type == "level symmetric") {
		quad = new LevelSymmetricQuadrature(order, dim); 
	}

	else if (type == "abu shumays") {
		quad = new AbuShumaysQuadrature(order, dim); 
	}

	else if (type == "product") {
		const std::string polar_type_str = io::GetAndValidateOption<std::string>(
			table, "polar_type", {"legendre", "lobatto"}, "legendre", root);
		out << YAML::Key << "polar type" << YAML::Value << polar_type_str;
		const bool tri = table["triangular"].get_or(false);
		int polar_type;
		if (polar_type_str == "legendre") polar_type = mfem::Quadrature1D::GaussLegendre;
		else if (polar_type_str == "lobatto") polar_type = mfem::Quadrature1D::GaussLobatto;
		if (tri) {
			out << YAML::Key << "azimuthal order" << YAML::Value << "triangular";
			quad = new ProductQuadrature(dim, order, order, polar_type, true);
		}
		else {
			const int az_order = table["azimuthal_order"].get_or(order); 
			out << YAML::Key << "azimuthal order" << YAML::Value << az_order;
			quad = new ProductQuadrature(dim, order, az_order, polar_type, false);
		}
	}
	sol::optional<sol::table> rotate_avail = table["rotation"]; 
	if (rotate_avail) {
		sol::table rotate = rotate_avail.value();
		const double theta_x = rotate["x"].get_or(0.0);
		const double theta_y = rotate["y"].get_or(0.0);
		const double theta_z = rotate["z"].get_or(0.0);
		quad->Rotate(theta_x, theta_y, theta_z); 
		out << YAML::Key << "rotation" << YAML::Value << YAML::Flow << YAML::BeginMap;
			out << YAML::Key << "x" << YAML::Value << theta_x;
			out << YAML::Key << "y" << YAML::Value << theta_y; 
			out << YAML::Key << "z" << YAML::Value << theta_z; 
		out << YAML::EndMap;
	}
	out << YAML::Key << "number of angles" << YAML::Key << quad->Size(); 
	const bool print = table["print"].get_or(true);
	if (print)
		PrintAngularQuadrature(out, *quad);
	out << YAML::EndMap;
	return quad;
}

void PrintAngularQuadrature(YAML::Emitter &out, const AngularQuadrature &quad)
{
	out << YAML::Key << "angular quadrature rule" << YAML::Value << YAML::BeginSeq; 
	for (int a=0; a<quad.Size(); a++) {
		const auto &Omega = quad.GetOmega(a); 
		const auto w = quad.GetWeight(a); 
		out << YAML::Flow << YAML::BeginSeq; 
		out << YAML::Flow << YAML::BeginSeq; 
		for (int d=0; d<Omega.Size(); d++) {
			out << Omega(d); 
		}
		out << YAML::EndSeq;
		out << w; 
		out << YAML::EndSeq; 
	}
	out << YAML::EndSeq; 
}

void PrintParallelInformation(YAML::Emitter &out, MPI_Comm comm)
{
	int ranks; 
	MPI_Comm_size(comm, &ranks); 
	const auto threads = omp_get_max_threads();
	const auto total = ranks * threads; 
	out << YAML::Key << "parallel" << YAML::Value << YAML::BeginMap; 
		out << YAML::Key << "MPI ranks" << YAML::Value << ranks; 
		out << YAML::Key << "OpenMP threads" << YAML::Value << threads; 
		out << YAML::Key << "total" << YAML::Value << total; 
	out << YAML::EndMap;
}

void PrintEnergyGridInformation(YAML::Emitter &out, const MultiGroupEnergyGrid &grid)
{
	out << YAML::Key << "groups" << YAML::Value << grid.Size(); 
	const auto &bounds = grid.Bounds();
	out << YAML::Key << "bounds" << YAML::Value << YAML::BeginSeq;
	for (int b=0; b<bounds.Size(); b++) {
		out << FormatScientific(bounds[b],6);
	}
	out << YAML::EndSeq;
}

MultiGroupEnergyGrid CreateEnergyGrid(sol::table &table, YAML::Emitter &out, bool root)
{
	MultiGroupEnergyGrid grid;
	sol::optional<sol::table> bounds_table_avail = table["bounds"];
	if (bounds_table_avail) {
		sol::table bounds_table = bounds_table_avail.value();
		mfem::Array<double> bounds(bounds_table.size());
		for (int i=0; i<bounds.Size(); i++) {
			bounds[i] = bounds_table[i+1];
		}
		grid = MultiGroupEnergyGrid(bounds);
	} else {
		const double Emin = table["min"];
		const double Emax = table["max"]; 
		const int G = table["num_groups"];

		out << YAML::Key << "Emin" << YAML::Value << FormatScientific(Emin); 
		out << YAML::Key << "Emax" << YAML::Value << FormatScientific(Emax);

		if (G == 1) {
			grid = MultiGroupEnergyGrid::MakeGray(Emin, Emax);
		} else {
			const bool extend_to_zero = table["extend_to_zero"].get_or(false);
			const auto spacing = io::GetAndValidateOption<std::string>(table, "spacing", 
				{"log", "equal"}, "log", root);
			out << YAML::Key << "group spacing" << YAML::Value << spacing;
			out << YAML::Key << "extend to zero" << YAML::Value << extend_to_zero;

			if (spacing == "log") {
				grid = MultiGroupEnergyGrid::MakeLogSpaced(Emin, Emax, G, extend_to_zero);
			} else if (spacing == "equal") {
				grid = MultiGroupEnergyGrid::MakeEqualSpaced(Emin, Emax, G, extend_to_zero);
			}			
		}
	}
	return grid;
}

MultiGroupEnergyGrid CreateEnergyGrid(sol::optional<sol::table> &table_avail, 
	YAML::Emitter &out, const IpcressData *ipcress, bool root)
{
	MultiGroupEnergyGrid energy_grid;
	if (table_avail) {
		sol::table table = table_avail.value();
		sol::optional<sol::table> bounds_avail = table["bounds"];
		if (ipcress) {
			int G;
			if (bounds_avail) {
				G = bounds_avail.value().size();
			} else {
				G = table["num_groups"];
			}
			if (G>1) MFEM_ABORT("ipcress opacity can only be used with ipcress group structure or gray");
			const auto &bounds = ipcress->GetGroupBounds();
			energy_grid = MultiGroupEnergyGrid::MakeGray(bounds[0], bounds.Last());			
		} else {
			energy_grid = io::CreateEnergyGrid(table, out, root);			
		}
	} else {
		if (ipcress) {
			energy_grid = MultiGroupEnergyGrid(ipcress->GetGroupBounds());
		} else {
			MFEM_ABORT("group bounds not defined");
		}
	}
	return energy_grid;
}

OpacityCoefficient *OpacityFactory::CreateOpacity(sol::table &table) const
{
	OpacityCoefficient *opac;
	std::string type = table["type"]; 
	io::ValidateOption<std::string>("opacity type", type, 
		{"constant", "analytic gray", "analytic", "analytic edge", "brunner", "ipcress"}, root); 
	out << YAML::Key << "type" << YAML::Value << type; 
	if (type == "constant") {
		sol::table values = table["values"];
		mfem::Vector vec(values.size()); 
		for (int i=0; i<vec.Size(); i++) { vec(i) = values[i+1]; }
		out << YAML::Key << "values" << YAML::Value << YAML::Flow << YAML::BeginSeq; 
		for (int i=0; i<vec.Size(); i++) {
			vec(i) = values[i+1];
			out << vec(i);
		}
		out << YAML::EndSeq;
		opac = new ConstantOpacityCoefficient(vec);  
	}

	else if (type == "analytic gray") {
		double coef = table["coef"]; 
		double nrho = table["nrho"]; 
		double nT = table["nT"]; 
		opac = new AnalyticGrayOpacityCoefficient(coef, nrho, nT); 

		out << YAML::Key << "coef" << YAML::Value << coef;
		out << YAML::Key << "nrho" << YAML::Value << nrho; 
		out << YAML::Key << "nT" << YAML::Value << nT;
	}

	else if (type == "analytic") {
		double coef = table["coef"]; 
		double nrho = table["nrho"]; 
		double nT = table["nT"]; 
		double Emin = table["Emin"].get_or(0.0);
		const int int_order = table["int_order"].get_or(1);
		sol::optional<sol::table> edge_avail = table["edge"]; 
		FleckCummingsOpacityFunction func(coef, nrho, nT, Emin);

		out << YAML::Key << "coef" << YAML::Value << coef;
		out << YAML::Key << "nrho" << YAML::Value << nrho; 
		out << YAML::Key << "nT" << YAML::Value << nT;
		out << YAML::Key << "Emin" << YAML::Value << FormatScientific(Emin); 
		out << YAML::Key << "integration order" << YAML::Value << int_order;
		if (edge_avail) {
			sol::table edge_table = edge_avail.value();
			double edge = edge_table["energy"];
			double coef = edge_table["coef"];
			func.SetEdge(edge, coef);
			out << YAML::Key << "edge" << YAML::Value << YAML::BeginMap; 
				out << YAML::Key << "energy" << YAML::Value << edge; 
				out << YAML::Key << "coef" << YAML::Value << coef; 
			out << YAML::EndMap;
		}
		auto *ptr = new MultiGroupFunctionOpacityCoefficient(grid.Bounds(), func);
		if (int_order > 1) ptr->SetIntegrationOrder(int_order);
		opac = ptr;
	} 

	else if (type == "analytic edge") {
		const double c0 = table["c0"];
		const double c1 = table["c1"];
		const double c2 = table["c2"];
		const double Emin = table["Emin"];
		const double Eedge = table["Eedge"];
		const double delta_s = table["delta_s"];
		const double delta_w = table["delta_w"];
		const double nT = table["nT"].get_or(-0.5);
		const int Nlines = table["Nlines"];
		const int int_order = table["int_order"].get_or(1);
		EdgeLineOpacityFunction func(c0, c1, c2, Emin, Eedge, delta_s, delta_w, nT, Nlines);
		std::function<double(double,double)> weight_func; 
		sol::optional<std::string> weight_type_avail = table["weight"];
		if (weight_type_avail) {
			const std::string weight_type = weight_type_avail.value();
			io::ValidateOption<std::string>("opacity weight", weight_type, {"planck"}, root);
			if (weight_type == "planck") {
				weight_func = PlanckFunction;
			}
		}
		auto *ptr = new MultiGroupFunctionOpacityCoefficient(grid.Bounds(), func, weight_func);
		if (int_order > 1) ptr->SetIntegrationOrder(int_order);
		out << YAML::Key << "c0" << YAML::Value << c0;
		out << YAML::Key << "c1" << YAML::Value << c1;
		out << YAML::Key << "c2" << YAML::Value << c2;
		out << YAML::Key << "Emin" << YAML::Value << FormatScientific(Emin);
		out << YAML::Key << "Eedge" << YAML::Value << FormatScientific(Eedge);
		out << YAML::Key << "delta_s" << YAML::Value << FormatScientific(delta_s);
		out << YAML::Key << "delta_w" << YAML::Value << FormatScientific(delta_w);
		out << YAML::Key << "nT" << YAML::Value << nT;
		out << YAML::Key << "Nlines" << YAML::Value << Nlines;
		out << YAML::Key << "integration order" << YAML::Value << int_order;
		opac = ptr;
	}

	else if (type == "brunner") {
		const double c0 = table["c0"];
		const double c1 = table["c1"];
		const double c2 = table["c2"];
		const double Emin = table["Emin"];
		const double Eedge = table["Eedge"];
		const double delta_s = table["delta_s"];
		const double delta_w = table["delta_w"];
		const int Nlines = table["Nlines"];
		const auto weight_str = io::GetAndValidateOption<std::string>(
			table, "weight", {"planck", "rosseland"}, "planck", root);
		const bool planck_weight = weight_str == "planck";
		opac = new BrunnerOpacityCoefficient(
			grid.Bounds(), c0, c1, c2, Emin, Eedge, delta_s, delta_w, Nlines, planck_weight);
		out << YAML::Key << "c0" << YAML::Value << c0;
		out << YAML::Key << "c1" << YAML::Value << c1;
		out << YAML::Key << "c2" << YAML::Value << c2;
		out << YAML::Key << "Emin" << YAML::Value << FormatScientific(Emin);
		out << YAML::Key << "Eedge" << YAML::Value << FormatScientific(Eedge);
		out << YAML::Key << "delta_s" << YAML::Value << FormatScientific(delta_s);
		out << YAML::Key << "delta_w" << YAML::Value << FormatScientific(delta_w);
		out << YAML::Key << "Nlines" << YAML::Value << Nlines;
		out << YAML::Key << "weight function" << YAML::Value << weight_str;
	}

	else if (type == "ipcress") {
		if (!ipcress_data) MFEM_ABORT("ipcress data not provided");
		const int mat_id = table["id"]; 
		const auto key = io::GetAndValidateOption<std::string>(
			table, "key", {"ramg", "pamg", "ragray", "pgray"}, "ramg", root);
		opac = new IpcressOpacityCoefficient(*ipcress_data, mat_id, key);

		out << YAML::Key << "ipcress id" << YAML::Value << mat_id;
		out << YAML::Key << "ipcress key" << YAML::Value << key;
	}
	return opac;
}

void PrintIpcressInformation(YAML::Emitter &out, const IpcressData &data)
{
	auto temperature = data.GetField(0, "tgrid");
	auto density = data.GetField(0, "rgrid");
	out << YAML::Key << "ipcress" << YAML::Value << YAML::BeginMap;
		out << YAML::Key << "file" << YAML::Value << ResolveRelativePath(data.FileName());
		out << YAML::Key << "materials" << YAML::Value << data.NumMaterials();
		out << YAML::Key << "temperature" << YAML::Value << YAML::BeginMap;
			out << YAML::Key << "size" << YAML::Value << temperature.size(); 
			out << YAML::Key << "min" << YAML::Value << std::exp(temperature.front());
			out << YAML::Key << "max" << YAML::Value << std::exp(temperature.back());
		out << YAML::EndMap;
		out << YAML::Key << "density" << YAML::Value << YAML::BeginMap;
			out << YAML::Key << "size" << YAML::Value << density.size(); 
			out << YAML::Key << "min" << YAML::Value << std::exp(density.front());
			out << YAML::Key << "max" << YAML::Value << std::exp(density.back());
		out << YAML::EndMap;
	out << YAML::EndMap;
}

NegativeFluxFixup *CreateNegativeFluxFixup(sol::table &table, bool root)
{
	NegativeFluxFixupOperator *op; 
	mfem::SLBQPOptimizer *optimizer = nullptr;
	std::string type = table["type"]; 
	io::ValidateOption<std::string>("fixup type", type, 
		{"zero", "zero and scale", "local optimization", "ryosuke"}, root); 
	double min = table["psi_min"].get_or(0.0); 
	if (type == "zero") {
		op = new ZeroFixupOperator(min);
	} else if (type == "zero and scale") {
		op = new ZeroAndScaleFixupOperator(min);
	} else if (type == "local optimization") {
		optimizer = new mfem::SLBQPOptimizer();
		double abstol = table["abstol"].get_or(1e-18); 
		double reltol = table["reltol"].get_or(1e-12); 
		int max_iter = table["max_iter"].get_or(20); 
		int print_level = table["print_level"].get_or(-1); 
		optimizer->SetAbsTol(abstol); 
		optimizer->SetRelTol(reltol); 
		optimizer->SetMaxIter(max_iter); 
		optimizer->SetPrintLevel(print_level); 
		optimizer->iterative_mode = table["iterative_mode"].get_or(true);
		op = new LocalOptimizationFixupOperator(*optimizer, min);
	} else if (type == "ryosuke") {
		double min = table["psi_min"].get_or(0.0);
		op = new RyosukeFixupOperator(min);
	}
	return new NegativeFluxFixup(op, optimizer);
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

void SetSweepOptions(sol::table &table, InverseAdvectionOperator &Linv, bool root)
{
	for (const auto &it : table) {
		auto key = it.first.as<std::string>(); 
		ValidateOption<std::string>("sweep", key, 
			{"write_graph", "send_buffer_size", "max_sends_per_recv", "parallel_block_jacobi", "num_pbj_sweeps"}, 
			root); 
	}
	bool pbj = table["parallel_block_jacobi"].get_or(false);
	int num_pbj_sweeps = table["num_pbj_sweeps"].get_or(1);
	Linv.UseParallelBlockJacobi(pbj);
	if (pbj) Linv.SetNumPBJSweeps(num_pbj_sweeps);
	bool write_graph = table["write_graph"].get_or(false); 
	if (write_graph) 
		Linv.WriteGraphToDot("graph"); 
	sol::optional<int> send_buffer_size = table["send_buffer_size"]; 
	if (send_buffer_size) 
		Linv.SetSendBufferSize(send_buffer_size.value()); 
	sol::optional<int> max_sends_per_recv = table["max_sends_per_recv"]; 
	if (max_sends_per_recv)
		Linv.SetMaxSendsPerReceive(max_sends_per_recv.value());
}

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

void SundialsErrorFunction(int error_code, const char *module, const char *function, char *msg, void *user_data)
{
	mfem::out << msg << std::endl; 
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

utils::InterpolatedTable1D *CreateInterpolatedTable(sol::table &table, YAML::Emitter &out, bool root) 
{
	utils::InterpolatedTable1D *ptr;
	sol::optional<std::string> file_avail = table["file"];
	sol::optional<sol::table> breaks_avail = table["breaks"];
	if (root and !file_avail and !breaks_avail) { MFEM_ABORT("must supply file or table of break points"); }

	if (file_avail) {
		const std::string file = file_avail.value();
		ptr = new utils::InterpolatedTable1D(file);
		out << YAML::Key << "file" << YAML::Value << ResolveRelativePath(file);
	} else {
		sol::table breaks = breaks_avail.value();
		mfem::Vector x(breaks.size()), y(breaks.size());
		for (int i=0; i<breaks.size(); i++) {
			sol::table row = breaks[i+1]; 
			x(i) = row[1];
			y(i) = row[2];
		}
		ptr = new utils::InterpolatedTable1D(x,y);
		out << YAML::Key << "breaks" << YAML::Value << YAML::BeginSeq;
		for (int i=1; i<=breaks.size(); i++) {
			sol::table row = breaks[i];
			const double x = row[1];
			const double y = row[2];
			out << YAML::Flow << YAML::BeginSeq << x << y << YAML::EndSeq;
		}
		out << YAML::EndSeq;
	}

	const bool log_x = table["log_x"].get_or(false);
	const bool log_y = table["log_y"].get_or(false);
	const bool piecewise = table["piecewise_constant"].get_or(false);
	ptr->UseLogX(log_x);
	ptr->UseLogY(log_y);
	ptr->UsePiecewiseConstant(piecewise);

	out << YAML::Key << "log x" << YAML::Value << log_x; 
	out << YAML::Key << "log y" << YAML::Value << log_y;
	out << YAML::Key << "piecewise constant" << YAML::Value << piecewise;

	return ptr;
}

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