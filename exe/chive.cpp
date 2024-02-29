#include "mfem.hpp"
#include "yaml-cpp/yaml.h"
#include "sol/sol.hpp"

#include "tvector.hpp"
#include "angular_quadrature.hpp"
#include "mip.hpp"
#include "p1diffusion.hpp"
#include "sweep.hpp"
#include "transport_op.hpp"
#include "block_smm_op.hpp"
#include "cons_smm_op.hpp"
#include "phase_coefficient.hpp"
#include "comment_stream.hpp"
#include "linalg.hpp"
#include "io.hpp"
#include <kinsol/kinsol.h>

template<typename K, typename V> 
void YAMLKeyValue(YAML::Emitter &out, K key, V value) {
	out << YAML::Key << key << YAML::Value << value; 
}

using LuaPhaseFunction = std::function<double(double,double,double,double,double,double)>; 

class TransportIterationMonitor : public mfem::IterativeSolverMonitor
{
private:
	mfem::IterativeSolver const * const inner_solver = nullptr; 
	const TransportOperator &T; 
	const DiffusionSyntheticAccelerationOperator * const dsa; 
	YAML::Emitter &out; 
	bool first = true; 
public:
	mfem::Array<int> inner_it; 

	TransportIterationMonitor(YAML::Emitter &yaml, const TransportOperator &t, 
		const DiffusionSyntheticAccelerationOperator * const d=nullptr, 
		const mfem::IterativeSolver * const inner=nullptr) 
		: out(yaml), T(t), dsa(d), inner_solver(inner) 
	{
	}
	void MonitorResidual(int it, double norm, const mfem::Vector &r, bool final) {
		if (first) {
			out << YAML::Key << "transport iterations" << YAML::Value << YAML::BeginSeq; 
			first = false; 
		}
		out << YAML::BeginMap; 
		out << YAML::Key << "it" << YAML::Value << it; 
		std::stringstream ss; 
		ss << std::scientific << std::setprecision(3) << norm; 
		out << YAML::Key << "norm" << YAML::Value << ss.str(); 
		if (inner_solver) {
			out << YAML::Key << "inner solver" << YAML::Value << YAML::BeginMap; 
				out << YAML::Key << "it" << YAML::Value << inner_solver->GetNumIterations(); 
				std::stringstream ss; 
				ss << std::scientific << std::setprecision(3) << inner_solver->GetFinalNorm(); 
				out << YAML::Key << "norm" << YAML::Value << ss.str(); 
			out << YAML::EndMap; 
			inner_it.Append(inner_solver->GetNumIterations()); 
		}
		out << YAML::Key << "timings" << YAML::Value << YAML::BeginMap; 
			out << YAML::Key << "sweep" << YAML::Value << io::FormatTimeString(T.GetStopWatch().RealTime()); 
			if (dsa)
				out << YAML::Key << "preconditioner" << YAML::Value << io::FormatTimeString(dsa->GetStopWatch().RealTime()); 
		out << YAML::EndMap; 
		out << YAML::EndMap << YAML::Newline; 

		if (final) {
			out << YAML::EndSeq; 
			first = true; 
		}
	}
};

class MomentMethodIterationMonitor : public mfem::IterativeSolverMonitor
{
private:
	YAML::Emitter &out; 
	MomentMethodFixedPointOperator &op; 
	mfem::IterativeSolver const * const inner_solver = nullptr; 
public:
	mfem::Array<int> inner_it; 
	MomentMethodIterationMonitor(YAML::Emitter &yaml, MomentMethodFixedPointOperator &_op, 
		const mfem::IterativeSolver * const inner=nullptr)
		: out(yaml), op(_op), inner_solver(inner)
	{
	}
	void MonitorResidual(int it, double norm, const mfem::Vector &r, bool final) {
		if (it==1) {
			out << YAML::Key << "transport iterations" << YAML::Value << YAML::BeginSeq; 
		}
		out << YAML::BeginMap; 
		out << YAML::Key << "it" << YAML::Value << it; 
		std::stringstream ss; 
		ss << std::scientific << std::setprecision(3) << norm; 
		out << YAML::Key << "norm" << YAML::Value << ss.str(); 
		if (inner_solver) {
			out << YAML::Key << "inner solver" << YAML::Value << YAML::BeginMap; 
				out << YAML::Key << "it" << YAML::Value << inner_solver->GetNumIterations(); 
				std::stringstream ss; 
				ss << std::scientific << std::setprecision(3) << inner_solver->GetFinalNorm(); 
				out << YAML::Key << "norm" << YAML::Value << ss.str(); 
			out << YAML::EndMap; 
			inner_it.Append(inner_solver->GetNumIterations()); 
		}
		out << YAML::Key << "timings" << YAML::Value << YAML::BeginMap; 
			out << YAML::Key << "total" << YAML::Value << io::FormatTimeString(op.TotalTimer().RealTime()); 
			out << YAML::Key << "sweep" << YAML::Value << io::FormatTimeString(op.SweepTimer().RealTime()); 
			out << YAML::Key << "moment" << YAML::Value << io::FormatTimeString(op.MomentTimer().RealTime()); 
		out << YAML::EndMap; 
		out << YAML::EndMap << YAML::Newline; 

		if (final) {
			out << YAML::EndSeq; 
		}		
	}
};

int main(int argc, char *argv[]) {
	// initialize MPI 
	// automatically calls MPI_Finalize 
	mfem::Mpi::Init(argc, argv); 
	// must call hypre init for BoomerAMG now? 
	mfem::Hypre::Init(); 

	mfem::StopWatch timer; 
	timer.Start(); 

	const auto rank = mfem::Mpi::WorldRank(); 
	const bool root = rank == 0; 

	if (argc == 1) { MFEM_ABORT("must supply input file"); }

	// stream for output to terminal
	mfem::OutStream par_out(std::cout);
	// enable only for root so non-root procs don't clutter cout 
	if (rank!=0) par_out.Disable();

	// make mfem print everything with 
	// yaml comment preceeding 
	// helps keep output yaml parse-able 
	CommentStreamBuf comm_buf(mfem::out, '#'); 

	if (root) {
		mfem::out << "      __                            \n"; 
		mfem::out << "     /\\ \\      __                   \n"; 
		mfem::out << "  ___\\ \\ \\___ /\\_\\  __  __     __   \n"; 
		mfem::out << " /'___\\ \\  _ `\\/\\ \\/\\ \\/\\ \\  /'__`\\ \n"; 
		mfem::out << "/\\ \\__/\\ \\ \\ \\ \\ \\ \\ \\ \\_/ |/\\  __/ \n"; 
		mfem::out << "\\ \\____\\\\ \\_\\ \\_\\ \\_\\ \\___/ \\ \\____\\ \n";
		mfem::out << " \\/____/ \\/_/\\/_/\\/_/\\/__/   \\/____/\n"; 
		mfem::out << "\n     a linear transport solver\n"; 
		mfem::out << std::endl; 
	}

	std::string input_file, lua_cmds; 
	mfem::OptionsParser args(argc, argv); 
	args.AddOption(&input_file, "-i", "--input", "input file name", true); 
	args.AddOption(&lua_cmds, "-l", "--lua", "lua commands to run", false); 
	args.Parse(); 
	if (!args.Good()) {
		args.PrintUsage(par_out); 
		return 1; 
	}
	if (root) { 
		args.PrintOptions(mfem::out); 
		mfem::out << std::endl; 
	}

	// YAML output 
	YAML::Emitter out(par_out);
	out.SetDoublePrecision(3); 
	out << YAML::BeginMap; 

	// --- load lua file --- 
	sol::state lua; 
	lua.open_libraries(); // allows using standard libraries (e.g. math) in input
	lua.script_file(input_file); // load from first cmd line argument 

	// overwrite input script with lua commands provided 
	// through cmdline 
	if (!lua_cmds.empty()) {
		lua.script(lua_cmds); 
	}

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
	out << YAML::Key << "input file" << YAML::Value << realpath(input_file.c_str(), nullptr); 

	// --- extract list of materials --- 
	std::vector<std::string> attr_list; 
	sol::table materials = lua["materials"]; 
	if (materials.valid()) {
		for (const auto &material : materials) {
			auto key = material.first.as<std::string>(); 
			attr_list.push_back(key); 
		}
	} else { MFEM_ABORT("materials not defined"); }

	// get data from lua 
	auto nattr = attr_list.size(); 
	mfem::Vector total_list(nattr), scattering_list(nattr); 
	// must store the lua object so data doesn't go out of scope 
	// for the source coefficients 
	std::vector<sol::object> lua_source_objs(nattr); 
	mfem::Array<PhaseSpaceCoefficient*> source_list(nattr); 
	for (auto i=0; i<attr_list.size(); i++) {
		sol::table data = materials[attr_list[i].c_str()]; 
		if (!data.valid()) MFEM_ABORT("material named " << attr_list[i] << " not found"); 
		total_list(i) = data["total"]; 
		scattering_list(i) = data["scattering"]; 
		lua_source_objs[i] = data["source"]; 
		if (lua_source_objs[i].get_type() == sol::type::number) {
			auto source_val = lua_source_objs[i].as<double>(); 
			source_list[i] = new ConstantPhaseSpaceCoefficient(source_val); 
		} else if (lua_source_objs[i].get_type() == sol::type::function) {
			LuaPhaseFunction lua_source = lua_source_objs[i].as<sol::function>(); 
			auto source_func = [lua_source](const mfem::Vector &x, const mfem::Vector &Omega) {
				return lua_source(x(0), x(1), x(2), Omega(0), Omega(1), Omega(2)); 
			};
			source_list[i] = new FunctionGrayCoefficient(source_func); 
		}
	}

	// print materials list to cout 
	out << YAML::Key << "materials" << YAML::Value << YAML::BeginSeq; 
	for (auto i=0; i<attr_list.size(); i++) {
		out << YAML::BeginMap << YAML::Key << "name" << YAML::Value << attr_list[i]; 
		out << YAML::Key << "attribute" << YAML::Value << i+1; 
		out << YAML::Key << "total" << YAML::Value << total_list(i); 
		out << YAML::Key << "scattering" << YAML::Value << scattering_list(i); 
		out << YAML::Key << "source" << YAML::Value; 
		if (lua_source_objs[i].get_type() == sol::type::number) {
			out << lua_source_objs[i].as<double>(); 
		} else {
			out << "function"; 
		}
		out << YAML::EndMap; 
	}
	out << YAML::EndSeq; 

	// map string material id to integer
	// start from 1 since MFEM expects attributes to be >0
	std::unordered_map<std::string,int> attr_map; 
	for (int i=0; i<attr_list.size(); i++) {
		attr_map[attr_list[i]] = i+1; 
	}

	// --- extract list of boundary conditions --- 
	std::vector<std::string> bdr_attr_list; 
	sol::table bcs = lua["boundary_conditions"]; 
	if (bcs.valid()) {
		for (const auto &bc : bcs) {
			auto key = bc.first.as<std::string>(); 
			bdr_attr_list.push_back(key); 
		}
	} else { MFEM_ABORT("boundary conditions not defined"); }

	// get values 
	auto nbattr = bdr_attr_list.size(); 
	std::vector<sol::object> lua_bc_objs(nbattr); 
	mfem::Array<PhaseSpaceCoefficient*> inflow_list(nbattr); 
	int reflection_bdr_attr = -1; 
	for (auto i=0; i<bdr_attr_list.size(); i++) {
		sol::table data = bcs[bdr_attr_list[i].c_str()]; 
		std::string type = data["type"]; 
		if (type == "inflow") {
			sol::object value = data["value"]; 
			lua_bc_objs[i] = value; // keep lua data in scope  
			if (value.get_type() == sol::type::number) {
				auto val = value.as<double>(); 
				inflow_list[i] = new ConstantPhaseSpaceCoefficient(val); 
			} else if (value.get_type() == sol::type::function) {
				auto lua_func = value.as<LuaPhaseFunction>();
				auto func = [lua_func](const mfem::Vector &x, const mfem::Vector &Omega) {
					return lua_func(x(0), x(1), x(2), Omega(0), Omega(1), Omega(2)); 
				};
				inflow_list[i] = new FunctionGrayCoefficient(func);  
			}			
		} else if (type == "reflective") {
			if (reflection_bdr_attr<0) {
				reflection_bdr_attr = i+1; 
			}
			inflow_list[i] = nullptr; 
		} 
	}

	// print list to screen 
	out << YAML::Key << "boundary conditions" << YAML::Value << YAML::BeginSeq; 
	for (auto i=0; i<bdr_attr_list.size(); i++) {
		out << YAML::BeginMap; 
		out << YAML::Key << "name" << YAML::Value << bdr_attr_list[i]; 
		out << YAML::Key << "type" << YAML::Value << ((inflow_list[i]) ? "inflow" : "reflective"); 
		if (inflow_list[i]) {
			out << YAML::Key << "value" << YAML::Value; 
			if (lua_bc_objs[i].get_type() == sol::type::number) {
				out << lua_bc_objs[i].as<double>(); 
			} else {
				out << "function"; 
			}			
		}
		out << YAML::Key << "bdr attribute" << YAML::Value << i+1; 
		out << YAML::EndMap; 
	}
	out << YAML::EndSeq; 

	// map string to bdr attribute integer 
	// start from 1 since MFEM expects >0
	std::unordered_map<std::string,int> bdr_attr_map; 
	for (int i=0; i<bdr_attr_list.size(); i++) {
		bdr_attr_map[bdr_attr_list[i]] = i+1; 
	}

	// --- make mesh and solution spaces --- 
	auto mesh_node = lua["mesh"]; 
	sol::optional<std::string> fname = mesh_node["file"]; 
	mfem::Mesh smesh; 
	out << YAML::Key << "mesh" << YAML::Value << YAML::BeginMap; 
	// load from a mesh file 
	if (fname) {
		smesh = mfem::Mesh::LoadFromFile(fname.value(), 1, 1);
		int refinements = mesh_node["refinements"].get_or(0); 
		for (int r=0; r<refinements; r++) smesh.UniformRefinement(); 			

		out << YAML::Key << "file name" << YAML::Value << realpath(fname.value().c_str(), nullptr); 
		out << YAML::Key << "uniform refinements" << YAML::Value << refinements; 
	} 

	// create a cartesian mesh from extents and elements/axis 
	else {
		sol::table ne = mesh_node["num_elements"]; 
		sol::table extents = mesh_node["extents"]; 
		assert(ne.size() == extents.size()); 
		auto num_dim = ne.size(); 
		sol::optional<std::string> element_type_avail = mesh_node["element_type"]; 
		std::string eltype_str = "segment"; 
		if (element_type_avail) eltype_str = element_type_avail.value(); 
		if (num_dim==1) {
			smesh = mfem::Mesh::MakeCartesian1D(ne[1], extents[1]); 
		} else if (num_dim==2) {
			mfem::Element::Type eltype = mfem::Element::QUADRILATERAL; 
			if (element_type_avail) {
				std::string type = element_type_avail.value(); 
				if (type == "quadrilateral") {
					eltype = mfem::Element::QUADRILATERAL; 
				} else if (type == "triangle") {
					eltype = mfem::Element::TRIANGLE; 
				} else { MFEM_ABORT("element type " << type << " not defined for dim = " << num_dim); }
			} else { eltype_str = "quadrilateral"; }
			smesh = mfem::Mesh::MakeCartesian2D(ne[1], ne[2], eltype, true, extents[1], extents[2], false); 
		} else if (num_dim==3) {
			mfem::Element::Type eltype = mfem::Element::HEXAHEDRON; 
			if (element_type_avail) {
				std::string type = element_type_avail.value(); 
				if (type == "hexahedron") {
					eltype = mfem::Element::HEXAHEDRON; 
				} else if (type == "tetrahedron") {
					eltype = mfem::Element::TETRAHEDRON; 
				} else { MFEM_ABORT("element type " << type << " not defined for dim = " << num_dim); }
			} else { eltype_str = "hexahedron"; }
			smesh = mfem::Mesh::MakeCartesian3D(ne[1], ne[2], ne[3], eltype, extents[1], extents[2], extents[3], false); 
		} else { MFEM_ABORT("dim = " << num_dim << " not supported"); }

		out << YAML::Key << "extents" << YAML::Value << YAML::Flow << YAML::BeginSeq; 
		for (auto i=1; i<=extents.size(); i++) {
			out << (double)extents[i]; 
		} 
		out << YAML::EndSeq; 

		out << YAML::Key << "elements/axis" << YAML::Value << YAML::Flow << YAML::BeginSeq; 
		for (auto i=1; i<=ne.size(); i++) {
			out << (int)ne[i]; 
		} 
		out << YAML::EndSeq; 		
		out << YAML::Key << "element type" << YAML::Value << eltype_str; 
	}
	const auto dim = smesh.Dimension(); 

	// print mesh characteristics 
	double hmin, hmax, kmin, kmax; 
	smesh.GetCharacteristics(hmin, hmax, kmin, kmax); 
	out << YAML::Key << "dim" << YAML::Value << dim; 
	out << YAML::Key << "elements" << YAML::Value << smesh.GetNE(); 
	out << YAML::Key << "min mesh size" << YAML::Value << hmin; 
	out << YAML::Key << "max mesh size" << YAML::Value << hmax; 
	out << YAML::Key << "min mesh conditioning" << YAML::Value << kmin; 
	out << YAML::Key << "max mesh conditioning" << YAML::Value << kmax; 
	out << YAML::Key << "MPI ranks" << YAML::Value << mfem::Mpi::WorldSize(); 
	out << YAML::EndMap; 

	// --- assign materials to elements --- 
	sol::function geom_func = lua["material_map"]; 
	for (int e=0; e<smesh.GetNE(); e++) {
		double c[3]; 
		mfem::Vector cvec(c, dim); 
		smesh.GetElementCenter(e, cvec);
		std::string attr_name = geom_func(c[0], c[1], c[2]); 
		if (!attr_map.contains(attr_name)) {
			MFEM_ABORT("material named \"" << attr_name << "\" not defined"); 
		}
		int attr = attr_map[attr_name]; 
		smesh.SetAttribute(e, attr); 
	}			

	// --- assign boundary conditions to boundary elements --- 
	sol::function bdr_func = lua["boundary_map"]; 
	for (int e=0; e<smesh.GetNBE(); e++) {
		const mfem::Element &el = *smesh.GetBdrElement(e); 
		int geom = smesh.GetBdrElementGeometry(e);
		mfem::ElementTransformation &trans = *smesh.GetBdrElementTransformation(e); 
		double c[3]; 
		mfem::Vector cvec(c, dim);  
		trans.Transform(mfem::Geometries.GetCenter(geom), cvec); 
		std::string attr_name = bdr_func(c[0], c[1], c[2]); 
		if (!bdr_attr_map.contains(attr_name)) {
			MFEM_ABORT("boundary condition named \"" << attr_name << "\" not defined"); 
		}
		smesh.SetBdrAttribute(e, bdr_attr_map[attr_name]); 
	}

	// --- create parallel mesh --- 
	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh); 
	mesh.ExchangeFaceNbrData(); // create parallel communication data needed for sweep 
	mesh.SetAttributes(); 

	// --- load algorithmic parameters --- 
	sol::table driver = lua["driver"]; 
	const int fe_order = driver["fe_order"]; 
	const int sn_order = driver["sn_order"]; 
	sol::optional<std::string> basis_type_avail = driver["basis_type"]; 
	std::string basis_type_string; 
	if (basis_type_avail) {
		basis_type_string = basis_type_avail.value(); 
	} 
	// default to lobatto
	// lobatto is better for the moment solver's preconditioners 
	else {
		basis_type_string = "lobatto"; 
	}

	// --- build solution space --- 
	// DG space for transport solution 
	int basis_type; 
	if (basis_type_string == "legendre") {
		basis_type = mfem::BasisType::GaussLegendre; 
	} else if (basis_type_string == "lobatto") {
		basis_type = mfem::BasisType::GaussLobatto; 
	} else { MFEM_ABORT("basis type " << basis_type_string << " not supported"); }
	mfem::L2_FECollection fec(fe_order, dim, basis_type); 
	mfem::ParFiniteElementSpace fes(&mesh, &fec); // scalar finite element space 
	mfem::ParFiniteElementSpace vfes(&mesh, &fec, dim); // vector finite element space, dim copies of fes 
	fes.ExchangeFaceNbrData(); // create parallel degree of freedom maps used in sweep 

	// piecewise constant used for plotting and storing cross section data 
	mfem::L2_FECollection fec0(0, dim); 
	mfem::ParFiniteElementSpace fes0(&mesh, &fec0); 

	// --- create coefficients for input material data ---
	// hack using grid function coefficients since attribute system 
	// broken in parallel 
	// MIP diffusion not getting correct diffusion coefficient on shared parallel faces 
	// instead project coefficient onto piecewise constant grid function, exchange parallel 
	// so value available across shared faces 
	// create total interaction cross section 
	mfem::PWConstCoefficient total_attr(total_list); 
	mfem::ParGridFunction total_gf(&fes0); 
	total_gf.ProjectCoefficient(total_attr); 
	total_gf.ExchangeFaceNbrData(); 
	mfem::GridFunctionCoefficient total(&total_gf); 

	// scattering 
	mfem::PWConstCoefficient scattering_attr(scattering_list); 
	mfem::ParGridFunction scattering_gf(&fes0); 
	scattering_gf.ProjectCoefficient(scattering_attr); 
	scattering_gf.ExchangeFaceNbrData(); 
	mfem::GridFunctionCoefficient scattering(&scattering_gf); 

	// piecewise constant isotropic source 
	mfem::Array<int> attrs(nattr); 
	for (int i=0; i<nattr; i++) { attrs[i] = i+1; }
	PWPhaseSpaceCoefficient source(attrs, source_list); 
	mfem::SumCoefficient absorption(total, scattering, 1, -1); 
	mfem::Array<int> battrs(nbattr); 
	for (int i=0; i<nbattr; i++) { battrs[i] = i+1; }
	PWPhaseSpaceCoefficient inflow(battrs, inflow_list); 

	// --- angular quadrature rule --- 
	LevelSymmetricQuadrature quad(sn_order, dim); 
	const auto Nomega = quad.Size(); 

	// --- transport vector setup --- 
	TransportVectorExtents psi_ext(1, Nomega, fes.GetVSize());
	const auto psi_size = TotalExtent(psi_ext); 
	MomentVectorExtents phi_ext(1, 1, fes.GetVSize());
	const auto phi_size = TotalExtent(phi_ext);
	MomentVectorExtents moments_ext(1, dim+1, fes.GetVSize()); // scalar flux and dim components of current 
	const auto moments_size = TotalExtent(moments_ext); 
	DiscreteToMoment D(quad, psi_ext, phi_ext); // integrates over angle psi -> phi 
	DiscreteToMoment Dlin_aniso(quad, psi_ext, moments_ext); // forms psi -> [phi,J] 

	const auto psi_size_global = mesh.ReduceInt(psi_size); 
	const auto phi_size_global = mesh.ReduceInt(phi_size); 

	// --- create solver objects from Lua input --- 
	sol::table solver = driver["solver"]; 
	sol::optional<sol::table> accel_avail = driver["acceleration"]; 
	sol::optional<sol::table> prec_avail = driver["preconditioner"]; 
	auto *outer_solver = io::CreateIterativeSolver(solver, MPI_COMM_WORLD);
	if (!outer_solver) { MFEM_ABORT("outer solver required"); }
	if (accel_avail and prec_avail) { MFEM_ABORT("cannot use both preconditioning and acceleration"); }
	sol::table inner_solver_table; 
	mfem::IterativeSolver *inner_it_solver = nullptr; 
	if (accel_avail) {
		if (not(dynamic_cast<FixedPointIterationSolver*>(outer_solver) 
			or dynamic_cast<mfem::KINSolver*>(outer_solver))) {
			MFEM_ABORT("must use fixed point-type solvers for moment methods"); 
		}
		sol::optional<sol::table> inner_solver_table_avail = accel_avail.value()["solver"]; 
		if (inner_solver_table_avail) {
			inner_solver_table = inner_solver_table_avail.value();
		}
		// default to direct if solver table not provided 
		else {
			inner_solver_table = accel_avail.value().create_with("type", "direct"); 
		}
		// can be nullptr if direct solver specified 
		inner_it_solver = io::CreateIterativeSolver(inner_solver_table, MPI_COMM_WORLD); 
	}
	if (prec_avail) {
		if (dynamic_cast<FixedPointIterationSolver*>(outer_solver) 
			or dynamic_cast<mfem::KINSolver*>(outer_solver)) {
			MFEM_ABORT("cannot use fixed point-type solvers for transport iteration"); 
		}
		sol::optional<sol::table> inner_solver_table_avail = prec_avail.value()["solver"]; 
		if (inner_solver_table_avail) {
			inner_solver_table = inner_solver_table_avail.value(); 
		} 
		// default to direct if solver table not provided 
		else {
			inner_solver_table = prec_avail.value().create_with("type", "direct"); 
		}
		// can be nullptr if direct solver specified 
		inner_it_solver = io::CreateIterativeSolver(inner_solver_table, MPI_COMM_WORLD); 
	}
	if (!accel_avail) {
		if (dynamic_cast<FixedPointIterationSolver*>(outer_solver) 
			or dynamic_cast<mfem::KINSolver*>(outer_solver)) {
			MFEM_ABORT("cannot use fixed point-type solvers for transport iteration"); 
		}
	}
	// generic operator in case inner solver is SuperLU or product operator etc 
	mfem::Operator *inner_solver = inner_it_solver; 
	// sweep setup options 
	sol::optional<sol::table> sweep_opts_avail = driver["sweep_opts"]; 

	// --- output algorithmic options used --- 
	out << YAML::Key << "driver" << YAML::Value << YAML::BeginMap; 
		out << YAML::Key << "fe order" << YAML::Value << fe_order; 
		out << YAML::Key << "sn order" << YAML::Value << sn_order; 
		out << YAML::Key << "num angles" << YAML::Value << Nomega; 			
		out << YAML::Key << "psi size" << YAML::Value << psi_size_global;
		out << YAML::Key << "phi size" << YAML::Value << phi_size_global;
		out << YAML::Key << "basis type" << YAML::Value << basis_type_string; 
		out << YAML::Key << "solver" << YAML::Value << solver; 
		if (sweep_opts_avail) {
			out << YAML::Key << "sweep options" << YAML::Value << sweep_opts_avail.value();
		}

	// --- sweep setup --- 
	// global scattering operator 
	mfem::BilinearForm Ms_form(&fes); 
	Ms_form.AddDomainIntegrator(new mfem::MassIntegrator(scattering)); 
	Ms_form.Assemble(); 
	Ms_form.Finalize();

	// allocate transport vector + views 
	mfem::Vector psi(psi_size);
	psi = 0.0; 
	mfem::Vector moment_solution(moments_size);  
	mfem::ParGridFunction phi(&fes), J(&vfes, moment_solution, phi_size);

	// initial guess 
	D.Mult(psi, phi); 

	// form fixed source term 
	mfem::Vector source_vec(psi_size); 
	source_vec = 0.0; 
	TransportVectorView source_vec_view(source_vec.GetData(), psi_ext); 
	FormTransportSource(fes, quad, source, inflow, source_vec_view); 

	// build sweep operator 
	InverseAdvectionOperator Linv(fes, quad, psi_ext, total, inflow, reflection_bdr_attr); 
	if (sweep_opts_avail) {
		sol::table sweep_opts = sweep_opts_avail.value(); 
		bool write_graph = sweep_opts["write_graph"].get_or(false); 
		if (write_graph) 
			Linv.WriteGraphToDot("graph"); 
		sol::optional<int> send_buffer_size = sweep_opts["send_buffer_size"]; 
		if (send_buffer_size) 
			Linv.SetSendBufferSize(send_buffer_size.value()); 
	}

	// common parameters to discretization
	mfem::Vector normal(dim); 
	normal = 0.0; normal(0) = 1.0; 
	double alpha = ComputeAlpha(quad, normal);
	mfem::Vector beta(dim); 
	for (int d=0; d<dim; d++) { beta(d) = d+1; }

	// standard sn iteration with preconditioning if available 
	if (!accel_avail) {
		DiffusionSyntheticAccelerationOperator *prec = nullptr; // applies I + D^{-1} Ms
		mfem::HypreParMatrix *dsa_mat = nullptr; // store diffusion system 
		mfem::HypreBoomerAMG *amg = nullptr; // preconditioner for diffusion system 
	#ifdef MFEM_USE_SUPERLU
		mfem::SuperLURowLocMatrix *slu_op = nullptr; // operator for direct solves 
	#endif
		BlockDiffusionDiscretization *block_disc = nullptr; 

		// build preconditioner from input spec 
		if (prec_avail) {
			out << YAML::Key << "preconditioner" << YAML::Value << YAML::BeginMap; 
			sol::table prec_table = prec_avail.value(); 
			std::string type = prec_table["type"]; 
			std::transform(type.begin(), type.end(), type.begin(), ::tolower); 
			out << YAML::Key << "type" << YAML::Value << type; 
			if (type == "p1sa") {
			#ifdef MFEM_USE_SUPERLU
				if (inner_it_solver) { MFEM_ABORT("only direct available for P1"); }
				auto p1disc = std::unique_ptr<mfem::BlockOperator>(CreateP1DiffusionDiscretization(
					fes, vfes, total, absorption, alpha, reflection_bdr_attr)); 
				auto mono = std::unique_ptr<mfem::HypreParMatrix>(BlockOperatorToMonolithic(*p1disc)); 
				slu_op = new mfem::SuperLURowLocMatrix(*mono); 
				auto *slu = new mfem::SuperLUSolver(*slu_op); 
				slu->SetPrintStatistics(false); 
				auto *ceo = new ComponentExtractionOperator(p1disc->RowOffsets(), 1); 
				auto *ceo_t = new mfem::TransposeOperator(*ceo); 
				// solve 2x2 but source and solution and scalar flux only 
				inner_solver = new mfem::TripleProductOperator(ceo, slu, ceo_t, true, true, true); 
			#else
				MFEM_ABORT("super LU required for P1"); 
			#endif
			} 
			else if (type == "ldgsa") {
				bool scale_stabilization = prec_table["scale_stabilization"].get_or(true); 
				const auto bc_type = io::GetDiffusionBCType(prec_table, "bc_type", "half range"); 
				block_disc = new BlockLDGDiffusionDiscretization(fes, vfes, total, absorption, alpha, 
					beta, scale_stabilization, reflection_bdr_attr, bc_type); 
				const auto &S = block_disc->SchurComplement(); 

				// iterative solve
				if (inner_it_solver) {
					amg = new mfem::HypreBoomerAMG(S); 
					inner_it_solver->SetOperator(S); 
					inner_it_solver->SetPreconditioner(*amg); 
				} 

				// direct solve 
				else {
				#ifdef MFEM_USE_SUPERLU
					slu_op = new mfem::SuperLURowLocMatrix(S); 
					auto *slu = new mfem::SuperLUSolver(*slu_op); 
					slu->SetPrintStatistics(false); 
					inner_solver = slu; 
				#else 
					MFEM_ABORT("superlu required for direct option"); 
				#endif 
				}

				out << YAML::Key << "scale stabilization" << YAML::Value << scale_stabilization; 
				out << YAML::Key << "boundary condition type" << YAML::Value << std::string(prec_table["bc_type"]); 
			}
			else if (type == "mip") {
				double kappa = prec_table["kappa"].get_or(pow(fe_order+1,2)); 
				bool scale_stabilization = prec_table["scale_stabilization"].get_or(true); 
				bool lower_bound = prec_table["bound_stabilization_below"].get_or(true); 
				const auto bc_type = io::GetDiffusionBCType(prec_table, "bc_type", "full range"); 
				block_disc = new BlockIPDiffusionDiscretization(fes, vfes, total, absorption, alpha, 
					kappa, lower_bound, scale_stabilization, reflection_bdr_attr, bc_type); 
				const auto &S = block_disc->SchurComplement(); 

				// iterative solve
				if (inner_it_solver) {
					amg = new mfem::HypreBoomerAMG(S); 
					inner_it_solver->SetOperator(S); 
					inner_it_solver->SetPreconditioner(*amg); 
				} 

				// direct solve 
				else {
				#ifdef MFEM_USE_SUPERLU
					slu_op = new mfem::SuperLURowLocMatrix(S); 
					auto *slu = new mfem::SuperLUSolver(*slu_op); 
					slu->SetPrintStatistics(false); 
					inner_solver = slu; 
				#else 
					MFEM_ABORT("superlu required for direct option"); 
				#endif 
				}

				out << YAML::Key << "kappa" << YAML::Value << kappa; 
				out << YAML::Key << "scale stabilization" << YAML::Value << scale_stabilization; 
				out << YAML::Key << "bound stabilization from below" << YAML::Value << lower_bound; 
				out << YAML::Key << "boundary condition type" << YAML::Value << std::string(prec_table["bc_type"]); 
			}
			else MFEM_ABORT("dsa type " << type << " not defined"); 

			// setup AMG object 
			if (amg) {
				amg->SetPrintLevel(0); 
				sol::optional<sol::table> amg_opts = prec_table["solver"]["amg_opts"]; 
				if (amg_opts) io::SetAMGOptions(amg_opts.value(), *amg); 				
			}

			// create DSA operator 
			// I + D^{-1} Ms 
			prec = new DiffusionSyntheticAccelerationOperator(*inner_solver, Ms_form); 

			out << YAML::Key << "solver" << YAML::Value << inner_solver_table; 
			out << YAML::EndMap; // end preconditioner map 
		}

		else {
			out << YAML::Key << "preconditioner" << YAML::Value << "none"; 
		}
		out << YAML::EndMap; // end driver map 

		// build main transport iteration operator 
		// applies I - DL^{-1} MS 
		// uses psi as storage vector 
		TransportOperator T(D, Linv, Ms_form, psi); 

		// form source for schur complement solve 
		// b -> D L^{-1} b
		Linv.Mult(source_vec, psi); 
		mfem::Vector schur_source(phi_size); 
		D.Mult(psi, schur_source);

		if (prec_avail) { outer_solver->SetPreconditioner(*prec); }
		outer_solver->SetOperator(T); 
		TransportIterationMonitor monitor(out, T, prec, dynamic_cast<mfem::IterativeSolver*>(inner_solver)); 
		outer_solver->SetMonitor(monitor); 
		outer_solver->Mult(schur_source, phi); 

		// extra sweep to get psi 
		mfem::Vector scat_source(phi_size); 
		Ms_form.Mult(phi, scat_source); 
		D.MultTranspose(scat_source, psi); 
		psi += source_vec; 
		Linv.Mult(psi, psi); 
		// compute phi and J 
		Dlin_aniso.Mult(psi, moment_solution); 

		// output iteration info 
		out << YAML::Key << "outer iterations" << YAML::Value << outer_solver->GetNumIterations();
		if (monitor.inner_it.Size()) {
			out << YAML::Key << "inner iteration" << YAML::Value << YAML::BeginMap; 
			out << YAML::Key << "max inner" << YAML::Value << monitor.inner_it.Max();
			out << YAML::Key << "min inner" << YAML::Value << monitor.inner_it.Min(); 
			out << YAML::Key << "avg inner" << YAML::Value << (double)monitor.inner_it.Sum()/monitor.inner_it.Size(); 		
			out << YAML::EndMap; 
		}
		if (prec) delete prec; 
		if (amg) delete amg; 
	#ifdef MFEM_USE_SUPERLU
		if (slu_op) delete slu_op; 
	#endif
		if (dsa_mat) delete dsa_mat; 
		if (block_disc) delete block_disc; 
	}

	// moment-based solve
	else {
		// space for current 
		mfem::Array<int> offsets; 
		mfem::BlockVector block_x; 
		mfem::Operator *smm = nullptr; 
		mfem::HypreBoomerAMG *amg = nullptr; 
		BlockDiffusionDiscretization *block_disc = nullptr; 
	#ifdef MFEM_USE_SUPERLU
		mfem::SuperLURowLocMatrix *slu_op = nullptr; 
	#endif

		sol::table accel = accel_avail.value(); 
		std::string type = accel["type"]; 
		out << YAML::Key << "acceleration" << YAML::Value << YAML::BeginMap; 
		out << YAML::Key << "type" << YAML::Value << type; 
		std::transform(type.begin(), type.end(), type.begin(), ::toupper); 
		if (type == "LDGSMM") {
			bool consistent = accel["consistent"].get_or(false); 
			bool scale_stabilization = accel["scale_stabilization"].get_or(true); 
			bool stab_bound = accel["bound_stabilization_below"].get_or(true); 
			auto bc_type = io::GetDiffusionBCType(accel, "bc_type", "full range"); 
			auto *block_ldg = new BlockLDGDiffusionDiscretization(fes, vfes, total, absorption, alpha, beta, 
				scale_stabilization, reflection_bdr_attr, bc_type); 
			block_disc = block_ldg; 
			const auto &S = block_disc->SchurComplement(); 

			mfem::Operator *source_op; 
			if (consistent) {
				source_op = new ConsistentLDGSMMSourceOperator(*block_ldg, quad, psi_ext, source_vec_view); 
			} else {
				source_op = new BlockDiffusionSMMSourceOperator(fes, vfes, quad, psi_ext, source, inflow, 
					alpha, reflection_bdr_attr, bc_type); 				
			}

			// iterative solve
			if (inner_it_solver) {
				amg = new mfem::HypreBoomerAMG(S); 
				inner_it_solver->SetOperator(S); 
				inner_it_solver->SetPreconditioner(*amg); 				
			} 

			// direct solve 
			else {
			#ifdef MFEM_USE_SUPERLU
				slu_op = new mfem::SuperLURowLocMatrix(S); 
				auto *slu = new mfem::SuperLUSolver(*slu_op); 
				slu->SetPrintStatistics(false); 
				inner_solver = slu;
			#else 
				MFEM_ABORT("superlu required for direct option"); 
			#endif 
			}

			auto *inv_ldg = new InverseBlockDiffusionOperator(*block_disc, *inner_solver); 
			offsets = block_disc->GetOffsets(); // copy offsets 
			block_x.Update(offsets); // set size of block_x 
			// extract scalar flux, storing [J,phi] in block_x 
			auto *block_extract = new SubBlockExtractionOperator(block_x, 1); 
			// psi -> SMM source -> diffusion solution -> extract phi from block vector 
			// use allium version of triple product operator to ensure temp vectors 
			// initialized to zero (for first initial guess when iterative_mode = true)
			smm = new TripleProductOperator(block_extract, inv_ldg, source_op, true, true, true); 

			// output LDG specific options 
			out << YAML::Key << "consistent" << YAML::Value << consistent; 
			out << YAML::Key << "scale stabilization" << YAML::Value << scale_stabilization; 
			out << YAML::Key << "bound stabilization from below" << YAML::Value << stab_bound; 
			out << YAML::Key << "boundary condition type" << YAML::Value << std::string(accel["bc_type"]); 
		} 
		else if (type == "IPSMM") {
			bool consistent = accel["consistent"].get_or(false); 
			double kappa = accel["kappa"].get_or(pow(fe_order+1,2)); 
			// scale interior penalty coefficient by diffusion coefficient 
			bool scale_stabilization = accel["scale_stabilization"].get_or(true); 
			// use kappa = alpha/2 if regular kappa goes below alpha/2 
			bool stab_bound = accel["bound_stabilization_below"].get_or(true); 
			const auto bc_type = io::GetDiffusionBCType(accel, "bc_type", "full range"); 

			auto *block_ip = new BlockIPDiffusionDiscretization(fes, vfes, total, absorption, alpha, kappa, stab_bound, 
				scale_stabilization, reflection_bdr_attr, bc_type); 
			block_disc = block_ip; 
			const auto &S = block_disc->SchurComplement(); 

			mfem::Operator *source_op; 
			if (consistent) {
				source_op = new ConsistentIPSMMSourceOperator(*block_ip, quad, psi_ext, source_vec_view); 
			} else {
				source_op = new BlockDiffusionSMMSourceOperator(fes, vfes, quad, psi_ext, source, inflow, alpha, 
					reflection_bdr_attr, bc_type); 				
			}

			// iterative solve
			if (inner_it_solver) {
				amg = new mfem::HypreBoomerAMG(S); 
				inner_it_solver->SetOperator(S); 
				inner_it_solver->SetPreconditioner(*amg); 				
			} 

			// direct solve 
			else {
			#ifdef MFEM_USE_SUPERLU
				slu_op = new mfem::SuperLURowLocMatrix(S); 
				auto *slu = new mfem::SuperLUSolver(*slu_op); 
				slu->SetPrintStatistics(false); 
				inner_solver = slu;
			#else 
				MFEM_ABORT("superlu required for direct option"); 
			#endif 
			}

			auto *inv_ip = new InverseBlockDiffusionOperator(*block_disc, *inner_solver); 
			offsets = block_disc->GetOffsets(); // copy offsets 
			block_x.Update(offsets); // set size of block_x 
			// extract scalar flux, storing [J,phi] in block_x 
			auto *block_extract = new SubBlockExtractionOperator(block_x, 1); 
			// psi -> SMM source -> diffusion solution -> extract phi from block vector 
			// use allium version of triple product operator to ensure temp vectors 
			// initialized to zero (for first initial guess when iterative_mode = true)
			smm = new TripleProductOperator(block_extract, inv_ip, source_op, true, true, true); 

			// output IP specific options 
			out << YAML::Key << "kappa" << YAML::Value << kappa; 
			out << YAML::Key << "consistent" << YAML::Value << consistent; 
			out << YAML::Key << "scale stabilization" << YAML::Value << scale_stabilization; 
			out << YAML::Key << "bound stabilization from below" << YAML::Value << stab_bound; 
			out << YAML::Key << "boundary condition type" << YAML::Value << std::string(accel["bc_type"]); 
		}
		else if (type == "P1SMM") {
		#ifdef MFEM_USE_SUPERLU
			if (inner_it_solver) { MFEM_ABORT("only direct supported for P1"); }
			sol::optional<bool> consistent_avail = accel["consistent"]; 
			if (consistent_avail) {
				if (consistent_avail.value() == false) { MFEM_ABORT("only consistent supported for P1"); }
			}
			auto p1disc = std::unique_ptr<mfem::BlockOperator>(CreateP1DiffusionDiscretization(
				fes, vfes, total, absorption, alpha, reflection_bdr_attr)); 
			auto mono = std::unique_ptr<mfem::HypreParMatrix>(BlockOperatorToMonolithic(*p1disc)); 
			slu_op = new mfem::SuperLURowLocMatrix(*mono); 
			auto *slu = new mfem::SuperLUSolver(*slu_op); 
			slu->SetPrintStatistics(false); 

			auto *source_op = new ConsistentSMMSourceOperator(fes, vfes, quad, psi_ext, source_vec_view, alpha, reflection_bdr_attr); 
			offsets = p1disc->RowOffsets(); // copy offsets 
			block_x.Update(offsets); // set size of block_x 
			// extract scalar flux, storing [J,phi] in block_x 
			auto *block_extract = new SubBlockExtractionOperator(block_x, 1); 
			// build SM source, invert P1, extract phi 
			smm = new mfem::TripleProductOperator(block_extract, slu, source_op, true, true, true); 

			out << YAML::Key << "consistent" << YAML::Value << true; 
		#else
			MFEM_ABORT("super LU required for P1"); 
		#endif
		} 
		else { MFEM_ABORT("acceleration type " << type << " not defined"); }

		// set AMG object 
		if (amg) {
			amg->SetPrintLevel(0); 
			sol::optional<sol::table> amg_opts = inner_solver_table["amg_opts"]; 
			if (amg_opts) io::SetAMGOptions(amg_opts.value(), *amg);	
		}

		out << YAML::Key << "solver" << YAML::Value << inner_solver_table; 
		out << YAML::EndMap; // end acceleration map 
		out << YAML::EndMap; // end driver map 

		MomentMethodFixedPointOperator G(D, Linv, Ms_form, *smm, source_vec, psi); 
		outer_solver->SetOperator(G); 
		io::SundialsUserCallbackData sundials_data(out, G, dynamic_cast<mfem::IterativeSolver*>(inner_solver)); 
		auto *sundials = dynamic_cast<mfem::SundialsSolver*>(outer_solver);
		if (sundials) {
			KINSetInfoHandlerFn(sundials->GetMem(), io::SundialsCallbackFunction, &sundials_data); 
			KINSetErrHandlerFn(sundials->GetMem(), io::SundialsErrorFunction, &sundials_data); 
		}

		MomentMethodIterationMonitor monitor(out, G, dynamic_cast<mfem::IterativeSolver*>(inner_solver)); 
		outer_solver->SetMonitor(monitor); 
		mfem::Vector blank; 
		outer_solver->Mult(blank, phi); 
		out << YAML::Key << "outer iterations" << YAML::Value << outer_solver->GetNumIterations();
		mfem::Array<int> *inner_it = nullptr; 
		if (sundials) {
			inner_it = &sundials_data.inner_it; 
		} else {
			inner_it = &monitor.inner_it; 
		}
		if (inner_it->Size()) {
			out << YAML::Key << "inner iteration" << YAML::Value << YAML::BeginMap; 
			out << YAML::Key << "min inner" << YAML::Value << inner_it->Min(); 
			out << YAML::Key << "max inner" << YAML::Value << inner_it->Max();
			out << YAML::Key << "avg inner" << YAML::Value << (double)inner_it->Sum()/inner_it->Size(); 		
			out << YAML::EndMap; 
		}

		// compute "consistency" between SN and moment solution 
		// for scalar flux and current 
		J = block_x.GetBlock(0); 
		mfem::Vector x_sn(moments_size); 
		Dlin_aniso.Mult(psi, x_sn); 
		mfem::ParGridFunction phi_sn(&fes, x_sn, 0); 
		mfem::ParGridFunction J_sn(&vfes, x_sn, fes.GetVSize()); 
		mfem::GridFunctionCoefficient phi_snc(&phi_sn); 
		double consistency_phi = phi.ComputeL2Error(phi_snc); 
		mfem::GridFunctionCoefficient J_snc(&J_sn); 
		double consistency_J = J.ComputeL2Error(J_snc); 
		std::stringstream ss; 
		out << YAML::Key << "consistency" << YAML::Value << YAML::BeginMap; 
			ss << std::setprecision(3) << std::scientific << consistency_phi; 
			out << YAML::Key << "scalar flux" << YAML::Value << ss.str(); 
			ss.str(""); 
			ss << consistency_J; 
			out << YAML::Key << "current" << YAML::Value << ss.str(); 
		out << YAML::EndMap; 

		if (smm) delete smm; 
		if (amg) delete amg; 
		if (block_disc) delete block_disc; 
	#ifdef MFEM_USE_SUPERLU
		if (slu_op) delete slu_op; 
	#endif
	}

	// --- clean up hanging pointers --- 
	delete outer_solver; 
	if (inner_solver) delete inner_solver; 
	for (int i=0; i<nattr; i++) { delete source_list[i]; }
	for (int i=0; i<nbattr; i++) { delete inflow_list[i]; }

	// --- compute error if exact solution provided --- 
	sol::optional<sol::function> solution_func_avail = lua["scalar_flux_solution"]; 
	sol::optional<sol::function> current_func_avail = lua["current_solution"]; 
	if (solution_func_avail or current_func_avail) {
		out << YAML::Key << "L2 error" << YAML::Value << YAML::BeginMap; 
	}

	if (solution_func_avail) {
		auto solution_func = solution_func_avail.value(); 
		auto solution_lam = [&solution_func](const mfem::Vector &x) {
			double pos[3]; 
			for (auto d=0; d<x.Size(); d++) { pos[d] = x(d); }
			return solution_func(pos[0], pos[1], pos[2]); 
		}; 
		mfem::FunctionCoefficient phi_coef(solution_lam); 
		double l2 = phi.ComputeL2Error(phi_coef); 
		std::stringstream ss; 
		ss << std::setprecision(3) << std::scientific << l2; 
		out << YAML::Key << "scalar flux" << YAML::Value << ss.str(); 
	}

	if (current_func_avail) {
		auto solution_func = current_func_avail.value(); 
		auto solution_lam = [&solution_func](const mfem::Vector &x) {
			double pos[3]; 
			for (auto d=0; d<x.Size(); d++) { pos[d] = x(d); }
			return solution_func(pos[0], pos[1], pos[2]); 
		}; 
		mfem::FunctionCoefficient Jcoef(solution_lam); 
		double l2 = J.ComputeL2Error(Jcoef); 
		std::stringstream ss; 
		ss << std::setprecision(3) << std::scientific << l2; 
		out << YAML::Key << "current" << YAML::Value << ss.str(); 		
	}

	if (solution_func_avail or current_func_avail) {
		out << YAML::EndMap; 
	}

	// --- output to paraview --- 
	sol::optional<sol::table> output_avail = lua["output"]; 
	if (output_avail) {
		out << YAML::Key << "output" << YAML::Value << YAML::BeginMap; 
		sol::table output = output_avail.value();

		// write to paraview 
		sol::optional<std::string> paraview_avail = output["paraview"]; 
		if (paraview_avail) {
			std::string output_name = paraview_avail.value(); 
			char output_name_resolve[PATH_MAX];
			realpath(output_name.c_str(), output_name_resolve);  	
			out << YAML::Key << "paraview" << YAML::Value << output_name_resolve; 
			mfem::ParGridFunction mesh_part(&fes0); 
			for (int i=0; i<mesh_part.Size(); i++) { mesh_part[i] = rank; }
			mfem::ParaViewDataCollection dc(output_name, &mesh); 
			dc.RegisterField("phi", &phi); 
			dc.RegisterField("J", &J); 
			dc.RegisterField("partition", &mesh_part); 
			dc.RegisterField("total", &total_gf); 
			dc.RegisterField("scattering", &scattering_gf); 
			dc.Save(); 			
		}


		sol::optional<sol::table> tracer_avail = output["tracer"]; 
		if (tracer_avail) {
		#ifdef MFEM_USE_GSLIB 
			sol::table tracer = tracer_avail.value(); 
			auto ntracers = tracer.size(); 
			mfem::Vector pos(dim*ntracers); 
			for (int i=0; i<ntracers; i++) {
				sol::table tracer_pts = tracer[i+1]; 
				for (int d=0; d<tracer_pts.size(); d++) {
					pos(dim*i + d) = tracer_pts[d+1]; 
				}
			}
			mfem::FindPointsGSLIB finder(MPI_COMM_WORLD);
			mesh.EnsureNodes();  
			finder.Setup(mesh); 
			finder.FindPoints(pos, mfem::Ordering::byVDIM); 
			const auto &gscodes = finder.GetCode(); 
			mfem::Vector phi_tracer, Jtracer; 
			finder.Interpolate(phi, phi_tracer); 
			finder.Interpolate(J, Jtracer); 
			out.SetDoublePrecision(16); 
			out << YAML::Key << "tracer" << YAML::Value << YAML::BeginSeq; 
			for (auto i=0; i<ntracers; i++) {
				if (gscodes[i] == 2) {
					if (mfem::Mpi::Root()) MFEM_WARNING("point " << i << " not found");
					continue; 
				}
				out << YAML::BeginMap << YAML::Value << "position" << YAML::Value << YAML::Flow << YAML::BeginSeq; 
				for (auto d=0; d<dim; d++) { out << pos(dim*i+d); }
				out << YAML::EndSeq; 
				out << YAML::Key << "scalar flux" << YAML::Value << phi_tracer(i); 
				out << YAML::Key << "current" << YAML::Value << YAML::Flow << YAML::BeginSeq; 
				for (auto d=0; d<dim; d++) { out << Jtracer(ntracers*d + i); }
				out << YAML::EndSeq; 
				out << YAML::EndMap; 
			}
			out << YAML::EndSeq; 
			finder.FreeData(); 
			out.SetDoublePrecision(3); 
		#else 
			if (root) MFEM_WARNING("gslib required for tracer output"); 
		#endif
		}

		sol::optional<sol::table> lineout_avail = output["lineout"]; 
		if (lineout_avail) {
		#ifdef MFEM_USE_GSLIB
			sol::table lineout = lineout_avail.value(); 
			std::string path = output["lineout_path"].get_or(std::string("lineout.yaml")); 
			out << YAML::Key << "lineout" << YAML::Value << realpath(path.c_str(), nullptr); 
			std::ofstream file_out(path); 
			mfem::OutStream file_out_par(file_out); 
			if (!root) file_out_par.Disable(); 
			YAML::Emitter fout(file_out_par); 
			fout.SetDoublePrecision(16); 
			fout << YAML::BeginMap; 
			for (const auto &it : lineout) {
				fout << YAML::Key << it.first.as<std::string>() << YAML::Value << YAML::BeginMap; 
				sol::table line = it.second; 
				mfem::Vector start(dim), end(dim), dir(dim);
				sol::table start_table = line["start_point"]; 
				sol::table end_table = line["end_point"]; 
				const int npoints = line["num_points"];  
				for (int d=0; d<dim; d++) {
					start(d) = start_table[d+1]; 
					end(d) = end_table[d+1]; 
				}
				subtract(end, start, dir); 
				double L = dir.Norml2();
				dir *= 1.0 / (npoints-1); 
				double h = 1.0 / (npoints-1);  
				mfem::Vector pts(npoints * dim); 
				mfem::Vector t(npoints); 
				auto pts_view = Kokkos::mdspan(pts.GetData(), npoints, dim); 
				for (int n=0; n<npoints; n++) {
					t(n) = h*n; 
					for (int d=0; d<dim; d++) {
						pts_view(n,d) = start(d) + dir(d) * n; 
					}
				}
				mfem::FindPointsGSLIB finder(MPI_COMM_WORLD);
				mesh.EnsureNodes();  
				finder.Setup(mesh); 
				finder.FindPoints(pts, mfem::Ordering::byVDIM); 
				const auto &gscodes = finder.GetCode(); 
				mfem::Vector phi_line, J_line; 
				finder.Interpolate(phi, phi_line);
				finder.Interpolate(J, J_line);  

				// output locations
				fout << YAML::Key << "x" << YAML::Value << YAML::BeginSeq; 
				for (auto i=0; i<npoints; i++) {
					if (gscodes[i] == 2) {
						if (root) MFEM_WARNING("lineout point " << i << " not found"); 
						continue; 
					}
					fout << YAML::Flow << YAML::BeginSeq; 
					for (auto d=0; d<dim; d++) {
						fout << pts_view(i,d); 						
					}
					fout << YAML::EndSeq; 
				}
				fout << YAML::EndSeq; 
				// solution values 
				fout << YAML::Key << "scalar flux" << YAML::Value << YAML::BeginSeq; 
				for (auto n=0; n<npoints; n++) {
					if (gscodes[n] == 2) continue; 
					fout << phi_line(n); 
				}
				fout << YAML::EndSeq;
				fout << YAML::Key << "current" << YAML::Value << YAML::BeginSeq;  
				for (auto n=0; n<npoints; n++) {
					if (gscodes[n] == 2) continue; 
					fout << YAML::Flow << YAML::BeginSeq; 
					for (auto d=0; d<dim; d++) {
						fout << J_line(d * npoints + n); 
					}
					fout << YAML::EndSeq; 
				}
				fout << YAML::EndSeq; 
				fout << YAML::EndMap; 
			}
			fout << YAML::EndMap; 
			file_out.close(); 
		#else 
			if (root) MFEM_WARNING("gslib required for lineouts"); 
		#endif
		}
		out << YAML::EndMap; // end output map 
	}

	// output wall clock time 
	MPI_Barrier(MPI_COMM_WORLD); 
	timer.Stop(); 
	double time = timer.RealTime(); 
	out << YAML::Key << "wall time" << YAML::Value << io::FormatTimeString(time); 

	// --- end yaml output --- 
	out << YAML::EndMap << YAML::Newline; 
}