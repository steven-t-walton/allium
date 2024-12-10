#include "mfem.hpp"
#include "yaml-cpp/yaml.h"
#include "sol/sol.hpp"

#include "bdr_conditions.hpp"
#include "tvector.hpp"
#include "angular_quadrature.hpp"
#include "sweep.hpp"
#include "transport_op.hpp"
#include "phase_coefficient.hpp"
#include "comment_stream.hpp"
#include "linalg.hpp"
#include "io.hpp"
#include "log.hpp"
#include "lumping.hpp"
#include "moment_discretization.hpp"
#include "smm_source.hpp"
#include "multigroup.hpp"
#include <kinsol/kinsol.h>
#include <regex>

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
	mfem::Array<double> sweep_time, prec_time; 

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
			const auto sweep = T.GetStopWatch().RealTime(); 
			sweep_time.Append(sweep); 
			out << YAML::Key << "sweep" << YAML::Value << io::FormatTimeString(sweep); 
			if (dsa) {
				const auto prec = dsa->GetStopWatch().RealTime(); 				
				prec_time.Append(prec); 
				out << YAML::Key << "preconditioner" << YAML::Value << io::FormatTimeString(prec); 
			}
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
	mfem::Array<double> sweep_time, moment_time; 
	MomentMethodIterationMonitor(YAML::Emitter &yaml, MomentMethodFixedPointOperator &_op, 
		const mfem::IterativeSolver * const inner=nullptr)
		: out(yaml), op(_op), inner_solver(inner)
	{
	}
	void MonitorResidual(int it, double norm, const mfem::Vector &r, bool final) {
		if (it==0) {
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
			const auto sweep = op.SweepTimer().RealTime(); 
			sweep_time.Append(sweep); 
			out << YAML::Key << "sweep" << YAML::Value << io::FormatTimeString(sweep);
			const auto moment = op.MomentTimer().RealTime(); 
			moment_time.Append(moment);  
			out << YAML::Key << "moment" << YAML::Value << io::FormatTimeString(moment); 
		out << YAML::EndMap; 
		out << YAML::EndMap << YAML::Newline; 

		if (final) {
			out << YAML::EndSeq; 
		}		
	}
};

struct SundialsUserCallbackData {
	YAML::Emitter *out; 
	const MomentMethodFixedPointOperator *G; 
	const mfem::IterativeSolver * const inner_solver; 
	mfem::Array<int> inner_it; 
	mfem::Array<double> sweep_time, moment_time; 
	SundialsUserCallbackData(YAML::Emitter &out, const MomentMethodFixedPointOperator &G, 
		const mfem::IterativeSolver * const isolver) 
		: out(&out), G(&G), inner_solver(isolver)
	{ }
};

void SundialsCallbackFunction(const char *module, const char *function, char *msg, void *user_data); 

int main(int argc, char *argv[]) {
	// initialize MPI 
	// automatically calls MPI_Finalize 
	mfem::Mpi::Init(argc, argv); 
	// must call hypre init for BoomerAMG now? 
	mfem::Hypre::Init(); 

	mfem::StopWatch wall_timer, setup_timer, solve_timer, output_timer; 
	wall_timer.Start(); 
	setup_timer.Start(); 

	const auto rank = mfem::Mpi::WorldRank(); 
	const bool root = rank == 0; 

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
	int par_ref = 0, ser_ref = 0; 
	mfem::OptionsParser args(argc, argv); 
	args.AddOption(&input_file, "-i", "--input", "input file name", true); 
	args.AddOption(&lua_cmds, "-l", "--lua", "lua commands to run", false); 
	args.AddOption(&ser_ref, "-sr", "--serial_refinements", "additional uniform refinements in serial"); 
	args.AddOption(&par_ref, "-pr", "--parallel_refinements", "additional uniform refinements in parallel"); 
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
	out.SetDoublePrecision(8); 
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

	// print git commit, branch, tag if available from CMake 
	io::PrintGitString(out);
	// output name of input file 
	out << YAML::Key << "input file" << YAML::Value << io::ResolveRelativePath(input_file); 

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
			io::LuaPhaseFunction lua_source = lua_source_objs[i].as<sol::function>(); 
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
	BoundaryConditionMap bc_map;
	for (auto i=0; i<bdr_attr_list.size(); i++) {
		sol::table data = bcs[bdr_attr_list[i].c_str()]; 
		std::string type = data["type"]; 
		io::ValidateOption<std::string>("boundary_conditions::type", type, {"inflow", "reflective", "vacuum"}); 
		if (type == "inflow") {
			sol::object value = data["value"]; 
			lua_bc_objs[i] = value; // keep lua data in scope  
			if (value.get_type() == sol::type::number) {
				auto val = value.as<double>(); 
				inflow_list[i] = new ConstantPhaseSpaceCoefficient(val); 
			} else if (value.get_type() == sol::type::function) {
				auto lua_func = value.as<io::LuaPhaseFunction>();
				auto func = [lua_func](const mfem::Vector &x, const mfem::Vector &Omega) {
					return lua_func(x(0), x(1), x(2), Omega(0), Omega(1), Omega(2)); 
				};
				inflow_list[i] = new FunctionGrayCoefficient(func);  
			}		
			bc_map[i+1] = BoundaryCondition::INFLOW;
		} else if (type == "reflective") {
			inflow_list[i] = nullptr; 
			bc_map[i+1] = BoundaryCondition::REFLECTIVE;
		} 
		else if (type == "vacuum") {
			inflow_list[i] = nullptr; 
			bc_map[i+1] = BoundaryCondition::INFLOW;
		}
	}

	// print list to screen 
	out << YAML::Key << "boundary conditions" << YAML::Value << YAML::BeginSeq; 
	for (auto i=0; i<bdr_attr_list.size(); i++) {
		out << YAML::BeginMap; 
		out << YAML::Key << "name" << YAML::Value << bdr_attr_list[i]; 
		out << YAML::Key << "type" << YAML::Value << std::string(bcs[bdr_attr_list[i].c_str()]["type"]); 
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
	sol::table mesh_node = lua["mesh"]; 
	auto ser_mesh = io::CreateMesh(mesh_node, out, root); 

	// --- assign materials to elements --- 
	sol::function geom_func = lua["material_map"]; 
	io::SetMeshAttributes(ser_mesh, geom_func, attr_map, root); 
	// --- assign boundary conditions to boundary elements --- 
	sol::function bdr_func = lua["boundary_map"]; 
	io::SetMeshBdrAttributes(ser_mesh, bdr_func, bdr_attr_map, root); 

	// apply serial refinements 
	// allow increasing lua refinement inputs with cmdline inputs -pr and -sr 
	ser_ref += mesh_node["serial_refinements"].get_or(0); 
	for (int sr=0; sr<ser_ref; sr++) {
		ser_mesh.UniformRefinement(); 
	}
	// need at minimum WorldSize() elements in serial mesh 
	if (ser_mesh.GetNE() < mfem::Mpi::WorldSize() and root) {
		MFEM_ABORT("serial mesh with " << ser_mesh.GetNE() << " elements too small to decompose on " 
			<< mfem::Mpi::WorldSize() << " processors"); 
	}
	const auto dim = ser_mesh.Dimension(); 

	// --- create parallel mesh --- 
	mfem::ParMesh mesh(MPI_COMM_WORLD, ser_mesh); 
	par_ref += mesh_node["parallel_refinements"].get_or(0); 
	for (int pr=0; pr<par_ref; pr++) {
		mesh.UniformRefinement(); 
	}
	mesh.ExchangeFaceNbrData(); // create parallel communication data needed for sweep 
	mesh.SetAttributes(); 
	// print mesh characteristics 
	double hmin, hmax, kmin, kmax; 
	mesh.GetCharacteristics(hmin, hmax, kmin, kmax); 
	const auto global_ne = mesh.ReduceInt(mesh.GetNE()); 
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
	out << YAML::Key << "MPI ranks" << YAML::Value << mfem::Mpi::WorldSize(); 
	out << YAML::Key << "refinements" << YAML::Value << YAML::BeginMap; 
		out << YAML::Key << "serial" << YAML::Value << ser_ref; 
		out << YAML::Key << "parallel" << YAML::Value << par_ref; 
	out << YAML::EndMap; 
	out << YAML::EndMap; 

	// --- load algorithmic parameters --- 
	sol::table driver = lua["driver"]; 
	const int fe_order = driver["fe_order"]; 
	std::string basis_type_str = io::GetAndValidateOption(driver, "basis_type", {"lobatto", "legendre"}, "lobatto", root); 

	// --- build solution space --- 
	// DG space for transport solution 
	int basis_type; 
	if (basis_type_str == "legendre") {
		basis_type = mfem::BasisType::GaussLegendre; 
	} else if (basis_type_str == "lobatto") {
		basis_type = mfem::BasisType::GaussLobatto; 
	} 
	mfem::L2_FECollection fec(fe_order, dim, basis_type); 
	mfem::ParFiniteElementSpace fes(&mesh, &fec); // scalar finite element space 
	mfem::ParFiniteElementSpace vfes(&mesh, &fec, dim); // vector finite element space, dim copies of fes 
	fes.ExchangeFaceNbrData(); // create parallel degree of freedom maps used in sweep 

	mfem::H1_FECollection cfec(fe_order, dim);
	mfem::ParFiniteElementSpace cfes(&mesh, &cfec);

	std::unique_ptr<mfem::FiniteElementCollection> rtfec;
	if (dim == 1) {
		rtfec = std::make_unique<mfem::H1_FECollection>(fe_order, dim);
	} else {
		rtfec = std::make_unique<mfem::RT_FECollection>(fe_order, dim);
	}
	mfem::ParFiniteElementSpace rtfes(&mesh, rtfec.get());

	mfem::DG_Interface_FECollection ifec(fe_order, std::max(dim,2));
	mfem::ParFiniteElementSpace ifes(&mesh, &ifec);

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
	// sweep requires multigroup coefficient 
	GridFunctionMGCoefficient total_mg(total_gf); 
	// rest of the code relies on scalar coefficient 
	auto total_ptr = std::unique_ptr<mfem::Coefficient>(total_mg.GetGroupCoefficient(0));
	auto &total = *total_ptr;

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
	sol::optional<sol::table> sn_table_avail = lua["sn"]; 
	if (!sn_table_avail) MFEM_ABORT("must define angular quadrature table");
	sol::table sn_table = sn_table_avail.value();
	auto quad = std::unique_ptr<AngularQuadrature>(io::CreateAngularQuadrature(sn_table, out, dim, root));
	const auto Nomega = quad->Size(); 

	// --- transport vector setup --- 
	TransportVectorExtents psi_ext(1, Nomega, fes.GetVSize());
	const auto psi_size = TotalExtent(psi_ext); 
	MomentVectorExtents phi_ext(1, 1, fes.GetVSize());
	const auto phi_size = TotalExtent(phi_ext);
	MomentVectorExtents moments_ext(1, dim+1, fes.GetVSize()); // scalar flux and dim components of current 
	const auto moments_size = TotalExtent(moments_ext); 
	DiscreteToMoment D(*quad, psi_ext, phi_ext); // integrates over angle psi -> phi 
	DiscreteToMoment Dlin_aniso(*quad, psi_ext, moments_ext); // forms psi -> [phi,J] 

	const auto psi_size_global = mesh.ReduceInt(psi_size); 
	const auto phi_size_global = mesh.ReduceInt(phi_size); 

	// --- create solver objects from Lua input --- 
	sol::table solver = driver["solver"]; 
	sol::optional<sol::table> accel_avail = driver["acceleration"]; 
	sol::optional<sol::table> prec_avail = driver["preconditioner"]; 
	if (accel_avail and prec_avail) { MFEM_ABORT("cannot use both preconditioning and acceleration"); }
	auto outer_solver = std::unique_ptr<mfem::IterativeSolver>(io::CreateIterativeSolver(solver, MPI_COMM_WORLD));
	if (!outer_solver) { MFEM_ABORT("outer solver required"); }
	if (accel_avail) {
		io::ValidateOption<std::string>("driver::solver::type", solver["type"], 
			{"fixed point", "fp", "kinsol"}, root); 		
	} 
	else {
		io::ValidateOption<std::string>("driver::solver::type", solver["type"], 
			{"cg", "conjugate gradient", "gmres", "fgmres", "sli", "bicg", "bicgstab", "direct", "superlu"}, root); 
	}
	sol::table inner_solver_table; 
	std::unique_ptr<mfem::Solver> inner_solver;
	if (accel_avail) {
		sol::optional<sol::table> inner_solver_table_avail = accel_avail.value()["solver"]; 
		if (inner_solver_table_avail) {
			inner_solver_table = inner_solver_table_avail.value();
		}
		// default to direct if solver table not provided 
		else {
			inner_solver_table = accel_avail.value().create_with("type", "direct"); 
		}
		inner_solver.reset(io::CreateSolver(inner_solver_table, MPI_COMM_WORLD));
	}
	if (prec_avail) {
		sol::optional<sol::table> inner_solver_table_avail = prec_avail.value()["solver"]; 
		if (inner_solver_table_avail) {
			inner_solver_table = inner_solver_table_avail.value(); 
		} 
		// default to direct if solver table not provided 
		else {
			inner_solver_table = prec_avail.value().create_with("type", "direct"); 
		}
		inner_solver.reset(io::CreateSolver(inner_solver_table, MPI_COMM_WORLD));
	}

	// generic operator in case inner solver is SuperLU or product operator etc 
	auto *inner_it_solver = dynamic_cast<mfem::IterativeSolver*>(inner_solver.get());
	// sweep setup options 
	sol::optional<sol::table> sweep_opts_avail = driver["sweep_opts"]; 

	const int lumping = driver["lumping"].get_or(0);

	// --- output algorithmic options used --- 
	out << YAML::Key << "driver" << YAML::Value << YAML::BeginMap; 
		out << YAML::Key << "fe order" << YAML::Value << fe_order; 
		out << YAML::Key << "psi size" << YAML::Value << psi_size_global;
		out << YAML::Key << "phi size" << YAML::Value << phi_size_global;
		out << YAML::Key << "basis type" << YAML::Value << basis_type_str; 
		out << YAML::Key << "solver" << YAML::Value << solver; 
		if (sweep_opts_avail) {
			out << YAML::Key << "sweep options" << YAML::Value << sweep_opts_avail.value();
		}
		out << YAML::Key << "lumping type" << YAML::Value << YAML::BeginMap; 
			out << YAML::Key << "mass" << YAML::Value << IsMassLumped(lumping); 
			out << YAML::Key << "gradient" << YAML::Value << IsGradientLumped(lumping); 
			out << YAML::Key << "faces" << YAML::Value << IsFaceLumped(lumping); 
		out << YAML::EndMap; 

	// --- sweep setup --- 
	// allocate transport vector + views 
	mfem::Vector psi(psi_size);
	psi = 0.0; 
	mfem::Vector moment_solution(moments_size), moment_solution_HO; 
	mfem::ParGridFunction phi(&fes), J(&vfes, moment_solution, phi_size);

	// initial guess 
	D.Mult(psi, phi); 

	// form fixed source term 
	mfem::Vector source_vec(psi_size); 
	source_vec = 0.0; 
	TransportVectorView source_vec_view(source_vec.GetData(), psi_ext); 
	mfem::Array<double> energy_grid(2); energy_grid[0] = 0.0; energy_grid[1] = 1.0; 
	FormTransportSource(fes, *quad, energy_grid, source, inflow, source_vec_view); 

	// build sweep operator 
	InverseAdvectionOperator Linv(fes, *quad, total_mg, bc_map, lumping); 
	if (sweep_opts_avail) {
		sol::table sweep_opts = sweep_opts_avail.value();
		io::SetSweepOptions(sweep_opts, Linv, root);
		out << YAML::Key << "sweep options" << YAML::Value << sweep_opts;
	}
	if (Linv.IsParallelBlockJacobi()) Linv.Exchange(psi);

	// common parameters to discretization
	mfem::Vector normal(dim); 
	normal = 0.0; normal(0) = 1.0; 
	const double alpha = ComputeAlpha(*quad, normal);
	mfem::Vector beta(dim); 
	for (int d=0; d<dim; d++) { beta(d) = d+1; }

	// store time spent sweeping/solving moment system at each iteration 
	mfem::Array<double> sweep_time, moment_time; 

	// standard sn iteration with preconditioning if available 
	if (!accel_avail) {
		std::unique_ptr<DiffusionSyntheticAccelerationOperator> prec;
		std::unique_ptr<mfem::Operator> dsa_op, dsa_op_extract;
		std::unique_ptr<mfem::HypreBoomerAMG> amg;

		// global scattering operator 
		mfem::BilinearForm Ms_form(&fes); 
		if (IsMassLumped(lumping))
			Ms_form.AddDomainIntegrator(new QuadratureLumpedIntegrator(new mfem::MassIntegrator(scattering)));
		else
			Ms_form.AddDomainIntegrator(new mfem::MassIntegrator(scattering)); 
		Ms_form.Assemble(); 
		Ms_form.Finalize();

		// build preconditioner from input spec 
		if (prec_avail) {
			out << YAML::Key << "preconditioner" << YAML::Value << YAML::BeginMap; 
			sol::table prec_table = prec_avail.value(); 
			std::string type = prec_table["type"]; 
			std::transform(type.begin(), type.end(), type.begin(), ::tolower); 
			out << YAML::Key << "type" << YAML::Value << type; 
			if (type == "p1sa") {
				if (inner_it_solver) { MFEM_ABORT("only direct available for P1"); }
				P1Discretization p1(fes, vfes, total, absorption, bc_map, lumping);
				dsa_op.reset(p1.GetOperator());
				inner_solver->SetOperator(*dsa_op);
				auto *ceo = new ComponentExtractionOperator(p1.GetOffsets(), 1); 
				auto *ceo_t = new mfem::TransposeOperator(*ceo); 
				// solve 2x2 but source and solution and scalar flux only 
				dsa_op_extract = std::make_unique<mfem::TripleProductOperator>(
					ceo, inner_solver.get(), ceo_t, true, false, true); 
				prec = std::make_unique<DiffusionSyntheticAccelerationOperator>(*dsa_op_extract, Ms_form);
			} 
			else if (type == "ldgsa") {
				BlockLDGDiscretization disc(fes, vfes, total, absorption, bc_map, lumping);
				disc.SetAlpha(alpha);
				auto *block_op = disc.GetOperator();
				dsa_op.reset(disc.FormSchurComplement(*block_op));
				delete block_op;

				// iterative solve
				if (inner_it_solver) {
					amg = std::make_unique<mfem::HypreBoomerAMG>();
					inner_it_solver->SetPreconditioner(*amg); 
				} 
				inner_solver->SetOperator(*dsa_op);
				prec = std::make_unique<DiffusionSyntheticAccelerationOperator>(*inner_solver, Ms_form);
			}
			else if (type == "mip") {
				double kappa = prec_table["kappa"].get_or(pow(fe_order+1,2)); 
				bool scale_stabilization = prec_table["scale_stabilization"].get_or(true); 
				bool lower_bound = prec_table["bound_stabilization_below"].get_or(true); 
				// const auto bc_type = io::GetDiffusionBCType(bc_str); 
				BlockIPDiscretization disc(fes, vfes, total, absorption, bc_map, lumping);
				disc.SetAlpha(alpha);
				disc.SetKappa(kappa);
				disc.SetScalePenalty(scale_stabilization);
				if (lower_bound)
					disc.SetPenaltyLowerBound(alpha/2);
				auto *block_op = disc.GetOperator();
				dsa_op.reset(disc.FormSchurComplement(*block_op));
				delete block_op;

				// iterative solve
				if (inner_it_solver) {
					amg = std::make_unique<mfem::HypreBoomerAMG>();
					inner_it_solver->SetPreconditioner(*amg); 
				} 

				inner_solver->SetOperator(*dsa_op);
				prec = std::make_unique<DiffusionSyntheticAccelerationOperator>(*inner_solver, Ms_form);

				out << YAML::Key << "kappa" << YAML::Value << kappa; 
				out << YAML::Key << "scale stabilization" << YAML::Value << scale_stabilization; 
				out << YAML::Key << "bound stabilization from below" << YAML::Value << lower_bound; 
			}
			else if (type == "scalar mip") {
				double kappa = prec_table["kappa"].get_or(pow(fe_order+1,2)); 
				double sigma = prec_table["sigma"].get_or(-1.0); 
				InteriorPenaltyDiscretization ipdisc(fes, total, absorption, bc_map, lumping);
				ipdisc.SetKappa(kappa);
				ipdisc.SetSigma(sigma);
				ipdisc.SetPenaltyLowerBound(alpha/2);
				dsa_op.reset(ipdisc.GetOperator());

				if (inner_it_solver) {
					amg = std::make_unique<mfem::HypreBoomerAMG>();
					inner_it_solver->SetPreconditioner(*amg); 
				}

				inner_solver->SetOperator(*dsa_op);
				prec = std::make_unique<DiffusionSyntheticAccelerationOperator>(*inner_solver, Ms_form);

				out << YAML::Key << "kappa" << YAML::Value << kappa; 
				out << YAML::Key << "sigma" << YAML::Value << sigma; 
			}
			else MFEM_ABORT("dsa type " << type << " not defined"); 

			// setup AMG object 
			if (amg) {
				amg->SetPrintLevel(0); 
				sol::optional<sol::table> amg_opts = prec_table["solver"]["amg_opts"]; 
				if (amg_opts) io::SetAMGOptions(amg_opts.value(), *amg, root); 				
			}

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
		// turn off PBJ for elimination of angular flux 
		const bool pbj = Linv.IsParallelBlockJacobi();
		Linv.UseParallelBlockJacobi(false);
		Linv.Mult(source_vec, psi); 
		mfem::Vector schur_source(phi_size); 
		D.Mult(psi, schur_source);

		if (prec_avail) { outer_solver->SetPreconditioner(*prec); }
		outer_solver->SetOperator(T); 
		// set PBJ to original state 
		Linv.UseParallelBlockJacobi(pbj);
		TransportIterationMonitor monitor(out, T, prec.get(), inner_it_solver); 
		outer_solver->SetMonitor(monitor); 
		setup_timer.Stop(); 
		solve_timer.Start(); 
		outer_solver->Mult(schur_source, phi); 
		solve_timer.Stop(); 

		// extra sweep to get psi 
		mfem::Vector scat_source(phi_size); 
		Ms_form.Mult(phi, scat_source); 
		D.MultTranspose(scat_source, psi); 
		psi += source_vec; 
		// ensure PBJ off for back solve 
		Linv.UseParallelBlockJacobi(false);
		Linv.Mult(psi, psi); 
		// compute phi and J 
		Dlin_aniso.Mult(psi, moment_solution); 

		// output iteration info 
		out << YAML::Key << "outer iterations" << YAML::Value << outer_solver->GetNumIterations();
		if (monitor.inner_it.Size()) {
			out << YAML::Key << "inner iteration" << YAML::Value << YAML::BeginMap; 
			out << YAML::Key << "max" << YAML::Value << monitor.inner_it.Max();
			out << YAML::Key << "min" << YAML::Value << monitor.inner_it.Min(); 
			out << YAML::Key << "avg" << YAML::Value << (double)monitor.inner_it.Sum()/monitor.inner_it.Size();
			out << YAML::Key << "total" << YAML::Value << monitor.inner_it.Sum(); 
			out << YAML::EndMap; 
		}

		sweep_time = monitor.sweep_time; 
		moment_time = monitor.prec_time; 
	}

	// moment-based solve
	else {
		// space for current 
		mfem::Array<int> offsets; 
		mfem::BlockVector block_x;
		std::unique_ptr<mfem::Operator> smm; 
		std::unique_ptr<mfem::HypreBoomerAMG> amg;
		std::unique_ptr<mfem::Operator> lo_op;

		sol::table accel = accel_avail.value(); 
		std::string type = accel["type"]; 
		out << YAML::Key << "acceleration" << YAML::Value << YAML::BeginMap; 
		out << YAML::Key << "type" << YAML::Value << type; 
		std::transform(type.begin(), type.end(), type.begin(), ::toupper); 
		if (type == "LDGSMM") {
			bool consistent = accel["consistent"].get_or(false); 
			bool scale_stabilization = accel["scale_stabilization"].get_or(false); 
			BlockLDGDiscretization disc(fes, vfes, total, absorption, bc_map, lumping);
			disc.SetAlpha(alpha);
			lo_op.reset(disc.GetOperator());

			// iterative solve
			if (inner_it_solver) {
				amg = std::make_unique<mfem::HypreBoomerAMG>();
				inner_it_solver->SetPreconditioner(*amg); 				
			} 

			auto *lo_solver = new BlockMomentDiscretization::Solver(vfes, *inner_solver);
			lo_solver->SetOperator(*lo_op);

			mfem::Operator *source_op;
			if (consistent) {
				source_op = new ConsistentLDGSMMOperator(disc, *quad, psi_ext, source_vec);
			} else {
				int source_lumping = accel["source_lumping"].get_or(0);
				source_op = new IndependentBlockSMMOperator(fes, vfes, *quad, psi_ext, source, inflow, 
					alpha, bc_map, source_lumping);
			}

			offsets = disc.GetOffsets(); // copy offsets 
			block_x.Update(offsets); // set size of block_x 
			// extract scalar flux, storing [J,phi] in block_x 
			auto *block_extract = new SubBlockExtractionOperator(block_x, 1); 
			// psi -> SMM source -> diffusion solution -> extract phi from block vector 
			// use allium version of triple product operator to ensure temp vectors 
			// initialized to zero (for first initial guess when iterative_mode = true)
			smm = std::make_unique<TripleProductOperator>(block_extract, lo_solver, source_op, true, true, true); 

			// output LDG specific options 
			out << YAML::Key << "consistent" << YAML::Value << consistent; 
			out << YAML::Key << "scale stabilization" << YAML::Value << scale_stabilization; 
		} 
		else if (type == "IPSMM") {
			bool consistent = accel["consistent"].get_or(false); 
			double kappa = accel["kappa"].get_or(pow(fe_order+1,2)); 
			// scale interior penalty coefficient by diffusion coefficient 
			bool scale_stabilization = accel["scale_stabilization"].get_or(true); 
			// use kappa = alpha/2 if regular kappa goes below alpha/2 
			bool stab_bound = accel["bound_stabilization_below"].get_or(true); 

			BlockIPDiscretization disc(fes, vfes, total, absorption, bc_map, lumping);
			disc.SetAlpha(alpha);
			if (stab_bound)
				disc.SetPenaltyLowerBound(alpha/2);
			disc.SetScalePenalty(scale_stabilization);
			lo_op.reset(disc.GetOperator());

			// iterative solve
			if (inner_it_solver) {
				amg = std::make_unique<mfem::HypreBoomerAMG>();
				inner_it_solver->SetPreconditioner(*amg); 				
			} 

			auto *lo_solver = new BlockMomentDiscretization::Solver(vfes, *inner_solver);
			lo_solver->SetOperator(*lo_op);

			mfem::Operator *source_op;
			if (consistent) 
				source_op = new ConsistentIPSMMOperator(disc, total, *quad, psi_ext, source_vec);
			else {
				const int source_lumping = accel["source_lumping"].get_or(0);
				source_op = new IndependentBlockSMMOperator(fes, vfes, *quad, psi_ext, source, inflow, 
					alpha, bc_map, source_lumping);
			}

			offsets = disc.GetOffsets(); // copy offsets 
			block_x.Update(offsets); // set size of block_x 
			// extract scalar flux, storing [J,phi] in block_x 
			auto *block_extract = new SubBlockExtractionOperator(block_x, 1); 
			// psi -> SMM source -> diffusion solution -> extract phi from block vector 
			// use allium version of triple product operator to ensure temp vectors 
			// initialized to zero (for first initial guess when iterative_mode = true)
			smm = std::make_unique<TripleProductOperator>(block_extract, lo_solver, source_op, true, true, true); 

			// output IP specific options 
			out << YAML::Key << "kappa" << YAML::Value << kappa; 
			out << YAML::Key << "consistent" << YAML::Value << consistent; 
			out << YAML::Key << "scale stabilization" << YAML::Value << scale_stabilization; 
			out << YAML::Key << "bound stabilization from below" << YAML::Value << stab_bound; 
		}
		else if (type == "P1SMM") {
			if (inner_it_solver) { MFEM_ABORT("only direct supported for P1"); }
			const bool consistent = accel["consistent"].get_or(true);
			if (!consistent) MFEM_ABORT("only consistent supported for P1");

			P1Discretization p1(fes, vfes, total, absorption, bc_map, lumping);
			lo_op.reset(p1.GetOperator());
			inner_solver->SetOperator(*lo_op);
			auto *source_op = new ConsistentP1SMMOperator(p1, *quad, psi_ext, source_vec);
			offsets = p1.GetOffsets();
			block_x.Update(offsets);
			auto *block_extract = new SubBlockExtractionOperator(block_x, 1);
			smm = std::make_unique<TripleProductOperator>(block_extract, inner_solver.get(), source_op, true, false, true); 
			out << YAML::Key << "consistent" << YAML::Value << true; 
		} 
		else if (type == "CGSMM") {
			phi.SetSpace(&cfes);
			H1DiffusionDiscretization disc(cfes, total, absorption, bc_map, lumping);
			disc.SetAlpha(alpha);
			lo_op.reset(disc.GetOperator());
			// iterative solve
			if (inner_it_solver) {
				amg = std::make_unique<mfem::HypreBoomerAMG>();
				inner_it_solver->SetPreconditioner(*amg); 				
			} 

			inner_solver->SetOperator(*lo_op);

			auto energy = MultiGroupEnergyGrid::MakeGray(0,1.0);
			auto *source_op = new IndependentSMMOperator(cfes, fes, *quad, psi_ext, energy, 
				total, source, inflow, alpha, bc_map, lumping);

			smm = std::make_unique<mfem::ProductOperator>(
				inner_solver.get(), source_op, false, true);
		}
		else if (type == "RTSMM") {
			J.SetSpace(&rtfes);
			RTDiffusionDiscretization disc(fes, rtfes, total, absorption, bc_map, lumping);
			disc.SetAlpha(alpha);
			lo_op.reset(disc.GetOperator());

			inner_solver->SetOperator(*lo_op);
			auto energy = MultiGroupEnergyGrid::MakeGray(0,1.0);
			auto *source_op = new IndependentRTSMMOperator(fes, rtfes, *quad, psi_ext, energy, 
				source, inflow, alpha, bc_map, lumping);

			offsets = disc.GetOffsets();
			block_x.Update(offsets);
			auto *block_extract = new SubBlockExtractionOperator(block_x, 1);
			smm = std::make_unique<TripleProductOperator>(
				block_extract, inner_solver.get(), source_op, true, false, true); 			
		}
		else if (type == "HRTSMM") {
			J.SetSpace(&rtfes);
			auto *disc = new HybridizedRTDiffusionDiscretization(fes, rtfes, ifes, total, absorption, bc_map, lumping);
			disc->SetAlpha(alpha);
			lo_op.reset(disc->GetOperator());

			if (inner_it_solver) {
				amg = std::make_unique<mfem::HypreBoomerAMG>();
				inner_it_solver->SetPreconditioner(*amg);
			}

			auto *hyb_solver = new HybridizedRTDiffusionDiscretization::Solver(*disc, *inner_solver);
			hyb_solver->SetOperator(*lo_op);

			auto energy = MultiGroupEnergyGrid::MakeGray(0,1.0);
			auto *source_op = new IndependentRTSMMOperator(fes, rtfes, *quad, psi_ext, energy, 
				source, inflow, alpha, bc_map, lumping);

			offsets = hyb_solver->GetOffsets();
			block_x.Update(offsets);
			auto *block_extract = new SubBlockExtractionOperator(block_x, 1);
			smm = std::make_unique<TripleProductOperator>(
				block_extract, hyb_solver, source_op, true, true, true); 
		}
		else { MFEM_ABORT("acceleration type " << type << " not defined"); }

		// set AMG object 
		if (amg) {
			amg->SetPrintLevel(0); 
			sol::optional<sol::table> amg_opts = inner_solver_table["amg_opts"]; 
			if (amg_opts) io::SetAMGOptions(amg_opts.value(), *amg, root);	
		}

		auto *triple_prod = dynamic_cast<TripleProductOperator*>(smm.get()); 
		if (triple_prod) {
			triple_prod->SetLoggingKeys("smm block extract", "smm solve", "smm source"); 
		}

		out << YAML::Key << "solver" << YAML::Value << inner_solver_table; 
		out << YAML::EndMap; // end acceleration map 
		bool diffusion_solve = accel["diffusion_solve"].get_or(false); 
		if (diffusion_solve) out << YAML::Key << "diffusion solve" << YAML::Value << diffusion_solve << YAML::Newline; 
		out << YAML::EndMap; // end driver map 

		// global scattering operator 
		mfem::ParMixedBilinearForm Ms_form(phi.ParFESpace(), &fes);
		if (IsMassLumped(lumping))
			Ms_form.AddDomainIntegrator(new QuadratureLumpedIntegrator(new mfem::MassIntegrator(scattering)));
		else
			Ms_form.AddDomainIntegrator(new mfem::MassIntegrator(scattering)); 
		Ms_form.Assemble(); 
		Ms_form.Finalize();
		auto Ms = std::unique_ptr<mfem::HypreParMatrix>(Ms_form.ParallelAssemble());

		MomentMethodFixedPointOperator G(D, Linv, *Ms, *smm, source_vec, psi); 
		outer_solver->SetOperator(G); 
		SundialsUserCallbackData sundials_data(out, G, inner_it_solver); 
		auto *sundials = dynamic_cast<mfem::SundialsSolver*>(outer_solver.get());
		if (sundials) {
			KINSetInfoHandlerFn(sundials->GetMem(), SundialsCallbackFunction, &sundials_data); 
			KINSetErrHandlerFn(sundials->GetMem(), io::SundialsErrorFunction, &sundials_data); 
		}

		MomentMethodIterationMonitor monitor(out, G, inner_it_solver); 
		outer_solver->SetMonitor(monitor); 

		setup_timer.Stop(); 
		solve_timer.Start(); 
		if (diffusion_solve) {
			smm->Mult(psi, phi); 
		}

		else {
			mfem::Vector blank, x(phi.ParFESpace()->GetTrueVSize()); 
			outer_solver->Mult(blank, x);
			phi.Distribute(x); 
			out << YAML::Key << "outer iterations" << YAML::Value << outer_solver->GetNumIterations();			
		}
		solve_timer.Stop(); 
		mfem::Array<int> *inner_it = nullptr; 
		if (sundials) {
			inner_it = &sundials_data.inner_it; 
			sweep_time = sundials_data.sweep_time; 
			moment_time = sundials_data.moment_time; 
		} else {
			inner_it = &monitor.inner_it; 
			sweep_time = monitor.sweep_time; 
			moment_time = monitor.moment_time; 
		}
		if (inner_it->Size()) {
			out << YAML::Key << "inner iteration" << YAML::Value << YAML::BeginMap; 
			out << YAML::Key << "max" << YAML::Value << inner_it->Max();
			out << YAML::Key << "min" << YAML::Value << inner_it->Min(); 
			out << YAML::Key << "avg" << YAML::Value << (double)inner_it->Sum()/inner_it->Size(); 
			out << YAML::Key << "total" << YAML::Value << inner_it->Sum(); 		
			out << YAML::EndMap; 
		}

		// compute "consistency" between SN and moment solution 
		// for scalar flux and current 
		if (block_x.Size()) {
			J.Distribute(block_x.GetBlock(0));
			moment_solution_HO.SetSize(moments_size);  
			Dlin_aniso.Mult(psi, moment_solution_HO); 
			mfem::ParGridFunction phi_sn(&fes, moment_solution_HO, 0); 
			mfem::ParGridFunction J_sn(&vfes, moment_solution_HO, fes.GetVSize()); 
			mfem::GridFunctionCoefficient phi_snc(&phi_sn); 
			double consistency_phi = phi.ComputeL2Error(phi_snc); 
			mfem::VectorGridFunctionCoefficient J_snc(&J_sn); 
			double consistency_J = J.ComputeL2Error(J_snc); 
			std::stringstream ss; 
			out << YAML::Key << "consistency" << YAML::Value << YAML::BeginMap; 
				ss << std::setprecision(3) << std::scientific << consistency_phi; 
				out << YAML::Key << "scalar flux" << YAML::Value << ss.str(); 
				ss.str(""); 
				ss << consistency_J; 
				out << YAML::Key << "current" << YAML::Value << ss.str(); 
			out << YAML::EndMap; 
		}
	}

	// --- clean up hanging pointers --- 
	for (int i=0; i<nattr; i++) { delete source_list[i]; }
	for (int i=0; i<nbattr; i++) { delete inflow_list[i]; }

	// time post processing and output 
	output_timer.Start(); 

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
		auto solution_lam = [&solution_func](const mfem::Vector &x, mfem::Vector &v) {
			double pos[3]; 
			for (auto d=0; d<x.Size(); d++) { pos[d] = x(d); }
			sol::table r = solution_func(pos[0], pos[1], pos[2]); 
			for (int i=0; i<x.Size(); i++) {
				v(i) = r[i+1]; 
			}
		}; 
		mfem::VectorFunctionCoefficient Jcoef(dim, solution_lam); 
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
			out << YAML::Key << "paraview" << YAML::Value << io::ResolveRelativePath(output_name); 
			mfem::ParGridFunction mesh_part(&fes0); 
			for (int i=0; i<mesh_part.Size(); i++) { mesh_part[i] = rank; }
			mfem::ParaViewDataCollection dc(output_name, &mesh); 
			dc.RegisterField("phi", &phi); 
			dc.RegisterField("J", &J); 
			mfem::ParGridFunction phi_ho, J_ho; 
			if (moment_solution_HO.Size()) {
				phi_ho.MakeRef(&fes, moment_solution_HO, 0); 
				J_ho.MakeRef(&vfes, moment_solution_HO, fes.GetVSize()); 
				dc.RegisterField("phi_ho", &phi_ho); 
				dc.RegisterField("J_ho", &J_ho); 
			}
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
			mfem::Vector phi_tracer, Jtracer, phi_ho_tracer, J_ho_tracer; 
			finder.Interpolate(phi, phi_tracer); 
			finder.Interpolate(J, Jtracer); 
			if (moment_solution_HO.Size()) {
				mfem::ParGridFunction phi_ho(&fes, moment_solution_HO, 0);
				mfem::ParGridFunction J_ho(&vfes, moment_solution_HO, fes.GetVSize()); 
				finder.Interpolate(phi_ho, phi_ho_tracer); 
				finder.Interpolate(J_ho, J_ho_tracer); 
			}
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
				if (phi_ho_tracer.Size()) 
					out << YAML::Key << "scalar flux (HO)" << YAML::Value << phi_ho_tracer(i); 
				if (J_ho_tracer.Size()) {
					out << YAML::Key << "current (HO)" << YAML::Value << YAML::Flow << YAML::BeginSeq; 
					for (auto d=0; d<dim; d++) { out << J_ho_tracer(ntracers*d + i); }
					out << YAML::EndSeq; 
				}
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
			mesh.EnsureNodes(); // required for FindPointsGSLIB
			sol::table lineout = lineout_avail.value(); 
			std::string path = output["lineout_path"].get_or(std::string("lineout.yaml")); 
			out << YAML::Key << "lineout" << YAML::Value << io::ResolveRelativePath(path); 
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
				sol::table start_table = line["from"]; 
				sol::table end_table = line["to"]; 
				for (int d=0; d<dim; d++) {
					start(d) = start_table[d+1]; 
					end(d) = end_table[d+1]; 
				}
				subtract(end, start, dir); 
				double L = dir.Norml2();
				// get (p+1) points per element on a uniform mesh 
				const int npoints = L/hmin*(fe_order + 1); 
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
				finder.Setup(mesh); 
				finder.FindPoints(pts, mfem::Ordering::byVDIM); 
				const auto &gscodes = finder.GetCode(); 
				mfem::Vector phi_line, J_line, phi_ho_line, J_ho_line; 
				finder.Interpolate(phi, phi_line);
				finder.Interpolate(J, J_line);  
				if (moment_solution_HO.Size()) {
					mfem::ParGridFunction phi_ho(&fes, moment_solution_HO, 0);
					mfem::ParGridFunction J_ho(&vfes, moment_solution_HO, fes.GetVSize()); 
					finder.Interpolate(phi_ho, phi_ho_line); 
					finder.Interpolate(J_ho, J_ho_line); 
				}

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
				if (phi_ho_line.Size()) {
					fout << YAML::Key << "scalar flux (HO)" << YAML::Value << YAML::BeginSeq; 
					for (auto n=0; n<npoints; n++) {
						if (gscodes[n]==2) continue; 
						fout << phi_ho_line(n); 
					}
					fout << YAML::EndSeq; 
				}
				if (J_ho_line.Size()) {
					fout << YAML::Key << "current (HO)" << YAML::Value << YAML::BeginSeq;  
					for (auto n=0; n<npoints; n++) {
						if (gscodes[n] == 2) continue; 
						fout << YAML::Flow << YAML::BeginSeq; 
						for (auto d=0; d<dim; d++) {
							fout << J_ho_line(d * npoints + n); 
						}
						fout << YAML::EndSeq; 
					}
					fout << YAML::EndSeq; 
				}
				fout << YAML::EndMap; 
				finder.FreeData(); // clean up find points data 
			}
			fout << YAML::EndMap; 
			file_out.close(); 
		#else 
			if (root) MFEM_WARNING("gslib required for lineouts"); 
		#endif
		}
		out << YAML::EndMap; // end output map 
	}

	TimingLog.Synchronize(); 
	EventLog.Synchronize(); 

	// output wall clock time 
	MPI_Barrier(MPI_COMM_WORLD); 
	output_timer.Stop(); 
	wall_timer.Stop(); 
	if (EventLog.size()) {
		out << YAML::Key << "event log" << YAML::Value << YAML::BeginMap; 
		for (const auto &it : EventLog) {
			out << YAML::Key << it.first << YAML::Value << it.second; 
		}
		out << YAML::EndMap; 
	}
	if (TimingLog.size()) {
		out << YAML::Key << "timing log" << YAML::Value << YAML::BeginMap; 
		for (const auto &it : TimingLog) {
			out << YAML::Key << it.first << YAML::Value << io::FormatTimeString(it.second); 
		}
		out << YAML::EndMap; 		
	}
	out << YAML::Key << "timings" << YAML::Value << YAML::BeginMap; 
		out << YAML::Key << "setup" << YAML::Value << io::FormatTimeString(setup_timer.RealTime()); 
		out << YAML::Key << "solve" << YAML::Value << YAML::BeginMap; 
			out << YAML::Key << "total" << YAML::Value << io::FormatTimeString(solve_timer.RealTime()); 
			out << YAML::Key << "sweep" << YAML::Value << io::FormatTimeString(sweep_time.Sum()); 
			out << YAML::Key << "moment" << YAML::Value << io::FormatTimeString(moment_time.Sum()); 
		out << YAML::EndMap; 
		out << YAML::Key << "output" << YAML::Value << io::FormatTimeString(output_timer.RealTime()); 
		out << YAML::Key << "wall" << YAML::Value << io::FormatTimeString(wall_timer.RealTime()); 
	out << YAML::EndMap; 

	// --- end yaml output --- 
	out << YAML::EndMap << YAML::Newline; 
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
		out << YAML::Key << "total" << YAML::Value << io::FormatTimeString(total_time); 
		out << YAML::Key << "sweep" << YAML::Value << io::FormatTimeString(sweep_time); 
		out << YAML::Key << "moment" << YAML::Value << io::FormatTimeString(moment_time); 
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