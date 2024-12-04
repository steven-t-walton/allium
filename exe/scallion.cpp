#include "mfem.hpp"
#include "yaml-cpp/yaml.h"
#include "sol/sol.hpp"
#include <deque>

#include "bdr_conditions.hpp"
#include "comment_stream.hpp"
#include "io.hpp"
#include "sweep.hpp"
#include "transport_op.hpp"
#include "constants.hpp"
#include "opacity.hpp"
#include "trt_integrators.hpp"
#include "trt_picard.hpp"
#include "trt_linearized.hpp"
#include "trt_ndsa.hpp"
#include "lumping.hpp"
#include "tracer.hpp"
#include "block_diag_op.hpp"
#include "moment_discretization.hpp"
#include "multigroup.hpp"
#include "mg_form.hpp"
#include "planck.hpp"
#include "restart.hpp"
#include "coefficient.hpp"
#include "linalg.hpp"
#include "fixed_point.hpp"
#include "utils.hpp"

class ScallionInnerSolverMonitor : public mfem::IterativeSolverMonitor
{
public:
	mfem::Array<int> iters; 
	double max_norm = 0.0; 
	double max_rel_norm = 0.0;

	ScallionInnerSolverMonitor() { iters.Reserve(50); }
	void Reset() { iters.SetSize(0); max_norm = 0.0; }

	friend YAML::Emitter &operator<<(YAML::Emitter &out, 
		const ScallionInnerSolverMonitor &monitor)
	{
		// for some reason mfem::Array::Sum isn't const... 
		auto &iters = *const_cast<mfem::Array<int>*>(&monitor.iters); 
		const int sum = iters.Sum(); 
		const double avg = (double)sum/iters.Size(); 
		out << YAML::BeginMap; 
			out << YAML::Key << "it" << YAML::Value << YAML::Flow << YAML::BeginMap; 
				out << YAML::Key << "max" << YAML::Value << iters.Max(); 
				out << YAML::Key << "min" << YAML::Value << iters.Min(); 
				std::stringstream ss; 
				ss << std::fixed << std::setw(3) << std::setprecision(2) << avg; 
				out << YAML::Key << "avg" << YAML::Value << ss.str();  
				out << YAML::Key << "total" << YAML::Value << sum; 
			out << YAML::EndMap; 
			out << YAML::Key << "max norm" << YAML::Value << YAML::Flow << YAML::BeginMap
				<< YAML::Key << "absolute" << YAML::Value << io::FormatScientific(monitor.max_norm)
				<< YAML::Key << "relative" << YAML::Value << io::FormatScientific(monitor.max_rel_norm);
			out << YAML::EndMap;
		out << YAML::EndMap; 
		return out; 
	}
};

class DiffusionAndEnergyBalanceMonitor : public mfem::IterativeSolverMonitor
{
private:
	ScallionInnerSolverMonitor &diffusion, &meb;
public:
	DiffusionAndEnergyBalanceMonitor(
		ScallionInnerSolverMonitor &diffusion, ScallionInnerSolverMonitor &meb)	
		: diffusion(diffusion), meb(meb)
	{ }
	void MonitorResidual(int it, double norm, const mfem::Vector &r, bool final) 
	{
		diffusion.MonitorResidual(it, norm, r, final);
		meb.MonitorResidual(it, norm, r, final);
	}
	friend YAML::Emitter &operator<<(YAML::Emitter &out, 
		const DiffusionAndEnergyBalanceMonitor &monitor)
	{
		out << monitor.diffusion << monitor.meb;
		return out;
	}
};

class InnerIterativeSolverMonitor : public ScallionInnerSolverMonitor
{
private:
	const mfem::IterativeSolver &solver; 
public:
	InnerIterativeSolverMonitor(const mfem::IterativeSolver &s) : solver(s)
	{ }
	void MonitorResidual(int it, double norm, const mfem::Vector &r, bool final) {
		if (it==0) {
			Reset(); 
			// hack to fix poor design of call to monitor in mfem::NewtonSolver 
			if (dynamic_cast<const mfem::NewtonSolver*>(iter_solver)) {
				return; 
			}
		}
		const auto iter = solver.GetNumIterations(); 
		if (iter > 0) {
			iters.Append(iter); 
			max_rel_norm = std::max(solver.GetFinalRelNorm(), max_rel_norm); 
			max_norm = std::max(solver.GetFinalNorm(), max_norm);
		}
	}
};

class BlockNonlinearSolverMonitor : public ScallionInnerSolverMonitor
{
private:
	const DenseBlockDiagonalNonlinearSolver &solver; 
public:
	BlockNonlinearSolverMonitor(const DenseBlockDiagonalNonlinearSolver &s) 
		: solver(s)
	{ 
	}
	void MonitorResidual(int it, double norm, const mfem::Vector &r, bool final) {
		if (it==0) {
			Reset(); 
		}
		const auto iter = solver.GetNumIterations(); 
		if (iter >= 0) {
			iters.Append(iter); 
			max_norm = std::max(max_norm, solver.GetFinalNorm()); 		
			max_rel_norm = std::max(max_rel_norm, solver.GetFinalRelNorm()); 		
		}
	}
};

struct KinsolCallbackData {
	mfem::IterativeSolverMonitor *monitor = nullptr; 
	int it = -1; 
	double norm = -1.0; 

	KinsolCallbackData(mfem::IterativeSolverMonitor *m) : monitor(m) { }
};

void KinsolCallbackFunction(const char *module, const char *function, char *msg, void *user_data)
{
	KinsolCallbackData *data = static_cast<KinsolCallbackData*>(user_data); 
	if (!data) MFEM_ABORT("did not get callback data struct"); 
	int it = 0; 
	double norm = 0.0;
	if (io::ParseKINSOLMessage(msg, it, norm)) {
		data->it = it; 
		data->norm = norm; 
	}
	if (data->monitor) {
		mfem::Vector blank;
		data->monitor->MonitorResidual(it-1, norm, blank, false); 
	}
}

double ComputeBalance(
	const InverseAdvectionOperator &Linv, const mfem::Operator &D, 
	const mfem::Operator &emission_form, const mfem::Vector &source, 
	const mfem::Vector &psi, const mfem::Vector &T)
{
	AdvectionOperator L(Linv);
	double local[2]; 
	local[0] = L.ComputeBalance(psi);
	mfem::Vector tmp(emission_form.Height());
	emission_form.Mult(T, tmp);
	local[1] = tmp.Sum();
	D.Mult(source, tmp);
	local[1] += tmp.Sum();
	double global[2];
	MPI_Reduce(local, global, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	return std::fabs(global[0] - global[1]) / global[1];
}

int main(int argc, char *argv[]) {
	mfem::StopWatch wall_timer; 
	wall_timer.Start(); 
	// initialize MPI 
	// automatically calls MPI_Finalize 
	mfem::Mpi::Init(argc, argv); 
	// must call hypre init for BoomerAMG now? 
	mfem::Hypre::Init(); 

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
		mfem::out << "                       ___    ___                           \n";
		mfem::out << "                      /\\_ \\  /\\_ \\    __                    \n";
		mfem::out << "  ____    ___     __  \\//\\ \\ \\//\\ \\  /\\_\\    ___     ___    \n";
		mfem::out << " /',__\\  /'___\\ /'__`\\  \\ \\ \\  \\ \\ \\ \\/\\ \\  / __`\\ /' _ `\\  \n";
		mfem::out << "/\\__, `\\/\\ \\__//\\ \\L\\.\\_ \\_\\ \\_ \\_\\ \\_\\ \\ \\/\\ \\L\\ \\ /\\ \\/\\ \n";
		mfem::out << "\\/\\____/\\ \\____\\ \\__/\\._\\/\\____\\/\\____\\\\ \\_\\ \\____/\\ \\_\\ \\_\\\n";
		mfem::out << " \\/___/  \\/____/\\/__/\\/_/\\/____/\\/____/ \\/_/\\/___/  \\/_/\\/_/\n";
		mfem::out << "\n                         a thermal radiative transfer solver\n"; 
		mfem::out << std::endl; 
	}

	// parse cmdline arguments 
	std::string input_file, lua_cmds; 
	int par_ref = 0, ser_ref = 0, max_cycles_override = 0; 
	mfem::OptionsParser args(argc, argv); 
	args.AddOption(&input_file, "-i", "--input", "input file name", true); 
	args.AddOption(&lua_cmds, "-l", "--lua", "lua commands to run", false); 
	args.AddOption(&ser_ref, "-sr", "--serial_refinements", "additional uniform refinements in serial"); 
	args.AddOption(&par_ref, "-pr", "--parallel_refinements", "additional uniform refinements in parallel"); 
	args.AddOption(&max_cycles_override, "-mc", "--max_cycles", "limit cycles"); 
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

	// --- print physical constants --- 
	out << YAML::Key << "physical constants" << YAML::Value << YAML::BeginMap; 
		out << YAML::Key << "speed of light" << YAML::Value << constants::SpeedOfLight; 
		out << YAML::Key << "radiation constant" << YAML::Value << constants::RadiationConstant; 
		out << YAML::Key << "planck" << YAML::Value << constants::Planck; 
	out << YAML::EndMap; 

	// --- load energy grid --- 
	// do this first since materials depend on energy 
	// discretization 
	out << YAML::Key << "energy" << YAML::Value << YAML::BeginMap; 
	sol::table energy_table = lua["energy"];
	if (!energy_table.valid()) MFEM_ABORT("must supply energy table");
	auto energy_grid = io::CreateEnergyGrid(energy_table, out, root);
	out << YAML::EndMap; // end energy block 
	const auto G = energy_grid.Size();

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
	mfem::Vector cv_list(nattr), density_list(nattr); 
	// must store the lua object so data doesn't go out of scope 
	// for the source coefficients 
	std::vector<sol::object> lua_source_objs(nattr); 
	mfem::Array<OpacityCoefficient*> total_list(nattr); 
	mfem::Array<PhaseSpaceCoefficient*> source_list(nattr); 
	out << YAML::Key << "materials" << YAML::Value << YAML::BeginMap; 
	for (auto i=0; i<attr_list.size(); i++) {
		sol::table data = materials[attr_list[i].c_str()]; 
		if (!data.valid()) MFEM_ABORT("material named " << attr_list[i] << " not found"); 
		out << YAML::Key << attr_list[i] << YAML::Value << YAML::BeginMap;
		out << YAML::Key << "attribute" << YAML::Value << i+1; 
		out << YAML::Key << "opacity" << YAML::Key << YAML::BeginMap; 
		sol::table total = data["total"]; 
		total_list[i] = io::CreateOpacity(total, energy_grid, out, root);
		out << YAML::EndMap;
		cv_list(i) = data["heat_capacity"]; 
		density_list(i) = data["density"].get_or(1.0); 
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

		out << YAML::Key << "heat capacity" << YAML::Value << cv_list(i); 
		out << YAML::Key << "density" << YAML::Value << density_list(i); 
		out << YAML::Key << "source" << YAML::Value; 
		if (lua_source_objs[i].get_type() == sol::type::number) {
			out << lua_source_objs[i].as<double>(); 
		} else {
			out << "function"; 
		}
		out << YAML::EndMap; 
	}
	out << YAML::EndMap; 

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
	mfem::Array<mfem::Coefficient*> inflow_base_list(nbattr);
	mfem::Array<PhaseSpaceCoefficient*> inflow_list(nbattr); 
	BoundaryConditionMap bc_map;
	out << YAML::Key << "boundary conditions" << YAML::Value << YAML::BeginMap; 
	for (auto i=0; i<bdr_attr_list.size(); i++) {
		sol::table data = bcs[bdr_attr_list[i].c_str()]; 
		std::string type = data["type"]; 
		io::ValidateOption<std::string>("boundary_conditions::type", type, {"inflow", "reflective", "vacuum"}, root); 
		double value;
		if (type == "inflow") {
			value = data["value"]; 
			inflow_base_list[i] = new mfem::ConstantCoefficient(value);
			inflow_list[i] = new PlanckEmissionPSCoefficient(*inflow_base_list[i]);
			bc_map[i+1] = BoundaryCondition::INFLOW;
		} else if (type == "reflective") {
			inflow_list[i] = nullptr; 
			inflow_base_list[i] = nullptr; 
			bc_map[i+1] = BoundaryCondition::REFLECTIVE;
		} 
		else if (type == "vacuum") {
			inflow_list[i] = nullptr; 
			inflow_base_list[i] = nullptr; 
			bc_map[i+1] = BoundaryCondition::INFLOW;
		}

		out << YAML::Key << bdr_attr_list[i] << YAML::Value << YAML::BeginMap;
			out << YAML::Key << "attribute" << YAML::Value << i+1; 
			out << YAML::Key << "type" << YAML::Value << type; 
			if (inflow_list[i]) out << YAML::Key << "value" << YAML::Value << value; 
		out << YAML::EndMap; 
	}
	out << YAML::EndMap; 

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
	sol::optional<int> part_type = mesh_node["partitioning"];
	int *partitioning = nullptr;
	if (part_type and part_type.value() > 1) {
		mfem::Array<int> attrs(nattr); 
		for (int i=0; i<nattr; i++) { attrs[i] = i+1; }
		PWOpacityCoefficient total_coef(attrs, total_list); 
		VectorComponentSumCoefficient total(total_coef);
		partitioning = utils::GenerateMetisPartitioning(ser_mesh, mfem::Mpi::WorldSize(), total, part_type.value());
	}
	mfem::ParMesh mesh(MPI_COMM_WORLD, ser_mesh, partitioning); 
	par_ref += mesh_node["parallel_refinements"].get_or(0); 
	for (int pr=0; pr<par_ref; pr++) {
		mesh.UniformRefinement(); 
	}
	mesh.ExchangeFaceNbrData(); // create parallel communication data needed for sweep 
	mesh.SetAttributes(); 
	// print info about mesh 
	io::PrintMeshCharacteristics(out, mesh, ser_ref, par_ref);
	out << YAML::EndMap; // end mesh map 

	// --- load algorithmic parameters --- 
	sol::table driver = lua["driver"]; 
	const int log_verbosity = driver["log_verbosity"].get_or(1);
	const int fe_order = driver["fe_order"]; 
	const int sigma_fe_order = driver["sigma_fe_order"].get_or(fe_order); 
	const int gray_sigma_fe_order = driver["gray_sigma_fe_order"].get_or(sigma_fe_order);
	std::string basis_type_str = io::GetAndValidateOption(driver, "basis_type", 
		{"lobatto", "legendre", "positive"}, "lobatto", root); 

	// --- build solution space --- 
	// DG space for transport solution 
	int basis_type; 
	if (basis_type_str == "legendre") {
		basis_type = mfem::BasisType::GaussLegendre; 
	} else if (basis_type_str == "lobatto") {
		basis_type = mfem::BasisType::GaussLobatto; 
	} else if (basis_type_str == "positive") {
		basis_type = mfem::BasisType::Positive; 
	}
	mfem::L2_FECollection fec(fe_order, dim, basis_type); 
	mfem::L2_FECollection fec0(0, dim); 
	mfem::ParFiniteElementSpace fes(&mesh, &fec); // scalar finite element space 
	mfem::ParFiniteElementSpace fes0(&mesh, &fec0); 
	mfem::ParFiniteElementSpace vfes(&mesh, &fec, dim); // vector finite element space, dim copies of fes 
	fes.ExchangeFaceNbrData(); // create parallel degree of freedom maps used in sweep 

	// project to piecewise constant 
	ProjectGridFunctionOperator pwc_op(fes, fes0);

	// piecewise constant used for plotting and storing cross section data 
	mfem::L2_FECollection sigma_fec(sigma_fe_order, dim, mfem::BasisType::Positive); 
	mfem::ParFiniteElementSpace sigma_fes(&mesh, &sigma_fec, G); // G copies 
	mfem::L2_FECollection gray_sigma_fec(gray_sigma_fe_order, dim, mfem::BasisType::Positive);
	mfem::ParFiniteElementSpace gray_sigma_fes(&mesh, &gray_sigma_fec);

	// --- create coefficients for input material data ---
	// list of attributes in order of total_list 
	mfem::Array<int> attrs(nattr); 
	for (int i=0; i<nattr; i++) { attrs[i] = i+1; }
	PWOpacityCoefficient total_coef(attrs, total_list); 

	// heat capacity 
	for (int i=0; i<cv_list.Size(); i++) cv_list(i) *= density_list(i); 
	mfem::PWConstCoefficient heat_capacity(cv_list); 
	mfem::PWConstCoefficient density(density_list); 

	// volumetric source 
	PWPhaseSpaceCoefficient source(attrs, source_list); 
	mfem::Array<int> battrs(nbattr); 
	for (int i=0; i<nbattr; i++) { battrs[i] = i+1; }
	PWPhaseSpaceCoefficient inflow(battrs, inflow_list); 

	// --- angular quadrature rule --- 
	const int sn_order = driver["sn_order"]; 
	const std::string sn_type = io::GetAndValidateOption(driver, "sn_quadrature_type", 
		{"level symmetric", "abu shumays"}, "level symmetric", root); 
	std::unique_ptr<AngularQuadrature> quad; 
	if (sn_type == "level symmetric") {
		quad = std::make_unique<LevelSymmetricQuadrature>(sn_order, dim); 
	} 
	else if (sn_type == "abu shumays") {
		quad = std::make_unique<AbuShumaysQuadrature>(sn_order, dim); 
	}
	const auto Nomega = quad->Size(); 

	// size of psi 
	TransportVectorExtents psi_ext(G, Nomega, fes.GetVSize());
	const auto psi_size = TotalExtent(psi_ext); 
	const auto psi_size_global = mesh.ReduceInt(psi_size); 

	// size of phi 
	MomentVectorExtents phi_ext(G, 1, fes.GetVSize()); 
	const auto phi_size = TotalExtent(phi_ext); 
	const auto phi_size_global = mesh.ReduceInt(phi_size); 

	// moments
	MomentVectorExtents moment_ext(G, dim+1, fes.GetVSize());

	// temporal parameters 
	const double final_time = driver["final_time"]; 

	// allocate time step control parameters 
	double time = 0.0;
	int cycle = 0, output_cycle = 0;

	// get initial time step
	// optionally get function to change time step 
	double time_step = driver["time_step"];
	sol::optional<sol::function> time_step_func_avail = driver["time_step_function"]; 

	int max_cycles = driver["max_cycles"].get_or(std::numeric_limits<int>::max()); 
	if (max_cycles_override>0) max_cycles = max_cycles_override; 
	const int lump = driver["lump"].get_or(0); 

	// --- output algorithmic options used --- 
	out << YAML::Key << "driver" << YAML::Value << YAML::BeginMap; 
		out << YAML::Key << "fe order" << YAML::Value << fe_order; 
		out << YAML::Key << "opacity fe order" << YAML::Value << sigma_fe_order; 
		out << YAML::Key << "sn order" << YAML::Value << sn_order; 
		out << YAML::Key << "sn quadrature type" << YAML::Value << sn_type; 
		out << YAML::Key << "num angles" << YAML::Value << Nomega; 			
		out << YAML::Key << "basis type" << YAML::Value << basis_type_str; 
		out << YAML::Key << "psi size" << YAML::Value << psi_size_global;
		out << YAML::Key << "time integration" << YAML::Value << YAML::BeginMap; 
			out << YAML::Key << "final time" << YAML::Value << final_time; 
			out << YAML::Key << "time step" << YAML::Key << YAML::BeginMap; 
				out << YAML::Key << "type" << YAML::Value; 
				if (time_step_func_avail) out << "function"; 
				else out << "constant"; 
				out << YAML::Key << "initial value" << YAML::Value << time_step; 
			out << YAML::EndMap; 
		out << YAML::EndMap; 

	// time mass matrix for psi 
	SNTimeMassMatrix Mpsi(fes, psi_ext, IsMassLumped(lump)); 
	// cv/dt matrix for temperature 
	mfem::BilinearForm Mcv(&fes); 
	if (lump)
		Mcv.AddDomainIntegrator(new mfem::LumpedIntegrator(new mfem::MassIntegrator(heat_capacity))); 
	else
		Mcv.AddDomainIntegrator(new mfem::MassIntegrator(heat_capacity)); 
	Mcv.Assemble(); 
	Mcv.Finalize(); 

	// --- allocate solution vectors ---
	enum SolutionIndex {
		PSI = 0, 
		TEMP = 1
	};
	mfem::Array<int> offsets(3); 
	offsets[0] = 0; 
	offsets[SolutionIndex::PSI+1] = psi_size; 
	offsets[SolutionIndex::TEMP+1] = fes.GetVSize(); 
	offsets.PartialSum(); 
	mfem::BlockVector x(offsets), x0(offsets); 
	mfem::Vector psi(x.GetBlock(SolutionIndex::PSI), 0, psi_size); 
	mfem::Vector psi0(x0.GetBlock(SolutionIndex::PSI), 0, psi_size); 
	mfem::ParGridFunction T(&fes, x.GetBlock(SolutionIndex::TEMP), 0); 
	mfem::GridFunctionCoefficient Tcoef(&T); 
	mfem::ParGridFunction T0(&fes, x0.GetBlock(SolutionIndex::TEMP), 0); 

	// moments of psi 
	mfem::Vector moments_nu(TotalExtent(moment_ext));
	mfem::Vector Enu(moments_nu, 0, phi_size);
	mfem::Vector Fnu(moments_nu, phi_size, phi_size*dim);
	FirstMomentCoefficient Fnu_coef(fes, moment_ext, moments_nu);

	// gray moments 
	mfem::Vector moments(fes.GetVSize() * (dim+1));
	mfem::ParGridFunction E(&fes, moments, 0);
	mfem::ParGridFunction F(&vfes, moments, fes.GetVSize());

	// piecewise constant temperature 
	mfem::ParGridFunction Tpw(&fes0); 

	// --- load initial conditions ---
	// two options are supported
	// 1) load from a restart file from a previous simulation 
	// 2) equilibrium between radiation and material temperature function 
	// input from lua 
	sol::optional<sol::table> restart_table_avail = driver["restart"];
	if (restart_table_avail) {
		sol::table restart_table = restart_table_avail.value();
		const std::string restart_file = restart_table["path"];
		const int id = restart_table["id"].get_or(0);
		// loads data, grabs cycle, time, time_step from restart file 
		LoadFromRestart(MPI_COMM_WORLD, restart_file, id, x, output_cycle, cycle, time, time_step);
		x0 = x;
		// account for cycle != 0 in max cycles 
		if (max_cycles < std::numeric_limits<int>::max() - cycle)
			max_cycles += cycle;

		out << YAML::Key << "restart" << YAML::Value << YAML::BeginMap;
			out << YAML::Key << "path" << YAML::Value << restart_file;
			out << YAML::Key << "id" << YAML::Value << id;
		out << YAML::EndMap;
	} 

	else {
		// project initial condition 
		sol::function ic_lua = lua["initial_condition"]; 
		if (!ic_lua.valid()) { MFEM_ABORT("must supply initial condition function"); }
		auto ic_func = [&ic_lua](const mfem::Vector &x) {
			double pos[3]; 
			for (int d=0; d<x.Size(); d++) { pos[d] = x(d); }
			return ic_lua(pos[0], pos[1], pos[2]); 
		};
		mfem::FunctionCoefficient ic_coef(ic_func);
		Tpw.ProjectCoefficient(ic_coef);
		T.ProjectGridFunction(Tpw);
		T0 = T; 

		// radiation initial condition
		// assume radiation and material in equilibrium 
		// => psi_g = 1/4pi B_g(T0)
		PlanckEmissionPSCoefficient planck_coef(Tcoef);
		ProjectPsi(fes, *quad, energy_grid, planck_coef, psi0);
		psi = psi0;
	}

	// initial opacity 
	total_coef.SetTemperature(Tcoef); 
	total_coef.SetDensity(density); 
	ProjectedVectorCoefficient total(sigma_fes, total_coef);
	total.Project();

	// kinetic to continuum operators 
	DiscreteToMoment D(*quad, psi_ext, phi_ext); 
	DiscreteToMoment Dlin(*quad, psi_ext, moment_ext);
	GroupCollapseOperator to_gray_op(phi_ext);
	GroupCollapseOperator to_gray_op_moments(moment_ext);
	Dlin.Mult(psi, moments_nu);
	to_gray_op_moments.Mult(moments_nu, moments);
	E *= 1.0/constants::SpeedOfLight;
	mfem::ProductOperator Dgray(&to_gray_op, &D, false, false);

	// opacity weighting operators 
	// energy density weighted 
	ZerothMomentCoefficient Enu_coef(fes, phi_ext, Enu);
	OpacityGroupCollapseCoefficient sigmaE(total, Enu_coef);
	ProjectedCoefficient totalE(gray_sigma_fes, sigmaE);
	totalE.Project();

	// planck weighted 
	PlanckSpectrumMGCoefficient planck_coef(energy_grid.Bounds(), Tcoef);
	OpacityGroupCollapseCoefficient sigmaP(total, planck_coef);
	ProjectedCoefficient totalP(gray_sigma_fes, sigmaP);
	totalP.Project();

	// rosseland weighted 
	RosselandSpectrumMGCoefficient rosseland_coef(energy_grid.Bounds(), Tcoef);
	OpacityGroupCollapseCoefficient sigmaR(total, rosseland_coef);
	InverseOpacityGroupCollapseCoefficient sigmaRinv(total, rosseland_coef);
	ProjectedCoefficient totalR(gray_sigma_fes, sigmaR);
	ProjectedCoefficient totalRinv(gray_sigma_fes, sigmaRinv);
	totalR.Project();
	totalRinv.Project();

	// collapse to gray with rosseland spectrum 
	WeightedGroupCollapseOperator to_gray_ross_op(fes, phi_ext, rosseland_coef);

	// form fixed source term 
	mfem::Vector source_vec(psi_size); 
	FormTransportSource(fes, *quad, energy_grid, source, inflow, source_vec); 

	// build sweep operator 
	InverseAdvectionOperator Linv(fes, *quad, total, bc_map, lump); 
	sol::optional<sol::table> sweep_opts_avail = driver["sweep_opts"]; 
	if (sweep_opts_avail) {
		sol::table sweep_opts = sweep_opts_avail.value();
		io::SetSweepOptions(sweep_opts, Linv, root);
		out << YAML::Key << "sweep options" << YAML::Value << sweep_opts;
	}
	out << YAML::Key << "lumping type" << YAML::Value << YAML::BeginMap; 
		out << YAML::Key << "mass" << YAML::Value << IsMassLumped(lump); 
		out << YAML::Key << "gradient" << YAML::Value << IsGradientLumped(lump); 
		out << YAML::Key << "faces" << YAML::Value << IsFaceLumped(lump); 
	out << YAML::EndMap; 
	if (sweep_opts_avail) out << YAML::Key << "sweep options" << YAML::Value << sweep_opts_avail.value(); 
	if (Linv.IsParallelBlockJacobi()) Linv.Exchange(psi);

	std::unique_ptr<io::NegativeFluxFixup> nff;
	sol::optional<sol::table> fixup_avail = driver["fixup"]; 
	if (fixup_avail) {
		sol::table fixup = fixup_avail.value(); 
		nff.reset(io::CreateNegativeFluxFixup(fixup, root));
		// nff = std::make_unique<io::NegativeFluxFixup>(fixup, out, root);
		Linv.SetFixupOperator(*nff); 
		out << YAML::Key << "negative flux fixup" << YAML::Value << fixup; 
	}
	Linv.SetTimeAbsorption(1.0/time_step/constants::SpeedOfLight); 

	// energy balance nonlinear form 
	// cv/dt + sigma B(T)
	mfem::ProductCoefficient Cvdt(1.0/time_step, heat_capacity); 
	DenseBlockDiagonalNonlinearForm meb_form(&fes);
	if (IsMassLumped(lump)) {
		meb_form.AddDomainIntegrator(
			new QuadratureLumpedNFIntegrator(new GrayPlanckEmissionNFI(energy_grid.Bounds(), total))); 
		meb_form.AddDomainIntegrator(new QuadratureLumpedIntegrator(new mfem::MassIntegrator(Cvdt))); 
	}
	else {
		meb_form.AddDomainIntegrator(new GrayPlanckEmissionNFI(energy_grid.Bounds(), total, 2, 2 + sigma_fe_order)); 
		meb_form.AddDomainIntegrator(new mfem::MassIntegrator(Cvdt)); 
	}

	DenseBlockDiagonalNonlinearForm meb_form_totalP(&fes);
	if (IsMassLumped(lump)) {
		meb_form_totalP.AddDomainIntegrator(new QuadratureLumpedNFIntegrator(new BlackBodyEmissionNFI(totalP))); 
		meb_form_totalP.AddDomainIntegrator(new QuadratureLumpedIntegrator(new mfem::MassIntegrator(Cvdt))); 
	}
	else {
		meb_form_totalP.AddDomainIntegrator(new QuadratureLumpedNFIntegrator(new BlackBodyEmissionNFI(totalP))); 
		meb_form_totalP.AddDomainIntegrator(new mfem::MassIntegrator(Cvdt)); 
	}	

	DenseBlockDiagonalNonlinearForm gr_emission_form(&fes);
	gr_emission_form.AddDomainIntegrator(new QuadratureLumpedNFIntegrator(new BlackBodyEmissionNFI(totalP)));

	// emission nonlinear form 
	// sigma_g B_g(T) 
	PlanckEmissionNFI planck_int(energy_grid.Bounds(), total);
	PlanckEmissionNonlinearForm emission_form(fes, phi_ext, planck_int, IsMassLumped(lump));

	// absorption mass matrix for multigroup energy density 
	MultiGroupBilinearForm Mtot(fes, G);
	if (IsMassLumped(lump))
		Mtot.AddDomainIntegrator(new QuadratureLumpedMGIntegrator(new MGMassIntegrator(total)));
	else
		Mtot.AddDomainIntegrator(new MGMassIntegrator(total));
	Mtot.Assemble();
	Mtot.Finalize(); 

	// computes sum_g sigma_g phi_g 
	mfem::ProductOperator Mtot_collapse(&to_gray_op, &Mtot, false, false);

	mfem::BilinearForm Mtot_gray(&fes);
	if (IsMassLumped(lump)) {
		Mtot_gray.AddDomainIntegrator(new QuadratureLumpedIntegrator(new mfem::MassIntegrator(totalE)));
	} else {
		Mtot_gray.AddDomainIntegrator(new mfem::MassIntegrator(totalE));
	}
	Mtot_gray.UsePrecomputedSparsity();
	Mtot_gray.Assemble();
	Mtot_gray.Finalize();

	// solver for local dense matrices 
	// assume diagonal if mass lumped
	std::unique_ptr<mfem::Solver> local_mat_inv; 
	if (IsMassLumped(lump)) 
		local_mat_inv = std::make_unique<DiagonalDenseMatrixInverse>(); 
	else
		local_mat_inv = std::make_unique<mfem::DenseMatrixInverse>(); 

	// --- useful components of moment discretizations --- 
	mfem::Vector nor(dim); 
	nor = 0.0; nor(0) = 1.0; 
	const double alpha = ComputeAlpha(*quad, nor); 
	LinearizedPseudoAbsorptionCoefficient pseudo_abs(Tcoef, Cvdt, totalR);

	sol::table solver = driver["solver"]; 
	if (!solver.valid()) MFEM_ABORT("must supply solver options"); 
	const std::string solver_type = solver["type"]; 
	io::ValidateOption<std::string>("solver type", solver_type, 
		{"picard", "linearized", "ndsa", "inexact newton"}, root); 
	std::unique_ptr<mfem::IterativeSolver> nonlinear_solver, local_meb_solver, linear_solver, lo_solver;
	std::unique_ptr<DenseBlockDiagonalNonlinearSolver> meb_solver;
	std::unique_ptr<DenseBlockDiagonalSolver> linearized_meb_solver;
	std::unique_ptr<mfem::Solver> dsa_solver, fp_wrap;
	std::unique_ptr<MomentDiscretization> Kform; 
	std::unique_ptr<mfem::HypreBoomerAMG> amg; 
	std::unique_ptr<PARSolver> lmfg_solver;
	std::unique_ptr<InverseMomentDiscretization> dsa_inv;
	std::unique_ptr<ScallionInnerSolverMonitor> inner_monitor, dsa_monitor, meb_monitor; 
	std::unique_ptr<DiffusionAndEnergyBalanceMonitor> dual_monitor;
	std::unique_ptr<KinsolCallbackData> kinsol_data;
	std::unique_ptr<mfem::Operator> oper, stepper;
	ComponentReductionOperator reducer(offsets, 1); // only compute residual on temperature 
	out << YAML::Key << "solver" << YAML::Value << YAML::BeginMap; 
	out << YAML::Key << "type" << YAML::Value << solver_type; 
	if (solver_type == "picard") {
		sol::table nonlin_solve_table = solver["nonlinear_solver"];
		if (!nonlin_solve_table.valid()) MFEM_ABORT("must supply nonlinear solver"); 
		io::ValidateOption<std::string>("nonlinear solver", nonlin_solve_table["type"], 
			{"fp", "kinsol"}, root); 
		nonlinear_solver.reset(io::CreateIterativeSolver(nonlin_solve_table, MPI_COMM_WORLD)); 

		// create energy balance solver 
		sol::table meb_solver_table = solver["energy_balance_solver"]; 
		if (!meb_solver_table.valid()) MFEM_ABORT("must supply solver for energy balance"); 
		local_meb_solver = std::make_unique<EnergyBalanceNewtonSolver>();
		io::SetIterativeSolverOptions(meb_solver_table, *local_meb_solver);
		local_meb_solver->SetPreconditioner(*local_mat_inv);
		meb_solver = std::make_unique<DenseBlockDiagonalNonlinearSolver>(*local_meb_solver);
		meb_solver->SetOperator(meb_form);
		inner_monitor = std::make_unique<BlockNonlinearSolverMonitor>(*meb_solver);

		oper = std::make_unique<PicardTRTOperator>(offsets, Linv, D, emission_form, Mtot_collapse, 
			*meb_solver, *nonlinear_solver);
		fp_wrap = std::make_unique<FixedPointSolverWrapper>(*nonlinear_solver);
		auto *rsolver = new ReducedSolver(*fp_wrap, reducer);
		rsolver->SetOperator(*oper);
		stepper.reset(rsolver);
		nonlinear_solver->SetMonitor(*inner_monitor); 

		auto *sundials = dynamic_cast<mfem::KINSolver*>(nonlinear_solver.get()); 
		if (sundials) {
			mfem::IdentityOperator identity(fes.GetVSize()); 
			sundials->SetOperator(identity); 
			kinsol_data = std::make_unique<KinsolCallbackData>(inner_monitor.get()); 
			KINSetInfoHandlerFn(sundials->GetMem(), KinsolCallbackFunction, kinsol_data.get()); 
			KINSetErrHandlerFn(sundials->GetMem(), io::SundialsErrorFunction, nullptr); 
		}

		out << YAML::Key << "nonlinear solver" << YAML::Value << nonlin_solve_table; 
		out << YAML::Key << "energy balance solver" << YAML::Value << meb_solver_table; 
	} 

	else if (solver_type == "linearized") {
		// create nonlinear solver 
		sol::optional<sol::table> nonlin_solve_table_avail = solver["nonlinear_solver"]; 
		if (nonlin_solve_table_avail) {
			sol::table nonlin_solve_table = nonlin_solve_table_avail.value();
			io::ValidateOption<std::string>("nonlinear solver", nonlin_solve_table["type"], 
				{"fp", "kinsol"}, root); 
			nonlinear_solver.reset(io::CreateIterativeSolver(nonlin_solve_table, MPI_COMM_WORLD)); 
			auto *sundials = dynamic_cast<mfem::KINSolver*>(nonlinear_solver.get());
			if (sundials) {
				const std::string strategy = nonlin_solve_table["strategy"]; 
				io::ValidateOption<std::string>("kinsol strategy", strategy, {"none", "linesearch", "picard"}, root); 		
			}
			out << YAML::Key << "nonlinear solver" << YAML::Key << nonlin_solve_table; 			
		}

		// create solver for transport operator 
		sol::table transport_solve_table = solver["transport_solver"]; 
		if (!transport_solve_table.valid()) MFEM_ABORT("must supply transport solver"); 
		linear_solver.reset(io::CreateIterativeSolver(transport_solve_table, MPI_COMM_WORLD)); 

		// create solver for dsa system 
		sol::optional<sol::table> prec_table_avail = transport_solve_table["preconditioner"]; 
		if (prec_table_avail) {
			sol::table prec_table = prec_table_avail.value(); 
			const std::string type = prec_table["type"]; 
			io::ValidateOption<std::string>("preconditioner type", type, {"mip", "ldg"}, root); 

			if (type == "mip") {
				const double kappa = prec_table["kappa"].get_or(pow(fe_order+1,2)); 				
				auto *disc = new InteriorPenaltyDiscretization(fes, totalR, pseudo_abs, bc_map, lump);
				disc->SetPenaltyLowerBound(alpha/2);
				disc->SetKappa(kappa);
				Kform.reset(disc);
			} else if (type == "ldg") {
				auto *disc = new LDGDiscretization(fes, totalR, pseudo_abs, bc_map, lump);
				Kform.reset(disc);
			}
			Kform->SetAlpha(alpha);
			Kform->SetTimeAbsorption(1.0/time_step/constants::SpeedOfLight);

			sol::table solver_table = prec_table["solver"]; 
			dsa_solver.reset(io::CreateSolver(solver_table, MPI_COMM_WORLD));
			auto *it_solver = dynamic_cast<mfem::IterativeSolver*>(dsa_solver.get());
			if (it_solver) {
				amg = std::make_unique<mfem::HypreBoomerAMG>();
				amg->SetPrintLevel(0);
				it_solver->SetPreconditioner(*amg);		
				dsa_monitor = std::make_unique<InnerIterativeSolverMonitor>(*it_solver); 
				linear_solver->SetMonitor(*dsa_monitor);
			}
			// LMFG restricts to gray, solves diffusion, prolongs back to multigroup 
			// rosseland spectrum is used for both restriction and prolongation
			lmfg_solver = std::make_unique<PARSolver>(to_gray_ross_op, *dsa_solver, to_gray_ross_op);
			dsa_inv = std::make_unique<InverseMomentDiscretization>(*Kform, *lmfg_solver);
		}
		out << YAML::Key << "transport solver" << YAML::Value << transport_solve_table; 

		linearized_meb_solver = std::make_unique<DenseBlockDiagonalSolver>(*local_mat_inv);
		if (nonlinear_solver) {
			inner_monitor = std::make_unique<InnerIterativeSolverMonitor>(*linear_solver); 
			nonlinear_solver->SetMonitor(*inner_monitor); 
		}
		auto *op = new LinearizedTRTOperator(
			offsets, Linv, D, emission_form, meb_form, 
			Mtot_collapse, *linear_solver, *linearized_meb_solver);		
		if (dsa_inv) op->SetDSAPreconditioner(*dsa_inv);
		sol::optional<sol::table> meb_solver_table_avail = solver["rebalance_solver"]; 
		if (meb_solver_table_avail) {
			// create energy balance solver 
			sol::table meb_solver_table = meb_solver_table_avail.value();
			local_meb_solver = std::make_unique<EnergyBalanceNewtonSolver>();
			io::SetIterativeSolverOptions(meb_solver_table, *local_meb_solver);
			local_meb_solver->SetPreconditioner(*local_mat_inv);
			meb_solver = std::make_unique<DenseBlockDiagonalNonlinearSolver>(*local_meb_solver);
			meb_solver->SetOperator(meb_form);
			op->SetRebalanceSolver(*meb_solver);
			out << YAML::Key << "rebalance solver" << YAML::Value << meb_solver_table;
		}
		if (nonlinear_solver) {
			if (Kform)
				op->SetGrayOpacities(Kform->GetTotal(), Kform->GetAbsorption());
			oper.reset(op);
			fp_wrap = std::make_unique<FixedPointSolverWrapper>(*nonlinear_solver);
			auto *rsolver = new ReducedSolver(*fp_wrap, reducer);
			rsolver->SetOperator(*oper);
			stepper.reset(rsolver);
		} else {
			stepper.reset(op);
		}
	}

	else if (solver_type == "inexact newton") {
		// create nonlinear solver 
		sol::optional<sol::table> nonlin_solve_table_avail = solver["nonlinear_solver"]; 
		if (nonlin_solve_table_avail) {
			sol::table nonlin_solve_table = nonlin_solve_table_avail.value();
			io::ValidateOption<std::string>("nonlinear solver", nonlin_solve_table["type"], 
				{"fp", "kinsol"}, root); 
			nonlinear_solver.reset(io::CreateIterativeSolver(nonlin_solve_table, MPI_COMM_WORLD)); 
			auto *sundials = dynamic_cast<mfem::KINSolver*>(nonlinear_solver.get());
			if (sundials) {
				const std::string strategy = nonlin_solve_table["strategy"]; 
				io::ValidateOption<std::string>("kinsol strategy", strategy, {"fp"}, root); 		
			}
			out << YAML::Key << "nonlinear solver" << YAML::Key << nonlin_solve_table; 			
		}

		// create solver for dsa system 
		sol::optional<sol::table> prec_table_avail = solver["preconditioner"]; 
		if (prec_table_avail) {
			sol::table prec_table = prec_table_avail.value(); 
			const std::string type = prec_table["type"]; 
			io::ValidateOption<std::string>("preconditioner type", type, {"mip", "ldg"}, root); 

			if (type == "mip") {
				const double kappa = prec_table["kappa"].get_or(pow(fe_order+1,2)); 				
				auto *disc = new InteriorPenaltyDiscretization(fes, totalR, pseudo_abs, bc_map, lump);
				disc->SetPenaltyLowerBound(alpha/2);
				disc->SetKappa(kappa);
				Kform.reset(disc);
			} else if (type == "ldg") {
				auto *disc = new LDGDiscretization(fes, totalR, pseudo_abs, bc_map, lump);
				Kform.reset(disc);
			}
			Kform->SetAlpha(alpha);
			Kform->SetTimeAbsorption(1.0/time_step/constants::SpeedOfLight);

			sol::table solver_table = prec_table["solver"]; 
			dsa_solver.reset(io::CreateSolver(solver_table, MPI_COMM_WORLD));
			auto *it_solver = dynamic_cast<mfem::IterativeSolver*>(dsa_solver.get());
			if (it_solver) {
				amg = std::make_unique<mfem::HypreBoomerAMG>();
				amg->SetPrintLevel(0);
				it_solver->SetPreconditioner(*amg);		
				dsa_monitor = std::make_unique<InnerIterativeSolverMonitor>(*it_solver); 
			}
			// LMFG restricts to gray, solves diffusion, prolongs back to multigroup 
			// rosseland spectrum is used for both restriction and prolongation
			lmfg_solver = std::make_unique<PARSolver>(to_gray_ross_op, *dsa_solver, to_gray_ross_op);
			dsa_inv = std::make_unique<InverseMomentDiscretization>(*Kform, *lmfg_solver);
			out << YAML::Key << "preconditioner" << YAML::Value << prec_table;
		}

		auto *sundials = dynamic_cast<mfem::KINSolver*>(nonlinear_solver.get()); 
		if (sundials) {
			mfem::IdentityOperator identity(reducer.Height()); 
			sundials->SetOperator(identity); 
			kinsol_data = std::make_unique<KinsolCallbackData>(dsa_monitor.get()); 
			KINSetInfoHandlerFn(sundials->GetMem(), KinsolCallbackFunction, kinsol_data.get()); 
			KINSetErrHandlerFn(sundials->GetMem(), io::SundialsErrorFunction, nullptr); 
		}

		linearized_meb_solver = std::make_unique<DenseBlockDiagonalSolver>(*local_mat_inv);

		sol::table meb_solver_table = solver["energy_balance_solver"]; 
		if (!meb_solver_table.valid()) MFEM_ABORT("must supply energy balance solver table");
		local_meb_solver = std::make_unique<EnergyBalanceNewtonSolver>();
		io::SetIterativeSolverOptions(meb_solver_table, *local_meb_solver);
		local_meb_solver->SetPreconditioner(*local_mat_inv);
		meb_solver = std::make_unique<DenseBlockDiagonalNonlinearSolver>(*local_meb_solver);
		meb_solver->SetOperator(meb_form_totalP);
		meb_monitor = std::make_unique<BlockNonlinearSolverMonitor>(*meb_solver);
		out << YAML::Key << "energy balance solver" << YAML::Value << meb_solver_table;

		dual_monitor = std::make_unique<DiffusionAndEnergyBalanceMonitor>(*dsa_monitor, *meb_monitor);
		nonlinear_solver->SetMonitor(*dual_monitor);

		auto *op = new InexactNewtonTRTOperator(offsets, Linv, D, emission_form, meb_form_totalP, 
			Mtot_collapse, *meb_solver, *linearized_meb_solver);
		if (dsa_inv) op->SetDSAPreconditioner(*dsa_inv);
		op->SetGrayOpacities(totalR, totalP);
		oper.reset(op);
		fp_wrap = std::make_unique<FixedPointSolverWrapper>(*nonlinear_solver);
		auto *rsolver = new ReducedSolver(*fp_wrap, reducer);
		rsolver->SetOperator(*oper);
		stepper.reset(rsolver);
	}

	else if (solver_type == "ndsa") {
		sol::table nonlin_solve_table = solver["outer_solver"];
		if (!nonlin_solve_table.valid()) MFEM_ABORT("must supply outer_solver"); 
		io::ValidateOption<std::string>("nonlinear solver", nonlin_solve_table["type"], 
			{"fp", "kinsol"}, root); 
		nonlinear_solver.reset(io::CreateIterativeSolver(nonlin_solve_table, MPI_COMM_WORLD)); 
		out << YAML::Key << "outer solver" << YAML::Value << nonlin_solve_table;

		sol::table meb_solver_table = solver["energy_balance_solver"]; 
		if (!meb_solver_table.valid()) MFEM_ABORT("must supply energy balance solver table");
		local_meb_solver = std::make_unique<EnergyBalanceNewtonSolver>();
		io::SetIterativeSolverOptions(meb_solver_table, *local_meb_solver);
		local_meb_solver->SetPreconditioner(*local_mat_inv);
		meb_solver = std::make_unique<DenseBlockDiagonalNonlinearSolver>(*local_meb_solver);
		meb_solver->SetOperator(meb_form_totalP);
		meb_monitor = std::make_unique<BlockNonlinearSolverMonitor>(*meb_solver);

		out << YAML::Key << "energy balance solver" << YAML::Value << meb_solver_table;

		sol::table lo_solver_table = solver["inner_solver"];
		if (!lo_solver_table.valid()) MFEM_ABORT("must supply lo solver");
		lo_solver.reset(io::CreateIterativeSolver(lo_solver_table, MPI_COMM_WORLD));
		inner_monitor = std::make_unique<InnerIterativeSolverMonitor>(*lo_solver); 
		auto *sundials = dynamic_cast<mfem::KINSolver*>(nonlinear_solver.get()); 
		if (sundials) {
			mfem::IdentityOperator identity(reducer.Height()); 
			sundials->SetOperator(identity); 
			kinsol_data = std::make_unique<KinsolCallbackData>(inner_monitor.get()); 
			KINSetInfoHandlerFn(sundials->GetMem(), KinsolCallbackFunction, kinsol_data.get()); 
			KINSetErrHandlerFn(sundials->GetMem(), io::SundialsErrorFunction, nullptr); 
		} else {
			nonlinear_solver->SetMonitor(*inner_monitor); 			
		}
		out << YAML::Key << "inner solver" << YAML::Value << lo_solver_table;		

		sol::table linear_solver_table = solver["linear_solver"];
		if (!linear_solver_table.valid()) MFEM_ABORT("must supply linear solver");
		dsa_solver.reset(io::CreateSolver(linear_solver_table, MPI_COMM_WORLD));
		auto *it_solver = dynamic_cast<mfem::IterativeSolver*>(dsa_solver.get());
		if (it_solver) {
			amg = std::make_unique<mfem::HypreBoomerAMG>();
			amg->SetPrintLevel(0);
			it_solver->SetPreconditioner(*amg);		
			dsa_monitor = std::make_unique<InnerIterativeSolverMonitor>(*it_solver); 
		}
		out << YAML::Key << "linear solver" << YAML::Value << linear_solver_table;

		dual_monitor = std::make_unique<DiffusionAndEnergyBalanceMonitor>(*dsa_monitor, *meb_monitor);
		lo_solver->SetMonitor(*dual_monitor);

		const auto lo_type = io::GetAndValidateOption<std::string>(solver, "lo_type", {"ldg", "mip"}, "mip", root);
		if (lo_type == "mip") {
			auto *disc = new InteriorPenaltyDiscretization(fes, totalR, totalE, bc_map, lump);
			disc->SetPenaltyLowerBound(alpha/2);
			disc->SetKappa(pow(fe_order+1,2));
			Kform.reset(disc);
		} else if (lo_type == "ldg") {
			auto *disc = new LDGDiscretization(fes, totalR, totalE, bc_map, lump);
			Kform.reset(disc);
		}
		Kform->SetAlpha(alpha);
		Kform->SetTimeAbsorption(1.0/time_step/constants::SpeedOfLight);
		out << YAML::Key << "lo type" << YAML::Value << lo_type;

		auto *ndsa = new NonlinearDSATRTOperator(offsets, Linv, D, to_gray_op,
			emission_form, gr_emission_form, meb_form_totalP, Mtot_gray, 
			*Kform, *dsa_solver, *lo_solver, *meb_solver, Enu, E);
		ndsa->SetGrayOpacities(Kform->GetTotal(), Kform->GetAbsorption(), totalP); // <-- update LO, gray opacities at each outer
		oper.reset(ndsa);
		// reduce ndsa from [psi, T] -> T  
		// solve fixed point problem and return full [psi,T]
		// this avoids computing norms and residuals on the full angular flux 
		fp_wrap = std::make_unique<FixedPointSolverWrapper>(*nonlinear_solver);
		auto *rsolver = new ReducedSolver(*fp_wrap, reducer);
		rsolver->SetOperator(*oper);
		stepper.reset(rsolver);
	}
	out << YAML::EndMap; // end solver output
	out << YAML::EndMap; // end driver output 

	// --- configure outputs --- 
	mfem::ParGridFunction cvgf(&fes0); cvgf.ProjectCoefficient(heat_capacity); 
	mfem::ParGridFunction density_gf(&fes0); density_gf.ProjectCoefficient(density); 
	mfem::ParGridFunction partition(&fes0); partition = rank; 
	sol::optional<sol::table> output_avail = lua["output"]; 
	std::unique_ptr<mfem::DataCollection> dc; 
	std::unique_ptr<TracerDataCollection> tracer_dc; 
	std::unique_ptr<RestartWriter> restart_dc;
	int output_freq, restart_freq;
	std::deque<double> write_times;
	if (output_avail) {
		out << YAML::Key << "output" << YAML::Value << YAML::BeginMap; 
		sol::table output = output_avail.value(); 
		const std::string output_root = output["root"];
		if (output_root == "" and root) MFEM_ABORT("must supply output root");  
		out << YAML::Key << "root" << YAML::Value << io::ResolveRelativePath(output_root); 
		sol::optional<sol::table> viz_avail = output["visualization"];
		if (viz_avail) {
			sol::table viz = viz_avail.value(); 
			const std::string type = viz["type"];
			dc.reset(io::CreateDataCollection(type, output_root, mesh, root));
			output_freq = viz["frequency"].get_or(std::numeric_limits<int>::max()); 
			const int precision = viz["precision"].get_or(6); 
			const bool restart_mode = viz["restart_mode"].get_or(false) and restart_table_avail;
			dc->SetPrecision(precision); 
			dc->RegisterField("E", &E); 
			dc->RegisterField("F", &F);
			dc->RegisterField("T", &T); 
			dc->RegisterField("Tpw", &Tpw); 
			dc->RegisterField("sigmaR", &totalR.GetGridFunction()); 
			dc->RegisterField("sigmaE", &totalE.GetGridFunction());
			dc->RegisterField("cv", &cvgf); 
			dc->RegisterField("density", &density_gf); 
			dc->RegisterField("partition", &partition); 
			// paraview has a "restart mode" 
			// that allows continuing the same output after a restart 
			if (restart_mode) {
				auto *paraview = dynamic_cast<mfem::ParaViewDataCollection*>(dc.get());
				if (paraview)
					paraview->UseRestartMode(true);
				else
					if (root) MFEM_ABORT("restart mode supported for paraview only");
			}

			// only save initial condition if restart mode is off 
			else {
				dc->SetCycle(output_cycle); dc->SetTime(time); dc->SetTimeStep(time_step); 
				dc->Save(); 
			}

			sol::optional<sol::table> times_avail = viz["times"];
			if (times_avail) {
				sol::table times = times_avail.value();
				for (int i=1; i<=times.size(); i++) {
					write_times.push_back(times[i]);
				}
			}

			out << YAML::Key << "visualization" << YAML::Value << YAML::BeginMap; 
				out << YAML::Key << "type" << YAML::Value << type; 
				out << YAML::Key << "frequency" << YAML::Value << output_freq; 
				out << YAML::Key << "precision" << YAML::Value << precision; 
				out << YAML::Key << "restart mode" << YAML::Value << restart_mode;
			if (write_times.size()) {
				out << YAML::Key << "write times" << YAML::Value << YAML::BeginSeq; 
				for (const auto &time : write_times) {
					out << time;
				}
				out << YAML::EndSeq;
			}
			out << YAML::EndMap; 
		} 

		sol::optional<sol::table> tracer_avail = output["tracer"]; 
		if (tracer_avail) {
			sol::table tracer = tracer_avail.value(); 
			sol::table locations = tracer["locations"]; 
			auto ntracers = locations.size(); 
			mfem::DenseMatrix pts(dim, ntracers); 
			for (int i=0; i<ntracers; i++) {
				sol::table tracer_pts = locations[i+1]; 
				for (int d=0; d<tracer_pts.size(); d++) {
					pts(d,i) = tracer_pts[d+1]; 
				}
			}
			const auto prefix = tracer["prefix"].get_or(std::string("tracer")); 
			const int precision = tracer["precision"].get_or(10); 
			const bool restart_mode = tracer["restart_mode"].get_or(false) and restart_table_avail;
			tracer_dc = std::make_unique<TracerDataCollection>(prefix, mesh, pts); 
			tracer_dc->SetPrefixPath(output_root); 
			tracer_dc->SetPrecision(precision); 
			tracer_dc->RegisterField("E", &E); 
			tracer_dc->RegisterField("F", &F);
			tracer_dc->RegisterField("T", &T); 
			tracer_dc->RegisterField("Tpwc", &Tpw); 
			tracer_dc->RegisterField("sigmaR", &totalR.GetGridFunction()); 
			tracer_dc->RegisterField("sigmaE", &totalE.GetGridFunction());
			tracer_dc->RegisterField("cv", &cvgf); 
			tracer_dc->RegisterField("density", &density_gf); 
			if (restart_mode)
				tracer_dc->UseRestartMode(true);
			else {
				tracer_dc->SetCycle(cycle); tracer_dc->SetTime(time); tracer_dc->SetTimeStep(time_step); 
				tracer_dc->Save(); 
			}

			out << YAML::Key << "tracer" << YAML::Value << YAML::BeginMap; 
				out << YAML::Key << "prefix" << YAML::Value << prefix; 
				out << YAML::Key << "precision" << YAML::Value << precision; 
				out << YAML::Key << "restart mode" << YAML::Value << restart_mode;
				out << YAML::Key << "locations" << YAML::Value << YAML::BeginSeq; 
				for (int i=0; i<ntracers; i++) {
					sol::table tracer_pts = locations[i+1]; 
					out << YAML::Flow << YAML::BeginSeq;  
					for (int d=0; d<tracer_pts.size(); d++) {
						out << (double)tracer_pts[d+1]; 
					}
					out << YAML::EndSeq; 
				}
				out << YAML::EndSeq; 
			out << YAML::EndMap; 
		}

		sol::optional<sol::table> restart_avail = output["restart"]; 
		if (restart_avail) {
			sol::table restart = restart_avail.value(); 
			auto restart_keep = restart["num_restarts"].get_or(2);
			restart_freq = restart["frequency"].get_or(std::numeric_limits<int>::max());
			std::string restart_root = restart["prefix"].get_or(std::string("restart"));
			restart_dc = std::make_unique<RestartWriter>(MPI_COMM_WORLD, output_root + "/" + restart_root);
			restart_dc->SetNumRestartFiles(restart_keep);

			out << YAML::Key << "restart" << YAML::Value << YAML::BeginMap; 
				out << YAML::Key << "prefix" << YAML::Value << restart_root;
				out << YAML::Key << "num restarts" << YAML::Value << restart_keep;
				out << YAML::Key << "frequency" << YAML::Value << restart_freq;
			out << YAML::EndMap;
		}
		out << YAML::EndMap; // end output map
	}

	mfem::StopWatch cycle_timer; 
	LogMap<int,MAX> log;
	out << YAML::Key << "time integration" << YAML::BeginSeq; 
	while (true) {
		cycle_timer.Restart(); 

		// --- apply mass matrix to previous time step --- 
		Mpsi.Mult(psi0, psi0); // operator designed to work in-place 
		add(source_vec, 1.0/time_step/constants::SpeedOfLight, psi0, psi0); // source_vec + 1/c/dt psi0 -> psi0
		Mcv.Mult(T, T0); // assume T = T0 to get Mcv T0 -> T0 
		T0 *= 1.0/time_step; 

		// --- get new time step solution --- 
		stepper->Mult(x0, x); 

		// --- post process new time solution --- 
		Dlin.Mult(psi, moments_nu); // update energy density 
		to_gray_op_moments.Mult(moments_nu, moments);

		// --- compute energy balance --- 
		const auto balance = ComputeBalance(Linv, D, emission_form, psi0, psi, T);

		E *= 1.0/constants::SpeedOfLight; 
		if (E.CheckFinite() > 0) MFEM_ABORT("infinite energy density"); 

		// get peicewise constant version of temperature 
		// used for comparison to other codes 
		Tpw.ProjectGridFunction(T); 

		// --- update time step info --- 
		time += time_step; 
		cycle++; 

		// --- output to file --- 
		// write data collection to file 
		bool done = time >= final_time - 1e-14 or cycle >= max_cycles; 
		double write_time = write_times.front();
		const bool write = std::fabs(write_time - time) < time_step / 2;
		if (write) write_times.pop_front();
		if (dc and (cycle % output_freq == 0 or write or done)) {
			output_cycle++;
			dc->SetCycle(output_cycle); dc->SetTime(time); dc->SetTimeStep(time_step); 
			dc->Save(); 			
		}

		// write tracer to file 
		// output every time step 
		if (tracer_dc) {
			tracer_dc->SetCycle(cycle); tracer_dc->SetTime(time); tracer_dc->SetTimeStep(time_step); 
			tracer_dc->Save(); 
		}

		// write restart to file 
		if (restart_dc and (cycle % restart_freq == 0 or done)) {
			restart_dc->SetOutputCycle(output_cycle); restart_dc->SetSimulationCycle(cycle);
			restart_dc->SetTime(time); restart_dc->SetTimeStep(time_step);
			restart_dc->Write(x);
		}

		// check for new time step size 
		bool time_step_changed = false; 
		const double time_step_old = time_step;
		// reduce time step to end at final_time, if needed 
		if (time + time_step > final_time) {
			time_step_changed = true;
			time_step = final_time - time;
		}
		// query time step function for new value 
		else if (time_step_func_avail) {
			double new_time_step = time_step_func_avail.value()(time, time_step); 
			time_step_changed = std::fabs(new_time_step - time_step) > 1e-14; 
			time_step = new_time_step; 
		}

		// --- update opacity and opacity-dependent terms --- 
		if (T.Min() < 0) MFEM_ABORT("negative temperature"); 
		// recompute opacities 
		total.Project(); 
		totalR.Project();
		totalRinv.Project();
		totalE.Project();

		// recompute sweep data 
		Linv.AssembleLocalMatrices(); 
		Linv.SetTimeAbsorption(1.0/time_step/constants::SpeedOfLight);

		// energy balance time step 
		Cvdt.SetAConst(1.0/time_step); 

		// total interaction mass matrix
		// depends on sigma and dt 
		Mtot.Assemble(); 
		Mtot.Finalize();

		Mtot_gray = 0.0;
		Mtot_gray.Assemble();

		// DSA matrix depends on sigma and dt 
		if (Kform) {
			// set DSA time step 
			Kform->SetTimeAbsorption(1.0/time_step/constants::SpeedOfLight);
		}

		// store time step 
		x0 = x; 

		cycle_timer.Stop(); 
		double cycle_time = cycle_timer.RealTime(); 

		// warn if temperature is such that the 
		// energy group structure can't fully integrate
		// the planck spectrum 
		CheckPlanckSpectrumCovered(MPI_COMM_WORLD, energy_grid.MinEnergy(), 
			energy_grid.MaxEnergy(), T, 1e-10);

		// --- output progress to terminal --- 
		const double radE_norm = sqrt(mfem::InnerProduct(MPI_COMM_WORLD, E, E)); 
		out << YAML::BeginMap; 
			out << YAML::Key << "cycle" << YAML::Value << cycle; 
			out << YAML::Key << "simulation time" << YAML::Value << time; 
			out << YAML::Key << "time step size" << YAML::Value << time_step_old; 
			out << YAML::Key << "||radE||" << YAML::Value << io::FormatScientific(radE_norm); 
			if (nonlinear_solver) {
				log.Log("max nonlinear iterations", nonlinear_solver->GetNumIterations());
				if (inner_monitor)
					log.Log("max inner iterations", inner_monitor->iters.Max());
				out << YAML::Key << "it" << YAML::Value << nonlinear_solver->GetNumIterations(); 
				out << YAML::Key << "norm" << YAML::Value; 
				if (kinsol_data) 
					out << io::FormatScientific(kinsol_data->norm); 
				else {
					out << YAML::Flow << YAML::BeginMap
						<< YAML::Key << "absolute" << YAML::Value << io::FormatScientific(nonlinear_solver->GetFinalNorm())
						<< YAML::Key << "relative" << YAML::Value << io::FormatScientific(nonlinear_solver->GetFinalRelNorm()); 
					out << YAML::EndMap;
				}
			}
			if (inner_monitor) {
				out << YAML::Key << "inner solver" << YAML::Value << *inner_monitor;
			} else if (linear_solver) {
				log.Log("max transport iterations", linear_solver->GetNumIterations());
				out << YAML::Key << "linear solver" << YAML::Value << YAML::BeginMap; 
					out << YAML::Key << "it" << YAML::Value << linear_solver->GetNumIterations(); 
					out << YAML::Key << "norm" << YAML::Value << io::FormatScientific(linear_solver->GetFinalRelNorm()); 
				out << YAML::EndMap; 
			}
			if (dsa_monitor) {
				log.Log("max DSA iterations", dsa_monitor->iters.Max());
				out << YAML::Key << "dsa solver" << YAML::Value << *dsa_monitor;
			}
			if (meb_monitor) {
				out << YAML::Key << "energy balance solver" << YAML::Value << *meb_monitor;
			}
			io::ProcessGlobalLogs(out, log_verbosity);
			out << YAML::Key << "energy balance" << YAML::Value << io::FormatScientific(balance);
			out << YAML::Key << "cycle time" << YAML::Value << io::FormatTimeString(cycle_time); 
		out << YAML::EndMap << YAML::Newline; 

		// warn if max cycles reached 
		if (cycle == max_cycles and root) 
			MFEM_WARNING("max cycles reached. simulation end time not equal to final time"); 

		// end time integration 
		if (done) break; 
	}
	out << YAML::EndSeq; // time integration sequence 

	if (log.size()) {
		out << YAML::Key << "log" << YAML::Value << log;
	}

	// print logs persistent across all time steps 
	if (io::TimingLogPersistent.size()) {
		out << YAML::Key << "timings" << YAML::Value;
		io::PrintTimingMap(out, io::TimingLogPersistent);
	}
	if (io::EventLogPersistent.size()) {
		out << YAML::Key << "event log" << YAML::Value << io::EventLogPersistent;
	}

	// --- clean up hanging pointers --- 
	for (int i=0; i<nattr; i++) { delete total_list[i]; } 
	for (int i=0; i<nattr; i++) { delete source_list[i]; }
	for (int i=0; i<nbattr; i++) { delete inflow_list[i]; }
	for (int i=0; i<nbattr; i++) { delete inflow_base_list[i]; }
	delete[] partitioning;

	wall_timer.Stop(); 
	double wall_time = wall_timer.RealTime(); 
	out << YAML::Key << "wall time" << YAML::Value << io::FormatTimeString(wall_time); 

	// --- end yaml output --- 
	out << YAML::EndMap << YAML::Newline; 
}