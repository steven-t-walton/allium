#include "mfem.hpp"
#include "yaml-cpp/yaml.h"
#include "sol/sol.hpp"

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
#include "lumping.hpp"
#include "tracer.hpp"
#include "block_diag_op.hpp"
#include "moment_discretization.hpp"
#include "smm_source.hpp"
#include "multigroup.hpp"

class ScallionInnerSolverMonitor : public mfem::IterativeSolverMonitor
{
public:
	mfem::Array<int> iters; 
	double max_norm = 0.0; 

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
			out << YAML::Key << "max norm" << YAML::Key << monitor.max_norm; 
		out << YAML::EndMap; 
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
			max_norm = std::max(solver.GetFinalRelNorm(), max_norm); 
		}
	}
};

class BlockNonlinearSolverMonitor : public ScallionInnerSolverMonitor
{
private:
	const BlockDiagonalByElementNonlinearSolver &solver; 
public:
	BlockNonlinearSolverMonitor(const BlockDiagonalByElementNonlinearSolver &s) 
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
			max_norm = std::max(max_norm, solver.GetFinalRelNorm()); 		
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
	bool temp_dependent_opacity = false; 
	std::vector<sol::object> lua_source_objs(nattr); 
	mfem::Array<OpacityCoefficient*> total_list(nattr); 
	mfem::Array<PhaseSpaceCoefficient*> source_list(nattr); 
	out << YAML::Key << "materials" << YAML::Value << YAML::BeginSeq; 
	for (auto i=0; i<attr_list.size(); i++) {
		sol::table data = materials[attr_list[i].c_str()]; 
		if (!data.valid()) MFEM_ABORT("material named " << attr_list[i] << " not found"); 
		sol::table total = data["total"]; 
		std::string type = total["type"]; 
		io::ValidateOption<std::string>("opacity type", type, {"constant", "analytic gray"}, root); 
		if (type == "constant") {
			sol::table values = total["values"];
			mfem::Vector vec(values.size()); 
			for (int i=0; i<vec.Size(); i++) { vec(i) = values[i+1]; }
			total_list[i] = new ConstantOpacityCoefficient(vec);  
		}

		else if (type == "analytic gray") {
			double coef = total["coef"]; 
			int nrho = total["nrho"]; 
			int nT = total["nT"]; 
			total_list[i] = new AnalyticGrayOpacityCoefficient(coef, nrho, nT); 
			temp_dependent_opacity = true; 
		}
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

		out << YAML::BeginMap; 
			out << YAML::Key << "name" << YAML::Value << attr_list[i]; 
			out << YAML::Key << "attribute" << YAML::Value << i+1; 
			out << YAML::Key << "opacity" << YAML::Value << type; 
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
	mfem::Array<PhaseSpaceCoefficient*> inflow_list(nbattr); 
	BoundaryConditionMap bc_map;
	out << YAML::Key << "boundary conditions" << YAML::Value << YAML::BeginSeq; 
	for (auto i=0; i<bdr_attr_list.size(); i++) {
		sol::table data = bcs[bdr_attr_list[i].c_str()]; 
		std::string type = data["type"]; 
		io::ValidateOption<std::string>("boundary_conditions::type", type, {"inflow", "reflective", "vacuum"}, root); 
		double value;
		if (type == "inflow") {
			value = data["value"]; 
			double planck_value = constants::StefanBoltzmann * pow(value, 4) / 4.0 / constants::pi; 
			inflow_list[i] = new ConstantPhaseSpaceCoefficient(planck_value); 
			bc_map[i+1] = BoundaryCondition::INFLOW;
		} else if (type == "reflective") {
			inflow_list[i] = nullptr; 
			bc_map[i+1] = BoundaryCondition::REFLECTIVE;
		} 
		else if (type == "vacuum") {
			inflow_list[i] = nullptr; 
			bc_map[i+1] = BoundaryCondition::INFLOW;
		}

		out << YAML::BeginMap; 
			out << YAML::Key << "name" << YAML::Value << bdr_attr_list[i]; 
			out << YAML::Key << "attribute" << YAML::Value << i+1; 
			out << YAML::Key << "type" << YAML::Value << type; 
			if (inflow_list[i]) out << YAML::Key << "value" << YAML::Value << value; 
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
	const int sigma_fe_order = driver["sigma_fe_order"].get_or(fe_order); 
	std::string basis_type_str = io::GetAndValidateOption(driver, "basis_type", 
		{"lobatto", "legendre", "positive"}, "lobatto", root); 

	// load energy grid 
	mfem::Array<double> energy_grid(2); 
	energy_grid[0] = 0.0; energy_grid[1] = 1.0; 

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

	// piecewise constant used for plotting and storing cross section data 
	mfem::L2_FECollection sigma_fec(sigma_fe_order, dim, mfem::BasisType::Positive); 
	mfem::ParFiniteElementSpace sigma_fes(&mesh, &sigma_fec); 

	// --- create coefficients for input material data ---
	// list of attributes in order of total_list 
	mfem::Array<int> attrs(nattr); 
	for (int i=0; i<nattr; i++) { attrs[i] = i+1; }
	PWOpacityCoefficient total_coef(attrs, total_list); 
	mfem::ParGridFunction total_gf(&sigma_fes); 

	// heat capacity 
	for (int i=0; i<cv_list.Size(); i++) cv_list(i) *= density_list(i); 
	mfem::PWConstCoefficient heat_capacity(cv_list); 
	mfem::PWConstCoefficient density(density_list); 

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

	TransportVectorExtents psi_ext(1, Nomega, fes.GetVSize());
	const auto psi_size = TotalExtent(psi_ext); 
	const auto psi_size_global = mesh.ReduceInt(psi_size); 

	MomentVectorExtents phi_ext(1, 1, fes.GetVSize()); 
	const auto phi_size = TotalExtent(phi_ext); 
	const auto phi_size_global = mesh.ReduceInt(phi_size); 

	// temporal parameters 
	const double final_time = driver["final_time"]; 

	sol::optional<sol::function> time_step_func_avail = driver["time_step"]; 
	sol::optional<double> time_step_value_avail = driver["time_step"]; 
	double time_step; 
	if (time_step_func_avail) {
		time_step = time_step_func_avail.value()(0.0); 
	} else if (time_step_value_avail) {
		time_step = time_step_value_avail.value(); 
	} else {
		MFEM_ABORT("must supply time step"); 
	}

	int max_cycles = driver["max_cycles"].get_or(std::numeric_limits<int>::max()); 
	if (max_cycles_override>0) max_cycles = max_cycles_override; 
	const int lump = driver["lump"].get_or(0); 
	sol::optional<sol::table> sweep_opts_avail = driver["sweep_opts"]; 

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

	SNTimeMassMatrix Mpsi(fes, psi_ext, lump); 
	mfem::BilinearForm Mcv(&fes); 
	if (IsMassLumped(lump))
		Mcv.AddDomainIntegrator(new mfem::LumpedIntegrator(new mfem::MassIntegrator(heat_capacity))); 
	else
		Mcv.AddDomainIntegrator(new mfem::MassIntegrator(heat_capacity)); 
	Mcv.Assemble(); 
	Mcv.Finalize(); 

	mfem::ParBilinearForm Mtime_form_s(&fes);
	if (IsMassLumped(lump)) 
		Mtime_form_s.AddDomainIntegrator(new mfem::LumpedIntegrator(new mfem::MassIntegrator));
	else
		Mtime_form_s.AddDomainIntegrator(new mfem::MassIntegrator);
	Mtime_form_s.Assemble(); 
	Mtime_form_s.Finalize();
	auto Mtime_s = std::unique_ptr<mfem::HypreParMatrix>(Mtime_form_s.ParallelAssemble());

	mfem::ParBilinearForm Mtime_form_v(&vfes);
	if (IsMassLumped(lump)) 
		Mtime_form_v.AddDomainIntegrator(new mfem::LumpedIntegrator(new mfem::VectorMassIntegrator));
	else
		Mtime_form_v.AddDomainIntegrator(new mfem::MassIntegrator);
	Mtime_form_v.Assemble(); 
	Mtime_form_v.Finalize();
	auto Mtime_v = std::unique_ptr<mfem::HypreParMatrix>(Mtime_form_v.ParallelAssemble());

	// NOTE: RADE variable currently represents the zeroth moment of psi 
	// not 1/c \int psi dOmega
	enum SolutionIndex {
		PSI = 0, 
		TEMP = 1, 
		RADF = 2, 
		RADE = 3
	};
	mfem::Array<int> offsets(5); 
	offsets[0] = 0; 
	offsets[SolutionIndex::PSI+1] = psi_size; 
	offsets[SolutionIndex::TEMP+1] = fes.GetVSize(); 
	offsets[SolutionIndex::RADE+1] = fes.GetVSize();
	offsets[SolutionIndex::RADF+1] = vfes.GetVSize();
	offsets.PartialSum(); 
	mfem::BlockVector x(offsets), x0(offsets); 
	mfem::Vector psi(x.GetBlock(SolutionIndex::PSI), 0, psi_size); 
	mfem::Vector psi0(x0.GetBlock(SolutionIndex::PSI), 0, psi_size); 
	mfem::ParGridFunction T(&fes, x.GetBlock(SolutionIndex::TEMP), 0); 
	mfem::ParGridFunction T0(&fes, x0.GetBlock(SolutionIndex::TEMP), 0); 
	mfem::ParGridFunction E(&fes, x.GetBlock(SolutionIndex::RADE), 0); 
	mfem::ParGridFunction E0(&fes, x0.GetBlock(SolutionIndex::RADE), 0);
	mfem::ParGridFunction F(&vfes, x.GetBlock(SolutionIndex::RADF), 0);
	mfem::ParGridFunction F0(&vfes, x0.GetBlock(SolutionIndex::RADF), 0);

	// piecewise constant temperature 
	mfem::ParGridFunction Tpw(&fes0); 

	// project initial condition 
	sol::function ic_lua = lua["initial_condition"]; 
	if (!ic_lua.valid()) { MFEM_ABORT("must supply initial condition function"); }
	auto ic_func = [&ic_lua](const mfem::Vector &x) {
		double pos[3]; 
		for (int d=0; d<x.Size(); d++) { pos[d] = x(d); }
		return ic_lua(pos[0], pos[1], pos[2]); 
	};
	mfem::FunctionCoefficient ic_coef(ic_func);
	T.ProjectCoefficient(ic_coef);  
	T0 = T; 
	Tpw.ProjectGridFunction(T); 

	mfem::GridFunctionCoefficient Tcoef(&T); 
	total_coef.SetTemperature(Tcoef); 
	total_coef.SetDensity(density); 
	total_gf.ProjectCoefficient(total_coef); 
	total_gf.ExchangeFaceNbrData(); 
	GridFunctionMGCoefficient total_mg(total_gf);
	auto total_ptr = std::unique_ptr<mfem::Coefficient>(total_mg.GetGroupCoefficient(0));
	auto &total = *total_ptr;

	for (int i=0; i<fes.GetVSize(); i++) {
		E0(i) = constants::StefanBoltzmann * pow(T(i), 4); // ac T^4 
	}
	E = E0;
	F = 0.0;
	F0 = 0.0;

	// form fixed source term 
	mfem::Vector source_vec(psi_size); 
	TransportVectorView source_vec_view(source_vec.GetData(), psi_ext); 
	FormTransportSource(fes, *quad, energy_grid, source, inflow, source_vec_view); 

	// build sweep operator 
	// total_gf += 1.0/time_step; // <-- hack, needs better design 
	InverseAdvectionOperator Linv(fes, *quad, total_mg, bc_map, lump); 
	if (sweep_opts_avail) {
		sol::table sweep_opts = sweep_opts_avail.value(); 
		bool write_graph = sweep_opts["write_graph"].get_or(false); 
		if (write_graph) 
			Linv.WriteGraphToDot("graph"); 
		sol::optional<int> send_buffer_size = sweep_opts["send_buffer_size"]; 
		if (send_buffer_size) 
			Linv.SetSendBufferSize(send_buffer_size.value()); 
	}
	out << YAML::Key << "lumping type" << YAML::Value << YAML::BeginMap; 
		out << YAML::Key << "mass" << YAML::Value << IsMassLumped(lump); 
		out << YAML::Key << "gradient" << YAML::Value << IsGradientLumped(lump); 
		out << YAML::Key << "faces" << YAML::Value << IsFaceLumped(lump); 
	out << YAML::EndMap; 
	if (sweep_opts_avail) out << YAML::Key << "sweep options" << YAML::Value << sweep_opts_avail.value(); 

	bool use_fixup = false; 
	std::unique_ptr<NegativeFluxFixupOperator> nff_op = nullptr; 
	std::unique_ptr<mfem::SLBQPOptimizer> nff_optimizer = nullptr; 
	sol::optional<sol::table> fixup_avail = driver["fixup"]; 
	if (fixup_avail) {
		sol::table fixup = fixup_avail.value(); 
		use_fixup = true; 
		std::string type = fixup["type"]; 
		io::ValidateOption<std::string>("fixup type", type, 
			{"zero and scale", "local optimization"}, root); 
		double min = fixup["psi_min"].get_or(0.0); 
		if (type == "zero and scale") {
			nff_op = std::make_unique<ZeroAndScaleFixupOperator>(min); 
		} else if (type == "local optimization") {
			nff_optimizer = std::make_unique<mfem::SLBQPOptimizer>(); 
			double abstol = fixup["abstol"].get_or(1e-18); 
			double reltol = fixup["reltol"].get_or(1e-12); 
			int max_iter = fixup["max_iter"].get_or(20); 
			int print_level = fixup["print_level"].get_or(-1); 
			nff_optimizer->SetAbsTol(abstol); 
			nff_optimizer->SetRelTol(reltol); 
			nff_optimizer->SetMaxIter(max_iter); 
			nff_optimizer->SetPrintLevel(print_level); 
			nff_op = std::make_unique<LocalOptimizationFixupOperator>(*nff_optimizer, min); 
		}
		Linv.SetFixupOperator(*nff_op); 
		out << YAML::Key << "negative flux fixup" << YAML::Value << fixup; 
	}
	Linv.SetTimeAbsorption(1.0/time_step/constants::SpeedOfLight); 

	DiscreteToMoment D(*quad, psi_ext, phi_ext); 
	D.MultTranspose(E, psi0);
	// psi0 *= constants::SpeedOfLight; 
	psi = psi0; 

	mfem::ProductCoefficient Cvdt(1.0/time_step, heat_capacity); 
	BlockDiagonalByElementNonlinearForm meb_form(&fes);
	if (IsMassLumped(lump)) {
		meb_form.AddDomainIntegrator(new QuadratureLumpedNFIntegrator(new BlackBodyEmissionNFI(total))); 
		meb_form.AddDomainIntegrator(new mfem::LumpedIntegrator(new mfem::MassIntegrator(Cvdt))); 
	}
	else {
		meb_form.AddDomainIntegrator(new BlackBodyEmissionNFI(total, 2, 2 + sigma_fe_order)); 
		meb_form.AddDomainIntegrator(new mfem::MassIntegrator(Cvdt)); 
	}

	BlockDiagonalByElementNonlinearForm emission_form(&fes); 
	if (IsMassLumped(lump)) 
		emission_form.AddDomainIntegrator(new QuadratureLumpedNFIntegrator(new BlackBodyEmissionNFI(total)));
	else
		emission_form.AddDomainIntegrator(new BlackBodyEmissionNFI(total, 2, 2 + sigma_fe_order));

	mfem::BilinearForm Mtot(&fes); 
	if (lump) 
		Mtot.AddDomainIntegrator(new mfem::LumpedIntegrator(new mfem::MassIntegrator(total))); 
	else
		Mtot.AddDomainIntegrator(new mfem::MassIntegrator(total)); 
	Mtot.Assemble(); 
	Mtot.Finalize(); 

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
	mfem::ConstantCoefficient alpha_c(alpha/2);
	mfem::RatioCoefficient diffco(1.0/3, total); 

	sol::table solver = driver["solver"]; 
	out << YAML::EndMap; // end driver output 

	// --- setup low order discretization --- 
	// defined in src/moment_discretization.hpp, src/smm_source.hpp
	BlockLDGDiscretization lo_disc(fes, vfes, bc_map, lump);
	const auto &lo_offsets = lo_disc.GetOffsets(); // size of [F, E]
	// set time absorption for scalar and vector variable 
	lo_disc.SetScalarTimeAbsorption(1.0/time_step/constants::SpeedOfLight, *Mtime_s);
	lo_disc.SetVectorTimeAbsorption(1.0/time_step/constants::SpeedOfLight, *Mtime_v);
	ConsistentLDGSMMOperator source_op(lo_disc, *quad, psi_ext, source_vec);
	// reference to moment vector part of x and x0 
	// does not copy/allocate data 
	mfem::BlockVector moments(x, offsets[SolutionIndex::RADF], lo_offsets);
	mfem::BlockVector moments0(x0, offsets[SolutionIndex::RADF], lo_offsets);

	// solver for low order system 
	mfem::CGSolver cg_solver(MPI_COMM_WORLD);
	cg_solver.SetRelTol(1e-6);
	cg_solver.iterative_mode = false;
	mfem::HypreBoomerAMG amg;
	amg.SetPrintLevel(0);
	cg_solver.SetPreconditioner(amg);
	// solve 2x2 system for [F,E] taking advantage of block diagonal
	// structure of F equation due to using LDG or IP 
	BlockMomentDiscretization::Solver lo_solver(vfes, cg_solver);

	// --- solver for energy balance equation --- 
	// local_meb_solver is applied on each element independently 
	// by meb_solver 
	EnergyBalanceNewtonSolver local_meb_solver;
	local_meb_solver.SetRelTol(1e-8);
	local_meb_solver.SetMaxIter(40);
	local_meb_solver.SetPreconditioner(*local_mat_inv);
	local_meb_solver.iterative_mode = true;
	BlockDiagonalByElementNonlinearSolver meb_solver(local_meb_solver);
	meb_solver.SetOperator(meb_form); // solves meb_form = (cv/dt + B(.))T 

	// --- configure outputs --- 
	mfem::ParGridFunction cvgf(&fes0); cvgf.ProjectCoefficient(heat_capacity); 
	mfem::ParGridFunction density_gf(&fes0); density_gf.ProjectCoefficient(density); 
	mfem::ParGridFunction partition(&fes0); partition = rank; 
	sol::optional<sol::table> output_avail = lua["output"]; 
	std::unique_ptr<mfem::DataCollection> dc; 
	std::unique_ptr<TracerDataCollection> tracer_dc; 
	int output_freq; 
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
			dc->SetPrecision(precision); 
			dc->RegisterField("E", &E); 
			dc->RegisterField("F", &F);
			dc->RegisterField("T", &T); 
			dc->RegisterField("Tpw", &Tpw); 
			dc->RegisterField("sigma", &total_gf); 
			dc->RegisterField("cv", &cvgf); 
			dc->RegisterField("density", &density_gf); 
			dc->RegisterField("partition", &partition); 
			dc->SetCycle(0); dc->SetTime(0.0); dc->SetTimeStep(time_step); 
			dc->Save(); 

			out << YAML::Key << "visualization" << YAML::Value << YAML::BeginMap; 
				out << YAML::Key << "type" << YAML::Value << type; 
				out << YAML::Key << "frequency" << YAML::Value << output_freq; 
				out << YAML::Key << "precision" << YAML::Value << precision; 
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
			tracer_dc = std::make_unique<TracerDataCollection>(prefix, mesh, pts); 
			tracer_dc->SetPrefixPath(output_root); 
			tracer_dc->SetPrecision(precision); 
			tracer_dc->RegisterField("E", &E); 
			tracer_dc->RegisterField("F", &F);
			tracer_dc->RegisterField("T", &T); 
			tracer_dc->RegisterField("Tpwc", &Tpw); 
			tracer_dc->RegisterField("sigma", &total_gf); 
			tracer_dc->SetCycle(0); tracer_dc->SetTime(0.0); tracer_dc->SetTimeStep(time_step); 
			tracer_dc->Save(); 

			out << YAML::Key << "tracer" << YAML::Value << YAML::BeginMap; 
				out << YAML::Key << "prefix" << YAML::Value << prefix; 
				out << YAML::Key << "precision" << YAML::Value << precision; 
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
		out << YAML::EndMap; // end output map
	}

	// working vectors for moment algorithm 
	mfem::Vector em_source(fes.GetVSize()), abs_source(fes.GetVSize()), dT(fes.GetVSize());
	mfem::BlockVector smm_source(lo_disc.GetOffsets()), lo_source(lo_disc.GetOffsets());

	mfem::StopWatch cycle_timer; 
	double time = 0.0; 
	out << YAML::Key << "time integration" << YAML::BeginSeq; 
	std::map<std::string,int> log; 
	int cycle = 0; 
	while (true) {
		cycle_timer.Restart(); 

		// --- apply mass matrix to previous time step --- 
		Mpsi.Mult(psi0, psi0); // operator designed to work in-place 
		add(source_vec, 1.0/time_step/constants::SpeedOfLight, psi0, psi0); // source_vec + 1/c/dt psi0 -> psi0
		Mcv.Mult(T, T0); // assume T = T0 to get Mcv T0 -> T0 
		T0 *= 1.0/time_step; 

		Mtime_s->Mult(E, E0);
		E0 *= 1.0/time_step/constants::SpeedOfLight;
		Mtime_v->Mult(F, F0);
		F0 *= 3.0/time_step/constants::SpeedOfLight;

		// --- get new time step solution --- 
		int outer;
		int inner_sum = 0;
		for (outer=0; outer<20; outer++) {
			mfem::Vector Tstar(T); // store previous temperature 
			emission_form.Mult(T, em_source); // comute emission term 
			D.MultTranspose(em_source, psi); // phi -> psi 
			psi += psi0; // add in time, fixed source 
			Linv.Mult(psi, psi); // sweep, in place to avoid extra memory

			// compute SMM closure 
			source_op.Mult(psi, smm_source);
			smm_source += moments0; // time source 
			lo_source.GetBlock(0) = smm_source.GetBlock(0); // copy F source, emission only effects E source
			int inner;
			for (inner=0; inner<100; inner++) {
				mfem::Vector prev(moments); // previous moment solution 

				// nonlinearly eliminate temperature 
				// compute sigma c E term 
				Mtot.Mult(E, abs_source); 
				// add cv/dt T0 
				add(abs_source, 1.0, T0, abs_source); // abs_source + 1.0 * T0 -> abs_source 
				meb_solver.Mult(abs_source, T);

				// form schur complement of jacobian 
				auto K = std::unique_ptr<mfem::BlockOperator>(lo_disc.GetOperator(total, total)); // <-- does some redundant assembly
				const auto &dB = emission_form.GetGradient(T); // gradient of emission
				auto &dBt_inv = meb_form.GetGradient(T); // gradient of energy balance equation
				dBt_inv.Invert(); // <-- for lumping >0, this is diagonal, could save work with a diagonal inverse
				auto product = Mult(dB, dBt_inv).AsSparseMatrix(); // element-wise multiplication of dB and dBt_inv 
				// triple product: dB dBt_inv Mtot 
				auto lin_emission = std::unique_ptr<mfem::SparseMatrix>(Mult(product, Mtot.SpMat())); 
				// add linearized portion to K_{2,2} to get schur complement 
				// of Jacobian 
				// lin_emission is all on-processor => add directly to "diagonal" (aka on processor) 
				// CSR for the hypre matrix 
				mfem::SparseMatrix diag;
				static_cast<mfem::HypreParMatrix*>(&K->GetBlock(1,1))->GetDiag(diag);
				diag.Add(-1.0, *lin_emission);

				// compute Newton residual 
				// applies inverse of "U" matrix to form 
				// the source for the Schur complement operator 
				emission_form.Mult(T, em_source);
				add(em_source, 1.0, smm_source.GetBlock(1), lo_source.GetBlock(1));
				meb_form.Mult(T, em_source);
				add(T0, -1.0, em_source, em_source);
				product.AddMult(em_source, lo_source.GetBlock(1));

				// solve linearized LO system 
				lo_solver.SetOperator(*K); // <-- requires AMG setup 
				lo_solver.Mult(lo_source, moments); // uses preconditioned CG to invert Schur complement 

				// stopping criterion 
				prev -= moments;
				double norm = sqrt(mfem::InnerProduct(MPI_COMM_WORLD, prev, prev)); 
				double mag = sqrt(mfem::InnerProduct(MPI_COMM_WORLD, moments, moments));
				norm /= mag; // relative norm 
				if (norm < 1e-5) break;
			}
			inner_sum += inner;	// count total inner iterations 
			Tstar -= T; // attempt at an outer stopping criterion 
			double norm = sqrt(mfem::InnerProduct(MPI_COMM_WORLD, Tstar, Tstar)); 
			if (norm < 1e-3) break;
		}

		// --- compute consistency --- 
		// compares phi computed from psi vs phi computed
		// by low order solve 
		// difference should be on the order of the iterative tolerance
		// and independent of the mesh size
		mfem::ParGridFunction phi_sn(&fes);
		D.Mult(psi, phi_sn);
		mfem::GridFunctionCoefficient phi_snc(&phi_sn);
		double consistency_E = E.ComputeL2Error(phi_snc);

		if (E.CheckFinite() > 0) MFEM_ABORT("infinite energy density"); 

		// --- post process --- 
		// get peicewise constant version of temperature 
		// used for comparison to other codes 
		Tpw.ProjectGridFunction(T); 

		// prepare for next time step 
		time += time_step; 
		cycle++; 
		bool done = time >= final_time - 1e-14 or cycle == max_cycles; 
		if (dc and (cycle % output_freq == 0 or done)) {
			dc->SetCycle(cycle); 
			dc->SetTime(time); 
			dc->SetTimeStep(time_step); 
			dc->Save(); 			
		}

		if (tracer_dc) {
			tracer_dc->SetCycle(cycle); tracer_dc->SetTime(time); tracer_dc->SetTimeStep(time_step); 
			tracer_dc->Save(); 
		}

		// check for new time step size 
		bool time_step_changed = false; 
		if (time_step_func_avail) {
			double new_time_step = time_step_func_avail.value()(time); 
			time_step_changed = std::fabs(new_time_step - time_step) > 1e-14; 
			time_step = new_time_step; 
			Cvdt.SetAConst(1.0/time_step); 
		}

		if (T.Min() < 0) MFEM_ABORT("negative temperature"); 
		// update opacity and opacity-dependent terms 
		if (temp_dependent_opacity or time_step_changed) {
			// recompute opacities 
			total_gf.ProjectCoefficient(total_coef); 
			total_gf.ExchangeFaceNbrData(); 

			// recompute sweep data 
			Linv.AssembleLocalMatrices(); 
			Linv.SetTimeAbsorption(1.0/time_step/constants::SpeedOfLight);

			// total interaction mass matrix
			// depends on sigma and dt 
			delete Mtot.LoseMat();
			Mtot.Assemble(); 
			Mtot.Finalize(); 	
		}

		// store time step 
		x0 = x; 

		// get statistics from library code in parallel 
		EventLog.Synchronize(); 

		cycle_timer.Stop(); 
		double cycle_time = cycle_timer.RealTime(); 

		const double radE_norm = sqrt(mfem::InnerProduct(MPI_COMM_WORLD, E, E)); 
		// output progress 
		out << YAML::BeginMap; 
			out << YAML::Key << "cycle" << YAML::Value << cycle; 
			out << YAML::Key << "simulation time" << YAML::Value << time; 
			out << YAML::Key << "time step size" << YAML::Value << time_step; 
			out << YAML::Key << "||radE||" << YAML::Value << radE_norm; 
			out << YAML::Key << "consistency" << YAML::Value << consistency_E / radE_norm;
			out << YAML::Key << "outer it" << YAML::Value << outer;
			out << YAML::Key << "total inner" << YAML::Value << inner_sum;
			if (EventLog.size()) {
				out << YAML::Key << "event log" << YAML::Value << YAML::BeginMap; 
				for (const auto &it : EventLog) {
					out << YAML::Key << it.first << YAML::Value << it.second; 
				}				
				out << YAML::EndMap; 
			}
			out << YAML::Key << "cycle time" << YAML::Value << io::FormatTimeString(cycle_time); 
		out << YAML::EndMap << YAML::Newline; 
		EventLog.clear(); 

		// warn if max cycles reached 
		if (cycle == max_cycles and root) 
			MFEM_WARNING("max cycles reached. simulation end time not equal to final time"); 

		// end time integration 
		if (done) break; 
	}
	out << YAML::EndSeq; 

	if (log.size()) {
		out << YAML::Key << "log" << YAML::Value << YAML::BeginMap; 
		for (const auto &it : log) {
			out << YAML::Key << it.first << YAML::Value << it.second; 
		}
		out << YAML::EndMap; 
	}

	// --- clean up hanging pointers --- 
	for (int i=0; i<nattr; i++) { delete total_list[i]; } 
	for (int i=0; i<nattr; i++) { delete source_list[i]; }
	for (int i=0; i<nbattr; i++) { delete inflow_list[i]; }

	wall_timer.Stop(); 
	double wall_time = wall_timer.RealTime(); 
	out << YAML::Key << "wall time" << YAML::Value << io::FormatTimeString(wall_time); 

	// --- end yaml output --- 
	out << YAML::EndMap << YAML::Newline; 
}