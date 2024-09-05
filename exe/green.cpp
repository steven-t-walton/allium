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
#include "multigroup.hpp"
#include "mg_form.hpp"
#include "planck.hpp"
#include "restart.hpp"
#include "smm_source.hpp"
#include "smm_integrators.hpp"
#include "smm_coef.hpp"
#include "coefficient.hpp"

class IterationMonitor {
public:
	mfem::Array<int> iters;
	double max_norm = 0.0;
	IterationMonitor() { iters.Reserve(50); }
	void Register(int it, double norm) {
		iters.Append(it); 
		max_norm = std::max(norm, max_norm);
	}
	friend YAML::Emitter &operator<<(YAML::Emitter &out, 
		const IterationMonitor &monitor)
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
	sol::optional<sol::table> energy_table_avail = lua["energy"];
	MultiGroupEnergyGrid energy_grid;
	if (energy_table_avail) {
		sol::table energy_table = energy_table_avail.value();
		const double Emin = energy_table["min"];
		const double Emax = energy_table["max"]; 
		const int G = energy_table["num_groups"];

		out << YAML::Key << "Emin" << YAML::Value << Emin; 
		out << YAML::Key << "Emax" << YAML::Value << Emax;
		out << YAML::Key << "groups" << YAML::Value << G;

		if (G == 1) {
			energy_grid = MultiGroupEnergyGrid::MakeGray(Emin, Emax);
		} else {
			const bool extend_to_zero = energy_table["extend_to_zero"].get_or(false);
			const auto spacing = io::GetAndValidateOption<std::string>(energy_table, "spacing", 
				{"log", "equal"}, "log", root);
			if (spacing == "log") {
				energy_grid = MultiGroupEnergyGrid::MakeLogSpaced(Emin, Emax, G, extend_to_zero);
			} else if (spacing == "equal") {
				energy_grid = MultiGroupEnergyGrid::MakeEqualSpaced(Emin, Emax, G, extend_to_zero);
			}			
			out << YAML::Key << "extend to zero" << YAML::Value << extend_to_zero;
		}

		const bool print_bounds = energy_table["print_bounds"].get_or(false);
		if (print_bounds) {
			out << YAML::Key << "bounds" << YAML::Value << YAML::Flow << energy_grid.Bounds();
		}
	} else {
		const double Emin = 0.0;
		const double Emax = 1e9;
		energy_grid = MultiGroupEnergyGrid::MakeGray(Emin, Emax);
		out << YAML::Key << "Emin" << YAML::Value << Emin; 
		out << YAML::Key << "Emax" << YAML::Value << Emax;
		out << YAML::Key << "groups" << YAML::Value << 1;		
		if (root) MFEM_WARNING("not specifying energy is deprecated, defaulting to gray");
	}
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
		io::ValidateOption<std::string>("opacity type", type, {"constant", "analytic gray", "analytic"}, root); 
		if (type == "constant") {
			sol::table values = total["values"];
			mfem::Vector vec(values.size()); 
			for (int i=0; i<vec.Size(); i++) { vec(i) = values[i+1]; }
			total_list[i] = new ConstantOpacityCoefficient(vec);  
		}

		else if (type == "analytic gray") {
			double coef = total["coef"]; 
			double nrho = total["nrho"]; 
			double nT = total["nT"]; 
			total_list[i] = new AnalyticGrayOpacityCoefficient(coef, nrho, nT); 
			temp_dependent_opacity = true; 
		}

		else if (type == "analytic") {
			double coef = total["coef"]; 
			double nrho = total["nrho"]; 
			double nT = total["nT"]; 
			total_list[i] = new AnalyticOpacityCoefficient(coef, nrho, nT, energy_grid.Midpoints());
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
	mfem::Array<mfem::Coefficient*> inflow_base_list(nbattr);
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
			inflow_base_list[i] = new mfem::ConstantCoefficient(value);
			inflow_list[i] = new PlanckEmissionPSCoefficient(*inflow_base_list[i]);
			bc_map[i+1] = BoundaryCondition::INFLOW;
		} else if (type == "reflective") {
			inflow_base_list[i] = nullptr;
			inflow_list[i] = nullptr; 
			bc_map[i+1] = BoundaryCondition::REFLECTIVE;
		} 
		else if (type == "vacuum") {
			inflow_base_list[i] = nullptr;
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
	// print info about mesh 
	io::PrintMeshCharacteristics(out, mesh, ser_ref, par_ref);
	out << YAML::EndMap; 

	// --- load algorithmic parameters --- 
	sol::table driver = lua["driver"]; 
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
	mfem::ParFiniteElementSpace fes_mg(&mesh, &fec, G);
	mfem::ParFiniteElementSpace fes0_mg(&mesh, &fec0, G);

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

	// size of moment vector
	MomentVectorExtents moment_ext(G, dim+1, fes.GetVSize());
	const auto moment_size = TotalExtent(moment_ext);

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
		out << YAML::Key << "gray opacity fe order" << YAML::Value << gray_sigma_fe_order;
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

	// kinetic to continuum operators 
	DiscreteToMoment D(*quad, psi_ext, phi_ext); // compute Enu 
	DiscreteToMoment Dlin(*quad, psi_ext, moment_ext); // [Enu, Fnu]
	GroupCollapseOperator to_gray_op(phi_ext); // Enu -> gray E 
	GroupCollapseOperator to_gray_op_moments(moment_ext); // moments_nu -> gray moments 
	GroupCollapseOperator<TransportVectorLayout> to_gray_op_psi(psi_ext); // psi_nu -> gray psi 

	// --- allocate solution vectors ---
	// moment algorithm time integrates 
	//   intensity 
	//   temperature 
	//   gray LO flux 
	//   gray LO energy density 
	// NOTE: RADE is really int I dOmega not 1/c int I dOmega 
	enum SolutionIndex {
		PSI = 0, 
		TEMP = 1, 
		RADF = 2, 
		RADE = 3,
		NVARS = 4
	};
	mfem::Array<int> offsets(SolutionIndex::NVARS+1); 
	offsets[0] = 0; 
	offsets[SolutionIndex::PSI+1] = psi_size; 
	offsets[SolutionIndex::TEMP+1] = fes.GetVSize(); 
	offsets[SolutionIndex::RADE+1] = fes.GetVSize();
	offsets[SolutionIndex::RADF+1] = vfes.GetVSize();
	offsets.PartialSum(); 

	// allocate storage 
	mfem::BlockVector x(offsets), x0(offsets); 

	// get references to data in x, x0
	mfem::Vector psi(x.GetBlock(SolutionIndex::PSI), 0, psi_size); 
	mfem::Vector psi0(x0.GetBlock(SolutionIndex::PSI), 0, psi_size); 
	mfem::ParGridFunction T(&fes, x.GetBlock(SolutionIndex::TEMP), 0); 
	mfem::ParGridFunction T0(&fes, x0.GetBlock(SolutionIndex::TEMP), 0); 
	mfem::ParGridFunction F(&vfes, x.GetBlock(SolutionIndex::RADF), 0); 
	mfem::ParGridFunction F0(&vfes, x0.GetBlock(SolutionIndex::RADF), 0); 	
	mfem::ParGridFunction E(&fes, x.GetBlock(SolutionIndex::RADE), 0); 
	mfem::ParGridFunction E0(&fes, x0.GetBlock(SolutionIndex::RADE), 0); 

	// coefficients representing solution vectors 
	mfem::GridFunctionCoefficient Tcoef(&T); 
	mfem::GridFunctionCoefficient Ecoef(&E);

	// store multigroup E, F from transport 
	mfem::Vector moments_nu(moment_size);
	mfem::ParGridFunction Enu(&fes_mg, moments_nu, 0); // multigroup energy density, reference to moments_nu
	mfem::Vector Fnu(moments_nu, phi_size, phi_size*dim); // multigroup flux, reference to moments_nu
	// coefficients representing Enu, Fnu
	ZerothMomentCoefficient Enu_coef(fes, phi_ext, Enu);
	FirstMomentCoefficient Fnu_coef(fes, moment_ext, moments_nu);

	// store gray E,F from transport 
	mfem::Vector moments_ho(fes.GetVSize() * (dim+1));
	// reference into block vector 
	mfem::ParGridFunction Eho(&fes, moments_ho, 0);
	mfem::ParGridFunction Fho(&vfes, moments_ho, fes.GetVSize());
	// coefficient representing gray HO E and F  
	mfem::GridFunctionCoefficient Eho_coef(&Eho);
	mfem::VectorGridFunctionCoefficient Fho_coef(&Fho);

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
		T.ProjectCoefficient(ic_coef);  
		T0 = T; 
		Tpw.ProjectGridFunction(T); 

		// radiation initial condition
		// assume radiation and material in equilibrium 
		// => psi_g = 1/4pi B_g(T0)
		PlanckEmissionPSCoefficient planck_coef(Tcoef);
		ProjectPsi(fes, *quad, energy_grid, planck_coef, psi0);
		psi = psi0;

		D.Mult(psi, Enu);
		to_gray_op.Mult(Enu, E);
		E0 = E;
		F = 0.0; 
		F0 = 0.0;
	}

	// allocate data for total opacity 
	ProjectedVectorCoefficient total(sigma_fes, total_coef);

	// project initial opacity 
	total_coef.SetTemperature(Tcoef); 
	total_coef.SetDensity(density); 
	total.Project();

	// --- opacity weighting operators --- 
	// energy density weighted 
	OpacityGroupCollapseCoefficient sigmaE_coef(total, Enu_coef);
	// rosseland weighted 
	RosselandSpectrumMGCoefficient rosseland_coef(energy_grid.Bounds(), Tcoef);
	OpacityGroupCollapseCoefficient sigmaR_coef(total, rosseland_coef);
	InverseOpacityGroupCollapseCoefficient sigmaRinv_coef(total, rosseland_coef);
	// planck weighted 
	PlanckSpectrumMGCoefficient planck_coef(energy_grid.Bounds(), Tcoef);
	OpacityGroupCollapseCoefficient sigmaP_coef(total, planck_coef);
	// flux collapse
	RowL2NormVectorCoefficient flux_collapse_coef(Fnu_coef);
	OpacityGroupCollapseCoefficient sigmaF_coef(total, flux_collapse_coef);
	// second derivative weighted 
	PlanckSecondDerivativeSpectrumMGCoefficient planck2_coef(energy_grid.Bounds(), Tcoef);
	OpacityGroupCollapseCoefficient sigmaP2_coef(total, planck2_coef);

	// compute and store initial gray opacities 
	ProjectedCoefficient totalE(gray_sigma_fes, sigmaE_coef);
	ProjectedCoefficient totalR(gray_sigma_fes, sigmaR_coef);
	ProjectedCoefficient totalRinv(gray_sigma_fes, sigmaRinv_coef);
	ProjectedCoefficient totalP(gray_sigma_fes, sigmaP_coef);
	ProjectedCoefficient totalP2(gray_sigma_fes, sigmaP2_coef); 
	ProjectedCoefficient totalF(gray_sigma_fes, sigmaF_coef);
	totalE.Project();
	totalR.Project();
	totalRinv.Project();
	totalP.Project();
	totalP2.Project();
	totalF.Project();

	// form fixed source term 
	mfem::Vector source_vec(psi_size); 
	FormTransportSource(fes, *quad, energy_grid, source, inflow, source_vec); 

	// build sweep operator 
	InverseAdvectionOperator Linv(fes, *quad, total, bc_map, lump); 
	sol::optional<sol::table> sweep_opts_avail = driver["sweep_opts"]; 
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
	mfem::ParGridFunction fixup_monitor(&fes0_mg);
	fixup_monitor = 0.0;
	if (fixup_avail) {
		sol::table fixup = fixup_avail.value(); 
		use_fixup = true; 
		std::string type = fixup["type"]; 
		io::ValidateOption<std::string>("fixup type", type, 
			{"zero and scale", "local optimization", "ryosuke"}, root); 
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
			nff_optimizer->iterative_mode = fixup["iterative_mode"].get_or(true);
			nff_op = std::make_unique<LocalOptimizationFixupOperator>(*nff_optimizer, min); 
		} else if (type == "ryosuke") {
			nff_op = std::make_unique<RyosukeFixupOperator>(min);
		}
		Linv.SetFixupOperator(*nff_op); 
		Linv.SetFixupMonitorData(fixup_monitor);
		out << YAML::Key << "negative flux fixup" << YAML::Value << fixup; 
	}
	Linv.SetTimeAbsorption(1.0/time_step/constants::SpeedOfLight); 

	sol::table solver = lua["smm"];
	const auto lo_type = io::GetAndValidateOption<std::string>(solver, "type", {"ldg", "p1"}, root);
	const bool consistent = solver["consistent"].get_or(true);
	const bool implicit_opacity = solver["implicit_opacity"].get_or(false);
	const bool floor_radE_LO = solver["floor_E"].get_or(false);
	const bool reset_to_ho = solver["reset_to_ho"].get_or(false);
	const std::string sigmaF_type = io::GetAndValidateOption<std::string>(solver, "sigmaF_weight", 
		{"flux", "rosseland"}, "flux", root);
	sol::table linear_solver_table = solver["solver"];
	sol::table ho_solver_table = solver["ho_solver"];
	sol::table lo_solver_table = solver["lo_solver"];
	sol::table meb_solver_table = solver["energy_balance_solver"];

	out << YAML::Key << "smm" << YAML::Value << YAML::BeginMap;
		out << YAML::Key << "type" << YAML::Value << lo_type;
		out << YAML::Key << "consistent" << YAML::Value << consistent;
		out << YAML::Key << "implicit opacity" << YAML::Value << implicit_opacity;
		out << YAML::Key << "sigmaF weight" << YAML::Value << sigmaF_type;

	// energy balance nonlinear form 
	// cv/dt + sigma B(T)
	mfem::ProductCoefficient Cvdt(1.0/time_step, heat_capacity); 
	DenseBlockDiagonalNonlinearForm meb_form(&fes);
	if (implicit_opacity)
		meb_form.AddDomainIntegrator(
			new QuadratureLumpedNFIntegrator(new GrayPlanckEmissionNFI(energy_grid.Bounds(), total)));
	else
		meb_form.AddDomainIntegrator(new QuadratureLumpedNFIntegrator(new BlackBodyEmissionNFI(totalP)));
	meb_form.AddDomainIntegrator(new QuadratureLumpedNFIntegrator(new mfem::MassIntegrator(Cvdt)));

	// emission nonlinear form 
	// sigma_g B_g(T) 
	PlanckEmissionNFI planck_int(energy_grid.Bounds(), total);
	PlanckEmissionNonlinearForm emission_form(fes, phi_ext, planck_int, IsMassLumped(lump));

	// gray emission 
	DenseBlockDiagonalNonlinearForm gr_emission_form(&fes);
	if (implicit_opacity)
		gr_emission_form.AddDomainIntegrator(
			new QuadratureLumpedNFIntegrator(new GrayPlanckEmissionNFI(energy_grid.Bounds(), total)));
	else
		gr_emission_form.AddDomainIntegrator(new QuadratureLumpedNFIntegrator(new BlackBodyEmissionNFI(totalP)));

	// sigmaE absorption mass matrix 
	// used in gray energy balance equation 
	// sigmaE is computed using HO MG energy density as weight function 
	mfem::BilinearForm Mtot_gray(&fes);
	if (IsMassLumped(lump))
		Mtot_gray.AddDomainIntegrator(new QuadratureLumpedIntegrator(new mfem::MassIntegrator(totalE)));
	else
		Mtot_gray.AddDomainIntegrator(new mfem::MassIntegrator(totalE));
	Mtot_gray.UsePrecomputedSparsity();
	Mtot_gray.Assemble(); 
	Mtot_gray.Finalize();

	// time mass matrix for LO energy density 
	mfem::ParBilinearForm Mtime_form_s(&fes);
	if (IsMassLumped(lump)) 
		Mtime_form_s.AddDomainIntegrator(new QuadratureLumpedIntegrator(new mfem::MassIntegrator));
	else
		Mtime_form_s.AddDomainIntegrator(new mfem::MassIntegrator);
	Mtime_form_s.Assemble(); 
	Mtime_form_s.Finalize();
	auto Mtime_s = std::unique_ptr<mfem::HypreParMatrix>(Mtime_form_s.ParallelAssemble());

	// time mass matrix for HO energy density 
	mfem::ParBilinearForm Mtime_form_v(&vfes);
	if (IsMassLumped(lump)) 
		Mtime_form_v.AddDomainIntegrator(new QuadratureLumpedIntegrator(new mfem::VectorMassIntegrator));
	else
		Mtime_form_v.AddDomainIntegrator(new mfem::MassIntegrator);
	Mtime_form_v.Assemble(); 
	Mtime_form_v.Finalize();
	auto Mtime_v = std::unique_ptr<mfem::HypreParMatrix>(Mtime_form_v.ParallelAssemble());

	// --- useful components of moment discretizations --- 
	mfem::Vector nor(dim); 
	nor = 0.0; nor(0) = 1.0; 
	const double alpha = ComputeAlpha(*quad, nor); 

	// solver for local dense matrices 
	// assume diagonal if mass lumped
	std::unique_ptr<mfem::Solver> local_mat_inv; 
	if (IsMassLumped(lump)) 
		local_mat_inv = std::make_unique<DiagonalDenseMatrixInverse>(); 
	else
		local_mat_inv = std::make_unique<mfem::DenseMatrixInverse>(); 

	// LO discretization, source, solver 
	std::unique_ptr<BlockMomentDiscretization> lo_disc;
	std::unique_ptr<mfem::Solver> lo_schur_solver, lo_solver;
	mfem::IterativeSolver *linear_it_solver = nullptr;
	std::unique_ptr<mfem::Operator> gr_source_op;
	std::unique_ptr<mfem::HypreBoomerAMG> amg;

	ProjectedCoefficient *first_moment_opac;
	if (sigmaF_type == "flux") first_moment_opac = &totalF;
	else if (sigmaF_type == "rosseland") first_moment_opac = &totalR;
	// source term 
	TransportVectorExtents psi_ext_gr(1, quad->Size(), fes.GetVSize());
	mfem::Vector source_vec_gr(TotalExtent(psi_ext_gr));
	to_gray_op_psi.Mult(source_vec, source_vec_gr);
	if (lo_type == "ldg") {
		auto *ptr = new BlockLDGDiscretization(fes, vfes, *first_moment_opac, 
			totalE, bc_map, lump);
		if (consistent)
			gr_source_op = std::make_unique<ConsistentLDGSMMOperator>(
				*ptr, *quad, psi_ext_gr, source_vec_gr);
		else
			gr_source_op = std::make_unique<IndependentBlockSMMOperator>(fes, vfes, *quad, energy_grid, 
				psi_ext_gr, source, inflow, alpha, bc_map, lump);			
		lo_disc.reset(ptr);

		lo_schur_solver.reset(io::CreateSolver(linear_solver_table, MPI_COMM_WORLD));
		lo_solver = std::make_unique<BlockMomentDiscretization::Solver>(vfes, *lo_schur_solver);

		linear_it_solver = dynamic_cast<mfem::IterativeSolver*>(lo_schur_solver.get());
		if (linear_it_solver) {
			amg = std::make_unique<mfem::HypreBoomerAMG>();
			amg->SetPrintLevel(0);
			linear_it_solver->SetPreconditioner(*amg);
		}
	}
	else if (lo_type == "p1") {
		auto *ptr = new P1Discretization(fes, vfes, *first_moment_opac, 
			totalE, bc_map, lump);
		if (!consistent) MFEM_ABORT("only consistent supported for P1");
		gr_source_op = std::make_unique<ConsistentP1SMMOperator>(
			*ptr, *quad, psi_ext_gr, source_vec_gr);
		lo_disc.reset(ptr);

		lo_solver.reset(io::CreateSolver(linear_solver_table, MPI_COMM_WORLD));
	}
	lo_disc->SetScalarTimeAbsorption(1.0/constants::SpeedOfLight/time_step, *Mtime_s);
	lo_disc->SetVectorTimeAbsorption(1.0/constants::SpeedOfLight/time_step, *Mtime_v);
	lo_disc->SetAlpha(alpha);

	// NOTE: product operator has temporary storage the size of 
	// one group of psi 
	mfem::ProductOperator source_op(gr_source_op.get(), &to_gray_op_psi, false, false);

	// print solver info 
	out << YAML::Key << "linear solver" << YAML::Value << linear_solver_table;
	const auto ho_solver_opts = io::GetIterativeSolverOptions(ho_solver_table);
	out << YAML::Key << "ho solver" << YAML::Value << ho_solver_table;
	const auto lo_solver_opts = io::GetIterativeSolverOptions(lo_solver_table);
	out << YAML::Key << "lo solver" << YAML::Value << lo_solver_table;

	// solution vector for LO system 
	// ordered as [F,E] 
	const auto &lo_offsets = lo_disc->GetOffsets(); 
	mfem::BlockVector moments(x, offsets[SolutionIndex::RADF], lo_offsets);
	mfem::BlockVector moments0(x0, offsets[SolutionIndex::RADF], lo_offsets);

	// solver for low order system 
	auto linear_solver = std::unique_ptr<mfem::Solver>(
		io::CreateSolver(linear_solver_table, MPI_COMM_WORLD));

	// --- solver for energy balance equation --- 
	// local_meb_solver is applied on each element independently 
	// by meb_solver 
	EnergyBalanceNewtonSolver local_meb_solver;
	io::SetIterativeSolverOptions(meb_solver_table, local_meb_solver);
	local_meb_solver.SetPreconditioner(*local_mat_inv);
	DenseBlockDiagonalNonlinearSolver meb_solver(local_meb_solver);
	meb_solver.SetOperator(meb_form); // solves meb_form = (cv/dt + B(.))T 
	out << YAML::Key << "meb solver" << YAML::Value << meb_solver_table;

	mfem::LinearForm opac_corr_form(&vfes);
	OpacityCorrectionCoefficient opac_corr_coef(lo_disc->GetTotal(), total, Fnu_coef);
	if (IsMassLumped(lump))
		opac_corr_form.AddDomainIntegrator(
			new QuadratureLumpedLFIntegrator(new mfem::VectorDomainLFIntegrator(opac_corr_coef)));
	else
		opac_corr_form.AddDomainIntegrator(
			new mfem::VectorDomainLFIntegrator(opac_corr_coef));

		out << YAML::EndMap; // end smm output 
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
			dc->RegisterField("Eho", &Enu);
			dc->RegisterField("F", &F);
			dc->RegisterField("Fho", &Fho);
			dc->RegisterField("T", &T); 
			dc->RegisterField("Tpw", &Tpw); 
			dc->RegisterField("sigmaR", &totalR.GetGridFunction()); 
			dc->RegisterField("sigmaE", &totalE.GetGridFunction());
			dc->RegisterField("sigmaP", &totalP.GetGridFunction());
			dc->RegisterField("sigmaP2", &totalP2.GetGridFunction());
			dc->RegisterField("sigmaRinv", &totalRinv.GetGridFunction());
			dc->RegisterField("total", &total.GetGridFunction());
			dc->RegisterField("cv", &cvgf); 
			dc->RegisterField("density", &density_gf); 
			dc->RegisterField("partition", &partition); 
			dc->RegisterField("fixup", &fixup_monitor);
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

			out << YAML::Key << "visualization" << YAML::Value << YAML::BeginMap; 
				out << YAML::Key << "type" << YAML::Value << type; 
				out << YAML::Key << "frequency" << YAML::Value << output_freq; 
				out << YAML::Key << "precision" << YAML::Value << precision; 
				out << YAML::Key << "restart mode" << YAML::Value << restart_mode;
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
			tracer_dc->RegisterField("sigmaP", &totalP.GetGridFunction());
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

	// working vectors for moment algorithm 
	mfem::Vector em_source(fes.GetVSize()*G), em_source_gr(fes.GetVSize()), abs_source(fes.GetVSize());
	mfem::BlockVector smm_source(lo_disc->GetOffsets()), lo_source(lo_disc->GetOffsets());
	DenseBlockDiagonalOperator dB_dBt_inv(fes);

	mfem::StopWatch cycle_timer; // times cost per time step 
	// log events across time steps 
	// such as number of outer iterations 
	// assumed to be data that does not require parallel 
	// reduction 
	LogMap<int,MAX> log;
	LogMap<double,MAX> value_log;
	LogMap<double,SUM,MAX> timing_log(MPI_COMM_WORLD);
	out << YAML::Key << "time integration" << YAML::BeginSeq; 
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
		int outer = 1;
		double outer_norm;
		IterationMonitor lo_monitor, meb_monitor;
		std::unique_ptr<IterationMonitor> linear_monitor;
		if (linear_it_solver) linear_monitor = std::make_unique<IterationMonitor>();
		while (true) {
			mfem::ParGridFunction Estar(E); // store previous energy density for stopping criterion

			mfem::tic();
			emission_form.Mult(T, em_source); // comute emission term 
			D.MultTranspose(em_source, psi); // phi -> psi 
			psi += psi0; // add in time, fixed source 
			Linv.Mult(psi, psi); // sweep, in place to avoid extra memory
			timing_log.Log("sweep", mfem::toc());

			// compute gray moments of psi 
			Dlin.Mult(psi, moments_nu);
			to_gray_op_moments.Mult(moments_nu, moments_ho);

			// compute E_HO-weighted opacity 
			mfem::tic(); 
			totalE.Project();
			// re-assemble mass matrix 
			Mtot_gray = 0.0; // set all entries to zero 
			// assemble into existing sparsity pattern 
			Mtot_gray.Assemble(); 
			timing_log.Log("sigmaE", mfem::toc());

			totalR.Project();
			totalRinv.Project();
			totalP.Project();
			totalF.Project();

			// compute SMM closure 
			mfem::tic();
			source_op.Mult(psi, smm_source);
			smm_source += moments0; // time source 
			timing_log.Log("SMM source", mfem::toc());
			int inner = 1;
			double inner_norm;
			while (true) {
				mfem::ParGridFunction prev(E); // previous temperature for stopping criterion 
				mfem::ParGridFunction Tprev(T);

				mfem::tic();
				// nonlinearly eliminate temperature 
				// compute sigma c E term 
				Mtot_gray.Mult(E, abs_source); 
				// add cv/dt T0 
				add(abs_source, 1.0, T0, abs_source); // abs_source + 1.0 * T0 -> abs_source 
				// nonlinearly solve for T given E 
				meb_solver.Mult(abs_source, T);
				timing_log.Log("meb solve", mfem::toc());
				meb_monitor.Register(meb_solver.GetNumIterations(), meb_solver.GetFinalRelNorm());

				if (implicit_opacity) {
					total.Project();
					totalR.Project();
					totalF.Project();
					totalRinv.Project();
					totalP.Project();
					totalE.Project();
					Mtot_gray = 0.0; // set all entries to zero 
					Mtot_gray.Assemble(); 					
				}

				// compute rosseland opacity-dependent 
				// term in opacity correction 
				mfem::tic();
				opac_corr_form.Assemble();
				timing_log.Log("opac correction", mfem::toc());

				// form schur complement of jacobian 
				mfem::tic();
				auto K = std::unique_ptr<mfem::BlockOperator>(lo_disc->GetOperator());
				const auto &dB = gr_emission_form.GetGradient(T); // gradient of emission
				auto &dBt_inv = meb_form.GetGradient(T); // gradient of energy balance equation
				dBt_inv.Invert(); // <-- for lumping >0, this is diagonal, could save work with a diagonal inverse
				Mult(dB, dBt_inv, dB_dBt_inv);
				// triple product: dB dBt_inv Mtot 
				auto lin_emission = std::unique_ptr<mfem::SparseMatrix>(Mult(dB_dBt_inv, Mtot_gray.SpMat())); 
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
				gr_emission_form.Mult(T, em_source_gr);
				add(em_source_gr, 1.0, smm_source.GetBlock(1), lo_source.GetBlock(1));
				meb_form.Mult(T, em_source_gr);
				add(T0, -1.0, em_source_gr, em_source_gr);
				dB_dBt_inv.AddMult(em_source_gr, lo_source.GetBlock(1));
				// add in opacity correction and SMM source
				add(smm_source.GetBlock(0), 3.0, opac_corr_form, lo_source.GetBlock(0));
				timing_log.Log("LO assembly", mfem::toc());

				// solve linearized LO system 
				mfem::tic();
				lo_solver->SetOperator(*K); // <-- requires AMG setup 
				lo_solver->Mult(lo_source, moments); // uses preconditioned CG to invert Schur complement 
				if (linear_it_solver and !linear_it_solver->GetConverged() and root)
					EventLog.Register("linear solver not converged");
				if (linear_it_solver) {
					linear_monitor->Register(linear_it_solver->GetNumIterations(), linear_it_solver->GetFinalRelNorm());
				}
				if (floor_radE_LO) {
					for (int i=0; i<E.Size(); i++) {
						if (E(i) < 1e-10) {
							E(i) = 1e-10; 
							EventLog.Register("floor E");
						}
					}					
				}
				timing_log.Log("LO solve", mfem::toc());

				// stopping criterion 
				// const auto norm = prev.ComputeL2Error(Tcoef);
				// if (norm < 1e-8 or inner >= 10) break;
				const auto norm = prev.ComputeL2Error(Ecoef) / sqrt(mfem::InnerProduct(MPI_COMM_WORLD, prev, prev));
				const auto Tnorm = Tprev.ComputeL2Error(Tcoef) / sqrt(mfem::InnerProduct(MPI_COMM_WORLD, Tprev, Tprev));
				inner_norm = norm + Tnorm;
				if (inner_norm < lo_solver_opts.reltol) break; 
				if (inner >= lo_solver_opts.max_iter) {
					if (root) EventLog.Register("inner solve not converged");
					break;
				}
				inner++;
			}
			lo_monitor.Register(inner, inner_norm);
			if (implicit_opacity)
				Linv.AssembleLocalMatrices();
			outer_norm = Estar.ComputeL2Error(Ecoef) / sqrt(mfem::InnerProduct(MPI_COMM_WORLD, Estar, Estar));
			if (outer_norm < ho_solver_opts.reltol or outer >= ho_solver_opts.max_iter) break;
			outer++;
		}

		if (E.CheckFinite() > 0) MFEM_ABORT("infinite energy density"); 
		// compute consistency term 
		// E,F should match gray moments of transport solution 
		// to iterative solver tolerances 
		const auto consistency_E = E.ComputeL2Error(Eho_coef) / sqrt(mfem::InnerProduct(MPI_COMM_WORLD, Eho, Eho));
		const auto consistency_F = F.ComputeL2Error(Fho_coef) / sqrt(mfem::InnerProduct(MPI_COMM_WORLD, Fho, Fho));
		value_log.Log("max consistency E", consistency_E);
		value_log.Log("max consistency F", consistency_F);

		// get peicewise constant version of temperature 
		// used for comparison to other codes 
		Tpw.ProjectGridFunction(T); 

		// --- update time step info --- 
		time += time_step; 
		cycle++; 

		// --- output to file --- 
		// write data collection to file 
		mfem::tic();
		bool done = time >= final_time - 1e-14 or cycle >= max_cycles; 
		if (dc and (cycle % output_freq == 0 or done)) {
			output_cycle++;
			dc->SetCycle(output_cycle); dc->SetTime(time); dc->SetTimeStep(time_step); 
			E *= 1.0/constants::SpeedOfLight;
			Enu *= 1.0/constants::SpeedOfLight;
			dc->Save();
			E *= constants::SpeedOfLight;
			Enu *= constants::SpeedOfLight;
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
		timing_log.Log("write", mfem::toc());

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
		if (temp_dependent_opacity or time_step_changed) {
			mfem::StopWatch assembly_timer;
			assembly_timer.Start();
			// recompute opacities 
			total.Project();
			totalP.Project();

			// recompute sweep data 
			Linv.AssembleLocalMatrices(); 
			Linv.SetTimeAbsorption(1.0/time_step/constants::SpeedOfLight);

			// energy balance time step 
			Cvdt.SetAConst(1.0/time_step); 

			// LO system time step 
			lo_disc->SetScalarTimeAbsorption(1.0/constants::SpeedOfLight/time_step, *Mtime_s);
			lo_disc->SetVectorTimeAbsorption(1.0/constants::SpeedOfLight/time_step, *Mtime_v);

			assembly_timer.Stop(); 
			timing_log.Log("assembly time", assembly_timer.RealTime());
		}

		fixup_monitor = 0.0;

		if (reset_to_ho) {
			F = Fho;
			E = Eho;			
		}

		// store time step 
		x0 = x; 

		cycle_timer.Stop(); 
		double cycle_time = cycle_timer.RealTime(); 

		// --- output progress to terminal --- 
		log.Log("max outer", outer);
		log.Log("max inner sum", lo_monitor.iters.Max());
		if (linear_monitor)
			log.Log("max linear iterations", linear_monitor->iters.Max());
		const double radE_norm = sqrt(mfem::InnerProduct(MPI_COMM_WORLD, E, E)) / constants::SpeedOfLight; 
		out << YAML::BeginMap; 
			out << YAML::Key << "cycle" << YAML::Value << cycle; 
			out << YAML::Key << "simulation time" << YAML::Value << time; 
			out << YAML::Key << "time step size" << YAML::Value << time_step_old; 
			out << YAML::Key << "||radE||" << YAML::Value << radE_norm; 
			out << YAML::Key << "outer" << YAML::Value << outer;
			out << YAML::Key << "norm" << YAML::Value << outer_norm;
			out << YAML::Key << "lo solve" << YAML::Value << lo_monitor;
			if (linear_monitor)
				out << YAML::Key << "linear solver" << YAML::Value << *linear_monitor;
			out << YAML::Key << "energy balance solver" << YAML::Value << meb_monitor;
			out << YAML::Key << "consistency" << YAML::Value << YAML::BeginMap;
				out << YAML::Key << "energy density" << YAML::Value << io::FormatScientific(consistency_E);
				out << YAML::Key << "flux" << YAML::Value << io::FormatScientific(consistency_F);
			out << YAML::EndMap;
			io::ProcessGlobalLogs(out);
			out << YAML::Key << "cycle time" << YAML::Value << io::FormatTimeString(cycle_time); 
		out << YAML::EndMap << YAML::Newline; 

		// warn if temperature is such that the 
		// energy group structure can't fully integrate
		// the planck spectrum 
		CheckPlanckSpectrumCovered(MPI_COMM_WORLD, energy_grid.MinEnergy(), 
			energy_grid.MaxEnergy(), T, 1e-10);

		// warn if max cycles reached 
		if (cycle == max_cycles and root) 
			MFEM_WARNING("max cycles reached. simulation end time not equal to final time"); 

		// end time integration 
		if (done) break; 
	}
	out << YAML::EndSeq; // time integration sequence 

	// print the "log" variable to YAML map 
	if (log.size()) {
		out << YAML::Key << "log" << YAML::Value << log;
	}

	if (value_log.size()) {
		out << YAML::Key << "value log" << YAML::Value << value_log;
	}

	// print the timing log to YAML map 
	timing_log.Synchronize(); // <-- get times in parallel 
	if (timing_log.size()) {
		out << YAML::Key << "timing log" << YAML::Value;
		io::PrintTimingMap(out, timing_log);
	}

	// --- clean up hanging pointers --- 
	for (int i=0; i<nattr; i++) { delete total_list[i]; } 
	for (int i=0; i<nattr; i++) { delete source_list[i]; }
	for (int i=0; i<nbattr; i++) { delete inflow_list[i]; }
	for (int i=0; i<nbattr; i++) { delete inflow_base_list[i]; }

	wall_timer.Stop(); 
	double wall_time = wall_timer.RealTime(); 
	out << YAML::Key << "wall time" << YAML::Value << io::FormatTimeString(wall_time); 

	// --- end yaml output --- 
	out << YAML::EndMap << YAML::Newline; 
}