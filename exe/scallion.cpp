#include "mfem.hpp"
#include "yaml-cpp/yaml.h"
#include "sol/sol.hpp"

#include "comment_stream.hpp"
#include "io.hpp"
#include "sweep.hpp"
#include "transport_op.hpp"
#include "constants.hpp"
#include "opacity.hpp"
#include "trt_op.hpp"
#include "trt_integrators.hpp"
#include "mip.hpp"

using LuaPhaseFunction = std::function<double(double,double,double,double,double,double)>; 

class MockSolver : public mfem::Solver {
public:
	void SetOperator(const mfem::Operator &op) { }
	void Mult(const mfem::Vector &x, mfem::Vector &y) const { y = x; }
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
		mfem::out << "\033[1;31m                       ___    ___                           \033[0m\n";
		mfem::out << "\033[1;31m                      /\\_ \\  /\\_ \\    __                    \033[0m\n";
		mfem::out << "\033[1;31m  ____    ___     __  \\//\\ \\ \\//\\ \\  /\\_\\    ___     ___    \033[0m\n";
		mfem::out << "\033[1;31m /',__\\  /'___\\ /'__`\\  \\ \\ \\  \\ \\ \\ \\/\\ \\  / __`\\ /' _ `\\  \033[0m\n";
		mfem::out << "\033[1;31m/\\__, `\\/\\ \\__//\\ \\L\\.\\_ \\_\\ \\_ \\_\\ \\_\\ \\ \\/\\ \\L\\ \\ /\\ \\/\\ \033[0m\n";
		mfem::out << "\033[1;31m\\/\\____/\\ \\____\\ \\__/\\._\\/\\____\\/\\____\\\\ \\_\\ \\____/\\ \\_\\ \\_\\\033[0m\n";
		mfem::out << "\033[1;31m \\/___/  \\/____/\\/__/\\/_/\\/____/\\/____/ \\/_/\\/___/  \\/_/\\/_/\033[0m\n";
		mfem::out << "\n                         a thermal radiative transfer solver\n"; 
		mfem::out << std::endl; 
	}

	// parse cmdline arguments 
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
			LuaPhaseFunction lua_source = lua_source_objs[i].as<sol::function>(); 
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
	int reflection_bdr_attr = -1; 
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
		} else if (type == "reflective") {
			if (reflection_bdr_attr<0) {
				reflection_bdr_attr = i+1; 
			}
			inflow_list[i] = nullptr; 
		} 
		else if (type == "vacuum") {
			inflow_list[i] = nullptr; 
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
	sol::optional<std::string> fname = mesh_node["file"]; 
	// allow increasing lua refinement inputs with cmdline inputs -pr and -sr 
	ser_ref += mesh_node["serial_refinements"].get_or(0); 
	par_ref += mesh_node["parallel_refinements"].get_or(0); 
	const int tot_ref = ser_ref + par_ref; 

	mfem::Mesh smesh; 
	out << YAML::Key << "mesh" << YAML::Value << YAML::BeginMap; 
	// load from a mesh file 
	if (fname) {
		smesh = mfem::Mesh::LoadFromFile(fname.value(), 1, 1);
		out << YAML::Key << "file name" << YAML::Value << io::ResolveRelativePath(fname.value()); 
	} 

	// create a cartesian mesh from extents and elements/axis 
	else {
		sol::table ne = mesh_node["num_elements"]; 
		sol::table extents = mesh_node["extents"]; 
		assert(ne.size() == extents.size()); 
		int num_dim = ne.size(); 
		io::ValidateOption("mesh::num_dim", num_dim, {1,2,3}, root); 
		std::string eltype_str; 
		mfem::Element::Type eltype; 
		bool sfc_ordering = mesh_node["sfc_ordering"].get_or(false); 
		if (num_dim==1) {
			eltype_str = io::GetAndValidateOption(mesh_node, "element_type", {"segment"}, "segment", root); 
			smesh = mfem::Mesh::MakeCartesian1D(ne[1], extents[1]);
		} else if (num_dim==2) {
			eltype_str = io::GetAndValidateOption(mesh_node, "element_type", 
				{"quadrilateral", "triangle"}, "quadrilateral", root); 
			if (eltype_str == "quadrilateral") {
				eltype = mfem::Element::QUADRILATERAL; 
			} 
			else if (eltype_str == "triangle") {
				eltype = mfem::Element::TRIANGLE; 
			} 
			smesh = mfem::Mesh::MakeCartesian2D(ne[1], ne[2], eltype, true, extents[1], extents[2], sfc_ordering); 
		} else if (num_dim==3) {
			eltype_str = io::GetAndValidateOption(mesh_node, "element_type", 
				{"hexahedron", "tetrahedron"}, "hexahedron", root); 
			if (eltype_str == "hexahedron") {
				eltype = mfem::Element::HEXAHEDRON; 
			} 
			else if (eltype_str == "tetrahedron") {
				eltype = mfem::Element::TETRAHEDRON; 
			}
			smesh = mfem::Mesh::MakeCartesian3D(ne[1], ne[2], ne[3], eltype, extents[1], extents[2], extents[3], sfc_ordering); 
		}

		out << YAML::Key << "extents" << YAML::Value << YAML::Flow << YAML::BeginSeq; 
		for (auto i=1; i<=extents.size(); i++) {
			out << (double)extents[i]; 
		} 
		out << YAML::EndSeq; 

		out << YAML::Key << "elements/axis" << YAML::Value << YAML::Flow << YAML::BeginSeq; 
		for (auto i=1; i<=ne.size(); i++) {
			out << (int)ne[i] * pow(2,tot_ref); 
		} 
		out << YAML::EndSeq; 		
		out << YAML::Key << "element type" << YAML::Value << eltype_str; 
		out << YAML::Key << "space filling ordering" << YAML::Value << sfc_ordering; 
	}
	// apply serial refinements 
	for (int sr=0; sr<ser_ref; sr++) {
		smesh.UniformRefinement(); 
	}
	// need at minimum WorldSize() elements in serial mesh 
	if (smesh.GetNE() < mfem::Mpi::WorldSize() and root) {
		MFEM_ABORT("serial mesh with " << smesh.GetNE() << " elements too small to decompose on " 
			<< mfem::Mpi::WorldSize() << " processors"); 
	}
	const auto dim = smesh.Dimension(); 

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
	AngularQuadrature *quad; 
	if (sn_type == "level symmetric") {
		quad = new LevelSymmetricQuadrature(sn_order, dim); 
	} 
	else if (sn_type == "abu shumays") {
		quad = new AbuShumaysQuadrature(sn_order, dim); 
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
	const double time_step = driver["time_step"]; 
	const int max_cycles = driver["max_cycles"].get_or(std::numeric_limits<int>::max()); 

	// implicit solver 
	sol::table solver = driver["solver"]; 
	sol::table newton_solver = driver["newton_solver"]; 
	auto *outer_solver = io::CreateIterativeSolver(solver, MPI_COMM_WORLD);
	const int max_iter = newton_solver["max_iter"]; 
	const double abs_tol = newton_solver["abstol"]; 
	const bool lump = driver["lump"].get_or(false); 
	sol::optional<sol::table> sweep_opts_avail = driver["sweep_opts"]; 

	// --- output algorithmic options used --- 
	out << YAML::Key << "driver" << YAML::Value << YAML::BeginMap; 
		out << YAML::Key << "fe order" << YAML::Value << fe_order; 
		out << YAML::Key << "data fe order" << YAML::Value << sigma_fe_order; 
		out << YAML::Key << "sn order" << YAML::Value << sn_order; 
		out << YAML::Key << "sn quadrature type" << YAML::Value << sn_type; 
		out << YAML::Key << "num angles" << YAML::Value << Nomega; 			
		out << YAML::Key << "basis type" << YAML::Value << basis_type_str; 
		out << YAML::Key << "lump" << YAML::Value << lump; 
		out << YAML::Key << "psi size" << YAML::Value << psi_size_global;
		out << YAML::Key << "final time" << YAML::Value << final_time; 
		out << YAML::Key << "time step size" << YAML::Key << time_step; 
		out << YAML::Key << "solver" << YAML::Value << solver; 
		if (sweep_opts_avail) out << YAML::Key << "sweep options" << YAML::Value << sweep_opts_avail.value(); 
	out << YAML::EndMap; 

	SNTimeMassMatrix Mpsi(fes, psi_ext); 
	mfem::BilinearForm Mcv(&fes); 
	if (lump)
		Mcv.AddDomainIntegrator(new mfem::LumpedIntegrator(new mfem::MassIntegrator(heat_capacity))); 
	else
		Mcv.AddDomainIntegrator(new mfem::MassIntegrator(heat_capacity)); 
	Mcv.Assemble(); 
	Mcv.Finalize(); 

	enum SolutionIndex {
		PHI = 0, 
		PSI = 1, 
		TEMP = 2
	};
	mfem::Array<int> offsets(4); 
	offsets[0] = 0; 
	offsets[1] = phi_size; 
	offsets[2] = psi_size; 
	offsets[3] = fes.GetVSize(); 
	offsets.PartialSum(); 
	mfem::BlockVector x(offsets), x0(offsets); 
	mfem::Vector psi(x.GetBlock(SolutionIndex::PSI), 0, psi_size); 
	mfem::Vector psi0(x0.GetBlock(SolutionIndex::PSI), 0, psi_size); 
	mfem::ParGridFunction phi(&fes, x.GetBlock(SolutionIndex::PHI), 0); 
	mfem::ParGridFunction phi0(&fes, x0.GetBlock(SolutionIndex::PHI), 0); 
	mfem::ParGridFunction T(&fes, x.GetBlock(SolutionIndex::TEMP), 0); 
	mfem::ParGridFunction T0(&fes, x0.GetBlock(SolutionIndex::TEMP), 0); 

	mfem::ParGridFunction Tpw(&fes0); 

	// working vectors 
	mfem::Vector temp_resid(fes.GetVSize()), em_source(fes.GetVSize()), phi_source(fes.GetVSize()), dT(fes.GetVSize()); 

	// project initial condition 
	sol::function ic_lua = lua["initial_condition"]; 
	auto ic_func = [&ic_lua](const mfem::Vector &x) {
		return ic_lua(x(0), x(1), x(2)); 
	};
	mfem::FunctionCoefficient ic_coef(ic_func);
	T.ProjectCoefficient(ic_coef);  
	T0 = T; 
	Tpw.ProjectGridFunction(T); 

	mfem::GridFunctionCoefficient Tcoef(&T); 
	total_coef.SetTemperature(Tcoef); 
	total_coef.SetDensity(density); 
	total_gf.ProjectCoefficient(total_coef); 
	mfem::GridFunctionCoefficient total(&total_gf); 

	for (int i=0; i<fes.GetVSize(); i++) {
		phi(i) = constants::StefanBoltzmann * pow(T(i), 4);
	}

	// form fixed source term 
	mfem::Vector source_vec(psi_size); 
	TransportVectorView source_vec_view(source_vec.GetData(), psi_ext); 
	FormTransportSource(fes, *quad, energy_grid, source, inflow, source_vec_view); 

	// build sweep operator 
	// total_gf += 1.0/time_step; // <-- hack, needs better design 
	mfem::SumCoefficient total_dt_coef(1.0/time_step/constants::SpeedOfLight, total); 
	mfem::ParGridFunction total_gf_dt(&sigma_fes); 
	total_gf_dt.ProjectCoefficient(total_dt_coef); 
	InverseAdvectionOperator Linv(fes, *quad, total_gf_dt, reflection_bdr_attr, lump); 
	bool use_fixup = false; 
	if (sweep_opts_avail) {
		sol::table sweep_opts = sweep_opts_avail.value(); 
		bool write_graph = sweep_opts["write_graph"].get_or(false); 
		if (write_graph) 
			Linv.WriteGraphToDot("graph"); 
		sol::optional<int> send_buffer_size = sweep_opts["send_buffer_size"]; 
		if (send_buffer_size) 
			Linv.SetSendBufferSize(send_buffer_size.value()); 
		use_fixup = sweep_opts["use_fixup"].get_or(false); 
		sol::optional<double> psi_min = sweep_opts["psi_min"]; 
		if (psi_min) Linv.SetMinimumSolution(psi_min.value()); 
	}
	Linv.UseFixup(use_fixup); 

	DiscreteToMoment D(*quad, psi_ext, phi_ext); 
	D.MultTranspose(phi, psi0);
	psi = psi0; 

	mfem::ProductCoefficient Cvdt(1.0/time_step, heat_capacity); 
	mfem::NonlinearForm meb_form(&fes);
	if (lump) {
		meb_form.AddDomainIntegrator(new EnergyBalanceNonlinearFormIntegrator(total, basis_type)); 
		meb_form.AddDomainIntegrator(
			new LinearCoefficientNFIntegrator(new mfem::LumpedIntegrator(new mfem::MassIntegrator(Cvdt)))); 
	}
	else {
		meb_form.AddDomainIntegrator(new EnergyBalanceNonlinearFormIntegrator(total, sigma_fe_order, 2, 2)); 
		meb_form.AddDomainIntegrator(new LinearCoefficientNFIntegrator(new mfem::MassIntegrator(Cvdt))); 
	}

	mfem::NonlinearForm emission_form(&fes); 
	if (lump) 
		emission_form.AddDomainIntegrator(new EnergyBalanceNonlinearFormIntegrator(total, basis_type)); 
	else
		emission_form.AddDomainIntegrator(new EnergyBalanceNonlinearFormIntegrator(total, sigma_fe_order, 2, 2)); 

	mfem::ParGridFunction cvgf(&fes0); cvgf.ProjectCoefficient(heat_capacity); 
	mfem::ParGridFunction density_gf(&fes0); density_gf.ProjectCoefficient(density); 
	sol::table output = lua["output"]; 
	const int output_freq = output["frequency"].get_or(std::numeric_limits<int>::max()); 
	mfem::ParaViewDataCollection dc(output["paraview"], &mesh); 
	dc.RegisterField("phi", &phi); 
	dc.RegisterField("T", &T); 
	dc.RegisterField("Tpw", &Tpw); 
	dc.RegisterField("sigma", &total_gf); 
	dc.RegisterField("cv", &cvgf); 
	dc.RegisterField("density", &density_gf); 
	dc.RegisterField("fixup diff", &phi0); 
	dc.SetCycle(0); dc.SetTime(0.0); dc.SetTimeStep(time_step); 
	dc.Save(); 

	mfem::BilinearForm Mtot(&fes); 
	if (lump) 
		Mtot.AddDomainIntegrator(new mfem::LumpedIntegrator(new mfem::MassIntegrator(total))); 
	else
		Mtot.AddDomainIntegrator(new mfem::MassIntegrator(total)); 
	Mtot.Assemble(); 
	Mtot.Finalize(); 

	mfem::Vector nor(dim); 
	nor = 0.0; nor(0) = 1.0; 
	const double alpha = ComputeAlpha(*quad, nor); 

	mfem::StopWatch cycle_timer; 
	double time = 0.0; 
	const double under_relax = 0.05; 
	out << YAML::Key << "time integration" << YAML::BeginSeq; 
	std::map<std::string,int> log; 
	int cycle = 0; 
	while (true) {
		cycle_timer.Restart(); 
		Mpsi.Mult(psi0, psi0); // operator designed to work in-place 
		add(source_vec, 1.0/time_step/constants::SpeedOfLight, psi0, psi0); // source_vec + 1/c/dt psi0 -> psi0
		Mcv.Mult(T, T0); // assume T = T0 to get Mcv T0 -> T0 
		T0 *= 1.0/time_step; 

		int outer = 0; 
		double norm; 
		double max_inner_norm = 0; 
		mfem::Array<int> inners; 
		inners.Reserve(max_iter); 
		// for (outer=1; outer<=max_iter; outer++) {
		// 	double temp_norm;
		// 	int it; 
		// 	for (it=1; it<=20; it++) {
		// 		// form residual 
		// 		meb_form.Mult(T, temp_resid); 
		// 		Mtot.AddMult(phi, temp_resid, -1.0); 
		// 		Mcv.AddMult(T0, temp_resid, -1.0/time_step); 

		// 		// form and solve Jacobian 
		// 		NonlinearFormBlockInverse inv(meb_form, T);
		// 		inv.Mult(temp_resid, dT); 

		// 		// update temperature 
		// 		for (int i=0; i<T.Size(); i++) {
		// 			double Tnew = T(i) - dT(i); 
		// 			if (Tnew < 0) {
		// 				EventLog["under relax"] += 1; 
		// 				T(i) = (1 - under_relax) * T(i) - under_relax * dT(i); 
		// 			} else {
		// 				T(i) = Tnew; 
		// 			}
		// 			if (T(i) < 0) MFEM_ABORT("negativity"); 
		// 		}
		// 		// T -= dT; 
		// 		double mag = sqrt(mfem::InnerProduct(MPI_COMM_WORLD, T, T)); 
		// 		temp_norm = sqrt(mfem::InnerProduct(MPI_COMM_WORLD, dT, dT)) / mag; 
		// 		if (temp_norm < abs_tol/100) break; 
		// 	}
		// 	max_temp_norm = std::max(max_temp_norm, temp_norm); 
		// 	inners.Append(it); 

		// 	emission_form.Mult(T, em_source); 
		// 	D.MultTranspose(em_source, psi); 
		// 	psi += psi0; 
		// 	Linv.Mult(psi, psi); 
		// 	D.Mult(psi, phi); 

		// 	phi0 -= phi; 
		// 	norm = sqrt(mfem::InnerProduct(MPI_COMM_WORLD, phi0, phi0)); 
		// 	norm /= sqrt(mfem::InnerProduct(MPI_COMM_WORLD, phi, phi)); 
		// 	if (norm < abs_tol) break; 
		// 	phi0 = phi; 
		// }

		// mfem::ParBilinearForm Kform(&fes); 
		// mfem::RatioCoefficient diffco(1.0/3, total); 
		// mfem::ConstantCoefficient alpha_c(alpha/2);
		// double kappa = pow(fe_order+1,2)*4; 
		// Kform.AddDomainIntegrator(new mfem::DiffusionIntegrator(diffco)); 
		// Kform.AddDomainIntegrator(new mfem::LumpedIntegrator(new mfem::MassIntegrator(total))); 
		// Kform.AddInteriorFaceIntegrator(new MIPDiffusionIntegrator(diffco, -1, kappa, alpha/2, lump)); 
		// Kform.AddBdrFaceIntegrator(new mfem::BoundaryMassIntegrator(alpha_c)); 
		// Kform.Assemble(); 
		// Kform.Finalize(); 
		// auto dsa_mat = std::unique_ptr<mfem::HypreParMatrix>(Kform.ParallelAssemble());

		// mfem::CGSolver cgsolver(MPI_COMM_WORLD); 
		// cgsolver.SetRelTol(1e-10); 
		// mfem::HypreBoomerAMG amg(*dsa_mat);
		// amg.SetPrintLevel(0); 
		// cgsolver.SetOperator(*dsa_mat); 
		// cgsolver.SetPreconditioner(amg); 
		// cgsolver.SetMaxIter(100); 

		Linv.UseFixup(false); 
		while (true) {
			// --- form RHS --- 
			meb_form.Mult(T, temp_resid); 
			add(T0, -1.0, temp_resid, temp_resid); // T0 - temp_resid -> temp_resid 

			// invert (cv/dt + 4a sigma T^3)
			NonlinearFormBlockInverse dplanck_dt_inv(meb_form, T); 
			// ( 4a sigma T^3 u, v)
			const auto &dplanck = emission_form.GetGradient(T); 
			mfem::ProductOperator linearized_elim(&dplanck, &dplanck_dt_inv, false, false); 

			// form transport residual 
			emission_form.Mult(T, em_source); 
			// eliminate down to phi
			linearized_elim.Mult(temp_resid, phi_source);
			em_source += phi_source; 
			D.MultTranspose(em_source, psi); 
			psi += psi0; 
			Linv.Mult(psi, psi);
			D.Mult(psi, phi_source); 

			// apply I - D Linv dB (dBt)^{-1} Msigma 
			mfem::TripleProductOperator Ms_form(&dplanck, &dplanck_dt_inv, &Mtot, false, false, false); 
			TransportOperator transport_op(D, Linv, Ms_form, psi); 
			// MockSolver mock_solver; 
			// outer_solver->SetPreconditioner(mock_solver); 
			outer_solver->SetOperator(transport_op); 
			// outer_solver->SetPreconditioner(cgsolver); 
			outer_solver->Mult(phi_source, phi); 
			inners.Append(outer_solver->GetNumIterations()); 
			max_inner_norm = std::max(max_inner_norm, outer_solver->GetFinalNorm()); 
			if (!outer_solver->GetConverged()) log["schur solve failed"] += 1; 

			// solve for temperature update 
			Mtot.Mult(phi, phi_source); 
			phi_source += temp_resid; 
			dplanck_dt_inv.Mult(phi_source, dT); 

			norm = sqrt(mfem::InnerProduct(MPI_COMM_WORLD, dT, dT)); 
			bool done = norm < abs_tol or outer == max_iter; 
			if (done) {
				Linv.UseFixup(use_fixup); 
				// sweep to recover psi 
				Ms_form.Mult(phi, phi_source); 
				em_source += phi_source;
				D.MultTranspose(em_source, psi); 
				psi += psi0; 
				Linv.Mult(psi, psi); 
				D.Mult(psi, phi0); 
				phi0 -= phi; 
				double mag = sqrt(mfem::InnerProduct(MPI_COMM_WORLD, phi, phi)); 
				phi0 *= (1.0/mag); 
				for (int i=0; i<phi0.Size(); i++) { phi0(i) = std::fabs(phi0(i)); }
				D.Mult(psi, phi); 
				if (phi.Min() < 0) MFEM_ABORT("negative energy density"); 

				// re-evaulate temperature given positive phi 
				Mtot.Mult(phi, phi_source); 
				temp_resid += phi_source;
				dplanck_dt_inv.Mult(temp_resid, dT); 
				for (int i=0; i<T.Size(); i++) {
					double Tnew = T(i) + dT(i); 
					if (Tnew < 0) {
						EventLog["under relax"] += 1; 
						T(i) = (1.0 - under_relax) * T(i) + under_relax*dT(i); 
						// T(i) = T(i) + dT(i)*under_relax; 
					} else {
						T(i) = Tnew; 
					}
				}
				break;
			} else {
				for (int i=0; i<T.Size(); i++) {
					double Tnew = T(i) + dT(i); 
					if (Tnew < 0) {
						EventLog["under relax"] += 1; 
						T(i) = (1.0 - under_relax) * T(i) + under_relax*dT(i); 
						// T(i) = T(i) + dT(i)*under_relax; 
					} else {
						T(i) = Tnew; 
					}
				}
			}
			outer++; 
		}

		// update opacity and opacity-dependent terms 
		if (temp_dependent_opacity) {
			total_gf.ProjectCoefficient(total_coef); 
			total_gf_dt.ProjectCoefficient(total_dt_coef); 
			Linv.AssembleLocalMatrices(); 
			delete Mtot.LoseMat();
			Mtot.Assemble(); 
			Mtot.Finalize(); 			
		}

		// prepare for next time step 
		time += time_step; 
		cycle++; 

		bool done = time >= final_time - 1e-12 or cycle == max_cycles; 

		if (cycle % output_freq == 0 or done) {
			Tpw.ProjectGridFunction(T); 
			dc.SetCycle(cycle); 
			dc.SetTime(time); 
			dc.SetTimeStep(time_step); 
			dc.Save(); 			
		}

		x0 = x; 

		EventLog.Synchronize(); 

		cycle_timer.Stop(); 
		double cycle_time = cycle_timer.RealTime(); 
		log["max schur solves"] = std::max(inners.Max(), log["max schur solves"]); 
		// output progress 
		out << YAML::BeginMap; 
			out << YAML::Key << "cycle" << YAML::Value << cycle; 
			out << YAML::Key << "simulation time" << YAML::Value << time; 
			out << YAML::Key << "||phi||" << YAML::Value << phi.Norml2(); 
			out << YAML::Key << "it" << YAML::Value << outer; 
			out << YAML::Key << "norm" << YAML::Value << norm;  
			out << YAML::Key << "inner iteration" << YAML::Value << YAML::BeginMap; 
				out << YAML::Key << "max" << YAML::Value << inners.Max(); 
				out << YAML::Key << "min" << YAML::Value << inners.Min(); 
				out << YAML::Key << "avg" << YAML::Value << (double)inners.Sum()/inners.Size(); 
				out << YAML::Key << "total" << YAML::Value << inners.Sum(); 
				out << YAML::Key << "max norm" << YAML::Value << max_inner_norm; 
			out << YAML::EndMap;
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
	delete outer_solver; 
	delete quad;
	for (int i=0; i<nattr; i++) { delete total_list[i]; } 
	for (int i=0; i<nattr; i++) { delete source_list[i]; }
	for (int i=0; i<nbattr; i++) { delete inflow_list[i]; }

	wall_timer.Stop(); 
	double wall_time = wall_timer.RealTime(); 
	out << YAML::Key << "wall time" << YAML::Value << io::FormatTimeString(wall_time); 

	// --- end yaml output --- 
	out << YAML::EndMap << YAML::Newline; 
}