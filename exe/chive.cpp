#include "mfem.hpp"
#include "yaml-cpp/yaml.h"
#include "sol/sol.hpp"

#include "tvector.hpp"
#include "angular_quadrature.hpp"
#include "mip.hpp"
#include "p1diffusion.hpp"
#include "sweep.hpp"
#include "transport_op.hpp"
#include "phase_coefficient.hpp"
#include "comment_stream.hpp"

class TransportIterationMonitor : public mfem::IterativeSolverMonitor
{
private:
	mfem::IterativeSolver const * const inner_solver; 
	YAML::Emitter &out; 
	mfem::StopWatch timer; 
public:
	mfem::Array<int> inner_it; 

	TransportIterationMonitor(YAML::Emitter &yaml, const mfem::IterativeSolver * const inner) 
		: out(yaml), inner_solver(inner) 
	{
		out << YAML::Key << "transport iterations" << YAML::Value << YAML::BeginSeq; 
	}
	void MonitorResidual(int it, double norm, const mfem::Vector &r, bool final) {
		// skip outputting initial norm
		if (it==0) {
			timer.Clear();
			timer.Start();
			return;  
		}

		timer.Stop(); 
		double time = timer.RealTime(); 

		out << YAML::BeginMap; 
		out << YAML::Key << "it" << YAML::Value << it; 
		out << YAML::Key << "norm" << YAML::Value << norm; 
		if (inner_solver) {
			out << YAML::Key << "inner it" << YAML::Value << inner_solver->GetNumIterations(); 
			out << YAML::Key << "inner norm" << YAML::Value << inner_solver->GetFinalNorm(); 
			inner_it.Append(inner_solver->GetNumIterations()); 
		}
		out << YAML::Key << "time" << YAML::Value << time; 
		out << YAML::EndMap << YAML::Newline; 

		timer.Clear();
		timer.Start(); 

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

	if (!lua_cmds.empty()) {
		lua.script(lua_cmds); 
	}

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
	mfem::Vector total_list(nattr), scattering_list(nattr), source_list(nattr); 
	for (auto i=0; i<attr_list.size(); i++) {
		sol::table data = materials[attr_list[i].c_str()]; 
		total_list(i) = data["total"]; 
		scattering_list(i) = data["scattering"]; 
		source_list(i) = data["source"]; 
	}

	// print materials list to cout 
	out << YAML::Key << "materials" << YAML::Value << YAML::BeginSeq; 
	for (auto i=0; i<attr_list.size(); i++) {
		out << YAML::BeginMap << YAML::Key << "name" << YAML::Value << attr_list[i]; 
		out << YAML::Key << "attribute" << YAML::Value << i+1; 
		out << YAML::Key << "total" << YAML::Value << total_list(i); 
		out << YAML::Key << "scattering" << YAML::Value << scattering_list(i); 
		out << YAML::Key << "source" << YAML::Value << source_list(i); 
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
	mfem::Vector inflow_list(nbattr); 
	for (auto i=0; i<bdr_attr_list.size(); i++) {
		double data = bcs[bdr_attr_list[i].c_str()]; 
		inflow_list(i) = data; 
	}

	// print list to screen 
	out << YAML::Key << "boundary conditions" << YAML::Value << YAML::BeginSeq; 
	for (auto i=0; i<bdr_attr_list.size(); i++) {
		out << YAML::BeginMap; 
		out << YAML::Key << "name" << YAML::Value << bdr_attr_list[i]; 
		out << YAML::Key << "bdr attribute" << YAML::Value << i+1; 
		out << YAML::Key << "inflow" << YAML::Value << inflow_list(i); 
		out << YAML::EndMap; 
	}
	out << YAML::EndSeq; 

	// map string to bdr attribute integer 
	// start from 1 since MFEM expects >0
	std::unordered_map<std::string,int> bdr_attr_map; 
	for (int i=0; i<bdr_attr_list.size(); i++) {
		bdr_attr_map[bdr_attr_list[i]] = i+1; 
	}

	// --- algorithmic parameters --- 
	sol::table sn = lua["sn"]; 
	int fe_order = sn["fe_order"]; 
	int sn_order = sn["sn_order"]; 
	int max_it = sn["max_it"]; 
	double tolerance = sn["tol"]; 
	sol::optional<sol::table> accel_avail = sn["acceleration"]; 
	std::string outer_type = sn["solver"].get_or(std::string("sli")); 

	// --- make mesh and solution spaces --- 
	auto mesh_node = lua["mesh"]; 
	sol::optional<std::string> fname = mesh_node["file"]; 
	mfem::Mesh smesh; 
	// load from a mesh file 
	if (fname) {
		smesh = mfem::Mesh::LoadFromFile(fname.value(), 1, 1);
		sol::optional<int> refinements = mesh_node["refinements"]; 
		if (refinements) {
			for (int r=0; r<refinements; r++) smesh.UniformRefinement(); 			
		}
	} 

	// create a cartesian mesh from extents and elements/axis 
	else {
		sol::table ne = mesh_node["num_elements"]; 
		sol::table extents = mesh_node["extents"]; 
		assert(ne.size() == extents.size()); 
		auto num_dim = ne.size(); 
		if (num_dim==1) {
			smesh = mfem::Mesh::MakeCartesian1D(ne[1], extents[1]); 
		} else if (num_dim==2) {
			smesh = mfem::Mesh::MakeCartesian2D(ne[1], ne[2], mfem::Element::QUADRILATERAL, true, extents[1], extents[2], false); 
		} else if (num_dim==3) {
			smesh = mfem::Mesh::MakeCartesian3D(ne[1], ne[2], ne[3], mfem::Element::HEXAHEDRON, extents[1], extents[2], extents[3], false); 
		} else { MFEM_ABORT("dim = " << num_dim << " not supported"); }
	}
	const auto dim = smesh.Dimension(); 

	// print mesh characteristics 
	double hmin, hmax, kmin, kmax; 
	smesh.GetCharacteristics(hmin, hmax, kmin, kmax); 
	out << YAML::Key << "mesh" << YAML::Value << YAML::BeginMap; 
	if (fname) {
		out << YAML::Key << "file name" << YAML::Value << fname.value(); 
		out << YAML::Key << "uniform refinements" << YAML::Value << (int)mesh_node["refinements"]; 
	}
	else {
		out << YAML::Key << "extents" << YAML::Value << YAML::Flow << YAML::BeginSeq; 
		sol::table extents = mesh_node["extents"]; 
		for (auto i=1; i<=extents.size(); i++) {
			out << (double)extents[i]; 
		} 
		out << YAML::EndSeq; 
		out << YAML::Key << "elements/axis" << YAML::Value << YAML::Flow << YAML::BeginSeq; 
		sol::table ne = mesh_node["num_elements"]; 
		for (auto i=1; i<=ne.size(); i++) {
			out << (int)ne[i]; 
		} 
		out << YAML::EndSeq; 		
	}
	out << YAML::Key << "dim" << YAML::Value << dim; 
	out << YAML::Key << "elements" << YAML::Value << smesh.GetNE(); 
	out << YAML::Key << "min mesh size" << YAML::Value << hmin; 
	out << YAML::Key << "max mesh size" << YAML::Value << hmax; 
	out << YAML::Key << "min mesh conditioning" << YAML::Value << kmin; 
	out << YAML::Key << "max mesh conditioning" << YAML::Value << kmax; 
	out << YAML::Key << "MPI ranks" << YAML::Value << mfem::Mpi::WorldSize(); 
	out << YAML::EndMap; 

	// --- assign materials to elements --- 
	mfem::Vector center(3); 
	center = 0.0; 
	sol::function geom_func = lua["material_map"]; 
	for (int e=0; e<smesh.GetNE(); e++) {
		mfem::Vector c; 
		smesh.GetElementCenter(e, c);
		for (int d=0; d<c.Size(); d++) { center(d) = c(d); } 
		std::string attr_name = geom_func(center(0), center(1), center(2)); 
		int attr = attr_map[attr_name]; 
		smesh.SetAttribute(e, attr); 
	}			

	// --- assign boundary conditions to boundary elements --- 
	sol::function bdr_func = lua["boundary_map"]; 
	for (int e=0; e<smesh.GetNBE(); e++) {
		const mfem::Element &el = *smesh.GetBdrElement(e); 
		int geom = smesh.GetBdrElementBaseGeometry(e);
		mfem::ElementTransformation &trans = *smesh.GetBdrElementTransformation(e); 
		mfem::Vector c(smesh.SpaceDimension()); 
		trans.Transform(mfem::Geometries.GetCenter(geom), c); 
		for (int d=0; d<c.Size(); d++) { center(d) = c(d); }
		std::string attr_name = bdr_func(center(0), center(1), center(2)); 
		smesh.SetBdrAttribute(e, bdr_attr_map[attr_name]); 
	}

	// --- create parallel mesh --- 
	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh); 
	mesh.ExchangeFaceNbrData(); // create parallel communication data needed for sweep 

	// --- build solution space --- 
	// DG space for transport solution 
	mfem::L2_FECollection fec(fe_order, dim, mfem::BasisType::GaussLegendre); 
	mfem::ParFiniteElementSpace fes(&mesh, &fec); 
	fes.ExchangeFaceNbrData(); // create parallel degree of freedom maps used in sweep 
	const auto Ndof = fes.GetVSize(); 
	const auto Ndof_fnbr = fes.GetFaceNbrVSize(); 

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
	mfem::PWConstCoefficient source(source_list); 
	mfem::SumCoefficient absorption(total, scattering, 1, -1); 
	mfem::PWConstCoefficient inflow(inflow_list); 

	// --- angular quadrature rule --- 
	LevelSymmetricQuadrature quad(sn_order, dim); 
	const auto Nomega = quad.Size(); 

	// --- transport vector setup --- 
	TransportVectorExtents psi_ext(1, Nomega, fes.GetVSize());
	const auto psi_size = TotalExtent(psi_ext); 
	MomentVectorExtents phi_ext(1, 1, fes.GetVSize());
	const auto phi_size = TotalExtent(phi_ext); 
	// integrates over angle psi -> phi 
	DiscreteToMoment D(quad, psi_ext, phi_ext); 

	// --- output algorithmic options used --- 
	out << YAML::Key << "sn" << YAML::Value << YAML::BeginMap; 
		out << YAML::Key << "fe order" << YAML::Value << fe_order; 
		out << YAML::Key << "sn order" << YAML::Value << sn_order; 
		out << YAML::Key << "num angles" << YAML::Value << Nomega; 			
		out << YAML::Key << "psi size" << YAML::Value << psi_size;
		out << YAML::Key << "phi size" << YAML::Value << phi_size;
		out << YAML::Key << "acceleration" << YAML::Value; 
		if (accel_avail) {
			sol::table accel = accel_avail.value();
			out << YAML::BeginMap; 
			for (const auto &it : accel) {
				out << YAML::Key << it.first.as<std::string>() << YAML::Value << it.second.as<std::string>(); 
			}
			out << YAML::EndMap; 
		}
		else out << YAML::Value << "none"; 
		out << YAML::Key << "solver" << YAML::Value << outer_type; 
		out << YAML::Key << "max iterations" << YAML::Value << max_it;
		out << YAML::Key << "fp tolerance" << YAML::Value << tolerance;
	out << YAML::EndMap << YAML::Newline; 

	// --- sweep setup --- 
	// global scattering operator 
	mfem::BilinearForm Ms_form(&fes); 
	Ms_form.AddDomainIntegrator(new mfem::MassIntegrator(scattering)); 
	Ms_form.Assemble(); 
	Ms_form.Finalize();

	// allocate transport vector + views 
	mfem::Vector psi(psi_size);
	psi = 0.0; 
	TransportVectorView psi_view(psi.GetData(), psi_ext); 
	mfem::ParGridFunction phi(&fes); 
	// initial guess 
	D.Mult(psi, phi); 

	// form fixed source term 
	// allows either space/angle dependent source function 
	sol::optional<sol::function> source_func_avail = lua["source_function"]; 
	sol::optional<sol::function> inflow_func_avail = lua["inflow_function"]; 
	PhaseSpaceCoefficient *source_coef, *inflow_coef; 
	if (source_func_avail) {
		std::function<double(double x, double y, double z, double mu, double eta, double xi)> lua_source
			= source_func_avail.value(); 		
		auto source_func = [lua_source](const mfem::Vector &x, const mfem::Vector &Omega) {
			return lua_source(x(0), x(1), x(2), Omega(0), Omega(1), Omega(2)); 
		};
		source_coef = new FunctionGrayCoefficient(source_func); 
	} 
	else {
		source_coef = new IsotropicGrayCoefficient(source); 
	}

	if (inflow_func_avail) {
		std::function<double(double x, double y, double z, double mu, double eta, double xi)> lua_inflow
			= inflow_func_avail.value(); 
		auto inflow_func = [lua_inflow](const mfem::Vector &x, const mfem::Vector &Omega) {
			return lua_inflow(x(0), x(1), x(2), Omega(0), Omega(1), Omega(2)); 
		};			
		inflow_coef = new FunctionGrayCoefficient(inflow_func); 		
	} 
	else {
		inflow_coef = new IsotropicGrayCoefficient(inflow); 
	}

	mfem::Vector source_vec(psi_size); 
	source_vec = 0.0; 
	TransportVectorView source_vec_view(source_vec.GetData(), psi_ext); 
	FormTransportSource(fes, quad, *source_coef, *inflow_coef, source_vec_view); 

	// build sweep operator 
	InverseAdvectionOperator Linv(fes, quad, psi_ext, total, inflow); 
	bool write_graph = lua["sn"]["write_graph"].get_or(false); 
	if (write_graph) Linv.WriteGraphToDot("graph"); 

	TransportOperator T(D, Linv, Ms_form, psi); 

	DiffusionSyntheticAccelerationOperator *prec = nullptr; 
	mfem::HypreParMatrix *dsa_mat = nullptr; 
	mfem::Operator *dsa_solver = nullptr, *dsa_prec = nullptr; 
	mfem::SuperLURowLocMatrix *slu_op = nullptr; 
	mfem::Vector normal(dim); 
	normal = 0.0; normal(0) = 1.0; 
	double alpha = ComputeAlpha(quad, normal)/2;
	if (accel_avail) {
		sol::table accel = accel_avail.value(); 
		std::string type = accel["type"]; 
		if (type == "MIP") {
			// build MIP DSA operator
			mfem::ParBilinearForm Dform(&fes); 
			mfem::RatioCoefficient diffco(1./3, total); 
			mfem::ConstantCoefficient alpha_c(alpha); 
			double dsa_kappa = accel["kappa"].get_or(pow(fe_order+1,2)); 
			Dform.AddDomainIntegrator(new mfem::DiffusionIntegrator(diffco)); 
			Dform.AddDomainIntegrator(new mfem::MassIntegrator(absorption)); 
			// Dform.AddInteriorFaceIntegrator(new mfem::DGDiffusionIntegrator(diffco, -1, dsa_kappa)); 
			// Dform.AddBdrFaceIntegrator(new mfem::DGDiffusionIntegrator(diffco, -1, dsa_kappa)); 
			Dform.AddInteriorFaceIntegrator(new MIPDiffusionIntegrator(diffco, -1, dsa_kappa, alpha)); 
			// Dform.AddBdrFaceIntegrator(new MIPDiffusionIntegrator(diffco, -1, dsa_kappa, alpha)); 
			Dform.AddBdrFaceIntegrator(new mfem::BoundaryMassIntegrator(alpha_c)); 
			Dform.Assemble(); 
			Dform.Finalize(); 
			dsa_mat = Dform.ParallelAssemble(); 

			auto *itsolve = new mfem::CGSolver(MPI_COMM_WORLD); 
			sol::optional<double> rel_avail = accel["reltol"]; 
			sol::optional<double> abs_avail = accel["abstol"]; 
			if (not(rel_avail or abs_avail)) MFEM_ABORT("must specify tolerance for iterative solver"); 
			if (rel_avail) itsolve->SetRelTol(rel_avail.value()); 
			if (abs_avail) itsolve->SetAbsTol(abs_avail.value()); 
			itsolve->SetMaxIter(accel["max_it"].get_or(50));
			itsolve->SetPrintLevel(0);
			auto *amg = new mfem::HypreBoomerAMG(*dsa_mat); 
			amg->SetPrintLevel(0); 
			itsolve->SetOperator(*dsa_mat); 
			itsolve->SetPreconditioner(*amg); 
			itsolve->iterative_mode = false; 
			dsa_solver = itsolve; 
			dsa_prec = amg; 
		} 
		else if (type == "P1SA") {
			std::string solve_type = accel["solver"]; 
			if (solve_type != "direct") MFEM_ABORT("only direct available for P1"); 
			// vector finite element space for current in P1SA 
			mfem::ParFiniteElementSpace vfes(&mesh, &fec, dim); 
			auto p1disc = std::unique_ptr<mfem::BlockOperator>(CreateP1DiffusionDiscretization(
				fes, vfes, total, absorption, alpha)); 
			auto mono = std::unique_ptr<mfem::HypreParMatrix>(BlockOperatorToMonolithic(*p1disc)); 
			slu_op = new mfem::SuperLURowLocMatrix(*mono); 
			auto *slu = new mfem::SuperLUSolver(*slu_op); 
			slu->SetPrintStatistics(false); 
			auto *ceo = new ComponentExtractionOperator(p1disc->RowOffsets(), 1); 
			auto *ceo_t = new mfem::TransposeOperator(*ceo); 
			dsa_solver = new mfem::TripleProductOperator(ceo, slu, ceo_t, true, true, true); 
		} 
		else if (type == "LDGSA") {
			mfem::ParFiniteElementSpace vfes(&mesh, &fec, dim); 
			mfem::Vector beta(dim); 
			for (int d=0; d<dim; d++) { beta(d) = d+1; }
			dsa_mat = CreateLDGDiffusionDiscretization(fes, vfes, total, absorption, alpha, &beta); 
			auto *itsolve = new mfem::CGSolver(MPI_COMM_WORLD); 
			sol::optional<double> rel_avail = accel["reltol"]; 
			sol::optional<double> abs_avail = accel["abstol"]; 
			if (not(rel_avail or abs_avail)) MFEM_ABORT("must specify tolerance for iterative solver"); 
			if (rel_avail) itsolve->SetRelTol(rel_avail.value()); 
			if (abs_avail) itsolve->SetAbsTol(abs_avail.value()); 
			itsolve->SetMaxIter(accel["max_it"].get_or(50));
			itsolve->SetPrintLevel(0);
			auto *amg = new mfem::HypreBoomerAMG(*dsa_mat); 
			amg->SetPrintLevel(0); 
			itsolve->SetOperator(*dsa_mat); 
			itsolve->SetPreconditioner(*amg); 
			itsolve->iterative_mode = false; 
			dsa_solver = itsolve; 
			dsa_prec = amg; 
		}
		else MFEM_ABORT("dsa type " << type << " not defined"); 

		prec = new DiffusionSyntheticAccelerationOperator(*dsa_solver, Ms_form); 
	}

	// form source for schur complement solve 
	// b -> D L^{-1} b
	Linv.Mult(source_vec, psi); 
	mfem::Vector schur_source(phi_size); 
	D.Mult(psi, schur_source);

	// create outer solver object 
	mfem::IterativeSolver *outer; 
	if (outer_type == "sli") {
		outer = new mfem::SLISolver(MPI_COMM_WORLD); 
	} 
	else if (outer_type == "gmres") {
		outer = new mfem::GMRESSolver(MPI_COMM_WORLD); 
	} 
	else if (outer_type == "bicg") {
		outer = new mfem::BiCGSTABSolver(MPI_COMM_WORLD); 
	}
	else if (outer_type == "fgmres") {
		outer = new mfem::FGMRESSolver(MPI_COMM_WORLD); 
	}
	else MFEM_ABORT("solver " << outer_type << " not defined"); 
	outer->SetRelTol(tolerance*tolerance); 
	outer->SetAbsTol(tolerance); 
	outer->SetMaxIter(max_it); 
	if (accel_avail) { outer->SetPreconditioner(*prec); }
	outer->SetOperator(T); 
	outer->SetPrintLevel(0); 
	TransportIterationMonitor monitor(out, dynamic_cast<mfem::IterativeSolver*>(dsa_solver)); 
	outer->SetMonitor(monitor); 
	outer->Mult(schur_source, phi); 
	out << YAML::Key << "outer iterations" << YAML::Value << outer->GetNumIterations();
	if (monitor.inner_it.Size()) {
		out << YAML::Key << "inner iteration" << YAML::Value << YAML::BeginMap; 
		out << YAML::Key << "min inner" << YAML::Value << monitor.inner_it.Min(); 
		out << YAML::Key << "max inner" << YAML::Value << monitor.inner_it.Max();
		out << YAML::Key << "avg inner" << YAML::Value << (double)monitor.inner_it.Sum()/monitor.inner_it.Size(); 		
		out << YAML::EndMap; 
	}

	// --- clean up hanging pointers --- 
	delete outer; 
	if (prec) delete prec; 
	if (dsa_prec) delete dsa_prec; 
	if (dsa_solver) delete dsa_solver; 
	if (slu_op) delete slu_op; 
	if (dsa_mat) delete dsa_mat; 
	delete source_coef; 
	delete inflow_coef; 

	// --- compute error if exact solution provided --- 
	sol::optional<sol::function> solution_func_avail = lua["solution"]; 
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
		out << YAML::Key << "l2 error" << YAML::Value << ss.str(); 
	}

	// --- output to paraview --- 
	mfem::ParGridFunction mesh_part(&fes0); 
	for (int i=0; i<mesh_part.Size(); i++) { mesh_part[i] = rank; }
	mfem::ParaViewDataCollection dc("solution", &mesh); 
	dc.RegisterField("phi", &phi); 
	dc.RegisterField("partition", &mesh_part); 
	dc.RegisterField("total", &total_gf); 
	dc.RegisterField("scattering", &scattering_gf); 
	dc.Save(); 

	timer.Stop(); 
	double time = timer.RealTime(); 
	out << YAML::Key << "solve time" << YAML::Value << time; 

	// --- end yaml output --- 
	out << YAML::EndMap << YAML::Newline; 
}