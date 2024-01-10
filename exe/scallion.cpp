#include "mfem.hpp"
#include "igraph.h"
#include "tvector.hpp"
#include "angular_quadrature.hpp"
#include "mip.hpp"
#include "sweep.hpp"
#include "comment_stream.hpp"
#include "yaml-cpp/yaml.h"
#include "sol/sol.hpp"
#include <deque>
#include <set>
#include <span>

void ParPrint(std::function<void(int)> f) {
	for (int p=0; p<mfem::Mpi::WorldSize(); p++) {
		if (mfem::Mpi::WorldRank()==p) {
			f(p); 
			std::cout << std::flush; 
		}
		MPI_Barrier(MPI_COMM_WORLD); 
	}
}

void ParPrint(std::string tag, std::function<void(void)> f) {
	for (int p=0; p<mfem::Mpi::WorldSize(); p++) {
		if (mfem::Mpi::WorldRank()==p) {
			std::cout << tag << " rank " << p << ":" << std::endl; 
			f(); 
			std::cout << std::flush; 
		}
		MPI_Barrier(MPI_COMM_WORLD); 
	}
}

int main(int argc, char *argv[]) {
	// initialize MPI 
	// automatically calls MPI_Finalize 
	mfem::Mpi::Init(argc, argv); 
	mfem::Hypre::Init(); 
	const auto rank = mfem::Mpi::WorldRank(); 
	const bool root = rank == 0; 

	// take only input file 
	assert(argc==2); 

	// stream for output to terminal
	mfem::OutStream par_out(std::cout);
	// enable only for root so non-root procs don't clutter cout 
	if (rank!=0) par_out.Disable();

	// make mfem print everything with 
	// yaml comment preceeding 
	// helps keep output yaml parse-able 
	CommentStreamBuf comm_buf(mfem::out, '#'); 

	// YAML output 
	YAML::Emitter out(par_out);
	out.SetDoublePrecision(3); 
	out << YAML::BeginMap; 

	// --- load lua file --- 
	sol::state lua; 
	lua.open_libraries(); // allows using standard libraries (e.g. math) in input
	lua.script_file(argv[1]); // load from first cmd line argument 

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
	sol::optional<sol::table> dsa = sn["dsa"]; 
	double inner_tol, dsa_kappa; 
	int max_inner_it; 
	if (dsa) {
		auto dsa_tab = dsa.value(); 
		inner_tol = dsa_tab["tol"]; 
		max_inner_it = dsa_tab["max_it"]; 
		dsa_kappa = dsa_tab["kappa"]; 
	}

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
		if (dsa) {
			out << YAML::BeginMap; 
			out << YAML::Key << "type" << YAML::Value << "DSA"; 
			out << YAML::Key << "kappa" << YAML::Value << dsa_kappa;
			out << YAML::Key << "max iterations" << YAML::Value << max_inner_it; 
			out << YAML::Key << "tolerance" << YAML::Value << inner_tol; 
			out << YAML::EndMap; 
		}
		else out << YAML::Value << "none"; 
		out << YAML::Key << "max iterations" << YAML::Value << max_it;
		out << YAML::Key << "fp tolerance" << YAML::Value << tolerance;
	out << YAML::EndMap; 

	// --- sweep setup --- 
	// build MIP DSA operator
	mfem::ParBilinearForm Dform(&fes); 
	mfem::RatioCoefficient diffco(1./3, total); 
	mfem::Vector normal(dim); 
	normal = 0.0; normal(0) = 1.0; 
	double alpha = ComputeAlpha(quad, normal)/2;
	mfem::ConstantCoefficient alpha_c(alpha); 
	Dform.AddDomainIntegrator(new mfem::DiffusionIntegrator(diffco)); 
	Dform.AddDomainIntegrator(new mfem::MassIntegrator(absorption)); 
	// Dform.AddInteriorFaceIntegrator(new mfem::DGDiffusionIntegrator(diffco, -1, dsa_kappa)); 
	// Dform.AddBdrFaceIntegrator(new mfem::DGDiffusionIntegrator(diffco, -1, dsa_kappa)); 
	Dform.AddInteriorFaceIntegrator(new MIPDiffusionIntegrator(diffco, -1, dsa_kappa, alpha)); 
	// Dform.AddBdrFaceIntegrator(new MIPDiffusionIntegrator(diffco, -1, dsa_kappa, alpha)); 
	Dform.AddBdrFaceIntegrator(new mfem::BoundaryMassIntegrator(alpha_c)); 
	Dform.Assemble(); 
	Dform.Finalize(); 
	mfem::HypreParMatrix *dsa_mat = Dform.ParallelAssemble(); 

	// build DSA solver 
	mfem::CGSolver solver(MPI_COMM_WORLD); 
	solver.SetAbsTol(inner_tol); 
	solver.SetMaxIter(max_inner_it);
	solver.SetPrintLevel(0);
	mfem::HypreBoomerAMG amg(*dsa_mat); 
	amg.SetPrintLevel(0); 
	solver.SetOperator(*dsa_mat); 
	solver.SetPreconditioner(amg); 
	solver.iterative_mode = false; 

	// dsa source vector 
	mfem::Vector scatsource(fes.GetVSize()); 
	scatsource = 0.0; 

	// global scattering operator 
	mfem::BilinearForm Ms_form(&fes); 
	Ms_form.AddDomainIntegrator(new mfem::MassIntegrator(scattering)); 
	Ms_form.Assemble(); 
	Ms_form.Finalize();

	// allocate transport vector + views 
	mfem::Vector psi(psi_size);
	psi = 0.0; 
	TransportVectorView psi_view(psi.GetData(), psi_ext); 
	mfem::ParGridFunction phi(&fes), phi_old(&fes), delta_phi(&fes); 
	// initial guess 
	D.Mult(psi, phi); 
	phi_old = phi; 

	// form fixed source term 
	// allows either space/angle dependent source function 
	sol::optional<sol::function> source_func_avail = lua["source_function"]; 
	mfem::Vector source_vec(psi_size); 
	TransportVectorView source_vec_view(source_vec.GetData(), psi_ext); 
	if (source_func_avail) {
		auto source_func = source_func_avail.value(); 
		for (auto a=0; a<Nomega; a++) {
			const auto &Omega = quad.GetOmega(a); 
			mfem::Vector Omega3(3); 
			for (auto d=0; d<dim; d++) { Omega3(d) = Omega(d); }
			auto source_for_angle = [&Omega3, &source_func, &dim](const mfem::Vector &x) {
				double pos[3]; 
				for (auto d=0; d<dim; d++) { pos[d] = x(d); }
				return source_func(pos[0], pos[1], pos[2], Omega3(0), Omega3(1), Omega3(2)); 
			};
			mfem::ParLinearForm bform(&fes); 
			mfem::FunctionCoefficient source_coef(source_for_angle); 
			bform.AddDomainIntegrator(new mfem::DomainLFIntegrator(source_coef)); 
			bform.Assemble(); 
			for (int i=0; i<bform.Size(); i++) {
				source_vec_view(0,a,i) = bform[i]; 
			}
		}
	} 
	// or isotropic source keyed from element attribute 
	else {
		mfem::ParLinearForm bform(&fes); 
		bform.AddDomainIntegrator(new mfem::DomainLFIntegrator(source)); 
		bform.Assemble(); 
		for (auto a=0; a<Nomega; a++) {
			for (auto i=0; i<bform.Size(); i++) {
				source_vec_view(0,a,i) = bform[i]; 
			}
		}
	}

	// build sweep operator 
	InvAdvectionOperator Linv(fes, quad, psi_ext, total, inflow); 
	bool write_graph = lua["sn"]["write_graph"].get_or(false); 
	if (write_graph) Linv.WriteGraphToDot("graph"); 

	// --- begin fixed point iteration --- 
	out << YAML::Key << "fixed point iterations" << YAML::Value << YAML::BeginSeq; 
	mfem::StopWatch timer; // time each iteration 
	for (auto it=0; it<max_it; it++) {
		timer.Clear(); 
		timer.Start(); 

		// form scattering source 
		Ms_form.Mult(phi_old, scatsource); 
		D.MultTranspose(scatsource, psi); 
		psi += source_vec; 
		// sweep in place 
		Linv.Mult(psi, psi); 

		// compute new phi 
		D.Mult(psi, phi); 

		// do DSA step 
		int inner_it; 
		double inner_norm; 
		if (dsa) {
			delta_phi = phi; 
			delta_phi -= phi_old;
			Ms_form.Mult(delta_phi, scatsource);  
			solver.Mult(scatsource, delta_phi); 
			phi += delta_phi;

			inner_it = solver.GetNumIterations();  			
			inner_norm = solver.GetFinalNorm(); 
		}

		// compute norms 
		phi_old -= phi; 
		double norm = sqrt(InnerProduct(MPI_COMM_WORLD, phi_old, phi_old)); 
		phi_old = phi; 

		// report iteration info 
		timer.Stop(); 
		double it_time = timer.RealTime(); 
		out << YAML::BeginMap; 
		out << YAML::Key << "it" << YAML::Value << it+1; 
		out << YAML::Key << "norm" << YAML::Value << norm; 
		if (dsa) {
			out << YAML::Key << "inner it" << YAML::Value << inner_it; 
			out << YAML::Key << "inner norm" << YAML::Value << inner_norm; 
		}
		out << YAML::Key << "seconds per iteration" << YAML::Value << it_time; 
		out << YAML::EndMap; 

		// break if phi converged 
		if (norm < tolerance) {
			break; 
		}

	}
	out << YAML::EndSeq; // end iteration sequence 

	// --- end yaml output --- 
	out << YAML::EndMap << YAML::Newline; 

	// --- clean up dangling pointers --- 
	delete dsa_mat; 

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
		par_out << "l2 error: " << std::scientific << std::setprecision(3) << l2 << std::endl; 
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
}