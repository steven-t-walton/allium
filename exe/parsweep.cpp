#include "mfem.hpp"
#include "igraph.h"
#include "tvector.hpp"
#include "angular_quadrature.hpp"
#include "mip.hpp"
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

	// print materials list to cout 
	out << YAML::Key << "materials" << YAML::Value << YAML::BeginMap; 
	for (auto i=0; i<attr_list.size(); i++) {
		out << YAML::Key << i+1 << YAML::Value << attr_list[i]; 
	}
	out << YAML::EndMap; 

	// get data from lua 
	auto nattr = attr_list.size(); 
	mfem::Vector total_list(nattr), scattering_list(nattr), source_list(nattr); 
	for (auto i=0; i<attr_list.size(); i++) {
		sol::table data = materials[attr_list[i].c_str()]; 
		total_list(i) = data["total"]; 
		scattering_list(i) = data["scattering"]; 
		source_list(i) = data["source"]; 
	}

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

	// print list to screen 
	out << YAML::Key << "boundary conditions" << YAML::Value << YAML::BeginMap; 
	for (auto i=0; i<bdr_attr_list.size(); i++) {
		out << YAML::Key << i+1;
		out << YAML::Value << bdr_attr_list[i]; 
	}
	out << YAML::EndMap; 

	// get values 
	auto nbattr = bdr_attr_list.size(); 
	mfem::Vector inflow_list(nbattr); 
	for (auto i=0; i<bdr_attr_list.size(); i++) {
		double data = bcs[bdr_attr_list[i].c_str()]; 
		inflow_list(i) = data; 
	}

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
	int dim = smesh.Dimension(); 
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
	mfem::Array<int> part(smesh.GetNE()); 
	mfem::Array<int> par_off(mfem::Mpi::WorldSize()+1);
	par_off = 0; 
	for (int e=0; e<smesh.GetNE(); e++) {
		par_off[mfem::Mpi::WorldSize() - e%mfem::Mpi::WorldSize()] += 1; 
	} 
	par_off.PartialSum(); 
	for (int i=0; i<par_off.Size()-1; i++) {
		for (int j=par_off[i]; j<par_off[i+1]; j++) {
			part[j] = i; 
		}
	}
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
	const auto global_Ne = mesh.GetGlobalNE(); 
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

	// --- build data needed for parallel graph not provided by ParMesh --- 
	// build local to global element map for elements straddling parallel faces 
	// maps index into the ghost element id to its global element number
	auto Ne = mesh.GetNE(); 
	auto nbr_el = mesh.GetNFaceNeighborElements(); 
	auto num_face_nbrs = mesh.GetNFaceNeighbors(); 
	auto nv = Ne + nbr_el; 
	const mfem::Table &send_face_nbr_elements = mesh.send_face_nbr_elements; 
	mfem::Table recv_face_nbr_elements(num_face_nbrs); 

	// exchange what each processor says is the global element number 
	MPI_Request *requests = new MPI_Request[2*num_face_nbrs];
	MPI_Request *send_requests = requests;
	MPI_Request *recv_requests = requests + num_face_nbrs;
	MPI_Status  *statuses = new MPI_Status[num_face_nbrs];

	// exchange how many messages will be sent 
	mfem::Array<int> mesh_sizes(num_face_nbrs);
	for (auto fn=0; fn<num_face_nbrs; fn++) {
		mesh_sizes[fn] = send_face_nbr_elements.RowSize(fn); 
	}
	for (auto fn=0; fn<num_face_nbrs; fn++) {
		auto nbr_rank = mesh.GetFaceNbrRank(fn); 
		auto size = send_face_nbr_elements.RowSize(fn); 
		// send, wait for message to clear so buffer doesn't deallocate first 
		MPI_Isend(&size, 1, MPI_INT, nbr_rank, 0, MPI_COMM_WORLD, &send_requests[fn]); 
		MPI_Wait(&send_requests[fn], &statuses[fn]); 

		MPI_Irecv(&recv_face_nbr_elements.GetI()[fn], 1, MPI_INT, nbr_rank, 0, MPI_COMM_WORLD, &recv_requests[fn]); 
	}
	MPI_Waitall(num_face_nbrs, recv_requests, statuses);

	// send actual data 
	recv_face_nbr_elements.MakeJ();  
	for (auto fn=0; fn<num_face_nbrs; fn++) {
		auto nbr_rank = mesh.GetFaceNbrRank(fn); 
		MPI_Isend(send_face_nbr_elements.GetRow(fn), send_face_nbr_elements.RowSize(fn), 
			MPI_INT, nbr_rank, 0, MPI_COMM_WORLD, &send_requests[fn]); 
		MPI_Irecv(recv_face_nbr_elements.GetRow(fn), recv_face_nbr_elements.RowSize(fn), 
			MPI_INT, nbr_rank, 0, MPI_COMM_WORLD, &recv_requests[fn]); 
	}
	MPI_Waitall(num_face_nbrs, send_requests, statuses);
	MPI_Waitall(num_face_nbrs, recv_requests, statuses);

	// --- generate local offsets --- 
	// this processor owns elements [mesh_offsets[0], mesh_offsets[1])
	mfem::Array<HYPRE_BigInt> mesh_offsets; 
	mfem::Array<HYPRE_BigInt> *ptr = &mesh_offsets; 
	mesh.GenerateOffsets(1, &Ne, &ptr); 

	// --- swap offsets with parallel neighbors --- 
	// from this info can compute global element number of ghost cells 
	mfem::Array<HYPRE_BigInt> face_nbr_offsets(num_face_nbrs), mesh_face_nbr_offsets(num_face_nbrs); 
	for (auto fn=0; fn<num_face_nbrs; fn++) {
		auto nbr_rank = mesh.GetFaceNbrRank(fn); 
		MPI_Isend(&mesh_offsets[0], 1, HYPRE_MPI_BIG_INT, nbr_rank, 0, MPI_COMM_WORLD, &send_requests[fn]); 
		MPI_Irecv(&mesh_face_nbr_offsets[fn], 1, HYPRE_MPI_BIG_INT, nbr_rank, 0, MPI_COMM_WORLD, &recv_requests[fn]); 
	}
	MPI_Waitall(num_face_nbrs, send_requests, statuses); 
	MPI_Waitall(num_face_nbrs, recv_requests, statuses); 

	// scale by Nomega to represent all owned dofs on each processor 
	face_nbr_offsets = mesh_face_nbr_offsets; 
	for (auto &i : face_nbr_offsets) { i *= Nomega; }

	delete[] requests; delete[] statuses; 

	// --- build the actual map with "inverse" via unorder_map --- 
	mfem::Array<int> mesh_l2g(nbr_el); 
	std::unordered_map<int,int> mesh_g2l; 
	for (auto fn=0; fn<num_face_nbrs; fn++) {
		auto idx_offset = mesh.face_nbr_elements_offset[fn]; 
		auto *row = recv_face_nbr_elements.GetRow(fn); 
		for (auto i=0; i<recv_face_nbr_elements.RowSize(fn); i++) {
			mesh_l2g[idx_offset + i] = row[i] + mesh_face_nbr_offsets[fn]; // local ghost id -> global element 
			mesh_g2l[row[i] + mesh_face_nbr_offsets[fn]] = idx_offset + i; // global element -> local ghost id 
		}
	}

	// --- store face neighbor elements to face neighbor number --- 
	// maps ghost id to which processor owns the data 
	mfem::Array<int> nbr_to_fn(nbr_el); 
	for (auto fn=0; fn<recv_face_nbr_elements.Size(); fn++) {
		auto idx_offset = mesh.face_nbr_elements_offset[fn]; 
		const auto *row = recv_face_nbr_elements.GetRow(fn); 
		for (auto i=0; i<recv_face_nbr_elements.RowSize(fn); i++) {
			nbr_to_fn[idx_offset + i] = fn; 
		}
	}

	// --- create graph --- 
	const auto &el2el = mesh.ElementToElementTable(); // element to element connectivity 
	// faces to elements, has 2 elements for interior faces, 1 for boundary 
	// includes parallel shared faces to ghost/face neighbor elements 
	// handles 1/2/3D 
	const mfem::Table *f2el = mesh.GetFaceToAllElementTable(); 
	// "invert" to go from element id to the list of faces associated with that element 
	const mfem::Table *el2f = Transpose(*f2el); // needed for sweep 

	// loop through faces, add directed edges based on Omega.n 
	// normal is assumed to be piecewise constant! Works for linear meshes only 
	mfem::Array<igraph_integer_t> edges; // list of edges 
	edges.Reserve(f2el->Size_of_connections()*Nomega*2);
	mfem::Vector normal(dim); // normal vector 
	for (int r=0; r<f2el->Size(); r++) {
		// get view into row of table 
		auto row = std::span(f2el->GetRow(r), f2el->RowSize(r)); 
		if (row.size()==2) { // two faces => interior face 
			// compute normal 
			auto ref_geom = mesh.GetFaceGeometry(r);
			const auto &int_rule = mfem::IntRules.Get(ref_geom, 1); 
			auto *trans = mesh.GetFaceElementTransformations(r); 
			trans->SetAllIntPoints(&int_rule[0]); 
			if (dim==1) {
				normal(0) = 2*trans->GetElement1IntPoint().x - 1.0;
			} else {
				CalcOrtho(trans->Jacobian(), normal); 				
			}

			// add directed edges based on Omega.n 
			for (int a=0; a<Nomega; a++) {
				auto e1 = row[0]; 
				auto e2 = row[1]; 
				const auto &Omega = quad.GetOmega(a); 
				double dot = normal*Omega; 
				if (dot < 0) std::swap(e1,e2); 
				auto dof1 = (e1<Ne) ? e1*Nomega + a : Ne*Nomega + (e1 - Ne)*Nomega + a; 
				auto dof2 = (e2<Ne) ? e2*Nomega + a : Ne*Nomega + (e2 - Ne)*Nomega + a; 
				edges.Append(dof1); 
				edges.Append(dof2); 
			}
		}
	}

	delete f2el; 

	// --- convert to igraph --- 
	igraph_t graph; 
	igraph_vector_int_t edges_igraph;
	// view into MFEM array data 
	// igraph does not own the data 
	igraph_vector_int_view(&edges_igraph, edges.GetData(), edges.Size()); 
	igraph_create(&graph, &edges_igraph, 0, true); 

	// use igraph to plot the graph for debugging 
	bool write_graph = lua["sn"]["write_graph"].get_or(false); 
	if (write_graph) {
		FILE* file = fopen(mfem::MakeParFilename("graph.", rank, ".dot").c_str(), "w"); 
		igraph_write_graph_dot(&graph, file); 
		fclose(file); 		
	}

	// ensure DAG 
	igraph_bool_t is_dag; 
	igraph_is_dag(&graph, &is_dag); 
	if (!is_dag) { MFEM_ABORT("graph is not a dag"); }

	// --- sweep setup --- 
	// build MIP DSA operator
	mfem::ParBilinearForm Dform(&fes); 
	mfem::RatioCoefficient diffco(1./3, total); 
	double alpha = ComputeAlpha(quad, normal)/2;
	mfem::ConstantCoefficient alpha_c(alpha); 
	Dform.AddDomainIntegrator(new mfem::DiffusionIntegrator(diffco)); 
	Dform.AddDomainIntegrator(new mfem::MassIntegrator(absorption)); 
	// Dform.AddInteriorFaceIntegrator(new mfem::DGDiffusionIntegrator(diffco, -1, dsa_kappa)); 
	// Dform.AddBdrFaceIntegrator(new mfem::DGDiffusionIntegrator(diffco, -1, dsa_kappa)); 
	Dform.AddInteriorFaceIntegrator(new MIPDiffusionIntegrator(diffco, 1, dsa_kappa, alpha)); 
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

	// global scattering operator DSA 
	mfem::BilinearForm Ms_form(&fes); 
	Ms_form.AddDomainIntegrator(new mfem::MassIntegrator(scattering)); 
	Ms_form.Assemble(); 
	Ms_form.Finalize();

	// allocate transport vector + views 
	mfem::Vector psi(psi_size + Ndof_fnbr*Nomega); // owned + ghost data 
	psi = 0.0; 
	TransportVectorView psi_view(psi.GetData(), psi_ext); 
	// face neighbor data 
	TransportVectorExtents psi_fnbr_ext(1, Nomega, Ndof_fnbr); 
	// view into end of psi 
	// index by face neighbor element id 
	TransportVectorView psi_fnbr_view(psi.GetData()+psi_size, psi_fnbr_ext); 
	// working vectors for zeroth moment solution 
	mfem::ParGridFunction phi(&fes), phi_old(&fes), delta_phi(&fes); 
	// initial guess 
	D.Mult(psi, phi); 
	phi_old = phi; 

	// form source term 
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

	// build sweep helper data structures 
	// convert a generic face id to the bdr face id
	// required to extract the proper bdr attribute used to determine inflow condition
	// Mesh::GetFaceToBdrElMap does not work in parallel due to addition of parallel shared faces 
	std::unordered_map<int,int> f2be; 
	for (auto i=0; i<mesh.GetNBE(); i++) {
		f2be[mesh.GetBdrElementFaceIndex(i)] = i; 
	}
 	// place to store face ids/element, dofs/element, and dofs of neighboring elements 
	mfem::Array<int> faces, dofs, dofs2; 

	// igraph vectors to store neighbors in graph traversal 
	igraph_vector_int_t nbrs, nbr_nbrs; 
	igraph_vector_int_init(&nbrs, 0); 
	igraph_vector_int_init(&nbr_nbrs, 0); 

	// --- begin fixed point iteration --- 
	out << YAML::Key << "fixed point iterations" << YAML::Value << YAML::BeginSeq; 
	mfem::StopWatch timer; // time each iteration 
	for (auto it=0; it<max_it; it++) {
		timer.Clear(); 
		timer.Start(); 

		// --- determine roots of graph --- 
		auto num_nodes = igraph_vcount(&graph); 
		mfem::Array<igraph_integer_t> degrees(num_nodes); 
		igraph_vector_int_t degrees_view; 
		igraph_vector_int_view(&degrees_view, degrees.GetData(), degrees.Size()); 
		// count edges incoming to each vertex 
		IGRAPH_CHECK(igraph_degree(&graph, &degrees_view, igraph_vss_all(), IGRAPH_IN, 0));
		// list of nodes to visit (not including those that depend on another processor) 
		// initialize with roots of graph 
		std::deque<int> local; 
		for (int i=0; i<degrees.Size(); i++) {
			if (degrees[i] == 0) {
				if (i < Ne*Nomega) {
					local.push_back(i); 
				} 
			}
		}

		// --- do sweep --- 
		mfem::Array<int> order; // store sweep order for diagnostic purposes 
		order.Reserve(num_nodes); 
		// vertices traversed, only count owned elements 
		int nsweep = 0; 
		while (nsweep < Ne*Nomega) {
			// --- sweep on processor-local domain --- 
			while (not(local.empty())) {
				nsweep++; //increment sweep counter to know when to stop 
				// get first element of queue 
				auto node = local.front();
				local.pop_front(); 
				// save processor local ordering as diagnostic 
				order.Append(node); 

				// --- do work --- 
				// deconstruct angle/space id
				auto a = node % Nomega; 
				auto e = node / Nomega; 
				const auto &Omega = quad.GetOmega(a); 
				const auto &el = *fes.GetFE(e); 
				auto &trans = *mesh.GetElementTransformation(e); 
				fes.GetElementDofs(e, dofs); 

				// -- assemble volumetric terms --
				// volumetric streaming term 
				mfem::DenseMatrix G; 
				mfem::VectorConstantCoefficient Q(Omega); 
				mfem::ConservativeConvectionIntegrator conv_int(Q, 1.0); 
				conv_int.AssembleElementMatrix(el, trans, G); 

				// collision term 
				mfem::DenseMatrix Mt; 
				mfem::MassIntegrator mt_int(total); 
				mt_int.AssembleElementMatrix(el, trans, Mt); 

				// get source term 
				mfem::Vector rhs(dofs.Size()); 
				for (auto i=0; i<dofs.Size(); i++) { rhs[i] = source_vec_view(0,a,dofs[i]); }

				// scattering
				mfem::DenseMatrix Ms; 
				mfem::MassIntegrator ms_int(scattering); 
				ms_int.AssembleElementMatrix(el, trans, Ms); 
				mfem::Vector phi_local(dofs.Size()); 
				phi.GetSubVector(dofs, phi_local); 
				Ms.AddMult(phi_local, rhs, 1./4/M_PI); 

				// -- do face terms -- 
				// get faces associated with mesh element e 
				// face are UNIQUE => normal is not always outward 
				el2f->GetRow(e, faces);
				mfem::DenseMatrix F(dofs.Size());
				F = 0.0; 
				double alpha = 1.0; 
				double beta = 0.5; 
				for (int fidx=0; fidx<faces.Size(); fidx++) {
					auto info = mesh.GetFaceInformation(faces[fidx]); 
					mfem::FaceElementTransformations *face_trans; 
					if (info.IsShared()) {
						face_trans = mesh.GetSharedFaceTransformationsByLocalIndex(faces[fidx], true); 
					} else {
						face_trans = mesh.GetFaceElementTransformations(faces[fidx]); 
					}
					mfem::DGTraceIntegrator upw_int(Q, alpha, beta); 
					mfem::DenseMatrix f; 
					const auto &el1 = *fes.GetFE(face_trans->Elem1No); 
					const auto *el2 = &el1; 
					if (face_trans->Elem2No >= 0) {
						el2 = fes.GetFE(face_trans->Elem2No); 
					}
					int ndofs1 = el1.GetDof(); 
					int ndofs2 = el2->GetDof(); 
					upw_int.AssembleFaceMatrix(el1, *el2, *face_trans, f); 
					mfem::DenseMatrix f11, f12; 
					int ep = (e==face_trans->Elem1No) ? face_trans->Elem2No : face_trans->Elem1No;
					if (e == face_trans->Elem1No) {
						f.GetSubMatrix(0, ndofs1, 0, ndofs1, f11); 
						f.GetSubMatrix(0, ndofs1, ndofs1, f.Width(), f12); 				
					} else {
						f.GetSubMatrix(ndofs1, f.Height(), ndofs1, f.Width(), f11); 
						f.GetSubMatrix(ndofs1, f.Height(), 0, ndofs1, f12); 
					}
					F += f11; 
					if (ep>=0) {
						mfem::Vector psi2; 
						// face neighbor data 
						if (info.IsShared()) {
							fes.GetFaceNbrElementVDofs(ep-Ne, dofs2); 
							psi2.SetSize(dofs2.Size()); 
							for (int i=0; i<dofs2.Size(); i++) { psi2(i) = psi_fnbr_view(0, a, dofs2[i]); }
						}
						// local data 
						else {
							fes.GetElementDofs(ep, dofs2); 
							psi2.SetSize(dofs2.Size()); 
							for (int i=0; i<dofs2.Size(); i++) { psi2(i) = psi_view(0, a, dofs2[i]); }						
						}
						f12.AddMult(psi2, rhs, -1.0); 									
					} else {
						auto be = f2be[face_trans->ElementNo]; 
						auto &bdr_face_trans = *mesh.GetBdrFaceTransformations(be); 
						mfem::BoundaryFlowIntegrator bdr_flow(inflow, Q, alpha, beta);
						mfem::Vector elvec; 
						bdr_flow.AssembleRHSElementVect(el, bdr_face_trans, elvec);  
						rhs -= elvec; 
					}
				}  

				// form local system 
				mfem::DenseMatrix A(G); 
				A += F; 
				A += Mt; 

				// solve, solution overwritten into rhs 
				mfem::LinearSolve(A, rhs.GetData()); 

				// scatter back 
				for (int i=0; i<dofs.Size(); i++) { psi_view(0, a, dofs[i]) = rhs(i); }


				// --- compute next + send data to other processors --- 
				degrees[node] = -1; // already visited 
				// get neighbors 
				igraph_neighbors(&graph, &nbrs, node, IGRAPH_OUT);  
				auto nbrs_view = std::span(VECTOR(nbrs), igraph_vector_int_size(&nbrs)); 
				std::set<int> send_fn_set; // store unique set of processors to send data to 
				for (const auto &nbr : nbrs_view) {
					degrees[nbr] -= 1; // reduce degree since node has been visited 
					// if nbr is off-processor, add to send_set regardless of degree 
					if (nbr >= Ne*Nomega) {
						assert(node < Ne*Nomega); 
						assert(nbr >= Ne*Nomega); 
						// get neighbor of neighbor to find number of processors to send data to 
						igraph_neighbors(&graph, &nbr_nbrs, node, IGRAPH_OUT);
						auto nbr_nbrs_view = std::span(VECTOR(nbr_nbrs), igraph_vector_int_size(&nbr_nbrs)); 	
						for (const auto &nbr_nbr : nbr_nbrs_view) {
							if (nbr_nbr < Ne*Nomega) continue; // only send to off-proc neighbors 
							auto nbr_id = nbr_nbr - Ne*Nomega; 
							auto fn = nbr_to_fn[nbr_id / Nomega]; 
							send_fn_set.insert(fn); 
						}
					} 
					// neighbor is local, add to queue 
					else {
						if (degrees[nbr] == 0) {
							local.push_back(nbr); 						
						}					
					}
				}

				// MPI send data if available 
				auto tag = mesh_offsets[0] * Nomega + node; 
				for (const auto &fn : send_fn_set) {
					auto nbr_rank = mesh.GetFaceNbrRank(fn); 
					MPI_Request send_request; 
					MPI_Isend(rhs.GetData(), dofs.Size(), MPI_DOUBLE, nbr_rank, tag, MPI_COMM_WORLD, &send_request); 
					MPI_Wait(&send_request, MPI_STATUS_IGNORE); 
				}
			}

			// --- receive data from other processors --- 
			// loop until no messages remain 
			while (true) {
				// probe to see if message available 
				int avail; 
				MPI_Status status; 
				MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &avail, &status); 
				// exit if no messages available 
				if (avail==0) break; 

				// iprobe has same status as if MPI_Recv was called 
				// extract info to reproduce call to MPI_Recv 
				// also used to find the right buffer to place data in 
				auto tag = status.MPI_TAG; 
				auto source = status.MPI_SOURCE; 
				// convert tag (global idx) to a local index 
				auto e = tag / Nomega; 
				auto a = tag % Nomega; 
				auto local_e = mesh_g2l[e] + Ne; 
				auto local_id = local_e*Nomega + a;		
				// size of message 
				int count; 
				MPI_Get_count(&status, MPI_DOUBLE, &count); 	

				// get the data 
				// use a buffer in case psi not striding in space first 
				double buffer[count]; 
				MPI_Recv(&buffer, count, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);

				mfem::Array<int> nbr_dofs; 
				fes.GetFaceNbrElementVDofs(local_e-Ne, nbr_dofs); 
				for (auto i=0; i<nbr_dofs.Size(); i++) { psi_fnbr_view(0,a,nbr_dofs[i]) = buffer[i]; }

				// mark data as recvd, traverse neighbors to add the locally owned tail to the queue 
				// for use in the local sweep 
				degrees[local_id] = -1; 
				igraph_neighbors(&graph, &nbrs, local_id, IGRAPH_OUT); 
				auto nbrs_view = std::span(VECTOR(nbrs), igraph_vector_int_size(&nbrs)); 
				for (const auto &nbr : nbrs_view) {
					degrees[nbr] -= 1; 
					if (degrees[nbr]==0) {
						local.push_back(nbr); // add the locally owned id to the queue 
					}
				}
			}
		}
		MPI_Barrier(MPI_COMM_WORLD); 

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