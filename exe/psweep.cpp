#include "mfem.hpp"
#include "igraph.h"
#include "tvector.hpp"
#include "angular_quadrature.hpp"
#include "mip.hpp"
#include "yaml-cpp/yaml.h"
#include "sol/sol.hpp"
#include <deque>

void ParPrint(std::function<void(int)> f) {
	for (int p=0; p<mfem::Mpi::WorldSize(); p++) {
		if (mfem::Mpi::WorldRank()==p) {
			f(p); 
		}
		MPI_Barrier(MPI_COMM_WORLD); 
	}
}

void ParPrint(std::string tag, std::function<void(void)> f) {
	for (int p=0; p<mfem::Mpi::WorldSize(); p++) {
		if (mfem::Mpi::WorldRank()==p) {
			std::cout << tag << " rank " << p << ":" << std::endl; 
			f(); 
		}
		MPI_Barrier(MPI_COMM_WORLD); 
	}
}

int main(int argc, char *argv[]) {
	mfem::Mpi::Init(argc, argv); 
	bool viz = true; 
	assert(argc==2); 

	// --- load lua file --- 
	sol::state lua; 
	lua.script_file(argv[1]); 

	// --- extract list of materials --- 
	std::vector<std::string> attr_list; 
	sol::table materials = lua["materials"]; 
	if (materials.valid()) {
		for (const auto &material : materials) {
			auto key = material.first.as<std::string>(); 
			attr_list.push_back(key); 
		}
	} else { MFEM_ABORT("materials not defined"); }

	auto nattr = attr_list.size(); 
	mfem::Vector total_list(nattr), scattering_list(nattr), source_list(nattr); 
	for (auto i=0; i<attr_list.size(); i++) {
		sol::table data = materials[attr_list[i].c_str()]; 
		total_list(i) = data["total"]; 
		scattering_list(i) = data["scattering"]; 
		source_list(i) = data["source"]; 
	}

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
	auto nbattr = bdr_attr_list.size(); 
	mfem::Vector inflow_list(nbattr); 
	for (auto i=0; i<bdr_attr_list.size(); i++) {
		double data = bcs[bdr_attr_list[i].c_str()]; 
		inflow_list(i) = data; 
	}

	std::unordered_map<std::string,int> bdr_attr_map; 
	for (int i=0; i<bdr_attr_list.size(); i++) {
		bdr_attr_map[bdr_attr_list[i]] = i+1; 
	}

	// --- create coefficients for input material data ---
	mfem::PWConstCoefficient total(total_list); 
	mfem::PWConstCoefficient scattering(scattering_list); 
	mfem::PWConstCoefficient source(source_list); 
	mfem::SumCoefficient absorption(total, scattering, 1, -1); 
	mfem::PWConstCoefficient inflow(inflow_list); 

	// --- algorithmic parameters --- 
	sol::table sn = lua["sn"]; 
	int fe_order = sn["fe_order"]; 
	int sn_order = sn["sn_order"]; 
	bool dsa = sn["use_dsa"]; 
	int max_it = sn["max_it"]; 
	double tolerance = sn["tol"]; 

	// --- make mesh and solution spaces --- 
	auto mesh_node = lua["mesh"]; 
	sol::optional<std::string> fname = mesh_node["file"]; 
	mfem::Mesh smesh; 
	if (fname) {
		smesh = mfem::Mesh::LoadFromFile(fname.value(), 1, 1);
		sol::optional<int> refinements = mesh_node["refinements"]; 
		if (refinements) {
			for (int r=0; r<refinements; r++) smesh.UniformRefinement(); 			
		}
	} else {
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
	if (mfem::Mpi::Root()) std::cout << "dim = " << dim << std::endl; 

	// --- assign materials to elements --- 
	mfem::Vector center(3); 
	center = 0.0; 
	sol::function geom_func = lua["material_map"]; 
	for (int e=0; e<smesh.GetNE(); e++) {
		mfem::Vector c; 
		smesh.GetElementCenter(e, c);
		for (int d=0; d<c.Size(); d++) { center(d) = c(d); } 
		std::string attr_name = geom_func(center(0), center(1), center(2)); 
		smesh.SetAttribute(e, attr_map[attr_name]); 
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
	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh, part.GetData()); 
	mesh.ExchangeFaceNbrData();

	// --- build solution space --- 
	mfem::L2_FECollection fec(fe_order, dim, mfem::BasisType::GaussLegendre); 
	mfem::ParFiniteElementSpace fes(&mesh, &fec); 
	auto Ndof = fes.GetVSize(); 

	// --- angular quadrature rule --- 
	LevelSymmetricQuadrature quad(sn_order, dim); 
	auto global_Ne = mesh.GetGlobalNE(); 
	// auto Nomega = quad.Size(); 
	auto Nomega = 3; 
	if (mfem::Mpi::Root()) std::cout << "num elements = " << global_Ne << std::endl; 
	if (mfem::Mpi::Root()) std::cout << "SN order = " << sn_order << ", angles = " << Nomega << std::endl; 

	// --- build local to global element map --- 
	auto Ne = mesh.GetNE(); 
	auto nbr_el = mesh.GetNFaceNeighborElements(); 
	auto num_face_nbrs = mesh.GetNFaceNeighbors(); 
	auto nv = Ne + nbr_el; 
	const mfem::Table &send_face_nbr_elements = mesh.send_face_nbr_elements; 
	mfem::Table recv_face_nbr_elements(num_face_nbrs); 

	MPI_Request *requests = new MPI_Request[2*num_face_nbrs];
	MPI_Request *send_requests = requests;
	MPI_Request *recv_requests = requests + num_face_nbrs;
	MPI_Status  *statuses = new MPI_Status[num_face_nbrs];

	for (auto fn=0; fn<num_face_nbrs; fn++) {
		auto nbr_rank = mesh.GetFaceNbrRank(fn); 
		auto size = send_face_nbr_elements.RowSize(fn); 
		MPI_Isend(&size, 1, MPI_INT, nbr_rank, 0, MPI_COMM_WORLD, &send_requests[fn]); 
		MPI_Wait(&send_requests[fn], &statuses[fn]); 
		MPI_Irecv(&recv_face_nbr_elements.GetI()[fn], 1, MPI_INT, nbr_rank, 0, MPI_COMM_WORLD, &recv_requests[fn]); 
	}
	MPI_Waitall(num_face_nbrs, recv_requests, statuses);
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
	mfem::Array<HYPRE_BigInt> mesh_offsets; 
	mfem::Array<HYPRE_BigInt> *ptr = &mesh_offsets; 
	mesh.GenerateOffsets(1, &Ne, &ptr); 

	// --- swap offsets --- 
	mfem::Array<HYPRE_BigInt> face_nbr_offsets(num_face_nbrs), mesh_face_nbr_offsets(num_face_nbrs); 
	for (auto fn=0; fn<num_face_nbrs; fn++) {
		auto nbr_rank = mesh.GetFaceNbrRank(fn); 
		MPI_Isend(&mesh_offsets[0], 1, HYPRE_MPI_BIG_INT, nbr_rank, 0, MPI_COMM_WORLD, &send_requests[fn]); 
		MPI_Irecv(&mesh_face_nbr_offsets[fn], 1, HYPRE_MPI_BIG_INT, nbr_rank, 0, MPI_COMM_WORLD, &recv_requests[fn]); 
	}
	MPI_Waitall(num_face_nbrs, send_requests, statuses); 
	MPI_Waitall(num_face_nbrs, recv_requests, statuses); 

	mfem::Array<int> mesh_l2g(nbr_el); 
	std::unordered_map<int,int> mesh_g2l; 
	for (auto fn=0; fn<num_face_nbrs; fn++) {
		auto idx_offset = mesh.face_nbr_elements_offset[fn]; 
		auto *row = recv_face_nbr_elements.GetRow(fn); 
		for (auto i=0; i<recv_face_nbr_elements.RowSize(fn); i++) {
			mesh_l2g[idx_offset + i] = row[i] + mesh_face_nbr_offsets[fn]; 
			mesh_g2l[row[i] + mesh_face_nbr_offsets[fn]] = idx_offset + i; 
		}
	}
	face_nbr_offsets = mesh_face_nbr_offsets; 
	for (auto &i : face_nbr_offsets) { i *= Nomega; }

	// --- setup transport vectors --- 
	TransportVectorExtents extents(1, Nomega, fes.GetVSize());
	auto psi_size = TotalExtent(extents); 
	MomentVectorExtents extents_phi(1, 1, fes.GetVSize());
	auto phi_size = TotalExtent(extents_phi); 
	DiscreteToMoment D(quad, extents, extents_phi); 

	// --- create graph --- 
	const auto &el2el = mesh.ElementToElementTable(); 
	const mfem::Table *f2el = mesh.GetFaceToAllElementTable(); 
	const mfem::Table *el2f = Transpose(*f2el); 

	std::unordered_map<int,int> lface_to_fn; 
	for (auto sf=0; sf<mesh.GetNSharedFaces(); sf++) {
		auto lf = mesh.GetSharedFace(sf); 
		auto fn = mesh.sface_to_fn[sf]; 
		lface_to_fn[lf] = fn; 
	}

	mfem::Array<mfem::Connection> send_elems_list; 
	mfem::Array<igraph_integer_t> edges; 
	edges.Reserve(el2f->Size_of_connections()*Nomega*2);
	mfem::Vector normal(dim); 
	auto nfaces = mesh.GetNumFaces(); 
	mfem::Array<int> row; 
	for (int r=0; r<f2el->Size(); r++) {
		f2el->GetRow(r, row); 
		if (row.Size()==2) {
			auto ref_geom = mesh.GetFaceGeometry(r);
			const auto &int_rule = mfem::IntRules.Get(ref_geom, 1); 
			auto *trans = mesh.GetFaceElementTransformations(r); 
			trans->SetAllIntPoints(&int_rule[0]); 
			if (dim==1) {
				normal(0) = 2*trans->GetElement1IntPoint().x - 1.0;
			} else {
				CalcOrtho(trans->Jacobian(), normal); 				
			}

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
				if (e1 < Ne and e2 >= Ne) {
					auto fn = lface_to_fn[r]; 
					send_elems_list.Append({fn, dof1}); 
				} 
			}
		}
	}

	delete f2el; 

	send_elems_list.Sort(); 
	send_elems_list.Unique(); 
	mfem::Table send_elems(num_face_nbrs, send_elems_list); 
	mfem::Table recv_elems; 
	recv_elems.MakeI(num_face_nbrs); 

	// --- exchange how many messages will be sent --- 
	mfem::Array<int> send_sizes(num_face_nbrs); 
	for (auto i=0; i<num_face_nbrs; i++) {
		send_sizes[i] = send_elems.RowSize(i); 
	}
	for (auto fn=0; fn<num_face_nbrs; fn++) {
		auto nbr_rank = mesh.GetFaceNbrRank(fn); 
		MPI_Isend(&send_sizes[fn], 1, MPI_INT, nbr_rank, 0, MPI_COMM_WORLD, &send_requests[fn]); 
		MPI_Irecv(&recv_elems.GetI()[fn], 1, MPI_INT, nbr_rank, 0, MPI_COMM_WORLD, &recv_requests[fn]); 
	}
	MPI_Waitall(num_face_nbrs, send_requests, statuses);
	MPI_Waitall(num_face_nbrs, recv_requests, statuses);

	// --- tell receiver what is being received --- 
	recv_elems.MakeJ(); 
	for (auto fn=0; fn<num_face_nbrs; fn++) {
		auto nbr_rank = mesh.GetFaceNbrRank(fn); 
		MPI_Isend(send_elems.GetRow(fn), send_elems.RowSize(fn), MPI_INT, nbr_rank, 0, MPI_COMM_WORLD, &send_requests[fn]); 
		MPI_Irecv(recv_elems.GetRow(fn), recv_elems.RowSize(fn), MPI_INT, nbr_rank, 0, MPI_COMM_WORLD, &recv_requests[fn]); 
	}
	MPI_Waitall(num_face_nbrs, send_requests, statuses);
	MPI_Waitall(num_face_nbrs, recv_requests, statuses);

	// --- create global to local map for face neighbors --- 
	mfem::Array<int> l2g_nbr(nbr_el*Nomega); 
	l2g_nbr = -1; 
	for (auto fn=0; fn<num_face_nbrs; fn++) {
		auto nbr_offset = face_nbr_offsets[fn]; 
		auto idx_offset = mesh.face_nbr_elements_offset[fn]; 
		auto *row = recv_elems.GetRow(fn); 
		for (auto i=0; i<recv_elems.RowSize(fn); i++) {
			auto a = row[i] % Nomega; 
			l2g_nbr[(idx_offset+i)*Nomega+a] = nbr_offset + row[i]; 
		}
	}
	std::unordered_map<int,int> g2l_nbr; 
	for (auto i=0; i<l2g_nbr.Size(); i++) {
		if (l2g_nbr[i] != -1) 
			g2l_nbr[l2g_nbr[i]] = i; 
	}

	// --- shift recieve to face neighbor ids ---
	for (auto fn=0; fn<num_face_nbrs; fn++) {
		auto *row = recv_elems.GetRow(fn); 
		for (auto i=0; i<recv_elems.RowSize(fn); i++) {
			auto g = row[i] + face_nbr_offsets[fn]; 
			row[i] = g2l_nbr[g] + Ne*Nomega; 
		}
	}

	ParPrint("l2g_nbr", [&l2g_nbr]() { l2g_nbr.Print(); }); 

	delete[] requests; delete[] statuses; 

	auto *send_elems_t = mfem::Transpose(send_elems); 


	// --- convert to igraph --- 
	igraph_t graph; 
	igraph_vector_int_t edges_igraph;
	igraph_vector_int_init_array(&edges_igraph, edges.GetData(), edges.Size());  
	// igraph_vector_int_view(&edges_view, edges.GetData(), edges.Size()); 
	igraph_create(&graph, &edges_igraph, 0, true); 

	// --- order DAG --- 
	igraph_bool_t is_dag; 
	igraph_is_dag(&graph, &is_dag); 
	if (!is_dag) { MFEM_ABORT("graph is not a dag"); }

	auto num_nodes = igraph_vcount(&graph); 
	mfem::Array<igraph_integer_t> degrees(num_nodes); 
	igraph_vector_int_t degrees_view; 
	igraph_vector_int_view(&degrees_view, degrees.GetData(), degrees.Size()); 
	auto mode = IGRAPH_OUT; 
	auto deg_mode = IGRAPH_IN; 
	IGRAPH_CHECK(igraph_degree(&graph, &degrees_view, igraph_vss_all(), deg_mode, 0));
	std::deque<int> local, send, recv; 
	for (int i=0; i<degrees.Size(); i++) {
		if (degrees[i] == 0) {
			if (i < Ne*Nomega) {
				local.push_back(i); 
			} else {
				recv.push_back(i); 
			}
		}
	}

	ParPrint("send/recv + parallel entry requirements", [&send_elems, &recv_elems, &recv]() {
		send_elems.Print();
		std::cout << std::endl; 
		recv_elems.Print(); 
		std::cout << std::endl; 
		for (const auto &i : recv) {
			std::cout << i << " "; 
		}
		std::cout << std::endl; 
	}); 
	// return 0; 

	mfem::Vector sums(nv * Nomega); 
	sums = 0.0; 
	double tmp = 0.0; 
	send_requests = new MPI_Request[send_elems.Size_of_connections()]; 
	recv_requests = new MPI_Request[recv_elems.Size_of_connections()]; 
	statuses = new MPI_Status[recv_elems.Size_of_connections()]; 
	for (auto fn=0; fn<num_face_nbrs; fn++) {
		mfem::Array<int> row; 
		recv_elems.GetRow(fn, row); 
		auto offset = recv_elems.GetI()[fn]; 
		auto nbr_rank = mesh.GetFaceNbrRank(fn); 
		for (int i=0; i<row.Size(); i++) {
			auto tag = row[i]; 
			MPI_Irecv(&sums(row[i]), 1, MPI_DOUBLE, nbr_rank, 0, MPI_COMM_WORLD, &recv_requests[offset+i]); 
		}
	}


	mfem::Array<int> order; 
	order.Reserve(num_nodes); 
	igraph_vector_int_t nbrs; 
	igraph_vector_int_init(&nbrs, 0); 
	auto nrecv = recv_elems.Size_of_connections(); 
	while (true) {
		while (not(local.empty())) {
			auto node = local.front();
			local.pop_front(); 

			order.Append(node); 
			sums(node) += 1; 
			degrees[node] = -1; 
			igraph_neighbors(&graph, &nbrs, node, mode);  
			auto num_nbrs = igraph_vector_int_size(&nbrs);
			for (auto i=0; i<num_nbrs; i++) {
				auto nbr = VECTOR(nbrs)[i]; 
				degrees[nbr] -= 1; 
				if (degrees[nbr] == 0) {
					if (nbr >= Ne*Nomega) {
						mfem::Array<int> row; 
						send_elems_t->GetRow(node, row); 
						for (auto i=0; i<row.Size(); i++) {
							auto fn = row[i]; 
							auto offset = send_elems_t->GetI()[node]; 
							auto nbr_rank = mesh.GetFaceNbrRank(fn); 
							MPI_Isend(&sums(node), 1, MPI_DOUBLE, nbr_rank, 0, MPI_COMM_WORLD, &send_requests[offset+i]); 
							MPI_Wait(&send_requests[offset+i], MPI_STATUS_IGNORE); 
						}
					} else {
						local.push_back(nbr); 						
					}
				}
			}
		}

		if (nrecv == 0) {
			break; 
		}

		int completed; 
		int indices[recv_elems.Size_of_connections()]; 
		MPI_Waitsome(recv_elems.Size_of_connections(), recv_requests, &completed, indices, MPI_STATUS_IGNORE); 
		nrecv -= completed; 
		for (int i=0; i<completed; i++) {
			auto idx = indices[i]; 
			auto node = recv_elems.GetJ()[idx]; 
			degrees[node] = -1; 
			igraph_neighbors(&graph, &nbrs, node, mode);  
			auto num_nbrs = igraph_vector_int_size(&nbrs);
			for (auto i=0; i<num_nbrs; i++) {
				auto nbr = VECTOR(nbrs)[i]; 
				degrees[nbr] -= 1; 
				if (degrees[nbr] == 0) {
					local.push_back(nbr); 
				}
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD); 
	ParPrint("sweep order", [&order, &quad, &nv, &Nomega, &mesh, &Ne]() { 
		for (auto a=0; a<Nomega; a++) {
			std::cout << a << ": "; 
			for (auto i : order) {
				if (i / Ne == a) {
					auto e = i % Ne; 
					std::cout << mesh.GetGlobalElementNum(e) << " "; 						
				}
			}
			std::cout << std::endl; 
		}
	});

	ParPrint("sums", [&sums]() { sums.Print(); }); 

#if 0

	// std::vector<igraph_integer_t> order(nv); 
	mfem::Array<igraph_integer_t> order(nv*Nomega); 
	igraph_vector_int_t order_view; 
	igraph_vector_int_view(&order_view, order.GetData(), order.Size()); 
	igraph_topological_sorting(&graph, &order_view, IGRAPH_OUT); 
	// for (int a=0; a<1; a++) {
	// 	std::cout << a << ": "; 
	// 	for (int i=0; i<order.size(); i++) {
	// 		if (order[i] / nv == a) {

	// 			// auto global = mesh.GetGlobalElementNum(order[i] % nv); 
	// 			std::cout << order[i] << " "; 
	// 		}
	// 	}		
	// 	std::cout << std::endl; 
	// }

	ParPrint("sweep order", [&order, &quad, nv]() { 
		for (auto a=0; a<Nomega; a++) {
			std::cout << a << ": "; 
			for (auto i : order) {
				if (i / nv == a) {
					std::cout << i % nv << " "; 
				}
			}
			std::cout << std::endl; 
		}
	});


	igraph_destroy(&graph); 

	mfem::L2_FECollection fec0(0, dim); 
	mfem::ParFiniteElementSpace fes0(&mesh, &fec0); 
	mfem::ParGridFunction gf(&fes0); 

	for (int e=0; e<Ne; e++) {
		gf[e] = mesh.GetGlobalElementNum(e); 
	}

	mfem::ParaViewDataCollection dc("solution", &mesh); 
	dc.RegisterField("id", &gf); 
	dc.Save(); 

	// --- form isotropic source term --- 
	mfem::LinearForm bform(&fes); 
	bform.AddDomainIntegrator(new mfem::DomainLFIntegrator(source)); 
	bform.Assemble(); 

	// --- build MIP DSA operator --- 
	mfem::BilinearForm Dform(&fes); 
	mfem::RatioCoefficient diffco(1./3, total); 
	double alpha = ComputeAlpha(quad, normal)/2;
	mfem::ConstantCoefficient alpha_c(alpha); 
	Dform.AddDomainIntegrator(new mfem::DiffusionIntegrator(diffco)); 
	Dform.AddDomainIntegrator(new mfem::MassIntegrator(absorption)); 
	Dform.AddInteriorFaceIntegrator(new MIPDiffusionIntegrator(diffco, -1, pow(fe_order+1,2), alpha)); 
	// Dform.AddBdrFaceIntegrator(new MIPDiffusionIntegrator(diffco, -1, pow(fe_order+1,2), alpha)); 
	Dform.AddBdrFaceIntegrator(new mfem::BoundaryMassIntegrator(alpha_c)); 
	Dform.Assemble(); 
	Dform.Finalize(); 

	mfem::BilinearForm Ms_form(&fes); 
	Ms_form.AddDomainIntegrator(new mfem::MassIntegrator(scattering)); 
	Ms_form.Assemble(); 
	Ms_form.Finalize(); 

	mfem::Vector scatsource(fes.GetVSize()); 
	mfem::KLUSolver solver(Dform.SpMat()); 

	mfem::Vector psi(psi_size); 
	psi = 0.0; 
	TransportVectorView psi_view(psi.GetData(), extents); 
	mfem::GridFunction phi(&fes), phi_old(&fes), delta_phi(&fes); 
	D.Mult(psi, phi); 
	phi_old = phi; 

	auto f2be = smesh.GetFaceToBdrElMap();
	for (int it=0; it<max_it; it++) {

		mfem::Array<int> faces, dofs, dofs2; 
		for (auto v : order) {
			const auto e = v % Ne; 
			const auto a = v / Ne; 
			const auto &Omega = quad.GetOmega(a); 
			const auto &el = *fes.GetFE(e); 
			auto &trans = *smesh.GetElementTransformation(e); 
			el2f->GetRow(e, faces); 
			fes.GetElementDofs(e, dofs); 

			// volumetric streaming term 
			mfem::DenseMatrix G; 
			mfem::VectorConstantCoefficient Q(Omega); 
			mfem::ConservativeConvectionIntegrator conv_int(Q, 1.0); 
			conv_int.AssembleElementMatrix(el, trans, G); 

			// collision term 
			mfem::DenseMatrix Mt; 
			mfem::MassIntegrator mt_int(total); 
			mt_int.AssembleElementMatrix(el, trans, Mt); 

			mfem::Vector rhs(dofs.Size()); 
			bform.GetSubVector(dofs, rhs); 

			// scattering 
			mfem::DenseMatrix Ms; 
			mfem::MassIntegrator ms_int(scattering); 
			ms_int.AssembleElementMatrix(el, trans, Ms); 
			mfem::Vector phi_local(dofs.Size()); 
			phi.GetSubVector(dofs, phi_local); 
			Ms.AddMult(phi_local, rhs, 1./4/M_PI); 

			mfem::DenseMatrix F(dofs.Size());
			F = 0.0; 
			double alpha = 1.0; 
			double beta = 0.5; 
			for (int fidx=0; fidx<faces.Size(); fidx++) {
				auto &face_trans = *smesh.GetFaceElementTransformations(faces[fidx]); 
				mfem::DGTraceIntegrator upw_int(Q, alpha, beta); 
				mfem::DenseMatrix f; 
				const auto &el1 = *fes.GetFE(face_trans.Elem1No); 
				const auto *el2 = &el1; 
				if (face_trans.Elem2No >= 0) {
					el2 = fes.GetFE(face_trans.Elem2No); 
				}
				int ndofs1 = el1.GetDof(); 
				int ndofs2 = el2->GetDof(); 
				upw_int.AssembleFaceMatrix(el1, *el2, face_trans, f); 
				mfem::DenseMatrix f11, f12; 
				int ep = (e==face_trans.Elem1No) ? face_trans.Elem2No : face_trans.Elem1No;
				if (e == face_trans.Elem1No) {
					f.GetSubMatrix(0, ndofs1, 0, ndofs1, f11); 
					f.GetSubMatrix(0, ndofs1, ndofs1, f.Width(), f12); 				
				} else {
					f.GetSubMatrix(ndofs1, f.Height(), ndofs1, f.Width(), f11); 
					f.GetSubMatrix(ndofs1, f.Height(), 0, ndofs1, f12); 
				}
				F += f11; 
				if (ep>=0) {
					fes.GetElementDofs(ep, dofs2); 
					mfem::Vector psi2(dofs2.Size()); 
					for (int i=0; i<dofs2.Size(); i++) { psi2(i) = psi_view(0, a, dofs2[i]); }
					f12.AddMult(psi2, rhs, -1.0); 									
				} else {
					auto be = f2be[face_trans.ElementNo]; 
					auto &bdr_face_trans = *smesh.GetBdrFaceTransformations(be); 
					mfem::ConstantCoefficient zero(1.0); 
					mfem::BoundaryFlowIntegrator bdr_flow(inflow, Q, alpha, beta);
					mfem::Vector elvec; 
					bdr_flow.AssembleRHSElementVect(el, bdr_face_trans, elvec);  
					rhs -= elvec; 
				}
			} 

			mfem::DenseMatrix A(G); 
			A += F; 
			A += Mt; 

			mfem::LinearSolve(A, rhs.GetData()); 
			for (int i=0; i<dofs.Size(); i++) { psi_view(0, a, dofs[i]) = rhs(i); }
		}

		D.Mult(psi, phi); 

		if (dsa) {
			delta_phi = phi; 
			delta_phi -= phi_old;
			Ms_form.Mult(delta_phi, scatsource);  

			solver.Mult(scatsource, delta_phi); 
			phi += delta_phi; 			
		}
		phi_old -= phi; 
		double norm = phi_old.Norml2(); 
		std::cout << "it = " << it << ", norm = " << norm << std::endl; 
		if (norm < tolerance) {
			break; 
		}
		phi_old = phi; 
	}

	if (viz) {
		mfem::ParaViewDataCollection dc("solution", &smesh); 
		dc.RegisterField("phi", &phi); 
		dc.Save(); 		
	}

	delete el2f; 
#endif
}