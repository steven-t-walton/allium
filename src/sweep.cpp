#include "sweep.hpp"
#include <deque>
#include "config.hpp"

// #define INFLOW_IN_SWEEP

InverseAdvectionOperator::InverseAdvectionOperator(mfem::ParFiniteElementSpace &_fes, const AngularQuadrature &_quad, 
	const TransportVectorExtents &_psi_ext, mfem::Coefficient &_total, mfem::Coefficient &_inflow)
	: fes(_fes), mesh(*_fes.GetParMesh()), quad(_quad), psi_ext(_psi_ext), total(_total), inflow(_inflow)
{
	// set operator sizes to size of psi 
	height = width = TotalExtent(psi_ext);

	// ensure exchanged data called 
	mesh.ExchangeFaceNbrData(); 
	fes.ExchangeFaceNbrData(); 

	// --- create data structures not provided by mfem::ParMesh ---
	// build local to global element map for elements straddling parallel faces 
	// maps index into the ghost element id to its global element number
	const auto dim = mesh.Dimension(); 
	const auto Ne = mesh.GetNE(); 
	const auto Ndof = fes.GetVSize(); 
	const auto Nomega = quad.Size(); 
	const auto nbr_el = mesh.GetNFaceNeighborElements(); 
	const auto num_face_nbrs = mesh.GetNFaceNeighbors(); 
	const mfem::Table &send_face_nbr_elements = mesh.send_face_nbr_elements; 
	// elements to be received by neighbor processors 
	// ids are the local id for that processor
	mfem::Table recv_face_nbr_elements(num_face_nbrs); 

	// exchange send/recv tables 
	MPI_Request *requests = new MPI_Request[2*num_face_nbrs];
	MPI_Request *send_requests = requests;
	MPI_Request *recv_requests = requests + num_face_nbrs;
	MPI_Status  *statuses = new MPI_Status[num_face_nbrs];

	// exchange how many messages will be sent 
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
	// and dofs [dof_offsets[0], dof_offsets[1])
	mfem::Array<HYPRE_BigInt> *ptr[2] = {&mesh_offsets, &dof_offsets}; 
	HYPRE_BigInt N_hypre[2]; 
	N_hypre[0] = Ne; N_hypre[1] = Ndof; 
	mesh.GenerateOffsets(2, N_hypre, ptr); 

	// --- swap offsets with parallel neighbors --- 
	// from this info can compute global element number of ghost cells 
	mesh_face_nbr_offsets.SetSize(num_face_nbrs); 
	for (auto fn=0; fn<num_face_nbrs; fn++) {
		auto nbr_rank = mesh.GetFaceNbrRank(fn); 
		MPI_Isend(&mesh_offsets[0], 1, HYPRE_MPI_BIG_INT, nbr_rank, 0, MPI_COMM_WORLD, &send_requests[fn]); 
		MPI_Irecv(&mesh_face_nbr_offsets[fn], 1, HYPRE_MPI_BIG_INT, nbr_rank, 0, MPI_COMM_WORLD, &recv_requests[fn]); 
	}
	MPI_Waitall(num_face_nbrs, send_requests, statuses); 
	MPI_Waitall(num_face_nbrs, recv_requests, statuses); 

	mfem::Array<HYPRE_BigInt> dof_face_nbr_offsets(num_face_nbrs); 
	for (auto fn=0; fn<num_face_nbrs; fn++) {
		auto nbr_rank = mesh.GetFaceNbrRank(fn); 
		MPI_Isend(&dof_offsets[0], 1, HYPRE_MPI_BIG_INT, nbr_rank, 0, MPI_COMM_WORLD, &send_requests[fn]); 
		MPI_Irecv(&dof_face_nbr_offsets[fn], 1, HYPRE_MPI_BIG_INT, nbr_rank, 0, MPI_COMM_WORLD, &recv_requests[fn]); 
	}
	MPI_Waitall(num_face_nbrs, send_requests, statuses); 
	MPI_Waitall(num_face_nbrs, recv_requests, statuses); 

	// --- build the actual map with "inverse" via unorder_map --- 
	fnbr_to_global.SetSize(nbr_el); 
	for (auto fn=0; fn<num_face_nbrs; fn++) {
		auto idx_offset = mesh.face_nbr_elements_offset[fn]; 
		auto *row = recv_face_nbr_elements.GetRow(fn); 
		for (auto i=0; i<recv_face_nbr_elements.RowSize(fn); i++) {
			fnbr_to_global[idx_offset + i] = row[i] + mesh_face_nbr_offsets[fn]; // local ghost id -> global element 
			global_to_fnbr[row[i] + mesh_face_nbr_offsets[fn]] = idx_offset + i; // global element -> local ghost id 
		}
	}

	// --- store face neighbor elements to face neighbor number --- 
	// maps ghost id to which processor owns the data 
	fnbr_to_fn.SetSize(nbr_el); 
	for (auto fn=0; fn<recv_face_nbr_elements.Size(); fn++) {
		auto idx_offset = mesh.face_nbr_elements_offset[fn]; 
		const auto *row = recv_face_nbr_elements.GetRow(fn); 
		for (auto i=0; i<recv_face_nbr_elements.RowSize(fn); i++) {
			fnbr_to_fn[idx_offset + i] = fn; 
		}
	}

	// --- map face id to boundary element id --- 
	// required in sweep for integrator to properly use the bdr_attribute for 
	// the inflow condition 
	for (auto i=0; i<mesh.GetNBE(); i++) {
		face_to_bdr_el[mesh.GetBdrElementFaceIndex(i)] = i; 
	}


	// --- create graph --- 
	const auto &el2el = mesh.ElementToElementTable(); // element to element connectivity 
	// faces to elements, has 2 elements for interior faces, 1 for boundary 
	// includes parallel shared faces to ghost/face neighbor elements 
	// handles 1/2/3D 
	std::unique_ptr<const mfem::Table> face_to_element(mesh.GetFaceToAllElementTable()); 
	// "invert" to go from element id to the list of faces associated with that element 
	// needed for sweep
	element_to_face = std::unique_ptr<const mfem::Table>(mfem::Transpose(*face_to_element)); 

	// loop through faces, add directed edges based on Omega.n 
	// normal is assumed to be piecewise constant! Works for linear meshes only 
	mfem::Array<igraph_integer_t> edges; // list of edges 
	edges.Reserve(face_to_element->Size_of_connections()*Nomega*2);
	mfem::Vector normal(dim); // normal vector 
	for (int r=0; r<face_to_element->Size(); r++) {
		// get view into row of table 
		auto row = std::span(face_to_element->GetRow(r), face_to_element->RowSize(r)); 
		if (row.size()==2) { // two faces => interior face 
			// compute normal 
			auto ref_geom = mesh.GetFaceGeometry(r);
			const auto &int_rule = mfem::IntRules.Get(ref_geom, 1); 
			auto *trans = mesh.GetFaceElementTransformations(r); 
			trans->SetAllIntPoints(&int_rule[0]); 
			if (dim==1) {
				normal(0) = 2*trans->GetElement1IntPoint().x - 1.0;
			} else {
				mfem::CalcOrtho(trans->Jacobian(), normal); 				
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

	// --- convert to igraph --- 
	igraph_vector_int_t edges_igraph;
	// view into MFEM array data 
	// igraph does not own the data 
	igraph_vector_int_view(&edges_igraph, edges.GetData(), edges.Size()); 
	igraph_create(&graph, &edges_igraph, 0, true); 

	// ensure DAG 
	igraph_bool_t is_dag; 
	igraph_is_dag(&graph, &is_dag); 
	if (!is_dag) { MFEM_ABORT("graph is not a dag"); }

	// map processor number to face number index 
	for (auto fn=0; fn<num_face_nbrs; fn++) {
		proc_to_fn[mesh.GetFaceNbrRank(fn)] = fn; 
	}

	delete[] requests; delete[] statuses; 

#ifdef SWEEP_PREASSEMBLE_LOCAL_MATRICES
	mass_matrices.SetSize(fes.GetNE()); 
	grad_matrices.SetSize(fes.GetNE()*dim); 
	auto grad_mat_view = Kokkos::mdspan(grad_matrices.GetData(), dim, fes.GetNE()); 
	mfem::MassIntegrator mi(total); 
	for (auto e=0; e<fes.GetNE(); e++) {
		const auto &fe = *fes.GetFE(e); 
		auto &trans = *mesh.GetElementTransformation(e);
		mass_matrices[e] = new mfem::DenseMatrix; 
		mi.AssembleElementMatrix(fe, trans, *mass_matrices[e]); 

		mfem::Vector Omega(dim); 
		for (auto d=0; d<dim; d++) {
			grad_mat_view(d,e) = new mfem::DenseMatrix; 
			Omega = 0.0; 
			Omega(d) = 1.0; 
			mfem::VectorConstantCoefficient Q(Omega); 
			mfem::ConservativeConvectionIntegrator conv_int(Q, 1.0); 
			conv_int.AssembleElementMatrix(fe, trans, *grad_mat_view(d,e)); 
		}
	}	

	face_matrices.SetSize(4*mesh.GetNumFaces()); 
	for (auto i=0; i<face_matrices.Size(); i++) {
		face_matrices[i] = new mfem::DenseMatrix; 
	}
	auto face_mat_view = Kokkos::mdspan(face_matrices.GetData(), mesh.GetNumFaces(), 2, 2); 
	FaceMassMatricesIntegrator fmi; 
	mfem::DenseMatrix elmat; 
	for (auto f=0; f<mesh.GetNumFaces(); f++) {
		const auto info = mesh.GetFaceInformation(f); 
		mfem::FaceElementTransformations *face_trans; 
		if (info.IsShared()) {
			face_trans = mesh.GetSharedFaceTransformationsByLocalIndex(f, true); 
		} else {
			face_trans = mesh.GetFaceElementTransformations(f); 
		}
		const auto &el1 = *fes.GetFE(face_trans->Elem1No); 
		const auto *el2 = &el1; 
		if (face_trans->Elem2No >= 0) {
			el2 = fes.GetFE(face_trans->Elem2No); 
		}
		fmi.AssembleFaceMatrix(el1, *el2, *face_trans, elmat);
		const auto dof1 = el1.GetDof(); 
		elmat.GetSubMatrix(0,dof1,0,dof1, *face_mat_view(f,0,0)); 
		if (face_trans->Elem2No >= 0) {
			const auto dof2 = el2->GetDof(); 
			elmat.GetSubMatrix(0,dof1,dof1,dof1+dof2, *face_mat_view(f,0,1)); 
			elmat.GetSubMatrix(dof1,dof1+dof2,0,dof1, *face_mat_view(f,1,0)); 
			elmat.GetSubMatrix(dof1,dof1+dof2,dof1,dof1+dof2, *face_mat_view(f,1,1)); 			
		}
	}
#endif
}

void InverseAdvectionOperator::SetSendBufferSize(int s) 
{
#ifdef SWEEP_BUFFER_SENDS
	send_buffer_size = s; 
#else 
	if (fes.GetMyRank() == 0) 
		MFEM_WARNING("send buffer is disabled in sweep"); 
#endif
}

InverseAdvectionOperator::~InverseAdvectionOperator()
{
	igraph_destroy(&graph); 
	for (auto &ptr : mass_matrices) { delete ptr; }
	for (auto &ptr : grad_matrices) { delete ptr; }
	for (auto &ptr : face_matrices) { delete ptr; }
}

void InverseAdvectionOperator::WriteGraphToDot(std::string prefix) const 
{
	auto rank = mesh.GetMyRank(); 
	FILE *file = fopen(mfem::MakeParFilename(prefix + ".", rank, ".dot").c_str(), "w"); 
	igraph_write_graph_dot(&graph, file); 
	fclose(file); 	
}

void InverseAdvectionOperator::Mult(const mfem::Vector &source, mfem::Vector &psi) const 
{
	assert(source.Size() == Width()); 
	assert(psi.Size() == Height()); 

	const auto num_face_nbrs = mesh.GetNFaceNeighbors(); 
	const auto dim = mesh.Dimension(); 
	const auto Ne = mesh.GetNE(); 
	const auto Nomega = quad.Size(); 
	const auto Ndof_fnbr = fes.GetFaceNbrVSize(); 
	// place to store face ids/element, dofs/element, and dofs of neighboring elements 
	mfem::Array<int> faces, dofs, dofs2; 

	// igraph vectors to store neighbors in graph traversal 
	igraph_vector_int_t nbrs, nbr_nbrs; 
	igraph_vector_int_init(&nbrs, 0); 
	igraph_vector_int_init(&nbr_nbrs, 0); 

	// mdspan views into data 
	TransportVectorView psi_view(psi.GetData(), psi_ext); 
	TransportVectorView source_view(source.GetData(), psi_ext); 

#ifdef SWEEP_PREASSEMBLE_LOCAL_MATRICES
	auto grad_mat_view = Kokkos::mdspan(grad_matrices.GetData(), dim, fes.GetNE()); 
	auto face_mat_view = Kokkos::mdspan(face_matrices.GetData(), mesh.GetNumFaces(), 2, 2); 
#endif

	// allocate face neighbor data 
	TransportVectorExtents psi_fnbr_ext(1, Nomega, Ndof_fnbr);
	psi_fnbr.SetSize(TotalExtent(psi_fnbr_ext)); 
	psi_fnbr = 0.0; 
	TransportVectorView psi_fnbr_view(psi_fnbr.GetData(), psi_fnbr_ext); 

	// --- determine roots of graph --- 
	const auto num_nodes = igraph_vcount(&graph); 
	mfem::Array<igraph_integer_t> degrees(num_nodes); 
	igraph_vector_int_t degrees_view; 
	igraph_vector_int_view(&degrees_view, degrees.GetData(), degrees.Size()); 
	// count edges incoming to each vertex 
	igraph_degree(&graph, &degrees_view, igraph_vss_all(), IGRAPH_IN, 0);
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

#ifdef SWEEP_BUFFER_SENDS
	mfem::Array<mfem::Connection> send_list; 
	send_list.Reserve(send_buffer_size); 
	auto message_count = 0; 
#endif
	// --- do sweep --- 
	// vertices traversed, only count owned elements 
	int nsweep = 0; 
	while (nsweep < Ne*Nomega) {
		// --- sweep on processor-local domain --- 
		while (not(local.empty())) {
			nsweep++; //increment sweep counter to know when to stop 
			// get first element of queue 
			const auto node = local.front();
			local.pop_front(); 

			// --- do work --- 
			// deconstruct angle/space id
			const auto a = node % Nomega; 
			const auto e = node / Nomega; 
			const auto &Omega = quad.GetOmega(a); 
			mfem::VectorConstantCoefficient Q(Omega); 
			const auto &el = *fes.GetFE(e); 
			auto &trans = *mesh.GetElementTransformation(e); 
			fes.GetElementDofs(e, dofs); 

			// -- assemble volumetric terms --
			// volumetric streaming term 
#ifdef SWEEP_PREASSEMBLE_LOCAL_MATRICES
			mfem::DenseMatrix G(dofs.Size()); 
			G = 0.0; 
			for (auto d=0; d<dim; d++) {
				G.Add(Omega(d), *grad_mat_view(d,e)); 
			}
#else 
			mfem::DenseMatrix G; 
			mfem::ConservativeConvectionIntegrator conv_int(Q, 1.0); 
			conv_int.AssembleElementMatrix(el, trans, G); 
#endif

			// collision term 
#ifdef SWEEP_PREASSEMBLE_LOCAL_MATRICES
			mfem::DenseMatrix &Mt = *mass_matrices[e]; 
#else
			mfem::DenseMatrix Mt; 
			mfem::MassIntegrator mt_int(total); 
			mt_int.AssembleElementMatrix(el, trans, Mt); 
#endif

			// get source term 
			mfem::Vector rhs(dofs.Size()); 
			for (auto i=0; i<dofs.Size(); i++) { rhs[i] = source_view(0,a,dofs[i]); }

#ifdef SWEEP_PREASSEMBLE_LOCAL_MATRICES
			mfem::Vector nor(dim); 
			element_to_face->GetRow(e, faces);
			mfem::DenseMatrix F(dofs.Size()); 
			F = 0.0; 
			for (auto f : faces) {
				const auto info = mesh.GetFaceInformation(f); 
				mfem::FaceElementTransformations *face_trans; 
				if (info.IsShared()) {
					face_trans = mesh.GetSharedFaceTransformationsByLocalIndex(f, true); 
				} else {
					face_trans = mesh.GetFaceElementTransformations(f); 
				}
				bool keep_order = e == face_trans->Elem1No; 
				const auto ep = (keep_order) ? face_trans->Elem2No : face_trans->Elem1No; 
				const auto &ref_cent = mfem::Geometries.GetCenter(face_trans->GetGeometryType()); 
				face_trans->SetAllIntPoints(&ref_cent); 
				if (dim==1) {
					nor(0) = 2*face_trans->GetElement1IntPoint().x - 1.0;
				} else {
					mfem::CalcOrtho(face_trans->Jacobian(), nor); 				
				}
				// normalize, ensure outward normal 
				nor.Set((keep_order ? 1.0 : -1.0)/nor.Norml2(), nor); 
				const double dot = Omega * nor; 
				const auto idx = keep_order ? 0 : 1; 
				const auto nbr_idx = (idx + 1) % 2; 

				// outflow 
				if (dot > 0) {
					const auto &elmat = *face_mat_view(f, idx, idx); 
					F.Add(dot, elmat); 
				}

				// inflow 
				else {
					// interior face 
					if (ep >= 0) {
						const auto &elmat = *face_mat_view(f, idx, nbr_idx); 
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
						elmat.AddMult(psi2, rhs, -dot); 	
					}
					#ifdef INFLOW_IN_SWEEP
					else {
						auto be = face_to_bdr_el.at(face_trans->ElementNo); 
						auto &bdr_face_trans = *mesh.GetBdrFaceTransformations(be); 
						mfem::BoundaryFlowIntegrator bdr_flow(inflow, Q, 1.0, 0.5);
						mfem::Vector elvec; 
						bdr_flow.AssembleRHSElementVect(el, bdr_face_trans, elvec);  
						rhs -= elvec; 
					}
					#endif
				}
			}
#else 
			// -- do face terms -- 
			// get faces associated with mesh element e 
			// face are UNIQUE => normal is not always outward 
			element_to_face->GetRow(e, faces);
			mfem::DenseMatrix F(dofs.Size());
			F = 0.0; 
			double alpha = 1.0; 
			double beta = 0.5; 
			for (int fidx=0; fidx<faces.Size(); fidx++) {
				const auto info = mesh.GetFaceInformation(faces[fidx]); 
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
				} 
				#ifdef INFLOW_IN_SWEEP
				else {
					auto be = face_to_bdr_el.at(face_trans->ElementNo); 
					auto &bdr_face_trans = *mesh.GetBdrFaceTransformations(be); 
					mfem::BoundaryFlowIntegrator bdr_flow(inflow, Q, alpha, beta);
					mfem::Vector elvec; 
					bdr_flow.AssembleRHSElementVect(el, bdr_face_trans, elvec);  
					rhs -= elvec; 
				}
				#endif
			}  
#endif

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
						auto fn = fnbr_to_fn[nbr_id / Nomega]; 
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
			for (const auto &fn : send_fn_set) {
			#ifdef SWEEP_BUFFER_SENDS
				send_list.Append({fn, node}); 
			#else 
				auto nbr_rank = mesh.GetFaceNbrRank(fn); 
				MPI_Request send_request; 
				MPI_Isend(rhs.GetData(), dofs.Size(), MPI_DOUBLE, nbr_rank, node, MPI_COMM_WORLD, &send_request); 
				MPI_Wait(&send_request, MPI_STATUS_IGNORE); 
			#endif
			}

		#ifdef SWEEP_BUFFER_SENDS
			if (send_list.Size() >= send_buffer_size or local.empty()) {
				send_list.Sort(); send_list.Unique(); 
				mfem::Table send_table(num_face_nbrs, send_list); 
				for (auto fn=0; fn<num_face_nbrs; fn++) {
					if (send_table.RowSize(fn) == 0) continue; 
					const auto nbr_rank = mesh.GetFaceNbrRank(fn); 
					auto buffer_size = 0;
					auto nodes = std::span<const int>(send_table.GetRow(fn), send_table.RowSize(fn)); 
					for (auto n : nodes) {
						fes.GetElementDofs(n / Nomega, dofs); 
						buffer_size += dofs.Size(); 
					}

					double buffer[buffer_size]; 
					auto idx = 0;
					for (auto n : nodes) {
						const auto e = n / Nomega; 
						const auto a = n % Nomega; 
						fes.GetElementDofs(e, dofs); 
						for (auto i=0; i<dofs.Size(); i++) {
							buffer[idx++] = psi_view(0,a,dofs[i]); 
						}
					}

					MPI_Request request[2]; 
					MPI_Isend(send_table.GetRow(fn), send_table.RowSize(fn), MPI_INT, nbr_rank, message_count++, 
						MPI_COMM_WORLD, &request[0]); 
					MPI_Isend(buffer, buffer_size, MPI_DOUBLE, nbr_rank, message_count++, MPI_COMM_WORLD, &request[1]); 
					MPI_Waitall(2, request, MPI_STATUS_IGNORE); 
				}
				send_list.SetSize(0); // set size to 0, keeps capacity from reserve call above 
			}
		#endif
		}

		// --- receive data from other processors --- 
		// loop until no messages remain 
		while (true) {
		#ifdef SWEEP_BUFFER_SENDS
			int avail; 
			MPI_Status status[2]; 
			MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &avail, &status[0]);
			if (avail==0) break; 

			const auto tag = status[0].MPI_TAG; 
			const auto source = status[0].MPI_SOURCE; 
			MPI_Probe(source, (tag % 2 == 0) ? tag + 1 : tag - 1, MPI_COMM_WORLD, &status[1]); 

			if (tag % 2 == 1) {
				std::swap(status[0], status[1]); 
			}

			int node_count, data_count; 
			MPI_Get_count(&status[0], MPI_INT, &node_count); 
			MPI_Get_count(&status[1], MPI_DOUBLE, &data_count); 

			int node_buffer[node_count]; 
			double data_buffer[data_count]; 

			MPI_Request request[2]; 
			assert(status[0].MPI_SOURCE == status[1].MPI_SOURCE); 
			MPI_Irecv(&node_buffer, node_count, MPI_INT, source, status[0].MPI_TAG, MPI_COMM_WORLD, &request[0]); 
			MPI_Irecv(&data_buffer, data_count, MPI_DOUBLE, source, status[1].MPI_TAG, MPI_COMM_WORLD, &request[1]); 
			MPI_Waitall(2, request, MPI_STATUSES_IGNORE); 

			const auto fn = proc_to_fn.at(source); 
			auto node_view = std::span<const int>(node_buffer, node_count); 
			mfem::Array<int> I(node_count+1); 
			I[0] = 0; 
			for (int n=0; n<node_count; n++) {
				const auto node = node_view[n]; 
				// convert to global id so that it can be mapped to local fnbr id 
				const auto global = mesh_face_nbr_offsets[fn] * Nomega + node;  
				// convert tag (global idx) to a local index 
				const auto e = global / Nomega; 
				const auto a = global % Nomega; 
				const auto local_e = global_to_fnbr.at(e) + Ne; 
				const auto local_id = local_e*Nomega + a;		
				fes.GetFaceNbrElementVDofs(local_e-Ne, dofs); 
				I[n+1] = dofs.Size(); 
			}	
			I.PartialSum(); 

			for (auto n=0; n<node_count; n++) {
				const auto node = node_view[n]; 
				// convert to global id so that it can be mapped to local fnbr id 
				const auto global = mesh_face_nbr_offsets[fn] * Nomega + node;  
				// convert tag (global idx) to a local index 
				const auto e = global / Nomega; 
				const auto a = global % Nomega; 
				const auto local_e = global_to_fnbr.at(e) + Ne; 
				const auto local_id = local_e*Nomega + a;		
				fes.GetFaceNbrElementVDofs(local_e-Ne, dofs); 	
				for (auto i=0; i<dofs.Size(); i++) { psi_fnbr_view(0,a,dofs[i]) = data_buffer[I[n] + i]; }

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

		#else
			// probe to see if message available 
			int avail; 
			MPI_Status status; 
			MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &avail, &status); 
			// exit if no messages available 
			if (avail==0) break; 

			// iprobe has same status as if MPI_Recv was called 
			// extract info to reproduce call to MPI_Recv 
			// also used to find the right buffer to place data in 
			auto tag = status.MPI_TAG; // local node id to source processor 
			auto source = status.MPI_SOURCE; 
			auto fn = proc_to_fn.at(source);
			// convert to global id so that it can be mapped to local fnbr id 
			auto global = mesh_face_nbr_offsets[fn] * Nomega + tag;  
			// convert tag (global idx) to a local index 
			auto e = global / Nomega; 
			auto a = global % Nomega; 
			auto local_e = global_to_fnbr.at(e) + Ne; 
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
		#endif
		}
	}
	// clean up data 
	igraph_vector_int_destroy(&nbrs); 
	igraph_vector_int_destroy(&nbr_nbrs); 

	// use barrier to avoid tag clash? 
	MPI_Barrier(MPI_COMM_WORLD); 
}

void FormTransportSource(mfem::ParFiniteElementSpace &fes, AngularQuadrature &quad, 
	const TransportVectorExtents &psi_ext,
	std::function<double(const mfem::Vector &x, const mfem::Vector &Omega)> source_func, 
	std::function<double(const mfem::Vector &x, const mfem::Vector &Omega)> inflow_func, 
	mfem::Vector &source_vec)
{
	const auto psi_size = TotalExtent(psi_ext); 
	source_vec.SetSize(psi_size);
	source_vec = 0.0;  
	TransportVectorView source_vec_view(source_vec.GetData(), psi_ext); 
	for (auto a=0; a<quad.Size(); a++) {
		const auto &Omega = quad.GetOmega(a); 
		auto source_for_angle = [&Omega, &source_func](const mfem::Vector &x) {
			return source_func(x,Omega); 
		};
		mfem::ParLinearForm bform(&fes); 
		mfem::FunctionCoefficient source_coef(source_for_angle); 
		bform.AddDomainIntegrator(new mfem::DomainLFIntegrator(source_coef)); 
		#ifndef INFLOW_IN_SWEEP
		auto inflow_for_angle = [&Omega, &inflow_func](const mfem::Vector &x) {
			return inflow_func(x,Omega); 
		};
		mfem::FunctionCoefficient inflow_coef(inflow_for_angle); 
		mfem::VectorConstantCoefficient Q(Omega); 
		bform.AddBdrFaceIntegrator(new mfem::BoundaryFlowIntegrator(inflow_coef, Q, -1.0, -0.5));
		#endif
		bform.Assemble(); 
		for (int i=0; i<bform.Size(); i++) {
			source_vec_view(0,a,i) = bform[i]; 
		}
	}
}

void FormTransportSource(mfem::ParFiniteElementSpace &fes, AngularQuadrature &quad, 
	PhaseSpaceCoefficient &source_coef, PhaseSpaceCoefficient &inflow_coef, 
	TransportVectorView source_view) 
{
	for (auto a=0; a<quad.Size(); a++) {
		const auto &Omega = quad.GetOmega(a); 
		source_coef.SetState(Omega); inflow_coef.SetState(Omega); 

		mfem::ParLinearForm bform(&fes); 
		bform.AddDomainIntegrator(new mfem::DomainLFIntegrator(source_coef)); 
		#ifndef INFLOW_IN_SWEEP
		mfem::VectorConstantCoefficient Q(Omega); 
		bform.AddBdrFaceIntegrator(new mfem::BoundaryFlowIntegrator(inflow_coef, Q, -1.0, -0.5));
		#endif
		bform.Assemble(); 
		for (int i=0; i<bform.Size(); i++) {
			source_view(0,a,i) = bform[i]; 
		}
	}
}

void FormTransportSource(mfem::ParFiniteElementSpace &fes, AngularQuadrature &quad, 
	const TransportVectorExtents &psi_ext,
	std::function<double(double x, double y, double z, double mu, double eta, double xi)> source_func, 
	std::function<double(double x, double y, double z, double mu, double eta, double xi)> inflow_func, 
	mfem::Vector &source_vec)
{
	const auto dim = fes.GetMesh()->Dimension(); 
	auto source = [&dim, &source_func](const mfem::Vector &x, const mfem::Vector &Omega) {
		double X[3]; 
		double O[3]; 
		for (auto d=0; d<x.Size(); d++) { X[d] = x(d); }
		for (auto d=0; d<Omega.Size(); d++) { O[d] = Omega(d); }			
		return source_func(X[0], X[1], X[2], O[0], O[1], O[2]); 
	};
	auto inflow = [&dim, &inflow_func](const mfem::Vector &x, const mfem::Vector &Omega) {
		double X[3]; 
		double O[3]; 
		for (auto d=0; d<x.Size(); d++) { X[d] = x(d); }
		for (auto d=0; d<Omega.Size(); d++) { O[d] = Omega(d); }			
		return inflow_func(X[0], X[1], X[2], O[0], O[1], O[2]); 
	};
	FormTransportSource(fes, quad, psi_ext, source, inflow, source_vec); 
}

void FormTransportSource(mfem::ParFiniteElementSpace &fes, AngularQuadrature &quad, 
	const TransportVectorExtents &psi_ext, mfem::Coefficient &source, mfem::Coefficient &inflow, mfem::Vector &source_vec)
{
	const auto psi_size = TotalExtent(psi_ext); 
	source_vec.SetSize(psi_size); 
	source_vec = 0.0;  
	TransportVectorView source_vec_view(source_vec.GetData(), psi_ext); 

	mfem::ParLinearForm bform(&fes); 
	bform.AddDomainIntegrator(new mfem::DomainLFIntegrator(source)); 
	bform.Assemble(); 

	for (auto a=0; a<quad.Size(); a++) {
		const auto &Omega = quad.GetOmega(a); 
		mfem::VectorConstantCoefficient Q(Omega); 
		mfem::ParLinearForm bform2(&fes); 
		bform2.AddBdrFaceIntegrator(new mfem::BoundaryFlowIntegrator(inflow, Q, -1.0, -0.5)); 
		bform2.Assemble(); 
		for (auto i=0; i<fes.GetVSize(); i++) {
			#ifdef INFLOW_IN_SWEEP
			source_vec_view(0,a,i) = bform[i]; 
			#else
			source_vec_view(0,a,i) = bform[i] + bform2[i]; 
			#endif
		}
	}
}

void FaceMassMatricesIntegrator::AssembleFaceMatrix(const mfem::FiniteElement &el1, const mfem::FiniteElement &el2, 
		mfem::FaceElementTransformations &trans, mfem::DenseMatrix &elmat)
{
	const auto dof1 = el1.GetDof(); 
	shape1.SetSize(dof1); 
	int dof2 = 0; 
	if (trans.Elem2No >= 0) {
		dof2 = el2.GetDof(); 
		shape2.SetSize(dof2); 
	}	
	const auto dof = dof1 + dof2; 
	elmat.SetSize(dof);
	elmat = 0.0; 

	const auto *ir = IntRule;
	if (ir == NULL) {
		int order; 
		if (trans.Elem2No >= 0) {
			order = 2*std::max(el1.GetOrder(), el2.GetOrder()); 
		}
		else {
			order = 2*el1.GetOrder(); 
		}
		ir = &mfem::IntRules.Get(trans.GetGeometryType(), order); 
	} 

	for (auto n=0; n<ir->GetNPoints(); n++) {
		const auto &ip = ir->IntPoint(n); 
		trans.SetAllIntPoints(&ip); 
		double w = ip.weight * trans.Weight(); 

		const auto &eip1 = trans.GetElement1IntPoint(); 
		el1.CalcShape(eip1, shape1); 
		for (auto i=0; i<dof1; i++) {
			for (auto j=0; j<dof1; j++) {
				elmat(i,j) += shape1(i) * shape1(j) * w; 
			}
		}

		if (trans.Elem2No >= 0) {
			const auto &eip2 = trans.GetElement2IntPoint(); 
			el2.CalcShape(eip2, shape2); 
			for (auto i=0; i<dof1; i++) {
				for (auto j=0; j<dof2; j++) {
					double val = shape1(i) * shape2(j) * w; 
					elmat(i,j+dof1) += val; 
					elmat(j+dof1,i) += val; 
				}
			}

			for (auto i=0; i<dof2; i++) {
				for (auto j=0; j<dof2; j++) {
					elmat(i+dof1,j+dof1) += shape2(i) * shape2(j) * w; 
				}
			}
		}
	}
}