#include "sweep.hpp"
#include <deque>

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
	mfem::Array<HYPRE_BigInt> *ptr = &mesh_offsets; 
	HYPRE_BigInt Ne_hypre = Ne; 
	mesh.GenerateOffsets(1, &Ne_hypre, &ptr); 

	// --- swap offsets with parallel neighbors --- 
	// from this info can compute global element number of ghost cells 
	mfem::Array<HYPRE_BigInt> mesh_face_nbr_offsets(num_face_nbrs); 
	for (auto fn=0; fn<num_face_nbrs; fn++) {
		auto nbr_rank = mesh.GetFaceNbrRank(fn); 
		MPI_Isend(&mesh_offsets[0], 1, HYPRE_MPI_BIG_INT, nbr_rank, 0, MPI_COMM_WORLD, &send_requests[fn]); 
		MPI_Irecv(&mesh_face_nbr_offsets[fn], 1, HYPRE_MPI_BIG_INT, nbr_rank, 0, MPI_COMM_WORLD, &recv_requests[fn]); 
	}
	MPI_Waitall(num_face_nbrs, send_requests, statuses); 
	MPI_Waitall(num_face_nbrs, recv_requests, statuses); 

	delete[] requests; delete[] statuses; 

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
}

InverseAdvectionOperator::~InverseAdvectionOperator()
{
	igraph_destroy(&graph); 
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

	// allocate face neighbor data 
	TransportVectorExtents psi_fnbr_ext(1, Nomega, Ndof_fnbr);
	mfem::Vector psi_fnbr(TotalExtent(psi_fnbr_ext)); 
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

	// --- do sweep --- 
	// vertices traversed, only count owned elements 
	int nsweep = 0; 
	while (nsweep < Ne*Nomega) {
		// --- sweep on processor-local domain --- 
		while (not(local.empty())) {
			nsweep++; //increment sweep counter to know when to stop 
			// get first element of queue 
			auto node = local.front();
			local.pop_front(); 

			// --- do work --- 
			// deconstruct angle/space id
			const auto a = node % Nomega; 
			const auto e = node / Nomega; 
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
			for (auto i=0; i<dofs.Size(); i++) { rhs[i] = source_view(0,a,dofs[i]); }

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
		}
	}
	MPI_Barrier(MPI_COMM_WORLD); 

	// clean up data 
	igraph_vector_int_destroy(&nbrs); 
	igraph_vector_int_destroy(&nbr_nbrs); 
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