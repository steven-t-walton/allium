#include "sweep.hpp"
#include <deque>
#include "config.hpp"
#include "log.hpp"
#include "lumping.hpp"
#include "multigroup.hpp"

InverseAdvectionOperator::InverseAdvectionOperator(mfem::ParFiniteElementSpace &_fes, const AngularQuadrature &_quad, 
	mfem::GridFunction &_total_data, const BoundaryConditionMap &bc_map, int use_lumping)
	: fes(_fes), mesh(*_fes.GetParMesh()), quad(_quad), total_data(_total_data), lump(use_lumping)
{
	if (lump) {
		const auto *fec = fes.FEColl(); 
		const auto btype = dynamic_cast<const mfem::L2_FECollection*>(fec)->GetBasisType(); 
		if (btype != mfem::BasisType::GaussLobatto) {
			MFEM_ABORT("only lobatto supported with lumping"); 
		}
	}
	const auto dim = mesh.Dimension(); // dimension of mesh 
	const auto Ne = mesh.GetNE(); // processor-local number of elements 
	const auto Ndof = fes.GetVSize(); // local degrees of freedom 
	const auto Nomega = quad.Size(); // number of angles 
	const auto G = total_data.FESpace()->GetVDim(); // number of energy groups 
	psi_ext = TransportVectorExtents(G, Nomega, Ndof); // multi index based on energy, angle, space 
	// set operator sizes to size of psi 
	height = width = TotalExtent(psi_ext);

	// ensure exchanged data called 
	mesh.ExchangeFaceNbrData(); 
	fes.ExchangeFaceNbrData(); 

	// --- create data structures not provided by mfem::ParMesh ---
	// build local to global element map for elements straddling parallel faces 
	// maps index into the ghost element id to its global element number
	const auto nbr_el = mesh.GetNFaceNeighborElements(); // number of "ghost" elements 
	const auto num_face_nbrs = mesh.GetNFaceNeighbors(); // number of neighboring processors 
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
	mfem::Array<HYPRE_BigInt> *ptr[1] = {&mesh_offsets}; 
	HYPRE_BigInt N_hypre[1]; 
	N_hypre[0] = Ne; 
	mesh.GenerateOffsets(1, N_hypre, ptr); 

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
	edge_to_face_id.Reserve(edges.Capacity()); 
	normals.SetSize(dim*mesh.GetNumFaces()); 
	for (int r=0; r<face_to_element->Size(); r++) {
		// get view into row of table 
		auto row = std::span(face_to_element->GetRow(r), face_to_element->RowSize(r)); 
		// compute normal 
		const auto ref_geom = mesh.GetFaceGeometry(r);
		const auto &ip = mfem::Geometries.GetCenter(ref_geom); 
		auto *trans = mesh.GetFaceElementTransformations(r); 
		trans->SetAllIntPoints(&ip);
		mfem::Vector normal(normals, r*dim, dim); // normal vector 
		if (dim==1) {
			normal(0) = 2*trans->GetElement1IntPoint().x - 1.0;
		} else {
			mfem::CalcOrtho(trans->Jacobian(), normal); 				
		}
		normal *= 1.0/normal.Norml2(); 

		if (row.size()==2) { // two elements => interior face 
			// add directed edges based on Omega.n 
			for (int a=0; a<Nomega; a++) {
				auto e1 = row[0]; 
				auto e2 = row[1]; 
				const auto &Omega = quad.GetOmega(a); 
				double dot = normal*Omega; 
				bool force_swap = false; 
				if (dot==0 and e2 >= Ne) { // dot=0 and parallel => arbitrarily choose one as upwind 
					const auto ge1 = mesh_offsets[0] + e1; 
					const auto ge2 = fnbr_to_global[e2 - Ne];
					if (ge2 > ge1) force_swap = true; 
				}
				if (dot < 0 or force_swap) std::swap(e1,e2); 
				auto dof1 = (e1<Ne) ? e1*Nomega + a : Ne*Nomega + (e1 - Ne)*Nomega + a; 
				auto dof2 = (e2<Ne) ? e2*Nomega + a : Ne*Nomega + (e2 - Ne)*Nomega + a; 
				edges.Append(dof1); 
				edges.Append(dof2); 
				edge_to_face_id.Append(r); 
			}
		}

		else { // one element => boundary face 
			const auto be = face_to_bdr_el.at(r); 
			auto *btrans = mesh.GetBdrFaceTransformations(be); 
			const auto attr = btrans->Attribute; 
			if (bc_map.at(attr) == BoundaryCondition::REFLECTIVE) {
				for (int a=0; a<Nomega; a++) {
					const auto &Omega = quad.GetOmega(a); 
					double dot = normal * Omega; 
					if (dot < 0) { // inflow => need to add edge to reflected angle 
						const auto reflect_idx = quad.GetReflectedAngleIndex(a, normal); 
						const auto dof1 = row[0] * Nomega + reflect_idx; 
						const auto dof2 = row[0] * Nomega + a; 
						edges.Append(dof1); 
						edges.Append(dof2); 
						edge_to_face_id.Append(r); 
					}
				}
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

	// count nodes, allocate degrees array 
	const auto num_nodes = igraph_vcount(&graph); 
	degrees.SetSize(num_nodes); 

	// allocate parallel exchange buffer 
	const auto Ndof_fnbr = fes.GetFaceNbrVSize(); // spatial DOF in ghost buffer 
	psi_fnbr_ext = TransportVectorExtents(G, Nomega, Ndof_fnbr); 
	psi_fnbr.SetSize(TotalExtent(psi_fnbr_ext)); 

	// allocate local buffer through call to set send buffer size 
	// call with default value for send_buffer_size 
	SetSendBufferSize(send_buffer_size); 

	// map processor number to face number index 
	for (auto fn=0; fn<num_face_nbrs; fn++) {
		proc_to_fn[mesh.GetFaceNbrRank(fn)] = fn; 
	}

	delete[] requests; delete[] statuses; 
	AssembleLocalMatrices(); 
}

InverseAdvectionOperator::~InverseAdvectionOperator()
{
	igraph_destroy(&graph); 
	for (auto &ptr : mass_matrices) { delete ptr; }
	for (auto &ptr : grad_matrices) { delete ptr; }
	for (auto &ptr : face_matrices) { delete ptr; }
}

void InverseAdvectionOperator::Mult(const mfem::Vector &source, mfem::Vector &psi) const 
{
	assert(source.Size() == Width()); 
	assert(psi.Size() == Height()); 
	assert(mass_matrices[0]); 

	const auto num_face_nbrs = mesh.GetNFaceNeighbors(); 
	const auto dim = mesh.Dimension(); 
	const auto Ne = mesh.GetNE(); 
	const auto Nomega = quad.Size(); 
	const auto G = psi_ext.extent(0); 
	// place to store face ids/element, dofs/element, and dofs of neighboring elements 
	mfem::Array<int> faces, dofs, dofs2; 

	// igraph vectors to store neighbors in graph traversal 
	igraph_vector_int_t nbrs, nbr_nbrs, edges; 
	igraph_vector_int_init(&nbrs, 0); 
	igraph_vector_int_init(&nbr_nbrs, 0); 
	igraph_vector_int_init(&edges, 0);

	// mdspan views into data 
	TransportVectorView psi_view(psi.GetData(), psi_ext); 
	TransportVectorView psi_fnbr_view(psi_fnbr.GetData(), psi_fnbr_ext); 
	TransportVectorView source_view(source.GetData(), psi_ext); 

	// mdspan views into pre-assembled matrices 
	auto mass_mat_view = Kokkos::mdspan(mass_matrices.GetData(), G, fes.GetNE()); 
	auto grad_mat_view = Kokkos::mdspan(grad_matrices.GetData(), dim, fes.GetNE()); 
	auto face_mat_view = Kokkos::mdspan(face_matrices.GetData(), mesh.GetNumFaces(), 2, 2); 

	// --- determine roots of graph --- 
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

	// list of {processor index, angle/space set} to exchange 
	mfem::Array<mfem::Connection> send_list; 
	send_list.Reserve(send_buffer_size); 
	// local buffer for recv meta data 
	mfem::Array<int> node_buffer(send_buffer_size); 

	// count messages sents 
	auto message_count = 0; // incremented at each message, used as message tag 

	// persistent storage of local system 
	mfem::DenseMatrix grad, A, lu; 
	mfem::Vector rhs, sol, psi2, psi_fixup, nor(dim), ones, fixup_weights; 

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
			rhs.SetSize(dofs.Size()); 

			// --- assemble gradient term ---
			grad.SetSize(dofs.Size()); 
			grad = 0.0; 
			for (auto d=0; d<dim; d++) {
				grad.Add(Omega(d), *grad_mat_view(d,e)); 
			}

			// --- assemble outflow face terms --- 
			element_to_face->GetRow(e, faces);
			for (auto f : faces) {
				const auto info = mesh.GetFaceInformation(f); 
				bool keep_order = e == info.element[0].index; 
				for (auto d=0; d<dim; d++) { nor(d) = normals(f*dim + d); }
				if (!keep_order) nor *= -1.0; 
				const double dot = Omega * nor; 
				const auto idx = keep_order ? 0 : 1; 

				// outflow 
				if (dot > 0) {
					const auto &elmat = *face_mat_view(f, idx, idx); 
					grad.Add(dot, elmat); 
				}
			}

			// --- assemble and solve all groups --- 
			igraph_incident(&graph, &edges, node, IGRAPH_IN); 
			auto edges_in_view = std::span(VECTOR(edges), igraph_vector_int_size(&edges)); 
			for (int g=0; g<G; g++) {
				// copy group-independent terms 
				A = grad; 
				// add total mass matrix for group g 
				A.Add(1.0, *mass_mat_view(g,e)); 

				// --- load source into local vector --- 
				for (auto i=0; i<dofs.Size(); i++) { rhs[i] = source_view(g,a,dofs[i]); }

				// add inflow terms to RHS 
				for (const auto eid : edges_in_view) {
					const auto nbr = IGRAPH_FROM(&graph, eid); 
					const auto nbr_e = nbr / Nomega; 
					const auto nbr_a = nbr % Nomega; 
					const auto fid = edge_to_face_id[eid]; 
					for (auto d=0; d<dim; d++) { nor(d) = normals(fid * dim + d); }
					const auto info = mesh.GetFaceInformation(fid); 
					bool keep_order = e == info.element[0].index; 
					if (!keep_order) nor *= -1.0; 
					const double dot = Omega * nor; 
					const auto idx = keep_order ? 0 : 1; 
					const auto nbr_idx = (idx + 1) % 2; 

					if (nbr_e == e) { // reflection 
						const auto &elmat = *face_mat_view(fid, idx, idx); 
						psi2.SetSize(dofs.Size()); 
						for (auto i=0; i<dofs.Size(); i++) { psi2(i) = psi_view(g, nbr_a, dofs[i]); }
						elmat.AddMult(psi2, rhs, -dot);
					} 

					else { // inflow from neighbor
						const auto &elmat = *face_mat_view(fid, idx, nbr_idx); 
						// face neighbor data 
						if (info.IsShared()) {
							fes.GetFaceNbrElementVDofs(nbr_e-Ne, dofs2); 
							psi2.SetSize(dofs2.Size()); 
							for (int i=0; i<dofs2.Size(); i++) { psi2(i) = psi_fnbr_view(g, a, dofs2[i]); }
						}
						// local data 
						else {
							fes.GetElementDofs(nbr_e, dofs2); 
							psi2.SetSize(dofs2.Size()); 
							for (int i=0; i<dofs2.Size(); i++) { psi2(i) = psi_view(g, a, dofs2[i]); }						
						}
						elmat.AddMult(psi2, rhs, -dot); 
					}
				}

				if (is_time_dependent) {
					A.Add(time_absorption, *time_mass_matrices[e]);
				}
				// solve, solution overwritten into rhs, matrix is modified 
				lu = A; 
				sol = rhs; 
				mfem::LinearSolve(lu, sol.GetData()); 

				if (fixup_op and apply_fixup) {
					fixup_op->SetLocalSystem(A, rhs); 
					fixup_op->Mult(sol, rhs); 
				} else {
					rhs = sol; 
				}

				// scatter back 
				for (int i=0; i<dofs.Size(); i++) { psi_view(g, a, dofs[i]) = rhs(i); }
			}

			// --- compute next + send data to other processors --- 
			degrees[node] = -1; // already visited 
			// get neighbors 
			igraph_neighbors(&graph, &nbrs, node, IGRAPH_OUT);  
			auto nbrs_view = std::span(VECTOR(nbrs), igraph_vector_int_size(&nbrs)); 
			std::set<int> send_fn_set; // store unique set of processors to send data to 
			for (const auto &nbr : nbrs_view) {
				degrees[(int)nbr] -= 1; // reduce degree since node has been visited 
				// if nbr is off-processor, add to send_set regardless of degree 
				if (nbr >= Ne*Nomega) {
					// get neighbor of neighbor to find number of processors to send data to 
					igraph_neighbors(&graph, &nbr_nbrs, node, IGRAPH_OUT);
					auto nbr_nbrs_view = std::span(VECTOR(nbr_nbrs), igraph_vector_int_size(&nbr_nbrs)); 	
					for (const auto &nbr_nbr : nbr_nbrs_view) {
						if (nbr_nbr < Ne*Nomega) continue; // only send to off-proc neighbors 
						const auto nbr_id = (int)nbr_nbr - Ne*Nomega; 
						const auto fn = fnbr_to_fn[nbr_id / Nomega]; 
						send_fn_set.insert(fn); 
					}
				} 
				// neighbor is local, add to queue 
				else {
					if (degrees[(int)nbr] == 0) {
						local.push_back(nbr); 						
					}					
				}
			}

			// append to send list 
			for (const auto &fn : send_fn_set) {
				send_list.Append({fn, node}); 
			}

			// send data if send buffer full or no more local work to do 
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

					buffer_size *= G; 
					assert(par_data_buffer.Size() >= buffer_size); 
					auto idx = 0;
					for (auto n : nodes) {
						const auto e = n / Nomega; 
						const auto a = n % Nomega; 
						fes.GetElementDofs(e, dofs); 
						for (auto g=0; g<G; g++) {
							for (auto i=0; i<dofs.Size(); i++) {
								par_data_buffer[idx++] = psi_view(g,a,dofs[i]); 
							}
						}
					}

					MPI_Request request[2]; 
					MPI_Isend(send_table.GetRow(fn), send_table.RowSize(fn), MPI_INT, nbr_rank, message_count++, 
						MPI_COMM_WORLD, &request[0]); 
					MPI_Isend(par_data_buffer.GetData(), buffer_size, MPI_DOUBLE, nbr_rank, message_count++, MPI_COMM_WORLD, &request[1]); 
					MPI_Waitall(2, request, MPI_STATUS_IGNORE); 
				}
				send_list.SetSize(0); // set size to 0, keeps capacity from reserve call above 
			}
		}

		while (true) {
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

			// make sure data can fit in buffers 
			assert(node_buffer.Size() >= node_count); 
			assert(par_data_buffer.Size() >= data_count); 

			MPI_Request request[2]; 
			assert(status[0].MPI_SOURCE == status[1].MPI_SOURCE); 
			MPI_Irecv(node_buffer.GetData(), node_count, MPI_INT, source, status[0].MPI_TAG, MPI_COMM_WORLD, &request[0]); 
			MPI_Irecv(par_data_buffer.GetData(), data_count, MPI_DOUBLE, source, status[1].MPI_TAG, MPI_COMM_WORLD, &request[1]); 
			MPI_Waitall(2, request, MPI_STATUSES_IGNORE); 

			const auto fn = proc_to_fn.at(source); 
			int buffer_idx = 0; 
			for (auto n=0; n<node_count; n++) {
				const auto node = node_buffer[n]; 
				// convert to global id so that it can be mapped to local fnbr id 
				const auto global = mesh_face_nbr_offsets[fn] * Nomega + node;  
				// convert tag (global idx) to a local index 
				const auto e = global / Nomega; 
				const auto a = global % Nomega; 
				const auto local_e = global_to_fnbr.at(e) + Ne; 
				const auto local_id = local_e*Nomega + a;		
				fes.GetFaceNbrElementVDofs(local_e-Ne, dofs); 	
				for (int g=0; g<G; g++) {
					for (auto i=0; i<dofs.Size(); i++) {
						psi_fnbr_view(g,a,dofs[i]) = par_data_buffer[buffer_idx++];
					}
				}

				// mark data as recvd, traverse neighbors to add the locally owned tail to the queue 
				// for use in the local sweep 
				degrees[local_id] = -1; 
				igraph_neighbors(&graph, &nbrs, local_id, IGRAPH_OUT); 
				auto nbrs_view = std::span(VECTOR(nbrs), igraph_vector_int_size(&nbrs)); 
				for (const auto &nbr : nbrs_view) {
					degrees[(int)nbr] -= 1; 
					if (degrees[(int)nbr]==0) {
						local.push_back(nbr); // add the locally owned id to the queue 
					}
				}				
			}
		}
	}
	// clean up data 
	igraph_vector_int_destroy(&nbrs); 
	igraph_vector_int_destroy(&nbr_nbrs); 

	// use barrier to avoid tag clash? 
	MPI_Barrier(MPI_COMM_WORLD); 
}

void InverseAdvectionOperator::AssembleLocalMatrices() 
{
	const auto dim = mesh.Dimension(); 
	const auto G = total_data.FESpace()->GetVDim(); 

	mass_matrices.SetSize(fes.GetNE()*G);
	auto mass_mat_view = Kokkos::mdspan(mass_matrices.GetData(), G, fes.GetNE());  
	grad_matrices.SetSize(fes.GetNE()*dim); 
	auto grad_mat_view = Kokkos::mdspan(grad_matrices.GetData(), dim, fes.GetNE()); 
	for (auto e=0; e<fes.GetNE(); e++) {
		const auto &fe = *fes.GetFE(e); 
		auto &trans = *mesh.GetElementTransformation(e);
		LumpedIntegrationRule lumped_ir(trans.GetGeometryType()); 

		for (int g=0; g<G; g++) {
			mfem::GridFunctionCoefficient total(&total_data, g+1); // component is 1-based :( 
			mfem::MassIntegrator mi(total); 
			mass_mat_view(g,e) = new mfem::DenseMatrix; 
			if (IsMassLumped(lump)) mi.SetIntegrationRule(lumped_ir); 
			mi.AssembleElementMatrix(fe, trans, *mass_mat_view(g,e)); 
		}

		mfem::Vector Omega(dim); 
		for (auto d=0; d<dim; d++) {
			grad_mat_view(d,e) = new mfem::DenseMatrix; 
			Omega = 0.0; 
			Omega(d) = 1.0; 
			mfem::VectorConstantCoefficient Q(Omega); 
			mfem::ConservativeConvectionIntegrator conv_int(Q, 1.0); 
			if (IsGradientLumped(lump)) conv_int.SetIntegrationRule(lumped_ir); 
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

		// setup lumped integration rule 
		LumpedIntegrationRule lumped_ir(face_trans->GetGeometryType()); 
		if (IsFaceLumped(lump)) fmi.SetIntegrationRule(lumped_ir); 

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
}

void InverseAdvectionOperator::SetTimeAbsorption(const double sigma)
{
	time_absorption = sigma; 
	is_time_dependent = true; 
	if (time_mass_matrices.Size() > 0) { return; }

	time_mass_matrices.SetSize(fes.GetNE()); 
	// no coefficient so time step can be changed 
	// without re-assembling 
	mfem::MassIntegrator mi; 
	for (auto e=0; e<fes.GetNE(); e++) {
		const auto &fe = *fes.GetFE(e); 
		auto &trans = *mesh.GetElementTransformation(e);
		LumpedIntegrationRule lumped_ir(trans.GetGeometryType()); 

		time_mass_matrices[e] = new mfem::DenseMatrix; 
		if (IsMassLumped(lump)) mi.SetIntegrationRule(lumped_ir); 
		mi.AssembleElementMatrix(fe, trans, *time_mass_matrices[e]); 
	}
}

void InverseAdvectionOperator::SetSendBufferSize(int s) 
{
	send_buffer_size = s; 

	// allocate maximum local buffer to send psi data in parallel 
	const auto &face_nbr_element_dof = fes.face_nbr_element_dof; 
	int max_send_per_fn = 0; 
	for (int nbr_el=0; nbr_el<face_nbr_element_dof.Size(); nbr_el++) {
		int size = face_nbr_element_dof.RowSize(nbr_el); 
		max_send_per_fn	= std::max(size, max_send_per_fn); 
	}
	par_data_buffer.SetSize(send_buffer_size * max_send_per_fn * psi_ext.extent(0)); 
}

void InverseAdvectionOperator::UseFixup(bool use) 
{
	apply_fixup = use;
	if (use and !fixup_op) MFEM_ABORT("no fixup operator set"); 
}

void InverseAdvectionOperator::WriteGraphToDot(std::string prefix) const 
{
	auto rank = mesh.GetMyRank(); 
	FILE *file = fopen(mfem::MakeParFilename(prefix + ".", rank, ".dot").c_str(), "w"); 
	igraph_write_graph_dot(&graph, file); 
	fclose(file); 	
}

void FormTransportSource(mfem::ParFiniteElementSpace &fes, AngularQuadrature &quad, 
	const mfem::Array<double> &energy_grid, PhaseSpaceCoefficient &source_coef, 
	PhaseSpaceCoefficient &inflow_coef, TransportVectorView source_view)
{
	const int G = energy_grid.Size() - 1; 
	for (int g=0; g<G; g++) {
		const auto e_low = energy_grid[g]; const auto e_high = energy_grid[g+1];
		const auto mid = (e_low + e_high)/2;
		source_coef.SetEnergy(e_low, e_high, mid); inflow_coef.SetEnergy(e_low, e_high, mid); 
		for (auto a=0; a<quad.Size(); a++) {
			const auto &Omega = quad.GetOmega(a); 
			source_coef.SetAngle(Omega); inflow_coef.SetAngle(Omega); 

			mfem::ParLinearForm bform(&fes); 
			bform.AddDomainIntegrator(new mfem::DomainLFIntegrator(source_coef)); 
			mfem::VectorConstantCoefficient Q(Omega); 
			bform.AddBdrFaceIntegrator(new mfem::BoundaryFlowIntegrator(inflow_coef, Q, -1.0, -0.5));
			bform.Assemble(); 
			for (int i=0; i<bform.Size(); i++) {
				source_view(g,a,i) = bform[i]; 
			}
		}		
	}
}

void FormTransportSource(mfem::FiniteElementSpace &fes, const AngularQuadrature &quad, 
	const MultiGroupEnergyGrid &energy_grid, PhaseSpaceCoefficient &source_coef, 
	PhaseSpaceCoefficient &inflow_coef, mfem::Vector &source)
{
	TransportVectorExtents psi_ext(energy_grid.Size(), quad.Size(), fes.GetVSize());
	auto source_view = TransportVectorView(source.GetData(), psi_ext);
	for (int g=0; g<energy_grid.Size(); g++) {
		source_coef.SetEnergy(energy_grid.LowerBound(g), energy_grid.UpperBound(g), 
			energy_grid.MeanEnergy(g));
		inflow_coef.SetEnergy(energy_grid.LowerBound(g), energy_grid.UpperBound(g), 
			energy_grid.MeanEnergy(g));		
		for (int a=0; a<quad.Size(); a++) {
			const auto &Omega = quad.GetOmega(a);
			source_coef.SetAngle(Omega); inflow_coef.SetAngle(Omega);

			mfem::LinearForm bform(&fes); 
			bform.AddDomainIntegrator(new mfem::DomainLFIntegrator(source_coef)); 
			mfem::VectorConstantCoefficient Q(Omega); 
			bform.AddBdrFaceIntegrator(new mfem::BoundaryFlowIntegrator(inflow_coef, Q, -1.0, -0.5));
			bform.Assemble(); 
			for (int i=0; i<bform.Size(); i++) {
				source_view(g,a,i) = bform[i]; 
			}
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

void AdvectionOperator::Mult(const mfem::Vector &x, mfem::Vector &y) const 
{
	auto &mesh = Linv.mesh;
	const auto dim = mesh.Dimension(); 
	auto &fes = Linv.fes; 
	const auto &psi_ext = Linv.psi_ext; 
	const auto &psi_fnbr_ext = Linv.psi_fnbr_ext; 
	const auto G = psi_ext.extent(0); 
	const auto &quad = Linv.quad; 
	const auto Nomega = quad.Size(); 
	const auto &normals = Linv.normals; 
	auto mass_mat_view = Kokkos::mdspan(Linv.mass_matrices.GetData(), G, fes.GetNE()); 
	auto grad_mat_view = Kokkos::mdspan(Linv.grad_matrices.GetData(), dim, fes.GetNE()); 
	auto face_mat_view = Kokkos::mdspan(Linv.face_matrices.GetData(), mesh.GetNumFaces(), 2, 2); 

	TransportVectorView xview(x.GetData(), psi_ext); 
	TransportVectorView yview(y.GetData(), psi_ext); 
	y = 0.0; 

	auto &psi_fnbr = Linv.psi_fnbr; 
	psi_fnbr = 0.0; 
	if (fes.GetFaceNbrVSize() > 0) {
		int *send_offset = fes.send_face_nbr_ldof.GetI(); 
		const int *d_send_offset = fes.send_face_nbr_ldof.GetJ(); 
		int *recv_offset = fes.face_nbr_ldof.GetI(); 
		MPI_Comm comm = fes.GetComm(); 
		const int num_face_nbrs = mesh.GetNFaceNeighbors(); 

		// load data into send buffer 
		const auto num_send_dofs = fes.send_face_nbr_ldof.Size_of_connections(); 
		mfem::Vector send_buffer(num_send_dofs * G * Nomega); 
		for (int i=0; i<fes.send_face_nbr_ldof.Size_of_connections(); i++) {
			for (int g=0; g<G; g++) {
				for (int a=0; a<Nomega; a++) {
					send_buffer(a + g*Nomega + Nomega*G*i) = xview(g,a,d_send_offset[i]); 
				}
			}
		}

		auto *requests = new MPI_Request[2*num_face_nbrs]; 
		auto *send_requests = requests; 
		auto *recv_requests = requests + num_face_nbrs; 
		auto *statuses = new MPI_Status[num_face_nbrs]; 
		for (int fn=0; fn<num_face_nbrs; fn++) {
			const int nbr_rank = mesh.GetFaceNbrRank(fn); 
			int tag = 0; 
			MPI_Isend(&send_buffer.GetData()[send_offset[fn]*G*Nomega], 
				(send_offset[fn+1] - send_offset[fn])*G*Nomega, MPI_DOUBLE, 
				nbr_rank, tag, comm, &send_requests[fn]); 
			MPI_Irecv(&psi_fnbr.GetData()[recv_offset[fn]*G*Nomega], 
				(recv_offset[fn+1] - recv_offset[fn])*G*Nomega,
				MPI_DOUBLE, nbr_rank, tag, comm, &recv_requests[fn]); 
		}
		MPI_Waitall(num_face_nbrs, send_requests, statuses); 
		MPI_Waitall(num_face_nbrs, recv_requests, statuses); 
		delete[] requests; delete[] statuses; 
	}

	mfem::Array<int> vdofs, vdofs2; 
	mfem::Vector xlocal, xlocal2, ylocal, ylocal2, nor(dim); 
	mfem::DenseMatrix grad; 
	for (int g=0; g<G; g++) {
		for (int a=0; a<quad.Size(); a++) {
			const auto &Omega = quad.GetOmega(a); 
			for (int e=0; e<mesh.GetNE(); e++) {
				fes.GetElementDofs(e, vdofs); 
				xlocal.SetSize(vdofs.Size()); 
				ylocal.SetSize(vdofs.Size()); 
				for (int i=0; i<vdofs.Size(); i++) { xlocal(i) = xview(g,a,vdofs[i]); }
				mass_mat_view(g,e)->Mult(xlocal, ylocal); 
				grad.SetSize(vdofs.Size()); 
				grad = 0.0; 
				for (int d=0; d<dim; d++) {
					grad.Add(Omega(d), *grad_mat_view(d,e)); 
				}
				grad.AddMult(xlocal, ylocal); 
				for (int i=0; i<vdofs.Size(); i++) { yview(g,a,vdofs[i]) += ylocal(i); }
			}

			for (int f=0; f<mesh.GetNumFaces(); f++) {
				for (auto d=0; d<dim; d++) { nor(d) = normals(f*dim + d); }
				const auto &info = mesh.GetFaceInformation(f); 
				mfem::FaceElementTransformations *trans; 
				if (info.IsShared()) {
					trans = mesh.GetSharedFaceTransformationsByLocalIndex(f, true); 
				} else {
					trans = mesh.GetFaceElementTransformations(f); 
				}
				const double dot = Omega * nor; 
				const auto e1 = trans->Elem1No; 
				fes.GetElementDofs(e1, vdofs); 
				xlocal.SetSize(vdofs.Size()); 
				ylocal.SetSize(vdofs.Size()); 
				for (int i=0; i<vdofs.Size(); i++) { xlocal(i) = xview(g,a,vdofs[i]); }

				ylocal = 0.0; 
				if (info.IsShared()) {
					const auto e2 = trans->Elem2No; 
					const auto idx = e2 - mesh.GetNE(); 
					fes.GetFaceNbrElementVDofs(idx, vdofs2); 
					xlocal2.SetSize(vdofs2.Size());  
					for (int i=0; i<vdofs2.Size(); i++) { xlocal2(i) = psi_fnbr[a + g*Nomega + Nomega*G*vdofs2[i]]; }
					if (dot >= 0) {
						face_mat_view(f,0,0)->AddMult(xlocal, ylocal, dot); 
					} else {
						face_mat_view(f,0,1)->AddMult(xlocal2, ylocal, dot); 
					}
				}

				else if (info.IsInterior()) {
					const auto e2 = trans->Elem2No; 
					fes.GetElementDofs(e2, vdofs2);
					xlocal2.SetSize(vdofs2.Size());  
					ylocal2.SetSize(vdofs2.Size()); 
					for (int i=0; i<vdofs2.Size(); i++) { xlocal2(i) = xview(g,a,vdofs2[i]); }
					ylocal2 = 0.0; 
					if (dot >= 0) { // Omega points 1 -> 2 
						// 1 gets outflow 
						face_mat_view(f,0,0)->AddMult(xlocal, ylocal, dot); 
						// 2 gets inflow 
						face_mat_view(f,1,0)->AddMult(xlocal, ylocal2, -dot); 
					} 

					else { // Omega points 2 -> 1 
						face_mat_view(f,0,1)->AddMult(xlocal2, ylocal, dot); 
						face_mat_view(f,1,1)->AddMult(xlocal2, ylocal2, -dot); 
					}
					for (int i=0; i<vdofs2.Size(); i++) { yview(g,a,vdofs2[i]) += ylocal2(i); }
				}

				else {
					if (dot >= 0) {
						face_mat_view(f,0,0)->AddMult(xlocal, ylocal, dot); 
					} 
				}
				for (int i=0; i<vdofs.Size(); i++) { yview(g,a,vdofs[i]) += ylocal(i); }
			}
		}
	}
}