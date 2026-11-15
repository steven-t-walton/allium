#include "sweep_scheduler.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>

SweepScheduler::SweepScheduler(mfem::ParMesh &mesh, const AngularQuadrature &quad,
	const BoundaryConditionMap &bc_map, std::vector<std::vector<int>> angle_sets)
	: mesh(mesh), quad(quad), sets(std::move(angle_sets))
{
	const auto dim = mesh.Dimension(); // dimension of mesh
	const auto Nomega = quad.Size(); // number of angles
	Ne = mesh.GetNE(); // processor-local number of elements

	// --- angle sets ---
	// default: one set per ordinate (standard Sn sweep)
	if (sets.empty()) {
		sets.resize(Nomega);
		for (int a=0; a<Nomega; a++) { sets[a] = {a}; }
	}
	S = sets.size();
	set_of_angle.assign(Nomega, -1);
	for (int s=0; s<S; s++) {
		for (const auto a : sets[s]) {
			if (a < 0 or a >= Nomega or set_of_angle[a] != -1) MFEM_ABORT("angle sets must partition the quadrature");
			set_of_angle[a] = s;
		}
	}
	if (std::count(set_of_angle.begin(), set_of_angle.end(), -1)) MFEM_ABORT("angle sets must partition the quadrature");

	// ensure exchanged data called
	mesh.ExchangeFaceNbrData();

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
		for (auto i=0; i<recv_face_nbr_elements.RowSize(fn); i++) {
			fnbr_to_fn[idx_offset + i] = fn;
		}
	}

	// map processor number to face number index
	for (auto fn=0; fn<num_face_nbrs; fn++) {
		proc_to_fn[mesh.GetFaceNbrRank(fn)] = fn;
	}

	// --- map face id to boundary element id ---
	// required to use the bdr_attribute for reflective conditions
	std::unordered_map<int,int> face_to_bdr_el;
	for (auto i=0; i<mesh.GetNBE(); i++) {
		face_to_bdr_el[mesh.GetBdrElementFaceIndex(i)] = i;
	}

	// --- create graph ---
	// faces to elements, has 2 elements for interior faces, 1 for boundary
	// includes parallel shared faces to ghost/face neighbor elements
	// handles 1/2/3D
	std::unique_ptr<const mfem::Table> face_to_element(mesh.GetFaceToAllElementTable());
	// "invert" to go from element id to the list of faces associated with that element
	// needed for sweep
	element_to_face = std::unique_ptr<const mfem::Table>(mfem::Transpose(*face_to_element));

	// upwind direction of angle set s across a face with normal nor
	// +1: out of the normal's element, -1: into it, 0: all ordinates are parallel to the face
	auto SetDirection = [&](int s, const mfem::Vector &nor) {
		int sign = 0;
		for (const auto a : sets[s]) {
			const double dot = quad.GetOmega(a) * nor;
			const int sa = (dot > 0) - (dot < 0);
			if (sa == 0) continue;
			if (sign != 0 and sa != sign) MFEM_ABORT("angle set " << s << " is not upwind consistent on a face");
			sign = sa;
		}
		return sign;
	};
	auto LocalNode = [&](int e, int s) { return (e<Ne) ? e*S + s : Ne*S + (e - Ne)*S + s; };

	// loop through faces, add directed edges based on Omega.n
	// normal is assumed to be piecewise constant! Works for linear meshes only
	std::vector<igraph_integer_t> edges; // list of edges
	std::vector<int> edge_to_face_id; // map an edge id to the mesh's face id
	edges.reserve(face_to_element->Size_of_connections()*S*2);
	edge_to_face_id.reserve(edges.capacity());
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
			for (int s=0; s<S; s++) {
				auto e1 = row[0];
				auto e2 = row[1];
				const int dir = SetDirection(s, normal);
				bool force_swap = false;
				if (dir==0 and e2 >= Ne) { // dot=0 and parallel => arbitrarily choose one as upwind
					const auto ge1 = mesh_offsets[0] + e1;
					const auto ge2 = fnbr_to_global[e2 - Ne];
					if (ge2 > ge1) force_swap = true;
				}
				if (dir < 0 or force_swap) std::swap(e1,e2);
				edges.push_back(LocalNode(e1, s));
				edges.push_back(LocalNode(e2, s));
				edge_to_face_id.push_back(r);
			}
		}

		else { // one element => boundary face
			const auto be = face_to_bdr_el.at(r);
			auto *btrans = mesh.GetBdrFaceTransformations(be);
			const auto attr = btrans->Attribute;
			if (bc_map.at(attr) == BoundaryCondition::REFLECTIVE) {
				for (int s=0; s<S; s++) {
					if (SetDirection(s, normal) >= 0) continue;
					// inflow => need to add edge from the reflected set
					// ordinates parallel to the face reflect onto themselves and do not couple
					int reflect_set = -1;
					for (const auto a : sets[s]) {
						if (quad.GetOmega(a) * normal == 0.0) continue;
						const auto rs = set_of_angle[quad.GetReflectedAngleIndex(a, normal)];
						if (reflect_set != -1 and rs != reflect_set) MFEM_ABORT("angle set " << s << " reflects onto multiple sets");
						reflect_set = rs;
					}
					edges.push_back(row[0]*S + reflect_set);
					edges.push_back(row[0]*S + s);
					edge_to_face_id.push_back(r);
				}
			}
		}
	}

	// --- convert to igraph ---
	const int num_nodes = (Ne + nbr_el)*S;
	igraph_vector_int_t edges_igraph;
	// view into edge data, igraph does not own the data
	igraph_vector_int_view(&edges_igraph, edges.data(), edges.size());
	igraph_create(&graph, &edges_igraph, num_nodes, true);

	// ensure DAG
	igraph_bool_t is_dag;
	igraph_is_dag(&graph, &is_dag);
	if (!is_dag) { MFEM_ABORT("graph is not a dag"); }

	// --- adjacency in CSR format, ordered by edge id ---
	// in edges are kept only for local nodes (ghost nodes are never solved here),
	// out edges for all nodes so received ghost data can release local nodes
	const int num_edges = edge_to_face_id.size();
	const int num_local = Ne*S;
	in_offsets.assign(num_local+1, 0);
	out_offsets.assign(num_nodes+1, 0);
	for (int i=0; i<num_edges; i++) {
		const auto from = edges[2*i], to = edges[2*i+1];
		if (to < num_local) in_offsets[to+1]++;
		out_offsets[from+1]++;
	}
	for (int n=0; n<num_local; n++) { in_offsets[n+1] += in_offsets[n]; }
	for (int n=0; n<num_nodes; n++) { out_offsets[n+1] += out_offsets[n]; }
	in_edges.resize(in_offsets.back());
	out_nodes.resize(out_offsets.back());
	std::vector<int> in_count(num_local, 0), out_count(num_nodes, 0);
	for (int i=0; i<num_edges; i++) {
		const int from = edges[2*i], to = edges[2*i+1];
		out_nodes[out_offsets[from] + out_count[from]++] = to;
		if (to >= num_local) continue;

		// classify the edge by where the upwind node lives
		const auto f = edge_to_face_id[i];
		const int e = to / S;
		SweepEdge edge;
		edge.face = f;
		edge.side = ElementSide(e, f);
		edge.set = from % S;
		if (from >= num_local) {
			edge.kind = SweepEdge::GHOST;
			edge.elem = (from - num_local) / S;
		} else {
			edge.elem = from / S;
			edge.kind = (edge.elem == e) ? SweepEdge::REFLECT : SweepEdge::LOCAL;
		}
		in_edges[in_offsets[to] + in_count[to]++] = edge;
	}

	// --- face neighbor processors each local node sends to ---
	// a node with several ghost downwind nodes on the same processor is sent once
	send_offsets.assign(num_local+1, 0);
	for (int n=0; n<num_local; n++) {
		std::vector<int> fns;
		for (const auto nbr : OutNodes(n)) {
			if (nbr >= num_local) fns.push_back(fnbr_to_fn[(nbr - num_local) / S]);
		}
		std::sort(fns.begin(), fns.end());
		fns.erase(std::unique(fns.begin(), fns.end()), fns.end());
		send_fns.insert(send_fns.end(), fns.begin(), fns.end());
		send_offsets[n+1] = send_fns.size();
	}
}

SweepScheduler::~SweepScheduler()
{
	igraph_destroy(&graph);
}

void SweepScheduler::OutwardNormal(int f, int side, mfem::Vector &nor) const
{
	const auto dim = mesh.Dimension();
	nor.SetSize(dim);
	for (auto d=0; d<dim; d++) { nor(d) = normals(f*dim + d); }
	if (side == 1) nor *= -1.0;
}

void SweepScheduler::Sweep(SweepKernel &kernel, bool lagged) const
{
	const auto num_face_nbrs = mesh.GetNFaceNeighbors();
	const int num_local = Ne*S;
	const auto comm = MPI_COMM_WORLD;

	// --- count edges incoming to each local vertex ---
	// lagged ghost data is available from the start
	std::vector<int> degrees(num_local);
	for (int n=0; n<num_local; n++) {
		degrees[n] = 0;
		for (const auto &edge : InEdges(n)) {
			if (!lagged or edge.kind != SweepEdge::GHOST) degrees[n]++;
		}
	}
	// list of nodes to visit (not including those that depend on another processor)
	// initialize with roots of graph
	std::deque<int> local;
	for (int n=0; n<num_local; n++) {
		if (degrees[n] == 0) local.push_back(n);
	}

	// local nodes to send to each face neighbor after the current wave front
	std::vector<std::vector<int>> send_lists(num_face_nbrs);

	// store packed messages and an MPI request
	// this allows keeping the buffer around past the
	// call to MPI_Isend. This prevents MPI hanging
	// when it has buffer issues and requires
	// the receiver to post a MPI_Recv before send
	// can complete
	struct SentMessage {
		MPI_Request request;
		std::vector<char> buffer;
	};
	std::deque<SentMessage> sent_messages;
	// incremented at each message, used as message tag
	int message_count = 0;

	// one message per face neighbor processor and wave front
	// message layout: [int count][int node x count][int64 size x count][payload x count]
	// node ids are the sender's local node ids, the receiver maps them to ghost ids
	auto SendWaveFront = [&]() {
		for (auto fn=0; fn<num_face_nbrs; fn++) {
			auto &nodes = send_lists[fn];
			if (nodes.empty()) continue;
			const int count = nodes.size();
			std::vector<std::int64_t> sizes(count);
			std::size_t payload = 0;
			for (int n=0; n<count; n++) {
				sizes[n] = kernel.PackSize(nodes[n] / S, nodes[n] % S);
				payload += sizes[n];
			}
			auto &message = sent_messages.emplace_back();
			message.buffer.resize(sizeof(int)*(count+1) + sizeof(std::int64_t)*count + payload);
			char *p = message.buffer.data();
			std::memcpy(p, &count, sizeof(int)); p += sizeof(int);
			std::memcpy(p, nodes.data(), sizeof(int)*count); p += sizeof(int)*count;
			std::memcpy(p, sizes.data(), sizeof(std::int64_t)*count); p += sizeof(std::int64_t)*count;
			for (int n=0; n<count; n++) {
				kernel.Pack(nodes[n] / S, nodes[n] % S, p);
				p += sizes[n];
			}
			MPI_Isend(message.buffer.data(), message.buffer.size(), MPI_BYTE, mesh.GetFaceNbrRank(fn),
				message_count++, comm, &message.request);
			nodes.clear();
		}
	};

	// non-blocking: unpack every message that has arrived, then return
	auto ReceiveMessages = [&]() {
		while (true) {
			int avail;
			MPI_Status status;
			MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &avail, &status);
			if (avail==0) break;

			int buffer_size;
			MPI_Get_count(&status, MPI_BYTE, &buffer_size);
			std::vector<char> buffer(buffer_size);
			MPI_Recv(buffer.data(), buffer_size, MPI_BYTE, status.MPI_SOURCE, status.MPI_TAG, comm, MPI_STATUS_IGNORE);

			const char *p = buffer.data();
			int count;
			std::memcpy(&count, p, sizeof(int)); p += sizeof(int);
			std::vector<int> nodes(count);
			std::vector<std::int64_t> sizes(count);
			std::memcpy(nodes.data(), p, sizeof(int)*count); p += sizeof(int)*count;
			std::memcpy(sizes.data(), p, sizeof(std::int64_t)*count); p += sizeof(std::int64_t)*count;

			const auto fn = proc_to_fn.at(status.MPI_SOURCE);
			for (int n=0; n<count; n++) {
				// convert the sender's local element to a local ghost id
				const auto s = nodes[n] % S;
				const auto ghost = global_to_fnbr.at(mesh_face_nbr_offsets[fn] + nodes[n] / S);
				kernel.Unpack(ghost, s, p);
				p += sizes[n];

				// mark data as recvd, add the locally owned downwind nodes to the queue
				for (const auto nbr : OutNodes(num_local + ghost*S + s)) {
					if (--degrees[nbr] == 0) local.push_back(nbr);
				}
			}
		}
	};

	// --- do sweep ---
	// all threads run the loop below; the omp for solves the current wave front
	// in parallel (implicit barrier at its end), then one thread advances the
	// front and communicates while the others wait at the end of omp single.
	// nsweep and local are only modified inside omp single.
	// vertices traversed, only count owned elements
	int nsweep = 0;
	#pragma omp parallel
	{
	while (nsweep < num_local) {
		const int wave_front_size = local.size();
		// --- sweep on processor-local domain ---
		#pragma omp for
		for (int sweep_idx=0; sweep_idx<wave_front_size; sweep_idx++) {
			const auto node = local[sweep_idx];
			kernel.Solve(node / S, node % S, InEdges(node));
		}

		#pragma omp single
		{
		// process computed wave front
		// find next wave front
		nsweep += wave_front_size;
		for (int sweep_idx=0; sweep_idx<wave_front_size; sweep_idx++) {
			const auto node = local.front();
			local.pop_front();
			for (const auto nbr : OutNodes(node)) {
				if (nbr < num_local and --degrees[nbr] == 0) local.push_back(nbr);
			}
			if (!lagged) {
				for (const auto fn : SendFaceNbrs(node)) { send_lists[fn].push_back(node); }
			}
		}

		if (!lagged) {
			SendWaveFront();
			ReceiveMessages();
			// clear sent buffers if message has been recvd
			std::erase_if(sent_messages, [](SentMessage &message) {
				int ready;
				MPI_Test(&message.request, &ready, MPI_STATUS_IGNORE);
				return ready != 0;
			});
		}
		} // end omp single
	}
	} // end omp parallel

	if (!lagged) {
		for (auto &message : sent_messages) {
			MPI_Wait(&message.request, MPI_STATUS_IGNORE);
		}
		// use barrier to avoid tag clash with the next sweep
		MPI_Barrier(comm);
	}
	if (nsweep != num_local) MFEM_WARNING("rank " << mesh.GetMyRank() << " didn't sweep all");
}

void SweepScheduler::WriteGraphToDot(std::string prefix) const
{
	auto rank = mesh.GetMyRank();
	FILE *file = fopen(mfem::MakeParFilename(prefix + ".", rank, ".dot").c_str(), "w");
	igraph_write_graph_dot(&graph, file);
	fclose(file);
}

void SweepScheduler::WriteGlobalGraphToDot(std::string prefix,
	const std::function<void(std::ostream&,int,int)> &attributes) const
{
	const auto rank = mesh.GetMyRank();
	const int num_local = Ne*S;
	const int num_nodes = igraph_vcount(&graph);
	auto GlobalNode = [&](igraph_integer_t n) -> HYPRE_BigInt {
		if (n < num_local) return mesh_offsets[0]*S + n;
		n -= num_local;
		return fnbr_to_global[(int)(n / S)]*S + n % S;
	};

	std::ofstream out(mfem::MakeParFilename(prefix + ".", rank, ".dot"));
	out << "digraph {\n";
	for (int i=0; i<num_local; i++) {
		out << "   " << GlobalNode(i) << " [";
		attributes(out, i / S, i % S);
		out << "];\n";
	}
	for (int i=num_local; i<num_nodes; i++) {
		const auto fn = fnbr_to_fn[(i - num_local) / S];
		out << "   " << GlobalNode(i) << " [rank=" << mesh.GetFaceNbrRank(fn) << "];\n";
	}

	out << "\n";

	igraph_eit_t edge_iterator;
	igraph_eit_create(&graph, igraph_ess_all(IGRAPH_EDGEORDER_ID), &edge_iterator);
	igraph_integer_t from, to;
	while (!IGRAPH_EIT_END(edge_iterator)) {
		igraph_edge(&graph, IGRAPH_EIT_GET(edge_iterator), &from, &to);
		out << "   " << GlobalNode(from) << " -> " << GlobalNode(to) << ";\n";
		IGRAPH_EIT_NEXT(edge_iterator);
	}
	igraph_eit_destroy(&edge_iterator);

	out << "}\n";
}
