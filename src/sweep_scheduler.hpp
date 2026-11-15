#pragma once

#include "mfem.hpp"
#include "igraph.h"
#include "angular_quadrature.hpp"
#include "bdr_conditions.hpp"
#include <functional>
#include <memory>
#include <ostream>
#include <span>
#include <unordered_map>
#include <vector>

// A dependency of a sweep node on an upwind node, stored on the downwind node.
//
// A sweep node is a pair (element e, angle set s). The node can be solved once
// the angular flux of every upwind node it depends on is known:
// 	LOCAL:   flux of (elem, s) enters e through `face`, elem is owned by this processor
// 	GHOST:   same, but elem is a face neighbor element owned by another processor;
// 	         its data arrives through SweepKernel::Unpack
// 	REFLECT: `face` is on a reflective boundary and the incoming flux of the
// 	         ordinates in s is the outgoing flux of their reflections, which all
// 	         belong to the set `set` on the same element e
// The kernel uses `face` and `side` to select the face matrices and the outward
// normal of e on that face (SweepScheduler::OutwardNormal(face, side, ...)).
struct SweepEdge {
	enum Kind {
		LOCAL,  // upwind element is owned by this processor
		GHOST,  // upwind element is a face neighbor (ghost) element
		REFLECT // reflective boundary: upwind node is (e, reflected set)
	};
	Kind kind;
	int face; // mesh face id shared by the two nodes
	int side; // side of the face the downwind element e is on (0 or 1)
	int elem; // upwind element: local id, face neighbor id (GHOST), or e (REFLECT)
	int set;  // upwind angle set, differs from the downwind set only for REFLECT
};

// Discretization-specific work done at each sweep node.
//
// The scheduler owns the traversal order and the parallel communication; the
// kernel owns the unknowns and the element-local solve. This split lets the
// same scheduler sweep different payloads, e.g. dense multigroup Sn fluxes
// (DenseSnKernel) or one compressed tensor per (element, angle set).
//
// Contract:
// 	- Solve(e, s, in) is called exactly once per local node per sweep, after
// 	  every LOCAL and REFLECT upwind node in `in` has been solved and, for a
// 	  non-lagged sweep, after every GHOST upwind node has been unpacked;
// 	- nodes of one wave front are solved concurrently from OpenMP threads, so
// 	  Solve must only write data owned by (e, s) and use per-thread scratch;
// 	- Pack/PackSize are called on the processor that owns e after Solve(e, s),
// 	  Unpack is called on the receiving processor with the face neighbor
// 	  (ghost) element id. Messages can have any size and need not be aligned.
class SweepKernel
{
public:
	virtual ~SweepKernel() = default;
	// solve on element e for angle set `set`, called once all upwind nodes `in` are available
	// called concurrently from OpenMP threads for nodes in the same wave front
	virtual void Solve(int e, int set, std::span<const SweepEdge> in) = 0;
	// bytes needed to send the outflow data of node (e, set) to a face neighbor processor
	virtual std::size_t PackSize(int e, int set) const = 0;
	// serialize node (e, set) into buf (buf may be unaligned)
	virtual void Pack(int e, int set, char *buf) const = 0;
	// store data for face neighbor element ghost_e received from another processor
	virtual void Unpack(int ghost_e, int set, const char *buf) = 0;
};

// Directed acyclic graph over sweep nodes (element, angle set) and its
// parallel traversal.
//
// An angle set is a group of ordinates swept together through the mesh, e.g. a
// single ordinate (standard Sn sweep, the default) or all ordinates of a
// half range / quadrant / octant. Every ordinate in a set must have the same
// upwind direction on every face, otherwise the constructor aborts. On
// Cartesian meshes the sectors keyed by the signs of Omega satisfy this.
//
// Graph edges:
// 	- interior and shared faces: upwind node -> downwind node for each set, the
// 	  upwind side is the one Omega.n > 0 points out of. A set parallel to a
// 	  shared face is oriented from the lower to the higher global element id so
// 	  both processors agree;
// 	- reflective boundary faces: (e, reflected set) -> (e, s) for each set s
// 	  that enters through the face. Reflection is resolved within the sweep, so
// 	  the graph must stay acyclic (e.g. opposing reflective faces are not
// 	  supported, as before this refactor).
//
// Node numbering: local nodes are e*NumSets() + s, face neighbor (ghost) nodes
// are NE*NumSets() + ghost_e*NumSets() + s.
//
// Traversal (Sweep): wave fronts of ready local nodes are solved with OpenMP;
// after each front the outflow of nodes with ghost downwind neighbors is packed
// into one message per face neighbor processor, and received messages release
// the local nodes that depend on them.
class SweepScheduler
{
private:
	mfem::ParMesh &mesh;
	const AngularQuadrature &quad;
	std::vector<std::vector<int>> sets;
	std::vector<int> set_of_angle;
	int Ne, S;

	// this processor owns element ids in the
	// range [mesh_offsets[0], mesh_offsets[1])
	mfem::Array<HYPRE_BigInt> mesh_offsets;
	// mesh_offsets[0] for each neighboring processor
	mfem::Array<HYPRE_BigInt> mesh_face_nbr_offsets;
	// map processor id to face neighbor index
	std::unordered_map<int,int> proc_to_fn;
	// map a ghost element id to its global id
	mfem::Array<HYPRE_BigInt> fnbr_to_global;
	// "inverse" map: global id to the ghost element id
	std::unordered_map<HYPRE_BigInt,int> global_to_fnbr;
	// face neighbor element id to face number
	mfem::Array<int> fnbr_to_fn;
	// table that maps element ids to face ids
	std::unique_ptr<const mfem::Table> element_to_face;
	// list of unit normals on each face, oriented out of element 1
	mfem::Vector normals;
	// the local graph object
	igraph_t graph;

	// incoming edges of each local node in CSR format
	std::vector<int> in_offsets;
	std::vector<SweepEdge> in_edges;
	// outgoing neighbors of each local and ghost node in CSR format
	std::vector<int> out_offsets, out_nodes;
	// unique face neighbor processors each local node sends its data to
	std::vector<int> send_offsets, send_fns;

	std::span<const int> OutNodes(int node) const {
		return {out_nodes.data() + out_offsets[node], out_nodes.data() + out_offsets[node+1]};
	}
	std::span<const int> SendFaceNbrs(int node) const {
		return {send_fns.data() + send_offsets[node], send_fns.data() + send_offsets[node+1]};
	}
public:
	// mesh: parallel mesh, ExchangeFaceNbrData is called here
	// quad: angular quadrature, ordinates are referenced by index
	// bc_map: boundary condition of every boundary attribute; only REFLECTIVE
	// 	faces add graph edges, all other boundary types are inflow data in the source
	// angle_sets: partition of the ordinates of quad; empty => one set per ordinate
	SweepScheduler(mfem::ParMesh &mesh, const AngularQuadrature &quad, const BoundaryConditionMap &bc_map,
		std::vector<std::vector<int>> angle_sets = {});
	~SweepScheduler();
	SweepScheduler(const SweepScheduler&) = delete;
	SweepScheduler &operator=(const SweepScheduler&) = delete;

	// Traverse all local nodes in upwind order calling kernel.Solve.
	// lagged = false: exact parallel sweep, ghost data is communicated as soon
	// 	as each wave front is done (collective, all processors must call)
	// lagged = true: GHOST dependencies are ignored and nothing is communicated,
	// 	the kernel uses whatever ghost data it already has (parallel block
	// 	Jacobi). LOCAL and REFLECT dependencies are still exact.
	void Sweep(SweepKernel &kernel, bool lagged=false) const;

	// number of angle sets
	int NumSets() const { return S; }
	// ordinates of angle set s
	const std::vector<int> &AngleSet(int s) const { return sets[s]; }
	// angle set containing ordinate a
	int SetOfAngle(int a) const { return set_of_angle[a]; }
	// upwind dependencies of local node e*NumSets() + s
	std::span<const SweepEdge> InEdges(int node) const {
		return {in_edges.data() + in_offsets[node], in_edges.data() + in_offsets[node+1]};
	}
	// faces of local element e
	std::span<const int> ElementFaces(int e) const {
		return {element_to_face->GetRow(e), static_cast<std::size_t>(element_to_face->RowSize(e))};
	}
	// side (0 or 1) of face f that element e is on
	int ElementSide(int e, int f) const { return mesh.GetFaceInformation(f).element[0].index == e ? 0 : 1; }
	// unit normal of face f pointing out of the element on `side`
	void OutwardNormal(int f, int side, mfem::Vector &nor) const;

	// write the local graph (local node ids) to prefix.<rank>.dot
	void WriteGraphToDot(std::string prefix) const;
	// write graph with global node ids, attributes(out, e, s) writes the
	// comma-separated dot attributes of local node (e, s)
	void WriteGlobalGraphToDot(std::string prefix,
		const std::function<void(std::ostream&,int,int)> &attributes) const;
};
