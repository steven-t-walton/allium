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

// a dependency of sweep node (e, set) on an upwind node
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

// discretization-specific work done at each sweep node
// the scheduler owns the traversal order and parallel communication,
// the kernel owns the unknowns and the element-local solve
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

// directed acyclic graph over sweep nodes (element, angle set) and its parallel traversal
// an angle set is a group of ordinates swept together through the mesh,
// e.g. a single ordinate (standard Sn sweep) or all ordinates in a sector/octant
// every ordinate in a set must have the same upwind direction on every face,
// otherwise the constructor aborts
// local node ids are e*NumSets() + s, face neighbor (ghost) nodes are
// NE*NumSets() + ghost_e*NumSets() + s
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
	// angle_sets partitions the angles of quad; empty => one set per angle
	SweepScheduler(mfem::ParMesh &mesh, const AngularQuadrature &quad, const BoundaryConditionMap &bc_map,
		std::vector<std::vector<int>> angle_sets = {});
	~SweepScheduler();
	SweepScheduler(const SweepScheduler&) = delete;
	SweepScheduler &operator=(const SweepScheduler&) = delete;

	// traverse all local nodes in upwind order calling kernel.Solve
	// lagged = true: data on face neighbor elements is not communicated during the sweep,
	// the kernel uses whatever ghost data it already has (parallel block Jacobi)
	void Sweep(SweepKernel &kernel, bool lagged=false) const;

	int NumSets() const { return S; }
	const std::vector<int> &AngleSet(int s) const { return sets[s]; }
	int SetOfAngle(int a) const { return set_of_angle[a]; }
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

	void WriteGraphToDot(std::string prefix) const;
	// write graph with global node ids, attributes(out, e, s) writes the
	// comma-separated dot attributes of local node (e, s)
	void WriteGlobalGraphToDot(std::string prefix,
		const std::function<void(std::ostream&,int,int)> &attributes) const;
};
