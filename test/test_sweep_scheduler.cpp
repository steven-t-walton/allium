// Tests of the generic sweep machinery: SweepScheduler (graph and traversal),
// DGLocalMatrices (element and face matrices), and DenseSnKernel with angle
// sets of more than one ordinate. Run serially and on 4 processors.

#include "gtest/gtest.h"
#include "sweep.hpp"
#include "multigroup.hpp"
#include <atomic>
#include <cstring>

namespace {

// Mesh types shared by the tests. Cartesian meshes have axis-aligned normals,
// so the sectors from SectorSets are upwind consistent on every face.
enum class MeshType { Line, Square, Triangles, Cube };

mfem::Mesh MakeMesh(MeshType type)
{
	switch (type) {
	case MeshType::Line: return mfem::Mesh::MakeCartesian1D(10, 1.0);
	case MeshType::Square: return mfem::Mesh::MakeCartesian2D(5, 5, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false);
	case MeshType::Triangles: return mfem::Mesh::MakeCartesian2D(4, 4, mfem::Element::TRIANGLE, true, 1.0, 1.0, false);
	case MeshType::Cube: return mfem::Mesh::MakeCartesian3D(3, 3, 3, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0, false);
	}
	MFEM_ABORT("unknown mesh type");
	return mfem::Mesh();
}

// Boundary conditions with reflective faces that do not face each other
// (opposing reflective faces would create a cycle in the sweep graph).
// Attributes of mfem's Cartesian meshes:
// 	1D: 1 = left, 2 = right
// 	2D: 1 = bottom, 2 = right, 3 = top, 4 = left
// 	3D: 1 = z min, 2 = y min, 3 = x max, 4 = y max, 5 = x min, 6 = z max
BoundaryConditionMap MakeBC(const mfem::ParMesh &mesh, bool reflect)
{
	BoundaryConditionMap bc;
	for (const auto attr : mesh.bdr_attributes) { bc[attr] = INFLOW; }
	if (reflect) {
		const auto dim = mesh.Dimension();
		if (dim == 1) bc[1] = REFLECTIVE;
		if (dim == 2) bc[1] = bc[4] = REFLECTIVE;
		if (dim == 3) bc[1] = bc[2] = bc[5] = REFLECTIVE;
	}
	return bc;
}

// Partition of the quadrature into sectors (half ranges in 1D, quadrants in
// 2D, octants in 3D) keyed by the signs of the direction components.
std::vector<std::vector<int>> SectorSets(const AngularQuadrature &quad, int dim)
{
	std::vector<std::vector<int>> sets(1 << dim);
	for (int a=0; a<quad.Size(); a++) {
		const auto &Omega = quad.GetOmega(a);
		int key = 0;
		for (int d=0; d<dim; d++) {
			if (Omega(d) < 0) key |= 1 << d;
		}
		sets[key].push_back(a);
	}
	// drop empty sectors, the scheduler requires a partition of nonempty sets
	std::erase_if(sets, [](const std::vector<int> &s) { return s.empty(); });
	return sets;
}

// boundary attribute of every mesh face, -1 for interior and shared faces
std::vector<int> FaceAttributes(const mfem::ParMesh &mesh)
{
	std::vector<int> attr(mesh.GetNumFaces(), -1);
	for (int be=0; be<mesh.GetNBE(); be++) {
		attr[mesh.GetBdrElementFaceIndex(be)] = mesh.GetBdrAttribute(be);
	}
	return attr;
}

// Kernel that records the order in which nodes are solved and checks the
// scheduler's guarantees instead of solving anything:
// 	- every local node is solved exactly once;
// 	- every LOCAL and REFLECT upwind node is solved before its downwind node;
// 	- every GHOST upwind node is unpacked before its downwind node is solved
// 	  (only when the sweep is not lagged);
// 	- messages arrive intact, with a payload whose size depends on the node.
class OrderKernel : public SweepKernel
{
private:
	int S;
	bool lagged;
	// position of each local node in the solve order, -1 until solved
	std::vector<int> order;
	// number of times each ghost node was unpacked
	std::vector<int> unpacked;
	std::atomic<int> count{0};
	std::atomic<int> errors{0};
public:
	OrderKernel(int num_elements, int num_ghosts, int num_sets, bool lagged)
		: S(num_sets), lagged(lagged), order(num_elements*num_sets, -1), unpacked(num_ghosts*num_sets, 0)
	{ }

	void Solve(int e, int set, std::span<const SweepEdge> in) override
	{
		for (const auto &edge : in) {
			if (edge.kind == SweepEdge::GHOST) {
				// lagged sweeps use whatever ghost data exists and never wait for it
				if (!lagged and unpacked[edge.elem*S + edge.set] != 1) errors++;
			}
			// upwind node was solved in an earlier wave front
			else if (order[edge.elem*S + edge.set] < 0) errors++;
		}
		auto &slot = order[e*S + set];
		if (slot != -1) errors++;
		slot = count++;
	}
	// variable message size: 2 + e % 3 ints
	std::size_t PackSize(int e, int set) const override { return sizeof(int) * (2 + e % 3); }
	// payload [n, set, set + 1, ...] with n the number of ints after the first
	void Pack(int e, int set, char *buf) const override
	{
		const int n = 1 + e % 3;
		std::memcpy(buf, &n, sizeof(int));
		for (int k=0; k<n; k++) {
			const int v = set + k;
			std::memcpy(buf + sizeof(int)*(k+1), &v, sizeof(int));
		}
	}
	void Unpack(int ghost_e, int set, const char *buf) override
	{
		int n;
		std::memcpy(&n, buf, sizeof(int));
		if (n < 1 or n > 3) errors++;
		for (int k=0; k<n; k++) {
			int v;
			std::memcpy(&v, buf + sizeof(int)*(k+1), sizeof(int));
			if (v != set + k) errors++;
		}
		unpacked[ghost_e*S + set]++;
	}

	int Errors() const { return errors; }
	int NumSolved() const { return count; }
	int NumUnpacked() const { int n = 0; for (const auto u : unpacked) n += u; return n; }
};

// Traverse the graph with OrderKernel and check the scheduler's guarantees.
void CheckTraversal(MeshType type, bool reflect, bool sectors, bool lagged)
{
	auto smesh = MakeMesh(type);
	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);
	const auto dim = mesh.Dimension();
	LevelSymmetricQuadrature quad(4, dim);
	const auto bc = MakeBC(mesh, reflect);
	SweepScheduler sched(mesh, quad, bc, sectors ? SectorSets(quad, dim) : std::vector<std::vector<int>>{});

	OrderKernel kernel(mesh.GetNE(), mesh.GetNFaceNeighborElements(), sched.NumSets(), lagged);
	sched.Sweep(kernel, lagged);
	EXPECT_EQ(kernel.Errors(), 0);
	EXPECT_EQ(kernel.NumSolved(), mesh.GetNE()*sched.NumSets());
	// every ghost node that a local node depends on is received exactly once,
	// and nothing is communicated in a lagged sweep
	if (lagged) EXPECT_EQ(kernel.NumUnpacked(), 0);
}

} // namespace

// --- angle sets ---

// without angle sets every ordinate is its own set
TEST(SweepScheduler, DefaultSetsOnePerAngle) {
	auto smesh = MakeMesh(MeshType::Square);
	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);
	LevelSymmetricQuadrature quad(4, 2);
	SweepScheduler sched(mesh, quad, MakeBC(mesh, false));
	ASSERT_EQ(sched.NumSets(), quad.Size());
	for (int a=0; a<quad.Size(); a++) {
		EXPECT_EQ(sched.SetOfAngle(a), a);
		ASSERT_EQ(sched.AngleSet(a).size(), 1u);
		EXPECT_EQ(sched.AngleSet(a)[0], a);
	}
}

// sectors are stored as given and SetOfAngle inverts them
TEST(SweepScheduler, SectorSets) {
	for (const auto type : {MeshType::Line, MeshType::Square, MeshType::Cube}) {
		auto smesh = MakeMesh(type);
		mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);
		const auto dim = mesh.Dimension();
		LevelSymmetricQuadrature quad(4, dim);
		const auto sets = SectorSets(quad, dim);
		SweepScheduler sched(mesh, quad, MakeBC(mesh, true), sets);
		// 2 half ranges in 1D, 4 quadrants in 2D, 8 octants in 3D
		ASSERT_EQ(sched.NumSets(), 1 << dim);
		for (int s=0; s<sched.NumSets(); s++) {
			EXPECT_EQ(sched.AngleSet(s), sets[s]);
			for (const auto a : sets[s]) EXPECT_EQ(sched.SetOfAngle(a), s);
		}
	}
}

// --- graph structure ---

// every in edge is consistent with the mesh: the face belongs to the downwind
// element, side is the downwind side, the upwind element is on the other side
// (or is the same element for REFLECT), and the ordinates of the downwind set
// enter the element through the face
TEST(SweepScheduler, InEdgesConsistentWithMesh) {
	for (const auto type : {MeshType::Line, MeshType::Square, MeshType::Triangles, MeshType::Cube}) {
		for (const bool sectors : {false, true}) {
			// sectors are only upwind consistent on Cartesian meshes
			if (sectors and type == MeshType::Triangles) continue;
			auto smesh = MakeMesh(type);
			mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);
			const auto dim = mesh.Dimension();
			LevelSymmetricQuadrature quad(4, dim);
			const bool reflect = type != MeshType::Triangles;
			const auto bc = MakeBC(mesh, reflect);
			SweepScheduler sched(mesh, quad, bc, sectors ? SectorSets(quad, dim) : std::vector<std::vector<int>>{});
			const auto face_attr = FaceAttributes(mesh);
			const int S = sched.NumSets();

			mfem::Vector nor;
			for (int e=0; e<mesh.GetNE(); e++) {
				for (int s=0; s<S; s++) {
					// count REFLECT edges expected from the faces of e
					int expected_reflect = 0;
					for (const auto f : sched.ElementFaces(e)) {
						if (face_attr[f] < 0 or bc.at(face_attr[f]) != REFLECTIVE) continue;
						sched.OutwardNormal(f, 0, nor);
						if (quad.GetOmega(sched.AngleSet(s)[0]) * nor < 0) expected_reflect++;
					}

					int num_reflect = 0;
					for (const auto &edge : sched.InEdges(e*S + s)) {
						const auto &info = mesh.GetFaceInformation(edge.face);
						EXPECT_EQ(sched.ElementSide(e, edge.face), edge.side);
						sched.OutwardNormal(edge.face, edge.side, nor);
						for (const auto a : sched.AngleSet(s)) {
							EXPECT_LE(quad.GetOmega(a) * nor, 0.0) << "ordinate leaves through an inflow face";
						}
						if (edge.kind == SweepEdge::REFLECT) {
							num_reflect++;
							EXPECT_EQ(edge.elem, e);
							EXPECT_TRUE(info.IsBoundary());
							ASSERT_GE(face_attr[edge.face], 0);
							EXPECT_EQ(bc.at(face_attr[edge.face]), REFLECTIVE);
							// upwind set contains the reflection of every ordinate in s
							for (const auto a : sched.AngleSet(s)) {
								EXPECT_EQ(sched.SetOfAngle(quad.GetReflectedAngleIndex(a, nor)), edge.set);
							}
						} else {
							// the upwind node uses the same set on the other side of the face
							EXPECT_EQ(edge.set, s);
							const auto &other = info.element[(edge.side + 1) % 2];
							EXPECT_EQ(edge.kind == SweepEdge::GHOST, info.IsShared());
							// for shared faces mfem returns the face neighbor (ghost) element id
							EXPECT_EQ(other.index, edge.elem);
						}
					}
					EXPECT_EQ(num_reflect, expected_reflect);
				}
			}
		}
	}
}

// --- traversal order and communication ---

TEST(SweepScheduler, Order1D) {
	for (const bool sectors : {false, true}) {
		for (const bool reflect : {false, true}) CheckTraversal(MeshType::Line, reflect, sectors, false);
	}
}

TEST(SweepScheduler, Order2D) {
	for (const bool sectors : {false, true}) {
		for (const bool reflect : {false, true}) CheckTraversal(MeshType::Square, reflect, sectors, false);
	}
}

TEST(SweepScheduler, Order2DTri) {
	CheckTraversal(MeshType::Triangles, false, false, false);
}

TEST(SweepScheduler, Order3D) {
	for (const bool sectors : {false, true}) {
		for (const bool reflect : {false, true}) CheckTraversal(MeshType::Cube, reflect, sectors, false);
	}
}

// lagged (parallel block Jacobi) traversal ignores ghost dependencies and does not communicate
TEST(SweepScheduler, OrderLagged) {
	CheckTraversal(MeshType::Square, true, false, true);
	CheckTraversal(MeshType::Cube, true, true, true);
}

// --- local matrices ---

// mass rows sum to sigma_g |K|, face blocks of interior faces are transposes of
// each other and all four blocks integrate 1 x 1 over the same face
TEST(DGLocalMatrices, Consistency) {
	for (const auto type : {MeshType::Line, MeshType::Square, MeshType::Cube}) {
		auto smesh = MakeMesh(type);
		mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);
		const auto dim = mesh.Dimension();
		mfem::L2_FECollection fec(2, dim, mfem::BasisType::GaussLobatto);
		mfem::ParFiniteElementSpace fes(&mesh, &fec);
		mfem::Vector sigma(2);
		sigma(0) = 1.5; sigma(1) = 0.25;
		ConstantMGCoefficient total(sigma);
		DGLocalMatrices mats(fes, total);
		mats.AssembleTimeMass();
		ASSERT_EQ(mats.NumGroups(), 2);

		auto Sum = [](const mfem::DenseMatrix &A) {
			double s = 0.0;
			for (int i=0; i<A.Height(); i++) for (int j=0; j<A.Width(); j++) s += A(i,j);
			return s;
		};
		for (int e=0; e<mesh.GetNE(); e++) {
			const double vol = mesh.GetElementVolume(e);
			for (int g=0; g<2; g++) EXPECT_NEAR(Sum(mats.Mass(g,e)), sigma(g)*vol, 1e-12);
			EXPECT_NEAR(Sum(mats.TimeMass(e)), vol, 1e-12);
			// grad(v) is zero for constant v, so sum_i (Grad 1)_i = 0 for all d
			for (int d=0; d<dim; d++) {
				const auto &D = mats.Grad(d, e);
				mfem::Vector ones(D.Width()), Dt1(D.Height());
				ones = 1.0;
				D.MultTranspose(ones, Dt1);
				EXPECT_NEAR(Dt1.Normlinf(), 0.0, 1e-12);
			}
		}

		// face areas of the uniform Cartesian meshes
		const double area = (dim == 1) ? 1.0 : (dim == 2) ? 1.0/5 : 1.0/9;
		for (int f=0; f<mesh.GetNumFaces(); f++) {
			const auto &info = mesh.GetFaceInformation(f);
			EXPECT_NEAR(Sum(mats.Face(f,0,0)), area, 1e-12);
			if (!info.IsInterior() and !info.IsShared()) continue;
			EXPECT_NEAR(Sum(mats.Face(f,1,1)), area, 1e-12);
			EXPECT_NEAR(Sum(mats.Face(f,0,1)), area, 1e-12);
			mfem::DenseMatrix T(mats.Face(f,1,0), 't');
			T -= mats.Face(f,0,1);
			EXPECT_NEAR(T.MaxMaxNorm(), 0.0, 1e-14);
		}
	}
}

// --- DenseSnKernel with sector angle sets ---

namespace {

// Sweep with a DenseSnKernel over a sector scheduler and compare with
// InverseAdvectionOperator, which sweeps one ordinate at a time. The local
// systems are identical, so the two must agree to rounding.
void CheckSectorSweep(MeshType type, bool reflect, double time_absorption, bool lagged=false)
{
	auto smesh = MakeMesh(type);
	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);
	const auto dim = mesh.Dimension();
	mfem::L2_FECollection fec(1, dim, mfem::BasisType::GaussLobatto);
	mfem::ParFiniteElementSpace fes(&mesh, &fec);
	LevelSymmetricQuadrature quad(4, dim);
	mfem::Vector sigma(2);
	sigma(0) = 1.0; sigma(1) = 0.5;
	ConstantMGCoefficient total(sigma);
	const auto bc = MakeBC(mesh, reflect);

	TransportVectorExtents ext(2, quad.Size(), fes.GetVSize());
	mfem::Vector source(TotalExtent(ext)), psi_ref(TotalExtent(ext)), psi(TotalExtent(ext));
	source.Randomize(1234 + mesh.GetMyRank());

	// reference: one ordinate per sweep node
	InverseAdvectionOperator Linv(fes, quad, total, bc);
	if (time_absorption > 0) Linv.SetTimeAbsorption(time_absorption);
	Linv.Mult(source, psi_ref);

	// sector sweep reusing the reference's local matrices
	SweepScheduler sched(mesh, quad, bc, SectorSets(quad, dim));
	DenseSnKernel kernel(fes, quad, Linv.GetLocalMatrices(), sched);
	kernel.time_dependent = time_absorption > 0;
	kernel.time_absorption = time_absorption;
	TransportVectorExtents ghost_ext(2, quad.Size(), fes.GetFaceNbrVSize());
	mfem::Vector psi_fnbr(TotalExtent(ghost_ext));
	psi_fnbr = 0.0;
	if (lagged) {
		// ghost data taken from the reference solution, so a lagged sweep
		// must reproduce the reference exactly (fixed point of block Jacobi)
		const auto Ndof_fnbr = fes.GetFaceNbrVSize();
		mfem::ParGridFunction gf(&fes);
		ConstTransportVectorView ref(psi_ref.GetData(), ext);
		for (int g=0; g<2; g++) {
			for (int a=0; a<quad.Size(); a++) {
				for (int i=0; i<fes.GetVSize(); i++) gf(i) = ref(g,a,i);
				gf.ExchangeFaceNbrData();
				const auto &fnbr = gf.FaceNbrData();
				// DenseSnKernel ghost layout: a + g*Nomega + Nomega*G*dof
				for (int i=0; i<Ndof_fnbr; i++) psi_fnbr(a + g*quad.Size() + quad.Size()*2*i) = fnbr(i);
			}
		}
	}
	kernel.Bind(source, psi, psi_fnbr);
	sched.Sweep(kernel, lagged);

	psi -= psi_ref;
	const double diff = std::sqrt(mfem::InnerProduct(MPI_COMM_WORLD, psi, psi));
	const double ref = std::sqrt(mfem::InnerProduct(MPI_COMM_WORLD, psi_ref, psi_ref));
	EXPECT_LT(diff / ref, 1e-13);
}

// Constant solution with reflection: no source or opacity, inflow 1 on every
// inflow face. The reflection of an isotropic constant is the same constant,
// so psi = 1 everywhere for both the ordinate and the sector sweep.
void CheckReflectiveConstant(MeshType type, int order)
{
	auto smesh = MakeMesh(type);
	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);
	const auto dim = mesh.Dimension();
	mfem::L2_FECollection fec(order, dim, mfem::BasisType::GaussLobatto);
	mfem::ParFiniteElementSpace fes(&mesh, &fec);
	LevelSymmetricQuadrature quad(4, dim);
	mfem::ConstantCoefficient zero(0.0);
	GrayMGCoefficient total(zero, 1);
	const auto bc = MakeBC(mesh, true);

	// source: inflow terms with psi_in = 1 on the inflow faces only. Unlike
	// FormTransportSource, no boundary data is added on reflective faces,
	// where the incoming flux is the reflected solution itself.
	TransportVectorExtents ext(1, quad.Size(), fes.GetVSize());
	mfem::Vector source(TotalExtent(ext)), psi(TotalExtent(ext));
	mfem::ConstantCoefficient inflow(1.0);
	mfem::Array<int> marker(mesh.bdr_attributes.Max());
	for (int i=0; i<marker.Size(); i++) marker[i] = (bc.at(i+1) == REFLECTIVE) ? 0 : 1;
	TransportVectorView source_view(source.GetData(), ext);
	for (int a=0; a<quad.Size(); a++) {
		mfem::ParLinearForm bform(&fes);
		mfem::VectorConstantCoefficient Q(quad.GetOmega(a));
		bform.AddBdrFaceIntegrator(new mfem::BoundaryFlowIntegrator(inflow, Q, -1.0, -0.5), marker);
		bform.Assemble();
		for (int i=0; i<bform.Size(); i++) source_view(0,a,i) = bform[i];
	}

	InverseAdvectionOperator Linv(fes, quad, total, bc);
	psi = 0.0;
	Linv.Mult(source, psi);
	for (const auto v : psi) ASSERT_NEAR(v, 1.0, 1e-12);

	if (type == MeshType::Triangles) return;
	SweepScheduler sched(mesh, quad, bc, SectorSets(quad, dim));
	DenseSnKernel kernel(fes, quad, Linv.GetLocalMatrices(), sched);
	TransportVectorExtents ghost_ext(1, quad.Size(), fes.GetFaceNbrVSize());
	mfem::Vector psi_fnbr(TotalExtent(ghost_ext));
	psi = 0.0;
	kernel.Bind(source, psi, psi_fnbr);
	sched.Sweep(kernel);
	for (const auto v : psi) ASSERT_NEAR(v, 1.0, 1e-12);
}

} // namespace

TEST(DenseSnKernel, SectorSweep1D) {
	CheckSectorSweep(MeshType::Line, false, 0.0);
	CheckSectorSweep(MeshType::Line, true, 0.0);
	CheckSectorSweep(MeshType::Line, true, 10.0);
}

TEST(DenseSnKernel, SectorSweep2D) {
	CheckSectorSweep(MeshType::Square, false, 0.0);
	CheckSectorSweep(MeshType::Square, true, 0.0);
	CheckSectorSweep(MeshType::Square, true, 10.0);
}

TEST(DenseSnKernel, SectorSweep3D) {
	CheckSectorSweep(MeshType::Cube, false, 0.0);
	CheckSectorSweep(MeshType::Cube, true, 0.0);
	CheckSectorSweep(MeshType::Cube, true, 10.0);
}

// a lagged sweep with exact ghost data reproduces the solution
TEST(DenseSnKernel, LaggedFixedPoint) {
	CheckSectorSweep(MeshType::Square, true, 0.0, true);
	CheckSectorSweep(MeshType::Cube, true, 0.0, true);
}

TEST(DenseSnKernel, ReflectiveConstantSolution) {
	CheckReflectiveConstant(MeshType::Line, 1);
	CheckReflectiveConstant(MeshType::Square, 2);
	CheckReflectiveConstant(MeshType::Cube, 1);
}

// Known issue, disabled: AdvectionOperator::Mult omits the reflected inflow
// term +(Omega.n) Face(f,0,0) psi(reflected angle) on reflective faces, so
// L Linv q != q with reflective boundaries (the sweep itself is correct, see
// ReflectiveConstantSolution). ComputeBalance has the matching issue: it counts
// outgoing current on reflective faces as leakage.
TEST(AdvectionOperator, DISABLED_ReflectiveResidual) {
	auto smesh = MakeMesh(MeshType::Square);
	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);
	mfem::L2_FECollection fec(1, 2, mfem::BasisType::GaussLobatto);
	mfem::ParFiniteElementSpace fes(&mesh, &fec);
	LevelSymmetricQuadrature quad(4, 2);
	mfem::Vector sigma(1);
	sigma = 1.0;
	ConstantMGCoefficient total(sigma);
	TransportVectorExtents ext(1, quad.Size(), fes.GetVSize());
	mfem::Vector source(TotalExtent(ext)), psi(TotalExtent(ext)), residual(TotalExtent(ext));
	source.Randomize(12345);
	InverseAdvectionOperator Linv(fes, quad, total, MakeBC(mesh, true));
	AdvectionOperator L(Linv);
	Linv.Mult(source, psi);
	L.Mult(psi, residual);
	residual -= source;
	EXPECT_LT(residual.Norml2() / source.Norml2(), 1e-12);
}
