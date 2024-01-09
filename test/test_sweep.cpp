#include "gtest/gtest.h"
#include "sweep.hpp"

// test sweeping with inflow, no source, and no collision term 
// solution is constant and equal to the inflow source 
bool SweepConstantSolution(mfem::Mesh &smesh, int fe_order) {
	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh); 
	const auto dim = mesh.Dimension();
	LevelSymmetricQuadrature quad(4, dim); 

	mfem::L2_FECollection fec(fe_order, dim, mfem::BasisType::GaussLegendre); 
	mfem::ParFiniteElementSpace fes(&mesh, &fec); 

	mfem::ConstantCoefficient zero(0.0); 
	mfem::ConstantCoefficient inflow(1.0); 

	TransportVectorExtents psi_ext(1, quad.Size(), fes.GetVSize()); 
	const auto psi_size = TotalExtent(psi_ext); 
	mfem::Vector source(psi_size), psi(psi_size); 
	source = 0.0; 

	InvAdvectionOperator Linv(fes, quad, psi_ext, zero, inflow); 
	Linv.Mult(source, psi); 
	bool all_ones = true; 
	for (const auto &i : psi) {
		if (fabs(i - 1.0) > 1e-10) all_ones = false; 
	}
	return all_ones; 
}

TEST(Sweep, ConstantSolution1Dp0) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D(10,1.0);
	EXPECT_TRUE(SweepConstantSolution(mesh, 0)); 	 
}

TEST(Sweep, ConstantSolution1Dp1) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D(10,1.0);
	EXPECT_TRUE(SweepConstantSolution(mesh, 1)); 	 
}

TEST(Sweep, ConstantSolution1Dp2) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D(10,1.0);
	EXPECT_TRUE(SweepConstantSolution(mesh, 2)); 	 
}

TEST(Sweep, ConstantSolution1Dp3) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D(10,1.0);
	EXPECT_TRUE(SweepConstantSolution(mesh, 3)); 	 
}

TEST(Sweep, ConstantSolution2Dp0) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(10,10,mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	EXPECT_TRUE(SweepConstantSolution(mesh, 0)); 
}

TEST(Sweep, ConstantSolution2Dp1) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(10,10,mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	EXPECT_TRUE(SweepConstantSolution(mesh, 1)); 
}

TEST(Sweep, ConstantSolution2Dp2) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(10,10,mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	EXPECT_TRUE(SweepConstantSolution(mesh, 2)); 
}

TEST(Sweep, ConstantSolution3Dp0) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(10,10,10, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0, false); 
	EXPECT_TRUE(SweepConstantSolution(mesh, 0)); 
}

TEST(Sweep, ConstantSolution3Dp1) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(6,6,6, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0, false); 
	EXPECT_TRUE(SweepConstantSolution(mesh, 1)); 
}

TEST(Sweep, ConstantSolution3Dp2) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(3,3,3, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0, false); 
	EXPECT_TRUE(SweepConstantSolution(mesh, 2)); 
}

// --- test with interaction term --- 
// exact solution is exponential 
double ExponentialSolution(mfem::Mesh &smesh, int fe_order, std::function<double(const mfem::Vector&, const mfem::Vector&)> exsol) {
	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh); 
	const auto dim = mesh.Dimension();
	LevelSymmetricQuadrature lvlquad(4, dim); 
	SingleAngleQuadratureRule quad(lvlquad, 0); 

	mfem::L2_FECollection fec(fe_order, dim, mfem::BasisType::GaussLegendre); 
	mfem::ParFiniteElementSpace fes(&mesh, &fec); 

	mfem::ConstantCoefficient total(1.0); 
	mfem::ConstantCoefficient inflow(1.0); 

	TransportVectorExtents psi_ext(1, quad.Size(), fes.GetVSize()); 
	const auto psi_size = TotalExtent(psi_ext); 
	mfem::Vector source(psi_size), psi(psi_size); 
	source = 0.0; 
	psi = 0.0; 

	InvAdvectionOperator Linv(fes, quad, psi_ext, total, inflow); 
	Linv.Mult(source, psi); 

	auto exsol_per_angle = [&quad, &exsol](const mfem::Vector &x) {
		const mfem::Vector &Omega = quad.GetOmega(0); 
		return exsol(x,Omega); 
	};
	mfem::FunctionCoefficient exact_c(exsol_per_angle); 

	mfem::ParGridFunction sol(&fes, psi.GetData()); 
	return sol.ComputeL2Error(exact_c);
}

auto exp_sol_1d = [](const mfem::Vector &x, const mfem::Vector &Omega) {
	return std::exp(-x(0)/Omega(0)); 
}; 

auto exp_sol_2d = [](const mfem::Vector &x, const mfem::Vector &Omega) {
	if (x(0) > x(1)) 
		return exp(-x(1)/Omega(0)); 
	else 
		return exp(-x(0)/Omega(1)); 
};

TEST(Sweep, ExponentialSolution1Dp1) {
	const auto fe_order = 1; 
	mfem::Mesh mesh1 = mfem::Mesh::MakeCartesian1D(10,1.0);
	mfem::Mesh mesh2 = mfem::Mesh::MakeCartesian1D(20,1.0);

	double E1 = ExponentialSolution(mesh1, fe_order, exp_sol_1d); 
	double E2 = ExponentialSolution(mesh2, fe_order, exp_sol_1d); 
	double ooa = log2(E1/E2); 
	bool within_bounds = fabs(fe_order+1 - ooa) < .1; 
	EXPECT_TRUE(within_bounds); 
}

TEST(Sweep, ExponentialSolution1Dp2) {
	const auto fe_order = 2; 
	mfem::Mesh mesh1 = mfem::Mesh::MakeCartesian1D(10,1.0);
	mfem::Mesh mesh2 = mfem::Mesh::MakeCartesian1D(20,1.0);

	double E1 = ExponentialSolution(mesh1, fe_order, exp_sol_1d); 
	double E2 = ExponentialSolution(mesh2, fe_order, exp_sol_1d); 
	double ooa = log2(E1/E2); 
	bool within_bounds = fabs(fe_order+1 - ooa) < .1; 
	EXPECT_TRUE(within_bounds); 
}

TEST(Sweep, ExponentialSolution1Dp3) {
	const auto fe_order = 3; 
	mfem::Mesh mesh1 = mfem::Mesh::MakeCartesian1D(10,1.0);
	mfem::Mesh mesh2 = mfem::Mesh::MakeCartesian1D(20,1.0);

	double E1 = ExponentialSolution(mesh1, fe_order, exp_sol_1d); 
	double E2 = ExponentialSolution(mesh2, fe_order, exp_sol_1d); 
	double ooa = log2(E1/E2); 
	bool within_bounds = fabs(fe_order+1 - ooa) < .1; 
	EXPECT_TRUE(within_bounds); 
}

TEST(Sweep, ExponentialSolution2Dp1) {
	const auto fe_order = 1; 
	auto Ne = 50; 
	mfem::Mesh mesh1 = mfem::Mesh::MakeCartesian2D(Ne,Ne, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	mfem::Mesh mesh2 = mfem::Mesh::MakeCartesian2D(2*Ne,2*Ne, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	double E1 = ExponentialSolution(mesh1, fe_order, exp_sol_2d); 
	double E2 = ExponentialSolution(mesh2, fe_order, exp_sol_2d); 
	double ooa = log2(E1/E2); 
	EXPECT_TRUE(fabs(ooa - 1.0) < .2); 
}

// memory issue in parallel? 
// TEST(Sweep, ExponentialSolution2DTRI) {
// 	const auto fe_order = 1; 
// 	auto Ne = 50; 
// 	mfem::Mesh mesh1 = mfem::Mesh::MakeCartesian2D(Ne,Ne, mfem::Element::TRIANGLE, true, 1.0, 1.0, false); 
// 	mfem::Mesh mesh2 = mfem::Mesh::MakeCartesian2D(2*Ne,2*Ne, mfem::Element::TRIANGLE, true, 1.0, 1.0, false); 
// 	double E1 = ExponentialSolution(mesh1, fe_order, exp_sol_2d); 
// 	double E2 = ExponentialSolution(mesh2, fe_order, exp_sol_2d); 
// 	double ooa = log2(E1/E2); 
// 	EXPECT_TRUE(fabs(ooa - 2.0) < .2); 
// }

TEST(Sweep, ExponentialSolution2Dp2) {
	const auto fe_order = 2; 
	auto Ne = 50; 
	mfem::Mesh mesh1 = mfem::Mesh::MakeCartesian2D(Ne,Ne, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	mfem::Mesh mesh2 = mfem::Mesh::MakeCartesian2D(2*Ne,2*Ne, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	double E1 = ExponentialSolution(mesh1, fe_order, exp_sol_2d); 
	double E2 = ExponentialSolution(mesh2, fe_order, exp_sol_2d); 
	double ooa = log2(E1/E2); 
	bool within_bounds = fabs(ooa - 1.0) < .3; 
	EXPECT_TRUE(within_bounds); 
}