#include "gtest/gtest.h"
#include "sweep.hpp"
#include "multigroup.hpp"

// test sweeping with inflow, no source, and no collision term 
// solution is constant and equal to the inflow source 
bool SweepConstantSolution(mfem::Mesh &smesh, int fe_order, bool lump=false) {
	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh); 
	const auto dim = mesh.Dimension();
	LevelSymmetricQuadrature quad(4, dim); 

	mfem::L2_FECollection fec(fe_order, dim, mfem::BasisType::GaussLobatto); 
	mfem::ParFiniteElementSpace fes(&mesh, &fec); 

	ConstantPhaseSpaceCoefficient szero(0.0); 
	ConstantPhaseSpaceCoefficient inflow(1.0); 

	mfem::Array<double> energy_grid(3); 
	energy_grid[0] = 0; 
	energy_grid[1] = 1; 
	energy_grid[2] = 2; 
	mfem::Vector zero_data(2); zero_data = 0.0; 
	mfem::ConstantCoefficient zero(0.0);
	GrayMGCoefficient sigma(zero, 2);

	TransportVectorExtents psi_ext(2, quad.Size(), fes.GetVSize()); 
	const auto psi_size = TotalExtent(psi_ext); 
	mfem::Vector source(psi_size), psi(psi_size); 
	TransportVectorView source_view(source.GetData(), psi_ext); 
	FormTransportSource(fes, quad, energy_grid, szero, inflow, source_view); 
	const auto &bdr_attr = mesh.bdr_attributes;
	BoundaryConditionMap bc_map;
	for (const auto &attr : bdr_attr) {
		bc_map[attr] = INFLOW;
	}
	InverseAdvectionOperator Linv(fes, quad, sigma, bc_map, (lump) ? 7 : 0); 
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

TEST(LumpSweep, ConstantSolution1Dp1) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D(10,1.0);
	EXPECT_TRUE(SweepConstantSolution(mesh, 1, true)); 	 
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

TEST(LumpSweep, ConstantSolution2Dp1) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(10,10,mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	EXPECT_TRUE(SweepConstantSolution(mesh, 1, true)); 
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

TEST(LumpSweep, ConstantSolution3Dp1) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(6,6,6, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0, false); 
	EXPECT_TRUE(SweepConstantSolution(mesh, 1, true)); 
}

TEST(Sweep, ConstantSolution3Dp2) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(3,3,3, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0, false); 
	EXPECT_TRUE(SweepConstantSolution(mesh, 2)); 
}

// --- test with interaction term --- 
// exact solution is exponential 
double ExponentialSolution(mfem::Mesh &smesh, int fe_order, 
	std::function<double(double, const mfem::Vector&, const mfem::Vector&)> exsol, bool lump=false) {
	smesh.Finalize(); 
	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh); 
	const auto dim = mesh.Dimension();
	LevelSymmetricQuadrature lvlquad(4, dim); 
	SingleAngleQuadratureRule quad(lvlquad, 0); 

	mfem::L2_FECollection fec(fe_order, dim, mfem::BasisType::GaussLobatto); 
	mfem::ParFiniteElementSpace fes(&mesh, &fec); 
	mfem::ParFiniteElementSpace sigma_fes(&mesh, &fec, 2); 
	mfem::Array<double> energy_grid(3); 
	energy_grid[0] = 0; energy_grid[1] = 0.5; energy_grid[2] = 1.0; 
	mfem::Vector coef_data(2); 
	coef_data[0] = 1.0; 
	coef_data[1] = 0.5; 
	ConstantMGCoefficient total(coef_data);

	ConstantPhaseSpaceCoefficient inflow(1.0); 
	ConstantPhaseSpaceCoefficient zero(0.0); 

	TransportVectorExtents psi_ext(2, quad.Size(), fes.GetVSize()); 
	const auto psi_size = TotalExtent(psi_ext); 
	mfem::Vector source(psi_size), psi(psi_size); 
	TransportVectorView source_view(source.GetData(), psi_ext); 
	FormTransportSource(fes, quad, energy_grid, zero, inflow, source_view); 
	psi = 0.0; 

	BoundaryConditionMap bc_map;
	const auto &bdr_attr = mesh.bdr_attributes;
	for (const auto &attr : bdr_attr) {
		bc_map[attr] = INFLOW;
	}
	InverseAdvectionOperator Linv(fes, quad, total, bc_map, (lump) ? 7 : 0); 
	Linv.Mult(source, psi); 

	double err = 0.0; 
	for (int g=0; g<2; g++) {
		auto exsol_per_angle = [val=coef_data[g], &quad, &exsol](const mfem::Vector &x) {
			const mfem::Vector &Omega = quad.GetOmega(0); 
			return exsol(val, x,Omega); 
		};
		mfem::FunctionCoefficient exact_c(exsol_per_angle); 
		mfem::ParGridFunction sol(&fes, psi, fes.GetVSize()*g); 
		err += sol.ComputeL2Error(exact_c); 
	}

	return err; 
}

auto exp_sol_1d = [](double alpha, const mfem::Vector &x, const mfem::Vector &Omega) {
	return std::exp(-x(0)/Omega(0)*alpha); 
}; 

auto exp_sol_2d = [](double alpha, const mfem::Vector &x, const mfem::Vector &Omega) {
	if (x(0) > x(1)) 
		return exp(-x(1)/Omega(0)*alpha); 
	else 
		return exp(-x(0)/Omega(1)*alpha); 
};

TEST(Sweep, ExponentialSolution1Dp1) {
	const auto fe_order = 1; 
	mfem::Mesh mesh1 = mfem::Mesh::MakeCartesian1D(10,1.0);
	mfem::Mesh mesh2 = mfem::Mesh::MakeCartesian1D(20,1.0);

	double E1 = ExponentialSolution(mesh1, fe_order, exp_sol_1d); 
	double E2 = ExponentialSolution(mesh2, fe_order, exp_sol_1d); 
	double ooa = log2(E1/E2); 
	EXPECT_NEAR(ooa, fe_order+1, 0.1); 
}

TEST(LumpSweep, ExponentialSolution1Dp1) {
	const auto fe_order = 1; 
	mfem::Mesh mesh1 = mfem::Mesh::MakeCartesian1D(10,1.0);
	mfem::Mesh mesh2 = mfem::Mesh::MakeCartesian1D(20,1.0);

	double E1 = ExponentialSolution(mesh1, fe_order, exp_sol_1d, true); 
	double E2 = ExponentialSolution(mesh2, fe_order, exp_sol_1d, true); 
	double ooa = log2(E1/E2); 
	EXPECT_NEAR(ooa, fe_order+1, 0.1); 
}

TEST(Sweep, ExponentialSolution1Dp2) {
	const auto fe_order = 2; 
	mfem::Mesh mesh1 = mfem::Mesh::MakeCartesian1D(10,1.0);
	mfem::Mesh mesh2 = mfem::Mesh::MakeCartesian1D(20,1.0);

	double E1 = ExponentialSolution(mesh1, fe_order, exp_sol_1d); 
	double E2 = ExponentialSolution(mesh2, fe_order, exp_sol_1d); 
	double ooa = log2(E1/E2); 
	EXPECT_NEAR(ooa, fe_order+1, 0.1); 
}

TEST(Sweep, ExponentialSolution1Dp3) {
	const auto fe_order = 3; 
	mfem::Mesh mesh1 = mfem::Mesh::MakeCartesian1D(10,1.0);
	mfem::Mesh mesh2 = mfem::Mesh::MakeCartesian1D(20,1.0);

	double E1 = ExponentialSolution(mesh1, fe_order, exp_sol_1d); 
	double E2 = ExponentialSolution(mesh2, fe_order, exp_sol_1d); 
	double ooa = log2(E1/E2); 
	EXPECT_NEAR(ooa, fe_order+1, 0.1); 
}

TEST(Sweep, ExponentialSolution2Dp1) {
	const auto fe_order = 1; 
	auto Ne = 50; 
	mfem::Mesh mesh1 = mfem::Mesh::MakeCartesian2D(Ne,Ne, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	mfem::Mesh mesh2 = mfem::Mesh::MakeCartesian2D(2*Ne,2*Ne, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	double E1 = ExponentialSolution(mesh1, fe_order, exp_sol_2d); 
	double E2 = ExponentialSolution(mesh2, fe_order, exp_sol_2d); 
	double ooa = log2(E1/E2); 
	EXPECT_NEAR(ooa, 1.0, 0.2); 
}

TEST(LumpSweep, ExponentialSolution2Dp1) {
	const auto fe_order = 1; 
	auto Ne = 50; 
	mfem::Mesh mesh1 = mfem::Mesh::MakeCartesian2D(Ne,Ne, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	mfem::Mesh mesh2 = mfem::Mesh::MakeCartesian2D(2*Ne,2*Ne, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	double E1 = ExponentialSolution(mesh1, fe_order, exp_sol_2d, true); 
	double E2 = ExponentialSolution(mesh2, fe_order, exp_sol_2d, true); 
	double ooa = log2(E1/E2); 
	EXPECT_NEAR(ooa, 1.0, 0.2); 
}

TEST(Sweep, ExponentialSolution2DTRIp0) {
	const auto fe_order = 0; 
	auto Ne = 10; 
	mfem::Mesh mesh1 = mfem::Mesh::MakeCartesian2D(Ne,Ne, mfem::Element::TRIANGLE, true, 1.0, 1.0, false); 
	mfem::Mesh mesh2 = mfem::Mesh::MakeCartesian2D(2*Ne,2*Ne, mfem::Element::TRIANGLE, true, 1.0, 1.0, false); 
	double E1 = ExponentialSolution(mesh1, fe_order, exp_sol_2d); 
	double E2 = ExponentialSolution(mesh2, fe_order, exp_sol_2d); 
	double ooa = log2(E1/E2); 
	EXPECT_NEAR(ooa, fe_order+1, 0.2); // --> mesh aligned with discontinuity => get full accuracy 
}

TEST(Sweep, ExponentialSolution2DTRIp1) {
	const auto fe_order = 1; 
	auto Ne = 10; 
	mfem::Mesh mesh1 = mfem::Mesh::MakeCartesian2D(Ne,Ne, mfem::Element::TRIANGLE, true, 1.0, 1.0, false); 
	mfem::Mesh mesh2 = mfem::Mesh::MakeCartesian2D(2*Ne,2*Ne, mfem::Element::TRIANGLE, true, 1.0, 1.0, false); 
	double E1 = ExponentialSolution(mesh1, fe_order, exp_sol_2d); 
	double E2 = ExponentialSolution(mesh2, fe_order, exp_sol_2d); 
	double ooa = log2(E1/E2); 
	EXPECT_NEAR(ooa, fe_order+1, 0.2); // --> mesh aligned with discontinuity => get full accuracy 
}

TEST(Sweep, ExponentialSolution2DTRIp2) {
	const auto fe_order = 2; 
	auto Ne = 10; 
	mfem::Mesh mesh1 = mfem::Mesh::MakeCartesian2D(Ne,Ne, mfem::Element::TRIANGLE, true, 1.0, 1.0, false); 
	mfem::Mesh mesh2 = mfem::Mesh::MakeCartesian2D(2*Ne,2*Ne, mfem::Element::TRIANGLE, true, 1.0, 1.0, false); 
	double E1 = ExponentialSolution(mesh1, fe_order, exp_sol_2d); 
	double E2 = ExponentialSolution(mesh2, fe_order, exp_sol_2d); 
	double ooa = log2(E1/E2); 
	EXPECT_NEAR(ooa, fe_order+1, 0.2); // --> mesh aligned with discontinuity => get full accuracy 
}


TEST(Sweep, ExponentialSolution2Dp2) {
	const auto fe_order = 2; 
	auto Ne = 50; 
	mfem::Mesh mesh1 = mfem::Mesh::MakeCartesian2D(Ne,Ne, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	mfem::Mesh mesh2 = mfem::Mesh::MakeCartesian2D(2*Ne,2*Ne, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	double E1 = ExponentialSolution(mesh1, fe_order, exp_sol_2d); 
	double E2 = ExponentialSolution(mesh2, fe_order, exp_sol_2d); 
	double ooa = log2(E1/E2); 
	EXPECT_NEAR(ooa, 1.0, 0.3); 
}

void L_Linv(mfem::Mesh &smesh, bool lump) {
	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(1, dim, mfem::BasisType::GaussLobatto); 
	mfem::ParFiniteElementSpace fes(&mesh, &fec); 
	LevelSymmetricQuadrature quad(4, dim); 

	mfem::ParFiniteElementSpace sigma_fes(&mesh, &fec, 2); 
	mfem::Array<double> energy_grid(3); 
	energy_grid[0] = 0; energy_grid[1] = 0.5; energy_grid[2] = 1.0; 
	mfem::Vector coef_data(2); 
	coef_data[0] = 1.0; 
	coef_data[1] = 0.5; 
	ConstantMGCoefficient total(coef_data);

	TransportVectorExtents psi_ext(2, quad.Size(), fes.GetVSize()); 
	const auto psi_size = TotalExtent(psi_ext); 
	mfem::Vector source(psi_size), psi(psi_size); 
	source.Randomize(12345); 
	mfem::Vector copy(source); 

	BoundaryConditionMap bc_map;
	const auto &bdr_attr = mesh.bdr_attributes;
	for (const auto &attr : bdr_attr) {
		bc_map[attr] = INFLOW;
	}
	InverseAdvectionOperator Linv(fes, quad, total, bc_map, (lump) ? 7 : 0); 
	AdvectionOperator L(Linv); 
	L.Mult(source, psi); 
	Linv.Mult(psi, psi); 
	copy -= psi; 
	EXPECT_NEAR(copy.Norml2(), 0.0, 1e-12); 	
}

TEST(AdvectionOperator, Operator1D) {
	auto mesh = mfem::Mesh::MakeCartesian1D(10, 1.0); 
	L_Linv(mesh, false); 
}

TEST(AdvectionOperator, Operator2D) {
	auto mesh = mfem::Mesh::MakeCartesian2D(3,3,mfem::Element::QUADRILATERAL, false, 1.0, 1.0); 
	L_Linv(mesh, false); 
}

TEST(AdvectionOperator, Operator2DTri) {
	auto mesh = mfem::Mesh::MakeCartesian2D(3,3,mfem::Element::TRIANGLE, false, 1.0, 1.0); 
	L_Linv(mesh, false); 
}

TEST(AdvectionOperator, Operator3D) {
	auto mesh = mfem::Mesh::MakeCartesian3D(3,3,3, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0, false); 
	L_Linv(mesh, false); 
}

TEST(LumpedAdvectionOperator, Operator1D) {
	auto mesh = mfem::Mesh::MakeCartesian1D(10, 1.0); 
	L_Linv(mesh, true); 
}

TEST(LumpedAdvectionOperator, Operator2D) {
	auto mesh = mfem::Mesh::MakeCartesian2D(3,3,mfem::Element::QUADRILATERAL, false, 1.0, 1.0); 
	L_Linv(mesh, true); 
}

TEST(LumpedAdvectionOperator, Operator2DTri) {
	auto mesh = mfem::Mesh::MakeCartesian2D(3,3,mfem::Element::TRIANGLE, false, 1.0, 1.0); 
	L_Linv(mesh, true); 
}

TEST(LumpedAdvectionOperator, Operator3D) {
	auto mesh = mfem::Mesh::MakeCartesian3D(3,3,3, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0, false); 
	L_Linv(mesh, true); 
}