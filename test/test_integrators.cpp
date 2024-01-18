#include "gtest/gtest.h"
#include "p1diffusion.hpp"
#include "smm_integrators.hpp"

TEST(Integrators, Penalty1D) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D(3, 1.0); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(1, dim, mfem::BasisType::GaussLobatto); 
	mfem::FiniteElementSpace fes(&mesh, &fec); 

	mfem::BilinearForm Pform(&fes); 
	Pform.AddInteriorFaceIntegrator(new PenaltyIntegrator(1.0, false)); 
	Pform.AddBdrFaceIntegrator(new PenaltyIntegrator(1.0, false)); 
	Pform.Assemble(); 
	Pform.Finalize(); 
	const auto &P = Pform.SpMat(); 

	auto dense = std::unique_ptr<mfem::DenseMatrix>(P.ToDenseMatrix()); 

	mfem::DenseMatrix exact(6); 
	exact = 0.0; 
	exact(0,0) = 1.0; exact(5,5) = 1.0; 
	exact(1,1) = 1.0; exact(1,2) = -1.0; 
	exact(2,1) = -1.0; exact(2,2) = 1.0; 
	exact(3,3) = 1.0; exact(3,4) = -1.0; 
	exact(4,3) = -1.0; exact(4,4) = 1.0; 
	(*dense) -= exact; 
	auto norm = dense->MaxMaxNorm(); 
	EXPECT_TRUE(norm < 1e-14); 
}

TEST(Integrators, Penalty1DScale) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D(3, 1.0); 
	double hmin, hmax, kmin, kmax; 
	mesh.GetCharacteristics(hmin, hmax, kmin, kmax); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(1, dim, mfem::BasisType::GaussLobatto); 
	mfem::FiniteElementSpace fes(&mesh, &fec); 

	mfem::BilinearForm Pform(&fes); 
	Pform.AddInteriorFaceIntegrator(new PenaltyIntegrator(1.0, true)); 
	Pform.AddBdrFaceIntegrator(new PenaltyIntegrator(1.0, true)); 
	Pform.Assemble(); 
	Pform.Finalize(); 
	const auto &P = Pform.SpMat(); 

	auto dense = std::unique_ptr<mfem::DenseMatrix>(P.ToDenseMatrix()); 

	mfem::DenseMatrix exact(6); 
	exact = 0.0; 
	exact(0,0) = 1.0; exact(5,5) = 1.0; 
	exact(1,1) = 1.0; exact(1,2) = -1.0; 
	exact(2,1) = -1.0; exact(2,2) = 1.0; 
	exact(3,3) = 1.0; exact(3,4) = -1.0; 
	exact(4,3) = -1.0; exact(4,4) = 1.0; 
	exact *= 1./hmin; 
	(*dense) -= exact; 
	auto norm = dense->MaxMaxNorm(); 
	EXPECT_TRUE(norm < 1e-14); 	
}

TEST(Integrators, JumpAverage2D) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(2, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(0, dim, mfem::BasisType::GaussLegendre); 
	mfem::FiniteElementSpace fes(&mesh, &fec), vfes(&mesh, &fec, dim); 

	mfem::MixedBilinearForm Pform(&vfes, &fes); 
	Pform.AddInteriorFaceIntegrator(new DGJumpAverageIntegrator); 
	Pform.Assemble(); 
	Pform.Finalize(); 
	const auto &P = Pform.SpMat(); 

	mfem::GridFunction u(&vfes), v(&fes), x(&fes); 
	u(0) = 1; 
	u(1) = 2; 
	u(2) = 5; 
	u(3) = 10; 
	v(0) = 5; 
	v(1) = 15; 
	P.Mult(u, x); 
	double inner = v*x; 
	double answer = (2+1)*(5-15)/2; 
	EXPECT_DOUBLE_EQ(inner, answer); 
}

TEST(Integrators, VectorJumpJump2DX) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(2, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(0, dim, mfem::BasisType::GaussLegendre); 
	mfem::FiniteElementSpace vfes(&mesh, &fec, dim); 

	mfem::BilinearForm Pform(&vfes); 
	Pform.AddInteriorFaceIntegrator(new DGVectorJumpJumpIntegrator); 
	Pform.Assemble(); 
	Pform.Finalize(); 
	const auto &P = Pform.SpMat(); 

	mfem::GridFunction u(&vfes), v(&vfes), x(&vfes); 
	u(0) = 1; 
	u(1) = 2; 
	u(2) = 10; 
	u(3) = 20; 
	v(0) = 3; 
	v(1) = 6; 
	v(2) = -10; 
	v(3) = -100; 
	P.Mult(u, x); 
	double inner = v*x; 
	double answer = (2-1)*(6-3); 
	EXPECT_DOUBLE_EQ(inner, answer); 
}

TEST(Integrators, VectorJumpJump2DY) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(1, 2, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(0, dim, mfem::BasisType::GaussLegendre); 
	mfem::FiniteElementSpace vfes(&mesh, &fec, dim); 

	mfem::BilinearForm Pform(&vfes); 
	Pform.AddInteriorFaceIntegrator(new DGVectorJumpJumpIntegrator); 
	Pform.Assemble(); 
	Pform.Finalize(); 
	const auto &P = Pform.SpMat(); 

	mfem::GridFunction u(&vfes), v(&vfes), x(&vfes); 
	u(2) = 1; 
	u(3) = 2; 
	u(0) = 10; 
	u(1) = 20; 
	v(2) = 3; 
	v(3) = 6; 
	v(0) = -10; 
	v(1) = -100; 
	P.Mult(u, x); 
	double inner = v*x; 
	double answer = (2-1)*(6-3); 
	EXPECT_DOUBLE_EQ(inner, answer); 
}

TEST(Integrators, SMMWeakDiv) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(1,1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(1, dim, mfem::BasisType::GaussLobatto); 
	mfem::FiniteElementSpace fes(&mesh, &fec); 

	mfem::DenseMatrix T(dim); 
	T(0,0) = 1.0; T(0,1) = T(1,0) = 0.25; T(1,1) = 2.0; 
	mfem::MatrixConstantCoefficient Tc(T); 
	WeakTensorDivergenceLFIntegrator wtd(Tc); 
	mfem::Vector elvec; 
	wtd.AssembleRHSElementVect(*fes.GetFE(0), *mesh.GetElementTransformation(0), elvec); 
	mfem::Vector exact({-0.625, 0.375, -0.375, 0.625, -1.125, -0.875, 0.875, 1.125}); 
	exact -= elvec; 
	double norm = exact.Norml2(); 
	EXPECT_DOUBLE_EQ(norm, 0.0); 
}