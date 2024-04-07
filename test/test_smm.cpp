#include "gtest/gtest.h"
#include "p1diffusion.hpp"
#include "smm_integrators.hpp"
#include "phase_coefficient.hpp"
#include "linalg.hpp"
#include "block_smm_op.hpp"
#include "sweep.hpp"
#include "transport_op.hpp"

TEST(SMM, CorrectionTensorIsotropic) {
	mfem::Mesh smesh = mfem::Mesh::MakeCartesian2D(5,5,mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(1, dim, mfem::BasisType::GaussLegendre); 
	mfem::ParFiniteElementSpace fes(&mesh, &fec); 
	LevelSymmetricQuadrature quad(4, dim); 

	TransportVectorExtents psi_ext(1,quad.Size(),fes.GetVSize()); 
	mfem::Vector psi(TotalExtent(psi_ext)); 
	auto f = [](const mfem::Vector &x, const mfem::Vector &Omega) {
		return sin(M_PI*x(0))*sin(M_PI*x(1)); 
	};
	FunctionGrayCoefficient psi_coef(f); 
	TransportVectorView psi_view(psi.GetData(), psi_ext); 
	ProjectPsi(fes, quad, psi_coef, psi_view); 
	SMMCorrectionTensorCoefficient T(fes, quad, psi_view);
	mfem::DenseMatrix t(dim); 
	double norm = 0.0; 
	for (auto e=0; e<mesh.GetNE(); e++) {
		auto &trans = *mesh.GetElementTransformation(e); 
		int geom = mesh.GetElementBaseGeometry(e);		
		auto &ref_cent = mfem::Geometries.GetCenter(geom);
		T.Eval(t, trans, ref_cent); 
		norm += t.FNorm(); 
	}

	EXPECT_NEAR(norm, 0.0, 1e-14); 
}

TEST(SMM, CorrectionTensorQuadratic) {
	mfem::Mesh smesh = mfem::Mesh::MakeCartesian2D(5,5,mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(1, dim, mfem::BasisType::GaussLegendre); 
	mfem::ParFiniteElementSpace fes(&mesh, &fec); 
	LevelSymmetricQuadrature quad(4, dim); 

	TransportVectorExtents psi_ext(1,quad.Size(),fes.GetVSize()); 
	mfem::Vector psi(TotalExtent(psi_ext)); 
	auto f = [](const mfem::Vector &x, const mfem::Vector &Omega) {
		return Omega(0)*Omega(0) + Omega(1)*Omega(1) + Omega(0)*Omega(1); 
	};
	FunctionGrayCoefficient psi_coef(f); 
	TransportVectorView psi_view(psi.GetData(), psi_ext); 
	ProjectPsi(fes, quad, psi_coef, psi_view); 

	SMMCorrectionTensorCoefficient T(fes, quad, psi_view);
	mfem::DenseMatrix t(dim); 
	T.Eval(t, *mesh.GetElementTransformation(0), mfem::Geometries.GetCenter(mesh.GetElementBaseGeometry(0))); 
	mfem::DenseMatrix ex({{8*M_PI/45, 4*M_PI/15}, {4*M_PI/15, 8*M_PI/45}}); 
	ex -= t; 
	double norm = ex.FNorm(); 
	EXPECT_NEAR(norm, 0.0, 1e-13); 
}

TEST(SMM, BdrCorrectionLinear) {
	mfem::Mesh smesh = mfem::Mesh::MakeCartesian2D(2,2, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh); 
	const auto dim = mesh.Dimension(); 
	
	mfem::L2_FECollection fec(1, dim, mfem::BasisType::GaussLegendre); 
	mfem::ParFiniteElementSpace fes(&mesh, &fec); 
	LevelSymmetricQuadrature quad(4, dim); 

	TransportVectorExtents psi_ext(1,quad.Size(),fes.GetVSize()); 
	mfem::Vector psi(TotalExtent(psi_ext)); 
	auto f = [](const mfem::Vector &x, const mfem::Vector &Omega) {
		return 5.0 + Omega(0) + Omega(1); 
	};
	FunctionGrayCoefficient psi_coef(f); 
	TransportVectorView psi_view(psi.GetData(), psi_ext); 
	ProjectPsi(fes, quad, psi_coef, psi_view); 

	mfem::Vector nor(dim); 
	nor = 0.0; 
	nor(0) = 1.0; 
	auto alpha = ComputeAlpha(quad, nor); 
	SMMBdrCorrectionFactorCoefficient beta_coef(fes, quad, psi_view, alpha); 
	auto &trans = *mesh.GetFaceElementTransformations(0); 
	const auto &ref_cent = mfem::Geometries.GetCenter(trans.GetGeometryType()); 
	auto beta = beta_coef.Eval(trans, ref_cent); 
	EXPECT_NEAR(beta, 0.0, 1e-14); 
}

// absolute value in beta => can't integrate exactly 
// verify that it converges roughly linearly in #angles 
std::tuple<double,int> BetaError(int sn_order) {
	mfem::Mesh smesh = mfem::Mesh::MakeCartesian2D(2,2, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh); 
	const auto dim = mesh.Dimension(); 
	
	mfem::L2_FECollection fec(1, dim, mfem::BasisType::GaussLegendre); 
	mfem::ParFiniteElementSpace fes(&mesh, &fec); 
	LevelSymmetricQuadrature quad(sn_order, dim); 

	TransportVectorExtents psi_ext(1,quad.Size(),fes.GetVSize()); 
	mfem::Vector psi(TotalExtent(psi_ext)); 
	auto f = [](const mfem::Vector &x, const mfem::Vector &Omega) {
		return Omega(0)*Omega(0) + Omega(1)*Omega(1) + Omega(0)*Omega(1); 
	};
	FunctionGrayCoefficient psi_coef(f); 
	TransportVectorView psi_view(psi.GetData(), psi_ext); 
	ProjectPsi(fes, quad, psi_coef, psi_view); 

	mfem::Vector nor(dim); 
	nor = 0.0; 
	nor(0) = 1.0; 
	auto alpha = ComputeAlpha(quad, nor); 
	SMMBdrCorrectionFactorCoefficient beta_coef(fes, quad, psi_view, alpha); 
	auto &trans = *mesh.GetFaceElementTransformations(0); 
	const auto &ref_cent = mfem::Geometries.GetCenter(trans.GetGeometryType()); 
	auto beta = beta_coef.Eval(trans, ref_cent); 
	auto err = std::fabs(beta - M_PI/6); 
	return {err, quad.Size()}; 
}

TEST(SMM, BdrCorrectionQuadratic) {
	auto [E1,N1] = BetaError(12); 
	auto [E2,N2] = BetaError(24); 
	double ooa = log(E1/E2) / log((double)N2/N1); 
	EXPECT_NEAR(ooa, 1.0, .3); 
}