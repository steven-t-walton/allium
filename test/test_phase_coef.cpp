#include "gtest/gtest.h"
#include "phase_coefficient.hpp"

TEST(PhaseCoef, Isotropic) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(2,2, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	auto f = [](const mfem::Vector &x) {
		return sin(M_PI*x(0))*sin(M_PI*x(1)); 
	};
	mfem::FunctionCoefficient coef(f); 
	IsotropicGrayCoefficient phase(coef); 

	auto &trans = *mesh.GetElementTransformation(0); 
	auto &ref_cent = mfem::Geometries.GetCenter(trans.GetGeometryType()); 
	mfem::Vector center; 
	trans.Transform(ref_cent, center); 
	double eval = phase.Eval(trans, ref_cent); 
	double exact = f(center); 
	EXPECT_DOUBLE_EQ(exact, eval); 
}

TEST(PhaseCoef, SpaceAngle) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(2,2, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	auto f = [](const mfem::Vector &x, const mfem::Vector &Omega) {
		return sin(M_PI*x(0))*sin(M_PI*x(1)) + Omega(0)*Omega(1) + Omega(2); 
	};
	FunctionGrayCoefficient phase(f); 
	mfem::Vector Omega(2); 
	Omega(0) = 1.0; 
	Omega(1) = 0.5; 
	phase.SetState(Omega); 

	auto &trans = *mesh.GetElementTransformation(0); 
	auto &ref_cent = mfem::Geometries.GetCenter(trans.GetGeometryType()); 
	mfem::Vector center; 
	trans.Transform(ref_cent, center); 
	double eval = phase.Eval(trans, ref_cent); 
	// check properly setting Omega to be 3D for all space dim
	double exact = sin(M_PI*center(0))*sin(M_PI*center(1)) + Omega(0)*Omega(1); 
	EXPECT_DOUBLE_EQ(exact, eval); 
}

TEST(PhaseCoef, InflowCurrentIsotropic) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(1,1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	const auto dim = mesh.Dimension(); 
	LevelSymmetricQuadrature quad1(8, dim), quad2(12, dim); 
	auto psi_in = [](const mfem::Vector &x, const mfem::Vector &Omega) {
		return 1./4/M_PI; 
	}; 
	FunctionGrayCoefficient coef(psi_in); 
	InflowPartialCurrentCoefficient Jin1(coef, quad1), Jin2(coef, quad2); 
	// test all faces 
	for (auto f=0; f<mesh.GetNBE(); f++) {
		auto &trans = *mesh.GetBdrFaceTransformations(f); 
		const auto &ip = mfem::Geometries.GetCenter(trans.GetGeometryType()); 
		double val1 = Jin1.Eval(trans, ip); 
		double val2 = Jin2.Eval(trans, ip); 
		double E1 = std::fabs(val1 + 0.25); 
		double E2 = std::fabs(val2 + 0.25); 
		double ooa = log(E1/E2) / log(quad2.Size() / quad1.Size()); 
		EXPECT_NEAR(ooa, 1.0, 0.2); 
	}
}

TEST(PhaseCoef, InflowCurrentQuadratic) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(1,1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	const auto dim = mesh.Dimension(); 
	LevelSymmetricQuadrature quad1(8, dim), quad2(12, dim); 
	auto psi_in = [](const mfem::Vector &x, const mfem::Vector &Omega) {
		return Omega*Omega/4/M_PI; 
	}; 
	FunctionGrayCoefficient coef(psi_in); 
	InflowPartialCurrentCoefficient Jin1(coef, quad1), Jin2(coef, quad2); 
	auto &trans = *mesh.GetBdrFaceTransformations(0); 
	const auto &ip = mfem::Geometries.GetCenter(trans.GetGeometryType()); 
	double val1 = Jin1.Eval(trans, ip); 
	double val2 = Jin2.Eval(trans, ip); 
	double E1 = std::fabs(val1 + 0.1875); 
	double E2 = std::fabs(val2 + 0.1875); 
	double ooa = log(E1/E2) / log(quad2.Size() / quad1.Size()); 
	EXPECT_NEAR(ooa, 1.0, 0.25); 
}