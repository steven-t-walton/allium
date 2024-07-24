#include "gtest/gtest.h"
#include "phase_coefficient.hpp"
#include "smm_integrators.hpp"

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
	phase.SetAngle(Omega); 

	auto &trans = *mesh.GetElementTransformation(0); 
	auto &ref_cent = mfem::Geometries.GetCenter(trans.GetGeometryType()); 
	mfem::Vector center; 
	trans.Transform(ref_cent, center); 
	double eval = phase.Eval(trans, ref_cent); 
	// check properly setting Omega to be 3D for all space dim
	double exact = sin(M_PI*center(0))*sin(M_PI*center(1)) + Omega(0)*Omega(1); 
	EXPECT_DOUBLE_EQ(exact, eval); 
}