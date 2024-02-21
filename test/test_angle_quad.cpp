#include "gtest/gtest.h"
#include "angular_quadrature.hpp"
#include "tvector.hpp"
#include "transport_op.hpp"

double Integrate(const AngularQuadrature &quad, std::function<double(const mfem::Vector &)> f) {
	double r = 0;
	for (auto a=0; a<quad.Size(); a++) {
		const mfem::Vector &Omega = quad.GetOmega(a); 
		r += f(Omega) * quad.GetWeight(a); 
	}
	return r; 
}

TEST(AngularQuadrature, isotropic) {
	auto f = [](const mfem::Vector &Omega) {
		return 1; 
	}; 

	for (auto d=1; d<4; d++) {
		for (auto order=2; order<25; order+=2) {
			LevelSymmetricQuadrature quad(order,d); 
			double val = Integrate(quad, f); 
			EXPECT_NEAR(val, 4*M_PI, 1e-13); 		
		}		
	}
}

TEST(AngularQuadrature, quadratic) {
	auto f = [](const mfem::Vector &Omega) {
		return 2*Omega(0)*Omega(0) + Omega(1)*Omega(1) + Omega(0)*Omega(1); 
	}; 

	for (auto d=2; d<4; d++) {
		for (auto order=2; order<25; order+=2) {
			LevelSymmetricQuadrature quad(order,d); 
			double val = Integrate(quad, f); 
			EXPECT_NEAR(val, 4*M_PI, 1e-13); 		
		}		
	}
}

TEST(AngularQuadrature, NPoints1D) {
	for (auto order=2; order<25; order+=2) {
		LevelSymmetricQuadrature quad(order, 1); 
		EXPECT_EQ(quad.Size(), order); 
	}	
}

TEST(AngularQuadrature, GaussLegendre1D) {
	auto f = [](const mfem::Vector &Omega) {
		return Omega(0)*Omega(0)/4/M_PI; 
	};
	for (auto order=2; order<25; order+=2) {
		LevelSymmetricQuadrature quad(order,1); 
		double val = Integrate(quad, f); 
		EXPECT_NEAR(val, 1./3, 1e-13); 
	}
}

TEST(AngularQuadrature, Reflection2D) {
	const auto dim = 2; 
	LevelSymmetricQuadrature quad(4,dim); 
	mfem::Vector nor(dim); 
	nor = 1.0; 

	for (int angle=0; angle<quad.Size(); angle++) {
		auto r = quad.GetReflectedAngleIndex(angle, nor); 
	}
}

TEST(AngularQuadrature, Reflection3D) {
	const auto dim = 3; 
	LevelSymmetricQuadrature quad(4,dim); 
	mfem::Vector nor(dim); 
	nor = 1.0; 

	for (int angle=0; angle<quad.Size(); angle++) {
		auto r = quad.GetReflectedAngleIndex(angle, nor); 
	}
}

TEST(DiscreteToMoment, Isotropic) {
	LevelSymmetricQuadrature quad(4, 2); 
	TransportVectorExtents psi_ext(1,quad.Size(), 1); 
	MomentVectorExtents phi_ext(1,1,1); 
	DiscreteToMoment D(quad, psi_ext, phi_ext); 

	const auto psi_size = TotalExtent(psi_ext); 
	const auto phi_size = TotalExtent(phi_ext); 
	mfem::Vector psi(psi_size), phi(phi_size); 
	TransportVectorView psi_view(psi.GetData(), psi_ext); 
	for (auto a=0; a<quad.Size(); a++) {
		const auto &Omega = quad.GetOmega(a); 
		psi_view(0,a,0) = 2*Omega(0)*Omega(0) + Omega(1)*Omega(1) + Omega(0)*Omega(1); 
	}
	D.Mult(psi, phi); 
	EXPECT_NEAR(phi(0), 4*M_PI, 1e-13); 
}

TEST(DiscreteToMoment, LinearlyAnisotropic) {
	LevelSymmetricQuadrature quad(4, 2); 
	TransportVectorExtents psi_ext(1,quad.Size(), 1); 
	MomentVectorExtents phi_ext(1,3,1); 
	DiscreteToMoment D(quad, psi_ext, phi_ext); 

	const auto psi_size = TotalExtent(psi_ext); 
	const auto phi_size = TotalExtent(phi_ext); 
	mfem::Vector psi(psi_size), phi(phi_size); 
	TransportVectorView psi_view(psi.GetData(), psi_ext); 
	MomentVectorView phi_view(phi.GetData(), phi_ext); 
	for (auto a=0; a<quad.Size(); a++) {
		const auto &Omega = quad.GetOmega(a); 
		psi_view(0,a,0) = (Omega*Omega + Omega(0) + Omega(1))/4/M_PI; 
	}
	D.Mult(psi, phi); 
	EXPECT_NEAR(phi_view(0,0,0), 2./3, 1e-13); 	
	EXPECT_NEAR(phi_view(0,1,0), 1./3, 1e-13); 	
	EXPECT_NEAR(phi_view(0,2,0), 1./3, 1e-13); 	
}