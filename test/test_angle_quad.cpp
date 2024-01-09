#include "gtest/gtest.h"
#include "angular_quadrature.hpp"

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

	std::vector<int> orders = {2,4,6,8,10,12,16}; 
	for (auto d=2; d<4; d++) {
		for (const auto &order : orders) {
			LevelSymmetricQuadrature quad(order,d); 
			double val = Integrate(quad, f); 
			EXPECT_FLOAT_EQ(val, 4*M_PI); 		
		}		
	}
}

TEST(AngularQuadrature, quadratic) {
	auto f = [](const mfem::Vector &Omega) {
		return 2*Omega(0)*Omega(0) + Omega(1)*Omega(1) + Omega(0)*Omega(1); 
	}; 

	std::vector<int> orders = {2,4,6,8,10,12,16}; 
	for (auto d=2; d<4; d++) {
		for (const auto &order : orders) {
			LevelSymmetricQuadrature quad(order,d); 
			double val = Integrate(quad, f); 
			EXPECT_FLOAT_EQ(val, 4*M_PI); 		
		}		
	}
}
