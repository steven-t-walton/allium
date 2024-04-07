#include "gtest/gtest.h"
#include "opacity.hpp"

double CollapseError(int G) {
	auto mesh = mfem::Mesh::MakeCartesian1D(3, 1.0); 
	mfem::Array<double> energy_grid(G+1); 
	for (int i=0; i<G+1; i++) {
		energy_grid[i] = (double)i/G; 
	}
	mfem::L2_FECollection fec(0, mesh.Dimension()); 
	mfem::FiniteElementSpace fes(&mesh, &fec, G);
	mfem::FiniteElementSpace fes_gray(&mesh, &fec); 
	auto func = [&energy_grid](const mfem::Vector &x, mfem::Vector &sigma) {
		for (int g=0; g<energy_grid.Size()-1; g++) {
			sigma[g] = pow((energy_grid[g+1] + energy_grid[g])/2, 3); 
		}
	};
	mfem::VectorFunctionCoefficient sigma(G, func); 
	mfem::GridFunction sigma_gf(&fes); 
	sigma_gf.ProjectCoefficient(sigma); 
	GroupCollapseOperator gco(fes, energy_grid); 
	mfem::GridFunction sigma_gray(&fes_gray); 
	gco.Mult(sigma_gf, sigma_gray); 
	return std::fabs(sigma_gray[0] - 0.25); 
}

TEST(Opacity, GroupCollapse) {
	double E1 = CollapseError(4); 
	double E2 = CollapseError(8); 
	double ooa = log2(E1/E2); 
	EXPECT_NEAR(ooa, 2.0, .1); 
}