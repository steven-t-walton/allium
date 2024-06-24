#include "gtest/gtest.h"
#include "opacity.hpp"
#include "multigroup.hpp"

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
	OpacityGroupCollapseOperator gco(fes, energy_grid); 
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

double WeightedCollapseError(int G) {
	MultiGroupEnergyGrid grid = MultiGroupEnergyGrid::MakeLogSpaced(1e-5, 1e6, G, false);
	const auto &bounds = grid.Bounds();

	auto mesh = mfem::Mesh::MakeCartesian1D(1, 1.0); 
	mfem::L2_FECollection fec(0, mesh.Dimension()); 
	mfem::FiniteElementSpace fes(&mesh, &fec, G);
	mfem::FiniteElementSpace fes_gray(&mesh, &fec); 

	mfem::GridFunction sigma_data(&fes);
	auto func = [&bounds](const mfem::Vector &x, mfem::Vector &y) {
		for (int g=0; g<bounds.Size() - 1; g++) {
			const double mid = (bounds[g] + bounds[g+1])/2;
			y(g) = exp(-pow(mid - bounds[3], 2)/100);
		}
	};
	mfem::VectorFunctionCoefficient sigma_func(G, func);
	sigma_data.ProjectCoefficient(sigma_func);

	MomentVectorExtents phi_ext(G, 1, fes_gray.GetVSize());
	mfem::Vector E(TotalExtent(phi_ext));
	MomentVectorView view(E.GetData(), phi_ext);
	for (int g=0; g<phi_ext.extent(MomentIndex::ENERGY); g++) {
		const double mid = (bounds[g+1] + bounds[g])/2;
		for (int s=0; s<phi_ext.extent(MomentIndex::SPACE); s++) {
			view(g,0,s) = 1.0/(mid+1.0);
		}
	}
	MomentVectorMultiGroupCoefficient Ecoef(fes, phi_ext, E);
	OpacityGroupCollapseOperator gco(fes, bounds, &Ecoef);
	mfem::GridFunction gray_data(&fes_gray);
	gco.Mult(sigma_data, gray_data);

	const double exact = 0.1599861728707553;
	return std::fabs(gray_data(0) - exact);
}

TEST(Opacity, WeightedCollapse) {
	double E1 = CollapseError(40); 
	double E2 = CollapseError(80); 
	double ooa = log2(E1/E2); 
	EXPECT_NEAR(ooa, 2.0, .1); 
}