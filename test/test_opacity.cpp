#include "gtest/gtest.h"
#include "opacity.hpp"
#include "multigroup.hpp"
#include "mg_form.hpp"

double CollapseError(int G) {
	auto mesh = mfem::Mesh::MakeCartesian1D(3, 1.0); 
	mfem::Array<double> energy_grid(G+1); 
	for (int i=0; i<G+1; i++) {
		energy_grid[i] = (double)i/G; 
	}
	mfem::Vector midpts(G), widths(G);
	for (int g=0; g<G; g++) {
		midpts(g) = (energy_grid[g] + energy_grid[g+1])/2;
		widths(g) = energy_grid[g+1] - energy_grid[g];
	}
	mfem::L2_FECollection fec(0, mesh.Dimension());
	mfem::FiniteElementSpace fes_gray(&mesh, &fec); 
	auto func = [&midpts](const mfem::Vector &x, mfem::Vector &sigma) {
		for (int g=0; g<midpts.Size(); g++) {
			sigma[g] = pow(midpts[g], 3); 
		}
	};
	mfem::VectorFunctionCoefficient sigma(G, func); 
	mfem::VectorConstantCoefficient weight(widths);
	OpacityGroupCollapseCoefficient gcc(sigma, weight);
	mfem::GridFunction sigma_gray(&fes_gray);
	sigma_gray.ProjectCoefficient(gcc);
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
	GridFunctionMGCoefficient sigma(sigma_data);

	MomentVectorExtents phi_ext(G, 1, fes_gray.GetVSize());
	mfem::Vector E(TotalExtent(phi_ext));
	MomentVectorView view(E.GetData(), phi_ext);
	for (int g=0; g<phi_ext.extent(MomentIndex::ENERGY); g++) {
		const double mid = (bounds[g+1] + bounds[g])/2;
		for (int s=0; s<phi_ext.extent(MomentIndex::SPACE); s++) {
			view(g,0,s) = 1.0/(mid+1.0);
		}
	}
	ZerothMomentCoefficient Ecoef(fes, phi_ext, E);
	OpacityGroupCollapseCoefficient gcc(sigma, Ecoef);
	mfem::GridFunction gray_data(&fes_gray);
	gray_data.ProjectCoefficient(gcc);

	const double exact = 0.1599861728707553;
	return std::fabs(gray_data(0) - exact);
}

TEST(Opacity, WeightedCollapse) {
	double E1 = CollapseError(40); 
	double E2 = CollapseError(80); 
	double ooa = log2(E1/E2); 
	EXPECT_NEAR(ooa, 2.0, .1); 
}

TEST(Opacity, WeightedCollapseMassMatrix) {
	const int G = 200;
	MultiGroupEnergyGrid grid = MultiGroupEnergyGrid::MakeLogSpaced(1e-2, 1e6, G, false);
	auto mesh = mfem::Mesh::MakeCartesian1D(160, 1.0);
	mfem::L2_FECollection fec(1, mesh.Dimension(), mfem::BasisType::GaussLobatto);
	mfem::L2_FECollection sigma_fec(1, mesh.Dimension(), mfem::BasisType::GaussLobatto);
	mfem::FiniteElementSpace fes(&mesh, &fec); 
	mfem::FiniteElementSpace sigma_fes(&mesh, &sigma_fec, G);
	mfem::FiniteElementSpace sigma_fes_gray(&mesh, &fec);

	MomentVectorExtents phi_ext(G, 1, fes.GetVSize());

	mfem::ConstantCoefficient rho(1.0);
	mfem::FunctionCoefficient T([](const mfem::Vector &x) { return 100*sin(M_PI*x(0)) + 1.0; });
	AnalyticOpacityCoefficient opacity(1e9, 1.0, -3.0, grid.Midpoints());
	opacity.SetTemperature(T);
	opacity.SetDensity(rho);
	mfem::GridFunction total(&sigma_fes);
	total.ProjectCoefficient(opacity);
	mfem::GridFunction total_gray(&sigma_fes_gray);

	GridFunctionMGCoefficient total_coef(total);
	mfem::GridFunctionCoefficient total_gray_coef(&total_gray);

	mfem::Vector Enu(TotalExtent(phi_ext));
	Enu.Randomize(12345);

	ZerothMomentCoefficient Ecoef(fes, phi_ext, Enu);
	OpacityGroupCollapseCoefficient gcc(total_coef, Ecoef);
	total_gray.ProjectCoefficient(gcc);

	mfem::BilinearForm gray(&fes);
	gray.AddDomainIntegrator(new mfem::MassIntegrator(total_gray_coef));
	gray.Assemble(); 
	gray.Finalize(); 

	MultiGroupBilinearForm mg(fes, G);
	mg.AddDomainIntegrator(new MGMassIntegrator(total_coef));
	mg.Assemble(); 
	mg.Finalize();

	GroupCollapseOperator to_gray(phi_ext);
	mfem::ProductOperator mg_collapse(&to_gray, &mg, false, false);
	mfem::Vector E(fes.GetVSize());
	to_gray.Mult(Enu, E);

	mfem::GridFunction abs_gray(&fes), abs_mg(&fes);
	gray.Mult(E, abs_gray);
	mg_collapse.Mult(Enu, abs_mg);

	mfem::GridFunctionCoefficient abs_mg_coef(&abs_mg);
	const double err = abs_gray.ComputeL2Error(abs_mg_coef);
	const double mag = abs_mg.Norml2();
	printf("%.3e\n", err/mag);
}