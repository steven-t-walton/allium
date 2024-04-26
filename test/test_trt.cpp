#include "gtest/gtest.h"
#include "trt_op.hpp"
#include "trt_integrators.hpp"
#include "lumped_intrule.hpp"
#include "constants.hpp"

TEST(LumpedTRT, Emission1D) {
	auto mesh = mfem::Mesh::MakeCartesian1D(1, 1.0); 
	auto fec = mfem::L2_FECollection(1, mesh.Dimension(), mfem::BasisType::GaussLobatto); 
	auto fes = mfem::FiniteElementSpace(&mesh, &fec); 
	LumpedIntegrationRule lumped_intrule(*fes.GetFE(0)); 
	mfem::ConstantCoefficient coef(1.0/constants::StefanBoltzmann); 
	auto nfi = BlackBodyEmissionNFI(coef, 2, 2); 
	nfi.SetIntegrationRule(lumped_intrule); 
	const auto &fe = *fes.GetFE(0); 
	mfem::Vector elfun(fe.GetDof()); 
	elfun(0) = 1.0; 
	elfun(1) = 2.0; 
	mfem::Vector elvec; 
	nfi.AssembleElementVector(fe, *fes.GetElementTransformation(0), elfun, elvec); 
	mfem::Vector ex({0.5, pow(2.0,4)/2}); 
	ex -= elvec; 
	EXPECT_NEAR(ex.Norml2(), 0.0, 1e-14); 
}

TEST(LumpedTRT, Emission2D) {
	auto mesh = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL, false, 1.0, 1.0); 
	auto fec = mfem::L2_FECollection(1, mesh.Dimension(), mfem::BasisType::GaussLobatto); 
	auto fes = mfem::FiniteElementSpace(&mesh, &fec); 
	LumpedIntegrationRule lumped_intrule(*fes.GetFE(0)); 
	mfem::ConstantCoefficient coef(1.0/constants::StefanBoltzmann); 
	auto nfi = BlackBodyEmissionNFI(coef, 2, 2); 
	nfi.SetIntegrationRule(lumped_intrule); 
	const auto &fe = *fes.GetFE(0); 
	mfem::Vector elfun(fe.GetDof()); 
	elfun(0) = 0.0; 
	elfun(1) = 1.0; 
	elfun(2) = 2.0; 
	elfun(3) = 3.0; 
	mfem::Vector elvec; 
	nfi.AssembleElementVector(fe, *fes.GetElementTransformation(0), elfun, elvec); 
	mfem::Vector ex({0.0, 0.25, 4.0, pow(3.0, 4)/4}); 
	ex -= elvec; 
	EXPECT_NEAR(ex.Norml2(), 0.0, 1e-13); 
}

TEST(LumpedTRT, EmissionJacobian1D) {
	auto mesh = mfem::Mesh::MakeCartesian1D(1, 1.0); 
	auto fec = mfem::L2_FECollection(1, mesh.Dimension(), mfem::BasisType::GaussLobatto); 
	auto fes = mfem::FiniteElementSpace(&mesh, &fec); 
	LumpedIntegrationRule lumped_intrule(*fes.GetFE(0)); 
	mfem::ConstantCoefficient coef(1.0); 
	auto nfi = BlackBodyEmissionNFI(coef, 2, 2); 
	nfi.SetIntegrationRule(lumped_intrule); 
	const auto &fe = *fes.GetFE(0); 
	mfem::Vector elfun(fe.GetDof()); 
	elfun = 1.0; 
	mfem::DenseMatrix grad; 
	nfi.AssembleElementGrad(fe, *fes.GetElementTransformation(0), elfun, grad); 
	grad *= 1.0/constants::StefanBoltzmann; 
	mfem::DenseMatrix ex({{2.0, 0.0}, {0.0, 2.0}}); 
	grad -= ex; 
	double norm = grad.FNorm(); 
	EXPECT_NEAR(norm, 0.0, 1e-14); 
}

TEST(LumpedTRT, EmissionJacobian2D) {
	auto mesh = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL, false, 1.0, 1.0); 
	auto fec = mfem::L2_FECollection(1, mesh.Dimension(), mfem::BasisType::GaussLobatto); 
	auto fes = mfem::FiniteElementSpace(&mesh, &fec); 
	LumpedIntegrationRule lumped_intrule(*fes.GetFE(0)); 
	mfem::ConstantCoefficient coef(1.0); 
	auto nfi = BlackBodyEmissionNFI(coef, 2, 2); 
	nfi.SetIntegrationRule(lumped_intrule); 
	const auto &fe = *fes.GetFE(0); 
	mfem::Vector elfun(fe.GetDof()); 
	elfun = 1.0; 
	mfem::DenseMatrix grad; 
	nfi.AssembleElementGrad(fe, *fes.GetElementTransformation(0), elfun, grad); 
	grad *= 1.0/constants::StefanBoltzmann; 
	mfem::DenseMatrix ex(4); 
	ex = 0.0; 
	for (int i=0; i<4; i++) ex(i,i) = 1.0; 
	grad -= ex; 
	double norm = grad.FNorm(); 
	EXPECT_NEAR(norm, 0.0, 1e-14); 
}

TEST(TRT, BlockInverse) {
	auto mesh = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL, false, 1.0, 1.0); 
	auto fec = mfem::L2_FECollection(1, mesh.Dimension(), mfem::BasisType::GaussLobatto); 
	auto fes = mfem::FiniteElementSpace(&mesh, &fec); 
	mfem::ConstantCoefficient coef(1.0); 
	mfem::NonlinearForm form(&fes); 
	form.AddDomainIntegrator(new BlackBodyEmissionNFI(coef, 2, 2)); 
	form.AddDomainIntegrator(new mfem::MassIntegrator(coef)); 
	mfem::GridFunction temperature(&fes); 
	temperature = 1.0; 
	const auto &grad = form.GetGradient(temperature); 
	NonlinearFormBlockInverse inv_grad(form, temperature); 
	mfem::Vector x(fes.GetVSize()), z(fes.GetVSize()); 
	x.Randomize(); 
	mfem::Vector y(x); 
	grad.Mult(y, z); 
	inv_grad.Mult(z, y); 
	y -= x; 
	EXPECT_NEAR(y.Norml2(), 0.0, 1e-12); 
}

TEST(LumpedTRT, BlockInverse) {
	auto mesh = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL, false, 1.0, 1.0); 
	auto fec = mfem::L2_FECollection(1, mesh.Dimension(), mfem::BasisType::GaussLobatto); 
	auto fes = mfem::FiniteElementSpace(&mesh, &fec); 
	mfem::ConstantCoefficient coef(1.0); 
	LumpedIntegrationRule lumped_intrule(*fes.GetFE(0)); 
	mfem::NonlinearForm form(&fes); 
	form.AddDomainIntegrator(new BlackBodyEmissionNFI(coef, 2, 2)); 
	form.AddDomainIntegrator(new mfem::MassIntegrator(coef)); 
	auto &dnfi = *form.GetDNFI(); 
	for (auto &ptr : dnfi) {
		ptr->SetIntegrationRule(lumped_intrule); 
	}
	mfem::GridFunction temperature(&fes); 
	temperature = 1.0; 
	const auto &grad = form.GetGradient(temperature); 
	NonlinearFormBlockInverse inv_grad(form, temperature, true); 
	mfem::Vector x(fes.GetVSize()), z(fes.GetVSize()); 
	x.Randomize(); 
	mfem::Vector y(x); 
	grad.Mult(y, z); 
	inv_grad.Mult(z, y); 
	y -= x; 
	EXPECT_NEAR(y.Norml2(), 0.0, 1e-12); 
}

// solve T^4 - T = 0 <=> T (T^3 - 1) = 0 => solutions are 0 and +1 
TEST(LumpedTRT, NewtonSolve) {
	auto mesh = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL, false, 1.0, 1.0); 
	auto fec = mfem::L2_FECollection(1, mesh.Dimension(), mfem::BasisType::GaussLobatto); 
	auto fes = mfem::FiniteElementSpace(&mesh, &fec); 
	mfem::ConstantCoefficient sbi(1.0/constants::StefanBoltzmann); 
	mfem::ConstantCoefficient none(-1.0); 
	LumpedIntegrationRule lumped_intrule(*fes.GetFE(0)); 
	mfem::NonlinearForm form(&fes); 
	form.AddDomainIntegrator(new BlackBodyEmissionNFI(sbi, 2, 2)); 
	form.AddDomainIntegrator(new mfem::MassIntegrator(none)); 
	auto &dnfi = *form.GetDNFI(); 
	for (auto &ptr : dnfi) {
		ptr->SetIntegrationRule(lumped_intrule); 
	}

	mfem::GridFunction T(&fes), T0(&fes); 
	mfem::Vector resid(fes.GetVSize()), dT(fes.GetVSize()); 
	T0(0) = 12.0; T0(1) = 7.3; T0(2) = 9.6; T0(3) = 10.0; 
	int it; 
	double norm; 
	for (it=0; it<20; it++) {
		form.Mult(T0, resid);
		NonlinearFormBlockInverse grad_inv(form, T0, true); 
		grad_inv.Mult(resid, dT); 
		add(T0, -1.0, dT, T); 
		norm = dT.Norml2(); 
		if (norm < 1e-6) break; 
		T0 = T; 
	}
	mfem::Vector ones(fes.GetVSize()); 
	ones = 1.0; 
	T -= ones; 
	EXPECT_NEAR(T.Norml2(), 0.0, 1e-12); 
}

TEST(TRT, NewtonSolve) {
	auto mesh = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL, false, 1.0, 1.0); 
	auto fec = mfem::L2_FECollection(1, mesh.Dimension(), mfem::BasisType::GaussLobatto); 
	auto fes = mfem::FiniteElementSpace(&mesh, &fec); 
	mfem::ConstantCoefficient sbi(1.0/constants::StefanBoltzmann); 
	mfem::ConstantCoefficient none(-1.0); 
	mfem::NonlinearForm form(&fes); 
	form.AddDomainIntegrator(new BlackBodyEmissionNFI(sbi, 2, 2)); 
	form.AddDomainIntegrator(new mfem::MassIntegrator(none)); 

	mfem::GridFunction T(&fes), T0(&fes); 
	mfem::Vector resid(fes.GetVSize()), dT(fes.GetVSize()); 
	T0(0) = 12.0; T0(1) = 7.3; T0(2) = 9.6; T0(3) = 10.0; 
	int it; 
	double norm; 
	for (it=0; it<20; it++) {
		form.Mult(T0, resid);
		NonlinearFormBlockInverse grad_inv(form, T0, false); 
		grad_inv.Mult(resid, dT); 
		add(T0, -1.0, dT, T); 
		norm = dT.Norml2(); 
		if (norm < 1e-6) break; 
		T0 = T; 
	}
	mfem::Vector ones(fes.GetVSize()); 
	ones = 1.0; 
	T -= ones; 
	EXPECT_NEAR(T.Norml2(), 0.0, 1e-12); 
}