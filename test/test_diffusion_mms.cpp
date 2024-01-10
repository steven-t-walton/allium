#include "gtest/gtest.h"
#include "p1diffusion.hpp"

double P1DiffusionError1D(int Ne, int fe_order) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D(Ne, 1.0); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(fe_order, dim, mfem::BasisType::GaussLegendre); 
	mfem::FiniteElementSpace fes(&mesh, &fec); 
	mfem::FiniteElementSpace vfes(&mesh, &fec, dim); 

	mfem::Array<int> offsets(3); 
	offsets[1] = vfes.GetVSize(); 
	offsets[2] = fes.GetVSize(); 
	offsets.PartialSum(); 

	mfem::ConstantCoefficient total(1.0); 
	mfem::ConstantCoefficient absorption(0.0); 

	mfem::BilinearForm Mtform(&vfes);
	mfem::ProductCoefficient total3(3.0, total); 
	Mtform.AddDomainIntegrator(new mfem::VectorMassIntegrator(total3));
	Mtform.AddInteriorFaceIntegrator(new DGVectorJumpJumpIntegrator); 
	Mtform.AddBdrFaceIntegrator(new DGVectorJumpJumpIntegrator); 
	Mtform.Assemble(); 
	Mtform.Finalize();  

	mfem::BilinearForm Maform(&fes); 
	mfem::ConstantCoefficient alpha(0.25); 
	Maform.AddDomainIntegrator(new mfem::MassIntegrator(absorption)); 
	Maform.AddInteriorFaceIntegrator(new PenaltyIntegrator(4, false)); 
	Maform.AddBdrFaceIntegrator(new mfem::BoundaryMassIntegrator(alpha)); 
	Maform.Assemble(); 
	Maform.Finalize(); 

	mfem::MixedBilinearForm Dform(&vfes, &fes); 
	mfem::ConstantCoefficient neg_one(-1.0); 
	Dform.AddDomainIntegrator(new mfem::TransposeIntegrator(new mfem::GradientIntegrator(neg_one))); 
	Dform.AddInteriorFaceIntegrator(new DGJumpAverageIntegrator); 
	Dform.AddBdrFaceIntegrator(new DGJumpAverageIntegrator(0.5)); 
	Dform.Assemble(); 
	Dform.Finalize(); 
	auto DT = std::unique_ptr<mfem::SparseMatrix>(Transpose(Dform.SpMat())); 
	(*DT) *= -1.0; 

	mfem::LinearForm fform(&fes); 
	auto source = [](const mfem::Vector &x) {
		return 1.0; 
	};
	mfem::FunctionCoefficient source_coef(source); 
	fform.AddDomainIntegrator(new mfem::DomainLFIntegrator(source_coef)); 
	fform.Assemble(); 


	mfem::BlockVector b(offsets), x(offsets); 
	b.GetBlock(0) = 0.0; 
	b.GetBlock(1) = fform; 

	mfem::BlockMatrix A(offsets); 
	A.SetBlock(0,0, &Mtform.SpMat()); 
	A.SetBlock(1,0, &Dform.SpMat()); 
	A.SetBlock(0,1, DT.get()); 
	A.SetBlock(1,1, &Maform.SpMat()); 
	A.Finalize(); 
	auto mono = std::unique_ptr<mfem::SparseMatrix>(A.CreateMonolithic()); 

	mfem::KLUSolver solver(*mono); 
	solver.Mult(b, x); 

	mfem::GridFunction u(&fes, x.GetBlock(1)); 
	mfem::GridFunction v(&vfes, x.GetBlock(0)); 

	auto exact_func = [](const mfem::Vector &x) {
		return 1 + 3*x(0)/2 - 3*x(0)*x(0)/2; 
	}; 
	mfem::FunctionCoefficient exact_coef(exact_func); 
	double err = u.ComputeL2Error(exact_coef); 

	return err; 
}

TEST(MMS, P1Diffusion1Dp1) {
	const auto fe_order = 1; 
	double E1 = P1DiffusionError1D(10, fe_order); 
	double E2 = P1DiffusionError1D(20, fe_order); 
	double ooa = log2(E1/E2); 
	EXPECT_TRUE(std::fabs(fe_order+1-ooa)<.2); 
}

// exact solution is quadratic => error should be zero independent of h 
TEST(MMS, P1Diffusion1Dp2) {
	const auto fe_order = 2; 
	double E = P1DiffusionError1D(10, fe_order); 
	EXPECT_TRUE(E < 1e-12); 
}